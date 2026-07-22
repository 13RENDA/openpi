"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


# Tiny placeholder image. B1KInputs/_parse_image only converts images to uint8 HWC and
# forwards them to outputs that this script never reads, so the content is irrelevant.
_DUMMY_IMAGE = np.zeros((3, 2, 2), dtype=np.float32)


class InjectDummyImages(transforms.DataTransformFn):
    """Inject placeholder images for camera keys whose video decoding was disabled."""

    def __init__(self, camera_keys: list[str]):
        self._camera_keys = camera_keys

    def __call__(self, x: dict) -> dict:
        for key in self._camera_keys:
            if key not in x:
                x[key] = _DUMMY_IMAGE
        return x


def disable_video_decoding(dataset) -> list[str]:
    """Strip video features from a LeRobotDataset's metadata so __getitem__ skips decoding.

    Norm stats only need `state` and `actions`, which live in the parquet files; decoding
    video frames dominates runtime otherwise. `meta.video_keys` is derived from
    `meta.info.features` at access time, so removing the video features here disables the
    `_query_videos` call in the reader.

    Must be called after the dataset is constructed (video keys are used during construction
    to validate/download video files) and before the DataLoader is created (spawned workers
    receive a pickled copy of the mutated metadata). The mutation is in-memory only; this
    dataset is read-only, so nothing writes it back to `meta/info.json`.

    Returns the removed camera keys, in dataset-key form (e.g. "observation.rgb.xxx").
    """
    ds = dataset
    while not hasattr(ds, "meta") and hasattr(ds, "_dataset"):
        ds = ds._dataset
    if not hasattr(ds, "meta"):
        return []
    info = ds.meta.info
    # `info` is a dict in older lerobot versions, a DatasetInfo dataclass in newer ones.
    features = info["features"] if isinstance(info, dict) else info.features
    video_keys = [key for key, feature in features.items() if feature["dtype"] == "video"]
    for key in video_keys:
        features.pop(key)
    return video_keys


def resolve_asset_id(data_config: _config.DataConfig) -> str:
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Data config must have an asset_id or repo_id")
    if isinstance(asset_id, list):
        if not asset_id:
            raise ValueError("Data config asset_id/repo_id list cannot be empty")
        return asset_id[0]
    return asset_id


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_b1k_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_b1k_dataset(data_config=data_config, action_horizon=action_horizon)
    # Norm stats only use state/actions; skip video decoding and substitute dummy images
    # so the repack/data transforms still find the camera keys they expect.
    removed_camera_keys = disable_video_decoding(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            InjectDummyImages(removed_camera_keys),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches

def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    elif data_config.dataset_root is not None:
        data_loader, num_batches = create_b1k_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.num_workers, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    asset_id = resolve_asset_id(data_config)
    output_path = config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
