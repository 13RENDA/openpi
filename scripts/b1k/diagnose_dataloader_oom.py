"""Standalone dataloader-only OOM diagnostic for pi05_b1k.

Reproduces the exact sample sequence the real training job sees (same
seed=42, same batch_size=64, no offset for --resume/start_step -- the
seed never changes for the shuffle, so the Nth batch since dataloader
creation is always the same regardless of what step you resumed from,
see scripts/b1k/train_b1k.py's `for step in range(start_step, ...)` loop
using a data_loader built once with a fixed seed).

Runs num_workers=0 on purpose: PyTorch's shuffle order comes from the
main-process sampler, not the workers, so num_workers doesn't change
which samples land in which batch -- only how fetching them is
parallelized. Running everything in one process makes it possible to
directly monkeypatch lerobot's VideoDecoderCache and correlate memory
growth with specific video files, without needing to aggregate RSS
across worker subprocesses.

No GPU/JAX training touched -- this only exercises the data pipeline.

Usage (on the cluster, from the openpi repo root):
    uv run python scripts/b1k/diagnose_dataloader_oom.py [--max-steps 1000] [--start-step 30000]

--start-step doesn't change which data is read (see above) -- it only
changes the printed step numbers, so they line up with the real training
log for comparison.
"""
import argparse
import dataclasses
import resource
import time

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from lerobot.datasets.video_utils import VideoDecoderCache

log_file = open("dataloader_oom_diagnostic.log", "a", buffering=1)


def log(msg: str) -> None:
    print(msg, flush=True)
    log_file.write(msg + "\n")


def current_rss_mb() -> float:
    # ru_maxrss is peak RSS so far (KB on Linux), monotonically non-decreasing --
    # exactly what we want to catch a spike even if it's momentary.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def patch_decoder_cache_logging():
    """Wrap VideoDecoderCache.get_decoder to log every new video it caches."""
    original_get_decoder = VideoDecoderCache.get_decoder

    def logged_get_decoder(self, video_path: str):
        is_new = str(video_path) not in self._cache
        decoder = original_get_decoder(self, video_path)
        if is_new:
            log(
                f"  [cache] +1 decoder for {video_path}  "
                f"(cache_size={self.size()}, rss={current_rss_mb():.0f}MB)"
            )
        return decoder

    VideoDecoderCache.get_decoder = logged_get_decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--start-step", type=int, default=0, help="cosmetic only, see module docstring")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    patch_decoder_cache_logging()

    # batch_size=64 matches the --batch_size=64 CLI flag in train_b1k.sh /
    # train_b1k.sbatch.sh -- pi05_b1k itself doesn't set it, so left alone
    # this would silently fall back to the base TrainConfig default of 32
    # and no longer reproduce the same batch boundaries as the real job.
    config = dataclasses.replace(_config.get_config("pi05_b1k"), num_workers=0, batch_size=64)

    log(f"Building data loader: seed={config.seed} batch_size={config.batch_size} num_workers=0")
    t0 = time.time()
    loader = _data_loader.create_b1k_data_loader(config, shuffle=True)
    log(f"Data loader built in {time.time() - t0:.1f}s")

    rss0 = current_rss_mb()
    log(f"Baseline RSS after loader init: {rss0:.0f}MB")

    it = iter(loader)
    for i in range(args.max_steps):
        step = args.start_step + i
        t0 = time.time()
        try:
            next(it)
        except Exception as e:
            log(f"*** step {step}: EXCEPTION after {i} batches: {e!r}")
            raise
        dt = time.time() - t0
        if i % args.log_every == 0:
            rss = current_rss_mb()
            log(f"step {step:6d}  batch_time={dt:6.2f}s  rss={rss:8.0f}MB  delta_from_start={rss - rss0:+8.0f}MB")

    log(f"Completed {args.max_steps} batches without crashing. Final RSS: {current_rss_mb():.0f}MB")


if __name__ == "__main__":
    main()
