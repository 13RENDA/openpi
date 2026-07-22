#!/bin/bash
#SBATCH --job-name=b1k_norm_stats
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=/vision/u/shiyuc/openpi/outputs/viscam/norm_stats_%j.log
#SBATCH --error=/vision/u/shiyuc/openpi/outputs/viscam/norm_stats_%j.log

set -euo pipefail

# Code lives on /viscam, all results (logs, caches, norm stats) go to /vision/u/shiyuc.
CODE_DIR=/viscam/u/shiyuc/openpi
RESULT_DIR=/vision/u/shiyuc/openpi

cd "$CODE_DIR"
mkdir -p "$RESULT_DIR/outputs/viscam" "$RESULT_DIR/outputs/assets" "$RESULT_DIR/hf_cache"
source .venv/bin/activate

CONFIG_NAME=${1:-pi05_b1k}
if [ $# -gt 0 ]; then
    shift
fi

export HF_DATASETS_CACHE=$RESULT_DIR/hf_cache/datasets
export HF_HOME=$RESULT_DIR/hf_cache/huggingface
export OPENPI_DATA_HOME=$RESULT_DIR/openpi_cache
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Started: $(date)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "Extra args: $*"

python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME" "$@"

echo "Finished: $(date)"
