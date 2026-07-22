#!/bin/bash
#SBATCH --job-name="b1k_openpi_train"
#SBATCH --account=viscam
#SBATCH --partition=viscam
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=32
#SBATCH --time=5-00:00:00
#SBATCH --output=/vision/u/shiyuc/openpi/outputs/viscam/openpi_%j.log
#SBATCH --error=/vision/u/shiyuc/openpi/outputs/viscam/openpi_%j.log

set -euo pipefail

# Code lives on /viscam; logs, caches, checkpoints and norm stats live on /vision/u/shiyuc.
# The pi05_b1k config must point assets_base_dir/checkpoint_base_dir at $RESULT_DIR/outputs
# (they are also passed explicitly below).
# NOTE: the SLURM log directory above must exist BEFORE submitting the job:
#   mkdir -p /vision/u/shiyuc/openpi/outputs/viscam
CODE_DIR=/viscam/u/shiyuc/openpi
RESULT_DIR=/vision/u/shiyuc/openpi

cd "$CODE_DIR"
mkdir -p "$RESULT_DIR/outputs/viscam" "$RESULT_DIR/outputs/checkpoints" "$RESULT_DIR/hf_cache" "$RESULT_DIR/wandb"
source .venv/bin/activate

CONFIG_NAME=${1:-pi05_b1k}
if [ $# -gt 0 ]; then
    shift
fi

EXP_NAME=${EXP_NAME:-${CONFIG_NAME}_full}

export HF_DATASETS_CACHE=$RESULT_DIR/hf_cache/datasets
export HF_HOME=$RESULT_DIR/hf_cache/huggingface
export OPENPI_DATA_HOME=$RESULT_DIR/openpi_cache
export WANDB_DIR=$RESULT_DIR/wandb
export PYTHONUNBUFFERED=1

# Fail fast if the precomputed norm stats are missing (training would otherwise
# error out only after the dataset is set up).
NORM_STATS_FILE=$RESULT_DIR/outputs/assets/$CONFIG_NAME/2026-challenge-demos/norm_stats.json
if [ ! -f "$NORM_STATS_FILE" ]; then
    echo "ERROR: norm stats not found at $NORM_STATS_FILE"
    echo "Run scripts/b1k/compute_norm_stats.sbatch_new.sh first."
    exit 1
fi

# Calculate total GPUs across all nodes
NUM_GPUS=$((${SLURM_GPUS_ON_NODE:-1} * ${SLURM_NNODES:-1}))

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NAME=$SLURM_JOB_NAME"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_CPU_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE"
echo "Number of nodes: ${SLURM_NNODES:-1}"
echo "GPUs per node: ${SLURM_GPUS_ON_NODE:-1}"
echo "Total GPUs: $NUM_GPUS"
echo "Current time: $(date)"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "EXP_NAME=$EXP_NAME"
echo "Norm stats: $NORM_STATS_FILE"
echo "Running with args: $@"

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/b1k/train_b1k.py "$CONFIG_NAME" \
    --exp_name="$EXP_NAME" \
    --resume \
    --batch_size=64 \
    --num_train_steps=100000 \
    --assets_base_dir="$RESULT_DIR/outputs/assets" \
    --checkpoint_base_dir="$RESULT_DIR/outputs/checkpoints" \
    --num_workers=8 \
    "$@"

echo "Job finished: $(date)"
