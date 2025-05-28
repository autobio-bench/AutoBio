#!/bin/bash

# Run this script with: sbatch -J thermal_mixer train.bash

#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00
#SBATCH --comment CLI
#SBATCH --output slurm/logs/stdout.%j
#SBATCH --error slurm/logs/stderr.%j

CONFIG_NAME=$SLURM_JOB_NAME
EXTRA_ARGS="$@"
XLA_PYTHON_CLIENT_MEM_FRACTION=.95 CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY= LEROBOT_HOME=$PWD \
  ~/.local/bin/uv run --no-sync --no-cache python scripts/train.py $CONFIG_NAME --exp-name PLACEHOLDER $EXTRA_ARGS
