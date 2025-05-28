#!/bin/bash

# Run this script with: sbatch -J thermal_mixer train.bash

#SBATCH --partition gpu
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --mem 64G
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00
#SBATCH --comment CLI
#SBATCH --output logs/stdout.%j
#SBATCH --error logs/stderr.%j

# LEROBOT_NUM_EPISODES=20
EXP_NAME=$SLURM_JOB_NAME-32-$SLURM_JOB_ID

LEROBOT_TASK=$SLURM_JOB_NAME LEROBOT_MULTI_TASKS=thermal_cycler_close,thermal_cycler_open LEROBOT_ROOT="$PWD/../openpi" EXP_NAME=$EXP_NAME CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY= \
    ~/miniforge3/condabin/conda run -n rdt --no-capture-output --live-stream bash finetune.sh
