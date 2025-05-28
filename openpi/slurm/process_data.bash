#!/bin/bash

# Run this script with: sbatch -J thermal_mixer process_data.bash

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 16
#SBATCH --mem 64G
#SBATCH --time 4:00:00
#SBATCH --comment CLI
#SBATCH --output slurm/logs/stdout.%j
#SBATCH --error slurm/logs/stderr.%j

DATA_NAME=$SLURM_JOB_NAME
LEROBOT_HOME=$PWD uv run scripts/convert.py --data_dir raw/$DATA_NAME --repo_id data/$DATA_NAME
LEROBOT_HOME=$PWD JAX_PLATFORMS=cpu uv run python scripts/compute_norm_stats.py --config-name $DATA_NAME
ln -sf $PWD/assets/$DATA_NAME $PWD/assets/$DATA_NAME-lora
