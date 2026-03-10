#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=04:00:00  # More time for training

# Run your actual training python code
apptainer exec --nv container.sif \
    python train.py --data_dir ./data