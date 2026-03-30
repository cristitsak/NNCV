#!/bin/bash
#SBATCH --job-name=segformer_b0_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

srun apptainer exec --nv --env-file .env container_v2.sif /bin/bash main.sh