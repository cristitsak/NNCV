#!/bin/bash
#SBATCH --job-name=segformer_b3_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

# Fixes memory fragmentation issues on A100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun apptainer exec --nv --env-file .env container_v2.sif /bin/bash main.sh