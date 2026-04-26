#!/bin/bash
#SBATCH --job-name=resnet56
#SBATCH --account=bgvu-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:59:59
#SBATCH --output=output/linear_decrease1_%j.out
#SBATCH --error=output/linear_decrease1_%j.err
# Load PyTorch (do NOT module purge, the default modules include CUDA)
module load pytorch-conda/2.8
# Run training
python main.py