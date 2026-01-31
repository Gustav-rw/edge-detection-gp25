#!/bin/bash
#SBATCH --job-name=gpu_opt
#SBATCH --account=project_2016196
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=output.txt

srun gpu_opt 1920 4320