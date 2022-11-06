#!/bin/bash
#SBATCH -J cudacheck
#SBATCH -w gpuc1
#SBATCH -o ./checkcuda_out
#SBATCH -e ./checkcuda_err
#SBATCH -p gpu
#SBATCH --gres=gpu:2

nvidia-smi
nvcc -V
python /home/yupei/workspaces/check/checkcuda_tf.py

