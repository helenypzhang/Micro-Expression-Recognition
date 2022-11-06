#!/bin/bash
#SBATCH -J MER_SAMM
#SBATCH -w gpuc1
#SBATCH -o ./run_out
#SBATCH -e ./run_err
#SBATCH -p gpu
#SBATCH --gres=gpu:1

python /home/yupei/workspaces/MER/MER_newstart_SAMM/Main.py

