#!/bin/bash
#SBATCH -o /home/s.fiscale/conda/Models/forest_diffusion/output_files/2024-10-11_forest-dr24_train_job-%J.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=xgpu
#SBATCH --nodelist=gnode01
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000
#SBATCH --job-name="dv-k24"

source /home/s.fiscale/anaconda3/etc/profile.d/conda.sh

conda activate torchcfm

python3 /home/s.fiscale/conda/Models/forest_diffusion/code/feature_extraction/train.py --config train_config.yaml