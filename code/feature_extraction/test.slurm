#!/bin/bash
#SBATCH -o /home/s.fiscale/conda/Models/forest_diffusion/output_files/2024-10-11_ForestFlowModel-dr24_test_de_job-%J.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=xgpu
#SBATCH --nodelist=gnode01
#SBATCH --gres=gpu:1
#SBATCH --mem=44000
#SBATCH --job-name="dr24-de"

source /home/s.fiscale/anaconda3/etc/profile.d/conda.sh

conda activate torchcfm

python3 /home/s.fiscale/conda/Models/forest_diffusion/code/test.py --config test_config.yaml