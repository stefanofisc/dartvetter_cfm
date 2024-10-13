#!/bin/bash
#SBATCH -o /path/to/dartvetter_cfm/output_files/today_modelname-datasetname_details_job-%J.out
#SBATCH --nodes=1
#SBATCH --ntasks=1 --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=xgpu
#SBATCH --nodelist=gnode01
#SBATCH --gres=gpu:tesla:1
#SBATCH --mem=44000
#SBATCH --job-name="jobname"

source /path/to/your/anaconda3/etc/profile.d/conda.sh

conda activate your_environment_name

python3 /path/to/train.py --config train_config.yaml
