#!/bin/bash 
#SBATCH -n 16
#SBATCH --mem-per-cpu=4G 
#SBATCH --gpus=1 
#SBATCH --job-name=imu_training
#SBATCH --time=28:00:00 
echo "1"
conda activate imuposer 
echo "2"
export PYTHONPATH="${PYTHONPATH}:/cluster/scratch/xintliu/IMUPoser" 
python scripts/1.\ Preprocessing/preprocess_dmpl.py 
