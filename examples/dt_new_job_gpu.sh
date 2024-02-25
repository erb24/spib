#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --job-name=hp35
#SBATCH -N 1
#SBATCH --ntasks=1
#SBTACH -n 1
#SBATCH --gpus=a100:1
#SBATCH -p gpu

source /scratch/zt1/project/tiwary-prj/user/ebeyerle/anaconda3/etc/profile.d/conda.sh

conda activate msmbuilder #/scratch/zt1/project/tiwary-prj/user/dwang97/.conda/envs/spib_msm

export OMP_NUM_THREADS=1

#python run_spib_msm_cv.py

python dt_run_spib_cv.py
