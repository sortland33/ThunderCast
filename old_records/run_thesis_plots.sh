#!/bin/bash

## you need permission to submit jobs to the salvador partition.
##SBATCH --partition=salvador
#SBATCH --gpus=0
#SBATCH --time=6:00:00
#SBATCH --output=/home/%u/output/sb_%j.log
#SBATCH --cpus-per-task=2

#NOTE - cpu-per-gpu and mem-per-gpu defaults are 16G of ram per GPU and 6 cores.
# these are reasonable defaults, but can be changed with --cpus-per-gpu and --mem-per-gpu

#output directory must exist or the job will silently fail

#source /etc/profile

# Set environment
#module load miniconda/3.7-base
source activate TF_env_v2

# Run script
python thesis_plots.py -m /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210705_MaxRef_bny/fit_conv_model.h5 -mconfig /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210705_MaxRef_bny/model_config.pkl -mversion 20210705_MaxRef_bny -preds -DTfile thesis_plot_DT_list.config -l /apollo/grain/sbradshaw/CopiedData_thesis/ -mm -z -fn band_elimination.png