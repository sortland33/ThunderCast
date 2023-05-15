#!/bin/bash

## you need permission to submit jobs to the salvador partition.
##SBATCH --partition=salvador
#SBATCH --gpus=0
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/output/sb_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --job-name=preds
##SBATCH -w r740-105-17

#NOTE - cpu-per-gpu and mem-per-gpu defaults are 16G of ram per GPU and 6 cores.
# these are reasonable defaults, but can be changed with --cpus-per-gpu and --mem-per-gpu

#output directory must exist or the job will silently fail

#source /etc/profile

# Set environment
#module load miniconda/3.7-base
#source activate deep_learning_env

#point at container
#CONTAINER=/home/shared/containers/pytorch_22.04-py3.sif
#CONTAINER=/home/sortland/ci_ltg/pytorch_22.04-py3.sif
#source /etc/profile
source activate tiny_torch
# Run script
#singularity run -B /ships19 -B $HOME/local-pytorch:$HOME/.local --nv $CONTAINER python torch_train.py
#python torchlightning_predict.py -DT 2021-08-23-00-01 -t 2 -lat 45.02 -lon -101.38
python torchlightning_predict.py -DT 2021-08-12-21-31 -t 2 -lat 32.62 -lon -83.27 -hs 255 -c C13 -f
#python train_eval_cnn.py -f -v -o /ships19/grain/convective_init/models/delete -s -sn 12 #-m /ships19/grain/convective_init/models/zarr_test_model/model-10-0.063190.h5
#python train_eval_cnn.py -v -o /ships19/grain/convective_init/models/zarr_test_model -s -sn 12 -m /ships19/grain/convective_init/models/zarr_test_model/fit_conv_model.h5
#python train_eval_cnn.py -f -v -o /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210807_MaxRef_C02C13 -s -sn 9 -m /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210807_MaxRef_C02C13/model-04-0.048309.h5
#echo hello_world