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
source activate INN

# Run script
python lrp.py /home/sbradshaw/ci_ltg/plot_list_LRP.config /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210705_MaxRef_bny/fit_conv_model.h5 -mc /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210705_MaxRef_bny/model_config.pkl -p 20200603_feature1_B2 -o /apollo/grain/saved_ci_ltg_data/MSthesis
#python lrp.py /home/sbradshaw/ci_ltg/plot_list_LRP.config /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210705_MaxRef_bny/fit_conv_model.h5 -mc /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210705_MaxRef_bny/model_config.pkl -p 20200603_feature1
#python lrp_john.py /home/sbradshaw/ci_ltg/plot_list_LRP.config /home/sbradshaw/probsevere_oconus/src/static/fit_conv_model.h5 -mc /home/sbradshaw/probsevere_oconus/src/static/model_config.pkl -d

#python train_eval_cnn.py -f -v -o /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210807_MaxRef_C02C13 -s -sn 9
#python train_eval_cnn.py -v -o /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210807_MaxRef_C02C13 -s -sn 9 -m /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210807_MaxRef_C02C13/model-06-0.048357.h5
#python train_eval_cnn.py -f -v -o /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210807_MaxRef_C02C13 -s -sn 9 -m /apollo/grain/sbradshaw/AI/CONUS/DeepLearningModels/20210807_MaxRef_C02C13/model-04-0.048309.h5
#echo hello_world