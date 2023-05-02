#!/bin/bash

#SBATCH --partition=salvador
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/home/%u/output/test_day_%j.log
#SBATCH --cpus-per-task=6
#SBATCH --job-name=test
##SBATCH --mem=4500
##SBATCH -w r740-105-17

nvidia-smi
hostname
squeue -q `hostname -s`
echo $CUDA_VISIBLE_DEVICES
echo "----"
group=$(dcgmi group -c allgpus --default)
if [ $? -eq 0 ]; then
groupid=$(echo $group | awk '{print $10}')
dcgmi stats -g $groupid -e
dcgmi stats -g $groupid -s $SLURM_JOB_ID
fi

# Set environment
#point at container
CONTAINER=/home/shared/containers/pytorch_22.04-py3.sif

source /etc/profile

# Run script
srun singularity run -B /ships19 -B /apollo -B $HOME/local-pytorch:$HOME/.local --nv $CONTAINER python torchlightning_main.py