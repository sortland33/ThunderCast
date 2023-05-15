#!/bin/bash

#SBATCH --job-name=my-job
#SBATCH --partition=salvador
#SBATCH --time=1:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/%u/output/sb_%j.log

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
# Note- you will need to have an environment with tbparse for this. tbparse is not included in the main environment.yml file
# because it is not able to be installed with conda. Recommended creating a separate environment for tbparse use.

srun singularity run -B /ships19 -B /apollo -B $HOME/local-tbparse:$HOME/.local --nv $CONTAINER python extract_TB_data.py