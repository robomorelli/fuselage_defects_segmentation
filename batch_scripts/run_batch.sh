#!/bin/bash
#PBS -N fus_defects
#PBS -o fus_defects.txt
#PBS -q gpu
#PBS -e fus_defects.txt
#PBS -k oe
#PBS -m e
#PBS -M roberto.morelli.ext@leonardocompany.com
#PBS -l select=1:ngpus=1:ncpus=24,walltime=72:00:00

echo "start training"
echo "config_name: $config_name"
echo "fold: $fold"

source $HOME/.bashrc

cd /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/

conda activate flai
python /davinci-1/home/morellir/artificial_intelligence/repos/fuselage_defects_segmentation/main_train.py #--config_name "$config_name" --fold "$fold"
 
