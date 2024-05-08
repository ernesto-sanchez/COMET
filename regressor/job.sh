#!/bin/bash
#SBATCH --job-name=reg_1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --output=output_%J.txt

 
source ~/.bashrc
enable_modules

source /cluster/work/medinfmk/STCS_swiss_transplant/ern_env/bin/activate
 
python /cluster/work/medinfmk/STCS_swiss_transplant/AI_Organ_Transplant_Matching/code/code_ernesto/comet_cluster/regressor/reg_sklearn.py



