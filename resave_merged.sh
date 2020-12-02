#!/bin/sh

#SBATCH --partition=bluemoon
#SBATCH --time=01:00:00
#SBATCH --job-name=Resave
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --array=1-50

source /users/s/a/sahahn/.bashrc

cd ${SLURM_SUBMIT_DIR}
python resave_merged.py