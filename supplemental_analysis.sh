#!/bin/sh

#SBATCH --partition=bluemoon
#SBATCH --time=30:00:00
#SBATCH --job-name=Supp
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err
#SBATCH --mem=32G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

source /users/s/a/sahahn/.bashrc
cd ${SLURM_SUBMIT_DIR}
python supplemental_analysis.py