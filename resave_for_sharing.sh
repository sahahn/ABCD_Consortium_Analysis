#!/bin/sh

#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --job-name=Resave_for_sharing
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err
#SBATCH --mem=4G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

source /users/s/a/sahahn/.bashrc

cd ${SLURM_SUBMIT_DIR}
python resave_for_sharing.py