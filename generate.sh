#!/bin/sh

#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --job-name=Generate
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err
#SBATCH --mem=8G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

source /users/s/a/sahahn/.bashrc
cd ${SLURM_SUBMIT_DIR}
python generate.py $1 $2 $3