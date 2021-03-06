#!/bin/sh

#SBATCH --partition=short
#SBATCH --time=3:00:00
#SBATCH --job-name=Plot_Collage
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err
#SBATCH --mem=30G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

source /users/s/a/sahahn/.bashrc
cd ${SLURM_SUBMIT_DIR}
python plot_collage.py