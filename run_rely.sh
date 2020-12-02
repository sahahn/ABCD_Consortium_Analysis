#!/bin/sh

#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --job-name=Run_Rely
#SBATCH --output=Job_Logs/%x_%j.out
#SBATCH --error=Job_Logs/%x_%j.err
#SBATCH --mem=16G

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-100

source /users/s/a/sahahn/.bashrc
cd ${SLURM_SUBMIT_DIR}
python run_rely.py $1 $2 $3 $SLURM_ARRAY_TASK_ID