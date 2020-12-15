#!/bin/sh
#SBATCH --job-name=
#SBATCH --ntasks=1
#SBATCH --time=
#SBATCH --output=taes-job-%j.io
#SBATCH --gres=gpu:
#SBATCH --mem=32GB
#SBATCH --partition=
echo "working Directory" = $SLURM_SUBMIT_DIR
nvidia-docker run --rm -v WORKING_DIRECTORY:/home -w /home ENVIRONMENT python PATH_FOR_MAIN > $SLURM_JOB_ID.output
