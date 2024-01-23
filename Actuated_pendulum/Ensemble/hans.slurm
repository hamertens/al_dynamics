#!/bin/bash
# See https://slurm.schedmd.com/job_array.html

#SBATCH --partition=gpu # gpu, kill-shared, shared

#SBATCH --gres=gpu:1 #1 GPU
#SBATCH --cpus-per-task=1 # Number of cores; probably don't need to change this
#SBATCH --mem=12gb ## max amount of memory per node you require in gb
#SBATCH --time=01-22:00:00 ## time format is DD-HH:MM:SS, 3day max on kill-shared

#SBATCH --job-name=actuated_pendulum_ensemble #Not necessary but any name
#SBATCH --output=./slurm-output-%A.out #Output file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hansm@hawaii.edu

# Load python profile, then call python script
source ~/auto.profile # Bash script that initializes Anaconda

#conda activate environment_name #active the environment

# If you want to run an ipython notebook:
# jupyter nbconvert ipynb_file.ipynb --to python
# python ipynb_file.py

# If you want to run a regular python script:
# python script_name.py

# If you want to run a bash script:

# Read the file paths from the temporary file into an array
mapfile -t filepaths < filepaths.txt

# Execute the bash script
source ${filepaths[0]}