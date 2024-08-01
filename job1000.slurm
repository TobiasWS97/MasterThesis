#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=nv-ikj
#SBATCH --job-name=100_job
#SBATCH --time=5-01:15:00     # 5 days 1 hour and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --gres=gpu:8          # 
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=tobiaws@stud.ntnu.no
#SBATCH --mail-type=ALL


module purge

module load GROMACS/2023.1-foss-2022a-CUDA-11.7.0

module load Python/3.11.3-GCCcore-12.3.0

python IdunJob100.py