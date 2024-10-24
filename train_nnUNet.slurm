#!/bin/bash
#SBATCH -N 1 
#SBATCH -C gpu  
#SBATCH -G 4 
#SBATCH -q debug  
#SBATCH -J train-nnunet
#SBATCH --mail-type=ALL  
#SBATCH --time=00:30:00  
#SBATCH -A m4055 
#SBATCH --error=slurm_error.txt
#SBATCH --output=slurm_output.txt

# Change directory to where the SLURM script is located
cd "$(dirname "$0")"

echo "ooohosodf"

rrff