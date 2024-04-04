#!/bin/bash
#SBATCH -J get_labels          # job name to display in squeue
#SBATCH -o get_labels.txt       # standard output file
#SBATCH -p standard-s    # requested partition
#SBATCH --exclusive      # do not share nodes
#SBATCH --mem 256GB
#SBATCH --mail-user esliheet@smu.edu
#SBATCH --mail-type=all

# load any modules you need and/or activate conda environments

# run python script
module load intel/2023.1
module load mpi
python get_labels.py