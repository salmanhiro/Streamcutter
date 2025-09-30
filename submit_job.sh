#!/bin/bash
#SBATCH --partition mem
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --job-name streamcutter
#SBATCH --output /data/salmanhiro/streamcutter-%J.log
#SBATCH --time 1-0

module purge
module load slurm_limit_threads
source activate streamcutter

srun ./run-streamcutter.sh
