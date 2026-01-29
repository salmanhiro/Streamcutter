#!/bin/bash
#SBATCH --partition mem
#SBATCH --nodelist m03 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 16GB
#SBATCH --job-name streamcutter
#SBATCH --output /data/salmanhiro/streamcutter-%J.log
#SBATCH --time 10-0

module purge
module load slurm
module load slurm_limit_threads
source activate streamcutter

srun python simulate_stream.py --gc NGC_5053 --n-orbits 100