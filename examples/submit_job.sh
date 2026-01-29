#!/bin/bash
#SBATCH --partition mem
#SBATCH --nodelist m03 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 192GB
#SBATCH --job-name streamcutter
#SBATCH --output /data/salmanhiro/streamcutter-%J.log
#SBATCH --time 10-0

module purge
module load slurm
module load slurm_limit_threads
source activate streamcutter

srun python get_tractor_footprint.py --sim-file simulated_streams/simulated_stream_NGC_5053.fits --min-stars 10 --concat --outdir /scratch/u/salmanhiro/tractor --target-radius 2.0 --env-radius 5.0