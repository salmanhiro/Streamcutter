#!/bin/bash
#SBATCH --partition mem
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 196G
#SBATCH --job-name streamcutter
#SBATCH --output /data/salmanhiro/streamcutter-%J.log
#SBATCH --time 10-0

module purge
module load slurm
module load slurm_limit_threads
source activate streamcutter

srun python get_tractor_footprint.py --sim-file simulated_streams/simulated_stream_Pal_12.fits --min-stars 100 --concat --outdir /data/salmanhiro/tractor --target-radius 1.0 --env-radius 2.0