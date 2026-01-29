#!/bin/bash

CSV="data/streamfinder_baumgardt_map.csv"
JOB_DIR="submitted_jobs"
mkdir -p "$JOB_DIR"  # Create directory if it doesn't exist

# Read GC names (skip header)
tail -n +2 "$CSV" | while IFS=',' read -r CLUSTER IN_STREAMFINDER SF_NAME
do
  if [[ "$IN_STREAMFINDER" == "TRUE" ]]; then
    JOB_NAME="job_${CLUSTER}.sh"
    JOB_PATH="${JOB_DIR}/${JOB_NAME}"
    LOG_FILE="/data/salmanhiro/streamcutter-${CLUSTER}.log"

    # Write SLURM job script
    cat <<EOF > "$JOB_PATH"
#!/bin/bash
#SBATCH --partition mem
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 64G
#SBATCH --job-name stream_${CLUSTER}
#SBATCH --output ${LOG_FILE}
#SBATCH --time 1-0

module purge
module load slurm_limit_threads
source activate streamcutter

python select_stream_candidates.py --gc ${CLUSTER}
EOF

    # Submit it
    sbatch "$JOB_PATH"
    echo "[v] Submitted job for $CLUSTER -> $JOB_PATH"
  fi
done
