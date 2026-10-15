#!/bin/bash
#SBATCH --job-name=art-landmatrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Array task i (1-based) computes Dijkstra for terminal chunk i-1 (64 sources each),
# parallel over all cores. Tasks past the last terminal exit immediately.
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
python -m arteries.build landmatrix --work "$WORK_DIR" --task-id $(( SLURM_ARRAY_TASK_ID - 1 ))
