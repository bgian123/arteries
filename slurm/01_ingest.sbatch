#!/bin/bash
#SBATCH --job-name=art-ingest
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Array task i (1-based) ingests line i of the static manifest countries.txt.
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
LINE=$(( ${OFFSET:-0} + SLURM_ARRAY_TASK_ID ))
ISO=$(sed -n "${LINE}p" "$WORK_DIR/countries.txt")
echo "array task $SLURM_ARRAY_TASK_ID -> manifest line $LINE -> $ISO on $(hostname)"
python -m arteries.build ingest --work "$WORK_DIR" --iso "$ISO"
