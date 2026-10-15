#!/bin/bash
#SBATCH --job-name=art-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Single job: merge (docs/METHODS.md).
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
python -m arteries.build merge --work "$WORK_DIR"
