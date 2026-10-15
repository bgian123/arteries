#!/bin/bash
#SBATCH --job-name=art-assemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Single job: assemble (docs/METHODS.md).
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
python -m arteries.build assemble --work "$WORK_DIR"
