#!/bin/bash
#SBATCH --job-name=art-simplify
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Single job: simplify (docs/METHODS.md).
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
python -m arteries.build simplify --work "$WORK_DIR"
