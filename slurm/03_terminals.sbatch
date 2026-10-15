#!/bin/bash
#SBATCH --job-name=art-terminals
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Single job: terminals (docs/METHODS.md).
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
python -m arteries.build terminals --work "$WORK_DIR"
