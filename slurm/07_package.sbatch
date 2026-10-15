#!/bin/bash
#SBATCH --job-name=art-package
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Assemble the land matrix, run checks, write release files and the validation report.
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
python -m arteries.build package --work "$WORK_DIR" --version "$DATA_VERSION"
python "$PROJECT_DIR/scripts/validate_release.py" "$WORK_DIR" "$DATA_VERSION" | tee "$WORK_DIR/validation_report.txt"
