#!/bin/bash
#SBATCH --job-name=art-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Whole pipeline on 4 small countries (LUX, BEL, NLD, DNK) in a separate work dir.
# Run this first; it takes a few minutes and exercises every stage + validation.
set -eo pipefail
source "$PROJECT_DIR/slurm/config.sh"
activate_env
SMOKE="$ROOT/work_smoke"; rm -rf "$SMOKE"; mkdir -p "$SMOKE/roads"
for c in lux bel nld dnk; do ln -s "$ROADS_DIR/${c}_roads.pkl.gz" "$SMOKE/roads/"; done
python -m arteries.build init --work "$SMOKE" --roads-dir "$SMOKE/roads" --cities "$CITIES_CSV" --ports "$PORTS_CSV"
python -m arteries.build all --work "$SMOKE" --set terminal_iso3_filter=LUX,BEL,NLD,DNK
python "$PROJECT_DIR/scripts/validate_release.py" "$SMOKE" "$DATA_VERSION" | tee "$SMOKE/validation_report.txt"
echo "SMOKE TEST OK"
