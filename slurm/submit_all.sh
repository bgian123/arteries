#!/bin/bash
# Submit the full pipeline with dependencies (run on the login node):
#   bash submit_all.sh            # full build
#   bash submit_all.sh smoke      # only the 4-country smoke test
#   bash submit_all.sh from-terminals   # rerun stages 3-7 only (reuses ingest + merge)
set -eo pipefail
cd "$(dirname "$0")"
source ./config.sh
mkdir -p "$LOG_DIR"                      # must exist before sbatch (guide: gotcha #1)

sb() {  # common options; job name comes from the script
  local name=$1; shift
  local pat="%j"
  case " $* " in *" --array="*) pat="%A_%a";; esac
  sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --time="$WALL" \
    --cpus-per-task="$CPUS" --mem="$MEM" \
    --output="$LOG_DIR/${name}_${pat}.out" --error="$LOG_DIR/${name}_${pat}.err" \
    --mail-user="$MAIL_USER" --mail-type=FAIL \
    --export=ALL,PROJECT_DIR="$PROJECT_DIR",OFFSET=0 "$@"
}

if [ "${1:-}" = "smoke" ]; then
  j=$(sb smoke --mail-type=END,FAIL 99_smoke_test.sbatch)
  echo "smoke test submitted: $j   (log: $LOG_DIR/smoke_${j}.out)"
  exit 0
fi

N=$(wc -l < "$WORK_DIR/countries.txt")
[ "$N" -gt 0 ] || { echo "empty manifest; run 00_setup.sh first"; exit 1; }
if [ "${1:-}" = "from-terminals" ]; then   # reuse ingest + merge outputs, redo stages 3-7
  j1=skipped; j2=skipped
  j3=$(sb terminals                                                03_terminals.sbatch)
else
j1=$(sb ingest     --array=1-"$N"                                 01_ingest.sbatch)
j2=$(sb merge      --dependency=afterok:"$j1"                      02_merge.sbatch)
j3=$(sb terminals  --dependency=afterok:"$j2"                      03_terminals.sbatch)
fi
j4=$(sb simplify   --dependency=afterok:"$j3"                      04_simplify.sbatch)
j5=$(sb assemble   --dependency=afterok:"$j4"                      05_assemble.sbatch)
# < 4,500 terminals -> at most 71 chunks of 64; 80 tasks is a safe upper bound
j6=$(sb landmatrix --dependency=afterok:"$j5" --array=1-80         06_landmatrix.sbatch)
j7=$(sb package    --dependency=afterok:"$j6" --mail-type=END,FAIL 07_package.sbatch)
cat <<MSG
submitted: ingest=$j1 (array 1-$N) merge=$j2 terminals=$j3 simplify=$j4 assemble=$j5 landmatrix=$j6 package=$j7
monitor:   squeue -u $USER
logs:      $LOG_DIR   and   $WORK_DIR/logs   (per-stage stats in $WORK_DIR/stats)
MSG
