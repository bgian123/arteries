#!/bin/bash
# Run ONCE on the login node:  bash 00_setup.sh
# Creates the Python environment, installs arteries, writes the build config and the
# static country manifest, and creates the log folder.
set -eo pipefail
cd "$(dirname "$0")"
source ./config.sh

# 1) find a Python >= 3.10 (module first, else Miniforge in $ROOT/miniforge3)
PY=""
if command -v module >/dev/null 2>&1; then
  module load gcc/9.2.0 2>/dev/null || true
  for m in $(module -t avail python 2>&1 | grep -iE '^python[^ ]*3\.(1[0-9])' | sort -V -r); do
    if module load "$m" 2>/dev/null && python3 -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
      PY=$(command -v python3); echo "$m" > "$ROOT/.python_module"; echo "using module $m"; break
    fi
    module purge 2>/dev/null || true
  done
fi
if [ -z "$PY" ]; then
  rm -f "$ROOT/.python_module"
  if [ ! -x "$CONDA_DIR/bin/python3" ]; then
    echo "no Python>=3.10 module found; installing Miniforge into $CONDA_DIR"
    curl -fsSL -o "$ROOT/miniforge.sh" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$ROOT/miniforge.sh" -b -p "$CONDA_DIR"
    rm -f "$ROOT/miniforge.sh"
  fi
  PY="$CONDA_DIR/bin/python3"
fi
"$PY" --version

# 2) virtual environment + package
"$PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -e "$PROJECT_DIR[build,plot]" pytest

# 3) unit tests (fast, ~10 s)
(cd "$PROJECT_DIR" && python -m pytest -q tests)

# 4) build config, static manifest, log folder
mkdir -p "$WORK_DIR" "$LOG_DIR"
python -m arteries.build init --work "$WORK_DIR" --roads-dir "$ROADS_DIR" \
  --cities "$CITIES_CSV" --ports "$PORTS_CSV"
python -m arteries.build list-countries --work "$WORK_DIR" > "$WORK_DIR/countries.txt"
echo "countries in manifest: $(wc -l < "$WORK_DIR/countries.txt")  (expected 236)"
echo "setup complete"
