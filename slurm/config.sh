# ---------------------------------------------------------------------------
# Arteries build: Picotte configuration (the only file you should need to edit).
# Following picotte_slurm_guide.md: free-tier partition def-sm, account
# lebowfreeprj, no array throttle, logs folder created before sbatch,
# static manifest (countries.txt) instead of directory listings.
# ---------------------------------------------------------------------------
export ACCOUNT="lebowfreeprj"
export PARTITION="def-sm"                 # free tier; do NOT use def (paid)
export MAIL_USER="bg654@drexel.edu"

# Resource requests: as large as a def-sm node allows (48 cores, all node memory).
# Wall time: def-sm maximum is 12 h (sinfo); every stage is expected to need minutes.
# To see the partition maximum:  sinfo -p def-sm -o "%P %l %c %m"
export CPUS=24                            # def-sm: MaxCPUsPerNode=24
export MEM="96000M"                       # def-sm: MaxMemPerNode=96000 (MB)
export WALL="12:00:00"

# Layout under /home/bg654/logistic_package (64 GiB home quota; build needs ~15 GB)
export ROOT="/home/bg654/logistic_package"
export PROJECT_DIR="$ROOT/arteries"                       # this repository
export ROADS_DIR="$ROOT/data/initial_graphs"              # *_roads.pkl.gz (+ optional *_links.pkl.gz)
export CITIES_CSV="$ROOT/data/cities/cities_merged.csv"
export PORTS_CSV="$ROOT/data/ports/PortWatch_ports_database.csv"
export WORK_DIR="$ROOT/work"                              # intermediates + release
export LOG_DIR="$WORK_DIR/slurm_logs"
export DATA_VERSION="0.1.0"

# Python: 00_setup.sh installs Miniforge into $ROOT/miniforge3 if no Python >= 3.10 module exists
export CONDA_DIR="$ROOT/miniforge3"
export VENV="$ROOT/venv_arteries"

activate_env() {
  module purge 2>/dev/null || true
  module load gcc/9.2.0 2>/dev/null || true          # python/gcc/* modules require gcc/9.2.0 first
  if [ -f "$ROOT/.python_module" ]; then module load $(cat "$ROOT/.python_module"); fi
  source "$VENV/bin/activate"
  export NUMBA_CACHE_DIR="$WORK_DIR/.numba_cache"
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # we parallelize with processes
  export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
  export TMPDIR="${TMPDIR:-/tmp}"
}
