"""Shared helpers for the build pipeline: paths, logging, config, npz I/O."""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BuildConfig:
    roads_dir: str
    cities_csv: str
    ports_csv: str
    work_dir: str
    # --- cities -----------------------------------------------------------
    city_min_population: float = 100_000
    city_max_year: int = 2026          # "most recent" = latest observation with year <= this
    # --- snapping / connectors (km) ----------------------------------------
    city_snap_km: float = 25.0         # city -> nearest road node; beyond: city is its own island
    port_snap_km: float = 80.4672      # 50 miles, as in the previous pipeline
    insert_crossings: bool = True      # add junctions where segments cross (link ramps absent)
    crossing_cell_deg: float = 0.005
    bridge_km: float = 2.0             # dead-end -> other component (data-gap repair)
    bridge_rounds: int = 3
    # --- ports ------------------------------------------------------------
    exclude_offshore_ports: bool = True
    terminal_iso3_filter: str = ""     # comma-separated ISO3 list (testing); empty = all countries
    # --- land distance matrix ---------------------------------------------
    sssp_chunk: int = 64               # sources per array task
    workers: int = 0                   # 0 -> all cores visible to the job

    @property
    def work(self) -> Path:
        return Path(self.work_dir)

    def save(self):
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "config.json").write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, work_dir) -> "BuildConfig":
        d = json.loads((Path(work_dir) / "config.json").read_text())
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class Log:
    """Timestamped logger that also appends to a per-stage log file."""

    def __init__(self, path: Path | None = None):
        self.t0 = time.time()
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        el = time.time() - self.t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_gb = rss / (1024**3 if sys.platform == "darwin" else 1024**2)
        line = f"[{time.strftime('%H:%M:%S')} +{el:8.1f}s peakRSS={rss_gb:6.2f}GB] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as fh:
                fh.write(line + "\n")


def env_info() -> dict:
    return {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def n_workers(cfg: BuildConfig) -> int:
    if cfg.workers:
        return cfg.workers
    s = os.environ.get("SLURM_CPUS_PER_TASK")
    if s:
        return int(s)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count() or 1


def save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)  # atomic: a crashed job never leaves a half-written file


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def country_files(roads_dir) -> list[Path]:
    return sorted(Path(roads_dir).glob("*_roads.pkl.gz"))


def iso_of(path: Path) -> str:
    return path.name.split("_")[0].upper()
