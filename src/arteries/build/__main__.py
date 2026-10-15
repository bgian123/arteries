"""Command-line entry point:  python -m arteries.build <stage> --work WORKDIR [...]

Examples
--------
python -m arteries.build init --work W --roads-dir R --cities C --ports P
python -m arteries.build list-countries --work W        # one ISO3 per line (for job arrays)
python -m arteries.build ingest --work W --task-id 7    # 7th country (0-based)
python -m arteries.build merge --work W
python -m arteries.build terminals --work W
python -m arteries.build simplify --work W
python -m arteries.build assemble --work W
python -m arteries.build n-landmatrix-tasks --work W
python -m arteries.build landmatrix --work W --task-id 0
python -m arteries.build package --work W --version 0.1.0
python -m arteries.build all --work W                   # everything sequentially (testing)
"""
import argparse
import sys

from . import stages
from .common import BuildConfig, Log, country_files, iso_of


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m arteries.build")
    ap.add_argument("stage")
    ap.add_argument("--work", required=True)
    ap.add_argument("--roads-dir")
    ap.add_argument("--cities")
    ap.add_argument("--ports")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--iso")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--set", nargs="*", default=[], help="override config: key=value")
    a = ap.parse_args(argv)

    if a.stage == "init":
        cfg = BuildConfig(roads_dir=a.roads_dir, cities_csv=a.cities, ports_csv=a.ports, work_dir=a.work)
    else:
        cfg = BuildConfig.load(a.work)
    for kv in a.set:
        k, v = kv.split("=", 1)
        typ = type(getattr(cfg, k))
        setattr(cfg, k, typ(v) if typ is not bool else v.lower() in ("1", "true", "yes"))
    if a.stage == "init" or a.set:
        cfg.save()
    if a.stage == "init":
        print(f"config written to {cfg.work / 'config.json'}"); return 0

    isos = [iso_of(p) for p in country_files(cfg.roads_dir)]
    log = Log(cfg.work / "logs" / f"{a.stage}{'' if a.task_id is None else '_' + str(a.task_id)}.log")
    if a.stage == "list-countries":
        print("\n".join(isos)); return 0
    if a.stage == "ingest":
        targets = [a.iso] if a.iso else ([isos[a.task_id]] if a.task_id is not None else isos)
        for iso in targets:
            stages.ingest(cfg, iso, log)
    elif a.stage in ("merge", "terminals", "simplify", "assemble"):
        getattr(stages, a.stage)(cfg, log)
    elif a.stage == "n-landmatrix-tasks":
        print(stages.n_landmatrix_tasks(cfg)); return 0
    elif a.stage == "landmatrix":
        tasks = [a.task_id] if a.task_id is not None else range(stages.n_landmatrix_tasks(cfg))
        for t in tasks:
            stages.landmatrix(cfg, t, log)
    elif a.stage == "package":
        stages.package(cfg, log, a.version)
    elif a.stage == "all":
        for iso in isos:
            stages.ingest(cfg, iso, log)
        for s in ("merge", "terminals", "simplify", "assemble"):
            getattr(stages, s)(cfg, log)
        for t in range(stages.n_landmatrix_tasks(cfg)):
            stages.landmatrix(cfg, t, log)
        stages.package(cfg, log, a.version)
    else:
        ap.error(f"unknown stage {a.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
