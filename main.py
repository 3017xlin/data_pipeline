"""
Top-level CLI for the HDB 3D wind data pipeline.

Two independent subtasks (declared via --do):
  - plot : EDA phase 1 + 2 aggregate plots
  - pt   : two-pass NPZ → PT generation (norm_stats then z-scored PT)

By default both run, plots first ("先去做画图，然后...求 pt"). Pick subset
with --do plot or --do pt.

Layout:
  --data_dir   root containing case_*.npz (recursively scanned)
  --out_dir    where everything lands (subfolders: pt/, plots/, *.json)
  --workers    pool size (default: cpu_count())

Example:
    python main.py --data_dir /data/hdb_npz --out_dir /scratch/out --workers 32
    python main.py --data_dir /data/hdb_npz --out_dir /scratch/out --do pt
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path


def _default_workers() -> int:
    """Respect SLURM_CPUS_PER_TASK on HPC, then TOTAL_CORES, then cpu_count()."""
    for var in ("SLURM_CPUS_PER_TASK", "TOTAL_CORES"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return cpu_count()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HDB 3D wind data pipeline (NPZ → PT, plus optional EDA plots).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", type=Path, default=Path("."),
                   help="Root scanned recursively for NPZ files.")
    p.add_argument("--out_dir", type=Path, default=Path("./outputs"),
                   help="Output directory: PT files, plots, and stats JSON.")
    p.add_argument("--workers", type=int, default=_default_workers(),
                   help="Parallel workers. Honours SLURM_CPUS_PER_TASK by default.")
    p.add_argument("--do", nargs="+", default=["pt"],
                   choices=["plot", "pt", "eda1", "eda2"],
                   help="Subtasks. 'plot' = eda1 + eda2. Pass multiple to chain.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cases whose .pt file already exists in pass 2.")
    return p.parse_args()


def main() -> None:
    args = _parse()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tasks = set()
    for t in args.do:
        if t == "plot":
            tasks |= {"eda1", "eda2"}
        else:
            tasks.add(t)

    print(f"=== HDB pipeline ===")
    print(f"data_dir = {args.data_dir}")
    print(f"out_dir  = {args.out_dir}")
    print(f"workers  = {args.workers}")
    print(f"tasks    = {sorted(tasks)}")
    sys.stdout.flush()

    # Plotting first (independent of PT)
    if "eda1" in tasks:
        from plotting.eda1 import run_eda1
        run_eda1(args.data_dir, args.out_dir, args.workers)
    if "eda2" in tasks:
        from plotting.eda2_plots import run_eda2
        run_eda2(args.data_dir, args.out_dir, args.workers)

    if "pt" in tasks:
        from pipeline.transform import run_pt_pipeline
        run_pt_pipeline(args.data_dir, args.out_dir, args.workers,
                        args.skip_existing)

    print("=== done ===")


if __name__ == "__main__":
    main()
