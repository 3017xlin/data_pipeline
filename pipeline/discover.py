"""
File discovery: walk a directory tree and return every CFD case NPZ.

We deliberately skip:
  - `*_filtered.npz` artifacts produced by the old cut_field.py pipeline
    (the new pipeline cuts in-memory and never writes intermediate files)
  - Anything that doesn't end in `.npz`
"""
from __future__ import annotations

import os
from pathlib import Path


def discover_npz(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".npz"):
                continue
            if fn.endswith("_filtered.npz"):
                continue
            out.append(Path(dirpath) / fn)
    out.sort()
    return out


def case_name_from_path(npz_path: Path) -> str:
    return npz_path.stem
