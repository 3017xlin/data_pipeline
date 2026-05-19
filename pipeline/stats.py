"""
Streaming statistics + diverged / anomaly bookkeeping.

Welford-Chan parallel update gives numerically stable mean/std for tensors
whose total row count won't fit in memory. We accumulate per column.

Divergence is detected on the raw, pre-transform fields (we only need
gross sanity: NaN/Inf, or Uy magnitude greater than `max_uy` × U_ref, or
surface pressure exceeding `max_p_surf` in absolute kinematic units).
These thresholds match the constants used in eda_phase1.py.

Anomalies are flagged on the post-transform per-case means with a global
3-sigma test. They are NOT excluded from PT generation by default — only
diverged cases are dropped. Downstream code can choose how to use the
anomaly list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --- Divergence thresholds (mirroring eda_phase1) -----------------------
DIVERGE_MAX_UY_FACTOR = 5.0  # |Uy| > factor * U_ref → diverged
DIVERGE_MAX_P_SURF = 10000.0  # kinematic units (m²/s²)
ANOMALY_SIGMA = 3.0


@dataclass
class WelfordAccum:
    """Per-column streaming mean/var via Chan's parallel update."""
    n: int = 0
    mean: np.ndarray | None = None
    M2: np.ndarray | None = None  # sum of squared deviations

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        b = np.asarray(batch, dtype=np.float64)
        if b.ndim == 1:
            b = b[:, None]
        bn = b.shape[0]
        if self.mean is None:
            d = b.shape[1]
            self.mean = np.zeros(d, dtype=np.float64)
            self.M2 = np.zeros(d, dtype=np.float64)

        b_mean = b.mean(axis=0)
        b_M2 = ((b - b_mean) ** 2).sum(axis=0)
        delta = b_mean - self.mean
        new_n = self.n + bn
        self.mean = self.mean + delta * (bn / new_n)
        self.M2 = self.M2 + b_M2 + (delta ** 2) * (self.n * bn / new_n)
        self.n = new_n

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n < 2 or self.mean is None:
            d = 1 if self.mean is None else self.mean.size
            return np.zeros(d), np.ones(d)
        std = np.sqrt(self.M2 / (self.n - 1))
        # Avoid zero std (constant column) — z-score becomes a no-op.
        std = np.where(std < 1e-12, 1.0, std)
        return self.mean.copy(), std


@dataclass
class PerCaseStats:
    case_name: str
    file: str
    diverged: bool
    error: str | None = None
    n_surface: int = 0
    n_volume_raw: int = 0  # before bbox cut
    n_volume_cut: int = 0  # after bbox cut
    n_stl_faces: int = 0
    z_max_building: float = 0.0
    U_ref: float = 0.0
    rho: float = 0.0
    # per-case post-transform means for anomaly detection
    volume_means: list[float] = field(default_factory=list)
    surface_means: list[float] = field(default_factory=list)


def check_divergence(
    U_volume: np.ndarray,
    p_surf_kin: np.ndarray,
    U_ref: float,
) -> tuple[bool, str | None]:
    """Return (is_diverged, reason_or_None)."""
    if not np.all(np.isfinite(U_volume)):
        return True, "non-finite U in volume"
    if not np.all(np.isfinite(p_surf_kin)):
        return True, "non-finite p on surface"
    uy = U_volume[:, 1]
    if np.abs(uy).max() > DIVERGE_MAX_UY_FACTOR * abs(U_ref):
        return True, f"|Uy|>{DIVERGE_MAX_UY_FACTOR}*U_ref"
    if np.abs(p_surf_kin).max() > DIVERGE_MAX_P_SURF:
        return True, f"|p_surf|>{DIVERGE_MAX_P_SURF}"
    return False, None


def detect_anomalies(
    per_case: list[PerCaseStats],
    sigma: float = ANOMALY_SIGMA,
) -> list[dict]:
    """3-sigma outlier check over per-case post-transform means."""
    clean = [p for p in per_case if not p.diverged]
    if len(clean) < 10:
        return []

    vol = np.array([p.volume_means for p in clean], dtype=np.float64)
    surf = np.array([p.surface_means for p in clean], dtype=np.float64)

    def _outliers(mat: np.ndarray, names: list[str], kind: str) -> list[dict]:
        if mat.size == 0:
            return []
        mu = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = np.abs(mat - mu) / sd
        out: list[dict] = []
        for i, p in enumerate(clean):
            for j, name in enumerate(names):
                if z[i, j] > sigma:
                    out.append(
                        {
                            "case": p.case_name,
                            "kind": kind,
                            "field": name,
                            "z": float(z[i, j]),
                            "value": float(mat[i, j]),
                        }
                    )
        return out

    from .physics import VOLUME_FIELD_NAMES, SURFACE_FIELD_NAMES

    anomalies = _outliers(vol, VOLUME_FIELD_NAMES, "volume") + _outliers(
        surf, SURFACE_FIELD_NAMES, "surface"
    )
    return anomalies


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
