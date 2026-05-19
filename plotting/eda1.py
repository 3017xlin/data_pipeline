"""
EDA phase 1: per-case aggregate stats + summary plots.

Runs over the raw NPZ corpus (independent of PT generation). For every
case we collect physical/statistical descriptors and the same multi-offset
shell vs surface diagnostics that the original eda_phase1.py produced.

Differences from the original:
  - Uses Open3D for SDF (robust signed distance, replaces the old
    cKDTree-on-stl_centers approximation).
  - Uses pipeline.shell + pipeline.idw for shell construction so the
    plotting and PT pipelines stay in lock-step.
  - The IDW k differs per offset:
        0.5 m → k=4    (close-in, sharp)
        2.0 m → k=8    (this is also the PT shell)
        3.0 m → k=4    (far enough that 4 neighbours are stable)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from pipeline import physics
from pipeline.discover import case_name_from_path, discover_npz
from pipeline.idw import build_tree, idw_query
from pipeline.sdf3d import SDFComputer
from pipeline.shell import generate_shell_points
from pipeline.stats import (
    DIVERGE_MAX_P_SURF,
    DIVERGE_MAX_UY_FACTOR,
    save_json,
)
from pipeline.transform import CUT_X, CUT_Y, CUT_Z

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OFFSETS_M = [0.5, 2.0, 3.0]
IDW_K_BY_OFFSET = {0.5: 4, 2.0: 8, 3.0: 4}


def _bbox_cut(volume_pos: np.ndarray, volume_fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = (
        (volume_pos[:, 0] >= CUT_X[0])
        & (volume_pos[:, 0] <= CUT_X[1])
        & (volume_pos[:, 1] >= CUT_Y[0])
        & (volume_pos[:, 1] <= CUT_Y[1])
        & (volume_pos[:, 2] >= CUT_Z[0])
        & (volume_pos[:, 2] <= CUT_Z[1])
    )
    return volume_pos[m], volume_fields[m]


def _process_single(npz_path_str: str) -> dict:
    npz_path = Path(npz_path_str)
    case_name = case_name_from_path(npz_path)
    rec: dict[str, Any] = {"file": npz_path.name, "case": case_name, "diverged": False}

    try:
        with np.load(npz_path, allow_pickle=True) as d:
            stl_vertices = np.ascontiguousarray(d["stl_coordinates"], dtype=np.float32)
            stl_faces = np.ascontiguousarray(d["stl_faces"]).astype(np.int64)
            stl_centers = np.ascontiguousarray(d["stl_centers"], dtype=np.float32)
            stl_areas = np.ascontiguousarray(d["stl_areas"], dtype=np.float32)
            surface_pos = np.ascontiguousarray(d["surface_mesh_centers"], dtype=np.float32)
            surface_normals = np.ascontiguousarray(d["surface_normals"], dtype=np.float32)
            surface_areas = np.ascontiguousarray(d["surface_areas"], dtype=np.float32)
            surface_fields = np.ascontiguousarray(d["surface_fields"], dtype=np.float32)
            volume_pos = np.ascontiguousarray(d["volume_mesh_centers"], dtype=np.float32)
            volume_fields = np.ascontiguousarray(d["volume_fields"], dtype=np.float32)
            global_params = np.ascontiguousarray(d["global_params_values"], dtype=np.float32)

        if stl_faces.ndim == 1:
            stl_faces = stl_faces.reshape(-1, 3)

        U_ref = float(global_params[0]) if global_params.size > 0 else 2.0
        if U_ref <= 0:
            U_ref = 2.0
        rho = float(global_params[1]) if global_params.size > 1 else 1.225
        if rho <= 0:
            rho = 1.225
        z_max = float(stl_vertices[:, 2].max())
        L_scale = max(z_max, 1.0)

        # Cut volume to bbox (same convention as PT pipeline)
        volume_pos, volume_fields = _bbox_cut(volume_pos, volume_fields)
        if volume_pos.shape[0] == 0:
            rec["diverged"] = True
            rec["error"] = "no volume points in bbox"
            return rec

        # Divergence
        if (
            np.abs(volume_fields[:, 1]).max() > DIVERGE_MAX_UY_FACTOR * U_ref
            or np.abs(surface_fields[:, 0]).max() > DIVERGE_MAX_P_SURF
            or not np.all(np.isfinite(volume_fields))
            or not np.all(np.isfinite(surface_fields))
        ):
            rec["diverged"] = True
            rec["error"] = "divergence check"
            return rec

        # Aerodynamic transforms
        p_surf_aero = physics.p_aero(surface_fields[:, 0], surface_pos[:, 2], U_ref)
        p_vol_aero = physics.p_aero(volume_fields[:, 3], volume_pos[:, 2], U_ref)
        wss_mag = np.sqrt((surface_fields[:, 1:4] ** 2).sum(axis=1)) / (U_ref ** 2)
        log_nut = physics.log_nut(volume_fields[:, 4], L_scale)

        windward = surface_normals[:, 1] < -0.5
        leeward = surface_normals[:, 1] > 0.5
        p_wind = float(p_surf_aero[windward].mean()) if windward.sum() else float("nan")
        p_lee = float(p_surf_aero[leeward].mean()) if leeward.sum() else float("nan")

        rec.update(
            {
                "z_max": z_max,
                "U_ref": U_ref,
                "rho": rho,
                "n_stl_pts": int(stl_vertices.shape[0]),
                "n_stl_faces": int(stl_faces.shape[0]),
                "n_surf_cells": int(surface_fields.shape[0]),
                "n_vol_cells": int(volume_fields.shape[0]),
                "surf_area_mean": float(stl_areas.mean()),
                "surf_area_std": float(stl_areas.std()),
                "p_aero_mean": float(p_surf_aero.mean()),
                "p_aero_std": float(p_surf_aero.std()),
                "p_aero_min": float(p_surf_aero.min()),
                "p_aero_max": float(p_surf_aero.max()),
                "delta_p_wl": float(p_wind - p_lee) if np.isfinite(p_wind) and np.isfinite(p_lee) else None,
                "wss_mag_mean": float(wss_mag.mean()),
                "wss_mag_max": float(wss_mag.max()),
                "Uy_mean": float(volume_fields[:, 1].mean()),
                "Uy_min": float(volume_fields[:, 1].min()),
                "Uy_max": float(volume_fields[:, 1].max()),
                "p_vol_aero_mean": float(p_vol_aero.mean()),
                "log_nut_mean": float(log_nut.mean()),
                "log_nut_max": float(log_nut.max()),
                "nut_mean": float(volume_fields[:, 4].mean()),
                "nut_max": float(volume_fields[:, 4].max()),
            }
        )

        # Shell analysis (Open3D-based SDF)
        sdf = SDFComputer(stl_vertices, stl_faces)

        tree = build_tree(volume_pos)
        surf_tree = build_tree(surface_pos)
        _, surf_neighbour = surf_tree.query(surface_pos, k=5, workers=1)

        for offset in OFFSETS_M:
            tag = f"{offset}m"
            k = IDW_K_BY_OFFSET[offset]
            # Scale the SDF filter with the offset so a 0.5m shell isn't
            # nuked by the default 1m threshold meant for 2m offsets.
            min_sdf = max(0.3, 0.5 * offset)
            # Shells are deviated from CFD wall mesh centers, NOT STL.
            shell_pts, _base_idx, shell_sdf = generate_shell_points(
                base_points=surface_pos,
                base_normals=surface_normals,
                offset_m=offset,
                sdf=sdf,
                min_sdf_m=min_sdf,
            )
            if shell_pts.shape[0] < 10:
                rec[f"shell_{tag}_dist_mean"] = float("nan")
                rec[f"shell_{tag}_smoothness"] = float("nan")
                rec[f"shell_{tag}_corr"] = float("nan")
                rec[f"shell_{tag}_delta_p_mean"] = float("nan")
                rec[f"shell_{tag}_delta_p_std"] = float("nan")
                rec[f"shell_{tag}_wind_consistent"] = float("nan")
                rec[f"shell_{tag}_n_valid"] = int(shell_pts.shape[0])
                continue

            # Interpolate raw kinematic p at shell points, then detrend
            p_shell_raw, _, dists = idw_query(
                tree=tree,
                source_values=volume_fields[:, 3],
                query_points=shell_pts,
                k=k,
            )
            p_shell_aero = physics.p_aero(p_shell_raw, shell_pts[:, 2], U_ref)

            # Distance to nearest CFD volume cell (sanity)
            dist_mean = float(dists[:, 0].mean())

            # Map shell pts back to the closest surface point for the
            # correlation / W-L diagnostics (cross-domain comparisons).
            _, surf_idx = surf_tree.query(shell_pts, k=1, workers=1)
            paired_surf_p = p_surf_aero[surf_idx]
            delta = p_shell_aero - paired_surf_p

            # Shell smoothness: local std of SHELL pressure around each
            # shell point's own neighbours. Must use a shell-specific
            # KDTree — surf_neighbour indices live in surface_pos space
            # and are NOT valid indices into p_shell_aero (different size,
            # different filtering).
            k_nbr = min(5, shell_pts.shape[0])
            if k_nbr >= 2:
                shell_tree = build_tree(shell_pts)
                _, shell_neighbour = shell_tree.query(
                    shell_pts, k=k_nbr, workers=1
                )
                if shell_neighbour.ndim == 1:
                    shell_neighbour = shell_neighbour[:, None]
                local_var = float(
                    np.mean(np.std(p_shell_aero[shell_neighbour[:, 1:]], axis=1))
                )
            else:
                local_var = float("nan")

            if len(paired_surf_p) >= 3:
                try:
                    corr, _ = pearsonr(paired_surf_p, p_shell_aero)
                except Exception:
                    corr = float("nan")
            else:
                corr = float("nan")

            wind_mask = surface_normals[surf_idx, 1] < -0.5
            lee_mask = surface_normals[surf_idx, 1] > 0.5
            if wind_mask.sum() > 0 and lee_mask.sum() > 0 and np.isfinite(p_wind) and np.isfinite(p_lee):
                shell_dp = float(p_shell_aero[wind_mask].mean() - p_shell_aero[lee_mask].mean())
                surf_dp = float(p_wind - p_lee)
                consistent = 1.0 if (shell_dp > 0) == (surf_dp > 0) else 0.0
            else:
                consistent = float("nan")

            rec[f"shell_{tag}_dist_mean"] = dist_mean
            rec[f"shell_{tag}_smoothness"] = local_var
            rec[f"shell_{tag}_corr"] = float(corr)
            rec[f"shell_{tag}_delta_p_mean"] = float(delta.mean())
            rec[f"shell_{tag}_delta_p_std"] = float(delta.std())
            rec[f"shell_{tag}_wind_consistent"] = consistent
            rec[f"shell_{tag}_n_valid"] = int(shell_pts.shape[0])

        return rec

    except Exception as e:
        rec["diverged"] = True
        rec["error"] = f"{e}\n{traceback.format_exc()}"
        return rec


# =====================================================================
# Plot helpers
# =====================================================================
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_plots(records: list[dict], out_dir: Path) -> None:
    clean = [r for r in records if not r.get("diverged", False)]
    if not clean:
        print("[eda1] no clean cases to plot")
        return

    plot_dir = out_dir / "plots" / "eda1"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def col(key: str) -> np.ndarray:
        return np.array(
            [float(r[key]) if r.get(key) is not None else np.nan for r in clean],
            dtype=np.float64,
        )

    counter = [0]

    def save_named(name: str, fig) -> None:
        counter[0] += 1
        _save(fig, plot_dir / f"{counter[0]:02d}_{name}.png")

    # 1. Building height
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(col("z_max"), bins=30, edgecolor="black")
    ax.set_xlabel("Max building height (m)")
    ax.set_ylabel("Count")
    ax.set_title(f"Building Height Distribution (n={len(clean)})")
    ax.grid(True, alpha=0.3)
    save_named("z_max_distribution", fig)

    # 2. Mesh sizes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k, lbl in zip(axes, ["n_stl_pts", "n_surf_cells", "n_vol_cells"],
                          ["STL vertices", "Surface cells", "Volume cells (cut)"]):
        ax.hist(col(k), bins=30, edgecolor="black")
        ax.set_xlabel(lbl)
        ax.set_title(f"{lbl} per case")
    fig.tight_layout()
    save_named("mesh_size_distributions", fig)

    # 3. Surface pressure stats
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, k, lbl in zip(axes, ["p_aero_mean", "p_aero_min", "p_aero_max"],
                          ["Mean p_aero", "Min p_aero", "Max p_aero"]):
        ax.hist(col(k), bins=30, edgecolor="black")
        ax.set_xlabel(lbl)
        ax.set_title(lbl)
    fig.tight_layout()
    save_named("p_aero_stats", fig)

    # 4. delta P W-L
    dp = col("delta_p_wl")
    dpc = dp[np.isfinite(dp)]
    fig, ax = plt.subplots(figsize=(8, 4))
    if dpc.size:
        ax.hist(dpc, bins=30, edgecolor="black")
        ax.axvline(dpc.mean(), color="r", linestyle="--", label=f"mean={dpc.mean():.3f}")
        ax.legend()
    ax.set_xlabel("ΔP windward-leeward")
    ax.set_title("Aerodynamic Pressure Difference")
    ax.grid(True, alpha=0.3)
    save_named("delta_p_wl", fig)

    # 5. ΔP vs height
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(dp)
    ax.scatter(col("z_max")[valid], dp[valid], s=10, alpha=0.5)
    ax.set_xlabel("Building height (m)")
    ax.set_ylabel("ΔP windward-leeward")
    ax.set_title("Pressure Difference vs Building Height")
    ax.grid(True, alpha=0.3)
    save_named("delta_p_vs_height", fig)

    # 6/7/8 Shell comparisons across offsets
    for metric, ylabel, name in [
        ("corr", "Pearson r", "shell_correlation"),
        ("smoothness", "Local std", "shell_smoothness"),
        ("dist_mean", "Mean dist to nearest vol cell (m)", "shell_dist_to_vol"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        means = []
        for off in OFFSETS_M:
            vals = col(f"shell_{off}m_{metric}")
            ax.scatter(np.full(vals.size, off), vals, s=10, alpha=0.3)
            means.append(np.nanmean(vals))
        ax.plot(OFFSETS_M, means, "k-o", linewidth=2, markersize=8, label="Mean")
        ax.set_xlabel("Offset (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(name.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_named(name, fig)

    # 9. Correlation vs building height
    fig, ax = plt.subplots(figsize=(8, 5))
    for off in OFFSETS_M:
        ax.scatter(col("z_max"), col(f"shell_{off}m_corr"), s=10, alpha=0.3, label=f"{off}m")
    ax.set_xlabel("Building height (m)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Shell Correlation vs Building Height")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_named("correlation_vs_height", fig)

    # 10. nut + WSS
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(col("log_nut_max"), bins=30, edgecolor="black")
    axes[0].set_xlabel("log_nut max (per case)")
    axes[0].set_title("log(nut · L_scale) max")
    axes[1].hist(col("wss_mag_mean"), bins=30, edgecolor="black")
    axes[1].set_xlabel("Mean |WSS|/U_ref²")
    axes[1].set_title("Wall Shear Stress (non-dim)")
    fig.tight_layout()
    save_named("nut_wss", fig)

    # 11. Surface area uniformity
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(col("surf_area_mean"), col("surf_area_std"), s=10, alpha=0.5)
    ax.set_xlabel("Mean cell area")
    ax.set_ylabel("Std cell area")
    ax.set_title("Surface Mesh Uniformity")
    ax.grid(True, alpha=0.3)
    save_named("surface_area_uniformity", fig)


def _make_summary(records: list[dict], out_dir: Path) -> None:
    clean = [r for r in records if not r.get("diverged", False)]
    diverged = [r for r in records if r.get("diverged", False)]
    if not clean:
        return

    def col(key: str) -> np.ndarray:
        return np.array(
            [float(r[key]) if r.get(key) is not None else np.nan for r in clean],
            dtype=np.float64,
        )

    lines = [
        f"Total cases: {len(records)}",
        f"Clean: {len(clean)}  Diverged: {len(diverged)}",
        "",
        "=== SHELL OFFSET COMPARISON ===",
        f"{'Metric':<35} {'0.5m':>10} {'2.0m':>10} {'3.0m':>10}",
        "-" * 67,
    ]
    for label, key in [
        ("Mean dist to vol cell (m)", "shell_{}m_dist_mean"),
        ("Local smoothness", "shell_{}m_smoothness"),
        ("Pearson corr (shell vs surface)", "shell_{}m_corr"),
        ("Mean Δp shell-surface", "shell_{}m_delta_p_mean"),
        ("Std Δp shell-surface", "shell_{}m_delta_p_std"),
        ("W-L consistency frac", "shell_{}m_wind_consistent"),
        ("# valid shell points (mean)", "shell_{}m_n_valid"),
    ]:
        vals = [f"{np.nanmean(col(key.format(o))):>10.4f}" for o in OFFSETS_M]
        lines.append(f"{label:<35} {vals[0]} {vals[1]} {vals[2]}")

    lines.append("")
    lines.append(f"=== GENERAL STATS ({len(clean)} clean cases) ===")
    lines.append(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    lines.append("-" * 67)
    for k in [
        "z_max", "n_surf_cells", "n_vol_cells",
        "p_aero_mean", "p_aero_std", "delta_p_wl",
        "wss_mag_mean", "Uy_mean", "log_nut_mean", "log_nut_max",
    ]:
        v = col(k)
        v = v[np.isfinite(v)]
        if v.size:
            lines.append(
                f"{k:<25} {v.mean():>10.4f} {v.std():>10.4f} {v.min():>10.4f} {v.max():>10.4f}"
            )

    text = "\n".join(lines)
    print(text)
    (out_dir / "plots" / "eda1").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "plots" / "eda1" / "summary.txt", "w") as f:
        f.write(text)


def run_eda1(data_dir: Path, out_dir: Path, workers: int) -> None:
    files = discover_npz(data_dir)
    print(f"[eda1] {len(files)} NPZ files, {workers} workers")
    if not files:
        return
    t0 = time.time()
    args = [str(p) for p in files]

    if workers <= 1:
        records = [_process_single(a) for a in tqdm(args, desc="eda1")]
    else:
        with mp.Pool(workers) as pool:
            records = []
            for r in tqdm(
                pool.imap_unordered(_process_single, args, chunksize=1),
                total=len(args),
                desc="eda1",
            ):
                records.append(r)

    save_json(records, out_dir / "plots" / "eda1" / "all_case_stats.json")
    _make_plots(records, out_dir)
    _make_summary(records, out_dir)

    print(f"[eda1] done in {time.time() - t0:.0f}s")
