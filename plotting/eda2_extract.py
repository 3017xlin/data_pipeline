"""
EDA phase 2 — extract per-floor windward/leeward shell-vs-surface pressures.

For every case, every detected building, every (x_sample, z_floor) on a
fixed grid, we record the surface pressure (p_b) and the shell pressure
at 0.5/2/3 m offsets (p_a05/p_a2/p_a3) on both windward (-Y) and leeward
(+Y) faces. The result is a flat (N_rows, 12) table written to a single
NPZ; eda2_plots.py reads it.

Differences from the legacy eda_phase2.py:
  - Reads NPZ directly (no dependency on PT generation).
  - SDF computed by Open3D (pipeline.sdf3d) for both the shell validity
    filter and the IDW-neighbour validity check.
  - Per-offset IDW k: 0.5 → 4, 2 → 8, 3 → 4.
  - Cuts volume to the same bbox the PT pipeline uses, so plots reflect
    what the model actually sees.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from pipeline import physics
from pipeline.discover import case_name_from_path, discover_npz
from pipeline.idw import idw_query
from pipeline.sdf3d import SDFComputer
from pipeline.transform import CUT_X, CUT_Y, CUT_Z


FLOOR_STEP = 3.0
X_BINS = 10
OFFSETS = [0.5, 2.0, 3.0]
IDW_K_BY_OFFSET = {0.5: 4, 2.0: 8, 3.0: 4}
EPS = 1e-10
SDF_MIN = 0.1  # neighbouring volume cell must be at least this far from surface


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


def _idw_with_sdf_validation(
    tree: cKDTree,
    p_vol: np.ndarray,
    vol_sdf: np.ndarray,
    targets: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """IDW for scalar p, with a mask flagging queries whose k neighbours
    are all in the fluid (i.e. min neighbour SDF > SDF_MIN)."""
    dists, idx = tree.query(targets, k=k, workers=1)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    safe = np.maximum(dists, EPS)
    w = 1.0 / safe
    w /= w.sum(axis=1, keepdims=True)
    p = (p_vol[idx] * w).sum(axis=1)
    valid = np.all(vol_sdf[idx] > SDF_MIN, axis=1)
    return p, valid


def _generate_floors(z_min: float, z_max: float, step: float = FLOOR_STEP) -> np.ndarray:
    floors = np.arange(z_min + step / 2, z_max, step)
    if floors.size == 0:
        floors = np.array([(z_min + z_max) / 2])
    return floors


def _process_case(npz_path_str: str) -> list[list]:
    npz_path = Path(npz_path_str)
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            stl_vertices = np.ascontiguousarray(d["stl_coordinates"], dtype=np.float32)
            stl_faces = np.ascontiguousarray(d["stl_faces"]).astype(np.int64)
            if stl_faces.ndim == 1:
                stl_faces = stl_faces.reshape(-1, 3)
            surface_pos = np.ascontiguousarray(d["surface_mesh_centers"], dtype=np.float64)
            surface_normals = np.ascontiguousarray(d["surface_normals"], dtype=np.float64)
            surface_fields = np.ascontiguousarray(d["surface_fields"], dtype=np.float64)
            volume_pos = np.ascontiguousarray(d["volume_mesh_centers"], dtype=np.float64)
            volume_fields = np.ascontiguousarray(d["volume_fields"], dtype=np.float64)
            global_params = np.ascontiguousarray(d["global_params_values"], dtype=np.float32)

        U_ref = float(global_params[0]) if global_params.size > 0 and global_params[0] > 0 else 2.0

        # Cut volume
        volume_pos, volume_fields = _bbox_cut(volume_pos.astype(np.float32), volume_fields.astype(np.float32))
        volume_pos = volume_pos.astype(np.float64)
        volume_fields = volume_fields.astype(np.float64)
        if volume_pos.shape[0] < 10:
            return []

        p_surf_aero = physics.p_aero(surface_fields[:, 0], surface_pos[:, 2], U_ref)
        p_vol_kin = volume_fields[:, 3]

        # SDF on volume points (Open3D)
        sdf = SDFComputer(stl_vertices, stl_faces)
        vol_sdf = sdf.signed_distance(volume_pos.astype(np.float32))

        # Side wall mask
        wall = np.abs(surface_normals[:, 2]) < 0.5
        if wall.sum() < 50:
            return []
        w_pos = surface_pos[wall]
        w_norm = surface_normals[wall]
        w_p = p_surf_aero[wall]

        # DBSCAN cluster on XY to find separate buildings
        clusters = DBSCAN(eps=5.0, min_samples=10).fit_predict(w_pos[:, :2])
        vol_tree = cKDTree(volume_pos)

        rows: list[list] = []
        for bid in np.unique(clusters):
            if bid == -1:
                continue
            bm = clusters == bid
            b_pos = w_pos[bm]
            b_norm = w_norm[bm]
            b_p = w_p[bm]
            b_ny = b_norm[:, 1]
            wind = b_ny < -0.5
            lee = b_ny > 0.5
            if wind.sum() < 5 or lee.sum() < 5:
                continue

            wi_pos = b_pos[wind]
            wi_norm = b_norm[wind]
            wi_p = b_p[wind]
            le_pos = b_pos[lee]
            le_norm = b_norm[lee]
            le_p = b_p[lee]

            z_min = float(b_pos[:, 2].min())
            z_max = float(b_pos[:, 2].max())
            thickness = float(abs(wi_pos[:, 1].mean() - le_pos[:, 1].mean()))

            x_lo = max(wi_pos[:, 0].min(), le_pos[:, 0].min())
            x_hi = min(wi_pos[:, 0].max(), le_pos[:, 0].max())
            if x_hi - x_lo < 2.0:
                continue
            x_samples = np.linspace(x_lo, x_hi, X_BINS + 2)[1:-1]
            floor_zs = _generate_floors(z_min, z_max, FLOOR_STEP)

            wi_xz_tree = cKDTree(wi_pos[:, [0, 2]])
            le_xz_tree = cKDTree(le_pos[:, [0, 2]])

            queries_xz = []
            floor_z_list = []
            for fz in floor_zs:
                for xs in x_samples:
                    queries_xz.append([xs, fz])
                    floor_z_list.append(fz)
            queries_xz = np.array(queries_xz)
            if queries_xz.size == 0:
                continue

            d_wi, idx_wi = wi_xz_tree.query(queries_xz, k=1)
            d_le, idx_le = le_xz_tree.query(queries_xz, k=1)
            valid_dist = (d_wi < 5.0) & (d_le < 5.0)

            # Build the (n_valid, 3, 3) target tensor: per valid query, 3 wind
            # shell points (one per offset) and 3 lee shell points.
            wind_targets: list[list[np.ndarray]] = [[] for _ in OFFSETS]
            lee_targets: list[list[np.ndarray]] = [[] for _ in OFFSETS]
            query_indices: list[int] = []
            for q in range(len(queries_xz)):
                if not valid_dist[q]:
                    continue
                w_pt = wi_pos[idx_wi[q]]
                w_nrm = wi_norm[idx_wi[q]]
                l_pt = le_pos[idx_le[q]]
                l_nrm = le_norm[idx_le[q]]
                for oi, o in enumerate(OFFSETS):
                    wind_targets[oi].append(w_pt + w_nrm * o)
                    lee_targets[oi].append(l_pt + l_nrm * o)
                query_indices.append(q)
            if not query_indices:
                continue

            # IDW each offset separately (different k!)
            p_wind_offsets = np.empty((len(OFFSETS), len(query_indices)))
            p_lee_offsets = np.empty_like(p_wind_offsets)
            valid_wind = np.ones((len(OFFSETS), len(query_indices)), dtype=bool)
            valid_lee = np.ones_like(valid_wind)

            for oi, o in enumerate(OFFSETS):
                k = IDW_K_BY_OFFSET[o]
                tw = np.asarray(wind_targets[oi])
                tl = np.asarray(lee_targets[oi])
                pw, vw = _idw_with_sdf_validation(vol_tree, p_vol_kin, vol_sdf, tw, k)
                pl, vl = _idw_with_sdf_validation(vol_tree, p_vol_kin, vol_sdf, tl, k)
                # Detrend with the shell's own z
                p_wind_offsets[oi] = physics.p_aero(pw, tw[:, 2], U_ref)
                p_lee_offsets[oi] = physics.p_aero(pl, tl[:, 2], U_ref)
                valid_wind[oi] = vw
                valid_lee[oi] = vl

            for i, q in enumerate(query_indices):
                if not (valid_wind[:, i].all() and valid_lee[:, i].all()):
                    continue
                p_b_wind = float(wi_p[idx_wi[q]])
                p_b_lee = float(le_p[idx_le[q]])
                fz = floor_z_list[q]
                x_frac = q % X_BINS
                rows.append([
                    p_b_wind,
                    float(p_wind_offsets[0, i]),
                    float(p_wind_offsets[1, i]),
                    float(p_wind_offsets[2, i]),
                    p_b_lee,
                    float(p_lee_offsets[0, i]),
                    float(p_lee_offsets[1, i]),
                    float(p_lee_offsets[2, i]),
                    fz, z_max, thickness, x_frac,
                ])

        return rows
    except Exception as e:
        print(f"[eda2] error in {npz_path.name}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []


def run_eda2_extract(data_dir: Path, out_dir: Path, workers: int) -> Path:
    files = discover_npz(data_dir)
    print(f"[eda2-extract] {len(files)} NPZ files, {workers} workers")
    if not files:
        return out_dir / "tensor_data.npz"
    args = [str(p) for p in files]

    t0 = time.time()
    all_rows: list[list] = []
    n_failed = 0
    if workers <= 1:
        for i, fp in enumerate(args):
            r = _process_case(fp)
            if r:
                all_rows.extend(r)
            else:
                n_failed += 1
            if (i + 1) % 25 == 0 or (i + 1) == len(args):
                print(f"  [{i+1}/{len(args)}] {len(all_rows):,} rows, {n_failed} failed")
    else:
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_process_case, args, chunksize=1)):
                if r:
                    all_rows.extend(r)
                else:
                    n_failed += 1
                if (i + 1) % 25 == 0 or (i + 1) == len(args):
                    print(
                        f"  [{i+1}/{len(args)}] {len(all_rows):,} rows, "
                        f"{n_failed} failed, {time.time()-t0:.0f}s"
                    )
                    sys.stdout.flush()

    data = np.array(all_rows, dtype=np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tensor_data.npz"
    np.savez_compressed(out_path, data=data)
    print(f"[eda2-extract] {data.shape} → {out_path} in {time.time()-t0:.0f}s")
    return out_path
