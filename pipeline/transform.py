"""
Two-pass NPZ → PT pipeline.

Pass 1 (stats):  walk every NPZ, apply physics transforms in-memory,
                 stream Welford accumulation, flag diverged cases.
                 Emits norm_stats.json + diverged/anomaly JSON.

Pass 2 (write):  walk every NPZ again, redo the physics + apply z-score
                 (using stats from pass 1) + compute SDF / SDF gradient /
                 2 m shell augmentation / latent grid SDF, then save one
                 .pt file per case.

Each case is processed by an independent worker. We hand workers
serializable arguments only (paths, the global stats dict in pass 2) so
the pool can use spawn cleanly.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from . import physics
from .discover import case_name_from_path, discover_npz
from .grid import build_grid_coords, grid_metadata
from .idw import build_tree, idw_query
from .sdf3d import SDFComputer
from .shell import generate_shell_points
from .stats import (
    PerCaseStats,
    WelfordAccum,
    check_divergence,
    detect_anomalies,
    save_json,
)


SCHEMA_VERSION = "hdb-3d-v1"
CUT_X = (-550.0, 550.0)
CUT_Y = (-550.0, 550.0)
CUT_Z = (0.0, 160.0)
SHELL_OFFSET_M = 2.0
SHELL_IDW_K = 8
DEFAULT_RHO = 1.225


# =====================================================================
# Helpers
# =====================================================================
def _faces_to_2d(faces: np.ndarray) -> np.ndarray:
    if faces.ndim == 1:
        if faces.size % 3 != 0:
            raise ValueError(f"flat faces size {faces.size} not divisible by 3")
        return faces.reshape(-1, 3)
    if faces.ndim == 2 and faces.shape[1] == 3:
        return faces
    raise ValueError(f"unsupported stl_faces shape {faces.shape}")


def _bbox_mask(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return (
        (x >= CUT_X[0])
        & (x <= CUT_X[1])
        & (y >= CUT_Y[0])
        & (y <= CUT_Y[1])
        & (z >= CUT_Z[0])
        & (z <= CUT_Z[1])
    )


def _load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as d:
        return {k: np.array(d[k]) for k in d.files}


def _extract_raw(raw: dict) -> dict:
    """Pull out the arrays we care about and normalise shapes."""
    stl_vertices = np.ascontiguousarray(raw["stl_coordinates"], dtype=np.float32)
    stl_centers = np.ascontiguousarray(raw["stl_centers"], dtype=np.float32)
    stl_faces = _faces_to_2d(np.ascontiguousarray(raw["stl_faces"])).astype(np.int64)
    stl_areas = np.ascontiguousarray(raw["stl_areas"], dtype=np.float32)

    surface_pos = np.ascontiguousarray(raw["surface_mesh_centers"], dtype=np.float32)
    surface_normals = np.ascontiguousarray(raw["surface_normals"], dtype=np.float32)
    surface_areas = np.ascontiguousarray(raw["surface_areas"], dtype=np.float32)
    surface_fields = np.ascontiguousarray(raw["surface_fields"], dtype=np.float32)

    volume_pos = np.ascontiguousarray(raw["volume_mesh_centers"], dtype=np.float32)
    volume_fields = np.ascontiguousarray(raw["volume_fields"], dtype=np.float32)

    global_params = np.ascontiguousarray(raw["global_params_values"], dtype=np.float32)

    return {
        "stl_vertices": stl_vertices,
        "stl_centers": stl_centers,
        "stl_faces": stl_faces,
        "stl_areas": stl_areas,
        "surface_pos": surface_pos,
        "surface_normals": surface_normals,
        "surface_areas": surface_areas,
        "surface_fields": surface_fields,
        "volume_pos": volume_pos,
        "volume_fields": volume_fields,
        "global_params": global_params,
    }


def _per_case_constants(d: dict) -> dict:
    U_ref = float(d["global_params"][0]) if d["global_params"].size > 0 else 2.0
    if U_ref <= 0:
        U_ref = 2.0
    rho = float(d["global_params"][1]) if d["global_params"].size > 1 else DEFAULT_RHO
    if rho <= 0:
        rho = DEFAULT_RHO
    # L_scale: building height from STL vertices (physical meters)
    z_max_building = float(d["stl_vertices"][:, 2].max())
    L_scale = max(z_max_building, 1.0)  # guard divisor
    return {
        "U_ref": U_ref,
        "rho": rho,
        "z_max_building": z_max_building,
        "L_scale": L_scale,
    }


def _apply_physics(d: dict, consts: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (volume_fields_phys, surface_fields_phys) post-transform but
    pre-z-score, for both pass1 (stats) and pass2 (output)."""
    vol = d["volume_fields"]  # (N,5) [Ux, Uy, Uz, p, nut]
    vol_phys = physics.transform_volume_fields(
        U=vol[:, 0:3],
        p_kin_vol=vol[:, 3],
        nut_vol=vol[:, 4],
        z_vol=d["volume_pos"][:, 2],
        U_ref=consts["U_ref"],
        L_scale=consts["L_scale"],
    )
    surf = d["surface_fields"]  # (S,4) [p, wssx, wssy, wssz]
    surf_phys = physics.transform_surface_fields(
        p_kin_surf=surf[:, 0],
        wss=surf[:, 1:4],
        z_surf=d["surface_pos"][:, 2],
        U_ref=consts["U_ref"],
    )
    return vol_phys, surf_phys


# =====================================================================
# Pass 1: streaming statistics
# =====================================================================
def _pass1_one(npz_path_str: str) -> tuple[dict, dict, dict]:
    """
    Worker for pass 1. Returns:
      per_case_dict   — dict form of PerCaseStats
      vol_partial     — partial Welford state for volume fields (or None)
      surf_partial    — partial Welford state for surface fields (or None)
    Volume / surface partials are empty dicts if the case is diverged.
    """
    npz_path = Path(npz_path_str)
    case_name = case_name_from_path(npz_path)
    pc = PerCaseStats(case_name=case_name, file=npz_path.name, diverged=False)
    empty: dict = {}

    try:
        raw = _load_npz(npz_path)
        d = _extract_raw(raw)
    except Exception as e:
        pc.diverged = True
        pc.error = f"load: {e}"
        return asdict(pc), empty, empty

    try:
        consts = _per_case_constants(d)
        pc.U_ref = consts["U_ref"]
        pc.rho = consts["rho"]
        pc.z_max_building = consts["z_max_building"]
        pc.n_stl_faces = int(d["stl_faces"].shape[0])
        pc.n_surface = int(d["surface_pos"].shape[0])
        pc.n_volume_raw = int(d["volume_pos"].shape[0])

        # Cut volume to bbox before any stats
        mask = _bbox_mask(d["volume_pos"])
        d["volume_pos"] = d["volume_pos"][mask]
        d["volume_fields"] = d["volume_fields"][mask]
        pc.n_volume_cut = int(d["volume_pos"].shape[0])

        # Divergence check on RAW Uy and RAW surface kinematic pressure
        diverged, reason = check_divergence(
            U_volume=d["volume_fields"][:, 0:3],
            p_surf_kin=d["surface_fields"][:, 0],
            U_ref=consts["U_ref"],
        )
        if diverged:
            pc.diverged = True
            pc.error = reason
            return asdict(pc), empty, empty

        # Apply physics
        vol_phys, surf_phys = _apply_physics(d, consts)

        if not (np.all(np.isfinite(vol_phys)) and np.all(np.isfinite(surf_phys))):
            pc.diverged = True
            pc.error = "non-finite after physics"
            return asdict(pc), empty, empty

        # Per-case post-transform means for anomaly check
        pc.volume_means = vol_phys.mean(axis=0).astype(float).tolist()
        pc.surface_means = surf_phys.mean(axis=0).astype(float).tolist()

        # Build mini-Welford locally per field set, then return its state
        wv = WelfordAccum()
        wv.update(vol_phys)
        ws = WelfordAccum()
        ws.update(surf_phys)

        vol_partial = {
            "n": int(wv.n),
            "mean": wv.mean.tolist() if wv.mean is not None else [],
            "M2": wv.M2.tolist() if wv.M2 is not None else [],
        }
        surf_partial = {
            "n": int(ws.n),
            "mean": ws.mean.tolist() if ws.mean is not None else [],
            "M2": ws.M2.tolist() if ws.M2 is not None else [],
        }

        return asdict(pc), vol_partial, surf_partial

    except Exception as e:
        pc.diverged = True
        pc.error = f"pass1: {e}\n{traceback.format_exc()}"
        return asdict(pc), empty, empty


def _merge_into_welford(global_w: WelfordAccum, partial: dict) -> None:
    """Chan-style merge of a partial Welford state into the global one."""
    if not partial or partial.get("n", 0) == 0:
        return
    n2 = partial["n"]
    mu2 = np.asarray(partial["mean"], dtype=np.float64)
    M2_2 = np.asarray(partial["M2"], dtype=np.float64)

    if global_w.mean is None:
        global_w.mean = mu2.copy()
        global_w.M2 = M2_2.copy()
        global_w.n = n2
        return

    n1 = global_w.n
    delta = mu2 - global_w.mean
    new_n = n1 + n2
    global_w.mean = global_w.mean + delta * (n2 / new_n)
    global_w.M2 = global_w.M2 + M2_2 + (delta ** 2) * (n1 * n2 / new_n)
    global_w.n = new_n


def run_pass1(
    npz_files: list[Path],
    out_dir: Path,
    workers: int,
) -> dict:
    """Returns the loaded norm_stats dict (also written to disk)."""
    print(f"[pass1] {len(npz_files)} cases, {workers} workers")
    t0 = time.time()

    per_case_records: list[dict] = []
    global_vol = WelfordAccum()
    global_surf = WelfordAccum()

    args = [str(p) for p in npz_files]

    if workers <= 1:
        iterator = (_pass1_one(a) for a in args)
    else:
        pool = mp.Pool(workers)
        iterator = pool.imap_unordered(_pass1_one, args, chunksize=1)

    try:
        for rec, vp, sp in tqdm(iterator, total=len(args), desc="pass1"):
            per_case_records.append(rec)
            _merge_into_welford(global_vol, vp)
            _merge_into_welford(global_surf, sp)
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    vol_mean, vol_std = global_vol.finalize()
    surf_mean, surf_std = global_surf.finalize()

    per_case_objs = [PerCaseStats(**r) for r in per_case_records]
    anomalies = detect_anomalies(per_case_objs)

    diverged_files = [r["file"] for r in per_case_records if r["diverged"]]

    norm_stats = {
        "schema": SCHEMA_VERSION,
        "volume_fields": {
            "names": physics.VOLUME_FIELD_NAMES,
            "mean": vol_mean.tolist(),
            "std": vol_std.tolist(),
            "n_points": int(global_vol.n),
        },
        "surface_fields": {
            "names": physics.SURFACE_FIELD_NAMES,
            "mean": surf_mean.tolist(),
            "std": surf_std.tolist(),
            "n_points": int(global_surf.n),
        },
        "physics_constants": {
            "G_gravity": physics.G_GRAVITY,
            "rho_default": DEFAULT_RHO,
        },
        "cut_box": {"x": CUT_X, "y": CUT_Y, "z": CUT_Z},
    }

    save_json(norm_stats, out_dir / "norm_stats.json")
    save_json(per_case_records, out_dir / "all_case_stats.json")
    save_json(diverged_files, out_dir / "diverged_cases.json")
    save_json(anomalies, out_dir / "anomaly_cases.json")

    elapsed = time.time() - t0
    print(
        f"[pass1] done in {elapsed:.0f}s — "
        f"{len(per_case_records) - len(diverged_files)} clean, "
        f"{len(diverged_files)} diverged, "
        f"{len(anomalies)} anomaly flags"
    )
    return norm_stats


# =====================================================================
# Pass 2: write PT files
# =====================================================================
_W_NORM_STATS: dict | None = None
_W_OUT_DIR: str | None = None
_W_SKIP_EXISTING: bool = False
_W_GRID_COORDS: np.ndarray | None = None
_W_ANOMALY_CASES: set[str] | None = None


def _pass2_init(
    norm_stats: dict,
    out_dir: str,
    skip_existing: bool,
    anomaly_cases: set[str] | None,
) -> None:
    global _W_NORM_STATS, _W_OUT_DIR, _W_SKIP_EXISTING, _W_GRID_COORDS, _W_ANOMALY_CASES
    _W_NORM_STATS = norm_stats
    _W_OUT_DIR = out_dir
    _W_SKIP_EXISTING = skip_existing
    _W_GRID_COORDS = build_grid_coords()
    _W_ANOMALY_CASES = anomaly_cases or set()

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _zscore(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((arr - mean) / std).astype(np.float32)


def _pass2_one(npz_path_str: str) -> tuple[str, str, str | None]:
    assert _W_NORM_STATS is not None and _W_OUT_DIR is not None
    npz_path = Path(npz_path_str)
    case_name = case_name_from_path(npz_path)
    out_path = Path(_W_OUT_DIR) / f"{case_name}.pt"

    if _W_SKIP_EXISTING and out_path.exists():
        return "skipped", case_name, None

    try:
        raw = _load_npz(npz_path)
        d = _extract_raw(raw)
        consts = _per_case_constants(d)

        # Cut volume to bbox
        mask = _bbox_mask(d["volume_pos"])
        volume_pos_cut = d["volume_pos"][mask]
        volume_raw_cut = d["volume_fields"][mask]
        if volume_pos_cut.shape[0] == 0:
            return "failed", case_name, "no volume points inside bbox"

        # Physics transforms (pre-zscore)
        vol_phys_cut = physics.transform_volume_fields(
            U=volume_raw_cut[:, 0:3],
            p_kin_vol=volume_raw_cut[:, 3],
            nut_vol=volume_raw_cut[:, 4],
            z_vol=volume_pos_cut[:, 2],
            U_ref=consts["U_ref"],
            L_scale=consts["L_scale"],
        )
        surf_phys = physics.transform_surface_fields(
            p_kin_surf=d["surface_fields"][:, 0],
            wss=d["surface_fields"][:, 1:4],
            z_surf=d["surface_pos"][:, 2],
            U_ref=consts["U_ref"],
        )

        # SDF over STL
        sdf = SDFComputer(d["stl_vertices"], d["stl_faces"])
        stl_face_normals = sdf.face_normals  # (F, 3) float32

        # 2m shell points (only side walls, with SDF + ground filters)
        shell_pts, shell_face_idx, shell_sdf = generate_shell_points(
            face_centers=d["stl_centers"],
            face_normals=stl_face_normals,
            offset_m=SHELL_OFFSET_M,
            sdf=sdf,
        )

        # IDW interpolate physical fields from cut volume → shell points (k=8)
        if shell_pts.shape[0] > 0:
            tree = build_tree(volume_pos_cut)
            shell_fields_phys, _idx, _dist = idw_query(
                tree=tree,
                source_values=vol_phys_cut,
                query_points=shell_pts,
                k=SHELL_IDW_K,
            )
            # SDF + gradient for shell points (we already have shell_sdf)
            _shell_sdf_check, shell_sdf_grad = sdf.sdf_and_grad(shell_pts)
        else:
            shell_fields_phys = np.zeros((0, vol_phys_cut.shape[1]), dtype=np.float32)
            shell_sdf_grad = np.zeros((0, 3), dtype=np.float32)
            shell_sdf = np.zeros((0,), dtype=np.float32)

        # SDF + gradient for cut volume points
        vol_sdf_cut, vol_sdf_grad_cut = sdf.sdf_and_grad(volume_pos_cut)

        # Concatenate volume + shell
        volume_pos_all = np.concatenate(
            [volume_pos_cut, shell_pts], axis=0
        ).astype(np.float32)
        volume_fields_all = np.concatenate(
            [vol_phys_cut, shell_fields_phys], axis=0
        ).astype(np.float32)
        volume_sdf_all = np.concatenate(
            [vol_sdf_cut, shell_sdf], axis=0
        ).astype(np.float32)
        volume_sdf_grad_all = np.concatenate(
            [vol_sdf_grad_cut, shell_sdf_grad], axis=0
        ).astype(np.float32)
        is_shell_all = np.concatenate(
            [
                np.zeros(volume_pos_cut.shape[0], dtype=bool),
                np.ones(shell_pts.shape[0], dtype=bool),
            ]
        )

        # Grid SDF + grad
        grid_sdf_flat, grid_sdf_grad_flat = sdf.sdf_and_grad(_W_GRID_COORDS)

        # Apply z-score using PASS-1 stats
        vol_mean = np.array(_W_NORM_STATS["volume_fields"]["mean"], dtype=np.float32)
        vol_std = np.array(_W_NORM_STATS["volume_fields"]["std"], dtype=np.float32)
        surf_mean = np.array(_W_NORM_STATS["surface_fields"]["mean"], dtype=np.float32)
        surf_std = np.array(_W_NORM_STATS["surface_fields"]["std"], dtype=np.float32)

        volume_fields_norm = _zscore(volume_fields_all, vol_mean, vol_std)
        surface_fields_norm = _zscore(surf_phys, surf_mean, surf_std)

        # ---- assemble PT ----
        gmeta = grid_metadata()

        # Surface SDF gradient is the surface normal (consistent with the
        # raycasting closest-point relation evaluated on the surface).
        is_anomaly = _W_ANOMALY_CASES is not None and case_name in _W_ANOMALY_CASES

        record: dict[str, Any] = {
            # metadata
            "schema": SCHEMA_VERSION,
            "case_name": case_name,
            "is_diverged": False,
            "is_anomaly": is_anomaly,
            # per-case constants
            "U_ref": consts["U_ref"],
            "rho": consts["rho"],
            "L_scale": consts["L_scale"],
            "z_max_building": consts["z_max_building"],
            "global_params": torch.from_numpy(d["global_params"]).float(),
            # STL geometry (physical meters)
            "stl_vertices": torch.from_numpy(d["stl_vertices"]).float(),
            "stl_faces": torch.from_numpy(d["stl_faces"]).long(),
            "stl_centers": torch.from_numpy(d["stl_centers"]).float(),
            "stl_areas": torch.from_numpy(d["stl_areas"]).float(),
            "stl_face_normals": torch.from_numpy(stl_face_normals).float(),
            "stl_centroid": torch.from_numpy(d["stl_vertices"].mean(axis=0)).float(),
            "stl_bbox_min": torch.from_numpy(d["stl_vertices"].min(axis=0)).float(),
            "stl_bbox_max": torch.from_numpy(d["stl_vertices"].max(axis=0)).float(),
            # surface (physical meters, fields z-scored)
            "surface_pos": torch.from_numpy(d["surface_pos"]).float(),
            "surface_normals": torch.from_numpy(d["surface_normals"]).float(),
            "surface_areas": torch.from_numpy(d["surface_areas"]).float(),
            "surface_fields": torch.from_numpy(surface_fields_norm).float(),
            "surface_field_names": physics.SURFACE_FIELD_NAMES,
            # volume = cut + 2m shell (physical meters; fields z-scored;
            # sdf in physical meters, signed; grad unit vector)
            "volume_pos": torch.from_numpy(volume_pos_all).float(),
            "volume_fields": torch.from_numpy(volume_fields_norm).float(),
            "volume_field_names": physics.VOLUME_FIELD_NAMES,
            "volume_sdf": torch.from_numpy(volume_sdf_all).float(),
            "volume_sdf_grad": torch.from_numpy(volume_sdf_grad_all).float(),
            "is_shell_point": torch.from_numpy(is_shell_all),
            "volume_bbox_min": torch.tensor([CUT_X[0], CUT_Y[0], CUT_Z[0]]).float(),
            "volume_bbox_max": torch.tensor([CUT_X[1], CUT_Y[1], CUT_Z[1]]).float(),
            # latent grid
            "grid_sdf": torch.from_numpy(grid_sdf_flat).float(),
            "grid_sdf_grad": torch.from_numpy(grid_sdf_grad_flat).float(),
            "grid_size": gmeta["grid_size"],
            "grid_x_range": gmeta["grid_x_range"],
            "grid_y_range": gmeta["grid_y_range"],
            "grid_z_range": gmeta["grid_z_range"],
            "grid_indexing": gmeta["grid_indexing"],
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, str(out_path))
        return "ok", case_name, None

    except Exception as e:
        return "failed", case_name, f"{e}\n{traceback.format_exc()}"


def run_pass2(
    npz_files: list[Path],
    norm_stats: dict,
    out_dir: Path,
    workers: int,
    skip_existing: bool,
    diverged_set: set[str] | None = None,
    anomaly_cases: set[str] | None = None,
) -> None:
    """Skip diverged files (by basename) and write PT for the rest."""
    diverged_set = diverged_set or set()
    anomaly_cases = anomaly_cases or set()
    todo = [str(p) for p in npz_files if p.name not in diverged_set]
    print(
        f"[pass2] {len(todo)} cases to write ({len(npz_files) - len(todo)} diverged skipped), "
        f"{workers} workers"
    )
    t0 = time.time()

    pt_dir = out_dir / "pt"
    pt_dir.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        _pass2_init(norm_stats, str(pt_dir), skip_existing, anomaly_cases)
        iterator = (_pass2_one(a) for a in todo)
    else:
        pool = mp.Pool(
            workers,
            initializer=_pass2_init,
            initargs=(norm_stats, str(pt_dir), skip_existing, anomaly_cases),
        )
        iterator = pool.imap_unordered(_pass2_one, todo, chunksize=1)

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    try:
        for status, name, msg in tqdm(iterator, total=len(todo), desc="pass2"):
            counts[status] += 1
            if status == "failed":
                failures.append((name, msg or ""))
    finally:
        if workers > 1:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    print(
        f"[pass2] done in {elapsed:.0f}s — "
        f"ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}"
    )
    if failures:
        save_json(
            [{"case": n, "error": m} for n, m in failures],
            out_dir / "pt_failures.json",
        )


def run_pt_pipeline(
    data_dir: Path,
    out_dir: Path,
    workers: int,
    skip_existing: bool,
) -> None:
    files = discover_npz(data_dir)
    print(f"Discovered {len(files)} NPZ files under {data_dir}")
    if not files:
        return

    norm_stats = run_pass1(files, out_dir, workers)

    diverged = set()
    diverged_path = out_dir / "diverged_cases.json"
    if diverged_path.exists():
        with open(diverged_path) as f:
            diverged = set(json.load(f))

    anomaly_cases: set[str] = set()
    anomaly_path = out_dir / "anomaly_cases.json"
    if anomaly_path.exists():
        with open(anomaly_path) as f:
            anomaly_list = json.load(f)
        anomaly_cases = {entry["case"] for entry in anomaly_list}

    run_pass2(
        files,
        norm_stats,
        out_dir,
        workers,
        skip_existing,
        diverged_set=diverged,
        anomaly_cases=anomaly_cases,
    )
