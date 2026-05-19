"""
End-to-end smoke test: synthesize a single fake "case" NPZ, run pass1 +
pass2 on it, verify the resulting PT has every field promised by the
schema with the right shapes / dtypes.

The fake building is a centred axis-aligned cube. It gives us a
non-trivial SDF (interior is negative, side normals are unit vectors
along ±X/±Y) and lets us assert the shell geometry directly.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.transform import run_pt_pipeline


def _box_mesh(cx, cy, hx, hy, hz):
    """Build a closed box: 8 verts, 12 outward-facing triangles."""
    verts = np.array([
        [cx - hx, cy - hy, 0.0],
        [cx + hx, cy - hy, 0.0],
        [cx + hx, cy + hy, 0.0],
        [cx - hx, cy + hy, 0.0],
        [cx - hx, cy - hy, hz],
        [cx + hx, cy - hy, hz],
        [cx + hx, cy + hy, hz],
        [cx - hx, cy + hy, hz],
    ], dtype=np.float32)
    faces = np.array([
        # Bottom (-Z), wound to face -Z
        [0, 2, 1], [0, 3, 2],
        # Top (+Z)
        [4, 5, 6], [4, 6, 7],
        # -Y face
        [0, 1, 5], [0, 5, 4],
        # +Y face
        [3, 6, 2], [3, 7, 6],
        # -X face
        [0, 4, 7], [0, 7, 3],
        # +X face
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return verts, faces


def _face_centers_areas_normals(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    centers = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    areas = 0.5 * norms[:, 0]
    normals = cross / np.maximum(norms, 1e-20)
    return centers, areas, normals


def _make_fake_npz(path: Path):
    verts, faces = _box_mesh(cx=0, cy=0, hx=20, hy=15, hz=40)
    centers, areas, normals = _face_centers_areas_normals(verts, faces)

    # Surface samples: re-use STL centers, perturbed slightly so they
    # aren't EXACTLY on the mesh (more realistic CFD-like output)
    rng = np.random.default_rng(42)
    surf_pos = centers + rng.normal(0, 0.05, centers.shape).astype(np.float32)
    surf_normals = normals.copy()
    surf_areas = areas.copy()
    # surface_fields = [p (kinematic m²/s²), wss_x, wss_y, wss_z]
    surf_fields = rng.normal(0, 5, (centers.shape[0], 4)).astype(np.float32)

    # Volume: random points in the cut box, but biased away from the cube
    n_vol = 5000
    vx = rng.uniform(-550, 550, n_vol).astype(np.float32)
    vy = rng.uniform(-550, 550, n_vol).astype(np.float32)
    vz = rng.uniform(0, 160, n_vol).astype(np.float32)
    vol_pos = np.stack([vx, vy, vz], axis=1)
    # Drop any volume points "inside" the box for a more realistic SDF
    inside = (np.abs(vx) < 20) & (np.abs(vy) < 15) & (vz < 40)
    vol_pos = vol_pos[~inside]
    n_vol = vol_pos.shape[0]
    # volume_fields = [Ux, Uy, Uz, p_kin, nut]
    vol_fields = np.column_stack([
        rng.normal(0, 0.3, n_vol),     # Ux
        rng.normal(-2.0, 0.5, n_vol),  # Uy (around -U_ref)
        rng.normal(0, 0.2, n_vol),     # Uz
        rng.normal(0, 3.0, n_vol),     # p_kin
        np.abs(rng.normal(0.01, 0.005, n_vol)),  # nut > 0
    ]).astype(np.float32)

    global_params = np.array([2.0, 1.225], dtype=np.float32)

    np.savez_compressed(
        path,
        filename=path.stem,
        stl_coordinates=verts,
        stl_centers=centers.astype(np.float32),
        stl_faces=faces,
        stl_areas=areas.astype(np.float32),
        surface_mesh_centers=surf_pos,
        surface_normals=surf_normals.astype(np.float32),
        surface_areas=surf_areas.astype(np.float32),
        surface_fields=surf_fields,
        volume_mesh_centers=vol_pos,
        volume_fields=vol_fields,
        global_params_values=global_params,
    )


def main():
    tmp = Path(tempfile.mkdtemp(prefix="hdb_smoke_"))
    data_dir = tmp / "data"
    out_dir = tmp / "out"
    data_dir.mkdir()
    print(f"workdir: {tmp}")

    # 3 synthetic cases so Welford has real data to merge
    for i in range(3):
        _make_fake_npz(data_dir / f"case_HDB_{i:03d}.npz")

    run_pt_pipeline(data_dir, out_dir, workers=1, skip_existing=False)

    # Inspect first PT
    pt_files = sorted((out_dir / "pt").glob("*.pt"))
    assert len(pt_files) == 3, f"expected 3 PT files, got {len(pt_files)}"
    rec = torch.load(pt_files[0], weights_only=False)

    expected = {
        "schema": str,
        "case_name": str,
        "is_diverged": bool,
        "is_anomaly": bool,
        "U_ref": float,
        "rho": float,
        "L_scale": float,
        "z_max_building": float,
        "global_params": torch.Tensor,
        "stl_vertices": torch.Tensor,
        "stl_faces": torch.Tensor,
        "stl_centers": torch.Tensor,
        "stl_areas": torch.Tensor,
        "stl_face_normals": torch.Tensor,
        "stl_centroid": torch.Tensor,
        "stl_bbox_min": torch.Tensor,
        "stl_bbox_max": torch.Tensor,
        "surface_pos": torch.Tensor,
        "surface_normals": torch.Tensor,
        "surface_areas": torch.Tensor,
        "surface_fields": torch.Tensor,
        "surface_field_names": list,
        "volume_pos": torch.Tensor,
        "volume_fields": torch.Tensor,
        "volume_field_names": list,
        "volume_sdf": torch.Tensor,
        "volume_sdf_grad": torch.Tensor,
        "is_shell_point": torch.Tensor,
        "volume_bbox_min": torch.Tensor,
        "volume_bbox_max": torch.Tensor,
        "grid_sdf": torch.Tensor,
        "grid_sdf_grad": torch.Tensor,
        "grid_size": tuple,
        "grid_x_range": tuple,
        "grid_y_range": tuple,
        "grid_z_range": tuple,
        "grid_indexing": str,
    }
    for k, t in expected.items():
        assert k in rec, f"missing key: {k}"
        assert isinstance(rec[k], t), f"wrong type for {k}: {type(rec[k])}, expected {t}"

    # Shape sanity
    N = rec["volume_pos"].shape[0]
    assert rec["volume_fields"].shape == (N, 5)
    assert rec["volume_sdf"].shape == (N,)
    assert rec["volume_sdf_grad"].shape == (N, 3)
    assert rec["is_shell_point"].shape == (N,)
    assert rec["grid_sdf"].shape == (128 * 128 * 32,)
    assert rec["grid_sdf_grad"].shape == (128 * 128 * 32, 3)

    # SDF gradient should be unit length
    g = rec["volume_sdf_grad"].numpy()
    g_norm = np.linalg.norm(g, axis=1)
    assert np.allclose(g_norm, 1.0, atol=1e-3), f"grad not unit: {g_norm.min()} – {g_norm.max()}"

    g2 = rec["grid_sdf_grad"].numpy()
    g2_norm = np.linalg.norm(g2, axis=1)
    assert np.allclose(g2_norm, 1.0, atol=1e-3), f"grid grad not unit"

    # Shell points exist and all carry is_shell=True
    shell_idx = rec["is_shell_point"].numpy()
    assert shell_idx.sum() > 0, "no shell points generated"
    assert shell_idx.dtype == bool

    # Shell positions should all be ~2m from STL (within tolerance because
    # we filter SDF >= 1.0 and the cube has flat faces with exact normals)
    shell_pos = rec["volume_pos"][shell_idx].numpy()
    shell_sdf = rec["volume_sdf"][shell_idx].numpy()
    assert shell_sdf.min() >= 1.0, f"shell sdf {shell_sdf.min()} violates filter"
    # Side-wall offsets only — z must be >= 1m
    assert shell_pos[:, 2].min() >= 1.0

    # Fields are z-scored: per-corpus mean ≈ 0, std ≈ 1
    norm_stats = json.loads((out_dir / "norm_stats.json").read_text())
    vol_mean = np.array(norm_stats["volume_fields"]["mean"])
    vol_std = np.array(norm_stats["volume_fields"]["std"])

    # Just sanity: pulled values are finite
    assert torch.isfinite(rec["volume_fields"]).all()
    assert torch.isfinite(rec["surface_fields"]).all()
    assert torch.isfinite(rec["volume_sdf"]).all()
    assert torch.isfinite(rec["volume_sdf_grad"]).all()

    # JSON artifacts present
    for f in ["norm_stats.json", "all_case_stats.json",
              "diverged_cases.json", "anomaly_cases.json"]:
        assert (out_dir / f).exists(), f"missing {f}"

    print("OK: all assertions passed.")
    print(f"  N_volume_total = {N}")
    print(f"  N_shell        = {int(shell_idx.sum())}")
    print(f"  norm_stats vol mean = {vol_mean}")
    print(f"  norm_stats vol std  = {vol_std}")

    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
