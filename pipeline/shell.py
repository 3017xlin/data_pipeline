"""
Shell-point generation: synthetic mesh nodes that "thicken" the wall layer
so the downstream model sees field samples right next to the building.

Construction — **inputs are the CFD wall mesh cell centers** (the points
with real CFD field values, NPZ `surface_mesh_centers` / `surface_normals`),
NOT the STL face centers. The CFD wall mesh density matches the simulation
resolution exactly so the resulting shell tracks the same sampling pattern.

  1. Take the CFD wall cell centers whose outward normal is mostly
     horizontal (`|n_z| < SIDE_WALL_NORMAL_Z_MAX`) — these are side walls;
     we skip roofs / floors / overhangs.
  2. Offset each center along its outward normal by `offset_m` meters.
  3. Validate the resulting point:
       a. Open3D SDF at the offset point must be >= `min_sdf_m`. This drops
          points that fold back inside a neighbouring building in concave
          gaps between towers.
       b. Height must be >= `min_z_m` — points near the ground (z < 1 m)
          are typically inside boundary layer mesh and produce degenerate
          IDW.

The function is used in two places:
  - PT generation: just the 2 m offset (one shell), high-fidelity IDW (k=8).
  - Plotting (EDA phase 1): 0.5/2/3 m offsets for shell-vs-surface analysis.
"""
from __future__ import annotations

import numpy as np

from .sdf3d import SDFComputer

SIDE_WALL_NORMAL_Z_MAX = 0.5
DEFAULT_MIN_SDF_M = 1.0
DEFAULT_MIN_Z_M = 1.0


def select_side_wall_indices(
    normals: np.ndarray,
    z_max_thresh: float = SIDE_WALL_NORMAL_Z_MAX,
) -> np.ndarray:
    """Indices of points whose outward normal is mostly horizontal."""
    return np.where(np.abs(normals[:, 2]) < z_max_thresh)[0]


def generate_shell_points(
    base_points: np.ndarray,
    base_normals: np.ndarray,
    offset_m: float,
    sdf: SDFComputer,
    min_sdf_m: float = DEFAULT_MIN_SDF_M,
    min_z_m: float = DEFAULT_MIN_Z_M,
    side_wall_normal_z_max: float = SIDE_WALL_NORMAL_Z_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a single offset shell from the CFD wall mesh.

    Args:
        base_points:  (S, 3) float — typically NPZ `surface_mesh_centers`.
        base_normals: (S, 3) float — matching outward unit normals (NPZ
                                     `surface_normals`).
        offset_m:     scalar distance to deviate along the normal.
        sdf:          SDFComputer built against the case's STL.
        min_sdf_m:    drop shell points whose actual SDF is below this.
        min_z_m:      drop shell points whose z is below this.

    Returns:
        shell_pts:   (M, 3) float32 — valid shell points only
        base_index:  (M,)   int64   — index into base_points
        shell_sdf:   (M,)   float32 — SDF at each kept shell point
    """
    side_idx = select_side_wall_indices(base_normals, side_wall_normal_z_max)
    if side_idx.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    side_pts = base_points[side_idx]
    side_normals = base_normals[side_idx]

    # Offset along the outward normal
    candidates = (side_pts + side_normals * offset_m).astype(np.float32)
    cand_sdf = sdf.signed_distance(candidates)

    keep = (cand_sdf >= min_sdf_m) & (candidates[:, 2] >= min_z_m)

    return (
        np.ascontiguousarray(candidates[keep], dtype=np.float32),
        np.ascontiguousarray(side_idx[keep], dtype=np.int64),
        np.ascontiguousarray(cand_sdf[keep], dtype=np.float32),
    )
