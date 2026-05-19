"""
Shell-point generation: synthetic mesh nodes that "thicken" the wall layer
so the downstream model sees field samples right next to the building.

Construction:
  1. Take the STL face centers whose outward normal is mostly horizontal
     (`|n_z| < SIDE_WALL_NORMAL_Z_MAX`) — these are side walls; we skip
     roofs / floors / ceilings.
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
  - Plotting (EDA phase 2): 0.5/2/3 m offsets for shell-vs-surface analysis.
"""
from __future__ import annotations

import numpy as np

from .sdf3d import SDFComputer

SIDE_WALL_NORMAL_Z_MAX = 0.5
DEFAULT_MIN_SDF_M = 1.0
DEFAULT_MIN_Z_M = 1.0


def select_side_wall_faces(
    face_normals: np.ndarray,
    z_max_thresh: float = SIDE_WALL_NORMAL_Z_MAX,
) -> np.ndarray:
    """Return indices of faces whose normal is mostly horizontal."""
    return np.where(np.abs(face_normals[:, 2]) < z_max_thresh)[0]


def generate_shell_points(
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    offset_m: float,
    sdf: SDFComputer,
    min_sdf_m: float = DEFAULT_MIN_SDF_M,
    min_z_m: float = DEFAULT_MIN_Z_M,
    side_wall_normal_z_max: float = SIDE_WALL_NORMAL_Z_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a single offset shell. Inputs are in physical meters.

    Returns:
        shell_pts:   (M, 3) float32 — valid shell points only
        face_index:  (M,)   int64   — index back into face_centers
        shell_sdf:   (M,)   float32 — SDF value at each kept shell point
    """
    side_idx = select_side_wall_faces(face_normals, side_wall_normal_z_max)
    if side_idx.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    side_centers = face_centers[side_idx]
    side_normals = face_normals[side_idx]

    # Offset along the outward normal
    candidates = (side_centers + side_normals * offset_m).astype(np.float32)
    cand_sdf = sdf.signed_distance(candidates)

    keep = (cand_sdf >= min_sdf_m) & (candidates[:, 2] >= min_z_m)

    return (
        np.ascontiguousarray(candidates[keep], dtype=np.float32),
        np.ascontiguousarray(side_idx[keep], dtype=np.int64),
        np.ascontiguousarray(cand_sdf[keep], dtype=np.float32),
    )
