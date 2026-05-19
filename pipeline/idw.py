"""
Inverse-distance-weighted interpolation on point clouds.

Uses scipy cKDTree for k-NN; weights are 1/d^power normalized to sum to 1.
A small epsilon avoids division-by-zero when a query coincides with a source.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def build_tree(points: np.ndarray) -> cKDTree:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    return cKDTree(points)


def idw_query(
    tree: cKDTree,
    source_values: np.ndarray,
    query_points: np.ndarray,
    k: int,
    power: float = 2.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        interpolated: (Q, D) — IDW result; D inferred from source_values
        neighbor_idx: (Q, k) — indices of k nearest neighbors
        neighbor_dist: (Q, k) — distances to k nearest neighbors
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    source_values = np.asarray(source_values)
    if source_values.ndim == 1:
        source_values = source_values[:, None]
        squeeze = True
    else:
        squeeze = False

    dist, idx = tree.query(query_points, k=k, workers=-1)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    # Weights = 1 / d^power, normalised
    safe = np.maximum(dist, eps)
    w = 1.0 / (safe ** power)
    w /= w.sum(axis=1, keepdims=True)

    # (Q, k, D)
    neighbor_values = source_values[idx]
    result = (w[..., None] * neighbor_values).sum(axis=1)

    if squeeze:
        result = result[:, 0]
    return result.astype(np.float32), idx, dist
