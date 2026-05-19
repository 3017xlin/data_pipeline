"""
Fixed dense grid for the latent encoder.

The grid is the same for every case (fixed domain, no per-case alignment):
  X ∈ [-550, 550], Y ∈ [-550, 550], Z ∈ [0, 160]  (physical meters)
  resolution 128 × 128 × 32  =  524 288 voxels

We store grid points in `ij_zyx` order, i.e.

    for k in 0..Nz-1:
        for j in 0..Ny-1:
            for i in 0..Nx-1:
                idx = k * Nx * Ny + j * Nx + i

This is what `np.reshape(flat, (Nz, Ny, Nx))` recovers, which lines up with
common volumetric backbones (channel-first 3D tensors are (B, C, Z, Y, X)).
"""
from __future__ import annotations

import numpy as np


GRID_NX = 128
GRID_NY = 128
GRID_NZ = 32
GRID_X_RANGE = (-550.0, 550.0)
GRID_Y_RANGE = (-550.0, 550.0)
GRID_Z_RANGE = (0.0, 160.0)
GRID_INDEXING = "ij_zyx"


def build_grid_coords() -> np.ndarray:
    """
    Returns the full set of voxel-center coordinates flattened in ij_zyx order:
    shape (Nx*Ny*Nz, 3) float32.
    """
    x = np.linspace(GRID_X_RANGE[0], GRID_X_RANGE[1], GRID_NX, dtype=np.float32)
    y = np.linspace(GRID_Y_RANGE[0], GRID_Y_RANGE[1], GRID_NY, dtype=np.float32)
    z = np.linspace(GRID_Z_RANGE[0], GRID_Z_RANGE[1], GRID_NZ, dtype=np.float32)

    # meshgrid with indexing='ij' so output is (Nz, Ny, Nx) when stacked z,y,x
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return np.ascontiguousarray(coords, dtype=np.float32)


def grid_metadata() -> dict:
    return {
        "grid_size": (GRID_NX, GRID_NY, GRID_NZ),
        "grid_x_range": GRID_X_RANGE,
        "grid_y_range": GRID_Y_RANGE,
        "grid_z_range": GRID_Z_RANGE,
        "grid_indexing": GRID_INDEXING,
    }
