"""
Physical pre-transforms applied BEFORE the global z-score.

Conventions (kept consistent with eda_phase1.py):
  - Free stream blows in the -Y direction with magnitude U_ref, so the
    streamwise residual is `Uy_residual = Uy + U_ref` (≈ 0 in the far field).
  - The NPZ pressure field is OpenFOAM kinematic pressure (units m²/s²,
    i.e. P_static / rho). The non-dimensional, hydrostatically detrended
    aerodynamic pressure is

        p_aero = (p_kin + g·z) / U_ref²

    which becomes Cp-like (O(1)) and removes the hydrostatic gradient.
  - Turbulent viscosity is compressed with log(nut · L_scale) where
    L_scale is the per-case building height (z_max). nut is non-negative
    by physics; we clamp tiny values with EPS_NUT before the log.
"""
from __future__ import annotations

import numpy as np

G_GRAVITY = 9.81
EPS_NUT = 1e-12


def p_aero(p_kin: np.ndarray, z: np.ndarray, U_ref: float) -> np.ndarray:
    """Non-dimensional, hydrostatically detrended pressure (Cp-like)."""
    return (p_kin + G_GRAVITY * z) / (U_ref ** 2)


def uy_residual(uy: np.ndarray, U_ref: float) -> np.ndarray:
    """Streamwise residual: subtracts the (-Y) free-stream component."""
    return uy + U_ref


def log_nut(nut: np.ndarray, L_scale: float) -> np.ndarray:
    """log(nut · L_scale) with floor at EPS_NUT to avoid -inf."""
    return np.log(np.maximum(nut * L_scale, EPS_NUT))


def transform_volume_fields(
    U: np.ndarray,  # (N, 3) [Ux, Uy, Uz]
    p_kin_vol: np.ndarray,  # (N,)
    nut_vol: np.ndarray,  # (N,)
    z_vol: np.ndarray,  # (N,)
    U_ref: float,
    L_scale: float,
) -> np.ndarray:
    """
    Returns (N, 5) float32 array with columns
        [Ux, Uy_residual, Uz, p_aero, log_nut]
    """
    out = np.empty((U.shape[0], 5), dtype=np.float32)
    out[:, 0] = U[:, 0]
    out[:, 1] = uy_residual(U[:, 1], U_ref)
    out[:, 2] = U[:, 2]
    out[:, 3] = p_aero(p_kin_vol, z_vol, U_ref)
    out[:, 4] = log_nut(nut_vol, L_scale)
    return out


def transform_surface_fields(
    p_kin_surf: np.ndarray,  # (S,)
    wss: np.ndarray,  # (S, 3)
    z_surf: np.ndarray,  # (S,)
    U_ref: float,
) -> np.ndarray:
    """
    Returns (S, 4) float32 array with columns
        [p_aero, wss_x_nondim, wss_y_nondim, wss_z_nondim]

    Wall shear stress (kinematic, units m²/s²) is non-dimensionalised by
    U_ref² so it sits in the same regime as p_aero.
    """
    out = np.empty((p_kin_surf.shape[0], 4), dtype=np.float32)
    out[:, 0] = p_aero(p_kin_surf, z_surf, U_ref)
    out[:, 1:4] = wss / (U_ref ** 2)
    return out


VOLUME_FIELD_NAMES = ["Ux", "Uy_residual", "Uz", "p_aero", "log_nut"]
SURFACE_FIELD_NAMES = ["p_aero", "wss_x", "wss_y", "wss_z"]
