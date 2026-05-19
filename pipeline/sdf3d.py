"""
Signed distance field + gradient via Open3D RaycastingScene.

The SDF convention here:
  - sdf > 0 outside the building, sdf < 0 inside
  - gradient is a unit vector pointing in the direction of *increasing* SDF
    (i.e. outward from the building surface for every query point)
  - On-surface queries (|sdf| < eps) fall back to the face normal of the
    closest triangle so the gradient stays defined.

All distances are in physical meters. STL must be a watertight (or at least
locally consistent) triangle mesh for the sign to be meaningful.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


_EPS_ON_SURFACE = 1e-6


def _as_float32_contig(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _as_int32_contig(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


class SDFComputer:
    """Open3D-backed SDF + gradient queries against a fixed STL mesh."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must be (V,3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be (F,3), got {faces.shape}")

        self._vertices = _as_float32_contig(vertices)
        self._faces = _as_int32_contig(faces)

        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(self._vertices)
        mesh.triangle.indices = o3d.core.Tensor(self._faces)

        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(mesh)

        self._face_normals = self._compute_face_normals()

    def _compute_face_normals(self) -> np.ndarray:
        v0 = self._vertices[self._faces[:, 0]]
        v1 = self._vertices[self._faces[:, 1]]
        v2 = self._vertices[self._faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-20)
        return (n / norms).astype(np.float32)

    @property
    def face_normals(self) -> np.ndarray:
        return self._face_normals

    def signed_distance(self, query: np.ndarray) -> np.ndarray:
        """Return only signed distance (N,) float32."""
        q = o3d.core.Tensor(_as_float32_contig(query))
        return self._scene.compute_signed_distance(q).numpy()

    def closest(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (closest_points (N,3), primitive_ids (N,)) on the mesh."""
        q = o3d.core.Tensor(_as_float32_contig(query))
        res = self._scene.compute_closest_points(q)
        return res["points"].numpy(), res["primitive_ids"].numpy()

    def sdf_and_grad(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            sdf:  (N,)   float32 — signed distance (positive outside)
            grad: (N, 3) float32 — unit vector pointing toward increasing SDF
        """
        query = _as_float32_contig(query)
        sdf = self.signed_distance(query)
        closest, prim_ids = self.closest(query)

        diff = query - closest
        dist = np.linalg.norm(diff, axis=1)
        on_surface = dist < _EPS_ON_SURFACE

        # Safe denominator for division
        safe_dist = np.maximum(dist, _EPS_ON_SURFACE)
        raw_grad = diff / safe_dist[:, None]

        # grad of signed distance points OUTWARD: for outside points (sdf>0),
        # diff already points outward; for inside points (sdf<0), diff points
        # inward so flip. Equivalent to sign(sdf) * diff/|diff|.
        sign = np.sign(sdf)
        # Points exactly on the surface have sign==0 which would zero the
        # gradient; we patch them below using the face normal.
        sign[on_surface] = 1.0
        grad = sign[:, None] * raw_grad

        if on_surface.any():
            grad[on_surface] = self._face_normals[prim_ids[on_surface]]

        # Renormalize defensively (numerical hygiene)
        norms = np.linalg.norm(grad, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-20)
        grad = grad / norms

        return sdf.astype(np.float32), grad.astype(np.float32)
