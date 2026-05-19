"""
extract.py v2 — Fixed floor detection + SDF validation
Changes from v1:
  - detect_floors: fixed 3m interval instead of gap detection
  - volume_sdf: filter out shell points that land inside buildings
  - Read global_params for U_REF per case
"""
import argparse, os, time, sys
import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from multiprocessing import Pool

G = 9.81
FLOOR_STEP = 3.0      # sample every 3m in z
X_BINS = 10
OFFSETS = [0.5, 2.0, 3.0]
IDW_K = 4
EPS = 1e-10
SDF_MIN = 0.1          # shell point must be at least this far from any building surface


def idw_with_sdf(vol_tree, p_vol, v_sdf, targets, k=IDW_K):
    """IDW interpolation with SDF validation.
    Returns (pressures, valid_mask)."""
    n = len(targets)
    d, idx = vol_tree.query(targets, k=k, workers=1)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    d = np.maximum(d, EPS)

    # Check SDF: all k neighbors must be in fluid domain (sdf > SDF_MIN)
    neighbor_sdf = v_sdf[idx]  # (n, k)
    valid = np.all(neighbor_sdf > SDF_MIN, axis=1)  # (n,)

    w = 1.0 / d
    w /= w.sum(axis=1, keepdims=True)
    p = (p_vol[idx] * w).sum(axis=1)

    return p, valid


def generate_floors(z_min, z_max, step=FLOOR_STEP):
    """Generate evenly spaced floor heights from z_min to z_max."""
    floors = np.arange(z_min + step / 2, z_max, step)
    if len(floors) == 0:
        floors = np.array([(z_min + z_max) / 2])
    return floors


def process_case(pt_path):
    try:
        data = torch.load(pt_path, weights_only=False, map_location='cpu')

        s_pos = data['surface_pos'].numpy().astype(np.float64)
        s_norm = data['surface_normals'].numpy().astype(np.float64)
        s_fields = data['surface_fields'].numpy().astype(np.float64)
        v_pos = data['volume_pos'].numpy().astype(np.float64)
        v_fields = data['volume_fields'].numpy().astype(np.float64)
        v_sdf = data['volume_sdf'].numpy().astype(np.float64)

        # Read per-case reference velocity
        gp = data['global_params'].numpy()
        u_ref = float(gp[0]) if gp[0] > 0 else 2.0

        p_surf = s_fields[:, 0]
        p_vol = v_fields[:, 3]

        # Detrend
        p_surf_aero = p_surf + G * s_pos[:, 2] / (u_ref ** 2)

        # Wall mask
        wall = np.abs(s_norm[:, 2]) < 0.5
        if wall.sum() < 50:
            return []

        w_pos = s_pos[wall]
        w_norm = s_norm[wall]
        w_p = p_surf_aero[wall]

        # DBSCAN clustering
        clusters = DBSCAN(eps=5.0, min_samples=10).fit_predict(w_pos[:, :2])

        # Volume KDTree (once per case)
        vol_tree = cKDTree(v_pos)

        rows = []

        for bldg_id in np.unique(clusters):
            if bldg_id == -1:
                continue

            bm = clusters == bldg_id
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

            # Overlapping x range
            x_lo = max(wi_pos[:, 0].min(), le_pos[:, 0].min())
            x_hi = min(wi_pos[:, 0].max(), le_pos[:, 0].max())
            if x_hi - x_lo < 2.0:
                continue

            x_samples = np.linspace(x_lo, x_hi, X_BINS + 2)[1:-1]

            # Fixed interval floor heights
            floor_zs = generate_floors(z_min, z_max, FLOOR_STEP)

            # KDTrees on xz
            wi_xz_tree = cKDTree(wi_pos[:, [0, 2]])
            le_xz_tree = cKDTree(le_pos[:, [0, 2]])

            # Collect all queries
            queries_xz = []
            floor_z_list = []
            for fz in floor_zs:
                for xs in x_samples:
                    queries_xz.append([xs, fz])
                    floor_z_list.append(fz)

            if not queries_xz:
                continue

            queries_xz = np.array(queries_xz)
            n_q = len(queries_xz)

            # Find nearest windward and leeward surface points
            d_wi, idx_wi = wi_xz_tree.query(queries_xz, k=1)
            d_le, idx_le = le_xz_tree.query(queries_xz, k=1)

            valid_dist = (d_wi < 5.0) & (d_le < 5.0)

            # Batch compute all shell points
            all_shell_pts = []
            query_indices = []

            for q in range(n_q):
                if not valid_dist[q]:
                    continue

                w_pt = wi_pos[idx_wi[q]]
                w_nrm = wi_norm[idx_wi[q]]
                l_pt = le_pos[idx_le[q]]
                l_nrm = le_norm[idx_le[q]]

                for o in OFFSETS:
                    all_shell_pts.append(w_pt + w_nrm * o)
                for o in OFFSETS:
                    all_shell_pts.append(l_pt + l_nrm * o)

                query_indices.append(q)

            if not query_indices:
                continue

            all_shell_pts = np.array(all_shell_pts)  # (n_valid * 6, 3)

            # Batch IDW with SDF validation
            p_shell_raw, shell_valid = idw_with_sdf(vol_tree, p_vol, v_sdf, all_shell_pts, k=IDW_K)

            # Detrend shell pressures
            p_shell_aero = p_shell_raw + G * all_shell_pts[:, 2] / (u_ref ** 2)

            # Assemble rows
            for i, q in enumerate(query_indices):
                # 6 shell values per query: wind[0.5,2,3], lee[0.5,2,3]
                base = i * 6
                s_valid = shell_valid[base:base + 6]

                # Skip if any shell point is invalid (inside building)
                if not s_valid.all():
                    continue

                p_b_wind = float(wi_p[idx_wi[q]])
                p_b_lee = float(le_p[idx_le[q]])

                fz = floor_z_list[q]
                x_frac = q % X_BINS

                rows.append([
                    p_b_wind,
                    float(p_shell_aero[base]),
                    float(p_shell_aero[base + 1]),
                    float(p_shell_aero[base + 2]),
                    p_b_lee,
                    float(p_shell_aero[base + 3]),
                    float(p_shell_aero[base + 4]),
                    float(p_shell_aero[base + 5]),
                    fz, z_max, thickness, x_frac
                ])

        return rows

    except Exception as e:
        print(f"ERROR {pt_path}: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', required=True)
    parser.add_argument('--out', default='tensor_data.npz')
    parser.add_argument('--workers', type=int, default=60)
    args = parser.parse_args()

    files = sorted([
        os.path.join(args.pt_dir, f)
        for f in os.listdir(args.pt_dir) if f.endswith('.pt')
    ])
    print(f"Found {len(files)} PT files, using {args.workers} workers")

    t0 = time.time()
    all_rows = []
    n_failed = 0

    with Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(process_case, files)):
            if rows:
                all_rows.extend(rows)
            else:
                n_failed += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(files)}] {elapsed:.0f}s, {len(all_rows):,} samples, {n_failed} failed")
                sys.stdout.flush()

    data = np.array(all_rows, dtype=np.float64)
    print(f"\nFinal tensor shape: {data.shape}")
    print(f"Columns: p_b_w, p_a05_w, p_a2_w, p_a3_w, p_b_l, p_a05_l, p_a2_l, p_a3_l, floor_z, z_max, thickness, x_frac")

    print(f"\nz_max range: {data[:,9].min():.1f} - {data[:,9].max():.1f} m")
    print(f"thickness range: {data[:,10].min():.1f} - {data[:,10].max():.1f} m")
    print(f"floor_z range: {data[:,8].min():.1f} - {data[:,8].max():.1f} m")
    print(f"Unique floor_z count: {len(np.unique(np.round(data[:,8], 1)))}")

    t = data[:, 10]
    print(f"Thickness percentiles: 25%={np.percentile(t,25):.1f}, 50%={np.percentile(t,50):.1f}, 75%={np.percentile(t,75):.1f}")

    # Count how many samples were filtered by SDF
    total_possible = len(files) * 10 * 10  # rough estimate
    print(f"SDF filter: kept {len(data):,} samples")

    np.savez_compressed(args.out, data=data)
    print(f"\nSaved to {args.out} ({os.path.getsize(args.out)/1024/1024:.1f} MB)")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()