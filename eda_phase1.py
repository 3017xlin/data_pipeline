"""
EDA HPC version: qualitative + quantitative combined
Runs on all cases, outputs aggregate stats + plots only
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count
import json
import time
import sys

# ============ CONFIG ============
DATA_DIR = Path("/home/ylin041/eda_data")
OUT_DIR = Path("/home/ylin041/eda_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RHO = 1.225
U_REF = 2.0
G = 9.81
OFFSETS = [0.5, 2.0, 3.0]
IDW_K = 4

# Divergence thresholds
MAX_UY = 10.0
MAX_P_SURF = 10000.0

N_WORKERS = int(os.environ.get("TOTAL_CORES", cpu_count()))
import os


# ============ HELPERS ============
def idw_interpolate(tree, values, targets, k=4):
    dists, idxs = tree.query(targets, k=k, workers=1)  # workers=1 inside subprocess
    if k == 1:
        dists, idxs = dists[:, None], idxs[:, None]
    dists = np.maximum(dists, 1e-10)
    w = 1.0 / dists
    w /= w.sum(axis=1, keepdims=True)
    return (values[idxs] * w).sum(axis=1)


def process_single_case(fpath):
    """Process one NPZ file, return dict of stats or None if diverged."""
    try:
        npz = np.load(fpath, allow_pickle=True)
        case_name = str(npz["filename"])

        sf = npz["surface_fields"]
        vf = npz["volume_fields"]
        sc = npz["surface_mesh_centers"]
        vc = npz["volume_mesh_centers"]
        sn = npz["surface_normals"]
        sa = npz["surface_areas"]
        stl = npz["stl_coordinates"]
        gp = npz["global_params_values"]

        # ---- Divergence check ----
        if (abs(vf[:, 1].min()) > MAX_UY or abs(vf[:, 1].max()) > MAX_UY or
            abs(sf[:, 0].min()) > MAX_P_SURF or abs(sf[:, 0].max()) > MAX_P_SURF):
            npz.close()
            return {"file": fpath.name, "case": case_name, "diverged": True}

        z_surf = sc[:, 2]
        z_vol = vc[:, 2]
        ny = sn[:, 1]

        # Detrend
        p_surf_raw = sf[:, 0]
        p_vol_raw = vf[:, 3]
        p_surf_aero = p_surf_raw + G * z_surf / (U_REF ** 2)
        p_vol_aero = p_vol_raw + G * z_vol / (U_REF ** 2)

        windward = ny < -0.5
        leeward = ny > 0.5
        wss_mag = np.sqrt((sf[:, 1:4] ** 2).sum(axis=1))

        p_wind = p_surf_aero[windward].mean() if windward.sum() > 0 else np.nan
        p_lee = p_surf_aero[leeward].mean() if leeward.sum() > 0 else np.nan

        rec = {
            "file": fpath.name,
            "case": case_name,
            "diverged": False,
            "z_max": float(stl[:, 2].max()),
            "n_stl_pts": len(stl),
            "n_surf_cells": len(sf),
            "n_vol_cells": len(vf),
            "surf_area_mean": float(sa.mean()),
            "surf_area_std": float(sa.std()),
            "p_aero_mean": float(p_surf_aero.mean()),
            "p_aero_std": float(p_surf_aero.std()),
            "p_aero_min": float(p_surf_aero.min()),
            "p_aero_max": float(p_surf_aero.max()),
            "delta_p_wl": float(p_wind - p_lee) if not (np.isnan(p_wind) or np.isnan(p_lee)) else None,
            "wss_mag_mean": float(wss_mag.mean()),
            "wss_mag_max": float(wss_mag.max()),
            "Uy_mean": float(vf[:, 1].mean()),
            "Uy_min": float(vf[:, 1].min()),
            "Uy_max": float(vf[:, 1].max()),
            "p_vol_aero_mean": float(p_vol_aero.mean()),
            "nut_mean": float(vf[:, 4].mean()),
            "nut_max": float(vf[:, 4].max()),
        }

        # Shell analysis
        vol_tree = cKDTree(vc)
        surf_tree = cKDTree(sc)
        _, surf_neighbor_idx = surf_tree.query(sc, k=5, workers=1)

        for offset in OFFSETS:
            tag = f"{offset}m"
            shell_pts = sc + sn * offset

            dists_to_vol, _ = vol_tree.query(shell_pts, k=1, workers=1)
            p_shell_raw = idw_interpolate(vol_tree, p_vol_raw, shell_pts, k=IDW_K)
            z_shell = shell_pts[:, 2]
            p_shell_aero = p_shell_raw + G * z_shell / (U_REF ** 2)
            delta_p = p_shell_aero - p_surf_aero

            neighbor_p = p_shell_aero[surf_neighbor_idx[:, 1:]]
            local_var = float(np.mean(np.std(neighbor_p, axis=1)))

            corr, _ = pearsonr(p_surf_aero, p_shell_aero)

            if windward.sum() > 0 and leeward.sum() > 0:
                shell_dp = float(p_shell_aero[windward].mean() - p_shell_aero[leeward].mean())
                surf_dp = float(p_wind - p_lee)
                consistent = 1.0 if (shell_dp > 0) == (surf_dp > 0) else 0.0
            else:
                shell_dp = None
                consistent = None

            rec[f"shell_{tag}_dist_mean"] = float(np.mean(dists_to_vol))
            rec[f"shell_{tag}_smoothness"] = local_var
            rec[f"shell_{tag}_corr"] = float(corr)
            rec[f"shell_{tag}_delta_p_mean"] = float(delta_p.mean())
            rec[f"shell_{tag}_delta_p_std"] = float(delta_p.std())
            rec[f"shell_{tag}_wind_consistent"] = consistent

        npz.close()
        return rec

    except Exception as e:
        return {"file": fpath.name, "case": "ERROR", "diverged": True, "error": str(e)}


# ============ MAIN ============
if __name__ == "__main__":
    all_files = sorted(DATA_DIR.glob("case_HDB_*.npz"))
    print(f"Found {len(all_files)} NPZ files")
    print(f"Using {N_WORKERS} workers\n")

    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        results = []
        for i, rec in enumerate(pool.imap_unordered(process_single_case, all_files)):
            results.append(rec)
            if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 60
                print(f"  [{i+1}/{len(all_files)}] {elapsed:.0f}s elapsed, {rate:.0f} cases/min")
                sys.stdout.flush()

    # Split clean / diverged
    clean = [r for r in results if not r.get("diverged", False)]
    diverged = [r for r in results if r.get("diverged", False)]

    print(f"\nDone processing: {len(clean)} clean, {len(diverged)} diverged")

    # Save all results
    with open(OUT_DIR / "all_case_stats.json", "w") as fp:
        json.dump(clean, fp, indent=2, default=str)
    with open(OUT_DIR / "diverged_cases.json", "w") as fp:
        json.dump([r["file"] for r in diverged], fp, indent=2)

    # ============ AGGREGATE & PLOT ============
    skip_keys = {"file", "case", "diverged", "error"}
    numeric_keys = [k for k in clean[0] if k not in skip_keys]
    data = {}
    for k in numeric_keys:
        vals = [float(r[k]) if r.get(k) is not None else np.nan for r in clean]
        data[k] = np.array(vals)

    fig_num = 0
    def savefig(name):
        global fig_num
        fig_num += 1
        path = OUT_DIR / f"{fig_num:02d}_{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    # 1. Building height
    plt.figure(figsize=(8, 4))
    plt.hist(data["z_max"], bins=30, edgecolor='black')
    plt.xlabel("Max building height (m)")
    plt.ylabel("Count")
    plt.title(f"Building Height Distribution (n={len(clean)})")
    plt.grid(True, alpha=0.3)
    savefig("z_max_distribution")

    # 2. Mesh sizes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes,
        ["n_stl_pts", "n_surf_cells", "n_vol_cells"],
        ["STL vertices", "Surface cells", "Volume cells"]):
        ax.hist(data[key], bins=30, edgecolor='black')
        ax.set_xlabel(label)
        ax.set_title(f"{label} per case")
    plt.tight_layout()
    savefig("mesh_size_distributions")

    # 3. Surface pressure stats
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes,
        ["p_aero_mean", "p_aero_min", "p_aero_max"],
        ["Mean p_aero", "Min p_aero", "Max p_aero"]):
        ax.hist(data[key], bins=30, edgecolor='black')
        ax.set_xlabel(label)
        ax.set_title(label)
    plt.tight_layout()
    savefig("p_aero_stats")

    # 4. Delta P W-L
    dp = data["delta_p_wl"]
    dp_clean = dp[~np.isnan(dp)]
    plt.figure(figsize=(8, 4))
    plt.hist(dp_clean, bins=30, edgecolor='black')
    plt.xlabel("ΔP windward-leeward")
    plt.title("Aerodynamic Pressure Difference")
    if len(dp_clean) > 0:
        plt.axvline(dp_clean.mean(), color='r', linestyle='--', label=f"mean={dp_clean.mean():.3f}")
        plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("delta_p_wl")

    # 5. ΔP vs building height
    valid = ~np.isnan(dp)
    plt.figure(figsize=(8, 5))
    plt.scatter(data["z_max"][valid], dp[valid], s=10, alpha=0.5)
    plt.xlabel("Max building height (m)")
    plt.ylabel("ΔP windward-leeward")
    plt.title("Pressure Difference vs Building Height")
    plt.grid(True, alpha=0.3)
    savefig("delta_p_vs_height")

    # 6. Shell correlation comparison
    plt.figure(figsize=(8, 5))
    for o in OFFSETS:
        vals = data[f"shell_{o}m_corr"]
        plt.scatter(np.full(len(vals), o), vals, s=10, alpha=0.3)
    means = [np.nanmean(data[f"shell_{o}m_corr"]) for o in OFFSETS]
    plt.plot(OFFSETS, means, 'k-o', linewidth=2, markersize=8, label="Mean")
    plt.xlabel("Offset (m)")
    plt.ylabel("Pearson r")
    plt.title("Shell-Surface Correlation by Offset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("shell_correlation")

    # 7. Shell smoothness comparison
    plt.figure(figsize=(8, 5))
    for o in OFFSETS:
        vals = data[f"shell_{o}m_smoothness"]
        plt.scatter(np.full(len(vals), o), vals, s=10, alpha=0.3)
    means = [np.nanmean(data[f"shell_{o}m_smoothness"]) for o in OFFSETS]
    plt.plot(OFFSETS, means, 'k-o', linewidth=2, markersize=8, label="Mean")
    plt.xlabel("Offset (m)")
    plt.ylabel("Spatial smoothness")
    plt.title("Shell Smoothness by Offset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("shell_smoothness")

    # 8. Shell dist to vol cell
    plt.figure(figsize=(8, 5))
    for o in OFFSETS:
        vals = data[f"shell_{o}m_dist_mean"]
        plt.scatter(np.full(len(vals), o), vals, s=10, alpha=0.3)
    means = [np.nanmean(data[f"shell_{o}m_dist_mean"]) for o in OFFSETS]
    plt.plot(OFFSETS, means, 'k-o', linewidth=2, markersize=8, label="Mean")
    plt.axhline(y=3.125/2, color='r', linestyle='--', label='Half cell (1.56m)')
    plt.xlabel("Offset (m)")
    plt.ylabel("Dist to nearest vol cell (m)")
    plt.title("Shell Proximity to Volume Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("shell_dist_to_vol")

    # 9. Correlation vs building height
    plt.figure(figsize=(8, 5))
    for o in OFFSETS:
        plt.scatter(data["z_max"], data[f"shell_{o}m_corr"], s=10, alpha=0.3, label=f"{o}m")
    plt.xlabel("Building height (m)")
    plt.ylabel("Pearson r")
    plt.title("Shell Correlation vs Building Height")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig("correlation_vs_height")

    # 10. nut and WSS
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data["nut_max"], bins=30, edgecolor='black')
    axes[0].set_xlabel("Max nut")
    axes[0].set_title("Turbulent Viscosity Max")
    axes[1].hist(data["wss_mag_mean"], bins=30, edgecolor='black')
    axes[1].set_xlabel("Mean |WSS|")
    axes[1].set_title("Wall Shear Stress")
    plt.tight_layout()
    savefig("nut_wss")

    # 11. Surface area uniformity
    plt.figure(figsize=(8, 4))
    plt.scatter(data["surf_area_mean"], data["surf_area_std"], s=10, alpha=0.5)
    plt.xlabel("Mean cell area")
    plt.ylabel("Std cell area")
    plt.title("Surface Mesh Uniformity")
    plt.grid(True, alpha=0.3)
    savefig("surface_area_uniformity")

    # ============ ANOMALY DETECTION ============
    print("\n=== ANOMALY CHECK (3-sigma) ===")
    anomalies = []
    check_keys = ["p_aero_mean", "p_aero_min", "p_aero_max",
                  "nut_max", "delta_p_wl", "wss_mag_max"]
    for key in check_keys:
        vals = data[key]
        vc = vals[~np.isnan(vals)]
        if len(vc) < 10:
            continue
        mu, sigma = vc.mean(), vc.std()
        if sigma == 0:
            continue
        for j, r in enumerate(clean):
            v = r.get(key)
            if v is not None and not np.isnan(float(v)) and abs(float(v) - mu) > 3 * sigma:
                anomalies.append({"file": r["file"], "metric": key, "value": float(v),
                                  "mean": float(mu), "threshold": float(3 * sigma)})

    with open(OUT_DIR / "anomalies.json", "w") as fp:
        json.dump(anomalies, fp, indent=2)
    print(f"  Found {len(anomalies)} anomalies across {len(set(a['file'] for a in anomalies))} cases")

    # ============ SUMMARY ============
    summary_lines = []
    summary_lines.append(f"Total files: {len(all_files)}")
    summary_lines.append(f"Clean: {len(clean)}, Diverged: {len(diverged)}")
    summary_lines.append(f"Anomalies: {len(anomalies)}")
    summary_lines.append("")
    summary_lines.append("=== SHELL OFFSET COMPARISON ===")
    summary_lines.append(f"{'Metric':<35} {'0.5m':>10} {'2.0m':>10} {'3.0m':>10}")
    summary_lines.append("-" * 67)

    for label, key_tmpl in [
        ("Mean dist to vol cell (m)", "shell_{}m_dist_mean"),
        ("Spatial smoothness (std)", "shell_{}m_smoothness"),
        ("Pearson corr with surface", "shell_{}m_corr"),
        ("Mean delta_p", "shell_{}m_delta_p_mean"),
        ("Std delta_p", "shell_{}m_delta_p_std"),
    ]:
        vals = []
        for o in OFFSETS:
            key = key_tmpl.format(o)
            arr = data[key]
            vals.append(f"{np.nanmean(arr):.4f}")
        summary_lines.append(f"{label:<35} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    # W-L consistency
    for o in OFFSETS:
        key = f"shell_{o}m_wind_consistent"
        arr = data[key]
        arr_clean = arr[~np.isnan(arr)]
    wl_vals = []
    for o in OFFSETS:
        arr = data[f"shell_{o}m_wind_consistent"]
        arr_clean = arr[~np.isnan(arr)]
        wl_vals.append(f"{arr_clean.mean()*100:.1f}")
    summary_lines.append(f"{'W-L consistency (%)':<35} {wl_vals[0]:>10} {wl_vals[1]:>10} {wl_vals[2]:>10}")

    summary_lines.append("")
    summary_lines.append(f"=== GENERAL STATS ({len(clean)} clean cases) ===")
    summary_lines.append(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    summary_lines.append("-" * 67)
    for key in ["z_max", "n_surf_cells", "n_vol_cells",
                "p_aero_mean", "p_aero_std", "delta_p_wl",
                "wss_mag_mean", "Uy_mean", "nut_mean", "nut_max"]:
        vals = data[key]
        vc = vals[~np.isnan(vals)]
        if len(vc) > 0:
            summary_lines.append(f"{key:<25} {vc.mean():>10.4f} {vc.std():>10.4f} {vc.min():>10.4f} {vc.max():>10.4f}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(OUT_DIR / "summary.txt", "w") as fp:
        fp.write(summary_text)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Plots saved to {OUT_DIR}")