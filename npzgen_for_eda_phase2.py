"""
plot.py v2 — All versions: 2 y-scales × 3 x-ranges = 6 per plot
Profile plots with consistent y-axis across all 7.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'savefig.bbox': 'tight', 'savefig.dpi': 200
})
COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#673AB7']
EPS = 1e-10

B_W, A05_W, A2_W, A3_W = 0, 1, 2, 3
B_L, A05_L, A2_L, A3_L = 4, 5, 6, 7
FZ, ZMAX, THICK, XFRAC = 8, 9, 10, 11

H_EDGES = [0, 40, 70, 9999]
H_LABELS = ['< 40m', '40–70m', '> 70m']

VERSIONS = [
    ('log_x2',  True,  (-2, 2)),
    ('log_x5',  True,  (-5, 5)),
    ('log_x20', True,  (-20, 20)),
    ('lin_x2',  False, (-2, 2)),
    ('lin_x5',  False, (-5, 5)),
    ('lin_x20', False, (-20, 20)),
]

def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p

def h_mask(d, i):
    return (d[:, ZMAX] >= H_EDGES[i]) & (d[:, ZMAX] < H_EDGES[i+1])

def t_edges(d):
    t = d[:, THICK]
    t25, t50, t75 = np.percentile(t, [25, 50, 75])
    return [t.min(), t25, t50, t75, t.max() + 0.1]

def t_mask(d, edges, i):
    return (d[:, THICK] >= edges[i]) & (d[:, THICK] < edges[i+1])

# ============================================================
# HIST GROUPED: 2x2 subplot
# ============================================================
def plot_hist_grouped(meta, values, title, xlabel, path, use_log, xlim, bin_type='height', te=None):
    if bin_type == 'height':
        labels = ['Overall'] + H_LABELS
        masks = [np.ones(len(meta), bool)] + [h_mask(meta, i) for i in range(3)]
        colors = ['#2196F3', '#2196F3', '#FF9800', '#4CAF50']
    else:
        edges = te
        labels = ['Overall'] + [f'{edges[i]:.0f}–{edges[i+1]:.0f}m' for i in range(len(edges)-1)]
        masks = [np.ones(len(meta), bool)]
        for i in range(len(edges)-1):
            masks.append(t_mask(meta, edges, i))
        colors = ['#673AB7'] * len(labels)

    n_panels = len(labels)
    cols = min(n_panels, 4)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (label, mask, color) in enumerate(zip(labels, masks, colors)):
        ax = axes[i]
        v = values[mask]
        v = v[np.isfinite(v)]
        if len(v) == 0:
            ax.set_title(label, fontweight='bold'); continue
        ax.hist(v, bins=80, color=color, alpha=0.85, edgecolor='white', linewidth=0.3, log=use_log,
                range=xlim)
        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.axvline(v.mean(), color='red', linestyle='--', linewidth=1.5)
        ax.text(0.97, 0.95, f'mean={v.mean():.3f}\nstd={v.std():.3f}\nn={len(v):,}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlim(xlim)
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel(xlabel)
        yl = 'Count (log)' if use_log else 'Count'
        ax.set_ylabel(yl)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    scale_tag = 'log' if use_log else 'linear'
    fig.suptitle(f'{title}  [{scale_tag}, x: {xlim[0]} to {xlim[1]}]', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")

# ============================================================
# OVERLAY
# ============================================================
def plot_overlay(datasets, title, xlabel, path, use_log, xlim):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, vals, c in datasets:
        ax.hist(vals, bins=100, alpha=0.5, color=c, label=label,
                log=use_log, edgecolor='white', linewidth=0.3, range=xlim)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    yl = 'Count (log)' if use_log else 'Count'
    ax.set_ylabel(yl)
    scale_tag = 'log' if use_log else 'linear'
    ax.set_title(f'{title}  [{scale_tag}, x: {xlim[0]} to {xlim[1]}]', fontweight='bold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")

# ============================================================
# PROFILE
# ============================================================
def plot_profile(meta, values, title, ylabel, path, x_col=FZ, x_label='Height z (m)',
                 bin_step=3.0, ylim=None):
    x = meta[:, x_col]
    x_bins = np.arange(x.min(), x.max() + bin_step, bin_step)
    x_mid = 0.5 * (x_bins[:-1] + x_bins[1:])
    means, stds = [], []
    for i in range(len(x_bins)-1):
        m = (x >= x_bins[i]) & (x < x_bins[i+1])
        v = values[m]; v = v[np.isfinite(v)]
        if len(v) > 5:
            means.append(v.mean()); stds.append(v.std())
        else:
            means.append(np.nan); stds.append(np.nan)
    means, stds = np.array(means), np.array(stds)
    ok = ~np.isnan(means)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(x_mid[ok], means[ok], yerr=stds[ok], marker='o', capsize=3,
                color='#673AB7', linewidth=1.5, markersize=4)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel(x_label); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path}")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out_dir', default='./plots')
    args = parser.parse_args()

    d = np.load(args.input)['data']
    print(f"Loaded: {d.shape}")
    o = args.out_dir

    te = t_edges(d)
    print(f"Thickness edges: {[f'{e:.1f}' for e in te]}")

    # === DERIVED QUANTITIES ===
    ab_05_w = d[:, A05_W] - d[:, B_W]
    ab_2_w  = d[:, A2_W]  - d[:, B_W]
    ab_3_w  = d[:, A3_W]  - d[:, B_W]
    ab_05_l = d[:, A05_L] - d[:, B_L]
    ab_2_l  = d[:, A2_L]  - d[:, B_L]
    ab_3_l  = d[:, A3_L]  - d[:, B_L]

    ab_05 = np.concatenate([ab_05_w, ab_05_l])
    ab_2  = np.concatenate([ab_2_w,  ab_2_l])
    ab_3  = np.concatenate([ab_3_w,  ab_3_l])
    d2 = np.vstack([d, d])

    bb    = d[:, B_W] - d[:, B_L]
    aa_05 = d[:, A05_W] - d[:, A05_L]
    aa_2  = d[:, A2_W]  - d[:, A2_L]
    aa_3  = d[:, A3_W]  - d[:, A3_L]

    b_comb = np.concatenate([d[:, B_W], d[:, B_L]])
    valid_b = np.abs(b_comb) > 0.5
    ab_skew_05 = np.where(valid_b, ab_05 / b_comb, np.nan)
    ab_skew_2  = np.where(valid_b, ab_2  / b_comb, np.nan)
    ab_skew_3  = np.where(valid_b, ab_3  / b_comb, np.nan)

    # ============================================================
    # 1. HISTOGRAMS BY HEIGHT — 6 versions each, 10 base plots
    # ============================================================
    print("\n=== Histograms by height (6 versions each) ===")

    hist_h_specs = [
        ('ab_0.5m', d2, ab_05, '(a-b) shell−surface 0.5m', 'Δp (shell − surface)'),
        ('ab_2.0m', d2, ab_2,  '(a-b) shell−surface 2.0m', 'Δp (shell − surface)'),
        ('ab_3.0m', d2, ab_3,  '(a-b) shell−surface 3.0m', 'Δp (shell − surface)'),
        ('aa_0.5m', d, aa_05,  '(a,a) shell W−L 0.5m',     'Δp (windward − leeward)'),
        ('aa_2.0m', d, aa_2,   '(a,a) shell W−L 2.0m',     'Δp (windward − leeward)'),
        ('aa_3.0m', d, aa_3,   '(a,a) shell W−L 3.0m',     'Δp (windward − leeward)'),
        ('bb',      d, bb,     '(b,b) surface W−L',         'Δp (windward − leeward)'),
        ('ab_skew_0.5m', d2, ab_skew_05, '(a,b) skewed 0.5m', '(shell−surface)/surface'),
        ('ab_skew_2.0m', d2, ab_skew_2,  '(a,b) skewed 2.0m', '(shell−surface)/surface'),
        ('ab_skew_3.0m', d2, ab_skew_3,  '(a,b) skewed 3.0m', '(shell−surface)/surface'),
    ]

    for vtag, use_log, xl in VERSIONS:
        vdir = ensure(os.path.join(o, 'hist_by_height', vtag))
        for fname, meta, vals, title, xlabel in hist_h_specs:
            plot_hist_grouped(meta, vals, title, xlabel,
                              os.path.join(vdir, f'{fname}.png'),
                              use_log, xl)

    # ============================================================
    # 2. HISTOGRAMS BY THICKNESS — 6 versions each, 4 base plots
    # ============================================================
    print("\n=== Histograms by thickness (6 versions each) ===")

    hist_t_specs = [
        ('aa_0.5m', d, aa_05, '(a,a) shell W−L 0.5m', 'Δp (windward − leeward)'),
        ('aa_2.0m', d, aa_2,  '(a,a) shell W−L 2.0m', 'Δp (windward − leeward)'),
        ('aa_3.0m', d, aa_3,  '(a,a) shell W−L 3.0m', 'Δp (windward − leeward)'),
        ('bb',      d, bb,    '(b,b) surface W−L',     'Δp (windward − leeward)'),
    ]

    for vtag, use_log, xl in VERSIONS:
        vdir = ensure(os.path.join(o, 'hist_by_thickness', vtag))
        for fname, meta, vals, title, xlabel in hist_t_specs:
            plot_hist_grouped(meta, vals, title, xlabel,
                              os.path.join(vdir, f'{fname}.png'),
                              use_log, xl, bin_type='thickness', te=te)

    # ============================================================
    # 3. OVERLAY — 6 versions each, 3 base plots
    # ============================================================
    print("\n=== Overlay plots (6 versions each) ===")

    ov_specs = [
        ('ab_overlay', '(a-b) overlay: 0.5m vs 2.0m vs 3.0m', 'Δp (shell − surface)', [
            (f'0.5m (mean={ab_05.mean():.3f})', ab_05, COLORS[0]),
            (f'2.0m (mean={ab_2.mean():.3f})',  ab_2,  COLORS[1]),
            (f'3.0m (mean={ab_3.mean():.3f})',  ab_3,  COLORS[2]),
        ]),
        ('aa_bb_overlay', '(a,a) vs (b,b): shell W−L vs surface W−L', 'Δp (windward − leeward)', [
            (f'shell 0.5m (mean={aa_05.mean():.3f})', aa_05, COLORS[0]),
            (f'shell 2.0m (mean={aa_2.mean():.3f})',  aa_2,  COLORS[1]),
            (f'shell 3.0m (mean={aa_3.mean():.3f})',  aa_3,  COLORS[2]),
            (f'surface (mean={bb.mean():.3f})',        bb,    COLORS[3]),
        ]),
        ('ab_skew_overlay', '(a,b) skewed overlay', '(shell−surface)/surface', [
            (f'0.5m (mean={np.nanmean(ab_skew_05):.3f})', ab_skew_05[np.isfinite(ab_skew_05)], COLORS[0]),
            (f'2.0m (mean={np.nanmean(ab_skew_2):.3f})',  ab_skew_2[np.isfinite(ab_skew_2)],   COLORS[1]),
            (f'3.0m (mean={np.nanmean(ab_skew_3):.3f})',  ab_skew_3[np.isfinite(ab_skew_3)],   COLORS[2]),
        ]),
    ]

    for vtag, use_log, xl in VERSIONS:
        vdir = ensure(os.path.join(o, 'overlay', vtag))
        for fname, title, xlabel, datasets in ov_specs:
            plot_overlay(datasets, title, xlabel,
                         os.path.join(vdir, f'{fname}.png'), use_log, xl)

    # ============================================================
    # 4. SCATTER — 3 plots (no versions needed)
    # ============================================================
    print("\n=== Scatter plots ===")
    sc_dir = ensure(os.path.join(o, 'scatter'))

    for tag, aa_vals, c in [('0.5m', aa_05, COLORS[0]), ('2.0m', aa_2, COLORS[1]), ('3.0m', aa_3, COLORS[2])]:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(bb, aa_vals, s=2, alpha=0.15, color=c)
        lim = max(np.percentile(np.abs(bb), 99.5), np.percentile(np.abs(aa_vals), 99.5))
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='y = x')
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel('Surface ΔP: (b,b)'); ax.set_ylabel(f'Shell ΔP: (a,a) at {tag}')
        ax.set_title(f'Surface vs shell W−L ({tag})', fontweight='bold')
        ax.set_aspect('equal'); ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(sc_dir, f'bb_vs_aa_{tag}.png'))
        plt.close(fig)
        print(f"  scatter bb_vs_aa_{tag}.png")

    # ============================================================
    # 5. PROFILE BY Z — 7 plots, consistent y-axis
    # ============================================================
    print("\n=== Profile by z (consistent y-axis) ===")
    pz_dir = ensure(os.path.join(o, 'profile_by_z'))

    pz_specs = [
        ('ab_0.5m', d, ab_05_w, '(a-b) windward 0.5m', 'Mean Δp (shell − surface)'),
        ('ab_2.0m', d, ab_2_w,  '(a-b) windward 2.0m', 'Mean Δp (shell − surface)'),
        ('ab_3.0m', d, ab_3_w,  '(a-b) windward 3.0m', 'Mean Δp (shell − surface)'),
        ('aa_0.5m', d, aa_05,   '(a,a) shell W−L 0.5m', 'Mean Δp (windward − leeward)'),
        ('aa_2.0m', d, aa_2,    '(a,a) shell W−L 2.0m', 'Mean Δp (windward − leeward)'),
        ('aa_3.0m', d, aa_3,    '(a,a) shell W−L 3.0m', 'Mean Δp (windward − leeward)'),
        ('bb',      d, bb,      '(b,b) surface W−L',     'Mean Δp (windward − leeward)'),
    ]

    # Compute global y range across all 7 profiles
    global_ymin, global_ymax = 0, 0
    fz_all = d[:, FZ]
    z_bins = np.arange(fz_all.min(), fz_all.max() + 3, 3)
    for _, meta, vals, _, _ in pz_specs:
        for i in range(len(z_bins)-1):
            m = (meta[:, FZ] >= z_bins[i]) & (meta[:, FZ] < z_bins[i+1])
            v = vals[m]; v = v[np.isfinite(v)]
            if len(v) > 5:
                mu, sd = v.mean(), v.std()
                global_ymin = min(global_ymin, mu - sd)
                global_ymax = max(global_ymax, mu + sd)

    y_pad = (global_ymax - global_ymin) * 0.1
    profile_ylim = (global_ymin - y_pad, global_ymax + y_pad)
    print(f"  Global profile y-axis: {profile_ylim[0]:.2f} to {profile_ylim[1]:.2f}")

    for fname, meta, vals, title, ylabel in pz_specs:
        plot_profile(meta, vals, title, ylabel,
                     os.path.join(pz_dir, f'{fname}.png'),
                     ylim=profile_ylim)

    # ============================================================
    # 6. PROFILE BY THICKNESS — 4 plots
    # ============================================================
    print("\n=== Profile by thickness ===")
    pt_dir = ensure(os.path.join(o, 'profile_by_thick'))

    pt_specs = [
        ('aa_0.5m', d, aa_05, '(a,a) shell W−L vs thickness 0.5m', 'Mean Δp (windward − leeward)'),
        ('aa_2.0m', d, aa_2,  '(a,a) shell W−L vs thickness 2.0m', 'Mean Δp (windward − leeward)'),
        ('aa_3.0m', d, aa_3,  '(a,a) shell W−L vs thickness 3.0m', 'Mean Δp (windward − leeward)'),
        ('bb',      d, bb,    '(b,b) surface W−L vs thickness',     'Mean Δp (windward − leeward)'),
    ]

    for fname, meta, vals, title, ylabel in pt_specs:
        plot_profile(meta, vals, title, ylabel,
                     os.path.join(pt_dir, f'{fname}.png'),
                     x_col=THICK, x_label='Building thickness (m)', bin_step=2.0)

    # ============================================================
    # SUMMARY
    # ============================================================
    total = (10 + 4) * 6 + 3 * 6 + 3 + 7 + 4
    print(f"\n{'='*50}")
    print(f"Total plots: {total}")
    print(f"  hist_by_height: 10 × 6 = 60")
    print(f"  hist_by_thickness: 4 × 6 = 24")
    print(f"  overlay: 3 × 6 = 18")
    print(f"  scatter: 3")
    print(f"  profile_by_z: 7")
    print(f"  profile_by_thick: 4")
    print(f"Total: {total}")

    print(f"\n{'Metric':<30} {'mean':>8} {'std':>8}")
    print('-' * 48)
    for label, vals in [
        ('(b,b) surface W-L', bb),
        ('(a,a) 0.5m W-L', aa_05), ('(a,a) 2.0m W-L', aa_2), ('(a,a) 3.0m W-L', aa_3),
        ('(a-b) 0.5m', ab_05), ('(a-b) 2.0m', ab_2), ('(a-b) 3.0m', ab_3),
    ]:
        v = vals[np.isfinite(vals)]
        print(f"{label:<30} {v.mean():>8.4f} {v.std():>8.4f}")

    print(f"\nAll saved under: {o}/")

if __name__ == '__main__':
    main()