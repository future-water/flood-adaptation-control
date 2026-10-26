"""Figure S2: diagnostics of the regularized estimator (python make_figure.py).

Reads data("regularization")/all_paths.csv (panels a, b; rows with alpha > 0) and
regularization_sensitivity.csv (panel c) and writes figS2_regularization.{pdf,png,svg} to
out("figS2_regularization"), with two small tables of the plotted values: selected_alpha_diagnostics.csv (the stars) and
coefficient_sensitivity_summary.csv (median, min and max of panel c). It does not rerun the
identification. The drawing follows the figure code of the paper (rcParams, palette and helpers).
Use --data DIR to read the two tables from another folder, e.g. $FAC_WORK_ROOT/regularization.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
import argparse
import numpy as np
import pandas as pd
from paths import data, out

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--data', type=Path, default=data('regularization'),
                    help='folder with all_paths.csv and regularization_sensitivity.csv')
DATA = parser.parse_args().data
OUT = out('figS2_regularization')

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import LogLocator

try:
    font_manager.findfont('Arial', fallback_to_default=False)
    FONT = 'Arial'
except ValueError:
    FONT = 'DejaVu Sans'

# ── Config, style and helpers of the paper figures ───────────────
#    (as in figures/main_figures.ipynb and figures/supplementary_figures.ipynb)
HYDROS = ['VIC5', 'PRMS']
COLORS = {'VIC5': '#2166AC', 'PRMS': '#d4820a'}
COLORS_LIGHT = {'VIC5': '#9ecae1', 'PRMS': '#fdd49e'}

MM = 1/25.4; W_FULL = 190*MM

mpl.rcParams.update({
    "font.family": FONT, "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.3, "ytick.minor.width": 0.3,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.fontsize": 6, "legend.frameon": False,
    "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
    "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "lines.linewidth": 0.8,
})

def lgrid(ax, alpha=0.10, lw=0.3):
    ax.grid(True, alpha=alpha, lw=lw, zorder=0); ax.set_axisbelow(True)

def plabel(ax, lab, x=-0.18, y=1.05):
    ax.text(x, y, lab, transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="top", ha="right")

def jitter_strip(ax, x_pos, values, color, width=0.25, ms=3.5, alpha=0.5):
    jit = np.random.default_rng(42).uniform(-width/2, width/2, len(values))
    ax.scatter(x_pos + jit, values, s=ms, alpha=alpha, color=color,
               edgecolors="white", linewidths=0.15, zorder=3)

def dot_whisker(ax, x_pos, values, color, lw=2.0, ms=7):
    med = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    ax.plot([x_pos, x_pos], [q25, q75], color=color, lw=lw,
            solid_capstyle="round", zorder=4)
    ax.plot(x_pos, med, "o", color=color, ms=ms,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    return med

def export(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor="white", dpi=600 if ext != "png" else 300)
    plt.close(fig)

# ── Inputs and checks ─────────────────────────────────────────────
PATH_COLUMNS = ['hydro', 'alpha', 'is_original', 'in_original_grid', 'n', 'p', 'df_per_response',
                'rho', 'margin_pass', 'rollout_complete', 'raw_NSE']
paths = pd.read_csv(DATA / 'all_paths.csv')
paths = paths.loc[paths.alpha > 0, PATH_COLUMNS]   # a rerun's table also has the alpha = 0 rows
noise_all = pd.read_csv(DATA / 'regularization_sensitivity.csv')
noise = noise_all[noise_all.reference == 'rcond_1e-8']
assert len(paths) == 92 and len(noise) == 80
assert paths.n.eq(3650).all() and paths.p.eq(14451).all()
original = paths[paths.is_original].copy()
assert len(original) == 2 and original.alpha.eq(0.01).all()
for h in HYDROS:   # selected alpha = smallest grid value with rho(A) < 0.995
    g = paths[(paths.hydro == h) & paths.in_original_grid].sort_values('alpha')
    assert g.margin_pass.eq(g.rho < 0.995).all() and g[g.margin_pass].alpha.iloc[0] == 0.01
original.to_csv(OUT / 'selected_alpha_diagnostics.csv', index=False)
summary = noise_all.groupby(['reference', 'hydro', 'metric']).ridge_index.agg(['median', 'min', 'max'])
summary.to_csv(OUT / 'coefficient_sensitivity_summary.csv')
pd.set_option('display.width', 160)
print(original[['hydro', 'alpha', 'n', 'p', 'df_per_response', 'rho', 'raw_NSE']].to_string(), '\n')
print(summary.to_string(), '\n')
print('Panel b, reach-median daily NSE on the regularization grid:')
print(paths[paths.in_original_grid].pivot(index='alpha', columns='hydro', values='raw_NSE')[HYDROS].to_string(), '\n')

# ── Figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 65*MM), gridspec_kw={"wspace": 0.42})
LS = {'VIC5': '-', 'PRMS': '--'}

# ── (a) Effective degrees of freedom along the regularization path ──
ax = axes[0]
ax.axhline(3650, color="#777", ls="--", lw=0.6, zorder=1)
for h in HYDROS:
    d = paths[(paths.hydro == h) & (paths.alpha > 0)].sort_values('alpha')
    ax.plot(d.alpha, d.df_per_response, color=COLORS[h], ls=LS[h], lw=1.4, label=h, zorder=3)
    o = original[original.hydro == h].iloc[0]
    ax.plot(o.alpha, o.df_per_response, "*", color=COLORS[h], ms=8,
            markeredgecolor="white", markeredgewidth=0.4, zorder=5)
ax.text(0.97, 3650 + 60, "n = 3,650 transitions", transform=ax.get_yaxis_transform(),
        fontsize=5.5, va="bottom", ha="right", color="#555")
ax.set_xscale("log"); ax.set_ylim(0, 4050)
ax.xaxis.set_major_locator(LogLocator(numticks=4))
ax.set_xlabel(r"Regularization strength $\alpha$")
ax.set_ylabel("Effective degrees of freedom\nper response")
ax.legend(loc="lower left", fontsize=6, handlelength=1.8)
lgrid(ax); plabel(ax, "a", x=-0.24)

# ── (b) Reconstruction for the nine penalties of the original grid ──
ax = axes[1]
for h in HYDROS:
    d = paths[(paths.hydro == h) & paths.in_original_grid].sort_values('alpha')
    assert len(d) == 9 and d.rollout_complete.all()
    ax.plot(d.alpha, d.raw_NSE, "o-", color=COLORS[h], ms=3.5, lw=1.0, label=h, zorder=3)
    o = d[d.is_original].iloc[0]
    ax.plot(o.alpha, o.raw_NSE, "*", color=COLORS[h], ms=8,
            markeredgecolor="white", markeredgewidth=0.4, zorder=5)
ax.set_xscale("log"); ax.set_xlim(0.007, 75); ax.set_ylim(0, 1)
ax.set_xticks([0.01, 0.1, 1, 10, 50]); ax.set_xticklabels(["0.01", "0.1", "1", "10", "50"])
ax.set_xlabel(r"Regularization strength $\alpha$")
ax.set_ylabel("Reach-median daily NSE")
lgrid(ax); plabel(ax, "b")

# ── (c) Coefficient sensitivity to 1% target perturbations ──────────
ax = axes[2]
ax.axhline(100, color="#777", ls="--", lw=0.6, zorder=1)
for j, metric in enumerate(['delta_A_norm', 'delta_E_norm']):
    for k, h in enumerate(HYDROS):
        a = noise[(noise.hydro == h) & (noise.metric == metric)].sort_values('seed').ridge_index.to_numpy()
        assert len(a) == 20
        x = j + (k - 0.5) * 0.34
        jitter_strip(ax, x, a, COLORS[h], width=0.14, ms=4, alpha=0.55)
        med = dot_whisker(ax, x, a, COLORS[h])
        ax.text(x, a.max() + 4, f"{med:.1f}", fontsize=6.5, fontweight="bold", ha="center", va="bottom")
ax.text(0.97, 100 + 2.5, "No ridge = 100", transform=ax.get_yaxis_transform(),
        fontsize=5.5, va="bottom", ha="right", color="#555")
ax.set_xticks([0, 1]); ax.set_xticklabels([r"State $\mathbf{A}$", r"Forcing $\mathbf{E}$"])
ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 118)
ax.set_ylabel("Coefficient-sensitivity index")
lgrid(ax); plabel(ax, "c")

export(fig, 'figS2_regularization')
print('wrote', OUT / 'figS2_regularization.pdf')
