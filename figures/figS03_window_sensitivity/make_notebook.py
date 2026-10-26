"""Write figS3_window_sensitivity.ipynb and execute it (python make_notebook.py).

The notebook reads the archived tables in data("window_sensitivity") (outputs_window_sensitivity.zip) and writes
Figure S3 (PDF, PNG, SVG), a per-setting summary and the caption values to out("figS3_window_sensitivity"), together
with the executed notebook.
"""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repository root (paths)
from paths import out  # noqa: E402

STEM = "figS3_window_sensitivity"
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md(r"""# Figure S3: sensitivity to the identification window

Window length and update interval for the 14 SSP2-4.5 members at a matched effort of 10 Gm3/yr. The notebook draws the
figure from the archived tables of `outputs_window_sensitivity.zip` (written by `analyses/window_sensitivity/package_tables.py`);
it does not rerun the identification or the control."""),
code(r"""from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'paths.py').is_file() and (p / '04_dmd_lqr_analysis.py').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Open this notebook from its folder in the repository.')
sys.path.insert(0, str(ROOT))
from paths import data, out
DATA = data('window_sensitivity')
OUT = out('figS3_window_sensitivity')

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.ticker import LogLocator

try:
    font_manager.findfont('Arial', fallback_to_default=False)
    FONT = 'Arial'
except ValueError:
    FONT = 'DejaVu Sans'

# ── Config, style and helpers of the paper figures ───────────────
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

def plabel_map(ax, lab):
    ax.text(0.03, 0.97, lab, transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

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
    display(Image(filename=str(OUT / f"{stem}.png"), width=1000))
    plt.close(fig)"""),
code(r"""win = pd.read_csv(DATA / 'supplementary_all_original_results.csv')
paired = pd.read_csv(DATA / 'supplementary_all_original_paired_results.csv')
order = ['5_2p5', '5_5', '10_5', '15_5', '20_5', '20_10']
LABELS = [s.replace('_', '/').replace('2p5', '2.5') for s in order]
assert len(win) == 84 and len(paired) == 84
assert win.budget_matched.all() and paired.paired_budget_matched.all()
assert ((win.actual_budget - 10).abs() <= 0.05).all()
assert win.groupby('setting').size().reindex(order).eq(14).all()
longer = ['15_5', '20_5', '20_10']
# Per-setting median and range of the plotted quantities (panels a-d)
summary = pd.concat([win.groupby('setting')[['nse_median', 'reduction_percent']].agg(['median', 'min', 'max']),
                     paired.groupby('setting')[['delta_reduction_pp', 'top10_overlap']].agg(['median', 'min', 'max'])],
                    axis=1).reindex(order)
summary.to_csv(OUT / 'figS3_setting_summary.csv')
# Caption values: effort matched within 0.5%; the only negative reduction (panel b); median top-decile overlap (panel d)
negative = win[win.reduction_percent < 0]
overlap = 100 * paired.groupby('setting').top10_overlap.median().reindex(order)
caption = {'runs': len(win), 'members': int(win.groupby(['hydro', 'gcm']).ngroups),
           'max_relative_effort_deviation': float((win.actual_budget / win.budget_target - 1).abs().max()),
           'negative_reduction': [f'{r.hydro}/{r.gcm} at {r.setting}' for r in negative.itertuples()],
           'median_top10_overlap_percent': {s: round(float(overlap[s]), 1) for s in order}}
assert caption['members'] == 14 and caption['max_relative_effort_deviation'] <= 0.005
assert negative[['hydro', 'gcm', 'setting']].values.tolist() == [['VIC5', 'ACCESS-CM2', '5_2p5']]
assert (round(overlap[longer].min()), round(overlap[longer].max())) == (95, 97)
(OUT / 'figS3_caption_values.json').write_text(json.dumps(caption, indent=2))
display(summary); display(caption)

def members(ax, frame, key, scale=1):
    # thin lines: members; thick lines with markers: hydrologic-model medians
    for h in HYDROS:
        d = frame[frame.hydro == h].pivot(index='gcm', columns='setting', values=key).reindex(columns=order) * scale
        assert d.shape == (7, len(order)) and d.notna().all().all()
        for _, row in d.iterrows():
            ax.plot(range(len(order)), row.to_numpy(), color=COLORS_LIGHT[h], lw=0.4, alpha=0.7, zorder=1)
    for h in HYDROS:
        d = frame[frame.hydro == h].pivot(index='gcm', columns='setting', values=key).reindex(columns=order) * scale
        ax.plot(range(len(order)), d.median().to_numpy(), "o-", color=COLORS[h], ms=3.5, lw=1.4, label=h, zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.35, len(order) - 0.65)"""),
code(r"""fig, axes = plt.subplots(2, 2, figsize=(W_FULL, 118*MM), gridspec_kw={"wspace": 0.26, "hspace": 0.46})
XLAB = "Window / update interval (years)"

# ── (a) Continuous reconstruction ────────────────────────────────
ax = axes[0, 0]
members(ax, win, 'nse_median')
ax.set_xlabel(XLAB); ax.set_ylabel("Reach-median daily NSE")
ax.set_ylim(0.4, 0.85)
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
ax.legend(loc="lower left", fontsize=6)
lgrid(ax); plabel(ax, "a", x=-0.10, y=1.10)

# ── (b) Reduction of network-summed flood excess ─────────────────
ax = axes[0, 1]
ax.axhline(0, color="#777", ls="--", lw=0.6, zorder=1)
members(ax, win, 'reduction_percent')
ax.set_xlabel(XLAB); ax.set_ylabel("Reduction from own baseline (%)")
lgrid(ax); plabel(ax, "b", x=-0.10, y=1.10)

# ── (c) Member-paired change relative to 10/5 years ──────────────
ax = axes[1, 0]
ax.axhline(0, color="#777", ls="--", lw=0.6, zorder=1)
for j, s in enumerate(longer):
    for k, h in enumerate(HYDROS):
        d = paired[(paired.setting == s) & (paired.hydro == h)].sort_values('gcm').delta_reduction_pp.to_numpy()
        assert len(d) == 7
        x = j + (k - 0.5) * 0.30
        jitter_strip(ax, x, d, COLORS[h], width=0.12, ms=5, alpha=0.6)
        dot_whisker(ax, x, d, COLORS[h], ms=6)
ax.set_xticks(range(3)); ax.set_xticklabels(["15/5", "20/5", "20/10"])
ax.set_xlim(-0.5, 2.5)
ax.set_xlabel(XLAB); ax.set_ylabel("Change in reduction\n(percentage points)")
lgrid(ax); plabel(ax, "c", x=-0.10, y=1.10)

# ── (d) Spatial priorities (top decile by effort) ─────────────────
ax = axes[1, 1]
members(ax, paired, 'top10_overlap', scale=100)
ax.set_ylim(50, 102)
ax.set_xlabel(XLAB); ax.set_ylabel("Top-decile overlap with 10/5 (%)")
lgrid(ax); plabel(ax, "d", x=-0.10, y=1.10)

export(fig, 'figS3_window_sensitivity')"""),
]

path = out(STEM) / f"{STEM}.ipynb"
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
nbf.write(nb, path)
print("wrote and executed", path)
