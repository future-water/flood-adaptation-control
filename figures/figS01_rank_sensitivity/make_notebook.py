"""Write figS1_rank_sensitivity.ipynb, execute it, and save Figure S1 (python make_notebook.py [--data DIR]).

Reads the outputs_rank_sensitivity archive (data('rank_sensitivity'), or --data, for example the output folder of
analyses/rank_sensitivity/analyze_ensemble.py) and data('common')/network_geometry_5070.npz. Writes the executed notebook,
figS1_rank_sensitivity.{pdf,png,svg}, rank_summary.csv, rank_spatial_overlap.csv and rank_mapped_priority_sets.csv
to out('figS1_rank_sensitivity'). It does not rerun POD/DMDc/LQR.
"""
import argparse
import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repository root (paths)
from paths import out  # noqa: E402

STEM = "figS1_rank_sensitivity"
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
md(r"""# Figure S1: reduced dimension and spatial priorities

Sensitivity to the reduced dimension $r$ for the 14 SSP2-4.5 members (Table S1). The drawing follows the figure code
of the paper (line panels as in the validation figures, maps as in the categorical map of Fig. 5d). The notebook
reproduces the figure from the archived analysis tables (outputs_rank_sensitivity, outputs_common); it does not
rerun POD/DMDc/LQR."""),
code(r"""from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image

REPO = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'paths.py').is_file() and (p / '04_dmd_lqr_analysis.py').is_file()), None)
if REPO is None:
    raise FileNotFoundError('Run this notebook from its folder in the repository (python make_notebook.py).')
sys.path.insert(0, str(REPO))
from paths import data, out
DATA = Path(os.environ.get('FIGS1_RANK_DATA') or data('rank_sensitivity'))
GEOMETRY = data('common') / 'network_geometry_5070.npz'
OUT = out('figS1_rank_sensitivity')
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplcache'))

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

def export(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor="white", dpi=600 if ext != "png" else 300)
    display(Image(filename=str(OUT / f"{stem}.png"), width=1000))
    plt.close(fig)"""),
code(r"""spec = pd.read_csv(DATA / 'singular_spectra_all14.csv')
status = pd.read_csv(DATA / 'member_rank_status.csv')
pairs = pd.read_csv(DATA / 'ensemble_rank_comparisons.csv')
ranks = [10, 20, 30, 50, 75, 100]
assert len(spec) == 4200 and len(status) == 84
counts = status.groupby('rank').strict_valid.sum().reindex(ranks).astype(int)
assert counts.tolist() == [14, 14, 14, 13, 7, 3]
SCOPE = 'full14_original_fallback'  # primary scope of the original analysis: all 14 members
core = pairs[(pairs.scope == SCOPE) & (pairs.metric == 'raw')].copy()
assert core.n_members.eq(14).all()
comparisons = [(20, 30), (30, 50), (20, 50)]
rows = []
for a, b in comparisons:
    d = core[(core.rank_a == a) & (core.rank_b == b)].iloc[0]
    rows.append(dict(rank_pair=f'{a} vs {b}', raw_top10=100 * d['top10_retention_a']))
overlap = pd.DataFrame(rows).set_index('rank_pair')
energy = spec[spec['mode'].isin(ranks)].groupby('mode').agg(
    energy_median=('cumulative_fraction', 'median'), energy_min=('cumulative_fraction', 'min'),
    energy_max=('cumulative_fraction', 'max'), error_median=('centered_reconstruction_relative_l2', 'median'))
energy['admissible_members'] = counts
energy.to_csv(OUT / 'rank_summary.csv')
display(energy); display(overlap)

# top decile of reaches by raw effort at r = 20, 30, 50 (all 14 members)
rank_data = {r: dict(np.load(DATA / 'aggregates' / f'{SCOPE}_r{r}.npz', allow_pickle=False)) for r in [20, 30, 50]}
with np.load(DATA / 'aggregates' / f'{SCOPE}_common_reach_universe.npz', allow_pickle=False) as z:
    reach_ids = z['reach_ids']
    raw_valid = z['raw']
assert raw_valid.sum() == 14421
selected = {}
for r, d in rank_data.items():
    np.testing.assert_array_equal(d['reach_ids'], reach_ids)
    assert int(d['n_members']) == 14
    np.testing.assert_array_equal(d['members'], rank_data[30]['members'])
    indices = np.flatnonzero(raw_valid & np.isfinite(d['effort']))
    order = np.lexsort((reach_ids[indices], -d['effort'][indices]))
    selected[r] = set(reach_ids[indices[order[:int(0.1 * len(indices))]]].tolist())
    assert len(selected[r]) == 1442
with np.load(GEOMETRY, allow_pickle=False) as z:
    coordinates = z['coordinates']
    offsets = z['offsets']
    map_ids = z['comid']
    bounds = z['bounds']
segments = [coordinates[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
assert len(segments) == len(map_ids) == 14415
assert set(map_ids).issubset(set(reach_ids))
assert all((not ids - set(map_ids) for ids in selected.values()))
map_rows = []
for a, b in comparisons:
    shared = selected[a] & selected[b]
    ratio = 100 * len(shared) / 1442
    np.testing.assert_allclose(ratio, overlap.loc[f'{a} vs {b}', 'raw_top10'])
    map_rows.append(dict(rank_a=a, rank_b=b, n_selected=1442, n_shared=len(shared), n_lower_only=len(selected[a] - selected[b]),
                         n_higher_only=len(selected[b] - selected[a]), overlap_percent=ratio, unmapped_selected=0))
map_stats = pd.DataFrame(map_rows)
map_stats.to_csv(OUT / 'rank_spatial_overlap.csv', index=False)
selection_table = pd.DataFrame({'COMID': reach_ids, **{f'top10_r{r}': np.isin(reach_ids, list(selected[r])) for r in selected}})
selection_table.to_csv(OUT / 'rank_mapped_priority_sets.csv', index=False)
display(map_stats)"""),
code(r"""fig = plt.figure(figsize=(W_FULL, 122*MM))
gs_top = fig.add_gridspec(1, 3, left=0.085, right=0.985, top=0.965, bottom=0.615, wspace=0.42)
gs_map = fig.add_gridspec(1, 3, left=0.005, right=0.995, top=0.535, bottom=0.005, wspace=0.02)

# ══ (a–c) POD compression ═══════════════════════════════════════
panels = [('singular_value_relative', 'Singular value / leading value', 1),
          ('cumulative_fraction', 'Retained centered energy (%)', 100),
          ('centered_reconstruction_relative_l2', r'Relative $L_2$ projection error (%)', 100)]
for j, (key, ylabel, mult) in enumerate(panels):
    ax = fig.add_subplot(gs_top[0, j])
    for h in HYDROS:
        d = spec[spec.hydro == h]
        if j > 0:
            d = d[d['mode'].isin(ranks)]
        for _, g in d.groupby('gcm'):
            ax.plot(g['mode'], g[key] * mult, color=COLORS_LIGHT[h], lw=0.4, alpha=0.7, zorder=1)
    for h in HYDROS:
        d = spec[spec.hydro == h]
        if j > 0:
            d = d[d['mode'].isin(ranks)]
        med = d.groupby('mode')[key].median() * mult
        ax.plot(med.index, med.to_numpy(), "-" if j == 0 else "o-", color=COLORS[h], lw=1.4, ms=3.5, label=h, zorder=3)
    ax.axvline(30, color="#777", ls="--", lw=0.6, zorder=2)
    ax.set_xlabel('Mode number' if j == 0 else 'Retained modes $r$')
    ax.set_ylabel(ylabel)
    if j == 0:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 300)
        ax.legend(loc="lower left", fontsize=6)
    else:
        ax.set_xticks(ranks); ax.set_xlim(4, 106)
    if j == 1:
        ymin = 100 * spec.loc[spec['mode'].isin(ranks), 'cumulative_fraction'].min()
        ax.set_ylim(min(98, ymin - 0.15), 100.02)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    if j == 2:
        ax.set_yscale("log")
    lgrid(ax); plabel(ax, "abc"[j], x=-0.22, y=1.06)

# ══ (d–f) Top decile of reaches by raw effort at two ranks ══════
x0, y0, x1, y1 = bounds
dx, dy = x1 - x0, y1 - y0
C_BOTH, C_LOW, C_HIGH = "#2d8e3d", "#2166AC", "#d4820a"
for j, (a, b) in enumerate(comparisons):
    ax = fig.add_subplot(gs_map[0, j])
    ax.set_aspect("equal"); ax.axis("off")
    cat = np.array([1 if k in selected[a] and k in selected[b] else 2 if k in selected[a] else 3 if k in selected[b] else 0
                    for k in map_ids])
    pick = lambda c: [seg for seg, v in zip(segments, cat) if v == c]
    # draw order as in Fig. 5d: background, shared set, then the two unique sets on a white halo
    ax.add_collection(LineCollection(segments, colors="#dcdcdc", linewidths=0.15, alpha=0.60, zorder=0, rasterized=True))
    ax.add_collection(LineCollection(pick(1), colors=C_BOTH, linewidths=0.9, alpha=0.90, zorder=1,
                                     capstyle="round", joinstyle="round", rasterized=True))
    for c, color, z in ((2, C_LOW, 2), (3, C_HIGH, 4)):
        ax.add_collection(LineCollection(pick(c), colors="white", linewidths=2.2, zorder=z,
                                         capstyle="round", joinstyle="round", rasterized=True))
        ax.add_collection(LineCollection(pick(c), colors=color, linewidths=1.4, alpha=0.92, zorder=z + 1,
                                         capstyle="round", joinstyle="round", rasterized=True))
    ax.set_xlim(x0 - 0.02 * dx, x1 + 0.02 * dx); ax.set_ylim(y0 - 0.02 * dy, y1 + 0.02 * dy)
    plabel_map(ax, "def"[j])

    row = map_stats.iloc[j]
    ax.text(0.97, 0.95, f"$r$ = {a} vs {b}", transform=ax.transAxes,
            fontsize=7, fontweight="bold", ha="right", va="top")
    ax.text(0.97, 0.88, f"{row.overlap_percent:.1f}% overlap", transform=ax.transAxes,
            fontsize=6.5, ha="right", va="top", color="#333")
    legend_handles = [
        Line2D([0], [0], color=C_BOTH, lw=1.6, label=f"Both ranks ({int(row.n_shared):,})"),
        Line2D([0], [0], color=C_LOW, lw=1.6, label=f"$r$ = {a} only ({int(row.n_lower_only)})"),
        Line2D([0], [0], color=C_HIGH, lw=1.6, label=f"$r$ = {b} only ({int(row.n_higher_only)})"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=6, frameon=False,
              handlelength=2.0, labelspacing=0.25, borderpad=0.3)
    if j == 0:   # scale bar as in add_scalebar (100 km; coordinates in metres, EPSG:5070)
        bar = 100e3
        sx, sy = x0 + dx * 0.47, y0 + dy * 0.055
        ax.plot([sx, sx + bar], [sy, sy], "k-", lw=0.8, solid_capstyle="butt", zorder=10)
        ax.text(sx + bar / 2, sy + dy * 0.015, "100 km", ha="center", fontsize=5, zorder=10)

export(fig, 'figS1_rank_sensitivity')"""),
]


def main():
    parser = argparse.ArgumentParser(description="Write and execute the Figure S1 notebook.")
    parser.add_argument("--data", type=Path, default=None,
                        help="folder with the rank_sensitivity files (default: data('rank_sensitivity'))")
    args = parser.parse_args()
    if args.data is not None:
        os.environ["FIGS1_RANK_DATA"] = str(args.data.resolve())   # read by the notebook kernel
    path = out(STEM) / f"{STEM}.ipynb"
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, path)
    print("wrote and executed", path)


if __name__ == "__main__":
    main()
