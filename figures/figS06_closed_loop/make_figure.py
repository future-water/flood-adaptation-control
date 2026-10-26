"""Figure S6: closed-loop test of the control law on the full-order routing model.

All 56 members (7 GCMs x 2 hydrologic models x 4 SSPs), 10 Gm3/yr budget. The production control law acts on the
routed discharge of the Muskingum routing model of the process-based simulations at the penalty weight at which it
spends the surrogate's own total effort; the surrogate's controlled discharge is its own closed-loop simulation.
Discharge is shown and scored as Q+ = max(Q, 0).

- (a, b) The two SSP2-4.5 members (VIC5, PRMS) of the GCM with the highest full-period mainstem NSE averaged over the two
  hydrologic models: the full 2020-2099 record of controlled discharge at the downstream mainstem reach (COMID 3125212),
  the five-year block whose daily NSE is closest to the full-period NSE, and the daily network attenuation effort
  sum_i |u_i| of the two loops over the same block.
- (c) Daily NSE and KGE by Strahler stream order, pooling all 56 members (median and interquartile range).
- (d) Annual attenuation effort per reach when the loop is closed on the routing model and on the surrogate, all
  reach-member pairs.
- (e) Map of the top decile of reaches by the ensemble-median effort of the 14 SSP2-4.5 members in both loops, in the
  routing-model loop only and in the surrogate loop only.
- (f) Percentile rank of the residual exceedance over 2020-2099 within each member, both loops, all reach-member pairs.

Reads only data("closed_loop") (or --data DIR, e.g. FAC_WORK_ROOT/closed_loop/data after a rerun) and writes
figS6_control_fidelity.{pdf,png,svg}, figS6_control_fidelity_statistics.json and
figS6_control_fidelity_timeseries_scores.csv to FAC_OUT_ROOT/figS6_closed_loop/. Style, palette and helpers (plabel,
lgrid) follow the figure code of the paper; the map follows Figure 5d and Figure S1 d-f.
Usage: python make_figure.py [--data DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import data, out  # noqa: E402

ap = argparse.ArgumentParser(description="Figure S6")
ap.add_argument("--data", type=Path, default=data("closed_loop"))
DATA = ap.parse_args().data
OUT = out("figS6_closed_loop")
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplcache'))

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.stats import spearmanr, rankdata  # noqa: E402

try:
    font_manager.findfont('Arial', fallback_to_default=False)
    FONT = 'Arial'
except ValueError:
    FONT = 'DejaVu Sans'

# ── Config, style and helpers of the paper figures ──
HYDROS = ['VIC5', 'PRMS']
SSP_LABELS = {126: 'SSP1-2.6', 245: 'SSP2-4.5', 370: 'SSP3-7.0', 585: 'SSP5-8.5'}
C_SUR, C_REF = '#2166AC', '#666'      # surrogate (as in Fig. 2) and reference (as the baseline in Fig. 3)
C_NSE, C_KGE = '#2166AC', '#d4820a'
C_BOTH, C_REFONLY, C_SURONLY = '#2d8e3d', '#d4820a', '#2166AC'   # map categories, as in Figure S1 d-f
STEM = 'figS6_control_fidelity'
EFF_RANGE = (-11, -1)                 # plotted range of panel (d), log10 Gm3/yr; contains every positive pair

MM = 1/25.4; W_FULL = 190*MM

mpl.rcParams.update({
    "font.family": FONT, "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "axes.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.fontsize": 6, "legend.frameon": False,
    "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
    "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "lines.linewidth": 0.8,
})

def plabel(ax, lab, x=-0.08, y=1.05):
    ax.text(x, y, lab, transform=ax.transAxes,
            fontweight="bold", fontsize=9, va="top", ha="right")

def plabel_map(ax, lab):
    ax.text(0.03, 0.97, lab, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

def lgrid(ax, alpha=0.10, lw=0.3):
    ax.grid(True, alpha=alpha, lw=lw, zorder=0); ax.set_axisbelow(True)

def daily_scores(reference, surrogate):
    x, y = np.asarray(reference, float), np.asarray(surrogate, float)
    r = np.corrcoef(x, y)[0, 1]
    return dict(NSE=float(1 - np.mean((y - x)**2) / x.var()),
                KGE=float(1 - np.sqrt((r - 1)**2 + (y.std()/x.std() - 1)**2 + (y.mean()/x.mean() - 1)**2)))

def export(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor="white", dpi=600 if ext != "png" else 300)
    plt.close(fig)


members = pd.read_csv(DATA / 'members_fidelity.csv')
summary = json.loads((DATA / 'summary_fidelity.json').read_text())
fd = np.load(DATA / 'figure_data.npz')
cases = fd['cases'].tolist()
assert len(cases) == 56 and members.case.tolist() == cases
T = fd['ms_ref_cl'].shape[1]
dates = pd.date_range('2020-01-01', periods=T, freq='D')
record = {'output_convention': 'Qplus = max(Q, 0)', 'panels_ab_timeseries': []}

# (a, b) SSP2-4.5 GCM with the highest full-period mainstem NSE averaged over the two hydrologic models
im = np.load(DATA / 'inputs_map.npz')
m245 = members[members.ssp == 245]
pair_mean = m245.pivot_table(index='gcm', columns='hydro', values='ms_nse')[HYDROS].mean(axis=1)
best_gcm = pair_mean.idxmax()
assert str(im['shown_gcm']) == best_gcm and im['shown_cases'].tolist() == [f'{h}_{best_gcm}_ssp245_budget10' for h in HYDROS]
in_dates = pd.to_datetime(im['dates'])
selected = {}
for j, h in enumerate(HYDROS):
    g = members[members.hydro == h]
    pick = g[(g.gcm == best_gcm) & (g.ssp == 245)].iloc[0]
    i = cases.index(pick.case)
    ref, sur = np.maximum(fd['ms_ref_cl'][i], 0).astype(float), np.maximum(fd['ms_sur_cl'][i], 0).astype(float)
    full = daily_scores(ref, sur)
    # representative block: the five-year block whose daily NSE is closest to the full-period NSE
    blocks = {b: (dates.year >= b) & (dates.year < b + 5) for b in range(2020, 2100, 5)}
    start = min(blocks, key=lambda b: abs(daily_scores(ref[blocks[b]], sur[blocks[b]])['NSE'] - full['NSE']))
    w = blocks[start]
    win = daily_scores(ref[w], sur[w])
    assert abs(full['NSE'] - pick.ms_nse) < 1e-6
    e_sur, e_ref = im['effort_sur'][j], im['effort_ref'][j]
    w_in = (in_dates.year >= start) & (in_dates.year < start + 5)
    r_full, r_win = float(np.corrcoef(e_sur, e_ref)[0, 1]), float(np.corrcoef(e_sur[w_in], e_ref[w_in])[0, 1])
    selected[h] = dict(case=pick.case, gcm=pick.gcm, ssp=int(pick.ssp), reference=ref, surrogate=sur, window=w,
                       years=(start, start + 4), full=full, win=win, e_sur=e_sur, e_ref=e_ref, w_in=w_in,
                       r_full=r_full, r_win=r_win)
    record['panels_ab_timeseries'] += [dict(hydro=h, gcm=pick.gcm, ssp=int(pick.ssp), interval='2020-2099', **full, effort_r=r_full),
                                       dict(hydro=h, gcm=pick.gcm, ssp=int(pick.ssp), interval=f'{start}-{start + 4}', **win, effort_r=r_win)]
record['panels_ab_selection'] = {h: dict(case=s['case'], rule='SSP2-4.5 GCM with the highest full-period mainstem NSE averaged over the '
                                                              'two hydrologic models; '
                                                              'block: five-year block with daily NSE closest to the full-period NSE',
                                         pair_mean_ms_nse=float(pair_mean.max())) for h, s in selected.items()}
record['mainstem_nse_all_members'] = dict(min=float(members.ms_nse.min()), max=float(members.ms_nse.max()))

# (c) daily scores by stream order, pooled over reaches and members
al = summary['all']
record['panel_c_stream_orders'] = [dict(stream_order=int(o), NSE=v['nse'], KGE=v['kge']) for o, v in al['by_order'].items()]
# (d) annual attenuation effort per reach, both closed loops, all reach-member pairs
es, ec = fd['es_reach'], fd['ec_reach']            # Gm3/yr per reach: surrogate loop, loop closed on the routing model
k = int(round(0.1 * es.shape[1]))
overlap = np.array([len(set(np.argsort(-es[i])[:k]) & set(np.argsort(-ec[i])[:k])) / k * 100 for i in range(len(cases))])
rho_eff = np.array([spearmanr(ec[i], es[i]).statistic for i in range(len(cases))])
assert np.allclose(overlap, members.alloc_top10, atol=1e-6) and np.allclose(rho_eff, members.alloc_rho, atol=1e-6)
pos = (es > 0) & (ec > 0)
lx, ly = np.log10(ec[pos]), np.log10(es[pos])
assert not ((lx < EFF_RANGE[0]) | (ly < EFF_RANGE[0]) | (lx > EFF_RANGE[1]) | (ly > EFF_RANGE[1])).any()
record['panel_d_effort'] = dict(quantity='annual attenuation effort per reach (Gm3/yr) over 2020-2099, both closed loops',
                                top_decile_overlap_pct=dict(median=float(np.median(overlap)), min=float(overlap.min()), max=float(overlap.max())),
                                spearman=dict(median=float(np.median(rho_eff)), min=float(rho_eff.min()), max=float(rho_eff.max())))
# (e) map: top decile of the ensemble-median effort of the 14 SSP2-4.5 members, both loops
map_cat = im['map_cat']; map_ids = im['map_comid']
offsets, coordinates, bounds = im['offsets'], im['coordinates'], im['bounds']
segments = [coordinates[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
assert len(segments) == len(map_ids) == 14415
n_top, n_both, n_ref, n_sur, n_unmapped = (int(v) for v in im['map_counts'])
assert n_unmapped == 0 and n_both == (map_cat == 1).sum() and n_ref == (map_cat == 2).sum() and n_sur == (map_cat == 3).sum()
record['panel_e_map'] = dict(members='14 SSP2-4.5 members, ensemble-median effort', n_top=n_top, both=n_both,
                             routing_only=n_ref, surrogate_only=n_sur, overlap_pct=100 * n_both / n_top)
# (f) percentile rank of residual exceedance within each member, both closed loops
rd = np.load(DATA / 'residual_exceedance.npz')
assert rd['cases'].tolist() == cases
res_ref, res_sur = rd['rank_ref'], rd['rank_sur']   # within-member ranks of residual exceedance
rho_res = np.array([spearmanr(res_ref[i], res_sur[i]).statistic for i in range(len(cases))])
assert np.allclose(rho_res, members.resid_rho, atol=1e-6)
n_reach = res_ref.shape[1]
pr_ref = np.stack([100 * (rankdata(v) - 0.5) / n_reach for v in res_ref]).ravel()   # ties averaged
pr_sur = np.stack([100 * (rankdata(v) - 0.5) / n_reach for v in res_sur]).ravel()
k_res = int(round(0.1 * n_reach))
ov_res = np.array([len(set(np.argsort(-res_ref[i])[:k_res]) & set(np.argsort(-res_sur[i])[:k_res])) / k_res * 100 for i in range(len(cases))])
record['panel_f_residual'] = dict(quantity='percentile rank, within each member, of the residual exceedance above the 2-year level over 2020-2099',
                                  spearman=dict(median=float(np.median(rho_res)), min=float(rho_res.min()), max=float(rho_res.max())),
                                  top_decile_overlap_pct=dict(median=float(np.median(ov_res)), min=float(ov_res.min()), max=float(ov_res.max())))
print(pd.DataFrame(record['panels_ab_timeseries']).to_string(index=False))
print(record['panel_d_effort'], record['panel_e_map'], record['panel_f_residual'])

fig = plt.figure(figsize=(W_FULL, 150*MM))
gs_top = fig.add_gridspec(1, 3, left=0.07, right=0.985, top=0.97, bottom=0.50, wspace=0.34, width_ratios=[1.5, 1.5, 1])
gs_bot = fig.add_gridspec(1, 3, left=0.07, right=0.985, top=0.395, bottom=0.07, wspace=0.42, width_ratios=[1, 1.65, 1])

# ══ (a, b) Mainstem controlled discharge and network attenuation effort ══
for i, h in enumerate(HYDROS):
    s = selected[h]
    w, (y0, y1) = s['window'], s['years']
    sub = gs_top[0, i].subgridspec(3, 1, height_ratios=[0.24, 1.0, 0.46], hspace=0.34)

    # complete record, with the enlarged block shaded
    strip = fig.add_subplot(sub[0])
    strip.plot(dates, s['reference'], color=C_REF, lw=0.35)
    strip.plot(dates, s['surrogate'], color=C_SUR, lw=0.25)
    strip.axvspan(dates[w][0], dates[w][-1], color="#bbb", alpha=0.45, lw=0, zorder=0)
    strip.set_xlim(dates[0], pd.Timestamp('2100-01-01')); strip.set_ylim(0, None)
    strip.set_xticks(pd.to_datetime([f'{y}-01-01' for y in (2020, 2040, 2060, 2080, 2100)]))
    strip.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    strip.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, integer=True))
    strip.tick_params(labelsize=5.5, pad=1.5)
    plabel(strip, "ab"[i], x=-0.04, y=1.35)

    # enlarged five-year block: controlled discharge of both loops
    ax = fig.add_subplot(sub[1])
    ax.plot(dates[w], s['reference'][w], color=C_REF, lw=1.0)
    ax.plot(dates[w], s['surrogate'][w], color=C_SUR, lw=0.9)
    ax.set_xlim(dates[w][0], dates[w][-1])
    ax.set_ylim(0, 1.42 * max(s['reference'][w].max(), s['surrogate'][w].max()))
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelbottom=False)
    ax.yaxis.grid(True, which="major", color="#e0e0e0", lw=0.4); ax.set_axisbelow(True)
    ax.text(0.97, 0.94, f"{h} / {s['gcm']} / {SSP_LABELS[s['ssp']]}", transform=ax.transAxes,
            fontsize=6.5, fontweight="bold", color=C_SUR, ha="right", va="top")
    ax.text(0.97, 0.83, f"NSE {s['win']['NSE']:.2f}, KGE {s['win']['KGE']:.2f}"
                        f"  (2020–2099: {s['full']['NSE']:.2f}, {s['full']['KGE']:.2f})",
            transform=ax.transAxes, fontsize=5.8, color="#555", ha="right", va="top")
    if i == 0:
        ax.set_ylabel(r"Controlled discharge (m$^3$ s$^{-1}$)")

    # control input: daily network attenuation effort of both loops over the same block
    axu = fig.add_subplot(sub[2], sharex=ax)
    wi = s['w_in']
    axu.plot(in_dates[wi], s['e_ref'][wi] / 1e3, color=C_REF, lw=0.9)
    axu.plot(in_dates[wi], s['e_sur'][wi] / 1e3, color=C_SUR, lw=0.8)
    axu.set_ylim(0, 1.3 * max(s['e_ref'][wi].max(), s['e_sur'][wi].max()) / 1e3)
    axu.xaxis.set_major_locator(mdates.YearLocator()); axu.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axu.yaxis.grid(True, which="major", color="#e0e0e0", lw=0.4); axu.set_axisbelow(True)
    axu.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    axu.text(0.97, 0.92, f"daily effort $r$ = {s['r_win']:.2f}  (2020–2099: {s['r_full']:.2f})",
             transform=axu.transAxes, fontsize=5.8, color="#555", ha="right", va="top")
    if i == 0:
        axu.set_ylabel("$\\sum_i |u_i|$\n" + r"(10$^3$ m$^3$ s$^{-1}$)")

fig.legend(handles=[Line2D([0], [0], color=C_REF, lw=1.0, label="Loop on routing model"),
                    Line2D([0], [0], color=C_SUR, lw=1.0, label="Surrogate loop")],
           loc="upper left", bbox_to_anchor=(0.115, 1.005), ncol=2, fontsize=6, frameon=False,
           handlelength=1.4, columnspacing=1.2, handletextpad=0.5)

# ══ (c) Daily NSE and KGE by stream order ════════════════════════
ax = fig.add_subplot(gs_top[0, 2])
yb = np.arange(7); off = 0.19
for j, row in enumerate(record['panel_c_stream_orders']):
    for metric, color, marker, y in (("NSE", C_NSE, "o", yb[j] + off), ("KGE", C_KGE, "s", yb[j] - off)):
        q = row[metric]
        ax.barh(y, q['p75'] - q['p25'], left=q['p25'], height=0.32,
                color=color, alpha=0.25, edgecolor="none")
        ax.plot(q['median'], y, marker, color=color, ms=4.2 if marker == "o" else 3.8,
                markeredgecolor="white", markeredgewidth=0.4, zorder=5)
    x_label = max(row['NSE']['p75'], row['KGE']['p75']) + 0.025
    ax.text(x_label, yb[j] + off, f"{row['NSE']['median']:.2f}", fontsize=5.5, va="center", color=C_NSE)
    ax.text(x_label, yb[j] - off, f"{row['KGE']['median']:.2f}", fontsize=5.5, va="center", color=C_KGE)
ax.set_yticks(yb); ax.set_yticklabels([str(r['stream_order']) for r in record['panel_c_stream_orders']])
ax.set_ylabel("Stream order")
ax.set_xlabel("Daily NSE, KGE of controlled\ndischarge (median ± IQR)")
ax.set_xlim(0, 1.12); ax.set_ylim(-0.6, 6.6)
lgrid(ax, alpha=0.08)
plabel(ax, "c", x=-0.26, y=1.06)
handles_dot = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NSE, ms=4.2,
           markeredgecolor="white", markeredgewidth=0.3, label="NSE"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_KGE, ms=3.8,
           markeredgecolor="white", markeredgewidth=0.3, label="KGE"),
]
ax.legend(handles=handles_dot, loc="lower left", fontsize=6, handletextpad=0.3)

# ══ (d) Attenuation effort per reach, both closed loops ══════════
ax = fig.add_subplot(gs_bot[0, 0])
# rasterized: the PDF backend of matplotlib 3.11 drops hexagons whose offsets from the data origin are negative
ax.hexbin(lx, ly, gridsize=45, extent=(*EFF_RANGE, *EFF_RANGE), bins="log", cmap="Blues", mincnt=1,
          linewidths=0, alpha=0.75, zorder=1, rasterized=True)
ax.plot(EFF_RANGE, EFF_RANGE, "--", color="#333", lw=0.6, zorder=5)
c = record['panel_d_effort']
ax.text(0.96, 0.04, f"$\\rho$ = {c['spearman']['median']:.3f}\ntop decile {c['top_decile_overlap_pct']['median']:.0f}%",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", lw=0.4, alpha=0.92))
ax.set(xlim=EFF_RANGE, ylim=EFF_RANGE, aspect="equal")
ax.set_xticks(range(-10, 0, 3)); ax.set_yticks(range(-10, 0, 3))
ax.set_xlabel("Effort, loop on routing model\n" + r"(log$_{10}$ Gm$^3$ yr$^{-1}$)")
ax.set_ylabel("Effort, surrogate loop\n" + r"(log$_{10}$ Gm$^3$ yr$^{-1}$)")
lgrid(ax); plabel(ax, "d", x=-0.28, y=1.06)

# ══ (e) Map of the top decile by effort in the two loops ═════════
ax = fig.add_subplot(gs_bot[0, 1])
ax.set_aspect("equal"); ax.axis("off")
x0, y0, x1, y1 = bounds; dx, dy = x1 - x0, y1 - y0
pick = lambda cc: [seg for seg, v in zip(segments, map_cat) if v == cc]
ax.add_collection(LineCollection(segments, colors="#c4c4c4", linewidths=0.2, alpha=0.75, zorder=0, rasterized=True))
ax.add_collection(LineCollection(pick(1), colors=C_BOTH, linewidths=0.95, alpha=0.92, zorder=1,
                                 capstyle="round", joinstyle="round", rasterized=True))
for cc, color, z in ((2, C_REFONLY, 2), (3, C_SURONLY, 4)):
    ax.add_collection(LineCollection(pick(cc), colors="white", linewidths=2.2, zorder=z,
                                     capstyle="round", joinstyle="round", rasterized=True))
    ax.add_collection(LineCollection(pick(cc), colors=color, linewidths=1.4, alpha=0.92, zorder=z + 1,
                                     capstyle="round", joinstyle="round", rasterized=True))
ax.set_xlim(x0 - 0.01 * dx, x1 + 0.01 * dx); ax.set_ylim(y0 - 0.01 * dy, y1 + 0.01 * dy)
plabel_map(ax, "e")
mp = record['panel_e_map']
ax.text(0.98, 0.97, f"Top decile by effort\n{mp['overlap_pct']:.1f}% in both loops", transform=ax.transAxes,
        fontsize=6.2, ha="right", va="top", color="#333")
ax.legend(handles=[Line2D([0], [0], color=C_BOTH, lw=1.8, label=f"Both loops ({mp['both']:,})"),
                   Line2D([0], [0], color=C_REFONLY, lw=1.8, label=f"Routing loop only ({mp['routing_only']})"),
                   Line2D([0], [0], color=C_SURONLY, lw=1.8, label=f"Surrogate loop only ({mp['surrogate_only']})")],
          loc="lower left", fontsize=6, frameon=False, handlelength=1.8, labelspacing=0.3, borderpad=0.1)
bar = 100e3; sx, sy = x0 + dx * 0.74, y0 + dy * 0.04
ax.plot([sx, sx + bar], [sy, sy], "k-", lw=0.8, solid_capstyle="butt", zorder=10)
ax.text(sx + bar / 2, sy + dy * 0.015, "100 km", ha="center", fontsize=5.5, zorder=10)

# ══ (f) Rank of residual exceedance, both closed loops ═══════════
ax = fig.add_subplot(gs_bot[0, 2])
ax.hexbin(pr_ref, pr_sur, gridsize=45, extent=(0, 100, 0, 100), bins="log", cmap="Blues", mincnt=1,
          linewidths=0, alpha=0.75, zorder=1, rasterized=True)
ax.plot([0, 100], [0, 100], "--", color="#333", lw=0.6, zorder=5)
dd = record['panel_f_residual']['spearman']
ax.text(0.96, 0.04, f"$\\rho$ = {dd['median']:.2f}\ntop decile {record['panel_f_residual']['top_decile_overlap_pct']['median']:.0f}%",
        transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", lw=0.4, alpha=0.92))
ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal")
ax.set_xticks(range(0, 101, 50)); ax.set_yticks(range(0, 101, 50))
ax.set_xlabel("Residual exceedance rank,\nloop on routing model (percentile)")
ax.set_ylabel("Residual exceedance\nrank, surrogate loop")
lgrid(ax); plabel(ax, "f", x=-0.28, y=1.06)

export(fig, STEM)

(OUT / f'{STEM}_statistics.json').write_text(json.dumps(record, indent=2) + '\n')
pd.DataFrame(record['panels_ab_timeseries']).to_csv(OUT / f'{STEM}_timeseries_scores.csv', index=False)
print('saved', sorted(p.name for p in OUT.glob(f'{STEM}*')), 'in', OUT)
