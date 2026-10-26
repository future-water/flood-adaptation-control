"""Figure S10 (HUC planning flexibility): peak reduction lost under HUC-outlet control.

VIC5/MPI-ESM1-2-HR, SSP2-4.5, in the style of the figure notebooks in figures/.
  * panel subtitles: the number of outlet control points is the number of HUC12/HUC10/HUC8 units
    that contain network reaches in the mapping used by the HUC run, i.e. the columns of B in
    analyses/huc_control/si_huc_comparison.py (947, 158, 25).
  * percentages are computed from the run summary (flood-volume reduction at 10 Gm3/yr, linear
    interpolation in effort, relative to reach-level control): 99.9/98.4/88.6%.
HUC10 and HUC8 boundaries are dissolved from the WBD HUC12 polygons by code prefix.
Called by figures/supplementary_figures.ipynb.

Inputs
  Zenodo record, extracted in FAC_DATA_ROOT (paths.DATA):
    shape/Flowlines_huc02_12_Brazos.shp, shape/WBDHU12_Brazos.shp   (shape.zip)
    huc_spatial_compact.csv                                          (per-reach peak reduction)
    outputs_huc_comparison/huc_comparison_VIC5_MPI-ESM1-2-HR_ssp245.csv (optional, 1.4 GB:
        only used to check the control-point counts against n_ctrl_units of the run)
  analysis archive outputs_huc_control.zip, data("huc_control"):
    huc_summary_VIC5_MPI-ESM1-2-HR_ssp245.csv   basin effort and residual per R_weight and level
    comid_huc_mapping_huc_run.csv               COMID -> HUC8/10/12 mapping used by the HUC run
Outputs, out("figS10_huc_flexibility"): figS10_huc_flexibility.pdf/.png, figS10_values.json

Run from anywhere:  python figures/figS10_huc_flexibility/make_figure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import DATA, data, out

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib import cm

SHP_PATH = DATA / "shape/Flowlines_huc02_12_Brazos.shp"
WBD12 = DATA / "shape/WBDHU12_Brazos.shp"
HUC_COMPACT = DATA / "huc_spatial_compact.csv"
HUC_FULL = DATA / "outputs_huc_comparison/huc_comparison_VIC5_MPI-ESM1-2-HR_ssp245.csv"
HUC_SUMMARY = data("huc_control") / "huc_summary_VIC5_MPI-ESM1-2-HR_ssp245.csv"
HUC_MAPPING = data("huc_control") / "comid_huc_mapping_huc_run.csv"

_missing = [p for p in (SHP_PATH, WBD12, HUC_COMPACT, HUC_SUMMARY, HUC_MAPPING) if not p.exists()]
if _missing:
    sys.exit("Missing inputs. Extract shape.zip, huc_spatial_compact.csv and outputs_huc_control.zip of the Zenodo "
             "record into FAC_DATA_ROOT:\n  "
             + "\n  ".join(str(p) for p in _missing))

MM = 1 / 25.4; W_FULL = 190 * MM
mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7,
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
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 0.8,
})


# ── helpers (as in figures/main_figures.ipynb) ──
def _segs(geom):
    if geom is None or geom.is_empty: return []
    if geom.geom_type == "LineString": return [np.asarray(geom.coords)]
    if geom.geom_type == "MultiLineString":
        return [np.asarray(g.coords) for g in geom.geoms]
    return []


def draw_net_cmap(ax, gdf, values, cmap, norm, lw, alpha=0.92):
    segs, cols, lws = [], [], []
    for geom, v, w in zip(gdf.geometry.values, values, lw):
        for part in _segs(geom):
            segs.append(part)
            cols.append(cmap(norm(v)) if np.isfinite(v) else (0.85, 0.85, 0.85, 0.3))
            lws.append(float(w))
    lc = LineCollection(segs, colors=cols, linewidths=lws,
                        capstyle="round", joinstyle="round", alpha=alpha, zorder=1)
    lc.set_rasterized(True); ax.add_collection(lc)


def add_bg(ax, gdf):
    gdf.plot(ax=ax, color="#f0f0f0", linewidth=0.02, alpha=0.22, zorder=0)


def add_scalebar(ax, gdf, km=100):
    b = gdf.total_bounds; bar = km / 96.0
    x0 = b[2] - bar - (b[2] - b[0]) * 0.04
    y0 = b[1] + (b[3] - b[1]) * 0.04
    ax.plot([x0, x0 + bar], [y0, y0], "k-", lw=0.8, solid_capstyle="butt", zorder=10)
    ax.text(x0 + bar / 2, y0 + (b[3] - b[1]) * 0.015, f"{km} km",
            ha="center", fontsize=5, zorder=10)


def tcmap(name, lo=0.0, hi=1.0, n=256):
    base = mpl.colormaps[name]
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_t", base(np.linspace(lo, hi, n)))


# ── data ──
gdf_net = gpd.read_file(SHP_PATH)
gdf_net["COMID"] = gdf_net["COMID"].astype(int)

df_huc = pd.read_csv(HUC_COMPACT)
huc_map = pd.read_csv(HUC_MAPPING)
huc_map["COMID"] = huc_map["COMID"].astype(int)
df_huc = df_huc.drop(columns=["HUC8", "HUC10", "HUC12"], errors="ignore")
df_huc = df_huc.merge(huc_map[["COMID", "HUC8", "HUC10", "HUC12"]],
                      left_on="reach_id", right_on="COMID", how="left").drop(columns=["COMID"], errors="ignore")
for col in ["HUC8", "HUC10", "HUC12"]:
    df_huc[col] = df_huc[col].dropna().astype(float).astype(np.int64).astype(str)
brazos_hucs = {lv: set(df_huc[lv].dropna().unique()) for lv in ["HUC8", "HUC10", "HUC12"]}
for lv in ["HUC12", "HUC10", "HUC8"]:
    df_huc[f"diff_{lv}"] = df_huc["peak_red_reach"] - df_huc[f"peak_red_{lv}"]
gdf_huc = gdf_net.merge(df_huc, left_on="COMID", right_on="reach_id", how="inner")
gdf_huc = gdf_huc.sort_values("TotDASqKM_x", ascending=True)

# control points: one outlet per HUC unit containing network reaches (columns of B in the run)
n_ctrl = {lv: len(brazos_hucs[lv]) for lv in ["HUC12", "HUC10", "HUC8"]}
if HUC_FULL.exists():
    run = (pd.read_csv(HUC_FULL, usecols=["huc_level", "n_ctrl_units"]).drop_duplicates()
           .set_index("huc_level")["n_ctrl_units"].to_dict())
    for lv in ["HUC12", "HUC10", "HUC8"]:
        assert run[lv] == n_ctrl[lv], (lv, run[lv], n_ctrl[lv])
    print("  control points match n_ctrl_units of the run output")

# % of reach-level reduction at 10 Gm3/yr (linear interpolation in effort)
s = pd.read_csv(HUC_SUMMARY)
red = {}
for lv in ["reach", "HUC12", "HUC10", "HUC8"]:
    sub = s[s["huc_level"] == lv].sort_values("effort_Gm3")
    red[lv] = sub["baseline_Gm3"].iloc[0] - np.interp(10, sub["effort_Gm3"], sub["residual_Gm3"])
pct = {lv: 100 * red[lv] / red["reach"] for lv in red}

# WBD boundaries: HUC12 polygons of the Brazos, dissolved by code prefix for HUC10 and HUC8
w12 = gpd.read_file(WBD12)
w12["HUC12"] = w12["huc12"].astype(str).str.strip()
if w12.crs is not None and gdf_net.crs is not None and w12.crs != gdf_net.crs:
    w12 = w12.to_crs(gdf_net.crs)
wbd_huc = {}
for lv, k in (("HUC12", 12), ("HUC10", 10), ("HUC8", 8)):
    w = w12.assign(**{lv: w12["HUC12"].str[:k]})
    w = w[w[lv].isin(brazos_hucs[lv])]
    w = w[[lv, "geometry"]].dissolve(by=lv).reset_index() if lv != "HUC12" else w[[lv, "geometry"]]
    wbd_huc[lv] = w
    print(f"  WBD {lv}: {len(w)} polygons (control points {n_ctrl[lv]})")

# ── figure ──
_bnds = gdf_net.total_bounds
_xp = (_bnds[2] - _bnds[0]) * 0.02; _yp = (_bnds[3] - _bnds[1]) * 0.02
_mxlim = (_bnds[0] - _xp, _bnds[2] + _xp)
_mylim = (_bnds[1] - _yp, _bnds[3] + _yp)

fig_s = plt.figure(figsize=(W_FULL, 85 * MM))
gs_s = fig_s.add_gridspec(1, 3, left=0.01, right=0.89, top=0.92, bottom=0.14, wspace=0.03)
cmap_diff = tcmap("RdBu_r", 0.05, 0.95)
diff_lim = 0.50
norm_diff = mcolors.TwoSlopeNorm(vcenter=0, vmin=-diff_lim, vmax=diff_lim)

_panels = [
    ("diff_HUC12", "a", f"HUC12 ({n_ctrl['HUC12']:,} outlets)", "HUC12", 0.3, f"{pct['HUC12']:.1f}%"),
    ("diff_HUC10", "b", f"HUC10 ({n_ctrl['HUC10']:,} outlets)", "HUC10", 0.6, f"{pct['HUC10']:.1f}%"),
    ("diff_HUC8", "c", f"HUC8 ({n_ctrl['HUC8']:,} outlets)", "HUC8", 1.0, f"{pct['HUC8']:.1f}%"),
]
for i, (col, lab, subtitle, bnd_lv, bnd_lw, p) in enumerate(_panels):
    ax = fig_s.add_subplot(gs_s[0, i])
    ax.set_aspect("equal"); ax.axis("off")
    add_bg(ax, gdf_net)
    gdf_p = gdf_huc.sort_values(col, key=lambda v: v.abs(), ascending=True)
    draw_net_cmap(ax, gdf_p, gdf_p[col].values, cmap_diff, norm_diff, np.full(len(gdf_p), 0.95))
    w = wbd_huc.get(bnd_lv)
    if w is not None and len(w) > 0:
        w.boundary.plot(ax=ax, color="#555555", linewidth=bnd_lw, alpha=0.45, zorder=5)
    ax.set_xlim(_mxlim); ax.set_ylim(_mylim)
    ax.text(-0.05, 0.97, lab, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    ax.text(0.50, 0.02, subtitle, transform=ax.transAxes, ha="center", fontsize=7, fontweight="bold",
            color="#444", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    ax.text(0.50, -0.04, f"{p} of reach-level baseline", transform=ax.transAxes,
            ha="center", fontsize=6.5, color="#666", style="italic")
    if i == 0: add_scalebar(ax, gdf_net)

cax_s = fig_s.add_axes([0.91, 0.18, 0.013, 0.65])
cb_s = fig_s.colorbar(cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff), cax=cax_s)
cb_s.set_label("Peak reduction lost under outlet control (pp)", fontsize=7, labelpad=8)
_dt = [-diff_lim, -diff_lim / 2, 0, diff_lim / 2, diff_lim]
cb_s.set_ticks(_dt)
cb_s.set_ticklabels([f"{t*100:+.0f}" for t in _dt])
cb_s.ax.tick_params(labelsize=6.5, width=0.3, length=2)
cb_s.outline.set_linewidth(0.3)

OUT = out("figS10_huc_flexibility")
for ext in ("png", "pdf"):
    fig_s.savefig(OUT / f"figS10_huc_flexibility.{ext}")
pd.Series({"n_ctrl_HUC12": n_ctrl["HUC12"], "n_ctrl_HUC10": n_ctrl["HUC10"], "n_ctrl_HUC8": n_ctrl["HUC8"],
           **{f"pct_{k}": round(v, 2) for k, v in pct.items()}}).to_json(OUT / "figS10_values.json", indent=1)
print("saved", OUT, {k: round(v, 1) for k, v in pct.items()}, n_ctrl)
