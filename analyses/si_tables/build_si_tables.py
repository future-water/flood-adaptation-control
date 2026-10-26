"""Supporting Information tables and the closed-loop numbers of the paper, from the archived data.

Light step (seconds). Reads only the extracted analysis archives (data(...)) and writes to out('si_tables'):

  table_s1.tex                  Table S1 as printed: top (reduced dimension r), middle (rank pairs) and bottom
                                (basis period) blocks
  table_s1_top.csv, table_s1_middle.csv, table_s1_bottom.csv   unrounded values of the three blocks
  table_s2.tex, table_s2.csv    Table S2 (5-year vs 2-year flood threshold at 5 and 10 Gm3/yr), printed and unrounded
  closed_loop_numbers.csv       numbers quoted in the Figure S6 caption and in Section 3.1: unrounded values, the
                                printed text and the file they come from
  closed_loop_numbers.tex       the same printed text as LaTeX macros

Every printed row and number is compared with the published text, and the script stops if one differs (--no-check
skips the comparison, for example for rerun data).

Inputs (archive folders FAC_DATA_ROOT/outputs_<name>/; each can be replaced by a rerun folder of the same layout):
  rank_sensitivity  singular_spectra_all14.csv, same_period_accuracy_all14.csv, member_rank_status.csv,
                    ensemble_rank_comparisons.csv                              Table S1 top and middle
  basis_period      results/<member>.npz, results_basis/<member>_b10.{npz,json};
                    outputs_common/reach_attributes.csv                        Table S1 bottom
  flood_threshold   effort/effort_rl5_r30_*_ssp245.npz, rl2_production_effort_ssp245.npz,
                    stream_order.csv                                           Table S2
  closed_loop       members_fidelity.csv, figure_data.npz,
                    figS6_control_fidelity_statistics.json, inputs_map.npz     Figure S6 caption, Section 3.1

The computations are those of the analysis scripts: Table S1 bottom uses basis_period/summarize_basis.py, Table S2
flood_threshold/compare_rl2_rl5.py and the closed-loop summary closed_loop/summarize_fidelity.py. The Figure S6
statistics file is written by figures/figS06_closed_loop/make_figure.py, which reproduces the archived copy exactly.

Usage: python analyses/si_tables/build_si_tables.py [--no-check] [--rank DIR] [--basis DIR] [--threshold DIR]
       [--closed-loop DIR] [--figs6-statistics FILE]
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ANALYSES = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSES.parent))   # repository root (paths)
for sub in ("basis_period", "flood_threshold", "closed_loop"):   # the analysis modules import their siblings
    sys.path.insert(0, str(ANALYSES / sub))
from paths import data, out  # noqa: E402
from summarize_basis import summarize  # noqa: E402
from compare_rl2_rl5 import compare, rl2_members, stream_order  # noqa: E402
from summarize_fidelity import group_stats  # noqa: E402

N_MEMBERS_SSP245 = 14
N_MEMBERS_ALL = 56
HYDROS = ("VIC5", "PRMS")
RANKS = [10, 20, 30, 50, 75, 100]
RANK_PAIRS = [(20, 30), (30, 50), (20, 50)]
RANK_SCOPE = "full14_original_fallback"   # all 14 members
BASIS_YEARS = 10
BUDGETS = (5.0, 10.0)                       # Gm3/yr, the two columns of Table S2

# ---------------------------------------------------------------- published text (Supporting Information, Section 3.1)
PUBLISHED_S1_TOP = [
    r"10 & 99.159 & 0.36 & 0.42 & 0.990 & 14/14 \\",
    r"20 & 99.801 & 0.59 & 0.63 & 0.983 & 14/14 \\",
    r"30 & 99.926 & 0.69 & 0.72 & 0.992 & 14/14 \\",
    r"50 & 99.981 & 0.77 & 0.72 & 0.994 & 13/14 \\",
    r"75 & 99.993 & 0.75 & 0.59 & 0.995 & 7/14 \\",
    r"100 & 99.997 & 0.65 & 0.43 & 0.995 & 3/14 \\",
]
PUBLISHED_S1_MIDDLE = [
    r"20 vs 30 & 0.994 & 91.4 \\",
    r"30 vs 50 & 0.984 & 87.0 \\",
    r"20 vs 50 & 0.981 & 88.0 \\",
]
PUBLISHED_S1_BOTTOM = [
    r"2020--2029 & 0.57 & 0.60 & 93.8 (86; 72--94) & 0.996 \\",
    r"2020--2099 (main analysis; scored 2030--2099) & 0.70 & 0.72 & -- & -- \\",
]
PUBLISHED_S2 = [
    r"Spearman $\rho$ & 0.997 & 0.993 \\",
    r"Top-10\% overlap (\%) & 93.1 & 90.7 \\",
    r"Effort share on stream order 7 (\%), 2-year $\rightarrow$ 5-year & 15.7 $\rightarrow$ 22.0 & "
    r"21.3 $\rightarrow$ 42.1 \\",
]
PUBLISHED_S2_CAPPED = {5.0: 0, 10.0: 2}     # Table S2 caption: two members evaluated at their maximum effort
PUBLISHED_CLOSED_LOOP = {
    "n_members": "56",
    "ab_member_a": "VIC5/MRI-ESM2-0/SSP2-4.5",
    "ab_member_b": "PRMS/MRI-ESM2-0/SSP2-4.5",
    "ab_mainstem_nse_range": "0.19--0.81",
    "ab_effort_correlation": "0.87 and 0.96",
    "d_effort_top_decile": r"92\% (85--97\%)",
    "d_effort_spearman": r"$\rho = 0.998$ (0.993--0.999)",
    "e_map_both": "1,359",
    "e_map_routing_only": "83",
    "e_map_surrogate_only": "83",
    "f_residual_spearman": "0.98 (0.96--0.99)",
    "f_residual_top_decile": r"95\% (87--97\%)",
    "sec31_nse_orders_5_7": "0.72--0.80",
    "sec31_effort_top_decile": r"92\%",
}
MACROS = {
    "n_members": "FigSixMembers", "ab_member_a": "FigSixMemberA", "ab_member_b": "FigSixMemberB",
    "ab_mainstem_nse_range": "FigSixMainstemNSERange", "ab_effort_correlation": "FigSixEffortCorrelation",
    "d_effort_top_decile": "FigSixEffortTopDecile", "d_effort_spearman": "FigSixEffortSpearman",
    "e_map_both": "FigSixMapBoth", "e_map_routing_only": "FigSixMapRoutingOnly",
    "e_map_surrogate_only": "FigSixMapSurrogateOnly", "f_residual_spearman": "FigSixResidualSpearman",
    "f_residual_top_decile": "FigSixResidualTopDecile", "sec31_nse_orders_5_7": "SecThreeOneNSEOrders",
    "sec31_effort_top_decile": "SecThreeOneEffortTopDecile",
}


def read_csv(path):
    # round_trip parsing gives the same floats as the text in the file
    return pd.read_csv(path, float_precision="round_trip")


def tabular(columns, header, rows):
    return [rf"\begin{{tabular}}{{{columns}}}", r"\hline", *header, r"\hline", *rows, r"\hline", r"\end{tabular}"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- Table S1
def table_s1_rank(data):
    """Top and middle blocks: medians over the 14 members per r, and ensemble-median allocations between ranks."""
    spec = read_csv(data / "singular_spectra_all14.csv")
    acc = read_csv(data / "same_period_accuracy_all14.csv")
    status = read_csv(data / "member_rank_status.csv")
    comp = read_csv(data / "ensemble_rank_comparisons.csv")
    top, top_tex = [], []
    for r in RANKS:
        e = spec.loc[spec["mode"] == r, "cumulative_fraction"]          # retained energy of the leading r modes
        a = acc[acc["rank"] == r]
        s = status[status["rank"] == r]
        assert len(e) == len(a) == len(s) == N_MEMBERS_SSP245, r
        row = dict(rank=r, retained_energy_percent=100 * e.median(),
                   recursive_nse=a["recursive_nse_median"].median(), recursive_kge=a["recursive_kge_median"].median(),
                   largest_spectral_radius=s["max_radius"].median(),
                   admissible_members=int(s["strict_valid"].astype(bool).sum()), members=N_MEMBERS_SSP245)
        top.append(row)
        top_tex.append(f"{r} & {row['retained_energy_percent']:.3f} & {row['recursive_nse']:.2f} & "
                       f"{row['recursive_kge']:.2f} & {row['largest_spectral_radius']:.3f} & "
                       f"{row['admissible_members']}/{N_MEMBERS_SSP245} \\\\")
    middle, middle_tex = [], []
    for a, b in RANK_PAIRS:
        c = comp[(comp["scope"] == RANK_SCOPE) & (comp["metric"] == "raw") & (comp["rank_a"] == a) & (comp["rank_b"] == b)]
        assert len(c) == 1 and int(c["n_members"].iloc[0]) == N_MEMBERS_SSP245, (a, b)
        c = c.iloc[0]
        row = dict(ranks=f"{a} vs {b}", spearman=c["spearman"], top10_retained_percent=100 * c["top10_retention_a"],
                   top10_n=int(c["top10_n_a"]), top10_shared=int(c["top10_common"]), n_reaches=int(c["n_reaches"]))
        middle.append(row)
        middle_tex.append(f"{a} vs {b} & {row['spearman']:.3f} & {row['top10_retained_percent']:.1f} \\\\")
    return top, top_tex, middle, middle_tex


def table_s1_basis(data):
    """Bottom block: first-decade basis against the main analysis, both scored over 2030-2099."""
    prod = [np.load(f) for f in sorted((data / "results").glob("*.npz"))]
    assert len(prod) == N_MEMBERS_SSP245
    pn = float(np.median([np.nanmedian(d["nse_production_test"]) for d in prod]))
    pk = float(np.median([np.nanmedian(d["kge_production_test"]) for d in prod]))
    b = summarize(data, BASIS_YEARS)
    assert b["n_members"] == N_MEMBERS_SSP245 and not b["excluded"]   # caption: all 14 members are admissible
    m, em = b["members"]["top_decile_overlap"], b["ensemble_median"]
    nse, kge = b["skill_2030_2099"]["nse_median"]["median"], b["skill_2030_2099"]["kge_median"]["median"]
    blank = dict.fromkeys(["top10_ensemble_median", "top10_members_median", "top10_members_min", "top10_members_max",
                           "spearman_ensemble_median"], "")
    rows = [dict(basis_period=f"2020-{2019 + BASIS_YEARS}", nse=nse, kge=kge,
                 top10_ensemble_median=em["top_decile_overlap"], top10_members_median=m["median"],
                 top10_members_min=m["min"], top10_members_max=m["max"], spearman_ensemble_median=em["spearman"],
                 n_members=b["n_members"]),
            dict(basis_period="2020-2099 (main analysis; scored 2030-2099)", nse=pn, kge=pk, **blank,
                 n_members=len(prod))]
    tex = [f"2020--{2019 + BASIS_YEARS} & {nse:.2f} & {kge:.2f} & {em['top_decile_overlap']:.1f} "
           f"({m['median']:.0f}; {m['min']:.0f}--{m['max']:.0f}) & {em['spearman']:.3f} \\\\",
           f"2020--2099 (main analysis; scored 2030--2099) & {pn:.2f} & {pk:.2f} & -- & -- \\\\"]
    return rows, tex


# ---------------------------------------------------------------- Table S2
def table_s2(data):
    rl2 = rl2_members(SimpleNamespace(results_dir=None, rl2=data / "rl2_production_effort_ssp245.npz"))
    order = stream_order(SimpleNamespace(flowlines=None, stream_order=data / "stream_order.csv"))
    rl5_max = {f.name: float(np.load(f)["eff"].sum(axis=1).max() / 1e9)
               for f in sorted((data / "effort").glob("effort_rl5_r30_*_ssp245.npz"))}
    assert len(rl5_max) == N_MEMBERS_SSP245
    res, capped = {}, {}
    for t in BUDGETS:
        res[t], ms = compare(t, rl2, order, data / "effort")
        capped[t] = sorted(ms.member[ms.effort5 < t * (1 - 1e-9)])
        assert len(capped[t]) == sum(v < t for v in rl5_max.values())   # capped members are at their maximum effort
    o7 = {t: res[t]["effort_share_by_order_pct"]["7"] for t in BUDGETS}
    a, b = BUDGETS
    tex = [f"Spearman $\\rho$ & {res[a]['ensemble_spearman']:.3f} & {res[b]['ensemble_spearman']:.3f} \\\\",
           f"Top-10\\% overlap (\\%) & {res[a]['top10_overlap_pct']:.1f} & {res[b]['top10_overlap_pct']:.1f} \\\\",
           f"Effort share on stream order 7 (\\%), 2-year $\\rightarrow$ 5-year & "
           f"{o7[a]['rl2']:.1f} $\\rightarrow$ {o7[a]['rl5']:.1f} & {o7[b]['rl2']:.1f} $\\rightarrow$ {o7[b]['rl5']:.1f} \\\\"]
    rows = [dict(budget_Gm3yr=t, n_reaches=res[t]["n_reaches"], spearman=res[t]["ensemble_spearman"],
                 top10_overlap_percent=res[t]["top10_overlap_pct"], order7_share_rl2_percent=o7[t]["rl2"],
                 order7_share_rl5_percent=o7[t]["rl5"], members_at_maximum_effort=len(capped[t]),
                 members_at_maximum_effort_names=" ".join(capped[t])) for t in BUDGETS]
    return rows, tex, {t: len(v) for t, v in capped.items()}


# ---------------------------------------------------------------- Figure S6 caption and Section 3.1
def spread(d, fmt, pct=False):
    """median (min--max) as printed; pct adds the percent signs of the caption."""
    p = "\\%" if pct else ""
    return f"{d['median']:{fmt}}{p} ({d['min']:{fmt}}--{d['max']:{fmt}}{p})"


def closed_loop_numbers(data, stats_file):
    members = pd.read_csv(data / "members_fidelity.csv")
    with np.load(data / "figure_data.npz") as fd:
        assert fd["cases"].tolist() == members.case.tolist()
        full = {k: fd[k] for k in ("order", "nse_q", "kge_q")}
    summ = group_stats(members.to_dict("records"), [{k: v[i] for k, v in full.items()} for i in range(len(members))])
    assert summ["n_members"] == N_MEMBERS_ALL
    stats = json.loads(Path(stats_file).read_text())
    with np.load(data / "inputs_map.npz") as im:
        map_counts = [int(v) for v in im["map_counts"]]
    al = summ["members"]
    d, e, f = stats["panel_d_effort"], stats["panel_e_map"], stats["panel_f_residual"]
    # the figure statistics agree with the member table (as the figure script checks)
    for key, panel in (("alloc_top10", d["top_decile_overlap_pct"]), ("alloc_rho", d["spearman"]),
                       ("resid_rho", f["spearman"])):
        assert all(abs(al[key][s] - panel[s]) < 1e-9 for s in ("median", "min", "max")), key
    assert map_counts == [e["n_top"], e["both"], e["routing_only"], e["surrogate_only"], 0]
    assert stats["mainstem_nse_all_members"] == al["ms_nse"]
    # panels (a,b): the SSP2-4.5 GCM with the highest mainstem NSE averaged over the two hydrologic models
    m245 = members[members.ssp == 245]
    best = m245.pivot_table(index="gcm", columns="hydro", values="ms_nse")[list(HYDROS)].mean(axis=1).idxmax()
    sel = stats["panels_ab_selection"]
    assert [sel[h]["case"] for h in HYDROS] == [f"{h}_{best}_ssp245_budget10" for h in HYDROS]
    r_full = {r["hydro"]: r["effort_r"] for r in stats["panels_ab_timeseries"] if r["interval"] == "2020-2099"}
    o57 = {o: summ["by_order"][o]["nse"]["median"] for o in (5, 6, 7)}
    ms = al["ms_nse"]
    sname = Path(stats_file).name
    items = [  # key, quantity, unrounded values, printed text, source
        ("n_members", "members in the closed-loop test (Figure S6; Sections 2.8, 3.1)",
         dict(n=summ["n_members"]), f"{summ['n_members']}", "members_fidelity.csv"),
        ("ab_member_a", "member shown in Figure S6a", dict(pair_mean_mainstem_nse=sel["VIC5"]["pair_mean_ms_nse"]),
         f"VIC5/{best}/SSP2-4.5", "members_fidelity.csv ms_nse"),
        ("ab_member_b", "member shown in Figure S6b", dict(pair_mean_mainstem_nse=sel["PRMS"]["pair_mean_ms_nse"]),
         f"PRMS/{best}/SSP2-4.5", "members_fidelity.csv ms_nse"),
        ("ab_mainstem_nse_range", "daily NSE at the mainstem reach, range across all members", ms,
         f"{ms['min']:.2f}--{ms['max']:.2f}", "members_fidelity.csv ms_nse"),
        ("ab_effort_correlation", "correlation of the daily network effort of the two loops over 2020-2099 (a, b)",
         r_full, f"{r_full['VIC5']:.2f} and {r_full['PRMS']:.2f}", sname + " panels_ab_timeseries"),
        ("d_effort_top_decile", "top decile of reaches by effort shared by the two loops (%), across members",
         al["alloc_top10"], spread(al["alloc_top10"], ".0f", pct=True), "members_fidelity.csv alloc_top10"),
        ("d_effort_spearman", "Spearman correlation of reach effort between the loops, across members",
         al["alloc_rho"], f"$\\rho = {al['alloc_rho']['median']:.3f}$ "
                          f"({al['alloc_rho']['min']:.3f}--{al['alloc_rho']['max']:.3f})", "members_fidelity.csv alloc_rho"),
        ("e_map_both", "reaches in the top decile of the 14-member ensemble-median effort in both loops",
         dict(n=e["both"], k=e["n_top"]), f"{e['both']:,}", "inputs_map.npz map_counts"),
        ("e_map_routing_only", "in the routing-model loop only", dict(n=e["routing_only"]), f"{e['routing_only']:,}",
         "inputs_map.npz map_counts"),
        ("e_map_surrogate_only", "in the surrogate loop only", dict(n=e["surrogate_only"]), f"{e['surrogate_only']:,}",
         "inputs_map.npz map_counts"),
        ("f_residual_spearman", "Spearman correlation of residual exceedance between the loops, across members",
         al["resid_rho"], spread(al["resid_rho"], ".2f"), "members_fidelity.csv resid_rho"),
        ("f_residual_top_decile", "top decile of reaches by residual exceedance shared by the loops (%), across members",
         f["top_decile_overlap_pct"], spread(f["top_decile_overlap_pct"], ".0f", pct=True), sname + " panel_f_residual"),
        ("sec31_nse_orders_5_7", "median daily NSE of the controlled discharge for stream orders 5, 6 and 7",
         o57, f"{min(o57.values()):.2f}--{max(o57.values()):.2f}", "figure_data.npz nse_q and order"),
        ("sec31_effort_top_decile", "top-decile effort overlap (%), median across members",
         dict(median=al["alloc_top10"]["median"]), f"{al['alloc_top10']['median']:.0f}\\%",
         "members_fidelity.csv alloc_top10"),
    ]
    return [dict(key=k, quantity=q, values="; ".join(f"{n}={v!r}" for n, v in vals.items()), text=t,
                 published=PUBLISHED_CLOSED_LOOP[k], source=s) for k, q, vals, t, s in items]


def main():
    p = argparse.ArgumentParser(description="Supporting Information tables and closed-loop numbers.")
    p.add_argument("--rank", type=Path, default=data("rank_sensitivity"))
    p.add_argument("--basis", type=Path, default=data("basis_period"))
    p.add_argument("--threshold", type=Path, default=data("flood_threshold"))
    p.add_argument("--closed-loop", type=Path, default=data("closed_loop"))
    p.add_argument("--figs6-statistics", type=Path, default=None,
                   help="statistics file of figures/figS06_closed_loop/make_figure.py "
                        "(default: figS6_control_fidelity_statistics.json in --closed-loop)")
    p.add_argument("--no-check", action="store_true", help="do not compare with the published text")
    a = p.parse_args()
    target = out("si_tables")

    top, top_tex, middle, middle_tex = table_s1_rank(a.rank)
    bottom, bottom_tex = table_s1_basis(a.basis)
    s1 = (tabular("lccccc", [r" & Retained & Recursive & Recursive & Largest & Admissible \\",
                             r"$r$ & energy (\%) & NSE & KGE & $\rho(\mathbf{A}^{(w)})$ & members \\"], top_tex)
          + ["", r"\vspace{1em}"]
          + tabular("lcc", [r"Ranks & Spearman $\rho$ & Top 10\% (\%) \\"], middle_tex)
          + ["", r"\vspace{1em}"]
          + tabular("lcccc", [r"Basis period & NSE & KGE & Top 10\% (\%) & Spearman $\rho$ \\"], bottom_tex))
    (target / "table_s1.tex").write_text("\n".join(s1) + "\n")
    write_csv(target / "table_s1_top.csv", top)
    write_csv(target / "table_s1_middle.csv", middle)
    write_csv(target / "table_s1_bottom.csv", bottom)

    s2_rows, s2_tex, capped = table_s2(a.threshold)
    s2 = tabular("lcc", [r" & 5~Gm$^3$\,yr$^{-1}$ & 10~Gm$^3$\,yr$^{-1}$ \\"], s2_tex)
    (target / "table_s2.tex").write_text("\n".join(s2) + "\n")
    write_csv(target / "table_s2.csv", s2_rows)

    stats_file = a.figs6_statistics or a.closed_loop / "figS6_control_fidelity_statistics.json"
    cl = closed_loop_numbers(a.closed_loop, stats_file)
    write_csv(target / "closed_loop_numbers.csv", cl)
    (target / "closed_loop_numbers.tex").write_text(
        "% Numbers quoted in the Figure S6 caption and Section 3.1 (written by analyses/si_tables/build_si_tables.py)\n"
        + "".join(f"% {r['quantity']}\n\\newcommand{{\\{MACROS[r['key']]}}}{{{r['text']}}}\n" for r in cl))

    print("Table S1\n" + "\n".join(s1) + "\n\nTable S2\n" + "\n".join(s2) + "\n")
    print("Table S2 caption, members at their maximum effort:", {f"{t:g} Gm3/yr": n for t, n in capped.items()})
    print("Figure S6 caption and Section 3.1:")
    for r in cl:
        print(f"  {r['key']:24s} {r['text']}")
    print("wrote", target)
    if a.no_check:
        return
    bad = [(got, want) for got, want in zip(top_tex + middle_tex + bottom_tex + s2_tex,
                                            PUBLISHED_S1_TOP + PUBLISHED_S1_MIDDLE + PUBLISHED_S1_BOTTOM + PUBLISHED_S2)
           if got != want]
    bad += [(f"{capped[t]} members at maximum effort at {t:g} Gm3/yr", f"{n}")
            for t, n in PUBLISHED_S2_CAPPED.items() if capped[t] != n]
    bad += [(r["text"], r["published"]) for r in cl if r["text"] != r["published"]]
    if bad:
        for got, want in bad:
            print("MISMATCH  produced:", got, "  published:", want)
        sys.exit(1)
    print(f"All values match the published text: Table S1 ({len(top_tex + middle_tex + bottom_tex)} rows), "
          f"Table S2 ({len(s2_tex)} rows and the caption), Figure S6 caption and Section 3.1 ({len(cl)} numbers).")


if __name__ == "__main__":
    main()
