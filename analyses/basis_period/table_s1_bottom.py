"""Table S1, bottom block: first-decade basis period (2020-2029) against the main analysis, both scored over 2030-2099.

Row 1 comes from summarize_basis.summarize(): median across members of the reach-median NSE and KGE of the
restricted-basis reconstruction, top-decile overlap of the ensemble-median allocation with the main analysis (median
and range over members in parentheses) and the Spearman correlation. Row 2 is the main analysis: median across members
of the reach-median open-loop NSE and KGE over 2030-2099 (results/<member>.npz, nse/kge_production_test). Formatting
is that of the SI table.

Writes table_s1_bottom.tex (tabular rows) and table_s1_bottom.csv (unrounded values) to the output folder.
Usage: python table_s1_bottom.py [--data DIR]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import data, out  # noqa: E402
from summarize_basis import summarize  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=data("basis_period"),
                   help="folder holding results/ and results_basis/ (default: the extracted archive)")
    a = p.parse_args()
    prod = [np.load(f) for f in sorted((a.data / "results").glob("*.npz"))]
    assert len(prod) == 14
    pn = float(np.median([np.nanmedian(d["nse_production_test"]) for d in prod]))
    pk = float(np.median([np.nanmedian(d["kge_production_test"]) for d in prod]))
    rows, tex = [], []
    for yrs in (10,):
        b = summarize(a.data, yrs)
        assert b["n_members"] == 14 and not b["excluded"]
        m = b["members"]["top_decile_overlap"]
        nse, kge = b["skill_2030_2099"]["nse_median"]["median"], b["skill_2030_2099"]["kge_median"]["median"]
        em = b["ensemble_median"]
        tex.append(f"2020--{2019 + yrs} & {nse:.2f} & {kge:.2f} & {em['top_decile_overlap']:.1f} "
                   f"({m['median']:.0f}; {m['min']:.0f}--{m['max']:.0f}) & {em['spearman']:.3f} \\\\")
        rows.append(dict(basis_period=f"2020-{2019 + yrs}", nse=nse, kge=kge, top10_ensemble_median=em["top_decile_overlap"],
                         top10_members_median=m["median"], top10_members_min=m["min"], top10_members_max=m["max"],
                         spearman_ensemble_median=em["spearman"], n_members=b["n_members"]))
    tex.append(f"2020--2099 (main analysis; scored 2030--2099) & {pn:.2f} & {pk:.2f} & -- & -- \\\\")
    rows.append(dict(basis_period="2020-2099 (main analysis; scored 2030-2099)", nse=pn, kge=pk, n_members=len(prod)))

    folder = out("basis_period")
    (folder / "table_s1_bottom.tex").write_text("\n".join(tex) + "\n")
    with open(folder / "table_s1_bottom.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(tex))
    print("wrote", folder / "table_s1_bottom.tex", "and", folder / "table_s1_bottom.csv")


if __name__ == "__main__":
    main()
