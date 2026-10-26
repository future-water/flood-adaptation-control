"""Table S1, top and middle blocks: sensitivity to the reduced dimension r for the 14 SSP2-4.5 members.

Light step (seconds). Reads the outputs_rank_sensitivity archive (data('rank_sensitivity'), or --data, for example
the output folder of analyze_ensemble.py) and writes to out('rank_sensitivity'):

  table_s1_rank_top.csv     r = 10, 20, 30, 50, 75, 100: retained energy (median over members of the cumulative
                            fraction of the 300-mode spectrum at r), reach-median NSE and KGE of the recursive
                            reconstruction over 2020-2099 (median over members), largest spectral radius of the window
                            operators (median over members) and the number of admissible members
  table_s1_rank_middle.csv  ensemble-median raw effort compared between r = 20 vs 30, 30 vs 50 and 20 vs 50: Spearman
                            correlation over all reaches and the share of the lower-rank top decile kept at the
                            higher rank (scope full14_original_fallback, 14 members)
  table_s1_rank.tex         the two tabular blocks as printed in the Supporting Information

The printed values are compared with the published Table S1; --no-check skips the comparison (for rerun data).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import data, out  # noqa: E402

RANKS = [10, 20, 30, 50, 75, 100]
PAIRS = [(20, 30), (30, 50), (20, 50)]
SCOPE = "full14_original_fallback"
N_MEMBERS = 14
PUBLISHED_TOP = [
    "10 & 99.159 & 0.36 & 0.42 & 0.990 & 14/14 \\\\",
    "20 & 99.801 & 0.59 & 0.63 & 0.983 & 14/14 \\\\",
    "30 & 99.926 & 0.69 & 0.72 & 0.992 & 14/14 \\\\",
    "50 & 99.981 & 0.77 & 0.72 & 0.994 & 13/14 \\\\",
    "75 & 99.993 & 0.75 & 0.59 & 0.995 & 7/14 \\\\",
    "100 & 99.997 & 0.65 & 0.43 & 0.995 & 3/14 \\\\",
]
PUBLISHED_MIDDLE = [
    "20 vs 30 & 0.994 & 91.4 \\\\",
    "30 vs 50 & 0.984 & 87.0 \\\\",
    "20 vs 50 & 0.981 & 88.0 \\\\",
]


def read(path):
    # round_trip parsing gives the same floats as the text in the file
    return pd.read_csv(path, float_precision="round_trip")


def main():
    parser = argparse.ArgumentParser(description="Table S1 top and middle blocks (reduced dimension r).")
    parser.add_argument("--data", type=Path, default=data("rank_sensitivity"),
                        help="folder with the rank_sensitivity files (default: data('rank_sensitivity'))")
    parser.add_argument("--no-check", action="store_true", help="do not compare with the published values")
    args = parser.parse_args()
    folder = args.data
    spec = read(folder / "singular_spectra_all14.csv")
    acc = read(folder / "same_period_accuracy_all14.csv")
    status = read(folder / "member_rank_status.csv")
    comp = read(folder / "ensemble_rank_comparisons.csv")

    top_rows, top_tex = [], []
    for r in RANKS:
        e = spec.loc[spec["mode"] == r, "cumulative_fraction"]
        a = acc[acc["rank"] == r]
        s = status[status["rank"] == r]
        assert len(e) == len(a) == len(s) == N_MEMBERS, r
        row = dict(rank=r, retained_energy_percent=100 * e.median(),
                   recursive_nse=a["recursive_nse_median"].median(), recursive_kge=a["recursive_kge_median"].median(),
                   largest_spectral_radius=s["max_radius"].median(),
                   admissible_members=int(s["strict_valid"].astype(bool).sum()), members=N_MEMBERS)
        top_rows.append(row)
        top_tex.append(f"{r} & {row['retained_energy_percent']:.3f} & {row['recursive_nse']:.2f} & "
                       f"{row['recursive_kge']:.2f} & {row['largest_spectral_radius']:.3f} & "
                       f"{row['admissible_members']}/{N_MEMBERS} \\\\")

    mid_rows, mid_tex = [], []
    for a, b in PAIRS:
        c = comp[(comp["scope"] == SCOPE) & (comp["metric"] == "raw") & (comp["rank_a"] == a) & (comp["rank_b"] == b)]
        assert len(c) == 1 and int(c["n_members"].iloc[0]) == N_MEMBERS, (a, b)
        c = c.iloc[0]
        row = dict(rank_pair=f"{a} vs {b}", spearman=c["spearman"], top10_retained_percent=100 * c["top10_retention_a"],
                   top10_n=int(c["top10_n_a"]), top10_shared=int(c["top10_common"]), n_reaches=int(c["n_reaches"]))
        mid_rows.append(row)
        mid_tex.append(f"{a} vs {b} & {row['spearman']:.3f} & {row['top10_retained_percent']:.1f} \\\\")

    target = out("rank_sensitivity")
    pd.DataFrame(top_rows).to_csv(target / "table_s1_rank_top.csv", index=False)
    pd.DataFrame(mid_rows).to_csv(target / "table_s1_rank_middle.csv", index=False)
    tex = "\n".join([
        r"\begin{tabular}{lccccc}", r"\hline",
        r" & Retained & Recursive & Recursive & Largest & Admissible \\",
        r"$r$ & energy (\%) & NSE & KGE & $\rho(\mathbf{A}^{(w)})$ & members \\",
        r"\hline", *top_tex, r"\hline", r"\end{tabular}",
        "",
        r"\vspace{1em}",
        r"\begin{tabular}{lcc}", r"\hline",
        r"Ranks & Spearman $\rho$ & Top 10\% (\%) \\",
        r"\hline", *mid_tex, r"\hline", r"\end{tabular}",
    ]) + "\n"
    (target / "table_s1_rank.tex").write_text(tex)
    print(tex)
    print("wrote", target)
    if not args.no_check:
        bad = [(got, want) for got, want in zip(top_tex + mid_tex, PUBLISHED_TOP + PUBLISHED_MIDDLE) if got != want]
        if bad:
            for got, want in bad:
                print("MISMATCH  produced:", got, "  published:", want)
            sys.exit(1)
        print("Table S1 top and middle blocks match the published values (9 rows).")


if __name__ == "__main__":
    main()
