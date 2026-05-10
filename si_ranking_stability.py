"""
Per-member ranking-stability diagnostics for FIG S4 panel (c).

For each (gcm, hydro, scenario) member this script
  1. reads only the columns it needs from results_{gcm}.parquet,
  2. picks two R_weight values (R[3] and R[len(R)//2] — same as the original
     analysis cell) to represent low- and mid-budget LQR runs,
  3. computes the Spearman rank correlation and the top-10% overlap of
     per-reach effort between those two R_weights.

Output: ranking_stability.csv (one row per member; ~50 rows total).

The original notebook computed these stats by `df_all.groupby(...)` over the
full ~5 GB long-format parquet, which forced the kernel above the OOM limit.
This script processes one parquet at a time and never holds more than one
member's data in memory, so it runs comfortably in <2 GB.

Usage:
    python si_ranking_stability.py
    python si_ranking_stability.py --top-frac 0.10 --out ranking_stability.csv
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RESULT_DIR_DEFAULT = Path("outputs_flood_rl2")
NEEDED_COLS = ["gcm", "hydro", "scenario", "R_weight", "reach_id", "eff_total_L1"]


def member_stats(g: pd.DataFrame, top_frac: float) -> dict | None:
    R = sorted(g["R_weight"].unique())
    if len(R) < 6:
        return None
    Rlo, Rmi = R[3], R[len(R) // 2]
    gl = g[g["R_weight"] == Rlo]
    gm = g[g["R_weight"] == Rmi]
    common = set(gl["reach_id"]) & set(gm["reach_id"])
    if len(common) < 50:
        return None

    gl_c = gl[gl["reach_id"].isin(common)].sort_values("reach_id")
    gm_c = gm[gm["reach_id"].isin(common)].sort_values("reach_id")
    rho, _ = spearmanr(gl_c["eff_total_L1"].values, gm_c["eff_total_L1"].values)

    n = min(len(gl_c), len(gm_c))
    k = max(1, int(top_frac * n))
    topA = set(gl_c.nlargest(k, "eff_total_L1")["reach_id"])
    topB = set(gm_c.nlargest(k, "eff_total_L1")["reach_id"])
    overlap = len(topA & topB) / k

    return {"R_low": float(Rlo), "R_mid": float(Rmi),
            "spearman": float(rho), "overlap": float(overlap)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULT_DIR_DEFAULT))
    parser.add_argument("--top-frac", type=float, default=0.10,
                        help="Top-N fraction for the overlap metric (default 0.10)")
    parser.add_argument("--out", default="ranking_stability.csv")
    args = parser.parse_args()

    result_dir = Path(args.results_dir)
    rows = []
    files = sorted(result_dir.glob("results_*.parquet"))
    if not files:
        raise SystemExit(f"No results parquets found in {result_dir}/")

    print(f"Scanning {len(files)} parquet files in {result_dir} ...")
    for fp in files:
        df = pd.read_parquet(fp, columns=NEEDED_COLS)
        if "hydro" not in df.columns:
            df["hydro"] = "VIC5"
        df["scenario"] = df["scenario"].astype(int)
        df["reach_id"] = df["reach_id"].astype(int)

        for (gcm, hydro, scn), grp in df.groupby(["gcm", "hydro", "scenario"]):
            stats = member_stats(grp, args.top_frac)
            if stats is None:
                continue
            stats.update({"gcm": gcm, "hydro": hydro, "scenario": int(scn),
                          "n_reaches": grp["reach_id"].nunique()})
            rows.append(stats)
            print(f"  {hydro}/{gcm}/SSP{scn}: rho={stats['spearman']:+.3f}, "
                  f"overlap={stats['overlap']:.2f}")

        del df
        gc.collect()

    if not rows:
        raise SystemExit("No members produced stats — check parquet contents.")

    out = pd.DataFrame(rows)[["hydro", "gcm", "scenario",
                               "R_low", "R_mid", "spearman", "overlap", "n_reaches"]]
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out)} members)")


if __name__ == "__main__":
    main()
