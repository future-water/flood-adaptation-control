"""
Aggregate per-member HUC comparison output into a reach-wise spatial CSV.

Reads the long-format CSV written by `si_huc_comparison.py`
(`huc_comparison_{hydro}_{gcm}_ssp{scn}.csv`) and writes
`huc_spatial_compact.csv` — one row per reach with peak-reduction values at
each control resolution (reach / HUC12 / HUC10 / HUC8) and HUC outlet flags.
This is the file consumed by `06_figures_supplementary.ipynb` (FIG S5).

Pipeline (mirrors the original analysis notebook, Cell 8):
  1. For each huc_level, find the R_weight whose total effort lies closest to
     a target budget (default 10 Gm3/yr).
  2. Subset the long CSV to those (huc_level, R_weight) rows.
  3. Pivot: one row per reach, columns peak_red_reach, peak_red_HUC12, ...
  4. Add HUC8/10/12 labels + outlet flags from the COMID->HUC mapping.

Usage:
    python si_huc_aggregate.py \\
        --input outputs_huc_comparison/huc_comparison_VIC5_MPI-ESM1-2-HR_ssp245.csv \\
        --mapping comid_huc_mapping_correct.csv \\
        --out outputs_huc_comparison/huc_spatial_compact.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HUC_LEVELS = ["reach", "HUC12", "HUC10", "HUC8"]


def find_best_rw(results_path: Path, target_budget_gm3: float) -> dict[str, float]:
    """Per huc_level, pick R_weight whose summed effort is closest to target."""
    df_rw = pd.read_csv(results_path, usecols=["huc_level", "R_weight", "eff_total_L1"])
    best = {}
    for level in HUC_LEVELS:
        sub = df_rw[df_rw["huc_level"] == level]
        if len(sub) == 0:
            continue
        effort = sub.groupby("R_weight")["eff_total_L1"].sum() / 1e9
        rw = effort.index[(effort - target_budget_gm3).abs().argmin()]
        best[level] = float(rw)
        print(f"  {level:>6s}: R_w={rw:.2e}, effort={effort[rw]:.1f} Gm3/yr")
    return best


def load_subset(results_path: Path, best_rw: dict[str, float], chunksize: int = 500_000) -> pd.DataFrame:
    """Stream the long CSV, keep only rows matching the best (level, R_w) per level."""
    keep_cols = [
        "huc_level", "R_weight", "reach_id",
        "vol_hi_baseline", "vol_hi_controlled", "vol_hi_red_rel",
        "peak_baseline", "peak_controlled", "peak_red_rel",
        "eff_total_L1", "residual_ratio",
    ]
    chunks = []
    for chunk in pd.read_csv(results_path, usecols=keep_cols, chunksize=chunksize):
        for level, rw in best_rw.items():
            mask = (chunk["huc_level"] == level) & np.isclose(chunk["R_weight"], rw, rtol=1e-6)
            if mask.any():
                chunks.append(chunk[mask].copy())
    df = pd.concat(chunks, ignore_index=True)
    print(f"  loaded {len(df)} rows")
    return df


def pivot_to_reach(df: pd.DataFrame, best_rw: dict[str, float]) -> pd.DataFrame:
    """One row per reach, columns peak_red_reach, peak_red_HUC12, ..."""
    result = None
    for level in HUC_LEVELS:
        if level not in best_rw:
            continue
        sub = df[df["huc_level"] == level][[
            "reach_id", "vol_hi_baseline", "vol_hi_controlled",
            "peak_red_rel", "eff_total_L1", "residual_ratio",
        ]].copy()
        if level == "reach":
            sub.columns = [
                "reach_id", "vol_bl", "vol_cl_reach",
                "peak_red_reach", "effort_reach", "resid_ratio_reach",
            ]
            result = sub
        else:
            sub.columns = [
                "reach_id", "_drop",
                f"vol_cl_{level}", f"peak_red_{level}",
                f"effort_{level}", f"resid_ratio_{level}",
            ]
            sub = sub.drop(columns="_drop")
            result = result.merge(sub, on="reach_id", how="left")
    return result


def add_derived(result: pd.DataFrame) -> pd.DataFrame:
    """Residual increase + peak-reduction loss when stepping up to coarser HUC."""
    for level in ["HUC12", "HUC10", "HUC8"]:
        if f"vol_cl_{level}" not in result.columns:
            continue
        result[f"resid_increase_{level}"] = (
            (result[f"vol_cl_{level}"] - result["vol_cl_reach"])
            / (result["vol_cl_reach"] + 1e-12)
        )
        result[f"peak_red_loss_{level}"] = (
            result["peak_red_reach"] - result[f"peak_red_{level}"]
        )
    return result


def attach_huc(result: pd.DataFrame, mapping_csv: Path) -> pd.DataFrame:
    """Merge HUC labels + outlet flags."""
    mapping = pd.read_csv(mapping_csv)
    result = result.merge(
        mapping[["COMID", "HUC8", "HUC10", "HUC12", "TotDASqKM", "StreamOrde"]],
        left_on="reach_id", right_on="COMID", how="left",
    ).drop(columns=["COMID"], errors="ignore")
    for level in ["HUC8", "HUC10", "HUC12"]:
        outlets = mapping.loc[mapping.groupby(level)["TotDASqKM"].idxmax(), "COMID"].values
        result[f"is_outlet_{level}"] = result["reach_id"].isin(outlets)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Long-format CSV from si_huc_comparison.py")
    parser.add_argument("--mapping", default="comid_huc_mapping_correct.csv")
    parser.add_argument("--out", default="huc_spatial_compact.csv")
    parser.add_argument("--budget", type=float, default=10.0,
                        help="Target effort budget in Gm3/yr for picking R_w (default: 10)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Picking R_w per level (target = {args.budget} Gm3/yr) ...")
    best_rw = find_best_rw(Path(args.input), args.budget)

    print("[2/4] Loading rows for chosen (level, R_w) ...")
    df = load_subset(Path(args.input), best_rw)

    print("[3/4] Pivoting + derived metrics ...")
    result = pivot_to_reach(df, best_rw)
    result = add_derived(result)
    result = attach_huc(result, Path(args.mapping))

    print(f"[4/4] Writing {out_path} ({result.shape[0]} reaches, {result.shape[1]} cols)")
    result.to_csv(out_path, index=False, float_format="%.6g")
    print(f"  size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
