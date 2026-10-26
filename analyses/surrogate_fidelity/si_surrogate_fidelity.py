"""
Compute the surrogate-fidelity diagnostics consumed by FIG S1 panels (b)/(c).

For each (hydro, gcm, ssp) member this script
  1. loads streamflow + precipitation,
  2. fits POD + windowed DMDc on the historical era,
  3. simulates the open-loop surrogate,
  4. derives annual maximum series (AMS) for the basin-aggregate hydrograph
     and for each stream-order aggregate hydrograph (sum across reaches at
     each timestep, then yearly maxima),
  5. computes the empirical RL bias of the surrogate vs. the reference at
     RP = 2, 5, 10, 20, 50, 100 years.

Output: surrogate_fidelity_summary.csv (one row per member; ~56 rows total).

Panel (a) of FIG S1 is **not** produced here — it reads the per-reach
`rl10_bias_openloop` column directly from `outputs_flood_rl2/results_*.parquet`.
Only the basin- and order-aggregate hydrograph diagnostics needed by panels
(b) and (c) are exported, because they cannot be derived from the per-reach
parquet columns without re-running the surrogate (Jensen's inequality on the
AMS-RL operator means flow-weighted reach-average ≠ basin-aggregate RL bias).

Usage:
    python si_surrogate_fidelity.py
    python si_surrogate_fidelity.py --gcms MPI-ESM1-2-HR ACCESS-CM2
    python si_surrogate_fidelity.py --hydros VIC5 --scenarios 245 585

Run as a script (not a Jupyter cell) — each member is memory-conservative
(<5 GB RSS), but the loop is too long for an interactive kernel.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


_pipeline = importlib.import_module("04_dmd_lqr_analysis")
RANK = _pipeline.RANK
ANALYSIS_ERA = _pipeline.ANALYSIS_ERA
AVAILABLE_GCMS = _pipeline.AVAILABLE_GCMS
HYDRO_MODELS = _pipeline.HYDRO_MODELS
SCENARIOS = _pipeline.SCENARIOS

load_scenario_data = _pipeline.load_scenario_data
match_to_reference = _pipeline.match_to_reference
compute_reference_pack = _pipeline.compute_reference_pack
compute_pod = _pipeline.compute_pod
normalize_forcing = _pipeline.normalize_forcing
fit_windowed_dmdc = _pipeline.fit_windowed_dmdc
simulate_openloop = _pipeline.simulate_openloop


SHP_PATH = Path("shape/Flowlines_huc02_12_Brazos.shp")
DEFAULT_OUT = Path("surrogate_fidelity_summary.csv")
RPS = [2, 5, 10, 20, 50, 100]
NSE_THRESHOLD = 0.5  # Match figure_supp.ipynb


def empirical_rl(ams: np.ndarray, T: float) -> float:
    return float(np.percentile(ams, (1.0 - 1.0 / T) * 100))


def load_stream_order(reach_ids: list[int]) -> np.ndarray:
    shp = gpd.read_file(SHP_PATH)
    so_map = dict(zip(shp["COMID"].astype(int), shp["StreamOrde"].astype(int)))
    return np.array([so_map.get(int(r), 0) for r in reach_ids])


def member_diagnostics(hydro: str, gcm: str, ssp: int,
                       stream_order_cache: dict) -> dict | None:
    try:
        ref = compute_reference_pack(gcm, hydro, verbose=False)
    except Exception as e:
        print(f"  SKIP ref {hydro}/{gcm}: {e}")
        return None

    try:
        X, U, dates, reach_ids = load_scenario_data(ssp, gcm, hydro)
        idx_ref, idx_this, _ = match_to_reference(ref.reach_ids, reach_ids)
        X = X[idx_this]
        U = U[idx_this]

        pod = compute_pod(X, RANK)
        fp = normalize_forcing(U)
        dmd = fit_windowed_dmdc(pod.Z, fp.V)
        z0 = pod.Z[:, 0:1]
        V_sim = fp.V[:, :-1]
        Z_ol = simulate_openloop(dmd, z0, V_sim)

        T_use = min(X.shape[1], Z_ol.shape[1])
        X_ref = X[:, :T_use]
        X_surr = np.maximum(pod.U_r @ Z_ol[:, :T_use] + pod.mean_x, 0.0)

        years = pd.DatetimeIndex(dates[:T_use]).year.values
        mask_era = (years >= ANALYSIS_ERA[0]) & (years <= ANALYSIS_ERA[1])
        years_era = years[mask_era]
        unique_years = np.unique(years_era)

        X_ref_era = X_ref[:, mask_era]
        X_surr_era = X_surr[:, mask_era]

        # Per-reach NSE for filtering reaches into stream-order aggregates.
        ss_res = np.sum((X_ref_era - X_surr_era) ** 2, axis=1)
        ss_tot = np.sum(
            (X_ref_era - X_ref_era.mean(axis=1, keepdims=True)) ** 2, axis=1
        ) + 1e-12
        nse_arr = 1 - ss_res / ss_tot
        keep_mask = nse_arr >= NSE_THRESHOLD

        row: dict = {"hydro": hydro, "gcm": gcm, "ssp": int(ssp),
                     "n_reaches_total": int(X_ref_era.shape[0]),
                     "n_reaches_kept": int(keep_mask.sum())}

        # (b) Basin-aggregate hydrograph: sum across kept reaches at each
        # timestep, then yearly maxima → empirical T-year return level.
        q_ref_basin = X_ref_era[keep_mask].sum(axis=0)
        q_surr_basin = X_surr_era[keep_mask].sum(axis=0)
        ams_ref_b = np.array([q_ref_basin[years_era == y].max() for y in unique_years])
        ams_surr_b = np.array([q_surr_basin[years_era == y].max() for y in unique_years])
        for rp in RPS:
            rl_r = empirical_rl(ams_ref_b, rp)
            rl_s = empirical_rl(ams_surr_b, rp)
            row[f"basin_rl{rp}_bias"] = (rl_s - rl_r) / (rl_r + 1e-10)

        # (c) Stream-order aggregate hydrograph (RL10 only, all kept reaches
        # in that order summed at each timestep).
        ref_reach_ids_member = [int(ref.reach_ids[i]) for i in idx_ref]
        cache_key = (hydro, gcm)
        if cache_key not in stream_order_cache:
            stream_order_cache[cache_key] = load_stream_order(ref_reach_ids_member)
        so_arr = stream_order_cache[cache_key]

        for order in sorted(set(so_arr)):
            if order <= 0:
                continue
            mask_o = (so_arr == order) & keep_mask
            if mask_o.sum() == 0:
                continue
            q_r_o = X_ref_era[mask_o].sum(axis=0)
            q_s_o = X_surr_era[mask_o].sum(axis=0)
            ams_r_o = np.array([q_r_o[years_era == y].max() for y in unique_years])
            ams_s_o = np.array([q_s_o[years_era == y].max() for y in unique_years])
            rl_r_o = empirical_rl(ams_r_o, 10)
            rl_s_o = empirical_rl(ams_s_o, 10)
            row[f"order{int(order)}_rl10_bias"] = (rl_s_o - rl_r_o) / (rl_r_o + 1e-10)

        print(f"  {hydro}/{gcm}/SSP{ssp}: basin RL10 = {row['basin_rl10_bias']:+.3f}")
        return row

    except Exception as e:
        print(f"  SKIP {hydro}/{gcm}/SSP{ssp}: {e}")
        return None
    finally:
        for name in ("X", "U", "X_ref", "X_surr", "X_ref_era", "X_surr_era",
                    "pod", "dmd", "Z_ol", "fp"):
            if name in locals():
                del locals()[name]
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcms", nargs="+", default=None,
                        help="Subset of GCMs (default: all available)")
    parser.add_argument("--hydros", nargs="+", default=None,
                        help="Subset of hydro models (default: all)")
    parser.add_argument("--scenarios", type=int, nargs="+", default=None,
                        help="Subset of SSP scenarios (default: all)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    gcms = args.gcms or AVAILABLE_GCMS
    hydros = args.hydros or HYDRO_MODELS
    scenarios = args.scenarios or SCENARIOS

    print(f"Members: {len(hydros)} hydro x {len(gcms)} GCM x {len(scenarios)} SSP "
          f"= {len(hydros)*len(gcms)*len(scenarios)} max")

    rows = []
    so_cache: dict = {}
    for hydro in hydros:
        for gcm in gcms:
            for ssp in scenarios:
                row = member_diagnostics(hydro, gcm, int(ssp), so_cache)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise SystemExit("No members produced output. Check input data paths.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(df)} members, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
