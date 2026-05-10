#!/usr/bin/env python3
"""
Compute physical-space attenuation fraction: |δ⁻_i| / |δ_i|
where δ_i(t) = max(x_cl_i(t), 0) - max(x_ol_i(t), 0)

This replaces the raw-u_i-based atten_fraction with a metric that
reflects actual flow change after dynamics propagation + clipping.

Usage:
    python compute_delta_atten_fraction.py --rl 2
    python compute_delta_atten_fraction.py --rl 2 --gcm ACCESS-CM2 --hydro VIC5
"""

from __future__ import annotations
import os, gc, argparse, warnings, time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Import pipeline ──────────────────────────────────────────
import importlib as _il
_pipeline = _il.import_module('04_dmd_lqr_analysis')
SCENARIOS = _pipeline.SCENARIOS
AVAILABLE_GCMS = _pipeline.AVAILABLE_GCMS
HYDRO_MODELS = _pipeline.HYDRO_MODELS
RANK = _pipeline.RANK
DTYPE = _pipeline.DTYPE
DT = _pipeline.DT
FLOOD_RL_PERIOD = _pipeline.FLOOD_RL_PERIOD
Q_WEIGHT = _pipeline.Q_WEIGHT
R_VALUES = _pipeline.R_VALUES
CHUNK = _pipeline.CHUNK
ANALYSIS_ERA = _pipeline.ANALYSIS_ERA
load_scenario_data = _pipeline.load_scenario_data
match_to_reference = _pipeline.match_to_reference
compute_reference_pack = _pipeline.compute_reference_pack
compute_pod = _pipeline.compute_pod
normalize_forcing = _pipeline.normalize_forcing
fit_windowed_dmdc = _pipeline.fit_windowed_dmdc
simulate_openloop = _pipeline.simulate_openloop
simulate_cl_flood_only = _pipeline.simulate_cl_flood_only
lqr_gain = _pipeline.lqr_gain

# ── Config ───────────────────────────────────────────────────
TARGET_BUDGET = 10.0          # Gm³/yr — match main analysis
OUT_DIR = None                # set in main()


def find_bracket_Rw(dmd, z0, V_sim, thr_flood, pod, Q_mat,
                    target_budget: float = TARGET_BUDGET):
    """
    Find two R_weight values that bracket the target budget,
    and return the interpolation weight.

    Returns
    -------
    Rlo, Rhi : float
        R_weight values bracketing target budget (Rlo has HIGHER effort,
        Rhi has LOWER effort, since effort is monotonically decreasing in R).
    w : float
        Interpolation weight in [0, 1]: final = (1-w)*lo + w*hi,
        where "lo" = high-effort side and "hi" = low-effort side.
    eff_lo, eff_hi : float
        Effort (Gm³/yr) at Rlo and Rhi.
    """
    efforts = {}
    for Rw in R_VALUES:
        try:
            K_list = [lqr_gain(A, Q_mat, float(Rw)) for A in dmd.A_list]
            Z_cl, U_cl = simulate_cl_flood_only(
                dmd, z0, V_sim, K_list, thr_flood, pod.U_r, pod.mean_x)

            n = pod.U_r.shape[0]
            T_use = U_cl.shape[1]
            total_eff = 0.0
            for i0 in range(0, n, CHUNK):
                i1 = min(i0 + CHUNK, n)
                Uphys = pod.U_r[i0:i1] @ U_cl[:, :T_use]
                total_eff += np.abs(Uphys).sum()
            n_years = max(1, T_use / 365.25)
            eff_gm3 = total_eff * DT / n_years / 1e9
            efforts[Rw] = eff_gm3
            del Z_cl, U_cl, K_list
            gc.collect()
        except Exception:
            continue

    if not efforts:
        raise RuntimeError("No R_weight succeeded")

    # Sort by effort ascending (= R_weight descending)
    rs = np.array(sorted(efforts.keys()))
    es = np.array([efforts[r] for r in rs])
    si = np.argsort(es)
    es_s, rs_s = es[si], rs[si]

    # Edge cases
    if target_budget <= es_s[0]:
        return rs_s[0], rs_s[0], 0.0, es_s[0], es_s[0]
    if target_budget >= es_s[-1]:
        return rs_s[-1], rs_s[-1], 0.0, es_s[-1], es_s[-1]

    # Bracket: find ia such that es_s[ia-1] <= target <= es_s[ia]
    ia = int(np.searchsorted(es_s, target_budget))
    ib = ia - 1
    if es_s[ia] == es_s[ib]:
        w = 0.5
    else:
        w = (target_budget - es_s[ib]) / (es_s[ia] - es_s[ib])

    # Rlo = high-effort side (= ib), Rhi = low-effort side (= ia)
    return rs_s[ib], rs_s[ia], float(w), es_s[ib], es_s[ia]


def simulate_for_Rw(dmd, z0, V_sim, thr_flood, pod, Q_mat, Rw):
    """Run closed-loop simulation for a given R_weight."""
    K_list = [lqr_gain(A, Q_mat, float(Rw)) for A in dmd.A_list]
    Z_cl, U_cl = simulate_cl_flood_only(
        dmd, z0, V_sim, K_list, thr_flood, pod.U_r, pod.mean_x)
    return Z_cl, U_cl


def compute_delta_metrics(
    pod, Z_ol, Z_cl, U_cl, thr_flood, dates,
) -> Dict[str, np.ndarray]:
    """
    Compute per-reach δ-based metrics:
      δ_i(t) = max(x_cl_i(t), 0) - max(x_ol_i(t), 0)

    Returns dict with per-reach arrays.
    """
    n = pod.U_r.shape[0]
    T_ctrl = min(Z_ol.shape[1], Z_cl.shape[1]) - 1

    # Analysis era mask
    dates_use = pd.DatetimeIndex(dates[:T_ctrl])
    mask_era = np.asarray(
        (dates_use.year >= ANALYSIS_ERA[0]) & (dates_use.year <= ANALYSIS_ERA[1]),
        dtype=bool,
    )
    n_years = max(len(np.unique(dates_use[mask_era].year)), 1)

    Z_ol_era = Z_ol[:, :T_ctrl][:, mask_era]
    Z_cl_era = Z_cl[:, :T_ctrl][:, mask_era]
    U_cl_era = U_cl[:, :T_ctrl][:, mask_era]

    thr = thr_flood.reshape(-1, 1)

    # Per-reach outputs
    delta_atten_L1   = np.zeros(n, dtype=DTYPE)  # Σ|δ⁻_i|
    delta_augment_L1 = np.zeros(n, dtype=DTYPE)  # Σ|δ⁺_i|  (δ>0 means flow increased)
    delta_total_L1   = np.zeros(n, dtype=DTYPE)  # Σ|δ_i|
    delta_atten_frac = np.zeros(n, dtype=DTYPE)  # |δ⁻|/|δ|

    # Also compute during exceedance only
    delta_atten_flood_L1   = np.zeros(n, dtype=DTYPE)
    delta_total_flood_L1   = np.zeros(n, dtype=DTYPE)
    delta_atten_flood_frac = np.zeros(n, dtype=DTYPE)

    # Raw u-based (for comparison)
    u_atten_L1  = np.zeros(n, dtype=DTYPE)
    u_total_L1  = np.zeros(n, dtype=DTYPE)
    u_atten_frac = np.zeros(n, dtype=DTYPE)

    # Ratio to flow during exceedance: |δ_i| / x_ol_i
    delta_ratio_flow_median = np.zeros(n, dtype=DTYPE)
    delta_ratio_flow_p95    = np.zeros(n, dtype=DTYPE)

    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        Uc = pod.U_r[i0:i1]
        mc = pod.mean_x[i0:i1]
        hi = thr[i0:i1]

        # Reconstruct physical-space discharge (clipped)
        X_ol = np.maximum(Uc @ Z_ol_era + mc, 0.0)
        X_cl = np.maximum(Uc @ Z_cl_era + mc, 0.0)

        # δ_i(t) = x_cl - x_ol (negative = attenuation)
        delta = X_cl - X_ol

        # Exceedance mask (based on open-loop)
        above_ol = X_ol > hi

        # δ-based metrics: all timesteps
        d_neg = np.minimum(delta, 0.0)
        d_pos = np.maximum(delta, 0.0)
        delta_atten_L1[i0:i1]   = np.abs(d_neg).sum(axis=1) * DT / n_years
        delta_augment_L1[i0:i1] = d_pos.sum(axis=1) * DT / n_years
        delta_total_L1[i0:i1]   = np.abs(delta).sum(axis=1) * DT / n_years
        delta_atten_frac[i0:i1] = (np.abs(d_neg).sum(axis=1) /
                                   (np.abs(delta).sum(axis=1) + 1e-12))

        # δ-based metrics: during exceedance only
        d_neg_flood = np.abs(d_neg) * above_ol
        d_abs_flood = np.abs(delta) * above_ol
        delta_atten_flood_L1[i0:i1]   = d_neg_flood.sum(axis=1) * DT / n_years
        delta_total_flood_L1[i0:i1]   = d_abs_flood.sum(axis=1) * DT / n_years
        delta_atten_flood_frac[i0:i1] = (d_neg_flood.sum(axis=1) /
                                         (d_abs_flood.sum(axis=1) + 1e-12))

        # Raw u-based (for comparison)
        Uphys = Uc @ U_cl_era
        u_neg = np.minimum(Uphys, 0.0)
        u_atten_L1[i0:i1]  = np.abs(u_neg).sum(axis=1) * DT / n_years
        u_total_L1[i0:i1]  = np.abs(Uphys).sum(axis=1) * DT / n_years
        u_atten_frac[i0:i1] = (np.abs(u_neg).sum(axis=1) /
                                (np.abs(Uphys).sum(axis=1) + 1e-12))

        # |δ_i| / x_ol_i during exceedance
        ratio_all = []
        for j in range(i1 - i0):
            exc_mask = above_ol[j]
            if exc_mask.sum() > 0:
                ratios = np.abs(delta[j, exc_mask]) / (X_ol[j, exc_mask] + 1e-12)
                ratio_all.append((np.median(ratios), np.percentile(ratios, 95)))
            else:
                ratio_all.append((0.0, 0.0))
        ra = np.array(ratio_all)
        delta_ratio_flow_median[i0:i1] = ra[:, 0]
        delta_ratio_flow_p95[i0:i1]    = ra[:, 1]

        del X_ol, X_cl, delta, d_neg, d_pos, Uphys, u_neg
        del d_neg_flood, d_abs_flood, above_ol
        gc.collect()

    return {
        # δ-based (all timesteps)
        "delta_atten_L1": delta_atten_L1,
        "delta_augment_L1": delta_augment_L1,
        "delta_total_L1": delta_total_L1,
        "delta_atten_frac": delta_atten_frac,
        # δ-based (exceedance only)
        "delta_atten_flood_L1": delta_atten_flood_L1,
        "delta_total_flood_L1": delta_total_flood_L1,
        "delta_atten_flood_frac": delta_atten_flood_frac,
        # raw-u-based (for comparison)
        "u_atten_L1": u_atten_L1,
        "u_total_L1": u_total_L1,
        "u_atten_frac": u_atten_frac,
        # δ-to-flow ratio during exceedance
        "delta_ratio_flow_median": delta_ratio_flow_median,
        "delta_ratio_flow_p95": delta_ratio_flow_p95,
    }


def process_member(gcm: str, hydro: str):
    """Process one (GCM, hydro) member across all SSPs."""
    print(f"\n{'='*60}")
    print(f"  {hydro} / {gcm}")
    print(f"{'='*60}")

    ref = compute_reference_pack(gcm, hydro, verbose=True)
    Q_mat = (Q_WEIGHT * np.eye(RANK)).astype(DTYPE)

    all_rows = []

    for scn in SCENARIOS:
        print(f"\n  SSP{scn} ...")
        try:
            X, U, dates, reach_ids = load_scenario_data(scn, gcm, hydro)
        except FileNotFoundError as e:
            print(f"    SKIP: {e}")
            continue

        idx_ref, idx_this, matched = match_to_reference(ref.reach_ids, reach_ids)
        X = X[idx_this]; U = U[idx_this]
        thr_flood = ref.thr_flood[idx_ref]

        print(f"    POD + DMDc ({X.shape[0]} reaches)...")
        pod = compute_pod(X, RANK)
        fp  = normalize_forcing(U)
        dmd = fit_windowed_dmdc(pod.Z, fp.V)

        z0    = pod.Z[:, 0:1]
        V_sim = fp.V[:, :-1]

        # Open-loop
        Z_ol = simulate_openloop(dmd, z0, V_sim)

        # Find R_weight bracket for target budget
        print(f"    Finding R_weight bracket for {TARGET_BUDGET} Gm³/yr budget...")
        Rlo, Rhi, w_interp, eff_lo, eff_hi = find_bracket_Rw(
            dmd, z0, V_sim, thr_flood, pod, Q_mat)
        actual_eff = (1 - w_interp) * eff_lo + w_interp * eff_hi
        print(f"    Rlo={Rlo:.2e} (eff={eff_lo:.1f}), Rhi={Rhi:.2e} (eff={eff_hi:.1f})")
        print(f"    Interpolation weight w={w_interp:.3f}, target effort={actual_eff:.1f} Gm³/yr")

        # Closed-loop at both bracket points
        if Rlo == Rhi:
            # Edge case: target outside bracket range
            Z_cl, U_cl = simulate_for_Rw(dmd, z0, V_sim, thr_flood, pod, Q_mat, Rlo)
            metrics_lo = compute_delta_metrics(pod, Z_ol, Z_cl, U_cl, thr_flood, dates)
            metrics = metrics_lo
            del Z_cl, U_cl
        else:
            # Run both and interpolate reach-level metrics
            print(f"    Simulating at Rlo...")
            Z_cl_lo, U_cl_lo = simulate_for_Rw(dmd, z0, V_sim, thr_flood, pod, Q_mat, Rlo)
            metrics_lo = compute_delta_metrics(pod, Z_ol, Z_cl_lo, U_cl_lo, thr_flood, dates)
            del Z_cl_lo, U_cl_lo
            gc.collect()

            print(f"    Simulating at Rhi...")
            Z_cl_hi, U_cl_hi = simulate_for_Rw(dmd, z0, V_sim, thr_flood, pod, Q_mat, Rhi)
            metrics_hi = compute_delta_metrics(pod, Z_ol, Z_cl_hi, U_cl_hi, thr_flood, dates)
            del Z_cl_hi, U_cl_hi
            gc.collect()

            # Linear interpolation of all per-reach metrics
            metrics = {}
            for key in metrics_lo.keys():
                metrics[key] = (1 - w_interp) * metrics_lo[key] + w_interp * metrics_hi[key]

        # Build dataframe
        df = pd.DataFrame(metrics)
        df["reach_id"] = matched
        df["gcm"] = gcm
        df["hydro"] = hydro
        df["ssp"] = int(scn)
        df["R_weight_lo"] = Rlo
        df["R_weight_hi"] = Rhi
        df["interp_weight"] = w_interp
        df["effort_Gm3yr"] = actual_eff
        all_rows.append(df)

        # Print summary
        print(f"    δ-based atten fraction:  median={np.median(metrics['delta_atten_frac'])*100:.1f}%  "
              f"P5={np.percentile(metrics['delta_atten_frac'], 5)*100:.1f}%  "
              f"P95={np.percentile(metrics['delta_atten_frac'], 95)*100:.1f}%")
        print(f"    raw-u  atten fraction:   median={np.median(metrics['u_atten_frac'])*100:.1f}%  "
              f"P5={np.percentile(metrics['u_atten_frac'], 5)*100:.1f}%  "
              f"P95={np.percentile(metrics['u_atten_frac'], 95)*100:.1f}%")
        print(f"    δ flood-only atten frac: median={np.median(metrics['delta_atten_flood_frac'])*100:.1f}%")
        print(f"    δ/flow ratio (exceed):   median={np.median(metrics['delta_ratio_flow_median'])*100:.1f}%  "
              f"P95={np.median(metrics['delta_ratio_flow_p95'])*100:.1f}%")

        # Basin-scale δ atten fraction
        basin_delta_atten = np.sum(metrics['delta_atten_L1'])
        basin_delta_total = np.sum(metrics['delta_total_L1'])
        basin_u_atten     = np.sum(metrics['u_atten_L1'])
        basin_u_total     = np.sum(metrics['u_total_L1'])
        print(f"    Basin δ-atten fraction:  {basin_delta_atten/(basin_delta_total+1e-12)*100:.1f}%")
        print(f"    Basin u-atten fraction:  {basin_u_atten/(basin_u_total+1e-12)*100:.1f}%")

        del X, U, pod, dmd, Z_ol
        gc.collect()

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main():
    global FLOOD_RL_PERIOD, OUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--rl", type=int, default=2)
    parser.add_argument("--gcm", default="all")
    parser.add_argument("--hydro", default="all")
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pipeline = _pipeline
    pipeline.FLOOD_RL_PERIOD = args.rl
    pipeline.OUT_DIR = f"outputs_flood_rl{args.rl}"

    global TARGET_BUDGET
    TARGET_BUDGET = args.budget

    out_dir = Path(f"outputs_flood_rl{args.rl}")
    out_dir.mkdir(exist_ok=True)

    gcms   = AVAILABLE_GCMS if args.gcm == "all" else [args.gcm]
    hydros = HYDRO_MODELS if args.hydro == "all" else [args.hydro]

    print(f"\n{'='*60}")
    print(f"  δ-based attenuation fraction computation")
    print(f"  RL threshold: {args.rl}-year")
    print(f"  Budget target: {TARGET_BUDGET} Gm³/yr")
    print(f"  Members: {len(gcms)} GCMs × {len(hydros)} hydro = {len(gcms)*len(hydros)}")
    print(f"{'='*60}")

    all_dfs = []
    summary_rows = []

    for hydro in hydros:
        for gcm in gcms:
            t0 = time.time()
            try:
                df = process_member(gcm, hydro)
                if len(df):
                    all_dfs.append(df)
                    # Per-member summary
                    for scn in df["ssp"].unique():
                        sub = df[df["ssp"] == scn]
                        summary_rows.append({
                            "gcm": gcm, "hydro": hydro, "ssp": int(scn),
                            "n_reaches": len(sub),
                            "delta_atten_frac_median": sub["delta_atten_frac"].median(),
                            "delta_atten_frac_p5": sub["delta_atten_frac"].quantile(0.05),
                            "delta_atten_frac_p95": sub["delta_atten_frac"].quantile(0.95),
                            "delta_atten_flood_frac_median": sub["delta_atten_flood_frac"].median(),
                            "u_atten_frac_median": sub["u_atten_frac"].median(),
                            "basin_delta_atten_frac": (sub["delta_atten_L1"].sum() /
                                                       (sub["delta_total_L1"].sum() + 1e-12)),
                            "basin_u_atten_frac": (sub["u_atten_L1"].sum() /
                                                   (sub["u_total_L1"].sum() + 1e-12)),
                            "delta_ratio_flow_median": sub["delta_ratio_flow_median"].median(),
                            "delta_ratio_flow_p95": sub["delta_ratio_flow_p95"].median(),
                            "effort_Gm3yr": sub["effort_Gm3yr"].iloc[0],
                        })
            except Exception as e:
                print(f"  ERROR {hydro}/{gcm}: {e}")
                import traceback; traceback.print_exc()

            elapsed = time.time() - t0
            print(f"  [{hydro}/{gcm}] done in {elapsed:.0f}s")
            gc.collect()

    # Save
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        reach_path = out_dir / "delta_atten_per_reach.parquet"
        df_all.to_parquet(reach_path, index=False)
        print(f"\nSaved per-reach: {reach_path} ({len(df_all)} rows)")

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        sum_path = out_dir / "delta_atten_summary.csv"
        df_sum.to_csv(sum_path, index=False)
        print(f"Saved summary:   {sum_path} ({len(df_sum)} rows)")

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"  COMPARISON: δ-based vs raw-u attenuation fraction")
        print(f"{'='*70}")
        print(f"  {'SSP':<12} {'δ per-reach':>14} {'δ basin':>14} {'u per-reach':>14} {'u basin':>14}")
        print(f"  {'-'*66}")
        for scn in sorted(df_sum["ssp"].unique()):
            s = df_sum[df_sum["ssp"] == scn]
            print(f"  SSP{scn:<8} "
                  f"{s['delta_atten_frac_median'].median()*100:>12.1f}% "
                  f"{s['basin_delta_atten_frac'].median()*100:>12.1f}% "
                  f"{s['u_atten_frac_median'].median()*100:>12.1f}% "
                  f"{s['basin_u_atten_frac'].median()*100:>12.1f}%")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()