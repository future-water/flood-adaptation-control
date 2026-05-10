from __future__ import annotations

"""
Stream-order-based control constraint comparison.

Instead of HUC-outlet placement, this script constrains the B matrix
by Strahler stream order:
  - band_N_M   : control only reaches with N ≤ order ≤ M  (default: 1-2 vs 3-4)
  - low_leq_N  : control only reaches with order ≤ N  (headwater / tributary)
  - high_geq_N : control only reaches with order ≥ N  (mainstem)

Each qualifying reach gets its own actuator (proper DARE formulation).

Usage (band comparison, default):
    python dmd_lqr_stream_order.py \
        --gcm MPI-ESM1-2-HR --hydro VIC5 --scenario 245 \
        --bands 1-2 3-4

Usage (cumulative thresholds):
    python dmd_lqr_stream_order.py \
        --gcm MPI-ESM1-2-HR --hydro VIC5 --scenario 245 \
        --low-thresholds 1 2 3 --high-thresholds 3 4 5 6 --bands
"""

import gc
import os
import argparse
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are

warnings.filterwarnings("ignore")

# ── Import from base pipeline ──
import importlib as _il
_pipeline = _il.import_module('04_dmd_lqr_analysis')
RANK = _pipeline.RANK
DTYPE = _pipeline.DTYPE
DT = _pipeline.DT
R_VALUES = _pipeline.R_VALUES
Q_WEIGHT = _pipeline.Q_WEIGHT
CHUNK = _pipeline.CHUNK
SCENARIOS = _pipeline.SCENARIOS
AVAILABLE_GCMS = _pipeline.AVAILABLE_GCMS
HYDRO_MODELS = _pipeline.HYDRO_MODELS
ANALYSIS_ERA = _pipeline.ANALYSIS_ERA
RETURN_PERIODS = _pipeline.RETURN_PERIODS
FLOOD_RL_PERIOD = _pipeline.FLOOD_RL_PERIOD
load_scenario_data = _pipeline.load_scenario_data
match_to_reference = _pipeline.match_to_reference
compute_reference_pack = _pipeline.compute_reference_pack
compute_pod = _pipeline.compute_pod
normalize_forcing = _pipeline.normalize_forcing
fit_windowed_dmdc = _pipeline.fit_windowed_dmdc
simulate_openloop = _pipeline.simulate_openloop
simulate_one_step_anchored = _pipeline.simulate_one_step_anchored
precompute_baseline = _pipeline.precompute_baseline
compute_controlled_metrics = _pipeline.compute_controlled_metrics
simulate_cl_flood_only = _pipeline.simulate_cl_flood_only
ReferencePack = _pipeline.ReferencePack
PODPack = _pipeline.PODPack
DMDcPack = _pipeline.DMDcPack
PrecompPack = _pipeline.PrecompPack
Profiler = _pipeline.Profiler

# ── Module-level profiler ──
PROF = Profiler()

# ============================================================
# Default column name for Strahler order in NHDPlus attributes
# ============================================================
DEFAULT_ORDER_COL = "StreamOrde"
DEFAULT_MAPPING_CSV = "comid_huc_mapping_correct.csv"

# Stream order thresholds to sweep
DEFAULT_LOW_THRESHOLDS = []      # control order ≤ N
DEFAULT_HIGH_THRESHOLDS = []     # control order ≥ N
DEFAULT_BANDS = [(1, 2), (3, 4)] # order bands for direct comparison


# ============================================================
# 1. Stream Order Mapping
# ============================================================

def load_stream_order_mapping(
    comid_list: np.ndarray,
    mapping_csv: str = DEFAULT_MAPPING_CSV,
    order_col: str = DEFAULT_ORDER_COL,
) -> pd.DataFrame:
    """
    Load COMID → stream order mapping aligned to the reach index order.
    Returns DataFrame with columns: COMID, reach_idx, stream_order.
    """
    mapping = pd.read_csv(mapping_csv)
    mapping["COMID"] = mapping["COMID"].astype(int)

    if order_col not in mapping.columns:
        available = [c for c in mapping.columns if "ord" in c.lower() or "strahler" in c.lower()]
        raise KeyError(
            f"Column '{order_col}' not found in {mapping_csv}. "
            f"Available candidates: {available}. All columns: {list(mapping.columns)}"
        )

    comid_df = pd.DataFrame({
        "COMID": np.asarray(comid_list, dtype=int),
        "reach_idx": np.arange(len(comid_list)),
    })

    merged = comid_df.merge(
        mapping[["COMID", order_col]].drop_duplicates(subset="COMID"),
        on="COMID",
        how="left",
    )
    merged = merged.rename(columns={order_col: "stream_order"})
    merged = merged.sort_values("reach_idx").reset_index(drop=True)

    n_total = len(comid_list)
    n_missing = merged["stream_order"].isna().sum()
    if n_missing > 0:
        print(f"  [stream order] {n_missing}/{n_total} COMIDs missing order → excluded from constrained runs")

    order_dist = merged["stream_order"].dropna().astype(int).value_counts().sort_index()
    print(f"  [stream order] Distribution:")
    for order_val, count in order_dist.items():
        print(f"      Order {int(order_val):>2d}: {count:>5d} reaches ({100*count/n_total:.1f}%)")

    return merged


# ============================================================
# 2. B Matrix Construction (Stream-Order Selection)
# ============================================================

def build_B_stream_order(
    order_mapping: pd.DataFrame,
    mode: str,
    threshold: int,
    n_reaches: int,
    threshold_hi: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Build physical-space B matrix by stream order selection.

    Parameters
    ----------
    order_mapping : DataFrame with reach_idx and stream_order columns
    mode : 'low' (order ≤ threshold), 'high' (order ≥ threshold),
           or 'band' (threshold ≤ order ≤ threshold_hi)
    threshold : Strahler order cutoff (lower bound for band)
    threshold_hi : upper bound for band mode (inclusive)
    n_reaches : total number of reaches (rows of B)

    Returns
    -------
    B_phys : (n_reaches, m) binary selection matrix
    selected_indices : 1-D array of selected reach indices
    label : human-readable label for this configuration
    """
    df = order_mapping.dropna(subset=["stream_order"]).copy()
    df["stream_order"] = df["stream_order"].astype(int)

    if mode == "low":
        mask = df["stream_order"] <= threshold
        label = f"low_leq_{threshold}"
    elif mode == "high":
        mask = df["stream_order"] >= threshold
        label = f"high_geq_{threshold}"
    elif mode == "band":
        if threshold_hi is None:
            raise ValueError("band mode requires threshold_hi")
        mask = (df["stream_order"] >= threshold) & (df["stream_order"] <= threshold_hi)
        label = f"band_{threshold}_{threshold_hi}"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    selected = df.loc[mask, "reach_idx"].values.astype(int)
    selected = selected[selected < n_reaches]  # safety clamp
    m = len(selected)

    if m == 0:
        print(f"  WARNING: {label} selects 0 reaches — skipping")
        return np.zeros((n_reaches, 0), dtype=DTYPE), np.array([], dtype=int), label

    # B_phys: each selected reach gets its own actuator column
    B_phys = np.zeros((n_reaches, m), dtype=DTYPE)
    for j, idx in enumerate(selected):
        B_phys[idx, j] = 1.0

    pct = 100 * m / n_reaches
    order_range = df.loc[mask, "stream_order"]
    print(f"  B [{label}]: {m}/{n_reaches} reaches ({pct:.1f}%), "
          f"orders {int(order_range.min())}–{int(order_range.max())}")

    return B_phys, selected, label


# ============================================================
# 3. B_r Compression + General LQR
# ============================================================

def compress_B_r(B_r: np.ndarray, tol: float = 1e-10) -> Tuple[np.ndarray, int]:
    """
    Compress B_r (r × m, where m can be thousands) to B_eff (r × k, k ≤ r).

    Since B_r = U_r^T @ B_phys where B_phys is a selection matrix and
    U_r has only r=30 columns, rank(B_r) ≤ r.  Thousands of actuators
    collapse into at most 30 independent control directions.

    Returns B_eff (r × k) and effective rank k.
    DARE with B_eff: O(r³) instead of O(m³).  Simulation: O(rk) instead of O(rm).
    """
    U_b, s, Vh = np.linalg.svd(B_r, full_matrices=False)
    k = int(np.sum(s > tol * s[0]))
    B_eff = (U_b[:, :k] * s[:k]).astype(DTYPE)  # (r × k)
    return B_eff, k


def lqr_gain_compressed(
    A: np.ndarray,
    B_eff: np.ndarray,
    Q: np.ndarray,
    rho: float,
) -> np.ndarray:
    """
    Discrete-time LQR gain using compressed B_eff (r × k, k ≤ r).
    Returns K_eff (k × r).  In simulation: u_z = B_eff @ (-K_eff @ z_excess).

    This is mathematically equivalent to the full (r × m) formulation
    but DARE sees R_eff = ρI_k instead of ρI_m.
    """
    k = B_eff.shape[1]
    R_eff = rho * np.eye(k, dtype=np.float64)

    P = solve_discrete_are(
        A.astype(np.float64),
        B_eff.astype(np.float64),
        Q.astype(np.float64),
        R_eff,
    )
    K_eff = np.linalg.solve(
        R_eff + B_eff.T @ P @ B_eff,
        B_eff.T @ P @ A,
    )
    return K_eff.astype(DTYPE)


# ============================================================
# 4. Closed-Loop Simulation (compressed B_eff)
# ============================================================

def simulate_cl_flood_compressed(
    dmd: DMDcPack,
    z0: np.ndarray,
    V_sim: np.ndarray,
    K_eff_list: List[np.ndarray],
    B_eff: np.ndarray,
    thr_flood: np.ndarray,
    U_r: np.ndarray,
    mean_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-loop flood-only simulation with compressed B_eff.

    Control law: u_eff = -K_eff @ z_excess  (k × 1)
                 u_z   =  B_eff @ u_eff     (r × 1)
    Dynamics:    z_{k+1} = A z_k + u_z + E v_k + b

    When B_eff comes from compress_B_r, this is mathematically
    equivalent to the full (r × m) formulation.
    """
    r, T = z0.shape[0], V_sim.shape[1]

    Z = np.zeros((r, T + 1), dtype=DTYPE)
    U_ctrl = np.zeros((r, T), dtype=DTYPE)
    Z[:, 0:1] = z0

    thr = thr_flood.reshape(-1, 1)

    for k in range(T):
        w = dmd.window_for_k[k]
        A, E, b = dmd.A_list[w], dmd.E_list[w], dmd.b_list[w]
        K_eff = K_eff_list[w]

        # Reconstruct physical state, compute flood excess
        x_k = U_r @ Z[:, k:k+1] + mean_x
        x_excess = np.maximum(x_k - thr, 0.0)
        z_excess = U_r.T @ x_excess

        # Compressed control: k-dim instead of m-dim
        u_eff = -K_eff @ z_excess    # (k, 1)
        u_z = B_eff @ u_eff          # (r, 1)

        U_ctrl[:, k:k+1] = u_z
        Z[:, k+1:k+2] = A @ Z[:, k:k+1] + u_z + E @ V_sim[:, k:k+1] + b

    return Z, U_ctrl


# ============================================================
# 5. Run One Stream-Order Configuration
# ============================================================

def run_stream_order_level(
    label: str,
    B_eff: np.ndarray,
    eff_rank: int,
    n_selected: int,
    gcm: str,
    scn: int,
    hydro: str,
    ref: ReferencePack,
    pod: PODPack,
    dmd: DMDcPack,
    pre: PrecompPack,
    V_sim: np.ndarray,
    z0: np.ndarray,
    thr_flood: np.ndarray,
    vol_hi_ref: np.ndarray,
    matched: list,
    R_values: list = None,
) -> pd.DataFrame:
    """
    Sweep R values for one stream-order B_eff configuration.
    B_eff is already compressed (r × k, k ≤ r).
    """
    PROF.start(f"sweep_{label}")

    if R_values is None:
        R_values = R_VALUES

    Q = (Q_WEIGHT * np.eye(RANK)).astype(DTYPE)
    results = []

    for Rw in R_values:
        try:
            K_eff_list = [lqr_gain_compressed(A, B_eff, Q, float(Rw))
                          for A in dmd.A_list]
        except Exception as e:
            print(f"      [{label}] R={Rw:.2e}: DARE failed ({e})")
            continue

        Z_cl, U_cl = simulate_cl_flood_compressed(
            dmd, z0, V_sim, K_eff_list, B_eff,
            thr_flood, pod.U_r, pod.mean_x,
        )

        m = compute_controlled_metrics(thr_flood, pod, pre, Z_cl, U_cl, vol_hi_ref)

        effort_gm3 = np.nansum(m["eff_total_L1"]) / 1e9
        residual_gm3 = np.nansum(m["vol_hi_controlled"]) / 1e9
        baseline_gm3 = np.nansum(m["vol_hi_baseline"]) / 1e9
        red_frac = (baseline_gm3 - residual_gm3) / (baseline_gm3 + 1e-12)

        atten_gm3 = np.nansum(m["eff_atten_L1"]) / 1e9
        augment_gm3 = np.nansum(m["eff_augment_L1"]) / 1e9

        print(f"      [{label:>12s}] R={Rw:>10.2e}: "
              f"eff={effort_gm3:>6.1f} Gm³, "
              f"atten={atten_gm3:>5.1f}/aug={augment_gm3:>5.1f}, "
              f"resid={residual_gm3:>6.1f}, "
              f"red={red_frac:>5.1%}")

        df_r = pd.DataFrame(m)
        df_r["constraint_label"] = label
        parts = label.split("_")
        df_r["constraint_mode"] = parts[0]  # 'low', 'high', or 'band'
        if parts[0] == "band":
            df_r["order_lo"] = int(parts[1])
            df_r["order_hi"] = int(parts[2])
            df_r["order_threshold"] = int(parts[1])  # keep for backward compat
        else:
            df_r["order_lo"] = int(parts[-1]) if parts[0] == "high" else 1
            df_r["order_hi"] = int(parts[-1]) if parts[0] == "low" else 99
            df_r["order_threshold"] = int(parts[-1])
        df_r["n_ctrl_reaches"] = n_selected
        df_r["eff_rank"] = eff_rank
        df_r["ctrl_fraction"] = n_selected / len(matched) if len(matched) > 0 else 0.0
        df_r["R_weight"] = float(Rw)
        df_r["reach_id"] = matched
        df_r["gcm"] = gcm
        df_r["hydro"] = hydro
        df_r["scenario"] = int(scn)

        results.append(df_r)

        del Z_cl, U_cl, K_eff_list, m
        gc.collect()

    PROF.stop(f"sweep_{label}")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ============================================================
# 6. Run Reach-Level Baseline (B = I, original formulation)
# ============================================================

def run_reach_baseline(
    gcm: str,
    scn: int,
    hydro: str,
    ref: ReferencePack,
    pod: PODPack,
    dmd: DMDcPack,
    pre: PrecompPack,
    V_sim: np.ndarray,
    z0: np.ndarray,
    thr_flood: np.ndarray,
    vol_hi_ref: np.ndarray,
    matched: list,
    R_values: list = None,
) -> pd.DataFrame:
    """
    Full reach-level control (B = I_r) as upper-bound baseline.
    Uses the original simulate_cl_flood_only for consistency.
    """
    PROF.start("sweep_reach")

    if R_values is None:
        R_values = R_VALUES

    lqr_gain = _pipeline.lqr_gain  # B=I version

    Q = (Q_WEIGHT * np.eye(RANK)).astype(DTYPE)
    results = []

    for Rw in R_values:
        try:
            K_list = [lqr_gain(A, Q, float(Rw)) for A in dmd.A_list]
        except Exception as e:
            print(f"      [reach] R={Rw:.2e}: Failed ({e})")
            continue

        Z_cl, U_cl = simulate_cl_flood_only(
            dmd, z0, V_sim, K_list,
            thr_flood, pod.U_r, pod.mean_x,
        )

        m = compute_controlled_metrics(thr_flood, pod, pre, Z_cl, U_cl, vol_hi_ref)

        effort_gm3 = np.nansum(m["eff_total_L1"]) / 1e9
        residual_gm3 = np.nansum(m["vol_hi_controlled"]) / 1e9
        baseline_gm3 = np.nansum(m["vol_hi_baseline"]) / 1e9
        red_frac = (baseline_gm3 - residual_gm3) / (baseline_gm3 + 1e-12)

        print(f"      [       reach] R={Rw:>10.2e}: "
              f"eff={effort_gm3:>6.1f} Gm³, "
              f"resid={residual_gm3:>6.1f}, "
              f"red={red_frac:>5.1%}")

        df_r = pd.DataFrame(m)
        df_r["constraint_label"] = "reach"
        df_r["constraint_mode"] = "reach"
        df_r["order_threshold"] = -1
        n_reaches = len(matched)
        df_r["n_ctrl_reaches"] = n_reaches
        df_r["ctrl_fraction"] = 1.0
        df_r["R_weight"] = float(Rw)
        df_r["reach_id"] = matched
        df_r["gcm"] = gcm
        df_r["hydro"] = hydro
        df_r["scenario"] = int(scn)

        results.append(df_r)

        del Z_cl, U_cl, K_list, m
        gc.collect()

    PROF.stop("sweep_reach")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ============================================================
# 7. Main Comparison Pipeline
# ============================================================

def run_stream_order_comparison(
    gcm: str,
    hydro: str,
    scn: int,
    low_thresholds: List[int],
    high_thresholds: List[int],
    bands: List[Tuple[int, int]] = None,
    mapping_csv: str = DEFAULT_MAPPING_CSV,
    order_col: str = DEFAULT_ORDER_COL,
    out_dir: str = "outputs_stream_order",
    include_reach_baseline: bool = True,
):
    """
    Run the full comparison across stream-order constraints for one
    (GCM, hydro, scenario) combination.

    bands: list of (lo, hi) tuples, e.g. [(1,2), (3,4)] → order 1-2 vs 3-4
    """
    if bands is None:
        bands = []

    global PROF
    PROF = Profiler()

    print(f"\n{'='*70}")
    print(f"STREAM ORDER COMPARISON: {hydro}/{gcm}/SSP{scn}")
    if low_thresholds:
        print(f"  Low (headwater) thresholds:  order ≤ {low_thresholds}")
    if high_thresholds:
        print(f"  High (mainstem) thresholds:  order ≥ {high_thresholds}")
    if bands:
        print(f"  Bands:  {['order ' + str(lo) + '–' + str(hi) for lo, hi in bands]}")
    print(f"  Mapping: {mapping_csv} (column: {order_col})")
    print(f"  DARE with per-reach actuators")
    print(f"{'='*70}")

    # ── 1. Reference pack ──
    print("\n[1] Reference pack ...")
    ref = compute_reference_pack(gcm, hydro, verbose=True)

    # ── 2. Load scenario data ──
    print("\n[2] Loading scenario data ...")
    X, U, dates, reach_ids = load_scenario_data(scn, gcm, hydro)
    idx_ref, idx_this, matched = match_to_reference(ref.reach_ids, reach_ids)
    X = X[idx_this]
    U = U[idx_this]
    thr_flood = ref.thr_flood[idx_ref]
    vol_hi_ref = ref.vol_hi_ref[idx_ref]
    n_reaches = X.shape[0]
    print(f"    {n_reaches} reaches, {X.shape[1]} time steps")

    # ── 3. POD + DMDc (computed once) ──
    print("\n[3] POD + DMDc ...")
    PROF.start("pod_dmdc")
    pod = compute_pod(X, RANK)
    V = normalize_forcing(U).V
    dmd = fit_windowed_dmdc(pod.Z, V)

    z0, V_sim = pod.Z[:, 0:1], V[:, :-1]
    Z_ol = simulate_openloop(dmd, z0, V_sim)
    Z_anch = simulate_one_step_anchored(dmd, pod.Z, V_sim)
    pre = precompute_baseline(X, thr_flood, dates, pod, Z_ol, Z_anch)
    PROF.stop("pod_dmdc")

    # Report open-loop skill
    print(f"    NSE open-loop  median: {np.nanmedian(pre.nse_openloop):.4f}")
    print(f"    KGE open-loop  median: {np.nanmedian(pre.kge_openloop):.4f}")
    print(f"    Spectral radii: {[f'{s:.3f}' for s in dmd.spectral_radii]}")

    # ── 4. Stream order mapping ──
    print("\n[4] Stream order mapping ...")
    comid_array = np.array(matched, dtype=int)
    order_mapping = load_stream_order_mapping(comid_array, mapping_csv, order_col)

    # ── 5. Collect all configurations ──
    configs = []  # list of (mode, label, threshold_lo, threshold_hi)

    for thr_val in sorted(set(low_thresholds)):
        configs.append(("low", f"low_leq_{thr_val}", thr_val, None))
    for thr_val in sorted(set(high_thresholds)):
        configs.append(("high", f"high_geq_{thr_val}", thr_val, None))
    for lo, hi in bands:
        configs.append(("band", f"band_{lo}_{hi}", lo, hi))

    # ── 6. Run each configuration ──
    all_results = []

    # 6a. Reach-level baseline
    if include_reach_baseline:
        print(f"\n{'─'*60}")
        print(f"  [reach] Full reach-level control (B = I_r, upper bound)")
        print(f"{'─'*60}")
        df_reach = run_reach_baseline(
            gcm=gcm, scn=scn, hydro=hydro, ref=ref,
            pod=pod, dmd=dmd, pre=pre,
            V_sim=V_sim, z0=z0,
            thr_flood=thr_flood, vol_hi_ref=vol_hi_ref,
            matched=matched,
        )
        if len(df_reach):
            all_results.append(df_reach)

    # 6b. Stream-order-constrained runs
    for mode, label, threshold, threshold_hi in configs:
        print(f"\n{'─'*60}")
        print(f"  [{label}]")
        print(f"{'─'*60}")

        B_phys, selected_idx, label_out = build_B_stream_order(
            order_mapping, mode, threshold, n_reaches, threshold_hi=threshold_hi,
        )

        if B_phys.shape[1] == 0:
            print(f"    → No reaches selected, skipping.")
            continue

        # Project to reduced space and compress: B_r (r × m) → B_eff (r × k)
        PROF.start(f"project_Br_{label}")
        B_r = (pod.U_r.T @ B_phys).astype(DTYPE)
        B_eff, eff_rank = compress_B_r(B_r)
        PROF.stop(f"project_Br_{label}")

        m_actuators = B_phys.shape[1]
        print(f"    B_r: ({RANK} × {m_actuators}) → B_eff: ({RANK} × {eff_rank})  "
              f"[{m_actuators} actuators compressed to {eff_rank} control directions]")

        df = run_stream_order_level(
            label=label,
            B_eff=B_eff,
            eff_rank=eff_rank,
            n_selected=len(selected_idx),
            gcm=gcm, scn=scn, hydro=hydro, ref=ref,
            pod=pod, dmd=dmd, pre=pre,
            V_sim=V_sim, z0=z0,
            thr_flood=thr_flood, vol_hi_ref=vol_hi_ref,
            matched=matched,
        )
        if len(df):
            all_results.append(df)

        del B_phys, B_r, B_eff
        gc.collect()

    # ── 7. Save ──
    print(f"\n{'='*70}")
    print(f"[7] Saving results ...")
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        fname = f"stream_order_{hydro}_{gcm}_ssp{scn}.csv"
        df_all.to_csv(out_path / fname, index=False)
        print(f"    Saved: {out_path / fname} ({len(df_all)} rows)")

        # ── Aggregate summary ──
        _print_summary(df_all, out_path, hydro, gcm, scn)
    else:
        print("    No results to save.")

    PROF.report(f"STREAM ORDER: {hydro}/{gcm}/SSP{scn}")

    del X, U, V, Z_ol, Z_anch, pod, dmd, pre
    gc.collect()


# ============================================================
# 8. Summary / Pareto Table
# ============================================================

def _print_summary(df_all: pd.DataFrame, out_path: Path, hydro: str, gcm: str, scn: int):
    """
    Print and save the effort-vs-reduction summary, grouped by constraint.
    This is the core comparison table.
    """
    summary = (
        df_all.groupby(["constraint_label", "constraint_mode", "order_threshold",
                         "n_ctrl_reaches", "ctrl_fraction", "R_weight"])
        .agg(
            effort_Gm3=("eff_total_L1", lambda x: x.sum() / 1e9),
            atten_Gm3=("eff_atten_L1", lambda x: x.sum() / 1e9),
            augment_Gm3=("eff_augment_L1", lambda x: x.sum() / 1e9),
            residual_Gm3=("vol_hi_controlled", lambda x: x.sum() / 1e9),
            baseline_Gm3=("vol_hi_baseline", lambda x: x.sum() / 1e9),
            median_vol_red=("vol_hi_red_rel", "median"),
            median_peak_red=("peak_red_rel", "median"),
            median_days_red=("days_hi_red_rel", "median"),
        )
        .reset_index()
    )

    summary["reduction_frac"] = (
        (summary["baseline_Gm3"] - summary["residual_Gm3"])
        / (summary["baseline_Gm3"] + 1e-12)
    )
    summary["efficiency"] = (
        (summary["baseline_Gm3"] - summary["residual_Gm3"])
        / (summary["effort_Gm3"] + 1e-12)
    )

    # Sort for readability
    mode_order = {"reach": 0, "low": 1, "band": 2, "high": 3}
    summary["_sort"] = summary["constraint_mode"].map(mode_order).fillna(4)
    summary = summary.sort_values(["_sort", "order_threshold", "R_weight"]).drop(columns="_sort")

    print(f"\n{'='*90}")
    print(f"  STREAM ORDER COMPARISON SUMMARY")
    print(f"{'='*90}")

    # Compact table: one line per (label, R)
    fmt = "{:<14s} {:>5d} ({:>4.0%})  R={:>8.1e}  eff={:>5.1f}  resid={:>5.1f}  red={:>5.1%}  η={:>5.2f}"
    print(f"  {'Config':<14s} {'nCtrl':>5s} {'(frac)':>6s}  {'R_weight':>10s}  "
          f"{'Effort':>5s}  {'Resid':>5s}  {'Reduct':>6s}  {'Effcy':>5s}")
    print(f"  {'─'*84}")

    for _, row in summary.iterrows():
        print(f"  " + fmt.format(
            row["constraint_label"],
            int(row["n_ctrl_reaches"]),
            row["ctrl_fraction"],
            row["R_weight"],
            row["effort_Gm3"],
            row["residual_Gm3"],
            row["reduction_frac"],
            row["efficiency"],
        ))

    fname = f"stream_order_summary_{hydro}_{gcm}_ssp{scn}.csv"
    summary.to_csv(out_path / fname, index=False)
    print(f"\n  Summary saved: {out_path / fname}")

    # ── Pareto frontier (per constraint label) ──
    print(f"\n  PARETO FRONTIER (non-dominated effort vs residual):")
    print(f"  {'─'*60}")

    pareto_rows = []
    for label, grp in summary.groupby("constraint_label"):
        # Sort by effort ascending
        grp_sorted = grp.sort_values("effort_Gm3").reset_index(drop=True)
        best_resid = float("inf")
        for _, row in grp_sorted.iterrows():
            if row["residual_Gm3"] < best_resid:
                best_resid = row["residual_Gm3"]
                pareto_rows.append(row)

    if pareto_rows:
        pareto_df = pd.DataFrame(pareto_rows)
        pareto_df = pareto_df.sort_values(["constraint_label", "effort_Gm3"])
        for _, row in pareto_df.iterrows():
            print(f"    {row['constraint_label']:<14s}  "
                  f"eff={row['effort_Gm3']:>6.1f}  "
                  f"resid={row['residual_Gm3']:>6.1f}  "
                  f"red={row['reduction_frac']:>5.1%}")
        pareto_fname = f"stream_order_pareto_{hydro}_{gcm}_ssp{scn}.csv"
        pareto_df.to_csv(out_path / pareto_fname, index=False)


# ============================================================
# 9. Multi-Scenario / Multi-GCM Runner
# ============================================================

def run_all_stream_order(
    gcm: str,
    hydro: str,
    scenarios: List[int],
    low_thresholds: List[int],
    high_thresholds: List[int],
    bands: List[Tuple[int, int]] = None,
    mapping_csv: str = DEFAULT_MAPPING_CSV,
    order_col: str = DEFAULT_ORDER_COL,
    out_dir: str = "outputs_stream_order",
):
    """Run stream order comparison for multiple scenarios."""
    if bands is None:
        bands = list(DEFAULT_BANDS)

    gcms = AVAILABLE_GCMS if gcm == "all" else [gcm]
    hydros = HYDRO_MODELS if hydro == "all" else [hydro]

    for h in hydros:
        for g in gcms:
            for s in scenarios:
                try:
                    run_stream_order_comparison(
                        gcm=g, hydro=h, scn=s,
                        low_thresholds=low_thresholds,
                        high_thresholds=high_thresholds,
                        bands=bands,
                        mapping_csv=mapping_csv,
                        order_col=order_col,
                        out_dir=out_dir,
                    )
                except Exception as e:
                    print(f"ERROR {h}/{g}/SSP{s}: {e}")
                    import traceback
                    traceback.print_exc()
                gc.collect()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stream-order-based control constraint comparison"
    )
    parser.add_argument("--gcm", default="MPI-ESM1-2-HR",
                        help="GCM name or 'all'")
    parser.add_argument("--hydro", default="VIC5",
                        help="Hydro model: VIC5, PRMS, or 'all'")
    parser.add_argument("--scenario", type=int, nargs="+", default=[245],
                        help="SSP scenarios (can pass multiple)")
    parser.add_argument("--low-thresholds", type=int, nargs="*",
                        default=DEFAULT_LOW_THRESHOLDS,
                        help="Control order ≤ N (headwater). Empty by default.")
    parser.add_argument("--high-thresholds", type=int, nargs="*",
                        default=DEFAULT_HIGH_THRESHOLDS,
                        help="Control order ≥ N (mainstem). Empty by default.")
    parser.add_argument("--bands", type=str, nargs="*",
                        default=None,
                        help="Order bands as lo-hi pairs, e.g. '1-2' '3-4'. "
                             "Default: 1-2 3-4")
    parser.add_argument("--order-col", default=DEFAULT_ORDER_COL,
                        help="Column name for Strahler order in mapping CSV")
    parser.add_argument("--mapping", default=DEFAULT_MAPPING_CSV,
                        help="COMID → stream order mapping CSV")
    parser.add_argument("--outdir", default="outputs_stream_order",
                        help="Output directory")
    parser.add_argument("--no-reach-baseline", action="store_true",
                        help="Skip the full reach-level baseline (B=I)")
    args = parser.parse_args()

    # Parse bands
    if args.bands is not None:
        bands = []
        for b in args.bands:
            lo, hi = b.split("-")
            bands.append((int(lo), int(hi)))
    else:
        bands = list(DEFAULT_BANDS)

    n_configs = (len(args.low_thresholds) + len(args.high_thresholds)
                 + len(bands) + (0 if args.no_reach_baseline else 1))
    n_combos = (
        len(AVAILABLE_GCMS if args.gcm == "all" else [args.gcm])
        * len(HYDRO_MODELS if args.hydro == "all" else [args.hydro])
        * len(args.scenario)
    )

    print(f"\n{'='*70}")
    print(f"STREAM ORDER CONTROL COMPARISON")
    if args.low_thresholds:
        print(f"  Low thresholds (order ≤ N): {args.low_thresholds}")
    if args.high_thresholds:
        print(f"  High thresholds (order ≥ N): {args.high_thresholds}")
    if bands:
        print(f"  Bands: {['order ' + str(lo) + '–' + str(hi) for lo, hi in bands]}")
    print(f"  + reach baseline: {not args.no_reach_baseline}")
    print(f"  Configurations per scenario: {n_configs}")
    print(f"  R grid: {len(R_VALUES)} values")
    print(f"  Total (GCM × hydro × scenario): {n_combos}")
    print(f"  Output: {args.outdir}")
    print(f"{'='*70}")

    _include_reach = not args.no_reach_baseline

    gcms = AVAILABLE_GCMS if args.gcm == "all" else [args.gcm]
    hydros = HYDRO_MODELS if args.hydro == "all" else [args.hydro]

    for h in hydros:
        for g in gcms:
            for s in args.scenario:
                try:
                    run_stream_order_comparison(
                        gcm=g, hydro=h, scn=s,
                        low_thresholds=args.low_thresholds,
                        high_thresholds=args.high_thresholds,
                        bands=bands,
                        mapping_csv=args.mapping,
                        order_col=args.order_col,
                        out_dir=args.outdir,
                        include_reach_baseline=_include_reach,
                    )
                except Exception as e:
                    print(f"ERROR {h}/{g}/SSP{s}: {e}")
                    import traceback
                    traceback.print_exc()
                gc.collect()


if __name__ == "__main__":
    main()