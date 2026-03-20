"""
Supplementary: HUC-Level Control Constraint Comparison
======================================================

Compares reach-level vs HUC-aggregated (HUC8/10/12) control placement.
Produces per-member CSVs and spatial compact output for Fig S (HUC Flexibility).
Used by 06_figures_supplementary.ipynb.

Requires comid_huc_mapping_correct.csv (from WBD spatial join).

Usage:
    python si_huc_comparison.py --gcm EC-Earth3 --hydro VIC5 --scenario 245
"""

from __future__ import annotations

import gc
import argparse
import importlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are

warnings.filterwarnings("ignore")

# Import main pipeline (04_dmd_lqr_analysis.py)
_pipeline = importlib.import_module("04_dmd_lqr_analysis")

RANK       = _pipeline.RANK
DTYPE      = _pipeline.DTYPE
DT         = _pipeline.DT
R_VALUES   = _pipeline.R_VALUES
Q_WEIGHT   = _pipeline.Q_WEIGHT
CHUNK      = _pipeline.CHUNK
SCENARIOS  = _pipeline.SCENARIOS
AVAILABLE_GCMS = _pipeline.AVAILABLE_GCMS
HYDRO_MODELS   = _pipeline.HYDRO_MODELS

load_scenario_data      = _pipeline.load_scenario_data
match_to_reference      = _pipeline.match_to_reference
compute_reference_pack  = _pipeline.compute_reference_pack
compute_pod             = _pipeline.compute_pod
normalize_forcing       = _pipeline.normalize_forcing
fit_windowed_dmdc       = _pipeline.fit_windowed_dmdc
simulate_openloop       = _pipeline.simulate_openloop
simulate_one_step_anchored = _pipeline.simulate_one_step_anchored
precompute_baseline     = _pipeline.precompute_baseline
compute_controlled_metrics = _pipeline.compute_controlled_metrics
lqr_gain                = _pipeline.lqr_gain
simulate_cl_flood_only  = _pipeline.simulate_cl_flood_only
PROF                    = _pipeline.PROF

ReferencePack = _pipeline.ReferencePack
PODPack       = _pipeline.PODPack
DMDcPack      = _pipeline.DMDcPack
PrecompPack   = _pipeline.PrecompPack


# ============================================================
# 1. HUC Mapping (from pre-computed WBD spatial join CSV)
# ============================================================

MAPPING_CSV = "comid_huc_mapping_correct.csv"


def build_huc_mapping(comid_list: np.ndarray, mapping_csv: str = MAPPING_CSV) -> pd.DataFrame:
    """Map COMID → HUC8/10/12, aligned to comid_list order (= U_r row order)."""
    mapping = pd.read_csv(mapping_csv)
    mapping['COMID'] = mapping['COMID'].astype(int)

    comid_df = pd.DataFrame({
        'COMID': np.asarray(comid_list, dtype=int),
        'reach_idx': np.arange(len(comid_list))
    })

    merged = comid_df.merge(mapping, on='COMID', how='left')
    n_missing = merged['HUC8'].isna().sum()
    if n_missing > 0:
        print(f"  [HUC mapping] {n_missing}/{len(comid_list)} COMIDs not found in mapping")

    return merged.sort_values('reach_idx').reset_index(drop=True)


# ============================================================
# 2. B Matrix Construction (Outlet-Only)
# ============================================================

def build_B_matrix(
    huc_mapping: pd.DataFrame,
    huc_level: str,
    n_reaches: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build B matrix: (n_reaches x m_units).
    Each HUC unit's outlet reach (max TotDASqKM) gets a 1; all others are 0.
    Physical meaning: one control facility at each HUC outlet.
    """
    df = huc_mapping.copy()
    huc_ids = sorted(df[huc_level].dropna().unique().tolist())
    m = len(huc_ids)
    huc_to_col = {h: j for j, h in enumerate(huc_ids)}

    # Outlet = reach with largest drainage area within each HUC
    outlet_idx = (
        df.dropna(subset=[huc_level])
        .groupby(huc_level)['TotDASqKM']
        .idxmax()
    )

    B = np.zeros((n_reaches, m), dtype=DTYPE)
    n_assigned = 0
    for huc_id, row_idx in outlet_idx.items():
        reach_idx = int(df.loc[row_idx, 'reach_idx'])
        if reach_idx < n_reaches and huc_id in huc_to_col:
            B[reach_idx, huc_to_col[huc_id]] = 1.0
            n_assigned += 1

    print(f"  B matrix [{huc_level}]: {n_reaches} reaches x {m} units "
          f"(outlet-only, {n_assigned} active reaches)")
    return B, huc_ids


# ============================================================
# 3. General LQR (arbitrary B_r)
# ============================================================

def lqr_gain_general(
    A: np.ndarray,
    B_r: np.ndarray,
    Q: np.ndarray,
    rho: float,
) -> np.ndarray:
    """
    General discrete-time LQR with arbitrary B_r.
    K = (R + B_r^T P B_r)^{-1} B_r^T P A
    """
    r = A.shape[0]
    m = B_r.shape[1]
    R = rho * np.eye(m, dtype=np.float64)

    P = solve_discrete_are(
        A.astype(np.float64),
        B_r.astype(np.float64),
        Q.astype(np.float64),
        R,
    )
    K = np.linalg.solve(R + B_r.T @ P @ B_r, B_r.T @ P @ A)
    return K.astype(DTYPE)


# ============================================================
# 4. General Closed-Loop Simulation (arbitrary B_r)
# ============================================================

def simulate_cl_flood_general(
    dmd: DMDcPack,
    z0: np.ndarray,
    V_sim: np.ndarray,
    K_list: List[np.ndarray],
    B_r: np.ndarray,
    thr_flood: np.ndarray,
    U_r: np.ndarray,
    mean_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalized closed-loop simulation with arbitrary B_r.
    Reduces to the standard (B=I) version when B_r = I_r.
    """
    r, T = z0.shape[0], V_sim.shape[1]
    m = K_list[0].shape[0]

    Z = np.zeros((r, T + 1), dtype=DTYPE)
    U_ctrl = np.zeros((r, T), dtype=DTYPE)
    Z[:, 0:1] = z0

    thr = thr_flood.reshape(-1, 1)

    for k in range(T):
        w = dmd.window_for_k[k]
        A, E, b, K = dmd.A_list[w], dmd.E_list[w], dmd.b_list[w], K_list[w]

        x_k = U_r @ Z[:, k:k+1] + mean_x
        x_excess = np.maximum(x_k - thr, 0.0)
        z_excess = U_r.T @ x_excess

        u_huc = -K @ z_excess
        u_z = B_r @ u_huc

        U_ctrl[:, k:k+1] = u_z
        Z[:, k+1:k+2] = A @ Z[:, k:k+1] + u_z + E @ V_sim[:, k:k+1] + b

    return Z, U_ctrl


# ============================================================
# 5. Single HUC-Level Run
# ============================================================

def run_huc_level(
    huc_level: str,
    B_r: np.ndarray,
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
    """Sweep R_weight values for a given B_r and return metrics."""
    if R_values is None:
        R_values = R_VALUES

    Q = (Q_WEIGHT * np.eye(RANK)).astype(DTYPE)
    results = []

    for Rw in R_values:
        try:
            K_list = [lqr_gain_general(A, B_r, Q, float(Rw)) for A in dmd.A_list]
        except Exception as e:
            print(f"      [{huc_level}] R={Rw:.2e}: DARE failed ({e})")
            continue

        Z_cl, U_cl = simulate_cl_flood_general(
            dmd, z0, V_sim, K_list, B_r,
            thr_flood, pod.U_r, pod.mean_x,
        )

        m = compute_controlled_metrics(thr_flood, pod, pre, Z_cl, U_cl, vol_hi_ref)

        effort_gm3 = np.nansum(m['eff_total_L1']) / 1e9
        residual_gm3 = np.nansum(m['vol_hi_controlled']) / 1e9
        baseline_gm3 = np.nansum(m['vol_hi_baseline']) / 1e9
        red_frac = (baseline_gm3 - residual_gm3) / (baseline_gm3 + 1e-12)

        print(f"      [{huc_level:>5s}] R={Rw:>10.2e}: "
              f"effort={effort_gm3:>6.1f} Gm3, "
              f"residual={residual_gm3:>6.1f} Gm3, "
              f"reduction={red_frac:>5.1%}")

        df_r = pd.DataFrame(m)
        df_r["huc_level"] = huc_level
        df_r["n_ctrl_units"] = B_r.shape[1]
        df_r["R_weight"] = float(Rw)
        df_r["reach_id"] = matched
        df_r["gcm"] = gcm
        df_r["hydro"] = hydro
        df_r["scenario"] = int(scn)

        results.append(df_r)

        del Z_cl, U_cl, K_list, m
        gc.collect()

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ============================================================
# 6. Main Comparison Pipeline
# ============================================================

def run_huc_comparison(
    gcm: str,
    hydro: str,
    scn: int,
    huc_levels: List[str] = ['reach', 'HUC12', 'HUC10', 'HUC8'],
    mapping_csv: str = MAPPING_CSV,
    out_dir: str = "outputs_huc_comparison",
):
    """Compare control across multiple HUC resolutions for one (GCM, hydro, scenario)."""
    print(f"\n{'='*70}")
    print(f"HUC COMPARISON: {hydro}/{gcm}/SSP{scn}")
    print(f"  Levels: {huc_levels}")
    print(f"  Mapping: {mapping_csv}")
    print(f"{'='*70}")

    # 1. Reference pack
    print("\n[1] Reference pack ...")
    ref = compute_reference_pack(gcm, hydro, verbose=True)

    # 2. Load data
    print("\n[2] Loading scenario data ...")
    X, U, dates, reach_ids = load_scenario_data(scn, gcm, hydro)
    idx_ref, idx_this, matched = match_to_reference(ref.reach_ids, reach_ids)
    X = X[idx_this]
    U = U[idx_this]
    thr_flood = ref.thr_flood[idx_ref]
    vol_hi_ref = ref.vol_hi_ref[idx_ref]
    n_reaches = X.shape[0]

    # 3. POD + DMDc (once)
    print("\n[3] POD + DMDc ...")
    pod = compute_pod(X, RANK)
    V = normalize_forcing(U).V
    dmd = fit_windowed_dmdc(pod.Z, V)

    z0, V_sim = pod.Z[:, 0:1], V[:, :-1]
    Z_ol = simulate_openloop(dmd, z0, V_sim)
    Z_anch = simulate_one_step_anchored(dmd, pod.Z, V_sim)
    pre = precompute_baseline(X, thr_flood, dates, pod, Z_ol, Z_anch)

    # 4. HUC mapping
    print("\n[4] Building HUC mapping ...")
    comid_array = np.array(matched, dtype=int)
    huc_mapping = build_huc_mapping(comid_array, mapping_csv)

    for level in ['HUC8', 'HUC10', 'HUC12']:
        n_units = huc_mapping[level].nunique()
        n_missing = huc_mapping[level].isna().sum()
        print(f"    {level}: {n_units} units, {n_missing} missing")

    # 5. Run each HUC level
    all_results = []

    for huc_level in huc_levels:
        print(f"\n{'─'*50}")
        print(f"  Processing: {huc_level}")
        print(f"{'─'*50}")

        if huc_level == 'reach':
            B_r = np.eye(RANK, dtype=DTYPE)
        else:
            B_phys, huc_ids = build_B_matrix(huc_mapping, huc_level, n_reaches)
            B_r = (pod.U_r.T @ B_phys).astype(DTYPE)
            print(f"  B_r shape: {B_r.shape}")

        df = run_huc_level(
            huc_level=huc_level,
            B_r=B_r,
            gcm=gcm, scn=scn, hydro=hydro, ref=ref,
            pod=pod, dmd=dmd, pre=pre,
            V_sim=V_sim, z0=z0,
            thr_flood=thr_flood, vol_hi_ref=vol_hi_ref,
            matched=matched,
        )
        if len(df):
            all_results.append(df)

    # 6. Save
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        fname = f"huc_comparison_{hydro}_{gcm}_ssp{scn}.csv"
        df_all.to_csv(out_path / fname, index=False)
        print(f"\n  Saved: {out_path / fname} ({len(df_all)} rows)")

        # Summary table
        print(f"\n{'='*70}")
        print(f"  SUMMARY: Effort-Residual at R_w grid")
        print(f"{'='*70}")
        summary = (
            df_all.groupby(['huc_level', 'R_weight'])
            .agg(
                effort_Gm3=('eff_total_L1', lambda x: x.sum() / 1e9),
                residual_Gm3=('vol_hi_controlled', lambda x: x.sum() / 1e9),
                baseline_Gm3=('vol_hi_baseline', lambda x: x.sum() / 1e9),
                median_peak_red=('peak_red_rel', 'median'),
            )
            .reset_index()
        )
        summary['reduction_frac'] = (
            (summary['baseline_Gm3'] - summary['residual_Gm3'])
            / (summary['baseline_Gm3'] + 1e-12)
        )
        print(summary.to_string(index=False))

        summary.to_csv(out_path / f"huc_summary_{hydro}_{gcm}_ssp{scn}.csv", index=False)

        # Spatial compact: per-reach peak reduction at each HUC level
        R_mid = sorted(df_all['R_weight'].unique())[len(df_all['R_weight'].unique())//2]
        df_mid = df_all[df_all['R_weight'] == R_mid].copy()

        spatial_rows = []
        for rid in df_mid[df_mid['huc_level'] == 'reach']['reach_id'].unique():
            row = {'reach_id': int(rid)}
            for lv in huc_levels:
                sub = df_mid[(df_mid['huc_level'] == lv) & (df_mid['reach_id'] == rid)]
                if len(sub) > 0:
                    row[f'peak_red_{lv}'] = float(sub['peak_red_rel'].iloc[0])
            spatial_rows.append(row)

        if spatial_rows:
            df_spatial = pd.DataFrame(spatial_rows)
            spatial_fname = f"huc_spatial_compact_{hydro}_{gcm}_ssp{scn}.csv"
            df_spatial.to_csv(out_path / spatial_fname, index=False)
            print(f"  Saved spatial compact: {out_path / spatial_fname}")

    del X, U, V, Z_ol, Z_anch, pod, dmd, pre
    gc.collect()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HUC-level control constraint comparison")
    parser.add_argument("--gcm", default="MPI-ESM1-2-HR")
    parser.add_argument("--hydro", default="VIC5")
    parser.add_argument("--scenario", type=int, default=245)
    parser.add_argument("--levels", nargs='+', default=['reach', 'HUC12', 'HUC10', 'HUC8'])
    parser.add_argument("--mapping", default=MAPPING_CSV)
    parser.add_argument("--outdir", default="outputs_huc_comparison")
    args = parser.parse_args()

    run_huc_comparison(
        gcm=args.gcm,
        hydro=args.hydro,
        scn=args.scenario,
        huc_levels=args.levels,
        mapping_csv=args.mapping,
        out_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
