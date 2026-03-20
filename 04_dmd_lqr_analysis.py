#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMDc + LQR Flood Control Analysis [Dual Hydro: VIC5 + PRMS]
-------------------------------------------------------------

Runs both VIC5 and PRMS hydrological models for each GCM, producing
14 ensemble members (7 GCMs × 2 hydro models) instead of 7.

Outputs (per GCM × hydro):
- results_{hydro}_{GCM}.parquet: reach-level metrics
- daily_{hydro}_{GCM}.parquet: daily basin time series
- fdc_{hydro}_{GCM}.parquet: flow duration curve statistics

Usage:
    python 04_dmd_lqr_analysis.py --stage run --gcm EC-Earth3 --hydro all --rl 2
    python 04_dmd_lqr_analysis.py --stage all --gcm all --hydro PRMS --rl 2
    python 04_dmd_lqr_analysis.py --stage all --gcm all --hydro all --rl 2
"""

import os
import gc
import argparse
import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from scipy.linalg import solve_discrete_are

warnings.filterwarnings("ignore")


# ============================================================
# PROFILING
# ============================================================

class Profiler:
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.current_start: Dict[str, float] = {}

    def start(self, name: str):
        self.current_start[name] = time.time()

    def stop(self, name: str):
        if name in self.current_start:
            elapsed = time.time() - self.current_start[name]
            self.times[name] = self.times.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
            del self.current_start[name]
            return elapsed
        return 0.0

    def report(self, title: str = "PROFILING REPORT"):
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
        total = sum(self.times.values())
        for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
            count = self.counts.get(name, 1)
            pct = 100 * t / total if total > 0 else 0
            print(f"  {name:<35} {t:>8.1f}s ({pct:>5.1f}%)  [n={count}]")
        print(f"  {'─'*55}")
        print(f"  {'TOTAL':<35} {total:>8.1f}s")
        print(f"{'='*60}\n")


PROF = Profiler()


# ============================================================
# CONFIG
# ============================================================

# SSP scenarios (radiative forcing in W/m²)
SCENARIOS = [126, 245, 370, 585]
REFERENCE_SCN = 245  # Baseline comparison scenario (SSP2-4.5)
AVAILABLE_GCMS = ["MPI-ESM1-2-HR", "ACCESS-CM2", "EC-Earth3", "CNRM-ESM2-1", "BCC-CSM2-MR", "MRI-ESM2-0", "NorESM2-MM"]

# Dual hydrological models for ensemble analysis
HYDRO_MODELS = ["PRMS", "VIC5"]

HISTORICAL_YEARS = (1980, 2019)  # Historical baseline period
ANALYSIS_ERA = (2020, 2099)      # Future projection period

DT = 86400.0  # Seconds per day (time step)

# Flood threshold return period in years (set by CLI: --rl)
FLOOD_RL_PERIOD = 2
RETURN_PERIODS = [2, 5, 10, 20, 50, 100]  # Years

# DMDc settings
RANK = 30            # Truncated SVD rank for POD basis
WINDOW = 365 * 10    # Sliding window size (days): 10 years
STEP = 365 * 5       # Sliding window step (days): 5 years

ALPHA_CANDIDATES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10, 25, 50]  # Ridge regularization candidates
SPECTRAL_RADIUS_LIMIT = 0.995  # Stability threshold for discrete-time LQR

DTYPE = np.float64

# LQR control weight grid: 19 log-spaced values from 1e-4 to 1e5
R_VALUES = np.logspace(-4, 5, 19).tolist()

Q_WEIGHT = 1.0       # State penalty weight in LQR cost function
CHUNK = 384           # Block size for batched matrix operations
USE_TRUNCATED_SVD = True

# Output and cache directories
OUT_DIR = "outputs_flood_rl2"
CACHE_DIR = Path("cache_pooled_return_levels")
CACHE_DIR.mkdir(exist_ok=True)

# Target water-release budgets for daily output (Gm³/yr)
TARGET_BUDGETS = [1, 3, 5, 10, 20]


# ============================================================
# DATA LOADING
# ============================================================

def _realization_id(model: str) -> str:
    return {
        "ACCESS-CM2": "r1i1p1f1",
        "CNRM-ESM2-1": "r1i1p1f2",
        "EC-Earth3": "r1i1p1f1",
        "MPI-ESM1-2-HR": "r1i1p1f1",
        "BCC-CSM2-MR": "r1i1p1f1",
        "MRI-ESM2-0": "r1i1p1f1",
        "NorESM2-MM": "r1i1p1f1",
    }[model]


def _streamflow_prefix(hydro: str) -> str:
    """Map hydro model name to file prefix: VIC5 -> VIC5_RAPID, PRMS -> PRMS_RAPID."""
    return f"{hydro}_RAPID"


def load_historical_data(model: str, scenario: int, hydro: str) -> Tuple[np.ndarray, pd.DatetimeIndex, List[int]]:
    PROF.start("load_historical")
    real_id = _realization_id(model)
    prefix = _streamflow_prefix(hydro)
    path = f"streamflow/{prefix}_{model}_ssp{scenario}_{real_id}_DBCCA_Daymet_{HISTORICAL_YEARS[0]}_{HISTORICAL_YEARS[1]}_daily.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df = df.loc[f"{HISTORICAL_YEARS[0]}-01-01":f"{HISTORICAL_YEARS[1]}-12-31"] / 35.315
    df.columns = df.columns.astype(float).astype(int)
    X = df.values.T.astype(DTYPE, copy=False)
    dates = pd.DatetimeIndex(df.index)
    reach_ids = df.columns.to_list()
    PROF.stop("load_historical")
    return X, dates, reach_ids


def load_scenario_data(
    scenario: int, model: str, hydro: str,
    start_year: int = 2020, end_year: int = 2099,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[int]]:
    PROF.start("load_data")
    real_id = _realization_id(model)
    prefix = _streamflow_prefix(hydro)

    flow_files = []
    if start_year <= 2059:
        flow_files.append(f"streamflow/{prefix}_{model}_ssp{scenario}_{real_id}_DBCCA_Daymet_2020_2059_daily.parquet")
    if end_year >= 2060:
        flow_files.append(f"streamflow/{prefix}_{model}_ssp{scenario}_{real_id}_DBCCA_Daymet_2060_2099_daily.parquet")
    df_flow = pd.concat([pd.read_parquet(f) for f in flow_files]).sort_index()
    df_flow = df_flow.loc[f"{start_year}-01-01":f"{end_year}-12-31"] / 35.315

    # Precipitation files are hydro-model-agnostic (no VIC5/PRMS prefix)
    prcp_files = []
    if start_year <= 2059:
        prcp_files.append(f"precipitation/{model}_ssp{scenario}_{real_id}_DBCCA_Daymet_prcp_2020_2059_daily.parquet")
    if end_year >= 2060:
        prcp_files.append(f"precipitation/{model}_ssp{scenario}_{real_id}_DBCCA_Daymet_prcp_2060_2099_daily.parquet")
    df_prcp = pd.concat([pd.read_parquet(f) for f in prcp_files]).sort_index()
    df_prcp = df_prcp.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

    df_flow.columns = df_flow.columns.astype(float).astype(int)
    df_prcp.columns = df_prcp.columns.astype(float).astype(int)
    df_prcp = df_prcp.reindex(columns=df_flow.columns, fill_value=0.0)

    idx = df_flow.index.intersection(df_prcp.index)
    df_flow = df_flow.loc[idx].fillna(0.0)
    df_prcp = df_prcp.loc[idx].fillna(0.0)

    X = df_flow.values.T.astype(DTYPE, copy=False)
    U = df_prcp.values.T.astype(DTYPE, copy=False)
    dates = pd.DatetimeIndex(idx)
    reach_ids = df_flow.columns.to_list()

    PROF.stop("load_data")
    return X, U, dates, reach_ids


def match_to_reference(reach_ids_ref: List[int], reach_ids_this: List[int]):
    pos_this = {rid: i for i, rid in enumerate(reach_ids_this)}
    idx_ref, idx_this, matched = [], [], []
    for i_ref, rid in enumerate(reach_ids_ref):
        if rid in pos_this:
            idx_ref.append(i_ref)
            idx_this.append(pos_this[rid])
            matched.append(rid)
    return np.asarray(idx_ref, dtype=int), np.asarray(idx_this, dtype=int), matched


# ============================================================
# POOLED AMS THRESHOLD
# ============================================================

def _get_annual_max(X: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    years = dates.year.values
    unique_years = np.unique(years)
    return np.array([X[:, years == y].max(axis=1) for y in unique_years]).T


def _cache_path_for_pooled(gcm: str, hydro: str) -> Path:
    return CACHE_DIR / f"pooled_return_levels_{hydro}_{gcm}_{HISTORICAL_YEARS[0]}-{HISTORICAL_YEARS[1]}_empirical.npz"


def compute_pooled_return_levels(gcm: str, hydro: str, verbose: bool = True, use_cache: bool = True) -> Dict[str, np.ndarray]:
    PROF.start("pooled_ams")
    cache_path = _cache_path_for_pooled(gcm, hydro)
    if use_cache and cache_path.exists():
        if verbose:
            print(f"  [cache] Loading pooled return levels: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        out = {k: data[k] for k in data.files}
        out["reach_ids"] = out["reach_ids"].astype(int)
        PROF.stop("pooled_ams")
        return out

    if verbose:
        print(f"  Computing pooled AMS threshold ({hydro})...")

    ams_list = []
    reach_ids: Optional[List[int]] = None
    used_ssps = []

    for ssp in SCENARIOS:
        try:
            X, dates, rids = load_historical_data(gcm, ssp, hydro)
        except FileNotFoundError:
            continue

        if reach_ids is None:
            reach_ids = rids
        else:
            idx_ref, idx_this, _ = match_to_reference(reach_ids, rids)
            ams_list = [ams[idx_ref, :] for ams in ams_list]
            X = X[idx_this, :]
            reach_ids = [reach_ids[i] for i in idx_ref]

        ams = _get_annual_max(X, dates)
        ams_list.append(ams)
        used_ssps.append(ssp)
        del X, ams
        gc.collect()

    if reach_ids is None or not ams_list:
        raise RuntimeError(f"No historical SSP files found for {hydro}/{gcm}.")

    ams_pooled = np.concatenate(ams_list, axis=1)
    n_reaches, n_samples_total = ams_pooled.shape

    out: Dict[str, np.ndarray] = {
        "reach_ids": np.asarray(reach_ids, dtype=int),
        "sample_n": np.full(n_reaches, n_samples_total, dtype=np.int32),
    }

    for T in RETURN_PERIODS:
        p = 1.0 - 1.0 / float(T)
        rl = np.nanpercentile(ams_pooled, 100 * p, axis=1).astype(DTYPE)
        rl = np.maximum(rl, 0.0)
        out[f"rl{T}"] = rl

    if use_cache:
        np.savez_compressed(cache_path, **out)

    del ams_pooled, ams_list
    gc.collect()
    PROF.stop("pooled_ams")
    return out


# ============================================================
# REFERENCE PACK
# ============================================================

@dataclass
class ReferencePack:
    reach_ids: List[int]
    thr_flood: np.ndarray
    vol_hi_ref: np.ndarray
    mean_flow: np.ndarray
    return_levels: Dict[str, np.ndarray]


def compute_reference_pack(gcm: str, hydro: str, verbose: bool = True) -> ReferencePack:
    PROF.start("reference_pack")

    rl = compute_pooled_return_levels(gcm, hydro, verbose=verbose, use_cache=True)
    reach_ids_pooled = rl["reach_ids"].astype(int).tolist()
    n_all = len(reach_ids_pooled)

    thr_flood = rl[f"rl{FLOOD_RL_PERIOD}"].astype(DTYPE)
    thr_flood = np.maximum(thr_flood, 0.0)

    return_levels: Dict[str, np.ndarray] = {}
    for T in RETURN_PERIODS:
        return_levels[f"rl{T}"] = np.maximum(rl[f"rl{T}"].astype(DTYPE), 0.0)
    return_levels["sample_n"] = rl["sample_n"].astype(np.int32)

    X_ref, _, dates_ref, reach_ids_ref = load_scenario_data(REFERENCE_SCN, gcm, hydro)
    idx_pool, idx_ref, _ = match_to_reference(reach_ids_pooled, reach_ids_ref)
    X_ref_m = X_ref[idx_ref, :]

    mask_bl = np.asarray((dates_ref.year >= 2020) & (dates_ref.year <= 2039), dtype=bool)
    mean_flow_m = np.nanmean(X_ref_m[:, mask_bl], axis=1).astype(DTYPE)

    mask_era = np.asarray((dates_ref.year >= ANALYSIS_ERA[0]) & (dates_ref.year <= ANALYSIS_ERA[1]), dtype=bool)
    n_years = max(len(np.unique(dates_ref[mask_era].year)), 1)

    thr_m = thr_flood[idx_pool].reshape(-1, 1)
    exceed = np.maximum(X_ref_m[:, mask_era] - thr_m, 0.0)
    vol_hi_ref_m = (np.nansum(exceed, axis=1) * DT / n_years).astype(DTYPE)

    vol_hi_ref_full = np.full(n_all, np.nan, dtype=DTYPE)
    mean_flow_full = np.full(n_all, np.nan, dtype=DTYPE)
    vol_hi_ref_full[idx_pool] = vol_hi_ref_m
    mean_flow_full[idx_pool] = mean_flow_m

    del X_ref, X_ref_m, exceed
    gc.collect()
    PROF.stop("reference_pack")

    return ReferencePack(
        reach_ids=reach_ids_pooled,
        thr_flood=thr_flood,
        vol_hi_ref=vol_hi_ref_full,
        mean_flow=mean_flow_full,
        return_levels=return_levels,
    )


# ============================================================
# POD
# ============================================================

@dataclass
class PODPack:
    mean_x: np.ndarray
    U_r: np.ndarray
    Z: np.ndarray


def compute_pod(X: np.ndarray, rank: int = RANK) -> PODPack:
    if USE_TRUNCATED_SVD:
        from sklearn.decomposition import TruncatedSVD
        PROF.start("pod_truncated_svd")
        mean_x = X.mean(axis=1, keepdims=True).astype(DTYPE)
        Xc = X - mean_x
        svd = TruncatedSVD(n_components=rank, algorithm="randomized", n_iter=5, random_state=42)
        Z = svd.fit_transform(Xc.T).T.astype(DTYPE)
        U_r = svd.components_.T.astype(DTYPE)
        PROF.stop("pod_truncated_svd")
    else:
        PROF.start("pod_full_svd")
        mean_x = X.mean(axis=1, keepdims=True).astype(DTYPE)
        Xc = X - mean_x
        U, _, _ = np.linalg.svd(Xc, full_matrices=False)
        U_r = U[:, :rank].astype(DTYPE)
        Z = (U_r.T @ Xc).astype(DTYPE)
        PROF.stop("pod_full_svd")
    return PODPack(mean_x=mean_x, U_r=U_r, Z=Z)


# ============================================================
# FORCING NORMALIZATION
# ============================================================

@dataclass
class ForcingPack:
    V: np.ndarray
    mu: Optional[np.ndarray] = None
    global_sigma: Optional[float] = None


def normalize_forcing(U: np.ndarray) -> ForcingPack:
    mu = U.mean(axis=1, keepdims=True).astype(DTYPE)
    V = U - mu
    gsig = float(np.std(V) + 1e-12)
    return ForcingPack(V=(V / gsig).astype(DTYPE), mu=mu, global_sigma=gsig)


# ============================================================
# DMDc
# ============================================================

@dataclass
class DMDcPack:
    A_list: List[np.ndarray]
    E_list: List[np.ndarray]
    b_list: List[np.ndarray]
    window_for_k: np.ndarray
    bounds: List[Tuple[int, int]]
    spectral_radii: List[float]
    alphas_used: List[float]


def fit_windowed_dmdc(Z: np.ndarray, V: np.ndarray) -> DMDcPack:
    PROF.start("dmdc_fit")

    Zk, Znext, Vk = Z[:, :-1], Z[:, 1:], V[:, :-1]
    r, Tm1 = Zk.shape

    A_list, E_list, b_list, bounds = [], [], [], []
    alphas_used, spectral_radii = [], []

    def fit_with_lowest_stable_alpha(Omega, Y):
        best_A, best_E, best_b, best_alpha, best_spec_rad = None, None, None, None, None
        for alpha_val in ALPHA_CANDIDATES:
            reg = Ridge(alpha=alpha_val, fit_intercept=False, solver="auto")
            reg.fit(Omega.T, Y.T)
            A = reg.coef_[:, :r]
            spec_rad = np.max(np.abs(np.linalg.eigvals(A)))
            if spec_rad < SPECTRAL_RADIUS_LIMIT:
                E = reg.coef_[:, r:]
                b = np.zeros((r, 1), dtype=DTYPE)
                return A.astype(DTYPE), E.astype(DTYPE), b, alpha_val, float(spec_rad)
            best_A, best_E = A, reg.coef_[:, r:]
            best_b = np.zeros((r, 1), dtype=DTYPE)
            best_alpha, best_spec_rad = alpha_val, float(spec_rad)
        return best_A.astype(DTYPE), best_E.astype(DTYPE), best_b, best_alpha, best_spec_rad

    for t0 in range(0, max(1, Tm1 - WINDOW), STEP):
        t1 = min(t0 + WINDOW, Tm1)
        Omega = np.vstack([Zk[:, t0:t1], Vk[:, t0:t1]])
        A, E, b, used_alpha, spec_rad = fit_with_lowest_stable_alpha(Omega, Znext[:, t0:t1])
        A_list.append(A)
        E_list.append(E)
        b_list.append(b)
        bounds.append((t0, t1))
        alphas_used.append(used_alpha)
        spectral_radii.append(spec_rad)

    if bounds and bounds[-1][1] < Tm1:
        t0, t1 = max(0, Tm1 - WINDOW), Tm1
        Omega = np.vstack([Zk[:, t0:t1], Vk[:, t0:t1]])
        A, E, b, used_alpha, spec_rad = fit_with_lowest_stable_alpha(Omega, Znext[:, t0:t1])
        A_list.append(A)
        E_list.append(E)
        b_list.append(b)
        bounds.append((t0, t1))
        alphas_used.append(used_alpha)
        spectral_radii.append(spec_rad)

    window_for_k = np.zeros(Tm1, dtype=int)
    for w, (a, bnd) in enumerate(bounds):
        window_for_k[a:bnd] = w

    PROF.stop("dmdc_fit")
    return DMDcPack(
        A_list=A_list, E_list=E_list, b_list=b_list,
        window_for_k=window_for_k, bounds=bounds,
        spectral_radii=spectral_radii, alphas_used=alphas_used,
    )


def simulate_openloop(dmd: DMDcPack, z0: np.ndarray, V_sim: np.ndarray) -> np.ndarray:
    PROF.start("sim_openloop")
    r, T = z0.shape[0], V_sim.shape[1]
    Zsim = np.zeros((r, T + 1), dtype=DTYPE)
    Zsim[:, 0:1] = z0
    for k in range(T):
        w = dmd.window_for_k[k]
        Zsim[:, k+1:k+2] = dmd.A_list[w] @ Zsim[:, k:k+1] + dmd.E_list[w] @ V_sim[:, k:k+1] + dmd.b_list[w]
    PROF.stop("sim_openloop")
    return Zsim


def simulate_one_step_anchored(dmd: DMDcPack, Z_true: np.ndarray, V_sim: np.ndarray) -> np.ndarray:
    PROF.start("sim_anchored")
    r, T = Z_true.shape[0], min(V_sim.shape[1], Z_true.shape[1] - 1)
    Zpred = np.zeros((r, T + 1), dtype=DTYPE)
    Zpred[:, 0:1] = Z_true[:, 0:1]
    for k in range(T):
        w = dmd.window_for_k[k]
        Zpred[:, k+1:k+2] = dmd.A_list[w] @ Z_true[:, k:k+1] + dmd.E_list[w] @ V_sim[:, k:k+1] + dmd.b_list[w]
    PROF.stop("sim_anchored")
    return Zpred


# ============================================================
# LQR
# ============================================================

def lqr_gain(A: np.ndarray, Q: np.ndarray, rho: float) -> np.ndarray:
    r = A.shape[0]
    I = np.eye(r)
    P = solve_discrete_are(A.astype(np.float64), I, Q.astype(np.float64), rho * I)
    return np.linalg.solve(rho * I + P, P @ A).astype(DTYPE)


# ============================================================
# FLOOD-ONLY CONTROL SIMULATION
# ============================================================

def simulate_cl_flood_only(
    dmd: DMDcPack,
    z0: np.ndarray,
    V_sim: np.ndarray,
    K_list: List[np.ndarray],
    thr_flood: np.ndarray,
    U_r: np.ndarray,
    mean_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    PROF.start("sim_cl_flood")

    r, T = z0.shape[0], V_sim.shape[1]
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
        u = -K @ z_excess

        U_ctrl[:, k:k+1] = u
        Z[:, k+1:k+2] = A @ Z[:, k:k+1] + u + E @ V_sim[:, k:k+1] + b

    PROF.stop("sim_cl_flood")
    return Z, U_ctrl


# ============================================================
# DAILY BASIN OUTPUT
# ============================================================

def compute_daily_basin_output(
    pod: PODPack,
    Z_baseline: np.ndarray,
    Z_controlled: np.ndarray,
    U_ctrl: np.ndarray,
    thr_flood: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    PROF.start("daily_basin")

    T = min(Z_baseline.shape[1], Z_controlled.shape[1], U_ctrl.shape[1], len(dates))
    thr = thr_flood.reshape(-1, 1)

    daily_data = {
        "date": [],
        "basin_exceed_baseline": [],
        "basin_exceed_controlled": [],
        "basin_effort": [],
        "basin_peak_baseline": [],
        "basin_peak_controlled": [],
    }

    for t in range(T):
        X_bl = np.maximum(pod.U_r @ Z_baseline[:, t:t+1] + pod.mean_x, 0.0)
        X_cl = np.maximum(pod.U_r @ Z_controlled[:, t:t+1] + pod.mean_x, 0.0)

        exceed_bl = np.maximum(X_bl - thr, 0.0).sum() * DT
        exceed_cl = np.maximum(X_cl - thr, 0.0).sum() * DT

        u_phys = pod.U_r @ U_ctrl[:, t:t+1]
        effort = np.abs(u_phys).sum()

        daily_data["date"].append(dates[t])
        daily_data["basin_exceed_baseline"].append(exceed_bl)
        daily_data["basin_exceed_controlled"].append(exceed_cl)
        daily_data["basin_effort"].append(effort)
        daily_data["basin_peak_baseline"].append(X_bl.max())
        daily_data["basin_peak_controlled"].append(X_cl.max())

    PROF.stop("daily_basin")
    return pd.DataFrame(daily_data)


# ============================================================
# FLOW DURATION CURVE
# ============================================================

def compute_fdc(
    pod: PODPack,
    Z_baseline: np.ndarray,
    Z_controlled: np.ndarray,
    percentiles: List[float] = None,
) -> Dict[str, float]:
    PROF.start("fdc")

    if percentiles is None:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    T = min(Z_baseline.shape[1], Z_controlled.shape[1])

    basin_bl = np.zeros(T)
    basin_cl = np.zeros(T)

    for t in range(T):
        X_bl = np.maximum(pod.U_r @ Z_baseline[:, t:t+1] + pod.mean_x, 0.0)
        X_cl = np.maximum(pod.U_r @ Z_controlled[:, t:t+1] + pod.mean_x, 0.0)
        basin_bl[t] = X_bl.sum()
        basin_cl[t] = X_cl.sum()

    fdc = {}
    for p in percentiles:
        fdc[f"fdc_{p}pct_baseline"] = np.percentile(basin_bl, 100 - p)
        fdc[f"fdc_{p}pct_controlled"] = np.percentile(basin_cl, 100 - p)

    PROF.stop("fdc")
    return fdc


# ============================================================
# METRICS HELPERS
# ============================================================

def _analysis_mask_and_year_groups(dates: pd.DatetimeIndex, T_ctrl: int):
    dates_ctrl = pd.DatetimeIndex(dates[:T_ctrl])
    mask = np.asarray(
        (dates_ctrl.year >= ANALYSIS_ERA[0]) & (dates_ctrl.year <= ANALYSIS_ERA[1]),
        dtype=bool
    )
    dates_era = dates_ctrl[mask]
    years_era = dates_era.year.values
    unique_years = np.unique(years_era)
    idx_by_year = [np.where(years_era == y)[0] for y in unique_years]
    return mask, dates_era, idx_by_year, max(len(unique_years), 1)


def _nse(obs, sim):
    mean_obs = obs.mean(axis=1, keepdims=True)
    return np.clip(
        1 - np.sum((obs - sim) ** 2, axis=1) / (np.sum((obs - mean_obs) ** 2, axis=1) + 1e-12),
        -10, 1
    ).astype(DTYPE)


def _kge(obs, sim):
    """Kling-Gupta Efficiency (2009 formulation), per reach."""
    mu_o = obs.mean(axis=1)
    mu_s = sim.mean(axis=1)
    sigma_o = obs.std(axis=1) + 1e-12
    sigma_s = sim.std(axis=1) + 1e-12
    # Pearson r per reach
    cov = ((obs - mu_o[:, None]) * (sim - mu_s[:, None])).mean(axis=1)
    r = cov / (sigma_o * sigma_s)
    alpha = sigma_s / sigma_o
    beta = mu_s / (mu_o + 1e-12)
    return np.clip(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), -10, 1).astype(DTYPE)


def _r2(obs, sim):
    """Coefficient of determination (R²), per reach."""
    mu_o = obs.mean(axis=1, keepdims=True)
    mu_s = sim.mean(axis=1, keepdims=True)
    ss_res = np.sum((obs - sim) ** 2, axis=1)
    ss_tot = np.sum((obs - mu_o) ** 2, axis=1) + 1e-12
    # Signed R² from correlation
    cov = np.sum((obs - mu_o) * (sim - mu_s), axis=1)
    r = cov / (np.sqrt(ss_tot * (np.sum((sim - mu_s) ** 2, axis=1) + 1e-12)))
    return np.clip(np.sign(r) * r ** 2, -1, 1).astype(DTYPE)


def _pbias(obs, sim):
    return (np.sum(sim - obs, axis=1) / (np.sum(obs, axis=1) + 1e-12)).astype(DTYPE)


def _peak_bias(obs, sim):
    """Relative bias of annual maxima (peak flow), per reach."""
    peak_o = obs.max(axis=1) + 1e-12
    peak_s = sim.max(axis=1)
    return ((peak_s - peak_o) / peak_o).astype(DTYPE)


def _ams_rl(amax, return_periods):
    return {
        f"rl{T}": np.percentile(amax, 100 * (1 - 1 / T), axis=1).astype(DTYPE)
        for T in return_periods
    }


def _count_events(bool_arr: np.ndarray, n_years: int) -> np.ndarray:
    if bool_arr.shape[1] == 0:
        return np.zeros(bool_arr.shape[0], dtype=DTYPE)
    prev = np.zeros((bool_arr.shape[0], 1), dtype=bool)
    shifted = np.concatenate([prev, bool_arr[:, :-1]], axis=1)
    starts = bool_arr & ~shifted
    return (starts.sum(axis=1) / n_years).astype(DTYPE)


# ============================================================
# PRECOMPUTE BASELINE
# ============================================================

@dataclass
class PrecompPack:
    T_ctrl: int
    mask_t: np.ndarray
    idx_by_year: List[np.ndarray]
    n_years: int

    # Open-loop skill metrics
    nse_openloop: np.ndarray
    nse_anchored: np.ndarray
    kge_openloop: np.ndarray
    kge_anchored: np.ndarray
    r2_openloop: np.ndarray
    r2_anchored: np.ndarray
    pbias_openloop: np.ndarray
    peak_bias_openloop: np.ndarray

    # Extreme-event bias (open-loop vs truth)
    rl_bias_openloop: Dict[str, np.ndarray]   # e.g. {"rl2": ..., "rl5": ...}
    vol_hi_bias_openloop: np.ndarray           # relative bias of exceedance volume

    rl10_bias_openloop: np.ndarray
    rl10_true: np.ndarray
    rl5_true: np.ndarray

    # Baseline flood metrics
    vol_hi_bl: np.ndarray
    days_hi_bl: np.ndarray
    peak_bl: np.ndarray
    events_bl: np.ndarray
    rl_bl: Dict[str, np.ndarray]


def precompute_baseline(X_true, thr_flood, dates, pod, Z_ol, Z_anch) -> PrecompPack:
    PROF.start("precompute_baseline")

    n, T_full = X_true.shape
    T_ctrl = min(T_full - 1, Z_ol.shape[1] - 1, Z_anch.shape[1] - 1)
    mask_t, _, idx_by_year, n_years = _analysis_mask_and_year_groups(dates, T_ctrl)

    Z_ol_use = Z_ol[:, :T_ctrl][:, mask_t]
    Z_an_use = Z_anch[:, :T_ctrl][:, mask_t]

    # Skill metrics
    nse_ol = np.full(n, np.nan, DTYPE)
    nse_an = np.full(n, np.nan, DTYPE)
    kge_ol = np.full(n, np.nan, DTYPE)
    kge_an = np.full(n, np.nan, DTYPE)
    r2_ol = np.full(n, np.nan, DTYPE)
    r2_an = np.full(n, np.nan, DTYPE)
    pbias_ol = np.full(n, np.nan, DTYPE)
    peak_bias_ol = np.full(n, np.nan, DTYPE)

    # Per-RL bias
    rl_bias_ol = {f"rl{T}": np.full(n, np.nan, DTYPE) for T in RETURN_PERIODS}
    vol_hi_bias_ol = np.full(n, np.nan, DTYPE)

    rl10_bias_ol = np.full(n, np.nan, DTYPE)
    rl10_true_arr = np.full(n, np.nan, DTYPE)
    rl5_true_arr = np.full(n, np.nan, DTYPE)

    vol_hi = np.zeros(n, DTYPE)
    vol_hi_true = np.zeros(n, DTYPE)
    days_hi = np.zeros(n, DTYPE)
    peak = np.zeros(n, DTYPE)
    events = np.zeros(n, DTYPE)
    rl_bl = {f"rl{T}": np.full(n, np.nan, DTYPE) for T in RETURN_PERIODS}

    thr = thr_flood.reshape(-1, 1)

    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        hi = thr[i0:i1]
        Uc, mc = pod.U_r[i0:i1, :], pod.mean_x[i0:i1, :]

        X_true_era = X_true[i0:i1, :T_ctrl][:, mask_t]
        X_ol = np.maximum(Uc @ Z_ol_use + mc, 0.0)
        X_an = np.maximum(Uc @ Z_an_use + mc, 0.0)

        # Full skill metrics
        nse_ol[i0:i1] = _nse(X_true_era, X_ol)
        nse_an[i0:i1] = _nse(X_true_era, X_an)
        kge_ol[i0:i1] = _kge(X_true_era, X_ol)
        kge_an[i0:i1] = _kge(X_true_era, X_an)
        r2_ol[i0:i1] = _r2(X_true_era, X_ol)
        r2_an[i0:i1] = _r2(X_true_era, X_an)
        pbias_ol[i0:i1] = _pbias(X_true_era, X_ol)
        peak_bias_ol[i0:i1] = _peak_bias(X_true_era, X_ol)

        # AMS and RL
        amax_true = np.array([
            X_true_era[:, idx].max(axis=1) if len(idx) > 0 else np.full(i1 - i0, np.nan)
            for idx in idx_by_year
        ]).T
        rl_true = _ams_rl(amax_true, RETURN_PERIODS)
        rl10_true_arr[i0:i1] = rl_true.get("rl10", np.full(i1 - i0, np.nan))
        rl5_true_arr[i0:i1] = rl_true.get("rl5", np.full(i1 - i0, np.nan))

        amax_ol = np.array([
            X_ol[:, idx].max(axis=1) if len(idx) > 0 else np.full(i1 - i0, np.nan)
            for idx in idx_by_year
        ]).T
        rl_ol = _ams_rl(amax_ol, RETURN_PERIODS)

        rl10_bias_ol[i0:i1] = (rl_ol["rl10"] - rl_true["rl10"]) / (rl_true["rl10"] + 1e-12)

        # Per-RL bias (open-loop vs truth)
        for T in RETURN_PERIODS:
            key = f"rl{T}"
            rl_bias_ol[key][i0:i1] = (rl_ol[key] - rl_true[key]) / (rl_true[key] + 1e-12)

        # Exceedance volume: true vs open-loop
        exceed_true = np.maximum(X_true_era - hi, 0.0)
        vol_hi_true[i0:i1] = exceed_true.sum(axis=1) * DT / n_years

        exceed = np.maximum(X_ol - hi, 0.0)
        above = X_ol > hi
        vol_hi[i0:i1] = exceed.sum(axis=1) * DT / n_years
        days_hi[i0:i1] = above.sum(axis=1) / n_years
        peak[i0:i1] = X_ol.max(axis=1)
        events[i0:i1] = _count_events(above, n_years)

        for key, arr in rl_ol.items():
            rl_bl[key][i0:i1] = arr

        del X_true_era, X_ol, X_an, exceed, exceed_true, above, amax_true, amax_ol
        gc.collect()

    # Volume exceedance bias: (open-loop - truth) / truth
    vol_hi_bias_ol = (vol_hi - vol_hi_true) / (vol_hi_true + 1e-12)

    PROF.stop("precompute_baseline")
    return PrecompPack(
        T_ctrl=T_ctrl, mask_t=mask_t, idx_by_year=idx_by_year, n_years=n_years,
        nse_openloop=nse_ol, nse_anchored=nse_an,
        kge_openloop=kge_ol, kge_anchored=kge_an,
        r2_openloop=r2_ol, r2_anchored=r2_an,
        pbias_openloop=pbias_ol, peak_bias_openloop=peak_bias_ol,
        rl_bias_openloop=rl_bias_ol, vol_hi_bias_openloop=vol_hi_bias_ol,
        rl10_bias_openloop=rl10_bias_ol, rl10_true=rl10_true_arr, rl5_true=rl5_true_arr,
        vol_hi_bl=vol_hi, days_hi_bl=days_hi, peak_bl=peak, events_bl=events, rl_bl=rl_bl,
    )


# ============================================================
# CONTROLLED METRICS
# ============================================================

def compute_controlled_metrics(thr_flood, pod, pre, Z_cl, U_cl, vol_hi_ref) -> Dict[str, np.ndarray]:
    PROF.start("compute_metrics")

    n = pod.U_r.shape[0]
    T_ctrl, mask_t, n_years, idx_by_year = pre.T_ctrl, pre.mask_t, pre.n_years, pre.idx_by_year

    Z_use = Z_cl[:, :T_ctrl][:, mask_t]
    Uuse = U_cl[:, :T_ctrl][:, mask_t]

    thr = thr_flood.reshape(-1, 1)

    vol_hi = np.zeros(n, DTYPE)
    days_hi = np.zeros(n, DTYPE)
    peak = np.zeros(n, DTYPE)
    events = np.zeros(n, DTYPE)
    rl_cl = {f"rl{T}": np.full(n, np.nan, DTYPE) for T in RETURN_PERIODS}

    eff_total = np.zeros(n, DTYPE)
    eff_flood = np.zeros(n, DTYPE)
    max_ctrl_cms = np.zeros(n, DTYPE)
    active_days = np.zeros(n, DTYPE)
    eff_atten = np.zeros(n, DTYPE)
    eff_augment = np.zeros(n, DTYPE)
    eff_atten_flood = np.zeros(n, DTYPE)
    eff_augment_flood = np.zeros(n, DTYPE)
    n_atten_days = np.zeros(n, DTYPE)
    n_augment_days = np.zeros(n, DTYPE)
    max_atten_cms = np.zeros(n, DTYPE)
    max_augment_cms = np.zeros(n, DTYPE)

    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        hi = thr[i0:i1]
        Uc, mc = pod.U_r[i0:i1, :], pod.mean_x[i0:i1, :]

        Xc = np.maximum(Uc @ Z_use + mc, 0.0)

        exceed = np.maximum(Xc - hi, 0.0)
        above = Xc > hi
        vol_hi[i0:i1] = exceed.sum(axis=1) * DT / n_years
        days_hi[i0:i1] = above.sum(axis=1) / n_years
        peak[i0:i1] = Xc.max(axis=1)
        events[i0:i1] = _count_events(above, n_years)

        amax = np.array([
            Xc[:, idx].max(axis=1) if len(idx) > 0 else np.full(i1 - i0, np.nan)
            for idx in idx_by_year
        ]).T
        rl_dict = _ams_rl(amax, RETURN_PERIODS)
        for key, arr in rl_dict.items():
            rl_cl[key][i0:i1] = arr

        Uphys = Uc @ Uuse
        eff_total[i0:i1] = np.abs(Uphys).sum(axis=1) * DT / n_years
        eff_flood[i0:i1] = (np.abs(Uphys) * above).sum(axis=1) * DT / n_years
        max_ctrl_cms[i0:i1] = np.abs(Uphys).max(axis=1)
        active_days[i0:i1] = (np.abs(Uphys) > 1e-6).sum(axis=1) / n_years
        u_neg = np.minimum(Uphys, 0.0)
        u_pos = np.maximum(Uphys, 0.0)

        eff_atten[i0:i1] = np.abs(u_neg).sum(axis=1) * DT / n_years
        eff_augment[i0:i1] = u_pos.sum(axis=1) * DT / n_years
        eff_atten_flood[i0:i1] = (np.abs(u_neg) * above).sum(axis=1) * DT / n_years
        eff_augment_flood[i0:i1] = (u_pos * above).sum(axis=1) * DT / n_years
        n_atten_days[i0:i1] = (Uphys < -1e-6).sum(axis=1) / n_years
        n_augment_days[i0:i1] = (Uphys > 1e-6).sum(axis=1) / n_years
        max_atten_cms[i0:i1] = np.abs(u_neg).max(axis=1)
        max_augment_cms[i0:i1] = u_pos.max(axis=1)

        del Xc, exceed, above, amax, Uphys, u_neg, u_pos
        gc.collect()

    eps = 1e-12
    vol_hi_bl, peak_bl, events_bl, rl_bl = pre.vol_hi_bl, pre.peak_bl, pre.events_bl, pre.rl_bl

    out = {
        # Open-loop skill metrics (full suite)
        "nse_openloop": pre.nse_openloop, "nse_anchored": pre.nse_anchored,
        "kge_openloop": pre.kge_openloop, "kge_anchored": pre.kge_anchored,
        "r2_openloop": pre.r2_openloop, "r2_anchored": pre.r2_anchored,
        "pbias_openloop": pre.pbias_openloop,
        "peak_bias_openloop": pre.peak_bias_openloop,
        "vol_hi_bias_openloop": pre.vol_hi_bias_openloop,
        "rl10_bias_openloop": pre.rl10_bias_openloop,
        "rl10_true": pre.rl10_true, "rl5_true": pre.rl5_true,
        "vol_hi_baseline": vol_hi_bl, "days_hi_baseline": pre.days_hi_bl,
        "peak_baseline": peak_bl, "events_baseline": events_bl,
        "vol_hi_controlled": vol_hi, "days_hi_controlled": days_hi,
        "peak_controlled": peak, "events_controlled": events,
        "vol_hi_red_abs": vol_hi_bl - vol_hi,
        "vol_hi_red_rel": (vol_hi_bl - vol_hi) / (vol_hi_bl + eps),
        "peak_red_abs": peak_bl - peak,
        "peak_red_rel": (peak_bl - peak) / (peak_bl + eps),
        "days_hi_red_rel": (pre.days_hi_bl - days_hi) / (pre.days_hi_bl + eps),
        "events_red_abs": events_bl - events,
        "events_red_rel": (events_bl - events) / (events_bl + eps),
        "eff_total_L1": eff_total, "eff_flood_L1": eff_flood,
        "eff_flood_ratio": eff_flood / (eff_total + eps),
        "max_ctrl_cms": max_ctrl_cms, "active_days_yr": active_days,
        
        "eff_atten_L1": eff_atten,
        "eff_augment_L1": eff_augment,
        "eff_atten_flood_L1": eff_atten_flood,
        "eff_augment_flood_L1": eff_augment_flood,
        "atten_fraction": eff_atten / (eff_total + eps),
        "augment_fraction": eff_augment / (eff_total + eps),
        "n_atten_days_yr": n_atten_days,
        "n_augment_days_yr": n_augment_days,
        "max_atten_cms": max_atten_cms,
        "max_augment_cms": max_augment_cms,

        "efficiency": (vol_hi_bl - vol_hi) / (eff_total + eps),
        "residual_ratio": vol_hi / (vol_hi_bl + eps),
        "vol_hi_ref": vol_hi_ref,
        "vol_vs_ref_baseline": vol_hi_bl / (vol_hi_ref + eps),
        "vol_vs_ref_controlled": vol_hi / (vol_hi_ref + eps),
    }

    for key in rl_bl.keys():
        out[f"{key}_baseline"] = rl_bl[key]
        out[f"{key}_controlled"] = rl_cl[key]
        out[f"{key}_red_rel"] = (rl_bl[key] - rl_cl[key]) / (rl_bl[key] + eps)

    # Per-RL open-loop bias (model skill for extremes)
    for key, arr in pre.rl_bias_openloop.items():
        out[f"{key}_bias_openloop"] = arr

    PROF.stop("compute_metrics")
    return out


# ============================================================
# SINGLE SCENARIO PROCESS
# ============================================================

def process_single_scenario(gcm: str, scn: int, hydro: str, ref: ReferencePack):
    print(f"\n  [{hydro} / SSP{scn}] Loading ...")
    X, U, dates, reach_ids = load_scenario_data(scn, gcm, hydro)
    print(f"    Shape: {X.shape[0]} × {X.shape[1]}")

    idx_ref, idx_this, matched = match_to_reference(ref.reach_ids, reach_ids)
    X = X[idx_this]
    U = U[idx_this]

    thr_flood = ref.thr_flood[idx_ref]
    vol_hi_ref = ref.vol_hi_ref[idx_ref]
    mean_flow_ref = ref.mean_flow[idx_ref]

    rl_ref = {k: v[idx_ref] for k, v in ref.return_levels.items()
              if isinstance(v, np.ndarray) and v.shape[0] == len(ref.reach_ids)}

    print(f"    POD (rank={RANK}) ...")
    pod = compute_pod(X, RANK)

    V = normalize_forcing(U).V
    print(f"    DMDc ...")
    dmd = fit_windowed_dmdc(pod.Z, V)

    z0, V_sim = pod.Z[:, 0:1], V[:, :-1]
    Z_ol = simulate_openloop(dmd, z0, V_sim)
    Z_anch = simulate_one_step_anchored(dmd, pod.Z, V_sim)

    pre = precompute_baseline(X, thr_flood, dates, pod, Z_ol, Z_anch)

    # ── Open-loop performance report ──
    print(f"\n    ╔{'═'*56}╗")
    print(f"    ║  OPEN-LOOP PERFORMANCE  ({hydro} / SSP{scn}){' '*(24-len(hydro)-len(str(scn)))}║")
    print(f"    ╠{'═'*56}╣")
    print(f"    ║  {'Metric':<22} {'Median':>10} {'Mean':>10} {'P25':>10} ║")
    print(f"    ╠{'─'*56}╣")
    for label, arr in [
        ("NSE (open-loop)",     pre.nse_openloop),
        ("NSE (anchored)",      pre.nse_anchored),
        ("KGE (open-loop)",     pre.kge_openloop),
        ("KGE (anchored)",      pre.kge_anchored),
        ("R² (open-loop)",      pre.r2_openloop),
        ("R² (anchored)",       pre.r2_anchored),
        ("PBIAS (open-loop)",   pre.pbias_openloop),
        ("Peak bias (OL)",      pre.peak_bias_openloop),
    ]:
        med = np.nanmedian(arr)
        mn = np.nanmean(arr)
        p25 = np.nanpercentile(arr, 25)
        print(f"    ║  {label:<22} {med:>10.4f} {mn:>10.4f} {p25:>10.4f} ║")
    print(f"    ╠{'─'*56}╣")
    print(f"    ║  {'Extreme Event Bias':<22} {'Median':>10} {'Mean':>10} {'P75':>10} ║")
    print(f"    ╠{'─'*56}╣")
    for T in RETURN_PERIODS:
        arr = pre.rl_bias_openloop[f"rl{T}"]
        med = np.nanmedian(arr)
        mn = np.nanmean(arr)
        p75 = np.nanpercentile(arr, 75)
        print(f"    ║  RL{T:<4} bias (OL)       {med:>10.4f} {mn:>10.4f} {p75:>10.4f} ║")
    vhb = pre.vol_hi_bias_openloop
    print(f"    ║  {'Vol exceed bias (OL)':<22} {np.nanmedian(vhb):>10.4f} {np.nanmean(vhb):>10.4f} {np.nanpercentile(vhb,75):>10.4f} ║")
    print(f"    ╠{'─'*56}╣")
    print(f"    ║  Spectral radii: {[f'{s:.3f}' for s in dmd.spectral_radii]}")
    print(f"    ║  Alphas used:    {dmd.alphas_used}")
    print(f"    ║  DMDc windows:   {len(dmd.bounds)}")
    print(f"    ╚{'═'*56}╝\n")

    print(f"    LQR sweeping {len(R_VALUES)} R values ...")

    Q = (Q_WEIGHT * np.eye(RANK)).astype(DTYPE)
    results = []
    effort_by_r = {}
    daily_outputs = {}
    fdc_outputs = []

    for Rw in R_VALUES:
        try:
            K_list = [lqr_gain(A, Q, float(Rw)) for A in dmd.A_list]
        except Exception as e:
            print(f"      R={Rw:.2e}: Failed ({e})")
            continue

        Z_cl, U_cl = simulate_cl_flood_only(dmd, z0, V_sim, K_list, thr_flood, pod.U_r, pod.mean_x)
        m = compute_controlled_metrics(thr_flood, pod, pre, Z_cl, U_cl, vol_hi_ref)

        effort_gm3 = np.nansum(m['eff_total_L1']) / 1e9
        effort_by_r[Rw] = effort_gm3

        df_r = pd.DataFrame(m)
        df_r["gcm"] = gcm
        df_r["hydro"] = hydro
        df_r["scenario"] = int(scn)
        df_r["R_weight"] = float(Rw)
        df_r["reach_id"] = matched
        df_r["vol_hi_ref"] = vol_hi_ref
        df_r["mean_flow_ref"] = mean_flow_ref
        df_r["thr_flood"] = thr_flood

        for k, v in rl_ref.items():
            df_r[f"{k}_pooled"] = v

        results.append(df_r)

        atten_gm3 = np.nansum(m['eff_atten_L1']) / 1e9
        augment_gm3 = np.nansum(m['eff_augment_L1']) / 1e9
        atten_pct = atten_gm3 / (effort_gm3 + 1e-12) * 100
        print(f"      R={Rw:>10.2e}: eff={effort_gm3:>5.1f}, "
              f"atten={atten_gm3:.1f}({atten_pct:.0f}%), "
              f"augment={augment_gm3:.1f}, ...")
        del Z_cl, U_cl, K_list, m, df_r
        gc.collect()

    # Daily outputs for target budgets
    print("    Computing daily outputs for target budgets...")

    for target in TARGET_BUDGETS:
        feasible = {r: e for r, e in effort_by_r.items() if e <= target * 1.1}

        if not feasible:
            best_r = min(effort_by_r.keys(), key=lambda r: effort_by_r[r])
        else:
            best_r = max(feasible.keys(), key=lambda r: feasible[r])

        actual_eff = effort_by_r[best_r]
        print(f"      Budget {target}: R={best_r:.2e}, actual_eff={actual_eff:.1f}")

        K_list = [lqr_gain(A, Q, float(best_r)) for A in dmd.A_list]
        Z_cl, U_cl = simulate_cl_flood_only(dmd, z0, V_sim, K_list, thr_flood, pod.U_r, pod.mean_x)

        df_daily = compute_daily_basin_output(pod, Z_ol, Z_cl, U_cl, thr_flood, dates)
        df_daily["gcm"] = gcm
        df_daily["hydro"] = hydro
        df_daily["scenario"] = scn
        df_daily["budget_target"] = target
        df_daily["R_weight"] = best_r
        df_daily["actual_effort_Gm3yr"] = actual_eff

        daily_outputs[(scn, target)] = df_daily

        fdc = compute_fdc(pod, Z_ol, Z_cl)
        fdc["gcm"] = gcm
        fdc["hydro"] = hydro
        fdc["scenario"] = scn
        fdc["budget_target"] = target
        fdc["R_weight"] = best_r
        fdc_outputs.append(fdc)

        del Z_cl, U_cl, K_list
        gc.collect()

    vrow = {
        "gcm": gcm, "hydro": hydro, "scenario": int(scn),
        "n_reaches": int(len(matched)),
        # Skill metrics (medians)
        "nse_openloop_median": float(np.nanmedian(pre.nse_openloop)),
        "nse_anchored_median": float(np.nanmedian(pre.nse_anchored)),
        "kge_openloop_median": float(np.nanmedian(pre.kge_openloop)),
        "kge_anchored_median": float(np.nanmedian(pre.kge_anchored)),
        "r2_openloop_median": float(np.nanmedian(pre.r2_openloop)),
        "r2_anchored_median": float(np.nanmedian(pre.r2_anchored)),
        "pbias_openloop_median": float(np.nanmedian(pre.pbias_openloop)),
        "peak_bias_openloop_median": float(np.nanmedian(pre.peak_bias_openloop)),
        # Extreme event bias (medians)
        "rl2_bias_openloop_median": float(np.nanmedian(pre.rl_bias_openloop["rl2"])),
        "rl5_bias_openloop_median": float(np.nanmedian(pre.rl_bias_openloop["rl5"])),
        "rl10_bias_openloop_median": float(np.nanmedian(pre.rl10_bias_openloop)),
        "rl50_bias_openloop_median": float(np.nanmedian(pre.rl_bias_openloop["rl50"])),
        "rl100_bias_openloop_median": float(np.nanmedian(pre.rl_bias_openloop["rl100"])),
        "vol_hi_bias_openloop_median": float(np.nanmedian(pre.vol_hi_bias_openloop)),
        # DMDc diagnostics
        "spectral_radius_max": float(max(dmd.spectral_radii)),
        "spectral_radius_mean": float(np.mean(dmd.spectral_radii)),
        "n_dmdc_windows": int(len(dmd.bounds)),
    }

    del X, U, V, Z_ol, Z_anch, pod, dmd, pre
    gc.collect()

    return (
        pd.concat(results, ignore_index=True) if results else pd.DataFrame(),
        vrow,
        daily_outputs,
        pd.DataFrame(fdc_outputs),
    )


# ============================================================
# RUNNER
# ============================================================

def run_ensemble(gcm: str, hydro: str):
    """Run a single (GCM, hydro model) ensemble member."""
    tag = f"{hydro}/{gcm}"
    print("\n" + "=" * 70)
    print(f"PROCESSING: {tag}")
    print("=" * 70)

    global PROF
    PROF = Profiler()

    print(f"\n[1] Reference pack ({hydro}) ...")
    ref = compute_reference_pack(gcm, hydro, verbose=True)

    print("\n[2] Scenarios ...")
    all_results, validation_rows = [], []
    all_daily, all_fdc = [], []

    for scn in SCENARIOS:
        df_scn, vrow, daily_out, fdc_df = process_single_scenario(gcm, scn, hydro, ref)
        if len(df_scn):
            all_results.append(df_scn)
        validation_rows.append(vrow)

        for key, df_d in daily_out.items():
            all_daily.append(df_d)

        if len(fdc_df):
            all_fdc.append(fdc_df)

    print("\n[3] Saving ...")
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    # File names include hydro model
    df_all = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    df_all.to_parquet(out_dir / f"results_{hydro}_{gcm}.parquet", index=False)
    print(f"    Results: {len(df_all)} rows")

    pd.DataFrame(validation_rows).to_csv(out_dir / f"validation_{hydro}_{gcm}.csv", index=False)

    if all_daily:
        df_daily_all = pd.concat(all_daily, ignore_index=True)
        df_daily_all.to_parquet(out_dir / f"daily_{hydro}_{gcm}.parquet", index=False)
        print(f"    Daily: {len(df_daily_all)} rows")

    if all_fdc:
        df_fdc_all = pd.concat(all_fdc, ignore_index=True)
        df_fdc_all.to_parquet(out_dir / f"fdc_{hydro}_{gcm}.parquet", index=False)
        print(f"    FDC: {len(df_fdc_all)} rows")

    PROF.report(f"PROFILING: {tag}")

    del df_all, all_results, ref
    gc.collect()


def run_all(stage: str, gcm: str, hydro: str):
    gcms = AVAILABLE_GCMS if gcm == "all" else [gcm]
    hydros = HYDRO_MODELS if hydro == "all" else [hydro]

    if stage in ("run", "all"):
        for h in hydros:
            for g in gcms:
                try:
                    run_ensemble(g, h)
                except Exception as e:
                    print(f"ERROR {h}/{g}: {e}")
                    import traceback
                    traceback.print_exc()
                gc.collect()


def main():
    global FLOOD_RL_PERIOD, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["run", "summary", "all"])
    parser.add_argument("--gcm", default="all")
    parser.add_argument("--hydro", default="all", help="Hydro model: VIC5, PRMS, or all")
    parser.add_argument("--rl", type=int, default=2, choices=RETURN_PERIODS)
    args = parser.parse_args()

    FLOOD_RL_PERIOD = int(args.rl)
    OUT_DIR = f"outputs_flood_rl{FLOOD_RL_PERIOD}"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Validate --hydro
    valid_hydro = HYDRO_MODELS + ["all"]
    if args.hydro not in valid_hydro:
        parser.error(f"--hydro must be one of {valid_hydro}")

    n_ensembles = len(AVAILABLE_GCMS if args.gcm == "all" else [args.gcm]) * \
                  len(HYDRO_MODELS if args.hydro == "all" else [args.hydro])

    print(f"\n{'='*70}")
    print(f"FLOOD CONTROL — RL{FLOOD_RL_PERIOD} Threshold")
    print(f"Hydro models: {HYDRO_MODELS if args.hydro == 'all' else [args.hydro]}")
    print(f"Ensemble members: {n_ensembles} (GCMs × hydro models)")
    print(f"R grid: {len(R_VALUES)} values ({min(R_VALUES):.0e} to {max(R_VALUES):.0e})")
    print(f"Target budgets: {TARGET_BUDGETS} Gm³/yr")
    print(f"Output: {OUT_DIR}")
    print(f"{'='*70}")

    run_all(args.stage, args.gcm, args.hydro)


if __name__ == "__main__":
    main()