#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary: Mass Conservation Diagnostic + Fig S5
=====================================================

Produces mass_conservation_summary.csv and plots Fig S5 (Surrogate Mass Fidelity).
Used by 06_figures_supplementary.ipynb.

Diagnostics:
  1) Open-loop mass fidelity: surrogate vs RAPID basin-total
  2) Clipping mass violation across R_w values
  3) δ consistency: reduced-space vs physical-space control effect
  4) Per-reach δ tracking
  5) Physical feasibility: |u_i(t)| vs x_i(t)

Usage:
    python si_mass_conservation.py --gcm ACCESS-CM2 --hydro VIC5 --ssp 245
    python si_mass_conservation.py --gcm all --hydro all --ssp all
"""

from __future__ import annotations

import os
import gc
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

import importlib
_pipeline = importlib.import_module("04_dmd_lqr_analysis")

SCENARIOS      = _pipeline.SCENARIOS
AVAILABLE_GCMS = _pipeline.AVAILABLE_GCMS
HYDRO_MODELS   = _pipeline.HYDRO_MODELS
RANK           = _pipeline.RANK
DTYPE          = _pipeline.DTYPE
DT             = _pipeline.DT
Q_WEIGHT       = _pipeline.Q_WEIGHT

load_scenario_data     = _pipeline.load_scenario_data
match_to_reference     = _pipeline.match_to_reference
compute_reference_pack = _pipeline.compute_reference_pack
compute_pod            = _pipeline.compute_pod
normalize_forcing      = _pipeline.normalize_forcing
fit_windowed_dmdc      = _pipeline.fit_windowed_dmdc
simulate_openloop      = _pipeline.simulate_openloop
simulate_cl_flood_only = _pipeline.simulate_cl_flood_only
lqr_gain               = _pipeline.lqr_gain
PODPack                = _pipeline.PODPack
DMDcPack               = _pipeline.DMDcPack
ReferencePack          = _pipeline.ReferencePack

OUT_DIR = Path("outputs_mass_conservation")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Test clipping across multiple R_w to show it shrinks with less aggressive control
RW_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


# ============================================================
# HELPERS
# ============================================================

def reconstruct_basin_totals(pod, Z, T_use, return_raw=False):
    """Compute basin-total mass from reduced states, with and without clipping."""
    n = pod.U_r.shape[0]
    M_clip = np.zeros(T_use, dtype=DTYPE)
    M_raw = np.zeros(T_use, dtype=DTYPE)

    CHUNK = 1024
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        Uc = pod.U_r[i0:i1, :]
        mc = pod.mean_x[i0:i1, :]
        X_raw = Uc @ Z[:, :T_use] + mc
        M_raw += X_raw.sum(axis=0)
        M_clip += np.maximum(X_raw, 0.0).sum(axis=0)
        del X_raw
    return M_clip, M_raw


def compute_physical_control(pod, U_ctrl_z, T_use):
    """Compute basin-total control signal in physical space."""
    n = pod.U_r.shape[0]
    M_ctrl = np.zeros(T_use, dtype=DTYPE)
    CHUNK = 1024
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        u_phys = pod.U_r[i0:i1, :] @ U_ctrl_z[:, :T_use]
        M_ctrl += u_phys.sum(axis=0)
        del u_phys
    return M_ctrl


# ============================================================
# DIAGNOSTIC 1: Open-loop mass fidelity
# ============================================================

def diag_openloop_mass(X_rapid, pod, Z_open, T_use, dates=None):
    """How well does the surrogate track RAPID basin-total mass?"""
    n = X_rapid.shape[0]
    M_rapid = X_rapid[:, :T_use].sum(axis=0)
    M_open, _ = reconstruct_basin_totals(pod, Z_open, T_use)

    resid = M_open - M_rapid
    mean_M = np.mean(np.abs(M_rapid)) + 1e-10

    # ── Volume ratio: total surrogate / total RAPID ──
    vol_ratio = float(np.sum(M_open) / (np.sum(M_rapid) + 1e-10))

    # ── Cumulative divergence time series ──
    cum_surr = np.cumsum(M_open)
    cum_rapid = np.cumsum(M_rapid)
    cum_divergence = (cum_surr - cum_rapid) / (cum_rapid + 1e-10)  # (T,)
    cum_div_final = float(cum_divergence[-1])
    cum_div_max = float(np.max(np.abs(cum_divergence)))
    cum_div_bounded = bool(cum_div_max < 0.20)  # stays within 20%

    # ── Annual volume agreement ──
    annual_corr = np.nan
    annual_r2 = np.nan
    annual_vol_surr = None
    annual_vol_rapid = None
    if dates is not None:
        years = pd.DatetimeIndex(dates[:T_use]).year.values
        unique_years = np.unique(years)
        if len(unique_years) >= 5:
            annual_vol_surr = np.array([M_open[years == y].sum() for y in unique_years])
            annual_vol_rapid = np.array([M_rapid[years == y].sum() for y in unique_years])
            annual_corr = float(np.corrcoef(annual_vol_surr, annual_vol_rapid)[0, 1])
            # R²
            ss_res = np.sum((annual_vol_surr - annual_vol_rapid) ** 2)
            ss_tot = np.sum((annual_vol_rapid - np.mean(annual_vol_rapid)) ** 2) + 1e-10
            annual_r2 = float(1 - ss_res / ss_tot)

    # ── Monthly volume agreement ──
    monthly_corr = np.nan
    if dates is not None:
        months = pd.DatetimeIndex(dates[:T_use]).to_period('M')
        unique_months = months.unique()
        if len(unique_months) >= 12:
            monthly_surr = np.array([M_open[months == m].sum() for m in unique_months])
            monthly_rapid = np.array([M_rapid[months == m].sum() for m in unique_months])
            monthly_corr = float(np.corrcoef(monthly_surr, monthly_rapid)[0, 1])

    return {
        'M_rapid': M_rapid,
        'M_open': M_open,
        # Original
        'ol_relative_rmse': float(np.sqrt(np.mean((resid / mean_M) ** 2))),
        'ol_relative_bias': float(np.mean(resid) / mean_M),
        'ol_abs_rmse': float(np.sqrt(np.mean(resid ** 2))),
        # Volume ratio
        'vol_ratio': vol_ratio,
        # Cumulative divergence
        'cum_divergence': cum_divergence,
        'cum_div_final': cum_div_final,
        'cum_div_max': cum_div_max,
        'cum_div_bounded': cum_div_bounded,
        # Annual
        'annual_corr': annual_corr,
        'annual_r2': annual_r2,
        'annual_vol_surr': annual_vol_surr,
        'annual_vol_rapid': annual_vol_rapid,
        # Monthly
        'monthly_corr': monthly_corr,
    }


# ============================================================
# DIAGNOSTIC 2: Clipping violation across R_w
# ============================================================

def diag_clipping_vs_rw(pod, dmd, z0, V_sim, thr_flood, Z_open, T_use):
    """
    For each R_w, measure:
    - clipping_fraction: mass added by max(x,0) as fraction of basin-total
    - This should decrease as R_w increases (less aggressive control)
    """
    # Open-loop clipping (baseline)
    M_ol_clip, M_ol_raw = reconstruct_basin_totals(pod, Z_open, T_use)
    mean_M = np.mean(np.abs(M_ol_clip)) + 1e-10
    clip_ol = float(np.mean(M_ol_clip - M_ol_raw) / mean_M)

    Q = (Q_WEIGHT * np.eye(RANK)).astype(DTYPE)
    results = [{'R_w': None, 'label': 'open-loop', 'clipping_frac': clip_ol}]

    for Rw in RW_VALUES:
        try:
            K_list = [lqr_gain(A, Q, float(Rw)) for A in dmd.A_list]
            Z_cl, U_cl = simulate_cl_flood_only(
                dmd, z0, V_sim, K_list, thr_flood, pod.U_r, pod.mean_x
            )
            M_cl_clip, M_cl_raw = reconstruct_basin_totals(pod, Z_cl, T_use)
            clip_frac = float(np.mean(M_cl_clip - M_cl_raw) / mean_M)
            results.append({
                'R_w': Rw,
                'label': f'R_w={Rw:.0e}',
                'clipping_frac': clip_frac,
                'Z_cl': Z_cl,
                'U_cl': U_cl,
                'M_cl_clip': M_cl_clip,
            })
            print(f"        R_w={Rw:.0e}: clipping={clip_frac*100:.3f}%")
        except Exception as e:
            print(f"        R_w={Rw:.0e}: FAILED ({e})")

    return results


# ============================================================
# DIAGNOSTIC 3: δ consistency (reduced vs physical)
# ============================================================

def diag_delta_consistency(pod, Z_open, Z_closed, T_use):
    """
    δ(t) = z_cl(t) - z_ol(t) in reduced space is exact.
    In physical space with clipping: Σ max(x_cl,0) - Σ max(x_ol,0) ≠ w_r^T δ exactly.
    Measure the deviation.
    """
    n = pod.U_r.shape[0]
    w_r = pod.U_r.T @ np.ones(n, dtype=DTYPE)  # sum-projection vector

    delta_z = Z_closed[:, :T_use] - Z_open[:, :T_use]
    M_delta_reduced = w_r @ delta_z  # (T,) without clipping

    # Physical with clipping
    M_cl_clip, _ = reconstruct_basin_totals(pod, Z_closed, T_use)
    M_ol_clip, _ = reconstruct_basin_totals(pod, Z_open, T_use)
    M_delta_physical = M_cl_clip - M_ol_clip  # (T,)

    # Correlation (robust to scale)
    corr = float(np.corrcoef(M_delta_reduced, M_delta_physical)[0, 1])

    # MAE relative to range (not dividing by near-zero values!)
    delta_range = np.max(np.abs(M_delta_physical)) + 1e-10
    mae_relative = float(np.mean(np.abs(M_delta_physical - M_delta_reduced)) / delta_range)

    # Fraction of variance explained
    ss_res = np.sum((M_delta_physical - M_delta_reduced) ** 2)
    ss_tot = np.sum((M_delta_physical - np.mean(M_delta_physical)) ** 2) + 1e-10
    r2 = float(1 - ss_res / ss_tot)

    return {
        'M_delta_reduced': M_delta_reduced,
        'M_delta_physical': M_delta_physical,
        'delta_corr': corr,
        'delta_mae_relative': mae_relative,
        'delta_r2': r2,
    }


# ============================================================
# DIAGNOSTIC 4: Per-reach δ tracking
# ============================================================

def diag_per_reach_delta(pod, Z_open, Z_closed, T_use):
    """
    Per reach: correlation between (x_cl_clip - x_ol_clip) and U_r @ δ.
    """
    n = pod.U_r.shape[0]
    delta_z = Z_closed[:, :T_use] - Z_open[:, :T_use]

    reach_corr = np.full(n, np.nan, dtype=DTYPE)
    CHUNK = 512

    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        Uc = pod.U_r[i0:i1, :]
        mc = pod.mean_x[i0:i1, :]

        X_ol = np.maximum(Uc @ Z_open[:, :T_use] + mc, 0.0)
        X_cl = np.maximum(Uc @ Z_closed[:, :T_use] + mc, 0.0)
        delta_phys = X_cl - X_ol       # with clipping
        delta_pred = Uc @ delta_z       # without clipping

        for j in range(i1 - i0):
            dp = delta_phys[j]
            dr = delta_pred[j]
            if np.std(dp) > 1e-10 and np.std(dr) > 1e-10:
                reach_corr[i0 + j] = np.corrcoef(dp, dr)[0, 1]

        del X_ol, X_cl, delta_phys, delta_pred
        gc.collect()

    valid = ~np.isnan(reach_corr)
    return {
        'reach_corr': reach_corr,
        'median': float(np.nanmedian(reach_corr)),
        'p25': float(np.nanpercentile(reach_corr, 25)),
        'gt90': int(np.nansum(reach_corr > 0.9)),
        'gt95': int(np.nansum(reach_corr > 0.95)),
        'n_valid': int(valid.sum()),
    }


# ============================================================
# DIAGNOSTIC 5: Physical feasibility of control signal
# ============================================================

def diag_physical_feasibility(pod, Z_open, Z_closed, U_ctrl_z, X_rapid, U_lateral,
                               thr_flood, T_use):
    """
    Check whether the control signal is physically plausible.

    Key insight: POD distributes reduced-space control to ALL reaches, but control
    is only *intended* at reaches exceeding the flood threshold. We therefore:
      (a) Compute basin-aggregate: Σ|u|(t) / Σx(t) across all reaches
      (b) Compute per-reach ratios ONLY at reaches with exceedance (x > τ)

    Two checks:
      1) |u_i(t)| / x_i(t) < 1  — can't remove more water than exists in channel
      2) |u_i(t)| / d_i(t) < 1  — can't detain more than lateral inflow
    """
    n = pod.U_r.shape[0]
    CHUNK = 512

    # Align dimensions
    T_ctrl = min(T_use, U_ctrl_z.shape[1], Z_open.shape[1],
                 U_lateral.shape[1], X_rapid.shape[1])

    # ── Basin-aggregate time series ──
    sum_abs_u = np.zeros(T_ctrl, dtype=DTYPE)
    sum_x_ol = np.zeros(T_ctrl, dtype=DTYPE)
    sum_d_lat = np.zeros(T_ctrl, dtype=DTYPE)
    sum_exceedance = np.zeros(T_ctrl, dtype=DTYPE)

    # ── Per-reach: only for reaches that experience exceedance ──
    n_exceed_steps = np.zeros(n, dtype=np.int64)   # how many timesteps x > τ
    sum_ratio_flow = np.zeros(n, dtype=DTYPE)       # sum of |u|/x during exceedance
    sum_ratio_lat = np.zeros(n, dtype=DTYPE)        # sum of |u|/d during exceedance
    max_ratio_flow = np.zeros(n, dtype=DTYPE)
    max_ratio_lat = np.zeros(n, dtype=DTYPE)
    n_violate_flow = np.zeros(n, dtype=np.int64)    # timesteps where |u| > x
    n_violate_lat = np.zeros(n, dtype=np.int64)

    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        Uc = pod.U_r[i0:i1, :]
        mc = pod.mean_x[i0:i1, :]

        # Physical-space control signal (absolute)
        u_phys = np.abs(Uc @ U_ctrl_z[:, :T_ctrl])  # (chunk, T)

        # Open-loop flow
        x_ol = np.maximum(Uc @ Z_open[:, :T_ctrl] + mc, 0.0)  # (chunk, T)

        # Lateral inflow (raw)
        d_lat = np.maximum(U_lateral[i0:i1, :T_ctrl], 0.0)  # (chunk, T)

        # Flood threshold for this chunk
        tau_chunk = thr_flood[i0:i1, np.newaxis]  # (chunk, 1)

        # Exceedance mask: where flow exceeds threshold
        exceed = x_ol > tau_chunk  # (chunk, T) bool

        # Basin-aggregate sums
        sum_abs_u += u_phys.sum(axis=0)
        sum_x_ol += x_ol.sum(axis=0)
        sum_d_lat += d_lat.sum(axis=0)
        sum_exceedance += (x_ol * exceed).sum(axis=0)

        # Per-reach statistics (only during exceedance)
        for j in range(i1 - i0):
            exc_j = exceed[j]
            n_exc = exc_j.sum()
            n_exceed_steps[i0 + j] = n_exc
            if n_exc == 0:
                continue

            u_exc = u_phys[j, exc_j]
            x_exc = x_ol[j, exc_j]
            d_exc = d_lat[j, exc_j]

            # |u|/x ratio
            r_flow = u_exc / (x_exc + 1e-6)
            sum_ratio_flow[i0 + j] = r_flow.sum()
            max_ratio_flow[i0 + j] = r_flow.max()
            n_violate_flow[i0 + j] = (r_flow > 1.0).sum()

            # |u|/d ratio
            r_lat = u_exc / (d_exc + 1e-6)
            sum_ratio_lat[i0 + j] = r_lat.sum()
            max_ratio_lat[i0 + j] = r_lat.max()
            n_violate_lat[i0 + j] = (r_lat > 1.0).sum()

        del u_phys, x_ol, d_lat, exceed
        gc.collect()

    # ── Basin-aggregate ratio time series ──
    basin_ratio_flow = sum_abs_u / (sum_x_ol + 1e-10)       # Σ|u| / Σx
    basin_ratio_exceedance = sum_abs_u / (sum_exceedance + 1e-10)  # Σ|u| / Σ(x where x>τ)

    # ── Per-reach mean ratios (only reaches with exceedance) ──
    has_exceed = n_exceed_steps > 0
    n_exceed_reaches = int(has_exceed.sum())

    mean_ratio_flow = np.where(has_exceed, sum_ratio_flow / (n_exceed_steps + 1e-10), np.nan)
    mean_ratio_lat = np.where(has_exceed, sum_ratio_lat / (n_exceed_steps + 1e-10), np.nan)
    frac_violate_flow = np.where(has_exceed, n_violate_flow / (n_exceed_steps + 1e-10), np.nan)
    frac_violate_lat = np.where(has_exceed, n_violate_lat / (n_exceed_steps + 1e-10), np.nan)

    # ── Summary ──
    valid_flow = mean_ratio_flow[has_exceed]
    valid_lat = mean_ratio_lat[has_exceed]

    return {
        # Basin-aggregate
        'basin_ratio_flow_median': float(np.median(basin_ratio_flow)),
        'basin_ratio_flow_p95': float(np.percentile(basin_ratio_flow, 95)),
        'basin_ratio_exceedance_median': float(np.median(basin_ratio_exceedance)),
        'basin_ratio_exceedance_p95': float(np.percentile(basin_ratio_exceedance, 95)),
        # Per-reach (during exceedance only)
        'n_exceed_reaches': n_exceed_reaches,
        'median_mean_ratio_flow': float(np.median(valid_flow)) if n_exceed_reaches > 0 else 0.0,
        'p95_mean_ratio_flow': float(np.percentile(valid_flow, 95)) if n_exceed_reaches > 0 else 0.0,
        'median_mean_ratio_lateral': float(np.median(valid_lat)) if n_exceed_reaches > 0 else 0.0,
        'p95_mean_ratio_lateral': float(np.percentile(valid_lat, 95)) if n_exceed_reaches > 0 else 0.0,
        'pct_reaches_flow_ok': float((valid_flow < 1.0).mean() * 100) if n_exceed_reaches > 0 else 100.0,
        'pct_reaches_lateral_ok': float((valid_lat < 1.0).mean() * 100) if n_exceed_reaches > 0 else 100.0,
        # Arrays for plotting
        'mean_ratio_flow': mean_ratio_flow,
        'mean_ratio_lateral': mean_ratio_lat,
        'basin_ratio_flow_ts': basin_ratio_flow,
        'basin_ratio_exceedance_ts': basin_ratio_exceedance,
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_all(d1, d2_list, d3, d4, d5, dates, gcm, hydro, ssp):
    """Generate comprehensive diagnostic figure."""

    T = len(d1['M_rapid'])
    time = np.arange(T) / 365.25 + 2020

    fig = plt.figure(figsize=(16, 26))
    gs = GridSpec(8, 2, figure=fig, hspace=0.45, wspace=0.3,
                  height_ratios=[1, 1, 1, 0.8, 1, 1, 1, 0.6])

    fig.suptitle(
        f"Mass Conservation: {hydro}/{gcm} SSP{ssp}",
        fontsize=14, fontweight='bold', y=0.98
    )

    # ── (a) Basin-total mass tracking ──
    ax = fig.add_subplot(gs[0, :])
    ax.plot(time, d1['M_rapid'] / 1e3, 'k-', alpha=0.4, lw=0.5, label='RAPID')
    ax.plot(time, d1['M_open'] / 1e3, 'b-', alpha=0.6, lw=0.5, label='Surrogate (OL)')
    ax.set_ylabel('Basin-total Q (×10³ m³/s)')
    ax.legend(fontsize=9)
    ax.set_title(f'(a) Open-loop mass fidelity '
                 f'(vol ratio={d1["vol_ratio"]:.4f}, '
                 f'bias={d1["ol_relative_bias"]*100:.1f}%)', fontsize=11)
    ax.set_xlim(2020, 2100)

    # ── (b) Cumulative divergence ──
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(time[:len(d1['cum_divergence'])], d1['cum_divergence'] * 100,
            'b-', lw=1)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.axhline(20, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(-20, color='red', lw=0.5, ls=':', alpha=0.5)
    ax.set_ylabel('Cumulative divergence (%)')
    ax.set_xlabel('Year')
    ax.set_title(f'(b) Cumulative mass divergence\n'
                 f'(final={d1["cum_div_final"]*100:.2f}%, '
                 f'max={d1["cum_div_max"]*100:.2f}%)', fontsize=10)
    ax.set_xlim(2020, 2100)

    # ── (c) Annual volume scatter ──
    ax = fig.add_subplot(gs[1, 1])
    if d1['annual_vol_surr'] is not None:
        ax.scatter(d1['annual_vol_rapid'] / 1e6,
                   d1['annual_vol_surr'] / 1e6,
                   s=20, c='steelblue', alpha=0.7, edgecolors='white', lw=0.5)
        lims = [min(d1['annual_vol_rapid'].min(), d1['annual_vol_surr'].min()) / 1e6,
                max(d1['annual_vol_rapid'].max(), d1['annual_vol_surr'].max()) / 1e6]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlabel('RAPID annual volume (×10⁶ m³/s·day)')
        ax.set_ylabel('Surrogate annual volume')
        ax.set_title(f'(c) Annual volume agreement\n'
                     f'(r={d1["annual_corr"]:.4f}, '
                     f'R²={d1["annual_r2"]:.4f})', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('(c) Annual volume agreement', fontsize=10)

    # ── (d) Clipping fraction vs R_w ──
    ax = fig.add_subplot(gs[2, 0])
    labels = [r['label'] for r in d2_list]
    values = [r['clipping_frac'] * 100 for r in d2_list]
    colors = ['gray'] + ['steelblue'] * (len(d2_list) - 1)
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Clipping mass fraction (%)')
    ax.set_title('(d) max(x,0) clipping vs control aggressiveness', fontsize=11)
    ax.axhline(1.0, color='red', ls='--', lw=0.8, alpha=0.5, label='1% reference')
    ax.legend(fontsize=8)

    # ── (e) Clipping summary ──
    ax = fig.add_subplot(gs[2, 1])
    # Find the entry with smallest R_w that has data
    cl_entries = [r for r in d2_list if r['R_w'] is not None and 'M_cl_clip' in r]
    if cl_entries:
        # Show OL and most aggressive CL
        _, M_ol_raw = reconstruct_basin_totals.__wrapped__(None, None, None) if False else (None, None)
        # Just show text summary
        ax.text(0.5, 0.5,
                f"Clipping increases with\ncontrol aggressiveness:\n\n"
                f"Open-loop: {d2_list[0]['clipping_frac']*100:.3f}%\n"
                + "\n".join(f"R_w={r['R_w']:.0e}: {r['clipping_frac']*100:.3f}%"
                           for r in cl_entries),
                transform=ax.transAxes, fontsize=10, va='center', ha='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax.set_title('(e) Clipping summary', fontsize=11)
    ax.axis('off')

    # ── (f) δ reduced vs physical (scatter) ──
    ax = fig.add_subplot(gs[3, 0])
    step = max(1, len(d3['M_delta_reduced']) // 3000)
    ax.scatter(d3['M_delta_reduced'][::step] / 1e3,
               d3['M_delta_physical'][::step] / 1e3,
               s=1, alpha=0.2, c='red', rasterized=True)
    lims = [min(d3['M_delta_reduced'].min(), d3['M_delta_physical'].min()) / 1e3,
            max(d3['M_delta_reduced'].max(), d3['M_delta_physical'].max()) / 1e3]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1')
    ax.set_xlabel('Reduced-space δ (×10³ m³/s)')
    ax.set_ylabel('Physical-space δ (×10³ m³/s)')
    ax.set_title(f'(f) δ consistency: r={d3["delta_corr"]:.4f}, '
                 f'R²={d3["delta_r2"]:.4f}', fontsize=11)
    ax.legend(fontsize=9)

    # ── (g) Per-reach δ correlation histogram ──
    ax = fig.add_subplot(gs[3, 1])
    valid = d4['reach_corr'][~np.isnan(d4['reach_corr'])]
    ax.hist(valid, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(d4['median'], color='red', ls='--', lw=1.5,
               label=f'Median={d4["median"]:.3f}')
    ax.set_xlabel('Correlation (clipped δ vs reduced δ)')
    ax.set_ylabel('Reaches')
    ax.set_title(f'(g) Per-reach δ tracking '
                 f'({d4["gt90"]}/{d4["n_valid"]} with r>0.9)', fontsize=11)
    ax.legend(fontsize=9)

    # ── (h) Physical feasibility: |u|/x during exceedance ──
    ax = fig.add_subplot(gs[4, 0])
    valid_flow = d5['mean_ratio_flow']
    valid_flow = valid_flow[~np.isnan(valid_flow)]
    if len(valid_flow) > 0:
        # Clip for visualization
        plot_vals = np.clip(valid_flow, 0, 3)
        ax.hist(plot_vals, bins=50,
                color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(1.0, color='red', ls='--', lw=1.5, label='|u|=x (violation)')
        ax.axvline(d5['median_mean_ratio_flow'], color='orange', ls='--', lw=1.5,
                   label=f'Median={d5["median_mean_ratio_flow"]:.3f}')
    ax.set_xlabel('Mean |u_i| / x_i  (during exceedance only)')
    ax.set_ylabel('Reaches')
    ax.set_title(f'(h) Control vs channel flow\n'
                 f'({d5["pct_reaches_flow_ok"]:.1f}% of exceeding reaches OK, '
                 f'median={d5["median_mean_ratio_flow"]:.3f})', fontsize=10)
    ax.legend(fontsize=8)

    # ── (i) Physical feasibility: basin-aggregate ratio time series ──
    ax = fig.add_subplot(gs[4, 1])
    T_ts = len(d5['basin_ratio_flow_ts'])
    time_ts = np.arange(T_ts) / 365.25 + 2020
    ax.plot(time_ts, d5['basin_ratio_flow_ts'] * 100, 'b-', alpha=0.3, lw=0.5,
            label='Σ|u|/Σx')
    ax.plot(time_ts, d5['basin_ratio_exceedance_ts'] * 100, 'r-', alpha=0.3, lw=0.5,
            label='Σ|u|/Σ(x>τ)')
    # Rolling mean
    win = 365
    if T_ts > win:
        roll1 = pd.Series(d5['basin_ratio_flow_ts'] * 100).rolling(win, center=True).mean()
        roll2 = pd.Series(d5['basin_ratio_exceedance_ts'] * 100).rolling(win, center=True).mean()
        ax.plot(time_ts, roll1, 'b-', lw=1.5)
        ax.plot(time_ts, roll2, 'r-', lw=1.5)
    ax.axhline(100, color='k', ls='--', lw=0.5, alpha=0.5)
    ax.set_ylabel('Control ratio (%)')
    ax.set_xlabel('Year')
    ax.set_title(f'(i) Basin-aggregate control ratio\n'
                 f'(median Σ|u|/Σx = {d5["basin_ratio_flow_median"]*100:.2f}%)',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(2020, 2100)

    # ── Summary table ──
    ax = fig.add_subplot(gs[5:, :])
    ax.axis('off')

    table_data = [
        ['1. Volume ratio (surr/RAPID)', f"{d1['vol_ratio']:.4f}",
         '✓' if abs(d1['vol_ratio'] - 1.0) < 0.05 else '△'],
        ['1. Cumulative divergence (max)', f"{d1['cum_div_max']*100:.2f}%",
         '✓' if d1['cum_div_bounded'] else '✗'],
        ['1. Annual volume corr', f"{d1['annual_corr']:.4f}",
         '✓' if d1['annual_corr'] > 0.95 else '△'],
        ['1. Annual volume R²', f"{d1['annual_r2']:.4f}",
         '✓' if d1['annual_r2'] > 0.90 else '△'],
        ['2. CL clipping (R_w=1.0)',
         f"{[r['clipping_frac'] for r in d2_list if r.get('R_w') == 1.0][0]*100:.3f}%"
         if any(r.get('R_w') == 1.0 for r in d2_list) else 'N/A',
         '✓' if any(r.get('R_w') == 1.0 and r['clipping_frac'] < 0.05 for r in d2_list) else '△'],
        ['2. CL clipping (R_w=100)',
         f"{[r['clipping_frac'] for r in d2_list if r.get('R_w') == 100.0][0]*100:.3f}%"
         if any(r.get('R_w') == 100.0 for r in d2_list) else 'N/A',
         '✓' if any(r.get('R_w') == 100.0 and r['clipping_frac'] < 0.01 for r in d2_list) else '△'],
        ['3. δ basin-total correlation', f"{d3['delta_corr']:.4f}",
         '✓' if d3['delta_corr'] > 0.95 else '△'],
        ['3. δ basin-total R²', f"{d3['delta_r2']:.4f}",
         '✓' if d3['delta_r2'] > 0.90 else '△'],
        ['4. Per-reach δ corr (median)', f"{d4['median']:.3f}",
         '✓' if d4['median'] > 0.8 else '△'],
        ['4. Reaches with r > 0.9', f"{d4['gt90']}/{d4['n_valid']}",
         '✓' if d4['gt90'] / max(d4['n_valid'], 1) > 0.2 else '△'],
        ['5. |u|/x at exceedance (median)', f"{d5['median_mean_ratio_flow']:.3f}",
         '✓' if d5['median_mean_ratio_flow'] < 1.0 else '✗'],
        ['5. Basin Σ|u|/Σx (median)', f"{d5['basin_ratio_flow_median']*100:.2f}%",
         '✓' if d5['basin_ratio_flow_median'] < 0.5 else '△'],
        ['5. Exceeding reaches with |u|<x', f"{d5['pct_reaches_flow_ok']:.1f}%",
         '✓' if d5['pct_reaches_flow_ok'] > 80 else '△'],
    ]

    table = ax.table(cellText=table_data,
                     colLabels=['Diagnostic', 'Value', ''],
                     cellLoc='left', colLoc='left',
                     loc='center', colWidths=[0.45, 0.20, 0.05])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    # Color status column
    for i, row in enumerate(table_data):
        cell = table[i + 1, 2]
        if row[2] == '✓':
            cell.set_facecolor('#d4edda')
        elif row[2] == '△':
            cell.set_facecolor('#fff3cd')
        else:
            cell.set_facecolor('#f8d7da')

    fname = f"mass_diag_{hydro}_{gcm}_ssp{ssp}"
    fig.savefig(OUT_DIR / f"{fname}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"{fname}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      Saved: {OUT_DIR / fname}")


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(d1, d2_list, d3, d4, d5, gcm, hydro, ssp):
    print(f"\n    ╔{'═'*62}╗")
    print(f"    ║  MASS CONSERVATION: {hydro}/{gcm} SSP{ssp}")
    print(f"    ╠{'═'*62}╣")
    print(f"    ║  1. OL mass fidelity:")
    print(f"    ║     Volume ratio (surr/RAPID):  {d1['vol_ratio']:>8.4f}")
    print(f"    ║     Rel. bias:                  {d1['ol_relative_bias']*100:>8.2f}%")
    print(f"    ║     Rel. RMSE (daily):          {d1['ol_relative_rmse']*100:>8.2f}%")
    print(f"    ║     Cum. divergence (final):    {d1['cum_div_final']*100:>8.2f}%")
    print(f"    ║     Cum. divergence (max |·|):  {d1['cum_div_max']*100:>8.2f}%")
    print(f"    ║     Cum. bounded (<20%):        {'✓' if d1['cum_div_bounded'] else '✗'}")
    print(f"    ║     Annual volume corr:         {d1['annual_corr']:>8.4f}")
    print(f"    ║     Annual volume R²:           {d1['annual_r2']:>8.4f}")
    print(f"    ║     Monthly volume corr:        {d1['monthly_corr']:>8.4f}")
    print(f"    ╠{'─'*62}╣")
    print(f"    ║  2. Clipping (max(x,0) mass added):")
    for r in d2_list:
        print(f"    ║     {r['label']:<15s}: {r['clipping_frac']*100:>8.4f}%")
    print(f"    ╠{'─'*62}╣")
    print(f"    ║  3. δ consistency (reduced ↔ physical):")
    print(f"    ║     Correlation: {d3['delta_corr']:>8.4f}")
    print(f"    ║     R²:          {d3['delta_r2']:>8.4f}")
    print(f"    ║     MAE/range:   {d3['delta_mae_relative']*100:>8.2f}%")
    print(f"    ╠{'─'*62}╣")
    print(f"    ║  4. Per-reach δ tracking:")
    print(f"    ║     Median r:    {d4['median']:>8.3f}")
    print(f"    ║     r > 0.90:    {d4['gt90']:>5d} / {d4['n_valid']}")
    print(f"    ║     r > 0.95:    {d4['gt95']:>5d} / {d4['n_valid']}")
    print(f"    ╠{'─'*62}╣")
    print(f"    ║  5. Physical feasibility of control:")
    print(f"    ║     Exceeding reaches: {d5['n_exceed_reaches']}")
    print(f"    ║     |u|/x at exceedance (median): {d5['median_mean_ratio_flow']:>8.3f}  (P95: {d5['p95_mean_ratio_flow']:.3f})")
    print(f"    ║     |u|/d at exceedance (median): {d5['median_mean_ratio_lateral']:>8.3f}  (P95: {d5['p95_mean_ratio_lateral']:.3f})")
    print(f"    ║     Basin Σ|u|/Σx (median):       {d5['basin_ratio_flow_median']*100:>8.2f}%")
    print(f"    ║     Reaches with |u|<x:           {d5['pct_reaches_flow_ok']:>7.1f}%")
    print(f"    ║     Reaches with |u|<d:           {d5['pct_reaches_lateral_ok']:>7.1f}%")
    flow_ok = d5['median_mean_ratio_flow'] < 1.0
    lat_ok = d5['median_mean_ratio_lateral'] < 1.0
    print(f"    ║     {'✓ Control < channel flow (median)' if flow_ok else '✗ Control exceeds channel flow'}")
    print(f"    ║     {'✓ Control < lateral inflow (median)' if lat_ok else '△ Control may exceed lateral inflow at some reaches'}")
    print(f"    ╚{'═'*62}╝\n")


# ============================================================
# MAIN RUNNER
# ============================================================

def run_diagnostic(gcm: str, hydro: str, ssp: int):
    tag = f"{hydro}/{gcm}/SSP{ssp}"
    print(f"\n{'─'*60}")
    print(f"  Mass diagnostic: {tag}")
    print(f"{'─'*60}")

    # ── Load ──
    print(f"    Loading...")
    ref = compute_reference_pack(gcm, hydro, verbose=False)
    X, U, dates, reach_ids = load_scenario_data(ssp, gcm, hydro)

    idx_ref, idx_this, matched = match_to_reference(ref.reach_ids, reach_ids)
    X = X[idx_this]
    U = U[idx_this]
    thr_flood = ref.thr_flood[idx_ref]

    n, T_full = X.shape
    print(f"    {n} reaches × {T_full} steps")

    # ── Build surrogate ──
    print(f"    POD + DMDc...")
    pod = compute_pod(X, RANK)
    fp = normalize_forcing(U)
    dmd = fit_windowed_dmdc(pod.Z, fp.V)

    z0 = pod.Z[:, 0:1]
    V_sim = fp.V[:, :-1]

    print(f"    Open-loop simulation...")
    Z_open = simulate_openloop(dmd, z0, V_sim)

    T_use = min(T_full, Z_open.shape[1])

    # ── Diagnostic 1 ──
    print(f"    Diag 1: OL mass fidelity...")
    d1 = diag_openloop_mass(X, pod, Z_open, T_use, dates=dates)

    # ── Diagnostic 2 ──
    print(f"    Diag 2: Clipping vs R_w...")
    d2_list = diag_clipping_vs_rw(pod, dmd, z0, V_sim, thr_flood, Z_open, T_use)

    # ── Get reference CL for remaining diagnostics (R_w=1.0) ──
    rw1_entry = next((r for r in d2_list if r.get('R_w') == 1.0), None)
    if rw1_entry and 'Z_cl' in rw1_entry:
        Z_closed = rw1_entry['Z_cl']
        U_ctrl_z = rw1_entry['U_cl']
    else:
        print(f"    Running CL with R_w=1.0...")
        Q = (Q_WEIGHT * np.eye(RANK)).astype(DTYPE)
        K_list = [lqr_gain(A, Q, 1.0) for A in dmd.A_list]
        Z_closed, U_ctrl_z = simulate_cl_flood_only(
            dmd, z0, V_sim, K_list, thr_flood, pod.U_r, pod.mean_x
        )

    # ── Diagnostic 3 ──
    print(f"    Diag 3: δ consistency...")
    d3 = diag_delta_consistency(pod, Z_open, Z_closed, T_use)

    # ── Diagnostic 4 ──
    print(f"    Diag 4: Per-reach δ tracking...")
    d4 = diag_per_reach_delta(pod, Z_open, Z_closed, T_use)

    # ── Diagnostic 5 ──
    print(f"    Diag 5: Physical feasibility...")
    d5 = diag_physical_feasibility(pod, Z_open, Z_closed, U_ctrl_z, X, U, thr_flood, T_use)

    # ── Report ──
    print_summary(d1, d2_list, d3, d4, d5, gcm, hydro, ssp)

    # ── Plot ──
    print(f"    Plotting...")
    plot_all(d1, d2_list, d3, d4, d5, dates, gcm, hydro, ssp)

    # ── Save ──
    summary_row = {
        'gcm': gcm, 'hydro': hydro, 'ssp': ssp,
        **{k: v for k, v in d1.items() if isinstance(v, (int, float, bool))},
        'clipping_ol': d2_list[0]['clipping_frac'],
        **{f'clipping_rw{r["R_w"]:.0e}': r['clipping_frac']
           for r in d2_list if r['R_w'] is not None},
        **{k: v for k, v in d3.items() if isinstance(v, (int, float))},
        **{f'reach_{k}': v for k, v in d4.items() if isinstance(v, (int, float))},
        **{f'feas_{k}': v for k, v in d5.items() if isinstance(v, (int, float))},
    }

    # Clean up Z_cl from d2_list to free memory
    for r in d2_list:
        r.pop('Z_cl', None)
        r.pop('U_cl', None)
        r.pop('M_cl_clip', None)

    del X, U, pod, dmd, Z_open, Z_closed, U_ctrl_z
    gc.collect()

    return summary_row


SUMMARY_CSV = OUT_DIR / "mass_conservation_summary.csv"


def _load_existing_summary() -> pd.DataFrame:
    """Load existing summary CSV if it exists."""
    if SUMMARY_CSV.exists():
        return pd.read_csv(SUMMARY_CSV)
    return pd.DataFrame()


def _save_summary(new_rows: List[Dict]):
    """Append new rows to summary CSV, replacing duplicates by (gcm, hydro, ssp)."""
    if not new_rows:
        return

    df_new = pd.DataFrame(new_rows)
    df_old = _load_existing_summary()

    if len(df_old) > 0:
        # Remove old entries that match new (gcm, hydro, ssp) combos
        keys = df_new[['gcm', 'hydro', 'ssp']].drop_duplicates()
        for _, row in keys.iterrows():
            mask = (
                (df_old['gcm'] == row['gcm']) &
                (df_old['hydro'] == row['hydro']) &
                (df_old['ssp'] == row['ssp'])
            )
            df_old = df_old[~mask]

        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.sort_values(['hydro', 'gcm', 'ssp'], inplace=True)
    df_all.to_csv(SUMMARY_CSV, index=False)
    print(f"\n  Summary saved: {SUMMARY_CSV}  ({len(df_all)} total rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcm", default="ACCESS-CM2",
                        help="GCM name or 'all'")
    parser.add_argument("--hydro", default="VIC5",
                        help="Hydro model or 'all'")
    parser.add_argument("--ssp", default="245",
                        help="SSP scenario: 245, 585, 'all', or comma-separated e.g. '245,585'")
    args = parser.parse_args()

    gcms = AVAILABLE_GCMS if args.gcm == "all" else [args.gcm]
    hydros = HYDRO_MODELS if args.hydro == "all" else [args.hydro]

    # Parse SSP: support 'all', single int, or comma-separated
    if args.ssp.lower() == "all":
        ssps = sorted(SCENARIOS.keys()) if isinstance(SCENARIOS, dict) else SCENARIOS
    else:
        ssps = [int(s.strip()) for s in args.ssp.split(",")]

    print(f"\n{'='*70}")
    print(f"  MASS CONSERVATION DIAGNOSTIC v4")
    print(f"  GCMs:  {gcms}")
    print(f"  Hydro: {hydros}")
    print(f"  SSPs:  {ssps}")
    print(f"  R_w tested: {RW_VALUES}")
    print(f"  Total combos: {len(gcms) * len(hydros) * len(ssps)}")
    print(f"{'='*70}")

    rows = []
    for ssp in ssps:
        for h in hydros:
            for g in gcms:
                try:
                    rows.append(run_diagnostic(g, h, ssp))
                except Exception as e:
                    print(f"  ERROR {h}/{g}/SSP{ssp}: {e}")
                    import traceback
                    traceback.print_exc()
                gc.collect()

    _save_summary(rows)

    # Print grand summary
    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'='*70}")
        print(f"  RUN COMPLETE: {len(df)} members")
        print(f"{'='*70}")
        for col in ['vol_ratio', 'annual_r2', 'delta_corr',
                     'feas_median_mean_ratio_flow', 'feas_pct_reaches_flow_ok']:
            if col in df.columns:
                vals = df[col]
                print(f"  {col}: median={vals.median():.4f}  "
                      f"range=[{vals.min():.4f}, {vals.max():.4f}]")
        print(f"{'='*70}\n")


# ============================================================
# FIG S5 — Surrogate Mass Fidelity (1×3)
# ============================================================

FIG_DIR = Path("figures_wrr_new")

SSP_LABELS = {126:"SSP1-2.6", 245:"SSP2-4.5", 370:"SSP3-7.0", 585:"SSP5-8.5"}
SSP_COLORS = {126:"#2d8e3d", 245:"#2166AC", 370:"#d4820a", 585:"#c72e29"}

def _jitter_strip(ax, x_pos, values, color, width=0.25, ms=3.5, alpha=0.5):
    jit = np.random.default_rng(42).uniform(-width/2, width/2, len(values))
    ax.scatter(x_pos + jit, values, s=ms, alpha=alpha, color=color,
               edgecolors="black", linewidths=0.15, zorder=3, rasterized=True)

def _dot_whisker(ax, x_pos, values, color, lw=2.0, ms=7):
    med = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    ax.plot([x_pos, x_pos], [q25, q75], color=color, lw=lw,
            solid_capstyle="round", zorder=4)
    ax.plot(x_pos, med, "o", color=color, ms=ms,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    return med


def plot_figS5(csv_path=None):
    """Plot FIG S5 — per-SSP dot-whisker: vol ratio, δ consistency, per-reach δ."""
    import matplotlib as mpl
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MM = 1/25.4; W_FULL = 190*MM

    mpl.rcParams.update({
        "font.family":"Arial", "font.size":7,
        "axes.labelsize":7, "axes.titlesize":7, "axes.linewidth":0.5,
        "axes.spines.top":False, "axes.spines.right":False,
        "xtick.labelsize":6, "ytick.labelsize":6,
        "xtick.major.width":0.4, "ytick.major.width":0.4,
        "xtick.major.size":2.5, "ytick.major.size":2.5,
        "xtick.direction":"out", "ytick.direction":"out",
        "legend.fontsize":5.5, "legend.frameon":False,
        "figure.dpi":200, "savefig.dpi":300,
        "savefig.bbox":"tight", "savefig.pad_inches":0.02,
        "pdf.fonttype":42, "ps.fonttype":42, "lines.linewidth":0.8,
    })
    _GA = 0.08; _PX = -0.10; _PY = 1.05

    def lgrid(ax):
        ax.grid(True, alpha=_GA, lw=0.3, zorder=0); ax.set_axisbelow(True)
    def plabel(ax, lab):
        ax.text(_PX, _PY, lab, transform=ax.transAxes,
                fontweight="bold", fontsize=9, va="top", ha="right")

    if csv_path is None:
        csv_path = SUMMARY_CSV
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} members from {csv_path}")
    scenarios = sorted(df["ssp"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 70*MM),
                              gridspec_kw={"wspace": 0.36})

    # (a) Volume ratio
    ax = axes[0]
    ax.axhline(1.0, color="#777", ls="--", lw=0.6, zorder=1)
    for i, scn in enumerate(scenarios):
        vals = df[df["ssp"]==scn]["vol_ratio"].dropna().values
        _jitter_strip(ax, i, vals, SSP_COLORS[scn], ms=4, alpha=0.55)
        _dot_whisker(ax, i, vals, SSP_COLORS[scn])
    all_vr = df["vol_ratio"].dropna().values
    ax.text(0.97, 0.05, f"all: {np.median(all_vr):.3f}\n(n={len(all_vr)})",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#999", lw=0.3, alpha=0.85))
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SSP_LABELS[s] for s in scenarios], fontsize=5.5)
    ax.set_ylabel("Volume ratio\n(surrogate / reference)")
    ax.set_xlim(-0.5, len(scenarios)-0.5); ax.set_ylim(0.95, 1.05)
    lgrid(ax); plabel(ax, "a")

    # (b) δ consistency: r + R²
    ax = axes[1]
    ax.axhline(1.0, color="#777", ls="--", lw=0.6, zorder=1)
    offset = 0.14
    for i, scn in enumerate(scenarios):
        sub = df[df["ssp"]==scn]; c = SSP_COLORS[scn]
        for j, (col, marker, ms, label) in enumerate([
            ("delta_corr", "o", 6, "Correlation $r$"),
            ("delta_r2",   "s", 5.5, "$R^2$"),
        ]):
            vals = sub[col].dropna().values
            xpos = i + (j - 0.5) * offset * 2
            _jitter_strip(ax, xpos, vals, c, width=0.12, ms=3, alpha=0.4)
            med = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            ax.plot([xpos, xpos], [q25, q75], color=c, lw=1.5,
                    solid_capstyle="round", zorder=3, alpha=0.7 if j==1 else 1.0)
            ax.plot(xpos, med, marker, color=c, ms=ms,
                    markeredgecolor="black", markeredgewidth=0.5,
                    zorder=4, label=label if i==0 else "")
    all_r  = df["delta_corr"].dropna().values
    all_r2 = df["delta_r2"].dropna().values
    ax.text(0.97, 0.05, f"$r$: {np.median(all_r):.3f}\n$R^2$: {np.median(all_r2):.3f}",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#999", lw=0.3, alpha=0.85))
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SSP_LABELS[s] for s in scenarios], fontsize=5.5)
    ax.set_ylabel(r"Basin-total $\delta$ consistency")
    ax.set_xlim(-0.5, len(scenarios)-0.5); ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=5, handlelength=1.0)
    lgrid(ax); plabel(ax, "b")

    # (c) Per-reach δ correlation median
    ax = axes[2]
    ax.axhline(1.0, color="#777", ls="--", lw=0.6, zorder=1)
    for i, scn in enumerate(scenarios):
        vals = df[df["ssp"]==scn]["reach_median"].dropna().values
        _jitter_strip(ax, i, vals, SSP_COLORS[scn], ms=4, alpha=0.55)
        _dot_whisker(ax, i, vals, SSP_COLORS[scn])
    all_rm = df["reach_median"].dropna().values
    ax.text(0.97, 0.05, f"all: {np.median(all_rm):.3f}\n(n={len(all_rm)})",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#999", lw=0.3, alpha=0.85))
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([SSP_LABELS[s] for s in scenarios], fontsize=5.5)
    ax.set_ylabel("Per-reach $\\delta$ correlation\n(median across reaches)")
    ax.set_xlim(-0.5, len(scenarios)-0.5); ax.set_ylim(-0.05, 1.05)
    lgrid(ax); plabel(ax, "c")

    for ext in ("png","pdf"): fig.savefig(FIG_DIR/f"figS2_mass_fidelity.{ext}")
    plt.close(fig)
    print(f"  Saved figS2_mass_fidelity to {FIG_DIR}")


if __name__ == "__main__":
    main()
    # Plot if summary CSV exists
    if SUMMARY_CSV.exists():
        print(f"\nPlotting FIG S5...")
        plot_figS5()