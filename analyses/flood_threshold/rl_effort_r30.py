"""Threshold sensitivity at the production configuration (Table S2, rerun path).

Rebuilds the surrogate of one SSP2-4.5 member exactly as in 04_dmd_lqr_analysis.py (r = 30, 10/5-year
windows, original ridge grid and spectral-radius rule), runs the threshold-triggered closed loop for
all 19 penalty weights with the flood threshold set to the T-year return level of the pooled
1980-2019 annual maxima (the same cache as the production run), and saves the per-reach effort
eff_total_L1 (m3/yr) for every weight. Only the threshold differs from the production RL2 run.

Inputs: FAC_RAW_ROOT/streamflow and FAC_RAW_ROOT/precipitation (ORNL HydroSource, SSP2-4.5, 2020-2099) and the pooled
return levels of the outputs_common archive. Output: effort_rl{T}_r30_{hydro}_{gcm}_ssp{scenario}.npz in --out-dir
(default FAC_WORK_ROOT/flood_threshold/effort) with reach_id, R_values, eff (19 x reaches), thr, rl, spectral_radii, alphas.
Usage: python rl_effort_r30.py --hydro VIC5 --gcm MPI-ESM1-2-HR --rl 5
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # analyses/ (common)
sys.path.insert(0, str(HERE.parents[1]))   # repository root (paths)
from paths import work  # noqa: E402
from common.pipeline_loader import raw_inputs  # noqa: E402
import surrogate_loop as rr  # noqa: E402  (load_pipeline, closed_loop_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hydro', required=True)
    ap.add_argument('--gcm', required=True)
    ap.add_argument('--rl', type=int, default=5)
    ap.add_argument('--scenario', type=int, default=245)
    ap.add_argument('--out-dir', type=Path, default=None, help='default: FAC_WORK_ROOT/flood_threshold/effort')
    a = ap.parse_args()
    OUT = a.out_dir if a.out_dir is not None else work('flood_threshold') / 'effort'
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'effort_rl{a.rl}_r30_{a.hydro}_{a.gcm}_ssp{a.scenario}.npz'
    if out.exists():
        print('exists', out); return
    t0 = time.time()
    m = rr.load_pipeline()
    m.PROF = m.Profiler()
    assert m.RANK == 30 and m.WINDOW == 3650 and m.STEP == 1825
    cache = np.load(m._cache_path_for_pooled(a.gcm, a.hydro), allow_pickle=True)
    pooled_ids = cache['reach_ids'].astype(int).tolist()
    with raw_inputs():
        X, U, dates, reach_ids = m.load_scenario_data(a.scenario, a.gcm, a.hydro)
    idx_ref, idx_this, matched = m.match_to_reference(pooled_ids, reach_ids)
    X, U = X[idx_this], U[idx_this]
    thr = np.maximum(cache[f'rl{a.rl}'].astype(m.DTYPE)[idx_ref], 0.0)
    n, T_full = X.shape
    pod = m.compute_pod(X, m.RANK)
    del X; gc.collect()
    forcing = m.normalize_forcing(U)
    del U; gc.collect()
    dmd = m.fit_windowed_dmdc(pod.Z, forcing.V)
    V_sim = forcing.V[:, :-1]
    Tm1 = V_sim.shape[1]
    D = np.empty((m.RANK, Tm1))
    for w in range(len(dmd.A_list)):
        ks = np.where(dmd.window_for_k[:Tm1] == w)[0]
        if len(ks):
            D[:, ks] = dmd.E_list[w] @ V_sim[:, ks] + dmd.b_list[w]
    del forcing, V_sim; gc.collect()
    Q = (m.Q_WEIGHT * np.eye(m.RANK)).astype(m.DTYPE)
    R_values = [float(r) for r in m.R_VALUES]
    K_by_window = [np.stack([m.lqr_gain(A, Q, Rw) for Rw in R_values]) for A in dmd.A_list]
    controls = rr.closed_loop_batch(dmd.A_list, K_by_window, dmd.window_for_k, D, pod.Z[:, 0:1], pod.U_r,
                                    pod.mean_x, thr)
    T_ctrl = min(T_full - 1, Tm1)
    years = np.asarray(dates.year)[:T_ctrl]
    n_years = len(np.unique(years))
    eff = np.zeros((len(R_values), n))
    for a0 in range(0, T_ctrl, 2000):
        b0 = min(a0 + 2000, T_ctrl)
        for j in range(len(R_values)):
            eff[j] += np.abs(pod.U_r @ controls[j][:, a0:b0]).sum(axis=1)
    eff *= m.DT / n_years
    np.savez_compressed(out, reach_id=np.asarray(matched, dtype=np.int64), R_values=np.array(R_values), eff=eff,
                        thr=thr, rl=a.rl, spectral_radii=np.array(dmd.spectral_radii), alphas=np.array(dmd.alphas_used))
    print(f'{a.hydro} {a.gcm} rl{a.rl}: basin effort by R (Gm3/yr) =', np.round(eff.sum(axis=1) / 1e9, 3).tolist(),
          f'| {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
