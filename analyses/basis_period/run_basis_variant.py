"""Allocation with the POD basis, temporal mean and forcing normalization taken from the first N years only.

Everything else is the paper's procedure: windowed DMDc operators identified within each 10-year window (5-year step)
on the restricted basis, one LQR gain per window and penalty weight, gated closed loop from 2020-01-01, and effort over
2030-2099 matched to 10 Gm3/yr by interpolation between the bracketing penalty weights. No operator is applied outside
its window; the only change from the production arm of run_member.py is that the basis, mean and forcing statistics
never see the years after the basis period. The open-loop (uncontrolled) reconstruction skill over 2030-2099 and the
relative projection error of the restricted basis over those years are recorded.
Raw inputs are read only (FAC_RAW_ROOT). Writes <FAC_WORK_ROOT>/basis_period/results_basis/<member>_b<N>.npz and .json.
Usage: python run_basis_variant.py --case 0 --basis-years 10 [--threads 6]
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))   # analyses/ (common)
sys.path.insert(0, str(HERE.parents[1]))   # repository root (paths)
from run_member import CASES, DT, GRID, RANK, TEST_START, TRAIN_TRANSITIONS, gains_for, log, match_budget, openloop, simulate, skill  # noqa: E402
from paths import work  # noqa: E402
from common.pipeline_loader import production, raw_inputs, return_levels_path  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--basis-years", type=int, default=10)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    threadpool_limits(limits=a.threads)
    h, g = CASES[a.case]
    key = f"{h}_{g}"
    outdir = work("basis_period") / "results_basis"
    outdir.mkdir(exist_ok=True)
    out = outdir / f"{key}_b{a.basis_years}.npz"
    if out.exists():
        log("already done", out.name)
        return
    tick = time.time()
    ns = production()
    assert ns["RANK"] == RANK and ns["WINDOW"] == TRAIN_TRANSITIONS and ns["STEP"] == 1825
    assert np.allclose(ns["R_VALUES"], GRID)

    # inputs and thresholds, exactly as run_member.py
    with raw_inputs():
        X, U, dates, reach_ids = ns["load_scenario_data"](245, g, h)
    rl = np.load(return_levels_path(h, g))
    idx_ref, idx_this, matched = ns["match_to_reference"](rl["reach_ids"].astype(int).tolist(), reach_ids)
    if not np.array_equal(idx_this, np.arange(len(reach_ids))):
        X, U = X[idx_this], U[idx_this]
    thr = np.maximum(rl["rl2"].astype(np.float64), 0.0)[idx_ref]
    assert dates[0] == pd.Timestamp("2020-01-01") and dates[-1] == pd.Timestamp("2099-12-31")
    T = X.shape[1]
    nt = T - 1
    s = int(dates.searchsorted(pd.Timestamp(f"{TEST_START}-01-01")))
    sb = int(dates.searchsorted(pd.Timestamp(f"{2020 + a.basis_years}-01-01")))
    ny_test = len(np.unique(dates[s:nt].year))
    process_peak = float(X[:, :nt].max())
    log("loaded", key, X.shape, "basis period", dates[0].date(), "to", dates[sb - 1].date(), round(time.time() - tick, 1), "s")

    # basis, mean and forcing statistics from the basis period only
    t = time.time()
    mu_b = U[:, :sb].mean(axis=1, keepdims=True)
    sig_b = float(np.std(U[:, :sb] - mu_b) + 1e-12)
    pod_b = ns["compute_pod"](np.ascontiguousarray(X[:, :sb]), RANK)
    Zb = np.empty((RANK, T))
    for lo in range(0, T, 2048):
        Zb[:, lo:lo + 2048] = pod_b.U_r.T @ (X[:, lo:lo + 2048] - pod_b.mean_x)
    num = den = 0.0                                                   # relative L2 projection error over the test years
    for lo in range(s, nt, 1024):
        blk = X[:, lo:min(lo + 1024, nt)] - pod_b.mean_x
        num += float(((blk - pod_b.U_r @ (pod_b.U_r.T @ blk)) ** 2).sum())
        den += float((blk ** 2).sum())
    proj_err_test = float(np.sqrt(num / den))
    U -= mu_b
    U /= sig_b
    dmd = ns["fit_windowed_dmdc"](Zb, U)                              # the paper's windows, on the restricted basis
    force = np.empty((RANK, nt))
    cuts = np.r_[0, np.flatnonzero(np.diff(dmd.window_for_k)) + 1, nt]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        w = dmd.window_for_k[lo]
        force[:, lo:hi] = dmd.E_list[w] @ U[:, lo:hi] + dmd.b_list[w]
    del U
    gc.collect()
    log("fit", round(time.time() - t, 1), "s; alphas", dmd.alphas_used, "max rho", round(max(dmd.spectral_radii), 4),
        "projection error 2030-2099", round(proj_err_test, 4))

    # closed loop from 2020 as in production; effort over 2030-2099
    t = time.time()
    K = gains_for(ns, dmd.A_list)
    ok = [j for j in range(len(GRID)) if all(K[w][j] is not None for w in range(len(dmd.A_list)))]
    windows = dmd.window_for_k.astype(np.int32)
    st1, z_s = simulate(pod_b.U_r, pod_b.mean_x, thr, Zb[:, 0], dmd.A_list, K, ok, force[:, :s], windows[:s])
    st2 = np.full_like(st1, np.nan)
    for j in ok:
        if np.isfinite(z_s[j]).all():
            st, _ = simulate(pod_b.U_r, pod_b.mean_x, thr, z_s[j], dmd.A_list, K, [j], force[:, s:], windows[s:])
            st2[:, j] = st[:, j]
    eff_test, peak_test = st2[0] * DT / ny_test, st2[5]
    log("closed loop", round(time.time() - t, 1), "s")

    z_ol = openloop(dmd.A_list, force, windows, Zb[:, 0])
    nse, kge = skill(ns, X, pod_b.U_r, pod_b.mean_x, z_ol[:, s:], s, nt)
    e_b, i_b = match_budget(eff_test, peak_test, process_peak)
    np.savez_compressed(out, reach_ids=np.asarray(matched, dtype=np.int64), effort_basis_test=e_b,
                        nse_basis=nse, kge_basis=kge)
    out.with_suffix(".json").write_text(json.dumps(dict(
        case=key, basis_years=a.basis_years, basis_period=[str(dates[0].date()), str(dates[sb - 1].date())],
        alphas=[float(v) for v in dmd.alphas_used], spectral_radii=[float(v) for v in dmd.spectral_radii],
        projection_error_test=proj_err_test, budget=i_b, nse_median=float(np.nanmedian(nse)),
        kge_median=float(np.nanmedian(kge)), seconds=time.time() - tick), indent=1))
    log("done", key, "status", i_b["status"], "nse", round(float(np.nanmedian(nse)), 3), "seconds", round(time.time() - tick, 1))


if __name__ == "__main__":
    main()
