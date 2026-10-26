"""Production allocation of one SSP2-4.5 member: the reference for the basis-period test and the cross-member agreement.

The 10 Gm3/yr reference allocation computed with the production functions of the pipeline (04_dmd_lqr_analysis.py)
and the production thresholds (pooled 1980-2019 2-year return levels): POD basis, mean and forcing normalization from
2020-2099; windowed DMDc exactly as in the paper (each 5-year block uses the 10-year window that starts with it); closed
loop from 2020. Effort is matched to 10 Gm3/yr by linear interpolation between the two bracketing penalty weights of the
production grid, as in the paper, once over 2030-2099 (effort_production_test, compared with the basis-period variant
in summarize_basis.py) and once over 2020-2099 (effort_production_full, used by cross_member/summarize_cross_member.py).
The open-loop (uncontrolled) reach NSE and KGE of the surrogate over 2030-2099 are recorded (main-analysis row of
Table S1, bottom block).

Raw inputs are read only (FAC_RAW_ROOT). Writes <FAC_WORK_ROOT>/basis_period/results/<member>.npz and .json.
Usage: python run_member.py --case 0 [--threads 6]
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")   # the compiled kernel uses Accelerate; one thread per process
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # analyses/ (common)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import work  # noqa: E402
from common import kernel  # noqa: E402
from common.pipeline_loader import production, raw_inputs, return_levels_path  # noqa: E402

GCMS = ["MPI-ESM1-2-HR", "ACCESS-CM2", "EC-Earth3", "CNRM-ESM2-1", "BCC-CSM2-MR", "MRI-ESM2-0", "NorESM2-MM"]
CASES = [(h, g) for g in GCMS for h in ["PRMS", "VIC5"]]
RANK = 30
GRID = np.logspace(-4, 5, 19)
TARGET = 10.0               # Gm3/yr
TEST_START = 2030
TRAIN_TRANSITIONS = 3650    # production window length (365 * 10)
DT = 86400.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gains_for(ns, A_list):
    """K[w][j] for every window and penalty weight; None where the DARE fails."""
    eye = np.eye(RANK)
    out = []
    for A in A_list:
        row = []
        for rho in GRID:
            try:
                row.append(ns["lqr_gain"](A, eye, float(rho)))
            except Exception:
                row.append(None)
        out.append(row)
    return out


def simulate(basis, mu, thr, z0, A_list, K, policies, force, windows):
    """Gated closed loop for the given policies from one shared initial state; failed policies are split off."""
    n = basis.shape[0]
    stats = np.full((9, len(GRID), n), np.nan)
    zend = np.full((len(GRID), RANK), np.nan)
    years = np.zeros(force.shape[1], dtype=np.int32)

    def run(idx):
        try:
            g = [[K[w][j] for j in idx] for w in range(len(A_list))]
            st, _, zz, _, _ = kernel.simulate(basis, mu, thr, z0, A_list, g, force, windows, years,
                                              annual=False, trace=True)
            for k, j in enumerate(idx):
                stats[:, j] = st[:, k]
                zend[j] = zz[-1, k]
        except FloatingPointError:
            if len(idx) > 1:
                run(idx[: len(idx) // 2])
                run(idx[len(idx) // 2:])
    run([j for j in policies])
    return stats, zend


def match_budget(effort, peak, process_peak):
    """Interpolate reach effort between the two valid policies bracketing the target budget."""
    budget = effort.sum(1) / 1e9
    ratio = peak.max(1) / process_peak
    bad = ~np.isfinite(budget) | ~np.isfinite(ratio) | (ratio > 1e6) | ~np.isfinite(effort).all(1)
    order = np.flatnonzero(~bad)
    order = order[np.argsort(budget[order])]
    info = dict(valid_policies=int(len(order)), budgets=[None if not np.isfinite(b) else float(b) for b in budget],
                status="no_bracket")
    if len(order) >= 2 and budget[order[0]] <= TARGET <= budget[order[-1]]:
        j = int(np.searchsorted(budget[order], TARGET))
        j = max(1, min(j, len(order) - 1))
        lo, hi = order[j - 1], order[j]
        w = float((TARGET - budget[lo]) / (budget[hi] - budget[lo]))
        gap = bool(bad[min(lo, hi):max(lo, hi) + 1].any())
        info.update(status="runaway_gap" if gap else "interpolated", lower_rho=float(GRID[lo]), upper_rho=float(GRID[hi]),
                    lower_budget=float(budget[lo]), upper_budget=float(budget[hi]), weight=w)
        if not gap:
            e = (1 - w) * effort[lo] + w * effort[hi]
            assert abs(e.sum() / 1e9 - TARGET) < 1e-8
            return e, info
    return np.full(effort.shape[1], np.nan), info


def openloop(A_list, force, windows, z0):
    z = np.empty((RANK, force.shape[1] + 1))
    z[:, 0] = z0
    for k in range(force.shape[1]):
        z[:, k + 1] = A_list[windows[k]] @ z[:, k] + force[:, k]
    return z


def skill(ns, X, basis, mean, z, lo, hi):
    """Reach NSE and KGE of the clipped surrogate flow against the process-based flow over days lo..hi-1."""
    n = X.shape[0]
    nse, kge = np.empty(n), np.empty(n)
    for a in range(0, n, 384):
        b = min(a + 384, n)
        obs = X[a:b, lo:hi]
        sim = np.maximum(basis[a:b] @ z[:, : hi - lo] + mean[a:b], 0.0)
        nse[a:b] = ns["_nse"](obs, sim)
        kge[a:b] = ns["_kge"](obs, sim)
    return nse, kge


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=int, required=True)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    threadpool_limits(limits=a.threads)
    h, g = CASES[a.case]
    key = f"{h}_{g}"
    outdir = work("basis_period") / "results"
    outdir.mkdir(exist_ok=True)
    out = outdir / f"{key}.npz"
    if out.exists():
        log("already done", key)
        return
    tick = time.time()
    ns = production()
    assert ns["RANK"] == RANK and ns["WINDOW"] == TRAIN_TRANSITIONS and ns["STEP"] == 1825
    assert np.allclose(ns["R_VALUES"], GRID)

    # ---------------- inputs and thresholds, exactly as process_single_scenario
    with raw_inputs():
        X, U, dates, reach_ids = ns["load_scenario_data"](245, g, h)
    rl = np.load(return_levels_path(h, g))
    idx_ref, idx_this, matched = ns["match_to_reference"](rl["reach_ids"].astype(int).tolist(), reach_ids)
    if not np.array_equal(idx_this, np.arange(len(reach_ids))):
        X, U = X[idx_this], U[idx_this]
    thr = np.maximum(np.maximum(rl["rl2"].astype(np.float64), 0.0)[idx_ref], 0.0)
    assert dates[0] == pd.Timestamp("2020-01-01") and dates[-1] == pd.Timestamp("2099-12-31")
    T = X.shape[1]
    nt = T - 1                                                      # production simulates T-1 steps
    s = int(dates.searchsorted(pd.Timestamp(f"{TEST_START}-01-01")))
    ny_test = len(np.unique(dates[s:nt].year))
    ny_full = len(np.unique(dates[:nt].year))
    process_peak = float(X[:, :nt].max())
    log("loaded", key, X.shape, "test from", dates[s].date(), "years", ny_test, round(time.time() - tick, 1), "s")

    # ---------------- production fit
    t = time.time()
    pod = ns["compute_pod"](X, RANK)
    V = ns["normalize_forcing"](U).V
    dmd = ns["fit_windowed_dmdc"](pod.Z, V)
    force = np.empty((RANK, nt))
    cuts = np.r_[0, np.flatnonzero(np.diff(dmd.window_for_k)) + 1, nt]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        w = dmd.window_for_k[lo]
        force[:, lo:hi] = dmd.E_list[w] @ V[:, lo:hi] + dmd.b_list[w]
    del V, U
    gc.collect()
    log("production fit", round(time.time() - t, 1), "s; alphas", dmd.alphas_used, "max rho", round(max(dmd.spectral_radii), 4))

    # ---------------- closed loop from 2020; effort over 2030-2099 and 2020-2099
    t = time.time()
    K = gains_for(ns, dmd.A_list)
    ok = [j for j in range(len(GRID)) if all(K[w][j] is not None for w in range(len(dmd.A_list)))]
    windows = dmd.window_for_k.astype(np.int32)
    st1, z_s = simulate(pod.U_r, pod.mean_x, thr, pod.Z[:, 0], dmd.A_list, K, ok, force[:, :s], windows[:s])
    st2 = np.full_like(st1, np.nan)
    for j in ok:
        if np.isfinite(z_s[j]).all():
            st, _ = simulate(pod.U_r, pod.mean_x, thr, z_s[j], dmd.A_list, K, [j], force[:, s:], windows[s:])
            st2[:, j] = st[:, j]
    eff_prod_test, peak_prod_test = st2[0] * DT / ny_test, st2[5]
    eff_prod_full, peak_prod_full = (st1[0] + st2[0]) * DT / ny_full, np.fmax(st1[5], st2[5])
    log("production closed loop", round(time.time() - t, 1), "s")

    z_ol = openloop(dmd.A_list, force, windows, pod.Z[:, 0])
    nse_prod, kge_prod = skill(ns, X, pod.U_r, pod.mean_x, z_ol[:, s:], s, nt)
    prod_alphas, prod_radii = list(dmd.alphas_used), list(dmd.spectral_radii)

    # ---------------- budget matching and saving
    e_pt, i_pt = match_budget(eff_prod_test, peak_prod_test, process_peak)
    e_pf, i_pf = match_budget(eff_prod_full, peak_prod_full, process_peak)
    np.savez_compressed(out, reach_ids=np.asarray(matched, dtype=np.int64),
                        effort_production_test=e_pt, effort_production_full=e_pf,
                        nse_production_test=nse_prod, kge_production_test=kge_prod)
    meta = dict(case=key, test_start=str(dates[s].date()), test_end=str(dates[nt - 1].date()), years_test=ny_test,
                production=dict(alphas=prod_alphas, spectral_radii=prod_radii, budget=i_pt, budget_full=i_pf),
                median_skill_test=dict(nse_production=float(np.nanmedian(nse_prod)), kge_production=float(np.nanmedian(kge_prod))),
                seconds=round(time.time() - tick, 1))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log("done", key, "status", i_pt["status"], i_pf["status"], "seconds", meta["seconds"])


if __name__ == "__main__":
    main()
