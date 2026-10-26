"""Closed-loop test of the production controller on the routing reference (Figure S6).

The production control law, u_z(k) = -K_w(k) U_r^T [x(k) - tau]^+, is applied with the saved gains,
windows, POD basis, and thresholds of each member at the 10 Gm3/yr budget. The state it observes is the
routing-reference discharge x_ref(k) = x_sim(k) + delta(k), and the lifted input is propagated with the
daily-coordinate recurrence delta(k+1) = F^96 delta(k) + U_r u_z(k) (Muskingum operator F of the RAPID network):
u_z(k) is recomputed every day from x_ref(k), and U_r u_z(k) is added once to the routed state at the end of the day.

Inputs are read-only: the RAPID network extract (layout.RAPID_NETWORK), the controller (recover_member.py) and the
uncontrolled discharge (make_q_ornl.py); --controller, --q-ornl and --out default to the work folders of layout.py.
Writes <tag>.json (penalty weight and effort of the run and of the surrogate), <tag>_reach.npz (residual exceedance
above the threshold and effort per reach, daily network effort) and <tag>_mainstem.npy (routed controlled discharge
at the mainstem reach). With --fidelity, the run also compares the routed controlled discharge with the surrogate's
own controlled discharge U_r z_controlled + mean (saved in the controller) day by day: per-reach sums for daily NSE
and KGE of Q+ = max(Q, 0), and the surrogate's controlled mainstem series (<tag>_fidelity.npz; run_allssp_fidelity.py).
Usage: python closed_loop.py --case VIC5_MPI-ESM1-2-HR_ssp245_budget10 [--R 100] [--fidelity]
"""
import argparse
import ctypes as ct
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # analyses/ (common)
import layout  # noqa: E402
from common.native import load_library  # noqa: E402
from rapid_model import RAPID, dp, ip  # noqa: E402

REFERENCE_NET = layout.RAPID_NETWORK
VOLUME = 86400.0 / 1e9  # m3/s-day -> Gm3
MAINSTEM = 3125212
OUT = layout.GRID

lib = load_library("matched_chunk")
lib.matched_chunk.argtypes = [ct.c_int, ct.c_int, ct.c_int, ip, ip, dp, dp, dp, dp, dp, dp, dp]
lib.matched_chunk.restype = None


def route_day(r, u_full, state):
    """One day of the matched recurrence: returns F^96 state + u_full."""
    out = np.empty((1, r.n))
    end = np.empty(r.n)
    lib.matched_chunk(r.n, 1, r.steps, r.order, r.down, r.c1, r.c2, r.c3,
                      np.ascontiguousarray(u_full[None, :]), np.ascontiguousarray(state), out, end)
    return end


def gains(A, R):
    """LQR gains with Q = I, B = I (reduced coordinates), as in the production code."""
    r = A.shape[1]
    I = np.eye(r)
    K = np.empty_like(A)
    for w in range(A.shape[0]):
        P = solve_discrete_are(A[w], I, I, R * I)
        K[w] = np.linalg.solve(R * I + P, P @ A[w])
    return K


def run(case, R=None, controller=None, q_ornl=None, out=OUT, fidelity=False):
    c = np.load(controller or layout.controller(case))
    U = np.ascontiguousarray(c["U_r"], dtype=np.float64)
    UT = np.ascontiguousarray(U.T)
    mean = c["mean_q"].astype(np.float64).ravel()
    tau = c["threshold_rl2_m3s"].astype(np.float64).ravel()
    win = c["window_for_k"].astype(np.int64)
    u_saved = c["u_reduced"]
    R_prod = float(c["R_weight"])  # weight of the saved production run (recover_member.py)
    K = c["K_lqr"].astype(np.float64)
    if R is not None and not np.isclose(R, R_prod):
        K = gains(c["A"].astype(np.float64), R)
    else:
        R = R_prod
        # the stored gains are the DARE solution for the production weight
        assert np.allclose(gains(c["A"][:1].astype(np.float64), R_prod)[0], K[0], rtol=1e-8, atol=1e-12)
    # the control law reproduces the saved surrogate inputs when it observes the surrogate state
    zc = c["z_controlled"]
    for k in (0, 1000, 5000, 12000, 20000, 29000):
        xh = U @ zc[:, k] + mean
        uz = -(K[win[k]] @ (UT @ np.maximum(xh - tau, 0.0)))
        if np.isclose(R, R_prod):
            assert np.allclose(uz, u_saved[:, k], rtol=1e-6, atol=1e-9 * (1 + np.abs(u_saved[:, k]).max()))

    q0 = np.load(q_ornl or layout.q_ornl(case), mmap_mode="r")
    T, n = q0.shape
    with np.load(REFERENCE_NET) as z:
        net = {k: np.array(z[k]) for k in ["comid", "downstream_comid", "K_seconds", "x"]}
    rt = RAPID(net["comid"], net["downstream_comid"], net["K_seconds"], net["x"])
    pos = {int(cid): i for i, cid in enumerate(net["comid"].astype(np.int64))}
    ids = c["comid"].astype(np.int64)
    ix = np.array([pos[int(cid)] for cid in ids], dtype=np.int64)
    ms = int(np.flatnonzero(ids == MAINSTEM)[0])

    state = np.zeros(rt.n)  # end-of-day routed change
    u_full = np.zeros(rt.n)
    exc_cl = np.zeros(n)
    eff = np.zeros(n)
    ms_cl = np.empty(T)  # routed controlled discharge at the mainstem
    daily_eff = np.zeros(T - 1)  # network-summed |input| per control day, m3/s
    if fidelity:
        assert zc.shape[1] == T
        # sums of (o - m), (o - m)^2, (s - m), (s - m)^2, (o - m)(s - m), (s - o)^2 with o the routed reference and
        # s the surrogate, both as Q+ = max(Q, 0), shifted by the time-mean m for numerical stability
        acc = np.zeros((6, n))
        ms_sur = np.empty(T)  # surrogate controlled discharge at the mainstem
    started = time.monotonic()
    for k in range(T):
        x0 = np.asarray(q0[k], dtype=np.float64)
        x = x0 + state[ix]
        exc_cl += np.maximum(x - tau, 0.0)
        ms_cl[k] = x[ms]
        if fidelity:
            s_cl = U @ zc[:, k] + mean
            o, sv = np.maximum(x, 0.0), np.maximum(s_cl, 0.0)
            do, ds = o - mean, sv - mean
            acc[0] += do
            acc[1] += do * do
            acc[2] += ds
            acc[3] += ds * ds
            acc[4] += do * ds
            acc[5] += (sv - o) ** 2
            ms_sur[k] = s_cl[ms]
        if k == T - 1:
            break
        e = np.maximum(x - tau, 0.0)
        uz = -(K[win[k]] @ (UT @ e))
        ureach = U @ uz
        eff += np.abs(ureach)
        daily_eff[k] = np.abs(ureach).sum()
        u_full[ix] = ureach
        state = route_day(rt, u_full, state)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"non-finite routed state at day {k}")
    secs = time.monotonic() - started
    eff_surr = float(np.abs(U @ u_saved).sum() * VOLUME / 80.0)
    summary = dict(
        case=case, R_weight=R, R_production=R_prod, days=T, reaches=n,
        effort_Gm3_per_yr=float(eff.sum() * VOLUME / 80.0),
        surrogate_effort_same_R_Gm3_per_yr=eff_surr,
        seconds=secs,
    )
    tag = f"{case}_closed" + ("" if np.isclose(R, R_prod) else f"_R{R:g}")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}_reach.npz", comid=ids, exc_cl=exc_cl * VOLUME,
                        effort=eff * VOLUME / 80.0, daily_effort=daily_eff)
    np.save(out / f"{tag}_mainstem.npy", ms_cl)
    if fidelity:
        np.savez_compressed(out / f"{tag}_fidelity.npz", comid=ids, mean=mean, T=T, acc=acc, ms_sur=ms_sur)
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=2))  # last: run_allssp_closed.py skips a tag with one
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--R", type=float, default=None)
    p.add_argument("--controller", type=Path, default=None)
    p.add_argument("--q-ornl", type=Path, default=None)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--fidelity", action="store_true")
    a = p.parse_args()
    run(a.case, a.R, a.controller, a.q_ornl, a.out, a.fidelity)
