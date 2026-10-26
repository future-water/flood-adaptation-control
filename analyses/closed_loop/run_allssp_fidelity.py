"""Closed-loop control-fidelity runs for all 56 members at the surrogate's own effort (Figure S6, Section 3.1).

For each member, the penalty weight R* at which the closed loop on the routing reference spends the surrogate's effort
is interpolated in log(effort)-log(R) between the bracketing closed-loop runs of run_allssp_closed.py
(WORK/results_grid/), refined once by the secant step if the realized effort differs by more than 3%, and
closed_loop.py is run at R* with --fidelity (controlled-discharge NSE/KGE sums, annual maxima, surrogate mainstem
series). The uncontrolled discharge is regenerated (make_q_ornl.py) and deleted after use.
A member is done once complete/<case>.json is written (after its one kept run is checked); otherwise any
{case}_closed_R* files of an interrupted attempt are moved to superseded/ and the member is rerun.
Usage: python run_allssp_fidelity.py [--workers 15] [--cases <case> ...]
"""
import argparse
import glob
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402

PY = sys.executable
OUT = layout.FIDELITY
LOGS = layout.WORK / "logs_fidelity"
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def sh(cmd, log):
    with open(log, "w") as f:
        subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, check=True)


def interp_R(pts, target):
    """R at which effort equals target, linear in log(effort)-log(R) between the bracketing (R, effort) points."""
    pts = sorted(pts)
    for (r0, e0), (r1, e1) in zip(pts, pts[1:]):
        if min(e0, e1) <= target <= max(e0, e1):
            w = (np.log(target) - np.log(e0)) / (np.log(e1) - np.log(e0))
            return float(np.exp(np.log(r0) + w * (np.log(r1) - np.log(r0))))
    raise ValueError("target effort not bracketed")


def files(tag):
    """The four files closed_loop.py --fidelity writes for one run (the JSON last)."""
    return [OUT / f"{tag}{s}" for s in ("_reach.npz", "_mainstem.npy", "_fidelity.npz", ".json")]


def kept(case):
    """The member's one remaining run, after checking that its arrays load with the expected keys."""
    js = sorted(OUT.glob(f"{case}_closed_R*.json"))
    assert len(js) == 1, (case, [j.name for j in js])
    tag = js[0].stem
    for f, keys in ((f"{tag}_reach.npz", {"comid", "exc_cl", "effort", "daily_effort"}),
                    (f"{tag}_fidelity.npz", {"comid", "mean", "T", "acc", "ms_sur"})):
        with np.load(OUT / f) as z:
            assert set(z.files) == keys and all(z[k].size for k in keys), (f, z.files)
    assert np.load(OUT / f"{tag}_mainstem.npy").ndim == 1, tag
    return tag, json.loads(js[0].read_text())


def member(case, ssp, pool, gen_lock):
    # the original SSP2-4.5 runs read a kept uncontrolled discharge; here every member regenerates it
    ctrl, prev = layout.controller(case), layout.GRID
    q, regen = layout.q_ornl(case), True
    runs = [json.loads(Path(g).read_text()) for g in glob.glob(str(prev / f"{case}_closed*.json"))]
    e_sur = runs[0]["surrogate_effort_same_R_Gm3_per_yr"]
    pts = [(r["R_weight"], r["effort_Gm3_per_yr"]) for r in runs]
    marker = OUT / "complete" / f"{case}.json"
    if marker.exists():
        print(case, "already done", flush=True)
        return
    for f in OUT.glob(f"{case}_closed_R*"):  # left by an interrupted attempt
        f.rename(OUT / "superseded" / f.name)
    if regen and not q.exists():
        with gen_lock:
            sh([PY, HERE / "make_q_ornl.py", "--case", case, "--controller", ctrl, "--out", q], LOGS / f"{case}_q.log")
    base = [PY, HERE / "closed_loop.py", "--case", case, "--fidelity",
            "--controller", ctrl, "--q-ornl", q, "--out", OUT]

    def closed(R):
        tag = f"{case}_closed_R{R:g}"
        pool.submit(sh, base + ["--R", repr(R)], LOGS / f"{tag}.log").result()
        return json.loads((OUT / f"{tag}.json").read_text())

    R1 = interp_R(pts, e_sur)
    r1 = closed(R1)
    if abs(np.log(r1["effort_Gm3_per_yr"] / e_sur)) > np.log(1.03):  # one secant refinement
        pts.append((R1, r1["effort_Gm3_per_yr"]))
        R2 = interp_R(pts, e_sur)
        r2 = closed(R2)
        keep = min([(R1, r1), (R2, r2)], key=lambda t: abs(np.log(t[1]["effort_Gm3_per_yr"] / e_sur)))
        for R, _ in [(R1, r1), (R2, r2)]:
            if R != keep[0]:  # keep only the run nearest the surrogate effort
                for f in files(f"{case}_closed_R{R:g}"):
                    f.rename(OUT / "superseded" / f.name)
    tag, r = kept(case)
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text(json.dumps(dict(case=case, run=tag, R_weight=r["R_weight"], effort_Gm3_per_yr=r["effort_Gm3_per_yr"],
                                   surrogate_effort_Gm3_per_yr=e_sur), indent=2))
    os.replace(tmp, marker)
    if regen:
        q.unlink(missing_ok=True)
    print(case, "done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--cases", nargs="+", default=None, help="members to run (default: all 56)")
    a = p.parse_args()
    (OUT / "superseded").mkdir(parents=True, exist_ok=True)
    (OUT / "complete").mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    cases = [(c, s) for s in (245, 126, 370, 585) for c in layout.cases([s]) if a.cases is None or c in a.cases]
    gen_lock = threading.Semaphore(1)
    failed = []
    with ThreadPoolExecutor(a.workers) as pool, ThreadPoolExecutor(a.workers + 2) as members:
        futs = {members.submit(member, c, s, pool, gen_lock): c for c, s in cases}
        for f, c in futs.items():
            try:
                f.result()
            except Exception as e:
                print(c, "FAILED", repr(e), flush=True)
                failed.append(c)
    if failed:
        print("failed members:", " ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
