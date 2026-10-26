"""Penalty-weight grid of the closed loop on the routing model for all 56 members (Figure S6).

For each member: regenerate the uncontrolled discharge (make_q_ornl.py) and run the closed loop at the production
weight and at 10^0.5, 10 and 10^1.5 times it, adding weights until the surrogate effort is bracketed. The original runs
used this script for the SSP1-2.6, SSP3-7.0 and SSP5-8.5 members and ran the same four weights for the SSP2-4.5
members from job lists; all four SSPs are included by default.
run_allssp_fidelity.py interpolates R* from the JSON files written here (WORK/results_grid/).
The regenerated discharge (3.4 GB per member) is deleted after the member's runs.
Usage: python run_allssp_closed.py [--workers 15] [--ssps 126 245 370 585] [--cases <case> ...]
"""
import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402

PY = sys.executable
OUT = layout.GRID
LOGS = layout.WORK / "logs_grid"
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def sh(cmd, log):
    with open(log, "w") as f:
        subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT, env=ENV, check=True)


def tag(case, R, R_prod):
    return f"{case}_closed" + ("" if np.isclose(R, R_prod) else f"_R{R:g}")


def member(case, pool, gen_lock):
    ctrl = layout.controller(case)
    R_prod = float(np.load(ctrl)["R_weight"])
    q = layout.q_ornl(case)
    base = [PY, HERE / "closed_loop.py", "--case", case, "--controller", ctrl, "--q-ornl", q, "--out", OUT]

    def closed(R):
        t = tag(case, R, R_prod)
        if not (OUT / f"{t}.json").exists():
            sh(base + ([] if np.isclose(R, R_prod) else ["--R", repr(R)]), LOGS / f"{t}.log")
        return json.loads((OUT / f"{t}.json").read_text())

    grid = [R_prod * 10 ** e for e in (0.0, 0.5, 1.0, 1.5)]
    todo = [t for t in [tag(case, R, R_prod) for R in grid] if not (OUT / f"{t}.json").exists()]
    if todo and not q.exists():
        with gen_lock:
            sh([PY, HERE / "make_q_ornl.py", "--case", case, "--controller", ctrl, "--out", q], LOGS / f"{case}_q.log")
    jobs = [pool.submit(closed, R) for R in grid]
    runs = [j.result() for j in jobs]
    e_sur = runs[0]["surrogate_effort_same_R_Gm3_per_yr"]
    lo, hi = 0.0, 1.5
    while len(runs) < 8 and not min(r["effort_Gm3_per_yr"] for r in runs) <= e_sur <= max(r["effort_Gm3_per_yr"] for r in runs):
        e = lo - 0.5 if e_sur > max(r["effort_Gm3_per_yr"] for r in runs) else hi + 0.5
        lo, hi = min(lo, e), max(hi, e)
        runs.append(closed(R_prod * 10 ** e))
    q.unlink(missing_ok=True)
    print(case, "done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--ssps", type=int, nargs="+", default=list(layout.SSPS))
    p.add_argument("--cases", nargs="+", default=None, help="members to run (default: all members of --ssps)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    cases = a.cases or layout.cases(a.ssps)
    gen_lock = threading.Semaphore(1)
    failed = []
    with ThreadPoolExecutor(a.workers) as pool, ThreadPoolExecutor(5) as members:
        futs = {members.submit(member, c, pool, gen_lock): c for c in cases}
        for f, c in futs.items():
            try:
                f.result()
            except Exception as e:  # keep going; the failing member is reported and can be rerun
                print(c, "FAILED", repr(e), flush=True)
                failed.append(c)
    if failed:
        print("failed members:", " ".join(failed), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
