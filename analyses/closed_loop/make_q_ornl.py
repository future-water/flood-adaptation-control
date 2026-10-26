"""Regenerate the uncontrolled daily discharge (q_ornl.npy) of one recovered member for the closed-loop test.

run_allssp.py deletes the copy that recover_member.py writes. This rebuilds it with the same lines as
recover_member.py (unmodified pipeline loader, exact RL2-cache reach alignment, X.T saved as float64) and checks it
against the recovered controller: same reach order, and its time mean equals the saved POD mean.
Raw inputs are read from FAC_RAW_ROOT and the pooled return levels from the outputs_common archive.
Usage: python make_q_ornl.py --case VIC5_ACCESS-CM2_ssp126_budget10 --controller <controller.npz> --out <q_ornl.npy>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analyses/ (common)
from common.pipeline_loader import load_pipeline, raw_inputs, return_levels_path  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--controller", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    hydro, gcm, ssp = a.case.split("_")[:3]
    scenario = int(ssp.removeprefix("ssp"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m = load_pipeline()  # 04_dmd_lqr_analysis.py, sha256 checked
    with raw_inputs():
        X, _, dates, original_ids = m.load_scenario_data(scenario, gcm, hydro)
    cache = return_levels_path(hydro, gcm)
    with np.load(cache, allow_pickle=False) as ref:
        _, it, ids = m.match_to_reference(ref["reach_ids"].astype(int).tolist(), original_ids)
    X = X[it]
    with np.load(a.controller) as c:
        assert np.array_equal(np.asarray(ids, dtype=np.int64), c["comid"]), "reach order differs from controller"
        assert X.shape[1] == c["z_ol"].shape[1], "day count differs from controller"
        mean_q = c["mean_q"].ravel()
    err = float(np.abs(X.mean(axis=1) - mean_q).max())
    assert err <= 1e-9 * (1 + np.abs(mean_q).max()), f"time mean differs from saved POD mean by {err}"
    tmp = a.out.with_suffix(".tmp.npy")
    np.save(tmp, X.T)
    tmp.rename(a.out)
    print(f"{a.case}: {X.shape[1]} days x {X.shape[0]} reaches, max |mean - POD mean| = {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
