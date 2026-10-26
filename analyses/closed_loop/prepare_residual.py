"""Collect the per-reach residual exceedance of both closed loops for the figure (Figure S6 f; read-only on all sources).

For each of the 56 members (order of WORK/data/figure_data.npz): the residual exceedance above the 2-year threshold
over 2020-2099 of the loop closed on the routing model (WORK/results_fidelity/<tag>_reach.npz, exc_cl) and of the
surrogate's own closed loop (summarize_fidelity.surrogate_exceedance of WORK/controllers/<case>/controller.npz, Q+
output), in the member's reach order. Writes WORK/data/residual_exceedance.npz.
Usage: python prepare_residual.py (after prepare_data.py)
"""
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402
from summarize_fidelity import surrogate_exceedance  # noqa: E402

DATA = layout.PREPARED


def main():
    fd = np.load(DATA / "figure_data.npz")
    cases = fd["cases"].tolist()
    res_ref, res_sur = [], []
    for i, case in enumerate(cases):
        js = glob.glob(str(layout.FIDELITY / f"{case}_closed_R*.json"))
        assert len(js) == 1, case
        tag = Path(js[0]).stem
        z = np.load(layout.FIDELITY / f"{tag}_reach.npz")
        assert np.array_equal(z["comid"], fd["comid"][i]), case
        ids, sur = surrogate_exceedance(layout.controller(case))
        assert np.array_equal(ids, z["comid"]), case
        res_ref.append(z["exc_cl"]); res_sur.append(sur)
    # Figure S6(f) uses only the within-member ranking of residual exceedance, so the ranks are stored
    # (average ranks for ties, as scipy.stats.rankdata) rather than the volumes.
    np.savez_compressed(DATA / "residual_exceedance.npz", cases=np.array(cases),
                        rank_ref=np.stack([rankdata(v) for v in res_ref]),
                        rank_sur=np.stack([rankdata(v) for v in res_sur]))
    print("saved", DATA / "residual_exceedance.npz")


if __name__ == "__main__":
    main()
