"""Surrogate closed loop used by rl_effort_r30.py.

load_pipeline() (common.pipeline_loader) returns the pipeline module, 04_dmd_lqr_analysis.py (m.load_scenario_data,
m.compute_pod, m.RANK, ...). The helper of the original runs switched into the data folder and imported the source
file; the loader checks the source hash and points the pooled return-level cache (m._cache_path_for_pooled) at the
shipped files. The streamflow and precipitation loaders read cwd-relative paths, so call them inside
common.pipeline_loader.raw_inputs().
closed_loop_batch() is copied unchanged from the helper module that ran the threshold runs; it is the threshold-gated
closed loop of the pipeline's simulate_cl_flood_only, evaluated for all penalty weights at once (closed_loop() in its
docstring is the per-weight version of the same recurrence, with the forcing term E^(w(k)) v(k) precomputed in D).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # analyses/ (common)
from common.pipeline_loader import load_pipeline  # noqa: E402,F401  (used by rl_effort_r30.py)


def closed_loop_batch(A_list, K_by_window, window_for_k, D, z0, U_r, mean_x, thr):
    """closed_loop() for all penalty weights at once (columns = weights); same recurrence.

    K_by_window[w] has shape (nR, r, r). Matrix-matrix products replace matrix-vector products,
    so results agree with the per-weight loop to rounding (checked against the original sweep).
    """
    r, T = D.shape
    nR = K_by_window[0].shape[0]
    Z = np.repeat(z0, nR, axis=1)
    Uc = np.zeros((nR, r, T))
    thr = thr.reshape(-1, 1)
    for k in range(T):
        w = window_for_k[k]
        z_excess = U_r.T @ np.maximum(U_r @ Z + mean_x - thr, 0.0)
        u = -np.einsum('jab,bj->aj', K_by_window[w], z_excess)
        Uc[:, :, k] = u.T
        Z = A_list[w] @ Z + u + D[:, k:k + 1]
    return Uc
