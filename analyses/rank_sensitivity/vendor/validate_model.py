#!/usr/bin/env python3
"""Pipeline loader and reach statistics used by run_ensemble.py (via run_approximation.py).

Release copy: only production_functions, statistics and summarize are kept from the original module. SOURCE is the
pipeline file 04_dmd_lqr_analysis.py at the repository root (sha256 checked by run_ensemble.configure).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # analyses/ (common)
from common.pipeline_loader import SOURCE, source_sha256  # noqa: E402,F401


def production_functions():
    # Only definitions are executed: no source directory changes, mkdirs, CLI,
    # production runs, or blanket warning filters from the module top level.
    import dataclasses
    import typing
    from sklearn.linear_model import Ridge
    selected = {'PODPack', 'ForcingPack', 'DMDcPack', 'compute_pod',
                'normalize_forcing', 'fit_windowed_dmdc',
                'simulate_openloop', 'simulate_one_step_anchored'}
    parsed = ast.parse(SOURCE.read_text())
    defs = ast.Module(body=[n for n in parsed.body
                           if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                           and n.name in selected], type_ignores=[])
    namespace = dict(np=np, pd=pd, Ridge=Ridge, dataclass=dataclasses.dataclass,
                     Optional=typing.Optional, List=typing.List, Tuple=typing.Tuple,
                     RANK=4, DTYPE=np.float64, USE_TRUNCATED_SVD=True,
                     WINDOW=1000, STEP=500, ALPHA_CANDIDATES=[0.01, 0.1, 1, 10],
                     SPECTRAL_RADIUS_LIMIT=0.995,
                     PROF=SimpleNamespace(start=lambda _: None, stop=lambda _: None))
    exec(compile(defs, str(SOURCE), 'exec'), namespace)
    return namespace


def statistics(obs, sim, q95):
    mo = obs.mean(1); ms = sim.mean(1)
    vo = ((obs-mo[:, None])**2).mean(1)
    vs = ((sim-ms[:, None])**2).mean(1)
    so = np.sqrt(vo); ss = np.sqrt(vs)
    cov = ((obs-mo[:, None])*(sim-ms[:, None])).mean(1)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        corr = cov / (so*ss)
        corr = np.clip(corr, -1, 1)
        constant_sim = np.ptp(sim, axis=1) <= 1e-12 * np.maximum(1, np.max(np.abs(sim), axis=1))
        corr[constant_sim] = np.nan
        ratio_std = ss/so
        ratio_mean = ms/mo
        nse = 1 - ((sim-obs)**2).mean(1)/vo
        kge = 1 - np.sqrt((corr-1)**2 + (ratio_std-1)**2 + (ratio_mean-1)**2)
        pbias = ratio_mean-1
        peak_bias = sim.max(1)/obs.max(1)-1
        high = obs > q95[:, None]
        high_sqerr = (((sim-obs)**2)*high).sum(1)
        high_sqobs = ((obs**2)*high).sum(1)
        high_nrmse = np.sqrt(high_sqerr/high_sqobs)
        excess_obs = np.maximum(obs-q95[:, None], 0).sum(1)
        excess_sim = np.maximum(sim-q95[:, None], 0).sum(1)
        excess_bias = (excess_sim-excess_obs)/excess_obs
    valid = (vo > 1e-16) & (mo > 1e-10)
    for arr in (nse, kge, corr, ratio_std, ratio_mean, pbias, peak_bias, high_nrmse, excess_bias):
        arr[~valid | ~np.isfinite(arr)] = np.nan
    return dict(nse=nse, kge=kge, correlation=corr, std_ratio=ratio_std,
                mean_ratio=ratio_mean, pbias=pbias, peak_bias=peak_bias,
                highflow_nrmse=high_nrmse, exceedance_volume_bias=excess_bias,
                high_sqerr=high_sqerr, high_sqobs=high_sqobs)


def summarize(arrays, mask=None):
    def finite(name):
        x = arrays[name] if mask is None else arrays[name][mask]
        return x[np.isfinite(x)]
    def med(name):
        x = finite(name)
        return float(np.median(x)) if len(x) else np.nan
    r = {name+'_median': med(name) for name in
         ['nse', 'kge', 'correlation', 'std_ratio', 'mean_ratio', 'peak_bias',
          'highflow_nrmse', 'exceedance_volume_bias']}
    for name in ['nse', 'kge']:
        x = finite(name)
        r[name+'_p25'] = float(np.quantile(x, .25)) if len(x) else np.nan
        r[name+'_p75'] = float(np.quantile(x, .75)) if len(x) else np.nan
        r[name+'_positive_fraction'] = float(np.mean(x > 0)) if len(x) else np.nan
    r['valid_kge_reaches'] = len(finite('kge'))
    for name in ['pbias', 'exceedance_volume_bias']:
        vals = finite(name)
        r['abs_'+name+'_median'] = float(np.median(np.abs(vals))) if len(vals) else np.nan
    if mask is None:
        mask = slice(None)
    denominator = np.sum(arrays['high_sqobs'][mask])
    r['pooled_highflow_nrmse'] = float(np.sqrt(np.sum(arrays['high_sqerr'][mask])/denominator)) if denominator > 0 else np.nan
    return r
