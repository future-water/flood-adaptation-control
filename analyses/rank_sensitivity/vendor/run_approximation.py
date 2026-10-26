#!/usr/bin/env python3
"""Same-scenario approximation audit of the pipeline (in-sample reconstruction, 2020-2099).

Source definitions are loaded by AST without executing module side effects.
Rank changes only; original centering, ridge selection, windows and assignment.

Release copy: only the parts run_ensemble.py uses are kept (functions, accelerate, audit). The driver of the
earlier three-member accuracy runs (thresholds, evaluate, main) is not included; run_ensemble.accuracy() scores
the recursive reconstruction with the same vm.statistics/vm.summarize definitions.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_model as vm

SOURCE = vm.SOURCE


def functions():
    ns = vm.production_functions()
    ns.update(WINDOW=3650, STEP=1825,
              ALPHA_CANDIDATES=[.01, .1, .5, 1., 2., 5., 10., 25., 50.],
              HISTORICAL_YEARS=(1980, 2019), os=os)
    names = {'_realization_id', '_streamflow_prefix', 'load_scenario_data',
             'load_historical_data', '_nse', '_kge'}
    parsed = ast.parse(SOURCE.read_text())
    body = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


def accelerate(dmd, z, v):
    """Same recurrences; batch only the prescribed additive forcing products."""
    n = z.shape[1]
    anchored = np.empty_like(z)
    recursive = np.empty_like(z)
    anchored[:, 0] = recursive[:, 0] = z[:, 0]
    cuts = np.r_[0, np.flatnonzero(np.diff(dmd.window_for_k)) + 1, n-1]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        w = dmd.window_for_k[lo]
        a = dmd.A_list[w]
        force = dmd.E_list[w] @ v[:, lo:hi] + dmd.b_list[w]
        anchored[:, lo+1:hi+1] = a @ z[:, lo:hi] + force
        for k in range(lo, hi):
            recursive[:, k+1] = a @ recursive[:, k] + force[:, k-lo]
    return anchored, recursive


def audit(ns):
    rng = np.random.default_rng(219)
    x = rng.normal(size=(17, 103))
    v = rng.normal(size=(20, 103))
    pod = ns['compute_pod'](x, 7)
    oldwin, oldstep = ns['WINDOW'], ns['STEP']
    ns['WINDOW'], ns['STEP'] = 40, 20
    dmd = ns['fit_windowed_dmdc'](pod.Z, v)
    ns['WINDOW'], ns['STEP'] = oldwin, oldstep
    an, rec = accelerate(dmd, pod.Z, v)
    expected_an = ns['simulate_one_step_anchored'](dmd, pod.Z, v[:, :-1])
    expected_rec = ns['simulate_openloop'](dmd, pod.Z[:, :1], v[:, :-1])
    np.testing.assert_allclose(an, expected_an, atol=1e-10)
    np.testing.assert_allclose(rec, expected_rec, atol=1e-10)
    residual = x - pod.mean_x - pod.U_r @ pod.Z
    dyn = pod.U_r @ (pod.Z - rec)
    total = x - pod.mean_x - pod.U_r @ rec
    np.testing.assert_allclose(np.sum(total**2), np.sum(residual**2)+np.sum(dyn**2), rtol=1e-12)
    return dict(accelerated_recurrence_parity='passed', orthogonal_decomposition='passed',
                source=SOURCE.name, source_sha256=vm.source_sha256(SOURCE))
