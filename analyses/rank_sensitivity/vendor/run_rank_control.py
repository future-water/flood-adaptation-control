#!/usr/bin/env python3
"""Fixed-budget rank robustness of original same-scenario gated LQR diagnostic.

Original alpha search preserved. No reanchoring, predictor training, change of the control law, or source-file edits.

Release copy: only the parts run_ensemble.py uses are kept (GRID, ORIGINAL_ALPHAS, source_functions,
pack_metrics, baseline_metrics, and the compiled kernel as rc.kernel). The driver of the earlier three-member runs
(get_case_reference, fit_model, evaluate, fixed_budgets, run_case, main) is not included.
"""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import ast, sys
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_are
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))   # analyses/ (common)
import run_approximation as ap
from common import kernel

ORIGINAL_ALPHAS=[.01,.1,.5,1.,2.,5.,10.,25.,50.]
GRID=np.logspace(-4,5,19)
DT=86400.


def source_functions():
    ns=ap.functions()
    ns['solve_discrete_are']=solve_discrete_are
    names={'lqr_gain','simulate_cl_flood_only'}
    tree=ast.parse(ap.SOURCE.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(ap.SOURCE),'exec'),ns)
    return ns


def pack_metrics(stats,am,ny):
    factor=DT/ny
    out=dict(effort=stats[0]*factor,attenuation=stats[1]*factor,augmentation=stats[2]*factor,
             residual=stats[3]*factor,exceedance_days=stats[4]/ny,peak=stats[5],
             negative_flow_days=stats[6]/ny,attenuation_on_flood_days=stats[7]*factor,
             augmentation_on_flood_days=stats[8]*factor)
    if am is not None:
        out['rl10']=np.percentile(am,90,axis=0)
        out['mean_annual_peak']=am.mean(0)
    return out


def baseline_metrics(x,dates,basis=None,z=None,mu=None):
    nt=len(dates)-1; years=dates[:nt].year
    cuts=np.r_[0,np.flatnonzero(np.diff(years))+1,nt]; ny=len(cuts)-1
    parts=[]
    for lo in range(0,x.shape[0],256):
        hi=min(lo+256,x.shape[0])
        q=x[lo:hi,:nt] if basis is None else np.maximum(basis[lo:hi]@z[:,:nt]+mu[lo:hi],0)
        am=np.stack([q[:,a:b].max(1) for a,b in zip(cuts[:-1],cuts[1:])],axis=1)
        parts.append(dict(peak=q.max(1),rl10=np.percentile(am,90,axis=1),mean_annual_peak=am.mean(1)))
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
