"""ctypes wrapper of the compiled closed-loop kernel used by the window runs.

The library is built from analyses/common/control_kernel.cpp by analyses/common/build_kernels.sh.
"""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # analyses/ (common)
from common.native import load_library

HERE=Path(__file__).resolve().parent
LIB=load_library('control_kernel')
FN=LIB.simulate_batch
FN.argtypes=[ct.c_int]*6+[ct.c_void_p]*13
FN.restype=ct.c_int


def matrix_stack(items):
    return np.concatenate([np.asarray(a).ravel(order='F') for a in items])


def simulate(basis, mu, thr, z0, aa, gains, force, windows, years, annual=True, trace=False):
    basis=np.asfortranarray(basis,dtype=np.float64)
    mu=np.ascontiguousarray(mu.ravel(),dtype=np.float64)
    thr=np.ascontiguousarray(thr,dtype=np.float64)
    z0=np.ascontiguousarray(z0.ravel(),dtype=np.float64)
    force=np.asfortranarray(force,dtype=np.float64)
    windows=np.ascontiguousarray(windows,dtype=np.int32)
    years=np.ascontiguousarray(years,dtype=np.int32)
    n,r=basis.shape; nt=force.shape[1]; nw=len(aa); m=len(gains[0]); ny=int(years.max())+1
    aflat=matrix_stack(aa)
    kflat=matrix_stack([g for group in gains for g in group])
    stats=np.zeros((9,m,n),dtype=np.float64)
    am=np.zeros((ny,m,n),dtype=np.float64) if annual else None
    zz=np.empty((nt+1,m,r),dtype=np.float64) if trace else None
    uu=np.empty((nt,m,r),dtype=np.float64) if trace else None
    arrays=[basis,mu,thr,z0,aflat,kflat,force,windows,years,stats,am,zz,uu]
    pointers=[None if a is None else a.ctypes.data for a in arrays]
    tick=time.time()
    status=FN(n,r,nt,m,nw,ny,*pointers)
    if status: raise FloatingPointError(f'Nonfinite control rollout or invalid metadata; kernel status {status}')
    return stats,am,zz,uu,time.time()-tick
