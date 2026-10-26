"""Members, ranks and helpers shared by run_ensemble.py and analyze_ensemble.py.

GCMS, CASES, RANKS and dump() are those of run_ensemble.py; norm2() and nested_pod() (the 300-mode spectrum behind
the retained-energy column of Table S1 and Figure S1a-c) are copied unchanged from the script of the original runs.
"""
import gc
import json

import numpy as np
from sklearn.decomposition import TruncatedSVD

GCMS=['MPI-ESM1-2-HR','ACCESS-CM2','EC-Earth3','CNRM-ESM2-1','BCC-CSM2-MR','MRI-ESM2-0','NorESM2-MM']
CASES=[(h,g) for g in GCMS for h in ['PRMS','VIC5']]
RANKS=[30,20,50,10,75,100]


def dump(path,d):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,default=str));tmp.replace(path)


def norm2(x):
    return sum(float(np.einsum('ij,ij->',x[i:i+256],x[i:i+256])) for i in range(0,len(x),256))

def nested_pod(x,n,iterations=7,seed=42):
    mu=x.mean(1,keepdims=True)
    xc=x-mu
    energy=norm2(xc)
    fit=TruncatedSVD(n_components=n,n_iter=iterations,random_state=seed)
    z=fit.fit_transform(xc.T).T
    b=fit.components_.T
    # Rayleigh-Ritz rotation makes the nested projected energies exactly ordered.
    ev,rot=np.linalg.eigh(z@z.T)
    ix=np.argsort(ev)[::-1]; b=b@rot[:,ix]; z=rot[:,ix].T@z
    singular=np.sqrt(np.maximum(ev[ix],0))
    assert np.max(np.abs(b.T@b-np.eye(n)))<1e-10
    del xc;gc.collect()
    return mu,b,z,singular,energy
