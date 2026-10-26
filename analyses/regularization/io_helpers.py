"""Helpers used by run_pilot.py and run_path.py.

Taken unchanged from the script the regularization runs imported them from (precipitation
loader, JSON writer, file hash, unreset rollout and reach scores). Only the data root changed:
precipitation is read from FAC_RAW_ROOT/precipitation (paths.py), and the returned
source names are relative to that folder.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
import hashlib, json
import numpy as np
import pandas as pd
from paths import RAW

DATA=RAW

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x

def write_json(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False)+'\n')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_precip(g,ids,dates):
    rid='r1i1p1f2' if g=='CNRM-ESM2-1' else 'r1i1p1f1'
    parts=[];sources=[];missing=set()
    for first,last in [(2020,2059),(2060,2099)]:
        if dates[-1].year<first or dates[0].year>last:continue
        p=DATA/'precipitation'/f'{g}_ssp245_{rid}_DBCCA_Daymet_prcp_{first}_{last}_daily.parquet'
        df=pd.read_parquet(p);df.columns=pd.Index(df.columns).astype(float).astype(int)
        assert df.index.is_unique and df.index.is_monotonic_increasing
        missing.update(set(ids)-set(df.columns));df=df.reindex(columns=ids,fill_value=0).fillna(0)
        df=df.loc[dates[0]:dates[-1]]
        parts.append(df);sources.append(str(p.relative_to(DATA)))
    df=pd.concat(parts)
    assert pd.DatetimeIndex(df.index).equals(dates)
    return df.to_numpy(dtype=np.float64).T,len(missing),sources

def rollout(a,forcing,initial):
    pred=np.empty_like(forcing);state=initial.copy()
    for i in range(forcing.shape[1]):
        state=a@state+forcing[:,i];pred[:,i]=state
        if not np.isfinite(state).all() or np.max(abs(state))>1e100:
            return None
    return pred

def reach_scores(obs,sim,q95):
    mean_o=obs.mean(1);mean_s=sim.mean(1)
    co=obs-mean_o[:,None];cs=sim-mean_s[:,None]
    den=np.sum(co**2,1);vs=np.sum(cs**2,1);err=sim-obs;sse=np.sum(err**2,1)
    r=np.divide(np.sum(co*cs,1),np.sqrt(den*vs),out=np.full(len(obs),np.nan),where=(den>1e-12)&(vs>1e-12))
    nse=1-np.divide(sse,den,out=np.full(len(obs),np.nan),where=den>1e-12)
    sd_ratio=np.sqrt(np.divide(vs,den,out=np.full(len(obs),np.nan),where=den>1e-12))
    mean_ratio=np.divide(mean_s,mean_o,out=np.full(len(obs),np.nan),where=abs(mean_o)>1e-12)
    kge=1-np.sqrt((r-1)**2+(sd_ratio-1)**2+(mean_ratio-1)**2)
    high=obs>q95[:,None];count=high.sum(1)
    high_sse=np.sum(err**2*high,1);high_norm=np.sum(obs**2*high,1)
    high_error=np.sqrt(np.divide(high_sse,high_norm,out=np.full(len(obs),np.nan),where=(count>=10)&(high_norm>1e-12)))
    excess_o=np.maximum(obs-q95[:,None],0).sum(1)
    excess_s=np.maximum(sim-q95[:,None],0).sum(1)
    bias=100*(mean_ratio-1)
    evbias=100*np.divide(excess_s-excess_o,excess_o,out=np.full(len(obs),np.nan),where=excess_o>1e-12)
    return dict(NSE=nse,KGE=kge,r=r,sd_ratio=sd_ratio,mean_ratio=mean_ratio,
        squared_error=sse,PBIAS_percent=bias,highflow_nrmse=high_error,highflow_days=count,
        highflow_squared_error=high_sse,highflow_squared_norm=high_norm,
        reference_excess_rate_sum=excess_o,prediction_excess_rate_sum=excess_s,
        excess_volume_bias_percent=evbias)
