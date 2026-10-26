#!/usr/bin/env python3
"""Original-coordinate ridge path and same-window continuous reconstruction.

Rerun path of Figure S2 (a, b). Run after run_pilot.py (default cutoff): it checks its regression
inputs and three-alpha fits against FAC_WORK_ROOT/regularization/pilot/. Reads the same controller.npz,
q_ornl.npy and precipitation inputs as run_pilot.py. Writes FAC_WORK_ROOT/regularization/path/<hydro>/
and FAC_WORK_ROOT/regularization/all_paths.csv.

  python run_path.py
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
import argparse, gc, hashlib, json, time
import numpy as np
import pandas as pd
from scipy.linalg import svd
from threadpoolctl import threadpool_limits
from paths import WORK
import io_helpers as base

CASES=WORK/'regularization'/'cases'
CONTROLLERS=CASES
QDIR=CASES
PILOT=WORK/'regularization'/'pilot'
OUT=WORK/'regularization'/'path'
CANDIDATES=[.01,.1,.5,1.,2.,5.,10.,25.,50.]
ALPHAS=sorted(set([0.]+list(np.logspace(-8,12,41))+CANDIDATES))

def rel(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def rollout(a,f,z0):
    z=z0.copy();out=np.empty_like(f)
    for k in range(f.shape[1]):
        z=a@z+f[:,k]
        if not np.isfinite(z).all() or np.max(abs(z))>1e100:
            return None,k+1
        out[:,k]=z
    return out,None

def scores(obs,z,U,mu):
    values={'raw':[],'clipped':[]}
    for lo in range(0,len(U),256):
        hi=min(lo+256,len(U));o=obs[lo:hi];q=U[lo:hi]@z+mu[lo:hi]
        om=o.mean(1);oc=o-om[:,None];ov=(oc*oc).sum(1)
        for mode in values:
            pred=q if mode=='raw' else np.maximum(q,0)
            pm=pred.mean(1);pc=pred-pm[:,None];pv=(pc*pc).sum(1)
            sse=((pred-o)**2).sum(1)
            nse=1-np.divide(sse,ov,out=np.full(len(o),np.nan),where=ov>1e-12)
            corr=np.divide((pc*oc).sum(1),np.sqrt(pv*ov),out=np.full(len(o),np.nan),where=(pv>1e-12)&(ov>1e-12))
            sd=np.sqrt(np.divide(pv,ov,out=np.full(len(o),np.nan),where=ov>1e-12))
            beta=np.divide(pm,om,out=np.full(len(o),np.nan),where=abs(om)>1e-12)
            kge=1-np.sqrt((corr-1)**2+(sd-1)**2+(beta-1)**2)
            values[mode].append(dict(nse=nse,kge=kge,sse=sse))
    result={}
    for mode,parts in values.items():
        nse=np.concatenate([p['nse'] for p in parts]);kge=np.concatenate([p['kge'] for p in parts])
        result.update({f'{mode}_NSE':float(np.nanmedian(nse)),f'{mode}_KGE':float(np.nanmedian(kge)),
            f'{mode}_RMSE':float(np.sqrt(sum(p['sse'].sum() for p in parts)/obs.size)),
            f'{mode}_valid_NSE':int(np.isfinite(nse).sum()),f'{mode}_valid_KGE':int(np.isfinite(kge).sum())})
    return result

def run(hydro):
    dest=OUT/hydro;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'completed.json').exists():return
    begin=time.monotonic();case=f'{hydro}_MPI-ESM1-2-HR_ssp245_budget10'
    source=CONTROLLERS/case/'controller.npz'
    with np.load(source) as d:
        U=d['U_r'].astype(float);mu=d['mean_q'].astype(float);ids=d['comid']
        pmu=d['forcing_mu'].astype(float);psig=float(d['forcing_global_sigma'])
        lo,hi=map(int,d['bounds'][0]);dates=pd.to_datetime(d['dates'][lo:hi+1])
        saved_a=d['A'][0].astype(float);saved_e=d['E'][0].astype(float)
    old_setup=json.loads((PILOT/f'{hydro}_setup.json').read_text())
    original=float(old_setup['alpha_original'])
    qpath=QDIR/case/'q_ornl.npy'
    q=np.load(qpath,mmap_mode='r');obs=np.asarray(q[lo:hi+1],dtype=float).T
    z=U.T@(obs-mu);y=z[:,1:]
    p,missing,paths=base.load_precip('MPI-ESM1-2-HR',ids,dates)
    v=(p[:,:-1]-pmu)/psig;del p
    x=np.ascontiguousarray(np.vstack([z[:,:-1],v]));n=x.shape[1]
    xhash=hashlib.sha256(x.tobytes()).hexdigest();yhash=hashlib.sha256(y.tobytes()).hexdigest()
    assert xhash==old_setup['x_sha256'] and yhash==old_setup['y_sha256']
    print(hydro,'LOADED',len(ALPHAS),'alphas',round(time.monotonic()-begin,1),flush=True)
    left,s,vt=svd(x,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    cutoff=np.finfo(float).eps*max(x.shape)*s[0];keep=s>cutoff;yt=y@vt.T
    old_s=np.load(PILOT/f'{hydro}_spectrum.npz')['singular_values']
    assert np.allclose(s,old_s,rtol=1e-8,atol=1e-9)
    np.savez_compressed(dest/'spectrum.npz',singular_values=s,numerical_cutoff=cutoff,
        response_spectral_energy=np.sum(yt**2,axis=0),original_alpha=original,
        original_filter_factors=s*s/(s*s+original))
    base.write_json(dest/'source.json',dict(case=case,controller=f'{case}/controller.npz',controller_sha256=base.sha(source),
        q_source=f'{case}/q_ornl.npy',forcing_sources=paths,missing_precipitation_channels=missing,
        n=n,p=x.shape[0],rank=30,date_start=str(dates[0].date()),date_end=str(dates[-1].date()),
        original_alpha=original,numerical_rank=int(keep.sum()),numerical_cutoff=cutoff,
        predictor_sha256=xhash,target_sha256=yhash))
    norms=float(np.linalg.norm(y));rows=[];verification={}
    for i,alpha in enumerate(ALPHAS):
        gain=np.divide(1.,s,out=np.zeros_like(s),where=keep) if alpha==0 else s/(s*s+alpha)
        shrink=s*gain;c=yt*gain;a=c@left[:30].T
        coeff_norm=float(np.linalg.norm(c));anorm=float(np.linalg.norm(a))
        residual=float(np.linalg.norm(yt*(1-shrink)))
        pred=(yt*shrink)@vt
        f=pred-a@z[:,:-1]
        sim,fail_step=rollout(a,f,z[:,0]);rho=float(max(abs(np.linalg.eigvals(a))))
        row=dict(hydro=hydro,alpha=alpha,original_alpha=original,is_original=alpha==original,
            in_original_grid=alpha in CANDIDATES,n=n,p=x.shape[0],
            residual_norm=residual,target_norm=norms,fit_relative_L2=residual/norms,
            coefficient_norm=coeff_norm,A_norm=anorm,E_norm=float(np.sqrt(max(0,coeff_norm**2-anorm**2))),
            rho=rho,margin_pass=rho<.995,df_per_response=float(shrink.sum()),
            normal_condition=None if alpha==0 else float((s[0]**2+alpha)/alpha),
            augmented_condition=None if alpha==0 else float(np.sqrt((s[0]**2+alpha)/alpha)),
            rollout_complete=sim is not None,abort_step=fail_step)
        if sim is not None:row.update(scores(obs[:,1:],sim,U,mu))
        if alpha==original:
            coef=c@left.T
            verification.update(saved_A_relative_difference=rel(a,saved_a),saved_E_relative_difference=rel(coef[:,30:],saved_e),
                saved_operator_action_relative_difference=rel(pred,saved_a@z[:,:-1]+saved_e@v),
                explicit_forcing_relative_difference=rel(f,coef[:,30:]@v),
                residual_spectral_vs_direct_absolute_difference=abs(residual-float(np.linalg.norm(pred-y))))
            np.savez_compressed(dest/'original_fit_check.npz',A=a,E=coef[:,30:],z_reconstructed=sim,dates=dates.to_numpy())
        rows.append(row)
        if i%5==0 or alpha in CANDIDATES:
            print(hydro,i+1,'/',len(ALPHAS),'alpha',alpha,'rho',round(rho,5),'NSE',row.get('raw_NSE','ABORT'),flush=True)
    df=pd.DataFrame(rows);df.to_csv(dest/'path.csv',index=False)
    prior=pd.read_csv(PILOT/f'{hydro}_fits.csv')
    for _,old in prior.iterrows():
        new=df.loc[df.alpha==old.alpha].iloc[0]
        assert np.isclose(new.rho,old.rho,rtol=1e-8,atol=1e-8)
        assert np.isclose(new.fit_relative_L2,old.fit_relative_l2,rtol=1e-8,atol=1e-10)
        if bool(new.rollout_complete):
            assert np.isclose(new.raw_NSE,old.rollout_raw_NSE_median,rtol=1e-5,atol=1e-5)
            assert np.isclose(new.raw_KGE,old.rollout_raw_KGE_median,rtol=1e-5,atol=1e-5)
        else:assert not bool(old.rollout_finite)
    assert verification['saved_operator_action_relative_difference']<1e-4
    assert verification['explicit_forcing_relative_difference']<1e-8
    pos=df.query('alpha>0').sort_values('alpha')
    assert (np.diff(pos.coefficient_norm)<=1e-7).all()
    assert (np.diff(pos.residual_norm)>=-1e-7).all()
    assert (np.diff(pos.df_per_response)<=1e-7).all()
    verification.update(archived_three_alpha_metric_parity=True,all_fits_included=True,
        coefficient_norm_decreases=True,residual_norm_increases=True,conditional_df_decreases=True,
        svd_reconstruction_relative_L2=rel((left*s)@vt[:,::500],x[:,::500]))
    base.write_json(dest/'verification.json',verification)
    base.write_json(dest/'completed.json',dict(hydro=hydro,n_fits=len(df),aborted_runs=int((~df.rollout_complete).sum()),
        elapsed_seconds=time.monotonic()-begin))
    print(hydro,'DONE',round(time.monotonic()-begin,1),flush=True)
    del obs,x,q,z,y,U,left,vt,v,yt;gc.collect()

def main():
    global CONTROLLERS,QDIR
    parser=argparse.ArgumentParser()
    parser.add_argument('--controllers',type=Path,default=CONTROLLERS,help='folder with <case>/controller.npz')
    parser.add_argument('--q-dir',type=Path,default=QDIR,help='folder with <case>/q_ornl.npy')
    args=parser.parse_args()
    CONTROLLERS=args.controllers.expanduser().resolve();QDIR=args.q_dir.expanduser().resolve()
    threadpool_limits(4);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(scope='Same two preselected MPI-ESM1-2-HR SSP245 cases as the preceding pilot: VIC5 and PRMS, first original3650-transition window.',
        fixed='Original full2020-2099 member POD, discharge mean, precipitation mean/global scale, r30 and all identification transitions. No noise perturbation; no control or LQR refit.',
        alpha_path=ALPHAS,original_candidate_grid=CANDIDATES,
        alpha_selection='The archived original-rule alpha is marked; path scores do not retune it and no L-curve corner is automatically selected.',
        SVD='Direct double-precision SVD of Omega, same raw scaling as fitted pipeline. Alpha0 uses eps*max(p,n)*smax cutoff; positive ridge uses all directions.',
        L_curve='Residual norm versus coefficient Frobenius norm, normalized by ||Y||F and ||G_original||F respectively for display. No candidate omitted.',
        rollout='Every alpha receives one actual projected initial state then3650 successive daily updates with original known forcing, no resets. Raw primary, Q+ secondary. Abort only if any state is nonfinite or exceeds1e100; preserve failed runs with no fabricated score.',
        metrics='Reachwise NSE and KGE (correlation, standard-deviation ratio, mean ratio) over the complete window, then median. Undefined denominators excluded with counts. No flooring or capping.',
        main_panel_C='Prespecified original alpha candidate grid only for readable NSE/stability comparison of the actual selection options. Full alpha path, including adverse results, is shown in the supplement.',
        condition='Coefficient-space normal matrix Omega Omega.T is singular without ridge because p>n. Ridge condition=(smax^2+alpha)/alpha; augmented least-squares condition is its square root. These are scaling-dependent numerical diagnostics, not sample-sufficiency tests.')
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:base.write_json(pp,protocol)
    for h in ['VIC5','PRMS']:run(h)
    pd.concat([pd.read_csv(OUT/h/'path.csv') for h in ['VIC5','PRMS']],ignore_index=True).to_csv(OUT.parent/'all_paths.csv',index=False)

if __name__=='__main__':main()
