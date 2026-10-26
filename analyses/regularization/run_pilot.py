#!/usr/bin/env python3
"""Conditional ridge sensitivity, original full-scenario coordinates, two fixed cases.

Rerun path of Figure S2 (c). For each case <hydro>_MPI-ESM1-2-HR_ssp245_budget10 it reads
<controllers>/<case>/controller.npz and <q-dir>/<case>/q_ornl.npy (default for both:
FAC_WORK_ROOT/regularization/cases) and the precipitation parquet under FAC_RAW_ROOT.
Writes FAC_WORK_ROOT/regularization/pilot/, or pilot/numerical_cutoff_1e-08/ with --rcond 1e-8.

  python run_pilot.py
  python run_pilot.py --rcond 1e-8
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

CASES = WORK / 'regularization' / 'cases'
CONTROLLERS = CASES
QDIR = CASES
OUT = WORK / 'regularization' / 'pilot'

def relative(a,b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))

def run(hydro,rcond=None):
    start=time.monotonic()
    case=f'{hydro}_MPI-ESM1-2-HR_ssp245_budget10'
    folder=CONTROLLERS/case
    with np.load(folder/'controller.npz') as d:
        ids=d['comid']; U=d['U_r'].astype(float); mu=d['mean_q'].astype(float)
        pmu=d['forcing_mu'].astype(float); psig=float(d['forcing_global_sigma'])
        lo,hi=map(int,d['bounds'][0]); dates=pd.to_datetime(d['dates'][lo:hi+1])
        saved_A=d['A'][0].astype(float); saved_E=d['E'][0].astype(float)
    qpath=QDIR/case/'q_ornl.npy'
    q=np.load(qpath,mmap_mode='r'); obs=np.asarray(q[lo:hi+1],dtype=float).T
    z=U.T@(obs-mu)
    p,missing,sources=base.load_precip('MPI-ESM1-2-HR',ids,dates)
    v=(p[:,:-1]-pmu)/psig; del p
    x=np.ascontiguousarray(np.vstack([z[:,:-1],v]))
    y=z[:,1:]; n=x.shape[1]; r=z.shape[0]
    print(hydro,'loaded',round(time.monotonic()-start,1),flush=True)
    # Direct SVD avoids squaring the condition number in X^T X for alpha=0.
    left,s,vt=svd(x,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    cutoff=(np.finfo(float).eps*max(x.shape) if rcond is None else rcond)*s[0]
    keep=s>cutoff
    yt=y@vt.T
    original_candidates=[.01,.1,.5,1.,2.,5.,10.,25.,50.]
    selection=[]
    for alpha in original_candidates:
        a=(yt*(s/(s*s+alpha)))@left[:r].T
        rho=float(max(abs(np.linalg.eigvals(a))))
        selection.append(dict(alpha=alpha,rho=rho))
        if rho<.995: break
    chosen=selection[-1]['alpha']
    # Freeze the selected original-rule alpha before drawing artificial noise.
    alphas=[0.,chosen,10*chosen]
    base.write_json(OUT/f'{hydro}_setup.json',dict(case=case,n=n,p=x.shape[0],bounds=[lo,hi],
        dates=[str(dates[0].date()),str(dates[-1].date())],alpha_original=chosen,grid=selection,
        singular_cutoff=cutoff,numerical_rank=int(keep.sum()),precip_missing=missing,precip_sources=sources,
        controller=f'{case}/controller.npz',controller_sha256=base.sha(folder/'controller.npz'),
        x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest()))
    np.savez_compressed(OUT/f'{hydro}_spectrum.npz',singular_values=s,cutoff=cutoff,alphas=alphas)
    rows=[]; fits={}; normy=float(np.linalg.norm(y))
    for alpha in alphas:
        filt=np.divide(1.,s,out=np.zeros_like(s),where=keep) if alpha==0 else s/(s*s+alpha)
        shrink=s*filt
        c=yt*filt
        a=c@left[:r].T
        coef=c@left.T
        e=coef[:,r:]
        pred=(yt*shrink)@vt
        forced=e@v
        sim=base.rollout(a,forced,z[:,0])
        residual=float(np.linalg.norm(pred-y)/normy)
        entry=dict(hydro=hydro,alpha=alpha,original_alpha=chosen,n=n,numerical_rank=int(keep.sum()),
            df_per_response=float(shrink.sum()),df_fraction=float(shrink.sum()/n),
            response_noise_fit_rms_gain=float(np.sqrt(np.sum(shrink**2)/n)),
            operator_isotropic_noise_gain=float(np.sqrt(np.sum(filt**2))),
            fit_relative_l2=residual,coefficient_norm=float(np.linalg.norm(c)),
            A_norm=float(np.linalg.norm(a)),E_norm=float(np.linalg.norm(e)),rho=float(max(abs(np.linalg.eigvals(a)))),
            rollout_latent_relative_l2=None if sim is None else relative(sim,y),
            rollout_finite=sim is not None)
        if sim is not None:
            reach=base.reach_scores(obs[:,1:],U@sim+mu,np.quantile(obs,.95,axis=1))
            entry['rollout_raw_NSE_median']=float(np.nanmedian(reach['NSE']))
            entry['rollout_raw_KGE_median']=float(np.nanmedian(reach['KGE']))
        if alpha==chosen:
            entry['saved_A_relative_difference']=relative(a,saved_A)
            entry['saved_E_relative_difference']=relative(e,saved_E)
            entry['saved_operator_action_relative_difference']=relative(pred,saved_A@z[:,:-1]+saved_E@v)
        rows.append(entry);fits[alpha]=dict(filt=filt,a=a,forced=forced,sim=sim)
        print(hydro,'alpha',alpha,'df',entry['df_per_response'],'rho',entry['rho'],flush=True)
        del coef,e
    noise_rows=[]
    for seed in range(1201,1221):
        rng=np.random.default_rng(seed)
        noise=rng.standard_normal(y.shape)*y.std(axis=1,keepdims=True)
        noise*=.01*np.linalg.norm(y-y.mean(axis=1,keepdims=True))/np.linalg.norm(noise)
        nw=noise@vt.T
        for alpha in alphas:
            f=fits[alpha];dc=nw*f['filt'];da=dc@left[:r].T
            normg=float(np.linalg.norm(dc));norma=float(np.linalg.norm(da))
            # (delta G) X - (delta A) Z = (delta E) V, without materializing E.
            de_v=(nw*(s*f['filt']))@vt-da@z[:,:-1]
            perturbed=base.rollout(f['a']+da,f['forced']+de_v,z[:,0])
            noise_rows.append(dict(hydro=hydro,seed=seed,alpha=alpha,original_alpha=chosen,
                noise_relative_centered_target=.01,delta_G_norm=normg,delta_A_norm=norma,
                delta_E_norm=float(np.sqrt(max(0,normg**2-norma**2))),
                delta_rollout_relative_l2=None if perturbed is None or f['sim'] is None else relative(perturbed,f['sim']),
                perturbed_rollout_finite=perturbed is not None))
    pd.DataFrame(rows).to_csv(OUT/f'{hydro}_fits.csv',index=False)
    pd.DataFrame(noise_rows).to_csv(OUT/f'{hydro}_noise.csv',index=False)
    base.write_json(OUT/f'{hydro}_verification.json',dict(
        svd_column_reconstruction_relative_l2=relative((left*s)@vt[:,::500],x[:,::500]),
        left_orthogonality_error=float(np.max(abs(left.T@left-np.eye(n)))),
        right_orthogonality_error=float(np.max(abs(vt@vt.T-np.eye(n)))),
        ridge_vs_saved={k:v for k,v in next(t for t in rows if t['alpha']==chosen).items() if k.startswith('saved_')},
        all_noise_G_norms_shrink=all(a['delta_G_norm']<=b['delta_G_norm']*(1+1e-10) for a,b in
            zip([t for t in noise_rows if t['alpha']==chosen],[t for t in noise_rows if t['alpha']==0])),
        elapsed_seconds=time.monotonic()-start))
    print(hydro,'DONE',round(time.monotonic()-start,1),flush=True)
    del x,left,s,vt,obs,z,q,v,yt,fits;gc.collect()

def main():
    global OUT,CONTROLLERS,QDIR
    parser=argparse.ArgumentParser();parser.add_argument('--rcond',type=float)
    parser.add_argument('--controllers',type=Path,default=CONTROLLERS,help='folder with <case>/controller.npz')
    parser.add_argument('--q-dir',type=Path,default=QDIR,help='folder with <case>/q_ornl.npy')
    args=parser.parse_args()
    CONTROLLERS=args.controllers.expanduser().resolve();QDIR=args.q_dir.expanduser().resolve()
    if args.rcond is not None:
        OUT=OUT/f'numerical_cutoff_{args.rcond:g}'
    threadpool_limits(4);OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(scope='Two fixed MPI-ESM1-2-HR SSP245 cases (VIC5, PRMS), first original3650-transition window.',
        purpose='Isolate conditional ridge shrinkage and sensitivity to synthetic target perturbations on the original full-scenario representation.',
        fixed='Original U, mean discharge, precipitation mean/global scale, rank30, window endpoints, regressors and all transitions. No control fitting or temporal holdout.',
        comparison='Minimum-norm unregularized SVD solution alpha0; original first-stable alpha; ten times that alpha as a labeled sensitivity comparison, not an adopted change.',
        alpha0='Moore-Penrose solution using eps*max(p,n)*smax numerical singular-value threshold; all ridge singular directions retained.',
        perturbation='Twenty paired seeds1201..1220. Independent Gaussian noise in response coordinates, weighted by each response temporal std and rescaled to1% of centered target Frobenius norm. Perturb Y only, hold regressors fixed. Same noise for all alphas; alpha frozen. Synthetic numerical probe, not physical observation uncertainty.',
        reporting='Report df, coefficient sensitivity for G/A/E, one-step fit error, whole-window unreset raw recursive reconstruction, and recursive perturbation sensitivity. Retain every trial including divergent ones.',
        limits='Conditional shrinkage is expected mathematically. Magnitude in actual original design is measured here. Neither shrinkage nor df proves sufficient independent samples, absence of estimator bias, or forecast skill. The two cases are a pilot, not ensemble evidence.')
    if args.rcond is not None:
        protocol['alpha0']=f'Post-hoc numerical sensitivity check: Moore-Penrose baseline with rcond={args.rcond:g}. Ridge calculations and paired noise unchanged. This truncation itself suppresses singular directions.'
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:base.write_json(path,protocol)
    for hydro in ['VIC5','PRMS']:
        if not (OUT/f'{hydro}_verification.json').exists():run(hydro,args.rcond)

if __name__=='__main__':main()
