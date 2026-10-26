#!/usr/bin/env python3
"""Moving-window sensitivity of the supplied Brazos diagnostic, float64 (Figure S3).

Same-scenario approximation and matched-budget control for the 14 SSP2-4.5 members
and six window / update-interval settings.
"""
from __future__ import annotations
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
import argparse, ast, dataclasses, gc, hashlib, json, sys, time, typing, warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # analyses/ (common)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import RAW, WORK as WORK_ROOT
from common.pipeline_loader import SOURCE, SOURCE_SHA256, source_sha256, return_levels_path
import data_support as ds
import kernel

HERE = Path(__file__).resolve().parent
OUT = WORK_ROOT/'window_sensitivity'   # primary/<member>/<setting>/ and implementation_audit.json
WORK = OUT                             # cache/<member>/<setting>.npz (fitted models)
DATA = RAW                             # streamflow/ and precipitation/ (ORNL HydroSource parquet files)
GCMS = ['MPI-ESM1-2-HR','ACCESS-CM2','EC-Earth3','CNRM-ESM2-1','BCC-CSM2-MR','MRI-ESM2-0','NorESM2-MM']
CASES = [(h,g) for h in ['PRMS','VIC5'] for g in GCMS]
SETTINGS = {'10_5':(10,5),'5_2p5':(5,2.5),'15_5':(15,5),'20_5':(20,5),'20_10':(20,10),'5_5':(5,5)}
ALPHAS = [.01,.1,.5,1.,2.,5.,10.,25.,50.]
RANK = 30
DT = 86400.
FACTOR = DT/80/1e9

def source_functions():
    selected = {'PODPack','ForcingPack','DMDcPack','compute_pod','normalize_forcing',
                'fit_windowed_dmdc','simulate_openloop','simulate_one_step_anchored',
                'lqr_gain','simulate_cl_flood_only','_nse','_kge'}
    if source_sha256()!=SOURCE_SHA256: raise RuntimeError(f'{SOURCE.name} differs from the pipeline the runs executed')
    tree=ast.parse(SOURCE.read_text())
    ns=dict(np=np,pd=pd,Ridge=Ridge,solve_discrete_are=solve_discrete_are,
            dataclass=dataclasses.dataclass,Optional=typing.Optional,List=typing.List,
            Tuple=typing.Tuple,RANK=30,DTYPE=np.float64,USE_TRUNCATED_SVD=True,
            WINDOW=3650,STEP=1825,ALPHA_CANDIDATES=ALPHAS,SPECTRAL_RADIUS_LIMIT=.995,
            PROF=SimpleNamespace(start=lambda _:None,stop=lambda _:None))
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in selected],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns

def bounds_for(nt, length, step):
    # Reproduce range(0,max(1,Tm1-WINDOW),STEP) and the terminal window.
    # Fractional step uses rounded cumulative offsets, never rounded increments.
    window=int(round(365*length)); stride=365*step
    starts=[]; j=0
    while j*stride < max(1,nt-window):
        starts.append(int(np.floor(j*stride+.5))); j+=1
    bounds=[(a,min(a+window,nt)) for a in starts]
    if bounds[-1][1]<nt: bounds.append((max(0,nt-window),nt))
    assert len(bounds)==len(set(bounds))
    return bounds

def fit_one(z,v,idx):
    design=np.concatenate([z[:,idx],v[:,idx]],axis=0).T
    target=z[:,idx+1].T
    notes=[]
    for alpha in ALPHAS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reg=Ridge(alpha=alpha,fit_intercept=False,solver='auto').fit(design,target)
        notes.extend(str(w.message) for w in caught)
        a=reg.coef_[:,:z.shape[0]].copy(); e=reg.coef_[:,z.shape[0]:].copy()
        radius=float(np.max(np.abs(np.linalg.eigvals(a))))
        if radius < .995: break
    return a,e,alpha,radius,notes

def fit_windows(ns,z,v,dates,setting):
    nt=z.shape[1]-1; bounds=bounds_for(nt,*SETTINGS[setting])
    aa=[];ee=[];rows=[];mapping=np.full(nt,-1,dtype=int)
    for w,(lo,hi) in enumerate(bounds):
        idx=np.arange(lo,hi)
        assert len(idx)>RANK
        cache=ns.setdefault('_fit_cache',{})
        cache_key=(lo,hi)
        reused=cache_key in cache
        if not reused:cache[cache_key]=fit_one(z,v,idx)
        a,e,alpha,radius,notes=cache[cache_key]
        row=dict(window=w,start_day=lo,end_day_exclusive=hi,start=str(dates[lo].date()),
                 last_target=str(dates[hi].date()),n_train_pairs=len(idx),alpha=alpha,
                 spectral_radius=radius,passes_stability_target=radius<.995,
                 warnings=' | '.join(notes),reused_identical_fit=reused)
        if aa:
            row['A_relative_change']=float(np.linalg.norm(a-aa[-1])/max(np.linalg.norm(aa[-1]),1e-30))
            row['E_relative_change']=float(np.linalg.norm(e-ee[-1])/max(np.linalg.norm(ee[-1]),1e-30))
            row['start_gap_years']=(lo-bounds[w-1][0])/365
        # A common five-year gap for comparing lengths; only naturally aligned starts.
        previous=[j for j,b in enumerate(bounds[:w]) if abs(lo-b[0]-1825)<=1]
        if previous:
            j=previous[-1]
            row['A_change_5yr']=float(np.linalg.norm(a-aa[j])/max(np.linalg.norm(aa[j]),1e-30))
            row['E_change_5yr']=float(np.linalg.norm(e-ee[j])/max(np.linalg.norm(ee[j]),1e-30))
        aa.append(a);ee.append(e);rows.append(row);mapping[lo:hi]=w
    assert np.all(mapping>=0)
    dmd=ns['DMDcPack'](aa,ee,[np.zeros((RANK,1)) for _ in aa],mapping,bounds,
                         [r['spectral_radius'] for r in rows],[r['alpha'] for r in rows])
    return dmd,pd.DataFrame(rows)

def force_for(dmd,v):
    nt=len(dmd.window_for_k); r=dmd.A_list[0].shape[0]
    force=np.empty((r,nt)); cuts=np.r_[0,np.flatnonzero(np.diff(dmd.window_for_k))+1,nt]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        w=dmd.window_for_k[lo]; force[:,lo:hi]=dmd.E_list[w]@v[:,lo:hi]
    return force

def rollout(dmd,z,force):
    pred=np.empty_like(z); pred[:,0]=z[:,0]
    for k,w in enumerate(dmd.window_for_k): pred[:,k+1]=dmd.A_list[w]@pred[:,k]+force[:,k]
    return pred

def score(ns,x,basis,mu,z,thr,selection=None,years=80):
    # Primary metrics omit final day to match source production accounting.
    selection=np.arange(x.shape[1]-1) if selection is None else np.asarray(selection)
    chunks=[]; netobs=np.zeros(len(selection)); netsim=np.zeros(len(selection)); neg=0
    for lo in range(0,len(x),256):
        hi=min(lo+256,len(x)); obs=x[lo:hi][:,selection]
        raw=basis[lo:hi]@z[:,selection]+mu[lo:hi]
        neg+=int(np.count_nonzero(raw<0)); sim=np.maximum(raw,0)
        mean=obs.mean(1); var=obs.var(1); so=obs.std(1); ss=sim.std(1)
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            nse=1-np.mean((sim-obs)**2,axis=1)/var
            corr=np.mean((obs-mean[:,None])*(sim-sim.mean(1)[:,None]),axis=1)/(so*ss)
            kge=1-np.sqrt((corr-1)**2+(ss/so-1)**2+(sim.mean(1)/mean-1)**2)
            pbias=(sim.mean(1)-mean)/mean
        valid=(var>1e-16)&(mean>1e-10)
        for vals in [nse,kge,pbias]: vals[~valid|~np.isfinite(vals)]=np.nan
        threshold=thr[lo:hi,None]
        q95=np.quantile(obs,.95,axis=1)[:,None]
        high=obs>q95
        chunks.append(dict(nse=nse,kge=kge,abs_pbias=np.abs(pbias),
            production_nse=ns['_nse'](obs,sim),production_kge=ns['_kge'](obs,sim),
            truth_excess=np.maximum(obs-threshold,0).sum(1)*DT/years/1e9,
            surrogate_excess=np.maximum(sim-threshold,0).sum(1)*DT/years/1e9,
            high_sse=(((sim-obs)**2)*high).sum(1),high_ss=(obs**2*high).sum(1)))
        netobs+=obs.sum(0); netsim+=sim.sum(0)
    per={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    row={k+'_median':float(np.nanmedian(per[k])) for k in ['nse','kge','abs_pbias','production_nse','production_kge']}
    row.update(nse_p25=float(np.nanquantile(per['nse'],.25)),kge_p25=float(np.nanquantile(per['kge'],.25)),
        nse_positive_fraction=float(np.nanmean(per['nse']>0)),
        truth_excess_Gm3yr=float(per['truth_excess'].sum()),surrogate_excess_Gm3yr=float(per['surrogate_excess'].sum()),
        highflow_nrmse=float(np.sqrt(per['high_sse'].sum()/per['high_ss'].sum())),
        network_sum_nse=float(1-np.sum((netsim-netobs)**2)/np.sum((netobs-netobs.mean())**2)),
        negative_before_clip_fraction=neg/(len(x)*len(selection)))
    row['excess_volume_bias']=row['surrogate_excess_Gm3yr']/row['truth_excess_Gm3yr']-1
    return row,per

def policies(ns,model,thr,years,rhos):
    gains=[[ns['lqr_gain'](a,np.eye(RANK),float(rho)) for rho in rhos] for a in model['A']]
    try:
        stats,_,_,_,elapsed=kernel.simulate(model['basis'],model['mu'],thr,model['z0'],model['A'],gains,
            model['force'],model['windows'],years,annual=False)
    except FloatingPointError:
        if len(rhos)>1:
            mid=len(rhos)//2
            return policies(ns,model,thr,years,rhos[:mid])+policies(ns,model,thr,years,rhos[mid:])
        return [dict(rho=float(rhos[0]),budget=np.nan,status='nonfinite_rollout')]
    points=[]
    for j,rho in enumerate(rhos):
        effort=stats[0,j]*FACTOR; residual=stats[3,j]*FACTOR
        points.append(dict(rho=float(rho),budget=float(effort.sum()),status='finite',
            effort=effort,residual=residual,attenuation=stats[1,j]*FACTOR,
            augmentation=stats[2,j]*FACTOR,peak=stats[5,j]))
    return points

def match_budget(ns,model,thr,years,folder,target=10.):
    started=time.time(); pointfile=folder/'policy_points.npz'
    rhos=[1.,10.,100.,1000.,10000.,100000.]
    points=policies(ns,model,thr,years,rhos)
    logs=[]; chosen=None
    # A penalty-grid endpoint is not an unattainable physical effort budget.
    # Extend only the weak-feedback endpoint, evaluating every new policy.
    for extension in range(4):
        finite=[p for p in points if p['status']=='finite' and np.isfinite(p['budget'])]
        if not finite or min(p['budget'] for p in finite)<=target:break
        rho=max(p['rho'] for p in points)*10
        new=policies(ns,model,thr,years,[rho])[0];points.append(new)
        logs.append(dict(stage='weak_feedback_bracket_extension',rho=rho,budget=new['budget']))
    # Never interpolate reach allocations: rerun feedback at each trial rho.
    for iteration in range(12):
        valid=sorted([p for p in points if p['status']=='finite' and np.isfinite(p['budget'])],key=lambda p:p['rho'])
        near=min(valid,key=lambda p:abs(p['budget']/target-1)) if valid else None
        if near is not None and abs(near['budget']/target-1)<=.005:
            chosen=near;break
        brackets=[(a,b) for a,b in zip(valid[:-1],valid[1:]) if (a['budget']-target)*(b['budget']-target)<0]
        if not brackets: break
        a,b=brackets[-1] # deterministic weakest-feedback crossing
        fraction=float(np.clip((target-a['budget'])/(b['budget']-a['budget']),.1,.9))
        rho=float(10**(np.log10(a['rho'])+fraction*np.log10(b['rho']/a['rho'])))
        new=policies(ns,model,thr,years,[rho])[0]
        logs.append(dict(iteration=iteration,rho=rho,budget=new['budget'],bracket_count=len(brackets)))
        points.append(new)
        if new['status']!='finite': break
    table=[{k:p[k] for k in ['rho','budget','status']}|({'residual_Gm3yr':float(p['residual'].sum()),'peak_max':float(p['peak'].max())} if p['status']=='finite' else {}) for p in points]
    pd.DataFrame(table).to_csv(folder/'policy_points.csv',index=False)
    (folder/'budget_match_log.json').write_text(json.dumps(logs,indent=2))
    if chosen is not None:
        np.savez_compressed(folder/'matched_budget10.npz',**{k:v for k,v in chosen.items() if k!='status'})
    return dict(budget_target=target,budget_matched=chosen is not None,
        actual_budget=chosen['budget'] if chosen else None,
        controlled_residual_Gm3yr=float(chosen['residual'].sum()) if chosen else None,
        matched_rho=chosen['rho'] if chosen else None,control_seconds=time.time()-started)

def audit():
    ns=source_functions(); rng=np.random.default_rng(1909)
    x=rng.normal(size=(17,105)); v=rng.normal(size=(17,105)); pod=ns['compute_pod'](x,5)
    old=SETTINGS.get('test'); SETTINGS['test']=(40/365,20/365)
    global RANK
    rank_save=RANK; RANK=5
    ns.update(WINDOW=40,STEP=20)
    d,rows=fit_windows(ns,pod.Z,v,pd.date_range('2020-01-01',periods=105),'test')
    original=ns['fit_windowed_dmdc'](pod.Z,v)
    assert d.bounds==original.bounds
    np.testing.assert_array_equal(d.window_for_k,original.window_for_k)
    for a,b in zip(d.A_list+d.E_list,original.A_list+original.E_list):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
    f=force_for(d,v); z=rollout(d,pod.Z,f)
    np.testing.assert_allclose(z,ns['simulate_openloop'](d,pod.Z[:,:1],v[:,:-1]),atol=1e-10)
    rhos=[.1,10,1000]; gains=[[ns['lqr_gain'](a,np.eye(5),r) for r in rhos] for a in d.A_list]
    thr=np.zeros(17)
    stats,am,zz,uu,_=kernel.simulate(pod.U_r,pod.mean_x,thr,pod.Z[:,0],d.A_list,gains,f,d.window_for_k,np.arange(104)//40,trace=True)
    for j,rho in enumerate(rhos):
        ez,eu=ns['simulate_cl_flood_only'](d,pod.Z[:,:1],v[:,:-1],[g[j] for g in gains],thr,pod.U_r,pod.mean_x)
        np.testing.assert_allclose(zz[:,j].T,ez,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(uu[:,j].T,eu,atol=1e-9,rtol=1e-9)
        flow=np.maximum(pod.U_r@ez[:,:-1]+pod.mean_x,0); control=pod.U_r@eu
        np.testing.assert_allclose(stats[0,j],np.abs(control).sum(1),atol=1e-8)
        np.testing.assert_allclose(stats[3,j],flow.sum(1),atol=1e-8)
    RANK=rank_save;del SETTINGS['test']
    calendar={}
    for name,(length,step) in SETTINGS.items():
        b=bounds_for(29219,length,step); mapping=np.full(29219,-1)
        for w,(lo,hi) in enumerate(b):mapping[lo:hi]=w
        assert (mapping>=0).all() and b[-1][1]==29219
        calendar[name]=dict(windows=len(b),last_start=b[-1][0],last_scheduled_gap=b[-1][0]-b[-2][0])
    return dict(source_sha256=source_sha256(),
        baseline_window_fit_parity='passed',recursive_forcing_parity='passed',
        gated_control_and_physical_metrics_parity='passed',complete_day_coverage='passed',windows=calendar)

def run_case(index,ssp,settings,threads):
    threadpool_limits(limits=threads); ns=source_functions(); hydro,gcm=CASES[index]
    mode='primary'; key=f'{hydro}_{gcm}_ssp{ssp}'
    caseout=OUT/mode/key; caseout.mkdir(parents=True,exist_ok=True)
    if all((caseout/s/'summary.json').exists() for s in settings):return
    tick=time.time(); print(f'BEGIN {mode} {key}',flush=True)
    flow,precip,dates,ids,unmatched=ds.load_data(SimpleNamespace(data_root=DATA,hydro=hydro,gcm=gcm,ssp=ssp),np.dtype('float64'))
    assert dates.equals(pd.date_range('2020-01-01','2099-12-31')) and len(set(ids))==len(ids)
    manifest=[]
    for path in sum(ds._future_paths(DATA,gcm,hydro,ssp),[]):
        manifest.append(dict(path=str(path.relative_to(DATA)),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
    bad=int(np.count_nonzero(~np.isfinite(flow))); assert not np.isinf(flow).any()
    if bad:np.nan_to_num(flow,copy=False)
    rlpath=return_levels_path(hydro,gcm)
    rl=np.load(rlpath)
    reference_ids=rl['reach_ids'].astype(int).tolist()
    reordered_to_reference=ids!=reference_ids
    if reordered_to_reference:
        assert set(ids)==set(reference_ids)
        reorder=pd.Index(ids).get_indexer(reference_ids)
        flow=flow[reorder];precip=precip[reorder];ids=reference_ids
        print(f'Realigned state and forcing to original threshold-reference reach order: {key}',flush=True)
    positions=pd.Index(rl['reach_ids']).get_indexer(ids); assert (positions>=0).all()
    thr=np.maximum(rl['rl2'][positions],0)
    pod=ns['compute_pod'](flow,RANK)
    # In-place algebra identical to normalize_forcing, avoiding two full copies.
    precip-=precip.mean(1,keepdims=True);sigma=np.std(precip)+1e-12;precip/=sigma;v=precip
    metadata=dict(hydro=hydro,gcm=gcm,ssp=ssp,n_reaches=len(ids),n_days=len(dates),
        missing_precip_reaches_filled_zero=unmatched,nonfinite_flow_filled_zero=bad,reordered_to_reference=reordered_to_reference,
        negative_flow_count=int(np.count_nonzero(flow<0)),input_manifest=manifest,
        historical_threshold_cache=f'pooled_return_levels/{rlpath.name}',threshold_sha256=hashlib.sha256(rlpath.read_bytes()).hexdigest(),
        pod_fitted_on='2020-2099 all days',
        pod_basis_sha256=hashlib.sha256(pod.U_r.tobytes()).hexdigest(),forcing_sigma=float(sigma))
    (caseout/'data_audit.json').write_text(json.dumps(metadata,indent=2))
    for setting in settings:
        folder=caseout/setting;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists():continue
        started=time.time();dmd,fitrows=fit_windows(ns,pod.Z,v,dates,setting)
        fitrows.to_csv(folder/'window_fits.csv',index=False)
        force=force_for(dmd,v)
        print(f'FIT {key} {setting} {time.time()-started:.1f}s max_radius={max(dmd.spectral_radii):.6f}',flush=True)
        common=dict(hydro=hydro,gcm=gcm,ssp=ssp,setting=setting,window_years=SETTINGS[setting][0],step_years=SETTINGS[setting][1],
            mode=mode,n_windows=len(dmd.bounds),stability_target_failures=int((fitrows.spectral_radius>=.995).sum()),
            unstable_windows=int((fitrows.spectral_radius>=1).sum()),spectral_radius_max=float(max(dmd.spectral_radii)),
            alpha_median=float(np.median(dmd.alphas_used)),
            A_change_5yr_median=float(fitrows.get('A_change_5yr',pd.Series([np.nan])).median()),
            E_change_5yr_median=float(fitrows.get('E_change_5yr',pd.Series([np.nan])).median()))
        zp=rollout(dmd,pod.Z,force)
        row,per=score(ns,flow,pod.U_r,pod.mean_x,zp,thr)
        np.savez_compressed(folder/'reach_metrics.npz',reach_ids=ids,**per)
        model=dict(basis=pod.U_r,mu=pod.mean_x,z0=pod.Z[:,0],A=np.stack(dmd.A_list),
            force=force,windows=dmd.window_for_k)
        # Store small fitted models for exact follow-up without reloading 7 GB.
        cache=WORK/'cache'/key;cache.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache/f'{setting}.npz',**model,threshold=thr,years=dates[:-1].year.to_numpy()-2020)
        control=match_budget(ns,model,thr,dates[:-1].year.to_numpy()-2020,folder)
        summary=dict(common,**row,**control)
        if control['budget_matched']:
            summary['reduction_Gm3yr']=row['surrogate_excess_Gm3yr']-control['controlled_residual_Gm3yr']
            summary['reduction_percent']=100*summary['reduction_Gm3yr']/row['surrogate_excess_Gm3yr']
        del model,zp,per
        summary['elapsed_seconds']=time.time()-started
        (folder/'summary.json').write_text(json.dumps(summary,indent=2))
        print(f'DONE {mode} {key} {setting}: NSE={summary.get("nse_median",np.nan):.4f}; elapsed={time.time()-started:.1f}s',flush=True)
        del dmd,force;gc.collect()
    print(f'CASE COMPLETE {mode} {key} {time.time()-tick:.1f}s',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default='0');p.add_argument('--ssp',type=int,default=245)
    p.add_argument('--settings',default=','.join(SETTINGS));p.add_argument('--threads',type=int,default=3)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--audit-only',action='store_true')
    args=p.parse_args(); threadpool_limits(limits=args.threads)
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    (OUT/'implementation_audit.json').write_text(json.dumps(audit(),indent=2))
    if args.audit_only: print('All implementation audits PASSED');return
    cases=list(map(int,args.cases.split(','))); settings=args.settings.split(',')
    if args.workers==1:
        for c in cases:run_case(c,args.ssp,settings,args.threads)
    else:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
            jobs=[pool.submit(run_case,c,args.ssp,settings,args.threads) for c in cases]
            for j in as_completed(jobs):j.result()

if __name__=='__main__':main()
