#!/usr/bin/env python3
"""All 14 SSP245 members; exact original fits and gated policy grid, explicit availability.

No model repair/alpha extension. The original fallback operator is also evaluated,
but a margin failure never enters the strict admissible ensemble.

Heavy rerun of the reduced-dimension sensitivity (Figure S1, Table S1 top and middle blocks): for every member and
r in RANKS, POD, the 16-window DMDc fit, same-period accuracy, the 300-mode spectrum, the 19-weight gated LQR grid
(compiled kernel, analyses/common/build_kernels.sh) and interpolation to 10 Gm3/yr. Reads the ORNL HydroSource
SSP2-4.5 streamflow and precipitation under FAC_RAW_ROOT and the pooled return levels of outputs_common; writes
<FAC_WORK_ROOT>/rank_sensitivity/members/<hydro>_<gcm>/ and a model cache in <FAC_WORK_ROOT>/rank_sensitivity/cache/.
Then run analyze_ensemble.py.

Three members (PRMS/EC-Earth3, PRMS/MPI-ESM1-2-HR, VIC5/NorESM2-MM) were taken from earlier runs of the same fits
and policy grid (reuse_old, get_reference, full_spectrum). Those runs are not archived; without
<FAC_WORK_ROOT>/rank_sensitivity/earlier_runs the three members are computed here like the other eleven.
"""
from pathlib import Path
import os,sys,argparse,json,time,gc,hashlib,shutil,warnings,traceback
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','5')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))   # analyses/ (common)
sys.path.insert(0,str(HERE.parents[1]))   # repository root (paths)
from paths import RAW, work
from common.pipeline_loader import raw_inputs, return_levels_path, SOURCE, SOURCE_SHA256, source_sha256
OUT=work('rank_sensitivity');WORK=OUT
OLD=OUT/'earlier_runs';SRC=RAW
sys.path.insert(0,str(HERE/'vendor'))
import run_rank_control as rc
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from rank_utils import GCMS,CASES,RANKS,dump,nested_pod,norm2

EXPECTED_SOURCE=SOURCE_SHA256
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(p):
    s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def configure():
    OUT.mkdir(parents=True,exist_ok=True)
    assert source_sha256()==EXPECTED_SOURCE
    paths=[]
    ns=rc.source_functions()
    for h,g in CASES:
        realization=ns['_realization_id'](g)
        names=[SRC/'streamflow'/f'{h}_RAPID_{g}_ssp245_{realization}_DBCCA_Daymet_{a}_{b}_daily.parquet' for a,b in [(2020,2059),(2060,2099)]]
        names += [SRC/'precipitation'/f'{g}_ssp245_{realization}_DBCCA_Daymet_prcp_{a}_{b}_daily.parquet' for a,b in [(2020,2059),(2060,2099)]]
        names += [return_levels_path(h,g)]
        for p in names:paths.append(stat(p))
    config=dict(cases=CASES,scenario=245,ranks=sorted(RANKS),n_members=14,target_budget_Gm3_year=10,
       period='2020-2099',n_days=29220,source_sha256=EXPECTED_SOURCE,source=SOURCE.name,
       original_alpha_grid=rc.ORIGINAL_ALPHAS,operator_margin=.995,operator_repair=False,
       policy_grid=rc.GRID.tolist(),closed_loop='original lifted, threshold-gated LQR; no reanchoring',
       budget='member-level original-grid interpolation; no endpoint substitution, no extrapolation',
       aggregation='equal-member reachwise median after individual budget matching; no renormalization after median',
       strict_membership='fixed intersection across compared ranks, all 16 windows below 0.995 and admissible B10',
       full14_diagnostic='original fallback fits retained with flags; unavailable budget means no complete 14-member map',
       priority_definition='top decile of reaches by ensemble-median raw effort, floor(N*0.1), ties broken by COMID',
       comparison_reach_universe='Fixed intersection of finite scores over all available ranks in each comparison scope.',
       historical_threshold='original pooled RL2 cache, or previously recomputed historical RL2 for reused cases; fixed across ranks; cache differences and hashes recorded',
       numerical_runaway='nonfinite or max physical peak / process max peak > 1e6, same as earlier audit',
       inputs=paths)
    dump(OUT/'configuration.json',config)

def get_reference(ns,h,g,case,x,dates,ids):
    p=return_levels_path(h,g)
    rl=dict(np.load(p));positions=pd.Index(rl['reach_ids']).get_indexer(ids);assert np.all(positions>=0)
    thr=rl['rl2'][positions]
    assert np.isfinite(thr).all() and (thr>=0).all()
    prior=OLD/'reused/rank_control'/f'{h}_{g}'/'reference.npz'
    difference={}
    if prior.exists():
        ref=dict(np.load(prior));assert np.array_equal(ids,ref['reach_ids'])
        difference=dict(maximum_absolute_cache_difference=float(np.max(abs(thr-ref['threshold']))),
            maximum_relative_cache_difference=float(np.max(abs(thr-ref['threshold'])/np.maximum(abs(thr),1e-10))))
        # Previously recomputed historical inputs round at about 1e-7 relative
        # to the old cache. Preserve their exact threshold at EVERY rank in
        # those cases so reuse does not introduce within-case threshold changes.
        np.testing.assert_allclose(thr,ref['threshold'],atol=1e-6,rtol=2e-7)
    else:
        ref=rc.baseline_metrics(x,dates)
        residual=[]
        for lo in range(0,len(ids),256):
            hi=min(lo+256,len(ids));residual.append(np.maximum(x[lo:hi,:-1]-thr[lo:hi,None],0).sum(1)*86400/80)
        ref.update(reach_ids=np.array(ids),threshold=thr,mean_flow_reference=x[:,dates.year<=2039].mean(1),
          mean_flow_full=x.mean(1),process_residual=np.concatenate(residual),years=dates[:-1].year.to_numpy()-2020)
    np.savez_compressed(case/'reference.npz',**ref)
    dump(case/'threshold_provenance.json',dict(**stat(p),sha256=sha(p),sample_counts=np.unique(rl.get('sample_n',[])).tolist(),
        uses_previously_recomputed_threshold=prior.exists(),previous_reference=str(prior) if prior.exists() else None,
        previous_reference_sha256=sha(prior) if prior.exists() else None,**difference))
    return ref

def full_spectrum(x,case,h,g):
    prior=OLD/'spectra'/f'{h}_{g}'
    if (case/'spectrum.csv').exists():return
    if (prior/'spectrum.csv').exists():
        for name in ['spectrum.csv','spectrum_metadata.json','spectrum_numerical_check.csv']:shutil.copy2(prior/name,case/name)
        return
    t=time.time();mu,b,z,s,energy=nested_pod(x,300)
    cum=np.cumsum(s*s)/energy
    pd.DataFrame(dict(mode=np.arange(1,301),singular_value=s,singular_value_relative=s/s[0],
        cumulative_fraction=cum,reconstruction_relative_l2=np.sqrt(np.maximum(1-cum,0)))).to_csv(case/'spectrum.csv',index=False)
    dump(case/'spectrum_metadata.json',dict(total_centered_energy=energy,modes=300,
         omitted_tail_fraction=1-cum[-1],seconds=time.time()-t))
    del mu,b,z,s;gc.collect()
    print('SPECTRUM',h,g,round(time.time()-t,1),flush=True)

def accuracy(x,pod,dmd,v,ref,case,r):
    an,recur=rc.ap.accelerate(dmd,pod.Z,v)
    n=x.shape[1]-1;tss=norm2(x[:,:n]-pod.mean_x)
    sse={k:0. for k in ['pod','one_step_raw','one_step','recursive_raw','recursive']}
    allmetrics={'one_step':[],'recursive':[]}
    for lo in range(0,len(x),256):
        hi=min(lo+256,len(x));obs=x[lo:hi,:n];mu=pod.mean_x[lo:hi];b=pod.U_r[lo:hi]
        sse['pod']+=norm2(obs-b@pod.Z[:,:n]-mu)
        for key,z in [('one_step',an),('recursive',recur)]:
            raw=b@z[:,:n]+mu;sse[key+'_raw']+=norm2(obs-raw);pred=np.maximum(raw,0);sse[key]+=norm2(obs-pred)
            allmetrics[key].append(rc.ap.vm.statistics(obs,pred,ref['threshold'][lo:hi]))
    row=dict(rank=r,days=n,operator_pass=bool(max(dmd.spectral_radii)<.995),max_radius=max(dmd.spectral_radii),
      retained_energy=1-sse['pod']/tss,pod_error=(sse['pod']/tss)**.5)
    for key in ['one_step','recursive']:
        arr={k:np.concatenate([a[k] for a in allmetrics[key]]) for k in allmetrics[key][0]}
        row.update({f'{key}_{k}':v for k,v in rc.ap.vm.summarize(arr).items()})
        row[key+'_RMSE']=(sse[key]/(len(x)*n))**.5;row[key+'_raw_relative_L2']=(sse[key+'_raw']/tss)**.5
    dump(case/f'accuracy_r{r}.json',row)
    del an,recur;gc.collect()

def model_fit(ns,x,v,dates,ref,case,cache,r):
    model_file=cache/f'model_r{r}.npz'
    if model_file.exists():return dict(np.load(model_file))
    start=time.time();pod=ns['compute_pod'](x,r)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');dmd=ns['fit_windowed_dmdc'](pod.Z,v)
    rows=[dict(window=i,start=str(dates[a].date()),end=str(dates[b].date()),alpha=alpha,spectral_radius=rho,
         applied_days=int(np.sum(dmd.window_for_k==i))) for i,((a,b),alpha,rho) in enumerate(zip(dmd.bounds,dmd.alphas_used,dmd.spectral_radii))]
    pd.DataFrame(rows).to_csv(case/f'window_fits_r{r}.csv',index=False)
    accuracy(x,pod,dmd,v,ref,case,r)
    force=np.empty((r,len(dates)-1));cuts=np.r_[0,np.flatnonzero(np.diff(dmd.window_for_k))+1,len(dates)-1]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        w=dmd.window_for_k[lo];force[:,lo:hi]=dmd.E_list[w]@v[:,lo:hi]+dmd.b_list[w]
    model=dict(basis=pod.U_r,mu=pod.mean_x,z0=pod.Z[:,0],force=force,A=np.stack(dmd.A_list),windows=dmd.window_for_k)
    if r==30:
        days=100;rho=1000.;gains=[[ns['lqr_gain'](a,np.eye(r),rho)] for a in dmd.A_list]
        _,_,zz,uu,_=rc.kernel.simulate(model['basis'],model['mu'],ref['threshold'],model['z0'],model['A'],gains,
            force[:,:days],model['windows'][:days],ref['years'][:days],trace=True)
        z0,u0=ns['simulate_cl_flood_only'](dmd,pod.Z[:,:1],v[:,:days],[a[0] for a in gains],ref['threshold'],pod.U_r,pod.mean_x)
        np.testing.assert_allclose(zz[:,0].T,z0,rtol=1e-7,atol=1e-6);np.testing.assert_allclose(uu[:,0].T,u0,rtol=1e-7,atol=1e-6)
        dump(case/'recurrence_parity.json',dict(passed=True,days=days,max_latent_abs_difference=float(np.max(abs(zz[:,0].T-z0)))))
    np.savez_compressed(model_file,**model)
    dump(case/f'model_r{r}.json',dict(rank=r,operator_pass=bool(max(dmd.spectral_radii)<.995),max_radius=max(dmd.spectral_radii),
        fit_seconds=time.time()-start,warnings=[str(a.message) for a in caught],alpha_extension=False))
    print('MODEL',case.name,r,round(time.time()-start,1),'rho',max(dmd.spectral_radii),flush=True)
    del pod,dmd;gc.collect();return model

def grid_run(ns,model,ref,case,r):
    target=case/f'grid_r{r}.npz'
    if target.exists():return
    start=time.time();n=len(ref['reach_ids']);results={};failures=[]
    def run(indices):
        try:
            gains=[[ns['lqr_gain'](a,np.eye(r),float(rc.GRID[j])) for j in indices] for a in model['A']]
            stats,am,_,_,runtime=rc.kernel.simulate(model['basis'],model['mu'],ref['threshold'],model['z0'],model['A'],
                 gains,model['force'],model['windows'],ref['years'])
            packed=rc.pack_metrics(stats,am,80)
            for k,j in enumerate(indices):results[j]={name:a[k] for name,a in packed.items()}
        except Exception as e:
            if len(indices)>1:
                mid=len(indices)//2;run(indices[:mid]);run(indices[mid:])
            else:
                failures.append(dict(policy_index=indices[0],rho=float(rc.GRID[indices[0]]),reason=repr(e)))
                print('POLICY FAILED',case.name,r,indices,repr(e),flush=True)
    run(list(range(len(rc.GRID))))
    names=next(iter(results.values())).keys() if results else ['effort','attenuation','augmentation','residual','peak','rl10']
    arrays={k:np.stack([results[j][k] if j in results else np.full(n,np.nan) for j in range(len(rc.GRID))]) for k in names}
    np.savez_compressed(target,rhos=rc.GRID,**arrays)
    dump(case/f'grid_audit_r{r}.json',dict(seconds=time.time()-start,failed_policies=failures))
    print('GRID',case.name,r,round(time.time()-start,1),flush=True)

def interpolate(case,ref,r):
    d=dict(np.load(case/f'grid_r{r}.npz'));budget=d['effort'].sum(1)/1e9
    ratio=d['peak'].max(1)/ref['peak'].max();bad=~np.isfinite(budget)|~np.isfinite(ratio)|(ratio>1e6)
    # Every metric must be finite, not only effort/peak.
    for k,v in d.items():
        if k!='rhos':bad |= ~np.isfinite(v).all(1)
    order=np.flatnonzero(~bad);order=order[np.argsort(budget[order])]
    fit=json.loads((case/f'model_r{r}.json').read_text())
    info=dict(rank=r,operator_pass=fit['operator_pass'],max_radius=fit['max_radius'],runaway_policies=int(bad.sum()),
        valid_grid_points=len(order),status='no_admissible_budget_bracket',strict_valid=False,finite_bracket_valid=False)
    if len(order):info.update(min_budget=float(budget[order[0]]),max_budget=float(budget[order[-1]]))
    if len(order)>=2 and budget[order[0]]<=10<=budget[order[-1]]:
        j=int(np.searchsorted(budget[order],10));j=max(1,min(j,len(order)-1));lo,hi=order[j-1],order[j]
        weight=(10-budget[lo])/(budget[hi]-budget[lo]);gap=bool(bad[min(lo,hi):max(lo,hi)+1].any())
        info.update(status='runaway_gap' if gap else 'interpolated',lower_budget=float(budget[lo]),upper_budget=float(budget[hi]),
          lower_rho=float(d['rhos'][lo]),upper_rho=float(d['rhos'][hi]),weight=float(weight),spans_runaway_gap=gap,
          finite_bracket_valid=not gap,strict_valid=bool(not gap and fit['operator_pass']))
        if not gap:
            metrics={k:(1-weight)*v[lo]+weight*v[hi] for k,v in d.items() if k!='rhos'}
            assert abs(metrics['effort'].sum()/1e9-10)<1e-8
            np.savez_compressed(case/f'interpolated_r{r}_B10.npz',**metrics)
    pd.DataFrame(dict(rho=d['rhos'],budget=budget,peak_ratio=ratio,runaway=bad)).to_csv(case/f'policies_r{r}.csv',index=False)
    dump(case/f'status_r{r}.json',info);return info

def reuse_old(case,h,g,r):
    prev=OLD/'reused/rank_control'/f'{h}_{g}'
    if not (prev/f'grid_r{r}.npz').exists():return False
    meta=json.loads((prev/f'model_r{r}.json').read_text())
    if meta.get('extension_used'):return False
    shutil.copy2(prev/f'grid_r{r}.npz',case/f'grid_r{r}.npz')
    oldwin=pd.read_csv(prev/f'window_fits_r{r}.csv')
    oldwin.rename(columns={'original_alpha':'alpha','original_spectral_radius':'spectral_radius'}).to_csv(case/f'window_fits_r{r}.csv',index=False)
    meta.update(operator_pass=bool(meta['max_spectral_radius']<.995),max_radius=meta['max_spectral_radius'],
       reused=True,reused_path=str(prev),source_grid_sha256=sha(prev/f'grid_r{r}.npz'))
    dump(case/f'model_r{r}.json',meta)
    oldacc=pd.read_csv(OLD/'reused/approximation_error/summary.csv')
    q=oldacc[(oldacc.hydro==h)&(oldacc.gcm==g)&(oldacc['rank']==r)]
    info=dict(rank=r,reused=True,operator_pass=meta['operator_pass'],max_radius=meta['max_radius'])
    for _,row in q.iterrows():
        name={'POD':'pod','anchored':'one_step','recursive':'recursive'}[row.method]
        if name=='pod':info.update(retained_energy=row.retained_energy_fraction,pod_error=row.pooled_raw_relative_l2_error)
        else:info.update({name+'_raw_relative_L2':row.pooled_raw_relative_l2_error,name+'_kge_median':row.kge_median,name+'_nse_median':row.nse_median})
    dump(case/f'accuracy_r{r}.json',info)
    return True

def run_case(ci,threads,ranks):
    threadpool_limits(limits=threads);h,g=CASES[ci];key=f'{h}_{g}'
    case=OUT/'members'/key;cache=WORK/'cache'/key
    for d in [case,cache]:d.mkdir(parents=True,exist_ok=True)
    if all((case/f'status_r{r}.json').exists() for r in ranks):print('ALREADY DONE',key,flush=True);return
    tick=time.time();ns=rc.source_functions();print('LOADING',ci,key,flush=True)
    prior=OLD/'reused/rank_control'/key
    if (prior/'reference.npz').exists() and all((prior/f'grid_r{r}.npz').exists() for r in ranks):
        ids=np.load(prior/'reference.npz')['reach_ids']
        ref=get_reference(ns,h,g,case,None,None,ids);full_spectrum(None,case,h,g)
        for r in ranks:
            assert reuse_old(case,h,g,r)
            info=interpolate(case,ref,r)
            print('REUSED',key,r,'strict',info['strict_valid'],'finite',info['finite_bracket_valid'],flush=True)
        dump(case/'complete.json',dict(seconds=time.time()-tick,ranks=ranks,reused=True))
        (case/'execution_error.json').unlink(missing_ok=True)
        return
    with raw_inputs():x,p,dates,ids=ns['load_scenario_data'](245,g,h)
    assert dates.equals(pd.date_range('2020-01-01','2099-12-31')) and np.isfinite(x).all() and np.isfinite(p).all()
    v=ns['normalize_forcing'](p).V;del p;gc.collect()
    ref=get_reference(ns,h,g,case,x,dates,ids)
    full_spectrum(x,case,h,g)
    for r in ranks:
        if (case/f'status_r{r}.json').exists():continue
        if not reuse_old(case,h,g,r):
            model=model_fit(ns,x,v,dates,ref,case,cache,r);grid_run(ns,model,ref,case,r);del model;gc.collect()
        info=interpolate(case,ref,r)
        print('DONE',key,r,'strict',info['strict_valid'],'finite',info['finite_bracket_valid'],'status',info['status'],flush=True)
    dump(case/'complete.json',dict(seconds=time.time()-tick,ranks=ranks))
    (case/'execution_error.json').unlink(missing_ok=True)
    del x,v;gc.collect();print('CASE COMPLETE',key,round(time.time()-tick,1),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',default=','.join(map(str,range(14))));p.add_argument('--ranks',default=','.join(map(str,RANKS)))
    p.add_argument('--threads',type=int,default=5);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    configure()
    if a.audit_only:
        dump(OUT/'implementation_audit.json',rc.ap.audit(rc.source_functions()));print('AUDIT PASSED',flush=True);return
    failed=[]
    for ci in map(int,a.cases.split(',')):
        try:run_case(ci,a.threads,list(map(int,a.ranks.split(','))))
        except Exception as e:
            failed.append(ci);traceback.print_exc();case=OUT/'members'/('_'.join(CASES[ci]));case.mkdir(parents=True,exist_ok=True)
            dump(case/'execution_error.json',dict(error=repr(e),traceback=traceback.format_exc()))
    rows=[]
    for h,g in CASES:
        for r in RANKS:
            path=OUT/'members'/f'{h}_{g}'/f'status_r{r}.json'
            if path.exists():rows.append(dict(hydro=h,gcm=g,**json.loads(path.read_text())))
    pd.DataFrame(rows).to_csv(OUT/'member_rank_status.csv',index=False)
    print('FINISHED',len(rows),'of',84,flush=True)
    if failed:print('FAILED cases',failed,flush=True)
    if failed or len(rows)<84:sys.exit(1)
if __name__=='__main__':main()
