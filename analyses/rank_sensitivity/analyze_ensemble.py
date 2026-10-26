"""Aggregate the member runs of run_ensemble.py into the data of Figure S1 and Table S1 (top and middle blocks).

Light step (seconds). Reads <FAC_WORK_ROOT>/rank_sensitivity/members (or --members) and the reach list of outputs_common
(reach_attributes.csv, the reach order of the published r = 30 results), and writes to --out (default
<FAC_WORK_ROOT>/rank_sensitivity/aggregated) the files of the outputs_rank_sensitivity archive under the same names:

  member_rank_status.csv              operator margin, closed-loop bracket and admissibility for 14 members x 6 ranks
  singular_spectra_all14.csv          300-mode spectrum of each member
  same_period_accuracy_all14.csv      reach-median NSE and KGE of the recursive reconstruction, 2020-2099
  ensemble_rank_comparisons.csv       ensemble-median raw effort compared between r = 20, 30, 50 (14 members)
  aggregates/full14_original_fallback_r{20,30,50}.npz, aggregates/full14_original_fallback_common_reach_universe.npz

Scope full14_original_fallback: all 14 members at every rank where all of them reach 10 Gm3/yr within the policy
grid without a runaway gap (r = 10, 20, 30, 50); the reach universe of the comparisons is fixed over those ranks.

Release copy of the original aggregation step, reduced to these outputs.
"""
from pathlib import Path
import json,itertools,argparse,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]))   # repository root (paths)
from paths import data, work
from rank_utils import CASES,RANKS

SCOPE='full14_original_fallback'
MAP_RANKS=[20,30,50]
PAIRS=[(20,30),(30,50),(20,50)]
STATUS_COLUMNS=['hydro','gcm','rank','operator_pass','max_radius','strict_valid','finite_bracket_valid']
SPECTRUM_COLUMNS=['mode','singular_value_relative','cumulative_fraction','centered_reconstruction_relative_l2','hydro','gcm']
ACCURACY_COLUMNS=['hydro','gcm','rank','recursive_kge_median','recursive_nse_median']
COMPARISON_COLUMNS=['n_reaches','spearman','top10_n_a','top10_n_b','top10_common','top10_retention_a',
                    'scope','n_members','rank_a','rank_b','metric']

def top(a,ids,fraction=.1,valid=None,count=None):
    valid=np.isfinite(a) if valid is None else valid&np.isfinite(a)
    ix=np.flatnonzero(valid);k=max(1,int(len(ix)*fraction)) if count is None else count
    return set(ix[np.lexsort((ids[ix],-a[ix]))[:k]])
def setstats(a,b):
    common=len(a&b)
    return dict(n_a=len(a),n_b=len(b),common=common,retention_a=common/len(a) if a else np.nan)
def compare(a,b,ids,valid=None):
    valid=np.isfinite(a)&np.isfinite(b) if valid is None else valid&np.isfinite(a)&np.isfinite(b)
    d=dict(n_reaches=int(valid.sum()),spearman=float(spearmanr(a[valid],b[valid]).statistic))
    aa=top(a,ids,.1,valid);bb=top(b,ids,.1,valid)
    d.update({f'top10_{k}':v for k,v in setstats(aa,bb).items()})
    return d

def main():
    p=argparse.ArgumentParser(description='Aggregate the member runs of run_ensemble.py (Figure S1, Table S1).')
    p.add_argument('--members',type=Path,help='folder of member runs (default <FAC_WORK_ROOT>/rank_sensitivity/members)')
    p.add_argument('--out',type=Path,help='output folder (default <FAC_WORK_ROOT>/rank_sensitivity/aggregated)')
    a=p.parse_args()
    members_root=a.members or work('rank_sensitivity')/'members';out=a.out or work('rank_sensitivity')/'aggregated'
    (out/'aggregates').mkdir(parents=True,exist_ok=True)
    statuses={};rows=[]
    for h,g in CASES:
        for r in sorted(RANKS):
            path=members_root/f'{h}_{g}'/f'status_r{r}.json'
            if path.exists():
                d=json.loads(path.read_text());statuses[(h,g,r)]=d;rows.append(dict(hydro=h,gcm=g,**d))
    assert len(rows)==84,'All member-rank attempts must finish before aggregate interpretation.'
    pd.DataFrame(rows)[STATUS_COLUMNS].to_csv(out/'member_rank_status.csv',index=False)
    ids=pd.read_csv(data('common')/'reach_attributes.csv',usecols=['COMID'])['COMID'].to_numpy()
    arrays={}
    for h,g in CASES:
        root=members_root/f'{h}_{g}';ref=dict(np.load(root/'reference.npz'));idx=pd.Index(ref['reach_ids']).get_indexer(ids)
        assert np.all(idx>=0)
        for r in RANKS:
            if not statuses[(h,g,r)]['finite_bracket_valid']:continue
            d=dict(np.load(root/f'interpolated_r{r}_B10.npz'))
            arrays[(h,g,r)]=d['effort'][idx]
            assert abs(arrays[(h,g,r)].sum()/1e9-10)<1e-8
    allr=sorted(RANKS)
    members=CASES;ranks=[r for r in allr if all(statuses[(*c,r)]['finite_bracket_valid'] for c in CASES)]
    print('scope',SCOPE,'members',len(members),'ranks',ranks,flush=True)
    sc={}
    for r in ranks:
        matrix=np.stack([arrays[(*c,r)] for c in members]);sc[r]=np.median(matrix,axis=0)
        if r in MAP_RANKS:
            np.savez_compressed(out/'aggregates'/f'{SCOPE}_r{r}.npz',reach_ids=ids,effort=sc[r],
                n_members=len(members),members=np.array(['_'.join(c) for c in members]))
    raw_mask=np.logical_and.reduce([np.isfinite(sc[r]) for r in ranks])
    np.savez_compressed(out/'aggregates'/f'{SCOPE}_common_reach_universe.npz',reach_ids=ids,raw=raw_mask)
    comp=[]
    for ra,rb in itertools.combinations(ranks,2):
        if (ra,rb) not in PAIRS:continue
        c=compare(sc[ra],sc[rb],ids,valid=raw_mask)
        c.update(scope=SCOPE,n_members=len(members),rank_a=ra,rank_b=rb,metric='raw')
        comp.append(c)
    pd.DataFrame(comp)[COMPARISON_COLUMNS].to_csv(out/'ensemble_rank_comparisons.csv',index=False)
    accuracy=[];spectra=[]
    for h,g in CASES:
        root=members_root/f'{h}_{g}'
        spectrum=pd.read_csv(root/'spectrum.csv').rename(columns={'reconstruction_relative_l2':'centered_reconstruction_relative_l2'})
        spectrum['hydro']=h;spectrum['gcm']=g;spectra.append(spectrum)
        for r in allr:accuracy.append(dict(hydro=h,gcm=g,**json.loads((root/f'accuracy_r{r}.json').read_text())))
    pd.concat(spectra,ignore_index=True)[SPECTRUM_COLUMNS].to_csv(out/'singular_spectra_all14.csv',index=False)
    pd.DataFrame(accuracy)[ACCURACY_COLUMNS].to_csv(out/'same_period_accuracy_all14.csv',index=False)
    print('COMPLETE AGGREGATION',len(comp),'pairwise ensemble rows ->',out,flush=True)
if __name__=='__main__':main()
