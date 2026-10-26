#!/usr/bin/env python3
"""Aggregate paired results without treating reaches/windows as replicates.

Reads the per-run outputs of run_sensitivity.py and refine_unmatched.py (<FAC_WORK_ROOT>/window_sensitivity/primary/)
and writes member_results.csv (one row per member and setting) and paired_changes_vs_10_5.csv (member-paired changes
relative to the 10/5-year setting) next to them. package_tables.py turns these into the archived tables.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import WORK
import json, argparse
import numpy as np
import pandas as pd

OUT=WORK/'window_sensitivity'

def rank_compare(current,ref,ids):
    good=np.isfinite(current)&np.isfinite(ref)
    count=int(np.ceil(.1*good.sum()))
    a=np.flatnonzero(good); first=a[np.lexsort((ids[a],-current[a]))[:count]]
    second=a[np.lexsort((ids[a],-ref[a]))[:count]]
    overlap=len(np.intersect1d(first,second))/count if count else np.nan
    return overlap

def analyze(allow_incomplete):
    records=[json.loads(p.read_text()) for p in sorted((OUT/'primary').glob('*/*/summary.json'))]
    df=pd.DataFrame(records)
    if not allow_incomplete:assert len(df)==84, f'Expected 84 primary runs, found {len(df)}'
    df.to_csv(OUT/'member_results.csv',index=False)
    paired=[]
    for (hydro,gcm,ssp),group in df.groupby(['hydro','gcm','ssp']):
        reference=group[group.setting=='10_5']
        if len(reference)!=1:continue
        ref=reference.iloc[0]; root=OUT/'primary'/f'{hydro}_{gcm}_ssp{ssp}'
        refmatch=root/'10_5/matched_budget10.npz'
        for _,row in group.iterrows():
            item=dict(hydro=hydro,gcm=gcm,ssp=ssp,setting=row.setting,
                paired_budget_matched=bool(row.budget_matched and ref.budget_matched))
            if item['paired_budget_matched']:
                item.update(delta_reduction_pp=row.reduction_percent-ref.reduction_percent)
                current=np.load(root/row.setting/'matched_budget10.npz');base=np.load(refmatch)
                ids=np.load(root/row.setting/'reach_metrics.npz')['reach_ids']
                # Top decile of reaches by annual absolute effort.
                item['top10_overlap']=rank_compare(current['effort'],base['effort'],ids)
            paired.append(item)
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'paired_changes_vs_10_5.csv',index=False)
    print(df.groupby('setting')[['nse_median','reduction_percent']].median().to_string())
    print(paired.groupby('setting')[['delta_reduction_pp','top10_overlap']].median().to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-incomplete',action='store_true');args=p.parse_args()
    analyze(args.allow_incomplete)
