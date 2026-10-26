#!/usr/bin/env python3
"""Write the data tables of the outputs_regularization archive from the run outputs.

Only what Figure S2 and its caption use is exported, and values are copied as text, so they are
unchanged:
- all_paths.csv: the rows with alpha > 0 and the columns read by make_figure.py. df_per_response is
  given for every alpha (panel a); rho, margin_pass (the selection rule rho(A) < 0.995),
  rollout_complete and raw_NSE (panel b) are given for the nine grid values only and left empty
  for the other alpha values.
- pilot/: the inputs of make_sensitivity_table.py (panel c), i.e. the 1% target-perturbation change
  norms of the selected-alpha ridge fit (default run) and of the alpha = 0 minimum-norm reference
  (rcond = 1e-8 run), the setup records of both runs, and the rcond = 1e-8 ratios for A and E.
  Setup records keep file names relative to FAC_RAW_ROOT (precipitation) and to the case folder
  (controller).

  python export_tables.py --src $FAC_WORK_ROOT/regularization --dest DIR
Then: python make_sensitivity_table.py --data DIR --out DIR
"""
import argparse, json
from pathlib import Path
import pandas as pd

HYDROS=['VIC5','PRMS']
REFERENCE='rcond_1e-8'
PATH_COLUMNS=['hydro','alpha','is_original','in_original_grid','n','p','df_per_response','rho','margin_pass',
    'rollout_complete','raw_NSE']
GRID_ONLY_COLUMNS=['rho','margin_pass','rollout_complete','raw_NSE']   # panel (b) and the selection rule
NOISE_COLUMNS=['hydro','seed','alpha','noise_relative_centered_target','delta_A_norm','delta_E_norm']
SETUP_KEYS=['case','n','p','bounds','dates','alpha_original','grid','precip_sources',
    'controller','controller_sha256','x_sha256','y_sha256']
REFERENCE_SETUP_KEYS=SETUP_KEYS+['singular_cutoff','numerical_rank']

def read_table(path):
    return pd.read_csv(path,dtype=str,keep_default_na=False)

def write_table(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,lineterminator='\n')

def tail(path,parts=2):
    return '/'.join(Path(path.replace('\\','/')).parts[-parts:])

def write_setup(src,dest,keys):
    d=json.loads(src.read_text())
    d={k:v for k,v in d.items() if k in keys}
    assert set(d)==set(keys),sorted(set(keys)-set(d))
    d['controller']=tail(d['controller'])
    d['precip_sources']=[tail(p) for p in d['precip_sources']]
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--src',type=Path,required=True,help='run folder with all_paths.csv and pilot/')
    parser.add_argument('--dest',type=Path,required=True,help='archive folder outputs_regularization')
    args=parser.parse_args()
    src=args.src/'pilot';cut=src/'numerical_cutoff_1e-08'
    dst=args.dest/'pilot';dcut=dst/'numerical_cutoff_1e-08'
    args.dest.mkdir(parents=True,exist_ok=True)
    paths=read_table(args.src/'all_paths.csv')
    paths=paths.loc[paths.alpha.astype(float)>0,PATH_COLUMNS].copy()
    assert len(paths)==92
    grid=paths.in_original_grid=='True'
    assert grid.sum()==18
    paths.loc[~grid,GRID_ONLY_COLUMNS]=''
    write_table(paths,args.dest/'all_paths.csv')
    for h in HYDROS:
        alpha=float(json.loads((src/f'{h}_setup.json').read_text())['alpha_original'])
        noise=read_table(src/f'{h}_noise.csv')
        ridge=noise.loc[noise.alpha.astype(float)==alpha,NOISE_COLUMNS]
        assert len(ridge)==20
        write_table(ridge,dst/f'{h}_noise.csv')
        ref=read_table(cut/f'{h}_noise.csv')
        ref=ref.loc[ref.alpha.astype(float)==0,NOISE_COLUMNS]
        assert len(ref)==20 and ref.seed.tolist()==ridge.seed.tolist()
        write_table(ref,dcut/f'{h}_noise.csv')
        write_setup(src/f'{h}_setup.json',dst/f'{h}_setup.json',SETUP_KEYS)
        write_setup(cut/f'{h}_setup.json',dcut/f'{h}_setup.json',REFERENCE_SETUP_KEYS)
    ratios=read_table(src/'all_paired_ratios.csv')
    keep=(ratios.reference==REFERENCE)&ratios.metric.isin(['delta_A_norm','delta_E_norm'])
    assert keep.sum()==80
    write_table(ratios.loc[keep],dst/'all_paired_ratios.csv')
    for p in sorted(args.dest.rglob('*')):
        if p.is_file():print(p.relative_to(args.dest))

if __name__=='__main__':main()
