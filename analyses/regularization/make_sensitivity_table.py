#!/usr/bin/env python3
"""Coefficient-sensitivity table behind Figure S2 (c); no fitting, selection or simulation.

Builds the table that panel (c) plots. For each case, operator M (A or E) and seed it computes the index
    ridge_index = 100 * ||dM_ridge(seed)||_F / ||dM_ref(seed)||_F,
where dM_ridge is the change of the selected-alpha ridge estimate and dM_ref the change of the
unregularized minimum-norm least-squares solution computed with relative singular-value cutoff
rcond = 1e-8, both under the same 1% perturbation of the regression targets. The selected alpha is
read from pilot/<hydro>_setup.json, and the two setup records must describe the same regression problem
(case, n, p, sha256 of the regressor and target matrices). Every ratio is checked against
pilot/all_paired_ratios.csv.

  python make_sensitivity_table.py                        # data("regularization") -> out("regularization")
  python make_sensitivity_table.py --data DIR --out DIR   # e.g. DIR = $FAC_WORK_ROOT/regularization after a rerun
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
import argparse, json
import numpy as np
import pandas as pd
from paths import data, out

HYDROS = ['VIC5', 'PRMS']
METRICS = {'delta_A_norm': 'State operator A', 'delta_E_norm': 'Forcing operator E'}
REFERENCE = 'rcond_1e-8'

def get_data(src):
    original = pd.read_csv(src/'all_paired_ratios.csv')
    assert not original.duplicated(['hydro','reference','metric','seed']).any()
    rows=[]
    for hydro in HYDROS:
        setup=json.loads((src/f'{hydro}_setup.json').read_text())
        reference=json.loads((src/'numerical_cutoff_1e-08'/f'{hydro}_setup.json').read_text())
        for key in ['case','n','p','alpha_original','x_sha256','y_sha256']:
            assert setup[key]==reference[key],key
        alpha=float(setup['alpha_original'])
        assert alpha==.01
        raw=pd.read_csv(src/f'{hydro}_noise.csv')
        ridge=raw.loc[raw.alpha==alpha].set_index('seed').sort_index()
        baseline=pd.read_csv(src/'numerical_cutoff_1e-08'/f'{hydro}_noise.csv').query('alpha==0').set_index('seed').sort_index()
        assert ridge.index.equals(baseline.index)
        assert ridge.index.tolist()==list(range(1201,1221))
        assert (ridge.noise_relative_centered_target==.01).all() and (baseline.noise_relative_centered_target==.01).all()
        for metric in METRICS:
            assert (baseline[metric]>0).all()
            ratio=ridge[metric]/baseline[metric]
            check=original.query('hydro==@hydro and reference==@REFERENCE and metric==@metric').set_index('seed').sort_index()
            assert check.index.equals(ratio.index)
            assert np.allclose(ratio,check.ratio,rtol=1e-12,atol=1e-14)
            for seed,val in ratio.items():
                rows.append(dict(hydro=hydro,reference=REFERENCE,metric=metric,seed=seed,
                    no_ridge_index=100.,ridge_index=100*float(val)))
    data=pd.DataFrame(rows)
    assert len(data)==80 and np.isfinite(data.ridge_index).all()
    return data

def main():
    parser=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data',type=Path,default=data('regularization'),help='analysis folder that contains pilot/')
    parser.add_argument('--out',type=Path,default=None,help='output folder (default: out("regularization"))')
    args=parser.parse_args()
    dest=args.out if args.out is not None else out('regularization')
    dest.mkdir(parents=True,exist_ok=True)
    table=get_data(args.data/'pilot')
    table.to_csv(dest/'regularization_sensitivity.csv',index=False)
    print(table.groupby(['hydro','metric']).ridge_index.agg(['median','min','max']).to_string())
    print('wrote',dest/'regularization_sensitivity.csv')

if __name__=='__main__':main()
