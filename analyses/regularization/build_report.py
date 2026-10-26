#!/usr/bin/env python3
"""Paired coefficient-sensitivity ratios from the run_pilot.py outputs (rerun path of Figure S2 (c)).

Reads FAC_WORK_ROOT/regularization/pilot/ (default cutoff run) and pilot/numerical_cutoff_1e-08/
(run_pilot.py --rcond 1e-8), checks the ridge algebra against scikit-learn on a small problem and
the refits against the saved operators, and writes pilot/all_paired_ratios.csv and pilot/verification.json.

  python build_report.py
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from paths import WORK

OUT=WORK/'regularization'/'pilot'

def verify_algebra():
    rng=np.random.default_rng(456)
    z=rng.normal(size=(2,10));v=rng.normal(size=(9,9));x=np.vstack([z[:,:-1],v]);y=z[:,1:]
    l,s,vt=np.linalg.svd(x,full_matrices=False);alpha=.01
    noise=.01*rng.normal(size=y.shape)
    filt=s/(s*s+alpha)
    g=((y@vt.T)*filt)@l.T;dg=((noise@vt.T)*filt)@l.T
    sklearn_g=Ridge(alpha=alpha,fit_intercept=False,solver='cholesky').fit(x.T,y.T).coef_
    sklearn_gp=Ridge(alpha=alpha,fit_intercept=False,solver='cholesky').fit(x.T,(y+noise).T).coef_
    assert np.allclose(g,sklearn_g,rtol=1e-9,atol=1e-11)
    assert np.allclose(dg,sklearn_gp-sklearn_g,rtol=1e-8,atol=1e-11)
    forced_formula=((noise@vt.T)*(s*filt))@vt-dg[:,:2]@z[:,:-1]
    assert np.allclose(forced_formula,dg[:,2:]@v,atol=1e-11)
    return {'direct_sklearn_ridge_parity':True,'target_perturbation_parity':True,'forcing_perturbation_identity':True}

def main():
    fits={};noise={};checks=verify_algebra();ratios=[]
    for h in ['VIC5','PRMS']:
        fits[h]=pd.read_csv(OUT/f'{h}_fits.csv')
        noise[h]=pd.read_csv(OUT/f'{h}_noise.csv')
        fn=pd.read_csv(OUT/'numerical_cutoff_1e-08'/f'{h}_fits.csv')
        nn=pd.read_csv(OUT/'numerical_cutoff_1e-08'/f'{h}_noise.csv')
        original=float(fits[h].original_alpha.iloc[0])
        fitted=fits[h].query('alpha==@original').iloc[0]
        nr=noise[h].query('alpha==@original').set_index('seed')
        assert len(nr)==20
        for label,nref in [('machine_cutoff',noise[h].query('alpha==0')),('rcond_1e-8',nn.query('alpha==0'))]:
            nref=nref.set_index('seed')
            for metric in ['delta_A_norm','delta_E_norm','delta_G_norm']:
                ratio=nr[metric]/nref[metric]
                for seed,value in ratio.items():ratios.append(dict(hydro=h,reference=label,metric=metric,seed=seed,ratio=value))
        chk=json.loads((OUT/f'{h}_verification.json').read_text())
        assert chk['svd_column_reconstruction_relative_l2']<1e-10
        assert chk['ridge_vs_saved']['saved_operator_action_relative_difference']<1e-4
        assert chk['ridge_vs_saved']['saved_A_relative_difference']<1e-4
        assert chk['ridge_vs_saved']['saved_E_relative_difference']<1e-4
        assert np.allclose(fitted[['rho','df_per_response','fit_relative_l2']].astype(float),fn.query('alpha==@original').iloc[0][['rho','df_per_response','fit_relative_l2']].astype(float),rtol=1e-8)
    pd.DataFrame(ratios).to_csv(OUT/'all_paired_ratios.csv',index=False)
    (OUT/'verification.json').write_text(json.dumps(dict(**checks,original_operator_action_parity_both_cases=True,cutoff_check_preserves_ridge_solution=True),indent=2)+'\n')
    print('wrote',OUT/'all_paired_ratios.csv',len(ratios),'rows')

if __name__=='__main__':main()
