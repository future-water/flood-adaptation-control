#!/usr/bin/env python3
"""Write the archived Figure S3 tables from the aggregated runs; do not refit models.

Reads member_results.csv and paired_changes_vs_10_5.csv written by analyze_sensitivity.py in
<FAC_WORK_ROOT>/window_sensitivity/ and writes, to <FAC_OUT_ROOT>/window_sensitivity/, the three files of the
outputs_window_sensitivity archive: supplementary_all_original_results.csv, supplementary_all_original_paired_results.csv
and final_configuration.json ('original' = the ridge grid of the main analysis).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import WORK, out
import numpy as np
import pandas as pd

SRC = WORK / 'window_sensitivity'
OUT = out('window_sensitivity')
LONG = ['10_5', '15_5', '20_5', '20_10']        # windows of 10 to 20 years
SHORT = ['5_2p5', '5_5']
ALL = LONG + SHORT
ORDER = ['5_2p5', '5_5', '10_5', '15_5', '20_5', '20_10']
KEYS = ['hydro', 'gcm', 'ssp', 'setting']
# Columns of the archived tables: the values drawn in Figure S3 and the effort matching behind its caption.
RESULT_COLUMNS = KEYS + ['window_years', 'step_years', 'nse_median', 'budget_target', 'budget_matched',
                         'actual_budget', 'reduction_percent']
PAIRED_COLUMNS = KEYS + ['paired_budget_matched', 'delta_reduction_pp', 'top10_overlap']

d = pd.read_csv(SRC / 'member_results.csv')
p = pd.read_csv(SRC / 'paired_changes_vs_10_5.csv')
assert len(d) == len(p) == 84
assert not d.duplicated(KEYS).any() and not p.duplicated(KEYS).any()
assert set(d.ssp) == {245} and set(d.setting) == set(ALL)
members = set(map(tuple, d[['hydro', 'gcm', 'ssp']].drop_duplicates().to_numpy()))
assert len(members) == 14
for setting in ALL:
    assert set(map(tuple, d.loc[d.setting.eq(setting), ['hydro', 'gcm', 'ssp']].to_numpy())) == members
assert d.budget_matched.all() and p.paired_budget_matched.all()
assert np.max(np.abs(d.actual_budget / 10 - 1)) <= 0.005
np.testing.assert_allclose(d.reduction_percent,
    100 * (1 - d.controlled_residual_Gm3yr / d.surrogate_excess_Gm3yr), atol=1e-10)
assert int(d.stability_target_failures.sum()) == 0          # every window operator has spectral radius < 0.995
assert (d[d.setting.isin(LONG)].reduction_percent > 0).all()
base = d[d.setting.eq('10_5')].set_index(KEYS[:-1])
for setting in ALL:
    rows = p[p.setting.eq(setting)].set_index(KEYS[:-1]).sort_index()
    selected = d[d.setting.eq(setting)].set_index(KEYS[:-1]).sort_index()
    np.testing.assert_allclose(rows.delta_reduction_pp, selected.reduction_percent - base.reduction_percent.reindex(rows.index), atol=1e-10)

d[RESULT_COLUMNS].to_csv(OUT / 'supplementary_all_original_results.csv', index=False, lineterminator='\n')
p[PAIRED_COLUMNS].to_csv(OUT / 'supplementary_all_original_paired_results.csv', index=False, lineterminator='\n')

final_config = dict(reference_window_years=10, reference_step_years=5,
    ridge_candidates=[0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50], spectral_radius_cutoff=0.995,
    ridge_selection='first candidate passing radius < 0.995; retain 50 and flag if none passes',
    settings=ORDER, primary_scenario='SSP2-4.5', n_members=14,
    period='2020–2099', target_budget_Gm3yr=10, budget_relative_tolerance=0.005,
    scope='same-scenario reconstruction and surrogate control')
(OUT / 'final_configuration.json').write_text(json.dumps(final_config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('wrote', *sorted(x.name for x in OUT.iterdir()), 'in', OUT)
