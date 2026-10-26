#!/usr/bin/env python3
"""Complete any target-budget searches requiring a weaker-feedback bracket."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
import run_sensitivity as r

threadpool_limits(limits=2)
ns=r.source_functions()
for path in sorted((r.OUT/'primary').glob('*/*/summary.json')):
    row=json.loads(path.read_text())
    if row['budget_matched']:continue
    key=path.parent.parent.name;setting=path.parent.name
    model=dict(np.load(r.WORK/'cache'/key/f'{setting}.npz'))
    result=r.match_budget(ns,model,model['threshold'],model['years'],path.parent)
    row.update(result)
    if result['budget_matched']:
        row['reduction_Gm3yr']=row['surrogate_excess_Gm3yr']-row['controlled_residual_Gm3yr']
        row['reduction_percent']=100*row['reduction_Gm3yr']/row['surrogate_excess_Gm3yr']
    row['budget_search_refined']=True
    path.write_text(json.dumps(row,indent=2))
    print(key,setting,result,'reduction_percent',row.get('reduction_percent'),flush=True)
