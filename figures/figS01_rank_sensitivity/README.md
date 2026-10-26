# Figure S1: reduced dimension

`make_notebook.py` writes and executes `figS1_rank_sensitivity.ipynb`, which draws Figure S1 (sensitivity to the reduced dimension r, 14 SSP2-4.5 members) from the archived tables; it does not rerun POD, DMDc or LQR.

**Data.** `outputs_rank_sensitivity.zip` (unpacks to `FAC_DATA_ROOT/outputs_rank_sensitivity/`) and `network_geometry_5070.npz` from `outputs_common.zip`.

**Run** (needs nbformat, nbclient and ipykernel from `environment.yml`; writes the executed notebook, `figS1_rank_sensitivity.{pdf,png,svg}`, `rank_summary.csv`, `rank_spatial_overlap.csv` and `rank_mapped_priority_sets.csv` to `FAC_OUT_ROOT/figS1_rank_sensitivity/`):
```bash
python figures/figS01_rank_sensitivity/make_notebook.py
```

**Rerun from raw inputs.** See `analyses/rank_sensitivity/README.md`; `--data DIR` reads the rerun tables.
