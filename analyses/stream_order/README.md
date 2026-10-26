# Control by stream order

Figure S8 (b, c): control restricted to reaches of given Strahler orders (VIC5/MPI-ESM1-2-HR, SSP2-4.5), effort and residual flood volume over the R_weight grid.

**Data.** Published `stream_order_summary_VIC5_MPI-ESM1-2-HR_ssp245.csv` and `stream_order_pareto_VIC5_MPI-ESM1-2-HR_ssp245.csv` under `FAC_DATA_ROOT`, read by `figures/supplementary_figures.ipynb`. The script reads stream orders from `comid_huc_mapping_correct.csv`.

**Run.** Nothing here on the light path.

**Rerun from raw inputs** (from the folder `04_dmd_lqr_analysis.py` runs in, which holds `streamflow/`, `precipitation/` and `cache_pooled_return_levels/`):
```bash
python analyses/stream_order/dmd_lqr_stream_order.py --gcm MPI-ESM1-2-HR --hydro VIC5 --scenario 245 --low-thresholds 1 2 3 4 5 6 7
```
Writes `stream_order_<member>.csv`, `stream_order_summary_<member>.csv` and `stream_order_pareto_<member>.csv` to `FAC_DATA_ROOT` (or `--outdir DIR`).
