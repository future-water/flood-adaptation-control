# Control at HUC outlets

Figure S10: control acting only at the outlets of HUC12, HUC10 or HUC8 units instead of at every reach (VIC5/MPI-ESM1-2-HR, SSP2-4.5).

| Script | Output (in `FAC_DATA_ROOT`) |
|--------|-----------------------------|
| `build_comid_huc_mapping.py` | `comid_huc_mapping_correct.csv`: COMID to HUC8/10/12 by spatial join with the WBD HUC12 polygons in `shape/` |
| `si_huc_comparison.py` | `outputs_huc_comparison/huc_comparison_<member>.csv` and `huc_summary_<member>.csv`: R_weight sweep per control resolution |
| `si_huc_aggregate.py` | `huc_spatial_compact.csv`: per-reach peak reduction at each resolution and HUC outlet flags |

**Data.** Published under `FAC_DATA_ROOT`: `shape/`, `comid_huc_mapping_correct.csv`, `outputs_huc_comparison/`, `huc_spatial_compact.csv`. `outputs_huc_control.zip` (unpacks to `FAC_DATA_ROOT/outputs_huc_control/`) holds the mapping of the reported run, `comid_huc_mapping_huc_run.csv`, and its `huc_summary`.

**Run.** Nothing here on the light path: `python figures/figS10_huc_flexibility/make_figure.py` draws the figure from these files.

**Rerun from raw inputs** (from the folder `04_dmd_lqr_analysis.py` runs in, which holds `streamflow/`, `precipitation/` and `cache_pooled_return_levels/`; copy `pooled_return_levels_VIC5_MPI-ESM1-2-HR_1980-2019_empirical.npz` from `outputs_common/pooled_return_levels/` into the latter to use the return levels of the reported run):
```bash
M=${FAC_DATA_ROOT:-.}/outputs_huc_control/comid_huc_mapping_huc_run.csv   # mapping of the reported run
python analyses/huc_control/si_huc_comparison.py --gcm MPI-ESM1-2-HR --hydro VIC5 --scenario 245 --mapping $M
python analyses/huc_control/si_huc_aggregate.py --mapping $M
```
Outputs replace the published files in `FAC_DATA_ROOT` (`--outdir` and `--out` write elsewhere); copy the rerun's `huc_summary_<member>.csv` to `outputs_huc_control/` before drawing the figure.
