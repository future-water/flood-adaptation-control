# Figure S10: HUC-outlet control

`make_figure.py` draws Figure S10: peak reduction lost when control acts only at HUC12, HUC10 or HUC8 outlets (VIC5/MPI-ESM1-2-HR, SSP2-4.5, 10 Gm3/yr). `figures/supplementary_figures.ipynb` calls it.

**Data.** Under `FAC_DATA_ROOT`: `shape/` (`shape.zip`) and `huc_spatial_compact.csv`; `outputs_huc_control.zip` (unpacks to `FAC_DATA_ROOT/outputs_huc_control/`): `huc_summary_VIC5_MPI-ESM1-2-HR_ssp245.csv`, `comid_huc_mapping_huc_run.csv`.

**Run** (about 10 s; writes `figS10_huc_flexibility.{pdf,png}` and `figS10_values.json` to `FAC_OUT_ROOT/figS10_huc_flexibility/`):
```bash
python figures/figS10_huc_flexibility/make_figure.py
```

**Rerun from raw inputs.** See `analyses/huc_control/README.md`.
