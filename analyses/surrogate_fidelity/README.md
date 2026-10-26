# Surrogate fidelity at high flows

Figure S4 (b, c): return-level bias of the open-loop surrogate for the basin-aggregate hydrograph (2- to 100-year) and for stream-order aggregates (10-year), one row per member. Panel (a) is read directly from the published `outputs_flood_rl2/results_*.parquet`.

**Data.** Published `surrogate_fidelity_summary.csv` under `FAC_DATA_ROOT`, read by `figures/supplementary_figures.ipynb`. The script also reads stream orders from `shape/`.

**Run.** Nothing here on the light path.

**Rerun from raw inputs** (from the folder `04_dmd_lqr_analysis.py` runs in, which holds `streamflow/`, `precipitation/` and `cache_pooled_return_levels/`):
```bash
python analyses/surrogate_fidelity/si_surrogate_fidelity.py
```
Writes `FAC_DATA_ROOT/surrogate_fidelity_summary.csv` (or `--out FILE`).
