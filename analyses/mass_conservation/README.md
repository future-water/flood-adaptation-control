# Mass conservation and control magnitude

Figure S5 (mass fidelity of the surrogate) and Figure S11 (c) (control magnitude relative to channel flow): mass-conservation and feasibility diagnostics, one row per member.

**Data.** Published `mass_conservation_summary.csv` under `FAC_DATA_ROOT`, read by `figures/supplementary_figures.ipynb`.

**Run.** Nothing here on the light path.

**Rerun from raw inputs** (from the folder `04_dmd_lqr_analysis.py` runs in, which holds `streamflow/`, `precipitation/` and `cache_pooled_return_levels/`):
```bash
python analyses/mass_conservation/si_mass_conservation.py --gcm all --hydro all --ssp all
```
Merges the rows into `FAC_DATA_ROOT/mass_conservation_summary.csv` (by gcm, hydro, ssp), writes per-member diagnostic plots to `FAC_OUT_ROOT/mass_conservation/` and draws `figS5_mass_fidelity.{png,pdf}` in `FAC_OUT_ROOT/figures/`.
