# Attenuation direction of the lifted inputs

Figure S11 (a, b) and the Section 4.2 attenuation fractions: share |δ⁻|/|δ| of the closed-loop change in surrogate discharge that attenuates flow, δ = max(x_cl, 0) − max(x_ol, 0), per reach during exceedance and basin-wide, at 10 Gm3/yr for all 56 members.

**Data.** Published `outputs_flood_rl2/delta_atten_per_reach.parquet` under `FAC_DATA_ROOT`, read by `figures/supplementary_figures.ipynb`. Panel (c) comes from `../mass_conservation/`.

**Run.** Nothing here on the light path; run the Figure S11 cell of the notebook.

**Rerun from raw inputs** (from the folder `04_dmd_lqr_analysis.py` runs in, which holds `streamflow/`, `precipitation/` and `cache_pooled_return_levels/`):
```bash
python analyses/attenuation_direction/compute_delta_atten_fraction.py --rl 2
```
Writes `delta_atten_per_reach.parquet` and `delta_atten_summary.csv` to `FAC_DATA_ROOT/outputs_flood_rl2/` (or `--output DIR`).
