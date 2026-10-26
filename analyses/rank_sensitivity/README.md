# Reduced dimension

Figure S1 and the top and middle blocks of Table S1: sensitivity to the reduced dimension r (14 SSP2-4.5 members).

**Data.** `outputs_rank_sensitivity.zip` (unpacks to `FAC_DATA_ROOT/outputs_rank_sensitivity/`): the CSV files and `aggregates/*.npz`. Also `network_geometry_5070.npz` from `outputs_common.zip`.

**Run** (outputs in `FAC_OUT_ROOT/rank_sensitivity/` and `FAC_OUT_ROOT/figS1_rank_sensitivity/`):
```bash
python analyses/rank_sensitivity/table_s1_rank.py
python figures/figS01_rank_sensitivity/make_notebook.py
```

**Rerun from raw inputs** (hours). Needs the ORNL HydroSource SSP2-4.5 streamflow and precipitation parquet files under `FAC_RAW_ROOT` and `bash analyses/common/build_kernels.sh`.
Run `python analyses/rank_sensitivity/run_ensemble.py` (writes `$FAC_WORK_ROOT/rank_sensitivity/`) and `python analyses/rank_sensitivity/analyze_ensemble.py`.
Then run the two commands above with `--data "$FAC_WORK_ROOT/rank_sensitivity/aggregated"` (`--no-check` skips the table's comparison with the published values).
