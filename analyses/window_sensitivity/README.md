# Identification window

Figure S3: sensitivity to the length and update interval of the identification window (14 SSP2-4.5 members).

**Data.** `outputs_window_sensitivity.zip` (unpacks to `FAC_DATA_ROOT/outputs_window_sensitivity/`). The figure reads `supplementary_all_original_results.csv` and `supplementary_all_original_paired_results.csv`.

**Run** (Figure S3 goes to `FAC_OUT_ROOT/figS3_window_sensitivity/`):
```bash
python figures/figS03_window_sensitivity/make_notebook.py
```

**Rerun from raw inputs** (hours). Needs the ORNL HydroSource SSP2-4.5 streamflow and precipitation parquet files under `FAC_RAW_ROOT`, `outputs_common.zip` extracted under `FAC_DATA_ROOT`, and a C++ compiler with BLAS.
`bash analyses/window_sensitivity/reproduce.sh` (`PYTHON_BIN=...` sets the interpreter) runs `build_kernels.sh` and every step; intermediates go to `$FAC_WORK_ROOT/window_sensitivity/` and the three archive files to `$FAC_OUT_ROOT/window_sensitivity/`.
Copy those files into `FAC_DATA_ROOT/outputs_window_sensitivity/` to draw the figure from the rerun.
