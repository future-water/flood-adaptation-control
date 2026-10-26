# Figure S3: sensitivity to the identification window

Draws Figure S3 from the archived tables.

**Data.** `outputs_window_sensitivity.zip` (unpacks to `FAC_DATA_ROOT/outputs_window_sensitivity/`): `supplementary_all_original_results.csv` and `supplementary_all_original_paired_results.csv`.

**Run** (needs nbformat, nbclient and ipykernel from `environment.yml`; writes and executes `figS3_window_sensitivity.ipynb` and saves `figS3_window_sensitivity.{pdf,png,svg}`, `figS3_setting_summary.csv` and `figS3_caption_values.json` to `FAC_OUT_ROOT/figS3_window_sensitivity/`):
```bash
python figures/figS03_window_sensitivity/make_notebook.py
```

**Rerun from raw inputs.** See `analyses/window_sensitivity/README.md`.
