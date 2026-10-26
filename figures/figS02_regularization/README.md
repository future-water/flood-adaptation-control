# Figure S2: ridge regularization

`make_figure.py` draws Figure S2 (diagnostics of the ridge-regularized DMDc estimator) from the archived tables; it does not rerun the identification.

**Data.** `outputs_regularization.zip` (unpacks to `FAC_DATA_ROOT/outputs_regularization/`): `all_paths.csv` (panels a, b) and `regularization_sensitivity.csv` (panel c).

**Run** (writes `figS2_regularization.{pdf,png,svg}`, `selected_alpha_diagnostics.csv` and `coefficient_sensitivity_summary.csv` to `FAC_OUT_ROOT/figS2_regularization/`):
```bash
python figures/figS02_regularization/make_figure.py
```

**Rerun from raw inputs.** See `analyses/regularization/README.md`; `--data DIR` reads the rerun tables.
