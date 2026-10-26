# Ridge regularization

Figure S2: diagnostics of the ridge-regularized DMDc estimator (VIC5 and PRMS forced by MPI-ESM1-2-HR, SSP2-4.5).

**Data.** `outputs_regularization.zip` (unpacks to `FAC_DATA_ROOT/outputs_regularization/`): `all_paths.csv` (panels a, b), `regularization_sensitivity.csv` (panel c) and `pilot/` (input of the optional table rebuild).

**Run** (Figure S2 goes to `FAC_OUT_ROOT/figS2_regularization/`):
```bash
python analyses/regularization/make_sensitivity_table.py   # optional: rebuilds regularization_sensitivity.csv from pilot/
python figures/figS02_regularization/make_figure.py
```

**Rerun from raw inputs** (minutes). Needs the MPI-ESM1-2-HR precipitation parquet under `FAC_RAW_ROOT` and, per case `<hydro>_MPI-ESM1-2-HR_ssp245_budget10`, `controller.npz` and `q_ornl.npy` in `$FAC_WORK_ROOT/regularization/cases/<case>/` (from `analyses/closed_loop/recover_member.py`).
Run the two `run_pilot.py` commands in its docstring, then `build_report.py` and `run_path.py`.
Then run `make_sensitivity_table.py --data $FAC_WORK_ROOT/regularization --out $FAC_WORK_ROOT/regularization` and `make_figure.py --data $FAC_WORK_ROOT/regularization`.
