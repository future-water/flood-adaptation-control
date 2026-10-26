# Basis period

Table S1, bottom block: basis, mean and forcing statistics from 2020-2029 only, compared with the main analysis (14 SSP2-4.5 members).

**Data.** `outputs_basis_period.zip` (unpacks to `FAC_DATA_ROOT/outputs_basis_period/`): `results/<member>.npz`, `results_basis/<member>_b10.{npz,json}`. Also `reach_attributes.csv` from `outputs_common.zip`.

**Run** (seconds; outputs in `FAC_OUT_ROOT/basis_period/`):
```bash
python analyses/basis_period/summarize_basis.py --basis-years 10
python analyses/basis_period/table_s1_bottom.py
```

**Rerun from raw inputs.** Needs the ORNL HydroSource SSP2-4.5 streamflow and precipitation of the 14 members under `FAC_RAW_ROOT`, `outputs_common.zip`, and `bash analyses/common/build_kernels.sh`.
`python analyses/basis_period/run_all.py` and `python analyses/basis_period/run_basis_all.py --basis-years 10` write to `FAC_WORK_ROOT/basis_period/`.
Then run the two commands above with `--data $FAC_WORK_ROOT/basis_period`.
