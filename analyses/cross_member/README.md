# Agreement across members

Section 3.3: top-decile agreement of the 10 Gm3/yr allocations across the 14 SSP2-4.5 members and with their ensemble median.

**Data.** `outputs_basis_period.zip` (unpacks to `FAC_DATA_ROOT/outputs_basis_period/`): `effort_production_full` in `results/<member>.npz`.

**Run** (seconds; writes `FAC_OUT_ROOT/cross_member/summary_cross_member.json`):
```bash
python analyses/cross_member/summarize_cross_member.py
```

**Rerun from raw inputs.** Run `analyses/basis_period/run_all.py` (see `../basis_period/README.md`), then `python analyses/cross_member/summarize_cross_member.py --data $FAC_WORK_ROOT/basis_period`.
