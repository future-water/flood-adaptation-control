# Figure S6: closed-loop test on the routing model

`make_figure.py` draws Figure S6 (the control law closed on the full-order routing model, 56 members, 10 Gm3/yr) and writes the statistics quoted in its caption and in Section 3.1.

**Data.** `outputs_closed_loop.zip` (unpacks to `FAC_DATA_ROOT/outputs_closed_loop/`): `members_fidelity.csv`, `summary_fidelity.json`, `figure_data.npz`, `inputs_map.npz`, `residual_exceedance.npz`.

**Run** (seconds; writes `figS6_control_fidelity.{pdf,png,svg}`, `figS6_control_fidelity_statistics.json` and `figS6_control_fidelity_timeseries_scores.csv` to `FAC_OUT_ROOT/figS6_closed_loop/`):
```bash
python figures/figS06_closed_loop/make_figure.py
```

**Rerun from raw inputs.** See `analyses/closed_loop/README.md`; `--data $FAC_WORK_ROOT/closed_loop/data` reads the rerun.
