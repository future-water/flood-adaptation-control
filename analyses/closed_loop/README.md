# Closed-loop test on the routing model

Figure S6 and the closed-loop numbers of Sections 2.8 and 3.1: the surrogate's feedback law closed on the full-order RAPID/Muskingum routing model (56 members, 10 Gm3/yr).

**Data.** `outputs_closed_loop.zip` (unpacks to `FAC_DATA_ROOT/outputs_closed_loop/`). The light path reads `members_fidelity.csv`, `summary_fidelity.json`, `figure_data.npz`, `inputs_map.npz` and `residual_exceedance.npz`.

**Run** (seconds; outputs in `FAC_OUT_ROOT/figS6_closed_loop/` and `FAC_OUT_ROOT/closed_loop/`):
```bash
python figures/figS06_closed_loop/make_figure.py                     # Figure S6 and its statistics
python analyses/closed_loop/summarize_fidelity.py --summary-only     # summary_fidelity.json
```

**Rerun from raw inputs** (hours). Needs the ORNL HydroSource streamflow and precipitation of all four SSPs under `FAC_RAW_ROOT`, `outputs_common.zip`, the published `outputs_flood_rl2/` under `FAC_DATA_ROOT`, and `bash analyses/common/build_kernels.sh`.
In `analyses/closed_loop/`, run `python run_allssp.py --scenario S --hydro H --gcm G` for each of the 56 members (`layout.py`), then `run_allssp_closed.py`, `run_allssp_fidelity.py`, `summarize_fidelity.py`, `prepare_data.py`, `prepare_residual.py` and `prepare_inputs_map.py` (all write to `FAC_WORK_ROOT/closed_loop/`).
Then `python figures/figS06_closed_loop/make_figure.py --data $FAC_WORK_ROOT/closed_loop/data` (from the repository root).
