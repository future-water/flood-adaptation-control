# Flood threshold

Table S2: allocation with the 5-year instead of the 2-year return level as flood threshold (14 SSP2-4.5 members).

**Data.** `outputs_flood_threshold.zip` (unpacks to `FAC_DATA_ROOT/outputs_flood_threshold/`): `effort/effort_rl5_r30_<hydro>_<gcm>_ssp245.npz`, `rl2_production_effort_ssp245.npz`, `stream_order.csv`.

**Run** (about 1 s; writes `summary_rl2_rl5_budget{5,10}.json` and `member_rl2_rl5_budget{5,10}.csv` to `FAC_OUT_ROOT/flood_threshold/` and prints the Table S2 values):
```bash
python analyses/flood_threshold/compare_rl2_rl5.py
```

**Rerun from raw inputs.** 2-year side: with the published `outputs_flood_rl2/` and `shape.zip` extracted under `FAC_DATA_ROOT`, `python analyses/flood_threshold/extract_rl2_effort.py` writes the RL2 file and `stream_order.csv` to `FAC_OUT_ROOT/flood_threshold/`.
5-year side: with the ORNL HydroSource SSP2-4.5 streamflow and precipitation under `FAC_RAW_ROOT` and `outputs_common.zip`, run `python analyses/flood_threshold/rl_effort_r30.py --hydro H --gcm G --rl 5` for each member (writes `FAC_WORK_ROOT/flood_threshold/effort/`).
Then `python analyses/flood_threshold/compare_rl2_rl5.py --rl5-dir $FAC_WORK_ROOT/flood_threshold/effort --rl2 $FAC_OUT_ROOT/flood_threshold/rl2_production_effort_ssp245.npz --stream-order $FAC_OUT_ROOT/flood_threshold/stream_order.csv`.
