# Tables S1 and S2

Builds Tables S1 and S2 and the Figure S6 caption and Section 3.1 numbers, and checks them against the paper.

**Data.** Archives `outputs_rank_sensitivity`, `outputs_basis_period`, `outputs_common`, `outputs_flood_threshold` and `outputs_closed_loop`, extracted under `FAC_DATA_ROOT`.

**Run** (writes .tex/.csv to `FAC_OUT_ROOT/si_tables/`; stops on a mismatch):
```bash
python analyses/si_tables/build_si_tables.py
```

**Rerun from raw inputs.** Pass the rerun folders with `--rank`, `--basis`, `--threshold`, `--closed-loop` and `--figs6-statistics`; `--no-check` skips the comparison with the paper.
