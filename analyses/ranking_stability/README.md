# Ranking stability

Figure S9 (c): stability of the per-reach effort ranking between two LQR weights (the fourth and the middle value of the R_weight grid): Spearman correlation and top-10% overlap, one row per member.

**Data.** Published `outputs_flood_rl2/results_*.parquet` under `FAC_DATA_ROOT`. The result, `ranking_stability.csv`, is also published at the top of `FAC_DATA_ROOT` and read by `figures/supplementary_figures.ipynb`.

**Run** (reads only the published results):
```bash
python analyses/ranking_stability/si_ranking_stability.py
```
Writes `FAC_DATA_ROOT/ranking_stability.csv` (`--out FILE` writes elsewhere, `--results-dir DIR` reads another results folder).
