# Data-driven flood adaptation control

Replication code for **"Data-driven control reveals distributed flood adaptation priorities across large river networks under climate change."**

## Pipeline

| Step | File | Description |
|------|------|-------------|
| 1 | `01_download_data.ipynb` | Download streamflow & precipitation from HydroSource |
| 2 | `02_process_streamflow.ipynb` | Process NetCDFs → parquet |
| 3 | `03_process_precipitation.ipynb` | Process precipitation → parquet |
| 4 | `04_dmd_lqr_analysis.py` | POD + DMDc + LQR flood control (7 GCMs × 2 hydro × 4 SSPs) |
| 5 | `05_figures_main.ipynb` | Main manuscript figures |
| 6 | `06_figures_supplementary.ipynb` | Supplementary figures (S1–S5) |

Supplementary data scripts (run before step 6):
- `si_mass_conservation.py` — mass conservation diagnostics → `mass_conservation_summary.csv`
- `si_huc_comparison.py` — HUC-level control comparison

## Usage

```bash
pip install -r requirements.txt

# Step 4
python 04_dmd_lqr_analysis.py --stage all --gcm all --hydro all --rl 2

# Supplementary data
python si_mass_conservation.py --gcm all --hydro all --ssp all
```

Then run notebooks.