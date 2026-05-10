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
| 6 | `06_figures_supplementary.ipynb` | Supplementary figures (S1–S7) |

Supplementary data scripts (run before step 6):

| Script | Output | Used by |
|--------|--------|---------|
| `build_comid_huc_mapping.py` | `comid_huc_mapping_correct.csv` | S5 |
| `si_surrogate_fidelity.py` | `surrogate_fidelity_summary.csv` | S1 (b,c) |
| `si_ranking_stability.py` | `ranking_stability.csv` | S4 (c) |
| `si_mass_conservation.py` | `mass_conservation_summary.csv` | S2, S6 (c) |
| `si_huc_comparison.py` | `outputs_huc_comparison/huc_comparison_*.csv` | S5 |
| `si_huc_aggregate.py` | `huc_spatial_compact.csv` | S5 |
| `compute_delta_atten_fraction.py` | `outputs_flood_rl2/delta_atten_per_reach.parquet` | S6 (a,b) |
| `dmd_lqr_stream_order.py` | `stream_order_{pareto,summary}_*.csv` | S7 (b,c) |

## Usage

```bash
pip install -r requirements.txt

# Step 4
python 04_dmd_lqr_analysis.py --stage all --gcm all --hydro all --rl 2

# SI data (run as needed for each SI panel)
python build_comid_huc_mapping.py
python si_surrogate_fidelity.py
python si_ranking_stability.py
python si_mass_conservation.py --gcm all --hydro all --ssp all
python si_huc_comparison.py --gcm MPI-ESM1-2-HR --hydro VIC5 --scenario 245
python si_huc_aggregate.py --input outputs_huc_comparison/huc_comparison_VIC5_MPI-ESM1-2-HR_ssp245.csv
python compute_delta_atten_fraction.py --rl 2
python dmd_lqr_stream_order.py --gcm MPI-ESM1-2-HR --hydro VIC5 --scenario 245 --low-thresholds 1 2 3 4 5 6 7
```

Then run notebooks 05 and 06.

## Data

This repo contains code only. The intermediate data archive
(`outputs_flood_rl2/`, `shape/`, derived CSVs, ~6 GB total) is published as a
separate Zenodo dataset record. Extract it next to the notebooks so that all
relative paths resolve.
