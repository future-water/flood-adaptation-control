# Data-driven control reveals distributed flood adaptation priorities across large river networks

[![concept DOI](https://img.shields.io/badge/concept%20DOI-10.5281%2Fzenodo.20103438-blue)](https://doi.org/10.5281/zenodo.20103438)

Code for the paper of the same title. `04_dmd_lqr_analysis.py` identifies a reduced-order surrogate of the Brazos
River network (POD and windowed DMDc) from CMIP6-forced streamflow and precipitation and applies threshold-triggered
LQR control to it for 56 members (7 GCMs × 2 hydrologic models × 4 SSPs). `analyses/<name>/` holds the analyses of
the Supporting Information, each with a short README, and `figures/` draws every figure; `paths.py` sets the data and
output folders. The figures, tables and numbers below are drawn in minutes from the derived data on Zenodo; the raw
forcings are needed only to rerun the computations.

## Setup

```bash
conda env create -f environment.yml && conda activate flood-adaptation-control   # or: pip install -r requirements.txt
export FAC_DATA_ROOT=/path/to/data      # optional; default: repository root
export FAC_OUT_ROOT=/path/to/outputs    # optional; default: outputs/
```

Download every file of the Zenodo record into `FAC_DATA_ROOT` and extract all zips there, keeping the CSV files at
its top level; each `outputs_<name>.zip` unpacks to `outputs_<name>/`. A separate data folder keeps the record's
`README.md` apart from this one. The compiled kernels (`bash analyses/common/build_kernels.sh`; C++ compiler and
BLAS, see `environment.yml`) are needed only for reruns.

## Figures, tables and numbers

Run the scripts from the repository root (`python <script>`) and the notebooks top to bottom. The notebooks write to
`FAC_OUT_ROOT/figures/`, the scripts to `FAC_OUT_ROOT/<analysis or figure>/`. Both notebooks also read `shape.zip` and
`outputs_flood_rl2.zip` in their first cells, also for Figures 2, S5 and S8; the data column names files of the Zenodo
record.

| Item | Script | Data |
|------|--------|------|
| Figure 1 | schematic, not drawn by code | – |
| Figure 2 | `figures/main_figures.ipynb` | `outputs_surrogate_comparison.zip` |
| Figures 3–8 | `figures/main_figures.ipynb` | `outputs_flood_rl2.zip` |
| Figure S1 | `figures/figS01_rank_sensitivity/make_notebook.py` | `outputs_rank_sensitivity.zip`, `outputs_common.zip` |
| Figure S2 | `figures/figS02_regularization/make_figure.py` | `outputs_regularization.zip` |
| Figure S3 | `figures/figS03_window_sensitivity/make_notebook.py` | `outputs_window_sensitivity.zip` |
| Figure S4 | `figures/supplementary_figures.ipynb` | `outputs_flood_rl2.zip`, `surrogate_fidelity_summary.csv` |
| Figure S5 | `figures/supplementary_figures.ipynb` | `mass_conservation_summary.csv` |
| Figure S6 | `figures/figS06_closed_loop/make_figure.py` | `outputs_closed_loop.zip` |
| Figure S7 | `figures/supplementary_figures.ipynb` | `outputs_flood_rl2.zip` |
| Figure S8 | `figures/supplementary_figures.ipynb` | `stream_order_{summary,pareto}_VIC5_MPI-ESM1-2-HR_ssp245.csv` |
| Figure S9 | `figures/supplementary_figures.ipynb` | `outputs_flood_rl2.zip`, `ranking_stability.csv` |
| Figure S10 | `figures/figS10_huc_flexibility/make_figure.py` (also run by the notebook) | `outputs_huc_control.zip`, `huc_spatial_compact.csv`, `shape.zip` |
| Figure S11 | `figures/supplementary_figures.ipynb` | `outputs_flood_rl2.zip`, `mass_conservation_summary.csv` |
| Table 1 | metric definitions, no data | – |
| Table S1, top and middle | `analyses/rank_sensitivity/table_s1_rank.py` | `outputs_rank_sensitivity.zip` |
| Table S1, bottom | `analyses/basis_period/summarize_basis.py --basis-years 10`, then `analyses/basis_period/table_s1_bottom.py` | `outputs_basis_period.zip`, `outputs_common.zip` |
| Table S2 | `analyses/flood_threshold/compare_rl2_rl5.py` | `outputs_flood_threshold.zip` |
| Tables S1 and S2, checked against the paper | `analyses/si_tables/build_si_tables.py` | as for Tables S1 and S2 |
| Sections 2.8 and 3.1: closed-loop test | `analyses/si_tables/build_si_tables.py`, `analyses/closed_loop/summarize_fidelity.py --summary-only` | `outputs_closed_loop.zip` |
| Section 3.1: surrogate fit and high-flow bias | `figures/main_figures.ipynb` (Figure 2), `figures/supplementary_figures.ipynb` (Figure S4) | as for Figures 2 and S4 |
| Section 3.3: effort scaling | `figures/supplementary_figures.ipynb` (Figure S7) | `outputs_flood_rl2.zip` |
| Section 3.3: LQR-unique reaches | `figures/main_figures.ipynb` (Figure 5) | `outputs_flood_rl2.zip` |
| Section 3.3: agreement across members | `analyses/cross_member/summarize_cross_member.py` | `outputs_basis_period.zip` |
| Section 3.5: ensemble agreement | `figures/main_figures.ipynb` (Figure 8) | `outputs_flood_rl2.zip` |
| Section 4.2 | `figures/figS10_huc_flexibility/make_figure.py`, `figures/supplementary_figures.ipynb` (Figure S11) | as for Figures S10 and S11 |

`analyses/` also holds the scripts that produced the published CSV files and `outputs_flood_rl2/delta_atten_per_reach.parquet`
(`huc_control`, `mass_conservation`, `ranking_stability`, `stream_order`, `surrogate_fidelity`, `attenuation_direction`).

## Rerun from raw data

1. `01_download_data.ipynb`, `02_process_streamflow.ipynb` and `03_process_precipitation.ipynb` download the ORNL
   HydroSource projections (tens of GB) and write the parquet files that `04_dmd_lqr_analysis.py` reads from
   `streamflow/` and `precipitation/` in its working directory.
2. `python 04_dmd_lqr_analysis.py --stage all --gcm all --hydro all --rl 2`, in that folder, writes
   `outputs_flood_rl2/` and `cache_pooled_return_levels/`.
3. Set `FAC_RAW_ROOT` to that folder (`FAC_WORK_ROOT` holds the intermediates; see `paths.py`), run
   `bash analyses/common/build_kernels.sh`, and follow the "Rerun from raw inputs" steps of each
   `analyses/<name>/README.md`. Then run the scripts above on the rerun outputs (options in `--help`).

## Data and license

Data: Zenodo, concept DOI [10.5281/zenodo.20103438](https://doi.org/10.5281/zenodo.20103438) (resolves to the latest
version; the record's `README.md` describes every file). Raw forcings: ORNL HydroSource CMIP6-based streamflow
projections (doi:10.13139/OLCF/2318650), not redistributed. Code and data are licensed under CC BY 4.0 (`LICENSE`);
cite them with `CITATION.cff`.
