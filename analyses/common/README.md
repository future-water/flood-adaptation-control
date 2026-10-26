# Shared pipeline loader and kernels

Code shared by the analyses that rerun the pipeline: the loader of `04_dmd_lqr_analysis.py` and the compiled closed-loop kernel.

| File | Purpose |
|------|---------|
| `pipeline_loader.py` | loads the functions of `../../04_dmd_lqr_analysis.py` (sha256 checked) without running it: `production()`, `raw_inputs()`, `return_levels_path()` |
| `control_kernel.cpp`, `kernel.py`, `native.py` | compiled closed-loop kernel and its ctypes loaders |
| `build_kernels.sh`, `compat/` | compiles every `.cpp` under `analyses/` into `build/` (`compat/`: Linux CBLAS shim) |

**Data.** `outputs_common.zip` (unpacks to `FAC_DATA_ROOT/outputs_common/`): `pooled_return_levels/`, `reach_attributes.csv`, `network_geometry_5070.npz`. Reruns also read the ORNL `streamflow/` and `precipitation/` parquet files in `FAC_RAW_ROOT`.

**Run.** Nothing on the light path. Reruns compile the kernels with `bash analyses/common/build_kernels.sh` (clang++ and Accelerate on macOS; g++ and a CBLAS on Linux).
