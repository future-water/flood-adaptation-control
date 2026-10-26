#!/usr/bin/env bash
# Rerun path of Figure S3: 14 SSP2-4.5 members x 6 window / update-interval settings, matched effort 10 Gm3/yr.
# Needs the ORNL HydroSource streamflow/ and precipitation/ parquet files under FAC_RAW_ROOT and the outputs_common
# archive (pooled return levels) under FAC_DATA_ROOT; see README.md. Intermediates go to FAC_WORK_ROOT/window_sensitivity,
# the archived tables to FAC_OUT_ROOT/window_sensitivity.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export VECLIB_MAXIMUM_THREADS=1
export OMP_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
bash "$HERE/../common/build_kernels.sh"
"$PYTHON_BIN" "$HERE/run_sensitivity.py" --audit-only
"$PYTHON_BIN" "$HERE/run_sensitivity.py" --cases 0,1,2,3,4,5,6,7,8,9,10,11,12,13 --workers 2 --threads 3
"$PYTHON_BIN" "$HERE/refine_unmatched.py"
"$PYTHON_BIN" "$HERE/analyze_sensitivity.py"
"$PYTHON_BIN" "$HERE/package_tables.py"
