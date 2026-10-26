"""Folders of the closed-loop test (Figure S6, Section 3.1), shared by the scripts in this folder.

PUBLISHED is the data of the outputs_closed_loop archive, data("closed_loop"). WORK is where a rerun writes,
FAC_WORK_ROOT/closed_loop. A member (case) is named <hydro>_<gcm>_ssp<scenario>_budget10.

  WORK/controllers/<case>/   controller.npz, recovery.json, recovered_daily.parquet    recover_member.py (run_allssp.py)
  WORK/recover/<case>/       work folder of recover_member.py; run_allssp.py deletes its q_ornl.npy
  WORK/q_ornl/<case>/        q_ornl.npy, the uncontrolled discharge, regenerated and deleted per member  make_q_ornl.py
  WORK/results_grid/         closed loops on the routing model over the penalty-weight grid      run_allssp_closed.py
  WORK/results_fidelity/     closed loops at R*, members_fidelity.csv, summary_fidelity.json, figure_data.npz
                             run_allssp_fidelity.py, summarize_fidelity.py
  WORK/data/                 the figure inputs, the same files as PUBLISHED                      prepare_*.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
import paths  # noqa: E402

GCMS = ["ACCESS-CM2", "BCC-CSM2-MR", "CNRM-ESM2-1", "EC-Earth3", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-MM"]
SSPS = (126, 245, 370, 585)
HYDROS = ("VIC5", "PRMS")

PUBLISHED = paths.data("closed_loop")
RAPID_NETWORK = PUBLISHED / "rapid_network.npz"            # comid, downstream_comid, K_seconds, x
STREAM_ORDER = PUBLISHED / "stream_order_mapping.npz"      # COMID, StreamOrde (NHDPlusV2)
GEOMETRY = paths.data("common") / "network_geometry_5070.npz"   # reach polylines for the map of panel (e)

WORK = paths.WORK / "closed_loop"
CONTROLLERS = WORK / "controllers"
RECOVER = WORK / "recover"
Q_ORNL = WORK / "q_ornl"
GRID = WORK / "results_grid"
FIDELITY = WORK / "results_fidelity"
PREPARED = WORK / "data"


def cases(ssps=SSPS):
    """Members in the order of members_fidelity.csv: by SSP, then GCM, VIC5 before PRMS."""
    return [f"{h}_{g}_ssp{s}_budget10" for s in ssps for g in GCMS for h in HYDROS]


def controller(case):
    return CONTROLLERS / case / "controller.npz"


def q_ornl(case):
    return Q_ORNL / case / "q_ornl.npy"


def saved_daily(hydro, gcm):
    """Daily basin output of the published production runs (Zenodo archive outputs_flood_rl2/)."""
    return paths.DATA / "outputs_flood_rl2" / f"daily_{hydro}_{gcm}.parquet"


def relative(path):
    """Path relative to WORK, or to the data root, as written in provenance records."""
    path = Path(path).resolve()
    for root in (WORK.resolve(), paths.DATA):
        if path.is_relative_to(root):
            return str(path.relative_to(root))
    return path.name
