"""Locations used by the scripts under analyses/ and figures/.

Every script resolves its files through this module, so nothing depends on the machine the analyses were first run
on. Override the defaults with environment variables:

  FAC_DATA_ROOT  folder where the Zenodo archives are extracted (default: repository root). The published files
                 (outputs_flood_rl2/, shape/, the loose CSVs) sit at its top level; each analysis archive
                 outputs_<name>.zip unpacks to <DATA>/outputs_<name>/, returned by data(name).
  FAC_RAW_ROOT   ORNL HydroSource streamflow and precipitation and the pooled return-level cache; needed only to
                 rerun analyses from raw inputs.
  FAC_WORK_ROOT  heavy intermediates written by reruns (controllers, routed discharge).
  FAC_OUT_ROOT   figures and tables written by the scripts.

Scripts two levels below the repository root (analyses/<name>/x.py, figures/<figure>/x.py) import it with
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root
    from paths import data, DATA, RAW, WORK, out, work
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FAC_DATA_ROOT", REPO)).expanduser().resolve()
RAW = Path(os.environ.get("FAC_RAW_ROOT", REPO / "raw")).expanduser().resolve()
WORK = Path(os.environ.get("FAC_WORK_ROOT", REPO / "work")).expanduser().resolve()
OUT = Path(os.environ.get("FAC_OUT_ROOT", REPO / "outputs")).expanduser().resolve()


def data(name):
    """Folder of one analysis archive (outputs_<name>.zip) inside the extracted Zenodo record."""
    return DATA / f"outputs_{name}"


def out(name):
    """Output folder for one analysis or figure, created on first use."""
    p = OUT / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def work(name):
    """Work folder for heavy intermediates of one analysis, created on first use."""
    p = WORK / name
    p.mkdir(parents=True, exist_ok=True)
    return p
