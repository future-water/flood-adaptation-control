"""Load the functions of the pipeline, ../../04_dmd_lqr_analysis.py, without running it.

The source is checked against SOURCE_SHA256 (sha256 after normalizing line endings to LF) and executed once per
process as a module registered under a private name. The only line skipped is the creation of the threshold cache
folder in the current working directory, and CACHE_DIR is pointed at the pooled return-level cache found by
return_levels_dir(). The pipeline reads streamflow/ and precipitation/ relative to the working directory, so call its
loaders inside `with raw_inputs():`, which switches into FAC_RAW_ROOT (paths.RAW) and back.

Usage, from analyses/<analysis>/x.py after analyses/ is on sys.path:

    from common.pipeline_loader import production, raw_inputs, return_levels_path
    ns = production()
    with raw_inputs():
        X, U, dates, reach_ids = ns["load_scenario_data"](245, gcm, hydro)
    rl = np.load(return_levels_path(hydro, gcm))
"""
import hashlib
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))   # repository root (paths)
from paths import RAW, REPO, data  # noqa: E402

SOURCE = REPO / "04_dmd_lqr_analysis.py"
SOURCE_SHA256 = "f76c6df788b2778b4b83aacf379eb53a62bf35e26a747102cd108605d5cc54aa"   # after CRLF -> LF
NAME = "pipeline_source"
DATA_ROOT = RAW            # folder holding streamflow/ and precipitation/ (ORNL HydroSource parquet files)
# Pooled 1980-2019 return levels used by every run: the copy in the outputs_common archive first, then the cache
# folder of the pipeline under FAC_RAW_ROOT.
RETURN_LEVEL_DIRS = (data("common") / "pooled_return_levels", RAW / "cache_pooled_return_levels")


def source_sha256(path=SOURCE):
    """sha256 of a pipeline source after normalizing line endings to LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def return_levels_dir():
    """First existing folder of RETURN_LEVEL_DIRS (the last one if none exists)."""
    return next((d for d in RETURN_LEVEL_DIRS if d.is_dir()), RETURN_LEVEL_DIRS[-1])


def return_levels_path(hydro, gcm):
    """Pooled 1980-2019 return-level file of one member (keys reach_ids, sample_n, rl2, rl5, ..., rl100)."""
    name = f"pooled_return_levels_{hydro}_{gcm}_1980-2019_empirical.npz"
    for folder in RETURN_LEVEL_DIRS:
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError(f"{name} not found; extract outputs_common.zip under FAC_DATA_ROOT "
                            f"(looked in {', '.join(str(d) for d in RETURN_LEVEL_DIRS)})")


@contextmanager
def raw_inputs():
    """Switch into FAC_RAW_ROOT for the pipeline's cwd-relative loaders, then back."""
    if not (RAW / "streamflow").is_dir():
        raise FileNotFoundError(f"{RAW / 'streamflow'} not found; set FAC_RAW_ROOT to the folder holding the "
                                "ORNL HydroSource streamflow/ and precipitation/ parquet files")
    previous = os.getcwd()
    os.chdir(RAW)
    try:
        yield RAW
    finally:
        os.chdir(previous)


def load_pipeline():
    """The pipeline as a module object."""
    if NAME in sys.modules:
        return sys.modules[NAME]
    raw = SOURCE.read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"{SOURCE.name} has sha256 {digest}, expected {SOURCE_SHA256}")
    text = raw.decode("utf-8")
    assert text.count("CACHE_DIR.mkdir(exist_ok=True)") == 1
    text = text.replace("CACHE_DIR.mkdir(exist_ok=True)", "pass")
    module = types.ModuleType(NAME)
    module.__file__ = str(SOURCE)
    sys.modules[NAME] = module            # dataclasses look the module up while the source runs
    try:
        exec(compile(text, str(SOURCE), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[NAME]
        raise
    module.CACHE_DIR = return_levels_dir()
    return module


def production():
    """Namespace of the pipeline (dict): ns["compute_pod"], ns["RANK"], ..."""
    return load_pipeline().__dict__
