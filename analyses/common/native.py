"""Locate and load the shared libraries that build_kernels.sh compiles into analyses/common/build/.

Each analyses/**/<stem>.cpp becomes build/<stem>.dylib on macOS and build/<stem>.so on Linux.

    from common.native import load_library
    lib = load_library("matched_chunk")     # ctypes.CDLL
"""
import ctypes
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parent / "build"
SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"


def library_path(stem):
    return BUILD / f"{stem}{SUFFIX}"


def load_library(stem):
    path = library_path(stem)
    if not path.exists():
        raise ImportError(f"{path.name} not found in analyses/common/build/; run analyses/common/build_kernels.sh first")
    return ctypes.CDLL(str(path))
