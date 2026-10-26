"""Collect the data for the closed-loop figure (Figure S6; read-only on all sources).

Copies the per-member results, the summary and the figure arrays of the closed-loop fidelity runs for all 56 members
(summarize_fidelity.py, WORK/results_fidelity/) into WORK/data/ and records their SHA-256 in provenance.json.
Usage: python prepare_data.py
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402

DATA = layout.PREPARED
SRC = layout.FIDELITY
FILES = ["members_fidelity.csv", "summary_fidelity.json", "figure_data.npz"]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    prov = {}
    for f in FILES:
        shutil.copy2(SRC / f, DATA / f)
        prov[f] = dict(source=layout.relative(SRC / f), sha256=hashlib.sha256((DATA / f).read_bytes()).hexdigest())
    (DATA / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
