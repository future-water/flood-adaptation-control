"""Run run_basis_variant.py for all 14 SSP2-4.5 members, a few at a time, one log per member.
Logs go to <FAC_WORK_ROOT>/basis_period/logs/.
Usage: python run_basis_all.py --basis-years 10 [--workers 3] [--threads 5] [--cases 0,1,...]
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))   # repository root (paths)
from paths import work  # noqa: E402
from run_member import CASES  # noqa: E402

LOGS = work("basis_period") / "logs"


def run(i, years, threads):
    key = "_".join(CASES[i])
    with open(LOGS / f"basis_{key}_b{years}.log", "w") as log:
        rc = subprocess.run([sys.executable, str(HERE / "run_basis_variant.py"), "--case", str(i), "--basis-years", str(years),
                             "--threads", str(threads)], stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{key} b{years} exit {rc}", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis-years", type=int, default=10)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--cases", default=",".join(map(str, range(len(CASES)))))
    a = p.parse_args()
    LOGS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        rcs = list(ex.map(lambda i: run(i, a.basis_years, a.threads), map(int, a.cases.split(","))))
    print("failures:", sum(rc != 0 for rc in rcs))
    if any(rcs):
        sys.exit(1)


if __name__ == "__main__":
    main()
