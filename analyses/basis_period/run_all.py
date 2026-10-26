"""Run run_member.py for all 14 SSP2-4.5 members, a few at a time, one log per member.

Logs go to <FAC_WORK_ROOT>/basis_period/logs/.
Usage: python run_all.py [--workers 3] [--threads 5] [--cases 0,1,...]
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


def run(i, threads):
    key = "_".join(CASES[i])
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"{key}.log", "w") as log:
        rc = subprocess.run([sys.executable, str(HERE / "run_member.py"), "--case", str(i), "--threads", str(threads)],
                            stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{key} exit {rc}", flush=True)
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--cases", default=",".join(map(str, range(len(CASES)))))
    a = p.parse_args()
    cases = [int(c) for c in a.cases.split(",")]
    with ThreadPoolExecutor(a.workers) as pool:
        codes = list(pool.map(lambda i: run(i, a.threads), cases))
    print("finished", sum(c == 0 for c in codes), "of", len(codes), flush=True)
    if any(codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
