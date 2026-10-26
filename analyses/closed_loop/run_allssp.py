"""Controller recovery for one member at the 10 Gm3/yr budget (Figure S6).

Recovers the saved controller with the unmodified production code (recover_member.py, which refuses to write inside
FAC_RAW_ROOT) into WORK/controllers/<case>/, where the closed-loop runs (run_allssp_closed.py, run_allssp_fidelity.py)
and summarize_fidelity.py read it, and deletes the uncontrolled discharge that recover_member.py also writes
(3.4 GB; make_q_ornl.py regenerates it when a closed-loop run needs it). All 56 members go through the same code.

Usage: python run_allssp.py --scenario 126 --hydro PRMS --gcm ACCESS-CM2
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=int, required=True)
    p.add_argument("--hydro", required=True)
    p.add_argument("--gcm", required=True)
    a = p.parse_args()
    case = f"{a.hydro}_{a.gcm}_ssp{a.scenario}_budget10"
    work, out = layout.RECOVER / case, layout.CONTROLLERS / case
    if (out / "controller.npz").exists():
        print(case, "already done"); return
    subprocess.run([sys.executable, str(HERE / "recover_member.py"), "--hydro", a.hydro,
                    "--gcm", a.gcm, "--scenario", str(a.scenario), "--budget", "10",
                    "--work-dir", str(work), "--out-dir", str(out)], check=True)
    (work / "q_ornl.npy").unlink(missing_ok=True)
    print(case, "done", flush=True)


if __name__ == "__main__":
    main()
