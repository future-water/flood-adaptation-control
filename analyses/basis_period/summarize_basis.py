"""Compare the restricted-basis allocations (results_basis/) with the production allocation over 2030-2099 (results/).

Per member and for the ensemble-median allocation: top-decile overlap and Spearman rho; open-loop reconstruction skill
over 2030-2099 and the projection error of the restricted basis. Reaches without a finite reference mean flow or
drainage area are left out (14,415 of 14,421 reaches; top decile k = 1,441).

Reads <data>/results/<member>.npz and <data>/results_basis/<member>_b<N>.{npz,json}, where <data> is the extracted
outputs_basis_period archive (default) or the work folder of a rerun. Writes summary_basis_b<N>.json to the output
folder. Usage: python summarize_basis.py --basis-years 10 [--data DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import data, out  # noqa: E402
from compare_utils import CORE, compare  # noqa: E402


def spread(v):
    return dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))


def summarize(data, years):
    files = sorted((data / "results_basis").glob(f"*_b{years}.npz"))
    keys = [f.name.replace(f"_b{years}.npz", "") for f in files]
    meta = {k: json.loads(f.with_suffix(".json").read_text()) for k, f in zip(keys, files)}
    ok = [k for k in keys if meta[k]["budget"]["status"] == "interpolated"]
    ids = np.load(data / "results" / f"{ok[0]}.npz")["reach_ids"]
    prod = {k: np.load(data / "results" / f"{k}.npz")["effort_production_test"] for k in ok}
    basis = {k: np.load(data / "results_basis" / f"{k}_b{years}.npz")["effort_basis_test"] for k in ok}
    assert all(np.array_equal(np.load(data / "results_basis" / f"{k}_b{years}.npz")["reach_ids"], ids) for k in ok)
    core = pd.read_csv(CORE).set_index("COMID").loc[ids]
    q, da = core.reference_mean_flow.to_numpy(), core.drainage_area.to_numpy()
    keep = np.isfinite(q) & np.isfinite(da)
    k = int(np.floor(0.1 * keep.sum()))
    per = {m: compare(prod[m][keep], basis[m][keep], k) for m in ok}
    med_p, med_b = np.median([prod[m] for m in ok], 0), np.median([basis[m] for m in ok], 0)
    return dict(basis_years=years, n_members=len(ok), excluded=[m for m in keys if m not in ok],
                ensemble_median=compare(med_p[keep], med_b[keep], k),
                members=dict(top_decile_overlap=spread([v["top_decile_overlap"] for v in per.values()]),
                             spearman=spread([v["spearman"] for v in per.values()])),
                skill_2030_2099=dict(nse_median=spread([meta[m]["nse_median"] for m in ok]), kge_median=spread([meta[m]["kge_median"] for m in ok])),
                projection_error_2030_2099=spread([meta[m]["projection_error_test"] for m in ok]),
                max_spectral_radius=spread([max(meta[m]["spectral_radii"]) for m in ok]),
                per_member={m: dict(**per[m], nse=meta[m]["nse_median"], kge=meta[m]["kge_median"],
                                    projection_error=meta[m]["projection_error_test"]) for m in ok})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis-years", type=int, default=10)
    p.add_argument("--data", type=Path, default=data("basis_period"),
                   help="folder holding results/ and results_basis/ (default: the extracted archive)")
    a = p.parse_args()
    result = summarize(a.data, a.basis_years)
    path = out("basis_period") / f"summary_basis_b{a.basis_years}.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "per_member"}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
