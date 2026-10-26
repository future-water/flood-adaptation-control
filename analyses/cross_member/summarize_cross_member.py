"""Cross-member agreement of the production allocations (Section 3.3: 83% (69-93%) and 88% (79-94%)).

For the 14 SSP2-4.5 members, each matched to 10 Gm3/yr over 2020-2099 (effort_production_full of
basis_period/run_member.py): the top-decile overlap of every pair of members (91 pairs) and of each member with the
equal-member reachwise median allocation. The top decile is the k = floor(0.1 N) reaches with the largest effort, with
N = all 14,421 reaches of the network (k = 1,442); no reach is masked. This is the definition that produced the
published numbers. The basis-period comparison (basis_period/summarize_basis.py) instead leaves out the 6 reaches
without a drainage area (k = 1,441); that changes the unrounded values here slightly but none of the rounded ones.

Reads <data>/results/<member>.npz (default: the extracted outputs_basis_period archive). Writes
summary_cross_member.json to the output folder. Usage: python summarize_cross_member.py [--data DIR]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import data, out  # noqa: E402


def top(values, k):
    """Indices of the k largest values (same definition as basis_period/compare_utils.top)."""
    return set(np.argsort(-values, kind="stable")[:k])


def summarize(data):
    files = sorted((data / "results").glob("*.npz"))
    assert len(files) == 14, f"expected the 14 SSP2-4.5 member files in {data / 'results'}, found {len(files)}"
    names = [f.stem for f in files]
    ids = np.load(files[0])["reach_ids"]
    assert all(np.array_equal(np.load(f)["reach_ids"], ids) for f in files)
    eff = {f.stem: np.load(f)["effort_production_full"] for f in files}
    assert all(np.isfinite(v).all() for v in eff.values())
    n = len(ids)
    k = int(np.floor(0.1 * n))
    tops = {m: top(eff[m], k) for m in names}
    pairs = np.array([100 * len(tops[a] & tops[b]) / k for a, b in itertools.combinations(names, 2)])
    med = np.median(np.stack([eff[m] for m in names]), axis=0)
    tmed = top(med, k)
    vs_med = np.array([100 * len(tops[m] & tmed) / k for m in names])
    result = {"n_members": len(names), "n_pairs": len(pairs),
              "pairwise_top10_median": float(np.median(pairs)), "pairwise_top10_min": float(pairs.min()),
              "pairwise_top10_max": float(pairs.max()),
              "member_vs_ensemble_median_top10_median": float(np.median(vs_med)),
              "member_vs_ensemble_median_top10_min": float(vs_med.min()),
              "member_vs_ensemble_median_top10_max": float(vs_med.max())}
    return result, n, k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=data("basis_period"),
                   help="folder holding results/<member>.npz (default: the extracted archive)")
    a = p.parse_args()
    result, n, k = summarize(a.data)
    path = out("cross_member") / "summary_cross_member.json"
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(f"reaches {n}, top decile k = {k}")
    print(f"Section 3.3: pairwise {result['pairwise_top10_median']:.0f}% "
          f"({result['pairwise_top10_min']:.0f}-{result['pairwise_top10_max']:.0f}%), "
          f"member vs ensemble median {result['member_vs_ensemble_median_top10_median']:.0f}% "
          f"({result['member_vs_ensemble_median_top10_min']:.0f}-{result['member_vs_ensemble_median_top10_max']:.0f}%)")
    print("wrote", path)


if __name__ == "__main__":
    main()
