"""Table S2: reach-level effort with the 5-year flood threshold compared with the 2-year threshold of the main analysis.

Both use r = 30 and the production windows and regularization; only the threshold differs.
RL2 effort is the production result (rl2_production_effort_ssp245.npz written by extract_rl2_effort.py, or the
published outputs_flood_rl2/results_*.parquet tables with --results-dir), RL5 effort comes from rl_effort_r30.py.
For each budget, every member is interpolated to the budget between the two bracketing penalty weights (a member that
cannot reach the budget is evaluated at its maximum effort), the ensemble median is taken per reach over the 14
SSP2-4.5 members, and the script reports the Spearman correlation, the top-10% overlap and the effort share by
Strahler stream order.

Writes summary_rl2_rl5_budget{T}.json and member_rl2_rl5_budget{T}.csv (effort reached by each member, Gm3/yr)
to --out-dir (default FAC_OUT_ROOT/flood_threshold) and prints the Table S2 column of each budget.
Usage: python compare_rl2_rl5.py [--budget 5 10] [--results-dir DIR] [--flowlines SHP]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repository root (paths)
from paths import data, out  # noqa: E402
from extract_rl2_effort import MEMBERS, flowline_order, rl2_effort  # noqa: E402

DATA = data('flood_threshold')


def at_budget(R_values, eff, target):
    """eff: (nR, n) per-reach effort; linear interpolation between bracketing weights (figure-code rule)."""
    tot = eff.sum(axis=1) / 1e9
    order = np.argsort(tot)
    es = tot[order]
    if target <= es[0]:
        return eff[order[0]]
    if target >= es[-1]:
        return eff[order[-1]]
    ia = int(np.searchsorted(es, target)); ib = ia - 1
    w = 0.5 if es[ia] == es[ib] else (target - es[ib]) / (es[ia] - es[ib])
    return eff[order[ib]] * (1 - w) + eff[order[ia]] * w


def topk(series, frac):
    k = int(np.ceil(len(series) * frac))
    return set(series.nlargest(k).index)


def rl2_members(a):
    """{hydro_gcm: DataFrame (rows R_weight ascending, columns reach_id ascending)} of the production RL2 effort."""
    if a.results_dir is not None:
        return {f'{h}_{g}': rl2_effort(a.results_dir, h, g) for h, g in MEMBERS}
    z = np.load(a.rl2, allow_pickle=False)
    R = pd.Index(z['R_weight'], name='R_weight')
    reaches = pd.Index(z['reach_id'], name='reach_id')
    eff = z['eff_total_L1']
    return {str(m): pd.DataFrame(eff[i], index=R, columns=reaches) for i, m in enumerate(z['members'])}


def stream_order(a):
    if a.flowlines is not None:
        return flowline_order(a.flowlines)
    t = pd.read_csv(a.stream_order)
    return t.set_index('COMID').StreamOrde


def compare(target, rl2, order, rl5_dir):
    rows2, rows5 = {}, {}
    member_stats = []
    for h, g in MEMBERS:
        piv = rl2[f'{h}_{g}']
        # C order, as the pivot of the results table, so the basin sums in at_budget add up in the same order.
        e2 = pd.Series(at_budget(piv.index.values, np.ascontiguousarray(piv.to_numpy()), target), index=piv.columns)
        z = np.load(Path(rl5_dir) / f'effort_rl5_r30_{h}_{g}_ssp245.npz')
        e5 = pd.Series(at_budget(z['R_values'], z['eff'], target), index=z['reach_id'])
        common = e2.index.intersection(e5.index)
        e2, e5 = e2.loc[common], e5.loc[common]
        rows2[f'{h}_{g}'], rows5[f'{h}_{g}'] = e2, e5
        member_stats.append(dict(member=f'{h}_{g}', effort2=float(e2.sum() / 1e9), effort5=float(e5.sum() / 1e9)))
    E2 = pd.DataFrame(rows2).median(axis=1)
    E5 = pd.DataFrame(rows5).median(axis=1)
    idx = E2.index.intersection(E5.index)
    E2, E5 = E2.loc[idx], E5.loc[idx]
    so = order.reindex(idx)
    ok = (E2 > 0) & (E5 > 0)
    res = {'n_reaches': int(len(idx)),
           'ensemble_spearman': float(spearmanr(E2[ok], E5[ok])[0])}
    k = int(np.ceil(len(idx) * 10 / 100))
    res['top10_overlap_pct'] = 100 * len(topk(E2, 0.1) & topk(E5, 0.1)) / k
    shares = {}
    for o in range(1, 8):
        msk = so == o
        shares[str(o)] = dict(rl2=float(100 * E2[msk].sum() / E2.sum()), rl5=float(100 * E5[msk].sum() / E5.sum()))
    res['effort_share_by_order_pct'] = shares
    ms = pd.DataFrame(member_stats)
    res['member_effort_check_Gm3yr'] = [float(ms.effort2.min()), float(ms.effort2.max()),
                                        float(ms.effort5.min()), float(ms.effort5.max())]
    return res, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, nargs='+', default=[5.0, 10.0], help='Gm3/yr')
    ap.add_argument('--rl5-dir', type=Path, default=DATA / 'effort')
    ap.add_argument('--rl2', type=Path, default=DATA / 'rl2_production_effort_ssp245.npz')
    ap.add_argument('--results-dir', type=Path, default=None,
                    help='read RL2 effort from results_{hydro}_{gcm}.parquet/.csv here instead of --rl2')
    ap.add_argument('--stream-order', type=Path, default=DATA / 'stream_order.csv')
    ap.add_argument('--flowlines', type=Path, default=None,
                    help='read StreamOrde from the NHDPlusV2 flowline shapefile instead of --stream-order')
    ap.add_argument('--out-dir', type=Path, default=None, help='default: FAC_OUT_ROOT/flood_threshold')
    a = ap.parse_args()
    out_dir = a.out_dir if a.out_dir is not None else out('flood_threshold')
    out_dir.mkdir(parents=True, exist_ok=True)
    rl2, order = rl2_members(a), stream_order(a)
    for target in a.budget:
        res, ms = compare(target, rl2, order, a.rl5_dir)
        (out_dir / f'summary_rl2_rl5_budget{target:g}.json').write_text(json.dumps(res, indent=2))
        ms.to_csv(out_dir / f'member_rl2_rl5_budget{target:g}.csv', index=False)
        o7 = res['effort_share_by_order_pct']['7']
        capped = ms[ms.effort5 < target * (1 - 1e-9)]
        print(f'Table S2, {target:g} Gm3/yr: Spearman rho {res["ensemble_spearman"]:.3f}; '
              f'top-10% overlap {res["top10_overlap_pct"]:.1f}%; order-7 effort share {o7["rl2"]:.1f} -> '
              f'{o7["rl5"]:.1f}% (2-year -> 5-year); {len(capped)} member(s) at maximum RL5 effort'
              + ''.join(f'; {m} {e:.2f}' for m, e in zip(capped.member, capped.effort5)))
    print(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
