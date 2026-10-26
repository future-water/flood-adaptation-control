"""Controlled-output fidelity of the surrogate against the independent routing reference in closed loop, all 56 members
(Figure S6, Section 3.1).

For each member, run_allssp_fidelity.py applied the production control law to the routed discharge at the penalty
weight R* at which the closed loop spends the surrogate's own effort (closed_loop.py --fidelity). The surrogate's
controlled discharge (U_r z_controlled + mean: the surrogate closed loop at the production weight) is compared with the
routed controlled discharge through
  - daily NSE and KGE (2009 form) at every reach for Q+ = max(Q, 0), with the routed discharge as reference, pooled
    over reach-member pairs by Strahler stream order (Figure S6 c; Section 3.1)
  - daily NSE at the downstream mainstem reach (COMID 3125212; Figure S6 a,b)
  - the control actions: top-decile overlap and Spearman correlation of reach effort (Figure S6 d)
  - the Spearman correlation of reach residual exceedance above the 2-year level (Figure S6 f)
Writes WORK/results_fidelity/members_fidelity.csv, summary_fidelity.json and figure_data.npz (the files of the
outputs_closed_loop archive; prepare_data.py copies them to WORK/data/).

--summary-only recomputes summary_fidelity.json from members_fidelity.csv and figure_data.npz in --data (default: the
archive, data("closed_loop")), writes it to FAC_OUT_ROOT/closed_loop/ and compares it with the file in --data.
Usage: python summarize_fidelity.py [--procs 10]
       python summarize_fidelity.py --summary-only [--data DIR]
"""
import argparse
import glob
import json
import sys
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import layout  # noqa: E402   (also puts the repository root on sys.path)
from paths import out  # noqa: E402

OUT = layout.FIDELITY
MAINSTEM = 3125212
CHUNK = 2000
BLOCK = 192
VOLUME = 86400.0 / 1e9  # m3/s-day -> Gm3
ORDERS = range(1, 8)
# statistics across members of the per-member columns, as reported (Figure S6 a,b, d and f)
MEMBER_STATS = {"ms_nse": ("min", "max"), "resid_rho": ("median", "min", "max"),
                "alloc_top10": ("median", "min", "max"), "alloc_rho": ("median", "min", "max")}


@lru_cache(maxsize=1)
def order_map():
    z = np.load(layout.STREAM_ORDER)
    return pd.Series(z["StreamOrde"], index=z["COMID"])


def surrogate_exceedance(path):
    """Residual exceedance above the threshold over 2020-2099 per reach in the surrogate's own closed loop, Gm3.

    sum over days of [max(U_r z_controlled + mean, 0) - tau]^+, computed in blocks of 192 reaches. The original runs
    read this quantity from an intermediate per-reach table that computed it from the controller alone with these
    operations; the result is identical bit for bit and needs no routing.
    """
    with np.load(path) as archive:
        c = {k: np.array(archive[k]) for k in ("comid", "U_r", "mean_q", "z_controlled", "threshold_rl2_m3s")}
    U = np.ascontiguousarray(c["U_r"], dtype=np.float64)
    mean = c["mean_q"].astype(np.float64).reshape(-1, 1)
    zc, tau = c["z_controlled"], c["threshold_rl2_m3s"].astype(np.float64).ravel()
    exc = np.empty(len(c["comid"]))
    for a in range(0, len(exc), BLOCK):
        b = min(a + BLOCK, len(exc))
        q = np.maximum(U[a:b] @ zc + mean[a:b], 0)
        exc[a:b] = np.maximum(q - tau[a:b, None], 0).sum(axis=1) * VOLUME
    return c["comid"].astype(np.int64), exc


def scores(a, mean, T):
    """NSE, KGE from the shifted sums of closed_loop.py --fidelity (reference o, surrogate s)."""
    So, So2, Ss, Ss2, Sos, Sd2 = a
    with np.errstate(divide="ignore", invalid="ignore"):
        vo, vs, cov = So2 - So ** 2 / T, Ss2 - Ss ** 2 / T, Sos - So * Ss / T
        nse = 1 - Sd2 / vo
        r = cov / np.sqrt(vo * vs)
        alpha = np.sqrt(vs / vo)
        beta = (Ss / T + mean) / (So / T + mean)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return nse, kge


def daily_scores(o, s):
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.mean((s - o) ** 2) / o.var(), 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def top(a, q=0.1):
    return set(np.argsort(-np.asarray(a))[: int(round(q * len(a)))])


def member(args):
    case, ssp = args
    js = glob.glob(str(OUT / f"{case}_closed_R*.json"))
    assert len(js) == 1, (case, js)
    tag = Path(js[0]).stem
    fid = np.load(OUT / f"{tag}_fidelity.npz")
    z = np.load(OUT / f"{tag}_reach.npz")
    ref_cl = np.load(OUT / f"{tag}_mainstem.npy")  # routed controlled discharge at the mainstem
    T, mean, acc = int(fid["T"]), fid["mean"], fid["acc"]
    comid = fid["comid"]
    nse_q, kge_q = scores(acc, mean, T)

    # mainstem daily series (Q+)
    ms = int(np.flatnonzero(comid == MAINSTEM)[0])
    sur_cl = fid["ms_sur"]
    ms_nse, _ = daily_scores(np.maximum(ref_cl, 0), np.maximum(sur_cl, 0))
    assert abs(ms_nse - nse_q[ms]) < 1e-8, (case, ms_nse, nse_q[ms])

    # control actions of the surrogate (saved inputs) and of the closed loop
    c = np.load(layout.controller(case))
    U, ur = c["U_r"], c["u_reduced"]
    assert np.array_equal(c["comid"], comid)
    es_reach = np.zeros(U.shape[0])
    for a in range(0, ur.shape[1], CHUNK):
        es_reach += np.abs(U @ ur[:, a:a + CHUNK]).sum(axis=1)

    # residual exceedance of the surrogate loop (from the controller) and of the loop on the routing model
    ids, dc = surrogate_exceedance(layout.controller(case))
    assert np.array_equal(ids, comid)
    cc = z["exc_cl"]
    so = order_map().reindex(comid).to_numpy()
    row = dict(
        case=case, ssp=ssp, hydro=case.split("_")[0], gcm=case.split("_")[1], ms_nse=ms_nse,
        resid_rho=spearmanr(dc, cc).statistic,
        alloc_top10=100 * len(top(es_reach) & top(z["effort"])) / len(top(es_reach)),
        alloc_rho=spearmanr(es_reach, z["effort"]).statistic)
    arrays = dict(comid=comid, order=so, nse_q=nse_q, kge_q=kge_q,
                  es_reach=es_reach * 86400 / 1e9 / 80, ec_reach=z["effort"],
                  ms_ref_cl=ref_cl.astype(np.float32), ms_sur_cl=sur_cl.astype(np.float32))
    return row, arrays


def pooled(arrs, key, mask_fn=None):
    v = np.concatenate([a[key][mask_fn(a)] if mask_fn else a[key] for a in arrs])
    v = v[np.isfinite(v)]
    return dict(zip(["p25", "median", "p75"], map(float, np.quantile(v, [.25, .5, .75]))))


def group_stats(rows, arrs):
    df = pd.DataFrame(rows)
    stat = dict(median=lambda s: float(s.median()), min=lambda s: float(s.min()), max=lambda s: float(s.max()))
    return dict(
        n_members=len(df),
        by_order={int(o): dict(nse=pooled(arrs, "nse_q", lambda a, o=o: a["order"] == o),
                               kge=pooled(arrs, "kge_q", lambda a, o=o: a["order"] == o)) for o in ORDERS},
        members={k: {name: stat[name](df[k]) for name in names} for k, names in MEMBER_STATS.items()})


def report(summ):
    al = summ["all"]
    print(f"{al['n_members']} members")
    print("daily NSE of controlled discharge by stream order (median):", {o: round(v["nse"]["median"], 3) for o, v in al["by_order"].items()})
    print("daily KGE of controlled discharge by stream order (median):", {o: round(v["kge"]["median"], 3) for o, v in al["by_order"].items()})
    for k, v in al["members"].items():
        print(f"{k:12s}" + (f" median {v['median']:.4f} " if "median" in v else " ") + f" range {v['min']:.4f}-{v['max']:.4f}")


def flatten(d, prefix=""):
    if isinstance(d, dict):
        return {k2: v2 for k, v in d.items() for k2, v2 in flatten(v, f"{prefix}{k}/").items()}
    return {prefix.rstrip("/"): d}


def summary_only(data):
    """summary_fidelity.json from the archived members_fidelity.csv and figure_data.npz (no routing outputs needed)."""
    members = pd.read_csv(data / "members_fidelity.csv")
    with np.load(data / "figure_data.npz") as fd:
        assert fd["cases"].tolist() == members.case.tolist()
        full = {k: fd[k] for k in ("order", "nse_q", "kge_q")}
    arrs = [{k: v[i] for k, v in full.items()} for i in range(len(members))]
    summ = {"all": group_stats(members.to_dict("records"), arrs)}
    path = out("closed_loop") / "summary_fidelity.json"
    path.write_text(json.dumps(summ, indent=2))
    report(summ)
    print("wrote", path)
    ref = data / "summary_fidelity.json"
    if ref.exists():
        a, b = flatten(json.loads(path.read_text())), flatten(json.loads(ref.read_text()))
        assert a.keys() == b.keys(), sorted(set(a) ^ set(b))
        rel = max(abs(a[k] - b[k]) / max(abs(b[k]), 1e-300) for k in a)
        print(f"compared with {ref.name}: {len(a)} values, "
              + ("identical" if a == b else f"max relative difference {rel:.1e}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--data", type=Path, default=layout.PUBLISHED)
    a = ap.parse_args()
    if a.summary_only:
        summary_only(a.data)
        return
    cases = [(c, s) for s in layout.SSPS for c in layout.cases([s])]
    with Pool(a.procs) as pool:
        res = pool.map(member, cases, chunksize=1)
    rows, arrs = [r for r, _ in res], [x for _, x in res]
    pd.DataFrame(rows).to_csv(OUT / "members_fidelity.csv", index=False)
    summ = {"all": group_stats(rows, arrs)}
    (OUT / "summary_fidelity.json").write_text(json.dumps(summ, indent=2))
    np.savez_compressed(OUT / "figure_data.npz", cases=np.array([r["case"] for r in rows]),
                        **{k: np.stack([x[k] for x in arrs]) for k in arrs[0]})
    report(summ)


if __name__ == "__main__":
    main()
