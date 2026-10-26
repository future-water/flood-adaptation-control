"""Collect the control-input series and the map data for the closed-loop figure (Figure S6; read-only on all sources).

Inputs: for the two SSP2-4.5 members shown in panels (a,b) (the GCM with the highest full-period mainstem NSE averaged
over the two hydrologic models), the daily network attenuation effort sum_i |u_i(t)| of the surrogate's closed loop
(WORK/controllers/<case>/controller.npz: |U_r u_r(t)| summed over reaches) and of the loop closed on the routing model
(WORK/results_fidelity/<tag>_reach.npz, daily_effort).
Map (panel e): the Brazos network geometry (EPSG:5070) of the outputs_common archive (network_geometry_5070.npz) and,
for every mapped reach, whether it is in the top decile of the ensemble-median effort of the 14 SSP2-4.5 members in
both loops, in the routing-model loop only, or in the surrogate loop only.
Writes WORK/data/inputs_map.npz. Usage: python prepare_inputs_map.py (after prepare_data.py)
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout  # noqa: E402

DATA = layout.PREPARED
GEOM = layout.GEOMETRY
SEC_PER_DAY, GM3, NYEARS = 86400.0, 1e9, 80


def main():
    members = pd.read_csv(DATA / "members_fidelity.csv")
    fd = np.load(DATA / "figure_data.npz")
    cases = fd["cases"].tolist()
    assert members.case.tolist() == cases

    # ---- (a,b): best SSP2-4.5 GCM by the mean mainstem NSE of its two hydrologic models
    m245 = members[members.ssp == 245]
    pair = m245.pivot_table(index="gcm", columns="hydro", values="ms_nse")[["VIC5", "PRMS"]].mean(axis=1)
    gcm = pair.idxmax()
    shown = {h: f"{h}_{gcm}_ssp245_budget10" for h in ("VIC5", "PRMS")}
    e_sur, e_cl, dates = {}, {}, None
    for h, case in shown.items():
        ctrl = layout.controller(case)
        c = np.load(ctrl)
        U, ur = c["U_r"], c["u_reduced"]
        d = np.empty(ur.shape[1])
        for a in range(0, ur.shape[1], 2000):
            d[a:a + 2000] = np.abs(U @ ur[:, a:a + 2000]).sum(axis=0)
        e_sur[h] = d
        if dates is None:
            dates = c["dates"][: ur.shape[1]]
        js = glob.glob(str(layout.FIDELITY / f"{case}_closed_R*.json"))
        assert len(js) == 1, case
        reach_npz = Path(js[0]).with_name(Path(js[0]).stem + "_reach.npz")
        z = np.load(reach_npz)
        e_cl[h] = z["daily_effort"]
        assert e_cl[h].shape == d.shape
        i = cases.index(case)  # the daily series add up to the member's annual effort per reach in figure_data.npz
        for name, series, ref in (("surrogate", d, fd["es_reach"][i].sum()), ("routing", e_cl[h], fd["ec_reach"][i].sum())):
            total = series.sum() * SEC_PER_DAY / GM3 / NYEARS
            assert abs(total / ref - 1) < 0.02, (case, name, total, ref)
        print(case, "daily effort r = %.3f" % np.corrcoef(d, e_cl[h])[0, 1])

    # ---- (e): top decile of the ensemble-median effort of the 14 SSP2-4.5 members, both loops
    is245 = np.array([c.endswith("_ssp245_budget10") for c in cases])
    assert is245.sum() == 14 and np.array_equal(fd["comid"][0], fd["comid"][is245][0])
    assert all(np.array_equal(fd["comid"][i], fd["comid"][0]) for i in range(len(cases)))
    comid = fd["comid"][0]
    es_med, ec_med = np.median(fd["es_reach"][is245], axis=0), np.median(fd["ec_reach"][is245], axis=0)
    k = int(0.1 * len(comid))
    top = lambda e: set(comid[np.lexsort((comid, -e))[:k]].tolist())
    sur_top, ref_top = top(es_med), top(ec_med)
    g = np.load(GEOM)
    map_ids = g["comid"]
    cat = np.array([1 if c in ref_top and c in sur_top else 2 if c in ref_top else 3 if c in sur_top else 0 for c in map_ids],
                   dtype=np.int8)
    unmapped = len((sur_top | ref_top) - set(map_ids.tolist()))
    counts = dict(n_top=k, both=int((cat == 1).sum()), routing_only=int((cat == 2).sum()), surrogate_only=int((cat == 3).sum()),
                  overlap_pct=100 * len(sur_top & ref_top) / k, unmapped_selected=unmapped)
    print(gcm, "shown, pair-mean mainstem NSE %.3f;" % pair.max(), counts)
    np.savez_compressed(DATA / "inputs_map.npz", shown_gcm=np.array(gcm), shown_cases=np.array([shown["VIC5"], shown["PRMS"]]),
                        dates=dates, effort_sur=np.stack([e_sur["VIC5"], e_sur["PRMS"]]), effort_ref=np.stack([e_cl["VIC5"], e_cl["PRMS"]]),
                        map_comid=map_ids, map_cat=cat, coordinates=g["coordinates"], offsets=g["offsets"], bounds=g["bounds"],
                        map_counts=np.array([counts[kk] for kk in ("n_top", "both", "routing_only", "surrogate_only", "unmapped_selected")]))
    print("saved", DATA / "inputs_map.npz")


if __name__ == "__main__":
    main()
