"""Slim inputs for compare_rl2_rl5.py (Table S2): the RL2 production effort and the stream order of each reach.

The 2-year side of Table S2 is the production run of the main analysis. Its per-reach results tables
(outputs_flood_rl2/results_{hydro}_{gcm}.parquet in the Zenodo record, written by 04_dmd_lqr_analysis.py; CSV
copies of the same tables are also read) hold about 80 metrics for 4 SSPs. Table S2 needs only eff_total_L1 for
SSP2-4.5 at the 19 penalty weights. This script keeps that, plus StreamOrde from the NHDPlusV2 flowlines of the
shape archive, so the comparison runs without either archive.

Outputs (--out-dir, default FAC_OUT_ROOT/flood_threshold):
  rl2_production_effort_ssp245.npz
      members       (14,)            hydro_gcm labels, in the order of MEMBERS
      reach_id      (14421,)         NHDPlusV2 COMID, ascending
      R_weight      (19,)            LQR penalty weights, ascending
      eff_total_L1  (14, 19, 14421)  mean annual control effort per reach (m3/yr), SSP2-4.5, 2-year threshold
  stream_order.csv  COMID, StreamOrde for the same reaches (empty for reaches absent from the flowline file)

Usage: python extract_rl2_effort.py [--results-dir DIR] [--flowlines SHP] [--out-dir DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (paths)
from paths import DATA, out  # noqa: E402

GCMS = ['MPI-ESM1-2-HR', 'ACCESS-CM2', 'EC-Earth3', 'CNRM-ESM2-1', 'BCC-CSM2-MR', 'MRI-ESM2-0', 'NorESM2-MM']
HYDROS = ['VIC5', 'PRMS']
MEMBERS = [(h, g) for g in GCMS for h in HYDROS]
COLUMNS = ['reach_id', 'scenario', 'R_weight', 'eff_total_L1']
SCENARIO = 245


def results_file(results_dir, hydro, gcm):
    """results_{hydro}_{gcm}.parquet (Zenodo record) or a .csv copy of the same table."""
    for ext in ('parquet', 'csv'):
        path = Path(results_dir) / f'results_{hydro}_{gcm}.{ext}'
        if path.is_file():
            return path
    raise FileNotFoundError(f'results_{hydro}_{gcm}.parquet or .csv not found in {results_dir}')


def rl2_effort(results_dir, hydro, gcm):
    """SSP2-4.5 eff_total_L1 of one member: rows R_weight (ascending), columns reach_id (ascending)."""
    path = results_file(results_dir, hydro, gcm)
    if path.suffix == '.parquet':
        d = pd.read_parquet(path, columns=COLUMNS)
    else:
        d = pd.read_csv(path, usecols=COLUMNS)
    d = d[d.scenario == SCENARIO]
    return d.pivot(index='R_weight', columns='reach_id', values='eff_total_L1').sort_index()


def flowline_order(flowlines):
    """StreamOrde by COMID from the NHDPlusV2 flowline shapefile."""
    import geopandas as gpd
    net = gpd.read_file(flowlines, columns=['COMID', 'StreamOrde'], ignore_geometry=True)
    return net.set_index(net.COMID.astype(int)).StreamOrde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=DATA / 'outputs_flood_rl2')
    ap.add_argument('--flowlines', type=Path, default=DATA / 'shape' / 'Flowlines_huc02_12_Brazos.shp')
    ap.add_argument('--out-dir', type=Path, default=None, help='default: FAC_OUT_ROOT/flood_threshold')
    a = ap.parse_args()
    out_dir = a.out_dir if a.out_dir is not None else out('flood_threshold')
    out_dir.mkdir(parents=True, exist_ok=True)

    effs, reach_id, R_weight = [], None, None
    for h, g in MEMBERS:
        piv = rl2_effort(a.results_dir, h, g)
        if reach_id is None:
            reach_id, R_weight = piv.columns.to_numpy(np.int64), piv.index.to_numpy(np.float64)
        assert np.array_equal(piv.columns.to_numpy(np.int64), reach_id), f'{h}_{g}: reach set differs'
        assert np.array_equal(piv.index.to_numpy(np.float64), R_weight), f'{h}_{g}: penalty weights differ'
        assert not piv.isna().any().any(), f'{h}_{g}: missing values'
        effs.append(piv.to_numpy(np.float64))
        print(f'{h}_{g}: {piv.shape[1]} reaches x {piv.shape[0]} weights', flush=True)
    labels = np.array([f'{h}_{g}' for h, g in MEMBERS])
    np.savez_compressed(out_dir / 'rl2_production_effort_ssp245.npz', members=labels, reach_id=reach_id,
                        R_weight=R_weight, eff_total_L1=np.stack(effs))

    order = flowline_order(a.flowlines).reindex(reach_id).astype('Int64')
    pd.DataFrame({'COMID': reach_id, 'StreamOrde': order.array}).to_csv(out_dir / 'stream_order.csv', index=False)
    print(f'wrote {out_dir}: {len(reach_id)} reaches, {int(order.isna().sum())} without stream order')


if __name__ == '__main__':
    main()
