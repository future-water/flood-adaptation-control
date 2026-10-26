"""Read-only loading helpers; arithmetic and alignment match supplied code."""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
REALIZATION = {'ACCESS-CM2': 'r1i1p1f1', 'CNRM-ESM2-1': 'r1i1p1f2', 'EC-Earth3': 'r1i1p1f1', 'MPI-ESM1-2-HR': 'r1i1p1f1', 'BCC-CSM2-MR': 'r1i1p1f1', 'MRI-ESM2-0': 'r1i1p1f1', 'NorESM2-MM': 'r1i1p1f1'}

def _future_paths(root: Path, gcm: str, hydro: str, ssp: int) -> tuple[list[Path], list[Path]]:
    rid = REALIZATION[gcm]
    flow_prefix = f'{hydro}_RAPID'
    flow = [root / 'streamflow' / f'{flow_prefix}_{gcm}_ssp{ssp}_{rid}_DBCCA_Daymet_2020_2059_daily.parquet', root / 'streamflow' / f'{flow_prefix}_{gcm}_ssp{ssp}_{rid}_DBCCA_Daymet_2060_2099_daily.parquet']
    precip = [root / 'precipitation' / f'{gcm}_ssp{ssp}_{rid}_DBCCA_Daymet_prcp_2020_2059_daily.parquet', root / 'precipitation' / f'{gcm}_ssp{ssp}_{rid}_DBCCA_Daymet_prcp_2060_2099_daily.parquet']
    missing = [str(path) for path in flow + precip if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing inputs:\n' + '\n'.join(missing))
    return (flow, precip)

def _read_flow(paths: list[Path], dtype: np.dtype) -> tuple[np.ndarray, pd.DatetimeIndex, list[int]]:
    chunks: list[np.ndarray] = []
    date_chunks: list[pd.DatetimeIndex] = []
    reach_ids: list[int] | None = None
    for path in paths:
        started = time.perf_counter()
        df = pd.read_parquet(path).sort_index()
        ids = pd.Index(df.columns).astype(float).astype(int).tolist()
        if reach_ids is None:
            reach_ids = ids
        elif ids != reach_ids:
            
            if set(ids) != set(reach_ids):
                raise ValueError(f'Flow reach sets differ across periods: {path}')
            order = pd.Index(ids).get_indexer(reach_ids)
            df = df.iloc[:, order]
            print(f'Realigned flow columns to first-period reach order: {path.name}', flush=True)
        chunks.append((df / 35.315).to_numpy(dtype=dtype, copy=True).T)
        date_chunks.append(pd.DatetimeIndex(df.index))
        print(f'loaded flow {path.name} in {time.perf_counter() - started:.1f}s', flush=True)
        del df
        gc.collect()
    assert reach_ids is not None
    return (np.concatenate(chunks, axis=1), date_chunks[0].append(date_chunks[1:]), reach_ids)

def _read_precip(paths: list[Path], dtype: np.dtype, reach_ids: list[int]) -> tuple[np.ndarray, pd.DatetimeIndex, int]:
    chunks: list[np.ndarray] = []
    date_chunks: list[pd.DatetimeIndex] = []
    flow_columns = pd.Index(reach_ids)
    unmatched: set[int] = set()
    for path in paths:
        started = time.perf_counter()
        df = pd.read_parquet(path).sort_index()
        df.columns = pd.Index(df.columns).astype(float).astype(int)
        unmatched.update(set(flow_columns) - set(df.columns))
        df = df.reindex(columns=flow_columns, fill_value=0.0).fillna(0.0)
        chunks.append(df.to_numpy(dtype=dtype, copy=True).T)
        date_chunks.append(pd.DatetimeIndex(df.index))
        print(f'loaded precip {path.name} in {time.perf_counter() - started:.1f}s', flush=True)
        del df
        gc.collect()
    return (np.concatenate(chunks, axis=1), date_chunks[0].append(date_chunks[1:]), len(unmatched))

def load_data(args: argparse.Namespace, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[int], int]:
    flow_paths, precip_paths = _future_paths(args.data_root, args.gcm, args.hydro, args.ssp)
    flow, flow_dates, reach_ids = _read_flow(flow_paths, dtype)
    precip, precip_dates, unmatched = _read_precip(precip_paths, dtype, reach_ids)
    if not flow_dates.equals(precip_dates):
        common = flow_dates.intersection(precip_dates)
        flow_idx = flow_dates.get_indexer(common)
        precip_idx = precip_dates.get_indexer(common)
        flow = flow[:, flow_idx]
        precip = precip[:, precip_idx]
        dates = common
    else:
        dates = flow_dates
    return (flow, precip, dates, reach_ids, unmatched)
