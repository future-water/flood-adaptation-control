#!/usr/bin/env python3
"""Recover one saved production controller using its original, member-specific R.

Example (run_allssp.py passes the folders of layout.py):
  python recover_member.py --hydro PRMS --gcm ACCESS-CM2 --scenario 126 \
    --work-dir <work>/recover/PRMS_ACCESS-CM2_ssp126_budget10 \
    --out-dir <work>/controllers/PRMS_ACCESS-CM2_ssp126_budget10

The pipeline is 04_dmd_lqr_analysis.py, loaded by common.pipeline_loader; the raw ORNL
streamflow and precipitation are read from FAC_RAW_ROOT, the pooled return levels from the outputs_common
archive and the saved production run (R weight and daily basin output) from the published archive outputs_flood_rl2/.
These inputs are read-only. All generated files are confined to the requested workspace directories. This
script does not sweep R or alter the original POD, DMDc, LQR, simulation, or daily-output functions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analyses/ (common)
import layout  # noqa: E402
from common.pipeline_loader import RAW, load_pipeline, raw_inputs, return_levels_path, source_sha256  # noqa: E402
AGGREGATES = (
    'basin_exceed_baseline', 'basin_exceed_controlled', 'basin_effort',
    'basin_peak_baseline', 'basin_peak_controlled',
)
RELATIVE_L2_WARNING = 0.01


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hydro', required=True, choices=('VIC5', 'PRMS'))
    parser.add_argument('--gcm', required=True)
    parser.add_argument('--scenario', type=int, default=245,
                        choices=(126, 245, 370, 585))
    parser.add_argument('--budget', type=float, default=10.0)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.budget) or args.budget <= 0:
        parser.error('--budget must be positive and finite')
    args.work_dir = args.work_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    for path in (args.work_dir, args.out_dir):
        if path.is_relative_to(RAW):
            parser.error('Generated files must not be written inside FAC_RAW_ROOT')
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_saved_case(args):
    path = layout.saved_daily(args.hydro, args.gcm)
    saved = pd.read_parquet(path)
    required = {'date', 'scenario', 'budget_target', 'R_weight', *AGGREGATES}
    missing = required.difference(saved.columns)
    if missing:
        raise ValueError(f'{path}: missing saved columns {sorted(missing)}')
    selected = (pd.to_numeric(saved.scenario) == args.scenario) & np.isclose(
        pd.to_numeric(saved.budget_target), args.budget, rtol=0, atol=1e-9)
    saved = saved.loc[selected].copy()
    if saved.empty:
        raise ValueError(f'No saved daily case for {args.hydro}, {args.gcm}, '
                         f'SSP{args.scenario}, budget {args.budget}')
    for column, expected in [('hydro', args.hydro), ('gcm', args.gcm)]:
        if column in saved and not saved[column].eq(expected).all():
            raise ValueError(f'{path}: selected rows have inconsistent {column}')
    weights = pd.to_numeric(saved.R_weight).unique()
    if len(weights) != 1 or not np.isfinite(weights[0]) or weights[0] <= 0:
        raise ValueError(f'The saved case must identify one finite positive R: {weights}')
    saved['date'] = pd.to_datetime(saved.date)
    if saved.date.duplicated().any():
        raise ValueError('Saved case contains duplicate dates; comparison would be ambiguous')
    saved = saved.sort_values('date').reset_index(drop=True)
    return saved, float(weights[0]), path


def compare_daily(daily, saved):
    if daily.date.duplicated().any():
        raise ValueError('Recovered daily output contains duplicate dates')
    joined = daily.merge(saved, on='date', how='inner',
                         suffixes=('_new', '_saved'), validate='one_to_one')
    if joined.empty:
        raise ValueError('Recovered and saved outputs have no overlapping dates')
    comparison = {}
    for key in AGGREGATES:
        a = joined[f'{key}_new'].to_numpy(dtype=float)
        b = joined[f'{key}_saved'].to_numpy(dtype=float)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.all():
            error = float(np.linalg.norm(a - b))
            norm = float(np.linalg.norm(b))
            relative = error / norm if norm > 0 else (0.0 if error == 0 else None)
            maximum = float(np.max(np.abs(a - b)))
        else:
            relative = maximum = None
        comparison[key] = {
            'relative_L2': relative,
            'max_absolute_error': maximum,
            'n_finite_pairs': int(finite.sum()),
            'significant_discrepancy': relative is None or relative > RELATIVE_L2_WARNING,
        }
    recovered_dates = pd.DatetimeIndex(daily.date)
    saved_dates = pd.DatetimeIndex(saved.date)
    coverage = {
        'n_recovered_daily_rows': len(daily),
        'n_saved_daily_rows': len(saved),
        'n_days_compared_to_saved': len(joined),
        'n_recovered_dates_missing_from_saved': len(recovered_dates.difference(saved_dates)),
        'n_saved_dates_missing_from_recovered': len(saved_dates.difference(recovered_dates)),
        'comparison_first_day': str(joined.date.min().date()),
        'comparison_last_day': str(joined.date.max().date()),
    }
    return comparison, coverage


def main(argv=None):
    args = parse_args(argv)
    begin = time.perf_counter()
    stages = {}

    @contextmanager
    def stage(label):
        start = time.perf_counter()
        print(f'[{time.perf_counter() - begin:8.1f}s] {label}', flush=True)
        yield
        stages[label] = time.perf_counter() - start
        print(f'[{time.perf_counter() - begin:8.1f}s] Finished {label} '
              f'({stages[label]:.1f}s)', flush=True)

    with stage('Read saved member, budget, and R weight'):
        saved, R, saved_path = load_saved_case(args)
        cache = return_levels_path(args.hydro, args.gcm)  # raises if the member-specific cache is missing
        print(f'Member: {args.hydro} / {args.gcm} / SSP{args.scenario}; '
              f'saved budget={args.budget:g}; saved R={R:.17g}', flush=True)

    with stage('Prepare workspace and load the unmodified pipeline'):
        args.work_dir.mkdir(parents=True, exist_ok=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        m = load_pipeline()  # checks the sha256 of 04_dmd_lqr_analysis.py
        src = Path(m.__file__)
        if args.gcm not in m.AVAILABLE_GCMS:
            raise ValueError(f'GCM not listed by the pipeline: {args.gcm}')

    with stage('Load original full-period data and align exact RL2 cache'):
        with raw_inputs():  # the loader reads streamflow/ and precipitation/ relative to FAC_RAW_ROOT
            X, P, dates, original_ids = m.load_scenario_data(args.scenario, args.gcm, args.hydro)
        with np.load(cache, allow_pickle=False) as ref:
            ir, it, ids = m.match_to_reference(ref['reach_ids'].astype(int).tolist(), original_ids)
            threshold = np.maximum(ref['rl2'].astype(m.DTYPE)[ir], 0.0)
            cache_n_reaches = len(ref['reach_ids'])
        if not len(ids):
            raise ValueError('No reaches matched the original pooled return-level cache')
        X, P = X[it], P[it]
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError('Original loader returned duplicate or unordered dates')
        q_path = args.work_dir / 'q_ornl.npy'
        np.save(q_path, X.T)
        print(f'Aligned {len(ids):,} reaches × {len(dates):,} days; saved {q_path}', flush=True)

    with stage('Original full-period POD and forcing normalization'):
        pod = m.compute_pod(X, m.RANK)
        del X
        gc.collect()
        forcing = m.normalize_forcing(P)
        del P
        gc.collect()

    with stage('Original windowed DMDc fit'):
        dmd = m.fit_windowed_dmdc(pod.Z, forcing.V)
        print(f'Windows={len(dmd.bounds)}; alphas={dmd.alphas_used}; '
              f'spectral radii={dmd.spectral_radii}', flush=True)

    with stage('Original LQR at saved R and continuous OL/controlled simulations'):
        Q = (m.Q_WEIGHT * np.eye(m.RANK)).astype(m.DTYPE)
        K = [m.lqr_gain(A, Q, R) for A in dmd.A_list]
        z0 = pod.Z[:, :1]
        v = forcing.V[:, :-1]
        zol = m.simulate_openloop(dmd, z0, v)
        zcl, u = m.simulate_cl_flood_only(dmd, z0, v, K, threshold, pod.U_r, pod.mean_x)

    with stage('Recover original daily aggregates and compare saved output'):
        daily = m.compute_daily_basin_output(pod, zol, zcl, u, threshold, dates)
        comparison, coverage = compare_daily(daily, saved)
        actual_effort = float(daily.basin_effort.sum() * m.DT / dates.year.nunique() / 1e9)
        for key, item in comparison.items():
            print(f'{key}: relative_L2={item["relative_L2"]}, '
                  f'max_absolute_error={item["max_absolute_error"]}', flush=True)
        for key, value in [('gcm', args.gcm), ('hydro', args.hydro),
                           ('scenario', args.scenario), ('budget_target', args.budget),
                           ('R_weight', R), ('actual_effort_Gm3yr', actual_effort)]:
            daily[key] = value

    with stage('Save compact controller and recovery outputs'):
        np.savez_compressed(args.out_dir / 'controller.npz',
            comid=np.asarray(ids, dtype=np.int64),
            dates=dates.values.astype('datetime64[D]').astype(str),
            U_r=pod.U_r, mean_q=pod.mean_x, z_ol=zol, z_controlled=zcl,
            u_reduced=u, A=np.asarray(dmd.A_list), E=np.asarray(dmd.E_list),
            K_lqr=np.asarray(K), threshold_rl2_m3s=threshold,
            bounds=np.asarray(dmd.bounds, dtype=np.int64),
            window_for_k=dmd.window_for_k, forcing_mu=forcing.mu,
            forcing_sigma=forcing.global_sigma, forcing_global_sigma=forcing.global_sigma,
            R_weight=R, scenario=args.scenario, budget_target=args.budget,
            hydro=args.hydro, gcm=args.gcm)
        daily.to_parquet(args.out_dir / 'recovered_daily.parquet', index=False)

    flagged = [key for key, value in comparison.items() if value['significant_discrepancy']]
    coverage_mismatch = bool(coverage['n_recovered_dates_missing_from_saved'] or
                             coverage['n_saved_dates_missing_from_recovered'])
    saved_effort = (pd.to_numeric(saved.actual_effort_Gm3yr).unique().tolist()
                    if 'actual_effort_Gm3yr' in saved else None)
    report = {
        'source': src.name, 'source_sha256': source_sha256(src),
        'hydro': args.hydro, 'gcm': args.gcm, 'scenario': args.scenario,
        'budget_target': args.budget,
        'member': f'{args.hydro} / {args.gcm} / SSP{args.scenario}',
        'R_weight': R, 'R_selection': 'Read from the exact saved daily member/scenario/budget case; no R sweep',
        'saved_daily_path': str(saved_path), 'saved_daily_sha256': sha256_file(saved_path),
        'threshold_cache': str(cache), 'threshold_cache_sha256': sha256_file(cache),
        'threshold_return_period': 2, 'n_input_reaches': len(original_ids),
        'n_cache_reaches': cache_n_reaches, 'n_matched_reaches': len(ids),
        'n_states': len(dates), 'n_controls': u.shape[1],
        'period': [str(dates[0].date()), str(dates[-1].date())],
        'window_bounds': [[int(a), int(b)] for a, b in dmd.bounds],
        'window_alphas': [float(x) for x in dmd.alphas_used],
        'window_spectral_radii': [float(x) for x in dmd.spectral_radii],
        'saved_daily_reproduction': comparison, **coverage,
        'relative_L2_warning_threshold': RELATIVE_L2_WARNING,
        'significant_discrepancy': bool(flagged or coverage_mismatch),
        'significant_discrepancy_aggregates': flagged,
        'date_coverage_mismatch': coverage_mismatch,
        'recovered_actual_effort_Gm3yr': actual_effort,
        'saved_actual_effort_Gm3yr_values': saved_effort,
        'POD_random_state_from_original_code': 42,
        'state_resets': 0, 'controller_refit_policy': 'Original identification functions rerun unchanged; fixed saved R',
        'work_dir': str(args.work_dir), 'out_dir': str(args.out_dir),
        'q_ornl_path': str(q_path), 'controller_path': str(args.out_dir / 'controller.npz'),
        'stage_seconds': stages, 'elapsed_seconds': time.perf_counter() - begin,
        'thread_environment': {key: os.environ.get(key) for key in (
            'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
    }
    (args.out_dir / 'recovery.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if report['significant_discrepancy']:
        print('WARNING: recovery comparison requires review: '
              f'aggregates above 1%/nonfinite={flagged}; date coverage mismatch={coverage_mismatch}. '
              'Outputs were retained and the discrepancy was recorded, not silently accepted.', flush=True)
    print(f'Completed {args.hydro}/{args.gcm}/SSP{args.scenario}, '
          f'R={R:.17g}, elapsed={time.perf_counter() - begin:.1f}s; '
          f'report={args.out_dir / "recovery.json"}', flush=True)


if __name__ == '__main__':
    main()
