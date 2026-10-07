"""Check the release and rerun the workflow analyses.

1. Every released file is checked against MANIFEST_SHA256.json (run unpack_data.py first).
2. In a temporary copy, prepare_native_analysis.py rebuilds the prepared annual tables, which are compared
   with the released ones.
3. The observation-construction, observation-date, urban-pressure, albedo and snow, and robustness analyses
   are rerun and every rebuilt result table is compared with the released one.
The XGBoost and random forest refits (run_native_analysis.py prediction) are not repeated here.
"""
from pathlib import Path
import hashlib, json, shutil, subprocess, sys, tempfile
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
WF = Path('analysis/workflow')
manifest = json.loads((R / 'MANIFEST_SHA256.json').read_text(encoding='utf-8'))


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


bad = [n for n, d in {**manifest['files'], **manifest['unpacked']}.items() if not (R / n).is_file() or digest(R / n) != d]
assert not bad, ('missing or changed files; run unpack_data.py first', bad[:10])
print('files checked:', len(manifest['files']) + len(manifest['unpacked']))


def same(a, b, rtol, atol):
    assert list(a.columns) == list(b.columns) and len(a) == len(b)
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            assert np.allclose(a[c], b[c], rtol=rtol, atol=atol, equal_nan=True), c
        else:
            assert a[c].fillna('NA').astype(str).equals(b[c].fillna('NA').astype(str)), c


tmp = Path(tempfile.mkdtemp(prefix='ftp_verify_'))
try:
    for d in ['analysis', 'data']:
        shutil.copytree(R / d, tmp / d, ignore=shutil.ignore_patterns('*.gz', '__pycache__'))
    run = lambda script: subprocess.run([sys.executable, '-X', 'utf8', str(WF / script)], cwd=tmp, check=True,
                                        capture_output=True, text=True, encoding='utf-8')
    prepared = ['annual_panel.csv', 'longterm_windows.csv', 'balanced_windows.csv', 'matched_annual_panel.csv',
                'matched_longterm_windows.csv', 'common_MOD16_windows.csv', 'common_PML_windows.csv']
    before = {n: pd.read_csv(tmp / WF / 'native_analysis' / n) for n in prepared}
    run('prepare_native_analysis.py')
    for n, expected in before.items():
        actual = pd.read_csv(tmp / WF / 'native_analysis' / n)
        keys = ['window_id', 'year'] if 'year' in actual else ['window_id']
        actual = actual.sort_values(keys).reset_index(drop=True); expected = expected.sort_values(keys).reset_index(drop=True)
        same(actual, expected[actual.columns], 1e-11, 1e-11)
    print('prepared tables rebuilt:', len(prepared))
    targets = ['native_decomposition/decomposition_FE.csv', 'native_decomposition/quarter_FE_full_cohort.csv',
               'native_decomposition/conditional_slopes.csv', 'native_decomposition/table_S7.csv', 'native_decomposition/table_S8.csv',
               'native_dates/date_matching_FE.csv', 'native_dates/date_matching_quarter_FE.csv', 'native_dates/conditional_slopes.csv',
               'native_dates/date_overlap.csv', 'native_dates/unshared_date_departures.csv', 'native_decomposition/urban_modulation_FE.csv',
               'native_decomposition/table_S9.csv', 'native_decomposition/table_S10.csv', 'native_process/process_FE.csv',
               'native_process/process_descriptives.csv', 'native_robustness/robustness_FE.csv', 'native_robustness/av_bins.csv',
               'native_robustness/spatial_bins.csv', 'native_robustness/date_dose_bins.csv', 'native_robustness/date_example.csv',
               'native_robustness/season_temperature_grid.csv', 'native_robustness/q1_snow_surface.csv',
               'native_robustness/q1_window_snow.csv', 'native_robustness/winter_decomposition.csv', 'native_decomposition/table_S11.csv']
    released = {t: pd.read_csv(R / WF / t) for t in targets}
    for script in ['analyze_native_decomposition.py', 'analyze_native_dates.py', 'analyze_unshared_dates.py', 'prepare_supplementary_tables.py',
                   'analyze_urban_modulation.py', 'analyze_process_evidence.py', 'analyze_robustness_checks.py', 'prepare_supplementary_tables.py']:
        run(script); print('rerun', script)
    for t, expected in released.items():
        same(pd.read_csv(tmp / WF / t), expected, 1e-10, 1e-12)
    print('result tables reproduced:', len(targets))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print('release verified')
