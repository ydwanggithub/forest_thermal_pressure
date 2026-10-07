"""Test whether the concurrent-construction sign change depends on matching dates
between the two forest portions.

Input: native_dates/dates_all.csv (one row per portion, year and composite date;
fetch_native_support_dates.py). Annual means are rebuilt from composite-date zone
means. Unmatched constructions let each portion use its own valid dates, as the
published concurrent construction does. Matched constructions use only dates on
which both portions have at least m valid cells. Models, scales, covariates,
fixed effects and inference follow analyze_native_observation_support.py.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as tdist
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
OUT = HERE / 'native_dates'
sys.path.insert(0, str(ROOT / 'analysis/pipeline'))
from analyze import twoway_demean, standardize

long = pd.read_csv(HERE / 'native_analysis/longterm_windows.csv')
annual = pd.read_csv(HERE / 'native_analysis/annual_panel.csv')
annual = annual[annual.window_id.isin(long.window_id)].copy()
s = json.loads((HERE / 'native_analysis/scaling.json').read_text())['primary']
keys = pd.read_csv(HERE / 'native_support/quarter_balanced_annual_panel.csv', usecols=['window_id', 'year'])
support = pd.read_csv(HERE / 'native_support/support_all.csv')
d = pd.read_csv(OUT / 'dates_all.csv')
context = annual.drop(columns=['LAI', 'day', 'night', 'LSTd', 'LSTn', 'mean', 'day_minus_night'])

# ---- Anchor: summed jointly valid cell counts reproduce the cell-level joint counts ----
tot = d.groupby(['window_id', 'zone', 'year']).LAI_n.sum().rename('date_sum').reset_index()
sup = support[support.zone.isin(['pui', 'bak'])].copy()
sup['cell_sum'] = sum(sup[f'joint_q{q}n_mean'] for q in range(1, 5)) * sup.local_n
a = tot.merge(sup[['window_id', 'zone', 'year', 'cell_sum']], on=['window_id', 'zone', 'year'], validate='one_to_one')
anchor_err = float((a.date_sum - a.cell_sum).abs().max())
assert len(a) == 1920 * 23 and anchor_err < 1e-6, anchor_err
for v in ['day', 'night']:
    assert (d[v + '_n'] == d.LAI_n).all()
assert (d.dayD_n == d.LAID_n).all() and (d.LAID_n >= d.LAI_n).all()

d['quarter'] = (pd.to_datetime(d.date, format='%Y_%m_%d').dt.month - 1) // 3 + 1
w = d.pivot_table(index=['window_id', 'year', 'date', 'quarter'], columns='zone',
                  values=['LAI', 'day', 'night', 'LAID', 'dayD', 'LAI_n', 'LAID_n'], aggfunc='first')
w.columns = [f'{v}_{z}' for v, z in w.columns]
w = w.reset_index()
for c in ['LAI_n_pui', 'LAI_n_bak', 'LAID_n_pui', 'LAID_n_bak']:
    w[c] = w[c].fillna(0)


def annual_means(frame, cnt, vars_, matched, m, weighting):
    """Near-minus-reference annual means for one date rule."""
    f = frame.copy()
    if matched:
        f = f[(f[cnt + '_pui'] >= m) & (f[cnt + '_bak'] >= m)]
        parts = {z: f for z in ['pui', 'bak']}
    else:
        parts = {z: f[f[f'{cnt}_{z}'] >= 1] for z in ['pui', 'bak']}
    res = {}
    for z, g in parts.items():
        cols = [f'{v}_{z}' for v in vars_]
        if weighting == 'composite':
            r = g.groupby(['window_id', 'year'])[cols].mean()
        else:  # equal quarter weights; all four quarters required
            qm = g.groupby(['window_id', 'year', 'quarter'])[cols].mean().groupby(level=[0, 1])
            r = qm.mean()[qm.size() == 4]
        r.columns = vars_
        res[z] = r
    idx = res['pui'].index.intersection(res['bak'].index)
    out = res['pui'].loc[idx] - res['bak'].loc[idx]
    n_dates = (f.groupby(['window_id', 'year']).size() if matched else None)
    return out, n_dates


def fe(data, targets, case, extra=None, center=0.0):  # identical to analyze_native_decomposition.fe
    a = standardize(data.reset_index(drop=True), s)
    a['T_background'] = (a.window_id.map(long.set_index('window_id')['T']) - s['T']['mean']) / s['T']['sd'] - center
    a['LAIxT_background'] = a.LAI * a.T_background
    terms = ['UP', 'LAI', 'T', 'P', 'wind', 'LAIxT_background']
    rows = []
    for target in targets:
        z = a[['window_id', 'year', 'spatial_block', target] + terms].dropna().reset_index(drop=True)
        wi = pd.factorize(z.window_id)[0]; yi = pd.factorize(z.year)[0]; bi = pd.factorize(z.spatial_block)[0]
        arr = twoway_demean(z[[target] + terms].to_numpy(), wi, yi)
        fit = sm.OLS(arr[:, 0], arr[:, 1:]).fit()
        cov, _, _ = cov_cluster_2groups(fit, bi, yi, use_correction=True)
        k = len(terms) + z.window_id.nunique() + z.year.nunique() - 1
        cov *= (len(z) - len(terms)) / (len(z) - k)
        se = np.sqrt(np.diag(cov)); df = min(len(np.unique(bi)), len(np.unique(yi))) - 1
        c = tdist.ppf(.975, df)
        for j, term in enumerate(terms):
            rows.append(dict(case=case, target=target, term=term, estimate=fit.params[j], se=se[j],
                             lo95=fit.params[j] - c * se[j], hi95=fit.params[j] + c * se[j],
                             p=2 * tdist.sf(abs(fit.params[j] / se[j]), df), df=df, n_rows=len(z),
                             n_windows=z.window_id.nunique(), n_blocks=z.spatial_block.nunique(),
                             within_r2=fit.rsquared, **(extra or {})))
    return rows


def panel(contr, idx):
    p = contr.reindex(idx).reset_index().merge(context, on=['window_id', 'year'], validate='one_to_one')
    for v in ['day', 'night']:
        if v in p: p[v] = p[v] + .0065 * p.ELEV
    return p


kidx = pd.MultiIndex.from_frame(keys)
specs = {}
for weighting in ['composite', 'quarter']:
    specs[f'J_unmatched_{weighting}'] = annual_means(w, 'LAI_n', ['LAI', 'day', 'night'], False, 1, weighting)
    specs[f'JD_unmatched_{weighting}'] = annual_means(w.rename(columns={'LAID_pui': 'LAIx_pui', 'LAID_bak': 'LAIx_bak'}),
                                                      'LAID_n', ['LAIx', 'dayD'], False, 1, weighting)
    for m in [1, 3, 5]:
        specs[f'J_matched{m}_{weighting}'] = annual_means(w, 'LAI_n', ['LAI', 'day', 'night'], True, m, weighting)
        specs[f'JD_matched{m}_{weighting}'] = annual_means(w.rename(columns={'LAID_pui': 'LAIx_pui', 'LAID_bak': 'LAIx_bak'}),
                                                          'LAID_n', ['LAIx', 'dayD'], True, m, weighting)
for k, (x, _) in list(specs.items()):
    if k.startswith('JD'):
        specs[k] = (x.rename(columns={'LAIx': 'LAI', 'dayD': 'day'}), _)

# Published constructions on the same keys
pub_annual = annual.set_index(['window_id', 'year'])[['LAI', 'LSTd', 'LSTn']].rename(columns={'LSTd': 'day', 'LSTn': 'night'})
qb = pd.read_csv(HERE / 'native_support/quarter_balanced_annual_panel.csv').set_index(['window_id', 'year'])
specs['published_independent'] = (pub_annual, None)
specs['published_concurrent_quarter'] = (qb[['LAI', 'LSTd', 'LSTn']].rename(columns={'LSTd': 'day', 'LSTn': 'night'}), None)

avail = pd.DataFrame({k: kidx.isin(x.dropna().index) for k, (x, _) in specs.items()}, index=kidx)
core = ['published_independent', 'published_concurrent_quarter', 'J_unmatched_quarter', 'J_matched1_quarter',
        'J_unmatched_composite', 'J_matched1_composite', 'JD_unmatched_quarter', 'JD_matched1_quarter']
common = kidx[avail[core].all(axis=1)]
rows = []
for k, (x, _) in specs.items():
    targets = ['day', 'night'] if 'night' in x.columns else ['day']
    rows += fe(panel(x, kidx[avail[k]]), targets, k, {'sample': 'all_defined'})
    rows += fe(panel(x, common), targets, k, {'sample': 'common'})
res = pd.DataFrame(rows)
for smp in ['all_defined', 'common']:
    i = res['sample'].eq(smp)
    res.loc[i, 'p_bh'] = multipletests(res.loc[i, 'p'], method='fdr_bh')[1]
res.to_csv(OUT / 'date_matching_FE.csv', index=False)

# Date-set diagnostics on the identical keys
g = w.set_index(['window_id', 'year'])
both = ((g.LAI_n_pui >= 1) & (g.LAI_n_bak >= 1)).groupby(level=[0, 1]).sum()
either = ((g.LAI_n_pui >= 1) | (g.LAI_n_bak >= 1)).groupby(level=[0, 1]).sum()
near = (g.LAI_n_pui >= 1).groupby(level=[0, 1]).sum(); ref = (g.LAI_n_bak >= 1).groupby(level=[0, 1]).sum()
bothD = ((g.LAID_n_pui >= 1) & (g.LAID_n_bak >= 1)).groupby(level=[0, 1]).sum()
eitherD = ((g.LAID_n_pui >= 1) | (g.LAID_n_bak >= 1)).groupby(level=[0, 1]).sum()
diag = pd.DataFrame({'near_J_dates': near, 'ref_J_dates': ref, 'shared_J_dates': both,
                     'J_date_overlap': both / either, 'shared_JD_dates': bothD, 'JD_date_overlap': bothD / eitherD}).reindex(kidx)
diag.reset_index().to_csv(OUT / 'date_overlap.csv', index=False)
summary = {'anchor_max_abs_count_error': anchor_err, 'identical_keys': len(kidx),
           'common_keys': int(len(common)), 'common_windows': int(common.get_level_values(0).nunique()),
           'defined_window_years': {k: int(v) for k, v in avail.sum().items()},
           'date_overlap_median': {c: float(diag[c].median()) for c in diag.columns},
           'date_overlap_mean': {c: float(diag[c].mean()) for c in diag.columns},
           'source_sha256': hashlib.sha256((OUT / 'dates_all.csv').read_bytes()).hexdigest(),
           'note': 'Date-level zone means average valid cells within each composite date before averaging dates; '
                   'the published concurrent construction averages cell-level quarter means. Matched rules use '
                   'dates with at least m valid cells in both forest portions.'}
(OUT / 'date_matching_provenance.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
pd.set_option('display.width', 220)
print(json.dumps(summary, indent=2))
print(res[res.term.isin(['LAI', 'LAIxT_background'])][['sample', 'case', 'target', 'term', 'estimate', 'lo95', 'hi95', 'n_rows', 'n_windows']].round(3).to_string(index=False))

# ---- Calendar-quarter models on dates matched between portions (m = 1) ----
lat = long.set_index('window_id').lat
qrows = []; qpanels = {}
for q in range(1, 5):
    wq = w[w.quarter.eq(q)]
    for tag, frame, cnt, vars_ in [('J', wq, 'LAI_n', ['LAI', 'day', 'night']),
                                   ('JD', wq.rename(columns={'LAID_pui': 'LAIx_pui', 'LAID_bak': 'LAIx_bak'}), 'LAID_n', ['LAIx', 'dayD'])]:
        x, _ = annual_means(frame, cnt, vars_, True, 1, 'composite')
        if tag == 'JD':
            x = x.rename(columns={'LAIx': 'LAI', 'dayD': 'day'})
        idx = kidx[kidx.isin(x.dropna().index)]
        p = panel(x, idx); qpanels[(tag, q)] = p
        for hemi, sub in [('all', p), ('north', p[p.window_id.map(lat).ge(0)])]:
            qrows += fe(sub, ['day', 'night'] if tag == 'J' else ['day'], f'{tag}_matched1_q{q}_{hemi}',
                        {'quarter': q, 'hemisphere': hemi, 'dates': tag})
qres = pd.DataFrame(qrows)
i = qres.hemisphere.eq('north')
qres.loc[i, 'p_bh'] = multipletests(qres.loc[i, 'p'], method='fdr_bh')[1]
qres.to_csv(OUT / 'date_matching_quarter_FE.csv', index=False)
print(qres[qres.term.eq('LAI') & qres.hemisphere.eq('north')][['case', 'target', 'estimate', 'lo95', 'hi95', 'n_rows', 'n_windows']].round(3).to_string(index=False))

# ---- Conditional LAI slopes at background temperature -1, 0 and +1 primary SD ----
cond = []
for center in [-1.0, 0.0, 1.0]:
    for k in ['J_unmatched_quarter', 'J_matched1_quarter', 'JD_unmatched_quarter', 'JD_matched1_quarter']:
        x = specs[k][0]
        targets = ['day', 'night'] if 'night' in x.columns else ['day']
        cond += [r for r in fe(panel(x, kidx[avail[k]]), targets, k, {'center_sd': center}, center) if r['term'] == 'LAI']
    for (tag, q), p in qpanels.items():
        cond += [r for r in fe(p[p.window_id.map(lat).ge(0)], ['day', 'night'] if tag == 'J' else ['day'],
                               f'{tag}_matched1_q{q}_north', {'center_sd': center, 'quarter': q}, center)
                 if r['term'] == 'LAI']
cond = pd.DataFrame(cond)
cond['T_background_C'] = s['T']['mean'] + cond.center_sd * s['T']['sd']
cond.to_csv(OUT / 'conditional_slopes.csv', index=False)
