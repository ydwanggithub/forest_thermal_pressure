"""Separate the components of the annual observation-construction sensitivity.

Input: native_decomposition/support_all.csv from fetch_native_support_decomposition.py.
All models use the specification, scales, covariates, fixed effects and
inference of analyze_native_observation_support.py (no modeled ET).

Constructions evaluated on exactly the same 960 windows and 21,386 window-years:
  A  independent annual means, variable-specific valid cells   (published "annual means")
  B  independent annual means, shared quarter-balanced cells    (cell set)
  C  independent quarterly means, equal quarter weights, B cells (+ quarter weighting)
  D  concurrent composites, composite-weighted, B cells          (+ date concurrence)
  E  concurrent composites, equal quarter weights, B cells       (published "quarter equal")
Daytime only (nighttime validity not required for dates or cells):
  F  LAI-day concurrent composites, equal quarter weights, B cells
  G  LAI-day concurrent composites, equal quarter weights, LAI-day balanced cells
Calendar-quarter models use the concurrent quarter means of E (and independent
quarterly means on B cells) as single-quarter responses and predictors.
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
OUT = HERE / 'native_decomposition'
sys.path.insert(0, str(ROOT / 'analysis/pipeline'))
from analyze import twoway_demean, standardize

long = pd.read_csv(HERE / 'native_analysis/longterm_windows.csv')
annual = pd.read_csv(HERE / 'native_analysis/annual_panel.csv')
annual = annual[annual.window_id.isin(long.window_id)].copy()
s = json.loads((HERE / 'native_analysis/scaling.json').read_text())['primary']
pub = pd.read_csv(HERE / 'native_sensitivity/observation_support_FE_sensitivity.csv')
qb_keys = pd.read_csv(HERE / 'native_support/quarter_balanced_annual_panel.csv', usecols=['window_id', 'year'])
old = pd.read_csv(HERE / 'native_support/support_all.csv')
d = pd.read_csv(OUT / 'support_all.csv')
assert len(d) == 4208 * 23 and not d.duplicated(['window_id', 'zone', 'year']).any()

# ---- 1. Anchors: the new extraction must reproduce the published inputs ----
anchor = []
m = d.merge(old[old.zone.isin(['pui', 'bak'])], on=['window_id', 'zone', 'year'], suffixes=('', '_old'), validate='one_to_one')
assert len(m) == len(d)
for v in ['LAI', 'day', 'night']:
    e = (m[v + '_orig_mean'] - m[v + '_original_mean']).abs()
    qe = sum(m[f'{v}_q{q}_J_bal_mean'] for q in range(1, 5)) / 4
    e2 = (qe - m[v + '_balanced_mean']).abs()
    anchor.append(dict(variable=v, independent_max_abs=float(e.max()),
                       independent_count_identical=bool((m[v + '_orig_count'] == m[v + '_original_count']).all()),
                       quarter_equal_max_abs=float(np.nanmax(e2)),
                       quarter_equal_missing_identical=bool((qe.isna() == m[v + '_balanced_mean'].isna()).all())))
    assert e.max() < 1e-9 and np.nanmax(e2) < 1e-9 and anchor[-1]['independent_count_identical']
assert (m.mbal_count == m.LAI_balanced_count).all()
for c in [c for c in d.columns if c.endswith('_bal_count')]:
    assert (d[c] == d.mbal_count).all(), c
for c in [c for c in d.columns if c.endswith('_balD_count')]:
    assert (d[c] == d.mbalD_count).all(), c
assert (d.mbalD_count >= d.mbal_count).all()

# ---- 2. Near-minus-reference contrasts for each construction ----
n = d[d.zone.eq('pui')].set_index(['window_id', 'year']); b = d[d.zone.eq('bak')].set_index(['window_id', 'year'])
assert n.index.equals(b.index)
def contrast(col):
    return n[col + '_mean'] - b[col + '_mean']
def qeq(fmt):  # zone means over a fixed all-valid cell mask are linear: quarter-equal of means
    return sum(contrast(fmt.format(q=q)) for q in range(1, 5)) / 4
cons = {
    'A': {v: contrast(f'{v}_orig') for v in ['LAI', 'day', 'night']},
    'B': {v: contrast(f'{v}_ind_bal') for v in ['LAI', 'day', 'night']},
    'C': {v: qeq(v + '_q{q}_ind_bal') for v in ['LAI', 'day', 'night']},
    'D': {v: contrast(f'{v}_Jc_bal') for v in ['LAI', 'day', 'night']},
    'E': {v: qeq(v + '_q{q}_J_bal') for v in ['LAI', 'day', 'night']},
    'F': {v: qeq(v + '_q{q}_JD_bal') for v in ['LAI', 'day']},
    'G': {v: qeq(v + '_q{q}_JD_balD') for v in ['LAI', 'day']},
}
labels = {'A': 'Independent annual means, variable cells',
          'B': 'Independent annual means, balanced cells',
          'C': 'Independent quarter means, equal weights, balanced cells',
          'D': 'Concurrent composites, composite weights, balanced cells',
          'E': 'Concurrent composites, equal quarter weights, balanced cells',
          'F': 'Day-LAI concurrent, equal quarter weights, balanced cells',
          'G': 'Day-LAI concurrent, equal quarter weights, day-balanced cells'}
keys = pd.MultiIndex.from_frame(qb_keys)
assert len(keys) == 21386 and qb_keys.window_id.nunique() == 960
eligible = (n.mbal_count >= 5) & (b.mbal_count >= 5)
assert eligible.reindex(keys).all()
context = annual.drop(columns=['LAI', 'day', 'night', 'LSTd', 'LSTn', 'mean', 'day_minus_night'])

def panel(construction, idx):
    p = pd.DataFrame({v: x.reindex(idx).to_numpy() for v, x in construction.items()}, index=idx).reset_index()
    p = p.merge(context, on=['window_id', 'year'], validate='one_to_one')
    for v in ['day', 'night']:
        if v in p: p[v] = p[v] + .0065 * p.ELEV
    return p

def fe(data, targets, case, extra=None, center=0.0):
    # center shifts fixed background temperature (in primary SD); the LAI coefficient is then
    # the conditional slope at that background temperature. center=0 is the reported model.
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
        assert np.diag(cov).min() > 0
        se = np.sqrt(np.diag(cov)); df = min(len(np.unique(bi)), len(np.unique(yi))) - 1
        c = tdist.ppf(.975, df)
        for j, term in enumerate(terms):
            rows.append(dict(case=case, target=target, term=term, estimate=fit.params[j], se=se[j],
                             lo95=fit.params[j] - c * se[j], hi95=fit.params[j] + c * se[j],
                             p=2 * tdist.sf(abs(fit.params[j] / se[j]), df), df=df, n_rows=len(z),
                             n_windows=z.window_id.nunique(), n_blocks=z.spatial_block.nunique(),
                             within_r2=fit.rsquared, **(extra or {})))
    return rows

# ---- 3. Factorial decomposition on identical window-years ----
rows = []
for k, cdict in cons.items():
    targets = ['day', 'night'] if 'night' in cdict else ['day']
    rows += fe(panel(cdict, keys), targets, k, {'construction': labels[k]})
dec = pd.DataFrame(rows)
dec['p_bh'] = multipletests(dec.p, method='fdr_bh')[1]
# Reproduce the Supplementary Table S5 comparison exactly (A = original_same_sample, E = concurrent_quarter_balanced)
for ours, theirs in [('A', 'original_same_sample'), ('E', 'concurrent_quarter_balanced')]:
    x = dec[dec.case.eq(ours)].set_index(['target', 'term'])
    y = pub[pub.case.eq(theirs)].set_index(['target', 'term'])
    for col in ['estimate', 'se']:
        err = (x[col] - y.loc[x.index, col]).abs().max()
        assert err < 1e-9, (ours, col, err)
dec.to_csv(OUT / 'decomposition_FE.csv', index=False)

# ---- 4. Calendar-quarter models on the same window-years ----
lat = long.set_index('window_id').lat
season_rows = []
for source, fmt, tag in [('concurrent', '{v}_q{q}_J_bal', 'J'), ('independent', '{v}_q{q}_ind_bal', 'ind')]:
    for q in range(1, 5):
        cdict = {v: contrast(fmt.format(v=v, q=q)) for v in ['LAI', 'day', 'night']}
        p = panel(cdict, keys)
        for hemi, sub in [('all', p), ('north', p[p.window_id.map(lat).ge(0)])]:
            season_rows += fe(sub, ['day', 'night'], f'{tag}_q{q}_{hemi}',
                              {'source': source, 'quarter': q, 'hemisphere': hemi})
sea = pd.DataFrame(season_rows)
sea['p_bh'] = np.nan
for src in ['concurrent', 'independent']:
    i = sea.source.eq(src) & sea.hemisphere.eq('north')
    sea.loc[i, 'p_bh'] = multipletests(sea.loc[i, 'p'], method='fdr_bh')[1]
sea.to_csv(OUT / 'quarter_FE.csv', index=False)

# ---- 5. Descriptive seasonal contrasts and effective quarter weights ----
desc = []
pq = {}
for q in range(1, 5):
    x = panel({v: contrast(f'{v}_q{q}_J_bal') for v in ['LAI', 'day', 'night']}, keys)
    x = x[x.window_id.map(lat).ge(0)]
    pq[q] = x
    for v in ['LAI', 'day', 'night']:
        desc.append(dict(quarter=q, variable=v, hemisphere='north', mean=x[v].mean(), sd=x[v].std(ddof=1),
                         within_window_sd=(x[v] - x.groupby('window_id')[v].transform('mean')).std(ddof=1)))
# Quarter shares of valid composites inside the shared balanced cells (descriptive)
pd.DataFrame(desc).to_csv(OUT / 'quarter_descriptives.csv', index=False)

# Archive the analysis panel (all constructions on identical keys)
arch = qb_keys.copy()
for k, cdict in cons.items():
    for v, x in cdict.items():
        arch[f'{k}_{v}'] = x.reindex(keys).to_numpy()
arch.to_csv(OUT / 'decomposition_panel.csv', index=False)

cells = pd.DataFrame({'near_balanced': n.mbal_count.reindex(keys).to_numpy(),
                      'ref_balanced': b.mbal_count.reindex(keys).to_numpy(),
                      'near_day_balanced': n.mbalD_count.reindex(keys).to_numpy(),
                      'ref_day_balanced': b.mbalD_count.reindex(keys).to_numpy(),
                      'near_cells': n.local_n.reindex(keys).to_numpy(),
                      'ref_cells': b.local_n.reindex(keys).to_numpy()})
# Day-only eligibility over the whole primary frame (>=5 day-balanced cells per side, >=20 years)
el = ((n.mbalD_count >= 5) & (b.mbalD_count >= 5)).rename('ok').reset_index()
el = el[el.window_id.isin(long.window_id) & el.ok]
yrs = el.groupby('window_id').year.nunique()
dayonly_ids = yrs[yrs >= 20].index
cell_summary = {
    'balanced_cells_per_portion_mean': float(cells[['near_balanced', 'ref_balanced']].stack().mean()),
    'day_balanced_cells_per_portion_mean': float(cells[['near_day_balanced', 'ref_day_balanced']].stack().mean()),
    'mapped_cells_per_portion_mean': float(cells[['near_cells', 'ref_cells']].stack().mean()),
    'day_only_eligible_windows_primary_frame': int(len(dayonly_ids)),
    'day_only_eligible_window_years': int(el[el.window_id.isin(dayonly_ids)].shape[0]),
}

# ---- 6. Calendar-quarter models on the full primary cohort (independent quarter means) ----
# Each quarter uses each variable's own valid cells; a portion-quarter requires >= 5 valid
# cells per variable, and a window >= 20 qualifying years (the primary-cohort rule).
prim = pd.MultiIndex.from_frame(annual[['window_id', 'year']])
full_rows = []; full_n = []; full_panels = {}
for q in range(1, 5):
    ok = pd.Series(True, index=n.index)
    for v in ['LAI', 'day', 'night']:
        ok &= (n[f'{v}_q{q}_ind_count'] >= 5) & (b[f'{v}_q{q}_ind_count'] >= 5)
    okk = ok[ok].index.intersection(prim)
    yrs = pd.Series(okk.get_level_values(0)).value_counts()
    keep = okk[pd.Index(okk.get_level_values(0)).isin(yrs[yrs >= 20].index)]
    cdict = {v: contrast(f'{v}_q{q}_ind') for v in ['LAI', 'day', 'night']}
    p = panel(cdict, keep); full_panels[q] = p
    for hemi, sub in [('all', p), ('north', p[p.window_id.map(lat).ge(0)])]:
        full_rows += fe(sub, ['day', 'night'], f'full_ind_q{q}_{hemi}', {'quarter': q, 'hemisphere': hemi})
    full_n.append(dict(quarter=q, window_years=len(keep), windows=int(pd.Index(keep.get_level_values(0)).nunique())))
full = pd.DataFrame(full_rows)
i = full.hemisphere.eq('north')
full.loc[i, 'p_bh'] = multipletests(full.loc[i, 'p'], method='fdr_bh')[1]
full.to_csv(OUT / 'quarter_FE_full_cohort.csv', index=False)
pd.DataFrame(full_n).to_csv(OUT / 'quarter_full_cohort_samples.csv', index=False)

# ---- 7. Conditional LAI slopes at cooler and warmer background temperature ----
# Refits with background temperature centred at -1, 0 and +1 primary SD (about 6.5, 12.7
# and 19.0 °C) give the LAI slope and its clustered interval at each temperature.
noet = pd.DataFrame(fe(annual, ['day', 'night'], 'primary_noET')).set_index(['target', 'term'])
pub_noet = pd.read_csv(HERE / 'native_sensitivity/within_window_without_ET.csv')
pub_noet = pub_noet[pub_noet.case.eq('original')].set_index(['target', 'term'])
assert (noet.estimate - pub_noet.loc[noet.index, 'estimate']).abs().max() < 1e-9
cond = []
for center in [-1.0, 0.0, 1.0]:
    def keep_lai(rows):
        return [r for r in rows if r['term'] == 'LAI']
    cond += keep_lai(fe(annual, ['day', 'night'], 'primary_noET', {'center_sd': center}, center))
    for k in ['A', 'E']:
        cond += keep_lai(fe(panel(cons[k], keys), ['day', 'night'], k, {'center_sd': center}, center))
    for q in range(1, 5):
        p = full_panels[q]
        cond += keep_lai(fe(p[p.window_id.map(lat).ge(0)], ['day', 'night'], f'full_ind_q{q}_north',
                            {'center_sd': center, 'quarter': q}, center))
cond = pd.DataFrame(cond)
cond['T_background_C'] = s['T']['mean'] + cond.center_sd * s['T']['sd']
cond.to_csv(OUT / 'conditional_slopes.csv', index=False)

key = dec[dec.term.isin(['LAI', 'LAIxT_background'])][['case', 'target', 'term', 'estimate', 'lo95', 'hi95', 'p_bh']]
report = {'status': 'passed', 'anchors': anchor, 'identical_keys': {'windows': 960, 'window_years': 21386},
          'table_S5_reproduction': 'A and E reproduce Table S5 estimates and standard errors to < 1e-9',
          'cells': cell_summary,
          'BH_families': {'decomposition': f'all {len(dec)} coefficients in decomposition_FE.csv',
                          'quarter': 'northern-hemisphere quarter models, separately for concurrent and independent quarter means'},
          'limits': 'Composite-level concurrence within cells; not matched daily acquisitions or dates between forest portions; '
                    'single-quarter covariates are annual; quarter models describe within-window annual covariation of one quarter.',
          'source_sha256': hashlib.sha256((OUT / 'support_all.csv').read_bytes()).hexdigest()}
(OUT / 'decomposition_provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
pd.set_option('display.width', 200)
print(json.dumps(report, indent=2))
print(key.to_string(index=False))
print(sea[sea.term.isin(['LAI', 'LAIxT_background'])][['case', 'target', 'term', 'estimate', 'lo95', 'hi95', 'p', 'n_rows', 'n_windows']].to_string(index=False))
print(pd.DataFrame(desc).to_string(index=False))
print(pd.DataFrame(full_n).to_string(index=False))
print(full[full.term.isin(['LAI', 'LAIxT_background'])][['case', 'target', 'term', 'estimate', 'lo95', 'hi95', 'p', 'n_rows', 'n_windows']].round(3).to_string(index=False))
