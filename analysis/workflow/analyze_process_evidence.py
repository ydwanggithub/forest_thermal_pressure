# -*- coding: utf-8 -*-
"""Radiative and snow evidence for the seasonal canopy-temperature associations.

Uses the Northern Hemisphere full-cohort quarterly panels of Supplementary Table S8
(independent quarterly means; portion-quarters with >= 5 valid cells for LAI and both
overpasses; windows with >= 20 qualifying years) and adds, from native_process:
  dAlb    near-minus-reference white-sky shortwave albedo (percentage points)
  dSWabs  -SW x dAlb, the absorbed shortwave contrast implied by the albedo contrast (W m-2)
  S       ERA5-Land snow cover of the window-quarter (percent; standardized within quarter)
  dLAIsf  LAI contrast from retrievals without the snow/ice flag (primary LAI scale)
All models are the no-ET within-window models of Table S8 (window and year fixed effects,
two-way block/year clustered covariance, t22) with optional added terms. Conditional LAI
slopes are reported at background temperature -1, 0 and +1 primary SD.
Outputs native_process/process_FE.csv, process_panel.csv and process_provenance.json.
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
OUT = HERE / 'native_process'
sys.path.insert(0, str(ROOT / 'analysis/pipeline'))
from analyze import twoway_demean, standardize

long = pd.read_csv(HERE / 'native_analysis/longterm_windows.csv')
annual = pd.read_csv(HERE / 'native_analysis/annual_panel.csv')
annual = annual[annual.window_id.isin(long.window_id)].copy()
s = json.loads((HERE / 'native_analysis/scaling.json').read_text())['primary']
d = pd.read_csv(HERE / 'native_decomposition/support_all.csv')
pr = pd.read_csv(OUT / 'process_all.csv')
assert len(pr) == len(d) and not pr.duplicated(['window_id', 'zone', 'year']).any()
# Anchor: the process extraction reproduces the quarterly LAI of the decomposition extraction
m = pr.merge(d[['window_id', 'zone', 'year'] + [f'LAI_q{q}_ind_mean' for q in range(1, 5)] + [f'LAI_q{q}_ind_count' for q in range(1, 5)]],
             on=['window_id', 'zone', 'year'], validate='one_to_one')
anchor = {f'q{q}': float(np.nanmax((m[f'LAI_q{q}_mean'] - m[f'LAI_q{q}_ind_mean']).abs())) for q in range(1, 5)}
assert max(anchor.values()) < 1e-9
assert all((m[f'LAI_q{q}_count'] == m[f'LAI_q{q}_ind_count']).all() for q in range(1, 5))

n = d[d.zone.eq('pui')].set_index(['window_id', 'year']); b = d[d.zone.eq('bak')].set_index(['window_id', 'year'])
pn = pr[pr.zone.eq('pui')].set_index(['window_id', 'year']).reindex(n.index)
pb = pr[pr.zone.eq('bak')].set_index(['window_id', 'year']).reindex(n.index)
context = annual.drop(columns=['LAI', 'day', 'night', 'LSTd', 'LSTn', 'mean', 'day_minus_night'])
lat = long.set_index('window_id').lat
Tb = (long.set_index('window_id')['T'] - s['T']['mean']) / s['T']['sd']


def window_level(col):
    """ERA5-Land variables: mean of the portions with land values (coastal sea cells are 0 or missing)."""
    a = pn[col + '_mean'].where(pn[col.replace('sw', 'snowera') + '_mean'].notna()) if col.startswith('sw') else pn[col + '_mean']
    c = pb[col + '_mean'].where(pb[col.replace('sw', 'snowera') + '_mean'].notna()) if col.startswith('sw') else pb[col + '_mean']
    return pd.concat([a, c], axis=1).mean(axis=1)


def quarter_panel(q):
    ok = pd.Series(True, index=n.index)
    for v in ['LAI', 'day', 'night']:
        ok &= (n[f'{v}_q{q}_ind_count'] >= 5) & (b[f'{v}_q{q}_ind_count'] >= 5)
    prim = pd.MultiIndex.from_frame(annual[['window_id', 'year']])
    okk = ok[ok].index.intersection(prim)
    yrs = pd.Series(okk.get_level_values(0)).value_counts()
    keep = okk[pd.Index(okk.get_level_values(0)).isin(yrs[yrs >= 20].index)]
    cols = {v: n[f'{v}_q{q}_ind_mean'] - b[f'{v}_q{q}_ind_mean'] for v in ['LAI', 'day', 'night']}
    alb_ok = (pn[f'alb_q{q}_count'] >= 5) & (pb[f'alb_q{q}_count'] >= 5)
    sf_ok = (pn[f'LAIsf_q{q}_count'] >= 5) & (pb[f'LAIsf_q{q}_count'] >= 5)
    cols['dAlb'] = (100 * (pn[f'alb_q{q}_mean'] - pb[f'alb_q{q}_mean'])).where(alb_ok)
    cols['SW'] = window_level(f'sw_q{q}')
    cols['dSWabs'] = -cols['SW'] * cols['dAlb'] / 100
    cols['S_raw'] = window_level(f'snowera_q{q}')
    cols['dSnow'] = 100 * (pn[f'snowf_q{q}_mean'] - pb[f'snowf_q{q}_mean'])
    cols['LAIsf_raw'] = (pn[f'LAIsf_q{q}_mean'] - pb[f'LAIsf_q{q}_mean']).where(sf_ok)
    cols['LAIsnow'] = (pn[f'LAIsnow_q{q}_mean'] + pb[f'LAIsnow_q{q}_mean']) / 2
    p = pd.DataFrame({k: v.reindex(keep).to_numpy() for k, v in cols.items()}, index=keep).reset_index()
    p = p.merge(context, on=['window_id', 'year'], validate='one_to_one')
    for v in ['day', 'night']:
        p[v] = p[v] + .0065 * p.ELEV
    p['quarter'] = q
    return p[p.window_id.map(lat).ge(0)].reset_index(drop=True)


def fe(data, target, case, extra_terms=(), lai_col='LAI', moderators=('T_background',), meta=None):
    a = standardize(data.reset_index(drop=True), s)
    if lai_col != 'LAI':
        a['LAI'] = (data[lai_col].to_numpy() - s['LAI']['mean']) / s['LAI']['sd']
    a['T_background'] = a.window_id.map(Tb)
    if 'S' in moderators or 'S' in extra_terms:
        sv = a['S_raw']; s_mean, s_sd = sv.mean(), sv.std(ddof=1); a['S'] = (sv - s_mean) / s_sd
    for mdr in moderators:
        a['LAIx' + mdr] = a.LAI * a[mdr]
    terms = ['UP', 'LAI', 'T', 'P', 'wind'] + ['LAIx' + mdr for mdr in moderators] + list(extra_terms)
    z = a[['window_id', 'year', 'spatial_block', target] + terms].dropna().reset_index(drop=True)
    wi = pd.factorize(z.window_id)[0]; yi = pd.factorize(z.year)[0]; bi = pd.factorize(z.spatial_block)[0]
    arr = twoway_demean(z[[target] + terms].to_numpy(), wi, yi)
    fit = sm.OLS(arr[:, 0], arr[:, 1:]).fit()
    cov, _, _ = cov_cluster_2groups(fit, bi, yi, use_correction=True)
    k = len(terms) + z.window_id.nunique() + z.year.nunique() - 1
    cov *= (len(z) - len(terms)) / (len(z) - k)
    se = np.sqrt(np.diag(cov)); df = min(len(np.unique(bi)), len(np.unique(yi))) - 1
    c = tdist.ppf(.975, df)
    base = dict(case=case, target=target, df=df, n_rows=len(z), n_windows=z.window_id.nunique(), within_r2=fit.rsquared, **(meta or {}))
    rows = [dict(term=t, estimate=fit.params[j], se=se[j], lo95=fit.params[j] - c * se[j], hi95=fit.params[j] + c * se[j],
                 p=2 * tdist.sf(abs(fit.params[j] / se[j]), df), **base) for j, t in enumerate(terms)]
    jl = terms.index('LAI'); jt = terms.index('LAIxT_background')
    for cval, label in [(-1.0, 'cool'), (0.0, 'mean'), (1.0, 'warm')]:
        g = np.zeros(len(terms)); g[jl] = 1; g[jt] = cval
        est = g @ fit.params; sev = np.sqrt(g @ cov @ g)
        rows.append(dict(term=f'LAI_at_{label}', estimate=est, se=sev, lo95=est - c * sev, hi95=est + c * sev,
                         p=2 * tdist.sf(abs(est / sev), df), **base))
    if 'S' in moderators:  # LAI coefficient over raw snow cover (percent) at the three background temperatures
        js = terms.index('LAIxS')
        for cval, label in [(-1.0, 'cool'), (0.0, 'mean'), (1.0, 'warm')]:
            for raw in np.arange(0, 100.01, 2.5):
                g = np.zeros(len(terms)); g[jl] = 1; g[jt] = cval; g[js] = (raw - s_mean) / s_sd
                est = g @ fit.params; sev = np.sqrt(g @ cov @ g)
                rows.append(dict(term=f'LAI_at_{label}_snow{raw:g}', estimate=est, se=sev, lo95=est - c * sev, hi95=est + c * sev,
                                 p=2 * tdist.sf(abs(est / sev), df), **base))
    return rows, z


s8 = pd.read_csv(HERE / 'native_decomposition/table_S8.csv')
rows, panels, samples = [], [], {}
for q in range(1, 5):
    p = quarter_panel(q); panels.append(p)
    meta = {'quarter': q}
    # anchor: the Table S8 full-cohort day and night models
    for target in ['day', 'night']:
        r, _ = fe(p, target, f'q{q}_baseline_full', meta=meta)
        est = [x for x in r if x['term'] == 'LAI'][0]['estimate']
        ref = s8[s8.source.eq('full') & s8.target.eq(target) & s8.quarter.eq(q)].LAI.iloc[0]
        assert abs(est - ref) < 1e-9, (q, target, est, ref)
        rows += r
    pa = p.dropna(subset=['dAlb', 'dSWabs'])
    samples[f'q{q}'] = {'full': int(len(p)), 'albedo': int(len(pa)), 'windows_albedo': int(pa.window_id.nunique())}
    # 1. canopy and albedo, snow
    r, _ = fe(pa, 'dAlb', f'q{q}_albedo_response', meta=meta); rows += r
    r, _ = fe(p, 'dSnow', f'q{q}_snow_response', meta=meta); rows += r
    # 2. daytime temperature on the albedo sample, without and with the absorbed shortwave contrast
    r, _ = fe(pa, 'day', f'q{q}_day_albedo_sample', meta=meta); rows += r
    r, _ = fe(pa, 'day', f'q{q}_day_plus_SWabs', extra_terms=('dSWabs',), meta=meta); rows += r
    r, _ = fe(pa, 'dSWabs', f'q{q}_SWabs_response', meta=meta); rows += r
    # 3. snow moderation and snow-free LAI
    r, _ = fe(p.dropna(subset=['S_raw']), 'day', f'q{q}_day_snow_sample', meta=meta); rows += r
    r, _ = fe(p, 'day', f'q{q}_day_snow_moderation', extra_terms=('S',), moderators=('T_background', 'S'), meta=meta); rows += r
    r, _ = fe(p.dropna(subset=['LAIsf_raw']), 'day', f'q{q}_day_snowfree_sample', meta=meta); rows += r
    r, _ = fe(p.dropna(subset=['LAIsf_raw']), 'day', f'q{q}_day_snowfree_LAI', lai_col='LAIsf_raw', meta=meta); rows += r
    ps = p.dropna(subset=['S_raw'])
    samples[f'q{q}'].update(snow_rows=int(len(ps)), snow_mean=float(ps.S_raw.mean()), snow_sd=float(ps.S_raw.std(ddof=1)),
                            snowfree_rows=int(p.LAIsf_raw.notna().sum()))
res = pd.DataFrame(rows)
fam = res.term.isin(['LAI', 'LAI_at_cool', 'LAI_at_warm', 'dSWabs', 'LAIxS']) & ~res.case.str.contains('baseline_full')
res.loc[fam, 'p_bh'] = multipletests(res.loc[fam, 'p'], method='fdr_bh')[1]
res.to_csv(OUT / 'process_FE.csv', index=False)
pd.concat(panels, ignore_index=True).to_csv(OUT / 'process_panel.csv', index=False)
desc = pd.concat(panels, ignore_index=True)
desc['Tgroup'] = pd.cut(desc.window_id.map(Tb), [-np.inf, -0.5, 0.5, np.inf], labels=['cool', 'middle', 'warm'])
summary = desc.groupby(['quarter', 'Tgroup'], observed=True).agg(S_raw=('S_raw', 'mean'), S_p05=('S_raw', lambda v: v.quantile(.05)),
                                                                   S_p50=('S_raw', 'median'), S_p95=('S_raw', lambda v: v.quantile(.95)),
                                                                   LAIsnow=('LAIsnow', 'mean'),
                                                                   SW=('SW', 'mean'), dAlb_sd=('dAlb', 'std'), n=('day', 'size')).round(4)
summary.reset_index().to_csv(OUT / 'process_descriptives.csv', index=False)
(OUT / 'process_provenance.json').write_text(json.dumps({
    'anchors': {'LAI_reextraction_max_abs': anchor, 'table_S8_reproduced': True},
    'samples': samples,
    'definitions': {'dAlb': 'near minus reference white-sky shortwave albedo, percentage points',
                    'dSWabs': '-SW x dAlb, W m-2; SW is ERA5-Land downward solar radiation of the window-quarter',
                    'S': 'ERA5-Land snow cover of the window-quarter, standardized within the quarter sample',
                    'dSnow': 'near minus reference MOD10A2 snow frequency (share of valid 8-day periods with snow), percentage points',
                    'Tgroup': 'background temperature below -0.5, within +/-0.5 and above +0.5 primary SD (descriptives only)',
                    'LAIsf': 'LAI contrast from good retrievals without the snow/ice flag, primary LAI scale'},
    'BH_family': 'LAI, conditional LAI, dSWabs and LAI x S coefficients of all non-anchor models',
    'sources_sha256': {'process_all.csv': hashlib.sha256((OUT / 'process_all.csv').read_bytes()).hexdigest(),
                       'support_all.csv': hashlib.sha256((HERE / 'native_decomposition/support_all.csv').read_bytes()).hexdigest()}},
    indent=2), encoding='utf-8')
pd.set_option('display.width', 230)
show = res[res.term.isin(['LAI_at_cool', 'LAI_at_mean', 'LAI_at_warm', 'dSWabs', 'LAIxS', 'S']) & ~res.case.str.contains('baseline_full')]
print(show[['case', 'term', 'estimate', 'lo95', 'hi95', 'p_bh', 'n_rows', 'n_windows']].round(4).to_string(index=False))
print(json.dumps(samples))
print(summary.to_string())
