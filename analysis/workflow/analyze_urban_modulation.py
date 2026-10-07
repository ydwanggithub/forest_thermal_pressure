# -*- coding: utf-8 -*-
"""Does urban pressure modulate the within-window canopy-temperature association?

Adds LAI x fixed long-term urban pressure contrast (UP_background, standardized by the
primary UP scale) to the no-ET within-window models, alongside the existing
LAI x fixed long-term background temperature term:
  annual: primary cohort, independent annual means (2,096 windows);
  quarterly: Northern Hemisphere full cohort, independent quarterly means (Table S8 rows).
A secondary specification interacts LAI with annual UP (time-varying).
Anchors: without the new term, estimates reproduce Table S4 and Table S8 exactly.
Outputs native_decomposition/urban_modulation_FE.csv and urban_modulation_provenance.json.
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
d = pd.read_csv(OUT / 'support_all.csv')
n = d[d.zone.eq('pui')].set_index(['window_id', 'year']); b = d[d.zone.eq('bak')].set_index(['window_id', 'year'])
assert n.index.equals(b.index)
context = annual.drop(columns=['LAI', 'day', 'night', 'LSTd', 'LSTn', 'mean', 'day_minus_night'])
lat = long.set_index('window_id').lat
Tb = (long.set_index('window_id')['T'] - s['T']['mean']) / s['T']['sd']
Ub = (long.set_index('window_id')['UP'] - s['UP']['mean']) / s['UP']['sd']


def contrast(col):
    return n[col + '_mean'] - b[col + '_mean']


def panel(cdict, idx):
    p = pd.DataFrame({v: x.reindex(idx).to_numpy() for v, x in cdict.items()}, index=idx).reset_index()
    p = p.merge(context, on=['window_id', 'year'], validate='one_to_one')
    for v in ['day', 'night']:
        p[v] = p[v] + .0065 * p.ELEV
    return p


def fe(data, targets, case, moderators, extra=None):
    """No-ET within-window model; moderators is a list of 'T_background', 'UP_background', 'UP_annual'."""
    a = standardize(data.reset_index(drop=True), s)
    a['T_background'] = a.window_id.map(Tb)
    a['UP_background'] = a.window_id.map(Ub)
    a['UP_annual'] = a['UP']
    for m in moderators:
        a['LAIx' + m] = a.LAI * a[m]
    terms = ['UP', 'LAI', 'T', 'P', 'wind'] + ['LAIx' + m for m in moderators]
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
                             n_windows=z.window_id.nunique(), **(extra or {})))
        # conditional LAI slopes at UP_background -1 and +1 SD (other moderators at 0)
        if 'UP_background' in moderators:
            jl = terms.index('LAI'); ju = terms.index('LAIxUP_background')
            for cval in [-1.0, 1.0]:
                g = np.zeros(len(terms)); g[jl] = 1; g[ju] = cval
                est = g @ fit.params; sev = np.sqrt(g @ cov @ g)
                rows.append(dict(case=case, target=target, term=f'LAI_at_UP_{"low" if cval < 0 else "high"}', estimate=est, se=sev,
                                 lo95=est - c * sev, hi95=est + c * sev, p=2 * tdist.sf(abs(est / sev), df), df=df,
                                 n_rows=len(z), n_windows=z.window_id.nunique(), **(extra or {})))
    return rows


# ---- anchors: the reported models are reproduced exactly ----
anchor = pd.DataFrame(fe(annual, ['day', 'night'], 'primary_noET', ['T_background']))
s4 = pd.read_csv(HERE / 'native_sensitivity/within_window_without_ET.csv'); s4 = s4[s4.case.eq('original')]
for target in ['day', 'night']:
    for term_new, term_old in [('LAI', 'LAI'), ('LAIxT_background', 'LAIxT_background')]:
        x = anchor[anchor.target.eq(target) & anchor.term.eq(term_new)].estimate.iloc[0]
        y = s4[s4.target.eq(target) & s4.term.eq(term_old)].estimate.iloc[0]
        assert abs(x - y) < 1e-9, (target, term_new, x, y)
prim = pd.MultiIndex.from_frame(annual[['window_id', 'year']])
qpanels = {}
s8 = pd.read_csv(OUT / 'table_S8.csv')
for q in range(1, 5):
    ok = pd.Series(True, index=n.index)
    for v in ['LAI', 'day', 'night']:
        ok &= (n[f'{v}_q{q}_ind_count'] >= 5) & (b[f'{v}_q{q}_ind_count'] >= 5)
    okk = ok[ok].index.intersection(prim)
    yrs = pd.Series(okk.get_level_values(0)).value_counts()
    keep = okk[pd.Index(okk.get_level_values(0)).isin(yrs[yrs >= 20].index)]
    p = panel({v: contrast(f'{v}_q{q}_ind') for v in ['LAI', 'day', 'night']}, keep)
    p = p[p.window_id.map(lat).ge(0)]
    qpanels[q] = p
    chk = pd.DataFrame(fe(p, ['day', 'night'], 'anchor', ['T_background']))
    for target in ['day', 'night']:
        x = chk[chk.target.eq(target) & chk.term.eq('LAI')].estimate.iloc[0]
        y = s8[s8.source.eq('full') & s8.target.eq(target) & s8.quarter.eq(q)].LAI.iloc[0]
        assert abs(x - y) < 1e-9, (q, target, x, y)

# ---- urban modulation ----
rows = []
rows += fe(annual, ['day', 'night'], 'annual_UPbackground', ['T_background', 'UP_background'], {'period': 'annual'})
rows += fe(annual, ['day', 'night'], 'annual_UPannual', ['T_background', 'UP_annual'], {'period': 'annual'})
for q, p in qpanels.items():
    rows += fe(p, ['day', 'night'], f'q{q}_UPbackground', ['T_background', 'UP_background'], {'period': f'q{q}'})
res = pd.DataFrame(rows)
fam = res.term.isin(['LAIxUP_background', 'LAIxUP_annual'])
res.loc[fam, 'p_bh'] = multipletests(res.loc[fam, 'p'], method='fdr_bh')[1]
res.to_csv(OUT / 'urban_modulation_FE.csv', index=False)
(OUT / 'urban_modulation_provenance.json').write_text(json.dumps({
    'anchors': 'Without the urban term, annual and quarterly estimates reproduce Tables S4 and S8 to < 1e-9',
    'UP_background': 'window long-term near-minus-reference UP, standardized by the primary pooled UP mean and SD',
    'UP_background_SD': float(s['UP']['sd']), 'corr_UPbackground_Tbackground': float(np.corrcoef(Ub.reindex(long.window_id), Tb.reindex(long.window_id))[0, 1]),
    'BH_family': 'all LAI x UP interaction coefficients in urban_modulation_FE.csv',
    'source_sha256': hashlib.sha256((OUT / 'support_all.csv').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
pd.set_option('display.width', 220)
show = res[res.term.str.contains('LAI')]
print(show[['case', 'target', 'term', 'estimate', 'lo95', 'hi95', 'p', 'p_bh', 'n_windows']].round(4).to_string(index=False))
