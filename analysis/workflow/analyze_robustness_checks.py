# -*- coding: utf-8 -*-
"""Robustness checks and figure data for the within-window models.

All models are the no-ET within-window models of Supplementary Tables S4 and S8 (window and year
fixed effects, two-way spatial-block/year clustered covariance, t22, LAI x background temperature).
Outputs in native_robustness/:
  robustness_FE.csv           trimming, forest-loss and date dose-response models (anchored to S4/S8 and to
                              date_matching_FE.csv)
  within_variation.json       within-window LAI variation and coefficient conversions
  season_temperature_grid.csv conditional LAI coefficients over background temperature, by quarter
  q1_snow_surface.csv         January-March conditional LAI coefficient over background temperature and snow cover,
                              with observation counts
  winter_decomposition.csv    January-March cool-setting coefficient under albedo and snow-flag adjustments
  av_bins.csv                 binned added-variable data (within-window residuals) by background-temperature group
  spatial_bins.csv            long-term window contrasts binned by LAI contrast within background-temperature groups
  date_dose_bins.csv          own-date minus shared-date annual contrasts binned by the unshared-date share
  date_example_candidates.csv outcome-blind candidate window-years for the observation-date example
  robustness_provenance.json
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
OUT = HERE / 'native_robustness'
sys.path.insert(0, str(ROOT / 'analysis/pipeline'))
from analyze import twoway_demean, standardize

long = pd.read_csv(HERE / 'native_analysis/longterm_windows.csv')
annual = pd.read_csv(HERE / 'native_analysis/annual_panel.csv')
annual = annual[annual.window_id.isin(long.window_id)].reset_index(drop=True)
s = json.loads((HERE / 'native_analysis/scaling.json').read_text())['primary']
lat = long.set_index('window_id').lat
Tb = (long.set_index('window_id')['T'] - s['T']['mean']) / s['T']['sd']
TbC = long.set_index('window_id')['T'] - 273.15
BASE = ['UP', 'LAI', 'T', 'P', 'wind']


def prep(data, lai_col='LAI'):
    a = standardize(data.reset_index(drop=True), s)
    if lai_col != 'LAI':
        a['LAI'] = (data[lai_col].to_numpy() - s['LAI']['mean']) / s['LAI']['sd']
    a['T_background'] = a.window_id.map(Tb)
    a['LAIxT_background'] = a.LAI * a.T_background
    return a


def fit_fe(a, target, terms):
    """Two-way FE OLS with block/year clustered covariance; returns params, cov, critical t and sample info."""
    z = a[['window_id', 'year', 'spatial_block', target] + terms].dropna().reset_index(drop=True)
    wi = pd.factorize(z.window_id)[0]; yi = pd.factorize(z.year)[0]; bi = pd.factorize(z.spatial_block)[0]
    arr = twoway_demean(z[[target] + terms].to_numpy(), wi, yi)
    fit = sm.OLS(arr[:, 0], arr[:, 1:]).fit()
    cov, _, _ = cov_cluster_2groups(fit, bi, yi, use_correction=True)
    k = len(terms) + z.window_id.nunique() + z.year.nunique() - 1
    cov = cov * (len(z) - len(terms)) / (len(z) - k)
    df = min(len(np.unique(bi)), len(np.unique(yi))) - 1
    return dict(params=np.asarray(fit.params), cov=cov, df=df, crit=tdist.ppf(.975, df), z=z, arr=arr,
                n_rows=len(z), n_windows=z.window_id.nunique(), terms=terms)


def combo(m, weights):
    g = np.zeros(len(m['terms']))
    for t, w in weights.items():
        g[m['terms'].index(t)] = w
    est = float(g @ m['params']); se = float(np.sqrt(g @ m['cov'] @ g))
    return dict(estimate=est, se=se, lo95=est - m['crit'] * se, hi95=est + m['crit'] * se,
                p=float(2 * tdist.sf(abs(est / se), m['df'])) if se > 0 else np.nan)


def report(m, case, target, extra=None, cond=True):
    rows = []
    base = dict(case=case, target=target, df=m['df'], n_rows=m['n_rows'], n_windows=m['n_windows'], **(extra or {}))
    for t in m['terms']:
        rows.append(dict(term=t, **combo(m, {t: 1}), **base))
    if cond and 'LAIxT_background' in m['terms']:
        for cval, label in [(-1.0, 'cool'), (0.0, 'mean'), (1.0, 'warm')]:
            rows.append(dict(term=f'LAI_at_{label}', **combo(m, {'LAI': 1, 'LAIxT_background': cval}), **base))
    return rows


rows = []
# ---------------------------------------------------------------- 1. anchors and within-window variation
a0 = prep(annual)
TERMS = BASE + ['LAIxT_background']
s4 = pd.read_csv(HERE / 'native_sensitivity/within_window_without_ET.csv'); s4 = s4[s4.case.eq('original')]
prim = {}
for target in ['day', 'night']:
    m = fit_fe(a0, target, TERMS); prim[target] = m
    ref = s4[s4.target.eq(target) & s4.term.eq('LAI')].estimate.iloc[0]
    assert abs(m['params'][1] - ref) < 1e-9, (target, m['params'][1], ref)
    rows += report(m, 'primary', target)
z = prim['day']['z']; arr = prim['day']['arr']
lai_dm = arr[:, 1 + TERMS.index('LAI')] * s['LAI']['sd']          # two-way demeaned LAI contrast, m2 m-2
per_window = pd.Series(lai_dm).groupby(z.window_id.to_numpy()).std(ddof=1)
within = {'pooled_LAI_sd': s['LAI']['sd'], 'twoway_within_LAI_sd': float(lai_dm.std(ddof=1)),
          'ratio_within_to_pooled': float(lai_dm.std(ddof=1) / s['LAI']['sd']),
          'per_window_sd_median': float(per_window.median()), 'per_window_sd_p10': float(per_window.quantile(.1)),
          'per_window_sd_p90': float(per_window.quantile(.9))}
for target in ['day', 'night']:
    b = prim[target]['params'][1]
    within[f'{target}_LAI_per_pooled_sd'] = float(b)
    within[f'{target}_LAI_per_within_sd'] = float(b * within['ratio_within_to_pooled'])
    within[f'{target}_LAI_per_0p1_LAI'] = float(b / s['LAI']['sd'] * 0.1)
    within[f'{target}_twoway_within_sd'] = float(prim[target]['arr'][:, 0].std(ddof=1))

# ---------------------------------------------------------------- 2. trimming extreme LAI anomalies
r_lai = pd.Series(lai_dm, index=pd.MultiIndex.from_frame(z[['window_id', 'year']]))
sd_r = r_lai.std(ddof=1)
flags = {'trim_3sd': r_lai.abs() > 3 * sd_r, 'trim_top1pct': r_lai.abs() > r_lai.abs().quantile(.99)}
wsd = per_window
flags['drop_top5pct_variable_windows'] = pd.Series(r_lai.index.get_level_values(0).isin(wsd[wsd > wsd.quantile(.95)].index), index=r_lai.index)
key = pd.MultiIndex.from_frame(annual[['window_id', 'year']])
trim_info = {}
for case, f in flags.items():
    drop = f[f].index
    sub = annual[~key.isin(drop)]
    trim_info[case] = {'dropped_window_years': int(len(drop)), 'share': float(len(drop) / len(r_lai))}
    for target in ['day', 'night']:
        rows += report(fit_fe(prep(sub), target, TERMS), case, target, {'group': 'trimming'})

# ---------------------------------------------------------------- 3. forest loss (Hansen GFC v1.13)
fl = pd.read_csv(OUT / 'forest_loss.csv')
yrs = range(2001, 2026)
L = fl.melt(id_vars=['window_id', 'zone'], value_vars=[f'loss_{y}' for y in yrs], var_name='year', value_name='loss')
L['year'] = L.year.str[5:].astype(int)
L = L.pivot_table(index=['window_id', 'year'], columns='zone', values='loss').rename(columns={'pui': 'lossN', 'bak': 'lossR'})
lag = L.groupby(level=0).shift(1).rename(columns={'lossN': 'lossN_l1', 'lossR': 'lossR_l1'})
L = L.join(lag).reset_index()
cum = fl.assign(cum=fl[[f'loss_{y}' for y in yrs]].sum(axis=1)).groupby('window_id').cum.max()
an = annual.merge(L, on=['window_id', 'year'], how='left', validate='one_to_one')
assert an[['lossN', 'lossR', 'lossN_l1', 'lossR_l1']].notna().all().all()
an['dLoss'] = 100 * (an.lossN - an.lossR); an['dLoss_l1'] = 100 * (an.lossN_l1 - an.lossR_l1)
recent = an[['lossN', 'lossR', 'lossN_l1', 'lossR_l1']].max(axis=1)
loss_info = {'portion_cumulative_2001_2025_median': float(fl[[f'loss_{y}' for y in yrs]].sum(axis=1).median()),
             'window_years_recent_loss_ge_1pct': float((recent >= .01).mean()),
             'windows_cumulative_ge_10pct': float((cum.reindex(long.window_id) >= .10).mean())}
loss_cases = {'loss_exclude_recent_1pct': an[recent < .01],
              'loss_exclude_cumulative_10pct': an[an.window_id.map(cum) < .10],
              'loss_exclude_both': an[(recent < .01) & (an.window_id.map(cum) < .10)]}
for case, sub in loss_cases.items():
    loss_info[case] = {'window_years': int(len(sub)), 'windows': int(sub.window_id.nunique())}
    for target in ['day', 'night']:
        rows += report(fit_fe(prep(sub), target, TERMS), case, target, {'group': 'forest_loss'})
a_l = prep(an); a_l['dLoss'] = an.dLoss.to_numpy(); a_l['dLoss_l1'] = an.dLoss_l1.to_numpy()
for target in ['day', 'night']:
    rows += report(fit_fe(a_l, target, TERMS + ['dLoss', 'dLoss_l1']), 'loss_contrast_controls', target, {'group': 'forest_loss'})
# quarterly models (Northern Hemisphere full cohort) without window-years of recent forest loss
pp_l = pd.read_csv(HERE / 'native_process/process_panel.csv').merge(L, on=['window_id', 'year'], how='left', validate='many_to_one')
rec_q = pp_l[['lossN', 'lossR', 'lossN_l1', 'lossR_l1']].max(axis=1)
for q in range(1, 5):
    sub = pp_l[pp_l.quarter.eq(q) & (rec_q < .01)]
    rows += report(fit_fe(prep(sub), 'day', TERMS), f'q{q}_loss_exclude_recent_1pct', 'day', {'group': 'forest_loss', 'quarter': q})
# linkage check: does recorded forest loss lower the LAI contrast?
a_l['LAIraw'] = an.LAI.to_numpy()
rows += report(fit_fe(a_l, 'LAIraw', ['dLoss', 'dLoss_l1', 'UP', 'T', 'P', 'wind']), 'loss_linkage', 'LAI', {'group': 'forest_loss'}, cond=False)

# ---------------------------------------------------------------- 4. observation-date dose-response (960 windows)
long_dates = pd.read_csv(HERE / 'native_dates/dates_all.csv')
long_dates['quarter'] = (pd.to_datetime(long_dates.date, format='%Y_%m_%d').dt.month - 1) // 3 + 1
w = long_dates.pivot_table(index=['window_id', 'year', 'date', 'quarter'], columns='zone',
                           values=['LAI', 'day', 'night', 'LAI_n'], aggfunc='first')
w.columns = [f'{v}_{zn}' for v, zn in w.columns]; w = w.reset_index()
for c in ['LAI_n_pui', 'LAI_n_bak']:
    w[c] = w[c].fillna(0)
keys = pd.read_csv(HERE / 'native_support/quarter_balanced_annual_panel.csv', usecols=['window_id', 'year'])
kidx = pd.MultiIndex.from_frame(keys)
context = annual.drop(columns=['LAI', 'day', 'night', 'LSTd', 'LSTn', 'mean', 'day_minus_night'])


def annual_quarter_means(frame, matched):
    """Equal-quarter-weight annual near-minus-reference means (analyze_native_dates.annual_means, m = 1)."""
    if matched:
        f = frame[(frame.LAI_n_pui >= 1) & (frame.LAI_n_bak >= 1)]; parts = {'pui': f, 'bak': f}
    else:
        parts = {zn: frame[frame[f'LAI_n_{zn}'] >= 1] for zn in ['pui', 'bak']}
    res = {}
    for zn, g in parts.items():
        cols = [f'{v}_{zn}' for v in ['LAI', 'day', 'night']]
        qm = g.groupby(['window_id', 'year', 'quarter'])[cols].mean().groupby(level=[0, 1])
        r = qm.mean()[qm.size() == 4]; r.columns = ['LAI', 'day', 'night']; res[zn] = r
    idx = res['pui'].index.intersection(res['bak'].index)
    return res['pui'].loc[idx] - res['bak'].loc[idx]


def date_panel(contr, idx):
    p = contr.reindex(idx).reset_index().merge(context, on=['window_id', 'year'], validate='one_to_one')
    for v in ['day', 'night']:
        p[v] = p[v] + .0065 * p.ELEV
    return p


own = annual_quarter_means(w, False); shared = annual_quarter_means(w, True)
common = kidx[kidx.isin(own.dropna().index) & kidx.isin(shared.dropna().index)]
dm = pd.read_csv(HERE / 'native_dates/date_matching_FE.csv')
for spec, x in [('J_unmatched_quarter', own), ('J_matched1_quarter', shared)]:
    m = fit_fe(prep(date_panel(x, common)), 'day', TERMS)
    ref = dm[dm['sample'].eq('common') & dm.case.eq(spec) & dm.target.eq('day') & dm.term.eq('LAI')].estimate.iloc[0]
    assert abs(m['params'][1] - ref) < 1e-9, (spec, m['params'][1], ref)
ov = pd.read_csv(HERE / 'native_dates/date_overlap.csv').set_index(['window_id', 'year'])
unshared = (1 - ov.J_date_overlap)
wmean = unshared.groupby(level=0).mean()
terc = pd.qcut(wmean, 3, labels=['low', 'middle', 'high'])
dose_info = {}
for tname in ['low', 'middle', 'high']:
    wins = terc[terc.eq(tname)].index
    idx = common[common.get_level_values(0).isin(wins)]
    dose_info[tname] = {'windows': int(len(wins)), 'mean_unshared_share': float(wmean[wins].mean()),
                        'min': float(wmean[wins].min()), 'max': float(wmean[wins].max())}
    for spec, x in [('own_dates', own), ('shared_dates', shared)]:
        for target in ['day', 'night']:
            rows += report(fit_fe(prep(date_panel(x, idx)), target, TERMS), f'dose_{tname}_{spec}', target,
                           {'group': 'date_dose', 'tercile': tname, 'dates': spec,
                            'mean_unshared_share': dose_info[tname]['mean_unshared_share']})
# window-year level: own-minus-shared annual contrasts against the unshared share
diff = (own.reindex(common) - shared.reindex(common))
dd = diff.join(unshared.rename('unshared').reindex(common))
dd['bin'] = pd.qcut(dd.unshared, 10, duplicates='drop')
bins = dd.groupby('bin', observed=True).agg(unshared=('unshared', 'mean'), n=('LAI', 'size'),
                                             dLAI=('LAI', 'mean'), dLAI_sd=('LAI', 'std'),
                                             dDay=('day', 'mean'), dDay_sd=('day', 'std'),
                                             dNight=('night', 'mean'), dNight_sd=('night', 'std')).reset_index(drop=True)
# covariance of the own-minus-shared LAI and temperature departures drives the coefficient shift
cv = dd.groupby('bin', observed=True).apply(lambda g: pd.Series({
    'corr_LAI_day': g.LAI.corr(g.day), 'corr_LAI_night': g.LAI.corr(g.night),
    'slope_day_on_LAI': np.cov(g.LAI, g.day)[0, 1] / g.LAI.var(), 'slope_night_on_LAI': np.cov(g.LAI, g.night)[0, 1] / g.LAI.var()})).reset_index(drop=True)
bins = pd.concat([bins, cv], axis=1)
bins.to_csv(OUT / 'date_dose_bins.csv', index=False)
# outcome-blind example candidates: NH windows in the middle background-temperature tercile, unshared share near the median
med = float(unshared.reindex(common).median())
cand = unshared.reindex(common).rename('unshared').reset_index()
cand['lat'] = cand.window_id.map(lat); cand['TbC'] = cand.window_id.map(TbC)
tq = TbC.reindex(cand.window_id.unique()).quantile([1 / 3, 2 / 3]).to_numpy()
cand = cand[cand.lat.ge(0) & cand.TbC.between(*tq)]
cand['dist'] = (cand.unshared - med).abs()
cand.sort_values(['dist', 'window_id', 'year']).head(25).to_csv(OUT / 'date_example_candidates.csv', index=False)
EXAMPLE = '156_738'   # midlatitude example window of Fig. 5 (outcome-blind selection, prepare_spatial.py)
ex = unshared.reindex(common).rename('unshared').reset_index()
ex = ex[ex.window_id.eq(EXAMPLE)].assign(dist=lambda v: (v.unshared - med).abs()).sort_values(['dist', 'year'])
EX_YEAR = int(ex.year.iloc[0])
xd = w[w.window_id.eq(EXAMPLE) & w.year.eq(EX_YEAR)].copy()
xd['status'] = np.select([(xd.LAI_n_pui >= 1) & (xd.LAI_n_bak >= 1), xd.LAI_n_pui >= 1, xd.LAI_n_bak >= 1],
                         ['shared', 'near_only', 'reference_only'], 'neither')
xd.to_csv(OUT / 'date_example.csv', index=False)
example_info = {'window_id': EXAMPLE, 'year': EX_YEAR, 'unshared': float(ex.unshared.iloc[0]), 'sample_median_unshared': med,
                'own_contrast': own.loc[(EXAMPLE, EX_YEAR)].to_dict(), 'shared_contrast': shared.loc[(EXAMPLE, EX_YEAR)].to_dict(),
                'status_counts': xd.status.value_counts().to_dict()}

# ---------------------------------------------------------------- 5. quarterly grids (Northern Hemisphere full cohort)
pp = pd.read_csv(HERE / 'native_process/process_panel.csv')
pfe = pd.read_csv(HERE / 'native_process/process_FE.csv')
grid_rows, sup_rows = [], []
TgridC = np.arange(-5, 30.01, 0.5)
for q in range(1, 5):
    p = pp[pp.quarter.eq(q)]
    tq = p.drop_duplicates('window_id').window_id.map(TbC)
    for target in ['day', 'night']:
        m = fit_fe(prep(p), target, TERMS)
        for label, cval in [('cool', -1.0), ('mean', 0.0), ('warm', 1.0)]:
            ref = pfe[pfe.case.eq(f'q{q}_baseline_full') & pfe.target.eq(target) & pfe.term.eq(f'LAI_at_{label}')].estimate.iloc[0]
            assert abs(combo(m, {'LAI': 1, 'LAIxT_background': cval})['estimate'] - ref) < 1e-9
        for tc in TgridC:
            c = (tc + 273.15 - s['T']['mean']) / s['T']['sd']
            grid_rows.append(dict(quarter=q, target=target, TbC=tc, **combo(m, {'LAI': 1, 'LAIxT_background': c}),
                                  support_p025=float(tq.quantile(.025)), support_p975=float(tq.quantile(.975)),
                                  n_windows=int(p.window_id.nunique())))
    sup_rows.append(dict(quarter=q, **{f'TbC_p{k}': float(tq.quantile(k / 100)) for k in [2.5, 5, 25, 50, 75, 95, 97.5]}))
for target in ['day', 'night']:  # annual reference line, primary cohort (both hemispheres)
    tq = annual.drop_duplicates('window_id').window_id.map(TbC)
    for tc in TgridC:
        c = (tc + 273.15 - s['T']['mean']) / s['T']['sd']
        grid_rows.append(dict(quarter=0, target=target, TbC=tc, **combo(prim[target], {'LAI': 1, 'LAIxT_background': c}),
                              support_p025=float(tq.quantile(.025)), support_p975=float(tq.quantile(.975)),
                              n_windows=int(annual.window_id.nunique())))
for q in range(1, 5):  # albedo contrast per LAI SD over background temperature (albedo sample)
    p = pp[pp.quarter.eq(q)].dropna(subset=['dAlb', 'dSWabs'])
    m = fit_fe(prep(p), 'dAlb', TERMS)
    ref = pfe[pfe.case.eq(f'q{q}_albedo_response') & pfe.term.eq('LAI_at_cool')].estimate.iloc[0]
    assert abs(combo(m, {'LAI': 1, 'LAIxT_background': -1.0})['estimate'] - ref) < 1e-9
    tq = p.drop_duplicates('window_id').window_id.map(TbC)
    for tc in TgridC:
        c = (tc + 273.15 - s['T']['mean']) / s['T']['sd']
        grid_rows.append(dict(quarter=q, target='dAlb', TbC=tc, **combo(m, {'LAI': 1, 'LAIxT_background': c}),
                              support_p025=float(tq.quantile(.025)), support_p975=float(tq.quantile(.975)),
                              n_windows=int(p.window_id.nunique())))
pd.DataFrame(grid_rows).to_csv(OUT / 'season_temperature_grid.csv', index=False)
gq = pd.DataFrame(grid_rows)
def crossing(q, target, col):
    g = gq[gq.quarter.eq(q) & gq.target.eq(target)].sort_values('TbC')
    g = g[g.TbC.between(g.support_p025.iloc[0], g.support_p975.iloc[0])]
    sgn = np.sign(g[col].to_numpy()); k = np.where(np.diff(sgn) != 0)[0]
    return [float(g.TbC.iloc[i] + (g.TbC.iloc[i + 1] - g.TbC.iloc[i]) * (0 - g[col].iloc[i]) / (g[col].iloc[i + 1] - g[col].iloc[i])) for i in k]
grid_crossings = {f'q{q}_{t}_{c}': crossing(q, t, c) for q in range(0, 5) for t in ['day', 'night'] for c in ['estimate', 'lo95', 'hi95']}

# ---------------------------------------------------------------- 6. January-March snow surface and winter decomposition
p1 = pp[pp.quarter.eq(1)].dropna(subset=['S_raw']).reset_index(drop=True)
a1 = prep(p1); s_mean, s_sd = p1.S_raw.mean(), p1.S_raw.std(ddof=1)
a1['S'] = (p1.S_raw - s_mean) / s_sd; a1['LAIxS'] = a1.LAI * a1.S
m_snow = fit_fe(a1, 'day', TERMS + ['LAIxS', 'S'])
ref = pfe[pfe.case.eq('q1_day_snow_moderation') & pfe.target.eq('day') & pfe.term.eq('LAIxS')].estimate.iloc[0]
assert abs(combo(m_snow, {'LAIxS': 1})['estimate'] - ref) < 1e-9
tb_obs = p1.window_id.map(TbC).to_numpy(); s_obs = p1.S_raw.to_numpy()
surf = []
for tc in np.arange(-2, 26.01, 1.0):
    for sv in np.arange(0, 100.01, 5.0):
        c = (tc + 273.15 - s['T']['mean']) / s['T']['sd']
        n_obs = int(((np.abs(tb_obs - tc) <= 1.0) & (np.abs(s_obs - sv) <= 5.0)).sum())
        surf.append(dict(TbC=tc, snow=sv, n_obs=n_obs,
                         **combo(m_snow, {'LAI': 1, 'LAIxT_background': c, 'LAIxS': (sv - s_mean) / s_sd})))
pd.DataFrame(surf).to_csv(OUT / 'q1_snow_surface.csv', index=False)
snow_support = {}
for lo, hi in [(-10, 6), (6, 8), (8, 10), (10, 13), (13, 40)]:
    k = (tb_obs >= lo) & (tb_obs < hi)
    snow_support[f'TbC_{lo}_{hi}'] = {'window_years': int(k.sum()), 'snow_min': float(s_obs[k].min()), 'snow_p05': float(np.quantile(s_obs[k], .05)),
                                      'share_snow_lt5': float((s_obs[k] < 5).mean()), 'share_snow_lt1': float((s_obs[k] < 1).mean())}
snow_support['coolest_TbC_with_snow_lt5'] = float(tb_obs[s_obs < 5].min())
snow_support['coolest_TbC_with_snow_lt1'] = float(tb_obs[s_obs < 1].min())
for tc in [10.0, 12.745, 15.0]:
    c = (tc + 273.15 - s['T']['mean']) / s['T']['sd']
    snow_support[f'coef_snow0_TbC{tc:g}'] = combo(m_snow, {'LAI': 1, 'LAIxT_background': c, 'LAIxS': (0 - s_mean) / s_sd})
p1w = p1.assign(TbC=p1.window_id.map(TbC))
p1w.groupby('window_id').agg(TbC=('TbC', 'first'), snow=('S_raw', 'mean'), lat=('lat', 'first')).reset_index() \
   .to_csv(OUT / 'q1_window_snow.csv', index=False)
# decomposition on the sample with albedo, snow-flag-free LAI and snow cover
pd1 = pp[pp.quarter.eq(1)].dropna(subset=['dAlb', 'dSWabs', 'LAIsf_raw', 'S_raw']).reset_index(drop=True)
dec = []
for case, lai_col, extra in [('baseline', 'LAI', []), ('plus_SWabs', 'LAI', ['dSWabs']),
                             ('snowflag_free_LAI', 'LAIsf_raw', []), ('snowflag_free_LAI_plus_SWabs', 'LAIsf_raw', ['dSWabs'])]:
    a = prep(pd1, lai_col); a['dSWabs'] = pd1.dSWabs.to_numpy()
    m = fit_fe(a, 'day', TERMS + extra)
    for label, cval in [('cool', -1.0), ('mean', 0.0)]:
        dec.append(dict(case=case, at=label, n_rows=m['n_rows'], n_windows=m['n_windows'],
                        **combo(m, {'LAI': 1, 'LAIxT_background': cval})))
    rows += report(m, f'q1_decomposition_{case}', 'day', {'group': 'winter_decomposition'})
pd.DataFrame(dec).to_csv(OUT / 'winter_decomposition.csv', index=False)

# ---------------------------------------------------------------- 7. added-variable bins by background-temperature group
av = []
tcut = TbC.reindex(long.window_id).quantile([1 / 3, 2 / 3]).to_numpy()


def tgroup(wid):
    v = wid.map(TbC)
    return pd.cut(v, [-np.inf, tcut[0], tcut[1], np.inf], labels=['cool', 'middle', 'warm'])


panels = [('annual', annual)] + [(f'q{q}', pp[pp.quarter.eq(q)]) for q in range(1, 5)]
for period, data in panels:
    g = tgroup(data.window_id)
    for gname in ['cool', 'middle', 'warm']:
        sub = data[g.eq(gname).to_numpy()]
        for target in ['day', 'night']:
            m = fit_fe(prep(sub), target, BASE)
            arr = m['arr']; X = arr[:, [1 + BASE.index(t) for t in BASE if t != 'LAI']]
            ry = arr[:, 0] - X @ np.linalg.lstsq(X, arr[:, 0], rcond=None)[0]
            rx = arr[:, 1 + BASE.index('LAI')]; rx = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
            slope = combo(m, {'LAI': 1})
            assert abs(np.dot(rx, ry) / np.dot(rx, rx) - slope['estimate']) < 1e-8
            b = pd.qcut(rx, 20, labels=False, duplicates='drop')
            dfb = pd.DataFrame({'b': b, 'x': rx * s['LAI']['sd'], 'y': ry})
            agg = dfb.groupby('b').agg(x=('x', 'mean'), y=('y', 'mean'), y_se=('y', lambda v: v.std(ddof=1) / np.sqrt(len(v))), n=('y', 'size'))
            for _, r in agg.iterrows():
                av.append(dict(period=period, group=gname, target=target, x_LAI=r.x, y_LST=r.y, y_se=r.y_se, n=int(r.n),
                               slope=slope['estimate'], slope_lo=slope['lo95'], slope_hi=slope['hi95'],
                               n_rows=m['n_rows'], n_windows=m['n_windows'], TbC_lo=None if gname == 'cool' else float(tcut[0 if gname == 'middle' else 1]),
                               TbC_hi=None if gname == 'warm' else float(tcut[0 if gname == 'cool' else 1])))
pd.DataFrame(av).to_csv(OUT / 'av_bins.csv', index=False)

# ---------------------------------------------------------------- 8. long-term window contrasts binned by LAI contrast
lw = long.assign(TbC=long.window_id.map(TbC))
lw['group'] = tgroup(lw.window_id).astype(str)
sp = []
for gname in ['cool', 'middle', 'warm']:
    sub = lw[lw.group.eq(gname)].copy()
    sub['bin'] = pd.qcut(sub.LAI, 10, labels=False)
    for target in ['day', 'night']:
        agg = sub.groupby('bin').agg(x=('LAI', 'mean'), y=(target, 'mean'), y_sd=(target, 'std'), n=(target, 'size'))
        for b_, r in agg.iterrows():
            sp.append(dict(group=gname, target=target, bin=int(b_), x_LAI=r.x, y_LST=r.y, y_se=r.y_sd / np.sqrt(r.n), n=int(r.n)))
pd.DataFrame(sp).to_csv(OUT / 'spatial_bins.csv', index=False)

# ---------------------------------------------------------------- outputs
res = pd.DataFrame(rows)
fam = (res.term.isin(['LAI', 'LAI_at_cool', 'LAI_at_warm', 'dLoss', 'dLoss_l1', 'dSWabs'])
       & res.group.isin(['trimming', 'forest_loss', 'date_dose', 'winter_decomposition']))
res.loc[fam, 'p_bh'] = multipletests(res.loc[fam, 'p'], method='fdr_bh')[1]
res.to_csv(OUT / 'robustness_FE.csv', index=False)
(OUT / 'within_variation.json').write_text(json.dumps(within, indent=2), encoding='utf-8')
(OUT / 'robustness_provenance.json').write_text(json.dumps({
    'anchors': 'primary no-ET models reproduce Table S4; quarterly models reproduce process_FE baseline_full; '
               'own- and shared-date annual models reproduce date_matching_FE (common sample); Q1 snow model reproduces process_FE',
    'trimming': trim_info, 'forest_loss': loss_info, 'date_dose': dose_info, 'date_example': example_info,
    'q1_snow_support': snow_support, 'grid_zero_crossings_TbC': grid_crossings,
    'BH_family': 'LAI, conditional LAI, forest-loss and absorbed-shortwave coefficients of the trimming, forest-loss, '
                 'date dose-response and winter decomposition models (Table S11)',
    'quarter_support_TbC': sup_rows, 'Tgroup_cuts_C': [float(v) for v in tcut],
    'q1_snow_scaling': {'mean': float(s_mean), 'sd': float(s_sd)},
    'forest_loss_definition': 'Hansen GFC v1.13 lossyear; share of all 30 m pixels in each forest portion lost in the year; '
                              'recent = year t or t-1 in either portion; cumulative = 2001-2025 sum, maximum of the two portions',
    'sources_sha256': {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in
                       ['native_robustness/forest_loss.csv', 'native_process/process_panel.csv', 'native_dates/dates_all.csv',
                        'native_analysis/annual_panel.csv']}}, indent=2), encoding='utf-8')
pd.set_option('display.width', 230)
show = res[res.term.isin(['LAI', 'LAI_at_cool', 'LAI_at_warm', 'dLoss', 'dLoss_l1', 'dSWabs'])]
print(show[['case', 'target', 'term', 'estimate', 'lo95', 'hi95', 'n_rows', 'n_windows']].round(3).to_string(index=False))
print(json.dumps(within, indent=1)); print(json.dumps(trim_info)); print(json.dumps(loss_info)); print(json.dumps(dose_info))
print(pd.DataFrame(dec).round(3).to_string(index=False))
print(bins.round(3).to_string(index=False))
print(json.dumps(example_info, indent=1, default=float)); print(json.dumps(snow_support, indent=1)); print(json.dumps(grid_crossings))
