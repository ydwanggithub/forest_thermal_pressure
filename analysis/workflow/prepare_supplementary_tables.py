"""Assemble the result tables of Supplementary Tables S7 to S11.

Sources: native_decomposition/decomposition_FE.csv, quarter_FE_full_cohort.csv,
quarter_full_cohort_samples.csv; native_dates/date_matching_FE.csv,
date_matching_quarter_FE.csv, date_overlap.csv. All rows describe within-window
annual models without modeled ET (window and year fixed effects, two-way clustered
covariance, t22 intervals). BH adjustment for each table covers every LAI and
LAI x background-Ta coefficient it displays.
"""
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
D, T = HERE / 'native_decomposition', HERE / 'native_dates'
dec = pd.read_csv(D / 'decomposition_FE.csv')
dm = pd.read_csv(T / 'date_matching_FE.csv'); dm = dm[dm['sample'].eq('all_defined')]
qf = pd.read_csv(D / 'quarter_FE_full_cohort.csv'); qm = pd.read_csv(T / 'date_matching_quarter_FE.csv')
ov = pd.read_csv(T / 'date_overlap.csv')

S7 = [  # (source, case, level, label)
    ('dec', 'A', 'cell', 'Independent annual means; variable-specific cells'),
    ('dec', 'B', 'cell', 'Independent annual means; quarter-balanced cells'),
    ('dec', 'C', 'cell', 'Independent quarterly means, equal quarter weights; quarter-balanced cells'),
    ('dec', 'F', 'cell', 'LD concurrent dates, equal quarter weights; quarter-balanced cells'),
    ('dec', 'G', 'cell', 'LD concurrent dates, equal quarter weights; LD balanced cells'),
    ('dec', 'E', 'cell', 'Three-variable concurrent dates, equal quarter weights; quarter-balanced cells'),
    ('dec', 'D', 'cell', 'Three-variable concurrent dates, composite weights; quarter-balanced cells'),
    ('dm', 'J_unmatched_quarter', 'date', 'Three-variable dates of each portion, equal quarter weights'),
    ('dm', 'J_matched1_quarter', 'date', 'Three-variable dates shared by both portions, equal quarter weights'),
    ('dm', 'JD_unmatched_quarter', 'date', 'LD dates of each portion, equal quarter weights'),
    ('dm', 'JD_matched1_quarter', 'date', 'LD dates shared by both portions, equal quarter weights'),
    ('dm', 'J_unmatched_composite', 'date', 'Three-variable dates of each portion, composite weights'),
    ('dm', 'J_matched1_composite', 'date', 'Three-variable dates shared by both portions, composite weights'),
    ('dm', 'JD_unmatched_composite', 'date', 'LD dates of each portion, composite weights'),
    ('dm', 'JD_matched1_composite', 'date', 'LD dates shared by both portions, composite weights'),
]
rows = []
for src, case, level, label in S7:
    tab = dec if src == 'dec' else dm
    for target in ['day', 'night']:
        z = tab[tab.case.eq(case) & tab.target.eq(target)]
        if z.empty: continue
        l = z[z.term.eq('LAI')].iloc[0]; i = z[z.term.eq('LAIxT_background')].iloc[0]
        rows.append(dict(order=len(rows), case=case, level=level, construction=label, target=target,
                         LAI=l.estimate, LAI_lo95=l.lo95, LAI_hi95=l.hi95, LAI_p=l.p,
                         LAIxT=i.estimate, LAIxT_lo95=i.lo95, LAIxT_hi95=i.hi95, LAIxT_p=i.p,
                         n_rows=int(l.n_rows), n_windows=int(l.n_windows), n_blocks=int(l.n_blocks),
                         within_r2=l.within_r2))
s7 = pd.DataFrame(rows)
p = np.r_[s7.LAI_p, s7.LAIxT_p]; adj = multipletests(p, method='fdr_bh')[1]
s7['LAI_p_bh'] = adj[:len(s7)]; s7['LAIxT_p_bh'] = adj[len(s7):]
s7.to_csv(D / 'table_S7.csv', index=False)

S8 = [('full', qf, 'full_ind_q{q}_north', 'Full cohort; independent quarterly means', ['day', 'night']),
      ('JD', qm, 'JD_matched1_q{q}_north', 'Observation-support sample; LD dates shared by both portions', ['day']),
      ('J', qm, 'J_matched1_q{q}_north', 'Observation-support sample; three-variable dates shared by both portions', ['day', 'night'])]
# Conditional LAI slopes at background temperature -1 and +1 primary SD (refits; see the analysis scripts)
cs = pd.concat([pd.read_csv(D / 'conditional_slopes.csv'), pd.read_csv(T / 'conditional_slopes.csv')], ignore_index=True)
def slope(case, target, center):
    z = cs[cs.case.eq(case) & cs.target.eq(target) & cs.center_sd.eq(center)]
    assert len(z) == 1, (case, target, center)
    return z.iloc[0]
rows = []
for tag, tab, fmt, label, targets in S8:
    for target in targets:
        for q in range(1, 5):
            case = fmt.format(q=q)
            z = tab[tab.case.eq(case) & tab.target.eq(target)]
            l = z[z.term.eq('LAI')].iloc[0]; i = z[z.term.eq('LAIxT_background')].iloc[0]
            assert abs(slope(case, target, 0.0).estimate - l.estimate) < 1e-9
            cool, warm = slope(case, target, -1.0), slope(case, target, 1.0)
            rows.append(dict(order=len(rows), source=tag, construction=label, target=target, quarter=q,
                             LAI=l.estimate, LAI_lo95=l.lo95, LAI_hi95=l.hi95, LAI_p=l.p,
                             LAIxT=i.estimate, LAIxT_lo95=i.lo95, LAIxT_hi95=i.hi95, LAIxT_p=i.p,
                             LAI_cool=cool.estimate, LAI_cool_lo95=cool.lo95, LAI_cool_hi95=cool.hi95,
                             LAI_warm=warm.estimate, LAI_warm_lo95=warm.lo95, LAI_warm_hi95=warm.hi95,
                             n_rows=int(l.n_rows), n_windows=int(l.n_windows), n_blocks=int(l.n_blocks)))
s8 = pd.DataFrame(rows)
p = np.r_[s8.LAI_p, s8.LAIxT_p]; adj = multipletests(p, method='fdr_bh')[1]
s8['LAI_p_bh'] = adj[:len(s8)]; s8['LAIxT_p_bh'] = adj[len(s8):]
# Annual rows: primary no-ET models (independent means). Their adjusted P values belong to
# the Table S4 family and are carried over, not recomputed.
s4 = pd.read_csv(HERE / 'native_sensitivity/within_window_without_ET.csv'); s4 = s4[s4.case.eq('original')]
annual_rows = []
for target in ['day', 'night']:
    l = s4[s4.target.eq(target) & s4.term.eq('LAI')].iloc[0]; i = s4[s4.target.eq(target) & s4.term.eq('LAIxT_background')].iloc[0]
    assert abs(slope('primary_noET', target, 0.0).estimate - l.estimate) < 1e-9
    cool, warm = slope('primary_noET', target, -1.0), slope('primary_noET', target, 1.0)
    annual_rows.append(dict(order=-2 + len(annual_rows), source='annual', construction='Full cohort; independent annual means',
                            target=target, quarter=0, LAI=l.estimate, LAI_lo95=l.lo95, LAI_hi95=l.hi95, LAI_p=l.p,
                            LAIxT=i.estimate, LAIxT_lo95=i.lo95, LAIxT_hi95=i.hi95, LAIxT_p=i.p,
                            LAI_cool=cool.estimate, LAI_cool_lo95=cool.lo95, LAI_cool_hi95=cool.hi95,
                            LAI_warm=warm.estimate, LAI_warm_lo95=warm.lo95, LAI_warm_hi95=warm.hi95,
                            n_rows=int(l.n_rows), n_windows=int(l.n_windows), n_blocks=201,
                            LAI_p_bh=l.p_bh, LAIxT_p_bh=i.p_bh))
s8 = pd.concat([pd.DataFrame(annual_rows), s8], ignore_index=True)
s8['T_cool_C'] = cs.T_background_C[cs.center_sd.eq(-1.0)].iloc[0]; s8['T_warm_C'] = cs.T_background_C[cs.center_sd.eq(1.0)].iloc[0]
s8.to_csv(D / 'table_S8.csv', index=False)

# ---- Table S9: urban-pressure modulation of the within-window LAI coefficient ----
um = pd.read_csv(D / 'urban_modulation_FE.csv')
rows9 = []
for case, period, spec in [('annual_UPbackground', 'Annual', 'Long-term UP contrast'), ('annual_UPannual', 'Annual', 'Annual UP contrast')] + \
        [(f'q{q}_UPbackground', ['Jan to Mar', 'Apr to Jun', 'Jul to Sep', 'Oct to Dec'][q - 1], 'Long-term UP contrast') for q in range(1, 5)]:
    for target in ['day', 'night']:
        z = um[um.case.eq(case) & um.target.eq(target)]
        term = 'LAIxUP_background' if 'background' in case else 'LAIxUP_annual'
        i = z[z.term.eq(term)].iloc[0]
        row = dict(order=len(rows9), case=case, period=period, specification=spec, target=target,
                   LAIxUP=i.estimate, LAIxUP_lo95=i.lo95, LAIxUP_hi95=i.hi95, LAIxUP_p_bh=i.p_bh,
                   n_rows=int(i.n_rows), n_windows=int(i.n_windows))
        for lab in ['low', 'high']:
            s_ = z[z.term.eq(f'LAI_at_UP_{lab}')]
            if len(s_):
                row.update({f'LAI_UP{lab}': s_.estimate.iloc[0], f'LAI_UP{lab}_lo95': s_.lo95.iloc[0], f'LAI_UP{lab}_hi95': s_.hi95.iloc[0]})
        rows9.append(row)
pd.DataFrame(rows9).to_csv(D / 'table_S9.csv', index=False)

# ---- Table S10: albedo, snow and snow-free-LAI tests of the quarterly daytime associations ----
PF = HERE / 'native_process/process_FE.csv'
if PF.exists():
    pf = pd.read_csv(PF)
    labels10 = [('albedo_response', 'Albedo contrast (percentage points)', None),
                ('snow_response', 'MODIS snow-frequency contrast (percentage points)', None),
                ('SWabs_response', 'Absorbed shortwave contrast (W m-2)', None),
                ('day_albedo_sample', 'Daytime LST, albedo sample', None),
                ('day_plus_SWabs', 'Daytime LST, absorbed shortwave added', 'dSWabs'),
                ('day_snow_sample', 'Daytime LST, snow-cover sample', None),
                ('day_snow_moderation', 'Daytime LST, LAI x snow cover added', 'LAIxS'),
                ('day_snowfree_sample', 'Daytime LST, snow-free LAI sample', None),
                ('day_snowfree_LAI', 'Daytime LST on snow-free LAI', None)]
    rows10 = []
    for q in range(1, 5):
        for key, label, extra in labels10:
            z = pf[pf.case.eq(f'q{q}_{key}')]
            g = lambda t: z[z.term.eq(t)].iloc[0]
            row = dict(order=len(rows10), quarter=q, model=key, label=label, n_rows=int(z.n_rows.iloc[0]), n_windows=int(z.n_windows.iloc[0]))
            for lab in ['cool', 'mean', 'warm']:
                r_ = g(f'LAI_at_{lab}')
                row.update({f'LAI_{lab}': r_.estimate, f'LAI_{lab}_lo95': r_.lo95, f'LAI_{lab}_hi95': r_.hi95, f'LAI_{lab}_p_bh': r_.p_bh})
            if extra:
                r_ = g(extra)
                row.update({'extra_term': extra, 'extra': r_.estimate, 'extra_lo95': r_.lo95, 'extra_hi95': r_.hi95, 'extra_p_bh': r_.p_bh})
            rows10.append(row)
    pd.DataFrame(rows10).to_csv(D / 'table_S10.csv', index=False)

# ---- Table S11: extreme anomalies, forest loss, date terciles and winter adjustments ----
RV = HERE / 'native_robustness/robustness_FE.csv'
if RV.exists():
    rv = pd.read_csv(RV)
    S11 = [  # (section, case, label, targets)
        ('Annual; extreme ΔLAI anomalies removed', 'trim_3sd', 'Two-way demeaned |ΔLAI| above 3 SD removed', ['day', 'night']),
        ('Annual; extreme ΔLAI anomalies removed', 'trim_top1pct', 'Top 1% of two-way demeaned |ΔLAI| removed', ['day', 'night']),
        ('Annual; extreme ΔLAI anomalies removed', 'drop_top5pct_variable_windows', '5% of windows with the most variable ΔLAI removed', ['day', 'night']),
        ('Annual; recorded forest loss', 'loss_exclude_recent_1pct', 'Window-years with loss ≥ 1% in the year or the year before removed', ['day', 'night']),
        ('Annual; recorded forest loss', 'loss_exclude_cumulative_10pct', 'Windows with cumulative loss ≥ 10% removed', ['day', 'night']),
        ('Annual; recorded forest loss', 'loss_exclude_both', 'Both removals', ['day', 'night']),
        ('Annual; recorded forest loss', 'loss_contrast_controls', 'ΔLoss of the year and the year before added', ['day', 'night']),
        ('Quarterly; recent forest loss removed', 'q1_loss_exclude_recent_1pct', 'Q1', ['day']),
        ('Quarterly; recent forest loss removed', 'q2_loss_exclude_recent_1pct', 'Q2', ['day']),
        ('Quarterly; recent forest loss removed', 'q3_loss_exclude_recent_1pct', 'Q3', ['day']),
        ('Quarterly; recent forest loss removed', 'q4_loss_exclude_recent_1pct', 'Q4', ['day']),
    ] + [('Annual; window terciles of the unshared-date share', f'dose_{t}_{d}', f'{t.capitalize()} u, {"own" if d == "own_dates" else "shared"} LDN dates', ['day', 'night'])
         for t in ['low', 'middle', 'high'] for d in ['own_dates', 'shared_dates']] + [
        ('Q1; adjustments of the daytime coefficient', 'q1_decomposition_baseline', 'Unadjusted', ['day']),
        ('Q1; adjustments of the daytime coefficient', 'q1_decomposition_plus_SWabs', 'ΔSWabs added', ['day']),
        ('Q1; adjustments of the daytime coefficient', 'q1_decomposition_snowflag_free_LAI', 'ΔLAIsf in place of ΔLAI', ['day']),
        ('Q1; adjustments of the daytime coefficient', 'q1_decomposition_snowflag_free_LAI_plus_SWabs', 'ΔLAIsf and ΔSWabs', ['day'])]
    rows11 = []
    for section, case, label, targets in S11:
        for target in targets:
            z = rv[rv.case.eq(case) & rv.target.eq(target)]
            g = lambda t: z[z.term.eq(t)].iloc[0]
            row = dict(order=len(rows11), section=section, case=case, label=label, target=target,
                       n_rows=int(z.n_rows.iloc[0]), n_windows=int(z.n_windows.iloc[0]))
            for lab, term in [('cool', 'LAI_at_cool'), ('mean', 'LAI'), ('warm', 'LAI_at_warm')]:
                r_ = g(term)
                row.update({f'LAI_{lab}': r_.estimate, f'LAI_{lab}_lo95': r_.lo95, f'LAI_{lab}_hi95': r_.hi95, f'LAI_{lab}_p_bh': r_.p_bh})
            extras = [t for t in ['dLoss', 'dLoss_l1', 'dSWabs'] if t in set(z.term)]
            row['extra'] = '; '.join(f"{ {'dLoss': 'ΔLoss', 'dLoss_l1': 'ΔLoss (year before)', 'dSWabs': 'ΔSWabs'}[t] } {g(t).estimate:.4f} ({g(t).lo95:.4f} to {g(t).hi95:.4f})" for t in extras)
            if 'mean_unshared_share' in z and z.mean_unshared_share.notna().any():
                row['mean_unshared_share'] = float(z.mean_unshared_share.iloc[0])
            rows11.append(row)
    link = rv[rv.case.eq('loss_linkage')]
    rows11.append(dict(order=len(rows11), section='Annual; recorded forest loss', case='loss_linkage', label='ΔLAI (m² m⁻²) regressed on ΔLoss', target='LAI',
                       n_rows=int(link.n_rows.iloc[0]), n_windows=int(link.n_windows.iloc[0]),
                       extra='; '.join(f"{ {'dLoss': 'ΔLoss', 'dLoss_l1': 'ΔLoss (year before)'}[t] } {r.estimate:.4f} ({r.lo95:.4f} to {r.hi95:.4f})" for t in ['dLoss', 'dLoss_l1'] for r in [link[link.term.eq(t)].iloc[0]])))
    pd.DataFrame(rows11).to_csv(D / 'table_S11.csv', index=False)

summary = {
    'unshared_three_variable_date_fraction_mean': float(1 - ov.J_date_overlap.mean()),
    'unshared_LAI_day_date_fraction_mean': float(1 - ov.JD_date_overlap.mean()),
    'three_variable_dates_near_mean': float(ov.near_J_dates.mean()), 'three_variable_dates_ref_mean': float(ov.ref_J_dates.mean()),
    'three_variable_dates_shared_mean': float(ov.shared_J_dates.mean()), 'LAI_day_dates_shared_mean': float(ov.shared_JD_dates.mean()),
    'matched_m3_m5': {c: float(dm[dm.case.eq(c) & dm.target.eq('day') & dm.term.eq('LAI')].estimate.iloc[0])
                      for c in ['J_matched3_quarter', 'J_matched5_quarter', 'JD_matched3_quarter', 'JD_matched5_quarter']},
    'unshared_date_departures_north': json.loads(pd.read_csv(T / 'unshared_date_departures.csv').round(4).to_json(orient='records')),
    'unshared_date_quarters_north': json.loads(pd.read_csv(T / 'unshared_date_quarters.csv').round(4).to_json(orient='records')),
    'sources': {f.parent.name + '/' + f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in
                [D / 'decomposition_FE.csv', D / 'quarter_FE_full_cohort.csv', D / 'conditional_slopes.csv',
                 T / 'date_matching_FE.csv', T / 'date_matching_quarter_FE.csv', T / 'date_overlap.csv',
                 T / 'conditional_slopes.csv', T / 'unshared_date_departures.csv', T / 'unshared_date_quarters.csv',
                 HERE / 'native_sensitivity/within_window_without_ET.csv', D / 'urban_modulation_FE.csv', HERE / 'native_process/process_FE.csv',
                 HERE / 'native_robustness/robustness_FE.csv']},
}
(D / 'tables_provenance.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
pd.set_option('display.width', 250)
print(s7[['case', 'target', 'LAI', 'LAI_lo95', 'LAI_hi95', 'LAIxT', 'LAIxT_lo95', 'LAIxT_hi95', 'LAI_p_bh', 'n_rows', 'within_r2']].round(3).to_string(index=False))
print(s8[['source', 'target', 'quarter', 'LAI_cool', 'LAI', 'LAI_warm', 'LAI_p_bh', 'LAIxT', 'LAIxT_p_bh', 'n_rows', 'n_windows']].round(3).to_string(index=False))
print(json.dumps(summary, indent=2))
