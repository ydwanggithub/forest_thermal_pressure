"""Describe the composite dates contributing to only one forest portion.

Input: native_dates/dates_all.csv (fetch_native_support_dates.py), restricted to the
21,386 window-years of the observation-support comparison. For the three-variable
(LAI, day and night LST) and the LAI-day date sets, a date is shared when both
portions have at least one valid cell and unshared when only one portion does.
Outputs the calendar-quarter distribution of shared and unshared dates and the
departure of values on unshared dates from the same portion's annual mean of
shared dates (Northern Hemisphere windows, quarters by composite start date).
"""
from pathlib import Path
import json, hashlib
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'native_dates'
long = pd.read_csv(HERE / 'native_analysis/longterm_windows.csv')
keys = pd.read_csv(HERE / 'native_support/quarter_balanced_annual_panel.csv', usecols=['window_id', 'year'])
d = pd.read_csv(OUT / 'dates_all.csv')
d = d.merge(keys, on=['window_id', 'year'])
d['quarter'] = (pd.to_datetime(d.date, format='%Y_%m_%d').dt.month - 1) // 3 + 1
north = set(long.loc[long.lat.ge(0), 'window_id'])

rows, dep = [], []
for tag, cnt, vars_ in [('three_variable', 'LAI_n', ['LAI', 'day', 'night']), ('LAI_day', 'LAID_n', ['LAID', 'dayD'])]:
    v = d[d[cnt] >= 1][['window_id', 'zone', 'year', 'date', 'quarter'] + vars_]
    n = v.groupby(['window_id', 'year', 'date']).zone.nunique().rename('portions').reset_index()
    v = v.merge(n, on=['window_id', 'year', 'date'])
    v['status'] = v.portions.map({2: 'shared', 1: 'unshared'})
    v = v[v.window_id.isin(north)]
    q = v.drop_duplicates(['window_id', 'year', 'date', 'status']).groupby(['status', 'quarter']).size()
    q = (q / q.groupby(level=0).transform('sum')).rename('share').reset_index()
    q['dates'] = tag
    rows.append(q)
    # departure of each date from the same portion's annual mean over its shared dates
    ref = v[v.status.eq('shared')].groupby(['window_id', 'zone', 'year'])[vars_].mean()
    u = v.set_index(['window_id', 'zone', 'year'])
    for var in vars_:
        u[var + '_departure'] = u[var] - ref[var].reindex(u.index).to_numpy()
    g = u.reset_index().groupby(['status', 'quarter'])[[x + '_departure' for x in vars_]].mean().reset_index()
    g['dates'] = tag
    dep.append(g)

share = pd.concat(rows, ignore_index=True)
departure = pd.concat(dep, ignore_index=True)
share.to_csv(OUT / 'unshared_date_quarters.csv', index=False)
departure.to_csv(OUT / 'unshared_date_departures.csv', index=False)
(OUT / 'unshared_date_provenance.json').write_text(json.dumps({
    'windows': 'Northern Hemisphere windows of the 960-window observation-support sample',
    'window_years': 'the 21,386 identical window-years, restricted to Northern Hemisphere windows',
    'shared': 'both portions have >= 1 valid cell for the stated variables on the composite date',
    'departure': 'value on a date minus the same portion-year mean over its shared dates',
    'source_sha256': hashlib.sha256((OUT / 'dates_all.csv').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
pd.set_option('display.width', 200)
print(share.round(3).to_string(index=False))
print(departure.round(3).to_string(index=False))
