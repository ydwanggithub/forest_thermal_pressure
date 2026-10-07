"""Build annual records, period means and strict-ring sensitivity inputs."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from common import ROOT, OUT, YEARS, PML_YEARS, BANDS, master, save, dump

KEY=['win_y','win_x']
PERIODS={'2003_2010':(2003,2010),'2011_2018':(2011,2018),'2019_2025':(2019,2025)}

def frozen():
    paths=[ROOT/'data/tables/master_delta_5km_v2.csv',ROOT/'data/tables/window_zone_means.parquet',
        ROOT/'outputs/completion_20260919/pml_common_sample.csv']
    paths+=sorted((ROOT/'outputs/afm_final_figures_20260922/figures').glob('*.png'))
    rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,
           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    dest=OUT/'qa/frozen_inputs.json'
    if dest.exists():assert json.loads(dest.read_text(encoding='utf-8'))==rows
    else:dump(rows,'qa/frozen_inputs.json')

def base_panel():
    m=master()
    statics=['window_id','win_y','win_x','lat','lon','biome','dELEV','dFOREST','FOREST_bak','pa_size_km2']
    a=[]
    for year in YEARS:
        d=m[statics].copy();d['year']=year
        for v in ['UP','LAI','ET','LSTd','LSTn']:d[v]=m[f'd{v}_{year}']
        d['ELEV']=d.dELEV
        d['day']=d.LSTd+.0065*d.ELEV;d['night']=d.LSTn+.0065*d.ELEV
        d['mean']=(d.day+d.night)/2;d['day_minus_night']=d.day-d.night
        d['log_pa_size']=np.log10(d.pa_size_km2.clip(lower=1))
        d['spatial_block']=(np.floor((d.lat+90)/5).astype(int).astype(str)+'_'+np.floor((d.lon+180)/5).astype(int).astype(str))
        a.append(d)
    d=pd.concat(a,ignore_index=True)
    d['valid_core']=d[['UP','LAI','ET','day','night','ELEV']].notna().all(axis=1)
    save(d,'annual_raw.csv')
    rows=[]
    for year,g in d.groupby('year'):
        for v in ['UP','LAI','ET','day','night','valid_core']:
            rows.append(dict(year=int(year),variable=v,n_windows=int(g[v].sum()) if v=='valid_core' else int(g[v].notna().sum())))
    save(rows,'annual_coverage.csv')
    return d

def period_mean(d, start, end, min_years, product='MOD16'):
    a=d[d.year.between(start,end)].copy()
    vars_=['UP','LAI','ET','ELEV','day','night','mean','day_minus_night','T','P','wind','dFOREST','FOREST_bak','log_pa_size']
    if product=='PML':a['ET']=a.PML
    a=a.dropna(subset=['UP','LAI','ET','ELEV','day','night','T','P','wind'])
    keys=['window_id','win_y','win_x','lat','lon','biome','spatial_block']
    z=a.groupby(keys,observed=True).agg(**{v:(v,'mean') for v in vars_},n_years=('year','size')).reset_index()
    z=z[z.n_years>=min_years].copy()
    z['start']=start;z['end']=end;z['product']=product
    for v in ['UP','LAI','ET','day','night']:
        # Trend uses the same valid annual records that define this mean.
        slopes=a.groupby('window_id')[['year',v]].apply(lambda x: np.polyfit(x.year-x.year.mean(),x[v],1)[0]*10 if len(x)>=min_years else np.nan)
        z[v+'_trend']=z.window_id.map(slopes)
    return z

def matched_panel(d):
    p=pd.read_parquet(ROOT/'data/tables/window_zone_means.parquet')
    p=p.merge(master()[KEY+['window_id']],on=KEY,how='inner')
    p['ring']=np.where(p.zone.eq(1),'inner',np.where(p.zone.eq(2),'outer','background'))
    p['weighted']=p['mean']*p.n
    a=p.groupby(['window_id','var','year','ring'])[['weighted','n']].sum()
    a['value']=(a.weighted/a.n).where(a.n>=5)
    a=a.value.unstack('ring')
    a['diff']=a.inner-a.outer
    values=a['diff'].unstack('var').reset_index()
    elev=values[values.year.eq(0)][['window_id','ELEV','FOREST']].rename(columns={'FOREST':'dFOREST'})
    z=values[values.year.isin(YEARS)].drop(columns=['ELEV','FOREST']).merge(elev,on='window_id')
    context=d[['window_id','year','win_y','win_x','lat','lon','biome','spatial_block','T','P','wind','FOREST_bak','log_pa_size']]
    z=z.merge(context,on=['window_id','year'],validate='one_to_one')
    # Preserve the earlier strict comparison: both rings must have a valid
    # common >10 km reference, and each ring must be within 500 m of it.
    el=a.xs(('ELEV',0),level=['var','year'])
    eligible=el.index[(el.inner-el.background).abs().le(500)&(el.outer-el.background).abs().le(500)]
    bg=a.background.unstack('var').reset_index()
    bg=bg[bg.year.isin(YEARS)].dropna(subset=['LSTd','LSTn'])[['window_id','year']]
    z=z[z.window_id.isin(eligible)].merge(bg,on=['window_id','year'],validate='one_to_one').copy()
    z['day']=z.LSTd+.0065*z.ELEV;z['night']=z.LSTn+.0065*z.ELEV
    z['mean']=(z.day+z.night)/2;z['day_minus_night']=z.day-z.night
    z=z.dropna(subset=['UP','LAI','ET','ELEV','day','night','T','P','wind'])
    ring_values=a[['inner','outer']].stack().unstack('var').reset_index().rename(columns={'ring':'ring'})
    ring_values=ring_values[ring_values.year.isin(YEARS)].merge(z[['window_id','year']],on=['window_id','year'])
    save(ring_values,'matched_ring_annual_observations.csv')
    return z

def finish():
    d=base_panel()
    climate=pd.read_csv(OUT/'climate_annual.csv').set_index('window_id')
    for name in ['T','P','U','V']:
        d[name]=[climate.loc[w,f'{name}_{y}_mean'] for w,y in zip(d.window_id,d.year)]
    d['wind']=np.hypot(d.U,d.V)
    d=d[d.valid_core].dropna(subset=['T','P','wind']).copy()
    save(d,'annual_panel.csv')
    long=period_mean(d,2003,2025,20)
    save(long,'longterm_windows.csv')
    save(long[long.n_years.eq(23)],'balanced_windows.csv')
    periods=[]
    for tag,(y0,y1) in PERIODS.items():
        q=period_mean(d,y0,y1,max(3,int(np.ceil(.8*(y1-y0+1)))))
        q['period']=tag;periods.append(q)
    save(pd.concat(periods,ignore_index=True),'period_windows.csv')
    ring=matched_panel(d)
    save(ring,'matched_annual_panel.csv');save(period_mean(ring,2003,2025,20),'matched_longterm_windows.csv')
    pml=pd.read_csv(OUT/'pml_annual.csv')
    pa=pml[pml.zone.eq('pui')].set_index('window_id');pb=pml[pml.zone.eq('bak')].set_index('window_id')
    pairs=[]
    for y in PML_YEARS:
        good=(pa[f'pml_{y}_count']>=5)&(pb[f'pml_{y}_count']>=5)
        value=(pa[f'pml_{y}_mean']-pb[f'pml_{y}_mean']).where(good)
        pairs.append(pd.DataFrame({'window_id':value.index,'year':y,'PML':value.values}))
    d=d.merge(pd.concat(pairs),on=['window_id','year'],how='left',validate='one_to_one')
    save(d,'annual_panel.csv')
    common=d[d.year.le(2023)].dropna(subset=['PML']).copy()
    zmod=period_mean(common,2003,2023,18);zpml=period_mean(common,2003,2023,18,'PML')
    assert zmod.window_id.tolist()==zpml.window_id.tolist()
    assert zmod.n_years.tolist()==zpml.n_years.tolist()
    save(common,'common_product_annual_panel.csv')
    save(zmod,'common_MOD16_windows.csv');save(zpml,'common_PML_windows.csv')
    # Independently re-evaluate the previous PML endpoint before using the extension.
    old=pd.read_csv(ROOT/'outputs/completion_20260919/pml_common_sample.csv')
    recent=pd.concat(pairs);recent=recent[recent.year.ge(2021)].groupby('window_id').PML.mean()
    diff=old.dPML-old.window_id.map(recent)
    assert float(diff.abs().max())<1e-6,float(diff.abs().max())
    summary=[]
    for name,q in [('annual_panel',d),('longterm_windows',long),('balanced_windows',long[long.n_years.eq(23)]),('common_MOD16_windows',zmod),('common_PML_windows',zpml),('matched_annual_panel',ring),('matched_longterm_windows',pd.read_csv(OUT/'matched_longterm_windows.csv'))]:
        summary.append({'dataset':name,'rows':len(q),'windows':q.window_id.nunique(),
            'start':int(q.year.min()) if 'year' in q else int(q.start.min()),
            'end':int(q.year.max()) if 'year' in q else int(q.end.max())})
    save(summary,'sample_summary.csv')
    save(d.groupby(['year','biome']).agg(n_windows=('window_id','size')).reset_index(),'annual_analysis_coverage.csv')
    dump({'pml_recent_reproduction_max_abs_mm':float(diff.abs().max()),'annual_keys_unique':not d.duplicated(['window_id','year']).any(),
        'common_product_windows_identical':True,'year_imputation':False,'fixed_forest_support':True,
        'earliest_full_Aqua_year':2003,'latest_complete_calendar_year_in_primary_data':2025},'qa/data_checks.json')
    frozen();print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-only',action='store_true');args=p.parse_args()
    frozen()
    if args.base_only:base_panel()
    else:finish()
