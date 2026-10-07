"""Summarize observation support and test equal-quarter concurrent MODIS means."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as tdist
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'native_support'; DEST=HERE/'native_sensitivity'
sys.path.insert(0,str(ROOT/'analysis/pipeline'))
from analyze import twoway_demean,standardize
long=pd.read_csv(HERE/'native_analysis/longterm_windows.csv')
annual=pd.read_csv(HERE/'native_analysis/annual_panel.csv')
annual=annual[annual.window_id.isin(long.window_id)].copy()
s=json.loads((HERE/'native_analysis/scaling.json').read_text())['primary']
d=pd.read_csv(OUT/'support_all.csv')
assert len(d)==5334*23 and not d.duplicated(['window_id','zone','year']).any()
d=d[d.window_id.isin(long.window_id)&d.zone.isin(['pui','bak'])].copy()
assert len(d)==len(long)*23*2
d=d.merge(long[['window_id','biome','T']],on='window_id',validate='many_to_one')
cut=np.quantile(long['T'],[1/3,2/3])
d['climate']=np.where(d['T']<=cut[0],'cool',np.where(d['T']<=cut[1],'middle','warm'))
stats=[];season=[];spatial=[]
for group in ['all','cool','middle','warm']:
    a=d if group=='all' else d[d.climate.eq(group)]
    for zone,g in a.groupby('zone'):
        for v in ['LAI','day','night']:
            x=g[v+'_n_mean']
            stats.append(dict(group=group,zone=zone,variable=v,n_zone_years=len(g),mean=x.mean(),
                              p05=x.quantile(.05),median=x.median(),p95=x.quantile(.95),min=x.min(),max=x.max()))
            for q in range(1,5):
                f=g[f'{v}_q{q}n_mean']/g[v+'_n_mean'].replace(0,np.nan)
                season.append(dict(group=group,zone=zone,variable=v,quarter=q,mean_fraction=f.mean(),
                                   median_fraction=f.median(),p05=f.quantile(.05),p95=f.quantile(.95)))
            f=g[v+'_original_count']/g.local_n
            spatial.append(dict(group=group,zone=zone,variable=v,mean_valid_cell_fraction=f.mean(),
                                p05=f.quantile(.05),median=f.median(),min=f.min()))
pd.DataFrame(stats).to_csv(OUT/'annual_composite_count_summary.csv',index=False)
pd.DataFrame(season).to_csv(OUT/'quarterly_composite_share_summary.csv',index=False)
pd.DataFrame(spatial).to_csv(OUT/'annual_valid_cell_summary.csv',index=False)

n=d[d.zone.eq('pui')].set_index(['window_id','year']); b=d[d.zone.eq('bak')].set_index(['window_id','year'])
assert n.index.equals(b.index)
pairs=n[['climate']].copy(); diffs=[]
for v in ['LAI','day','night']:
    x=n[v+'_n_mean']-b[v+'_n_mean'];pairs[v+'_n_difference']=x
    for group in ['all','cool','middle','warm']:
        z=x if group=='all' else x[n.climate.eq(group)]
        diffs.append(dict(group=group,variable=v,mean_near_minus_reference=z.mean(),median_abs_difference=z.abs().median(),
                          p95_abs_difference=z.abs().quantile(.95),max_abs_difference=z.abs().max()))
pd.DataFrame(diffs).to_csv(OUT/'paired_composite_count_differences.csv',index=False)
pairs.reset_index().to_csv(OUT/'paired_observation_support.csv',index=False)

# Check native annual means against the newly rebuilt, unrounded analysis panel.
reconstruction=[]; original=annual.set_index(['window_id','year'])
idx=original.index.intersection(n.index)
for raw,v,tol in [('LAI','LAI',1e-10),('LSTd','day',1e-10),('LSTn','night',1e-10)]:
    e=(n.loc[idx,v+'_original_mean']-b.loc[idx,v+'_original_mean'])-original.loc[idx,raw]
    reconstruction.append(dict(variable=raw,n=len(e),max_abs_error=float(e.abs().max()),
                               median_abs_error=float(e.abs().median()),tolerance=tol,
                               outside_tolerance=int((e.abs()>tol).sum())))
    assert e.abs().max()<tol,reconstruction[-1]

# Common-product common-cell observations with coverage in every calendar quarter.
eligible=(n.LAI_balanced_count>=5)&(b.LAI_balanced_count>=5)
balanced=pd.DataFrame(index=n.index)
for v in ['LAI','day','night']:
    balanced[v]=n[v+'_balanced_mean']-b[v+'_balanced_mean']
balanced=balanced.loc[eligible].dropna().reset_index()
context=annual.drop(columns=['LAI','day','night','LSTd','LSTn','mean','day_minus_night'])
balanced=balanced.merge(context,on=['window_id','year'],validate='one_to_one')
balanced['LSTd']=balanced['day'];balanced['LSTn']=balanced['night']
balanced['day']+=.0065*balanced.ELEV;balanced['night']+=.0065*balanced.ELEV
balanced['mean']=(balanced.day+balanced.night)/2
balanced['day_minus_night']=balanced.day-balanced.night
years=balanced.groupby('window_id').year.nunique()
ids=set(years[years>=20].index)
balanced=balanced[balanced.window_id.isin(ids)].copy()
same=annual.merge(balanced[['window_id','year']],on=['window_id','year'],validate='one_to_one')
balanced.to_csv(OUT/'quarter_balanced_annual_panel.csv',index=False)
same.to_csv(OUT/'original_on_quarter_balanced_sample.csv',index=False)
assert len(same)==len(balanced)

rows=[];fitchecks=[]
for case,data in [('original_same_sample',same),('concurrent_quarter_balanced',balanced)]:
    a=standardize(data,s)
    a['T_background']=(a.window_id.map(long.set_index('window_id')['T'])-s['T']['mean'])/s['T']['sd']
    a['LAIxT_background']=a.LAI*a.T_background
    wi=pd.factorize(a.window_id)[0]; yi=pd.factorize(a.year)[0]; bi=pd.factorize(a.spatial_block)[0]
    terms=['UP','LAI','T','P','wind','LAIxT_background']
    for target in ['day','night']:
        arr=twoway_demean(a[[target]+terms].to_numpy(),wi,yi)
        fit=sm.OLS(arr[:,0],arr[:,1:]).fit()
        cov,_,_=cov_cluster_2groups(fit,bi,yi,use_correction=True)
        k=len(terms)+a.window_id.nunique()+a.year.nunique()-1
        cov*=(len(a)-len(terms))/(len(a)-k)
        assert np.diag(cov).min()>0
        se=np.sqrt(np.diag(cov));df=min(len(np.unique(bi)),len(np.unique(yi)))-1
        c=tdist.ppf(.975,df)
        for j,term in enumerate(terms):
            rows.append(dict(case=case,target=target,term=term,estimate=fit.params[j],se=se[j],
                lo95=fit.params[j]-c*se[j],hi95=fit.params[j]+c*se[j],p=2*tdist.sf(abs(fit.params[j]/se[j]),df),
                df=df,n_rows=len(a),n_windows=a.window_id.nunique(),n_blocks=a.spatial_block.nunique()))
        fitchecks.append(dict(case=case,target=target,within_r2=fit.rsquared,
              max_window_mean=float(pd.DataFrame(arr).groupby(wi).mean().abs().max().max()),
              max_year_mean=float(pd.DataFrame(arr).groupby(yi).mean().abs().max().max())))
fe=pd.DataFrame(rows);fe['p_bh']=multipletests(fe.p,method='fdr_bh')[1]
fe.to_csv(DEST/'observation_support_FE_sensitivity.csv',index=False)
assert max(max(c['max_window_mean'],c['max_year_mean']) for c in fitchecks)<1e-8
cohort=long.assign(quarter_balanced_eligible=long.window_id.isin(ids))
cohort.groupby(['biome','quarter_balanced_eligible']).size().rename('n').reset_index().to_csv(OUT/'quarter_balanced_sample_composition.csv',index=False)
report={'status':'passed','original_reconstruction':reconstruction,'support_rows':len(d),
        'balanced_n_windows':len(ids),'balanced_n_window_years':len(balanced),
        'balanced_min_years':int(balanced.groupby('window_id').year.nunique().min()),
        'specification':'Without modeled ET; corrected primary scales and fixed long-term climate; 5 degree block and year clustered covariance',
        'original_same_sample':'Unrounded native-quality annual means, using identical window-year rows to the concurrent quarterly comparison',
        'derived_temperature_columns':'LSTd, LSTn, day, night, mean and day_minus_night all recomputed from the relevant observation construction',
        'BH_family':'All 24 coefficients in the two-outcome two-input sensitivity comparison',
        'scope':'Annual composite-level concurrence within cells and equal-quarter weighting, not matching individual clear-sky days or dates between forest portions',
        'cautions':'This changes seasonal weighting and sample; all comparisons reported on exactly the same eligible window-years; original full-sample estimates retained',
        'fit_checks':fitchecks,'source_sha256':hashlib.sha256((OUT/'support_all.csv').read_bytes()).hexdigest()}
(OUT/'support_analysis_provenance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
print(fe[fe.term.isin(['LAI','LAIxT_background'])].to_string(index=False))
print(pd.DataFrame(stats).query("group == 'all'").to_string(index=False))
print(pd.DataFrame(diffs).query("group == 'all'").to_string(index=False))
