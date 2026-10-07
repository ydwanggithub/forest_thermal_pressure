"""Additional-control sensitivity analyses; the primary analysis outputs are left unchanged."""
from pathlib import Path
import sys, json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as tdist
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent / 'native_sensitivity'
DEST.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT/'analysis/pipeline'))
from analyze import cluster_boot, twoway_demean, scales, standardize

SOURCE = Path(__file__).resolve().parent/'native_analysis'
long = pd.read_csv(SOURCE/'longterm_windows.csv')
s = json.loads((SOURCE/'scaling.json').read_text())['primary']
for v in ['dFOREST','FOREST_bak']:
    s[v]={'mean':float(long[v].mean()),'sd':float(long[v].std())}
a = standardize(long,s)
for v in ['dFOREST','FOREST_bak']:
    a[v]=(a[v]-s[v]['mean'])/s[v]['sd']
a['UPxLAI']=(a.UP*a.LAI-s['interaction_mean'])/s['interaction_sd']
assert a[['UP','LAI','ET','ELEV','dFOREST','FOREST_bak','T','P','wind']].notna().all().all()

rows=[]; coef=[]; diagnostics=[]
seed=20260922+sum((j+1)*ord(c) for j,c in enumerate('global'))
for name, extras in [('original',[]),('forest_cover',['dFOREST','FOREST_bak']),
                     ('forest_cover_climate',['dFOREST','FOREST_bak','T','P','wind'])]:
    ef=['UP','LAI']+extras
    tf=['UP','LAI','ET','ELEV','UPxLAI']+extras
    xe=np.column_stack([np.ones(len(a)),a[ef]])
    xt=np.column_stack([np.ones(len(a)),a[tf]])
    pe,be=cluster_boot(xe,a[['ET']].to_numpy(),a.spatial_block,1000,seed)
    pt,bt=cluster_boot(xt,a[['day','night']].to_numpy(),a.spatial_block,1000,seed)
    for target,k in [('day',0),('night',1)]:
        j=tf.index('LAI')+1; q=tf.index('ET')+1; e=ef.index('LAI')+1
        values={'canopy_conditional':(pt[j,k],bt[:,j,k]),
                'canopy_via_ET':(pe[e,0]*pt[q,k],be[:,e,0]*bt[:,q,k]),
                'canopy_path_sum':(pt[j,k]+pe[e,0]*pt[q,k],bt[:,j,k]+be[:,e,0]*bt[:,q,k])}
        for term,(v,b) in values.items():
            lo,hi=np.quantile(b,[.025,.975])
            rows.append(dict(specification=name,target=target,term=term,estimate=v,lo95=lo,hi95=hi,
                             n_windows=len(a),n_blocks=a.spatial_block.nunique(),n_boot=1000))
        for idx,term in enumerate(tf,1):
            lo,hi=np.quantile(bt[:,idx,k],[.025,.975])
            coef.append(dict(specification=name,equation=target,term=term,estimate=pt[idx,k],lo95=lo,hi95=hi))
    for idx,term in enumerate(ef,1):
        lo,hi=np.quantile(be[:,idx,0],[.025,.975])
        coef.append(dict(specification=name,equation='ET',term=term,estimate=pe[idx,0],lo95=lo,hi95=hi))
    diagnostics.append(dict(specification=name,ET_terms=ef,LST_terms=tf,
                            ET_condition_number=float(np.linalg.cond(xe)),LST_condition_number=float(np.linalg.cond(xt))))

effects=pd.DataFrame(rows)
base=pd.read_csv(SOURCE/'path_effects.csv')
base=base[(base.case=='longterm')&(base.group=='global')]
for r in effects[effects.specification=='original'].itertuples():
    q=base[(base.target==r.target)&(base.term==r.term)].iloc[0]
    assert max(abs(q[k]-getattr(r,k)) for k in ['estimate','lo95','hi95'])<1e-10
effects.to_csv(DEST/'forest_climate_path_sensitivity.csv',index=False)
pd.DataFrame(coef).to_csv(DEST/'forest_climate_coefficients.csv',index=False)

# The original continuous-climate model with and without fixed forest-cover controls.
a['LAIxT']=a.LAI*a['T']; a['LAIxP']=a.LAI*a.P
a['UPxLAI_raw']=a.UP*a.LAI
mods=[]
for name, extras in [('original',[]),('forest_cover',['dFOREST','FOREST_bak'])]:
    terms=['UP','LAI','ET','ELEV','T','P','wind','LAIxT','LAIxP','UPxLAI_raw']+extras
    x=np.column_stack([np.ones(len(a)),a[terms]])
    p,b=cluster_boot(x,a[['day','night']].to_numpy(),a.spatial_block,1000)
    for k,target in enumerate(['day','night']):
        for j,term in enumerate(terms,1):
            lo,hi=np.quantile(b[:,j,k],[.025,.975])
            mods.append(dict(specification=name,target=target,term=term,estimate=p[j,k],lo95=lo,hi95=hi))
pd.DataFrame(mods).to_csv(DEST/'forest_cover_climate_moderation.csv',index=False)

# Direct within-window canopy-temperature associations, omitting modeled ET.
fe=[]; checks=[]
for case,pf,lf in [('original','annual_panel.csv','longterm_windows.csv'),
                   ('matched','matched_annual_panel.csv','matched_longterm_windows.csv')]:
    d=pd.read_csv(SOURCE/pf); l=pd.read_csv(SOURCE/lf)
    d=d[d.window_id.isin(l.window_id)].copy()
    z=standardize(d,s)
    z['T_background']=(z.window_id.map(l.set_index('window_id')['T'])-s['T']['mean'])/s['T']['sd']
    z['LAIxT_background']=z.LAI*z.T_background
    wi=pd.factorize(z.window_id)[0]; yi=pd.factorize(z.year)[0]; bi=pd.factorize(z.spatial_block)[0]
    terms=['UP','LAI','T','P','wind','LAIxT_background']
    for target in ['day','night']:
        arr=twoway_demean(z[[target]+terms].to_numpy(),wi,yi)
        fit=sm.OLS(arr[:,0],arr[:,1:]).fit()
        cov,_,_=cov_cluster_2groups(fit,bi,yi,use_correction=True)
        k=len(terms)+len(np.unique(wi))+len(np.unique(yi))-1
        cov*=(len(z)-len(terms))/(len(z)-k)
        assert np.diag(cov).min()>0
        se=np.sqrt(np.diag(cov)); df=min(len(np.unique(bi)),len(np.unique(yi)))-1
        crit=tdist.ppf(.975,df)
        for j,term in enumerate(terms):
            fe.append(dict(case=case,target=target,term=term,estimate=fit.params[j],se=se[j],
                lo95=fit.params[j]-crit*se[j],hi95=fit.params[j]+crit*se[j],
                p=2*tdist.sf(abs(fit.params[j]/se[j]),df),df=df,n_rows=len(z),n_windows=z.window_id.nunique()))
        checks.append(dict(case=case,target=target,within_r2=fit.rsquared,
            window_mean_max=float(pd.DataFrame(arr).groupby(wi).mean().abs().max().max()),
            year_mean_max=float(pd.DataFrame(arr).groupby(yi).mean().abs().max().max())))
fe=pd.DataFrame(fe)
fe['p_bh']=multipletests(fe.p,method='fdr_bh')[1] # all 24 coefficients in this separate sensitivity family
fe.to_csv(DEST/'within_window_without_ET.csv',index=False)
assert max(max(c['window_mean_max'],c['year_mean_max']) for c in checks)<1e-8

# Recover raw mean contrasts and quantify the known lapse-rate contribution.
des=[]
for target in ['day','night']:
    y=np.column_stack([long[target]-.0065*long.ELEV,long[target],.0065*long.ELEV])
    p,b=cluster_boot(np.ones((len(long),1)),y,long.spatial_block,1000)
    for j,typ in enumerate(['raw','adjusted','adjustment']):
        lo,hi=np.quantile(b[:,0,j],[.025,.975])
        des.append(dict(target=target,quantity=typ,estimate=p[0,j],lo95=lo,hi95=hi))
pd.DataFrame(des).to_csv(DEST/'raw_adjusted_thermal_means.csv',index=False)
inputs=['longterm_windows.csv','annual_panel.csv','matched_longterm_windows.csv','matched_annual_panel.csv','scaling.json','path_effects.csv']
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'n_windows':len(long),'n_boot':1000,
 'method':f'Same {len(long)} windows, pooled primary scales, identical spatial-block resampling across equations and specifications',
 'interpretation':'Conditional association and path algebra; not causal mediation; exploratory sensitivity',
 'additional_FE_BH_family':'All 24 coefficients from two outcomes and two samples without ET; original 46-test family unchanged',
 'scales':s,'diagnostics':diagnostics,'FE_checks':checks,
 'inputs':{f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in inputs}}
(DEST/'additional_analysis_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(effects.to_string(index=False))
print(fe[fe.term.isin(['LAI','LAIxT_background'])].to_string(index=False))
print(pd.DataFrame(mods).query("term == 'LAIxT'").to_string(index=False))
print(pd.DataFrame(des).to_string(index=False))
