"""Spatial paths, annual persistence and within-window temporal responses."""
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from common import OUT, YEARS, BANDS, SEED, save, dump
threadpool_limits(2)

TARGETS=['day','night']
TERMS=['UP','LAI','ET','ELEV','UPxLAI']
SEM=[];EFFECTS=[];DRAWS=[];R2=[];PROFILES=[];MODERATION=[]

def scales(d):
    return {v:{'mean':float(d[v].mean()),'sd':float(d[v].std())} for v in ['UP','LAI','ET','ELEV','T','P','wind','day','night']}

def standardize(d,s):
    a=d.copy()
    for v in ['UP','LAI','ET','ELEV','T','P','wind']:a[v]=(a[v]-s[v]['mean'])/s[v]['sd']
    return a

def cluster_boot(X,Y,blocks,B,seed=SEED):
    Y=np.asarray(Y);Y=Y[:,None] if Y.ndim==1 else Y
    codes,unique=pd.factorize(blocks,sort=True)
    gx=np.stack([X[codes==i].T@X[codes==i] for i in range(len(unique))])
    gy=np.stack([X[codes==i].T@Y[codes==i] for i in range(len(unique))])
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=B)
    bx=np.einsum('bg,gij->bij',weights,gx);by=np.einsum('bg,gij->bij',weights,gy)
    estimates=np.linalg.solve(bx,by)
    point=np.linalg.lstsq(X,Y,rcond=None)[0]
    return point,estimates

def sem(d,s,case,group='global',B=1000,store_draws=True,year=None):
    a=standardize(d,s)
    raw_product=a.UP*a.LAI
    # Fixed pooled interaction scale makes all subgroup and yearly coefficients comparable.
    whole=pd.read_csv(OUT/f'{case}_windows.csv') if (OUT/f'{case}_windows.csv').exists() else None
    if whole is None: im=s.get('interaction_mean',0.);isd=s.get('interaction_sd',1.)
    else:
        iz=((whole.UP-s['UP']['mean'])/s['UP']['sd'])*((whole.LAI-s['LAI']['mean'])/s['LAI']['sd'])
        im=float(iz.mean());isd=float(iz.std())
    a['UPxLAI']=(raw_product-im)/isd
    blocks=a.spatial_block
    equations=[('dLAI',['UP'],a[['LAI']].to_numpy()),('dET',['UP','LAI'],a[['ET']].to_numpy()),
               ('dLST',TERMS,a[TARGETS].to_numpy())]
    fits={};meta=dict(case=case,group=group,n=len(a),n_blocks=a.spatial_block.nunique(),year=year,
                     product='PML' if 'PML' in case else 'MOD16')
    for node,features,y in equations:
        X=np.column_stack([np.ones(len(a)),a[features].to_numpy()])
        group_seed=SEED+sum((j+1)*ord(c) for j,c in enumerate(group))
        point,bb=cluster_boot(X,y,blocks,B,seed=group_seed)
        fits[node]=(point[1:],bb[:,1:])
        r2=1-((y-X@point)**2).sum(axis=0)/((y-y.mean(axis=0))**2).sum(axis=0)
        for k in range(y.shape[1]):
            target=TARGETS[k] if node=='dLST' else 'both'
            R2.append(dict(**meta,node=node,target=target,r2=r2[k]))
            for j,v in enumerate(features):
                c=point[j+1,k];b=bb[:,j+1,k]
                thermal_sd=s[target]['sd'] if node=='dLST' else 1.
                lo,hi=np.quantile(b,[.025,.975])
                SEM.append(dict(**meta,target=target,path=node+'<-d'+v,coefficient_C=c if node=='dLST' else np.nan,
                    estimate=c/thermal_sd,lo95=lo/thermal_sd,hi95=hi/thermal_sd,ci_excludes_zero=bool(lo*hi>0),n_boot=B))
    av=fits['dLAI'][0][0,0];ab=fits['dLAI'][1][:,0,0]
    cu,cl=fits['dET'][0][:,0];cub,clb=fits['dET'][1][:,:,0].T
    for k,target in enumerate(TARGETS):
        p=fits['dLST'][0][:,k];b=fits['dLST'][1][:,:,k]
        defs={'canopy_conditional':(p[1],b[:,1]),'canopy_via_ET':(cl*p[2],clb*b[:,2]),
            'canopy_path_sum':(p[1]+cl*p[2],b[:,1]+clb*b[:,2]),
            'direct':(p[0],b[:,0]),'all_canopy_routes':(av*(p[1]+cl*p[2]),ab*(b[:,1]+clb*b[:,2])),
            'via_ET':(cu*p[2],cub*b[:,2]),'UPxLAI':(p[4],b[:,4])}
        for term,(value,draw) in defs.items():
            lo,hi=np.quantile(draw,[.025,.975])
            EFFECTS.append(dict(**meta,target=target,term=term,estimate=value,lo95=lo,hi95=hi,n_boot=B))
            if store_draws:
                DRAWS.extend(dict(**meta,target=target,term=term,draw=i,estimate=x) for i,x in enumerate(draw))
    return fits

def climate(d,s,case,group='global'):
    a=standardize(d,s)
    a['LAIxT']=a.LAI*a['T'];a['LAIxP']=a.LAI*a.P;a['UPxLAI']=a.UP*a.LAI
    terms=['UP','LAI','ET','ELEV','T','P','wind','LAIxT','LAIxP','UPxLAI']
    X=np.column_stack([np.ones(len(a)),a[terms].to_numpy()])
    point,bb=cluster_boot(X,a[TARGETS].to_numpy(),a.spatial_block,1000)
    for k,target in enumerate(TARGETS):
        for j,v in enumerate(terms,1):
            lo,hi=np.quantile(bb[:,j,k],[.025,.975])
            MODERATION.append(dict(case=case,group=group,target=target,term=v,estimate=point[j,k],lo95=lo,hi95=hi,n=len(a)))
        for temp in np.linspace(*np.quantile(d['T'],[.05,.95]),80):
            z=(temp-s['T']['mean'])/s['T']['sd']
            value=point[terms.index('LAI')+1,k]+z*point[terms.index('LAIxT')+1,k]
            draws=bb[:,terms.index('LAI')+1,k]+z*bb[:,terms.index('LAIxT')+1,k]
            lo,hi=np.quantile(draws,[.025,.975])
            PROFILES.append(dict(case=case,target=target,temperature_C=temp-273.15,estimate=value,lo95=lo,hi95=hi))

def twoway_demean(A,wi,yi):
    out=A.copy()
    for _ in range(100):
        old=out.copy()
        for codes in [wi,yi]:
            n=np.bincount(codes)
            means=np.stack([np.bincount(codes,weights=out[:,j],minlength=len(n))/n for j in range(out.shape[1])],axis=1)
            out-=means[codes]
        if np.max(np.abs(out-old))<1e-10:break
    return out

def panel_model(d,long,s,case):
    d=d[d.window_id.isin(long.window_id)].copy()
    a=standardize(d,s)
    # The moderator is the fixed long-record climate of each forest window.
    bg=long.set_index('window_id')['T']
    a['T_background']=(a.window_id.map(bg)-s['T']['mean'])/s['T']['sd']
    a['LAIxT_background']=a.LAI*a.T_background
    wi=pd.factorize(a.window_id)[0];yi=pd.factorize(a.year)[0];blocks=pd.factorize(a.spatial_block)[0]
    models=[('LAI',['UP','T','P','wind']),('ET',['LAI','UP','T','P','wind']),
            ('day',['UP','LAI','ET','T','P','wind','LAIxT_background']),
            ('night',['UP','LAI','ET','T','P','wind','LAIxT_background'])]
    rows=[];fitrows=[]
    for target,terms in models:
        arr=twoway_demean(a[[target]+terms].to_numpy(),wi,yi)
        y=arr[:,0];X=arr[:,1:]
        fit=sm.OLS(y,X).fit()
        cov,_,_=cov_cluster_2groups(fit,blocks,yi,use_correction=True)
        # Account for absorbed window/year parameters in the finite-sample correction.
        total_parameters=len(terms)+len(np.unique(wi))+len(np.unique(yi))-1
        cov*= (len(a)-len(terms))/(len(a)-total_parameters)
        assert np.diag(cov).min()>0, (case,target,np.diag(cov))
        se=np.sqrt(np.diag(cov));df=min(len(np.unique(blocks)),len(np.unique(yi)))-1
        critical=tdist.ppf(.975,df)
        for j,v in enumerate(terms):
            rows.append(dict(case=case,target=target,term=v,estimate=fit.params[j],se=se[j],
                lo95=fit.params[j]-critical*se[j],hi95=fit.params[j]+critical*se[j],
                p=2*tdist.sf(abs(fit.params[j]/se[j]),df),df=df,n_rows=len(a),n_windows=a.window_id.nunique(),
                n_years=a.year.nunique(),n_spatial_blocks=a.spatial_block.nunique()))
        fitrows.append(dict(case=case,target=target,within_r2=fit.rsquared,n_rows=len(a),
            max_abs_window_residual_mean=float(pd.Series(arr[:,0]).groupby(wi).mean().abs().max()),
            max_abs_year_residual_mean=float(pd.Series(arr[:,0]).groupby(yi).mean().abs().max())))
    return rows,fitrows

def main():
    long=pd.read_csv(OUT/'longterm_windows.csv')
    annual=pd.read_csv(OUT/'annual_panel.csv')
    cut=np.quantile(long['T'],[1/3,2/3]);s0=scales(long)
    interaction=((long.UP-s0['UP']['mean'])/s0['UP']['sd'])*((long.LAI-s0['LAI']['mean'])/s0['LAI']['sd'])
    s0['interaction_mean']=float(interaction.mean());s0['interaction_sd']=float(interaction.std())
    dump({'primary':s0,'climate_tercile_boundaries_K':cut.tolist(),'spatial_block_degrees':5,
          'bootstrap_replicates_main':1000,'bootstrap_replicates_annual':300},'scaling.json')
    cases={'longterm':long,'common_MOD16':pd.read_csv(OUT/'common_MOD16_windows.csv'),
           'common_PML':pd.read_csv(OUT/'common_PML_windows.csv'),
           'matched_longterm':pd.read_csv(OUT/'matched_longterm_windows.csv')}
    for case,d in cases.items():
        s=scales(d)
        d['climate']=np.where(d['T']<=cut[0],'cool',np.where(d['T']<=cut[1],'mid','warm'))
        save(d,f'{case}_windows.csv')
        for group in ['global']+BANDS+['cool','mid','warm']:
            a=d if group=='global' else d[d.biome.eq(group)] if group in BANDS else d[d.climate.eq(group)]
            sem(a,s,case,group)
        climate(d,s,case)
        print('PATHS',case,len(d),flush=True)
    means=[]
    fixed=annual[annual.window_id.isin(long.window_id)].copy()
    for year in YEARS:
        g=fixed[fixed.year.eq(year)]
        sem(g,s0,'annual',B=300,store_draws=False,year=year)
        for group in ['global']+BANDS:
            a=g if group=='global' else g[g.biome.eq(group)]
            X=np.ones((len(a),1));vars_=['day','night','LAI','ET','UP','T','P']
            point,bb=cluster_boot(X,a[vars_].to_numpy(),a.spatial_block,1000)
            for k,v in enumerate(vars_):
                lo,hi=np.quantile(bb[:,0,k],[.025,.975])
                means.append(dict(year=year,group=group,variable=v,estimate=point[0,k],lo95=lo,hi95=hi,n=len(a)))
        print('YEAR',year,len(g),flush=True)
    periods=pd.read_csv(OUT/'period_windows.csv')
    for tag,g in periods.groupby('period'):
        g=g[g.window_id.isin(long.window_id)]
        sem(g,s0,'period_'+tag,B=1000,store_draws=False)
    # Balanced-record and stricter elevation checks on the same 23-year specification.
    for tag,g in [('balanced23',long[long.n_years.eq(23)]),('elevation200',long[long.ELEV.abs().le(200)]),('elevation100',long[long.ELEV.abs().le(100)])]:
        sem(g,s0,tag,B=1000,store_draws=False)
    save(SEM,'sem_paths.csv');save(EFFECTS,'path_effects.csv');save(DRAWS,'path_bootstrap.csv');save(R2,'equation_r2.csv')
    save(MODERATION,'climate_moderation.csv');save(PROFILES,'climate_profiles.csv');save(means,'annual_means.csv')
    rows=[];qa=[]
    for case,d,summary in [('original',annual,long),('matched',pd.read_csv(OUT/'matched_annual_panel.csv'),cases['matched_longterm'])]:
        r,q=panel_model(d,summary,s0,case);rows+=r;qa+=q
    fe=pd.DataFrame(rows);fe['p_bh']=multipletests(fe.p,method='fdr_bh')[1]
    save(fe,'within_window_effects.csv');save(qa,'qa/panel_fit_checks.csv')
    assert pd.DataFrame(qa).filter(like='max_abs').to_numpy().max()<1e-8
    # All periods and bootstrap results remain visible regardless of significance.
    dump({'status':'complete','years':YEARS,'main_units':'degrees Celsius per pooled long-record LAI SD',
        'annual_panel_covariance':'two-way spatial-block and year clustering with window and year fixed effects',
        'spatial_paths_uncertainty':'5 degree spatial-block percentile bootstrap',
        'climate_groups':'fixed terciles of 2003 to 2025 window air temperature'},'qa/analysis_completion.json')
    print(fe[['case','target','term','estimate','lo95','hi95']].to_string(index=False),flush=True)

if __name__=='__main__':main()
