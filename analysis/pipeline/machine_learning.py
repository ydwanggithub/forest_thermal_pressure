"""Leakage-controlled spatial and temporal validation with complete explanations."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from threadpoolctl import threadpool_limits
from common import ROOT, OUT, BANDS, save, dump
sys.path.insert(0,str(ROOT/'analysis/pipeline'))
from responses import pdp, ale1, joint_support, verify_algorithms

threadpool_limits(2)
XGB=dict(n_estimators=600,learning_rate=.03,max_depth=4,subsample=.8,
         colsample_bytree=.8,reg_lambda=1.,random_state=42,n_jobs=2)
SIX=['UP','LAI','ET','ELEV','T','P']
FULL=SIX+['wind','dFOREST','FOREST_bak','log_pa_size','UP_trend']+['b_'+b for b in BANDS]

def metrics(y,p):
    return dict(r2=r2_score(y,p),rmse=np.sqrt(mean_squared_error(y,p)),mae=mean_absolute_error(y,p))

def raw_linear(X,train,test,y):
    mu=X.iloc[train].mean();sd=X.iloc[train].std().replace(0,1)
    A=(X-mu)/sd
    A=A.assign(LAIxT=A.LAI*A['T'],LAIxP=A.LAI*A.P,intercept=1.)
    beta=np.linalg.lstsq(A.iloc[train],y[train],rcond=None)[0]
    return A.iloc[test].to_numpy()@beta

def responses(model,X,folder):
    ales=[];pds=[];ices=[];grids={}
    for v in ['LAI','ET','T','UP']:
        curve,bins=ale1(model,X,v,n_bins=20);curve['feature']=v;ales.append(curve)
        grid=np.unique(np.quantile(X[v],np.linspace(.025,.975,41)));grids[v]=grid
        values,individual=pdp(model,X,[v],[grid],ice=True)
        pds.extend(dict(feature=v,x=x,pdp=p) for x,p in zip(grid,values))
        # Deterministic evenly spaced rows, with their identities retained.
        ids=np.unique(np.linspace(0,len(X)-1,min(120,len(X))).astype(int))
        center=np.argmin(np.abs(grid-X[v].median()))
        for j in ids:
            ices.extend(dict(feature=v,window_id=str(X.index[j]),x=x,prediction=individual[i,j],
                centered=individual[i,j]-individual[center,j]) for i,x in enumerate(grid))
    pd.concat(ales).to_csv(folder/'ale_1d.csv',index=False)
    pd.DataFrame(pds).to_csv(folder/'pdp_1d.csv',index=False)
    pd.DataFrame(ices).to_csv(folder/'ice.csv',index=False)
    ga=np.unique(np.r_[np.quantile(X.LAI,np.linspace(.05,.95,21)),X.LAI.median()])
    gt=np.unique(np.quantile(X['T'],np.linspace(.05,.95,21)))
    surface=pdp(model,X,['LAI','T'],[ga,gt]);_,mask=joint_support(X,['LAI','T'],[ga,gt])
    center=np.where(ga==X.LAI.median())[0][0]
    rows=[dict(LAI=aa,T=tt,pdp=surface[i,j],centered=surface[i,j]-surface[center,j],
         supported=bool(mask[i,j])) for i,aa in enumerate(ga) for j,tt in enumerate(gt)]
    pd.DataFrame(rows).to_csv(folder/'pdp_2d.csv',index=False)

def run_spatial(case):
    d=pd.read_csv(OUT/f'{case}_windows.csv')
    for b in BANDS:d['b_'+b]=d.biome.eq(b).astype(int)
    features=FULL if case=='longterm' else SIX
    X=d.set_index('window_id')[features]
    assert np.isfinite(X).all().all()
    folder=OUT/'models'/case;folder.mkdir(parents=True,exist_ok=True)
    previous=pd.read_csv(folder/'inputs.csv') if (folder/'inputs.csv').exists() else None
    same_input=previous is not None and previous.window_id.tolist()==d.window_id.tolist() and np.allclose(previous[features],d[features])
    d.to_csv(folder/'inputs.csv',index=False)
    folds=list(GroupKFold(5).split(X,groups=d.spatial_block))
    fold_ids=np.empty(len(d),int)
    for i,(tr,te) in enumerate(folds):
        assert not set(d.spatial_block.iloc[tr])&set(d.spatial_block.iloc[te])
        fold_ids[te]=i
    pd.DataFrame(dict(window_id=d.window_id,spatial_block=d.spatial_block,fold=fold_ids)).to_csv(folder/'fold_assignment.csv',index=False)
    summaries=[]
    for target in ['day','night']:
        p=folder/target;p.mkdir(exist_ok=True)
        if same_input and (p/'validation_summary.csv').exists():
            summaries.extend(pd.read_csv(p/'validation_summary.csv').to_dict('records'));continue
        y=d[target].to_numpy();oof={k:np.empty(len(d)) for k in ['XGB','RF','climate_linear']}
        sv=np.empty((len(d),len(features)));base=np.empty(len(d));foldstat=[]
        for k,(tr,te) in enumerate(folds):
            for name,model in [('XGB',xgb.XGBRegressor(**XGB)),('RF',RandomForestRegressor(n_estimators=400,min_samples_leaf=5,random_state=42,n_jobs=2))]:
                model.fit(X.iloc[tr],y[tr]);oof[name][te]=model.predict(X.iloc[te])
                foldstat.append(dict(fold=k,model=name,n_train=len(tr),n_test=len(te),**metrics(y[te],oof[name][te])))
                if name=='XGB':
                    model.save_model(p/f'fold_{k}.ubj')
                    e=shap.TreeExplainer(model);sv[te]=e.shap_values(X.iloc[te]);base[te]=float(e.expected_value)
            oof['climate_linear'][te]=raw_linear(X,tr,te,y)
        check=float(np.max(np.abs(sv.sum(axis=1)+base-oof['XGB'])))
        assert check<1e-4
        pd.DataFrame(dict(window_id=d.window_id,fold=fold_ids,observed=y,**oof)).to_csv(p/'oof_predictions.csv',index=False)
        sh=pd.DataFrame(sv,columns=features);sh.insert(0,'window_id',d.window_id);sh['base']=base;sh.to_csv(p/'shap_oof.csv',index=False)
        pd.DataFrame(dict(feature=features,mean_abs_shap_oof=np.abs(sv).mean(axis=0))).to_csv(p/'importance_oof.csv',index=False)
        model=xgb.XGBRegressor(**XGB).fit(X,y);model.save_model(p/'xgboost.ubj');responses(model,X,p)
        stats=[dict(case=case,target=target,model=name,validation='5_degree_spatial_block',n=len(d),
                    n_windows=len(d),group='global',**metrics(y,pred)) for name,pred in oof.items()]
        pd.DataFrame(stats).to_csv(p/'validation_summary.csv',index=False)
        pd.DataFrame(foldstat).to_csv(p/'cv_folds.csv',index=False)
        (p/'shap_check.txt').write_text(str(check),encoding='utf-8')
        summaries.extend(stats);print('ML',case,target,[(r['model'],round(r['r2'],3)) for r in stats],flush=True)
    return summaries

def run_annual():
    d=pd.read_csv(OUT/'annual_panel.csv')
    for b in BANDS:d['b_'+b]=d.biome.eq(b).astype(int)
    features=[f for f in FULL if f!='UP_trend']+['year']
    X=d[features];folder=OUT/'models/annual';folder.mkdir(exist_ok=True)
    folds=list(GroupKFold(5).split(X,groups=d.spatial_block));rows=[]
    for target in ['day','night']:
        y=d[target].to_numpy();oof=np.zeros(len(d));fi=np.zeros(len(d),int)
        for k,(tr,te) in enumerate(folds):
            assert not set(d.window_id.iloc[tr])&set(d.window_id.iloc[te])
            model=xgb.XGBRegressor(**XGB).fit(X.iloc[tr],y[tr]);oof[te]=model.predict(X.iloc[te]);fi[te]=k
        rows.append(dict(case='annual',target=target,model='XGB',validation='5_degree_spatial_block',n=len(d),n_windows=d.window_id.nunique(),group='global',**metrics(y,oof)))
        result=d[['window_id','year','spatial_block']].copy();result['observed']=y;result['prediction']=oof;result['fold']=fi
        result.to_csv(folder/f'{target}_spatial_predictions.csv',index=False)
        tr=np.flatnonzero(d.year.le(2020));te=np.flatnonzero(d.year.ge(2021))
        assert d.year.iloc[tr].max()<d.year.iloc[te].min()
        model=xgb.XGBRegressor(**XGB).fit(X.iloc[tr],y[tr]);pred=model.predict(X.iloc[te])
        model.save_model(folder/f'{target}_temporal_model.ubj')
        rows.append(dict(case='annual',target=target,model='XGB',validation='train_2003_2020_test_2021_2025',n=len(te),n_windows=d.window_id.iloc[te].nunique(),group='global',**metrics(y[te],pred)))
        q=d.iloc[te][['window_id','year']].copy();q['observed']=y[te];q['prediction']=pred;q.to_csv(folder/f'{target}_temporal_predictions.csv',index=False)
        print('ANNUAL ML',target,[(r['validation'],round(r['r2'],3)) for r in rows if r['target']==target],flush=True)
    return rows

def main():
    checks=verify_algorithms();save(checks,'qa/response_algorithm_checks.csv')
    rows=[]
    for case in ['longterm','common_MOD16','common_PML','matched_longterm']:
        rows+=run_spatial(case);save(rows,'validation.csv')
    rows+=run_annual();save(rows,'validation.csv')
    dump({'status':'complete','primary_predictors':FULL,'annual_predictors':[f for f in FULL if f!='UP_trend']+['year'],
        'late_only_night_lights_excluded':True,'spatial_cv':'GroupKFold with 5 degree block groups',
        'temporal_validation':'train 2003 to 2020; predict 2021 to 2025','annual_window_leakage':False,
        'XGB_parameters':XGB},'qa/ml_completion.json')

if __name__=='__main__':main()
