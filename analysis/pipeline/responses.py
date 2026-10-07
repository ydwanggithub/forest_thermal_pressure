"""Numerically checked model response summaries, with explicit estimands."""
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


def pdp(model, X, features, grids, ice=False):
    """Brute empirical marginal averaging, not tree recursion."""
    shape=tuple(len(g) for g in grids)
    points=np.array(np.meshgrid(*grids,indexing='ij')).reshape(len(features),-1).T
    averages=[];individual=[]
    for point in points:
        xx=X.copy()
        for col,value in zip(features,point):xx[col]=value
        prediction=np.asarray(model.predict(xx),dtype=float)
        averages.append(prediction.mean())
        if ice:individual.append(prediction)
    result=np.array(averages).reshape(shape)
    return (result,np.array(individual)) if ice else result


def ale1(model,X,feature,n_bins=20):
    x=X[feature].to_numpy()
    edges=np.unique(np.quantile(x,np.linspace(0,1,n_bins+1)))
    assert len(edges)>1
    bins=np.clip(np.searchsorted(edges,x,side='right')-1,0,len(edges)-2)
    effects=[];counts=[]
    for i in range(len(edges)-1):
        xx=X.iloc[np.flatnonzero(bins==i)].copy()
        low=xx.copy();high=xx.copy();low[feature]=edges[i];high[feature]=edges[i+1]
        counts.append(len(xx))
        assert len(xx)>0, 'Quantile bin cannot be empty'
        effects.append(float(np.mean(model.predict(high)-model.predict(low))))
    accum=np.r_[0.,np.cumsum(effects)]
    # Empirical centering of the piecewise-linear accumulated curve.
    center=float(np.interp(x,edges,accum).mean())
    curve=pd.DataFrame(dict(x=edges,ale=accum-center))
    local=pd.DataFrame(dict(low=edges[:-1],high=edges[1:],n=counts,local_effect=effects))
    return curve,local


def joint_support(X,features,grids):
    values=X[features].to_numpy()
    edges=[]
    for grid in grids:
        edges.append(np.r_[-np.inf,(grid[:-1]+grid[1:])/2,np.inf])
    counts=np.histogram2d(values[:,0],values[:,1],bins=edges)[0]
    low,high=np.quantile(values,[.025,.975],axis=0)
    interior=values[((values>=low)&(values<=high)).all(1)]
    hull=Delaunay(interior)
    points=np.array(np.meshgrid(*grids,indexing='ij')).reshape(2,-1).T
    supported=(hull.find_simplex(points)>=0).reshape(counts.shape)
    return counts,supported


def mixed_contrast(model,X,features,points,scales):
    low_a,high_a,low_t,high_t=points
    f=pdp(model,X,features,[np.array([low_a,high_a]),np.array([low_t,high_t])])
    cool=float(f[1,0]-f[0,0]);warm=float(f[1,1]-f[0,1])
    normalizer=(high_a-low_a)/scales[0]*(high_t-low_t)/scales[1]
    return dict(effect_cool_C=cool,effect_warm_C=warm,mixed_contrast_C=warm-cool,
                mixed_contrast_per_SD=(warm-cool)/normalizer,
                pred_lowLAI_cool=float(f[0,0]),pred_highLAI_cool=float(f[1,0]),
                pred_lowLAI_warm=float(f[0,1]),pred_highLAI_warm=float(f[1,1]))


def verify_algorithms():
    from sklearn.inspection import partial_dependence
    from sklearn.ensemble import RandomForestRegressor
    rng=np.random.default_rng(821)
    X=pd.DataFrame(rng.normal(size=(600,3)),columns=['A','T','Z'])
    class Analytic:
        def __init__(self,interaction): self.interaction=interaction
        def predict(self,q):
            return (2*q['A']-3*q['T']+.7*q['Z']+self.interaction*q['A']*q['T']).to_numpy()
    checks=[]
    m=Analytic(0);curve,_=ale1(m,X,'A')
    err=float(np.max(np.abs(curve.ale-2*(curve.x-X.A.mean()))))
    checks.append(dict(check='ALE_linear_known_solution',error=err,tolerance=1e-10,passed=err<1e-10))
    pts=[-.6,.8,-.9,.5]
    for gamma in [0,1.3]:
        r=mixed_contrast(Analytic(gamma),X,['A','T'],pts,[1,1])
        err=abs(r['mixed_contrast_per_SD']-gamma)
        checks.append(dict(check=f'mixed_contrast_known_interaction_{gamma}',error=err,tolerance=1e-10,passed=err<1e-10))
    grid=np.linspace(-1,1,7)
    surf=pdp(m,X,['A','T'],[grid,grid]);nonadd=surf-surf.mean(0)[None,:]-surf.mean(1)[:,None]+surf.mean()
    err=float(np.max(np.abs(nonadd)))
    checks.append(dict(check='additive_double_centered_surface_zero',error=err,tolerance=1e-10,passed=err<1e-10))
    rf=RandomForestRegressor(n_estimators=20,max_depth=3,random_state=42).fit(X,m.predict(X))
    ref=partial_dependence(rf,X,[0],grid_resolution=9,method='brute')
    own=pdp(rf,X,['A'],ref['grid_values'])
    err=float(np.max(np.abs(ref.average[0]-own)))
    checks.append(dict(check='brute_PDP_matches_sklearn',error=err,tolerance=1e-10,passed=err<1e-10))
    assert all(c['passed'] for c in checks)
    return checks
