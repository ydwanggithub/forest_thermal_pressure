"""Build revised analysis tables from explicit-native MODIS retrievals.

Keeps the original frozen forest-location frame and static design. Reapplies annual
cell and long-record eligibility, and preserves the former tables for comparison.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'native_analysis';OUT.mkdir(exist_ok=True)
for s in ['qa','models','source_data','figures','editable']: (OUT/s).mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'analysis/pipeline'))
import common
common.OUT=OUT
import prepare
prepare.OUT=OUT

def pairs(q,left,right):
    a=q[q.zone.eq(left)].set_index(['window_id','year'])
    b=q[q.zone.eq(right)].set_index(['window_id','year'])
    idx=a.index.intersection(b.index)
    a=a.loc[idx];b=b.loc[idx]
    out=pd.DataFrame(index=idx)
    for var,source in [('LAI','LAI'),('ET','ET'),('LSTd','day'),('LSTn','night')]:
        n=source+'_original_count';v=source+'_original_mean'
        good=a[n].ge(5)&b[n].ge(5)
        out[var]=(a[v]-b[v]).where(good)
    return out.reset_index()

def main():
    s=pd.read_csv(HERE/'native_support/support_all.csv')
    assert len(s)==5334*23 and not s.duplicated(['window_id','zone','year']).any()
    assert s.LAI_original_mean.dropna().between(0,10).all()
    assert s.LAI_balanced_mean.dropna().between(0,10).all()
    assert (s.LAI_balanced_count==s.day_balanced_count).all()
    assert (s.LAI_balanced_count==s.night_balanced_count).all()
    long_old=pd.read_csv(ROOT/'data/longterm/longterm_windows.csv')
    d=prepare.base_panel()
    d=d[d.window_id.isin(long_old.window_id)].drop(columns=['LAI','ET','LSTd','LSTn'])
    d=d.merge(pairs(s,'pui','bak'),on=['window_id','year'],validate='one_to_one')
    climate=pd.read_csv(ROOT/'data/longterm/climate_annual.csv').set_index('window_id')
    for name in ['T','P','U','V']:
        d[name]=[climate.loc[w,f'{name}_{y}_mean'] for w,y in zip(d.window_id,d.year)]
    d['wind']=np.hypot(d.U,d.V)
    d['day']=d.LSTd+.0065*d.ELEV;d['night']=d.LSTn+.0065*d.ELEV
    d['mean']=(d.day+d.night)/2;d['day_minus_night']=d.day-d.night
    d['valid_core']=d[['UP','LAI','ET','day','night','ELEV','T','P','wind']].notna().all(axis=1)
    common.save(d,'annual_raw.csv')
    coverage=[]
    for year,g in d.groupby('year'):
        for v in ['UP','LAI','ET','day','night','valid_core']:
            coverage.append(dict(year=int(year),variable=v,n_windows=int(g[v].sum()) if v=='valid_core' else int(g[v].notna().sum())))
    common.save(coverage,'annual_coverage.csv')
    d=d[d.valid_core].copy()
    long=prepare.period_mean(d,2003,2025,20)
    # The final panel and its models use exactly the qualifying long-record cohort.
    d=d[d.window_id.isin(long.window_id)].copy()
    common.save(long,'longterm_windows.csv')
    common.save(long[long.n_years.eq(23)],'balanced_windows.csv')
    periods=[]
    for tag,(y0,y1) in prepare.PERIODS.items():
        z=prepare.period_mean(d,y0,y1,max(3,int(np.ceil(.8*(y1-y0+1)))))
        z['period']=tag;periods.append(z)
    common.save(pd.concat(periods,ignore_index=True),'period_windows.csv')

    # Strict ring: retain the original static elevation criteria, but replace all
    # four MODIS annual variables and reapply minimum native-valid cell counts.
    oldring=pd.read_csv(ROOT/'data/longterm/matched_annual_panel.csv')
    oldlong=pd.read_csv(ROOT/'data/longterm/matched_longterm_windows.csv')
    c=oldring[oldring.window_id.isin(oldlong.window_id)].drop(columns=['LAI','ET','LSTd','LSTn','day','night','mean','day_minus_night'])
    ring=c.merge(pairs(s,'pui','outer'),on=['window_id','year'],validate='one_to_one')
    # >10 km cells are the >5 km union minus the disjoint 5-10 km union.
    ba=s[s.zone.eq('bak')].set_index(['window_id','year'])
    ou=s[s.zone.eq('outer')].set_index(['window_id','year'])
    far=(ba.loc[ou.index,['day_original_count','night_original_count']]-ou[['day_original_count','night_original_count']]).ge(5).all(axis=1)
    ring=ring.merge(far.rename('far_support_ok').reset_index(),on=['window_id','year'],validate='one_to_one')
    ring=ring[ring.far_support_ok].dropna(subset=['UP','LAI','ET','LSTd','LSTn','ELEV','T','P','wind'])
    ring['day']=ring.LSTd+.0065*ring.ELEV;ring['night']=ring.LSTn+.0065*ring.ELEV
    ring['mean']=(ring.day+ring.night)/2;ring['day_minus_night']=ring.day-ring.night
    strict=prepare.period_mean(ring,2003,2025,20)
    ring=ring[ring.window_id.isin(strict.window_id)]
    common.save(ring,'matched_annual_panel.csv');common.save(strict,'matched_longterm_windows.csv')

    pml=pd.read_csv(ROOT/'data/longterm/pml_annual.csv')
    pa=pml[pml.zone.eq('pui')].set_index('window_id');pb=pml[pml.zone.eq('bak')].set_index('window_id')
    pp=[]
    for y in common.PML_YEARS:
        good=pa[f'pml_{y}_count'].ge(5)&pb[f'pml_{y}_count'].ge(5)
        values=(pa[f'pml_{y}_mean']-pb[f'pml_{y}_mean']).where(good)
        pp.append(pd.DataFrame({'window_id':values.index,'year':y,'PML':values.values}))
    d=d.merge(pd.concat(pp),on=['window_id','year'],how='left',validate='one_to_one')
    common.save(d,'annual_panel.csv')
    common.save(d.groupby(['year','biome']).agg(n_windows=('window_id','size')).reset_index(),'annual_analysis_coverage.csv')
    cc=d[d.year.le(2023)].dropna(subset=['PML'])
    cm=prepare.period_mean(cc,2003,2023,18);cp=prepare.period_mean(cc,2003,2023,18,'PML')
    assert cm.window_id.tolist()==cp.window_id.tolist()
    cc=cc[cc.window_id.isin(cm.window_id)]
    common.save(cc,'common_product_annual_panel.csv');common.save(cm,'common_MOD16_windows.csv');common.save(cp,'common_PML_windows.csv')
    summary=[]
    for name,q in [('annual_panel',d),('longterm_windows',long),('balanced_windows',long[long.n_years.eq(23)]),('common_MOD16_windows',cm),('common_PML_windows',cp),('matched_annual_panel',ring),('matched_longterm_windows',strict)]:
        summary.append({'dataset':name,'rows':len(q),'windows':q.window_id.nunique(),'start':int(q.year.min()) if 'year' in q else int(q.start.min()),'end':int(q.year.max()) if 'year' in q else int(q.end.max())})
    common.save(summary,'sample_summary.csv')
    before=pd.read_csv(ROOT/'data/longterm/annual_panel.csv')
    delta=d.merge(before,on=['window_id','year'],suffixes=('_native','_legacy'),validate='one_to_one')
    changes=[]
    for v in ['LAI','ET','LSTd','LSTn','UP','T','P','wind']:
        x=delta[v+'_native']-delta[v+'_legacy']
        changes.append(dict(variable=v,n=len(x),mean_delta=x.mean(),median_absolute_delta=x.abs().median(),p95_absolute_delta=x.abs().quantile(.95),max_absolute_delta=x.abs().max()))
    common.save(changes,'native_minus_legacy_input_differences.csv')
    common.dump({'status':'passed','frame_windows':len(long_old),'final_windows':len(long),'native_support_sha256':hashlib.sha256((HERE/'native_support/support_all.csv').read_bytes()).hexdigest(),
      'cohort_dropped':sorted(set(long_old.window_id)-set(long.window_id)),
      'ranges_checked':{'LAI':[0,10]},'common_balanced_cell_counts_equal_for_all_three_variables':True,
      'reason':'Correct native-resolution quality evaluation before grid transfer; all downstream fits must use this new version',
      'original_data_and_results_preserved':True},'qa/native_data_checks.json')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(changes).to_string(index=False))

if __name__=='__main__':main()
