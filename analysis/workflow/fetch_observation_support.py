"""Reconstruct MODIS composite support on frozen cell unions, without new global rasters.

The sensitivity uses concurrent LAI/day/night composite dates at each cell and
equal calendar-quarter weights, requiring >=1 concurrent observation per quarter.
It does not claim identical clear-sky dates across the two forest portions.
"""
import os
from pathlib import Path
import argparse, json, time, hashlib, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore',category=FutureWarning)
import ee
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'observation_support'; OUT.mkdir(exist_ok=True)
CACHE=OUT/'cache'; CACHE.mkdir(exist_ok=True)
CRS='EPSG:4326'; TRANSFORM=[.01,0,-180,0,-.01,84]
YEARS=list(range(2003,2026))

def sources():
    p=ROOT/'data/geometry/frozen_zones.geojson'
    fs=json.loads(p.read_text(encoding='utf-8'))['features']
    ids=set(pd.read_csv(ROOT/'data/longterm/longterm_windows.csv').window_id)
    fs=[f for f in fs if f['properties']['window_id'] in ids]
    assert len(fs)==2*len(ids)==4208
    return fs,p

def image(year):
    dates=(f'{year}-01-01',f'{year+1}-01-01')
    lc=ee.ImageCollection('MODIS/061/MOD15A2H').filterDate(*dates)
    tc=ee.ImageCollection('MODIS/061/MYD11A2').filterDate(*dates)
    def laigood(i):
        return i.select('Lai_500m').multiply(.1).updateMask(i.select('FparLai_QC').bitwiseAnd(1).eq(0)).rename('LAI').copyProperties(i,['system:time_start'])
    def tgood(i):
        d=i.select('LST_Day_1km').multiply(.02).updateMask(i.select('QC_Day').bitwiseAnd(3).lte(1)).rename('day')
        n=i.select('LST_Night_1km').multiply(.02).updateMask(i.select('QC_Night').bitwiseAnd(3).lte(1)).rename('night')
        return d.addBands(n).copyProperties(i,['system:time_start'])
    l=lc.map(laigood); t=tc.map(tgood)
    # Pair only dates present in both products; missing collection dates are recorded.
    def concurrent(i):
        li=ee.Image(l.filter(ee.Filter.eq('system:time_start',i.get('system:time_start'))).first())
        both=i.addBands(li)
        return both.updateMask(both.mask().reduce(ee.Reducer.min())).copyProperties(i,['system:time_start'])
    joint=t.filter(ee.Filter.inList('system:time_start',l.aggregate_array('system:time_start'))).map(concurrent)
    bands=[]; qmeans=[]; qvalid=[]
    for name,c in [('LAI',l),('day',t.select('day')),('night',t.select('night'))]:
        bands.append(c.mean().rename(name+'_original'))
        bands.append(c.count().unmask(0).rename(name+'_n'))
        for q in range(4):
            sub=c.filter(ee.Filter.calendarRange(q*3+1,q*3+3,'month'))
            bands.append(sub.count().unmask(0).rename(f'{name}_q{q+1}n'))
    for q in range(4):
        sub=joint.filter(ee.Filter.calendarRange(q*3+1,q*3+3,'month'))
        count=sub.select('LAI').count().unmask(0)
        bands.append(count.rename(f'joint_q{q+1}n'))
        qmeans.append(sub.mean());qvalid.append(count.gte(1))
    allq=ee.ImageCollection.fromImages(qvalid).min()
    balanced=ee.ImageCollection.fromImages(qmeans).mean().updateMask(allq)
    bands.append(balanced.rename(['day_balanced','night_balanced','LAI_balanced']))
    bands.append(ee.Image.constant(1).rename('grid_check'))
    return ee.Image.cat(bands)

def fetch(year,i,fs,img):
    p=CACHE/f'y{year}_b{i:03d}.json'
    if p.exists():
        rows=json.loads(p.read_text())
        assert [(r['window_id'],r['zone']) for r in rows]==[(f['properties']['window_id'],f['properties']['zone']) for f in fs]
        return rows
    fc=ee.FeatureCollection([ee.Feature(ee.Geometry(f['geometry'],proj=CRS,geodesic=False),f['properties']) for f in fs])
    reducer=ee.Reducer.mean().unweighted().combine(ee.Reducer.count(),sharedInputs=True)
    query=img.reduceRegions(collection=fc,reducer=reducer,crs=CRS,crsTransform=TRANSFORM,tileScale=4)
    query=query.map(lambda f:ee.Feature(None,f.toDictionary()))
    t0=time.time()
    rows=[f['properties'] for f in query.getInfo()['features']]
    assert len(rows)==len(fs)
    for r in rows:
        assert r['grid_check_count']==r['local_n'],(year,r['window_id'],r['grid_check_count'],r['local_n'])
        r['year']=year
    p.write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
    print('DONE',year,i,len(rows),'seconds',round(time.time()-t0,1),flush=True)
    return rows

def main(pilot=False):
    ee.Initialize(project=os.environ['EE_PROJECT']);ee.data.setDeadline(180000)
    fs,source=sources(); batch=256
    years=[2020] if pilot else YEARS
    meta_path=OUT/'collection_metadata.json'
    if not meta_path.exists():
        ld=ee.ImageCollection('MODIS/061/MOD15A2H').filterDate('2003-01-01','2026-01-01').aggregate_array('system:time_start').getInfo()
        td=ee.ImageCollection('MODIS/061/MYD11A2').filterDate('2003-01-01','2026-01-01').aggregate_array('system:time_start').getInfo()
        from datetime import datetime, timezone
        def iso(v):return datetime.fromtimestamp(v/1000,timezone.utc).date().isoformat()
        def counts(seq):
            dates=[iso(v) for v in seq]
            return {str(y):{f'q{q+1}':sum(x.startswith(str(y)) and q*3+1<=int(x[5:7])<=q*3+3 for x in dates) for q in range(4)} for y in YEARS}
        meta={'LAI_collection':'MODIS/061/MOD15A2H','LST_collection':'MODIS/061/MYD11A2',
          'dates_identical':sorted(ld)==sorted(td),'nominal_composites_per_year':46,
          'LAI_total_dates':len(ld),'LST_total_dates':len(td),'common_total_dates':len(set(ld)&set(td)),
          'LAI_only_dates':[iso(v) for v in sorted(set(ld)-set(td))],
          'LST_only_dates':[iso(v) for v in sorted(set(td)-set(ld))],
          'LAI_available':counts(ld),'LST_available':counts(td),'common_available':counts(set(ld)&set(td)),
          'grid_transform':TRANSFORM,'resampling':'Earth Engine default nearest-neighbor, as original annual raster export',
          'spatial_support':'Frozen near/reference common-grid cell unions; unweighted mean at grid centers',
          'support_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'sensitivity':'Common LAI/day/night dates within each cell, equal quarter means, all four quarters required; at least five balanced cells per side at pairing',
          'not_established':'Identical daily clear-sky sampling across forest portions, or independent satellite/process observations'}
        meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    jobs=[]
    for y in years:
        img=image(y)
        starts=[0] if pilot else range(0,len(fs),batch)
        for start in starts:jobs.append((y,start//batch,fs[start:start+batch],img))
    rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(fetch,*j):(j[0],j[1]) for j in jobs}
        for f in as_completed(pending):
            try:rows.extend(f.result())
            except Exception as e:
                errors.append(dict(job=pending[f],type=type(e).__name__,message=str(e)[:600]))
                print('FAILED',errors[-1],flush=True)
    if errors:
        (OUT/'fetch_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
        raise RuntimeError(f'{len(errors)} requests failed; successful cache retained')
    d=pd.DataFrame(rows).sort_values(['window_id','zone','year'])
    d.to_csv(OUT/('support_pilot.csv' if pilot else 'support_all.csv'),index=False)
    print('COMPLETE',len(d),'rows',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
