"""Fetch annual PML and climate tables on the existing spatial support.

No global raster download. At most three simultaneous Earth Engine requests.
Existing masks, products and raw tables remain unchanged.
"""
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import ee
import pandas as pd
from common import ROOT, OUT, YEARS, PML_YEARS, master, save, dump

CACHE = OUT / 'remote_cache'
CRS = 'EPSG:4326'
TRANSFORM = [.01, 0, -180, 0, -.01, 84]

def geometries(tag):
    if tag == 'pml':
        p = ROOT / 'data/geometry/frozen_zones.geojson'
        return json.loads(p.read_text(encoding='utf-8'))['features']
    fs = []
    for r in master().itertuples():
        x = -180 + r.win_x * .25
        y = 84 - r.win_y * .25
        fs.append({'type':'Feature', 'geometry':{'type':'Polygon',
            'coordinates':[[[x,y],[x+.25,y],[x+.25,y-.25],[x,y-.25],[x,y]]]},
            'properties':{'window_id':r.window_id,'local_n':625}})
    return fs

def image(tag):
    bands = []
    if tag == 'pml':
        col = ee.ImageCollection('CAS/IGSNRR/PML/V2_v018')
        projection = col.first().select('Ec').projection()
        for y in PML_YEARS:
            sub = col.filterDate(f'{y}-01-01', f'{y+1}-01-01')
            end = ee.Date.fromYMD(y+1, 1, 1)
            def amount(i):
                duration = end.difference(ee.Date(i.get('system:time_start')), 'day').min(8)
                return i.select(['Ec','Es','Ei']).reduce(ee.Reducer.sum()).multiply(duration)
            annual = sub.map(amount).sum().setDefaultProjection(projection)
            bands.append(annual.reduceResolution(ee.Reducer.mean(),maxPixels=1024)
                .reproject(crs=CRS,crsTransform=TRANSFORM).rename(f'pml_{y}'))
    else:
        col = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
        for y in YEARS:
            sub = col.filterDate(f'{y}-01-01', f'{y+1}-01-01')
            t = sub.select('temperature_2m').mean().rename(f'T_{y}')
            p = sub.select('total_precipitation_sum').sum().rename(f'P_{y}')
            u = sub.select('u_component_of_wind_10m').mean().rename(f'U_{y}')
            v = sub.select('v_component_of_wind_10m').mean().rename(f'V_{y}')
            bands.extend([t,p,u,v])
    return ee.Image.cat(bands)

def fetch(tag, i, fs, img):
    path = CACHE / f'{tag}_{i:04d}.json'
    if path.exists():
        rows = json.loads(path.read_text(encoding='utf-8'))
        assert [r['window_id'] for r in rows] == [f['properties']['window_id'] for f in fs]
        return rows
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f['geometry'],proj=CRS,geodesic=False),f['properties']) for f in fs])
    reducer = ee.Reducer.mean().unweighted().combine(ee.Reducer.count(), sharedInputs=True)
    query = img.addBands(ee.Image.constant(1).rename('grid_check')).reduceRegions(
        collection=fc, reducer=reducer, crs=CRS,crsTransform=TRANSFORM,tileScale=4)
    query = query.map(lambda f: ee.Feature(None,f.toDictionary()))
    for attempt in range(3):
        try:
            start=time.time()
            rows=[f['properties'] for f in query.getInfo()['features']]
            assert len(rows)==len(fs)
            for row in rows:
                assert row['grid_check_count']==row['local_n'], (tag,row['window_id'])
            path.write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
            print(tag,i,'complete',len(rows),'seconds',round(time.time()-start,1),flush=True)
            return rows
        except Exception as e:
            print(tag,i,'attempt',attempt+1,type(e).__name__,str(e)[:180],flush=True)
            if attempt==2: raise
            time.sleep(2*(attempt+1))

def main(pilot=False, only=None):
    ee.Initialize(project=os.environ['EE_PROJECT'])
    ee.data.setDeadline(180000)
    tags = [only] if only else ['pml','climate']
    jobs=[]
    for tag in tags:
        source='CAS/IGSNRR/PML/V2_v018' if tag=='pml' else 'ECMWF/ERA5_LAND/MONTHLY_AGGR'
        ys=PML_YEARS if tag=='pml' else YEARS
        c=ee.ImageCollection(source).filterDate(f'{min(ys)}-01-01',f'{max(ys)+1}-01-01')
        counts=c.map(lambda im: im.set('year',ee.Date(im.get('system:time_start')).format('YYYY'))).aggregate_histogram('year').getInfo()
        assert all(counts[str(y)]==(46 if tag=='pml' else 12) for y in ys),counts
        dump({'source':source,'years':ys,'annual_image_counts':counts,'grid':TRANSFORM,
              'spatial_support':'frozen forest-zone cell unions' if tag=='pml' else 'full 0.25 degree window',
              'annual_aggregation':'8-day flux times actual days, then annual sum' if tag=='pml' else 'monthly temperature/u/v means and precipitation sum'},f'remote_cache/{tag}_metadata.json')
        img=image(tag)
        fs=geometries(tag)
        size=64 if tag=='pml' else 128
        for i,start in enumerate(range(0,len(fs),size)):
            if pilot and i>0:break
            jobs.append((tag,i,fs[start:start+size],img))
    result={tag:[] for tag in tags}
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks={pool.submit(fetch,*job):job[0] for job in jobs}
        for f in as_completed(tasks):result[tasks[f]].extend(f.result())
    for tag,rows in result.items():
        d=pd.DataFrame(rows).sort_values(['window_id']+(['zone'] if tag=='pml' else []))
        save(d,f'{tag}_annual'+('_pilot' if pilot else '')+'.csv')
        print('FINISHED',tag,len(d),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--only',choices=['pml','climate'])
    a=p.parse_args();main(a.pilot,a.only)
