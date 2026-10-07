# -*- coding: utf-8 -*-
"""Calendar-quarter albedo, snow and radiation for the near and reference forest portions.

Process evidence for the seasonal canopy-temperature associations. For all 2,104 candidate
windows (4,208 frozen portion geometries) and 2003 to 2025, each calendar quarter gets:
  LAI      MOD15A2H, MODLAND bit 0 = 0 (anchor: must equal native_decomposition LAI_q{q}_ind)
  LAIsf    the same LAI restricted to retrievals without the snow/ice flag (FparExtra_QC bit 2 = 0)
  LAIsnow  share of good LAI retrievals carrying the snow/ice flag
  alb      MCD43A3 white-sky shortwave albedo, mandatory quality 0 or 1, sampled at the centre
           (start + 4 days) of every LAI/LST composite period
  snowf    MOD10A2 maximum snow extent: share of valid 8-day periods (snow 200 / no snow 25) with snow
  sw       ERA5-Land downward surface solar radiation, W m-2 (monthly sums / seconds in month)
  snowera  ERA5-Land snow cover, percent (model-based; independent of the forest portions)
MODIS layers are screened at native resolution and transferred to the common 0.01 degree grid
by nearest neighbour, as in fetch_native_observation_support.common_grid.
"""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, hashlib, inspect, argparse
import pandas as pd
import fetch_observation_support as old
import fetch_native_observation_support as nat

ee = old.ee
HERE = Path(__file__).resolve().parent
OUT = HERE / 'native_process'; CACHE = OUT / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
GEOM = HERE / 'native_support' / 'frozen_zones_with_outer.geojson'
PROJECT = os.environ.get('EE_PROJECT')  # your Earth Engine project
BATCH = 256
DAY_MS = 86400000


def sources():
    fs = json.loads(GEOM.read_text(encoding='utf-8'))['features']
    fs = [f for f in fs if f['properties']['zone'] in ('pui', 'bak')]
    assert len(fs) == 4208
    return fs


def quarter(c, q):
    return c.filter(ee.Filter.calendarRange(q * 3 + 1, q * 3 + 3, 'month'))


def to_grid(img, proj):
    """Composite of native-grid images: restore the native projection, then nearest-neighbour transfer."""
    return img.setDefaultProjection(proj).reproject(proj).reproject(crs=old.CRS, crsTransform=old.TRANSFORM)


def image(year):
    dates = (f'{year}-01-01', f'{year + 1}-01-01')
    lc = ee.ImageCollection('MODIS/061/MOD15A2H').filterDate(*dates)
    starts = lc.aggregate_array('system:time_start')
    mid = starts.map(lambda t: ee.Number(t).add(4 * DAY_MS))
    ac = ee.ImageCollection('MODIS/061/MCD43A3').filterDate(*dates).filter(ee.Filter.inList('system:time_start', mid))
    sc = ee.ImageCollection('MODIS/061/MOD10A2').filterDate(*dates)
    era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate(*dates)
    p_modis = ee.Image(lc.first()).select('Lai_500m').projection()
    p_alb = ee.Image(ac.first()).select('Albedo_WSA_shortwave').projection()
    p_snow = ee.Image(sc.first()).select('Maximum_Snow_Extent').projection()
    p_era = ee.Image(era.first()).select('surface_solar_radiation_downwards_sum').projection()

    def lai(i):  # identical screen to the primary extraction, plus the snow/ice flag
        v = i.select('Lai_500m'); ex = i.select('FparExtra_QC')
        good = i.select('FparLai_QC').bitwiseAnd(1).eq(0).And(v.gte(0)).And(v.lte(100))
        a = nat.common_grid(v, good).multiply(.1).rename('LAI')
        sf = nat.common_grid(v, good.And(ex.bitwiseAnd(4).eq(0))).multiply(.1).rename('LAIsf')
        fl = nat.common_grid(ex.bitwiseAnd(4).neq(0), good).rename('LAIsnow')
        return a.addBands(sf).addBands(fl).copyProperties(i, ['system:time_start'])

    def alb(i):
        v = i.select('Albedo_WSA_shortwave')
        ok = i.select('BRDF_Albedo_Band_Mandatory_Quality_shortwave').lte(1).And(v.gte(0)).And(v.lte(1000))
        return v.multiply(.001).updateMask(ok).rename('alb').copyProperties(i, ['system:time_start'])

    def snow(i):
        v = i.select('Maximum_Snow_Extent')
        valid = v.eq(200).Or(v.eq(25))
        return v.eq(200).updateMask(valid).rename('snowf').copyProperties(i, ['system:time_start'])

    def era_q(i):
        secs = i.date().advance(1, 'month').difference(i.date(), 'second')
        sw = i.select('surface_solar_radiation_downwards_sum').divide(secs).rename('sw')
        return sw.addBands(i.select('snow_cover').rename('snowera')).copyProperties(i, ['system:time_start'])

    l = lc.map(lai); a = ac.map(alb); s = sc.map(snow); e = era.map(era_q)
    bands = []
    for q in range(4):
        k = q + 1
        lq = quarter(l, q)
        bands += [lq.select('LAI').mean().rename(f'LAI_q{k}'), lq.select('LAIsf').mean().rename(f'LAIsf_q{k}'),
                  lq.select('LAIsnow').mean().rename(f'LAIsnow_q{k}')]
        bands.append(to_grid(quarter(a, q).mean(), p_alb).rename(f'alb_q{k}'))
        bands.append(to_grid(quarter(s, q).mean(), p_snow).rename(f'snowf_q{k}'))
        eq = quarter(e, q).mean()
        bands.append(to_grid(eq.select('sw'), p_era).rename(f'sw_q{k}'))
        bands.append(to_grid(eq.select('snowera'), p_era).rename(f'snowera_q{k}'))
    bands.append(ee.Image.constant(1).rename('grid_check'))
    return ee.Image.cat(bands)


def fetch(year, i, fs, img):
    p = CACHE / f'y{year}_b{i:03d}.json'
    if p.exists():
        rows = json.loads(p.read_text())
        assert [(r['window_id'], r['zone']) for r in rows] == [(f['properties']['window_id'], f['properties']['zone']) for f in fs]
        return rows
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f['geometry'], proj=old.CRS, geodesic=False), f['properties']) for f in fs])
    reducer = ee.Reducer.mean().unweighted().combine(ee.Reducer.count(), sharedInputs=True)
    query = img.reduceRegions(collection=fc, reducer=reducer, crs=old.CRS, crsTransform=old.TRANSFORM, tileScale=4)
    query = query.map(lambda f: ee.Feature(None, f.toDictionary()))
    t0 = time.time()
    rows = [f['properties'] for f in query.getInfo()['features']]
    assert len(rows) == len(fs)
    for r in rows:
        assert r['grid_check_count'] == r['local_n'], (year, r['window_id'])
        r['year'] = year
    p.write_text(json.dumps(rows, separators=(',', ':')), encoding='utf-8')
    print('DONE', year, i, len(rows), 'seconds', round(time.time() - t0, 1), flush=True)
    return rows


def fingerprint():
    return {'pipeline': 'native-quality-process-quarterly-v1',
            'processing_sha256': hashlib.sha256((inspect.getsource(nat.common_grid) + inspect.getsource(image) + inspect.getsource(to_grid)
                                                 + inspect.getsource(fetch)).encode()).hexdigest(),
            'support_sha256': hashlib.sha256(GEOM.read_bytes()).hexdigest(),
            'crs': old.CRS, 'transform': old.TRANSFORM, 'batch_size': BATCH, 'project': PROJECT}


def main(pilot=False):
    version = OUT / 'cache_version.json'; current = fingerprint()
    if version.exists():
        assert json.loads(version.read_text()) == current, 'Processing changed; use a new output directory'
    else:
        assert not list(CACHE.glob('*.json')), 'Existing cache lacks a processing-version record'
        version.write_text(json.dumps(current, indent=2), encoding='utf-8')
    ee.Initialize(project=PROJECT); ee.data.setDeadline(180000)
    fs = sources()
    years = [2020] if pilot else old.YEARS
    jobs = []
    for y in years:
        img = image(y)
        for start in ([0] if pilot else range(0, len(fs), BATCH)):
            jobs.append((y, start // BATCH, fs[start:start + BATCH], img))
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch, *j): (j[0], j[1]) for j in jobs}
        for f in as_completed(pending):
            try:
                rows.extend(f.result())
            except Exception as e:
                errors.append(dict(job=pending[f], type=type(e).__name__, message=str(e)[:600]))
                print('FAILED', errors[-1], flush=True)
    if errors:
        (OUT / 'fetch_errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
        raise RuntimeError(f'{len(errors)} requests failed; successful cache retained')
    d = pd.DataFrame(rows).sort_values(['window_id', 'zone', 'year'])
    d.to_csv(OUT / ('process_pilot.csv' if pilot else 'process_all.csv'), index=False)
    print('COMPLETE', len(d), 'portion-year rows', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pilot', action='store_true'); a = p.parse_args()
    main(a.pilot)
