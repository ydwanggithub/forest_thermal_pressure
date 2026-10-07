"""Composite-date zone means for the 960-window observation-support sample.

For each near/reference forest portion, year and MODIS composite start date, this
extracts the mean LAI, daytime LST and nighttime LST over cells in which all
three are valid on that date (J), and the mean LAI and daytime LST over cells in
which LAI and daytime LST are valid (JD), with the corresponding cell counts.
It allows annual means to be rebuilt from dates shared by both forest portions.

Same products, native-grid QA, common-grid transfer, frozen cell unions and
years as fetch_native_support_decomposition.py; nothing existing is overwritten.
Anchor: summed J cell counts over dates equal sum_q(joint_q{q}n_mean) x local_n
import os
from native_support/support_all.csv for every portion-year.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, hashlib, inspect, argparse
import pandas as pd
import fetch_observation_support as old
import fetch_native_observation_support as nat

ee = old.ee
HERE = Path(__file__).resolve().parent
OUT = HERE / 'native_dates'; CACHE = OUT / 'cache'; CACHE.mkdir(parents=True, exist_ok=True)
GEOM = HERE / 'native_support' / 'frozen_zones_with_outer.geojson'
PROJECT = os.environ.get('EE_PROJECT')  # your Earth Engine project
BATCH = 128


def sources():
    ids = set(pd.read_csv(HERE / 'native_support/quarter_balanced_annual_panel.csv', usecols=['window_id']).window_id)
    fs = json.loads(GEOM.read_text(encoding='utf-8'))['features']
    fs = [f for f in fs if f['properties']['zone'] in ('pui', 'bak') and f['properties']['window_id'] in ids]
    assert len(fs) == 2 * len(ids) == 1920
    return fs


def image(year):
    dates = (f'{year}-01-01', f'{year + 1}-01-01')
    lc = ee.ImageCollection('MODIS/061/MOD15A2H').filterDate(*dates)
    tc = ee.ImageCollection('MODIS/061/MYD11A2').filterDate(*dates)

    def laigood(i):  # identical to fetch_native_support_decomposition.image
        v = i.select('Lai_500m')
        quality = i.select('FparLai_QC').bitwiseAnd(1).eq(0).And(v.gte(0)).And(v.lte(100))
        return nat.common_grid(v, quality).multiply(.1).rename('LAI').copyProperties(i, ['system:time_start'])

    def tgood(i):
        ds = []
        for band, qb, name in [('LST_Day_1km', 'QC_Day', 'day'), ('LST_Night_1km', 'QC_Night', 'night')]:
            v = i.select(band)
            quality = i.select(qb).bitwiseAnd(3).lte(1).And(v.gte(7500)).And(v.lte(65535))
            ds.append(nat.common_grid(v, quality).multiply(.02).rename(name))
        return ds[0].addBands(ds[1]).copyProperties(i, ['system:time_start'])

    l = lc.map(laigood); t = tc.map(tgood)
    shared = t.filter(ee.Filter.inList('system:time_start', l.aggregate_array('system:time_start')))

    def concurrent(i):
        li = ee.Image(l.filter(ee.Filter.eq('system:time_start', i.get('system:time_start'))).first())
        both = i.addBands(li)
        return both.updateMask(both.mask().reduce(ee.Reducer.min())).copyProperties(i, ['system:time_start'])

    def concurrent_day(i):
        li = ee.Image(l.filter(ee.Filter.eq('system:time_start', i.get('system:time_start'))).first())
        both = i.select('day').addBands(li)
        both = both.updateMask(both.mask().reduce(ee.Reducer.min()))
        return both.select(['day', 'LAI'], ['dayD', 'LAID']).copyProperties(i, ['system:time_start'])

    joint = shared.map(concurrent).toBands()        # '<YYYY_MM_DD>_day', '_night', '_LAI'
    joint_d = shared.map(concurrent_day).toBands()  # '<YYYY_MM_DD>_dayD', '_LAID'
    return joint.addBands(joint_d).addBands(ee.Image.constant(1).rename('grid_check'))


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
    return {'pipeline': 'native-quality-composite-dates-v1',
            'processing_sha256': hashlib.sha256((inspect.getsource(nat.common_grid) + inspect.getsource(image)
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
    d = to_long(rows)
    d.to_csv(OUT / ('dates_pilot.csv' if pilot else 'dates_all.csv'), index=False)
    print('COMPLETE', len(d), 'portion-year-date rows', flush=True)


def to_long(rows):
    """One row per portion, year and composite start date (dates differ among years)."""
    out = []
    for r in rows:
        dates = sorted({k[:10] for k in r if k[:4].isdigit() and k[4] == '_'})
        for dt in dates:
            rec = dict(window_id=r['window_id'], zone=r['zone'], year=r['year'], date=dt, local_n=r['local_n'])
            for v in ['LAI', 'day', 'night', 'LAID', 'dayD']:
                rec[v] = r.get(f'{dt}_{v}_mean'); rec[v + '_n'] = r.get(f'{dt}_{v}_count', 0)
            out.append(rec)
    return pd.DataFrame(out).sort_values(['window_id', 'zone', 'year', 'date'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pilot', action='store_true'); a = p.parse_args()
    main(a.pilot)
