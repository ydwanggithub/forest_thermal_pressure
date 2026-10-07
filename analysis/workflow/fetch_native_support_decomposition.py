"""Factorial re-aggregation of the native-quality MODIS observations.

Purpose: separate the components that the concurrent, quarter-equal sensitivity
changes jointly (contributing cells, concurrent composite dates, quarter weights,
and the nighttime-validity requirement imposed on the daytime analysis), and
provide calendar-quarter means for a seasonal check.

Same products, quality screens, native-grid evaluation, common-grid transfer,
near/reference cell unions and years as fetch_native_observation_support.py.
Nothing in native_support/ is read for writing or overwritten. Two anchor groups
must reproduce native_support/support_all.csv exactly:
  *_orig (independent annual means)  ->  *_original_mean
  quarter-equal mean of *_q{q}_J_bal ->  *_balanced_mean
"""
from pathlib import Path
import argparse, json, hashlib, inspect
import fetch_observation_support as old
import fetch_native_observation_support as nat

ee = old.ee
HERE = Path(__file__).resolve().parent
OUT = HERE / 'native_decomposition'; OUT.mkdir(exist_ok=True)
GEOM = HERE / 'native_support' / 'frozen_zones_with_outer.geojson'


def sources():
    fs = json.loads(GEOM.read_text(encoding='utf-8'))['features']
    fs = [f for f in fs if f['properties']['zone'] in ('pui', 'bak')]
    assert len(fs) == 4208
    return fs, GEOM


def quarter(c, q):
    return c.filter(ee.Filter.calendarRange(q * 3 + 1, q * 3 + 3, 'month'))


def image(year):
    dates = (f'{year}-01-01', f'{year + 1}-01-01')
    lc = ee.ImageCollection('MODIS/061/MOD15A2H').filterDate(*dates)
    tc = ee.ImageCollection('MODIS/061/MYD11A2').filterDate(*dates)

    # Quality screens copied verbatim from fetch_native_observation_support.image
    def laigood(i):
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

    def concurrent(i):  # LAI, day and night all valid in the cell on this composite date
        li = ee.Image(l.filter(ee.Filter.eq('system:time_start', i.get('system:time_start'))).first())
        both = i.addBands(li)
        return both.updateMask(both.mask().reduce(ee.Reducer.min())).copyProperties(i, ['system:time_start'])

    def concurrent_day(i):  # LAI and day valid; night validity not required
        li = ee.Image(l.filter(ee.Filter.eq('system:time_start', i.get('system:time_start'))).first())
        both = i.select('day').addBands(li)
        return both.updateMask(both.mask().reduce(ee.Reducer.min())).copyProperties(i, ['system:time_start'])

    joint = shared.map(concurrent)
    joint_d = shared.map(concurrent_day)

    one = ee.Image.constant(1)
    mbal = ee.ImageCollection.fromImages(
        [quarter(joint, q).select('LAI').count().unmask(0).gte(1) for q in range(4)]).min()
    mbal_d = ee.ImageCollection.fromImages(
        [quarter(joint_d, q).select('LAI').count().unmask(0).gte(1) for q in range(4)]).min()

    var = [('LAI', l), ('day', t.select('day')), ('night', t.select('night'))]
    bands = []
    # Anchors and masks
    for name, c in var:
        bands.append(c.mean().rename(f'{name}_orig'))
    bands.append(one.updateMask(mbal).rename('mbal'))
    bands.append(one.updateMask(mbal_d).rename('mbalD'))
    # Annual constructions on the shared quarter-balanced cell set
    for name, c in var:
        bands.append(c.mean().updateMask(mbal).rename(f'{name}_ind_bal'))
    bands.append(joint.mean().updateMask(mbal).select(['LAI', 'day', 'night'], ['LAI_Jc_bal', 'day_Jc_bal', 'night_Jc_bal']))
    # Daytime-only concurrence (night validity not required)
    bands.append(joint_d.mean().updateMask(mbal_d).select(['LAI', 'day'], ['LAI_JDc_balD', 'day_JDc_balD']))
    for name, c in var[:2]:
        bands.append(c.mean().updateMask(mbal_d).rename(f'{name}_ind_balD'))
    for q in range(4):
        k = q + 1
        jq = quarter(joint, q); dq = quarter(joint_d, q)
        for name, c in var:
            cq = quarter(c, q)
            bands.append(cq.mean().updateMask(mbal).rename(f'{name}_q{k}_ind_bal'))
            bands.append(cq.mean().rename(f'{name}_q{k}_ind'))
        bands.append(jq.mean().updateMask(mbal).select(['LAI', 'day', 'night'],
                     [f'LAI_q{k}_J_bal', f'day_q{k}_J_bal', f'night_q{k}_J_bal']))
        bands.append(dq.mean().updateMask(mbal).select(['LAI', 'day'], [f'LAI_q{k}_JD_bal', f'day_q{k}_JD_bal']))
        bands.append(dq.mean().updateMask(mbal_d).select(['LAI', 'day'], [f'LAI_q{k}_JD_balD', f'day_q{k}_JD_balD']))
        bands.append(dq.select('LAI').count().unmask(0).rename(f'JD_q{k}n'))
    bands.append(one.rename('grid_check'))
    return ee.Image.cat(bands)


def fingerprint():
    return {'pipeline': 'native-quality-decomposition-v1',
            'processing_sha256': hashlib.sha256((inspect.getsource(nat.common_grid) + inspect.getsource(image)
                                                 + inspect.getsource(old.fetch)).encode()).hexdigest(),
            'support_sha256': hashlib.sha256(GEOM.read_bytes()).hexdigest(),
            'crs': old.CRS, 'transform': old.TRANSFORM, 'batch_size': 256}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pilot', action='store_true'); a = p.parse_args()
    old.OUT = OUT; old.CACHE = OUT / 'cache'; old.CACHE.mkdir(exist_ok=True)
    version = OUT / 'cache_version.json'; current = fingerprint()
    if version.exists():
        assert json.loads(version.read_text()) == current, 'Processing changed; use a new output directory'
    else:
        assert not list(old.CACHE.glob('*.json')), 'Existing cache lacks a processing-version record'
        version.write_text(json.dumps(current, indent=2), encoding='utf-8')
    old.sources = sources; old.image = image
    old.main(pilot=a.pilot)
