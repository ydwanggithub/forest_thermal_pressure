"""Rebuild MODIS observations with quality masks evaluated at native resolution.

The explicit native reprojection prevents continuous-value and categorical-QA
pyramids from being read at different effective supports. The common-grid transfer
then uses nearest neighbour, as intended in the original design. Original outputs
are retained. This implementation also extracts the 5-10 km strict-ring support.
"""
from pathlib import Path
import argparse,json,hashlib,inspect
import fetch_observation_support as old
ee=old.ee
HERE=Path(__file__).resolve().parent
old.ROOT=HERE.parents[1]
OUT=HERE/'native_support';OUT.mkdir(exist_ok=True)

def sources():
    p=OUT/'frozen_zones_with_outer.geojson'
    fs=json.loads(p.read_text(encoding='utf-8'))['features']
    assert len(fs)==5334
    return fs,p

def common_grid(v,quality):
    # Force the numeric value AND all QA/range checks at the source native grid.
    # Merely calling updateMask without this step leaves scale-dependent pyramids.
    return (v.updateMask(quality).reproject(v.projection())
            .reproject(crs=old.CRS,crsTransform=old.TRANSFORM))

def image(year):
    dates=(f'{year}-01-01',f'{year+1}-01-01')
    lc=ee.ImageCollection('MODIS/061/MOD15A2H').filterDate(*dates)
    tc=ee.ImageCollection('MODIS/061/MYD11A2').filterDate(*dates)
    ec=ee.ImageCollection('MODIS/061/MOD16A2GF').filterDate(*dates)
    def laigood(i):
        v=i.select('Lai_500m')
        quality=i.select('FparLai_QC').bitwiseAnd(1).eq(0).And(v.gte(0)).And(v.lte(100))
        return common_grid(v,quality).multiply(.1).rename('LAI').copyProperties(i,['system:time_start'])
    def tgood(i):
        ds=[]
        for band,qb,name in [('LST_Day_1km','QC_Day','day'),('LST_Night_1km','QC_Night','night')]:
            v=i.select(band)
            quality=i.select(qb).bitwiseAnd(3).lte(1).And(v.gte(7500)).And(v.lte(65535))
            ds.append(common_grid(v,quality).multiply(.02).rename(name))
        return ds[0].addBands(ds[1]).copyProperties(i,['system:time_start'])
    def egood(i):
        v=i.select('ET')
        return common_grid(v,v.gte(-32767).And(v.lte(32700))).multiply(.1).rename('ET').copyProperties(i,['system:time_start'])
    l=lc.map(laigood);t=tc.map(tgood);e=ec.map(egood)
    def concurrent(i):
        li=ee.Image(l.filter(ee.Filter.eq('system:time_start',i.get('system:time_start'))).first())
        both=i.addBands(li)
        return both.updateMask(both.mask().reduce(ee.Reducer.min())).copyProperties(i,['system:time_start'])
    joint=t.filter(ee.Filter.inList('system:time_start',l.aggregate_array('system:time_start'))).map(concurrent)
    bands=[];qmeans=[];qvalid=[]
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
    # Select by name, avoiding reliance on ImageCollection band ordering.
    bands.append(balanced.select(['day','night','LAI'],['day_balanced','night_balanced','LAI_balanced']))
    bands.append(e.sum().rename('ET_original'))
    bands.append(e.count().unmask(0).rename('ET_n'))
    bands.append(ee.Image.constant(1).rename('grid_check'))
    return ee.Image.cat(bands)

def fingerprint():
    return {'pipeline':'native-quality-common-grid-v1',
      'processing_sha256':hashlib.sha256((inspect.getsource(common_grid)+inspect.getsource(image)+inspect.getsource(old.fetch)).encode()).hexdigest(),
      'support_sha256':hashlib.sha256((OUT/'frozen_zones_with_outer.geojson').read_bytes()).hexdigest(),
      'crs':old.CRS,'transform':old.TRANSFORM,'batch_size':256}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args()
    old.OUT=OUT;old.CACHE=OUT/'cache';old.CACHE.mkdir(exist_ok=True)
    version=OUT/'cache_version.json';current=fingerprint()
    if version.exists():assert json.loads(version.read_text())==current,'Cache processing version changed; use a new output directory'
    else:
        assert not list(old.CACHE.glob('*.json')),'Existing cache must have a verified processing-version record'
        version.write_text(json.dumps(current,indent=2),encoding='utf-8')
    old.sources=sources;old.image=image
    old.main(pilot=a.pilot)
    mp=OUT/'collection_metadata.json';meta=json.loads(mp.read_text())
    meta.update({'resampling':'Native-grid quality and valid-range checks forced by explicit native reprojection, followed by nearest-neighbor transfer to original common-grid centers',
      'spatial_support':'4,208 original near/reference cell unions plus 1,126 original 5-10 km ring cell unions',
      'valid_ranges_raw':{'Lai_500m':[0,100],'LST_Day_1km':[7500,65535],'LST_Night_1km':[7500,65535],'ET':[-32767,32700]},
      'correction_reason':'Direct native-versus-default-pyramid probe found out-of-range LAI codes at common scale that were absent at native scale; numeric and QA pyramids must not be treated as equivalent source retrievals',
      'LAI_QA':'MODLAND bit 0 = 0','LST_QA':'mandatory bits 0-1 = 0 or 1; retains original quality threshold','ET_QA':'Supplied masking and valid numeric range; no additional quality-bit screen'})
    mp.write_text(json.dumps(meta,indent=2),encoding='utf-8')
