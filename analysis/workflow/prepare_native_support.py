"""Freeze an additional 5-10 km cell union for the already selected strict sample."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np,rasterio
from rasterio.features import shapes
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=HERE/'native_support';OUT.mkdir(exist_ok=True)
p=ROOT/'data/geometry/frozen_zones.geojson'
d=json.loads(p.read_text(encoding='utf-8'))
main=pd.read_csv(ROOT/'data/longterm/longterm_windows.csv')
strict=pd.read_csv(ROOT/'data/longterm/matched_longterm_windows.csv')
ids=set(main.window_id)
fs=[f for f in d['features'] if f['properties']['window_id'] in ids]
assert len(fs)==4208 and set(strict.window_id).issubset(ids)
with rasterio.open(ROOT/'data/zones/zones_001deg.tif') as src:
    for row in strict.itertuples():
        w=Window(int(row.win_x)*25,int(row.win_y)*25,25,25)
        z=src.read(1,window=w); mask=z==2
        assert mask.sum()>=5
        coords=[g['coordinates'] for g,v in shapes(mask.astype('uint8'),mask=mask,transform=src.window_transform(w)) if v==1]
        fs.append({'type':'Feature','geometry':{'type':'MultiPolygon','coordinates':coords},
           'properties':{'window_id':row.window_id,'zone':'outer','local_n':int(mask.sum())}})
o=OUT/'frozen_zones_with_outer.geojson'
o.write_text(json.dumps({'type':'FeatureCollection','features':fs},separators=(',',':')),encoding='utf-8')
(OUT/'geometry_provenance.json').write_text(json.dumps({'original_frozen_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
    'original_main_features':4208,'added_outer_features':len(strict),'extended_sha256':hashlib.sha256(o.read_bytes()).hexdigest(),
    'scope':'Original long-record cohort only; strict-ring eligibility from the existing static design; no new locations'},indent=2),encoding='utf-8')
print(len(fs),o)
