# -*- coding: utf-8 -*-
"""Annual forest loss in the near and reference forest portions (Hansen Global Forest Change v1.13).

Disturbance check for the within-window models: does the annual canopy-temperature association
depend on disturbance years? For each of the 4,208 frozen portion geometries, the 30 m lossyear
band (1 to 25 = 2001 to 2025) is tabulated over all pixels and over pixels with treecover2000 >= 30%.
Output native_robustness/forest_loss.csv: window_id, zone, n_pixels, n_tree, and per-year loss
fractions loss_YYYY (share of all pixels) and treeloss_YYYY (share of tree-covered pixels).
"""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, time, hashlib, argparse
import pandas as pd
import fetch_observation_support as old

ee = old.ee
HERE = Path(__file__).resolve().parent
OUT = HERE / 'native_robustness'; CACHE = OUT / 'forest_loss_cache'; CACHE.mkdir(parents=True, exist_ok=True)
GEOM = HERE / 'native_support' / 'frozen_zones_with_outer.geojson'
PROJECT = os.environ.get('EE_PROJECT')  # your Earth Engine project
ASSET = 'UMD/hansen/global_forest_change_2025_v1_13'
BATCH = 128


def sources():
    fs = json.loads(GEOM.read_text(encoding='utf-8'))['features']
    fs = [f for f in fs if f['properties']['zone'] in ('pui', 'bak')]
    assert len(fs) == 4208
    return fs


def image():
    h = ee.Image(ASSET)
    loss = h.select('lossyear').unmask(0).rename('loss')
    tree = h.select('treecover2000').gte(30)
    treeloss = loss.updateMask(tree).rename('treeloss')
    return loss.addBands(treeloss), h.select('lossyear').projection()


def fetch(i, fs, img, proj):
    p = CACHE / f'b{i:03d}.json'
    if p.exists():
        rows = json.loads(p.read_text())
        assert [(r['window_id'], r['zone']) for r in rows] == [(f['properties']['window_id'], f['properties']['zone']) for f in fs]
        return rows
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f['geometry'], proj=old.CRS, geodesic=False), f['properties']) for f in fs])
    query = img.reduceRegions(collection=fc, reducer=ee.Reducer.frequencyHistogram(), crs=proj, tileScale=4)
    query = query.map(lambda f: ee.Feature(None, f.toDictionary()))
    t0 = time.time()
    rows = [f['properties'] for f in query.getInfo()['features']]
    assert len(rows) == len(fs)
    p.write_text(json.dumps(rows, separators=(',', ':')), encoding='utf-8')
    print('DONE', i, len(rows), 'seconds', round(time.time() - t0, 1), flush=True)
    return rows


def tidy(rows):
    out = []
    for r in rows:
        loss = {int(float(k)): v for k, v in (r.get('loss') or {}).items()}
        tl = {int(float(k)): v for k, v in (r.get('treeloss') or {}).items()}
        n, nt = sum(loss.values()), sum(tl.values())
        rec = dict(window_id=r['window_id'], zone=r['zone'], local_n=r['local_n'], n_pixels=n, n_tree=nt)
        for k in range(1, 26):
            rec[f'loss_{2000 + k}'] = loss.get(k, 0) / n if n else float('nan')
            rec[f'treeloss_{2000 + k}'] = tl.get(k, 0) / nt if nt else float('nan')
        out.append(rec)
    return pd.DataFrame(out)


def main(pilot=False):
    version = OUT / 'forest_loss_version.json'
    current = {'asset': ASSET, 'batch': BATCH, 'support_sha256': hashlib.sha256(GEOM.read_bytes()).hexdigest(), 'project': PROJECT}
    if version.exists():
        assert json.loads(version.read_text()) == current
    else:
        version.write_text(json.dumps(current, indent=2), encoding='utf-8')
    ee.Initialize(project=PROJECT); ee.data.setDeadline(180000)
    fs = sources(); img, proj = image()
    jobs = [(0, fs[:8])] if pilot else [(k // BATCH, fs[k:k + BATCH]) for k in range(0, len(fs), BATCH)]
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch, i, f, img, proj): i for i, f in jobs}
        for f in as_completed(pending):
            try:
                rows.extend(f.result())
            except Exception as e:
                errors.append(dict(job=pending[f], type=type(e).__name__, message=str(e)[:600]))
                print('FAILED', errors[-1], flush=True)
    if errors:
        (OUT / 'forest_loss_errors.json').write_text(json.dumps(errors, indent=2), encoding='utf-8')
        raise RuntimeError(f'{len(errors)} requests failed; cache retained')
    d = tidy(rows).sort_values(['window_id', 'zone'])
    if pilot:
        print(d.iloc[:, :8].to_string()); (CACHE / 'b000.json').unlink()
        return
    d.to_csv(OUT / 'forest_loss.csv', index=False)
    print('COMPLETE', len(d), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--pilot', action='store_true')
    main(a.parse_args().pilot)
