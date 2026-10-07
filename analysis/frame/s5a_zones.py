"""S5a: Build the analysis-zone raster — PUI / background-PA classification.

Inputs (all on the 0.01-deg grid, 36000 x 14400, lon -180..180, lat -60..84):
  data/gee_exports/static_pa_forest_dem.vrt  (pa_fraction, forest_fraction, dem)
  data/gub/GUB_2018/GUB_Global_2018.shp      (urban polygons, EPSG:4326)

Outputs (data/zones/):
  urban_2018_001deg.tif   uint8   urban presence (any GUB polygon touches cell)
  dist_km_001deg.tif      float32 distance to nearest urban cell, km
                                  (latitude-banded EDT, 10-deg strips + 1-deg halo)
  zones_001deg.tif        uint8   0=none; PA-forest cells: 1=PUI<=5km, 2=<=10,
                                  3=<=25, 4=<=50, 5=background PA (>50km)

Definitions (Huang-faithful):
  PA cell     : pa_fraction >= 0.99  ("pure" PA pixel)
  forest cell : forest_fraction >= 0.50 (v4: CGLS forest-CLASS majority,
                Huang's "predominantly forest" semantics)
Usage: python s5a_zones.py
"""
import os

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRID_W, GRID_H = 36000, 14400
RES = 0.01
LAT_TOP = 84.0
TRANSFORM = from_origin(-180.0, LAT_TOP, RES, RES)
OUT = os.path.join(BASE, "data", "zones")
CELL_M = 111_320 * RES  # N-S cell size in meters (~1113 m)


def rasterize_urban():
    import fiona
    out = os.path.join(OUT, "urban_2018_001deg.tif")
    if os.path.exists(out):
        print("urban raster exists, skip", flush=True)
        with rasterio.open(out) as ds:
            return ds.read(1).astype(bool)
    shp = os.path.join(BASE, "data", "gub", "GUB_2018", "GUB_Global_2018.shp")
    print("rasterizing GUB 2018 ...", flush=True)
    with fiona.open(shp) as src:
        shapes = ((f["geometry"], 1) for f in src)
        arr = features.rasterize(
            shapes, out_shape=(GRID_H, GRID_W), transform=TRANSFORM,
            fill=0, dtype="uint8", all_touched=True)
    prof = dict(driver="GTiff", width=GRID_W, height=GRID_H, count=1,
                dtype="uint8", crs="EPSG:4326", transform=TRANSFORM,
                compress="deflate", tiled=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr, 1)
    print(f"urban cells: {arr.sum():,}", flush=True)
    return arr.astype(bool)


def distance_km(urban):
    """Latitude-banded EDT: 10-deg strips with 1-deg halo, per-strip dx."""
    out = os.path.join(OUT, "dist_km_001deg.tif")
    if os.path.exists(out):
        print("distance raster exists, skip", flush=True)
        with rasterio.open(out) as ds:
            return ds.read(1)
    print("computing distance transform ...", flush=True)
    dist = np.empty((GRID_H, GRID_W), dtype="float32")
    band_rows = 1000   # 10 deg
    halo = 100         # 1 deg  (>= 50 km at any latitude)
    for r0 in range(0, GRID_H, band_rows):
        r1 = min(r0 + band_rows, GRID_H)
        h0, h1 = max(0, r0 - halo), min(GRID_H, r1 + halo)
        mid_lat = LAT_TOP - RES * (h0 + h1) / 2
        dx = CELL_M * max(np.cos(np.radians(mid_lat)), 0.05)
        strip = urban[h0:h1]
        d = distance_transform_edt(~strip, sampling=(CELL_M, dx))
        dist[r0:r1] = (d[(r0 - h0):(r0 - h0) + (r1 - r0)] / 1000.0)
        print(f"  rows {r0}-{r1} (midlat {mid_lat:.0f}) done", flush=True)
    prof = dict(driver="GTiff", width=GRID_W, height=GRID_H, count=1,
                dtype="float32", crs="EPSG:4326", transform=TRANSFORM,
                compress="deflate", tiled=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(dist, 1)
    return dist


def main():
    os.makedirs(OUT, exist_ok=True)
    urban = rasterize_urban()
    dist = distance_km(urban)

    print("classifying zones ...", flush=True)
    with rasterio.open(os.path.join(
            BASE, "data", "gee_exports", "static_pa_forest_dem.vrt")) as ds:
        pa = ds.read(1)
        forest = ds.read(2)

    paf = (pa >= 0.99) & (forest >= 0.50) & (~urban)
    zones = np.zeros((GRID_H, GRID_W), dtype="uint8")
    zones[paf & (dist <= 5)] = 1
    zones[paf & (dist > 5) & (dist <= 10)] = 2
    zones[paf & (dist > 10) & (dist <= 25)] = 3
    zones[paf & (dist > 25) & (dist <= 50)] = 4
    zones[paf & (dist > 50)] = 5

    prof = dict(driver="GTiff", width=GRID_W, height=GRID_H, count=1,
                dtype="uint8", crs="EPSG:4326", transform=TRANSFORM,
                compress="deflate", tiled=True)
    with rasterio.open(os.path.join(OUT, "zones_001deg.tif"), "w", **prof) as dst:
        dst.write(zones, 1)

    labels = {1: "PUI<=5km", 2: "5-10km", 3: "10-25km", 4: "25-50km",
              5: "background>50km"}
    total_pa_forest = int(paf.sum())
    print(f"PA-forest cells total: {total_pa_forest:,}")
    for v, lab in labels.items():
        n = int((zones == v).sum())
        print(f"  zone {v} ({lab}): {n:,} ({100*n/max(total_pa_forest,1):.1f}%)")


if __name__ == "__main__":
    main()
