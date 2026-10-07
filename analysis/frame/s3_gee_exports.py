"""S3: GEE-side exports — all GEE-hosted variables on the local-aligned grid.

Grid contract (identical to s2 local products):
  0.01 deg, lon -180..180, lat -60..84, 36000 x 14400 cells,
  crsTransform = [0.01, 0, -180, 0, -0.01, 84], EPSG:4326.

Exports (Drive folder GEE_PA_thermal):
  static  : pa_fraction (WDPA I-VI >1km2), forest_fraction (WorldCover Trees),
            elevation (SRTM mean)                          -> 1 image, 3 bands
  lst_day / lst_night : MYD11A2 QC-filtered annual means,
            23 bands (2003-2025), scaled int16 (K*50)      -> 2 images
  lai     : MOD15A2H annual mean, 23 bands, int16 (*100)   -> 1 image
  et      : MOD16A2GF annual sum, 23 bands, int16 (mm/yr)  -> 1 image

Annual bands are masked to cells with pa_fraction > 0 (PA cells only) to keep
files small; static exports globally.

Usage:
  python s3_gee_exports.py test     # small-region getInfo sanity checks
  python s3_gee_exports.py submit   # submit the 5 export tasks
"""
import os
import sys
import time

import ee

ee.Initialize(project=os.environ["EE_PROJECT"])

YEARS = list(range(2003, 2026))
CRS = "EPSG:4326"
TRANSFORM = [0.01, 0, -180, 0, -0.01, 84]
DIMENSIONS = "36000x14400"
FOLDER = "GEE_PA_thermal_v4"  # v4: Huang-aligned PA (IUCN I-VI) + CGLS forest class


# ---------- static layers ----------
def pa_fraction():
    """WDPA IUCN I-VI, area > 1 km2 — EXACTLY Huang 2026's PA definition
    (their Methods: "IUCN category of I to VI... We excluded 'not
    reported', 'not assigned' or 'not applicable'"). China is sparse in
    BOTH studies because Chinese WDPA records mostly lack I-VI labels.
    """
    wdpa = (ee.FeatureCollection("WCMC/WDPA/current/polygons")
            .filter(ee.Filter.inList("IUCN_CAT", ["Ia", "Ib", "II", "III",
                                                  "IV", "V", "VI"]))
            .filter(ee.Filter.gte("GIS_AREA", 1)))
    # paint at ~100 m then average to 0.01 deg -> fraction
    img = ee.Image(0).byte().paint(wdpa, 1).setDefaultProjection(
        crs=CRS, scale=100)
    return (img.reduceResolution(ee.Reducer.mean(), maxPixels=1024)
            .rename("pa_fraction").toFloat())


def forest_fraction():
    """Forest-CLASS fraction from a 100 m classification product
    (Copernicus CGLS-LC100 2019, classes 111-126 = closed+open forest).
    Aligns with Huang's ESA CCI "predominantly forest" semantics; the
    old WorldCover >=60% pure-tree-canopy rule was far stricter and cut
    83% of China's PA cells (agro-forest mosaics)."""
    lc = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019") \
        .select("discrete_classification")
    forest = lc.gte(111).And(lc.lte(126))
    return (forest.setDefaultProjection(crs=CRS, scale=100)
            .reduceResolution(ee.Reducer.mean(), maxPixels=1024)
            .rename("forest_fraction").toFloat())


def elevation():
    dem = ee.Image("USGS/SRTMGL1_003")
    return (dem.setDefaultProjection(crs=CRS, scale=30)
            .reduceResolution(ee.Reducer.mean(), maxPixels=4096)
            .rename("elevation").toFloat())


# ---------- annual variables ----------
def lst_annual(year, band, qc_band):
    """MYD11A2 annual mean of one LST band, QC-filtered, in Kelvin."""
    col = (ee.ImageCollection("MODIS/061/MYD11A2")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))

    def good(img):
        qa = img.select(qc_band)
        ok = qa.bitwiseAnd(3).lte(1)  # mandatory QA 0 or 1
        return img.select(band).multiply(0.02).updateMask(ok)

    return col.map(good).mean().rename(f"y{year}")


def lai_annual(year):
    col = (ee.ImageCollection("MODIS/061/MOD15A2H")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))

    def good(img):
        qa = img.select("FparLai_QC")
        ok = qa.bitwiseAnd(1).eq(0)  # MODLAND good quality
        return img.select("Lai_500m").multiply(0.1).updateMask(ok)

    return col.map(good).mean().rename(f"y{year}")


def et_annual(year):
    col = (ee.ImageCollection("MODIS/061/MOD16A2GF")
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    # ET scale 0.1 kg/m2/8day -> annual sum in mm
    def scaled(img):
        return img.select("ET").multiply(0.1)

    return col.map(scaled).sum().rename(f"y{year}")


def stack(builder, **kw):
    return ee.Image.cat([builder(y, **kw) for y in YEARS])


def submit(image, name, scale_factor=None, mask=None):
    img = image
    if mask is not None:
        img = img.updateMask(mask)
    if scale_factor:
        img = img.multiply(scale_factor).round().toInt16()
    task = ee.batch.Export.image.toDrive(
        image=img.unmask(-32768) if scale_factor else img,
        description=name,
        folder=FOLDER,
        fileNamePrefix=name,
        crs=CRS,
        crsTransform=TRANSFORM,
        dimensions=DIMENSIONS,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": False},
    )
    task.start()
    print(f"submitted: {name} ({task.id})", flush=True)
    return task


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    static_img = ee.Image.cat([pa_fraction(), forest_fraction(), elevation()])

    if mode == "test":
        # small window over an area with PAs + forest + city (S China)
        box = ee.Geometry.Rectangle([113.0, 23.0, 113.5, 23.5])
        stats = static_img.reduceRegion(
            ee.Reducer.mean(), box, crs=CRS, crsTransform=TRANSFORM,
            maxPixels=1e9, bestEffort=True).getInfo()
        print("static means (S China box):", stats)
        lst_test = lst_annual(2020, "LST_Day_1km", "QC_Day").reduceRegion(
            ee.Reducer.mean(), box, crs=CRS, crsTransform=TRANSFORM,
            maxPixels=1e9, bestEffort=True).getInfo()
        print("LST day 2020 mean (K):", lst_test)
        lai_test = lai_annual(2020).reduceRegion(
            ee.Reducer.mean(), box, crs=CRS, crsTransform=TRANSFORM,
            maxPixels=1e9, bestEffort=True).getInfo()
        print("LAI 2020 mean:", lai_test)
        et_test = et_annual(2020).reduceRegion(
            ee.Reducer.mean(), box, crs=CRS, crsTransform=TRANSFORM,
            maxPixels=1e9, bestEffort=True).getInfo()
        print("ET 2020 annual (mm):", et_test)
        return

    if mode == "submit":
        pa_mask = static_img.select("pa_fraction").gt(0)
        submit(static_img, "static_pa_forest_dem")
        submit(stack(lst_annual, band="LST_Day_1km", qc_band="QC_Day"),
               "lst_day_annual_2003_2025", scale_factor=50, mask=pa_mask)
        submit(stack(lst_annual, band="LST_Night_1km", qc_band="QC_Night"),
               "lst_night_annual_2003_2025", scale_factor=50, mask=pa_mask)
        submit(stack(lai_annual), "lai_annual_2003_2025",
               scale_factor=100, mask=pa_mask)
        submit(stack(et_annual), "et_annual_2003_2025",
               scale_factor=10, mask=pa_mask)
        print("5 tasks submitted; monitor with ee.batch.Task.list()", flush=True)


if __name__ == "__main__":
    main()
