"""S5b: Window-level paired differencing — the master cross-section table.

Grid: 0.01 deg, 36000 x 14400. Windows: 0.25 deg -> 1440 x 576 windows (25x25).

For every window and every zone class (1: PUI<=5km ... 4: 25-50km,
5: background >50km), computes the zone-mean of each variable, then forms
paired differences Delta = zone_k - zone_5 within the same window.

Variables:
  UP      2003-2025  (data/up/UP_*.tif, uint16/1e4)
  LST_day 2003-2025  (gee_exports vrt, int16/50, K)
  LST_night          (int16/50, K)
  LAI                (int16/100)
  ET                 (int16/10, mm/yr)
  elevation, forest_fraction (static)

Outputs (data/tables/):
  window_zone_means.parquet  long: window_id, zone, var, year, mean, n
  master_delta_5km.csv       wide: one row per window with zone1&zone5 valid,
                             all Delta variables + statics + biome
Usage: python s5b_pairing.py
"""
import os

import numpy as np
import pandas as pd
import rasterio

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W, H = 36000, 14400
WIN = 25                      # 0.25 deg
NWX, NWY = W // WIN, H // WIN  # 1440 x 576
YEARS = list(range(2003, 2026))
OUT = os.path.join(BASE, "data", "tables")
MIN_CELLS = 5
LAT_TOP = 84.0


def win_reduce(arr, valid):
    """Sum of arr and count of valid over 25x25 blocks -> (576,1440) each."""
    a = np.where(valid, arr, 0.0).reshape(NWY, WIN, NWX, WIN)
    v = valid.reshape(NWY, WIN, NWX, WIN)
    return a.sum(axis=(1, 3)), v.sum(axis=(1, 3))


def main():
    os.makedirs(OUT, exist_ok=True)
    zpath = os.path.join(BASE, "data", "zones", "zones_001deg.tif")
    with rasterio.open(zpath) as ds:
        zones = ds.read(1)

    zone_masks = {z: zones == z for z in range(1, 6)}
    counts = {z: zone_masks[z].reshape(NWY, WIN, NWX, WIN).sum(axis=(1, 3))
              for z in range(1, 6)}

    records = []   # (var, year, zone) -> mean grids kept sparse via records

    def add_variable(name, year, arr):
        for z in range(1, 6):
            s, n = win_reduce(arr, zone_masks[z] & np.isfinite(arr))
            with np.errstate(invalid="ignore", divide="ignore"):
                # keep every window with >=1 cell; MIN_CELLS is enforced later
                # at pairing stage on the COMBINED background, so zone means
                # must stay exactly recombinable via n-weighting
                m = np.where(n >= 1, s / np.maximum(n, 1), np.nan)
            records.append((name, year, z, m.astype("float32"), n))
        print(f"  {name} {year} done", flush=True)

    # ---- UP (local products) ----
    for y in YEARS:
        with rasterio.open(os.path.join(BASE, "data", "up", f"UP_{y}.tif")) as ds:
            up = ds.read(1).astype("float32") / 10000.0
        add_variable("UP", y, up)
        del up

    # ---- GEE stacks ----
    stacks = [("LSTd", "lst_day_annual_2003_2025.vrt", 50.0),
              ("LSTn", "lst_night_annual_2003_2025.vrt", 50.0),
              ("LAI", "lai_annual_2003_2025.vrt", 100.0),
              ("ET", "et_annual_2003_2025.vrt", 10.0)]
    for name, fn, sc in stacks:
        with rasterio.open(os.path.join(BASE, "data", "gee_exports", fn)) as ds:
            for bi, y in enumerate(YEARS, start=1):
                a = ds.read(bi).astype("float32")
                a[a == -32768] = np.nan
                a /= sc
                add_variable(name, y, a)
                del a

    # ---- statics ----
    with rasterio.open(os.path.join(BASE, "data", "gee_exports",
                                    "static_pa_forest_dem.vrt")) as ds:
        add_variable("FOREST", 0, ds.read(2).astype("float32"))
        add_variable("ELEV", 0, ds.read(3).astype("float32"))

    # ---- long table ----
    rows = []
    wy, wx = np.mgrid[0:NWY, 0:NWX]
    lat_c = LAT_TOP - (wy + 0.5) * 0.25
    for name, year, z, m, n in records:
        sel = np.isfinite(m)
        if not sel.any():
            continue
        rows.append(pd.DataFrame({
            "win_y": wy[sel], "win_x": wx[sel], "zone": z,
            "var": name, "year": year,
            "mean": m[sel], "n": n[sel].astype("int32")}))
    long = pd.concat(rows, ignore_index=True)
    long.to_parquet(os.path.join(OUT, "window_zone_means.parquet"))
    print(f"long table: {len(long):,} rows", flush=True)

    # ---- master delta table: PUI (<=5 km, zone 1) vs background (>5 km,
    # zones 2-5 n-weighted combination inside the SAME window) — Huang's
    # actual definition: background = the PA portion outside the buffer.
    pm = long.pivot_table(index=["win_y", "win_x"],
                          columns=["var", "year", "zone"], values="mean")
    pn = long.pivot_table(index=["win_y", "win_x"],
                          columns=["var", "year", "zone"], values="n")

    def zone_mean(name, year, z):
        m, n = pm.get((name, year, z)), pn.get((name, year, z))
        return (m, n) if m is not None else (None, None)

    def combined_bak(name, year):
        s = None
        n_tot = None
        for z in (2, 3, 4, 5):
            m, n = zone_mean(name, year, z)
            if m is None:
                continue
            part = (m * n).fillna(0.0)
            cnt = n.where(m.notna(), 0).fillna(0.0)
            s = part if s is None else s + part
            n_tot = cnt if n_tot is None else n_tot + cnt
        if s is None:
            return None, None
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = s / n_tot.replace(0, np.nan)
        return mean, n_tot

    idx = pm.index
    lat_center = LAT_TOP - (idx.get_level_values(0).values + 0.5) * 0.25
    lon_center = -180 + (idx.get_level_values(1).values + 0.5) * 0.25
    master = pd.DataFrame({"lat": lat_center, "lon": lon_center}, index=idx)

    for name in ["UP", "LSTd", "LSTn", "LAI", "ET"]:
        for y in YEARS:
            m1, n1 = zone_mean(name, y, 1)
            mb, nb = combined_bak(name, y)
            if m1 is None or mb is None:
                continue
            valid = (n1 >= MIN_CELLS) & (nb >= MIN_CELLS)
            master[f"d{name}_{y}"] = (m1 - mb).where(valid)
    for name in ["ELEV", "FOREST"]:
        m1, n1 = zone_mean(name, 0, 1)
        mb, nb = combined_bak(name, 0)
        if m1 is None or mb is None:
            continue
        valid = (n1 >= MIN_CELLS) & (nb >= MIN_CELLS)
        master[f"d{name}"] = (m1 - mb).where(valid)
        master[f"{name}_pui"] = m1.where(valid)
        master[f"{name}_bak"] = mb.where(valid)
    # validity: paired in the latest epoch + elevation constraint
    has_pair = master[[f"dUP_{y}" for y in (2021, 2022, 2023, 2024, 2025)]].notna().any(axis=1)
    elev_ok = master["dELEV"].abs() <= 500
    master = master[has_pair & elev_ok].reset_index()
    # biome bands (window center latitude)
    lat = master["lat"]
    master["biome"] = np.select(
        [(lat >= 50) & (lat <= 70), (lat >= 25) & (lat < 50),
         (lat > -25) & (lat < 25)],
        ["boreal", "temperate", "tropical"], default="other")
    master.to_csv(os.path.join(OUT, "master_delta_5km.csv"), index=False)
    print(f"master table: {len(master):,} windows | biomes:", flush=True)
    print(master["biome"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
