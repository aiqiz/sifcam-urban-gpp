"""Mosaic the Sentinel-2 tiles onto one 10 m grid per date, then build 8-day
NIRv composites and classify pixels (trend / background / invalid).

in:  <s2_dir>/<tile>/<year>/nirv_YYYY-MM-DD.tif
out: 0.NIRv_cloud_mask/<year>/nirv_YYYY-MM-DD.tif   (mosaic, one per date)
     1.NIRv_8day_year/nirv_inputs_<year>.npz
"""

import re
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

import utils as C

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

RES_M = 10.0
DATE_RE = re.compile(r"nirv_(\d{4}-\d{2}-\d{2})\.tif$")


def mosaic_year(year):
    """Every date onto the union grid of all tiles. Overlaps are averaged
    (nanmean), so a cloud in one tile gets filled by the other."""
    out_dir = C.DIR_NIRV_CLOUD / str(year)
    src_dirs = [C.s2_tile_dir(t, year) for t in C.REGION_CFG["s2_tiles"]]

    by_date = {}
    for d in src_dirs:
        for f in d.glob("nirv_*.tif") if d.exists() else []:
            m = DATE_RE.search(f.name)
            if m and not f.name.startswith("."):
                by_date.setdefault(m.group(1), []).append(f)
    dates = sorted(by_date)
    if not dates:
        print(f"{year}: no Sentinel-2 files found, run 1_s2_download.py first")
        return None

    # union grid
    bounds = []
    for d in src_dirs:
        fs = sorted(p for p in d.glob("nirv_*.tif") if not p.name.startswith(".")) if d.exists() else []
        if fs:
            with rasterio.open(fs[0]) as s:
                bounds.append(s.bounds)
    minx = min(b.left for b in bounds); maxx = max(b.right for b in bounds)
    miny = min(b.bottom for b in bounds); maxy = max(b.top for b in bounds)
    mid_lat = (miny + maxy) / 2
    px_lat = RES_M / 111320.0
    px_lon = RES_M / (111320.0 * np.cos(np.radians(mid_lat)))
    W = int(np.ceil((maxx - minx) / px_lon))
    H = int(np.ceil((maxy - miny) / px_lat))
    transform = from_origin(minx, maxy, px_lon, px_lat)

    if C.outputs_present(out_dir, "nirv_*.tif", len(dates)) and not C.FORCE_RERUN:
        print(f"{year}: mosaics already done")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{year}: {len(dates)} dates, grid {W} x {H}")

    records = []
    for date in dates:
        layers = []
        for f in by_date[date]:
            with rasterio.open(f) as src:
                dst = np.full((H, W), np.nan, dtype="float32")
                reproject(source=rasterio.band(src, 1), destination=dst,
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=transform, dst_crs="EPSG:4326",
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
            layers.append(dst)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            arr = np.nanmean(np.stack(layers, axis=0), axis=0).astype("float32")
        with rasterio.open(out_dir / f"nirv_{date}.tif", "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                           nodata=np.nan, compress="lzw") as dst:
            dst.write(arr, 1)
        valid = int(np.isfinite(arr).sum())
        records.append({"date": date, "n_domains": len(by_date[date]),
                        "valid_px": valid, "valid_frac_grid": valid / (H * W)})
    pd.DataFrame(records).to_csv(out_dir / f"grouped_metadata_{year}.csv", index=False)
    return out_dir


def composite_year(year):
    out_path = C.nirv_npz(year)
    if out_path.exists() and not C.FORCE_RERUN:
        print(f"{year}: {out_path.name} already done")
        return
    nirv_dir = C.DIR_NIRV_CLOUD / str(year)
    meta = pd.read_csv(nirv_dir / f"grouped_metadata_{year}.csv")
    meta["date"] = pd.to_datetime(meta["date"])
    meta["filepath"] = meta["date"].apply(lambda d: nirv_dir / f"nirv_{d.strftime('%Y-%m-%d')}.tif")
    meta = meta[meta["filepath"].apply(lambda p: p.exists())].sort_values("date").reset_index(drop=True)
    meta["doy"] = meta["date"].dt.dayofyear

    with rasterio.open(meta["filepath"].iloc[0]) as src0:
        nrows, ncols = src0.height, src0.width
    cube = np.full((len(meta), nrows, ncols), np.nan, dtype=np.float32)
    for i, fp in enumerate(meta["filepath"]):
        with rasterio.open(fp) as src:
            band = src.read(1).astype(np.float32)
            if src.nodata is not None:
                band = np.where(band == src.nodata, np.nan, band)
            cube[i] = band

    # 8-day bins (empty bins and all-NaN pixels raise harmless numpy warnings)
    warnings.simplefilter("ignore", RuntimeWarning)
    is_leap = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)
    n_doy = 366 if is_leap else 365
    bin_starts = np.arange(1, n_doy + 1, 8)
    bin_centers = bin_starts + 3.5
    bin_centers[bin_centers > n_doy] = n_doy
    nirv_8d = np.full((len(bin_starts), nrows, ncols), np.nan, dtype=np.float32)
    doys = meta["doy"].values
    for bi, start in enumerate(bin_starts):
        m = (doys >= start) & (doys <= min(start + 7, n_doy))
        if m.any():
            with np.errstate(all="ignore"):
                nirv_8d[bi] = np.nanmean(cube[m], axis=0)

    # pixel classes
    amp = np.nanmax(nirv_8d, axis=0) - np.nanmin(nirv_8d, axis=0)
    std = np.nanstd(nirv_8d, axis=0)
    n_valid = np.sum(np.isfinite(nirv_8d), axis=0)
    trend_mask = (amp > C.AMP_THR_STRONG) & (std > C.STD_THR_STRONG) & (n_valid >= C.MIN_VALID_STRONG)
    background_mask = (~trend_mask) & (n_valid >= C.MIN_VALID_BG) & (amp > 0.0)
    n = nrows * ncols
    print(f"{year}: trend {trend_mask.sum() / n:.1%}, background {background_mask.sum() / n:.1%}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, nirv_8d_obs=nirv_8d, trend_mask=trend_mask.astype(bool),
             background_mask=background_mask.astype(bool),
             bin_centers=bin_centers.astype(np.float32), bin_starts=bin_starts.astype(np.int16))


if __name__ == "__main__":
    for yr in C.YEARS:
        if mosaic_year(yr) is not None:
            composite_year(yr)
