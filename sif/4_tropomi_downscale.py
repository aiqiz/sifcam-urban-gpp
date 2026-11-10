"""Downscale TROPOMI SIF to 10 m by spreading each footprint over the 10 m
NIRv inside it (weights: 1/sigma^2 x overlap area), then apply the
negative-SIF bias correction (Koehler et al. 2018, Eq. 1) with a 30-day window.
Pixels without usable NIRv (NaN or <= 0) get no SIF.

in:  2.NIRv_gapfilled/<year>/, ESA TROPOMI granules (<esa_sif_dir>/<year>/)
out: 3.SIF_downscaled/<year>/SIF10m_8day_YYYYMMDD.nc
     4.SIF_winter_corrected/<year>/SIF10m_8day_YYYYMMDD.nc
"""

import os
import re
import glob
from datetime import datetime, date, time, timedelta

import numpy as np

import utils as C
from utils import (banner, info, outputs_present, load_sif_stack,
                            save_sif_netcdf, negative_bias_correction)


_DOY_RX  = re.compile(r"DOY(\d{3})")
_NCDT_RX = re.compile(r"S5P_PAL__L2B_SIF_{2,4}(\d{8})T.*\.nc$")


# ------------------------------------------------------------------ #
#  TROPOMI granule indexing / loading
# ------------------------------------------------------------------ #
def list_nirv_tiles(nirv_dir, year):
    items = []
    for f in sorted(nirv_dir.glob("nirv_gapfilled_8day_DOY*.tif")):
        if f.name.startswith("."):
            continue
        m = _DOY_RX.search(f.name)
        if m:
            doy = int(m.group(1))
            items.append((datetime(year, 1, 1) + timedelta(days=doy - 1), doy, f))
    items.sort()
    return items


def find_esa_files(esa_dir, year=None):
    out = []
    for p in glob.glob(os.path.join(str(esa_dir), "**", "*.nc"), recursive=True):
        m = _NCDT_RX.search(os.path.basename(p))
        if m and (year is None or m.group(1).startswith(str(year))):
            out.append((p, datetime.strptime(m.group(1), "%Y%m%d").date()))
    return sorted(out, key=lambda x: x[1])


def _to_datetime(x):
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime.combine(x, time(0, 0))
    raise TypeError(type(x))


def files_in_window(file_index, center_date, half_window):
    c = _to_datetime(center_date)
    d0, d1 = c - timedelta(days=half_window), c + timedelta(days=half_window)
    return [p for p, d in file_index if d0 <= _to_datetime(d) <= d1]


def time_weight(kind, d, center_date, half_window):
    dd = abs((_to_datetime(d) - _to_datetime(center_date)).days)
    if dd > half_window:
        return 0.0
    if kind in ("none", "uniform"):
        return 1.0
    if kind == "gaussian":
        return float(np.exp(-0.5 * (dd / float(half_window)) ** 2))
    raise ValueError(kind)


def load_sif_single_file(path, bbox=None, filter_negative=True):
    import xarray as xr
    import geopandas as gpd
    from shapely.geometry import Polygon
    prod = xr.open_dataset(path, group="PRODUCT", decode_times=False, mask_and_scale=True)
    geo = xr.open_dataset(path, group="PRODUCT/SUPPORT_DATA/GEOLOCATIONS",
                          decode_times=False, mask_and_scale=True)
    lon = prod["longitude"].values.ravel()
    lat = prod["latitude"].values.ravel()
    sif = prod["SIF_Corr_743"].values.ravel()
    unc = prod["SIF_ERROR_743"].values.ravel()
    sza = geo["solar_zenith_angle"].values.ravel()
    co = np.cos(np.deg2rad(sza)).astype(float)
    lonb = geo["longitude_bounds"].values.reshape(-1, 4)
    latb = geo["latitude_bounds"].values.reshape(-1, 4)
    keep = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(sif)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        keep &= (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    geoms, vs, vu, vc = [], [], [], []
    for i in np.where(keep)[0]:
        xs, ys = lonb[i], latb[i]
        poly = Polygon([(xs[0], ys[0]), (xs[1], ys[1]),
                        (xs[2], ys[2]), (xs[3], ys[3])]).buffer(0)
        if poly.is_empty:
            continue
        geoms.append(poly); vs.append(float(sif[i]))
        vu.append(float(unc[i]) if np.isfinite(unc[i]) else np.nan)
        vc.append(float(co[i]) if np.isfinite(co[i]) else np.nan)
    prod.close(); geo.close()
    m = _NCDT_RX.search(os.path.basename(path))
    file_date = datetime.strptime(m.group(1), "%Y%m%d").date() if m else None
    gdf = gpd.GeoDataFrame({"sif": vs, "unc": vu, "co": vc},
                           geometry=geoms, crs="EPSG:4326")
    gdf["date"] = file_date
    if filter_negative and len(gdf) > 0 and file_date is not None \
            and file_date.month in (12, 1, 2, 3):
        gdf.loc[gdf["sif"] < 0, "sif"] = 0.0
    return gdf


# ------------------------------------------------------------------ #
#  Footprint rasterization + downscaler
# ------------------------------------------------------------------ #
def footprint_frac_grid(poly, out_transform, rr0, rr1, cc0, cc1,
                        supersample=8, all_touched=True):
    from rasterio.transform import Affine
    from rasterio.features import rasterize
    from rasterio.enums import MergeAlg
    H, W = rr1 - rr0, cc1 - cc0
    if H <= 0 or W <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    SS = int(max(1, supersample))
    win_tf = out_transform * Affine.translation(cc0, rr0)
    tf_ss = win_tf * Affine.scale(1 / SS, 1 / SS)
    Hs, Ws = H * SS, W * SS
    shapes = []
    if hasattr(poly, "geoms"):
        for g in poly.geoms:
            if not g.is_empty:
                shapes.append((g, 1.0))
    elif not poly.is_empty:
        shapes = [(poly, 1.0)]
    if not shapes:
        return np.zeros((H, W), dtype=np.float32)
    mask_ss = rasterize(shapes=shapes, out_shape=(Hs, Ws), transform=tf_ss,
                        all_touched=all_touched, dtype="float32",
                        merge_alg=MergeAlg.add)
    return mask_ss.reshape(H, SS, W, SS).mean(axis=(1, 3)).astype(np.float32)


def downscale_one_date(center_date, nirv, nir_transform, nir_crs, bbox_lonlat,
                       esa_files, eps=1e-8):
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    H0, W0 = nirv.shape
    bounds_nir = transform_bounds(CRS.from_epsg(4326), nir_crs, *bbox_lonlat,
                                  densify_pts=21)
    minx, miny, maxx, maxy = bounds_nir
    win = from_bounds(minx, miny, maxx, maxy, nir_transform, height=H0, width=W0)
    r0 = max(0, int(round(win.row_off)))
    c0 = max(0, int(round(win.col_off)))
    r1 = min(H0, int(round(win.row_off + win.height)))
    c1 = min(W0, int(round(win.col_off + win.width)))
    out_transform = nir_transform * Affine.translation(c0, r0)
    nirv_sub = nirv[r0:r1, c0:c1].astype(np.float32, copy=True)
    Hs, Ws = nirv_sub.shape
    aoi_poly = box(*bounds_nir)

    num = np.zeros((Hs, Ws), dtype=np.float64)
    den = np.zeros((Hs, Ws), dtype=np.float64)
    diag = {"n_loaded": 0, "n_intersect": 0, "n_contrib": 0}
    if not esa_files:
        return np.full((Hs, Ws), np.nan, np.float32), out_transform, nir_crs, diag

    gdfs = []
    for f in esa_files:
        g = load_sif_single_file(f, bbox=bbox_lonlat,
                                 filter_negative=C.FILTER_NEGATIVE)
        diag["n_loaded"] += len(g)
        gdfs.append(g)
    if not any(len(g) > 0 for g in gdfs):
        return np.full((Hs, Ws), np.nan, np.float32), out_transform, nir_crs, diag
    sif_gdf = pd.concat([g for g in gdfs if len(g) > 0], ignore_index=True)
    sif_gdf = gpd.GeoDataFrame(sif_gdf, geometry="geometry",
                               crs="EPSG:4326").to_crs(nir_crs)
    sif_gdf = sif_gdf.loc[sif_gdf.intersects(aoi_poly)].copy()
    diag["n_intersect"] = len(sif_gdf)

    for row in sif_gdf.itertuples(index=False):
        poly = row.geometry.intersection(aoi_poly)
        if poly.is_empty:
            continue
        S_k = float(row.sif)
        er_k = float(row.unc) if np.isfinite(row.unc) and row.unc > 0 else np.nan
        if not np.isfinite(er_k):
            continue
        w_t = time_weight(C.TEMPORAL_WEIGHT, row.date, center_date,
                          C.HALF_WINDOW_DAYS)
        if w_t <= 0:
            continue
        minx, miny, maxx, maxy = poly.bounds
        inv = ~out_transform
        cmin, rmin = inv * (minx, miny)
        cmax, rmax = inv * (maxx, maxy)
        rr0 = max(0, int(np.floor(min(rmin, rmax))))
        rr1 = min(Hs, int(np.ceil(max(rmin, rmax))))
        cc0 = max(0, int(np.floor(min(cmin, cmax))))
        cc1 = min(Ws, int(np.ceil(max(cmin, cmax))))
        if (rr1 - rr0) <= 0 or (cc1 - cc0) <= 0:
            continue
        frac = footprint_frac_grid(poly, out_transform, rr0, rr1, cc0, cc1,
                                   supersample=C.SUPERSAMPLE, all_touched=True)
        if frac.size == 0 or float(np.nanmax(frac)) <= 0:
            continue
        nir_win = nirv_sub[rr0:rr1, cc0:cc1]
        # pixels with no NIRv (NaN: outside the Sentinel-2 tiles, or never
        # observed) or NIRv <= 0 are left out of the footprint mean and get no
        # SIF; the footprint's SIF is spread over the usable pixels only
        valid = np.isfinite(nir_win) & (nir_win > 0)
        if C.MIN_INTERSECT_FRACTION > 0:
            valid &= (frac >= C.MIN_INTERSECT_FRACTION)
        nir_ok = np.where(valid, nir_win, 0.0)     # NaN must not get into the sums
        w_area = frac * valid
        denom_area = float(w_area.sum())
        if denom_area <= 0:
            continue
        vbar_k = float((nir_ok * w_area).sum() / denom_area)
        if not (np.isfinite(vbar_k) and vbar_k > 0):
            continue
        invsig2 = 1.0 / (er_k ** 2)
        r_ik = nir_ok / (vbar_k + eps)
        w_ik = np.where(valid, w_t * invsig2 * frac, 0.0)
        num[rr0:rr1, cc0:cc1] += w_ik * (S_k * r_ik)
        den[rr0:rr1, cc0:cc1] += w_ik
        diag["n_contrib"] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        sif_10m = (num / den).astype(np.float32)
    sif_10m[~np.isfinite(sif_10m)] = np.nan
    return sif_10m, out_transform, nir_crs, diag


def save_sif_to_netcdf(out_path, sif_10m, transform, crs, center_date, bbox):
    from netCDF4 import Dataset
    H, W = sif_10m.shape
    minx, miny, maxx, maxy = bbox
    dx, dy = (maxx - minx) / W, (maxy - miny) / H
    lon_1d = minx + dx * (0.5 + np.arange(W))
    lat_1d = maxy - dy * (0.5 + np.arange(H))
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    with Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.createDimension("y", H); nc.createDimension("x", W)
        vlat = nc.createVariable("latitude", "f4", ("y", "x"))
        vlon = nc.createVariable("longitude", "f4", ("y", "x"))
        vsif = nc.createVariable("SIF", "f4", ("y", "x"), zlib=True, complevel=4,
                                 fill_value=np.float32(np.nan))
        vlat[:] = lat2d.astype("float32")
        vlon[:] = lon2d.astype("float32")
        vsif[:] = sif_10m.astype("float32")
        nc.center_date = str(center_date)
        nc.crs = str(crs)
        nc.description = f"Downscaled 10 m SIF ({C.REGION})"
        nc.spatial_resolution = "10 m"
        nc.transform = repr(transform)


def _plot_downscaled(out_dir, year):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        cube, doys, lon, lat, _ = load_sif_stack(out_dir)
    except FileNotFoundError:
        return
    with np.errstate(invalid="ignore"):
        ts = np.array([np.nanmean(cube[t]) for t in range(cube.shape[0])])
    mid = int(np.nanargmax(ts)) if np.isfinite(ts).any() else cube.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(doys, ts, "-o", ms=4, color="tab:orange")
    ax[0].set_xlabel("DOY"); ax[0].set_ylabel("mean 10 m SIF")
    ax[0].set_title(f"Downscaled SIF seasonal cycle ({year})")
    ax[0].grid(alpha=0.3)
    im = ax[1].imshow(cube[mid], cmap="viridis",
                      vmin=0, vmax=np.nanpercentile(cube[mid], 98))
    ax[1].set_title(f"Downscaled 10 m SIF (DOY {int(doys[mid])})")
    ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="SIF")
    fig.tight_layout()
    pdir = out_dir / "plots"; pdir.mkdir(exist_ok=True)
    fig.savefig(pdir / f"step2_sif_downscaled_{year}.png", dpi=140)
    plt.close(fig)


def _plot_winter(raw_cube, corr_cube, bias_mean, doys, out_dir, year):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.errstate(invalid="ignore"):
        raw_ts = np.array([np.nanmean(raw_cube[t]) for t in range(raw_cube.shape[0])])
        cor_ts = np.array([np.nanmean(corr_cube[t]) for t in range(corr_cube.shape[0])])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(doys, raw_ts, "-o", ms=4, color="0.6", label="downscaled (raw)")
    ax[0].plot(doys, cor_ts, "-o", ms=4, color="tab:red",
               label="Eq.1 bias-corrected (raw + b)")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xlabel("DOY"); ax[0].set_ylabel("mean 10 m SIF")
    ax[0].set_title(f"Negative-SIF bias correction ({year})\n"
                    f"raw kept (no smoothing); offset only where 30-day mean < 0")
    ax[0].grid(alpha=0.3); ax[0].legend()
    finite = np.isfinite(bias_mean)
    vmax = np.nanpercentile(bias_mean, 98) if finite.any() and np.nanmax(bias_mean) > 0 else 1
    im = ax[1].imshow(bias_mean, cmap="magma", vmin=0, vmax=vmax)
    ax[1].set_title("Mean bias offset b added (annual mean)"); ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="SIF")
    fig.tight_layout()
    pdir = out_dir / "plots"; pdir.mkdir(exist_ok=True)
    fig.savefig(pdir / f"step2_bias_corrected_{year}.png", dpi=140)
    plt.close(fig)


def run(year, force=False):
    C.require_dir(C.esa_dir(year), f"ESA TROPOMI granules for {year}")
    import rasterio

    nirv_dir = C.nirv_gapfill(year)
    ds_dir = C.sif_downscaled(year)
    wc_dir = C.sif_winter(year)
    if not nirv_dir.is_dir() or not outputs_present(
            nirv_dir, "nirv_gapfilled_8day_DOY*.tif", 1):
        info(f"[{year}] skip: gap-filled NIRv not found (run 3_nirv_gapfill.py)")
        return

    # ---- (a) DOWNSCALE -------------------------------------------------
    do_downscale = force or not C.is_done(ds_dir)
    if do_downscale:
        banner(f"4a  TROPOMI -> 10 m downscaling   year={year}")
        ds_dir.mkdir(parents=True, exist_ok=True)
        nirv_items = list_nirv_tiles(nirv_dir, year)
        info(f"{len(nirv_items)} gap-filled NIRv tiles")
        with rasterio.open(nirv_items[0][2]) as src:
            b = src.bounds
        BBOX = [b.left, b.bottom, b.right, b.top]
        file_index = find_esa_files(C.esa_dir(year), year=year)
        info(f"{len(file_index)} ESA TROPOMI granules for {year} "
             f"(from {C.ESA_ROOT})")
        n_written = 0
        for d, doy, nirv_path in nirv_items:
            with rasterio.open(nirv_path) as src:
                nirv = src.read(1).astype(np.float32)
                nir_transform = src.transform
                nir_crs = src.crs
            nirv = nirv + 1e-4      # small offset so a pixel with NIRv exactly 0 still counts; NaN stays NaN
            wfiles = files_in_window(file_index, d, C.HALF_WINDOW_DAYS)
            sif_10m, tf, crs_out, diag = downscale_one_date(
                d, nirv, nir_transform, nir_crs, BBOX, wfiles)
            if diag["n_contrib"] == 0:
                continue
            save_sif_to_netcdf(ds_dir / f"SIF10m_8day_{d.strftime('%Y%m%d')}.nc",
                               sif_10m, tf, crs_out, d, BBOX)
            n_written += 1
        info(f"downscaled {n_written}/{len(nirv_items)} composites -> {ds_dir}")
        _plot_downscaled(ds_dir, year)
        C.mark_done(ds_dir)
    else:
        n = len([p for p in ds_dir.glob('SIF10m_8day_*.nc')])
        info(f"[{year}] 4a downscaling already done ({n} files) -> skip")

    # ---- (b) WINTER CORRECTION (30-day mean + per-pixel baseline) -------
    if (not force) and C.is_done(wc_dir):
        n = len([p for p in wc_dir.glob('SIF10m_8day_*.nc')])
        info(f"[{year}] 4b bias correction already done ({n}) -> skip")
        return
    if not outputs_present(ds_dir, "SIF10m_8day_*.nc", 1):
        info(f"[{year}] skip 4b: no downscaled SIF to correct")
        return

    banner(f"4b  negative-SIF bias correction (Eq. 1)   year={year}")
    wc_dir.mkdir(parents=True, exist_ok=True)
    cube, doys, lon, lat, dates = load_sif_stack(ds_dir)
    info(f"stack {cube.shape}  finite={np.isfinite(cube).mean():.3f}")
    # Eq. (1): add b = |sbar| where the 30-day moving-window mean is negative.
    # Not a smoothing: raw per-composite values (and noise negatives) are kept.
    corrected, bias = negative_bias_correction(
        cube, n_composites=C.BIAS_WINDOW_COMPOSITES)
    info(f"bias correction: pixels lifted={np.mean(bias > 0)*100:.2f}%  "
         f"mean b(where>0)={float(np.nanmean(bias[bias > 0])) if np.any(bias > 0) else 0.0:.4f}")
    lon2d, lat2d = (lon, lat) if lon.ndim == 2 else np.meshgrid(lon, lat)
    for t, d in enumerate(dates):
        save_sif_netcdf(wc_dir / f"SIF10m_8day_{d.strftime('%Y%m%d')}.nc",
                        corrected[t], lon2d, lat2d, d,
                        description="Bias-corrected downscaled 10 m SIF "
                                    "(Eq.1 spatiotemporal negative-SIF correction)")
    with np.errstate(invalid="ignore"):
        bias_mean = np.nanmean(bias, axis=0)        # mean offset map (H, W)
    np.save(wc_dir / "bias_mean.npy", bias_mean.astype(np.float32))
    _plot_winter(cube, corrected, bias_mean, doys, wc_dir, year)
    C.mark_done(wc_dir)
    info(f"done -> {wc_dir}")


if __name__ == "__main__":
    for yr in C.YEARS:
        run(yr, force=C.FORCE_RERUN)
