"""Shoreline correction. Near Lake Ontario the coarse SIF footprints mix land
and water, so per date we fit SIF = a * peakNIRv + b on inland pixels and
blend shoreline pixels toward that fit. Permanent water is set to NaN.

in:  6.SIF_gapfilled/<year>/, 2.NIRv_gapfilled/<year>/, WorldCover tile
out: 7.SIF_shoreline_corrected/<year>/SIF10m_8day_YYYYMMDD.nc   (final 10 m SIF)
"""

import os, re, glob
from pathlib import Path
from datetime import datetime

import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from netCDF4 import Dataset
from scipy.ndimage import label, binary_erosion, distance_transform_edt

import utils as C
from utils import banner, info, outputs_present

# ----- settings (from utils) -----
SHORELINE_THRESHOLD_DEG    = C.SHORELINE_THRESHOLD_DEG
MIN_WATER_COMPONENT_PIXELS = C.MIN_WATER_COMPONENT_PIXELS
PERMANENT_WATER_CLASSES    = C.PERMANENT_WATER_CLASSES
USE_INTERCEPT  = C.USE_INTERCEPT
MIN_REG_POINTS = C.MIN_REG_POINTS
PRED_CLIP_MIN  = C.PRED_CLIP_MIN
PRED_CLIP_MAX  = C.PRED_CLIP_MAX
FILL_VALUE = np.float32(99999.0)


# ============================================================ #
#  IO helpers
# ============================================================ #
def list_sif_nc_files(in_dir):
    files = sorted([
        f for f in glob.glob(str(Path(in_dir) / "*.nc"))
        if not os.path.basename(f).startswith("._")
    ])
    if not files:
        raise ValueError(f"No .nc files found in {in_dir}")
    return [Path(f) for f in files]


def load_sif_nc(nc_path):
    ds  = xr.open_dataset(nc_path)
    sif  = ds["SIF"].values.astype(np.float32)
    lat2d = ds["latitude"].values.astype(np.float32)
    lon2d = ds["longitude"].values.astype(np.float32)
    ds.close()
    sif[~np.isfinite(sif)] = np.nan
    sif[sif >= 99999] = np.nan
    return sif, lat2d, lon2d


def extract_date_from_filename(path):
    m = re.search(r"(\d{8})", path.name)
    return datetime.strptime(m.group(1), "%Y%m%d") if m else None


def save_sif_to_netcdf(out_path, sif_2d, lat2d, lon2d, center_date, source_file=None):
    H, W = sif_2d.shape
    write_arr = sif_2d.astype(np.float32).copy()
    write_arr[~np.isfinite(write_arr)] = FILL_VALUE

    with Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.createDimension("y", H)
        nc.createDimension("x", W)
        vlat = nc.createVariable("latitude",  "f4", ("y", "x"))
        vlon = nc.createVariable("longitude", "f4", ("y", "x"))
        vsif = nc.createVariable(
            "SIF", "f4", ("y", "x"),
            zlib=True, complevel=4,
            fill_value=FILL_VALUE
        )
        vlat[:] = lat2d.astype("float32")
        vlon[:] = lon2d.astype("float32")
        vsif[:] = write_arr
        vsif.missing_value = FILL_VALUE
        vsif.units         = "mW m-2 sr-1 nm-1"
        nc.description = "Final downscaled SIF with shoreline correction"
        nc.center_date = str(center_date)
        nc.history     = "Shoreline corrected: peak NIRv regression"
        if source_file is not None:
            nc.source_file = str(source_file)


# ============================================================ #
#  Ancillary data: peak NIRv + water mask + shoreline distance
# ============================================================ #
def load_peak_nirv_to_target_grid(peak_nirv_path, target_lat2d, target_lon2d):
    H, W = target_lat2d.shape
    lon_min = float(np.nanmin(target_lon2d)); lon_max = float(np.nanmax(target_lon2d))
    lat_min = float(np.nanmin(target_lat2d)); lat_max = float(np.nanmax(target_lat2d))
    dx = (lon_max - lon_min) / W
    dy = (lat_max - lat_min) / H
    target_transform = rasterio.transform.from_origin(lon_min, lat_max, dx, dy)
    out = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(peak_nirv_path) as src:
        src_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan
        reproject(
            source=src_arr, destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=target_transform, dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
    out[~np.isfinite(out)] = np.nan
    out[out <= 0] = np.nan
    return out


def connected_components_keep_large(mask_bool, min_pixels):
    lbl, nlab = label(mask_bool, structure=np.ones((3, 3), dtype=np.int8))
    if nlab == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    counts  = np.bincount(lbl.ravel())
    keep_ids = np.where(counts >= min_pixels)[0]
    keep_ids = keep_ids[keep_ids != 0]
    return np.isin(lbl, keep_ids)


def load_large_water_mask_to_target_grid(
    landcover_path, target_lat2d, target_lon2d,
    water_classes, min_component_pixels
):
    H, W = target_lat2d.shape
    lon_min = float(np.nanmin(target_lon2d)); lon_max = float(np.nanmax(target_lon2d))
    lat_min = float(np.nanmin(target_lat2d)); lat_max = float(np.nanmax(target_lat2d))
    dx = (lon_max - lon_min) / W
    dy = (lat_max - lat_min) / H
    target_transform = rasterio.transform.from_origin(lon_min, lat_max, dx, dy)
    lc_reproj = np.full((H, W), -9999, dtype=np.int32)
    with rasterio.open(landcover_path) as src:
        src_arr = src.read(1).astype(np.int32)
        reproject(
            source=src_arr, destination=lc_reproj,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=target_transform, dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=src.nodata if src.nodata is not None else -9999,
            dst_nodata=-9999,
        )
    water_mask       = np.isin(lc_reproj, water_classes)
    large_water_mask = connected_components_keep_large(water_mask, min_component_pixels)
    diag = {
        "n_water_pixels_all":        int(water_mask.sum()),
        "n_water_pixels_large_only": int(large_water_mask.sum()),
    }
    return large_water_mask, diag


def compute_shoreline_distance_deg(large_water_mask, lat2d, lon2d):
    eroded = binary_erosion(large_water_mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    shoreline_boundary = large_water_mask ^ eroded
    if not shoreline_boundary.any():
        return shoreline_boundary, np.full_like(lat2d, np.nan, dtype=np.float32)
    dx_deg = float(np.nanmedian(np.abs(np.diff(lon2d, axis=1))))
    dy_deg = float(np.nanmedian(np.abs(np.diff(lat2d, axis=0))))
    dx_deg = dx_deg if (dx_deg > 0 and np.isfinite(dx_deg)) else 1e-4
    dy_deg = dy_deg if (dy_deg > 0 and np.isfinite(dy_deg)) else 1e-4
    dist_deg = distance_transform_edt(
        ~shoreline_boundary, sampling=(dy_deg, dx_deg)
    ).astype(np.float32)
    return shoreline_boundary, dist_deg


def build_land_and_shore_masks(large_water_mask, dist_deg, threshold_deg):
    land_mask         = ~large_water_mask
    shoreline_mask    = land_mask & np.isfinite(dist_deg) & (dist_deg < threshold_deg)
    nonshore_land_mask= land_mask & np.isfinite(dist_deg) & (dist_deg >= threshold_deg)
    return land_mask, shoreline_mask, nonshore_land_mask


# ============================================================ #
#  Regression + correction
# ============================================================ #
def fit_peaknirv_to_sif_regression(peak_nirv, sif, fit_mask, use_intercept=True):
    x = peak_nirv[fit_mask].astype(np.float64)
    y = sif[fit_mask].astype(np.float64)
    if x.size < MIN_REG_POINTS:
        return np.nan, np.nan, np.nan
    if use_intercept:
        A = np.column_stack([x, np.ones_like(x)])
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        yhat = a * x + b
    else:
        denom = np.sum(x * x)
        if denom <= 0:
            return np.nan, np.nan, np.nan
        a = float(np.sum(x * y) / denom)
        b = 0.0
        yhat = a * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def predict_sif_from_peaknirv(peak_nirv, a, b):
    pred = (a * peak_nirv + b).astype(np.float32)
    pred[~np.isfinite(pred)] = np.nan
    pred = np.clip(pred, PRED_CLIP_MIN, PRED_CLIP_MAX)
    return pred


def shoreline_weight(dist_deg, threshold_deg):
    """Linear weight: 1 at shoreline, 0 at threshold distance."""
    w = np.clip(1.0 - dist_deg / threshold_deg, 0.0, 1.0)
    w[~np.isfinite(w)] = 0.0
    return w.astype(np.float32)


def correct_one_sif_date(
    sif_orig, peak_nirv,
    land_mask, shoreline_mask, nonshore_land_mask,
    dist_deg, use_intercept=True, shoreline_threshold_deg=0.012
):
    fit_mask = nonshore_land_mask & np.isfinite(peak_nirv) & np.isfinite(sif_orig)
    a, b, r2 = fit_peaknirv_to_sif_regression(peak_nirv, sif_orig, fit_mask, use_intercept)

    sif_corr = sif_orig.copy().astype(np.float32)
    if not np.isfinite(a):
        return sif_corr, np.full_like(sif_orig, np.nan, dtype=np.float32), {
            "n_fit_points": int(fit_mask.sum()), "reg_slope": np.nan,
            "reg_intercept": np.nan, "reg_r2": np.nan,
            "n_shoreline_blended": 0, "n_missing_land_recreated": 0,
        }

    pred_sif = predict_sif_from_peaknirv(peak_nirv, a, b)
    w        = shoreline_weight(dist_deg, shoreline_threshold_deg)

    # blend shoreline pixels where original exists
    mask_blend = shoreline_mask & np.isfinite(sif_orig) & np.isfinite(pred_sif)
    sif_corr[mask_blend] = (
        w[mask_blend] * pred_sif[mask_blend]
        + (1.0 - w[mask_blend]) * sif_orig[mask_blend]
    ).astype(np.float32)

    # recreate missing land pixels
    mask_recreate = land_mask & ~np.isfinite(sif_orig) & np.isfinite(pred_sif)
    sif_corr[mask_recreate] = pred_sif[mask_recreate]

    diag = {
        "n_fit_points": int(fit_mask.sum()),
        "reg_slope": float(a), "reg_intercept": float(b),
        "reg_r2": float(r2) if np.isfinite(r2) else np.nan,
        "n_shoreline_blended": int(mask_blend.sum()),
        "n_missing_land_recreated": int(mask_recreate.sum()),
    }
    return sif_corr, pred_sif, diag


# ============================================================ #
#  Plot (one composite: before / predicted / corrected)
# ============================================================ #
def _plot(out_dir, year, sif_orig, pred, sif_corr, large_water_mask, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.cm.viridis.copy(); cmap.set_bad("black")
    disp_pred = pred.copy(); disp_pred[large_water_mask] = np.nan
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a_, arr, t in zip(
        ax, [sif_orig, disp_pred, sif_corr],
        [f"Original ({tag})", "Predicted from peak NIRv",
         "Shoreline-corrected"]):
        im = a_.imshow(arr, cmap=cmap, vmin=0, vmax=1.0, interpolation="nearest")
        a_.set_title(t, fontsize=10); a_.axis("off")
        plt.colorbar(im, ax=a_, shrink=0.7)
    fig.suptitle(f"Shoreline correction {year}", y=1.02)
    fig.tight_layout()
    pdir = out_dir / "plots"; pdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdir / f"step5_shoreline_{year}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================ #
#  Runner (per year; paths from config)
# ============================================================ #
def run(year, force=False):
    SIF_DIR        = C.sif_gapfilled(year)
    PEAK_NIRV_PATH = (C.nirv_gapfill(year)
                      / f"nirv_gapfilled_8day_DOY{C.SHORE_PEAK_DOY:03d}.tif")
    LANDCOVER_PATH = C.WORLDCOVER_TILE
    OUT_DIR        = C.sif_shoreline(year)

    if not outputs_present(SIF_DIR, "*.nc", 1):
        info(f"[{year}] skip: gap-filled SIF not found (run 6_merge_gapfill.py)")
        return
    if (not force) and C.is_done(OUT_DIR):
        n = len([p for p in OUT_DIR.glob('SIF10m_8day_*.nc')])
        info(f"[{year}] already done ({n} files) -> skip")
        return
    if not Path(PEAK_NIRV_PATH).exists():
        info(f"[{year}] skip: peak NIRv tile missing: {PEAK_NIRV_PATH.name}")
        return
    if not Path(LANDCOVER_PATH).exists():
        info(f"[{year}] skip: WorldCover tile missing: {LANDCOVER_PATH}")
        return

    banner(f"7  shoreline correction   year={year}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- setup (once per year) ---
    sif_files = list_sif_nc_files(SIF_DIR)
    _, lat2d, lon2d = load_sif_nc(sif_files[0])
    peak_nirv = load_peak_nirv_to_target_grid(PEAK_NIRV_PATH, lat2d, lon2d)
    large_water_mask, water_diag = load_large_water_mask_to_target_grid(
        LANDCOVER_PATH, lat2d, lon2d,
        water_classes=PERMANENT_WATER_CLASSES,
        min_component_pixels=MIN_WATER_COMPONENT_PIXELS)
    shoreline_boundary, dist_deg = compute_shoreline_distance_deg(
        large_water_mask, lat2d, lon2d)
    land_mask, shoreline_mask, nonshore_land_mask = build_land_and_shore_masks(
        large_water_mask, dist_deg, SHORELINE_THRESHOLD_DEG)
    info(f"peak NIRv=DOY{C.SHORE_PEAK_DOY}  water_px={int(large_water_mask.sum()):,}  "
         f"shoreline_zone={int(shoreline_mask.sum()):,}  "
         f"nonshore_land={int(nonshore_land_mask.sum()):,}")

    # --- main loop ---
    plot_cache = None
    r2s = []
    for sif_path in sif_files:
        center_date = extract_date_from_filename(sif_path)
        sif_orig, lat_i, lon_i = load_sif_nc(sif_path)
        if sif_orig.shape != peak_nirv.shape:
            info(f"  {center_date}: shape {sif_orig.shape} != peak {peak_nirv.shape}, skip")
            continue
        sif_corr, pred_sif, diag = correct_one_sif_date(
            sif_orig=sif_orig, peak_nirv=peak_nirv,
            land_mask=land_mask, shoreline_mask=shoreline_mask,
            nonshore_land_mask=nonshore_land_mask, dist_deg=dist_deg,
            use_intercept=USE_INTERCEPT,
            shoreline_threshold_deg=SHORELINE_THRESHOLD_DEG)
        sif_corr[large_water_mask] = np.nan          # permanent water -> NaN
        save_sif_to_netcdf(
            OUT_DIR / f"SIF10m_8day_{center_date.strftime('%Y%m%d')}.nc",
            sif_corr, lat_i, lon_i, center_date, source_file=sif_path)
        if np.isfinite(diag["reg_r2"]):
            r2s.append(diag["reg_r2"])
        if plot_cache is None and np.isfinite(diag["reg_slope"]):
            plot_cache = (sif_orig.copy(), pred_sif.copy(), sif_corr.copy(),
                          center_date.strftime("%Y%m%d"))

    info(f"corrected {len(sif_files)} composites"
         + (f"  median R2={np.median(r2s):.3f}" if r2s else ""))
    if plot_cache:
        _plot(OUT_DIR, year, plot_cache[0], plot_cache[1], plot_cache[2],
              large_water_mask, plot_cache[3])
    C.mark_done(OUT_DIR)
    info(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    for yr in C.YEARS:
        run(yr, force=C.FORCE_RERUN)
