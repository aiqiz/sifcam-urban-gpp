"""Settings, paths and small helpers shared by the sif/ scripts.

Paths and years come from ../config.yaml. The numbers below are the method
settings used in the paper; normally you don't need to touch them.
"""

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
_cfg_file = ROOT / "config.local.yaml"          # your own copy of config.yaml, not in git
CFG = yaml.safe_load(open(_cfg_file if _cfg_file.exists() else ROOT / "config.yaml"))


def path(p):
    """Absolute paths are used as they are. Relative paths are inside data_dir,
    except `data/...`, the small files shipped with this repository."""
    p = Path(p)
    if p.is_absolute():
        return p
    if p.parts[0] == "data":
        return ROOT / p
    return Path(CFG["data_dir"]) / p
_S = CFG["sif"]

REGION      = _S["region"]
REGION_CFG  = _S["regions"][REGION]
YEARS       = REGION_CFG.get("years") or _S["years"]   # a region can override the years
FORCE_RERUN = _S["force_rerun"]
N_PROCS     = _S["n_procs"]

# ---- paths ------------------------------------------------------------------
SITE_DIR       = Path(CFG["results_dir"]) / "sif" / REGION
S2_DIR         = path(_S["s2_dir"])
ESA_ROOT       = path(_S["esa_sif_dir"])
GOSIF_ROOT     = path(_S["gosif_dir"])
WORLDCOVER_TILE = path(_S["worldcover_tile"])

DIR_NIRV_CLOUD    = SITE_DIR / "0.NIRv_cloud_mask"
DIR_NIRV_NPZ      = SITE_DIR / "1.NIRv_8day_year"
DIR_NIRV_GAPFILL  = SITE_DIR / "2.NIRv_gapfilled"
DIR_SIF_DOWNSCL   = SITE_DIR / "3.SIF_downscaled"        # ESA, raw winter clamp
DIR_SIF_WINTER    = SITE_DIR / "4.SIF_winter_corrected"  # + 30-day mean bias correction
DIR_GOSIF_DOWNSCL = SITE_DIR / "5.GOSIF_downscaled"
DIR_SIF_GAPFILL   = SITE_DIR / "6.SIF_gapfilled"
DIR_SIF_SHORE     = SITE_DIR / "7.SIF_shoreline_corrected"   # final 10 m SIF
DIR_GPP           = SITE_DIR / "8.GPP_10m"                   # SMUrF output


def s2_tile_dir(tile, year): return S2_DIR / tile / str(year)
def nirv_npz(year):         return DIR_NIRV_NPZ / f"nirv_inputs_{year}.npz"
def nirv_gapfill(year):     return DIR_NIRV_GAPFILL / str(year)
def sif_downscaled(year):   return DIR_SIF_DOWNSCL / str(year)
def sif_winter(year):       return DIR_SIF_WINTER / str(year)
def gosif_downscaled(year): return DIR_GOSIF_DOWNSCL / str(year)
def sif_gapfilled(year):    return DIR_SIF_GAPFILL / str(year)
def sif_shoreline(year):    return DIR_SIF_SHORE / str(year)
def esa_dir(year):          return ESA_ROOT / str(year)
def gosif_dir(year):        return GOSIF_ROOT / str(year)


def require_dir(path, what):
    """Stop with a clear message instead of running on with no input."""
    if not Path(path).is_dir():
        raise SystemExit(f"{what} not found: {path}\n    check config.yaml (is the drive mounted?)")


# ---- method settings --------------------------------------------------------
# NIRv 8-day composite: pixel classes
AMP_THR_STRONG   = 0.01   # trend pixel: amplitude and std above these
STD_THR_STRONG   = 0.01
MIN_VALID_STRONG = 11     # ... and at least this many valid 8-day bins
MIN_VALID_BG     = 6      # background pixel: fewer points ok

# NIRv gap-fill (Savitzky-Golay)
SG_WINDOW      = 5        # odd, in 8-day steps
SG_POLYORDER   = 2
SG_MIN_POINTS  = 8        # fewer valid obs -> linear interpolation only
BG_HALF_WINDOW = 4        # triangular smooth half window for background pixels

# TROPOMI downscaling
HALF_WINDOW_DAYS       = 8        # +/- days of granules per composite
COMPOSITE_STRIDE_DAYS  = 8
TEMPORAL_WEIGHT        = "none"   # "none" | "uniform" | "gaussian"
SUPERSAMPLE            = 8        # footprint sub-pixel rasterization factor
MIN_INTERSECT_FRACTION = 0.0
FILTER_NEGATIVE        = False    # keep negative footprints, the bias correction needs them
BIAS_WINDOW_COMPOSITES = 4        # ~30-day window = 4 x 8-day composites

# GOSIF
GOSIF_SCALE      = 0.0001
GOSIF_FILL_WATER = 32767
GOSIF_FILL_LAND  = 32766
GOSIF_MATCH_DAYS = 4
GAPFILL_MAX_DAYS = 8

# shoreline correction
SHORELINE_THRESHOLD_DEG    = 0.012   # blend band half-width (~1.3 km)
MIN_WATER_COMPONENT_PIXELS = 5000
PERMANENT_WATER_CLASSES    = [80]    # WorldCover permanent water
USE_INTERCEPT  = True
MIN_REG_POINTS = 200
PRED_CLIP_MIN  = 0.0
PRED_CLIP_MAX  = 5.0
SHORE_PEAK_DOY = 188


# ---- logging / skip helpers -------------------------------------------------
def banner(msg):
    print(f"\n--- {msg}", flush=True)


def info(msg):
    print(f"    {msg}", flush=True)


def count_files(directory, pattern):
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return len([p for p in directory.glob(pattern) if not p.name.startswith(".")])


def outputs_present(directory, pattern, min_count=1):
    return count_files(directory, pattern) >= min_count


# A step writes an empty `_complete` file in its output folder when it has
# finished a year. Only that counts as done: a folder with some files in it
# (a run that was stopped) is redone. Delete the marker to redo a year.
def is_done(directory):
    return (Path(directory) / "_complete").exists()


def mark_done(directory):
    (Path(directory) / "_complete").touch()


def doy_to_date(year, doy):
    return datetime(year, 1, 1) + timedelta(days=int(doy) - 1)


# ---- raster I/O -------------------------------------------------------------
def read_geotiff(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        prof = src.profile.copy()
    return arr, prof


def write_geotiff(path, arr, template_profile):
    import rasterio
    prof = template_profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(np.float32), 1)


def template_profile_from(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.profile.copy()


# ---- negative-SIF bias correction -------------------------------------------
# Koehler et al. (2018) Eq. 1: b = |sbar| where the 30-day window mean sbar < 0,
# else 0. b is added to the raw SIF. Not a smoothing - only the local mean is
# shifted to be non-negative.
def moving_window_mean(stack, n_composites=4):
    import warnings
    T = stack.shape[0]
    half = max(1, n_composites) // 2
    out = np.full_like(stack, np.nan, dtype=np.float32)
    for t in range(T):
        t0 = max(0, t - half)
        t1 = min(T, t + half + 1)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out[t] = np.nanmean(stack[t0:t1], axis=0)
    return out


def negative_bias_correction(stack, n_composites=4):
    sbar = moving_window_mean(stack, n_composites=n_composites)
    with np.errstate(invalid="ignore"):
        neg = np.isfinite(sbar) & (sbar < 0.0)
    bias = np.where(neg, -sbar, 0.0).astype(np.float32)
    corrected = (stack + bias).astype(np.float32)
    return corrected, bias


# ---- downscaled SIF NetCDF ----------------------------------------------------
_DATE8_RX = re.compile(r"(\d{8})")
FILL_SENTINEL = 99999.0   # some older products store NaN as this


def load_sif_stack(sif_dir, pattern="SIF10m_8day_*.nc", varname="SIF"):
    """All matching NetCDFs in sif_dir -> (cube, doys, lon, lat, dates)."""
    import xarray as xr
    items = []
    for p in sorted(Path(sif_dir).glob(pattern)):
        if p.name.startswith("."):
            continue
        m = _DATE8_RX.search(p.name)
        if m:
            items.append((datetime.strptime(m.group(1), "%Y%m%d"), p))
    items.sort()
    if not items:
        raise FileNotFoundError(f"No {pattern} in {sif_dir}")
    arrs, doys, dates, lon, lat = [], [], [], None, None
    for d, p in items:
        with xr.open_dataset(p) as ds:
            a = ds[varname].values.astype(np.float32)
            a[~np.isfinite(a)] = np.nan
            a[a >= FILL_SENTINEL] = np.nan
            arrs.append(a)
            if lon is None:
                lon = ds["longitude"].values
                lat = ds["latitude"].values
        doys.append(d.timetuple().tm_yday)
        dates.append(d)
    cube = np.stack(arrs, axis=0)
    return cube, np.array(doys), lon, lat, dates


def save_sif_netcdf(out_path, sif_2d, lon2d, lat2d, date, varname="SIF",
                    description="Downscaled 10 m SIF"):
    from netCDF4 import Dataset
    H, W = sif_2d.shape
    with Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.createDimension("y", H)
        nc.createDimension("x", W)
        vlat = nc.createVariable("latitude", "f4", ("y", "x"))
        vlon = nc.createVariable("longitude", "f4", ("y", "x"))
        vsif = nc.createVariable(varname, "f4", ("y", "x"),
                                 zlib=True, complevel=4,
                                 fill_value=np.float32(np.nan))
        vlat[:] = lat2d.astype("float32")
        vlon[:] = lon2d.astype("float32")
        vsif[:] = sif_2d.astype("float32")
        nc.center_date = str(date)
        nc.description = description
        nc.spatial_resolution = "10 m"


def list_dated_files(directory, pattern, date_rx=_DATE8_RX):
    """(datetime, Path) for files matching pattern, sorted by YYYYMMDD in the name."""
    items = []
    for p in sorted(Path(directory).glob(pattern)):
        if p.name.startswith("."):
            continue
        m = date_rx.search(p.name)
        if m:
            items.append((datetime.strptime(m.group(1), "%Y%m%d"), p))
    items.sort()
    return items


def read_sif_nc(path, varname="SIF"):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        arr = ds[varname].values.astype(np.float32)
        lon = ds["longitude"].values
        lat = ds["latitude"].values
    return arr, lon, lat


def affine_from_lonlat(lon2d, lat2d):
    """North-up Affine transform from a regular lon/lat grid."""
    from rasterio.transform import Affine
    lon1d = lon2d[0, :] if lon2d.ndim == 2 else lon2d
    lat1d = lat2d[:, 0] if lat2d.ndim == 2 else lat2d
    dx = float(lon1d[1] - lon1d[0])
    dy = float(lat1d[1] - lat1d[0])
    x0 = float(lon1d[0]) - dx / 2.0
    y0 = float(lat1d[0]) - dy / 2.0
    return Affine(dx, 0.0, x0, 0.0, dy, y0)


def load_worldcover_water(worldcover_dir, ref_transform, ref_crs, H, W,
                          water_class=80):
    """WorldCover water class reprojected onto a reference grid -> bool mask."""
    import glob
    import os
    import rasterio
    from rasterio.warp import reproject, Resampling
    water = np.zeros((H, W), dtype=bool)
    for tif in sorted(glob.glob(os.path.join(str(worldcover_dir), "*.tif"))):
        with rasterio.open(tif) as wsrc:
            wc = wsrc.read(1)
            dst = np.zeros((H, W), dtype=np.uint8)
            reproject(source=wc, destination=dst,
                      src_transform=wsrc.transform, src_crs=wsrc.crs,
                      dst_transform=ref_transform, dst_crs=ref_crs,
                      resampling=Resampling.nearest)
        water |= (dst == water_class)
    return water
