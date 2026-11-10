"""Calibrate GOSIF against the downscaled TROPOMI SIF (linear fit on the
GOSIF grid), then spread each GOSIF pixel over the 10 m NIRv inside it.

in:  2.NIRv_gapfilled/<year>/, 3.SIF_downscaled/<year>/, <gosif_dir>/<year>/
out: 5.GOSIF_downscaled/<year>/GOSIF10m_8day_YYYYMMDD.nc
"""

import re
from datetime import datetime, timedelta

import numpy as np

import utils as C
from utils import (banner, info, outputs_present, load_sif_stack,
                            save_sif_netcdf, read_sif_nc, affine_from_lonlat)

_DOY_RX = re.compile(r"DOY(\d{3})")
_GOSIF_RAW_FILL = (99999, 65535, 32767, 32766)


# ------------------------------------------------------------------ #
#  File listing
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


def list_gosif_files(gosif_dir, year):
    rx = re.compile(rf"GOSIF_{year}(\d{{3}})\.tif$", re.IGNORECASE)
    out = {}
    if not gosif_dir.is_dir():
        return out
    for f in gosif_dir.iterdir():
        if f.name.startswith("."):
            continue
        m = rx.search(f.name)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def find_gosif_for_doy(gosif_dict, doy):
    best = min(gosif_dict.keys(), key=lambda d: abs(d - doy))
    return gosif_dict[best], best


# ------------------------------------------------------------------ #
#  GOSIF reading / masking
# ------------------------------------------------------------------ #
def read_gosif_physical(path):
    """Read GOSIF DN -> physical SIF with fill masking + range filter."""
    import rasterio
    with rasterio.open(path) as src:
        raw = src.read(1).astype(np.float32)
        tfm, crs, nd = src.transform, src.crs, src.nodata
        gh, gw = src.height, src.width
    if nd is not None:
        raw[raw == np.float32(nd)] = np.nan
    for fv in _GOSIF_RAW_FILL:
        raw[raw >= fv] = np.nan
    raw *= C.GOSIF_SCALE
    raw[(raw < 0.0) | (raw > 3.0)] = np.nan
    return raw, tfm, crs, gh, gw


# ------------------------------------------------------------------ #
#  Calibration GOSIF -> ESA
# ------------------------------------------------------------------ #
def calibrate_gosif_to_esa(esa_dir, gosif_files, year):
    """Linear regression raw GOSIF -> downscaled ESA SIF, aggregated to the
    GOSIF coarse grid.  Returns (slope, intercept, r2, n)."""
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    try:
        cube, doys, lon, lat, dates = load_sif_stack(esa_dir)
    except FileNotFoundError:
        return 1.0, 0.0, np.nan, 0
    esa_tfm = affine_from_lonlat(lon, lat)
    esa_crs = CRS.from_epsg(4326)
    gx, ex = [], []
    for t, d in enumerate(dates):
        doy = d.timetuple().tm_yday
        if not gosif_files:
            continue
        gpath, _ = find_gosif_for_doy(gosif_files, doy)
        raw, g_tfm, g_crs, gh, gw = read_gosif_physical(gpath)
        esa_arr = cube[t].astype(np.float32)
        esa_on_g = np.full((gh, gw), np.nan, dtype=np.float32)
        reproject(source=np.where(np.isfinite(esa_arr), esa_arr, np.nan),
                  destination=esa_on_g,
                  src_transform=esa_tfm, src_crs=esa_crs,
                  dst_transform=g_tfm, dst_crs=g_crs,
                  resampling=Resampling.average,
                  src_nodata=np.nan, dst_nodata=np.nan)
        valid = np.isfinite(raw) & np.isfinite(esa_on_g)
        if valid.any():
            gx.extend(raw[valid].tolist())
            ex.extend(esa_on_g[valid].tolist())
    if len(gx) < 10:
        info("  calibration: too few paired footprints -> identity (slope=1)")
        return 1.0, 0.0, np.nan, len(gx)
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(np.array(gx), np.array(ex))
    return float(slope), float(intercept), float(r ** 2), len(gx)


# ------------------------------------------------------------------ #
#  Downscale one GOSIF tile to the NIRv grid
# ------------------------------------------------------------------ #
def downscale_gosif(gosif_path, nirv, nir_transform, nir_crs,
                    cal_slope, cal_intercept, eps=1e-8):
    from rasterio.warp import reproject, Resampling
    H, W = nirv.shape
    raw, g_tfm, g_crs, gh, gw = read_gosif_physical(gosif_path)
    raw = cal_slope * raw + cal_intercept
    raw[raw < 0] = np.nan

    gosif_reproj = np.full((H, W), np.nan, dtype=np.float32)
    reproject(source=raw, destination=gosif_reproj,
              src_transform=g_tfm, src_crs=g_crs,
              dst_transform=nir_transform, dst_crs=nir_crs,
              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)

    labels_src = np.arange(gh * gw, dtype=np.int32).reshape(gh, gw)
    labels_src[~np.isfinite(raw)] = -1
    label_grid = np.full((H, W), -1, dtype=np.int32)
    reproject(source=labels_src, destination=label_grid,
              src_transform=g_tfm, src_crs=g_crs,
              dst_transform=nir_transform, dst_crs=nir_crs,
              resampling=Resampling.nearest, src_nodata=-1, dst_nodata=-1)

    valid_nirv = np.isfinite(nirv) & (nirv > 0)
    valid_gosif = (label_grid >= 0) & np.isfinite(gosif_reproj)
    sif_10m = np.full((H, W), np.nan, dtype=np.float32)
    n_used = 0
    for lbl in np.unique(label_grid[valid_gosif & valid_nirv]):
        if lbl < 0:
            continue
        mask = (label_grid == lbl) & valid_nirv & valid_gosif
        if not mask.any():
            continue
        nirv_mean = float(nirv[mask].mean())
        if nirv_mean <= 0:
            continue
        S_k = float(gosif_reproj[mask][0])
        sif_10m[mask] = (S_k * nirv[mask] / (nirv_mean + eps)).astype(np.float32)
        n_used += 1
    return sif_10m, n_used


def _plot(out_dir, year, slope, intercept, r2, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        cube, doys, lon, lat, _ = load_sif_stack(out_dir, pattern="GOSIF10m_8day_*.nc")
    except FileNotFoundError:
        return
    with np.errstate(invalid="ignore"):
        ts = np.array([np.nanmean(cube[t]) for t in range(cube.shape[0])])
    mid = int(np.nanargmax(ts)) if np.isfinite(ts).any() else cube.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(doys, ts, "-o", ms=4, color="teal")
    ax[0].set_xlabel("DOY"); ax[0].set_ylabel("mean 10 m GOSIF SIF")
    ax[0].set_title(f"Downscaled GOSIF ({year})\n"
                    f"calib ESA={slope:.3f}*GOSIF+{intercept:.3f}  R2={r2:.3f} n={n}")
    ax[0].grid(alpha=0.3)
    im = ax[1].imshow(cube[mid], cmap="viridis",
                      vmin=0, vmax=np.nanpercentile(cube[mid], 98))
    ax[1].set_title(f"Downscaled 10 m GOSIF (DOY {int(doys[mid])})"); ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="SIF")
    fig.tight_layout()
    pdir = out_dir / "plots"; pdir.mkdir(exist_ok=True)
    fig.savefig(pdir / f"step3_gosif_downscaled_{year}.png", dpi=140)
    plt.close(fig)


def run(year, force=False):
    C.require_dir(C.GOSIF_ROOT, "GOSIF folder")   # a missing year is fine (ESA only), a missing folder is not
    import rasterio

    nirv_dir = C.nirv_gapfill(year)
    esa_dir = C.sif_downscaled(year)
    gosif_dir = C.gosif_dir(year)
    out_dir = C.gosif_downscaled(year)

    if not outputs_present(nirv_dir, "nirv_gapfilled_8day_DOY*.tif", 1):
        info(f"[{year}] skip: gap-filled NIRv not found (run 3_nirv_gapfill.py)")
        return
    gosif_files = list_gosif_files(gosif_dir, year)
    if not gosif_files:
        info(f"[{year}] skip: no GOSIF tiles in {gosif_dir} "
             f"(GOSIF not available for {year}; 6_merge_gapfill.py will be ESA-only)")
        return
    if (not force) and C.is_done(out_dir):
        n = len([p for p in out_dir.glob('GOSIF10m_8day_*.nc')])
        info(f"[{year}] already done ({n} files) -> skip")
        return

    banner(f"5  GOSIF -> 10 m downscaling   year={year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    info(f"{len(gosif_files)} GOSIF tiles for {year} (from {C.GOSIF_ROOT})")

    slope, intercept, r2, n = calibrate_gosif_to_esa(esa_dir, gosif_files, year)
    info(f"calibration: ESA = {slope:.4f}*GOSIF + {intercept:.4f}  "
         f"R2={r2:.3f}  n={n}")

    nirv_items = list_nirv_tiles(nirv_dir, year)
    n_written = 0
    for d, doy, nirv_path in nirv_items:
        gpath, gdoy = find_gosif_for_doy(gosif_files, doy)
        with rasterio.open(nirv_path) as src:
            nirv = src.read(1).astype(np.float32)
            nir_transform, nir_crs = src.transform, src.crs
            lon0, lat_top = src.bounds.left, src.bounds.top
            res = src.res
        nirv[~np.isfinite(nirv)] = np.nan
        nirv[nirv <= 0] = np.nan
        sif_10m, n_used = downscale_gosif(gpath, nirv, nir_transform, nir_crs,
                                          slope, intercept)
        if n_used == 0:
            continue
        H, W = sif_10m.shape
        lon1d = lon0 + res[0] * (0.5 + np.arange(W))
        lat1d = lat_top - res[1] * (0.5 + np.arange(H))
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        save_sif_netcdf(out_dir / f"GOSIF10m_8day_{d.strftime('%Y%m%d')}.nc",
                        sif_10m, lon2d, lat2d, d,
                        description=f"Downscaled 10 m GOSIF (cal {slope:.3f}x+{intercept:.3f})")
        n_written += 1
    info(f"downscaled {n_written}/{len(nirv_items)} GOSIF composites -> {out_dir}")
    np.savez(out_dir / f"gosif_calibration_{year}.npz",
             slope=slope, intercept=intercept, r2=r2, n=n)
    _plot(out_dir, year, slope, intercept, r2, n)
    C.mark_done(out_dir)
    info(f"done -> {out_dir}")


if __name__ == "__main__":
    for yr in C.YEARS:
        run(yr, force=C.FORCE_RERUN)
