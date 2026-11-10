"""Gap-fill and smooth the 8-day NIRv per pixel. Trend pixels: linear
interpolation then Savitzky-Golay. Background pixels: broad triangular smooth,
only filling gaps. This keeps the double-peak shape of urban grass.

in:  1.NIRv_8day_year/nirv_inputs_<year>.npz
out: 2.NIRv_gapfilled/<year>/nirv_gapfilled_8day_DOY###.tif
"""

import os
from multiprocessing import Pool

import numpy as np

import utils as C
from utils import (banner, info, outputs_present, doy_to_date)

# scipy / rasterio imported lazily inside functions so `import utils` stays cheap


# ------------------------------------------------------------------ #
#  Per-pixel 1-D smoothers
# ------------------------------------------------------------------ #
def simple_temporal_fill_1d(y):
    y = np.asarray(y, dtype=float)
    idx = np.arange(y.size)
    good = np.isfinite(y)
    if good.sum() == 0:
        return y.copy()
    if good.sum() == 1:
        out = np.full_like(y, np.nan)
        out[:] = y[good][0]
        return out
    out = y.copy()
    out[~good] = np.interp(idx[~good], idx[good], y[good])
    return out


def big_window_smooth_nan(y, half_window=4):
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() == 0:
        return np.full_like(y, np.nan)
    if valid.sum() == 1:
        out = np.full_like(y, np.nan)
        out[:] = y[valid][0]
        return out
    y_fill = np.where(valid, y, 0.0)
    w_valid = valid.astype(float)
    ramp = np.arange(1, half_window + 2, dtype=float)
    kernel = np.concatenate([ramp, ramp[-2::-1]])
    num = np.convolve(y_fill, kernel, mode="same")
    den = np.convolve(w_valid, kernel, mode="same")
    out = np.full_like(y, np.nan, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def fit_sg_pixel(y, min_points, window, polyorder):
    from scipy.signal import savgol_filter
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < min_points:
        return simple_temporal_fill_1d(y)
    y_filled = simple_temporal_fill_1d(y)
    if y_filled.size < window:
        return y_filled
    return savgol_filter(y_filled, window_length=window, polyorder=polyorder)


# ------------------------------------------------------------------ #
#  Multiprocess worker (module globals set per-worker by initializer)
# ------------------------------------------------------------------ #
_nirv = None
_coords = None
_mode = None
_params = None


def _init_worker(nirv, coords, mode, params):
    global _nirv, _coords, _mode, _params
    _nirv, _coords, _mode, _params = nirv, coords, mode, params


def _process_pixel(idx):
    r, c = _coords[idx]
    y = _nirv[:, r, c]
    if not np.isfinite(y).any():
        return int(r), int(c), np.full_like(y, np.nan, dtype=np.float32)
    if _mode == "trend":
        out = fit_sg_pixel(y, _params["min_points"],
                           _params["window"], _params["polyorder"])
    elif _mode == "background":
        y_sm = big_window_smooth_nan(y, half_window=_params["bg_half"])
        out = y.copy()
        miss = ~np.isfinite(out)
        out[miss] = y_sm[miss]
    else:
        out = simple_temporal_fill_1d(y)
    return int(r), int(c), out.astype(np.float32)


# ------------------------------------------------------------------ #
#  Plot
# ------------------------------------------------------------------ #
def _plot(out_cube, raw_cube, bin_centers, trend_mask, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = out_cube.shape[0]
    # spatial-mean seasonal cycle over trend pixels
    tm = trend_mask
    with np.errstate(invalid="ignore"):
        raw_mean = np.array([np.nanmean(raw_cube[t][tm]) for t in range(T)])
        fil_mean = np.array([np.nanmean(out_cube[t][tm]) for t in range(T)])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(bin_centers, raw_mean, "o", ms=4, color="0.6",
                 label="raw 8-day (trend px mean)")
    axes[0].plot(bin_centers, fil_mean, "-", lw=2.2, color="tab:green",
                 label="SG gap-filled")
    axes[0].set_xlabel("DOY"); axes[0].set_ylabel("NIRv")
    axes[0].set_title("Mean seasonal cycle over trend pixels")
    axes[0].grid(alpha=0.3); axes[0].legend()

    mid = T // 2
    im = axes[1].imshow(out_cube[mid], cmap="YlGn", vmin=0,
                        vmax=np.nanpercentile(out_cube[mid], 98))
    axes[1].set_title(f"Gap-filled NIRv  (DOY {int(round(bin_centers[mid]))})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="NIRv")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Public entry point
# ------------------------------------------------------------------ #
def run(year, force=False):
    import rasterio

    out_dir = C.nirv_gapfill(year)
    plots_dir = out_dir / "plots"
    npz_path = C.nirv_npz(year)

    if not npz_path.exists():
        info(f"[{year}] skip: missing input {npz_path.name} (run 2_nirv_composite.py)")
        return

    # ---- skip-if-exists -------------------------------------------------
    if (not force) and C.is_done(out_dir):
        n = len([p for p in out_dir.glob('nirv_gapfilled_8day_DOY*.tif')])
        info(f"[{year}] already done ({n} tiles) -> skip")
        return

    banner(f"3  NIRv Savitzky-Golay gap-fill   year={year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    nirv = data["nirv_8d_obs"].astype(np.float32)
    trend_mask = data["trend_mask"]
    background_mask = data["background_mask"]
    bin_centers = data["bin_centers"]
    T, R, Cc = nirv.shape
    info(f"cube {nirv.shape}  trend={int(trend_mask.sum()):,}  "
         f"background={int((background_mask & ~trend_mask).sum()):,}")

    raw_cube = nirv.copy()
    out_cube = nirv.copy()
    params = dict(min_points=C.SG_MIN_POINTS, window=C.SG_WINDOW,
                  polyorder=C.SG_POLYORDER, bg_half=C.BG_HALF_WINDOW)

    # ---- TREND pixels (SG) ---------------------------------------------
    trend_coords = np.argwhere(trend_mask)
    if len(trend_coords):
        info(f"SG on {len(trend_coords):,} trend pixels "
             f"(win={C.SG_WINDOW}, poly={C.SG_POLYORDER}, procs={C.N_PROCS})")
        with Pool(processes=C.N_PROCS, initializer=_init_worker,
                  initargs=(nirv, trend_coords, "trend", params)) as pool:
            for r, c, y_out in pool.imap_unordered(
                    _process_pixel, range(len(trend_coords)), chunksize=2000):
                out_cube[:, r, c] = y_out

    # ---- BACKGROUND pixels (triangular smooth) -------------------------
    bg_coords = np.argwhere(background_mask & (~trend_mask))
    if len(bg_coords):
        info(f"triangular smooth on {len(bg_coords):,} background pixels")
        with Pool(processes=C.N_PROCS, initializer=_init_worker,
                  initargs=(nirv, bg_coords, "background", params)) as pool:
            for r, c, y_out in pool.imap_unordered(
                    _process_pixel, range(len(bg_coords)), chunksize=2000):
                out_cube[:, r, c] = y_out

    # ---- write GeoTIFFs -------------------------------------------------
    # georef template = the first per-date NIRv tile of this year
    cloud_dir = C.DIR_NIRV_CLOUD / str(year)
    templ = sorted(p for p in cloud_dir.glob("nirv_*.tif")
                   if not p.name.startswith("."))
    if not templ:
        raise FileNotFoundError(f"No template NIRv tile in {cloud_dir}")
    with rasterio.open(templ[0]) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)

    info(f"writing {T} gap-filled GeoTIFFs")
    for t in range(T):
        doy = int(round(bin_centers[t]))
        with rasterio.open(out_dir / f"nirv_gapfilled_8day_DOY{doy:03d}.tif",
                           "w", **profile) as dst:
            dst.write(out_cube[t], 1)

    np.savez(out_dir / "nirv_gapfilled_cube.npz",
             nirv_8d_filled=out_cube,
             bin_centers=bin_centers.astype(np.float32),
             trend_mask=trend_mask.astype(bool),
             background_mask=background_mask.astype(bool))

    out_png = plots_dir / f"step1_nirv_gapfill_{year}.png"
    _plot(out_cube, raw_cube, bin_centers, trend_mask, out_png)
    C.mark_done(out_dir)
    info(f"done -> {out_dir}")
    info(f"plot -> {out_png}")


if __name__ == "__main__":
    for yr in C.YEARS:
        run(yr, force=C.FORCE_RERUN)
