"""Fill pixels where TROPOMI SIF is missing with the downscaled GOSIF of the
same composite date.

in:  4.SIF_winter_corrected/<year>/, 5.GOSIF_downscaled/<year>/
out: 6.SIF_gapfilled/<year>/SIF10m_8day_YYYYMMDD.nc
"""

import numpy as np

import utils as C
from utils import (banner, info, outputs_present, list_dated_files,
                            read_sif_nc, save_sif_netcdf)


def _plot(out_dir, year, n_esa, n_filled):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils import load_sif_stack
    try:
        cube, doys, lon, lat, _ = load_sif_stack(out_dir)
    except FileNotFoundError:
        return
    with np.errstate(invalid="ignore"):
        ts = np.array([np.nanmean(cube[t]) for t in range(cube.shape[0])])
        cov = np.array([np.isfinite(cube[t]).mean() for t in range(cube.shape[0])])
    mid = int(np.nanargmax(ts)) if np.isfinite(ts).any() else cube.shape[0] // 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(doys, cov, "-o", ms=4, color="darkgreen")
    ax[0].set_xlabel("DOY"); ax[0].set_ylabel("valid pixel fraction")
    ax[0].set_title(f"Gap-filled SIF coverage ({year})\n"
                    f"ESA-valid px {n_esa:,} -> filled {n_filled:,}")
    ax[0].set_ylim(0, 1); ax[0].grid(alpha=0.3)
    im = ax[1].imshow(cube[mid], cmap="viridis",
                      vmin=0, vmax=np.nanpercentile(cube[mid], 98))
    ax[1].set_title(f"Gap-filled 10 m SIF (DOY {int(doys[mid])})"); ax[1].axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="SIF")
    fig.tight_layout()
    pdir = out_dir / "plots"; pdir.mkdir(exist_ok=True)
    fig.savefig(pdir / f"step4_gapfilled_{year}.png", dpi=140)
    plt.close(fig)


def run(year, force=False):
    esa_dir = C.sif_winter(year)
    gosif_dir = C.gosif_downscaled(year)
    out_dir = C.sif_gapfilled(year)

    if not outputs_present(esa_dir, "SIF10m_8day_*.nc", 1):
        info(f"[{year}] skip: bias-corrected SIF not found (run 4_tropomi_downscale.py)")
        return
    if (not force) and C.is_done(out_dir):
        n = len([p for p in out_dir.glob('SIF10m_8day_*.nc')])
        info(f"[{year}] already done ({n} files) -> skip")
        return

    banner(f"6  gap-fill ESA SIF with GOSIF   year={year}")
    out_dir.mkdir(parents=True, exist_ok=True)

    esa_items = {d.strftime("%Y%m%d"): p
                 for d, p in list_dated_files(esa_dir, "SIF10m_8day_*.nc")}
    gosif_items = {d.strftime("%Y%m%d"): p
                   for d, p in list_dated_files(gosif_dir, "GOSIF10m_8day_*.nc")}
    if not gosif_items:
        info(f"[{year}] WARNING: no downscaled GOSIF available -> "
             f"gap-fill is ESA-only (no fallback) for {len(esa_items)} composites")

    tags = sorted(esa_items)            # ESA defines the output composites
    tot_esa = tot_filled = n_filled_from_gosif = 0
    for tag in tags:
        from datetime import datetime
        d = datetime.strptime(tag, "%Y%m%d")
        esa, lon, lat = read_sif_nc(esa_items[tag])
        esa[~np.isfinite(esa)] = np.nan
        esa[esa >= 99999.0] = np.nan
        filled = esa.copy()
        if tag in gosif_items:
            gos, glon, glat = read_sif_nc(gosif_items[tag])
            gos[~np.isfinite(gos)] = np.nan
            gos[gos >= 99999.0] = np.nan
            if gos.shape == esa.shape:
                miss = ~np.isfinite(filled) & np.isfinite(gos)
                filled[miss] = gos[miss]
                n_filled_from_gosif += int(miss.sum())
            else:
                info(f"  {tag}: GOSIF shape {gos.shape} != ESA {esa.shape}, skip fill")
        tot_esa += int(np.isfinite(esa).sum())
        tot_filled += int(np.isfinite(filled).sum())
        lon2d, lat2d = (lon, lat) if lon.ndim == 2 else np.meshgrid(lon, lat)
        save_sif_netcdf(out_dir / f"SIF10m_8day_{tag}.nc",
                        filled, lon2d, lat2d, d,
                        description="Gap-filled 10 m SIF (ESA primary, GOSIF fallback)")
    info(f"composites={len(tags)}  ESA-valid={tot_esa:,}  "
         f"after-fill={tot_filled:,}  gaps closed by GOSIF={n_filled_from_gosif:,}")
    _plot(out_dir, year, tot_esa, tot_filled)
    C.mark_done(out_dir)
    info(f"done -> {out_dir}")


if __name__ == "__main__":
    for yr in C.YEARS:
        run(yr, force=C.FORCE_RERUN)
