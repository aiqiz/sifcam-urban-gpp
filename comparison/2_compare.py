"""Compare camera GPP with SIF (SMUrF) GPP on the same footprints.

  agreement.csv  8-day pairs pooled over sites, per season and all year:
                 n, slope, intercept, r2, RMSE, MAE, mean bias (camera - SIF)
  late_summer.csv  August / June GPP per site and year, both methods
  phenology.csv  SOS / EOS per site, year and series (20 % and 50 % of the
                 seasonal amplitude, smoothed as in the paper)
  summary.csv    city medians of the above, per year

in:  comparison/<traffic|phenocam>/per_camera_csv/
out: comparison/summary/*.csv
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

ROOT = Path(__file__).resolve().parents[1]
_cfg_file = ROOT / "config.local.yaml"          # your own copy of config.yaml, not in git
CFG = yaml.safe_load(open(_cfg_file if _cfg_file.exists() else ROOT / "config.yaml"))
YEARS = CFG["camera"]["years"]
CMP_DIR = Path(CFG["results_dir"]) / "comparison"
OUT = CMP_DIR / "summary"

SEASONS = {"DJF": {12, 1, 2}, "MAM": {3, 4, 5}, "JJA": {6, 7, 8}, "SON": {9, 10, 11},
           "all": set(range(1, 13))}

# phenology settings (same as the paper)
THRESHOLDS = [0.2, 0.5]
SG_WIN, SG_POLY = 7, 2
SPIKE_MED, SPIKE_VALUE = 5, 3.0
ENV_ITER, ADAPT = 3, 3
PRE_SPRING_CUT = 100      # hold values before this DOY at the dormant baseline
MIN_VALID = 12
MAX_GAP_DAYS = 32         # no SOS / EOS if the series has a longer hole in the growing season
SOS_RANGE, EOS_RANGE = (80, 180), (200, 345)
DOY_FINE = np.arange(1, 366)
QC = {"gpp": dict(amp_min=0.8, rel_amp=0.35), "gcc": dict(amp_min=0.005, rel_amp=0.0)}
SERIES = {"cam_whole": ("gpp_camera_mean", "gpp"), "cam_tree": ("gpp_camera_tree", "gpp"),
          "cam_grass": ("gpp_camera_grass", "gpp"), "gcc_tree": ("gcc_tree", "gcc"),
          "gcc_grass": ("gcc_grass", "gcc"), "sif_whole": ("gpp_SMUrF", "gpp"),
          "sif_tree": ("gpp_SMUrF_tree", "gpp"), "sif_grass": ("gpp_SMUrF_grass", "gpp")}


def read_site(path):
    rows = Path(path).read_text().splitlines()
    ms = rows.index("# META"); ts = rows.index("# TIME_SERIES")
    meta = pd.read_csv(StringIO("\n".join(rows[ms + 1:ts])))
    df = pd.read_csv(StringIO("\n".join(rows[ts + 1:])), parse_dates=["date"], na_values=["--"])
    return dict(zip(meta["key"], meta["value"])), df


# ---- agreement -----------------------------------------------------------------
def fit_stats(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(n=len(x), slope=slope, intercept=intercept, r2=r * r,
                rmse_fit=np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)),
                rmse_11=np.sqrt(np.mean((y - x) ** 2)), mae_11=np.mean(np.abs(y - x)),
                mbe=np.mean(y - x))


def agreement(sites, kind):
    allp = pd.concat([df.assign(site=s) for s, (_, df) in sites.items()])
    allp = allp.dropna(subset=["gpp_SMUrF", "gpp_camera_mean"])
    rows = []
    for yr in YEARS + ["all"]:
        d = allp if yr == "all" else allp[allp["date"].dt.year == yr]
        for sname, months in SEASONS.items():
            p = d[d["date"].dt.month.isin(months)]
            if len(p) >= 2:
                rows.append({"cameras": kind, "year": yr, "season": sname,
                             **fit_stats(p["gpp_SMUrF"].to_numpy(), p["gpp_camera_mean"].to_numpy())})
    return pd.DataFrame(rows)


# ---- late summer ------------------------------------------------------------------
def late_summer(sites, kind):
    rows = []
    for s, (_, df) in sites.items():
        for yr in YEARS:
            d = df[df["date"].dt.year == yr]
            jun, aug = d[d["date"].dt.month == 6], d[d["date"].dt.month == 8]
            rows.append({"cameras": kind, "site": s, "year": yr,
                         "sif_aug_over_jun": aug["gpp_SMUrF"].mean() / jun["gpp_SMUrF"].mean(),
                         "camera_aug_over_jun": aug["gpp_camera_mean"].mean() / jun["gpp_camera_mean"].mean()})
    return pd.DataFrame(rows)


# ---- phenology ------------------------------------------------------------------------
def adaptive_sg(Y):
    """Spike removal, then upper-envelope Savitzky-Golay (TIMESAT style)."""
    y = Y.copy()
    med = median_filter(y, size=(SPIKE_MED, 1), mode="nearest")
    dev = y - med
    sd = np.nanstd(dev, axis=0, keepdims=True)
    spike = np.abs(dev) > (SPIKE_VALUE * np.where(sd == 0, np.inf, sd))
    y[spike] = med[spike]
    pull = min(1.0, ADAPT / 3.0)
    for _ in range(ENV_ITER):
        f = savgol_filter(y, SG_WIN, SG_POLY, axis=0, mode="interp")
        y = np.where(y < f, y + pull * (f - y), y)
    return savgol_filter(y, SG_WIN, SG_POLY, axis=0, mode="interp")


def smooth_daily(doy, y):
    doy = np.asarray(doy, float); Y = np.array(y, dtype=float)[:, None]
    if np.isfinite(Y).sum() < MIN_VALID:
        return None
    m = np.isfinite(Y[:, 0])
    # a long hole (camera down, no images) would be bridged by the interpolation
    # below and could invent a plausible-looking SOS or EOS, so refuse instead
    season = doy[m][(doy[m] >= SOS_RANGE[0]) & (doy[m] <= EOS_RANGE[1])]
    edges = np.concatenate([[SOS_RANGE[0]], season, [EOS_RANGE[1]]])
    if np.diff(edges).max() > MAX_GAP_DAYS:
        return None
    Y[:, 0] = np.interp(doy, doy[m], Y[m, 0])
    pre = doy < PRE_SPRING_CUT
    Y[pre] = np.minimum(Y[pre], np.nanpercentile(Y, 10, axis=0)[None, :])
    F = adaptive_sg(Y)
    return interp1d(doy, F, axis=0, bounds_error=False, fill_value=(F[0], F[-1]))(DOY_FINE)[:, 0]


def timing(F, frac, amp_min, rel_amp):
    if F is None:
        return np.nan, np.nan
    base, peak = F.min(), F.max(); amp = peak - base
    above = F >= base + frac * amp
    if not above.any():
        return np.nan, np.nan
    s = DOY_FINE[above.argmax()]; e = DOY_FINE[len(F) - 1 - above[::-1].argmax()]
    ok = (amp >= amp_min and amp / max(abs(peak), 1e-9) >= rel_amp and SOS_RANGE[0] <= s <= SOS_RANGE[1]
          and EOS_RANGE[0] <= e <= EOS_RANGE[1] and e > s)
    return (float(s), float(e)) if ok else (np.nan, np.nan)


def phenology(sites, kind):
    rows = []
    for s, (_, df) in sites.items():
        for yr in YEARS:
            d = df[df["date"].dt.year == yr]
            if d.empty:
                continue
            for name, (col, qk) in SERIES.items():
                if col not in d:
                    continue
                F = smooth_daily(d["date"].dt.dayofyear, d[col])
                for thr in THRESHOLDS:
                    sos, eos = timing(F, thr, **QC[qk])
                    rows.append({"cameras": kind, "site": s, "year": yr, "series": name,
                                 "thr": thr, "SOS": sos, "EOS": eos})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    agr, ls, ph = [], [], []
    for kind in ["traffic", "phenocam"]:
        files = sorted(f for f in (CMP_DIR / kind / "per_camera_csv").glob("*.csv") if not f.name.startswith("."))
        sites = {f.stem: read_site(f) for f in files}
        if not sites:
            continue
        agr.append(agreement(sites, kind)); ls.append(late_summer(sites, kind)); ph.append(phenology(sites, kind))
    agr, ls, ph = pd.concat(agr), pd.concat(ls), pd.concat(ph)
    agr.to_csv(OUT / "agreement.csv", index=False)
    ls.to_csv(OUT / "late_summer.csv", index=False)
    ph.to_csv(OUT / "phenology.csv", index=False)

    # city medians, the numbers quoted in the paper
    med_ls = ls.groupby(["cameras", "year"])[["sif_aug_over_jun", "camera_aug_over_jun"]].median()
    med_ph = ph.pivot_table(index=["cameras", "year"], columns=["series", "thr"],
                            values=["SOS", "EOS"], aggfunc="median")
    med_ph.columns = [f"{m}_{s}_{int(t * 100)}" for m, s, t in med_ph.columns]
    summary = med_ls.join(med_ph).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(2).to_string(index=False))
    print(agr[agr["season"] == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
