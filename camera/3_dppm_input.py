"""Daily DPPM input per camera: mean GCC of the images that passed QC, plus
daily air temperature from the nearest weather station and daytime PAR. DPPM
was fitted with daytime means, so the 24 h mean PAR in the met csv is scaled
by 24 / day length (hours of sun above the horizon at the camera latitude).

in:  camera/traffic/<year>/per_loc/locXXXX.csv, camera/phenocam/<camera>/gcc.csv,
     met csv (date, Temp_C_<STATION>..., PAR_Wm2), camera -> station mapping
out: camera/dppm_input/traffic/<year>/locXXXX.csv
     camera/dppm_input/phenocam/<camera>.csv
     columns: date, GCC_tree, GCC_grass (traffic) or GCC (phenocam), TA_C, PPFD_umol
     camera/dppm_input/settings.json   (paths for dppm_run.m, MATLAB can't read yaml)
"""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import utils as U

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

PAR_TO_PPFD = 4.57   # W m-2 -> umol m-2 s-1


def day_length_h(dates, lat_deg):
    """Hours between sunrise and sunset (sun centre at the horizon)."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)
    cos_w = -np.tan(np.radians(lat_deg)) * np.tan(decl)
    return 2 * np.degrees(np.arccos(np.clip(cos_w, -1, 1))) / 15.0


met = pd.read_csv(U.path(U.CAM["met_csv"]), parse_dates=["date"])
met["date"] = met["date"].dt.normalize()
met["PPFD_umol"] = met["PAR_Wm2"] * PAR_TO_PPFD * 24.0 / day_length_h(met["date"], U.CAM["lat"])   # daytime mean
stations = pd.read_csv(U.path(U.CAM["station_mapping_csv"])).set_index("location")["nearest_station_name"]


def temp_col(station):
    s = re.sub(r"[^A-Z0-9]+", "_", station.strip().upper()).strip("_")
    return "Temp_C_" + s


def add_met(daily, station):
    m = met[["date", temp_col(station), "PPFD_umol"]].rename(columns={temp_col(station): "TA_C"})
    return daily.merge(m, on="date", how="left")


def traffic():
    for year in U.YEARS:
        src = U.OUT / "traffic" / str(year) / "per_loc"
        out = U.OUT / "dppm_input" / "traffic" / str(year)
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(src.glob("loc*.csv")):
            loc = f.stem
            if loc not in stations.index:
                print(f"  {loc}: no weather station mapped, skipped")
                continue
            df = pd.read_csv(f, parse_dates=["datetime"])
            df = df[df["qc_passed"].astype(str).str.lower() == "true"]
            df["date"] = df["datetime"].dt.normalize()
            daily = (df.groupby("date")[["gcc_tree_mean", "gcc_grass_mean"]].mean()
                     .rename(columns={"gcc_tree_mean": "GCC_tree", "gcc_grass_mean": "GCC_grass"})
                     .reset_index())
            daily = daily[daily["date"].dt.year == year]
            daily = daily[daily[["GCC_tree", "GCC_grass"]].notna().any(axis=1)]
            if daily.empty:
                continue
            add_met(daily, stations[loc]).to_csv(out / f"{loc}.csv", index=False)
            n += 1
        print(f"traffic {year}: {n} cameras -> {out}")


def phenocam():
    out = U.OUT / "dppm_input" / "phenocam"
    out.mkdir(parents=True, exist_ok=True)
    for name in U.CAM["phenocams"]:
        df = pd.read_csv(U.OUT / "phenocam" / name / "gcc.csv", parse_dates=["datetime"])
        df["date"] = df["datetime"].dt.normalize()
        daily = df.groupby("date")["mean"].mean().rename("GCC").reset_index()
        daily = daily[daily["date"].dt.year.isin(U.YEARS) & daily["GCC"].notna()]
        add_met(daily, U.CAM["phenocam_station"]).to_csv(out / f"{name}.csv", index=False)
        print(f"phenocam {name}: {len(daily)} days")


def settings():
    s = {"input_dir": str(U.OUT / "dppm_input"), "output_dir": str(U.OUT / "gpp"),
         "dppm_dir": str(U.path(U.CAM["dppm_dir"])),
         "params_csv": str(Path(__file__).resolve().parent / U.CAM["dppm_params"]),   # relative to camera/
         "n_mc": U.CAM["n_mc"], "years": U.YEARS}
    json.dump(s, open(U.OUT / "dppm_input" / "settings.json", "w"), indent=2)


if __name__ == "__main__":
    traffic()
    phenocam()
    settings()
