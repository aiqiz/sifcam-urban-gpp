"""Config and small image helpers shared by the camera/ scripts."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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

CAM = CFG["camera"]
YEARS = CAM["years"]
OUT = Path(CFG["results_dir"]) / "camera"


def solar_zenith(times, lat=CAM["lat"], lon=CAM["lon"]):
    """Solar zenith angle (deg). `times` are naive local Toronto times, which is
    how the camera filenames are stamped, so localize before computing."""
    from pvlib.location import Location
    idx = pd.DatetimeIndex(times).tz_localize("America/Toronto", ambiguous="NaT",
                                             nonexistent="shift_forward")
    site = Location(latitude=lat, longitude=lon, tz="America/Toronto")
    return site.get_solarposition(times=idx)["zenith"].to_numpy()


def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def edge_density(img_bgr, clahe=True):
    """Fraction of Canny edge pixels. Low values = fog, blur, dark or blank frames."""
    proc = apply_clahe(img_bgr) if clahe else img_bgr
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).sum() / edges.size)
