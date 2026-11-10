"""Put the two GPP estimates on the same footprint.

Traffic cameras: 100 m circle around the camera. Phenocams: 800 m wedge from
the tower in the camera's view direction. Inside each footprint we
  - average SMUrF GPP (and SIF) over the 10 m pixels, split into tree / grass /
    other pixels by the City of Toronto 2018 land cover;
  - weight camera GPP by land cover:
      GPP_tree   = 0.9 GPP_Deciduous + 0.1 GPP_Evergreen
      GPP_camera = p_tree GPP_tree + p_grass GPP_Grasses
    and average it over the +/- 4 days of each 8-day SIF composite.

in:  sif/<region>/8.GPP_10m/easternCONUS/, camera/gpp/
out: comparison/<traffic|phenocam>/per_camera_daily_csv/<site>.csv   (camera, daily)
     comparison/<traffic|phenocam>/per_camera_csv/<site>.csv         (both, 8-day; META block on top)
     comparison/<traffic|phenocam>/landcover_fractions.csv
"""

from pathlib import Path

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import pyproj
import yaml
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

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

CMP = CFG["comparison"]
YEARS = CFG["camera"]["years"]
RES = Path(CFG["results_dir"])
GPP_DIR = RES / "sif" / "toronto" / "8.GPP_10m" / "easternCONUS"
OUT = RES / "comparison"

FILL = -999.0
UMOL_TO_GC_DAY = 12.011 * 86400 * 1e-6   # umol CO2 m-2 s-1 (24 h mean) -> gC m-2 d-1
F_DEC = CMP["frac_deciduous"]
LC_NAMES = {1: "Tree", 2: "Grass", 3: "Bare", 4: "Water", 5: "Building",
            6: "Road", 7: "Other paved", 8: "Shrub"}
TREE_CLS, GRASS_CLS = {"Tree"}, {"Grass", "Shrub"}   # shrub counts as grass

to_mtm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2952", always_xy=True)
to_wgs = pyproj.Transformer.from_crs("EPSG:2952", "EPSG:4326", always_xy=True)


# ---- footprints ---------------------------------------------------------------
def circle(lon, lat, r):
    x, y = to_mtm.transform(lon, lat)
    return Point(x, y).buffer(r)


def wedge(lon, lat, r, az, half, n_arc=192):
    """Azimuth clockwise from north, polygon in EPSG:2952."""
    cx, cy = to_mtm.transform(lon, lat)
    a = np.radians(90.0 - np.linspace(az - half, az + half, n_arc))
    return Polygon([(cx, cy)] + list(zip(cx + r * np.cos(a), cy + r * np.sin(a))) + [(cx, cy)])


def circle_pixels(lon, lat, r, glat, glon):
    """(rows, cols) of grid cells whose centre is within r metres (great circle)."""
    m = r / 111_320.0 * 3
    li = np.where((glat >= lat - m) & (glat <= lat + m))[0]
    oi = np.where((glon >= lon - m) & (glon <= lon + m))[0]
    if li.size == 0 or oi.size == 0:
        return np.array([], int), np.array([], int)
    mlon, mlat = np.meshgrid(glon[oi], glat[li])
    dlat, dlon = np.radians(mlat - lat), np.radians(mlon - lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat)) * np.cos(np.radians(mlat)) * np.sin(dlon / 2) ** 2
    rr, cc = np.where(6_371_000.0 * 2 * np.arcsin(np.sqrt(a)) <= r)
    return li[rr], oi[cc]


def pixels_in(poly, glat, glon):
    """(rows, cols) of grid cells whose centre is inside poly."""
    x0, y0, x1, y1 = poly.bounds
    lons, lats = to_wgs.transform([x0, x1, x1, x0], [y0, y0, y1, y1])
    # small margin so cells right on the bounding box edge are tested too
    m = 0.0005
    li = np.where((glat >= min(lats) - m) & (glat <= max(lats) + m))[0]
    oi = np.where((glon >= min(lons) - m) & (glon <= max(lons) + m))[0]
    if li.size == 0 or oi.size == 0:
        return np.array([], int), np.array([], int)
    mlon, mlat = np.meshgrid(glon[oi], glat[li])
    xs, ys = to_mtm.transform(mlon.ravel(), mlat.ravel())
    pp = prep(poly)
    inside = np.array([pp.contains(Point(x, y)) for x, y in zip(xs, ys)]).reshape(mlon.shape)
    rr, cc = np.where(inside)
    return li[rr], oi[cc]


def area_fractions(lc, poly, tree_cls, grass_cls):
    x0, y0, x1, y1 = poly.bounds
    clip = lc.cx[x0:x1, y0:y1].copy()
    if clip.empty:
        return 0.0, 0.0
    clip["a"] = clip.geometry.intersection(poly).area
    return (clip.loc[clip["class_name"].isin(tree_cls), "a"].sum() / poly.area,
            clip.loc[clip["class_name"].isin(grass_cls), "a"].sum() / poly.area)


def pixel_classes(lc, glon_sel, glat_sel):
    xs, ys = to_mtm.transform(glon_sel, glat_sel)
    pts = gpd.GeoDataFrame({"i": np.arange(len(xs))}, geometry=gpd.points_from_xy(xs, ys), crs="EPSG:2952")
    x0, y0, x1, y1 = pts.total_bounds
    clip = lc.cx[x0:x1, y0:y1]
    if clip.empty:
        return np.full(len(xs), None, dtype=object)
    j = gpd.sjoin(pts, clip[["class_name", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j.set_index("i")["class_name"].reindex(np.arange(len(xs))).values


# ---- SMUrF ------------------------------------------------------------------------
def smurf_series(sites):
    """8-day SMUrF GPP and SIF inside each footprint, split by pixel class."""
    files = sorted(f for y in YEARS for f in (GPP_DIR / "daily" / str(y)).glob("GPP10m_*.nc"))
    dates = pd.to_datetime([f.stem.split("_")[1] for f in files], format="%Y%m%d")

    with nc.Dataset(files[0]) as ds:
        glat0 = float(ds.variables["latitude"][0, 0]); glon0 = float(ds.variables["longitude"][0, 0])
    sif_ds, sif_idx, sif_off = {}, {}, {}
    for y in YEARS:
        p = GPP_DIR / f"daily_mean_SIF_GPP_uncert_easternCONUS_{y}_10m.nc"
        if p.exists():
            sif_ds[y] = nc.Dataset(p)
            # the yearly file may cover a larger grid than the daily files: find the offset
            ylat = sif_ds[y].variables["lat"][:]; ylon = sif_ds[y].variables["lon"][:]
            sif_off[y] = (int(np.argmin(np.abs(ylat - glat0))), int(np.argmin(np.abs(ylon - glon0))))
            for i, t in enumerate(pd.to_datetime(sif_ds[y].variables["time"][:], unit="s")):
                sif_idx[pd.Timestamp(t).normalize()] = (y, i)

    recs = {s: [] for s in sites}
    for f, d in zip(files, dates):
        with nc.Dataset(f) as ds:
            gpp = np.ma.filled(ds.variables["GPP_mean"][:], np.nan).astype(float)
            sd = np.ma.filled(ds.variables["GPP_sd"][:], np.nan).astype(float)
        gpp[gpp == FILL] = np.nan; sd[sd == FILL] = np.nan
        gpp *= UMOL_TO_GC_DAY; sd *= UMOL_TO_GC_DAY
        sif = None
        if d.normalize() in sif_idx:
            y, i = sif_idx[d.normalize()]
            r0, c0 = sif_off[y]
            sif = np.ma.filled(sif_ds[y].variables["SIF_mean"][i, r0:r0 + gpp.shape[0], c0:c0 + gpp.shape[1]],
                               np.nan).astype(float)
            sif[sif == FILL] = np.nan

        for s, info in sites.items():
            r, c = info["rows"], info["cols"]
            g, e = gpp[r, c], sd[r, c]
            ok = ~np.isnan(g)
            if not ok.any():
                continue
            n = ok.sum()
            t, gr = info["tree_px"] & ok, info["grass_px"] & ok
            o = ok & ~info["tree_px"] & ~info["grass_px"]
            # tree / grass / other are per m2 of the whole footprint, they add up to the mean
            rec = {"date": d,
                   "gpp_SMUrF": g[ok].mean(), "gpp_SMUrF_sd": np.nanmean(e[ok]),
                   "gpp_SMUrF_tree": g[t].sum() / n, "gpp_SMUrF_tree_sd": np.nansum(e[t]) / n,
                   "gpp_SMUrF_grass": g[gr].sum() / n, "gpp_SMUrF_grass_sd": np.nansum(e[gr]) / n,
                   "gpp_SMUrF_other": g[o].sum() / n, "gpp_SMUrF_other_sd": np.nansum(e[o]) / n}
            if sif is not None:
                v = sif[r, c]; okv = ~np.isnan(v); nv = okv.sum()
                rec.update({"sif_SMUrF": v[okv].mean() if nv else np.nan,
                            "sif_SMUrF_tree": v[info["tree_px"] & okv].sum() / nv if nv else np.nan,
                            "sif_SMUrF_grass": v[info["grass_px"] & okv].sum() / nv if nv else np.nan,
                            "sif_SMUrF_other": v[okv & ~info["tree_px"] & ~info["grass_px"]].sum() / nv if nv else np.nan})
            recs[s].append(rec)
    return {s: pd.DataFrame(v).set_index("date") for s, v in recs.items() if v}, dates


# ---- camera -------------------------------------------------------------------------
def camera_daily(df, p_tree, p_grass, stream_tree, stream_grass):
    out = pd.DataFrame(index=df.index)
    for sfx, col in [("mean", ""), ("lo", "_p025"), ("hi", "_p975")]:
        tree = F_DEC * df[f"GPP_{stream_tree}_Deciduous{col}"] + (1 - F_DEC) * df[f"GPP_{stream_tree}_Evergreen{col}"]
        grass = df[f"GPP_{stream_grass}_Grasses{col}"]
        out[f"GPP_tree_{sfx}"] = tree
        out[f"GPP_grass_{sfx}"] = grass
        out[f"cam_tree_{sfx}"] = p_tree * tree
        out[f"cam_grass_{sfx}"] = p_grass * grass
        out[f"cam_{sfx}"] = out[f"cam_tree_{sfx}"] + out[f"cam_grass_{sfx}"]
    return out


def window_mean(daily, centres, half=4):
    rows = [daily.loc[c - pd.Timedelta(days=half):c + pd.Timedelta(days=half)].mean() for c in centres]
    return pd.DataFrame(rows, index=centres)


def write_site_csv(path, ts, info, extra):
    """Small META block, then the 8-day time series (the format the paper
    figure scripts read)."""
    meta = {**extra, "n_smurf_pixels": len(info["rows"]), "n_tree_pixels": int(info["tree_px"].sum()),
            "n_grass_pixels": int(info["grass_px"].sum()),
            "tree_fraction": info["p_tree"], "grass_fraction": info["p_grass"],
            "frac_deciduous_of_tree": F_DEC, "frac_evergreen_of_tree": 1 - F_DEC,
            "years": "+".join(map(str, YEARS)), "gpp_unit": "gC m-2 d-1",
            "sif_unit": "mW m-2 nm-1 sr-1"}
    with open(path, "w") as fh:
        fh.write("# META\n")
        pd.DataFrame({"key": list(meta), "value": list(meta.values())}).to_csv(fh, index=False)
        fh.write("# TIME_SERIES\n")
        ts.rename_axis("date").reset_index().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d")) \
          .to_csv(fh, index=False)


def run(kind, footprints, read_camera, lc):
    tree_cls, grass_cls = TREE_CLS, GRASS_CLS
    with nc.Dataset(next((GPP_DIR / "daily" / str(YEARS[0])).glob("GPP10m_*.nc"))) as ds:
        glat = ds.variables["latitude"][:][:, 0].astype(float)
        glon = ds.variables["longitude"][:][0, :].astype(float)

    sites = {}
    for s, fp in footprints.items():
        poly = fp["poly"]
        r, c = circle_pixels(*fp["circle"], glat, glon) if "circle" in fp else pixels_in(poly, glat, glon)
        if r.size == 0:
            print(f"  {s}: outside the SMUrF grid")
            continue
        cls = pixel_classes(lc, glon[c], glat[r])
        pt, pg = area_fractions(lc, poly, tree_cls, grass_cls)
        sites[s] = {"rows": r, "cols": c, "p_tree": pt, "p_grass": pg,
                    "tree_px": np.isin(cls, list(tree_cls)), "grass_px": np.isin(cls, list(grass_cls))}

    field, dates = smurf_series(sites)
    out = OUT / kind
    (out / "per_camera_daily_csv").mkdir(parents=True, exist_ok=True)
    (out / "per_camera_csv").mkdir(parents=True, exist_ok=True)
    fr = []
    for s, info in sites.items():
        fr.append({"site": s, "p_tree": info["p_tree"], "p_grass": info["p_grass"],
                   "p_impervious": 1 - info["p_tree"] - info["p_grass"], "n_pixels": len(info["rows"])})
        cam = read_camera(s)
        if cam is None or s not in field:
            continue
        daily = cam["daily_fn"](info["p_tree"], info["p_grass"])
        daily.rename_axis("date").to_csv(out / "per_camera_daily_csv" / f"{s}.csv", date_format="%Y-%m-%d")
        w = window_mean(daily, dates)
        ts = field[s].reindex(dates).copy()
        for a, b in [("gpp_camera_mean", "cam_mean"), ("gpp_camera_lo", "cam_lo"), ("gpp_camera_hi", "cam_hi"),
                     ("gpp_camera_tree", "cam_tree_mean"), ("gpp_camera_tree_lo", "cam_tree_lo"),
                     ("gpp_camera_tree_hi", "cam_tree_hi"), ("gpp_camera_grass", "cam_grass_mean"),
                     ("gpp_camera_grass_lo", "cam_grass_lo"), ("gpp_camera_grass_hi", "cam_grass_hi")]:
            ts[a] = w[b].values
        for g in [c for c in daily.columns if c.startswith("GCC")]:
            ts[g.lower()] = w[g].values
        write_site_csv(out / "per_camera_csv" / f"{s}.csv", ts, info, footprints[s].get("meta", {}))
    pd.DataFrame(fr).to_csv(out / "landcover_fractions.csv", index=False)
    print(f"{kind}: {len(sites)} sites -> {out}")


def main():
    lc = gpd.read_file(path(CMP["landcover_gdb"]))
    lc = lc[lc["gridcode"] > 0].copy()
    lc["class_name"] = lc["gridcode"].map(LC_NAMES)
    lc = lc[["class_name", "geometry"]]

    # traffic cameras
    locs = pd.read_csv(path(CMP["traffic_sites_csv"])).drop_duplicates("location")
    rad = CMP["traffic_radius_m"]
    fps = {r.location: {"poly": circle(r.lon, r.lat, rad), "circle": (r.lon, r.lat, rad),
                        "meta": {"loc_index": r.location, "lon": r.lon, "lat": r.lat, "radius_m": rad}}
           for r in locs.itertuples()}

    def traffic_cam(loc):
        parts = [pd.read_csv(f, parse_dates=["date"]) for y in YEARS
                 if (f := RES / "camera" / "gpp" / "traffic" / str(y) / f"{loc}.csv").exists()]
        if not parts:
            return None
        df = pd.concat(parts).set_index("date").sort_index()
        def fn(pt, pg):
            d = camera_daily(df, pt, pg, "tree", "grass")
            d["GCC_tree"], d["GCC_grass"] = df["GCC_tree"], df["GCC_grass"]
            return d
        return {"daily_fn": fn}
    run("traffic", fps, traffic_cam, lc)

    # phenocams
    fps = {cam: {"poly": wedge(CMP["tower_lon"], CMP["tower_lat"], CMP["wedge_radius_m"], az, half),
                 "meta": {"loc_index": cam, "label": cam, "tower_lat": CMP["tower_lat"], "tower_lon": CMP["tower_lon"],
                          "wedge_azimuth_deg_from_N": az, "wedge_half_angle_deg": half,
                          "wedge_radius_m": CMP["wedge_radius_m"]}}
           for cam, (az, half) in CMP["wedges"].items()}

    def pheno_cam(cam):
        f = RES / "camera" / "gpp" / "phenocam" / f"{cam}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        def fn(pt, pg):
            d = camera_daily(df, pt, pg, "gcc", "gcc")
            d["GCC_wedge"] = df["GCC"]
            return d
        return {"daily_fn": fn}
    run("phenocam", fps, pheno_cam, lc)


if __name__ == "__main__":
    main()
