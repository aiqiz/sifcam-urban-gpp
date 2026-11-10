"""Download Sentinel-2 L2A NIRv for every tile of the region, one GeoTIFF per
acquisition date. Clouds, shadows and snow (SCL 3, 8, 9, 10, 11) are set to NaN.

in:  Copernicus Data Space account (env vars SH_CLIENT_ID, SH_CLIENT_SECRET)
out: <s2_dir>/<tile>/<year>/nirv_YYYY-MM-DD.tif
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from sentinelhub import (SHConfig, CRS, BBox, DataCollection, SentinelHubRequest,
                         SentinelHubCatalog, MimeType, bbox_to_dimensions)

import utils as C

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

MAX_WORKERS = 5   # keep modest, the API rate-limits

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"]
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
S2 = DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url)
catalog = SentinelHubCatalog(config=config)

evalscript = """
//VERSION=3
function setup() {
  return {input: [{bands: ["B04", "B08", "SCL", "dataMask"]}],
          output: {bands: 1, sampleType: "FLOAT32"}};
}
function evaluatePixel(sample) {
  if (sample.dataMask === 0) { return [NaN]; }
  if ([3,8,9,10,11].includes(sample.SCL)) { return [NaN]; }
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return [sample.B08 * ndvi];
}
"""


def acquisition_dates(bbox, year):
    # the catalog search is free, so we only request dates that have data
    search = catalog.search(S2, bbox=bbox, time=(f"{year}-01-01", f"{year}-12-31"),
                            fields={"include": ["properties.datetime"], "exclude": []})
    return sorted({f["properties"]["datetime"][:10] for f in search})


def fetch_one(date, bbox, size, out_dir):
    fpath = out_dir / f"nirv_{date}.tif"
    if fpath.exists():
        return date, "skip", None
    t1 = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(data_collection=S2, time_interval=(date, t1))],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox, size=size, config=config)
    try:
        data = req.get_data()[0]
        if data.ndim == 3:
            data = data[:, :, 0]
        if not np.isfinite(data).any():
            return date, "empty", None
        with rasterio.open(fpath, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                           count=1, dtype="float32", crs="EPSG:4326", nodata=np.nan,
                           transform=rasterio.transform.from_bounds(*bbox, data.shape[1], data.shape[0])) as dst:
            dst.write(data.astype(np.float32), 1)
        return date, "ok", float(np.isfinite(data).mean())
    except Exception as e:
        return date, f"FAILED: {e}", None


def download_tile(tile, tile_bbox, year):
    bbox = BBox(bbox=tile_bbox, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=10)
    if max(size) > 2500:
        print(f"{tile}: {size} px is over the 2500 px limit, split the tile")
        return
    out_dir = C.s2_tile_dir(tile, year)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = acquisition_dates(bbox, year)
    print(f"{tile} {year}: {len(dates)} acquisitions")

    records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fetch_one, d, bbox, size, out_dir) for d in dates]
        for i, fut in enumerate(as_completed(futures), 1):
            date, status, vr = fut.result()
            print(f"  [{i}/{len(dates)}] {date} {status}")
            if status == "ok":
                records.append({"date": date, "filename": f"nirv_{date}.tif", "valid_ratio": vr})
    pd.DataFrame(records, columns=["date", "filename", "valid_ratio"]).sort_values("date") \
        .to_csv(out_dir / f"metadata_{tile}_{year}.csv", index=False)


if __name__ == "__main__":
    for year in C.YEARS:
        for tile, tile_bbox in C.REGION_CFG["s2_tiles"].items():
            download_tile(tile, tile_bbox, year)
