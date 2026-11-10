"""Traffic cameras: segment tree and grass in every image with SegFormer
(ADE20K classes), compute GCC = G / (R + G + B) inside each class, and write
one time series per camera. Images failing QC (time of day, sun angle, edge
density) are kept as rows with qc_passed = False and no GCC.

in:  <traffic_image_dir>/<year>/<YYYYMMDD>/locXXXX.YYYYMMDD_HHMMSS.jpg
out: camera/traffic/<year>/daily/YYYYMMDD.csv     (all cameras, one file per day)
     camera/traffic/<year>/per_loc/locXXXX.csv    (one file per camera)
"""

import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import utils as U

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

CKPT = "nvidia/segformer-b0-finetuned-ade-512-512"
QC = U.CAM["traffic_qc"]
LOCS = set(U.CAM["traffic_locations"] or [])

MIN_REGION_AREA = 64      # drop segmented blobs smaller than this (px)
# camera name / timestamp overlays in the bottom corners, as fractions of a 400 x 255 frame
_REF_W, _REF_H = 400, 255
_HIDE_RECTS = [(55 / _REF_W, 65 / _REF_H, "left-bottom"),
               (350 / _REF_W, 25 / _REF_H, "right-bottom")]
FNAME_RE = re.compile(r"^(loc\d+)\.(\d{8}_\d{6})\.jpg$", re.IGNORECASE)

COLS = ["location", "datetime",
        "gcc_tree_mean", "gcc_tree_std", "gcc_tree_min", "gcc_tree_max", "gcc_tree_count",
        "gcc_grass_mean", "gcc_grass_std", "gcc_grass_min", "gcc_grass_max", "gcc_grass_count",
        "labeled_area_tree", "labeled_area_grass", "total_pixels",
        "sza_deg", "edge_density", "qc_passed"]

_seg = None   # one model per worker process


class VegSegmenter:
    def __init__(self, device="cpu"):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(CKPT, use_fast=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(CKPT).eval().to(device)
        label2id = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.tree_id = label2id.get("tree")
        self.grass_id = label2id.get("grass")

    def segment(self, image_pil):
        H, W = image_pil.size[1], image_pil.size[0]
        with self.torch.no_grad():
            inputs = self.processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pred = self.model(**inputs).logits.argmax(dim=1)[0].cpu().numpy().astype("uint8")
            pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)
        return pred == self.tree_id, pred == self.grass_id


def pick_device():
    """camera.device in config.yaml: auto (default), cpu, mps or cuda."""
    import torch
    want = U.CAM.get("device", "auto")
    if want != "auto":
        return want
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _init_worker():
    global _seg
    warnings.filterwarnings("ignore")
    _seg = VegSegmenter(pick_device())


def load_image(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = Image.open(path); im.load()
    except Exception:
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            im = Image.open(path); im.load()
        except Exception:
            im = None
    if im is None:   # some files only open through cv2
        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return im.convert("RGB") if im else None


def roi_mask(H, W):
    """Whole frame minus the text overlays."""
    mask = np.ones((H, W), dtype=bool)
    for w_frac, h_frac, align in _HIDE_RECTS:
        wp = max(1, int(round(w_frac * W))); hp = max(1, int(round(h_frac * H)))
        if align == "left-bottom":
            mask[H - hp:H, 0:wp] = False
        else:
            mask[H - hp:H, W - wp:W] = False
    return mask


def clean_mask(mask):
    m = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_REGION_AREA:
            out[labels == i] = 255
    return out > 0


def gcc_stats(image_pil, mask):
    none = {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    if mask is None or mask.sum() == 0:
        return none
    rgb = np.array(image_pil).astype(np.float32)
    denom = rgb.sum(axis=2)
    denom[denom == 0] = np.nan
    vals = (rgb[..., 1] / denom)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return none
    return {"mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()),
            "max": float(vals.max()), "count": int(vals.size)}


def process_day(day_dir, out_path):
    if out_path.exists():
        return f"{day_dir.name} done before"
    files = sorted(p for p in day_dir.iterdir() if FNAME_RE.match(p.name))
    if LOCS:
        files = [p for p in files if FNAME_RE.match(p.name).group(1) in LOCS]
    if not files:
        return f"{day_dir.name} no images"

    stamps = [datetime.strptime(FNAME_RE.match(p.name).group(2), "%Y%m%d_%H%M%S") for p in files]
    szas = U.solar_zenith(stamps)

    rows = []
    for p, dt, sza in zip(files, stamps, szas):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        ed = U.edge_density(bgr)   # whole frame (the ROI is the full frame)
        qc = (QC["time_min"] <= dt.hour <= QC["time_max"]) and (sza <= QC["sza_max"]) \
            and (ed >= QC["edge_density_min"])

        st = sg = {"mean": None, "std": None, "min": None, "max": None, "count": 0}
        area_tree = area_grass = None
        total = H * W
        if qc:
            im = load_image(p)
            if im is None:
                continue
            ih, iw = im.size[1], im.size[0]
            roi = roi_mask(ih, iw)
            tree_m, grass_m = _seg.segment(im)
            tree_m, grass_m = clean_mask(tree_m & roi), clean_mask(grass_m & roi)
            st, sg = gcc_stats(im, tree_m), gcc_stats(im, grass_m)
            area_tree, area_grass, total = int(tree_m.sum()), int(grass_m.sum()), ih * iw

        rows.append([FNAME_RE.match(p.name).group(1), dt.isoformat(),
                     st["mean"], st["std"], st["min"], st["max"], st["count"],
                     sg["mean"], sg["std"], sg["min"], sg["max"], sg["count"],
                     area_tree, area_grass, total, round(float(sza), 2), round(ed, 6), qc])

    pd.DataFrame(rows, columns=COLS).to_csv(out_path, index=False)
    return f"{day_dir.name} {len(rows)} images"


def run_year(year):
    in_dir = U.path(U.CAM["traffic_image_dir"]) / str(year)
    day_out = U.OUT / "traffic" / str(year) / "daily"
    loc_out = U.OUT / "traffic" / str(year) / "per_loc"
    day_out.mkdir(parents=True, exist_ok=True)
    loc_out.mkdir(parents=True, exist_ok=True)

    days = sorted(d for d in in_dir.iterdir() if d.is_dir() and re.fullmatch(r"\d{8}", d.name))
    print(f"{year}: {len(days)} days, {U.CAM['n_workers']} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=U.CAM["n_workers"], initializer=_init_worker) as pool:
        futs = {pool.submit(process_day, d, day_out / f"{d.name}.csv"): d for d in days}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"{futs[fut].name} ERROR {e}"
            print(f"  [{i}/{len(days)}] {msg}")
    print(f"  segmentation took {(time.time() - t0) / 3600:.1f} h")

    # one file per camera
    frames = [pd.read_csv(p, low_memory=False) for p in sorted(day_out.glob("????????.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["location", "datetime"])
    for loc, g in df.groupby("location"):
        g.to_csv(loc_out / f"{loc}.csv", index=False)
    print(f"  {df['location'].nunique()} cameras -> {loc_out}")


if __name__ == "__main__":
    for yr in U.YEARS:
        run_year(yr)
