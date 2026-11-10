"""Phenocams: the cameras drift a little over time, so register images to a
reference image (AKAZE features + MAGSAC). Only every n-th usable image is
matched; 2_phenocam_gcc.py gives each image the nearest good transform.

in:  phenocam image folder + reference image (config: camera.phenocams)
out: camera/phenocam/<camera>/transformation.csv
"""

import os
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from skimage.restoration import wiener

import utils as U

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

QC = U.CAM["phenocam_qc"]
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TIME_RE = re.compile(r"(?:(\d{8})[_-](\d{6}|\d{4}))|(?:_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2}))")


def parse_time(fname):
    """YYYYMMDD_HHMMSS, YYYYMMDD_HHMM or _YYYY_MM_DD_HH_MM_SS in the filename."""
    m = TIME_RE.search(fname)
    if not m:
        return None
    if m.group(1):
        t = m.group(2) + ("00" if len(m.group(2)) == 4 else "")
        s = m.group(1) + t
    else:
        s = "".join(m.group(i) for i in range(3, 9))
    try:
        return datetime.strptime(s, "%Y%m%d%H%M%S")
    except ValueError:
        return None


def usable_images(folder, lat, lon):
    """(datetime, path) for images inside the time-of-day and sun-angle window."""
    items = [(parse_time(f), os.path.join(folder, f)) for f in os.listdir(folder)
             if f.lower().endswith(VALID_EXT) and not f.startswith(".")]
    items = sorted((t, p) for t, p in items if t is not None)
    if not items:
        return []
    sza = U.solar_zenith([t for t, _ in items], lat, lon)
    return [(t, p) for (t, p), z in zip(items, sza)
            if QC["time_min"] <= t.hour <= QC["time_max"] and z <= QC["sza_max"]]


def prep(img):
    img = U.apply_clahe(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    psf = np.ones((5, 5)) / 25
    deblurred = wiener(gray.astype(np.float32) / 255.0, psf, balance=0.1)
    return np.clip(deblurred * 255, 0, 255).astype(np.uint8)


def match(ref_img, cur_img, k, ransac_px, ratio):
    """Homography that maps cur_img onto ref_img, or None."""
    h, w = ref_img.shape[:2]
    ref, cur = prep(ref_img), prep(cur_img)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // k:(h // k) * (k - 1), :] = 255     # skip sky and timestamp strip
    akaze = cv2.AKAZE_create(threshold=0.0002, nOctaves=6, nOctaveLayers=8,
                             diffusivity=cv2.KAZE_DIFF_CHARBONNIER)
    kp1, des1 = akaze.detectAndCompute(ref, mask)
    kp2, des2 = akaze.detectAndCompute(cur, mask)
    try:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
        good = [m for m, n in pairs if m.distance < ratio * n.distance]
    except cv2.error:
        return None
    if len(good) < 4:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    A, inl = cv2.findHomography(dst, src, method=cv2.USAC_MAGSAC, ransacReprojThreshold=ransac_px)
    if A is None or inl is None:
        return None
    return A, dst, src, inl


def match_quality(A, dst, src, inl, k):
    """Reprojection error of the inliers (lower part of the frame) and inlier ratio."""
    d, s = dst[inl.ravel() == 1], src[inl.ravel() == 1]
    keep = s[:, 0, 1] > np.max(s[:, :, 1]) / k
    d, s = d[keep], s[keep]
    err = np.linalg.norm(cv2.perspectiveTransform(d, A).squeeze() - s.squeeze(), axis=1)
    return err.mean(), np.median(err), err.max(), err.std(), inl.sum() / len(inl)


def align_camera(name, c):
    out_dir = U.OUT / "phenocam" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = cv2.imread(str(U.path(c["ref_image"])), cv2.IMREAD_COLOR)
    k = QC["ignore_top_bottom"]
    imgs = usable_images(U.path(c["image_dir"]), U.CAM["phenocam_lat"], U.CAM["phenocam_lon"])
    imgs = imgs[::QC["align_every_n"]]
    print(f"{name}: matching {len(imgs)} images")

    rows = []
    for t, p in imgs:
        img = cv2.imread(p)
        if img is None or U.edge_density(img, clahe=False) <= c["edge_density_min"]:
            continue
        res = match(ref, img, k, c["ransac_px"], c["ratio"])
        if res is None:
            continue
        A, dst, src, inl = res
        rows.append([t, *match_quality(A, dst, src, inl, k), *A.flatten().tolist()])
    cols = ["datetime", "e_mean", "e_median", "e_max", "e_std", "inlier_ratio"] + \
           [f"A{i}{j}" for i in range(3) for j in range(3)]
    pd.DataFrame(rows, columns=cols).to_csv(out_dir / "transformation.csv", index=False)
    print(f"  {len(rows)} transforms -> {out_dir / 'transformation.csv'}")


if __name__ == "__main__":
    for name, c in U.CAM["phenocams"].items():
        align_camera(name, c)
