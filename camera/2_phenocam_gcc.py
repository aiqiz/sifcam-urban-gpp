"""Phenocams: warp every usable image onto the reference view and compute GCC
inside the hand-drawn vegetation ROIs (labelme polygons). Pixels near the ROI
edge are down-weighted by how well the image was aligned.

in:  phenocam images, ROI json, camera/phenocam/<camera>/transformation.csv
out: camera/phenocam/<camera>/gcc.csv   (datetime, mean, std, min, max)
"""

import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

import utils as U
from importlib import import_module

import sys
sys.stdout.reconfigure(line_buffering=True)   # progress shows up even when piped to a file

align = import_module("1_phenocam_align")   # reuse the file listing / time parsing
QC = U.CAM["phenocam_qc"]


def roi_union(roi_json, W, H):
    """Union of all labelme polygons (same rasterization as labelme.utils.shape_to_mask)."""
    data = json.load(open(roi_json))
    mask = np.zeros((H, W), dtype=bool)
    for s in data["shapes"]:
        m = Image.fromarray(np.zeros((H, W), dtype=np.uint8))
        ImageDraw.Draw(m).polygon([tuple(p) for p in s["points"]], outline=1, fill=1)
        mask |= np.array(m, dtype=bool)
    return mask


def soft_interior_mask(union_mask, e_mean, e_std, min_weight=0.05):
    """Weight 1 inside the ROI, falling off toward the edge over ~e_mean px
    (the alignment error), so misalignment doesn't pull in non-ROI pixels."""
    weight = np.zeros(union_mask.shape, dtype=float)
    for region in regionprops(label(union_mask)):
        r0, c0, r1, c1 = region.bbox
        rmask = region.image.astype(bool)
        d = distance_transform_edt(~find_boundaries(rmask, mode="inner"))
        w = np.ones_like(d)
        zone = (d >= 0) & (d < e_mean)
        w[zone] = np.clip(np.exp(-0.5 * ((e_mean - d[zone]) / e_std) ** 2), min_weight, 1)
        w[d < 0] = 0
        w *= rmask
        weight[r0:r1, c0:c1] = np.maximum(weight[r0:r1, c0:c1], w)
    return weight


def weighted_gcc(img, weight):
    # cv2 loads BGR; GCC only needs the green channel (index 1) and the sum
    img = img.astype(np.float32)
    black = (img == 0).all(axis=2)
    denom = img.sum(axis=2)
    denom[denom == 0] = np.nan
    gcc = img[:, :, 1] / denom
    gcc[black] = np.nan
    ok = (weight > 0) & ~black & ~np.isnan(gcc)
    if not ok.any():
        return np.nan, np.nan, np.nan, np.nan
    v, w = gcc[ok], weight[ok]
    mu = np.average(v, weights=w)
    return mu, np.sqrt(np.average((v - mu) ** 2, weights=w)), v.min(), v.max()


def gcc_camera(name, c):
    cam_dir = U.OUT / "phenocam" / name
    t_df = pd.read_csv(cam_dir / "transformation.csv", parse_dates=["datetime"])
    ref = cv2.imread(str(U.path(c["ref_image"])))
    H, W = ref.shape[:2]
    union = roi_union(U.path(c["roi_json"]), W, H)

    rows = []
    imgs = align.usable_images(U.path(c["image_dir"]), U.CAM["phenocam_lat"], U.CAM["phenocam_lon"])
    imgs = [(t, p) for t, p in imgs if t.year in U.YEARS]
    print(f"{name}: {len(imgs)} images in the time / sun window")
    masks = {}   # the soft mask only depends on which transform row is used, so compute each once
    for t, p in imgs:
        img = cv2.imread(p)
        if img is None or U.edge_density(img, clahe=False) <= c["edge_density_min"]:
            continue
        # nearest transform, stepping back in time until one is good enough
        i = int(np.abs(t_df["datetime"] - t).idxmin())
        good = [j for j in range(i, -1, -1) if t_df.loc[j, "inlier_ratio"] > QC["inlier_ratio_min"]]
        if not good:
            continue
        r = t_df.loc[good[0]]
        # NOTE: uses the top 2 rows of the homography as an affine warp
        A = np.array([[r["A00"], r["A01"], r["A02"]], [r["A10"], r["A11"], r["A12"]]])
        warped = cv2.warpAffine(img, A, (W, H))
        if good[0] not in masks:
            masks[good[0]] = soft_interior_mask(union, r["e_mean"], r["e_std"])
        mean, std, vmin, vmax = weighted_gcc(warped, masks[good[0]])
        rows.append([t, mean, std, vmin, vmax])

    pd.DataFrame(rows, columns=["datetime", "mean", "std", "min", "max"]).to_csv(cam_dir / "gcc.csv", index=False)
    print(f"  {len(rows)} images -> {cam_dir / 'gcc.csv'}")


if __name__ == "__main__":
    for name, c in U.CAM["phenocams"].items():
        gcc_camera(name, c)
