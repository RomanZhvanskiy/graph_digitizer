#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Digitize a purple curve from a PNG with three red axis markers.

Features
--------
1) Color ID via a small k-NN classifier in HSV space using sample swatches
   for purple and red.
2) Red pixels define the three axis reference points:
   - if exactly 3 red pixels are found, use them directly;
   - if more than 3 are found, group them into 3 groups and use the mean
     of each group.
3) X and Y axes are linear by default, with optional logarithmic mapping.
4) Purple curve is extracted and converted into data coordinates.
5) Curve is simplified to K representative points using farthest-point sampling.
6) Optional monotone non-increasing Y enforcement.
7) Can run either:
   - from command line with arguments, or
   - from a hard-coded batch routine (main_caller).

Usage (CLI)
-----------
#Auto-select number of colors, reject grayscale, reject pure-red curve groups:
python Digitize1.py input.png --xmin 0 --xmax 1 --ymin 1e-7 --ymax 1 --auto-colors --exclude-grayscale --reject-marker-red --monotone-y --out out.csv --debug
#Force exactly 6 color groups:
python Digitize1.py input.png --xmin 0 --xmax 1 --ymin 1e-7 --ymax 1 --auto-colors --num-colors 6 --exclude-grayscale --reject-marker-red --out out.csv
Batch usage
-----------
Edit main_caller(), then call:
    from Digitize1 import main_caller
    main_caller()
"""

import sys
import math
import argparse
import os
import re
import unicodedata
import difflib
from pathlib import Path

import numpy as np
import cv2


# --------------------------- Tunable color-filter settings ---------------------------

COLOR_FILTER = {
    # strict red axis-marker detection
    "red_r_min": 240,
    "red_g_max": 10,
    "red_b_max": 10,

    # grayscale rejection for palette discovery
    "gray_max_channel_spread": 15,
    "gray_max_sat": 25,
    "gray_min_value_for_sat_test": 25,

    # luminance gate for palette discovery
    "lum_min": 45,
    "lum_max": 150,

    # near-pure-red rejection for palette discovery
    "marker_red_tol": 4.0,

    # k-means sampling
    "sample_limit": 50000,
    "kmin": 2,
    "kmax": 20,
    
    #iterative spatial relabelling settings
    "spatial_refine_iters": 620,
    "spatial_color_weight": 1.0,
    "spatial_neighbor_weight": 12000.0,
    "spatial_use_lab": True,
    
    # post-clustering geometric cleanup
    "curve_snap_enable": True,
    "curve_snap_dilate_radius": 2,
    "curve_snap_search_radius": 3,
    
    # exclude near axes distance
    "axes_exclude_dist_for_clustering": 10.0,
    "axes_exclude_dist_for_curve_masks": 3.0,

    "prepend_origin_point": True,
    "append_xmax_zero_point": True,    
    "extend_tail_from_overlapping_curve": True,
    "tail_overlap_lookback_points": 10,
    "tail_overlap_distance_tol": 2.5, 
}

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------- Args ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Graph digitizer with k-NN color ID (x linear, y log).")

    p.add_argument("--batch", action="store_true",
                   help="Run main_caller() and ignore all other command-line arguments.")

    p.add_argument("image", nargs="?", default="",
                   help="Path to PNG image.")
    p.add_argument("--xmin", type=float, default=None)
    p.add_argument("--xmax", type=float, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)

    p.add_argument("--k", type=int, default=100, help="Target number of representative points (default: 100).")
    p.add_argument("--min-sat", type=int, default=30, help="Rough prefilter: min saturation (HSV) for candidate pixels.")
    p.add_argument("--min-val", type=int, default=30, help="Rough prefilter: min value/brightness (HSV) for candidate pixels.")
    p.add_argument("--knn-k", type=int, default=3, help="k for k-NN (default: 3).")
    p.add_argument("--knn-thresh", type=float, default=35.0,
                   help="Reject color if distance to nearest class prototype > threshold (HSV metric).")
    p.add_argument("--out", type=str, default="", help="Write CSV to this path (if omitted, prints to stdout).")
    p.add_argument("--debug", action="store_true", help="Save debug overlays next to input image.")
    p.add_argument("--monotone-y", action="store_true",
                   help="After simplification, enforce y[i+1] <= y[i] (non-increasing).")
    p.add_argument("--monotone-method", choices=["pava", "clip"], default="pava",
                   help="Method for monotonic y enforcement: 'pava' or 'clip'.")
    p.add_argument("--auto-colors", action="store_true",
                   help="Automatically discover distinct non-grayscale colors and digitize the main curve.")
    p.add_argument("--exclude-grayscale", action="store_true",
                   help="Exclude grayscale-ish pixels before color grouping.")
    p.add_argument("--gray-abs", type=float, default=2.0,
                   help="Absolute grayscale tolerance in RGB units (default: 2.0).")
    p.add_argument("--gray-rel", type=float, default=0.05,
                   help="Relative grayscale tolerance (default: 0.05 = 5%%).")

    p.add_argument("--num-colors", type=int, default=0,
                   help="If > 0, use exactly this many discovered color groups instead of auto-selecting 2..20.")
    p.add_argument("--reject-marker-red", action="store_true",
                   help="Reject near-pure-red clusters as curve candidates.")
    p.add_argument("--marker-red-tol", type=float, default=4.0,
                   help="Tolerance around pure red RGB=(255,0,0) for rejecting marker-red clusters.")
    p.add_argument("--xscale_log", action="store_true",
                   help="Use logarithmic x-axis mapping. Default is linear.")
    p.add_argument("--yscale_log", action="store_true",
                   help="Use logarithmic y-axis mapping. Default is linear.")
                   
    p.add_argument("--keep-outside-axes", action="store_true",
                   help="Do not reject points outside the axes box after clustering.")
    return p.parse_args()

# --------------------------- Robust path handling ---------------------------

_HYPHENS = "-\u2010\u2011\u2012\u2013\u2014\u2212\u2043\uFE63\uFF0D"
_SPACES = " \u00A0\u2007\u202F"

def _norm_name(s: str) -> str:
    """Normalize filename for fuzzy matching."""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate({ord(h): '-' for h in _HYPHENS})
    s = s.translate({ord(sp): ' ' for sp in _SPACES})
    s = re.sub(r"[ \-]+", "-", s.strip().lower())
    return s

def resolve_image_path(path_str: str) -> str:
    """Resolve likely path mismatches due to Unicode hyphens/spaces."""
    p = Path(path_str)
    if p.exists():
        return str(p)

    parent = p.parent if str(p.parent) not in ("", ".") else Path.cwd()
    if not parent.exists():
        return str(p)

    needle = _norm_name(p.name)
    best = None
    best_score = 0.0

    for c in parent.iterdir():
        score = difflib.SequenceMatcher(None, _norm_name(c.name), needle).ratio()
        if score > best_score:
            best = c
            best_score = score

    if best is not None and best_score > 0.72:
        return str(best)

    return str(p)


# --------------------------- Color swatches → HSV prototypes ---------------------------

def _rgb_list_to_hsv(rgb_list):
    """Convert list of (R,G,B) into OpenCV HSV."""
    arr = np.array(rgb_list, dtype=np.uint8).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    return hsv

def get_color_prototypes_hsv():
    purple_rgb = [
        (95, 58, 153), (116, 58, 160), (42, 2, 75),
        (113, 46, 152), (103, 58, 139), (88, 44, 139),
        (103, 53, 140), (53, 0, 96), (109, 54, 156),
    ]
    red_rgb = [
        (237, 28, 36), (216, 116, 120), (237, 28, 36),
        (199, 99, 103), (207, 135, 138), (246, 146, 150), (237, 28, 36),
    ]
    purple_hsv = _rgb_list_to_hsv(purple_rgb)
    red_hsv = _rgb_list_to_hsv(red_rgb)
    return purple_hsv, red_hsv


# --------------------------- k-NN color classifier ---------------------------

def hsv_dist2(a, b, wH=2.0, wS=1.0, wV=1.0):
    """
    Squared distance between HSV vectors with circular hue.
    a: [N,3], b: [M,3]
    returns: [N,M]
    """
    Ha, Sa, Va = a[:, 0:1], a[:, 1:2], a[:, 2:3]
    Hb, Sb, Vb = b[None, :, 0], b[None, :, 1], b[None, :, 2]
    dH = np.abs(Ha - Hb)
    dH = np.minimum(dH, 180.0 - dH)
    dS = Sa - Sb
    dV = Va - Vb
    return (wH * dH) ** 2 + (wS * dS) ** 2 + (wV * dV) ** 2

def knn_classify_hsv(hsv_img, knn_k=3, min_sat=30, min_val=30, dist_thresh=35.0):
    """
    Classify pixels as purple, red, or other using k-NN in HSV space.
    """
    H, W, _ = hsv_img.shape
    purple_proto, red_proto = get_color_prototypes_hsv()

    S = hsv_img[:, :, 1]
    V = hsv_img[:, :, 2]
    candidate = (S >= min_sat) & (V >= min_val)

    idxs = np.flatnonzero(candidate)
    if idxs.size == 0:
        return np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8), np.ones((H, W), np.uint8)

    hsv_flat = hsv_img.reshape(-1, 3).astype(np.float32)
    hsv_sel = hsv_flat[idxs]

    d2_p = hsv_dist2(hsv_sel, purple_proto)
    d2_r = hsv_dist2(hsv_sel, red_proto)

    kP = min(knn_k, d2_p.shape[1])
    kR = min(knn_k, d2_r.shape[1])

    d_p = np.sqrt(np.partition(d2_p, kth=kP - 1, axis=1)[:, :kP].mean(axis=1))
    d_r = np.sqrt(np.partition(d2_r, kth=kR - 1, axis=1)[:, :kR].mean(axis=1))

    closer_is_purple = d_p < d_r
    min_d = np.where(closer_is_purple, d_p, d_r)
    accept = min_d <= dist_thresh

    purple_sel = accept & closer_is_purple
    red_sel = accept & (~closer_is_purple)

    purple_mask = np.zeros(H * W, np.uint8)
    red_mask = np.zeros(H * W, np.uint8)
    purple_mask[idxs[purple_sel]] = 255
    red_mask[idxs[red_sel]] = 255

    purple_mask = purple_mask.reshape(H, W)
    red_mask = red_mask.reshape(H, W)

    other_mask = np.ones((H, W), np.uint8)
    other_mask[purple_mask > 0] = 0
    other_mask[red_mask > 0] = 0

    return purple_mask, red_mask, other_mask

def strict_red_mask_bgr(
    img_bgr,
    r_min=COLOR_FILTER["red_r_min"],
    g_max=COLOR_FILTER["red_g_max"],
    b_max=COLOR_FILTER["red_b_max"],
    ):
    
    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]
    return (r >= r_min) & (g <= g_max) & (b <= b_max)
    
    
def find_red_centroids_strict(img_bgr):
    red_bool = strict_red_mask_bgr(img_bgr)
    ys, xs = np.where(red_bool)
    n = xs.size

    if n < 3:
        raise ValueError(
            f"Only {n} strict-red pixel(s) detected; need at least 3."
        )

    X = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    if n == 3:
        centers = X.copy()
    else:
        centers, _ = _cluster_red_into_three(X)

    idx_sorted_x = np.argsort(centers[:, 0])
    left_two = centers[idx_sorted_x[:2]]
    right_one = centers[idx_sorted_x[2]]

    tl = left_two[np.argmin(left_two[:, 1])]
    bl = left_two[np.argmax(left_two[:, 1])]
    br = right_one

    red_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    red_mask[red_bool] = 255

    return bl, br, tl, red_mask
    
    
    
    
    
    
    
    
def detect_axes_red_points(img_bgr, hsv=None, min_sat=30, min_val=30, knn_k=3, knn_thresh=35.0):
    """
    Detect the three red axis reference points using strict RGB thresholds.
    """
    bl_px, br_px, tl_px, red_mask = find_red_centroids_strict(img_bgr)
    return bl_px, br_px, tl_px, red_mask
    
def mask_near_axes_region(img_shape, origin, ax_vec, ay_vec, axis_dist_px=10.0):
    """
    Return boolean mask of pixels lying within axis_dist_px of either:
      - the x-axis segment from origin to origin + ax_vec
      - the y-axis segment from origin to origin + ay_vec
    """
    H, W = img_shape[:2]
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 2)

    origin = np.asarray(origin, dtype=np.float64)
    x_end = origin + np.asarray(ax_vec, dtype=np.float64)
    y_end = origin + np.asarray(ay_vec, dtype=np.float64)

    def point_to_segment_distance(p, a, b):
        ab = b - a
        ab_n2 = np.dot(ab, ab)
        if ab_n2 < 1e-12:
            return np.sqrt(np.sum((p - a) ** 2, axis=1))
        t = np.einsum('ij,j->i', p - a, ab) / ab_n2
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        return np.sqrt(np.sum((p - proj) ** 2, axis=1))

    dx = point_to_segment_distance(pts, origin, x_end)
    dy = point_to_segment_distance(pts, origin, y_end)

    near = (dx <= axis_dist_px) | (dy <= axis_dist_px)
    return near.reshape(H, W)
    
# --------------------------- Red grouping ---------------------------

def _fps3_indices(X):
    """Pick 3 well-separated points by farthest-point sampling."""
    i1 = int(np.argmin(X[:, 0]))
    d2_1 = np.sum((X - X[i1]) ** 2, axis=1)
    i2 = int(np.argmax(d2_1))
    d2_2 = np.sum((X - X[i2]) ** 2, axis=1)
    d2_min = np.minimum(d2_1, d2_2)
    i3 = int(np.argmax(d2_min))
    return np.array([i1, i2, i3], dtype=np.int64)

def _cluster_red_into_three(X):
    """
    Group all red pixels into 3 clusters using FPS seeds and one assignment step.
    """
    seed_idx = _fps3_indices(X)
    seeds = X[seed_idx]
    d2 = np.sum((X[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1)

    centers = np.zeros((3, 2), dtype=np.float64)
    for i in range(3):
        centers[i] = X[labels == i].mean(axis=0)
    return centers, labels



# --------------------------- Coordinate mapping ---------------------------

def build_affine_axes(bl_px, br_px, tl_px):
    origin = np.array(bl_px, dtype=np.float64)
    ax_vec = np.array(br_px, dtype=np.float64) - origin
    ay_vec = np.array(tl_px, dtype=np.float64) - origin
    ax_n2 = float(np.dot(ax_vec, ax_vec))
    ay_n2 = float(np.dot(ay_vec, ay_vec))
    if ax_n2 < 1e-6 or ay_n2 < 1e-6:
        raise ValueError("Axis vectors too small; check red dot clusters.")
    return origin, ax_vec, ay_vec, ax_n2, ay_n2

def pixels_to_params(points_px, origin, ax_vec, ay_vec, ax_n2, ay_n2):
    rel = points_px - origin[None, :]
    tx = np.clip(np.einsum('ij,j->i', rel, ax_vec) / ax_n2, 0.0, 1.0)
    ty = np.clip(np.einsum('ij,j->i', rel, ay_vec) / ay_n2, 0.0, 1.0)
    return tx, ty
    
def mask_inside_axes_region(img_shape, origin, ax_vec, ay_vec, ax_n2, ay_n2):
    """
    Return boolean mask of pixels whose normalized axis coordinates lie in [0,1]x[0,1].
    """
    H, W = img_shape[:2]
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 2)

    rel = pts - origin[None, :]
    tx = np.einsum('ij,j->i', rel, ax_vec) / ax_n2
    ty = np.einsum('ij,j->i', rel, ay_vec) / ay_n2

    inside = (tx >= 0.0) & (tx <= 1.0) & (ty >= 0.0) & (ty <= 1.0)
    return inside.reshape(H, W)
    
def filter_points_inside_axes(points_px, origin, ax_vec, ay_vec, ax_n2, ay_n2):
    """
    Project points to normalized axis coordinates and keep only those inside [0,1]x[0,1].
    Returns:
      pts_in, tx_in, ty_in
    """
    rel = points_px - origin[None, :]
    tx = np.einsum('ij,j->i', rel, ax_vec) / ax_n2
    ty = np.einsum('ij,j->i', rel, ay_vec) / ay_n2
    inside = (tx >= 0.0) & (tx <= 1.0) & (ty >= 0.0) & (ty <= 1.0)
    return points_px[inside], tx[inside], ty[inside]
    
def params_to_data(tx, ty, xmin, xmax, ymin, ymax, xscale_log=False, yscale_log=False):
    if xscale_log:
        if xmin <= 0 or xmax <= 0:
            raise ValueError("Log-scale x requires xmin > 0 and xmax > 0.")
        logx = math.log(xmin) + tx * (math.log(xmax) - math.log(xmin))
        x = np.exp(logx)
    else:
        x = xmin + tx * (xmax - xmin)

    if yscale_log:
        if ymin <= 0 or ymax <= 0:
            raise ValueError("Log-scale y requires ymin > 0 and ymax > 0.")
        logy = math.log(ymin) + ty * (math.log(ymax) - math.log(ymin))
        y = np.exp(logy)
    else:
        y = ymin + ty * (ymax - ymin)

    return x, y


# --------------------------- Purple curve extraction ---------------------------

def extract_purple_points_knn(img_bgr, hsv, purple_mask, red_mask):
    pm = cv2.bitwise_and(purple_mask, cv2.bitwise_not(red_mask))
    ys, xs = np.where(pm > 0)
    if xs.size == 0:
        raise ValueError("No purple pixels detected. Adjust k-NN thresholds or check the image.")

    N = xs.size
    if N > 500_000:
        step = int(np.ceil(N / 500_000))
        xs = xs[::step]
        ys = ys[::step]

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    return pts, pm


# --------------------------- Simplification ---------------------------

def farthest_point_sampling_xy(xy, k):
    n = xy.shape[0]
    if n <= k:
        return np.arange(n, dtype=np.int64)

    start_idx = int(np.argmin(xy[:, 0]))
    selected = [start_idx]
    d2 = np.sum((xy - xy[start_idx]) ** 2, axis=1)

    for _ in range(1, k):
        far_idx = int(np.argmax(d2))
        selected.append(far_idx)
        new_d2 = np.sum((xy - xy[far_idx]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return np.array(selected, dtype=np.int64)


# --------------------------- Monotone Y enforcement ---------------------------

def _pava_increasing(y):
    """
    Pool-Adjacent-Violators Algorithm for non-decreasing fit.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size

    vals = []
    lens = []

    for i in range(n):
        vals.append(y[i])
        lens.append(1)
        while len(vals) >= 2 and vals[-2] > vals[-1]:
            v = (vals[-2] * lens[-2] + vals[-1] * lens[-1]) / (lens[-2] + lens[-1])
            l = lens[-2] + lens[-1]
            vals[-2] = v
            lens[-2] = l
            vals.pop()
            lens.pop()

    y_fit = np.empty(n, dtype=np.float64)
    pos = 0
    for v, l in zip(vals, lens):
        y_fit[pos:pos + l] = v
        pos += l

    return y_fit

def enforce_monotone_nonincreasing(y, method="pava"):
    """
    Ensure y[i+1] <= y[i].
    """
    y = np.asarray(y, dtype=np.float64)

    if method == "clip":
        out = y.copy()
        for i in range(1, out.size):
            if out[i] > out[i - 1]:
                out[i] = out[i - 1]
        return out

    return -_pava_increasing(-y)

# --------------------------- Grayscale filtering ---------------------------

def grayscale_mask_bgr(
    img_bgr,
    max_channel_spread=COLOR_FILTER["gray_max_channel_spread"],
    max_sat=COLOR_FILTER["gray_max_sat"],
    min_value_for_sat_test=COLOR_FILTER["gray_min_value_for_sat_test"],
    ):
    """
    True where pixel is grayscale-ish.

    A pixel is grayscale if EITHER:
      1) its RGB channels are close together, OR
      2) its HSV saturation is low enough (for non-dark pixels)
    """
    b = img_bgr[:, :, 0].astype(np.int16)
    g = img_bgr[:, :, 1].astype(np.int16)
    r = img_bgr[:, :, 2].astype(np.int16)

    rgb_spread = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    rgb_gray = rgb_spread <= max_channel_spread

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)

    hsv_gray = (sat <= max_sat) & (val >= min_value_for_sat_test)

    gray = rgb_gray | hsv_gray
    return gray

def luminance_mask_bgr(
    img_bgr,
    lum_min=COLOR_FILTER["lum_min"],
    lum_max=COLOR_FILTER["lum_max"],
    ):

    """
    True where pixel luminance is within [lum_min, lum_max].

    Uses standard approximate perceptual luminance on RGB:
        Y = 0.299 R + 0.587 G + 0.114 B
    """
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)

    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return (lum >= lum_min) & (lum <= lum_max)
    
# --------------------------- K-means in color space ---------------------------

def kmeans_pp_init_general(X, k, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)

    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]

    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        s = d2.sum()
        if not np.isfinite(s) or s <= 0:
            centers[i] = X[rng.integers(0, n)]
        else:
            probs = d2 / s
            centers[i] = X[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, np.sum((X - centers[i]) ** 2, axis=1))
    return centers

def kmeans_nd(X, k, max_iter=50, tol=1e-4, rng=None):
    X = np.asarray(X, dtype=np.float64)
    centers = kmeans_pp_init_general(X, k, rng=rng)
    labels = np.zeros(X.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.copy()
        for i in range(k):
            m = labels == i
            if np.any(m):
                new_centers[i] = X[m].mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break

    d2 = np.sum((X - centers[labels]) ** 2, axis=1)
    inertia = float(d2.sum())
    return centers, labels, inertia

def silhouette_sample(X, labels, max_points=4000, rng=None):
    """
    Approximate silhouette score on a sample.
    Returns a score in [-1, 1]; higher is better.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)

    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        X = X[idx]
        labels = labels[idx]

    uniq = np.unique(labels)
    if uniq.size < 2:
        return -1.0

    # pairwise squared distances
    d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    s_vals = []

    for i in range(X.shape[0]):
        own = labels == labels[i]
        own[i] = False

        if np.any(own):
            a = d[i, own].mean()
        else:
            a = 0.0

        b = np.inf
        for lab in uniq:
            if lab == labels[i]:
                continue
            m = labels == lab
            if np.any(m):
                b = min(b, d[i, m].mean())

        denom = max(a, b)
        s = 0.0 if denom <= 0 else (b - a) / denom
        s_vals.append(s)

    return float(np.mean(s_vals))

def choose_k_by_separation(X, kmin=2, kmax=20, rng=None):
    """
    Try k = 2..20 and choose the one with best approximate silhouette.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    best = None
    results = []

    for k in range(kmin, min(kmax, X.shape[0]) + 1):
        centers, labels, inertia = kmeans_nd(X, k, rng=rng)
        sil = silhouette_sample(X, labels, rng=rng)
        results.append((k, centers, labels, inertia, sil))
        if best is None or sil > best[-1]:
            best = (k, centers, labels, inertia, sil)

    return best, results

def pixels_to_params_unclipped(points_px, origin, ax_vec, ay_vec, ax_n2, ay_n2):
    rel = points_px - origin[None, :]
    tx = np.einsum('ij,j->i', rel, ax_vec) / ax_n2
    ty = np.einsum('ij,j->i', rel, ay_vec) / ay_n2
    return tx, ty
    
# --------------------------- Automatic color discovery ---------------------------

def discover_color_clusters(
    img_bgr,
    exclude_grayscale=False,
    gray_abs=2.0,
    gray_rel=0.05,
    sample_limit=COLOR_FILTER["sample_limit"],
    kmin=COLOR_FILTER["kmin"],
    kmax=COLOR_FILTER["kmax"],
    num_colors=0,
    reject_marker_red=False,
    marker_red_tol=COLOR_FILTER["marker_red_tol"],
    inside_mask=None,
    axes_exclude_mask=None,
    lum_min=COLOR_FILTER["lum_min"],
    lum_max=COLOR_FILTER["lum_max"],
    ):
    """
    Cluster image colors after filtering, then refine labels spatially so nearby
    eligible pixels prefer to share the same group.

    Returns:
      cluster_masks: list of HxW uint8 masks
      cluster_centers_rgb: [K,3]
      chosen_k: int
      model_scores: list of tuples (k, inertia, silhouette)
    """
    H, W, _ = img_bgr.shape

    valid = np.ones((H, W), dtype=bool)

    if inside_mask is not None:
        valid &= inside_mask

    if exclude_grayscale:
        valid &= ~grayscale_mask_bgr(img_bgr)

    if axes_exclude_mask is not None:
        valid &= ~axes_exclude_mask

    if reject_marker_red:
        valid &= ~marker_red_mask_bgr(img_bgr, tol=marker_red_tol)

    valid &= luminance_mask_bgr(img_bgr, lum_min=lum_min, lum_max=lum_max)

    save_candidate_pixels_debug_png(
        img_bgr,
        valid,
        str(SCRIPT_DIR / "debug_candidate_pixels.png")
    )

    ys, xs = np.where(valid)
    if xs.size == 0:
        raise ValueError("No eligible non-grayscale/non-marker-red pixels available for color discovery.")

    # features for clustering/refinement
    use_lab = COLOR_FILTER["spatial_use_lab"]
    feat_img = get_feature_image_for_clustering(img_bgr, use_lab=use_lab)
    X = feat_img[ys, xs].astype(np.float64)

    rng = np.random.default_rng(0)
    if X.shape[0] > sample_limit:
        idx = rng.choice(X.shape[0], size=sample_limit, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    if num_colors and num_colors > 0:
        chosen_k = min(int(num_colors), X.shape[0])
        centers, labels, inertia = kmeans_nd(X, chosen_k, rng=rng)
        model_scores = [(chosen_k, inertia, float("nan"))]
    else:
        best, results = choose_k_by_separation(X_sample, kmin=kmin, kmax=kmax, rng=rng)
        chosen_k, _, _, _, _ = best
        centers, labels, inertia = kmeans_nd(X, chosen_k, rng=rng)
        model_scores = [(k, inertia_, sil_) for (k, _, _, inertia_, sil_) in results]

    # spatial refinement
    labels, centers = refine_labels_spatial_icm(
        feat_img=feat_img,
        ys=ys,
        xs=xs,
        labels=labels,
        centers=centers,
        n_iter=COLOR_FILTER["spatial_refine_iters"],
        color_weight=COLOR_FILTER["spatial_color_weight"],
        neighbor_weight=COLOR_FILTER["spatial_neighbor_weight"],
    )

    # build masks from refined labels
    cluster_masks = []
    for i in range(chosen_k):
        mask = np.zeros((H, W), dtype=np.uint8)
        m = labels == i
        mask[ys[m], xs[m]] = 255
        cluster_masks.append(mask)

    # convert final centers to RGB for palette/debug display
    if use_lab:
        centers_u8 = np.clip(np.round(centers), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        centers_rgb = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float64)
    else:
        centers_rgb = centers.copy()


    return cluster_masks, centers_rgb, centers, chosen_k, model_scores

# --------------------------- Cluster interpretation ---------------------------

def _connected_components_summary(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]
        comps.append(dict(label=i, area=area, x=x, y=y, w=w, h=h, cx=float(cx), cy=float(cy)))
    return comps, labels

def pick_red_cluster_from_discovered(centers_rgb, cluster_masks):
    """
    Choose the most red-like discovered cluster:
    large R, and R clearly exceeds G/B.
    """
    best_idx = None
    best_score = -np.inf
    for i, c in enumerate(centers_rgb):
        r, g, b = c.astype(float)
        score = (r - 0.6 * g - 0.6 * b) + 0.2 * r
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx

def red_centroids_from_mask(red_mask):
    ys, xs = np.where(red_mask > 0)
    n = xs.size
    if n < 3:
        raise ValueError(f"Only {n} red pixel(s) detected; need at least 3.")
    X = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    if n == 3:
        centers = X.copy()
    else:
        centers, _ = _cluster_red_into_three(X)

    idx_sorted_x = np.argsort(centers[:, 0])
    left_two = centers[idx_sorted_x[:2]]
    right_one = centers[idx_sorted_x[2]]
    tl = left_two[np.argmin(left_two[:, 1])]
    bl = left_two[np.argmax(left_two[:, 1])]
    br = right_one
    return bl, br, tl



# --------------------------- post-clustering curve de-jagging ---------------------------

def dilate_mask(mask, radius=2):
    """
    Binary dilation with an elliptical kernel.
    """
    if radius <= 0:
        return ((mask > 0).astype(np.uint8) * 255)

    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel, iterations=1)


def build_reference_curve_set(cluster_masks, dilate_radius=2):
    """
    Build common thickened reference set from all curve masks together.

    Returns:
      ref_mask_u8 : uint8 mask, 255 where reference curve set exists
      ref_dt      : float32 distance transform inside ref_mask
    """
    union_mask = make_union_mask(cluster_masks)
    if union_mask is None:
        return None, None

    ref_mask = dilate_mask(union_mask, radius=dilate_radius)
    ref_bin = (ref_mask > 0).astype(np.uint8)

    # distance to nearest non-line pixel
    ref_dt = cv2.distanceTransform(ref_bin, cv2.DIST_L2, 3)

    return ref_mask, ref_dt



def _local_samecurve_support(mask01, x, y):
    """
    Count same-curve support in 8-neighbourhood.
    mask01 is 0/1 uint8 or bool.
    """
    H, W = mask01.shape
    x0 = max(0, x - 1)
    x1 = min(W, x + 2)
    y0 = max(0, y - 1)
    y1 = min(H, y + 2)

    patch = mask01[y0:y1, x0:x1]
    s = int(np.count_nonzero(patch))
    if 0 <= y < H and 0 <= x < W and mask01[y, x]:
        s -= 1
    return s

def snap_curve_mask_to_reference(curve_mask, ref_mask, ref_dt, search_radius=3):
    """
    Move each foreground point by at most search_radius pixels if doing so
    increases its distance from the nearest non-line point.

    Criterion:
      maximize ref_dt[new_y, new_x]

    Only candidate positions inside ref_mask are allowed.
    Each source pixel is moved at most once.
    """
    if curve_mask is None or ref_mask is None or ref_dt is None:
        return curve_mask

    H, W = curve_mask.shape[:2]
    curve01 = (curve_mask > 0).astype(np.uint8)
    ref01 = (ref_mask > 0).astype(np.uint8)

    ys, xs = np.where(curve01 > 0)
    if xs.size == 0:
        return curve_mask.copy()

    out = np.zeros_like(curve01, dtype=np.uint8)

    # candidate offsets inside a disk of radius search_radius
    offsets = [(0, 0)]
    r2 = search_radius * search_radius
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= r2:
                offsets.append((dy, dx))

    # Prefer bigger dt first, but for equal dt prefer shorter move.
    # We implement that by scanning all candidates and tie-breaking by move length.
    for y, x in zip(ys, xs):
        best_x = x
        best_y = y
        best_dt = -1.0
        best_move2 = 10**18

        current_dt = float(ref_dt[y, x]) if ref01[y, x] else -1.0

        for dy, dx in offsets:
            yy = y + dy
            xx = x + dx

            if xx < 0 or xx >= W or yy < 0 or yy >= H:
                continue
            if not ref01[yy, xx]:
                continue

            cand_dt = float(ref_dt[yy, xx])
            move2 = dx * dx + dy * dy

            if (cand_dt > best_dt) or (cand_dt == best_dt and move2 < best_move2):
                best_dt = cand_dt
                best_move2 = move2
                best_x = xx
                best_y = yy

        # only move if strictly better than current location
        if best_dt > current_dt:
            out[best_y, best_x] = 1
        else:
            out[y, x] = 1

    return (out * 255).astype(np.uint8)



#def snap_all_curve_masks_to_reference(cluster_masks,
#                                      dilate_radius=2,
#                                      search_radius=3,
#                                      samecurve_weight=0.35):
#    if not cluster_masks:
#        return cluster_masks, None
#
#    ref_mask, ref_dt = build_reference_curve_set(
#        cluster_masks,
#        dilate_radius=dilate_radius,
#    )
#
#    if ref_mask is None:
#        return cluster_masks, None
#
#    cleaned = []
#    for mask in cluster_masks:
#        cleaned_mask = snap_curve_mask_to_reference(
#            mask,
#            ref_mask=ref_mask,
#            ref_dt=ref_dt,
#            search_radius=search_radius,
#            samecurve_weight=samecurve_weight,
#        )
#        cleaned_mask = dilate_mask(cleaned_mask, radius=1)
#        cleaned.append(cleaned_mask)
#
#    return cleaned, ref_mask
#    
def snap_all_curve_masks_to_reference(cluster_masks,
                                      dilate_radius=2,
                                      search_radius=3):
    """
    Snap every curve mask toward the middle of the shared thickened reference set.
    """
    if not cluster_masks:
        return cluster_masks, None

    ref_mask, ref_dt = build_reference_curve_set(
        cluster_masks,
        dilate_radius=dilate_radius,
    )
    if ref_mask is None:
        return cluster_masks, None

    cleaned = []
    for mask in cluster_masks:
        cleaned_mask = snap_curve_mask_to_reference(
            mask,
            ref_mask=ref_mask,
            ref_dt=ref_dt,
            search_radius=search_radius,
        )
        cleaned.append(cleaned_mask)

    return cleaned, ref_mask
    
    
def save_reference_curve_set_debug_png(img_bgr, ref_mask, out_path):
    """
    Save overlay of the thickened reference curve set.
    """
    if ref_mask is None:
        return

    dbg = img_bgr.copy()
    overlay = dbg.copy()
    overlay[ref_mask > 0] = (0, 255, 255)
    dbg = cv2.addWeighted(overlay, 0.25, dbg, 0.75, 0.0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, dbg)



# --------------------------- Combined legend detection ---------------------------

def save_legend_box_debug_png(img_bgr, legend_box, out_path):
    dbg = img_bgr.copy()
    if legend_box is not None:
        x0, y0, x1, y1 = legend_box
        cv2.rectangle(
            dbg,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (0, 255, 255),
            2
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, dbg)
    
def make_union_mask(cluster_masks):
    """
    Union of all cluster masks into one binary mask.
    """
    if not cluster_masks:
        return None
    union = np.zeros_like(cluster_masks[0], dtype=np.uint8)
    for m in cluster_masks:
        union = cv2.bitwise_or(union, m)
    return union


def rect_intersects_canvas(x0, y0, w, h, W, H):
    """
    True if rectangle intersects the image canvas at all.
    """
    return not (x0 + w <= 0 or y0 + h <= 0 or x0 >= W or y0 >= H)


def clip_rect_to_canvas(x0, y0, w, h, W, H):
    """
    Clip rectangle to canvas. Returns (cx0, cy0, cx1, cy1) or None if no overlap.
    """
    cx0 = max(0, x0)
    cy0 = max(0, y0)
    cx1 = min(W, x0 + w)
    cy1 = min(H, y0 + h)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    return cx0, cy0, cx1, cy1


def point_to_rect_distance(points_xy, x0, y0, x1, y1):
    """
    Distance from points to axis-aligned rectangle.
    Rectangle boundaries are inclusive in the geometric sense.
    points_xy: [N,2] with columns x,y
    returns: [N]
    """
    px = points_xy[:, 0]
    py = points_xy[:, 1]

    dx = np.maximum.reduce([x0 - px, np.zeros_like(px), px - x1])
    dy = np.maximum.reduce([y0 - py, np.zeros_like(py), py - y1])
    return np.sqrt(dx * dx + dy * dy)

def detect_legend_box_sliding(
    union_mask,
    plot_w_px,
    plot_h_px,
    box_frac_x_values=(0.15, 0.20, 0.25, 0.30, 0.35),
    box_frac_y_values=(0.15, 0.20, 0.25, 0.30, 0.35),
    isolation_frac=0.07,
    step_frac=0.01,
    min_points_in_box=20,
    min_fill_ratio=0.002,
):
    """
    Detect an isolated legend-like region by sliding a box over the canvas
    across multiple box sizes.

    Early exit: return the first box that satisfies the legend criteria.
    """

    H, W = union_mask.shape[:2]

    ys, xs = np.where(union_mask > 0)
    if xs.size == 0:
        return None

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)

    iso_thr = isolation_frac * max(plot_w_px, plot_h_px)
    slide_step = max(1, int(round(step_frac * min(plot_w_px, plot_h_px))))

    for frac_x in box_frac_x_values:
        for frac_y in box_frac_y_values:
            box_w = max(1, int(round(frac_x * plot_w_px)))
            box_h = max(1, int(round(frac_y * plot_h_px)))

            x_starts = range(-box_w + 1, W, slide_step)
            y_starts = range(-box_h + 1, H, slide_step)

            for y0 in y_starts:
                for x0 in x_starts:
                    if not rect_intersects_canvas(x0, y0, box_w, box_h, W, H):
                        continue

                    x1 = x0 + box_w
                    y1 = y0 + box_h

                    inside = (
                        (pts[:, 0] >= x0) & (pts[:, 0] < x1) &
                        (pts[:, 1] >= y0) & (pts[:, 1] < y1)
                    )
                    n_in = int(np.count_nonzero(inside))
                    if n_in < min_points_in_box:
                        continue

                    clipped = clip_rect_to_canvas(x0, y0, box_w, box_h, W, H)
                    if clipped is None:
                        continue

                    cx0, cy0, cx1, cy1 = clipped
                    clipped_area = float((cx1 - cx0) * (cy1 - cy0))
                    if clipped_area <= 0:
                        continue

                    fill_ratio = n_in / clipped_area
                    if fill_ratio < min_fill_ratio:
                        continue

                    outside_pts = pts[~inside]
                    if outside_pts.shape[0] == 0:
                        continue

                    d = point_to_rect_distance(outside_pts, x0, y0, x1, y1)
                    min_outside_dist = float(d.min())

                    if min_outside_dist >= iso_thr:
                        return (x0, y0, x1, y1)

    return None
    
def remove_legend_box_from_masks(cluster_masks, legend_box):
    """
    Remove all pixels inside legend_box from every cluster mask.
    legend_box: (x0, y0, x1, y1)
    """
    if legend_box is None:
        return cluster_masks

    x0, y0, x1, y1 = legend_box
    out_masks = []

    for mask in cluster_masks:
        cleaned = mask.copy()

        H, W = cleaned.shape[:2]
        cx0 = max(0, int(np.floor(x0)))
        cy0 = max(0, int(np.floor(y0)))
        cx1 = min(W, int(np.ceil(x1)))
        cy1 = min(H, int(np.ceil(y1)))

        if cx0 < cx1 and cy0 < cy1:
            cleaned[cy0:cy1, cx0:cx1] = 0

        out_masks.append(cleaned)

    return out_masks


def choose_curve_cluster(cluster_masks, centers_rgb, red_cluster_idx,
                         reject_marker_red=False, marker_red_tol=4.0):
    """
    Choose the main curve cluster among non-red discovered colors.
    Heuristic: the curve should have one dominant connected component,
    decent area, and not be too horizontal like a legend.
    """
    best_idx = None
    best_score = -np.inf

    for i, mask in enumerate(cluster_masks):
        if i == red_cluster_idx:
            continue

        if reject_marker_red and is_marker_red_rgb(centers_rgb[i], tol=marker_red_tol):
            continue

        comps, _ = _connected_components_summary(mask)
        if not comps:
            continue

        main = max(comps, key=lambda c: c["area"])
        if main["area"] < 10:
            continue

        extent_score = (
            np.log1p(main["area"])
            + 0.4 * np.log1p(max(main["h"], 1))
            + 0.2 * np.log1p(max(main["w"], 1))
        )
        horiz_penalty = 1.5 if main["w"] >= 4 * max(main["h"], 1) else 0.0

        score = extent_score - horiz_penalty
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        raise ValueError("Could not identify a main non-red curve color cluster.")

    return best_idx


def save_candidate_pixels_debug_png(img_bgr, valid_mask, out_path):
    """
    Save an image showing only the pixels that remain eligible for color clustering.
    Rejected pixels are painted white.
    """
    dbg = np.full_like(img_bgr, 255)
    dbg[valid_mask] = img_bgr[valid_mask]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, dbg)
    
    
def is_marker_red_rgb(rgb, tol=4.0):
    """
    True if rgb is near pure red (255,0,0) within per-channel tolerance.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    target = np.array([255.0, 0.0, 0.0], dtype=np.float64)
    return np.all(np.abs(rgb - target) <= tol)

def marker_red_mask_bgr(img_bgr, tol=COLOR_FILTER["marker_red_tol"]):
    """
    True where pixel is near pure red RGB=(255,0,0), interpreted from BGR image.
    """
    b = img_bgr[:, :, 0].astype(np.float32)
    g = img_bgr[:, :, 1].astype(np.float32)
    r = img_bgr[:, :, 2].astype(np.float32)
    return (np.abs(r - 255.0) <= tol) & (np.abs(g - 0.0) <= tol) & (np.abs(b - 0.0) <= tol)
    
#def auto_detect_red_and_curve(
#    img_bgr,
#    exclude_grayscale=False,
#    gray_abs=2.0,
#    gray_rel=0.05,
#    num_colors=0,
#    reject_marker_red=False,
#    marker_red_tol=4.0,
#    ):
#    bl_px, br_px, tl_px, red_mask = find_red_centroids_strict(img_bgr)
#    origin, ax_vec, ay_vec, ax_n2, ay_n2 = build_affine_axes(bl_px, br_px, tl_px)
#
#    axes_exclude_mask = mask_near_axes_region(
#        img_bgr.shape,
#        origin,
#        ax_vec,
#        ay_vec,
#        axis_dist_px=10.0,
#    )
#
#    inside_axes_mask = mask_inside_axes_region(
#        img_bgr.shape, origin, ax_vec, ay_vec, ax_n2, ay_n2
#    )
#
#    cluster_masks, centers_rgb, centers_feat, chosen_k, model_scores = discover_color_clusters(
#        img_bgr,
#        exclude_grayscale=exclude_grayscale,
#        gray_abs=gray_abs,
#        gray_rel=gray_rel,
#        num_colors=num_colors,
#        reject_marker_red=reject_marker_red,
#        marker_red_tol=marker_red_tol,
#        inside_mask=inside_axes_mask,
#        axes_exclude_mask=axes_exclude_mask,
#    )
#    
#  
#
#    plot_w_px = float(np.sqrt(ax_n2))
#    plot_h_px = float(np.sqrt(ay_n2))
#
#    union_mask = make_union_mask(cluster_masks)
#    legend_box = detect_legend_box_sliding(
#        union_mask,
#        plot_w_px=plot_w_px,
#        plot_h_px=plot_h_px,
#        box_frac_x_values=(0.15, 0.20, 0.25, 0.30, 0.35),
#        box_frac_y_values=(0.15, 0.20, 0.25, 0.30, 0.35),
#        isolation_frac=0.07,
#        step_frac=0.01,
#        min_points_in_box=20,
#        min_fill_ratio=0.002,
#        )
#
#    if legend_box is not None:
#        cluster_masks = remove_legend_box_from_masks(cluster_masks, legend_box)
#
#    red_idx = pick_red_cluster_from_discovered(centers_rgb, cluster_masks)
#
#    curve_idx = choose_curve_cluster(
#        cluster_masks,
#        centers_rgb,
#        red_idx,
#        reject_marker_red=reject_marker_red,
#        marker_red_tol=marker_red_tol,
#    )
#    curve_mask = cluster_masks[curve_idx].copy()
#
#    return bl_px, br_px, tl_px, red_mask, curve_mask, centers_rgb, chosen_k, model_scores
#    
def save_cluster_palette_png(cluster_centers_rgb, out_path, swatch_w=120, swatch_h=80):
    """
    Save a horizontal palette image showing the discovered cluster colors.
    cluster_centers_rgb: array-like of shape [K, 3], values in RGB
    """
    centers = np.asarray(cluster_centers_rgb, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("cluster_centers_rgb must have shape [K,3].")

    K = centers.shape[0]
    if K == 0:
        raise ValueError("No cluster centers to draw.")

    palette = np.zeros((swatch_h, swatch_w * K, 3), dtype=np.uint8)

    for i, c in enumerate(centers):
        rgb = np.clip(np.round(c), 0, 255).astype(np.uint8)
        bgr = rgb[::-1]  # convert RGB -> BGR for OpenCV image writing
        x0 = i * swatch_w
        x1 = (i + 1) * swatch_w
        palette[:, x0:x1, :] = bgr

        # draw border
        cv2.rectangle(palette, (x0, 0), (x1 - 1, swatch_h - 1), (0, 0, 0), 1)

        # label cluster index
        label = f"{i+1}"
        cv2.putText(
            palette, label, (x0 + 10, swatch_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            palette, label, (x0 + 10, swatch_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, palette)    
    
# --------------------------- extractor for an already-known curve mask ---------------------------

def extract_points_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("No curve pixels found in mask.")
    N = xs.size
    if N > 500_000:
        step = int(np.ceil(N / 500_000))
        xs = xs[::step]
        ys = ys[::step]
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    return pts
    



def digitize_curve_mask(mask, origin, ax_vec, ay_vec, ax_n2, ay_n2,
                        xmin, xmax, ymin, ymax, k,
                        xscale_log=False, yscale_log=False,
                        monotone_y=False, monotone_method="pava",
                        keep_outside_axes=False):
    """
    Digitize one curve mask into x_out, y_out.
    Returns (x_out, y_out) or (None, None) if no valid points remain.
    """
    curve_pts_px = extract_points_from_mask(mask)

    if keep_outside_axes:
        tx, ty = pixels_to_params_unclipped(curve_pts_px, origin, ax_vec, ay_vec, ax_n2, ay_n2)
    else:
        _, tx, ty = filter_points_inside_axes(curve_pts_px, origin, ax_vec, ay_vec, ax_n2, ay_n2)
    if tx.size == 0:
        return None, None

    x, y = params_to_data(
        tx, ty, xmin, xmax, ymin, ymax,
        xscale_log=xscale_log,
        yscale_log=yscale_log
    )

    xy = np.stack([x, y], axis=1)
    sel_idx = farthest_point_sampling_xy(xy, k)
    x_sel = x[sel_idx]
    y_sel = y[sel_idx]

    order = np.argsort(x_sel)
    x_out = x_sel[order]
    y_out = y_sel[order]

    if monotone_y:
        y_out = enforce_monotone_nonincreasing(y_out, method=monotone_method)

    return x_out, y_out
    
# --------------------------- spatial refinement after k-means ---------------------------

def get_feature_image_for_clustering(img_bgr, use_lab=True):
    """
    Return per-pixel colour features for clustering/refinement.
    Uses Lab by default because it behaves better than RGB for colour distance.
    Output shape: [H, W, 3], dtype float64
    """
    if use_lab:
        feat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    else:
        feat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    return feat


def refine_labels_spatial_icm(
    feat_img,
    ys,
    xs,
    labels,
    centers,
    n_iter=8,
    color_weight=1.0,
    neighbor_weight=6.0,
):
    """
    Refine cluster labels by minimizing:
        color_weight * distance-to-center
      - neighbor_weight * number-of-neighbors-with-same-label

    Uses 8-connected neighbours on the image grid, but only among eligible pixels.
    This encourages nearby pixels to belong to the same cluster.
    """
    ys = np.asarray(ys, dtype=np.int32)
    xs = np.asarray(xs, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32).copy()
    centers = np.asarray(centers, dtype=np.float64).copy()

    H, W = feat_img.shape[:2]
    K = centers.shape[0]
    N = ys.size

    if N == 0 or K <= 1:
        return labels, centers

    X = feat_img[ys, xs].astype(np.float64)   # [N, 3]

    # map image pixel -> eligible-point index, else -1
    index_img = -np.ones((H, W), dtype=np.int32)
    index_img[ys, xs] = np.arange(N, dtype=np.int32)

    neigh_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    for _ in range(n_iter):
        changed = 0

        # current label image for neighbour lookups
        label_img = -np.ones((H, W), dtype=np.int32)
        label_img[ys, xs] = labels

        for i in range(N):
            y = ys[i]
            x = xs[i]
            xi = X[i]  # [3]

            # color cost to each center
            d2 = np.sum((centers - xi[None, :]) ** 2, axis=1)  # [K]

            # neighbour agreement term
            neigh_counts = np.zeros(K, dtype=np.float64)
            for dy, dx in neigh_offsets:
                yy = y + dy
                xx = x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    lab = label_img[yy, xx]
                    if lab >= 0:
                        neigh_counts[lab] += 1.0

            # minimize energy = color cost - reward for matching neighbours
            costs = color_weight * d2 - neighbor_weight * neigh_counts
            new_lab = int(np.argmin(costs))

            if new_lab != labels[i]:
                labels[i] = new_lab
                label_img[y, x] = new_lab
                changed += 1

        # recompute centers
        new_centers = centers.copy()
        for k in range(K):
            m = labels == k
            if np.any(m):
                new_centers[k] = X[m].mean(axis=0)
        centers = new_centers

        if changed == 0:
            break

    return labels, centers
    
    
# --------------------------- writer for multiple CSVs ---------------------------

def write_combined_curves_csv(curves_xy, out_path):
    """
    Write multiple curves into one CSV with columns:
      curve1_x, curve1_y, curve2_x, curve2_y, ...
    curves_xy: list of (x_out, y_out)
    """
    import csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n_curves = len(curves_xy)
    max_len = max((len(x) for x, y in curves_xy), default=0)

    header = []
    for i in range(n_curves):
        header.extend([f"curve{i+1}_x", f"curve{i+1}_y"])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for row_idx in range(max_len):
            row = []
            for x_out, y_out in curves_xy:
                if row_idx < len(x_out):
                    row.extend([f"{x_out[row_idx]:.12g}", f"{y_out[row_idx]:.12g}"])
                else:
                    row.extend(["", ""])
            w.writerow(row)
  
def assign_pixels_to_known_centers(
    img_bgr,
    centers,
    inside_mask=None,
    axes_exclude_mask=None,
    exclude_grayscale=False,
    reject_marker_red=False,
    marker_red_tol=4.0,
    lum_min=COLOR_FILTER["lum_min"],
    lum_max=COLOR_FILTER["lum_max"],
    use_lab=COLOR_FILTER["spatial_use_lab"],
):
    """
    Build cluster masks by assigning eligible pixels to precomputed color centers,
    then restore the iterative spatial relabelling step.
    """
    H, W, _ = img_bgr.shape

    valid = np.ones((H, W), dtype=bool)

    if inside_mask is not None:
        valid &= inside_mask

    if exclude_grayscale:
        valid &= ~grayscale_mask_bgr(img_bgr)

    if axes_exclude_mask is not None:
        valid &= ~axes_exclude_mask

    if reject_marker_red:
        valid &= ~marker_red_mask_bgr(img_bgr, tol=marker_red_tol)

    valid &= luminance_mask_bgr(img_bgr, lum_min=lum_min, lum_max=lum_max)

    ys, xs = np.where(valid)
    K = centers.shape[0]

    cluster_masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(K)]
    if xs.size == 0:
        return cluster_masks

    feat_img = get_feature_image_for_clustering(img_bgr, use_lab=use_lab)
    X = feat_img[ys, xs].astype(np.float64)

    # initial assignment to known centers
    d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int32)

    # bring back iterative spatial reassignment
    labels, centers_refined = refine_labels_spatial_icm(
        feat_img=feat_img,
        ys=ys,
        xs=xs,
        labels=labels,
        centers=np.asarray(centers, dtype=np.float64).copy(),
        n_iter=COLOR_FILTER["spatial_refine_iters"],
        color_weight=COLOR_FILTER["spatial_color_weight"],
        neighbor_weight=COLOR_FILTER["spatial_neighbor_weight"],
    )

    for i in range(K):
        m = labels == i
        cluster_masks[i][ys[m], xs[m]] = 255

    return cluster_masks
  
def maybe_prepend_origin_point(x_out, y_out, enabled=False):
    if not enabled:
        return x_out, y_out

    if x_out.size > 0 and np.isclose(x_out[0], 0.0) and np.isclose(y_out[0], 0.0):
        return x_out, y_out

    x_out = np.concatenate([np.array([0.0]), x_out])
    y_out = np.concatenate([np.array([0.0]), y_out])
    return x_out, y_out


def add_synthetic_endpoint_points(curves_xy, xmin, xmax, xscale_log=False, yscale_log=False):
    """
    Add synthetic left and right endpoint points after all other curve processing.
    Left point:  (0, 0) for linear-x, or (xmin, y_zero_like) for log-x
    Right point: (xmax, y_zero_like)
    For log-y, y_zero_like is float32 tiny.
    """
    y_zero_like = float(np.finfo(np.float32).tiny) if yscale_log else 0.0
    x_left = xmin if xscale_log else 0.0

    out = []
    for x_out, y_out in curves_xy:
        x_out = np.asarray(x_out, dtype=np.float64).copy()
        y_out = np.asarray(y_out, dtype=np.float64).copy()

        if COLOR_FILTER.get("prepend_origin_point", False):
            has_left_point = np.any(np.isclose(x_out, x_left) & np.isclose(y_out, y_zero_like))
            if not has_left_point:
                x_out = np.concatenate([x_out, np.array([x_left])])
                y_out = np.concatenate([y_out, np.array([y_zero_like])])

        if COLOR_FILTER.get("append_xmax_zero_point", False):
            has_xmax_point = np.any(np.isclose(x_out, xmax) & np.isclose(y_out, y_zero_like))
            if not has_xmax_point:
                x_out = np.concatenate([x_out, np.array([xmax])])
                y_out = np.concatenate([y_out, np.array([y_zero_like])])

        order = np.argsort(x_out)
        x_out = x_out[order]
        y_out = y_out[order]

        keep = np.ones(x_out.size, dtype=bool)
        for k in range(1, x_out.size):
            if np.isclose(x_out[k], x_out[k - 1]) and np.isclose(y_out[k], y_out[k - 1]):
                keep[k] = False

        out.append((x_out[keep], y_out[keep]))

    return out

def _tail_points_overlap(xa, ya, xb, yb, lookback_points=10, dist_tol=2.5):
    """
    Check whether the last few real points of curve A are close to curve B.
    Uses nearest-neighbour distance in data space.
    """
    if xa.size == 0 or xb.size == 0:
        return False

    n = min(int(lookback_points), xa.size)
    if n <= 0:
        return False

    tail_a = np.stack([xa[-n:], ya[-n:]], axis=1)
    pts_b = np.stack([xb, yb], axis=1)

    d2 = np.sum((tail_a[:, None, :] - pts_b[None, :, :]) ** 2, axis=2)
    min_d = np.sqrt(np.min(d2, axis=1))

    return bool(np.all(min_d <= dist_tol))


def extend_curve_tails_from_overlaps(curves_xy,
                                     lookback_points=10,
                                     dist_tol=2.5):
    """
    If curve A ends while closely following curve B, and curve B continues to larger x,
    append B's beyond-A tail points onto A.

    curves_xy: list of (x_out, y_out)
    returns: list of (x_out, y_out)
    """
    if not curves_xy:
        return curves_xy

    out = [(np.asarray(x, dtype=np.float64).copy(),
            np.asarray(y, dtype=np.float64).copy())
           for (x, y) in curves_xy]

    n_curves = len(out)

    for i in range(n_curves):
        xa, ya = out[i]
        if xa.size == 0:
            continue

        xmax_a = float(np.max(xa))

        best_j = None
        best_extra = None
        best_extra_count = 0

        for j in range(n_curves):
            if j == i:
                continue

            xb, yb = out[j]
            if xb.size == 0:
                continue

            extra_mask = xb > xmax_a
            if not np.any(extra_mask):
                continue

            if not _tail_points_overlap(
                xa, ya, xb, yb,
                lookback_points=lookback_points,
                dist_tol=dist_tol,
            ):
                continue

            extra_x = xb[extra_mask]
            extra_y = yb[extra_mask]

            if extra_x.size > best_extra_count:
                best_j = j
                best_extra = (extra_x, extra_y)
                best_extra_count = extra_x.size

        if best_extra is not None:
            extra_x, extra_y = best_extra
            xa2 = np.concatenate([xa, extra_x])
            ya2 = np.concatenate([ya, extra_y])

            order = np.argsort(xa2)
            xa2 = xa2[order]
            ya2 = ya2[order]

            # remove exact duplicate x,y pairs
            keep = np.ones(xa2.size, dtype=bool)
            for k in range(1, xa2.size):
                if np.isclose(xa2[k], xa2[k - 1]) and np.isclose(ya2[k], ya2[k - 1]):
                    keep[k] = False

            out[i] = (xa2[keep], ya2[keep])

    return out
    
# --------------------------- Main logic & entrypoints ---------------------------

def main_logic(*, image, xmin, xmax, ymin, ymax, k=100, min_sat=30, min_val=30,
               knn_k=3, knn_thresh=35.0, out="", debug=False,
               monotone_y=False, monotone_method="pava",
               auto_colors=False, exclude_grayscale=False,
               gray_abs=2.0, gray_rel=0.05,
               num_colors=0, reject_marker_red=False, marker_red_tol=4.0,
               xscale_log=False, yscale_log=False,
               keep_outside_axes=False):            
    """
    Core digitization pipeline.
    Returns (x_out, y_out).
    """
    img_path = resolve_image_path(image)
    if img_path != image:
        print(f"[digitizer] resolved path:\n  given : {image}\n  actual: {img_path}", file=sys.stderr)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        try:
            data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None

    if img is None:
        print(f"Could not read image: {img_path}", file=sys.stderr)
        return None, None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if auto_colors:

        # 1) detect axis reference points from red markers
        bl_px, br_px, tl_px, red_mask = detect_axes_red_points(
            img, hsv, min_sat=min_sat, min_val=min_val, knn_k=knn_k, knn_thresh=knn_thresh
        )

        origin, ax_vec, ay_vec, ax_n2, ay_n2 = build_affine_axes(bl_px, br_px, tl_px)
        
        #axes_exclude_mask = mask_near_axes_region(
        #    img.shape,
        #    origin,
        #    ax_vec,
        #    ay_vec,
        #    axis_dist_px=10.0,
        #)
        
        
        axes_exclude_mask_clustering = mask_near_axes_region(
            img.shape,
            origin,
            ax_vec,
            ay_vec,
            axis_dist_px=COLOR_FILTER["axes_exclude_dist_for_clustering"],
        )
        
        axes_exclude_mask_curve = mask_near_axes_region(
            img.shape,
            origin,
            ax_vec,
            ay_vec,
            axis_dist_px=COLOR_FILTER["axes_exclude_dist_for_curve_masks"],
        )
                
        
        
        inside_axes_mask = None if keep_outside_axes else mask_inside_axes_region(
                                         img.shape, origin, ax_vec, ay_vec, ax_n2, ay_n2
                                     )


        # 2) discover non-gray, non-marker-red color clusters

        
        cluster_masks, centers_rgb, centers_feat, chosen_k, model_scores = discover_color_clusters(
            img,
            exclude_grayscale=exclude_grayscale,
            gray_abs=gray_abs,
            gray_rel=gray_rel,
            num_colors=num_colors,
            reject_marker_red=reject_marker_red,
            marker_red_tol=marker_red_tol,
            inside_mask=inside_axes_mask,
            axes_exclude_mask=axes_exclude_mask_clustering,
        )
        
        cluster_masks = assign_pixels_to_known_centers(
            img,
            centers_feat,
            inside_mask=inside_axes_mask,
            axes_exclude_mask=axes_exclude_mask_curve,
            exclude_grayscale=exclude_grayscale,
            reject_marker_red=reject_marker_red,
            marker_red_tol=marker_red_tol,
        )
        
        
        
        #remove legend
        # 2b) combined legend detection on all groups together
        plot_w_px = float(np.sqrt(ax_n2))
        plot_h_px = float(np.sqrt(ay_n2))

        union_mask = make_union_mask(cluster_masks)

        legend_box = detect_legend_box_sliding(
            union_mask,
            plot_w_px=plot_w_px,
            plot_h_px=plot_h_px,
            box_frac_x_values=(0.15, 0.20, 0.25, 0.30, 0.35),
            box_frac_y_values=(0.15, 0.20, 0.25, 0.30, 0.35),
            isolation_frac=0.07,
            step_frac=0.01,
            min_points_in_box=20,
            min_fill_ratio=0.002,
        )

        if legend_box is not None:
            cluster_masks = remove_legend_box_from_masks(cluster_masks, legend_box)
        
            if debug:
                base = img_path.rsplit(".", 1)[0]
                save_legend_box_debug_png(img, legend_box, f"{base}_debug_legend_box.png")
                

        debug = 1
        if debug:
            base = img_path.rsplit(".", 1)[0]
            save_cluster_palette_png(centers_rgb, f"{base}_debug_palette.png")
            
##################################
        # 2c) geometric snap-back cleanup to reduce jagged line swapping
        if COLOR_FILTER.get("curve_snap_enable", True):

                        
            cluster_masks, ref_curve_mask = snap_all_curve_masks_to_reference(
                cluster_masks,
                dilate_radius=COLOR_FILTER["curve_snap_dilate_radius"],
                search_radius=COLOR_FILTER["curve_snap_search_radius"],
            )
            

            if debug:
                base = img_path.rsplit(".", 1)[0]
                save_reference_curve_set_debug_png(
                    img,
                    ref_curve_mask,
                    f"{base}_debug_reference_curve_set.png"
                )
##################################            
        
        print(f"[digitizer] auto-colors chose K={chosen_k}", file=sys.stderr)
        
        
        

        # 3) digitize every discovered color cluster
        all_results = []
        base_out = out

        for i, curve_mask in enumerate(cluster_masks, start=1):

            x_out, y_out = digitize_curve_mask(
                curve_mask,
                origin, ax_vec, ay_vec, ax_n2, ay_n2,
                xmin, xmax, ymin, ymax, k,
                xscale_log=xscale_log, yscale_log=yscale_log,
                monotone_y=monotone_y, monotone_method=monotone_method,
                keep_outside_axes=keep_outside_axes
            )

            if x_out is None:
                print(f"[digitizer] cluster {i}: no valid in-box points after filtering; skipped", file=sys.stderr)
                continue

            
            all_results.append((x_out, y_out, curve_mask))            
            
            
        curves_xy = [(x_out, y_out) for (x_out, y_out, _) in all_results]

        if COLOR_FILTER.get("extend_tail_from_overlapping_curve", False):
            curves_xy = extend_curve_tails_from_overlaps(
                curves_xy,
                lookback_points=COLOR_FILTER.get("tail_overlap_lookback_points", 10),
                dist_tol=COLOR_FILTER.get("tail_overlap_distance_tol", 2.5),
            )

        curves_xy = add_synthetic_endpoint_points(
            curves_xy,
            xmin=xmin,
            xmax=xmax,
            xscale_log=xscale_log,
            yscale_log=yscale_log,
        )

        if base_out:
            write_combined_curves_csv(curves_xy, base_out)
            
            
            
            
            print(f"[digitizer] wrote combined CSV {base_out}", file=sys.stderr)
        else:
            # Print a combined CSV-like table to stdout
            n_curves = len(curves_xy)
            max_len = max((len(x) for x, y in curves_xy), default=0)

            header = []
            for i in range(n_curves):
                header.extend([f"curve{i+1}_x", f"curve{i+1}_y"])
            print(",".join(header))

            for row_idx in range(max_len):
                row = []
                for x_out, y_out in curves_xy:
                    if row_idx < len(x_out):
                        row.extend([f"{x_out[row_idx]:.12g}", f"{y_out[row_idx]:.12g}"])
                    else:
                        row.extend(["", ""])
                print(",".join(row))

        if debug:
            base = img_path.rsplit(".", 1)[0]
            cv2.imwrite(f"{base}_debug_redmask.png", red_mask)

            dbg = img.copy()
            for p in (bl_px, br_px, tl_px):
                cv2.circle(dbg, (int(round(p[0])), int(round(p[1]))), 8, (0, 0, 255), -1)
            p_bl = (int(round(origin[0])), int(round(origin[1])))
            p_br = (int(round(origin[0] + ax_vec[0])), int(round(origin[1] + ax_vec[1])))
            p_tl = (int(round(origin[0] + ay_vec[0])), int(round(origin[1] + ay_vec[1])))
            cv2.line(dbg, p_bl, p_br, (0, 255, 0), 2)
            cv2.line(dbg, p_bl, p_tl, (255, 0, 0), 2)
            cv2.imwrite(f"{base}_debug_axes.png", dbg)

            for i, (_, _, curve_mask) in enumerate(all_results, start=1):
                cv2.imwrite(f"{base}_debug_curve_mask_{i}.png", curve_mask)

        return all_results
    else:
        bl_px, br_px, tl_px, red_mask = find_red_centroids_strict(img)
        purple_mask, _, _ = knn_classify_hsv(hsv, knn_k, min_sat, min_val, knn_thresh)
        curve_pts_px, purple_mask_clean = extract_purple_points_knn(img, hsv, purple_mask, red_mask)
    
    
    origin, ax_vec, ay_vec, ax_n2, ay_n2 = build_affine_axes(bl_px, br_px, tl_px)

    if keep_outside_axes:
        tx, ty = pixels_to_params_unclipped(curve_pts_px, origin, ax_vec, ay_vec, ax_n2, ay_n2)
    else:
        _, tx, ty = filter_points_inside_axes(curve_pts_px, origin, ax_vec, ay_vec, ax_n2, ay_n2)
    
    
        
    x, y = params_to_data(
        tx, ty, xmin, xmax, ymin, ymax,
        xscale_log=xscale_log,
        yscale_log=yscale_log
    )
    

    xy = np.stack([x, y], axis=1)
    sel_idx = farthest_point_sampling_xy(xy, k)
    x_sel = x[sel_idx]
    y_sel = y[sel_idx]

    order = np.argsort(x_sel)
    x_out = x_sel[order]
    y_out = y_sel[order]

    if monotone_y:
        y_out = enforce_monotone_nonincreasing(y_out, method=monotone_method)

    curves_xy = add_synthetic_endpoint_points(
        [(x_out, y_out)],
        xmin=xmin,
        xmax=xmax,
        xscale_log=xscale_log,
        yscale_log=yscale_log,
    )
    x_out, y_out = curves_xy[0]

    if out:
        import csv
        out_path = out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y"])
            for xi, yi in zip(x_out, y_out):
                w.writerow([f"{xi:.12g}", f"{yi:.12g}"])
        print(f"[digitizer] wrote {out_path}", file=sys.stderr)
    else:
        print("x,y")
        for xi, yi in zip(x_out, y_out):
            print(f"{xi:.12g},{yi:.12g}")

    if debug:
        base = img_path.rsplit(".", 1)[0]

        cv2.imwrite(f"{base}_debug_redmask.png", red_mask)
        cv2.imwrite(f"{base}_debug_curve_mask.png", purple_mask_clean)

        dbg = img.copy()
        for p in (bl_px, br_px, tl_px):
            cv2.circle(dbg, (int(round(p[0])), int(round(p[1]))), 8, (0, 0, 255), -1)

        p_bl = (int(round(origin[0])), int(round(origin[1])))
        p_br = (int(round(origin[0] + ax_vec[0])), int(round(origin[1] + ax_vec[1])))
        p_tl = (int(round(origin[0] + ay_vec[0])), int(round(origin[1] + ay_vec[1])))

        cv2.line(dbg, p_bl, p_br, (0, 255, 0), 2)
        cv2.line(dbg, p_bl, p_tl, (255, 0, 0), 2)
        cv2.imwrite(f"{base}_debug_axes.png", dbg)

        overlay = img.copy()
        sel_px = curve_pts_px[sel_idx]
        for (px, py) in sel_px.astype(int):
            cv2.circle(overlay, (px, py), 1, (255, 255, 0), -1)
        cv2.imwrite(f"{base}_debug_selected_overlay.png", overlay)

    return x_out, y_out

def main_cli():
    """Command-line entrypoint."""
    args = parse_args()

    if args.batch:
        return main_caller()

    missing = []
    if not args.image:
        missing.append("image")
    if args.xmin is None:
        missing.append("--xmin")
    if args.xmax is None:
        missing.append("--xmax")
    if args.ymin is None:
        missing.append("--ymin")
    if args.ymax is None:
        missing.append("--ymax")

    if missing:
        raise SystemExit(
            "Missing required arguments for single-file mode: "
            + ", ".join(missing)
            + "\nUse --batch to run the hard-coded batch jobs."
        )

    arg_dict = vars(args).copy()
    arg_dict.pop("batch", None)
    return main_logic(**arg_dict)


def main_caller():
    """
    Batch entrypoint: hard-code jobs here.
    Each dict uses the same keys as main_logic / parse_args.
    """
    
    
    def make_job_like(base_job, image, xmin, xmax, ymin, ymax):
        job = base_job.copy()
        job["image"] = image
        job["xmin"] = xmin
        job["xmax"] = xmax
        job["ymin"] = ymin
        job["ymax"] = ymax
        job["out"] = str(Path(image).with_suffix(".csv"))
        return job
    
    jobs = [
        dict(
    image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig1_a.png",
    xmin=0, xmax=200, ymin=0, ymax=140,
    xscale_log=False,
    yscale_log=False,    
    k=1000,#was100
    auto_colors=True,
    exclude_grayscale=True,
    keep_outside_axes=False,
    gray_abs=2.0,
    gray_rel=0.05,
    num_colors=5,
    reject_marker_red=True,
    marker_red_tol=4.0,
    monotone_y=False,
    monotone_method="pava",
    out=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig1_a.csv",
    debug=True,
        ),
    ]
    
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3figl_b.png",  xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3figl_c.png",  xmin=0, xmax=200, ymin=0, ymax=2000, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3figl_d.png",  xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3figl_e.png",  xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3figl_f.png",  xmin=0, xmax=200, ymin=0, ymax=2000, ))

    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig2a.png",   xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig2b.png",   xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig2c.png",   xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig2d.png",   xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig2e.png",   xmin=0, xmax=200, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig2f.png",   xmin=0, xmax=200, ymin=0, ymax=2500, ))
                                                                                                                  
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig3a.png",   xmin=0, xmax=400, ymin=0, ymax=3000, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig3b.png",   xmin=0, xmax=400, ymin=0, ymax=2500, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig3c.png",   xmin=0, xmax=400, ymin=0, ymax=3000, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig3d.png",   xmin=0, xmax=400, ymin=0, ymax=3000, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig3e.png",   xmin=0, xmax=400, ymin=0, ymax=3000, ))
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig3f.png",   xmin=0, xmax=400, ymin=0, ymax=3000, ))


    

    for j in jobs:
        print(f"[digitizer] processing {j['image']} → {j.get('out', '(stdout)')}", file=sys.stderr)
        main_logic(**j)


if __name__ == "__main__":
    main_cli()
