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
#Auto-select number of colors, reject grayscale:
python Digitize1.py input.png --xmin 0 --xmax 1 --ymin 1e-7 --ymax 1 --auto-colors --exclude-grayscale --monotone-y --out out.csv --debug
#Force exactly 6 color groups:
python Digitize1.py input.png --xmin 0 --xmax 1 --ymin 1e-7 --ymax 1 --auto-colors --num-colors 6 --exclude-grayscale --out out.csv
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
from dataclasses import dataclass
from itertools import product
from joblib import Parallel, delayed



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
    
    # colour feature representation used everywhere downstream:
    #   "lab"         -> old perceptual path
    #   "rgb"         -> old RGB path
    "feature_space": "lab",
    
    "feature_norm_min_scale": 1e-6,    
    
    "rolling_prune_enable": True,
    "rolling_prune_max_remove_frac": 0.20, #was 0.2
    "rolling_prune_window": 51, #was 51, 
    "rolling_prune_sigma_thresh": 2.0, #was 1.25, was cutting off tops of curves
    "rolling_prune_max_iters": 5,
    "rolling_prune_protect_endpoints": True,

    "autotune_min_preprune_points_frac": 0.2,
        
        
    "autotune_sat_grid_fast": [0.80, 1.00, 1.20, 1.45, 1.8, 3.6, 8.0],
    "autotune_val_grid_fast": [1.00],
    "autotune_contrast_grid_fast": [1.00],
    "autotune_certainty_ratio_grid_fast": [0.55, 0.65, 0.75, 0.85, 0.95],
    
    "autotune_sat_grid_full": [0.80, 1.00, 1.20, 1.45, 1.8, 3.6, 8.0, 12, 16, 20],
    "autotune_val_grid_full": [0.7, 0.8, 0.9, 1.00, 1.1, 1.2, 1.3],
    "autotune_contrast_grid_full": [0.7, 0.8, 0.9, 1.00, 1.1, 1.2, 1.3],
    "autotune_certainty_ratio_grid_full": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    
    
    
    
    
    "autotune_abs_dist_grid": [None],        
    "autotune_max_acceptable_roughness": 5000, #greater than this causes dither between curves - but can try 20000
    
    "autotune_parallel_n_jobs": -1,
    "autotune_parallel_backend": "loky",
    "autotune_parallel_batch_size": "auto",

    "roughness_method": "mean_squared_curvature",   # or "median_abs_curvature"
    "roughness_resample_n": 200,
    
    "spike_prune_enable": True,

    # 1-point spike removal
    "one_point_spike_factor": 3.0,
    "one_point_neighbor_pair_factor": 1.0,
    "one_point_max_passes": 10,
    "one_point_neighbor_rel_tol": 0.30,

    # 2-point spike removal
    "two_point_spike_factor": 3.0,
    "two_point_pair_tight_factor": 1.0,
    "two_point_baseline_tight_factor": 1.0,
    "two_point_max_passes": 10,
    "two_point_baseline_rel_tol": 0.30,
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
    
    
    
    p.add_argument("--debug", action="store_true",
               help="Print debug info to stderr.")
    
    p.add_argument("--monotone-y", action="store_true",
                   help="After simplification, enforce y[i+1] <= y[i] (non-increasing).")
    p.add_argument("--monotone-method", choices=["pava", "clip"], default="pava",
                   help="Method for monotonic y enforcement: 'pava' or 'clip'.")
    p.add_argument("--auto-colors", action="store_true",
                   help="Automatically discover distinct non-grayscale colors and digitize the main curve.")
    p.add_argument("--exclude-grayscale", action="store_true",
                   help="Exclude grayscale-ish pixels before color grouping.")
    #p.add_argument("--gray-abs", type=float, default=2.0,
    #               help="Absolute grayscale tolerance in RGB units (default: 2.0).")
    #p.add_argument("--gray-rel", type=float, default=0.05,
    #               help="Relative grayscale tolerance (default: 0.05 = 5%%).")

    p.add_argument("--num-colors", type=int, default=0,
                   help="If > 0, use exactly this many discovered color groups instead of auto-selecting 2..20.")
    #p.add_argument("--reject-marker-red", action="store_true",
    #               help="Reject near-pure-red clusters as curve candidates.")
    #p.add_argument("--marker-red-tol", type=float, default=4.0,
    #               help="Tolerance around pure red RGB=(255,0,0) for rejecting marker-red clusters.")
    p.add_argument("--xscale_log", action="store_true",
                   help="Use logarithmic x-axis mapping. Default is linear.")
    p.add_argument("--yscale_log", action="store_true",
                   help="Use logarithmic y-axis mapping. Default is linear.")
                   
    p.add_argument("--keep-outside-axes", action="store_true",
                   help="Do not reject points outside the axes box after clustering.")
                   
    p.add_argument("--debug-images", action="store_true",
                   help="Save debug PNG overlays next to input image.")
               
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

# --------------------------- Feature normalization ---------------------------

@dataclass
class FeatureNormalizer:
    feature_space: str
    offset: np.ndarray   # shape [3]
    scale: np.ndarray    # shape [3]

    def transform(self, feat_img):
        feat_img = np.asarray(feat_img, dtype=np.float64)
        return (feat_img - self.offset.reshape(1, 1, 3)) / self.scale.reshape(1, 1, 3)


        
    def inverse_transform_points(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale.reshape(1, 3) + self.offset.reshape(1, 3)        


def fit_feature_normalizer_from_valid_pixels(feat_img_raw, valid_mask, feature_space, min_scale=None):
    """
    Fit per-channel affine normalization using only valid pixels:
        X_norm = (X - offset) / scale

    Returns a FeatureNormalizer.
    """
    if min_scale is None:
        min_scale = COLOR_FILTER.get("feature_norm_min_scale", 1e-6)
        
    ys, xs = np.where(valid_mask)
    if xs.size == 0:
        raise ValueError("No valid pixels available for fitting feature normalizer.")

    X = feat_img_raw[ys, xs].astype(np.float64)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scale = np.maximum(x_max - x_min, float(min_scale))

    return FeatureNormalizer(
        feature_space=feature_space,
        offset=x_min,
        scale=scale,
    )
    

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
    
    

    
    
# --------------------------- Corner-based red marker detection ---------------------------

def _corner_scan_order(H, W, corner):
    """
    Yield pixel coordinates in diagonal shells from a corner inward.

    corner:
      - "tl" : top-left
      - "bl" : bottom-left
      - "br" : bottom-right

    Order is shell-by-shell, where shell index d increases by 1 pixel.
    """
    if corner not in {"tl", "bl", "br"}:
        raise ValueError(f"Unknown corner: {corner}")

    max_d = (H - 1) + (W - 1)

    for d in range(max_d + 1):
        pts = []

        if corner == "tl":
            # x + y = d
            y_min = max(0, d - (W - 1))
            y_max = min(H - 1, d)
            for y in range(y_min, y_max + 1):
                x = d - y
                pts.append((x, y))

        elif corner == "bl":
            # x + (H-1-y) = d
            t_min = max(0, d - (W - 1))
            t_max = min(H - 1, d)
            for t in range(t_min, t_max + 1):
                y = (H - 1) - t
                x = d - t
                pts.append((x, y))

        elif corner == "br":
            # (W-1-x) + (H-1-y) = d
            t_min = max(0, d - (W - 1))
            t_max = min(H - 1, d)
            for t in range(t_min, t_max + 1):
                y = (H - 1) - t
                x = (W - 1) - (d - t)
                pts.append((x, y))

        yield d, pts


def find_corner_red_point(red_bool, corner, min_pixels=4):
    """
    Find one red corner marker by scanning diagonal shells from the chosen corner inward.

    Returns:
      center_xy : np.array([x, y], dtype=float64)

    Strategy:
      - accumulate strict-red pixels shell by shell
      - stop at the first shell where at least min_pixels have been found
      - return the mean of all accumulated red pixels up to that shell
    """
    H, W = red_bool.shape
    hits = []

    for d, pts in _corner_scan_order(H, W, corner):
        for x, y in pts:
            if red_bool[y, x]:
                hits.append((x, y))

        if len(hits) >= min_pixels:
            arr = np.asarray(hits, dtype=np.float64)
            return arr.mean(axis=0)

    raise ValueError(
        f"Could not find {min_pixels} strict-red pixels from corner '{corner}'."
    )


def find_red_corners_by_corner_scan(img_bgr, min_pixels=4):
    """
    Detect the three axis reference markers by scanning from plot corners inward.

    Returns:
      bl, br, tl, red_mask
    """
    red_bool = strict_red_mask_bgr(img_bgr)

    red_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    red_mask[red_bool] = 255

    tl = find_corner_red_point(red_bool, "tl", min_pixels=min_pixels)
    bl = find_corner_red_point(red_bool, "bl", min_pixels=min_pixels)
    br = find_corner_red_point(red_bool, "br", min_pixels=min_pixels)

    return bl, br, tl, red_mask
    
    
    
    
def detect_axes_red_points(
    img_bgr,
    hsv=None,
    min_sat=30,
    min_val=30,
    knn_k=3,
    knn_thresh=35.0,
    corner_min_pixels=4,
    ):
    """
    Detect the three red axis reference points by scanning from the
    top-left, bottom-left, and bottom-right corners inward.
    """
    bl_px, br_px, tl_px, red_mask = find_red_corners_by_corner_scan(
        img_bgr,
        min_pixels=corner_min_pixels,
    )
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
    sample_limit=COLOR_FILTER["sample_limit"],
    kmin=COLOR_FILTER["kmin"],
    kmax=COLOR_FILTER["kmax"],
    num_colors=0,
    inside_mask=None,
    axes_exclude_mask=None,
    lum_min=COLOR_FILTER["lum_min"],
    lum_max=COLOR_FILTER["lum_max"],
    save_debug_images=False,
    ):
    """
    Cluster image colors after filtering, then refine labels spatially so nearby
    eligible pixels prefer to share the same group.

    Returns:
      cluster_masks: list of HxW uint8 masks
      cluster_centers_rgb: [K,3]
      cluster_centers_feat: [K,3] in normalized feature space
      feature_normalizer: FeatureNormalizer used for both clustering and reassignment
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

    #if reject_marker_red:
    #    valid &= ~marker_red_mask_bgr(img_bgr, tol=marker_red_tol)

    valid &= luminance_mask_bgr(img_bgr, lum_min=lum_min, lum_max=lum_max)
    if save_debug_images:
    
        save_candidate_pixels_debug_png(
            img_bgr,
            valid,
            str(SCRIPT_DIR / "debug_candidate_pixels.png")
        )

    ys, xs = np.where(valid)
    if xs.size == 0:
        raise ValueError("No eligible non-grayscale/non-marker-red pixels available for color discovery.")

    # features for clustering/refinement
    feature_space = COLOR_FILTER.get("feature_space", "lab")

    feat_img_raw = get_feature_image_for_clustering(
        img_bgr,
        feature_space=feature_space,
    )

    feature_normalizer = fit_feature_normalizer_from_valid_pixels(
        feat_img_raw,
        valid,
        feature_space=feature_space,
    )

    feat_img = feature_normalizer.transform(feat_img_raw)
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
    
    centers_raw = feature_normalizer.inverse_transform_points(centers)

    if feature_space == "rgb":
        centers_rgb = np.clip(np.round(255.0 * centers_raw), 0, 255).astype(np.float64)
    
    elif feature_space == "lab":
        centers_u8 = np.clip(np.round(255.0 * centers_raw), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
        centers_rgb = cv2.cvtColor(centers_u8, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float64)

    else:
        raise ValueError(f"Unknown feature_space: {feature_space}")






    return cluster_masks, centers_rgb, centers, feature_normalizer, chosen_k, model_scores
    


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

def legend_box_to_mask(img_shape, legend_box):
    """
    Convert legend_box = (x0, y0, x1, y1) into a boolean exclusion mask.
    """
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=bool)

    if legend_box is None:
        return mask

    x0, y0, x1, y1 = legend_box

    cx0 = max(0, int(np.floor(x0)))
    cy0 = max(0, int(np.floor(y0)))
    cx1 = min(W, int(np.ceil(x1)))
    cy1 = min(H, int(np.ceil(y1)))

    if cx0 < cx1 and cy0 < cy1:
        mask[cy0:cy1, cx0:cx1] = True

    return mask

def save_candidate_pixels_debug_png(img_bgr, valid_mask, out_path):
    """
    Save an image showing only the pixels that remain eligible for color clustering.
    Rejected pixels are painted white.
    """
    dbg = np.full_like(img_bgr, 255)
    dbg[valid_mask] = img_bgr[valid_mask]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, dbg)
    

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
def get_feature_image_for_clustering(img_bgr, feature_space=None):
    """
    Return per-pixel raw colour features, before post-exclusion normalization.

    feature_space options:
      - "rgb" : RGB in [0,1]
      - "lab" : OpenCV Lab in [0,1]

    Output shape: [H, W, 3], dtype float64
    """
    if feature_space is None:
        feature_space = COLOR_FILTER.get("feature_space", "lab")

    if feature_space == "rgb":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    if feature_space == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64) / 255.0

    raise ValueError(f"Unknown feature_space: {feature_space}")
    
    
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
    feature_normalizer,
    inside_mask=None,
    axes_exclude_mask=None,
    exclude_grayscale=False,
    #reject_marker_red=False,
    #marker_red_tol=4.0,
    lum_min=COLOR_FILTER["lum_min"],
    lum_max=COLOR_FILTER["lum_max"],
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

    #if reject_marker_red:
    #    valid &= ~marker_red_mask_bgr(img_bgr, tol=marker_red_tol)

    valid &= luminance_mask_bgr(img_bgr, lum_min=lum_min, lum_max=lum_max)

    ys, xs = np.where(valid)
    K = centers.shape[0]

    cluster_masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(K)]
    if xs.size == 0:
        return cluster_masks

    feat_img_raw = get_feature_image_for_clustering(
        img_bgr,
        feature_space=feature_normalizer.feature_space,
    )
    feat_img = feature_normalizer.transform(feat_img_raw)
    X = feat_img[ys, xs].astype(np.float64)

    # initial assignment to known centers
    d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int32)

    # bring back iterative spatial reassignment
    labels, _ = refine_labels_spatial_icm(
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
    
# --------------------------- Remove spikes ---------------------------
def remove_isolated_y_spikes(
    x_out,
    y_out,
    spike_factor=6.0,
    neighbor_pair_factor=2.0,
    max_passes=10,
):
    """
    Remove 1-point spikes with a floating 3-point stencil.

    Middle point is removed when:
      - the two neighbours are close to each other
      - the middle point is far from their local baseline

    This catches both upward and downward 1-point spikes.
    """
    x = np.asarray(x_out, dtype=np.float64).copy()
    y = np.asarray(y_out, dtype=np.float64).copy()

    if y.size < 3:
        return x, y

    for _ in range(max_passes):
        keep = np.ones(y.size, dtype=bool)
        removed_any = False

        for i in range(1, y.size - 1):
            y_left = y[i - 1]
            y_mid = y[i]
            y_right = y[i + 1]

            baseline_mean = 0.5 * (y_left + y_right)
            baseline_spread = abs(y_left - y_right)
            mid_offset = abs(y_mid - baseline_mean)

            amp_scale = max(abs(baseline_mean), abs(y_left), abs(y_right), 1.0)
            neighbour_tol = neighbor_pair_factor * COLOR_FILTER["one_point_neighbor_rel_tol"] * amp_scale
            spike_tol = spike_factor * max(baseline_spread, 1e-12)

            if baseline_spread <= neighbour_tol and mid_offset > spike_tol:
                keep[i] = False
                removed_any = True

        x = x[keep]
        y = y[keep]

        if not removed_any or y.size < 3:
            break

    return x, y
def remove_two_point_y_spikes(
    x_out,
    y_out,
    spike_factor=6.0,
    pair_tight_factor=2.0,
    baseline_tight_factor=2.0,
    max_passes=10,
):
    """
    Remove 2-point spikes with a floating 5-point stencil.

    For each 5-point window, test every possible choice of 2 points out of 5.
    Remove a candidate pair when:
      - the pair is tight internally
      - the remaining 3 baseline points are tight
      - the pair mean is far from the baseline mean

    This catches both upward and downward 2-point spikes, even when the
    offending pair is not centered in the window.
    """
    x = np.asarray(x_out, dtype=np.float64).copy()
    y = np.asarray(y_out, dtype=np.float64).copy()

    if y.size < 5:
        return x, y

    pair_choices = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 4),
    ]

    for _ in range(max_passes):
        keep = np.ones(y.size, dtype=bool)
        removed_any = False

        for start in range(0, y.size - 4):
            win_idx = np.arange(start, start + 5, dtype=np.int64)
            win_y = y[win_idx]

            best_pair = None
            best_strength = -np.inf

            for a, b in pair_choices:
                pair_mask = np.zeros(5, dtype=bool)
                pair_mask[a] = True
                pair_mask[b] = True
                base_mask = ~pair_mask

                pair_y = win_y[pair_mask]
                base_y = win_y[base_mask]

                pair_mean = float(np.mean(pair_y))
                base_mean = float(np.mean(base_y))

                pair_spread = float(np.max(pair_y) - np.min(pair_y))
                base_spread = float(np.max(base_y) - np.min(base_y))
                pair_offset = abs(pair_mean - base_mean)

                amp_scale = max(np.max(np.abs(base_y)), abs(base_mean), 1.0)

                baseline_tol = baseline_tight_factor * COLOR_FILTER["two_point_baseline_rel_tol"] * amp_scale
                pair_tol = pair_tight_factor * max(base_spread, baseline_tol)
                spike_tol = spike_factor * max(base_spread, baseline_tol)

                baseline_is_tight = base_spread <= baseline_tol
                pair_is_tight = pair_spread <= pair_tol
                pair_is_far = pair_offset > spike_tol

                if baseline_is_tight and pair_is_tight and pair_is_far:
                    strength = pair_offset / max(spike_tol, 1e-12)
                    if strength > best_strength:
                        best_strength = strength
                        best_pair = (win_idx[a], win_idx[b])

            if best_pair is not None:
                i0, i1 = best_pair
                keep[i0] = False
                keep[i1] = False
                removed_any = True

        x = x[keep]
        y = y[keep]

        if not removed_any or y.size < 5:
            break

    return x, y

# --------------------------- Fast rolling local outlier pruning ---------------------------

def remove_local_outlier_points_rolling(
    x_out,
    y_out,
    max_remove_frac=0.20, 
    window=51,
    sigma_thresh=1.25, 
    max_iters=5, 
    protect_endpoints=True,
):
    """
    Remove locally abnormal y-points using rolling mean/std.

    Strategy
    --------
    - for each point, compare y[i] against the rolling local mean/std
      computed from neighbouring points
    - mark points outside mean +/- sigma_thresh * std
    - remove them in batches
    - repeat for a small number of iterations
    - stop once max_remove_frac of original points has been removed

    Notes
    -----
    - window is forced odd
    - statistics exclude the point itself
    - endpoints can be protected
    - this is intentionally simple and fast
    """
    x = np.asarray(x_out, dtype=np.float64).copy()
    y = np.asarray(y_out, dtype=np.float64).copy()

    if x.size != y.size:
        raise ValueError("x_out and y_out must have same length.")

    n0 = x.size
    if n0 < 5:
        return x, y

    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1

    max_remove = int(np.floor(max_remove_frac * n0))
    if max_remove <= 0:
        return x, y

    total_removed = 0
    half = window // 2

    for _ in range(max_iters):
        n = y.size
        if n < 5:
            break

        keep = np.ones(n, dtype=bool)
        remove_mask = np.zeros(n, dtype=bool)

        start_idx = 1 if protect_endpoints else 0
        end_idx = n - 1 if protect_endpoints else n

        for i in range(start_idx, end_idx):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)

            idx = np.arange(lo, hi, dtype=np.int64)
            idx = idx[idx != i]

            if idx.size < 5:
                continue

            y_neigh = y[idx]
            mu = float(np.mean(y_neigh))
            sd = float(np.std(y_neigh))

            if not np.isfinite(sd) or sd <= 1e-12:
                continue

            if abs(y[i] - mu) > sigma_thresh * sd:
                remove_mask[i] = True

        n_remove_now = int(np.count_nonzero(remove_mask))
        if n_remove_now == 0:
            break

        remaining_budget = max_remove - total_removed
        if remaining_budget <= 0:
            break

        if n_remove_now > remaining_budget:
            # keep only the strongest offenders this round
            candidate_idx = np.flatnonzero(remove_mask)
            severities = np.zeros(candidate_idx.size, dtype=np.float64)

            for k, i in enumerate(candidate_idx):
                lo = max(0, i - half)
                hi = min(y.size, i + half + 1)

                idx = np.arange(lo, hi, dtype=np.int64)
                idx = idx[idx != i]
                if idx.size < 5:
                    severities[k] = -np.inf
                    continue

                y_neigh = y[idx]
                mu = float(np.mean(y_neigh))
                sd = float(np.std(y_neigh))
                if not np.isfinite(sd) or sd <= 1e-12:
                    severities[k] = -np.inf
                else:
                    severities[k] = abs(y[i] - mu) / sd

            order = np.argsort(-severities)
            chosen = candidate_idx[order[:remaining_budget]]

            remove_mask[:] = False
            remove_mask[chosen] = True
            n_remove_now = int(np.count_nonzero(remove_mask))

        keep[remove_mask] = False
        x = x[keep]
        y = y[keep]
        total_removed += n_remove_now

        if total_removed >= max_remove:
            break

    return x, y
    
# --------------------------- Per-curve feature adjustment search ---------------------------

def adjust_curve_feature_image(
    feat_img_raw,
    feature_space,
    sat_scale=1.0,
    val_scale=1.0,
    contrast_scale=1.0,
):
    """
    Apply per-curve feature adjustments before normalization/assignment.

    For now:
      - RGB feature space: approximate saturation/brightness/contrast in RGB
      - LAB feature space: adjust lightness contrast and chroma strength

    Returns adjusted raw feature image with same shape as feat_img_raw.
    """
    X = np.asarray(feat_img_raw, dtype=np.float64).copy()

    if feature_space == "rgb":
        # X in [0,1], shape [H,W,3]
        mean_rgb = X.mean(axis=2, keepdims=True)

        # saturation-like adjustment: move away from gray
        X = mean_rgb + sat_scale * (X - mean_rgb)

        # brightness / value-like scaling
        X = val_scale * X

        # contrast around mid-gray
        X = 0.5 + contrast_scale * (X - 0.5)

        X = np.clip(X, 0.0, 1.0)
        return X

    elif feature_space == "lab":
        # OpenCV Lab stored in [0,1] after your conversion
        # channel 0 = lightness-like
        # channels 1,2 = chroma-like
        L = X[:, :, 0:1]
        AB = X[:, :, 1:3]

        # lightness / brightness
        L = val_scale * L

        # contrast on lightness around middle
        L = 0.5 + contrast_scale * (L - 0.5)

        # chroma strength as saturation analogue
        AB_center = 0.5
        AB = AB_center + sat_scale * (AB - AB_center)

        X[:, :, 0:1] = L
        X[:, :, 1:3] = AB
        X = np.clip(X, 0.0, 1.0)
        return X

    else:
        raise ValueError(f"Unknown feature_space: {feature_space}")

def digitized_curve_roughness(x_out, y_out, resample_n=None, method=None):
    """
    Estimate jaggedness via second finite differences after resampling y(x)
    onto a uniform x-grid.

    Methods
    -------
    - "median_abs_curvature"   : old robust metric
    - "mean_squared_curvature" : penalizes large spikes more strongly
    """
    if resample_n is None:
        resample_n = COLOR_FILTER.get("roughness_resample_n", 200)
    if method is None:
        method = COLOR_FILTER.get("roughness_method", "mean_squared_curvature")

    x = np.asarray(x_out, dtype=np.float64)
    y = np.asarray(y_out, dtype=np.float64)

    if x.size < 5:
        return float("inf")

    # require strictly increasing x for interpolation
    uniq_x, uniq_idx = np.unique(x, return_index=True)
    x = x[uniq_idx]
    y = y[uniq_idx]

    if x.size < 5:
        return float("inf")

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return float("inf")

    grid_n = int(min(max(resample_n, 20), max(20, x.size)))
    xg = np.linspace(x_min, x_max, grid_n)
    yg = np.interp(xg, x, y)

    d2 = yg[2:] - 2.0 * yg[1:-1] + yg[:-2]

    if method == "median_abs_curvature":
        return float(np.median(np.abs(d2)))

    if method == "mean_squared_curvature":
        return float(np.mean(d2 * d2))

    raise ValueError(f"Unknown roughness_method: {method}")


def score_digitized_curve(
    x_out,
    y_out,
    xmin,
    xmax,
    min_coverage=0.40,
    min_points=12,
):
    """
    Score a digitized curve candidate.

    Hard rejects:
      - too few points
      - too small x coverage

    Lower score is better.
    """
    x = np.asarray(x_out, dtype=np.float64)
    y = np.asarray(y_out, dtype=np.float64)

    if x.size < min_points:
        return float("inf"), {"coverage": 0.0, "roughness": float("inf"), "n_points": int(x.size)}

    x_span_total = float(xmax - xmin)
    if not np.isfinite(x_span_total) or x_span_total <= 0:
        return float("inf"), {"coverage": 0.0, "roughness": float("inf"), "n_points": int(x.size)}

    coverage = float((np.max(x) - np.min(x)) / x_span_total)
    if coverage < min_coverage:
        return float("inf"), {"coverage": coverage, "roughness": float("inf"), "n_points": int(x.size)}

    roughness = digitized_curve_roughness(x, y)

    # Prefer smooth curves, but mildly reward coverage and usable point count
    score = roughness - 0.02 * coverage - 0.0005 * min(x.size, 500)

    return score, {
        "coverage": coverage,
        "roughness": roughness,
        "n_points": int(x.size),
    }


def assign_one_curve_with_adjustment(
    img_bgr,
    target_center,
    all_centers,
    feature_normalizer,
    inside_mask=None,
    axes_exclude_mask=None,
    legend_exclude_mask=None,
    exclude_grayscale=False,
    lum_min=COLOR_FILTER["lum_min"],
    lum_max=COLOR_FILTER["lum_max"],
    sat_scale=1.0,
    val_scale=1.0,
    contrast_scale=1.0,
    certainty_ratio=0.85,
    abs_dist_thresh=None,
):

    """
    Build a mask for one specific curve color using one-vs-rest competition
    after applying per-curve feature adjustments.
    """
    H, W, _ = img_bgr.shape

    valid = np.ones((H, W), dtype=bool)

    if inside_mask is not None:
        valid &= inside_mask

    if exclude_grayscale:
        valid &= ~grayscale_mask_bgr(img_bgr)

    if axes_exclude_mask is not None:
        valid &= ~axes_exclude_mask

    if legend_exclude_mask is not None:
        valid &= ~legend_exclude_mask

    valid &= luminance_mask_bgr(img_bgr, lum_min=lum_min, lum_max=lum_max)

    ys, xs = np.where(valid)
    mask = np.zeros((H, W), dtype=np.uint8)

    if xs.size == 0:
        return mask

    feat_img_raw = get_feature_image_for_clustering(
        img_bgr,
        feature_space=feature_normalizer.feature_space,
    )

    feat_img_adj_raw = adjust_curve_feature_image(
        feat_img_raw,
        feature_space=feature_normalizer.feature_space,
        sat_scale=sat_scale,
        val_scale=val_scale,
        contrast_scale=contrast_scale,
    )

    feat_img = feature_normalizer.transform(feat_img_adj_raw)
    X = feat_img[ys, xs].astype(np.float64)

    target_center = np.asarray(target_center, dtype=np.float64).reshape(1, -1)
    centers = np.asarray(all_centers, dtype=np.float64)

    d2_all = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    target_d2 = np.sum((X - target_center[0]) ** 2, axis=1)

    # nearest competing center other than target
    # use exact equality of row index outside, so caller should pass the real center row
    eq_rows = np.all(np.isclose(centers, target_center, atol=1e-12), axis=1)
    if not np.any(eq_rows):
        # fallback: nearest full-center assignment with certainty gating
        target_idx = int(np.argmin(np.sum((centers - target_center[0]) ** 2, axis=1)))

        other_d2 = d2_all.copy()
        target_d2 = d2_all[:, target_idx]
        other_d2[:, target_idx] = np.inf
        best_other_d2 = np.min(other_d2, axis=1)

        target_d = np.sqrt(np.maximum(target_d2, 0.0))
        best_other_d = np.sqrt(np.maximum(best_other_d2, 1e-12))

        chosen = target_d2 < best_other_d2

        if certainty_ratio is not None:
            chosen &= (target_d <= certainty_ratio * best_other_d)

        if abs_dist_thresh is not None:
            chosen &= (target_d <= abs_dist_thresh)

        mask[ys[chosen], xs[chosen]] = 255
        return mask

    target_idx = int(np.flatnonzero(eq_rows)[0])

    other_d2 = d2_all.copy()
    other_d2[:, target_idx] = np.inf
    best_other_d2 = np.min(other_d2, axis=1)

    target_d = np.sqrt(np.maximum(target_d2, 0.0))
    best_other_d = np.sqrt(np.maximum(best_other_d2, 1e-12))

    # nearest-center condition
    chosen = target_d2 < best_other_d2

    # certainty condition:
    # target must be sufficiently closer than the next-best alternative
    if certainty_ratio is not None:
        chosen &= (target_d <= certainty_ratio * best_other_d)

    # optional absolute closeness condition
    if abs_dist_thresh is not None:
        chosen &= (target_d <= abs_dist_thresh)

    mask[ys[chosen], xs[chosen]] = 255

    return mask

def evaluate_autotune_candidate(
    sat_scale,
    val_scale,
    contrast_scale,
    certainty_ratio,
    abs_dist_thresh,
    *,
    img_bgr,
    target_center,
    all_centers,
    feature_normalizer,
    origin,
    ax_vec,
    ay_vec,
    ax_n2,
    ay_n2,
    xmin,
    xmax,
    ymin,
    ymax,
    k,
    inside_mask,
    axes_exclude_mask,
    legend_exclude_mask,
    exclude_grayscale,
    xscale_log,
    yscale_log,
    monotone_y,
    monotone_method,
    keep_outside_axes,
):
    curve_mask = assign_one_curve_with_adjustment(
        img_bgr=img_bgr,
        target_center=target_center,
        all_centers=all_centers,
        feature_normalizer=feature_normalizer,
        inside_mask=inside_mask,
        axes_exclude_mask=axes_exclude_mask,
        legend_exclude_mask=legend_exclude_mask,
        exclude_grayscale=exclude_grayscale,
        sat_scale=sat_scale,
        val_scale=val_scale,
        contrast_scale=contrast_scale,
        certainty_ratio=certainty_ratio,
        abs_dist_thresh=abs_dist_thresh,
    )

    if not np.any(curve_mask):
        return None

    x_out, y_out = digitize_curve_mask(
        curve_mask,
        origin, ax_vec, ay_vec, ax_n2, ay_n2,
        xmin, xmax, ymin, ymax, k,
        xscale_log=xscale_log,
        yscale_log=yscale_log,
        monotone_y=monotone_y,
        monotone_method=monotone_method,
        keep_outside_axes=keep_outside_axes,
    )

    if x_out is None or y_out is None:
        return None

    min_preprune_points = max(
        1,
        int(math.ceil(
            COLOR_FILTER.get("autotune_min_preprune_points_frac", 0.25) * k
        ))
    )
    if len(x_out) < min_preprune_points:
        return None

    score, info = score_digitized_curve(
        x_out, y_out,
        xmin=xmin,
        xmax=xmax,
        min_coverage=0.40,
        min_points=12,
    )

    if not np.isfinite(score):
        return None

    max_acceptable_roughness = COLOR_FILTER.get("autotune_max_acceptable_roughness", None)
    if max_acceptable_roughness is not None:
        if info["roughness"] > max_acceptable_roughness:
            return None

    return {
        "score": score,
        "x_out": x_out,
        "y_out": y_out,
        "curve_mask": curve_mask,
        "info": {
            "sat_scale": sat_scale,
            "val_scale": val_scale,
            "contrast_scale": contrast_scale,
            "certainty_ratio": certainty_ratio,
            "abs_dist_thresh": abs_dist_thresh,
            **info,
        },
    }
    
    
def optimize_curve_mask_by_feature_adjustment(
    img_bgr,
    target_center,
    all_centers,
    feature_normalizer,
    origin,
    ax_vec,
    ay_vec,
    ax_n2,
    ay_n2,
    xmin,
    xmax,
    ymin,
    ymax,
    k,
    inside_mask=None,
    axes_exclude_mask=None,
    legend_exclude_mask=None,
    exclude_grayscale=False,
    xscale_log=False,
    yscale_log=False,
    monotone_y=False,
    monotone_method="pava",
    keep_outside_axes=False,
    debug=False,
    sat_grid=None,
    val_grid=None,
    contrast_grid=None,
    certainty_ratio_grid=None,
    abs_dist_grid=None,
    ):
    """
    Search over per-curve feature adjustments and keep the smoothest valid curve.
    """
    if sat_grid is None:
        sat_grid = COLOR_FILTER["autotune_sat_grid_full"]
    if val_grid is None:
        val_grid = COLOR_FILTER["autotune_val_grid_full"]
    if contrast_grid is None:
        contrast_grid = COLOR_FILTER["autotune_contrast_grid_full"]
    if certainty_ratio_grid is None:
        certainty_ratio_grid = COLOR_FILTER["autotune_certainty_ratio_grid_full"]
    if abs_dist_grid is None:
        abs_dist_grid = COLOR_FILTER["autotune_abs_dist_grid"]
####################################
    
    

    param_grid = list(product(
        sat_grid,
        val_grid,
        contrast_grid,
        certainty_ratio_grid,
        abs_dist_grid,
    ))

    results = []
    for sat_scale, val_scale, contrast_scale, certainty_ratio, abs_dist_thresh in param_grid:
        r = evaluate_autotune_candidate(
            sat_scale,
            val_scale,
            contrast_scale,
            certainty_ratio,
            abs_dist_thresh,
            img_bgr=img_bgr,
            target_center=target_center,
            all_centers=all_centers,
            feature_normalizer=feature_normalizer,
            origin=origin,
            ax_vec=ax_vec,
            ay_vec=ay_vec,
            ax_n2=ax_n2,
            ay_n2=ay_n2,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            k=k,
            inside_mask=inside_mask,
            axes_exclude_mask=axes_exclude_mask,
            legend_exclude_mask=legend_exclude_mask,
            exclude_grayscale=exclude_grayscale,
            xscale_log=xscale_log,
            yscale_log=yscale_log,
            monotone_y=monotone_y,
            monotone_method=monotone_method,
            keep_outside_axes=keep_outside_axes,
        )
        if r is not None:
            results.append(r)



    results = [r for r in results if r is not None]
    if not results:
        return None

    best_result = max(
        results,
        key=lambda r: (
            r["info"]["n_points"],   # first: keep the fullest curve
            -r["info"]["roughness"], # second: among equal sizes, prefer smoother
            r["info"]["coverage"],   # third: prefer larger x-span
        )
    )
    x_out = best_result["x_out"]
    y_out = best_result["y_out"]
    curve_mask = best_result["curve_mask"]
    best_info = best_result["info"]
    
####################################
    if COLOR_FILTER.get("spike_prune_enable", True):
        x_out, y_out = remove_isolated_y_spikes(
            x_out,
            y_out,
            spike_factor=COLOR_FILTER["one_point_spike_factor"],
            neighbor_pair_factor=COLOR_FILTER["one_point_neighbor_pair_factor"],
            max_passes=COLOR_FILTER["one_point_max_passes"],
        )

        x_out, y_out = remove_two_point_y_spikes(
            x_out,
            y_out,
            spike_factor=COLOR_FILTER["two_point_spike_factor"],
            pair_tight_factor=COLOR_FILTER["two_point_pair_tight_factor"],
            baseline_tight_factor=COLOR_FILTER["two_point_baseline_tight_factor"],
            max_passes=COLOR_FILTER["two_point_max_passes"],
        )
        
        
    if COLOR_FILTER.get("rolling_prune_enable", True):
        x_out_pruned, y_out_pruned = remove_local_outlier_points_rolling(
            x_out,
            y_out,
            max_remove_frac=COLOR_FILTER["rolling_prune_max_remove_frac"],
            window=COLOR_FILTER["rolling_prune_window"],
            sigma_thresh=COLOR_FILTER["rolling_prune_sigma_thresh"],
            max_iters=COLOR_FILTER["rolling_prune_max_iters"],
            protect_endpoints=COLOR_FILTER["rolling_prune_protect_endpoints"],
        )
        x_out, y_out = x_out_pruned, y_out_pruned
    
    
    if debug and best_info is not None:
        print(
            "[digitizer] best curve adjustment:"
            f" sat={best_info['sat_scale']}"
            f" val={best_info['val_scale']}"
            f" contrast={best_info['contrast_scale']}"
            f" certainty_ratio={best_info['certainty_ratio']}"
            f" abs_dist_thresh={best_info['abs_dist_thresh']}"
            f" coverage={best_info['coverage']:.3f}"
            f" roughness={best_info['roughness']:.6g}"
            f" points={best_info['n_points']}"
            f" final_points={len(x_out)}",
            file=sys.stderr
        )
    
    return x_out, y_out, curve_mask
    
    
def evaluate_autotune_candidate_for_curve(
    curve_idx,
    target_center,
    sat_scale,
    val_scale,
    contrast_scale,
    certainty_ratio,
    abs_dist_thresh,
    *,
    img_bgr,
    all_centers,
    feature_normalizer,
    origin,
    ax_vec,
    ay_vec,
    ax_n2,
    ay_n2,
    xmin,
    xmax,
    ymin,
    ymax,
    k,
    inside_mask,
    axes_exclude_mask,
    legend_exclude_mask,
    exclude_grayscale,
    xscale_log,
    yscale_log,
    monotone_y,
    monotone_method,
    keep_outside_axes,
):
    r = evaluate_autotune_candidate(
        sat_scale,
        val_scale,
        contrast_scale,
        certainty_ratio,
        abs_dist_thresh,
        img_bgr=img_bgr,
        target_center=target_center,
        all_centers=all_centers,
        feature_normalizer=feature_normalizer,
        origin=origin,
        ax_vec=ax_vec,
        ay_vec=ay_vec,
        ax_n2=ax_n2,
        ay_n2=ay_n2,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        k=k,
        inside_mask=inside_mask,
        axes_exclude_mask=axes_exclude_mask,
        legend_exclude_mask=legend_exclude_mask,
        exclude_grayscale=exclude_grayscale,
        xscale_log=xscale_log,
        yscale_log=yscale_log,
        monotone_y=monotone_y,
        monotone_method=monotone_method,
        keep_outside_axes=keep_outside_axes,
    )
    return curve_idx, r

def pick_best_result_per_curve(candidate_results):
    results_by_idx = {}

    for curve_idx, r in candidate_results:
        if r is None:
            continue

        old = results_by_idx.get(curve_idx)
        if old is None:
            results_by_idx[curve_idx] = r
            continue

        old_key = (
            old["info"]["n_points"],
            -old["info"]["roughness"],
            old["info"]["coverage"],
        )
        new_key = (
            r["info"]["n_points"],
            -r["info"]["roughness"],
            r["info"]["coverage"],
        )

        if new_key > old_key:
            results_by_idx[curve_idx] = r

    return results_by_idx
  
def local_roughness_contributions(x, y):
    """
    Per-point local roughness score.

    For interior points:
        contribution[k] = vertical deviation from the straight line
        joining points k-1 and k+1, evaluated at x[k].

    Endpoints get 0 because they do not have two-sided support.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)
    contrib = np.zeros(n, dtype=np.float64)

    if n < 3:
        return contrib

    for k in range(1, n - 1):
        x0, y0 = x[k - 1], y[k - 1]
        x1, y1 = x[k + 1], y[k + 1]
        xk, yk = x[k], y[k]

        dx = x1 - x0
        if abs(dx) <= 1e-12:
            # fallback if neighbors have same x
            y_line = 0.5 * (y0 + y1)
        else:
            t = (xk - x0) / dx
            y_line = y0 + t * (y1 - y0)

        contrib[k] = abs(yk - y_line)

    return contrib

def point_to_curve_vertical_distance(xp, yp, x_curve, y_curve):
    """
    Vertical distance from point (xp, yp) to another curve, using interpolation.

    Returns inf if xp lies outside the x-range of the other curve or the curve
    has fewer than 2 points.
    """
    x_curve = np.asarray(x_curve, dtype=np.float64)
    y_curve = np.asarray(y_curve, dtype=np.float64)

    if len(x_curve) < 2:
        return float("inf")

    xmin = np.min(x_curve)
    xmax = np.max(x_curve)
    if xp < xmin or xp > xmax:
        return float("inf")

    y_other = np.interp(xp, x_curve, y_curve)
    return abs(yp - y_other)

def reassign_rough_points_between_curves(
    curves_xy,
    frac=0.05,
    margin=1.0,
    max_passes=3,
):
    """
    For each curve:
      - score each point by local roughness contribution
      - take the worst frac of points
      - if a point is closer to another curve than to its own local neighbors,
        move it to that other curve

    No lower limit on number of points.

    Parameters
    ----------
    frac:
        Fraction of worst-contributing points to test on each curve.
    margin:
        Reassign when other_err < margin * own_err.
        Use 1.0 for the exact rule you described.
        Use <1.0 to make reassignment stricter.
    max_passes:
        Number of global cleanup passes.
    """
    

    
    curves = [
        (
            np.asarray(x, dtype=np.float64).copy(),
            np.asarray(y, dtype=np.float64).copy(),
        )
        for (x, y) in curves_xy
    ]
    

    
    
    if len(curves) < 2:
        return curves

    for _ in range(max_passes):
        moved_any = False

        for i in range(len(curves)):
            xi, yi = curves[i]
            
            order = np.argsort(xi)
            xi = xi[order]
            yi = yi[order]
            curves[i] = (xi, yi)
    
            n = len(xi)
            if n == 0:
                continue

            contrib = local_roughness_contributions(xi, yi)

            
            candidate_pool = np.arange(1, n - 1, dtype=np.int64)
            if candidate_pool.size == 0:
                continue
            
            m = max(1, int(math.ceil(frac * candidate_pool.size)))
            candidate_idx = candidate_pool[np.argsort(contrib[candidate_pool])[-m:]]
            

            # batch edits so indices do not shift while processing
            remove_mask = np.zeros(n, dtype=bool)
            points_to_add = {j: [] for j in range(len(curves)) if j != i}

            for k in candidate_idx:
                xk = xi[k]
                yk = yi[k]
                own_err = contrib[k]

                # cannot judge local fit for endpoints or tiny curves
                if not np.isfinite(own_err) or own_err <= 0:
                    continue

                best_j = None
                best_other_err = float("inf")

                for j in range(len(curves)):
                    if j == i:
                        continue

                    xj, yj = curves[j]
                    other_err = point_to_curve_vertical_distance(xk, yk, xj, yj)

                    if other_err < best_other_err:
                        best_other_err = other_err
                        best_j = j

                if best_j is None or not np.isfinite(best_other_err):
                    continue

                if best_other_err < margin * own_err:
                    remove_mask[k] = True
                    points_to_add[best_j].append((xk, yk))

            if np.any(remove_mask):
                xi2 = xi[~remove_mask]
                yi2 = yi[~remove_mask]
                curves[i] = (xi2, yi2)
                moved_any = True

            for j, pts in points_to_add.items():
                if not pts:
                    continue

                xj, yj = curves[j]
                add_x = np.array([p[0] for p in pts], dtype=np.float64)
                add_y = np.array([p[1] for p in pts], dtype=np.float64)

                xj2 = np.concatenate([xj, add_x])
                yj2 = np.concatenate([yj, add_y])

                order = np.argsort(xj2)
                xj2 = xj2[order]
                yj2 = yj2[order]

                curves[j] = (xj2, yj2)
                moved_any = True

        if not moved_any:
            break

    return curves

# --------------------------- Main logic & entrypoints ---------------------------

def main_logic(*, image, xmin, xmax, ymin, ymax, k=100, min_sat=30, min_val=30,
               knn_k=3, knn_thresh=35.0, out="", debug=False, debug_images=False,
               monotone_y=False, monotone_method="pava",
               auto_colors=False, exclude_grayscale=False,
               num_colors=0,
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

        
        cluster_masks, centers_rgb, centers_feat, feature_normalizer, chosen_k, model_scores = discover_color_clusters(
            img,
            exclude_grayscale=exclude_grayscale,
            num_colors=num_colors,
            inside_mask=inside_axes_mask,
            axes_exclude_mask=axes_exclude_mask_clustering,
            save_debug_images=debug_images,
        )
        
        cluster_masks = assign_pixels_to_known_centers(
            img,
            centers_feat,
            feature_normalizer,
            inside_mask=inside_axes_mask,
            axes_exclude_mask=axes_exclude_mask_curve,
            exclude_grayscale=exclude_grayscale,
        )
        
        
        
        #remove legend
        # 2b) combined legend detection on all groups together
        plot_w_px = float(np.sqrt(ax_n2))
        plot_h_px = float(np.sqrt(ay_n2))

        union_mask = make_union_mask(cluster_masks)

        legend_box = None
        legend_exclude_mask = np.zeros(img.shape[:2], dtype=bool)
        
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

        legend_exclude_mask = legend_box_to_mask(img.shape, legend_box)

        if legend_box is not None:
            cluster_masks = remove_legend_box_from_masks(cluster_masks, legend_box)

            if debug_images:
                base = img_path.rsplit(".", 1)[0]
                save_legend_box_debug_png(img, legend_box, f"{base}_debug_legend_box.png")
            
##################################
        # 2c) geometric snap-back cleanup to reduce jagged line swapping
        if COLOR_FILTER.get("curve_snap_enable", True):

                        
            cluster_masks, ref_curve_mask = snap_all_curve_masks_to_reference(
                cluster_masks,
                dilate_radius=COLOR_FILTER["curve_snap_dilate_radius"],
                search_radius=COLOR_FILTER["curve_snap_search_radius"],
            )
            

            if debug_images:
                base = img_path.rsplit(".", 1)[0]
                save_reference_curve_set_debug_png(
                    img,
                    ref_curve_mask,
                    f"{base}_debug_reference_curve_set.png"
                )
##################################            
        
        print(f"[digitizer] auto-colors chose K={chosen_k}", file=sys.stderr)
        
        
        
        # 3) digitize every discovered color cluster, with per-curve feature adjustment search
        
        
####################################

        all_results = []
        base_out = out
        
        fast_sat_grid = COLOR_FILTER["autotune_sat_grid_fast"]
        fast_val_grid = COLOR_FILTER["autotune_val_grid_fast"]
        fast_contrast_grid = COLOR_FILTER["autotune_contrast_grid_fast"]
        fast_certainty_ratio_grid = COLOR_FILTER["autotune_certainty_ratio_grid_fast"]
        
        full_sat_grid = COLOR_FILTER["autotune_sat_grid_full"]
        full_val_grid = COLOR_FILTER["autotune_val_grid_full"]
        full_contrast_grid = COLOR_FILTER["autotune_contrast_grid_full"]
        full_certainty_ratio_grid = COLOR_FILTER["autotune_certainty_ratio_grid_full"]
        
        abs_dist_grid = COLOR_FILTER["autotune_abs_dist_grid"]
        
        # pass 1: fast search for all curves
        fast_param_grid = list(product(
            fast_sat_grid,
            fast_val_grid,
            fast_contrast_grid,
            fast_certainty_ratio_grid,
            abs_dist_grid,
        ))
        
        pass1_jobs = [
            (i, centers_feat[i - 1], sat_scale, val_scale, contrast_scale, certainty_ratio, abs_dist_thresh)
            for i in range(1, len(centers_feat) + 1)
            for sat_scale, val_scale, contrast_scale, certainty_ratio, abs_dist_thresh in fast_param_grid
        ]
        
        pass1_results = Parallel(
            n_jobs=COLOR_FILTER["autotune_parallel_n_jobs"],
            backend=COLOR_FILTER["autotune_parallel_backend"],
            batch_size=COLOR_FILTER["autotune_parallel_batch_size"],
            prefer="processes",
        )(
            delayed(evaluate_autotune_candidate_for_curve)(
                curve_idx,
                target_center,
                sat_scale,
                val_scale,
                contrast_scale,
                certainty_ratio,
                abs_dist_thresh,
                img_bgr=img,
                all_centers=centers_feat,
                feature_normalizer=feature_normalizer,
                origin=origin,
                ax_vec=ax_vec,
                ay_vec=ay_vec,
                ax_n2=ax_n2,
                ay_n2=ay_n2,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                k=k,
                inside_mask=inside_axes_mask,
                axes_exclude_mask=axes_exclude_mask_curve,
                legend_exclude_mask=legend_exclude_mask,
                exclude_grayscale=exclude_grayscale,
                xscale_log=xscale_log,
                yscale_log=yscale_log,
                monotone_y=monotone_y,
                monotone_method=monotone_method,
                keep_outside_axes=keep_outside_axes,
            )
            for curve_idx, target_center, sat_scale, val_scale, contrast_scale, certainty_ratio, abs_dist_thresh in pass1_jobs
        )
        
        results_by_idx = pick_best_result_per_curve(pass1_results)
        failed = [i for i in range(1, len(centers_feat) + 1) if i not in results_by_idx]
        
        # pass 2: full search only for failed curves
        if failed:
            if debug:
                print(f"[digitizer] full autotune retry for failed clusters: {failed}", file=sys.stderr)
        


            full_param_grid = list(product(
                full_sat_grid,
                full_val_grid,
                full_contrast_grid,
                full_certainty_ratio_grid,
                abs_dist_grid,
            ))
            
            pass2_jobs = [
                (i, centers_feat[i - 1], sat_scale, val_scale, contrast_scale, certainty_ratio, abs_dist_thresh)
                for i in failed
                for sat_scale, val_scale, contrast_scale, certainty_ratio, abs_dist_thresh in full_param_grid
            ]
            
            pass2_results = Parallel(
                n_jobs=COLOR_FILTER["autotune_parallel_n_jobs"],
                backend=COLOR_FILTER["autotune_parallel_backend"],
                batch_size=COLOR_FILTER["autotune_parallel_batch_size"],
                prefer="processes",
            )(
                delayed(evaluate_autotune_candidate_for_curve)(
                    curve_idx,
                    target_center,
                    sat_scale,
                    val_scale,
                    contrast_scale,
                    certainty_ratio,
                    abs_dist_thresh,
                    img_bgr=img,
                    all_centers=centers_feat,
                    feature_normalizer=feature_normalizer,
                    origin=origin,
                    ax_vec=ax_vec,
                    ay_vec=ay_vec,
                    ax_n2=ax_n2,
                    ay_n2=ay_n2,
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    k=k,
                    inside_mask=inside_axes_mask,
                    axes_exclude_mask=axes_exclude_mask_curve,
                    legend_exclude_mask=legend_exclude_mask,
                    exclude_grayscale=exclude_grayscale,
                    xscale_log=xscale_log,
                    yscale_log=yscale_log,
                    monotone_y=monotone_y,
                    monotone_method=monotone_method,
                    keep_outside_axes=keep_outside_axes,
                )
                for curve_idx, target_center, sat_scale, val_scale, contrast_scale, certainty_ratio, abs_dist_thresh in pass2_jobs
            )
            
            pass2_best = pick_best_result_per_curve(pass2_results)
            
            for curve_idx in failed:
                best_result = pass2_best.get(curve_idx)
                if best_result is None:
                    print(f"[digitizer] cluster {curve_idx}: no valid candidate after full adjustment search; skipped", file=sys.stderr)
                else:
                    results_by_idx[curve_idx] = best_result


        
        # collect in original cluster order
        for i in range(1, len(centers_feat) + 1):
            best_result = results_by_idx.get(i)
            if best_result is None:
                continue
            
            x_out = best_result["x_out"]
            y_out = best_result["y_out"]
            curve_mask = best_result["curve_mask"]
            
            if debug:
                best_info = best_result["info"]
                print(
                    f"[digitizer] cluster {i}:"
                    f" sat={best_info['sat_scale']}"
                    f" val={best_info['val_scale']}"
                    f" contrast={best_info['contrast_scale']}"
                    f" certainty_ratio={best_info['certainty_ratio']}"
                    f" abs_dist_thresh={best_info['abs_dist_thresh']}"
                    f" coverage={best_info['coverage']:.3f}"
                    f" roughness={best_info['roughness']:.6g}"
                    f" points={best_info['n_points']}",
                    file=sys.stderr
                )
                
            all_results.append((x_out, y_out, curve_mask))            



####################################

            
        curves_xy = [(x_out, y_out) for (x_out, y_out, _) in all_results]

        if COLOR_FILTER.get("extend_tail_from_overlapping_curve", False):
            curves_xy = extend_curve_tails_from_overlaps(
                curves_xy,
                lookback_points=COLOR_FILTER.get("tail_overlap_lookback_points", 10),
                dist_tol=COLOR_FILTER.get("tail_overlap_distance_tol", 2.5),
            )

        curves_xy = reassign_rough_points_between_curves(
            curves_xy,
            frac=0.05,
            margin=0.8,
            max_passes=3,
        )
        all_results = [
            (x_out, y_out, curve_mask)
            for (x_out, y_out), (_, _, curve_mask) in zip(curves_xy, all_results)
        ]
        
        curves_xy = add_synthetic_endpoint_points(
            curves_xy,
            xmin=xmin,
            xmax=xmax,
            xscale_log=xscale_log,
            yscale_log=yscale_log,
        )
        
        all_results = [
            (x_out, y_out, curve_mask)
            for (x_out, y_out), (_, _, curve_mask) in zip(curves_xy, all_results)
        ]







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

        if debug_images:
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
        #bl_px, br_px, tl_px, red_mask = find_red_centroids_strict(img)
        bl_px, br_px, tl_px, red_mask = detect_axes_red_points(
            img, hsv, min_sat=min_sat, min_val=min_val, knn_k=knn_k, knn_thresh=knn_thresh
        )
        
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
    
    if COLOR_FILTER.get("spike_prune_enable", True):
        x_out, y_out = remove_isolated_y_spikes(
            x_out,
            y_out,
            spike_factor=COLOR_FILTER["one_point_spike_factor"],
            neighbor_pair_factor=COLOR_FILTER["one_point_neighbor_pair_factor"],
            max_passes=COLOR_FILTER["one_point_max_passes"],
        )
    
        x_out, y_out = remove_two_point_y_spikes(
            x_out,
            y_out,
            spike_factor=COLOR_FILTER["two_point_spike_factor"],
            pair_tight_factor=COLOR_FILTER["two_point_pair_tight_factor"],
            baseline_tight_factor=COLOR_FILTER["two_point_baseline_tight_factor"],
            max_passes=COLOR_FILTER["two_point_max_passes"],
        )    
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

    if debug_images:
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
    #gray_abs=2.0,
    #gray_rel=0.05,
    num_colors=5,
    #reject_marker_red=True,
    #marker_red_tol=4.0,
    monotone_y=False,
    monotone_method="pava",
    out=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3fig1_a.csv",
    debug=True,
        ),
    ]
    
    jobs.append(make_job_like( jobs[0], image=r"/home/romanz/code/RomanZhvanskiy/graph_digitizer/ref3figl_a.png",  xmin=0, xmax=200, ymin=0, ymax=140, ))
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


    

    for j in jobs[1:]:
        print(f"[digitizer] processing {j['image']} → {j.get('out', '(stdout)')}", file=sys.stderr)
        main_logic(**j)


if __name__ == "__main__":
    main_cli()
