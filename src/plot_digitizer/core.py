"""Public API for plot digitization.

This module wraps the original single-file digitizer engine and exposes a
small, typed interface that can be used by a GUI, CLI, notebooks, or tests.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from . import _legacy


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class CalibrationPoint:
    """One reference point connecting image pixels to plot data coordinates."""

    pixel: Point2D
    data: Point2D


@dataclass(frozen=True)
class AxisCalibration:
    """Three-point axis calibration.

    The expected roles match the common plot-corner workflow:

    - origin: bottom-left plot corner
    - x_axis: bottom-right plot corner
    - y_axis: top-left plot corner

    For log axes, provide the real data coordinates here; interpolation is
    handled by the underlying engine when xscale_log or yscale_log is set.
    """

    origin: CalibrationPoint
    x_axis: CalibrationPoint
    y_axis: CalibrationPoint

    def as_legacy_ranges(self) -> Tuple[float, float, float, float]:
        """Return xmin, xmax, ymin, ymax for the legacy engine."""
        xmin = float(self.origin.data[0])
        xmax = float(self.x_axis.data[0])
        ymin = float(self.origin.data[1])
        ymax = float(self.y_axis.data[1])
        return xmin, xmax, ymin, ymax

    def validation_warnings(self, tolerance: float = 1e-9) -> List[str]:
        """Validate that the calibration matches rectangular plot axes."""
        warnings: List[str] = []

        if abs(float(self.x_axis.data[1]) - float(self.origin.data[1])) > tolerance:
            warnings.append(
                "x-axis reference y coordinate differs from origin y; "
                "only origin.y is used as ymin."
            )

        if abs(float(self.y_axis.data[0]) - float(self.origin.data[0])) > tolerance:
            warnings.append(
                "y-axis reference x coordinate differs from origin x; "
                "only origin.x is used as xmin."
            )

        return warnings


@dataclass
class DigitizeOptions:
    """Digitization settings shared by CLI and GUI."""

    k: int = 100
    min_sat: int = 30
    min_val: int = 30
    knn_k: int = 3
    knn_thresh: float = 35.0
    monotone_y: bool = False
    monotone_method: str = "pava"
    auto_colors: bool = False
    exclude_grayscale: bool = True
    num_colors: int = 0
    xscale_log: bool = False
    yscale_log: bool = False
    keep_outside_axes: bool = False
    debug: bool = False
    debug_images: bool = False


@dataclass
class DigitizedCurve:
    """One digitized curve."""

    x: np.ndarray
    y: np.ndarray
    name: str = "curve1"


@dataclass
class DigitizeResult:
    """Result container returned by digitize."""

    curves: List[DigitizedCurve]
    warnings: List[str]

    def to_dataframe(self):
        """Return a pandas DataFrame with curve columns.

        Single-curve results use x/y.
        Multi-curve results use curve1_x, curve1_y, etc.
        """
        import pandas as pd

        if len(self.curves) == 1:
            return pd.DataFrame(
                {
                    "x": self.curves[0].x,
                    "y": self.curves[0].y,
                }
            )

        max_len = max((len(curve.x) for curve in self.curves), default=0)
        data = {}

        for idx, curve in enumerate(self.curves, start=1):
            x_col = np.full(max_len, np.nan)
            y_col = np.full(max_len, np.nan)

            x_col[: len(curve.x)] = curve.x
            y_col[: len(curve.y)] = curve.y

            data[f"curve{idx}_x"] = x_col
            data[f"curve{idx}_y"] = y_col

        return pd.DataFrame(data)

    def to_csv(self, path: Union[str, Path]) -> Path:
        """Write the result to CSV and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path


def _red_mask_with_manual_points(
    img_bgr: np.ndarray,
    points: Sequence[Point2D],
) -> np.ndarray:
    """Build a legacy-style red mask, including manual calibration points."""
    red_mask = _legacy.strict_red_mask_bgr(img_bgr).astype(np.uint8) * 255

    for x, y in points:
        cv2.circle(
            red_mask,
            (int(round(x)), int(round(y))),
            5,
            255,
            -1,
        )

    return red_mask


@contextmanager
def _override_axis_detection(calibration: Optional[AxisCalibration]):
    """Temporarily replace legacy red-marker detection with manual points."""
    if calibration is None:
        yield
        return

    original_detect_axes_red_points = _legacy.detect_axes_red_points

    def detect_from_calibration(
        img_bgr,
        hsv=None,
        min_sat=30,
        min_val=30,
        knn_k=3,
        knn_thresh=35.0,
        corner_min_pixels=4,
    ):
        origin_px = np.asarray(calibration.origin.pixel, dtype=np.float64)
        x_axis_px = np.asarray(calibration.x_axis.pixel, dtype=np.float64)
        y_axis_px = np.asarray(calibration.y_axis.pixel, dtype=np.float64)

        red_mask = _red_mask_with_manual_points(
            img_bgr,
            [origin_px, x_axis_px, y_axis_px],
        )

        # Legacy naming: bl, br, tl.
        return origin_px, x_axis_px, y_axis_px, red_mask

    _legacy.detect_axes_red_points = detect_from_calibration

    try:
        yield
    finally:
        _legacy.detect_axes_red_points = original_detect_axes_red_points


def _normalise_legacy_output(raw) -> List[DigitizedCurve]:
    """Convert legacy return shapes into a list of DigitizedCurve objects."""
    curves: List[DigitizedCurve] = []

    # Auto-colors path returns [(x, y, mask), ...]
    if isinstance(raw, list):
        for idx, item in enumerate(raw, start=1):
            if item is None:
                continue

            x, y = item[0], item[1]

            if x is None or y is None:
                continue

            curves.append(
                DigitizedCurve(
                    x=np.asarray(x, dtype=float),
                    y=np.asarray(y, dtype=float),
                    name=f"curve{idx}",
                )
            )

        return curves

    # Single-color path returns (x_out, y_out)
    if isinstance(raw, tuple) and len(raw) >= 2:
        x, y = raw[0], raw[1]

        if x is not None and y is not None:
            curves.append(
                DigitizedCurve(
                    x=np.asarray(x, dtype=float),
                    y=np.asarray(y, dtype=float),
                    name="curve1",
                )
            )

    return curves


def digitize(
    image: Union[str, Path],
    calibration: Optional[AxisCalibration] = None,
    *,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    options: Optional[DigitizeOptions] = None,
) -> DigitizeResult:
    """Digitize curves from an image.

    Parameters
    ----------
    image:
        Path to a PNG/JPEG/etc. image readable by OpenCV.
    calibration:
        Optional manual three-point calibration. When omitted, the original
        strict-red corner scanner is used.
    xmin, xmax, ymin, ymax:
        Axis limits used when calibration is omitted. Ignored when calibration
        is supplied.
    options:
        Algorithm options.
    """
    options = options or DigitizeOptions()
    warnings: List[str] = []

    if calibration is not None:
        xmin, xmax, ymin, ymax = calibration.as_legacy_ranges()
        warnings.extend(calibration.validation_warnings())
    else:
        missing = [
            name
            for name, value in {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
            }.items()
            if value is None
        ]

        if missing:
            raise ValueError(
                "Missing axis limits without manual calibration: "
                + ", ".join(missing)
            )

    with _override_axis_detection(calibration):
        raw = _legacy.main_logic(
            image=str(image),
            xmin=float(xmin),
            xmax=float(xmax),
            ymin=float(ymin),
            ymax=float(ymax),
            k=options.k,
            min_sat=options.min_sat,
            min_val=options.min_val,
            knn_k=options.knn_k,
            knn_thresh=options.knn_thresh,
            out="",
            debug=options.debug,
            debug_images=options.debug_images,
            monotone_y=options.monotone_y,
            monotone_method=options.monotone_method,
            auto_colors=options.auto_colors,
            exclude_grayscale=options.exclude_grayscale,
            num_colors=options.num_colors,
            xscale_log=options.xscale_log,
            yscale_log=options.yscale_log,
            keep_outside_axes=options.keep_outside_axes,
        )

    curves = _normalise_legacy_output(raw)

    if not curves:
        raise RuntimeError(
            "No digitized curve points were returned. "
            "Try changing color settings or calibration points."
        )

    return DigitizeResult(curves=curves, warnings=warnings)


def detect_strict_red_marker_centers(
    image_path: Union[str, Path],
    *,
    min_area: int = 20,
) -> List[Point2D]:
    """Detect exactly three strict-red marker blobs and return their centres.

    Uses the same strict-red threshold as the legacy backend:

        R >= 240, G <= 10, B <= 10

    Returns centres as (x, y) pixel coordinates.
    """
    image_path = Path(image_path)

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]

    red = ((r >= 240) & (g <= 10) & (b <= 10)).astype(np.uint8)

    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        red,
        connectivity=8,
    )

    components: List[Tuple[int, float, float]] = []

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= min_area:
            cx, cy = centroids[label]
            components.append((area, float(cx), float(cy)))

    if len(components) != 3:
        raise ValueError(
            f"Expected exactly 3 strict-red marker blobs, found {len(components)}. "
            "Select exactly three corners in the GUI. "
            "If you see this despite selecting three points, increase marker size "
            "or check that original red pixels were removed."
        )

    components.sort(reverse=True, key=lambda item: item[0])

    return [(cx, cy) for _area, cx, cy in components]


def _assign_detected_points_to_plot_corners(
    points: Sequence[Point2D],
) -> dict[str, Point2D]:
    """Assign three detected pixel points to TL/TR/BL/BR.

    Assumes the plot is axis-aligned or near-axis-aligned in image space.
    Missing fourth corner is inferred by parallelogram geometry.

    Returns all four corners:
        tl, tr, bl, br
    """
    if len(points) != 3:
        raise ValueError(f"Expected exactly 3 points, got {len(points)}.")

    pts = np.asarray(points, dtype=float)

    min_x = float(np.min(pts[:, 0]))
    max_x = float(np.max(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_y = float(np.max(pts[:, 1]))

    ideal = {
        "tl": np.array([min_x, min_y], dtype=float),
        "tr": np.array([max_x, min_y], dtype=float),
        "bl": np.array([min_x, max_y], dtype=float),
        "br": np.array([max_x, max_y], dtype=float),
    }

    assigned: dict[str, Point2D] = {}
    used_indices: set[int] = set()

    candidate_pairs: List[Tuple[float, str, int]] = []

    for corner_name, corner_xy in ideal.items():
        for idx, pt in enumerate(pts):
            dist = float(np.linalg.norm(pt - corner_xy))
            candidate_pairs.append((dist, corner_name, idx))

    candidate_pairs.sort(key=lambda item: item[0])

    for _dist, corner_name, idx in candidate_pairs:
        if corner_name in assigned or idx in used_indices:
            continue

        assigned[corner_name] = (
            float(pts[idx, 0]),
            float(pts[idx, 1]),
        )
        used_indices.add(idx)

        if len(assigned) == 3:
            break

    missing = {"tl", "tr", "bl", "br"} - set(assigned)

    if len(missing) != 1:
        raise ValueError(
            "Could not uniquely assign selected points to plot corners. "
            "Try selecting three well-separated corners."
        )

    missing_corner = next(iter(missing))

    arr = {
        key: np.asarray(value, dtype=float)
        for key, value in assigned.items()
    }

    if missing_corner == "tl":
        arr["tl"] = arr["bl"] + arr["tr"] - arr["br"]
    elif missing_corner == "tr":
        arr["tr"] = arr["tl"] + arr["br"] - arr["bl"]
    elif missing_corner == "bl":
        arr["bl"] = arr["tl"] + arr["br"] - arr["tr"]
    elif missing_corner == "br":
        arr["br"] = arr["bl"] + arr["tr"] - arr["tl"]

    out: dict[str, Point2D] = {
        key: (float(value[0]), float(value[1]))
        for key, value in arr.items()
    }

    for key in ["tl", "tr", "bl", "br"]:
        if key not in out:
            raise ValueError(f"Could not infer plot corner {key}.")

        if not all(np.isfinite(value) for value in out[key]):
            raise ValueError(f"Could not infer finite coordinates for plot corner {key}.")

    return out


def calibration_from_marked_corners(
    image_path: Union[str, Path],
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> AxisCalibration:
    """Build AxisCalibration from three strict-red corner markers.

    The marked image should contain exactly three red marker blobs.
    Original red pixels should already have been removed by the frontend.
    """
    centers = detect_strict_red_marker_centers(image_path)
    corners = _assign_detected_points_to_plot_corners(centers)

    return AxisCalibration(
        origin=CalibrationPoint(
            pixel=corners["bl"],
            data=(float(xmin), float(ymin)),
        ),
        x_axis=CalibrationPoint(
            pixel=corners["br"],
            data=(float(xmax), float(ymin)),
        ),
        y_axis=CalibrationPoint(
            pixel=corners["tl"],
            data=(float(xmin), float(ymax)),
        ),
    )


def digitize_marked_image(
    image_path: Union[str, Path],
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    options: Optional[DigitizeOptions] = None,
) -> DigitizeResult:
    """Digitize an image where the GUI has drawn three strict-red corner markers."""
    calibration = calibration_from_marked_corners(
        image_path,
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
    )

    return digitize(
        image_path,
        calibration=calibration,
        options=options,
    )