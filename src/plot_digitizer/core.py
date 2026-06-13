"""Public API for plot digitization.

This module wraps the original single-file digitizer engine and exposes a
small, typed interface that can be used by a GUI, CLI, notebooks, or tests.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

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
    * ``origin`` is usually the bottom-left plot corner.
    * ``x_axis`` is a point along the x-axis, usually bottom-right.
    * ``y_axis`` is a point along the y-axis, usually top-left.

    For log axes, provide the real data coordinates here; interpolation is
    handled by the underlying engine when ``xscale_log`` or ``yscale_log`` is set.
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
            warnings.append("x-axis reference y coordinate differs from origin y; only origin.y is used as ymin.")
        if abs(float(self.y_axis.data[0]) - float(self.origin.data[0])) > tolerance:
            warnings.append("y-axis reference x coordinate differs from origin x; only origin.x is used as xmin.")
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
    """Result container returned by :func:`digitize`."""

    curves: List[DigitizedCurve]
    warnings: List[str]

    def to_dataframe(self):
        """Return a pandas DataFrame with curve columns.

        Single-curve results use ``x``/``y``. Multi-curve results use
        ``curve1_x``, ``curve1_y``, etc.
        """
        import pandas as pd

        if len(self.curves) == 1:
            return pd.DataFrame({"x": self.curves[0].x, "y": self.curves[0].y})

        max_len = max((len(c.x) for c in self.curves), default=0)
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


def _red_mask_with_manual_points(img_bgr: np.ndarray, points: Sequence[Point2D]) -> np.ndarray:
    red_mask = _legacy.strict_red_mask_bgr(img_bgr).astype(np.uint8) * 255
    for x, y in points:
        cv2.circle(red_mask, (int(round(x)), int(round(y))), 5, 255, -1)
    return red_mask


@contextmanager
def _override_axis_detection(calibration: Optional[AxisCalibration]):
    """Temporarily replace legacy red-marker detection with manual points."""
    if calibration is None:
        yield
        return

    original = _legacy.detect_axes_red_points

    def detect_from_calibration(img_bgr, hsv=None, min_sat=30, min_val=30, knn_k=3, knn_thresh=35.0, corner_min_pixels=4):
        origin_px = np.asarray(calibration.origin.pixel, dtype=np.float64)
        x_axis_px = np.asarray(calibration.x_axis.pixel, dtype=np.float64)
        y_axis_px = np.asarray(calibration.y_axis.pixel, dtype=np.float64)
        red_mask = _red_mask_with_manual_points(img_bgr, [origin_px, x_axis_px, y_axis_px])
        # Legacy naming: bl, br, tl.
        return origin_px, x_axis_px, y_axis_px, red_mask

    _legacy.detect_axes_red_points = detect_from_calibration
    try:
        yield
    finally:
        _legacy.detect_axes_red_points = original


def _normalise_legacy_output(raw) -> List[DigitizedCurve]:
    """Convert legacy return shapes into a list of DigitizedCurve objects."""
    curves: List[DigitizedCurve] = []

    # auto-colors path returns [(x, y, mask), ...]
    if isinstance(raw, list):
        for idx, item in enumerate(raw, start=1):
            if item is None:
                continue
            x, y = item[0], item[1]
            curves.append(DigitizedCurve(np.asarray(x, dtype=float), np.asarray(y, dtype=float), f"curve{idx}"))
        return curves

    # single-color path returns (x_out, y_out)
    if isinstance(raw, tuple) and len(raw) >= 2:
        x, y = raw[0], raw[1]
        if x is not None and y is not None:
            curves.append(DigitizedCurve(np.asarray(x, dtype=float), np.asarray(y, dtype=float), "curve1"))
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
        Axis limits used when ``calibration`` is omitted. Ignored when a
        calibration is supplied.
    options:
        Algorithm options.
    """
    options = options or DigitizeOptions()
    warnings: List[str] = []

    if calibration is not None:
        xmin, xmax, ymin, ymax = calibration.as_legacy_ranges()
        warnings.extend(calibration.validation_warnings())
    else:
        missing = [name for name, value in {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}.items() if value is None]
        if missing:
            raise ValueError(f"Missing axis limits without manual calibration: {', '.join(missing)}")

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
        raise RuntimeError("No digitized curve points were returned. Try changing color settings or calibration points.")

    return DigitizeResult(curves=curves, warnings=warnings)
