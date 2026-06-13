from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from .core import AxisCalibration, CalibrationPoint, DigitizeOptions, digitize


def _pair(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 'x,y'.")
    return float(parts[0]), float(parts[1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Digitize plot curves from an image.")
    p.add_argument("image", help="Input image path")
    p.add_argument("--out", required=True, help="Output CSV path")

    p.add_argument("--xmin", type=float)
    p.add_argument("--xmax", type=float)
    p.add_argument("--ymin", type=float)
    p.add_argument("--ymax", type=float)

    p.add_argument("--origin-px", type=_pair, help="Manual origin pixel as x,y")
    p.add_argument("--origin-data", type=_pair, help="Manual origin data coordinate as x,y")
    p.add_argument("--x-px", type=_pair, help="Manual x-axis point pixel as x,y")
    p.add_argument("--x-data", type=_pair, help="Manual x-axis point data coordinate as x,y")
    p.add_argument("--y-px", type=_pair, help="Manual y-axis point pixel as x,y")
    p.add_argument("--y-data", type=_pair, help="Manual y-axis point data coordinate as x,y")

    p.add_argument("--k", type=int, default=100)
    p.add_argument("--auto-colors", action="store_true")
    p.add_argument("--num-colors", type=int, default=0)
    p.add_argument("--exclude-grayscale", action="store_true")
    p.add_argument("--monotone-y", action="store_true")
    p.add_argument("--xscale-log", action="store_true")
    p.add_argument("--yscale-log", action="store_true")
    p.add_argument("--keep-outside-axes", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    manual_values = [args.origin_px, args.origin_data, args.x_px, args.x_data, args.y_px, args.y_data]
    calibration = None
    if any(v is not None for v in manual_values):
        if not all(v is not None for v in manual_values):
            raise SystemExit("Manual calibration requires all of --origin-px, --origin-data, --x-px, --x-data, --y-px, --y-data.")
        calibration = AxisCalibration(
            origin=CalibrationPoint(args.origin_px, args.origin_data),
            x_axis=CalibrationPoint(args.x_px, args.x_data),
            y_axis=CalibrationPoint(args.y_px, args.y_data),
        )

    options = DigitizeOptions(
        k=args.k,
        auto_colors=args.auto_colors,
        num_colors=args.num_colors,
        exclude_grayscale=args.exclude_grayscale,
        monotone_y=args.monotone_y,
        xscale_log=args.xscale_log,
        yscale_log=args.yscale_log,
        keep_outside_axes=args.keep_outside_axes,
        debug=args.debug,
    )
    result = digitize(args.image, calibration, xmin=args.xmin, xmax=args.xmax, ymin=args.ymin, ymax=args.ymax, options=options)
    result.to_csv(args.out)
    for warning in result.warnings:
        print(f"warning: {warning}")
    print(f"wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
