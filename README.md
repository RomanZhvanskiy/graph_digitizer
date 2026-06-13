# Plot Digitizer

A small Python library and Streamlit frontend for extracting numeric data from plot images.

This project wraps the original single-file OpenCV/Numpy digitizer into a reusable package. It keeps the existing colour-based extraction pipeline, but adds manual three-point calibration so users can click the red/corner points and enter their real plot coordinates.

## Features

- Installable Python package with a typed public API.
- Streamlit GUI for image upload, point selection, coordinate entry, preview, and CSV download.
- CLI for repeatable digitization jobs.
- Manual calibration using three points:
  - origin / bottom-left
  - x-axis point / bottom-right
  - y-axis point / top-left
- Optional automatic multi-colour curve extraction.
- Linear or logarithmic x/y axes.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/plot-digitizer-portfolio.git
cd plot-digitizer-portfolio
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[gui]
```

## Run the GUI

```bash
streamlit run src/plot_digitizer/app.py
```

Workflow:

1. Upload a plot image.
2. Select which reference point you are setting: origin, x-axis corner, or y-axis corner.
3. Click the image to fill that point's pixel coordinate.
4. Enter the real data coordinate for each reference point.
5. Choose extraction options and click **Digitize**.
6. Download the CSV.

## CLI examples

Use the original red marker auto-detection:

```bash
plot-digitizer plot.png --xmin 0 --xmax 1 --ymin 1e-7 --ymax 1 --out out.csv
```

Use manual calibration points:

```bash
plot-digitizer plot.png \
  --origin-px 105,730 --origin-data 0,0 \
  --x-px 980,730 --x-data 1,0 \
  --y-px 105,90 --y-data 0,1 \
  --out out.csv
```

Multi-colour extraction:

```bash
plot-digitizer plot.png \
  --origin-px 105,730 --origin-data 0,0 \
  --x-px 980,730 --x-data 1,0 \
  --y-px 105,90 --y-data 0,1 \
  --auto-colors --exclude-grayscale --out curves.csv
```

## Python API

```python
from plot_digitizer import AxisCalibration, CalibrationPoint, DigitizeOptions, digitize

calibration = AxisCalibration(
    origin=CalibrationPoint(pixel=(105, 730), data=(0, 0)),
    x_axis=CalibrationPoint(pixel=(980, 730), data=(1, 0)),
    y_axis=CalibrationPoint(pixel=(105, 90), data=(0, 1)),
)

result = digitize(
    "plot.png",
    calibration=calibration,
    options=DigitizeOptions(k=150, auto_colors=False, yscale_log=False),
)

result.to_csv("digitized.csv")
print(result.to_dataframe().head())
```

## Project structure

```text
src/plot_digitizer/
  __init__.py      public package exports
  _legacy.py       original digitizer engine, preserved for compatibility
  core.py          typed API and manual calibration wrapper
  cli.py           command-line interface
  app.py           Streamlit frontend
```

## Notes for portfolio polish

Suggested next steps before publishing:

- Add screenshots or a short GIF of the GUI workflow.
- Add a few sample plot images under `examples/`.
- Add regression tests once you have known input/output images.
- Consider splitting `_legacy.py` into smaller modules over time: `calibration.py`, `colors.py`, `curves.py`, `export.py`.

## License

MIT
