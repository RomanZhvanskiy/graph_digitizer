from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:  # pragma: no cover - import guard for friendly app error
    streamlit_image_coordinates = None

from .core import AxisCalibration, CalibrationPoint, DigitizeOptions, digitize


st.set_page_config(page_title="Plot Digitizer", layout="wide")
st.title("Plot Digitizer")
st.caption("Upload a plot, click the three red/corner calibration points, enter their data coordinates, then export CSV.")

if "points" not in st.session_state:
    st.session_state.points = {
        "origin": {"pixel": None, "data": (0.0, 0.0)},
        "x_axis": {"pixel": None, "data": (1.0, 0.0)},
        "y_axis": {"pixel": None, "data": (0.0, 1.0)},
    }

uploaded = st.file_uploader("Plot image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])

left, right = st.columns([1.35, 1.0])

with left:
    if uploaded is None:
        st.info("Upload an image to start.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    st.write(f"Image size: {image.width} × {image.height} px")

    if streamlit_image_coordinates is None:
        st.error("streamlit-image-coordinates is not installed. Install the project with GUI extras: `pip install -e .[gui]`.")
        st.stop()

    role = st.radio(
        "Click assignment",
        ["origin", "x_axis", "y_axis"],
        format_func={"origin": "Origin / bottom-left", "x_axis": "X-axis corner / bottom-right", "y_axis": "Y-axis corner / top-left"}.get,
        horizontal=True,
    )
    coords = streamlit_image_coordinates(image, key="image_coords")
    if coords is not None:
        st.session_state.points[role]["pixel"] = (float(coords["x"]), float(coords["y"]))
        st.success(f"Set {role} pixel to ({coords['x']}, {coords['y']}).")

with right:
    st.subheader("Calibration")
    labels = {
        "origin": "Origin / bottom-left",
        "x_axis": "X-axis corner / bottom-right",
        "y_axis": "Y-axis corner / top-left",
    }
    for key, label in labels.items():
        st.markdown(f"**{label}**")
        px = st.session_state.points[key]["pixel"]
        c1, c2, c3, c4 = st.columns(4)
        default_x = 0.0 if px is None else float(px[0])
        default_y = 0.0 if px is None else float(px[1])
        px_x = c1.number_input(f"{key} px x", value=default_x, key=f"{key}_px_x")
        px_y = c2.number_input(f"{key} px y", value=default_y, key=f"{key}_px_y")
        data_x_default, data_y_default = st.session_state.points[key]["data"]
        data_x = c3.number_input(f"{key} data x", value=float(data_x_default), format="%.12g", key=f"{key}_data_x")
        data_y = c4.number_input(f"{key} data y", value=float(data_y_default), format="%.12g", key=f"{key}_data_y")
        st.session_state.points[key] = {"pixel": (px_x, px_y), "data": (data_x, data_y)}

    st.subheader("Options")
    auto_colors = st.checkbox("Auto-detect multiple colour curves", value=False)
    exclude_grayscale = st.checkbox("Exclude grayscale pixels", value=True)
    k = st.slider("Representative points per curve", min_value=10, max_value=1000, value=100, step=10)
    num_colors = st.number_input("Force number of colours/clusters (0 = auto)", min_value=0, max_value=20, value=0)
    xscale_log = st.checkbox("Log x-axis", value=False)
    yscale_log = st.checkbox("Log y-axis", value=False)
    monotone_y = st.checkbox("Force monotone non-increasing y", value=False)

    run = st.button("Digitize", type="primary", use_container_width=True)

if run:
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / uploaded.name
        image.save(image_path)
        p = st.session_state.points
        calibration = AxisCalibration(
            origin=CalibrationPoint(p["origin"]["pixel"], p["origin"]["data"]),
            x_axis=CalibrationPoint(p["x_axis"]["pixel"], p["x_axis"]["data"]),
            y_axis=CalibrationPoint(p["y_axis"]["pixel"], p["y_axis"]["data"]),
        )
        options = DigitizeOptions(
            k=int(k),
            auto_colors=auto_colors,
            exclude_grayscale=exclude_grayscale,
            num_colors=int(num_colors),
            xscale_log=xscale_log,
            yscale_log=yscale_log,
            monotone_y=monotone_y,
        )
        with st.spinner("Digitizing..."):
            result = digitize(image_path, calibration=calibration, options=options)

    for warning in result.warnings:
        st.warning(warning)

    df = result.to_dataframe()
    st.subheader("Result")
    st.dataframe(df, use_container_width=True)

    if len(result.curves) == 1:
        st.line_chart(pd.DataFrame({"x": result.curves[0].x, "y": result.curves[0].y}).set_index("x"))

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="digitized_curves.csv",
        mime="text/csv",
        use_container_width=True,
    )
