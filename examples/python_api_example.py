from plot_digitizer import AxisCalibration, CalibrationPoint, DigitizeOptions, digitize

calibration = AxisCalibration(
    origin=CalibrationPoint(pixel=(105, 730), data=(0, 0)),
    x_axis=CalibrationPoint(pixel=(980, 730), data=(1, 0)),
    y_axis=CalibrationPoint(pixel=(105, 90), data=(0, 1)),
)

result = digitize(
    "plot.png",
    calibration=calibration,
    options=DigitizeOptions(k=150),
)
result.to_csv("digitized.csv")
