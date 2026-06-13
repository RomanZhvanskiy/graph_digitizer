from plot_digitizer import AxisCalibration, CalibrationPoint, DigitizeOptions


def test_calibration_ranges():
    cal = AxisCalibration(
        origin=CalibrationPoint((0, 10), (0, 0)),
        x_axis=CalibrationPoint((10, 10), (1, 0)),
        y_axis=CalibrationPoint((0, 0), (0, 2)),
    )
    assert cal.as_legacy_ranges() == (0.0, 1.0, 0.0, 2.0)


def test_options_defaults():
    opts = DigitizeOptions()
    assert opts.k == 100
    assert opts.knn_k == 3
