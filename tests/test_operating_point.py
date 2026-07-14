import numpy as np

from rc_irstd.episodes.builder import default_threshold_grid
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold


def test_threshold_grid_contains_empty_prediction_action() -> None:
    thresholds = default_threshold_grid()
    assert thresholds[-1] > 1.0


def test_empty_prediction_action_is_reported_as_rejection() -> None:
    thresholds = np.asarray([0.0, 0.5, 1.000001], dtype=np.float32)
    pixel_log = np.log10(np.asarray([1.0, 0.2, 1e-12]))
    peak_log = np.log10(np.asarray([10.0, 2.0, 1e-6]))
    point = select_dual_budget_threshold(
        thresholds,
        pixel_log,
        peak_log,
        pixel_budget=0.1,
        peak_budget_per_mp=1.0,
    )
    assert point.index == 2
    assert point.rejected
