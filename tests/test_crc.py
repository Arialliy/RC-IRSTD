import numpy as np

from rc_irstd.calibration.crc import (
    adaptive_offset_loss_matrix,
    select_crc_parameter,
)


def test_crc_detects_small_sample_infeasibility() -> None:
    losses = np.zeros((5, 3), dtype=np.float64)
    result = select_crc_parameter(losses, np.asarray([0, 1, 2]), alpha=0.1)
    assert not result.feasible
    assert np.isclose(result.minimum_possible_corrected_risk, 1.0 / 6.0)


def test_adaptive_joint_loss_is_nested() -> None:
    pixel = np.asarray([[0.2, 0.1, 0.01], [0.3, 0.05, 0.0]])
    peak = np.asarray([[3.0, 1.0, 0.0], [4.0, 2.0, 0.0]])
    losses, selected = adaptive_offset_loss_matrix(
        pixel,
        peak,
        base_indices=np.asarray([0, 0]),
        offsets=np.asarray([0, 1, 2]),
        pixel_budget=0.1,
        peak_budget=1.5,
    )
    assert selected.shape == losses.shape
    assert np.all(np.diff(losses, axis=1) <= 0)
