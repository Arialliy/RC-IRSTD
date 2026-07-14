import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays
from rc_irstd.episodes.splits import grouped_calibration_test_split


def _arrays() -> EpisodeArrays:
    n = 12
    t = 4
    return EpisodeArrays(
        features=np.zeros((n, 3), dtype=np.float32),
        pixel_log_risk=np.zeros((n, t), dtype=np.float32),
        peak_log_risk=np.zeros((n, t), dtype=np.float32),
        pixel_risk=np.zeros((n, t), dtype=np.float32),
        peak_risk=np.zeros((n, t), dtype=np.float32),
        pd=np.zeros((n, t), dtype=np.float32),
        context_pixel_upper=np.zeros((n, t), dtype=np.float32),
        context_peak_upper=np.zeros((n, t), dtype=np.float32),
        thresholds=np.linspace(0, 1, t, dtype=np.float32),
        domains=np.asarray(["target"] * n),
        sequences=np.asarray(["s0"] * 4 + ["s1"] * 4 + ["s2"] * 4),
        context_ids=np.asarray(["[]"] * n),
        future_ids=np.asarray(["[]"] * n),
        feature_names=("a", "b", "c"),
    )


def test_calibration_test_are_sequence_disjoint() -> None:
    arrays = _arrays()
    calibration, test = grouped_calibration_test_split(arrays, calibration_size=3, seed=7)
    assert len(calibration) == 3
    assert len(test) > 0
    assert set(arrays.sequences[calibration]).isdisjoint(set(arrays.sequences[test]))
