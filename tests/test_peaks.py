import numpy as np

from rc_irstd.candidates.peaks import build_fixed_peak_set, fixed_peak_curves


def test_fixed_peak_false_count_is_monotone():
    score = np.zeros((16, 16), dtype=np.float32)
    score[3, 3] = 0.9
    score[7, 7] = 0.8
    score[12, 12] = 0.7
    mask = np.zeros_like(score, dtype=np.uint8)
    mask[2:5, 2:5] = 1
    peaks = build_fixed_peak_set(score, mask, min_distance=1, min_score=0.1)
    thresholds = np.linspace(0.0, 1.0, 21, dtype=np.float32)
    total, false, matched = fixed_peak_curves(peaks, thresholds)
    assert np.all(np.diff(total) <= 0)
    assert np.all(np.diff(false) <= 0)
    assert np.all(np.diff(matched) <= 0)


def test_duplicate_candidates_near_one_target_count_as_false() -> None:
    score = np.zeros((20, 20), dtype=np.float32)
    score[8, 8] = 0.90
    score[8, 12] = 0.80
    mask = np.zeros_like(score, dtype=np.uint8)
    mask[8, 10] = 1

    peaks = build_fixed_peak_set(
        score,
        mask,
        min_distance=1,
        min_score=0.1,
        tolerance=3.0,
    )
    assert int((peaks.gt_ids > 0).sum()) == 1
    assert int((peaks.gt_ids == 0).sum()) == 1

    thresholds = np.asarray([0.0, 0.85, 0.95], dtype=np.float32)
    _, false, matched = fixed_peak_curves(peaks, thresholds)
    assert false.tolist() == [1, 0, 0]
    assert matched.tolist() == [1, 1, 0]
