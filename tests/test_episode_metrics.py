import numpy as np

from rc_irstd.evaluation.curves import compute_image_curves, rates_from_counts


def test_pixel_and_fixed_peak_risks_are_monotone():
    score = np.zeros((20, 20), dtype=np.float32)
    score[2, 2] = 0.9
    score[10, 10] = 0.7
    score[15, 15] = 0.5
    mask = np.zeros_like(score, dtype=np.uint8)
    mask[1:4, 1:4] = 1
    thresholds = np.linspace(0, 1, 101, dtype=np.float32)
    counts = compute_image_curves(score, mask, thresholds, peak_min_distance=1)
    rates = rates_from_counts(counts)
    assert np.all(np.diff(rates["pixel_false_rate"]) <= 0)
    assert np.all(np.diff(rates["peak_false_per_mp"]) <= 0)
