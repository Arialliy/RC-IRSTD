import numpy as np

from rc_irstd.features.window_stats import WindowFeatureConfig


def test_window_feature_config_round_trip() -> None:
    config = WindowFeatureConfig(
        survival_thresholds=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        quantiles=np.asarray([0.5, 0.95], dtype=np.float32),
        peak_min_distance=3,
        peak_min_score=0.02,
        peak_border=1,
        max_candidates_per_image=None,
    )
    restored = WindowFeatureConfig.from_dict(config.to_dict())

    assert np.array_equal(restored.survival_thresholds, config.survival_thresholds)
    assert np.array_equal(restored.quantiles, config.quantiles)
    assert restored.peak_min_distance == config.peak_min_distance
    assert restored.peak_min_score == config.peak_min_score
    assert restored.peak_border == config.peak_border
    assert restored.max_candidates_per_image is None
