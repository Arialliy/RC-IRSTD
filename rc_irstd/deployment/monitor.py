from __future__ import annotations

import numpy as np

from rc_irstd.models.risk_curve import FeatureNormaliser


def feature_ood_score(
    feature: np.ndarray,
    normaliser: FeatureNormaliser,
    clip: float = 50.0,
) -> float:
    """RMS standardised distance from the meta-training feature centre."""
    transformed = normaliser.transform(np.asarray(feature, dtype=np.float32)[None])[0]
    transformed = np.clip(transformed, -float(clip), float(clip))
    return float(np.sqrt(np.mean(np.square(transformed, dtype=np.float64))))


def score_drift(previous: np.ndarray, current: np.ndarray) -> float:
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.shape != current.shape:
        raise ValueError("Feature vectors must have equal shapes")
    denominator = max(float(np.linalg.norm(previous)), 1e-12)
    return float(np.linalg.norm(current - previous) / denominator)
