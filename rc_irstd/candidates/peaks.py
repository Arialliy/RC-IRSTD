from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FixedPeakSet:
    scores: np.ndarray
    ys: np.ndarray
    xs: np.ndarray
    gt_ids: np.ndarray
    num_gt: int

    def __post_init__(self) -> None:
        lengths = {len(self.scores), len(self.ys), len(self.xs), len(self.gt_ids)}
        if len(lengths) != 1:
            raise ValueError("Peak arrays must have the same length")


def _one_point_per_plateau(mask: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(mask)
    ys: list[int] = []
    xs: list[int] = []
    for component_id in range(1, count + 1):
        coords = np.argwhere(labels == component_id)
        if coords.size == 0:
            continue
        values = score[coords[:, 0], coords[:, 1]]
        best_value = values.max()
        best = coords[values == best_value]
        # Lexicographic tie-breaking is deterministic across platforms.
        order = np.lexsort((best[:, 1], best[:, 0]))
        y, x = best[order[0]]
        ys.append(int(y))
        xs.append(int(x))
    return np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)


def extract_fixed_peaks(
    score_map: np.ndarray,
    min_distance: int = 2,
    min_score: float = 0.0,
    border: int = 0,
    max_candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a threshold-independent set of deterministic local maxima.

    Once this set is extracted, increasing a score threshold can only remove
    candidates. Consequently, candidate count, false-candidate count and matched
    target count are monotone with respect to the threshold.
    """
    score = np.asarray(score_map, dtype=np.float32).squeeze()
    if score.ndim != 2:
        raise ValueError(f"score_map must be 2-D, got shape {score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("score_map contains NaN or infinity")
    if min_distance < 0:
        raise ValueError("min_distance must be non-negative")
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be positive or None")
    size = 2 * min_distance + 1
    local_max = score >= ndimage.maximum_filter(score, size=size, mode="nearest")
    candidate_mask = local_max & (score >= float(min_score))
    if border > 0:
        candidate_mask[:border, :] = False
        candidate_mask[-border:, :] = False
        candidate_mask[:, :border] = False
        candidate_mask[:, -border:] = False
    ys, xs = _one_point_per_plateau(candidate_mask, score)
    scores = score[ys, xs] if len(ys) else np.empty((0,), dtype=np.float32)
    order = np.lexsort((xs, ys, -scores))
    scores, ys, xs = scores[order], ys[order], xs[order]
    if max_candidates is not None:
        scores = scores[:max_candidates]
        ys = ys[:max_candidates]
        xs = xs[:max_candidates]
    return scores.astype(np.float32), ys, xs


def assign_peaks_to_gt(
    ys: np.ndarray,
    xs: np.ndarray,
    gt_mask: np.ndarray,
    tolerance: float = 2.0,
) -> tuple[np.ndarray, int]:
    mask = np.asarray(gt_mask).squeeze() > 0
    if mask.ndim != 2:
        raise ValueError("gt_mask must be 2-D")
    gt_labels, num_gt = ndimage.label(mask)
    gt_ids = np.zeros(len(ys), dtype=np.int32)
    if num_gt == 0 or len(ys) == 0:
        return gt_ids, int(num_gt)

    distance, nearest = ndimage.distance_transform_edt(~mask, return_indices=True)
    height, width = mask.shape
    for index, (y, x) in enumerate(zip(ys, xs, strict=True)):
        y_i, x_i = int(y), int(x)
        if not (0 <= y_i < height and 0 <= x_i < width):
            raise ValueError("Peak coordinate is outside the mask")
        direct = int(gt_labels[y_i, x_i])
        if direct > 0:
            gt_ids[index] = direct
            continue
        if distance[y_i, x_i] <= tolerance:
            near_y = int(nearest[0, y_i, x_i])
            near_x = int(nearest[1, y_i, x_i])
            gt_ids[index] = int(gt_labels[near_y, near_x])
    return gt_ids, int(num_gt)


def keep_one_peak_per_gt(
    gt_ids: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Turn a provisional many-to-one assignment into a fixed one-to-one match.

    Several local maxima can fall inside, or within the tolerance radius of, the
    same target. Counting every such maximum as a true candidate would make the
    false-candidate metric artificially permissive. We therefore retain only the
    highest-scoring candidate for each GT component and mark all duplicate
    candidates as background. The assignment is computed once, before any score
    threshold is applied, so the resulting true/false candidate labels remain a
    nested family as the threshold increases.
    """

    assignments = np.asarray(gt_ids, dtype=np.int32).copy()
    candidate_scores = np.asarray(scores, dtype=np.float32)
    if assignments.shape != candidate_scores.shape:
        raise ValueError("gt_ids and scores must have the same shape")
    for gt_id in np.unique(assignments[assignments > 0]):
        members = np.flatnonzero(assignments == gt_id)
        if len(members) <= 1:
            continue
        # Stable first-index tie-breaking is deterministic because extracted
        # candidates are already sorted by score, then coordinates.
        best = int(members[np.argmax(candidate_scores[members])])
        assignments[members] = 0
        assignments[best] = int(gt_id)
    return assignments


def build_fixed_peak_set(
    score_map: np.ndarray,
    gt_mask: np.ndarray,
    min_distance: int = 2,
    min_score: float = 0.0,
    border: int = 0,
    tolerance: float = 2.0,
    max_candidates: int | None = None,
) -> FixedPeakSet:
    scores, ys, xs = extract_fixed_peaks(
        score_map,
        min_distance=min_distance,
        min_score=min_score,
        border=border,
        max_candidates=max_candidates,
    )
    gt_ids, num_gt = assign_peaks_to_gt(ys, xs, gt_mask, tolerance=tolerance)
    gt_ids = keep_one_peak_per_gt(gt_ids, scores)
    return FixedPeakSet(scores=scores, ys=ys, xs=xs, gt_ids=gt_ids, num_gt=num_gt)


def fixed_peak_curves(
    peak_set: FixedPeakSet,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if np.any(np.diff(thresholds) < 0):
        raise ValueError("thresholds must be ascending")
    total = np.zeros(len(thresholds), dtype=np.int64)
    false = np.zeros(len(thresholds), dtype=np.int64)
    matched = np.zeros(len(thresholds), dtype=np.int64)
    for index, threshold in enumerate(thresholds):
        active = peak_set.scores >= threshold
        active_ids = peak_set.gt_ids[active]
        total[index] = int(active.sum())
        false[index] = int((active_ids == 0).sum())
        matched[index] = len(np.unique(active_ids[active_ids > 0]))
    return total, false, matched
