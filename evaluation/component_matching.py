"""Deterministic 8-connected, one-to-one IRSTD component matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
from skimage.measure import label, regionprops


MatchingRule = Literal["overlap", "centroid"]


@dataclass(frozen=True)
class MatchResult:
    """Object and false-alarm counts for one image.

    ``fp_pixels`` is deliberately independent of component assignment: it is
    the count of *all predicted pixels outside the GT foreground*.
    ``matched_pairs`` contains zero-based ``(prediction, ground_truth)`` IDs.
    """

    num_gt: int
    num_pred_components: int
    num_tp_objects: int
    num_fp_components: int
    num_fp_pixels: int
    total_pixels: int
    matched_pairs: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedTarget:
    """Immutable GT labels and object geometry reusable across thresholds."""

    binary: np.ndarray
    labels: np.ndarray
    num_gt: int
    centroids: tuple[tuple[float, float], ...]


def prepare_target(target: np.ndarray) -> PreparedTarget:
    """Label an invariant target once for an entire threshold sweep."""

    binary = np.ascontiguousarray(_as_binary_2d(target, "target")).copy()
    labels = np.ascontiguousarray(label(binary, connectivity=2))
    centroids = tuple(
        (float(prop.centroid[0]), float(prop.centroid[1]))
        for prop in regionprops(labels)
    )
    binary.setflags(write=False)
    labels.setflags(write=False)
    return PreparedTarget(
        binary=binary,
        labels=labels,
        num_gt=int(labels.max()),
        centroids=centroids,
    )


def match_components(
    prediction: np.ndarray,
    target: np.ndarray | PreparedTarget,
    *,
    rule: MatchingRule = "overlap",
    centroid_distance: float = 3.0,
) -> MatchResult:
    """Match predicted and GT components one-to-one.

    Both maps use 8-connectivity.  Overlap matching creates an edge when two
    components share at least one pixel.  Centroid matching creates an edge
    when Euclidean distance is strictly below ``centroid_distance``, matching
    the historical MSHNet ``distance < 3`` convention by default.  A maximum
    cardinality bipartite assignment prevents one prediction from detecting
    multiple GT objects (and vice versa).
    """

    pred = _as_binary_2d(prediction, "prediction")
    prepared = target if isinstance(target, PreparedTarget) else prepare_target(target)
    gt = prepared.binary
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {gt.shape}")
    if rule not in {"overlap", "centroid"}:
        raise ValueError("rule must be 'overlap' or 'centroid'")
    if centroid_distance <= 0:
        raise ValueError("centroid_distance must be positive")

    pred_labels = label(pred, connectivity=2)
    num_pred = int(pred_labels.max())
    num_gt = prepared.num_gt

    if rule == "overlap":
        edges = _overlap_edges(pred_labels, prepared.labels, num_gt)
    else:
        edges = _centroid_edges(
            pred_labels,
            prepared.centroids,
            centroid_distance,
        )
    matched_pairs = _maximum_cardinality_pairs(edges, num_pred)

    num_tp = len(matched_pairs)
    return MatchResult(
        num_gt=num_gt,
        num_pred_components=num_pred,
        num_tp_objects=num_tp,
        num_fp_components=num_pred - num_tp,
        num_fp_pixels=int(np.count_nonzero(pred & ~gt)),
        total_pixels=int(pred.size),
        matched_pairs=matched_pairs,
    )


def aggregate_match_results(results: Iterable[MatchResult]) -> dict[str, float | int]:
    """Aggregate image-level results into Pd and two false-alarm measures."""

    result_list = list(results)
    total_gt = sum(result.num_gt for result in result_list)
    total_tp = sum(result.num_tp_objects for result in result_list)
    total_fp_components = sum(result.num_fp_components for result in result_list)
    total_fp_pixels = sum(result.num_fp_pixels for result in result_list)
    total_pixels = sum(result.total_pixels for result in result_list)
    pd = total_tp / total_gt if total_gt else 0.0
    fa_pixel = total_fp_pixels / total_pixels if total_pixels else 0.0
    fa_component_mp = (
        total_fp_components / (total_pixels / 1_000_000.0)
        if total_pixels
        else 0.0
    )
    return {
        "pd": float(pd),
        "fa_pixel": float(fa_pixel),
        "fa_component_mp": float(fa_component_mp),
        "tp_objects": int(total_tp),
        "gt_objects": int(total_gt),
        "pred_components": int(
            sum(result.num_pred_components for result in result_list)
        ),
        "fp_components": int(total_fp_components),
        "fp_pixels": int(total_fp_pixels),
        "total_pixels": int(total_pixels),
        "num_images": len(result_list),
    }


def _as_binary_2d(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"{name} must resolve to a 2D map, got shape {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return value.astype(bool, copy=False)


def _overlap_edges(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_gt: int,
) -> list[list[tuple[int, float]]]:
    edges: list[list[tuple[int, float]]] = [[] for _ in range(num_gt)]
    overlap = (pred_labels > 0) & (gt_labels > 0)
    if not np.any(overlap):
        return edges

    pair_ids = np.stack(
        (pred_labels[overlap] - 1, gt_labels[overlap] - 1),
        axis=1,
    )
    unique_pairs, counts = np.unique(pair_ids, axis=0, return_counts=True)
    for (pred_id, gt_id), count in zip(unique_pairs, counts):
        edges[int(gt_id)].append((int(pred_id), float(count)))
    for candidates in edges:
        candidates.sort(key=lambda item: (-item[1], item[0]))
    return edges


def _centroid_edges(
    pred_labels: np.ndarray,
    gt_centroids: tuple[tuple[float, float], ...],
    max_distance: float,
) -> list[list[tuple[int, float]]]:
    pred_centroids = [np.asarray(prop.centroid) for prop in regionprops(pred_labels)]
    edges: list[list[tuple[int, float]]] = [[] for _ in range(len(gt_centroids))]
    for gt_id, gt_centroid in enumerate(gt_centroids):
        gt_centroid_array = np.asarray(gt_centroid)
        for pred_id, pred_centroid in enumerate(pred_centroids):
            distance = float(np.linalg.norm(pred_centroid - gt_centroid_array))
            if distance < max_distance:
                # Larger scores are preferred by the deterministic matcher.
                edges[gt_id].append((pred_id, max_distance - distance))
        edges[gt_id].sort(key=lambda item: (-item[1], item[0]))
    return edges


def _maximum_cardinality_pairs(
    edges: list[list[tuple[int, float]]],
    num_predictions: int,
) -> tuple[tuple[int, int], ...]:
    """Kuhn augmenting-path matching with deterministic candidate ordering."""

    pred_to_gt = [-1] * num_predictions

    def augment(gt_id: int, visited_predictions: set[int]) -> bool:
        for pred_id, _ in edges[gt_id]:
            if pred_id in visited_predictions:
                continue
            visited_predictions.add(pred_id)
            previous_gt = pred_to_gt[pred_id]
            if previous_gt < 0 or augment(previous_gt, visited_predictions):
                pred_to_gt[pred_id] = gt_id
                return True
        return False

    # Constrained GT nodes first reduces unnecessary reassignment while Kuhn's
    # algorithm still guarantees maximum cardinality.
    gt_order = sorted(range(len(edges)), key=lambda gt_id: (len(edges[gt_id]), gt_id))
    for gt_id in gt_order:
        augment(gt_id, set())

    return tuple(
        sorted(
            (
                (pred_id, gt_id)
                for pred_id, gt_id in enumerate(pred_to_gt)
                if gt_id >= 0
            ),
            key=lambda pair: (pair[0], pair[1]),
        )
    )
