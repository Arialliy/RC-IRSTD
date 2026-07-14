from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SegmentationMetrics:
    intersection: int
    union: int
    true_positive_pixels: int
    false_positive_pixels: int
    false_negative_pixels: int
    gt_objects: int
    detected_objects: int
    false_components: int


def evaluate_binary_segmentation(
    prediction: np.ndarray,
    target: np.ndarray,
    object_tolerance: float = 2.0,
) -> SegmentationMetrics:
    pred = np.asarray(prediction).squeeze() > 0
    gt = np.asarray(target).squeeze() > 0
    if pred.shape != gt.shape:
        raise ValueError("prediction and target must have equal shapes")
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    union = int((pred | gt).sum())

    gt_labels, gt_count = ndimage.label(gt)
    pred_labels, pred_count = ndimage.label(pred)
    detected: set[int] = set()
    false_components = 0
    if gt_count > 0:
        distance, nearest = ndimage.distance_transform_edt(~gt, return_indices=True)
    else:
        distance = np.full(gt.shape, np.inf)
        nearest = np.zeros((2,) + gt.shape, dtype=np.int64)

    for component_id in range(1, pred_count + 1):
        coords = np.argwhere(pred_labels == component_id)
        overlapping = np.unique(gt_labels[pred_labels == component_id])
        overlapping = overlapping[overlapping > 0]
        if len(overlapping):
            detected.update(int(value) for value in overlapping)
            continue
        centroid = np.rint(coords.mean(axis=0)).astype(int)
        y, x = int(centroid[0]), int(centroid[1])
        if distance[y, x] <= object_tolerance:
            near_y = int(nearest[0, y, x])
            near_x = int(nearest[1, y, x])
            gt_id = int(gt_labels[near_y, near_x])
            if gt_id > 0:
                detected.add(gt_id)
                continue
        false_components += 1

    return SegmentationMetrics(
        intersection=tp,
        union=union,
        true_positive_pixels=tp,
        false_positive_pixels=fp,
        false_negative_pixels=fn,
        gt_objects=int(gt_count),
        detected_objects=len(detected),
        false_components=false_components,
    )
