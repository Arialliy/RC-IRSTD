from __future__ import annotations

"""Literature-compatible IRSTD segmentation and object metrics."""

from dataclasses import asdict, dataclass

import numpy as np

from rc_irstd.evaluation.segmentation import evaluate_binary_segmentation


@dataclass(frozen=True)
class IRSTDMetrics:
    iou: float
    niou: float
    hiou: float
    foreground_iou: float
    background_iou: float
    precision: float
    recall: float
    f1: float
    pd: float
    false_components_per_mp: float
    false_pixel_rate: float
    gt_objects: int
    detected_objects: int
    false_components: int
    total_pixels: int
    num_images: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def evaluate_irstd_at_threshold(
    probabilities: list[np.ndarray],
    masks: list[np.ndarray],
    threshold: float,
    object_tolerance: float = 2.0,
) -> IRSTDMetrics:
    if len(probabilities) != len(masks) or not probabilities:
        raise ValueError("probabilities and masks must be non-empty and equally sized")
    tp = fp = fn = tn = 0
    union = 0
    total_pixels = 0
    gt_objects = detected_objects = false_components = 0
    per_image_iou: list[float] = []

    for probability, mask in zip(probabilities, masks, strict=True):
        pred = np.asarray(probability).squeeze() >= float(threshold)
        target = np.asarray(mask).squeeze() > 0
        if pred.shape != target.shape:
            raise ValueError("Probability and mask shapes differ")
        metrics = evaluate_binary_segmentation(pred, target, object_tolerance)
        current_tp = metrics.true_positive_pixels
        current_fp = metrics.false_positive_pixels
        current_fn = metrics.false_negative_pixels
        current_tn = int(target.size - current_tp - current_fp - current_fn)
        tp += current_tp
        fp += current_fp
        fn += current_fn
        tn += current_tn
        union += metrics.union
        total_pixels += int(target.size)
        gt_objects += metrics.gt_objects
        detected_objects += metrics.detected_objects
        false_components += metrics.false_components
        # nIoU is the mean image IoU over non-empty unions. Fully empty images
        # are excluded and reported through false-alarm metrics instead.
        if metrics.union > 0:
            per_image_iou.append(metrics.intersection / metrics.union)

    foreground_iou = _safe_ratio(tp, tp + fp + fn)
    background_iou = _safe_ratio(tn, tn + fp + fn)
    hiou = (
        2.0 * foreground_iou * background_iou / (foreground_iou + background_iou)
        if foreground_iou + background_iou > 0
        else 0.0
    )
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return IRSTDMetrics(
        iou=_safe_ratio(tp, union),
        niou=float(np.mean(per_image_iou)) if per_image_iou else 0.0,
        hiou=float(hiou),
        foreground_iou=foreground_iou,
        background_iou=background_iou,
        precision=precision,
        recall=recall,
        f1=f1,
        pd=_safe_ratio(detected_objects, gt_objects),
        false_components_per_mp=_safe_ratio(
            false_components, total_pixels / 1_000_000.0
        ),
        false_pixel_rate=_safe_ratio(fp, total_pixels),
        gt_objects=gt_objects,
        detected_objects=detected_objects,
        false_components=false_components,
        total_pixels=total_pixels,
        num_images=len(probabilities),
    )
