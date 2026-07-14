from __future__ import annotations

"""Label-free prefix-to-future deployment for the final no-reject path."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rc_irstd.data.score_records import load_score_record
from rc_irstd.features.source_distance import source_distances_from_reference
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.models.calibrator_io import load_monotone_calibrator, predict_threshold_curve
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an unlabeled prefix to freeze thresholds for future query images."
    )
    parser.add_argument("--score-directory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--budget", action="append", type=float, required=True)
    parser.add_argument("--source-reference", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.context_size <= 0:
        raise ValueError("context-size must be positive")
    device = resolve_device(args.device)
    loaded = load_monotone_calibrator(args.checkpoint, device)

    paths = list_npz(args.score_directory)
    records = [load_score_record(path, load_mask=False) for path in paths]
    ordered = sorted(
        zip(paths, records, strict=True),
        key=lambda pair: (pair[1].sequence_id, pair[1].frame_index, pair[1].image_id),
    )
    if len(ordered) <= args.context_size:
        raise ValueError("Deployment requires at least one future query image")
    support = [record for _, record in ordered[: args.context_size]]
    query = [record for _, record in ordered[args.context_size :]]
    checkpoints = {record.source_checkpoint for record in support}
    if len(checkpoints) > 1:
        raise ValueError("Support score records were produced by different detector checkpoints")

    config = WindowFeatureConfig.from_dict(loaded.feature_config)
    features, names = WindowFeatureExtractor(config).extract(support)
    if names != loaded.feature_names:
        raise ValueError("Deployment feature schema differs from calibrator checkpoint")
    source_distances = None
    if args.source_reference:
        source_distances = source_distances_from_reference(
            features, args.source_reference, names
        )[None, :]
    curve = predict_threshold_curve(
        loaded,
        features[None, :],
        device,
        source_distances=source_distances,
    )
    requested = torch.tensor(args.budget, device=device, dtype=torch.float32)
    with torch.inference_mode():
        eta = loaded.model.interpolate_logit(
            torch.from_numpy(curve).to(device), requested
        ).cpu().numpy()[0]
    threshold = 1.0 / (1.0 + np.exp(-eta))
    payload = {
        "method": "two_stage_no_reject_monotone_inverse_pixel_risk",
        "protocol": "unlabeled_prefix_to_future_query",
        "support_labels_read": False,
        "requested_budgets": [float(value) for value in args.budget],
        "threshold_logits": eta.astype(float).tolist(),
        "thresholds": threshold.astype(float).tolist(),
        "training_budget_grid": loaded.budgets.astype(float).tolist(),
        "interpolation": "linear_in_log10_budget_no_extrapolation",
        "support_image_ids": [record.image_id for record in support],
        "query_image_ids": [record.image_id for record in query],
        "detector_checkpoint": next(iter(checkpoints)) if checkpoints else "",
        "score_directory": str(Path(args.score_directory).resolve()),
        "calibrator_checkpoint": str(Path(args.checkpoint).resolve()),
        "main_risk": "original_resolution_pixel_false_alarm_rate",
        "connected_component_fa": "compatibility_evaluation_only",
        "guarantee": "empirical_meta_calibration_not_certified",
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
