from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rc_irstd.data.score_records import load_score_record
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a deployment threshold from a genuinely unlabeled score-map window."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--last-n", type=int, default=32)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    paths = list_npz(args.score_dir)
    records = [load_score_record(path, require_mask=False) for path in paths]
    records = sorted(records, key=lambda item: (item.sequence_id, item.frame_index, item.image_id))
    if args.last_n > 0:
        records = records[-args.last_n :]
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    feature_config = WindowFeatureConfig.from_dict(
        loaded.metadata.get("feature_config")
    )
    extractor = WindowFeatureExtractor(feature_config)
    features, names = extractor.extract(records)
    if names != loaded.feature_names:
        raise ValueError(
            "Unlabeled-window feature schema differs from the risk-model checkpoint. "
            "Use the same peak and statistics configuration used to build training episodes."
        )
    predictions = predict_risk_curves(loaded, features, device=device)
    point = select_dual_budget_threshold(
        loaded.thresholds,
        predictions["pixel_log_risk"][0],
        predictions["peak_log_risk"][0],
        args.pixel_budget,
        args.peak_budget,
    )
    result = {
        "mode": "zero_label_deployment",
        "num_warmup_images": len(records),
        "threshold_index": point.index,
        "threshold": point.threshold,
        "rejected": point.rejected,
        "predicted_pixel_risk": point.predicted_pixel_risk,
        "predicted_peak_risk_per_mp": point.predicted_peak_risk,
        "pixel_budget": args.pixel_budget,
        "peak_budget_per_mp": args.peak_budget,
        "formal_guarantee": False,
        "source_score_directory": str(Path(args.score_dir).resolve()),
    }
    atomic_json_dump(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
