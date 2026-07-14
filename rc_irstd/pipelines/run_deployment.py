from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rc_irstd.candidates.peaks import extract_fixed_peaks
from rc_irstd.data.score_records import ScoreRecord, load_score_record
from rc_irstd.deployment.monitor import feature_ood_score
from rc_irstd.deployment.session import DeploymentState, ThresholdUpdate
from rc_irstd.evaluation.operating_point import select_dual_budget_threshold
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir, list_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run causal zero-label threshold adaptation and apply it to future scores."
    )
    parser.add_argument("--score-dir", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--warmup-size", type=int, default=32)
    parser.add_argument(
        "--update-every",
        type=int,
        default=0,
        help="0 freezes one threshold; positive values use a past-only rolling window.",
    )
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--offset-index", type=int, default=0)
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=8.0,
        help="Reject a window if RMS feature z-score exceeds this value; <=0 disables.",
    )
    parser.add_argument("--peak-min-distance", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def _group(records: list[ScoreRecord]) -> dict[str, list[ScoreRecord]]:
    groups: dict[str, list[ScoreRecord]] = {}
    for record in records:
        groups.setdefault(record.sequence_id, []).append(record)
    for values in groups.values():
        values.sort(key=lambda item: (item.frame_index, item.image_id))
    return dict(sorted(groups.items()))


def _select_update(
    sequence: str,
    update_index: int,
    warmup: list[ScoreRecord],
    loaded,
    extractor: WindowFeatureExtractor,
    args: argparse.Namespace,
    device,
) -> ThresholdUpdate:
    feature, names = extractor.extract(warmup)
    if names != loaded.feature_names:
        raise ValueError("Deployment feature schema differs from risk-model checkpoint")
    predictions = predict_risk_curves(loaded, feature, device=device)
    point = select_dual_budget_threshold(
        loaded.thresholds,
        predictions["pixel_log_risk"][0],
        predictions["peak_log_risk"][0],
        args.pixel_budget,
        args.peak_budget,
    )
    final_index = min(point.index + max(args.offset_index, 0), len(loaded.thresholds) - 1)
    ood = feature_ood_score(feature, loaded.normaliser)
    rejected = point.rejected or bool(loaded.thresholds[final_index] > 1.0)
    if args.ood_threshold > 0 and ood > args.ood_threshold:
        final_index = len(loaded.thresholds) - 1
        rejected = True
    return ThresholdUpdate(
        sequence_id=sequence,
        update_index=update_index,
        warmup_ids=tuple(item.image_id for item in warmup),
        base_threshold_index=point.index,
        offset_index=max(args.offset_index, 0),
        final_threshold_index=final_index,
        threshold=float(loaded.thresholds[final_index]),
        predicted_pixel_risk=float(10 ** predictions["pixel_log_risk"][0, final_index]),
        predicted_peak_risk_per_mp=float(10 ** predictions["peak_log_risk"][0, final_index]),
        rejected=rejected,
        feature_ood_score=ood,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.warmup_size <= 0 or args.offset_index < 0:
        raise ValueError("warmup-size must be positive and offset-index non-negative")
    device = resolve_device(args.device)
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    extractor = WindowFeatureExtractor(
        WindowFeatureConfig.from_dict(loaded.metadata.get("feature_config"))
    )
    records = [load_score_record(path, require_mask=False) for path in list_npz(args.score_dir)]
    output_dir = ensure_dir(args.output_dir)
    mask_root = ensure_dir(output_dir / "masks")
    state = DeploymentState(
        detector_checkpoint=(records[0].source_checkpoint if records else ""),
        curve_checkpoint=str(Path(args.curve_checkpoint).resolve()),
        score_directory=str(Path(args.score_dir).resolve()),
        pixel_budget=args.pixel_budget,
        peak_budget_per_mp=args.peak_budget,
        warmup_size=args.warmup_size,
        offset_index=args.offset_index,
    )
    candidate_rows: list[dict[str, object]] = []
    processed = rejected_images = 0

    for sequence, sequence_records in _group(records).items():
        if len(sequence_records) <= args.warmup_size:
            continue
        update = _select_update(
            sequence,
            args.warmup_size,
            sequence_records[: args.warmup_size],
            loaded,
            extractor,
            args,
            device,
        )
        state.add(update)
        active = update
        for index in range(args.warmup_size, len(sequence_records)):
            if (
                args.update_every > 0
                and index > args.warmup_size
                and (index - args.warmup_size) % args.update_every == 0
            ):
                start = max(0, index - args.warmup_size)
                active = _select_update(
                    sequence,
                    index,
                    sequence_records[start:index],
                    loaded,
                    extractor,
                    args,
                    device,
                )
                state.add(active)
            record = sequence_records[index]
            threshold = active.threshold
            binary = np.zeros_like(record.probability, dtype=np.uint8)
            if not active.rejected:
                binary = (record.probability >= threshold).astype(np.uint8)
            else:
                rejected_images += 1
            mask_path = mask_root / f"{record.image_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(binary * 255).save(mask_path)
            if not active.rejected:
                scores, ys, xs = extract_fixed_peaks(
                    record.probability,
                    min_distance=args.peak_min_distance,
                    min_score=max(threshold, 0.0),
                )
                for score, y, x in zip(scores, ys, xs, strict=True):
                    if score < threshold:
                        continue
                    candidate_rows.append(
                        {
                            "image_id": record.image_id,
                            "sequence_id": sequence,
                            "frame_index": record.frame_index,
                            "y": int(y),
                            "x": int(x),
                            "score": float(score),
                            "threshold": threshold,
                            "update_index": active.update_index,
                        }
                    )
            processed += 1

    candidate_path = output_dir / "candidates.csv"
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "image_id", "sequence_id", "frame_index", "y", "x", "score",
            "threshold", "update_index",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    atomic_json_dump(state.to_dict(), output_dir / "deployment_state.json")
    summary = {
        "mode": "causal_zero_label_deployment",
        "formal_guarantee": False,
        "num_input_images": len(records),
        "num_processed_future_images": processed,
        "num_rejected_future_images": rejected_images,
        "num_threshold_updates": len(state.updates),
        "num_candidates": len(candidate_rows),
        "state": str((output_dir / "deployment_state.json").resolve()),
        "masks": str(mask_root.resolve()),
        "candidates": str(candidate_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
