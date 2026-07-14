from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rc_irstd.episodes.dataset import concatenate_episode_files
from rc_irstd.evaluation.risk_curve_metrics import evaluate_risk_curve_predictions
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate zero-label risk-curve operating points.")
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    arrays = concatenate_episode_files(args.episode)
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    if not np.array_equal(arrays.thresholds, loaded.thresholds):
        raise ValueError("Episode threshold grid differs from the risk-model checkpoint")
    if arrays.feature_names != loaded.feature_names:
        raise ValueError("Episode feature schema differs from the risk-model checkpoint")

    predictions = predict_risk_curves(
        loaded,
        arrays.features,
        device=device,
        batch_size=args.batch_size,
    )
    metrics, indices, rejected = evaluate_risk_curve_predictions(
        arrays.thresholds,
        predictions["pixel_log_risk"],
        predictions["peak_log_risk"],
        arrays.pixel_log_risk,
        arrays.peak_log_risk,
        arrays.pixel_risk,
        arrays.peak_risk,
        arrays.pd,
        arrays.domains,
        args.pixel_budget,
        args.peak_budget,
    )

    output_dir = ensure_dir(args.output_dir)
    rows = np.arange(len(indices))
    result_path = output_dir / "selected_points.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode_index",
                "domain",
                "sequence",
                "threshold_index",
                "threshold",
                "rejected",
                "predicted_pixel_risk",
                "predicted_peak_risk",
                "true_pixel_risk",
                "true_peak_risk",
                "pd",
                "joint_budget_satisfied",
            ]
        )
        for row, index in enumerate(indices):
            true_pixel = float(arrays.pixel_risk[row, index])
            true_peak = float(arrays.peak_risk[row, index])
            writer.writerow(
                [
                    row,
                    arrays.domains[row],
                    arrays.sequences[row],
                    int(index),
                    float(arrays.thresholds[index]),
                    bool(rejected[row]),
                    float(10.0 ** predictions["pixel_log_risk"][row, index]),
                    float(10.0 ** predictions["peak_log_risk"][row, index]),
                    true_pixel,
                    true_peak,
                    float(arrays.pd[row, index]),
                    bool(true_pixel <= args.pixel_budget and true_peak <= args.peak_budget),
                ]
            )

    np.savez_compressed(
        output_dir / "zero_label_outputs.npz",
        base_indices=indices,
        rejected=rejected,
        predicted_pixel_log_risk=predictions["pixel_log_risk"],
        predicted_peak_log_risk=predictions["peak_log_risk"],
        thresholds=arrays.thresholds,
        domains=arrays.domains,
        sequences=arrays.sequences,
    )
    summary = {
        "mode": "zero_label_empirical_adaptation",
        "formal_guarantee": False,
        "pixel_budget": args.pixel_budget,
        "peak_budget": args.peak_budget,
        "num_episodes": len(indices),
        "metrics": metrics.to_dict(),
        "selected_points_csv": str(result_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
