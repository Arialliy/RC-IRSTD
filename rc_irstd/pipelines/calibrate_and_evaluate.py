from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rc_irstd.calibration.crc import (
    adaptive_offset_loss_matrix,
    minimum_calibration_size,
    raw_global_threshold_loss_matrix,
    select_crc_parameter,
)
from rc_irstd.calibration.samples import (
    CalibrationSamples,
    episode_calibration_samples,
    image_calibration_samples,
    split_calibration_samples,
)
from rc_irstd.episodes.dataset import concatenate_episode_files
from rc_irstd.episodes.splits import grouped_calibration_test_split
from rc_irstd.evaluation.budget import summarise_selected_points
from rc_irstd.evaluation.risk_curve_metrics import select_indices_from_predictions
from rc_irstd.models.risk_io import load_risk_model, predict_risk_curves
from rc_irstd.utils.device import resolve_device
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate nested threshold offsets with explicit statistical units."
    )
    parser.add_argument("--episode", action="append", required=True)
    parser.add_argument("--curve-checkpoint", required=True)
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--calibration-sizes", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--calibration-unit",
        choices=["episode", "image"],
        default="episode",
        help="image means the requested size is exactly the number of labelled images.",
    )
    parser.add_argument("--offset-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Debug only: allow episode calibration/test image overlap.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def _episode_ids(arrays, indices: np.ndarray) -> set[str]:
    ids: set[str] = set()
    for index in indices:
        for field in (arrays.context_ids[int(index)], arrays.future_ids[int(index)]):
            values = json.loads(str(field))
            ids.update(str(value) for value in values)
    return ids


def _summary_at_indices(
    samples: CalibrationSamples,
    sample_indices: np.ndarray,
    threshold_indices: np.ndarray,
    rejected: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
):
    rows = np.arange(len(sample_indices))
    return summarise_selected_points(
        samples.pixel_risk[sample_indices][rows, threshold_indices],
        samples.peak_risk[sample_indices][rows, threshold_indices],
        samples.pd[sample_indices][rows, threshold_indices],
        rejected,
        samples.domains[sample_indices],
        pixel_budget,
        peak_budget,
    )


def _empirical_offset(
    losses: np.ndarray, offsets: np.ndarray, alpha: float
) -> tuple[int, bool]:
    empirical = losses.mean(axis=0)
    feasible = np.flatnonzero(empirical <= alpha)
    if len(feasible):
        return int(offsets[int(feasible[0])]), True
    return int(offsets[-1]), False


def _split(
    args: argparse.Namespace,
    arrays,
    samples: CalibrationSamples,
    calibration_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.calibration_unit == "episode":
        calibration, test = grouped_calibration_test_split(
            arrays, calibration_size, seed
        )
        overlap = _episode_ids(arrays, calibration).intersection(
            _episode_ids(arrays, test)
        )
        if overlap and not args.allow_overlap:
            raise ValueError(
                f"Episode split shares {len(overlap)} image IDs. Build evaluation "
                "episodes with non-overlapping windows or use --allow-overlap only for smoke/debug."
            )
        return calibration, test, {
            "strategy": "independent_episode_groups",
            "overlapping_image_ids": sorted(overlap),
            "allow_overlap": bool(args.allow_overlap),
        }
    calibration, test, metadata = split_calibration_samples(
        samples, calibration_size, seed
    )
    return calibration, test, metadata


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.offset_step <= 0:
        raise ValueError("offset-step must be positive")
    if args.pixel_budget <= 0 or args.peak_budget <= 0:
        raise ValueError("budgets must be positive")
    device = resolve_device(args.device)
    arrays = concatenate_episode_files(args.episode)
    loaded = load_risk_model(args.curve_checkpoint, device=device)
    if not np.array_equal(arrays.thresholds, loaded.thresholds):
        raise ValueError("Episode threshold grid differs from risk-model checkpoint")
    if arrays.feature_names != loaded.feature_names:
        raise ValueError("Episode feature schema differs from risk-model checkpoint")

    predictions = predict_risk_curves(
        loaded, arrays.features, device=device, batch_size=args.batch_size
    )
    episode_base_indices, episode_base_rejected = select_indices_from_predictions(
        arrays.thresholds,
        predictions["pixel_log_risk"],
        predictions["peak_log_risk"],
        args.pixel_budget,
        args.peak_budget,
    )
    if args.calibration_unit == "image":
        samples = image_calibration_samples(
            arrays, episode_base_indices, episode_base_rejected
        )
    else:
        samples = episode_calibration_samples(
            arrays, episode_base_indices, episode_base_rejected
        )

    num_thresholds = len(arrays.thresholds)
    offsets = np.arange(0, num_thresholds, args.offset_step, dtype=np.int64)
    if offsets[-1] != num_thresholds - 1:
        offsets = np.append(offsets, num_thresholds - 1)
    threshold_parameters = np.arange(num_thresholds, dtype=np.int64)

    output_dir = ensure_dir(args.output_dir)
    all_results: list[dict[str, Any]] = []
    split_records: dict[str, Any] = {}

    for calibration_size in args.calibration_sizes:
        for seed in args.seeds:
            calibration, test, split_metadata = _split(
                args, arrays, samples, calibration_size, seed
            )
            num_labeled_images = int(samples.label_count_per_sample[calibration].sum())
            if args.calibration_unit == "image" and num_labeled_images != calibration_size:
                raise RuntimeError(
                    "Image calibration count mismatch: requested "
                    f"{calibration_size}, obtained {num_labeled_images}"
                )
            split_key = f"{samples.unit}_m{calibration_size}_seed{seed}"
            split_records[split_key] = {
                "calibration_indices": calibration.tolist(),
                "test_indices": test.tolist(),
                "calibration_sample_ids": samples.sample_ids[calibration].tolist(),
                "test_sample_ids": samples.sample_ids[test].tolist(),
                "num_calibration_samples": int(len(calibration)),
                "num_labeled_images": num_labeled_images,
                **split_metadata,
            }

            adaptive_losses, _ = adaptive_offset_loss_matrix(
                samples.pixel_risk[calibration],
                samples.peak_risk[calibration],
                samples.base_indices[calibration],
                offsets,
                args.pixel_budget,
                args.peak_budget,
            )
            adaptive_crc = select_crc_parameter(adaptive_losses, offsets, args.alpha)
            adaptive_test_indices = np.minimum(
                samples.base_indices[test] + adaptive_crc.selected_parameter,
                num_thresholds - 1,
            )
            adaptive_rejected = samples.base_rejected[test] | (
                arrays.thresholds[adaptive_test_indices] > 1.0
            )
            adaptive_summary = _summary_at_indices(
                samples,
                test,
                adaptive_test_indices,
                adaptive_rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            common = {
                "calibration_unit": samples.unit,
                "calibration_size": calibration_size,
                "num_calibration_samples": int(len(calibration)),
                "num_labeled_images": num_labeled_images,
                "num_test_samples": int(len(test)),
                "seed": seed,
            }
            all_results.append(
                {
                    "method": "adaptive_risk_curve_crc_offset",
                    **common,
                    "formal_crc_feasible": adaptive_crc.feasible,
                    "selected_parameter": adaptive_crc.selected_parameter,
                    "selected_threshold": None,
                    "crc": adaptive_crc.to_dict(),
                    **adaptive_summary.to_dict(),
                }
            )

            empirical_offset, empirical_feasible = _empirical_offset(
                adaptive_losses, offsets, args.alpha
            )
            empirical_test_indices = np.minimum(
                samples.base_indices[test] + empirical_offset, num_thresholds - 1
            )
            empirical_rejected = samples.base_rejected[test] | (
                arrays.thresholds[empirical_test_indices] > 1.0
            )
            empirical_summary = _summary_at_indices(
                samples,
                test,
                empirical_test_indices,
                empirical_rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            all_results.append(
                {
                    "method": "adaptive_risk_curve_empirical_offset",
                    **common,
                    "formal_crc_feasible": False,
                    "empirical_feasible": empirical_feasible,
                    "selected_parameter": empirical_offset,
                    "selected_threshold": None,
                    **empirical_summary.to_dict(),
                }
            )

            raw_losses = raw_global_threshold_loss_matrix(
                samples.pixel_risk[calibration],
                samples.peak_risk[calibration],
                args.pixel_budget,
                args.peak_budget,
            )
            raw_crc = select_crc_parameter(
                raw_losses, threshold_parameters, args.alpha
            )
            raw_test_indices = np.full(
                len(test), raw_crc.selected_parameter, dtype=np.int64
            )
            raw_rejected = arrays.thresholds[raw_test_indices] > 1.0
            raw_summary = _summary_at_indices(
                samples,
                test,
                raw_test_indices,
                raw_rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            all_results.append(
                {
                    "method": "raw_global_threshold_crc",
                    **common,
                    "formal_crc_feasible": raw_crc.feasible,
                    "selected_parameter": raw_crc.selected_parameter,
                    "selected_threshold": float(
                        arrays.thresholds[raw_crc.selected_parameter]
                    ),
                    "crc": raw_crc.to_dict(),
                    **raw_summary.to_dict(),
                }
            )

            zero_summary = _summary_at_indices(
                samples,
                test,
                samples.base_indices[test],
                samples.base_rejected[test],
                args.pixel_budget,
                args.peak_budget,
            )
            all_results.append(
                {
                    "method": "zero_label_no_calibration",
                    **common,
                    "formal_crc_feasible": False,
                    "selected_parameter": 0,
                    "selected_threshold": None,
                    **zero_summary.to_dict(),
                }
            )

    result_path = output_dir / "results.csv"
    flat_keys = [
        "method",
        "calibration_unit",
        "calibration_size",
        "num_calibration_samples",
        "num_labeled_images",
        "num_test_samples",
        "seed",
        "formal_crc_feasible",
        "selected_parameter",
        "selected_threshold",
        "joint_bsr",
        "pixel_bsr",
        "peak_bsr",
        "pixel_excess",
        "peak_excess",
        "mean_pd_selected",
        "effective_pd_with_rejects",
        "conditional_pd_non_rejected",
        "worst_domain_pd_selected",
        "rejection_rate",
        "count",
    ]
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys)
        writer.writeheader()
        for result in all_results:
            writer.writerow({key: result.get(key) for key in flat_keys})

    summary = {
        "risk_control_target": "marginal probability of violating pixel or fixed-peak budget",
        "loss": "binary joint budget violation",
        "alpha": args.alpha,
        "minimum_calibration_samples_for_any_crc_solution": minimum_calibration_size(args.alpha),
        "pixel_budget": args.pixel_budget,
        "peak_budget": args.peak_budget,
        "calibration_unit": samples.unit,
        "counting_rule": (
            "calibration_size is exactly labelled images"
            if samples.unit == "image"
            else "calibration_size is labelled future blocks; num_labeled_images is also reported"
        ),
        "exchangeability_requirement": (
            "CRC requires exchangeability at the declared statistical unit. IID image mode "
            "uses unique images; temporal/block mode must be interpreted at the blocked unit."
        ),
        "results": all_results,
        "results_csv": str(result_path.resolve()),
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    atomic_json_dump(split_records, output_dir / "splits.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
