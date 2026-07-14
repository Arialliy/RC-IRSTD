from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rc_irstd.episodes.dataset import EpisodeArrays, concatenate_episode_files
from rc_irstd.evaluation.budget import summarise_selected_points
from rc_irstd.models.risk_curve import FeatureNormaliser
from rc_irstd.utils.io import atomic_json_dump, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed, source-derived, nearest-source, all-detection "
            "upper-bound and oracle operating-point baselines."
        )
    )
    parser.add_argument("--target-episode", action="append", required=True)
    parser.add_argument(
        "--source-episode",
        action="append",
        default=None,
        help="Episodes produced by the same final detector on source domains.",
    )
    parser.add_argument("--fixed-threshold", action="append", type=float, default=[0.5])
    parser.add_argument("--pixel-budget", type=float, default=1e-6)
    parser.add_argument("--peak-budget", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    return parser


def _first_feasible_indices(
    pixel_curves: np.ndarray,
    peak_curves: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel = np.asarray(pixel_curves, dtype=np.float64)
    peak = np.asarray(peak_curves, dtype=np.float64)
    if pixel.shape != peak.shape or pixel.ndim != 2:
        raise ValueError("Risk curves must share shape [episodes, thresholds]")
    feasible = (pixel <= pixel_budget) & (peak <= peak_budget)
    any_feasible = feasible.any(axis=1)
    indices = np.argmax(feasible, axis=1).astype(np.int64)
    indices[~any_feasible] = pixel.shape[1] - 1
    return indices, ~any_feasible


def _mark_empty_action(
    thresholds: np.ndarray,
    indices: np.ndarray,
    rejected: np.ndarray,
) -> np.ndarray:
    result = np.asarray(rejected, dtype=bool).copy()
    result |= np.asarray(thresholds)[np.asarray(indices, dtype=np.int64)] > 1.0
    return result


def _constant_curve_index(
    pixel_curve: np.ndarray,
    peak_curve: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[int, bool]:
    indices, rejected = _first_feasible_indices(
        np.asarray(pixel_curve)[None],
        np.asarray(peak_curve)[None],
        pixel_budget,
        peak_budget,
    )
    return int(indices[0]), bool(rejected[0])


def _summarise(
    arrays: EpisodeArrays,
    indices: np.ndarray,
    rejected: np.ndarray,
    pixel_budget: float,
    peak_budget: float,
) -> dict[str, Any]:
    rows = np.arange(len(indices))
    summary = summarise_selected_points(
        arrays.pixel_risk[rows, indices],
        arrays.peak_risk[rows, indices],
        arrays.pd[rows, indices],
        rejected,
        arrays.domains,
        pixel_budget,
        peak_budget,
    )
    return summary.to_dict()


def _nearest_source_indices(
    source: EpisodeArrays,
    target: EpisodeArrays,
    pixel_budget: float,
    peak_budget: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normaliser = FeatureNormaliser.fit(source.features)
    source_features = normaliser.transform(source.features)
    target_features = normaliser.transform(target.features)
    domains = np.unique(source.domains)
    centroids = np.stack(
        [source_features[source.domains == domain].mean(axis=0) for domain in domains]
    )
    domain_pixel = np.stack(
        [source.pixel_risk[source.domains == domain].mean(axis=0) for domain in domains]
    )
    domain_peak = np.stack(
        [source.peak_risk[source.domains == domain].mean(axis=0) for domain in domains]
    )
    domain_indices: list[int] = []
    domain_rejected: list[bool] = []
    for pixel_curve, peak_curve in zip(domain_pixel, domain_peak, strict=True):
        index, rejected = _constant_curve_index(
            pixel_curve,
            peak_curve,
            pixel_budget,
            peak_budget,
        )
        domain_indices.append(index)
        domain_rejected.append(rejected)
    squared = ((target_features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(squared, axis=1)
    return (
        np.asarray(domain_indices, dtype=np.int64)[nearest],
        np.asarray(domain_rejected, dtype=bool)[nearest],
        domains[nearest],
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    target = concatenate_episode_files(args.target_episode)
    source = (
        concatenate_episode_files(args.source_episode)
        if args.source_episode
        else None
    )
    if source is not None:
        if not np.array_equal(source.thresholds, target.thresholds):
            raise ValueError("Source and target episodes use different threshold grids")
        if source.feature_names != target.feature_names:
            raise ValueError("Source and target feature schemas differ")

    methods: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for threshold in args.fixed_threshold:
        index = int(np.searchsorted(target.thresholds, threshold, side="left"))
        index = min(index, len(target.thresholds) - 1)
        methods[f"fixed_{threshold:g}"] = (
            np.full(len(target.features), index, dtype=np.int64),
            np.full(
                len(target.features),
                bool(target.thresholds[index] > 1.0),
                dtype=bool,
            ),
            None,
        )

    # Strong label-free baseline: every context detection is provisionally
    # counted as false. It is an exact upper bound on context false detections,
    # but only an empirical predictor of the disjoint future block.
    if np.isfinite(target.context_pixel_upper).all() and np.isfinite(
        target.context_peak_upper
    ).all():
        index, rejected = _first_feasible_indices(
            target.context_pixel_upper,
            target.context_peak_upper,
            args.pixel_budget,
            args.peak_budget,
        )
        rejected = _mark_empty_action(target.thresholds, index, rejected)
        methods["context_all_detection_upper"] = (index, rejected, None)

    # Target-label oracle is reported only as an upper benchmark.
    index, rejected = _first_feasible_indices(
        target.pixel_risk,
        target.peak_risk,
        args.pixel_budget,
        args.peak_budget,
    )
    rejected = _mark_empty_action(target.thresholds, index, rejected)
    methods["target_future_oracle"] = (index, rejected, None)

    if source is not None:
        pooled_index, pooled_rejected = _constant_curve_index(
            source.pixel_risk.mean(axis=0),
            source.peak_risk.mean(axis=0),
            args.pixel_budget,
            args.peak_budget,
        )
        pooled_rejected = pooled_rejected or bool(
            target.thresholds[pooled_index] > 1.0
        )
        methods["source_pooled"] = (
            np.full(len(target.features), pooled_index, dtype=np.int64),
            np.full(len(target.features), pooled_rejected, dtype=bool),
            None,
        )

        domain_pixel = np.stack(
            [
                source.pixel_risk[source.domains == domain].mean(axis=0)
                for domain in np.unique(source.domains)
            ]
        )
        domain_peak = np.stack(
            [
                source.peak_risk[source.domains == domain].mean(axis=0)
                for domain in np.unique(source.domains)
            ]
        )
        worst_index, worst_rejected = _constant_curve_index(
            domain_pixel.max(axis=0),
            domain_peak.max(axis=0),
            args.pixel_budget,
            args.peak_budget,
        )
        worst_rejected = worst_rejected or bool(
            target.thresholds[worst_index] > 1.0
        )
        methods["source_worst_domain"] = (
            np.full(len(target.features), worst_index, dtype=np.int64),
            np.full(len(target.features), worst_rejected, dtype=bool),
            None,
        )

        nearest_index, nearest_rejected, nearest_domain = _nearest_source_indices(
            source,
            target,
            args.pixel_budget,
            args.peak_budget,
        )
        nearest_rejected = _mark_empty_action(
            target.thresholds,
            nearest_index,
            nearest_rejected,
        )
        methods["nearest_source_curve"] = (
            nearest_index,
            nearest_rejected,
            nearest_domain,
        )

    output_dir = ensure_dir(args.output_dir)
    results: dict[str, Any] = {}
    selected_path = output_dir / "selected_points.csv"
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "episode_index",
                "domain",
                "sequence",
                "threshold_index",
                "threshold",
                "rejected",
                "reference_source_domain",
                "true_pixel_risk",
                "true_peak_risk",
                "pd",
                "joint_budget_satisfied",
            ]
        )
        rows = np.arange(len(target.features))
        for name, (indices, rejected, references) in methods.items():
            summary = _summarise(
                target,
                indices,
                rejected,
                args.pixel_budget,
                args.peak_budget,
            )
            results[name] = summary
            for row, index in enumerate(indices):
                reference = "" if references is None else str(references[row])
                pixel = float(target.pixel_risk[row, index])
                peak = float(target.peak_risk[row, index])
                writer.writerow(
                    [
                        name,
                        row,
                        target.domains[row],
                        target.sequences[row],
                        int(index),
                        float(target.thresholds[index]),
                        bool(rejected[row]),
                        reference,
                        pixel,
                        peak,
                        float(target.pd[row, index]),
                        bool(pixel <= args.pixel_budget and peak <= args.peak_budget),
                    ]
                )

    summary = {
        "pixel_budget": args.pixel_budget,
        "peak_budget": args.peak_budget,
        "num_target_episodes": len(target.features),
        "source_episodes_provided": source is not None,
        "methods": results,
        "selected_points_csv": str(selected_path.resolve()),
        "interpretation": {
            "context_all_detection_upper": (
                "Label-free and deterministic on the warm-up context; all predicted "
                "pixels/peaks are counted as false. Its use for a future block still "
                "requires temporal stability and has no distribution-free guarantee."
            ),
            "target_future_oracle": "Uses target future labels and is not deployable.",
        },
    }
    atomic_json_dump(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
