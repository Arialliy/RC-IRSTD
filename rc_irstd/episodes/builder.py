from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rc_irstd.data.score_records import load_score_record
from rc_irstd.data.windows import build_causal_windows, build_iid_windows
from rc_irstd.evaluation.curves import (
    aggregate_curve_counts,
    compute_image_curves,
    rates_from_counts,
)
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.utils.io import list_npz


@dataclass(frozen=True)
class EpisodeBuildConfig:
    context_size: int = 32
    horizon: int = 16
    stride: int = 16
    protocol: str = "auto"  # auto | iid | temporal
    seed: int = 0
    peak_min_distance: int = 2
    peak_min_score: float = 0.0
    peak_border: int = 0
    peak_tolerance: float = 2.0
    max_candidates_per_image: int | None = None
    pixel_epsilon: float = 1e-12
    peak_epsilon: float = 1e-6

    def validate(self) -> None:
        if self.context_size <= 0 or self.horizon <= 0 or self.stride <= 0:
            raise ValueError("context_size, horizon and stride must be positive")
        if self.protocol not in {"auto", "iid", "temporal"}:
            raise ValueError("protocol must be auto, iid or temporal")


def default_threshold_grid() -> np.ndarray:
    empty_threshold = np.nextafter(np.float32(1.0), np.float32(2.0))
    return np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.90, 64, endpoint=False),
                np.linspace(0.90, 0.99, 64, endpoint=False),
                np.linspace(0.99, 0.999, 64, endpoint=False),
                np.linspace(0.999, 1.0, 61),
                np.asarray([empty_threshold], dtype=np.float32),
            ]
        )
    ).astype(np.float32)


def _resolve_protocol(records, requested: str) -> str:
    if requested != "auto":
        return requested
    types = {getattr(record, "dataset_type", "iid_images") for record in records}
    return "temporal" if types == {"temporal"} else "iid"


def build_episode_file(
    score_directory: str | Path,
    output_path: str | Path,
    thresholds: np.ndarray | None = None,
    config: EpisodeBuildConfig | None = None,
) -> Path:
    config = config or EpisodeBuildConfig()
    config.validate()
    thresholds = (
        default_threshold_grid()
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float32)
    )
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be a strictly increasing 1-D array")

    records = [load_score_record(path, require_mask=True) for path in list_npz(score_directory)]
    ordered = sorted(records, key=lambda item: (item.sequence_id, item.frame_index, item.image_id))
    protocol = _resolve_protocol(ordered, config.protocol)
    if protocol == "temporal":
        windows = build_causal_windows(
            [item.sequence_id for item in ordered],
            [item.frame_index for item in ordered],
            context_size=config.context_size,
            horizon=config.horizon,
            stride=config.stride,
        )
    else:
        windows = build_iid_windows(
            len(ordered),
            context_size=config.context_size,
            horizon=config.horizon,
            stride=config.stride,
            seed=config.seed,
        )
    if not windows:
        raise ValueError(
            f"No {protocol} support/query windows from {len(ordered)} records with "
            f"context={config.context_size}, horizon={config.horizon}, stride={config.stride}"
        )

    feature_config = WindowFeatureConfig(
        peak_min_distance=config.peak_min_distance,
        peak_min_score=config.peak_min_score,
        peak_border=config.peak_border,
        max_candidates_per_image=config.max_candidates_per_image,
    )
    feature_extractor = WindowFeatureExtractor(feature_config)

    image_curves = []
    image_rates: list[dict[str, np.ndarray]] = []
    for record in ordered:
        assert record.mask is not None
        curve = compute_image_curves(
            record.probability,
            record.mask,
            thresholds,
            peak_min_distance=config.peak_min_distance,
            peak_min_score=config.peak_min_score,
            peak_border=config.peak_border,
            peak_tolerance=config.peak_tolerance,
            max_candidates=config.max_candidates_per_image,
        )
        image_curves.append(curve)
        image_rates.append(
            rates_from_counts(
                curve,
                pixel_epsilon=config.pixel_epsilon,
                peak_epsilon=config.peak_epsilon,
            )
        )

    feature_rows: list[np.ndarray] = []
    pixel_log_rows: list[np.ndarray] = []
    peak_log_rows: list[np.ndarray] = []
    pixel_rate_rows: list[np.ndarray] = []
    peak_rate_rows: list[np.ndarray] = []
    pd_rows: list[np.ndarray] = []
    context_pixel_upper_rows: list[np.ndarray] = []
    context_peak_upper_rows: list[np.ndarray] = []
    future_pixel_rows: list[np.ndarray] = []
    future_peak_rows: list[np.ndarray] = []
    future_pd_rows: list[np.ndarray] = []
    future_gt_rows: list[np.ndarray] = []
    domains: list[str] = []
    sequences: list[str] = []
    protocols: list[str] = []
    context_ids: list[str] = []
    future_ids: list[str] = []
    feature_names: tuple[str, ...] | None = None

    for window in windows:
        context_records = [ordered[index] for index in window.context_indices]
        future_records = [ordered[index] for index in window.future_indices]
        context_counts = aggregate_curve_counts(
            [image_curves[index] for index in window.context_indices]
        )
        future_counts = aggregate_curve_counts(
            [image_curves[index] for index in window.future_indices]
        )
        rates = rates_from_counts(
            future_counts,
            pixel_epsilon=config.pixel_epsilon,
            peak_epsilon=config.peak_epsilon,
        )
        features, names = feature_extractor.extract(context_records)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature schema changed between episodes")

        feature_rows.append(features)
        pixel_log_rows.append(rates["pixel_log_risk"].astype(np.float32))
        peak_log_rows.append(rates["peak_log_risk"].astype(np.float32))
        pixel_rate_rows.append(rates["pixel_false_rate"].astype(np.float32))
        peak_rate_rows.append(rates["peak_false_per_mp"].astype(np.float32))
        pd_rows.append(rates["pd"].astype(np.float32))
        context_pixel_upper_rows.append(
            (context_counts.predicted_pixels / max(context_counts.total_pixels, 1)).astype(np.float32)
        )
        context_peak_upper_rows.append(
            (
                context_counts.predicted_peaks
                / max(context_counts.total_pixels / 1_000_000.0, 1e-12)
            ).astype(np.float32)
        )
        future_pixel_rows.append(
            np.stack(
                [image_rates[index]["pixel_false_rate"] for index in window.future_indices]
            ).astype(np.float32)
        )
        future_peak_rows.append(
            np.stack(
                [image_rates[index]["peak_false_per_mp"] for index in window.future_indices]
            ).astype(np.float32)
        )
        future_pd_rows.append(
            np.stack([image_rates[index]["pd"] for index in window.future_indices]).astype(np.float32)
        )
        future_gt_rows.append(
            np.asarray([image_curves[index].total_gt for index in window.future_indices], dtype=np.int32)
        )
        domains.append(future_records[0].dataset_name)
        sequences.append(window.sequence_id)
        protocols.append(window.protocol)
        context_ids.append(json.dumps([item.image_id for item in context_records]))
        future_ids.append(json.dumps([item.image_id for item in future_records]))

    assert feature_names is not None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.stack(feature_rows).astype(np.float32),
        pixel_log_risk=np.stack(pixel_log_rows).astype(np.float32),
        peak_log_risk=np.stack(peak_log_rows).astype(np.float32),
        pixel_risk=np.stack(pixel_rate_rows).astype(np.float32),
        peak_risk=np.stack(peak_rate_rows).astype(np.float32),
        pd=np.stack(pd_rows).astype(np.float32),
        context_pixel_upper=np.stack(context_pixel_upper_rows).astype(np.float32),
        context_peak_upper=np.stack(context_peak_upper_rows).astype(np.float32),
        future_pixel_risk=np.stack(future_pixel_rows).astype(np.float32),
        future_peak_risk=np.stack(future_peak_rows).astype(np.float32),
        future_pd=np.stack(future_pd_rows).astype(np.float32),
        future_gt_count=np.stack(future_gt_rows).astype(np.int32),
        thresholds=thresholds,
        domains=np.asarray(domains, dtype=np.str_),
        sequences=np.asarray(sequences, dtype=np.str_),
        protocols=np.asarray(protocols, dtype=np.str_),
        context_ids=np.asarray(context_ids, dtype=np.str_),
        future_ids=np.asarray(future_ids, dtype=np.str_),
        feature_names=np.asarray(feature_names, dtype=np.str_),
        feature_config_json=np.asarray(json.dumps(feature_config.to_dict(), sort_keys=True)),
        build_config_json=np.asarray(json.dumps(asdict(config), sort_keys=True)),
        context_size=np.asarray(config.context_size, dtype=np.int64),
        horizon=np.asarray(config.horizon, dtype=np.int64),
        stride=np.asarray(config.stride, dtype=np.int64),
        protocol=np.asarray(protocol),
        risk_definition=np.asarray("pixel_false_rate_and_fixed_false_peaks_per_mp"),
        context_upper_definition=np.asarray("all_context_detections_treated_as_false_upper_bound"),
    )
    return output_path
