from __future__ import annotations

"""Grouped multi-budget meta episodes for the final no-reject RC-IRSTD model.

Each row is one disjoint support/query episode and contains the *entire* budget
vector ``[J]``.  Support features are label-free.  Query labels are used only to
construct risk-aligned supervision and exact hard-replay metadata.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from rc_irstd.data.score_records import ScoreRecord, load_score_record
from rc_irstd.data.windows import CausalWindow, build_causal_windows, build_iid_windows
from rc_irstd.episodes.builder import default_threshold_grid
from rc_irstd.evaluation.curves import (
    aggregate_curve_counts,
    compute_image_curves,
    rates_from_counts,
)
from rc_irstd.features.window_stats import WindowFeatureConfig, WindowFeatureExtractor
from rc_irstd.models.monotone_pixel_calibrator import validate_budget_grid
from rc_irstd.utils.io import list_npz


_LOGIT_EPS = 1e-6


def probability_to_logit(probability: np.ndarray | float, eps: float = _LOGIT_EPS) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(value, eps, 1.0 - eps)
    return (np.log(clipped) - np.log1p(-clipped)).astype(np.float32)


@dataclass(frozen=True)
class MetaEpisodeBuildConfig:
    context_size: int = 32
    horizon: int = 64
    stride: int | None = None
    protocol: str = "auto"  # auto | iid | temporal
    seed: int = 0
    background_sample_limit: int = 65_536
    object_top_fraction: float = 0.25
    peak_min_distance: int = 2
    peak_min_score: float = 0.0
    peak_border: int = 0
    peak_tolerance: float = 2.0
    max_candidates_per_image: int | None = None
    enforce_global_disjoint: bool = True
    split_role: str = "official_train_meta"

    def validate(self) -> None:
        if self.context_size <= 0 or self.horizon <= 0:
            raise ValueError("context_size and horizon must be positive")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive or None")
        if self.protocol not in {"auto", "iid", "temporal"}:
            raise ValueError("protocol must be auto, iid or temporal")
        if self.background_sample_limit <= 0:
            raise ValueError("background_sample_limit must be positive")
        if not 0.0 < self.object_top_fraction <= 1.0:
            raise ValueError("object_top_fraction must be in (0, 1]")
        if not self.split_role:
            raise ValueError("split_role must be non-empty")


@dataclass(frozen=True)
class MetaEpisodeArrays:
    features: np.ndarray
    budgets: np.ndarray
    oracle_threshold_logit: np.ndarray
    background_logits: np.ndarray
    background_valid: np.ndarray
    background_fraction: np.ndarray
    object_scores: np.ndarray
    object_valid: np.ndarray
    hard_thresholds: np.ndarray
    hard_pixel_risk: np.ndarray
    hard_pd: np.ndarray
    domains: np.ndarray
    sequences: np.ndarray
    support_ids: np.ndarray
    query_ids: np.ndarray
    query_paths: np.ndarray
    feature_names: tuple[str, ...]
    feature_config: dict[str, object] = field(default_factory=dict)
    build_config: dict[str, object] = field(default_factory=dict)
    source_distances: np.ndarray | None = None
    source_distance_valid: np.ndarray | None = None
    score_roots: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = len(self.features)
        row_fields = {
            "oracle_threshold_logit": self.oracle_threshold_logit,
            "background_logits": self.background_logits,
            "background_valid": self.background_valid,
            "background_fraction": self.background_fraction,
            "object_scores": self.object_scores,
            "object_valid": self.object_valid,
            "hard_pixel_risk": self.hard_pixel_risk,
            "hard_pd": self.hard_pd,
            "domains": self.domains,
            "sequences": self.sequences,
            "support_ids": self.support_ids,
            "query_ids": self.query_ids,
            "query_paths": self.query_paths,
        }
        for name, value in row_fields.items():
            if len(value) != count:
                raise ValueError(f"{name} has {len(value)} rows; expected {count}")
        if self.oracle_threshold_logit.shape[1] != len(self.budgets):
            raise ValueError("oracle threshold width must equal budget count")
        if self.hard_pixel_risk.shape != self.hard_pd.shape:
            raise ValueError("hard risk and Pd curves must share shape")
        if self.hard_pixel_risk.shape[1] != len(self.hard_thresholds):
            raise ValueError("hard curve width must equal threshold count")
        if (self.source_distances is None) != (self.source_distance_valid is None):
            raise ValueError("source distance arrays must both be present or absent")
        if self.score_roots is not None and len(self.score_roots) != count:
            raise ValueError("score_roots must have one entry per episode")

    def subset(self, indices: np.ndarray) -> "MetaEpisodeArrays":
        index = np.asarray(indices, dtype=np.int64)
        optional = lambda value: None if value is None else value[index]
        return MetaEpisodeArrays(
            features=self.features[index],
            budgets=self.budgets,
            oracle_threshold_logit=self.oracle_threshold_logit[index],
            background_logits=self.background_logits[index],
            background_valid=self.background_valid[index],
            background_fraction=self.background_fraction[index],
            object_scores=self.object_scores[index],
            object_valid=self.object_valid[index],
            hard_thresholds=self.hard_thresholds,
            hard_pixel_risk=self.hard_pixel_risk[index],
            hard_pd=self.hard_pd[index],
            domains=self.domains[index],
            sequences=self.sequences[index],
            support_ids=self.support_ids[index],
            query_ids=self.query_ids[index],
            query_paths=self.query_paths[index],
            feature_names=self.feature_names,
            feature_config=dict(self.feature_config),
            build_config=dict(self.build_config),
            source_distances=optional(self.source_distances),
            source_distance_valid=optional(self.source_distance_valid),
            score_roots=optional(self.score_roots),
        )


def _pad_rows(rows: list[np.ndarray], fill: float, dtype) -> np.ndarray:
    width = max((len(row) for row in rows), default=0)
    output = np.full((len(rows), width), fill, dtype=dtype)
    for index, row in enumerate(rows):
        output[index, : len(row)] = np.asarray(row, dtype=dtype)
    return output


def _load_source_reference(
    path: str | Path | None,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as payload:
        centres = np.asarray(payload["centres"], dtype=np.float32)
        scale = np.asarray(payload.get("scale", np.ones(centres.shape[1])), dtype=np.float32)
        names = tuple(np.asarray(payload["feature_names"]).astype(str).tolist())
        domains = tuple(np.asarray(payload.get("domains", np.arange(len(centres)))).astype(str).tolist())
    if names != feature_names:
        raise ValueError("Source-reference feature schema does not match support features")
    if centres.ndim != 2 or centres.shape[1] != len(feature_names):
        raise ValueError("Invalid source-reference centres")
    if scale.shape != (centres.shape[1],):
        raise ValueError("Invalid source-reference scale")
    return centres, np.maximum(scale, 1e-6), domains


def _resolve_protocol(records: Sequence[ScoreRecord], requested: str) -> str:
    if requested != "auto":
        return requested
    types = {record.dataset_type for record in records}
    return "temporal" if types == {"temporal"} else "iid"


def _build_windows(records: Sequence[ScoreRecord], config: MetaEpisodeBuildConfig) -> list[CausalWindow]:
    protocol = _resolve_protocol(records, config.protocol)
    total = config.context_size + config.horizon
    stride = total if config.stride is None else config.stride
    if protocol == "temporal":
        windows = build_causal_windows(
            [record.sequence_id for record in records],
            [record.frame_index for record in records],
            context_size=config.context_size,
            horizon=config.horizon,
            stride=stride,
        )
    else:
        windows = build_iid_windows(
            len(records),
            context_size=config.context_size,
            horizon=config.horizon,
            stride=stride,
            seed=config.seed,
        )
    if config.enforce_global_disjoint:
        seen: set[int] = set()
        for window in windows:
            current = set(window.context_indices) | set(window.future_indices)
            if seen.intersection(current):
                raise ValueError(
                    "Episode windows overlap globally. Use stride >= context+horizon "
                    "or explicitly disable enforce_global_disjoint for diagnostics."
                )
            seen.update(current)
    return windows


def _sample_background_logits(
    records: Sequence[ScoreRecord],
    limit: int,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    rows: list[np.ndarray] = []
    total_background = 0
    total_pixels = 0
    maximum = -np.inf
    for record in records:
        if record.mask is None:
            raise ValueError("Query records require masks")
        logits = probability_to_logit(record.probability)
        mask = np.asarray(record.mask).squeeze() > 0
        background = logits[~mask]
        rows.append(background.astype(np.float32, copy=False))
        total_background += int(background.size)
        total_pixels += int(logits.size)
        maximum = max(maximum, float(np.max(logits)))
    all_background = np.concatenate(rows) if rows else np.empty((0,), dtype=np.float32)
    if len(all_background) > limit:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(all_background), size=limit, replace=False)
        all_background = all_background[np.sort(indices)]
    fraction = float(total_background / max(total_pixels, 1))
    return all_background.astype(np.float32), fraction, float(maximum)


def _query_object_scores(
    records: Sequence[ScoreRecord],
    top_fraction: float,
) -> np.ndarray:
    scores: list[float] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for record in records:
        if record.mask is None:
            raise ValueError("Query records require masks")
        logits = probability_to_logit(record.probability)
        labels, count = ndimage.label(np.asarray(record.mask).squeeze() > 0, structure=structure)
        for component_id in range(1, count + 1):
            values = logits[labels == component_id]
            if values.size == 0:
                continue
            k = max(1, int(np.ceil(top_fraction * values.size)))
            partition = np.partition(values, values.size - k)[-k:]
            scores.append(float(np.mean(partition, dtype=np.float64)))
    return np.asarray(scores, dtype=np.float32)


def oracle_threshold_logits(
    thresholds: np.ndarray,
    pixel_risk: np.ndarray,
    pd: np.ndarray,
    budgets: np.ndarray,
    empty_logit: float,
) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    risk = np.asarray(pixel_risk, dtype=np.float64)
    utility = np.asarray(pd, dtype=np.float64)
    output: list[float] = []
    for budget in budgets:
        feasible = np.flatnonzero(risk <= float(budget))
        if len(feasible) == 0:
            raise ValueError(
                "No budget-feasible hard threshold. The threshold grid must include "
                "an explicit empty-prediction sentinel."
            )
        # Primary: maximum Pd. Tie: lower threshold (more utility-preserving).
        order = np.lexsort((thresholds[feasible], -utility[feasible]))
        index = int(feasible[order[0]])
        threshold = float(thresholds[index])
        if threshold > 1.0:
            output.append(float(empty_logit))
        else:
            output.append(float(probability_to_logit(threshold)))
    return np.asarray(output, dtype=np.float32)


def build_meta_episode_file(
    score_directory: str | Path,
    output_path: str | Path,
    *,
    budgets: Sequence[float] = (1e-4, 1e-5, 1e-6),
    thresholds: np.ndarray | None = None,
    config: MetaEpisodeBuildConfig | None = None,
    source_reference: str | Path | None = None,
) -> Path:
    config = config or MetaEpisodeBuildConfig()
    config.validate()
    budget_array = validate_budget_grid(list(budgets))
    threshold_array = (
        default_threshold_grid()
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float32)
    )
    if threshold_array.ndim != 1 or np.any(np.diff(threshold_array) <= 0):
        raise ValueError("thresholds must be a strictly increasing 1-D array")
    if threshold_array[-1] <= 1.0:
        raise ValueError("threshold grid must end with an empty-prediction sentinel > 1")

    score_root = Path(score_directory).resolve()
    paths = list_npz(score_root)
    if not paths:
        raise ValueError(f"No score records found in {score_root}")
    # Label-free load: support construction never accesses the mask array.
    label_free_records = [
        load_score_record(path, require_mask=False, load_mask=False) for path in paths
    ]
    pairs = sorted(
        zip(paths, label_free_records, strict=True),
        key=lambda pair: (pair[1].sequence_id, pair[1].frame_index, pair[1].image_id),
    )
    ordered_paths = [Path(path) for path, _ in pairs]
    ordered = [record for _, record in pairs]
    windows = _build_windows(ordered, config)
    if not windows:
        raise ValueError(
            f"No episodes from {len(ordered)} records with context={config.context_size} "
            f"and horizon={config.horizon}"
        )

    feature_config = WindowFeatureConfig(
        peak_min_distance=config.peak_min_distance,
        peak_min_score=config.peak_min_score,
        peak_border=config.peak_border,
        max_candidates_per_image=config.max_candidates_per_image,
    )
    extractor = WindowFeatureExtractor(feature_config)

    feature_rows: list[np.ndarray] = []
    oracle_rows: list[np.ndarray] = []
    background_rows: list[np.ndarray] = []
    background_fraction_rows: list[float] = []
    object_rows: list[np.ndarray] = []
    hard_risk_rows: list[np.ndarray] = []
    hard_pd_rows: list[np.ndarray] = []
    domain_rows: list[str] = []
    sequence_rows: list[str] = []
    support_id_rows: list[str] = []
    query_id_rows: list[str] = []
    query_path_rows: list[str] = []
    distance_rows: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    source_payload = None

    for episode_index, window in enumerate(windows):
        support_records = [ordered[index] for index in window.context_indices]
        support_ids = [record.image_id for record in support_records]
        query_ids = [ordered[index].image_id for index in window.future_indices]
        if set(support_ids).intersection(query_ids):
            raise RuntimeError("Support and query IDs overlap")
        features, names = extractor.extract(support_records)
        if feature_names is None:
            feature_names = names
            source_payload = _load_source_reference(source_reference, feature_names)
        elif names != feature_names:
            raise RuntimeError("Feature schema changed between episodes")

        # Query labels are opened only after the support features are finalised.
        query_records = [
            load_score_record(ordered_paths[index], require_mask=True, load_mask=True)
            for index in window.future_indices
        ]
        image_curves = [
            compute_image_curves(
                record.probability,
                record.mask,
                threshold_array,
                peak_min_distance=config.peak_min_distance,
                peak_min_score=config.peak_min_score,
                peak_border=config.peak_border,
                peak_tolerance=config.peak_tolerance,
                max_candidates=config.max_candidates_per_image,
            )
            for record in query_records
        ]
        rates = rates_from_counts(aggregate_curve_counts(image_curves))
        background, background_fraction, max_query_logit = _sample_background_logits(
            query_records,
            limit=config.background_sample_limit,
            seed=config.seed + 104729 * episode_index,
        )
        objects = _query_object_scores(query_records, config.object_top_fraction)
        empty_logit = max_query_logit + 1.0
        oracle = oracle_threshold_logits(
            threshold_array,
            rates["pixel_false_rate"],
            rates["pd"],
            budget_array,
            empty_logit=empty_logit,
        )

        feature_rows.append(features.astype(np.float32))
        oracle_rows.append(oracle)
        background_rows.append(background)
        background_fraction_rows.append(background_fraction)
        object_rows.append(objects)
        hard_risk_rows.append(rates["pixel_false_rate"].astype(np.float32))
        hard_pd_rows.append(rates["pd"].astype(np.float32))
        domain_rows.append(query_records[0].dataset_name)
        sequence_rows.append(window.sequence_id)
        support_id_rows.append(json.dumps(support_ids, ensure_ascii=False))
        query_id_rows.append(json.dumps(query_ids, ensure_ascii=False))
        query_path_rows.append(
            json.dumps(
                [str(ordered_paths[index].resolve().relative_to(score_root)) for index in window.future_indices],
                ensure_ascii=False,
            )
        )
        if source_payload is not None:
            centres, scale, _ = source_payload
            normalised = (features[None, :] - centres) / scale[None, :]
            distance_rows.append(np.sqrt(np.mean(normalised**2, axis=1)).astype(np.float32))

    assert feature_names is not None
    background_values = _pad_rows(background_rows, fill=0.0, dtype=np.float32)
    background_valid = _pad_rows(
        [np.ones(len(row), dtype=np.uint8) for row in background_rows],
        fill=0,
        dtype=np.uint8,
    ).astype(bool)
    object_values = _pad_rows(object_rows, fill=0.0, dtype=np.float32)
    object_valid = _pad_rows(
        [np.ones(len(row), dtype=np.uint8) for row in object_rows],
        fill=0,
        dtype=np.uint8,
    ).astype(bool)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "features": np.stack(feature_rows).astype(np.float32),
        "budgets": budget_array.astype(np.float32),
        "oracle_threshold_logit": np.stack(oracle_rows).astype(np.float32),
        "background_logits": background_values,
        "background_valid": background_valid.astype(np.uint8),
        "background_fraction": np.asarray(background_fraction_rows, dtype=np.float32),
        "object_scores": object_values,
        "object_valid": object_valid.astype(np.uint8),
        "hard_thresholds": threshold_array.astype(np.float32),
        "hard_pixel_risk": np.stack(hard_risk_rows).astype(np.float32),
        "hard_pd": np.stack(hard_pd_rows).astype(np.float32),
        "domains": np.asarray(domain_rows, dtype=np.str_),
        "sequences": np.asarray(sequence_rows, dtype=np.str_),
        "support_ids": np.asarray(support_id_rows, dtype=np.str_),
        "query_ids": np.asarray(query_id_rows, dtype=np.str_),
        "query_paths": np.asarray(query_path_rows, dtype=np.str_),
        "feature_names": np.asarray(feature_names, dtype=np.str_),
        "feature_config_json": np.asarray(json.dumps(feature_config.to_dict(), sort_keys=True)),
        "build_config_json": np.asarray(json.dumps(asdict(config), sort_keys=True)),
        "score_roots": np.asarray([str(score_root)] * len(feature_rows), dtype=np.str_),
        "artifact_type": np.asarray("grouped_multi_budget_query_risk_meta_episode_v1"),
        "support_label_access": np.asarray("forbidden_and_not_loaded"),
        "query_label_access": np.asarray("meta_training_loss_and_exact_replay_only"),
        "split_role": np.asarray(config.split_role),
    }
    if distance_rows:
        distances = _pad_rows(distance_rows, fill=0.0, dtype=np.float32)
        payload["source_distances"] = distances
        payload["source_distance_valid"] = np.ones_like(distances, dtype=np.uint8)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_meta_episode_file(path: str | Path) -> MetaEpisodeArrays:
    with np.load(path, allow_pickle=False) as payload:
        source_distances = (
            np.asarray(payload["source_distances"], dtype=np.float32)
            if "source_distances" in payload
            else None
        )
        source_valid = (
            np.asarray(payload["source_distance_valid"], dtype=np.uint8).astype(bool)
            if "source_distance_valid" in payload
            else None
        )
        return MetaEpisodeArrays(
            features=np.asarray(payload["features"], dtype=np.float32),
            budgets=np.asarray(payload["budgets"], dtype=np.float32),
            oracle_threshold_logit=np.asarray(payload["oracle_threshold_logit"], dtype=np.float32),
            background_logits=np.asarray(payload["background_logits"], dtype=np.float32),
            background_valid=np.asarray(payload["background_valid"], dtype=np.uint8).astype(bool),
            background_fraction=np.asarray(payload["background_fraction"], dtype=np.float32),
            object_scores=np.asarray(payload["object_scores"], dtype=np.float32),
            object_valid=np.asarray(payload["object_valid"], dtype=np.uint8).astype(bool),
            hard_thresholds=np.asarray(payload["hard_thresholds"], dtype=np.float32),
            hard_pixel_risk=np.asarray(payload["hard_pixel_risk"], dtype=np.float32),
            hard_pd=np.asarray(payload["hard_pd"], dtype=np.float32),
            domains=np.asarray(payload["domains"]).astype(str),
            sequences=np.asarray(payload["sequences"]).astype(str),
            support_ids=np.asarray(payload["support_ids"]).astype(str),
            query_ids=np.asarray(payload["query_ids"]).astype(str),
            query_paths=np.asarray(payload["query_paths"]).astype(str),
            feature_names=tuple(np.asarray(payload["feature_names"]).astype(str).tolist()),
            feature_config=json.loads(str(np.asarray(payload["feature_config_json"]).item())),
            build_config=json.loads(str(np.asarray(payload["build_config_json"]).item())),
            source_distances=source_distances,
            source_distance_valid=source_valid,
            score_roots=(
                np.asarray(payload["score_roots"]).astype(str)
                if "score_roots" in payload
                else np.asarray([str(np.asarray(payload["score_root"]).item())] * len(payload["features"]), dtype=np.str_)
            ),
        )


def _pad_matrix_width(array: np.ndarray, width: int, fill=0) -> np.ndarray:
    if array.shape[1] == width:
        return array
    result = np.full((array.shape[0], width), fill, dtype=array.dtype)
    result[:, : array.shape[1]] = array
    return result


def concatenate_meta_episode_files(paths: Sequence[str | Path]) -> MetaEpisodeArrays:
    arrays = [load_meta_episode_file(path) for path in paths]
    if not arrays:
        raise ValueError("At least one meta-episode file is required")
    reference = arrays[0]
    for current in arrays[1:]:
        if not np.array_equal(current.budgets, reference.budgets):
            raise ValueError("Meta files use different budget grids")
        if not np.array_equal(current.hard_thresholds, reference.hard_thresholds):
            raise ValueError("Meta files use different hard threshold grids")
        if current.feature_names != reference.feature_names:
            raise ValueError("Meta files use different feature schemas")
        if current.feature_config != reference.feature_config:
            raise ValueError("Meta files use different feature configurations")
    max_bg = max(item.background_logits.shape[1] for item in arrays)
    max_obj = max(item.object_scores.shape[1] for item in arrays)
    has_source = [item.source_distances is not None for item in arrays]
    if any(has_source) and not all(has_source):
        raise ValueError("Meta files disagree on source-distance availability")
    max_source = (
        max(item.source_distances.shape[1] for item in arrays if item.source_distances is not None)
        if all(has_source)
        else 0
    )
    return MetaEpisodeArrays(
        features=np.concatenate([item.features for item in arrays]),
        budgets=reference.budgets,
        oracle_threshold_logit=np.concatenate([item.oracle_threshold_logit for item in arrays]),
        background_logits=np.concatenate(
            [_pad_matrix_width(item.background_logits, max_bg, 0.0) for item in arrays]
        ),
        background_valid=np.concatenate(
            [_pad_matrix_width(item.background_valid, max_bg, False) for item in arrays]
        ),
        background_fraction=np.concatenate([item.background_fraction for item in arrays]),
        object_scores=np.concatenate(
            [_pad_matrix_width(item.object_scores, max_obj, 0.0) for item in arrays]
        ),
        object_valid=np.concatenate(
            [_pad_matrix_width(item.object_valid, max_obj, False) for item in arrays]
        ),
        hard_thresholds=reference.hard_thresholds,
        hard_pixel_risk=np.concatenate([item.hard_pixel_risk for item in arrays]),
        hard_pd=np.concatenate([item.hard_pd for item in arrays]),
        domains=np.concatenate([item.domains for item in arrays]),
        sequences=np.concatenate([item.sequences for item in arrays]),
        support_ids=np.concatenate([item.support_ids for item in arrays]),
        query_ids=np.concatenate([item.query_ids for item in arrays]),
        query_paths=np.concatenate([item.query_paths for item in arrays]),
        feature_names=reference.feature_names,
        feature_config=dict(reference.feature_config),
        build_config={"concatenated_files": [str(Path(path)) for path in paths]},
        source_distances=(
            np.concatenate(
                [
                    _pad_matrix_width(item.source_distances, max_source, 0.0)
                    for item in arrays
                    if item.source_distances is not None
                ]
            )
            if all(has_source)
            else None
        ),
        source_distance_valid=(
            np.concatenate(
                [
                    _pad_matrix_width(item.source_distance_valid, max_source, False)
                    for item in arrays
                    if item.source_distance_valid is not None
                ]
            )
            if all(has_source)
            else None
        ),
        score_roots=np.concatenate([
            item.score_roots if item.score_roots is not None else np.asarray([""] * len(item.features), dtype=np.str_)
            for item in arrays
        ]),
    )


class MultiBudgetMetaDataset(Dataset[dict[str, torch.Tensor]]):
    """One dataset item per support/query episode, never per scalar budget."""

    def __init__(
        self,
        arrays: MetaEpisodeArrays,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        self.arrays = arrays
        mean = np.asarray(feature_mean, dtype=np.float32)
        std = np.maximum(np.asarray(feature_std, dtype=np.float32), 1e-6)
        self.features = ((arrays.features - mean) / std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            "features": torch.from_numpy(self.features[index]),
            "budgets": torch.from_numpy(self.arrays.budgets),
            "oracle_threshold_logit": torch.from_numpy(
                self.arrays.oracle_threshold_logit[index]
            ),
            "background_logits": torch.from_numpy(self.arrays.background_logits[index]),
            "background_valid": torch.from_numpy(self.arrays.background_valid[index]),
            "background_fraction": torch.tensor(
                self.arrays.background_fraction[index], dtype=torch.float32
            ),
            "object_scores": torch.from_numpy(self.arrays.object_scores[index]),
            "object_valid": torch.from_numpy(self.arrays.object_valid[index]),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.arrays.source_distances is not None:
            assert self.arrays.source_distance_valid is not None
            item["source_distances"] = torch.from_numpy(self.arrays.source_distances[index])
            item["source_distance_valid"] = torch.from_numpy(
                self.arrays.source_distance_valid[index]
            )
        return item
