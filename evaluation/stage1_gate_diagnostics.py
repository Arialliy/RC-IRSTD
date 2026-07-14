"""Development-only raw-logit diagnostics for paired Stage-1 D0/D3 runs.

The command consumes only score manifests whose frozen split role is
``detector_diagnostic`` and whose opt-in raw-logit artifacts are fully bound by
path, SHA-256, dtype, shape and checkpoint provenance.  Official-test score
manifests are rejected before any score or raw-logit artifact is analysed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data_ext.dataset_identity import ordered_sample_ids_sha256, sha256_file
from data_ext.development_split_contract import DETECTOR_DIAGNOSTIC_ROLE
from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts


DIAGNOSTIC_SCHEMA_VERSION = "rc-irstd.stage1-raw-logit-diagnostic.v1"
RAW_LOGIT_ARTIFACT_SCHEMA_VERSION = 1
RAW_LOGIT_CONTENT_ALGORITHM = (
    "sha256-length-prefixed-image-raw-logit-dtype-shape-v1"
)
RAW_LOGIT_DTYPE = "float64"
RAW_LOGIT_SPACE = "native_original_hw_spatially_aligned_restored_model_logit"
RAW_LOGIT_SCORE_RELATION = (
    "score = restore(float64_sigmoid(model_grid_logit)); raw_logit = "
    "restore(float64(model_grid_logit)); sigmoid(raw_logit) is "
    "diagnostic_only_not_pointwise_equal"
)
QUANTILES = (0.0, 0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999, 1.0)
_SIGMOID_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class RawLogitItem:
    image_id: str
    path: Path
    sha256: str
    dtype: str
    shape: tuple[int, int]
    gray_file_sha256: str


@dataclass(frozen=True)
class VerifiedStage1Manifest:
    path: Path
    manifest_sha256: str
    payload: Mapping[str, Any]
    variant: str
    items: tuple[RawLogitItem, ...]
    raw_logit_content_sha256: str


def raw_logit_manifest_content_sha256(
    items: Sequence[Mapping[str, object]],
) -> str:
    """Recompute the ordered raw-logit aggregate recorded by the exporter."""

    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("raw-logit manifest items must be a non-empty ordered list")
    digest = hashlib.sha256()
    _update_hash_frame(digest, RAW_LOGIT_CONTENT_ALGORITHM)
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"raw-logit manifest item {index} must be a mapping")
        image_id = _nonempty_string(item.get("image_id"), f"items[{index}].image_id")
        if image_id in seen:
            raise ValueError(f"duplicate raw-logit image_id: {image_id!r}")
        seen.add(image_id)
        file_value = _relative_path_string(
            item.get("raw_logit_file"), f"items[{index}].raw_logit_file"
        )
        raw_sha = _sha256_value(
            item.get("raw_logit_file_sha256"),
            f"items[{index}].raw_logit_file_sha256",
        )
        dtype = _nonempty_string(
            item.get("raw_logit_dtype"), f"items[{index}].raw_logit_dtype"
        )
        if dtype != RAW_LOGIT_DTYPE:
            raise ValueError(
                f"items[{index}].raw_logit_dtype must equal {RAW_LOGIT_DTYPE!r}"
            )
        height, width = _parse_shape(
            item.get("raw_logit_shape"), f"items[{index}].raw_logit_shape"
        )
        for value in (
            image_id,
            file_value,
            raw_sha,
            dtype,
            str(height),
            str(width),
        ):
            _update_hash_frame(digest, value)
    return digest.hexdigest()


def load_verified_stage1_manifest(
    manifest_path: str | Path,
    *,
    expected_variant: str,
) -> VerifiedStage1Manifest:
    """Load a D0/D3 manifest and fail closed on scope or artifact ambiguity."""

    if expected_variant not in {"D0", "D3"}:
        raise ValueError("expected_variant must be 'D0' or 'D3'")
    verified = verify_score_manifest_artifacts(
        manifest_path,
        required_split_role=DETECTOR_DIAGNOSTIC_ROLE,
    )
    payload = verified.payload
    _require_exact_scope(payload)
    if payload.get("raw_logits_exported") is not True:
        raise ValueError("score manifest does not contain an enabled raw-logit export")
    if payload.get("raw_logit_artifact_schema_version") != (
        RAW_LOGIT_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported raw-logit artifact schema version")
    if payload.get("raw_logit_content_sha256_algorithm") != (
        RAW_LOGIT_CONTENT_ALGORITHM
    ):
        raise ValueError("raw-logit content SHA-256 algorithm mismatch")
    if payload.get("raw_logit_dtype") != RAW_LOGIT_DTYPE:
        raise ValueError("raw-logit manifest dtype must be float64")
    if payload.get("raw_logit_space") != RAW_LOGIT_SPACE:
        raise ValueError("raw-logit coordinate/semantic space mismatch")
    if payload.get("raw_logit_score_relation") != RAW_LOGIT_SCORE_RELATION:
        raise ValueError("raw-logit/final-score relation contract mismatch")

    weight_value = _relative_path_string(
        payload.get("weight_path"), "score manifest weight_path"
    )
    weight_path = (verified.path.parent / weight_value).resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"score manifest detector checkpoint is missing: {weight_path}")
    declared_weight_sha = _sha256_value(
        payload.get("weight_sha256"), "score manifest weight_sha256"
    )
    if sha256_file(weight_path) != declared_weight_sha:
        raise ValueError("score manifest detector checkpoint SHA-256 mismatch")

    raw_provenance = payload.get("raw_logit_provenance")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("score manifest is missing raw_logit_provenance")
    _verify_raw_provenance(raw_provenance, payload, expected_variant)

    raw_items = list(verified.items)
    calculated_content_sha = raw_logit_manifest_content_sha256(raw_items)
    declared_content_sha = _sha256_value(
        payload.get("raw_logit_content_sha256"), "raw_logit_content_sha256"
    )
    if declared_content_sha != calculated_content_sha:
        raise ValueError(
            "raw_logit_content_sha256 does not match ordered raw-logit artifacts"
        )

    root = verified.path.parent
    resolved_items: list[RawLogitItem] = []
    for index, item in enumerate(raw_items):
        image_id = _nonempty_string(item.get("image_id"), f"items[{index}].image_id")
        relative = _relative_path_string(
            item.get("raw_logit_file"), f"items[{index}].raw_logit_file"
        )
        raw_path = (root / relative).resolve()
        try:
            raw_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"raw-logit path escapes the manifest directory: {relative!r}"
            ) from error
        if raw_path.suffix.lower() != ".npy":
            raise ValueError(f"raw-logit artifact must be .npy: {raw_path}")
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing raw-logit artifact: {raw_path}")
        declared_sha = _sha256_value(
            item.get("raw_logit_file_sha256"),
            f"items[{index}].raw_logit_file_sha256",
        )
        before = sha256_file(raw_path)
        if before != declared_sha:
            raise ValueError(f"raw-logit SHA-256 mismatch for {image_id!r}")
        declared_dtype = _nonempty_string(
            item.get("raw_logit_dtype"), f"items[{index}].raw_logit_dtype"
        )
        declared_shape = _parse_shape(
            item.get("raw_logit_shape"), f"items[{index}].raw_logit_shape"
        )
        original_hw = _parse_shape(
            item.get("original_hw"), f"items[{index}].original_hw"
        )
        if declared_shape != original_hw:
            raise ValueError(
                f"raw-logit/original shape mismatch for {image_id!r}: "
                f"{declared_shape} != {original_hw}"
            )
        array = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.ndim != 2:
            raise ValueError(f"raw-logit .npy must contain one 2D array: {raw_path}")
        if str(array.dtype) != declared_dtype or declared_dtype != RAW_LOGIT_DTYPE:
            raise ValueError(f"raw-logit dtype mismatch for {image_id!r}")
        if tuple(int(value) for value in array.shape) != declared_shape:
            raise ValueError(f"raw-logit shape mismatch for {image_id!r}")
        if not np.isfinite(array).all():
            raise ValueError(f"raw-logit artifact contains NaN/Inf: {raw_path}")
        del array
        after = sha256_file(raw_path)
        if after != before:
            raise RuntimeError(f"raw-logit artifact changed while verified: {raw_path}")
        resolved_items.append(
            RawLogitItem(
                image_id=image_id,
                path=raw_path,
                sha256=declared_sha,
                dtype=declared_dtype,
                shape=declared_shape,
                gray_file_sha256=_sha256_value(
                    item.get("gray_file_sha256"),
                    f"items[{index}].gray_file_sha256",
                ),
            )
        )

    return VerifiedStage1Manifest(
        path=verified.path,
        manifest_sha256=verified.manifest_sha256,
        payload=payload,
        variant=expected_variant,
        items=tuple(resolved_items),
        raw_logit_content_sha256=calculated_content_sha,
    )


def validate_paired_stage1_manifests(
    d0: VerifiedStage1Manifest,
    d3: VerifiedStage1Manifest,
) -> dict[str, object]:
    """Require exact sample and protocol pairing while allowing model changes."""

    if d0.variant != "D0" or d3.variant != "D3":
        raise ValueError("paired diagnostics require D0 first and D3 second")
    d0_ids = [item.image_id for item in d0.items]
    d3_ids = [item.image_id for item in d3.items]
    if d0_ids != d3_ids:
        raise ValueError("D0/D3 image IDs or their order do not match")
    if not d0_ids:
        raise ValueError("D0/D3 paired diagnostic manifests are empty")
    for left, right in zip(d0.items, d3.items):
        if left.gray_file_sha256 != right.gray_file_sha256:
            raise ValueError(
                f"D0/D3 source-image provenance mismatch for {left.image_id!r}"
            )
        if left.shape != right.shape:
            raise ValueError(f"D0/D3 raw-logit shape mismatch for {left.image_id!r}")

    matched_fields = (
        "target_dataset",
        "outer_fold_id",
        "outer_target",
        "detector_source_domains",
        "held_out_domains",
        "checkpoint_selection",
        "protocol_scope",
        "detector_source_records",
    )
    for field in matched_fields:
        if d0.payload.get(field) != d3.payload.get(field):
            raise ValueError(f"D0/D3 provenance mismatch for {field}")

    _require_matching_mapping_fields(
        d0.payload.get("target_dataset_record"),
        d3.payload.get("target_dataset_record"),
        "target_dataset_record",
        (
            "dataset_identity_sha256",
            "split_sha256",
            "ordered_sample_ids_sha256",
            "split_image_artifact_sha256",
            "split_image_artifact_items",
        ),
    )
    _require_matching_mapping_fields(
        d0.payload.get("split_contract"),
        d3.payload.get("split_contract"),
        "split_contract",
        (
            "role",
            "selected_split_sha256",
            "selected_num_images",
            "selected_ids_sha256",
            "derived_split_manifest_sha256",
            "partition_scope",
        ),
    )
    d0_raw = _mapping(d0.payload.get("raw_logit_provenance"), "D0 raw provenance")
    d3_raw = _mapping(d3.payload.get("raw_logit_provenance"), "D3 raw provenance")
    for field in (
        "proof_mode",
        "training_seed",
        "segmentation_loss_implementation",
        "same_dataset_partition_audit",
    ):
        if d0_raw.get(field) != d3_raw.get(field):
            raise ValueError(f"D0/D3 raw-logit provenance mismatch for {field}")

    d0_weight = _sha256_value(d0.payload.get("weight_sha256"), "D0 weight_sha256")
    d3_weight = _sha256_value(d3.payload.get("weight_sha256"), "D3 weight_sha256")
    return {
        "pair_checks_passed": True,
        "matched_image_count": len(d0_ids),
        "matched_image_ids_sha256": ordered_sample_ids_sha256(d0_ids),
        "matched_provenance_fields": list(matched_fields),
        "d0_detector_weight_sha256": d0_weight,
        "d3_detector_weight_sha256": d3_weight,
        "detector_weights_distinct": d0_weight != d3_weight,
        "development_partition_role": DETECTOR_DIAGNOSTIC_ROLE,
        "official_test_scores_consumed": False,
    }


def summarise_raw_logits(items: Sequence[RawLogitItem]) -> dict[str, object]:
    """Compute exact global raw/sigmoid quantiles and tie sizes."""

    if not items:
        raise ValueError("raw-logit summary requires at least one item")
    total = sum(int(np.prod(item.shape, dtype=np.int64)) for item in items)
    if total <= 0:
        raise ValueError("raw-logit summary has no pixels")
    values = np.empty(total, dtype=np.float64)
    offset = 0
    for item in items:
        array = _load_item_array(item)
        count = int(array.size)
        values[offset : offset + count] = array.reshape(-1)
        offset += count
    if offset != total:
        raise RuntimeError("raw-logit pixel count changed during collection")

    raw_min = float(np.min(values))
    raw_max = float(np.max(values))
    raw_mean = float(np.mean(values, dtype=np.float64))
    raw_std = float(np.std(values, dtype=np.float64))
    values.sort(kind="quicksort")
    raw_quantiles = _quantiles_from_sorted(values, QUANTILES)
    raw_tie = _maximum_tie_from_sorted(values)

    _sigmoid_float64_in_place(values)
    exact_zero_count = int(np.count_nonzero(values == 0.0))
    exact_one_count = int(np.count_nonzero(values == 1.0))
    values.sort(kind="quicksort")
    sigmoid_tie = _maximum_tie_from_sorted(values)
    sigmoid_quantiles = _quantiles_from_sorted(values, QUANTILES)
    del values

    return {
        "num_images": len(items),
        "num_pixels": total,
        "dtype": RAW_LOGIT_DTYPE,
        "raw_logit": {
            "min": raw_min,
            "max": raw_max,
            "range": raw_max - raw_min,
            "mean": raw_mean,
            "std": raw_std,
            "quantiles": raw_quantiles,
            "maximum_tie": raw_tie,
        },
        "float64_sigmoid_of_raw_logit_diagnostic": {
            "pointwise_equal_to_saved_score": False,
            "exact_zero_count": exact_zero_count,
            "exact_zero_fraction": exact_zero_count / total,
            "exact_one_count": exact_one_count,
            "exact_one_fraction": exact_one_count / total,
            "quantiles": sigmoid_quantiles,
            "maximum_tie": sigmoid_tie,
        },
    }


def fit_and_compare_raw_logits(
    d0_items: Sequence[RawLogitItem],
    d3_items: Sequence[RawLogitItem],
) -> dict[str, object]:
    """Fit ``D3 = a * D0 + b`` and measure deterministic ranking changes."""

    if len(d0_items) != len(d3_items) or not d0_items:
        raise ValueError("paired raw-logit analysis requires equal non-empty items")
    count = 0
    mean_x = 0.0
    mean_y = 0.0
    m2_x = 0.0
    m2_y = 0.0
    covariance = 0.0
    rank_changed_pixels = 0
    max_rank_displacement = 0
    images_with_rank_change = 0
    spearman_values: list[float] = []
    top_tail_overlaps: dict[str, list[float]] = {"0.001": [], "0.01": []}

    for d0_item, d3_item in zip(d0_items, d3_items):
        if d0_item.image_id != d3_item.image_id or d0_item.shape != d3_item.shape:
            raise ValueError("paired raw-logit items differ in ID or shape")
        x = _load_item_array(d0_item).reshape(-1)
        y = _load_item_array(d3_item).reshape(-1)
        (
            count,
            mean_x,
            mean_y,
            m2_x,
            m2_y,
            covariance,
        ) = _merge_pair_moments(
            count,
            mean_x,
            mean_y,
            m2_x,
            m2_y,
            covariance,
            x,
            y,
        )

        order_x = np.argsort(x, kind="stable")
        order_y = np.argsort(y, kind="stable")
        ranks_x = np.empty(x.size, dtype=np.int64)
        ranks_y = np.empty(y.size, dtype=np.int64)
        ordinal = np.arange(x.size, dtype=np.int64)
        ranks_x[order_x] = ordinal
        ranks_y[order_y] = ordinal
        displacement = np.abs(ranks_x - ranks_y)
        changed = int(np.count_nonzero(displacement))
        rank_changed_pixels += changed
        if changed:
            images_with_rank_change += 1
        if displacement.size:
            max_rank_displacement = max(
                max_rank_displacement, int(np.max(displacement))
            )
        spearman_values.append(_ordinal_spearman(ranks_x, ranks_y))
        for rendered_fraction, values_for_fraction in top_tail_overlaps.items():
            fraction = float(rendered_fraction)
            k = max(1, int(math.ceil(x.size * fraction)))
            overlap = np.intersect1d(
                order_x[-k:], order_y[-k:], assume_unique=True
            ).size
            values_for_fraction.append(float(overlap / k))

    if count <= 1 or m2_x <= 0.0:
        raise ValueError("D0 raw logits have insufficient variance for linear fit")
    scale = covariance / m2_x
    shift = mean_y - scale * mean_x
    sse = max(m2_y - 2.0 * scale * covariance + scale * scale * m2_x, 0.0)
    if m2_y > 0.0:
        r2 = 1.0 - sse / m2_y
    else:
        r2 = 1.0 if sse == 0.0 else 0.0
    rmse = math.sqrt(sse / count)

    max_abs_residual = 0.0
    for d0_item, d3_item in zip(d0_items, d3_items):
        x = _load_item_array(d0_item)
        y = _load_item_array(d3_item)
        residual = y - (scale * x + shift)
        max_abs_residual = max(max_abs_residual, float(np.max(np.abs(residual))))

    return {
        "num_images": len(d0_items),
        "num_pixels": count,
        "linear_fit_d3_from_d0": {
            "model": "D3 = scale_a * D0 + shift_b",
            "scale_a": float(scale),
            "shift_b": float(shift),
            "r2": float(r2),
            "rmse": float(rmse),
            "max_abs_residual": float(max_abs_residual),
        },
        "ranking_change": {
            "definition": "stable ordinal per-image ranks in native pixel space",
            "rank_changed_pixel_count": rank_changed_pixels,
            "rank_changed_pixel_fraction": rank_changed_pixels / count,
            "images_with_any_rank_change": images_with_rank_change,
            "ordering_identical_for_all_images": images_with_rank_change == 0,
            "maximum_absolute_rank_displacement": max_rank_displacement,
            "stable_ordinal_spearman_mean": float(np.mean(spearman_values)),
            "stable_ordinal_spearman_min": float(np.min(spearman_values)),
            "top_tail_overlap_fraction_mean": {
                key: float(np.mean(values))
                for key, values in top_tail_overlaps.items()
            },
        },
    }


def diagnose_stage1_gate(
    d0_manifest_path: str | Path,
    d3_manifest_path: str | Path,
) -> dict[str, object]:
    """Verify, pair and analyse two development-only raw-logit manifests."""

    d0 = load_verified_stage1_manifest(d0_manifest_path, expected_variant="D0")
    d3 = load_verified_stage1_manifest(d3_manifest_path, expected_variant="D3")
    provenance = validate_paired_stage1_manifests(d0, d3)
    paired = fit_and_compare_raw_logits(d0.items, d3.items)
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_type": "stage1_d0_d3_raw_logit_gate_diagnostic",
        "development_only": True,
        "official_test_scores_consumed": False,
        "claim_bearing": False,
        "provenance": provenance,
        "variants": {
            "D0": {
                "manifest_path": str(d0.path),
                "manifest_sha256": d0.manifest_sha256,
                "raw_logit_content_sha256": d0.raw_logit_content_sha256,
                "statistics": summarise_raw_logits(d0.items),
            },
            "D3": {
                "manifest_path": str(d3.path),
                "manifest_sha256": d3.manifest_sha256,
                "raw_logit_content_sha256": d3.raw_logit_content_sha256,
                "statistics": summarise_raw_logits(d3.items),
            },
        },
        "paired": paired,
    }


def _require_exact_scope(payload: Mapping[str, Any]) -> None:
    expected = {
        "development_only": True,
        "official_test_artifact": False,
        "final_evaluation_eligible": False,
        "claim_bearing_final_evaluation": False,
    }
    for field, value in expected.items():
        if payload.get(field) is not value:
            raise ValueError(f"development manifest {field} must be exactly {value!r}")
    split_contract = _mapping(payload.get("split_contract"), "split_contract")
    if split_contract.get("role") != DETECTOR_DIAGNOSTIC_ROLE:
        raise ValueError("Stage-1 diagnostics reject non-development split roles")


def _verify_raw_provenance(
    provenance: Mapping[str, Any],
    payload: Mapping[str, Any],
    expected_variant: str,
) -> None:
    expected = {
        "schema_version": RAW_LOGIT_ARTIFACT_SCHEMA_VERSION,
        "status": "verified",
        "eligible": True,
        "split_role": DETECTOR_DIAGNOSTIC_ROLE,
        "development_only": True,
        "official_test_scores_consumed": False,
        "checkpoint_provenance_level": "checkpoint_verified",
        "stage1_variant": expected_variant,
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(
                f"raw-logit provenance field {field!r} must equal {value!r}"
            )
    if payload.get("provenance_level") != "checkpoint_verified":
        raise ValueError("raw-logit manifest detector provenance is not verified")
    risk = _mapping(provenance.get("risk_objective_contract"), "risk contract")
    if risk.get("stage1_variant") != expected_variant:
        raise ValueError("raw-logit risk contract Stage-1 variant mismatch")
    if not isinstance(provenance.get("segmentation_loss_implementation"), Mapping):
        raise ValueError("raw-logit provenance lacks segmentation-loss identity")
    proof_mode = provenance.get("proof_mode")
    if proof_mode == "verified_held_out_domain_exclusion":
        if payload.get("target_exclusion_verified") is not True:
            raise ValueError("held-out raw-logit provenance lacks target exclusion")
    elif proof_mode == "verified_same_dataset_detector_fit_disjointness":
        audit = _mapping(
            provenance.get("same_dataset_partition_audit"),
            "same-dataset partition audit",
        )
        if (
            audit.get("sample_id_overlap_count") != 0
            or audit.get("image_content_overlap_count") != 0
            or audit.get("disjointness_verified") is not True
        ):
            raise ValueError("same-dataset raw-logit partition audit is not disjoint")
    else:
        raise ValueError("unsupported raw-logit provenance proof_mode")


def _require_matching_mapping_fields(
    left: object,
    right: object,
    name: str,
    fields: Iterable[str],
) -> None:
    left_mapping = _mapping(left, f"D0 {name}")
    right_mapping = _mapping(right, f"D3 {name}")
    for field in fields:
        if left_mapping.get(field) != right_mapping.get(field):
            raise ValueError(f"D0/D3 provenance mismatch for {name}.{field}")


def _load_item_array(item: RawLogitItem) -> np.ndarray:
    before = sha256_file(item.path)
    if before != item.sha256:
        raise ValueError(f"raw-logit SHA-256 mismatch for {item.image_id!r}")
    mapped = np.load(item.path, mmap_mode="r", allow_pickle=False)
    if str(mapped.dtype) != item.dtype or tuple(mapped.shape) != item.shape:
        raise ValueError(f"raw-logit dtype/shape changed for {item.image_id!r}")
    array = np.asarray(mapped, dtype=np.float64).copy()
    del mapped
    after = sha256_file(item.path)
    if after != before:
        raise RuntimeError(f"raw-logit changed while read: {item.path}")
    if not np.isfinite(array).all():
        raise ValueError(f"raw-logit contains NaN/Inf: {item.path}")
    return array


def _merge_pair_moments(
    count: int,
    mean_x: float,
    mean_y: float,
    m2_x: float,
    m2_y: float,
    covariance: float,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[int, float, float, float, float, float]:
    batch_count = int(x.size)
    if batch_count != int(y.size) or batch_count <= 0:
        raise ValueError("paired raw-logit batch sizes differ or are empty")
    batch_mean_x = float(np.mean(x, dtype=np.float64))
    batch_mean_y = float(np.mean(y, dtype=np.float64))
    centered_x = x - batch_mean_x
    centered_y = y - batch_mean_y
    batch_m2_x = float(np.dot(centered_x, centered_x))
    batch_m2_y = float(np.dot(centered_y, centered_y))
    batch_covariance = float(np.dot(centered_x, centered_y))
    if count == 0:
        return (
            batch_count,
            batch_mean_x,
            batch_mean_y,
            batch_m2_x,
            batch_m2_y,
            batch_covariance,
        )
    combined = count + batch_count
    delta_x = batch_mean_x - mean_x
    delta_y = batch_mean_y - mean_y
    cross_weight = count * batch_count / combined
    return (
        combined,
        mean_x + delta_x * batch_count / combined,
        mean_y + delta_y * batch_count / combined,
        m2_x + batch_m2_x + delta_x * delta_x * cross_weight,
        m2_y + batch_m2_y + delta_y * delta_y * cross_weight,
        covariance + batch_covariance + delta_x * delta_y * cross_weight,
    )


def _ordinal_spearman(ranks_x: np.ndarray, ranks_y: np.ndarray) -> float:
    count = int(ranks_x.size)
    if count != int(ranks_y.size) or count <= 0:
        raise ValueError("rank arrays must have the same non-zero size")
    if count == 1:
        return 1.0
    difference = ranks_x.astype(np.float64) - ranks_y.astype(np.float64)
    squared = float(np.dot(difference, difference))
    return float(1.0 - 6.0 * squared / (count * (count * count - 1.0)))


def _quantiles_from_sorted(
    sorted_values: np.ndarray,
    probabilities: Sequence[float],
) -> dict[str, float]:
    if sorted_values.ndim != 1 or sorted_values.size == 0:
        raise ValueError("quantiles require a non-empty sorted 1D array")
    result: dict[str, float] = {}
    last = sorted_values.size - 1
    for probability in probabilities:
        if probability < 0.0 or probability > 1.0:
            raise ValueError("quantile probability must lie in [0, 1]")
        position = probability * last
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            value = float(sorted_values[lower])
        else:
            weight = position - lower
            value = float(
                sorted_values[lower] * (1.0 - weight)
                + sorted_values[upper] * weight
            )
        result[format(probability, ".4g")] = value
    return result


def _maximum_tie_from_sorted(sorted_values: np.ndarray) -> dict[str, object]:
    if sorted_values.ndim != 1 or sorted_values.size == 0:
        raise ValueError("tie analysis requires a non-empty sorted 1D array")
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.concatenate((np.asarray([0], dtype=np.int64), boundaries))
    ends = np.concatenate(
        (boundaries, np.asarray([sorted_values.size], dtype=np.int64))
    )
    lengths = ends - starts
    index = int(np.argmax(lengths))
    count = int(lengths[index])
    return {
        "count": count,
        "fraction": count / int(sorted_values.size),
        "value": float(sorted_values[int(starts[index])]),
        "distinct_value_count": int(lengths.size),
    }


def _sigmoid_float64_in_place(values: np.ndarray) -> None:
    if values.dtype != np.float64 or values.ndim != 1:
        raise TypeError("in-place sigmoid requires a 1D float64 array")
    for start in range(0, values.size, _SIGMOID_CHUNK_SIZE):
        chunk = values[start : start + _SIGMOID_CHUNK_SIZE]
        positive = chunk >= 0.0
        if np.any(positive):
            positive_values = chunk[positive]
            chunk[positive] = 1.0 / (1.0 + np.exp(-positive_values))
        negative = ~positive
        if np.any(negative):
            exponent = np.exp(chunk[negative])
            chunk[negative] = exponent / (1.0 + exponent)


def _parse_shape(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} must contain [height, width]")
    array = np.asarray(value)
    if array.size != 2:
        raise ValueError(f"{name} must contain exactly two values")
    dimensions: list[int] = []
    for raw in array.reshape(-1):
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError(f"{name} values must be positive integers")
        try:
            integer = int(raw)
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} values must be positive integers") from error
        if not np.isfinite(numeric) or numeric != float(integer) or integer <= 0:
            raise ValueError(f"{name} values must be positive integers")
        dimensions.append(integer)
    return dimensions[0], dimensions[1]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _relative_path_string(value: object, name: str) -> str:
    rendered = _nonempty_string(value, name)
    if Path(rendered).expanduser().is_absolute():
        raise ValueError(f"{name} must be relative to the manifest directory")
    return rendered


def _sha256_value(value: object, name: str) -> str:
    rendered = _nonempty_string(value, name).lower()
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return rendered


def _update_hash_frame(digest: "hashlib._Hash", value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d0-manifest", required=True)
    parser.add_argument("--d3-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"diagnostic output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = diagnose_stage1_gate(args.d0_manifest, args.d3_manifest)
    _write_json_atomic(output, report)
    print(
        "Wrote development-only Stage-1 raw-logit diagnostics for "
        f"{report['paired']['num_images']} images to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
