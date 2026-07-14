"""Causal zero-label adaptation from a context prefix to a later query suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from data_ext.dataset_identity import sha256_file
from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts
from model.monotone_pixel_calibrator import (
    MonotonePixelRiskCalibrator,
    pixel_budget_from_spec,
)
from model.threshold_calibrator import ThresholdCalibrator

from .domain_statistics import (
    extract_unlabeled_statistics,
    load_probability_and_grayscale,
    load_source_reference,
)
from .meta_dataset import FeatureStandardizer
from .schema import (
    BudgetSpec,
    CAUSAL_PARTITION_RULE,
    DEFAULT_CENTROID_DISTANCE,
    DEFAULT_MATCHING_RULE,
    DeploymentProtocolContract,
    EVALUATION_MATCHING_CONTRACT_VERSION,
    FoldContract,
    ONLINE_DECISION_CONTRACT_VERSION,
    REJECT_COMPARISON_RULE,
    REJECT_SCORE_RULE,
    SCHEMA_VERSION,
    SourceReference,
    StatisticsConfig,
)


def causal_partition(
    records: Sequence[Mapping[str, Any]],
    *,
    context_size: int,
    query_size: int | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split an ordered stream at one strict context/query boundary."""

    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if context_size >= len(records):
        raise ValueError("at least one record must remain after the context boundary")
    if query_size is None:
        query_size = len(records) - context_size
    if query_size <= 0 or context_size + query_size > len(records):
        raise ValueError("query_size is outside the available post-context suffix")
    selected = list(records[: context_size + query_size])
    image_ids = [str(record["image_id"]) for record in selected]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("ordered score records must have unique image_id values")
    context = selected[:context_size]
    query = selected[context_size:]
    if {str(item["image_id"]) for item in context}.intersection(
        str(item["image_id"]) for item in query
    ):
        raise RuntimeError("internal context/query overlap")
    return context, query


def load_ordered_score_records(
    manifest_or_directory: str | Path,
    *,
    image_dir: str | Path | None = None,
    required_split_role: str | None = None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    source = Path(manifest_or_directory).expanduser().resolve()
    if image_dir is not None:
        raise ValueError(
            "audited score manifests bind explicit image_path values; "
            "--image-dir overrides are not supported"
        )
    if source.is_dir():
        manifest_path = source / "manifest.json"
        if manifest_path.exists():
            source = manifest_path
        else:
            records = [
                {"image_id": path.stem, "prob_path": str(path), "gray_path": None}
                for path in sorted(source.glob("*.npz"))
            ]
            if not records:
                raise FileNotFoundError(f"no score maps found under {source}")
            raise ValueError("audited online adaptation requires a score manifest, not a bare directory")
    verified = verify_score_manifest_artifacts(
        source,
        require_mask=False,
        require_native_contract=True,
        verify_artifact_bytes=False,
        required_split_role=required_split_role,
    )
    records = []
    for item in verified.items:
        score_value = item.get("file", item.get("prob_path", item.get("score_path")))
        if score_value is None:
            raise KeyError(f"score manifest item {item.get('image_id')!r} lacks a file")
        score_path = (verified.path.parent / str(score_value)).resolve()
        gray_path = (verified.path.parent / str(item["image_path"])).resolve()
        records.append(
            {
                "image_id": str(item["image_id"]),
                "prob_path": str(score_path),
                "gray_path": str(gray_path),
            }
        )
    return records, verified.payload


def _torch_load(path: str | Path, device: torch.device) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError("calibrator checkpoint must be a mapping")
    return payload


def _manifest_contract_value(payload: Mapping[str, Any], key: str) -> Any:
    nested = payload.get("detector_provenance")
    nested_present = isinstance(nested, Mapping) and key in nested
    top_present = key in payload
    if top_present and nested_present and payload[key] != nested[key]:
        raise ValueError(f"score manifest has conflicting top-level/nested {key}")
    if top_present:
        return payload[key]
    return nested[key] if nested_present else None


def _score_manifest_checkpoint_sha(payload: Mapping[str, Any]) -> str:
    values = []
    for key in (
        "weight_sha256",
        "detector_checkpoint_sha",
        "detector_weight_sha256",
        "checkpoint_sha256",
    ):
        value = _manifest_contract_value(payload, key)
        if value is not None:
            values.append(str(value).lower())
    if not values:
        raise KeyError("score manifest is missing detector checkpoint SHA-256")
    if len(set(values)) != 1:
        raise ValueError("score manifest contains conflicting detector checkpoint SHA fields")
    value = values[0]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("score manifest detector checkpoint SHA must be 64 hexadecimal characters")
    return value


def _deployment_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[FoldContract, SourceReference]:
    reference = SourceReference.from_dict(checkpoint["deployment_source_reference"])
    fold = FoldContract(
        outer_fold_id=str(checkpoint["outer_fold_id"]),
        outer_target=str(checkpoint["outer_target"]),
        detector_source_domains=tuple(
            str(value) for value in checkpoint["deployment_detector_source_domains"]
        ),
        detector_checkpoint_sha=str(checkpoint["deployment_detector_checkpoint_sha"]),
        held_out_domains=tuple(
            str(value) for value in checkpoint["deployment_held_out_domains"]
        ),
        protocol_scope=str(checkpoint["deployment_protocol_scope"]),
    )
    fold.assert_matches_source_reference(reference)
    if fold.protocol_scope != "multi_source_protocol_candidate":
        raise ValueError("calibrator deployment contract is not main-protocol eligible")
    return fold, reference


def _checkpoint_reject_cutoff(checkpoint: Mapping[str, Any]) -> float:
    """Resolve one cutoff and reject conflicting duplicated metadata."""

    values: list[tuple[str, float]] = []
    if "reject_probability" in checkpoint:
        values.append(
            ("reject_probability", float(checkpoint["reject_probability"]))
        )
    training_config = checkpoint.get("training_config")
    if isinstance(training_config, Mapping) and "reject_probability" in training_config:
        values.append(
            (
                "training_config.reject_probability",
                float(training_config["reject_probability"]),
            )
        )
    if not values:
        raise KeyError("calibrator checkpoint is missing the frozen reject cutoff")
    for name, value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"checkpoint {name} must lie in [0, 1]")
    cutoff = values[0][1]
    if any(
        not math.isclose(value, cutoff, rel_tol=0.0, abs_tol=1e-12)
        for _, value in values[1:]
    ):
        raise ValueError("calibrator checkpoint contains conflicting reject cutoffs")
    return cutoff


def _deployment_protocol_contract(
    checkpoint: Mapping[str, Any],
    *,
    required: bool,
) -> DeploymentProtocolContract | None:
    payload = checkpoint.get("deployment_protocol_contract")
    if payload is None:
        if required:
            raise ValueError(
                "claim-bearing online adaptation requires a frozen "
                "deployment_protocol_contract; this legacy checkpoint is diagnostic-only"
            )
        return None
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint deployment_protocol_contract must be a mapping")
    contract = DeploymentProtocolContract.from_dict(payload)
    checkpoint_cutoff = _checkpoint_reject_cutoff(checkpoint)
    if not math.isclose(
        contract.reject_cutoff,
        checkpoint_cutoff,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "deployment protocol reject cutoff conflicts with calibrator checkpoint"
        )
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, Mapping):
        raise TypeError("checkpoint training_config must be a mapping")
    if "evaluation_matching_rule" in training_config and str(
        training_config["evaluation_matching_rule"]
    ) != contract.matching_rule:
        raise ValueError(
            "deployment protocol matching rule conflicts with training_config"
        )
    if "evaluation_centroid_distance" in training_config and not math.isclose(
        float(training_config["evaluation_centroid_distance"]),
        contract.centroid_distance,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "deployment protocol centroid distance conflicts with training_config"
        )
    return contract


def _validated_monotone_budget_contract(
    checkpoint: Mapping[str, Any],
    model: MonotonePixelRiskCalibrator,
) -> dict[str, Any]:
    value = checkpoint.get("monotone_budget_contract")
    if not isinstance(value, Mapping):
        raise TypeError("schema-v4 checkpoint requires monotone_budget_contract")
    contract = dict(value)
    required = {
        "schema_version",
        "risk",
        "component_budget_supported",
        "grid",
        "grid_order",
        "grid_policy_sha256",
        "interpolation",
        "extrapolation_allowed",
        "curve_compute_dtype",
        "train_supervision",
        "validation_supervision",
    }
    missing = required.difference(contract)
    if missing:
        raise KeyError(
            f"monotone budget contract is missing: {sorted(missing)}"
        )
    if contract["schema_version"] != "rc-irstd.monotone-pixel-budget.v1":
        raise ValueError("unsupported monotone budget contract schema")
    if contract["risk"] != "fa_pixel" or contract["component_budget_supported"] is not False:
        raise ValueError("schema-v4 monotone contract must be pixel-only")
    grid = tuple(float(value) for value in contract["grid"])
    model_grid = tuple(float(value) for value in model.pixel_budget_grid.cpu())
    if grid != model_grid:
        raise ValueError("monotone budget contract grid differs from model_config")
    if (
        contract["grid_order"] != "loose_to_strict"
        or contract["interpolation"] != "piecewise_linear_log10"
        or contract["extrapolation_allowed"] is not False
        or contract["curve_compute_dtype"] != "float64"
    ):
        raise ValueError("unsupported monotone budget interpolation contract")
    canonical = json.dumps(
        {
            "risk": "fa_pixel",
            "grid": list(grid),
            "grid_order": "loose_to_strict",
            "interpolation": "piecewise_linear_log10",
            "extrapolation_allowed": False,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected_sha = hashlib.sha256(canonical).hexdigest()
    if contract["grid_policy_sha256"] != expected_sha:
        raise ValueError("monotone budget grid policy SHA-256 is invalid")
    for split in ("train_supervision", "validation_supervision"):
        supervision = contract[split]
        if not isinstance(supervision, Mapping) or supervision.get(
            "all_grid_points_supervised"
        ) is not True:
            raise ValueError(
                f"monotone budget contract lacks complete {split} grid coverage"
            )
    return contract


def load_calibrator_bundle(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, FeatureStandardizer, Mapping[str, Any]]:
    checkpoint = _torch_load(checkpoint_path, device)
    required = {
        "format_version",
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "dropout",
        "standardizer",
        "statistics_feature_names",
        "input_feature_names",
        "statistics_config",
        "p_min",
        "outer_fold_id",
        "outer_target",
        "episode_collection_provenance",
        "episode_collection_sha256",
        "training_config",
        "calibration_pseudo_targets",
        "deployment_detector_source_domains",
        "deployment_detector_checkpoint_sha",
        "deployment_held_out_domains",
        "deployment_protocol_scope",
        "deployment_source_reference",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"calibrator checkpoint is missing: {sorted(missing)}")
    format_version = str(checkpoint["format_version"])
    if format_version not in {
        "rc-irstd.calibrator.v3",
        "rc-irstd.calibrator.v4",
    }:
        raise ValueError("unsupported calibrator checkpoint format_version")
    collection_sha = str(checkpoint["episode_collection_sha256"])
    if len(collection_sha) != 64 or any(
        character not in "0123456789abcdef" for character in collection_sha
    ):
        raise ValueError("checkpoint episode_collection_sha256 is invalid")
    if not isinstance(checkpoint["episode_collection_provenance"], Mapping):
        raise TypeError("checkpoint episode_collection_provenance must be a mapping")
    if not isinstance(checkpoint["training_config"], Mapping):
        raise TypeError("checkpoint training_config must be a mapping")
    calibrator_model = str(
        checkpoint.get(
            "calibrator_model",
            "direct" if format_version == "rc-irstd.calibrator.v3" else "",
        )
    )
    if format_version == "rc-irstd.calibrator.v3":
        if calibrator_model != "direct":
            raise ValueError("schema-v3 calibrator checkpoint must use direct model")
        model: nn.Module = ThresholdCalibrator(
            int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            dropout=float(checkpoint["dropout"]),
        )
    else:
        if calibrator_model != "monotone_pixel":
            raise ValueError(
                "schema-v4 calibrator checkpoint must use monotone_pixel model"
            )
        raw_model_config = checkpoint.get("model_config")
        if not isinstance(raw_model_config, Mapping):
            raise TypeError("schema-v4 checkpoint model_config must be a mapping")
        model_config = dict(raw_model_config)
        model = MonotonePixelRiskCalibrator(**model_config)
        if model.context_feature_dim != int(checkpoint["input_dim"]):
            raise ValueError("schema-v4 model_config input dimension disagrees")
        capability = checkpoint.get("capability_contract")
        if not isinstance(capability, Mapping):
            raise TypeError(
                "schema-v4 checkpoint capability_contract must be a mapping"
            )
        expected_capability = model.capability_contract()
        if dict(capability) != expected_capability:
            raise ValueError("schema-v4 calibrator capability contract disagrees")
        _validated_monotone_budget_contract(checkpoint, model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    standardizer = FeatureStandardizer.from_dict(checkpoint["standardizer"])
    if tuple(checkpoint["input_feature_names"]) != standardizer.feature_names:
        raise ValueError("checkpoint input feature names disagree with standardizer")
    _deployment_contract(checkpoint)
    _checkpoint_reject_cutoff(checkpoint)
    _deployment_protocol_contract(checkpoint, required=False)
    return model, standardizer, checkpoint


@torch.no_grad()
def adapt_context_to_query(
    *,
    model: nn.Module,
    standardizer: FeatureStandardizer,
    checkpoint_metadata: Mapping[str, Any],
    context_records: Sequence[Mapping[str, Any]],
    query_records: Sequence[Mapping[str, Any]],
    budgets: BudgetSpec,
    source_reference: SourceReference,
    score_manifest: Mapping[str, Any],
    score_manifest_sha256: str,
    device: torch.device,
    target_domain: str,
    reject_probability: float | None = None,
    matching_rule: str | None = None,
    centroid_distance: float | None = None,
    temporal_order_asserted: bool = False,
    claim_bearing: bool = True,
) -> dict[str, Any]:
    """Estimate one threshold using context only, then bind it to later query IDs."""

    context_ids = tuple(str(record["image_id"]) for record in context_records)
    query_ids = tuple(str(record["image_id"]) for record in query_records)
    if not context_ids or not query_ids:
        raise ValueError("online context and query must both be non-empty")
    overlap = set(context_ids).intersection(query_ids)
    if overlap:
        raise ValueError(f"context/query image IDs overlap: {sorted(overlap)}")
    outer_target = str(checkpoint_metadata["outer_target"])
    if target_domain != outer_target:
        raise ValueError("online target_domain must equal checkpoint outer_target")
    deployment_fold, embedded_reference = _deployment_contract(checkpoint_metadata)
    if not isinstance(claim_bearing, bool):
        raise TypeError("claim_bearing must be boolean")
    checkpoint_cutoff = _checkpoint_reject_cutoff(checkpoint_metadata)
    frozen_protocol = _deployment_protocol_contract(
        checkpoint_metadata,
        required=claim_bearing,
    )
    override_requested = reject_probability is not None
    requested_cutoff = (
        checkpoint_cutoff
        if reject_probability is None
        else float(reject_probability)
    )
    if not math.isfinite(requested_cutoff) or not 0.0 <= requested_cutoff <= 1.0:
        raise ValueError("reject_probability must lie in [0, 1]")
    if claim_bearing:
        assert frozen_protocol is not None
        frozen_protocol.assert_runtime_sizes(
            context_size=len(context_ids),
            query_size=len(query_ids),
        )
        if not math.isclose(
            requested_cutoff,
            frozen_protocol.reject_cutoff,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "claim-bearing final-target adaptation cannot override the "
                "checkpoint reject cutoff"
            )
        cutoff = frozen_protocol.reject_cutoff
        cutoff_source = "checkpoint.deployment_protocol_contract.reject_cutoff"
    else:
        cutoff = requested_cutoff
        cutoff_source = (
            "diagnostic_runtime_override"
            if override_requested
            else (
                "checkpoint.reject_probability"
                if "reject_probability" in checkpoint_metadata
                else "checkpoint.training_config.reject_probability"
            )
        )
    matching_override_requested = (
        matching_rule is not None or centroid_distance is not None
    )
    default_matching_rule = (
        frozen_protocol.matching_rule
        if frozen_protocol is not None
        else DEFAULT_MATCHING_RULE
    )
    default_centroid_distance = (
        frozen_protocol.centroid_distance
        if frozen_protocol is not None
        else DEFAULT_CENTROID_DISTANCE
    )
    requested_matching_rule = (
        default_matching_rule if matching_rule is None else str(matching_rule)
    )
    requested_centroid_distance = (
        default_centroid_distance
        if centroid_distance is None
        else float(centroid_distance)
    )
    if requested_matching_rule not in {"overlap", "centroid"}:
        raise ValueError("matching_rule must be 'overlap' or 'centroid'")
    if (
        not math.isfinite(requested_centroid_distance)
        or requested_centroid_distance <= 0.0
    ):
        raise ValueError("centroid_distance must be finite and positive")
    if claim_bearing:
        assert frozen_protocol is not None
        if requested_matching_rule != frozen_protocol.matching_rule or not math.isclose(
            requested_centroid_distance,
            frozen_protocol.centroid_distance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "claim-bearing final-target adaptation cannot override the "
                "checkpoint evaluation matching contract"
            )
        effective_matching_rule = frozen_protocol.matching_rule
        effective_centroid_distance = frozen_protocol.centroid_distance
        matching_source = "checkpoint.deployment_protocol_contract.evaluation_matching"
    else:
        effective_matching_rule = requested_matching_rule
        effective_centroid_distance = requested_centroid_distance
        matching_source = (
            "diagnostic_runtime_override"
            if matching_override_requested
            else (
                "checkpoint.deployment_protocol_contract.evaluation_matching"
                if frozen_protocol is not None
                else "diagnostic_legacy_default"
            )
        )
    calibration_targets = tuple(
        str(value) for value in checkpoint_metadata["calibration_pseudo_targets"]
    )
    if target_domain in calibration_targets:
        raise ValueError("online target leaked into calibration pseudo-targets")
    manifest_target = str(score_manifest.get("target_dataset", ""))
    if manifest_target != target_domain:
        raise ValueError("score manifest target_dataset must equal online target")
    if str(_manifest_contract_value(score_manifest, "outer_fold_id") or "") != str(
        checkpoint_metadata["outer_fold_id"]
    ):
        raise ValueError("score manifest outer_fold_id differs from calibrator contract")
    if str(_manifest_contract_value(score_manifest, "outer_target") or "") != outer_target:
        raise ValueError("score manifest outer_target differs from calibrator contract")
    manifest_protocol_scope = _manifest_contract_value(score_manifest, "protocol_scope")
    if manifest_protocol_scope != deployment_fold.protocol_scope:
        raise ValueError("score manifest protocol_scope differs from calibrator contract")
    if manifest_protocol_scope != "multi_source_protocol_candidate":
        raise ValueError("online main protocol requires a multi-source detector checkpoint")
    if _manifest_contract_value(score_manifest, "target_exclusion_verified") is not True:
        raise ValueError("score manifest does not verify target-domain exclusion")
    manifest_checkpoint_sha = _score_manifest_checkpoint_sha(score_manifest)
    expected_checkpoint_sha = str(
        checkpoint_metadata["deployment_detector_checkpoint_sha"]
    ).lower()
    if manifest_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("score manifest detector checkpoint differs from calibrator contract")
    manifest_source_value = _manifest_contract_value(
        score_manifest, "detector_source_domains"
    )
    if manifest_source_value is None:
        raise ValueError("score manifest must record detector_source_domains")
    manifest_sources = tuple(str(value) for value in manifest_source_value)
    expected_sources = tuple(
        str(value) for value in checkpoint_metadata["deployment_detector_source_domains"]
    )
    if manifest_sources != expected_sources:
        raise ValueError("score manifest detector_source_domains differ from checkpoint")
    if target_domain in manifest_sources:
        raise ValueError("online target must not be a detector source domain")
    manifest_held_out = tuple(
        str(value)
        for value in (
            _manifest_contract_value(score_manifest, "held_out_domains") or ()
        )
    )
    if manifest_held_out != deployment_fold.held_out_domains:
        raise ValueError("score manifest held_out_domains differ from checkpoint")
    if target_domain not in manifest_held_out:
        raise ValueError("online target must occur in detector held_out_domains")
    manifest_items = score_manifest.get("items", score_manifest.get("records"))
    if not isinstance(manifest_items, list):
        raise ValueError("score manifest requires ordered items/records")
    manifest_ids = tuple(str(item["image_id"]) for item in manifest_items)
    expected_prefix = context_ids + query_ids
    if manifest_ids[: len(expected_prefix)] != expected_prefix:
        raise ValueError("context+query IDs must exactly equal the score manifest prefix")
    if source_reference != embedded_reference:
        raise ValueError("online source reference differs from checkpoint deployment reference")
    statistics_config = StatisticsConfig.from_dict(
        checkpoint_metadata["statistics_config"]
    )
    probabilities = []
    grays = []
    for record in context_records:
        probability, grayscale = load_probability_and_grayscale(
            record["prob_path"], record.get("gray_path")
        )
        probabilities.append(probability)
        grays.append(grayscale)
    if any(value is None for value in grays) and not all(value is None for value in grays):
        raise ValueError("context grayscale availability must be all-or-none")
    statistics = extract_unlabeled_statistics(
        probabilities,
        None if all(value is None for value in grays) else grays,
        source_reference=source_reference,
        statistics_config=statistics_config,
    )
    expected_statistics_names = tuple(checkpoint_metadata["statistics_feature_names"])
    if statistics.feature_names != expected_statistics_names:
        raise ValueError("online statistics schema differs from calibrator checkpoint")
    if isinstance(model, MonotonePixelRiskCalibrator):
        pixel_budget = pixel_budget_from_spec(budgets)
        raw_feature_values = tuple(float(value) for value in statistics.vector)
    elif isinstance(model, ThresholdCalibrator):
        pixel_budget = None
        raw_feature_values = (
            tuple(float(value) for value in statistics.vector) + budgets.encoded()
        )
    else:
        raise TypeError(f"unsupported calibrator model: {type(model).__name__}")
    raw_features = np.asarray(raw_feature_values, dtype=np.float64)[None, :]
    normalised = standardizer.transform(raw_features).astype(np.float32)
    features = torch.from_numpy(normalised).to(device)
    if isinstance(model, MonotonePixelRiskCalibrator):
        assert pixel_budget is not None
        budget_model_contract: dict[str, Any] | None = (
            _validated_monotone_budget_contract(checkpoint_metadata, model)
        )
        output = model(
            features,
            pixel_budgets=torch.tensor(
                [[pixel_budget]], dtype=torch.float64, device=device
            ),
        )
        if (
            output.requested_thresholds is None
            or output.requested_reject_probabilities is None
        ):
            raise RuntimeError("monotone calibrator did not return requested outputs")
        threshold_value = float(output.requested_thresholds[0, 0].cpu())
        probability = float(output.requested_reject_probabilities[0, 0].cpu())
    else:
        budget_model_contract = None
        threshold, reject_logit = model(features)
        threshold_value = float(threshold[0].cpu())
        probability = float(torch.sigmoid(reject_logit)[0].cpu())
    evaluation_contract = {
        "schema_version": EVALUATION_MATCHING_CONTRACT_VERSION,
        "matching_rule": effective_matching_rule,
        "centroid_distance": effective_centroid_distance,
        "source": matching_source,
        "target_override_allowed": False if claim_bearing else True,
        "runtime_override_requested": matching_override_requested,
    }
    decision_contract = {
        "schema_version": ONLINE_DECISION_CONTRACT_VERSION,
        "claim_bearing": claim_bearing,
        "claim_eligibility": (
            "claim_eligible_frozen_checkpoint_protocol"
            if claim_bearing
            else "diagnostic_only"
        ),
        "context_size": len(context_ids),
        "query_size": len(query_ids),
        "partition_rule": (
            frozen_protocol.partition_rule
            if frozen_protocol is not None
            else CAUSAL_PARTITION_RULE
        ),
        "size_contract_source": (
            "checkpoint.deployment_protocol_contract"
            if claim_bearing
            else "diagnostic_runtime"
        ),
        "reject_rule": {
            "score": (
                frozen_protocol.reject_score
                if frozen_protocol is not None
                else REJECT_SCORE_RULE
            ),
            "comparison": (
                frozen_protocol.reject_comparison
                if frozen_protocol is not None
                else REJECT_COMPARISON_RULE
            ),
            "cutoff": cutoff,
            "cutoff_source": cutoff_source,
            "target_override_allowed": False if claim_bearing else True,
            "runtime_override_requested": override_requested,
        },
        "evaluation_matching": evaluation_contract,
        "budget_model": budget_model_contract,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "asserted_temporal_prefix" if temporal_order_asserted else "prefix_holdout",
        "protocol": (
            "user_asserted_temporal_order"
            if temporal_order_asserted
            else "manifest_order_prefix_holdout"
        ),
        # A CLI assertion is useful provenance, but it is not independent
        # verification against acquisition timestamps or a signed source log.
        "temporal_order_asserted": bool(temporal_order_asserted),
        "target_domain": target_domain,
        "outer_fold_id": str(checkpoint_metadata["outer_fold_id"]),
        "outer_target": outer_target,
        "detector_source_domains": list(expected_sources),
        "detector_checkpoint_sha": expected_checkpoint_sha,
        "held_out_domains": list(deployment_fold.held_out_domains),
        "protocol_scope": deployment_fold.protocol_scope,
        "score_manifest_sha256": score_manifest_sha256,
        "score_manifest_target_dataset": manifest_target,
        "score_manifest_detector_checkpoint_sha": manifest_checkpoint_sha,
        "calibration_pseudo_targets": list(calibration_targets),
        "calibrator_format_version": str(checkpoint_metadata["format_version"]),
        "calibrator_model": str(
            checkpoint_metadata.get("calibrator_model", "direct")
        ),
        "calibrator_capability_contract": dict(
            checkpoint_metadata.get("capability_contract", {})
        ),
        "calibrator_budget_contract": budget_model_contract,
        "episode_collection_sha256": str(
            checkpoint_metadata["episode_collection_sha256"]
        ),
        "claim_bearing": claim_bearing,
        "deployment_protocol_contract": (
            None if frozen_protocol is None else frozen_protocol.to_dict()
        ),
        "decision_contract": decision_contract,
        "evaluation_contract": evaluation_contract,
        "context_size": len(context_ids),
        "query_size": len(query_ids),
        "context_image_ids": list(context_ids),
        "query_image_ids": list(query_ids),
        "causal_boundary": {
            "context_start_index": 0,
            "context_end_index_inclusive": len(context_ids) - 1,
            "query_start_index": len(context_ids),
            "query_end_index_inclusive": len(context_ids) + len(query_ids) - 1,
            "context_last_image_id": context_ids[-1],
            "query_first_image_id": query_ids[0],
        },
        "budgets": budgets.to_dict(),
        "p_min": float(checkpoint_metadata["p_min"]),
        "statistics_config": statistics_config.to_dict(),
        "source_reference": source_reference.to_dict(),
        "threshold": threshold_value,
        "reject_probability": probability,
        "reject_cutoff": cutoff,
        "reject": probability >= cutoff,
        "statistics_feature_names": list(statistics.feature_names),
        "statistics_metadata": dict(statistics.metadata or {}),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--score-manifest", dest="manifest", required=True)
    parser.add_argument("--calibrator-checkpoint", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--query-size", type=int)
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument("--source-reference")
    parser.add_argument(
        "--reject-probability",
        type=float,
        help=(
            "Diagnostic cutoff override. In claim-bearing mode, an explicitly "
            "supplied value must equal the checkpoint-frozen cutoff."
        ),
    )
    parser.add_argument(
        "--diagnostic-unfrozen-protocol",
        action="store_true",
        help=(
            "Permit legacy checkpoints, alternate context/query sizes, or a cutoff "
            "override; output is explicitly marked diagnostic-only."
        ),
    )
    parser.add_argument(
        "--matching-rule",
        choices=("overlap", "centroid"),
        help=(
            "Evaluation matching rule to bind without reading labels. Formal mode "
            "requires equality with the calibrator checkpoint contract."
        ),
    )
    parser.add_argument(
        "--centroid-distance",
        type=float,
        help=(
            "Centroid matching radius to bind without reading labels. Formal mode "
            "requires equality with the calibrator checkpoint contract."
        ),
    )
    parser.add_argument(
        "--assert-temporal-order",
        "--temporal-order-verified",
        dest="temporal_order_asserted",
        action="store_true",
        help=(
            "Record the user's assertion that manifest order follows acquisition time. "
            "This is not independent verification; otherwise report prefix_holdout."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    return parser


def _select_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    budgets = BudgetSpec.from_optional(args.pixel_budget, args.component_budget)
    claim_bearing = not args.diagnostic_unfrozen_protocol
    required_split_role = "official_test" if claim_bearing else None
    records, manifest = load_ordered_score_records(
        args.manifest,
        required_split_role=required_split_role,
    )
    context, query = causal_partition(
        records, context_size=args.context_size, query_size=args.query_size
    )
    manifest_path = Path(args.manifest).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    context_ids = tuple(str(record["image_id"]) for record in context)
    verified_context = verify_score_manifest_artifacts(
        manifest_path,
        image_ids=context_ids,
        require_mask=False,
        require_native_contract=True,
        required_split_role=required_split_role,
    )
    if (
        verified_context.manifest_sha256 != manifest_sha256
        or verified_context.payload != manifest
    ):
        raise RuntimeError("score manifest changed while verifying the context prefix")
    verified_context_paths = tuple(
        (str(item.score_path), str(item.gray_path))
        for item in verified_context.selected_items
    )
    record_context_paths = tuple(
        (str(record["prob_path"]), str(record["gray_path"])) for record in context
    )
    if verified_context_paths != record_context_paths:
        raise RuntimeError("verified context paths differ from manifest record binding")
    device = _select_device(args.device)
    calibrator_checkpoint_path = Path(args.calibrator_checkpoint).expanduser().resolve()
    model, standardizer, checkpoint = load_calibrator_bundle(
        calibrator_checkpoint_path, device=device
    )
    statistics_config = StatisticsConfig.from_dict(checkpoint["statistics_config"])
    _, embedded_reference = _deployment_contract(checkpoint)
    if args.source_reference is None:
        source_reference = embedded_reference
    else:
        source_reference = load_source_reference(
            args.source_reference,
            statistics_config=statistics_config,
        )
        if source_reference != embedded_reference:
            raise ValueError("provided source reference differs from checkpoint")
    result = adapt_context_to_query(
        model=model,
        standardizer=standardizer,
        checkpoint_metadata=checkpoint,
        context_records=context,
        query_records=query,
        budgets=budgets,
        source_reference=source_reference,
        score_manifest=manifest,
        score_manifest_sha256=manifest_sha256,
        device=device,
        target_domain=args.target_domain,
        reject_probability=args.reject_probability,
        matching_rule=args.matching_rule,
        centroid_distance=args.centroid_distance,
        temporal_order_asserted=args.temporal_order_asserted,
        claim_bearing=claim_bearing,
    )
    result["score_manifest"] = Path(args.manifest).name
    result["calibrator_checkpoint"] = calibrator_checkpoint_path.name
    result["calibrator_checkpoint_sha256"] = sha256_file(
        calibrator_checkpoint_path
    )
    result["score_manifest_target"] = manifest.get("target_dataset")
    _write_json_atomic(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
