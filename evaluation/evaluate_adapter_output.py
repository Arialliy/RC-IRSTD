"""Replay an online-adapter decision on its cryptographically bound query set.

The adapter consumes only an unlabeled context and emits one threshold (or a
rejection) for a disjoint query suffix.  This module is the label-using,
offline half of that protocol: it verifies the adapter/manifest binding before
loading any query labels, applies exactly ``probability > threshold``, and
reports native-resolution object and false-alarm counts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data_ext.dataset_identity import sha256_file
from data_ext.label_manifest_artifacts import (
    load_label_mask,
    verify_label_attachment,
)
from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts
from rc.schema import EVALUATION_MATCHING_CONTRACT_VERSION
from .budget_metrics import relative_budget_excess
from .component_matching import aggregate_match_results, match_components
from .threshold_sweep import THRESHOLD_SEMANTICS


ADAPTER_EVALUATION_SCHEMA_VERSION = "rc-irstd.adapter-evaluation.v1"
ADAPTER_SUMMARY_SCHEMA_VERSION = "rc-irstd.adapter-evaluation-summary.v1"
_RAW_METRIC_FIELDS = (
    "pd",
    "fa_pixel",
    "fa_component_mp",
    "tp_objects",
    "gt_objects",
    "pred_components",
    "fp_components",
    "fp_pixels",
    "total_pixels",
    "num_images",
)


def evaluate_adapter_output(
    adapter_output: str | Path | Mapping[str, Any],
    score_manifest: str | Path,
    *,
    calibrator_checkpoint: str | Path,
    label_manifest: str | Path,
    matching_rule: str | None = None,
    centroid_distance: float | None = None,
) -> dict[str, Any]:
    """Verify and replay one online adapter result on its declared query IDs.

    Rejected outputs deliberately contain no Pd/FA values or raw label-derived
    counts.  They remain useful records for coverage-aware aggregation.
    """

    adapter, adapter_name = _load_adapter_output(adapter_output)
    claim_bearing = _require_bool(adapter, "claim_bearing")
    required_split_role = "official_test" if claim_bearing else None
    evaluation_contract = _normalise_evaluation_contract(
        adapter.get("evaluation_contract")
    )
    if matching_rule is not None and matching_rule != evaluation_contract["matching_rule"]:
        raise ValueError(
            "requested matching_rule differs from the adapter evaluation contract"
        )
    if centroid_distance is not None:
        requested_distance = float(centroid_distance)
        if not math.isfinite(requested_distance) or requested_distance <= 0.0:
            raise ValueError("centroid_distance must be finite and positive")
        if not math.isclose(
            requested_distance,
            float(evaluation_contract["centroid_distance"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "requested centroid_distance differs from the adapter evaluation contract"
            )
    effective_matching_rule = str(evaluation_contract["matching_rule"])
    effective_centroid_distance = float(evaluation_contract["centroid_distance"])
    manifest_path = Path(score_manifest).expanduser().resolve()
    verified_manifest = verify_score_manifest_artifacts(
        manifest_path,
        require_mask=False,
        require_native_contract=True,
        verify_artifact_bytes=False,
        required_split_role=required_split_role,
    )
    manifest = verified_manifest.payload
    manifest_sha256 = verified_manifest.manifest_sha256
    calibrator_path = Path(calibrator_checkpoint).expanduser().resolve()
    if not calibrator_path.is_file():
        raise FileNotFoundError(
            f"Calibrator checkpoint does not exist: {calibrator_path}"
        )
    calibrator_sha256 = sha256_file(calibrator_path)

    target_domain, context_ids, query_ids = _verify_binding(
        adapter,
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        calibrator_sha256=calibrator_sha256,
    )
    budgets = _normalise_budgets(adapter.get("budgets"))
    rejected = _require_bool(adapter, "reject")
    threshold = _finite_probability(adapter.get("threshold"), "threshold")
    _verify_recomputed_calibrator_decision(
        adapter,
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        calibrator_path=calibrator_path,
        target_domain=target_domain,
        context_ids=context_ids,
        query_ids=query_ids,
        budgets=budgets,
        required_split_role=required_split_role,
    )
    result: dict[str, Any] = {
        "schema_version": ADAPTER_EVALUATION_SCHEMA_VERSION,
        "adapter_output_file": adapter_name,
        "outer_fold_id": _nonempty_string(adapter["outer_fold_id"], "outer_fold_id"),
        "target_domain": target_domain,
        "score_manifest_file": manifest_path.name,
        "score_manifest_sha256": manifest_sha256,
        "calibrator_checkpoint_file": calibrator_path.name,
        "calibrator_checkpoint_sha256": calibrator_sha256,
        "calibrator_replay_verified": True,
        "calibrator_model": str(adapter["calibrator_model"]),
        "calibrator_capability_contract": dict(
            adapter["calibrator_capability_contract"]
        ),
        "calibrator_budget_contract": adapter["calibrator_budget_contract"],
        "claim_bearing": claim_bearing,
        "decision_contract": dict(adapter["decision_contract"]),
        "evaluation_contract": evaluation_contract,
        "query_image_ids": list(query_ids),
        "num_query_images": len(query_ids),
        "budgets": budgets,
        "threshold": threshold,
        "threshold_semantics": THRESHOLD_SEMANTICS,
        "matching_rule": effective_matching_rule,
        "centroid_distance": effective_centroid_distance,
        "rejected": rejected,
    }
    if rejected:
        return result

    # The label artifact is deliberately not resolved, opened, or hashed
    # until the context-only decision has been replayed and accepted.
    label_manifest_path = Path(label_manifest).expanduser().resolve()
    attachment = verify_label_attachment(
        manifest_path,
        label_manifest_path,
        image_ids=query_ids,
    )
    if attachment.score_manifest.manifest_sha256 != manifest_sha256:
        raise RuntimeError("score manifest changed between decision replay and label replay")
    if tuple(item.image_id for item in attachment.selected_items) != tuple(query_ids):
        raise RuntimeError("verified query-label order differs from adapter binding")
    if tuple(
        item.image_id for item in attachment.score_manifest.selected_items
    ) != tuple(query_ids):
        raise RuntimeError("verified query-score order differs from adapter binding")

    matches = []
    for expected_id, score_item, label_item in zip(
        query_ids,
        attachment.score_manifest.selected_items,
        attachment.selected_items,
    ):
        if score_item.image_id != expected_id or label_item.image_id != expected_id:
            raise RuntimeError("score/label/query image-ID binding changed after verification")
        with np.load(score_item.score_path, allow_pickle=False) as score_payload:
            probability = np.asarray(score_payload["prob"], dtype=np.float64)
        mask = load_label_mask(label_item)
        if probability.shape != mask.shape:
            raise ValueError(
                f"verified score/label shape mismatch for {expected_id!r}: "
                f"{probability.shape} != {mask.shape}"
            )
        matches.append(
            match_components(
                probability > threshold,
                mask,
                rule=effective_matching_rule,
                centroid_distance=effective_centroid_distance,
            )
        )
    result.update(
        {
            "label_manifest_file": label_manifest_path.name,
            "label_manifest_sha256": attachment.manifest_sha256,
            "label_manifest_content_sha256": attachment.content_sha256,
            "label_attachment_verified": True,
        }
    )
    result.update(aggregate_match_results(matches))
    return result


def summarise_adapter_evaluations(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate coverage, per-record budget safety, excess, and covered Pd.

    Budgets may differ between adapter records because they are model inputs.
    BSR and excess therefore use each record's own active-budget contract.
    Covered Pd is a micro-average reconstructed from raw TP/GT counts, never
    an average of rounded per-record Pd values.
    """

    materialised = list(records)
    if not materialised:
        raise ValueError("At least one adapter evaluation is required")

    covered: list[Mapping[str, Any]] = []
    joint_satisfied = 0
    joint_excesses: list[float] = []
    pixel_satisfied = 0
    pixel_excesses: list[float] = []
    component_satisfied = 0
    component_excesses: list[float] = []
    covered_tp = 0
    covered_gt = 0
    covered_images = 0

    for record in materialised:
        if "rejected" not in record or not isinstance(record["rejected"], bool):
            raise TypeError("Each evaluation record requires a boolean 'rejected' field")
        budgets = _normalise_budgets(record.get("budgets"))
        if record["rejected"]:
            leaked = [field for field in _RAW_METRIC_FIELDS if field in record]
            if leaked:
                raise ValueError(
                    "Rejected records must not contain label-derived metrics: "
                    f"{leaked}"
                )
            continue

        _validate_covered_record(record)
        covered.append(record)
        values = (float(record["fa_pixel"]), float(record["fa_component_mp"]))
        active_satisfied: list[bool] = []
        active_excesses: list[float] = []
        for index, (value, budget, active) in enumerate(
            zip(values, budgets["values"], budgets["active"])
        ):
            if not active:
                continue
            satisfied = value <= budget
            excess = relative_budget_excess(value, budget)
            active_satisfied.append(satisfied)
            active_excesses.append(excess)
            if index == 0:
                pixel_satisfied += int(satisfied)
                pixel_excesses.append(excess)
            else:
                component_satisfied += int(satisfied)
                component_excesses.append(excess)
        joint_satisfied += int(all(active_satisfied))
        joint_excesses.append(max(active_excesses))
        covered_tp += int(record["tp_objects"])
        covered_gt += int(record["gt_objects"])
        covered_images += int(record["num_images"])

    num_total = len(materialised)
    num_covered = len(covered)
    return {
        "schema_version": ADAPTER_SUMMARY_SCHEMA_VERSION,
        "num_records": num_total,
        "num_covered": num_covered,
        "num_rejected": num_total - num_covered,
        "coverage": num_covered / num_total,
        "bsr": _safe_ratio(joint_satisfied, num_covered),
        "joint_bsr": _safe_ratio(joint_satisfied, num_covered),
        "unconditional_bsr": joint_satisfied / num_total,
        "excess": _safe_mean(joint_excesses),
        "joint_excess": _safe_mean(joint_excesses),
        "pixel_bsr": _safe_ratio(pixel_satisfied, len(pixel_excesses)),
        "pixel_excess": _safe_mean(pixel_excesses),
        "component_bsr": _safe_ratio(
            component_satisfied, len(component_excesses)
        ),
        "component_excess": _safe_mean(component_excesses),
        "covered_pd": (covered_tp / covered_gt) if covered_gt else None,
        "covered_tp_objects": covered_tp,
        "covered_gt_objects": covered_gt,
        "covered_num_images": covered_images,
        "budget_scope": "per_record_active_budgets",
    }


def _verify_binding(
    adapter: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    calibrator_sha256: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    required = (
        "outer_fold_id",
        "outer_target",
        "target_domain",
        "detector_source_domains",
        "detector_checkpoint_sha",
        "calibrator_checkpoint_sha256",
        "score_manifest_sha256",
        "score_manifest_target_dataset",
        "score_manifest_detector_checkpoint_sha",
        "context_image_ids",
        "query_image_ids",
        "context_size",
        "query_size",
        "claim_bearing",
        "deployment_protocol_contract",
        "decision_contract",
        "evaluation_contract",
        "calibrator_model",
        "calibrator_capability_contract",
        "calibrator_budget_contract",
    )
    missing = [field for field in required if field not in adapter]
    if missing:
        raise KeyError(f"Adapter output is missing binding fields: {missing}")

    expected_manifest_sha = _sha256_value(
        adapter["score_manifest_sha256"], "score_manifest_sha256"
    )
    if expected_manifest_sha != manifest_sha256:
        raise ValueError(
            "Score manifest SHA-256 mismatch: adapter output is not bound to "
            f"{manifest_path.name!r}"
        )
    expected_calibrator_sha = _sha256_value(
        adapter["calibrator_checkpoint_sha256"],
        "calibrator_checkpoint_sha256",
    )
    if expected_calibrator_sha != calibrator_sha256:
        raise ValueError(
            "Calibrator checkpoint SHA-256 mismatch: adapter output is not bound "
            "to the supplied calibrator"
        )

    target_domain = _nonempty_string(adapter["target_domain"], "target_domain")
    target_values = {
        "target_domain": target_domain,
        "outer_target": _nonempty_string(adapter["outer_target"], "outer_target"),
        "score_manifest_target_dataset": _nonempty_string(
            adapter["score_manifest_target_dataset"],
            "score_manifest_target_dataset",
        ),
        "manifest.target_dataset": _nonempty_string(
            manifest.get("target_dataset"), "manifest.target_dataset"
        ),
    }
    if len(set(target_values.values())) != 1:
        raise ValueError(f"Target-domain binding mismatch: {target_values}")

    outer_fold_id = _nonempty_string(adapter["outer_fold_id"], "outer_fold_id")
    if _nonempty_string(manifest.get("outer_fold_id"), "manifest.outer_fold_id") != outer_fold_id:
        raise ValueError("Outer-fold binding mismatch")
    if _nonempty_string(manifest.get("outer_target"), "manifest.outer_target") != target_domain:
        raise ValueError("Manifest outer_target binding mismatch")
    raw_adapter_sources = adapter["detector_source_domains"]
    raw_manifest_sources = manifest.get("detector_source_domains")
    if not isinstance(raw_adapter_sources, (list, tuple)) or not isinstance(
        raw_manifest_sources, (list, tuple)
    ):
        raise TypeError("detector_source_domains must be ordered lists")
    adapter_sources = tuple(
        _nonempty_string(value, "detector_source_domains")
        for value in raw_adapter_sources
    )
    manifest_sources = tuple(
        _nonempty_string(value, "manifest.detector_source_domains")
        for value in raw_manifest_sources
    )
    if (
        not adapter_sources
        or len(set(adapter_sources)) != len(adapter_sources)
        or adapter_sources != manifest_sources
    ):
        raise ValueError("Detector-source-domain binding mismatch")
    if target_domain in manifest_sources:
        raise ValueError("Target domain occurs in detector source domains")
    if manifest.get("protocol_scope") != "multi_source_protocol_candidate":
        raise ValueError("Adapter replay requires a multi-source detector artifact")
    if manifest.get("target_exclusion_verified") is not True:
        raise ValueError("Manifest does not verify target-domain exclusion")

    adapter_detector_sha = _sha256_value(
        adapter["detector_checkpoint_sha"], "detector_checkpoint_sha"
    )
    score_detector_sha = _sha256_value(
        adapter["score_manifest_detector_checkpoint_sha"],
        "score_manifest_detector_checkpoint_sha",
    )
    manifest_detector_sha = _sha256_value(
        manifest.get("weight_sha256"), "manifest.weight_sha256"
    )
    if len({adapter_detector_sha, score_detector_sha, manifest_detector_sha}) != 1:
        raise ValueError("Detector-checkpoint SHA-256 binding mismatch")

    if manifest.get("threshold_semantics") not in {None, THRESHOLD_SEMANTICS}:
        raise ValueError(
            "Score manifest threshold semantics disagree with replay semantics"
        )
    if manifest.get("score_type") not in {None, "sigmoid_probability"}:
        raise ValueError("Score manifest must contain sigmoid probabilities")

    context_ids = _image_ids(adapter["context_image_ids"], "context_image_ids")
    query_ids = _image_ids(adapter["query_image_ids"], "query_image_ids")
    overlap = set(context_ids).intersection(query_ids)
    if overlap:
        raise ValueError(f"Context/query image IDs overlap: {sorted(overlap)}")
    context_size = adapter["context_size"]
    query_size = adapter["query_size"]
    for name, value in (("context_size", context_size), ("query_size", query_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TypeError(f"Adapter {name} must be a positive integer")
    if context_size != len(context_ids) or query_size != len(query_ids):
        raise ValueError("Adapter context/query sizes do not match bound image IDs")
    claim_bearing = _require_bool(adapter, "claim_bearing")
    deployment_contract = adapter["deployment_protocol_contract"]
    if claim_bearing and not isinstance(deployment_contract, Mapping):
        raise TypeError(
            "Claim-bearing adapter output requires a deployment protocol contract"
        )
    decision_contract = adapter["decision_contract"]
    if not isinstance(decision_contract, Mapping):
        raise TypeError("Adapter decision_contract must be a mapping")
    calibrator_model = _nonempty_string(
        adapter["calibrator_model"], "calibrator_model"
    )
    budget_contract = adapter["calibrator_budget_contract"]
    if calibrator_model == "monotone_pixel":
        if not isinstance(budget_contract, Mapping):
            raise TypeError(
                "monotone_pixel adapter requires calibrator_budget_contract"
            )
    elif calibrator_model == "direct":
        if budget_contract is not None:
            raise ValueError("direct adapter must not declare a monotone budget contract")
    else:
        raise ValueError("unsupported adapter calibrator_model")
    if decision_contract.get("budget_model") != budget_contract:
        raise ValueError("Adapter decision/budget model contracts disagree")
    if decision_contract.get("claim_bearing") is not claim_bearing:
        raise ValueError("Adapter decision contract claim-bearing flag mismatch")
    if (
        decision_contract.get("context_size") != context_size
        or decision_contract.get("query_size") != query_size
    ):
        raise ValueError("Adapter decision contract context/query size mismatch")
    evaluation_contract = _normalise_evaluation_contract(
        adapter["evaluation_contract"]
    )
    if decision_contract.get("evaluation_matching") != evaluation_contract:
        raise ValueError("Adapter decision/evaluation matching contracts disagree")
    if claim_bearing:
        assert isinstance(deployment_contract, Mapping)
        deployment_matching = deployment_contract.get("evaluation_matching")
        if not isinstance(deployment_matching, Mapping):
            raise TypeError(
                "Claim-bearing deployment contract requires evaluation_matching"
            )
        frozen_pair = (
            deployment_matching.get("matching_rule"),
            deployment_matching.get("centroid_distance"),
        )
        observed_pair = (
            evaluation_contract["matching_rule"],
            evaluation_contract["centroid_distance"],
        )
        if frozen_pair != observed_pair:
            raise ValueError(
                "Adapter evaluation matching contract differs from deployment contract"
            )
    return target_domain, context_ids, query_ids


def _verify_recomputed_calibrator_decision(
    adapter: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    calibrator_path: Path,
    target_domain: str,
    context_ids: Sequence[str],
    query_ids: Sequence[str],
    budgets: Mapping[str, Any],
    required_split_role: str | None,
) -> None:
    """Replay context inference on CPU before any query label is consumed."""

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - deployment guard
        raise RuntimeError("Calibrator replay requires PyTorch") from error

    from rc.online_adapter import (
        adapt_context_to_query,
        load_calibrator_bundle,
        load_ordered_score_records,
    )
    from rc.schema import BudgetSpec, DeploymentProtocolContract, SourceReference

    for field in (
        "reject_probability",
        "reject_cutoff",
        "temporal_order_asserted",
        "source_reference",
        "statistics_config",
        "calibration_pseudo_targets",
        "held_out_domains",
        "protocol_scope",
        "p_min",
        "calibrator_format_version",
        "calibrator_model",
        "calibrator_capability_contract",
        "calibrator_budget_contract",
        "episode_collection_sha256",
        "claim_bearing",
        "deployment_protocol_contract",
        "decision_contract",
        "evaluation_contract",
        "context_size",
        "query_size",
    ):
        if field not in adapter:
            raise KeyError(f"Adapter output is missing replay field: {field}")
    if not isinstance(adapter["temporal_order_asserted"], bool):
        raise TypeError("temporal_order_asserted must be boolean")

    device = torch.device("cpu")
    model, standardizer, checkpoint = load_calibrator_bundle(
        calibrator_path,
        device=device,
    )
    records, replay_manifest = load_ordered_score_records(
        manifest_path,
        required_split_role=required_split_role,
    )
    if replay_manifest != manifest:
        raise RuntimeError("score manifest changed while preparing calibrator replay")
    by_id = {str(record["image_id"]): record for record in records}
    missing = [
        image_id
        for image_id in tuple(context_ids) + tuple(query_ids)
        if image_id not in by_id
    ]
    if missing:
        raise KeyError(f"Adapter context/query IDs are absent from manifest: {missing}")
    context_records = [by_id[image_id] for image_id in context_ids]
    query_records = [by_id[image_id] for image_id in query_ids]
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
        raise RuntimeError("score manifest changed while verifying replay context")
    verified_context_paths = tuple(
        (str(item.score_path), str(item.gray_path))
        for item in verified_context.selected_items
    )
    replay_context_paths = tuple(
        (str(record["prob_path"]), str(record["gray_path"]))
        for record in context_records
    )
    if verified_context_paths != replay_context_paths:
        raise RuntimeError("verified replay context paths differ from manifest binding")
    source_reference = SourceReference.from_dict(
        checkpoint["deployment_source_reference"]
    )
    replay_budgets = BudgetSpec(
        values=tuple(float(value) for value in budgets["values"]),  # type: ignore[arg-type]
        active=tuple(bool(value) for value in budgets["active"]),  # type: ignore[arg-type]
    )
    reject_cutoff = _finite_probability(
        adapter["reject_cutoff"], "reject_cutoff"
    )
    claim_bearing = _require_bool(adapter, "claim_bearing")
    evaluation_contract = _normalise_evaluation_contract(
        adapter["evaluation_contract"]
    )
    raw_decision_contract = adapter["decision_contract"]
    if not isinstance(raw_decision_contract, Mapping):
        raise TypeError("decision_contract must be a mapping")
    raw_reject_rule = raw_decision_contract.get("reject_rule")
    if not isinstance(raw_reject_rule, Mapping):
        raise TypeError("decision_contract.reject_rule must be a mapping")
    reject_override_requested = raw_reject_rule.get("runtime_override_requested")
    if not isinstance(reject_override_requested, bool):
        raise TypeError(
            "decision_contract.reject_rule.runtime_override_requested must be boolean"
        )
    matching_override_requested = evaluation_contract[
        "runtime_override_requested"
    ]
    frozen_protocol = None
    if claim_bearing:
        raw_frozen_protocol = checkpoint.get("deployment_protocol_contract")
        if not isinstance(raw_frozen_protocol, Mapping):
            raise TypeError(
                "claim-bearing checkpoint requires deployment_protocol_contract"
            )
        frozen_protocol = DeploymentProtocolContract.from_dict(raw_frozen_protocol)
    replay_reject_cutoff = None
    if reject_override_requested:
        replay_reject_cutoff = (
            frozen_protocol.reject_cutoff
            if frozen_protocol is not None
            else reject_cutoff
        )
    replay_matching_rule = None
    replay_centroid_distance = None
    if matching_override_requested:
        replay_matching_rule = (
            frozen_protocol.matching_rule
            if frozen_protocol is not None
            else str(evaluation_contract["matching_rule"])
        )
        replay_centroid_distance = (
            frozen_protocol.centroid_distance
            if frozen_protocol is not None
            else float(evaluation_contract["centroid_distance"])
        )
    recomputed = adapt_context_to_query(
        model=model,
        standardizer=standardizer,
        checkpoint_metadata=checkpoint,
        context_records=context_records,
        query_records=query_records,
        budgets=replay_budgets,
        source_reference=source_reference,
        score_manifest=manifest,
        score_manifest_sha256=manifest_sha256,
        device=device,
        target_domain=target_domain,
        # Claim-bearing replay resolves frozen parameters from the checkpoint,
        # never from potentially edited adapter values.
        reject_probability=replay_reject_cutoff,
        matching_rule=replay_matching_rule,
        centroid_distance=replay_centroid_distance,
        temporal_order_asserted=bool(adapter["temporal_order_asserted"]),
        claim_bearing=claim_bearing,
    )

    for field in ("threshold", "reject_probability", "reject_cutoff", "p_min"):
        observed = float(adapter[field])
        expected = float(recomputed[field])
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"Adapter {field} differs from deterministic calibrator replay: "
                f"{observed} != {expected}"
            )
    if _require_bool(adapter, "reject") != bool(recomputed["reject"]):
        raise ValueError("Adapter reject decision differs from deterministic calibrator replay")
    for field in (
        "source_reference",
        "statistics_config",
        "calibration_pseudo_targets",
        "held_out_domains",
        "protocol_scope",
        "calibrator_format_version",
        "calibrator_model",
        "calibrator_capability_contract",
        "calibrator_budget_contract",
        "episode_collection_sha256",
        "deployment_protocol_contract",
        "decision_contract",
        "evaluation_contract",
        "claim_bearing",
        "context_size",
        "query_size",
    ):
        if adapter[field] != recomputed[field]:
            raise ValueError(
                f"Adapter {field} differs from deterministic calibrator replay"
            )


def _normalise_budgets(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("budgets must be a mapping")
    names = tuple(value.get("names", ("pixel", "component")))
    if names != ("pixel", "component"):
        raise ValueError("budget names must be ['pixel', 'component']")
    raw_values = value.get("values")
    raw_active = value.get("active")
    if not isinstance(raw_values, (list, tuple)) or len(raw_values) != 2:
        raise ValueError("budgets.values must contain pixel and component values")
    if not isinstance(raw_active, (list, tuple)) or len(raw_active) != 2:
        raise ValueError("budgets.active must contain two booleans")
    if not all(isinstance(item, bool) for item in raw_active):
        raise TypeError("budgets.active values must be booleans")
    values = tuple(float(item) for item in raw_values)
    if not all(math.isfinite(item) and item >= 0.0 for item in values):
        raise ValueError("budget values must be finite and non-negative")
    active = tuple(bool(item) for item in raw_active)
    if not any(active):
        raise ValueError("at least one budget must be active")
    if any(enabled and budget <= 0.0 for budget, enabled in zip(values, active)):
        raise ValueError("active budgets must be positive")
    return {"names": list(names), "values": list(values), "active": list(active)}


def _normalise_evaluation_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("evaluation_contract must be a mapping")
    required = (
        "schema_version",
        "matching_rule",
        "centroid_distance",
        "source",
        "target_override_allowed",
        "runtime_override_requested",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise KeyError(f"evaluation_contract is missing: {missing}")
    if value["schema_version"] != EVALUATION_MATCHING_CONTRACT_VERSION:
        raise ValueError("unsupported evaluation_contract schema_version")
    matching_rule = value["matching_rule"]
    if matching_rule not in {"overlap", "centroid"}:
        raise ValueError("evaluation_contract matching_rule is unsupported")
    centroid_distance = float(value["centroid_distance"])
    if not math.isfinite(centroid_distance) or centroid_distance <= 0.0:
        raise ValueError("evaluation_contract centroid_distance must be positive")
    source = _nonempty_string(value["source"], "evaluation_contract.source")
    for field in ("target_override_allowed", "runtime_override_requested"):
        if not isinstance(value[field], bool):
            raise TypeError(f"evaluation_contract.{field} must be boolean")
    return {
        "schema_version": EVALUATION_MATCHING_CONTRACT_VERSION,
        "matching_rule": matching_rule,
        "centroid_distance": centroid_distance,
        "source": source,
        "target_override_allowed": bool(value["target_override_allowed"]),
        "runtime_override_requested": bool(value["runtime_override_requested"]),
    }


def _validate_covered_record(record: Mapping[str, Any]) -> None:
    missing = [field for field in _RAW_METRIC_FIELDS if field not in record]
    if missing:
        raise KeyError(f"Covered evaluation is missing metrics: {missing}")
    for field in ("pd", "fa_pixel", "fa_component_mp"):
        value = float(record[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    counts: dict[str, int] = {}
    for field in _RAW_METRIC_FIELDS[3:]:
        value = record[field]
        integer = int(value)
        if isinstance(value, bool) or float(value) != integer or integer < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        counts[field] = integer
    if counts["tp_objects"] > counts["gt_objects"]:
        raise ValueError("tp_objects cannot exceed gt_objects")
    if counts["tp_objects"] > counts["pred_components"]:
        raise ValueError("tp_objects cannot exceed pred_components")
    if counts["fp_components"] != (
        counts["pred_components"] - counts["tp_objects"]
    ):
        raise ValueError("fp_components disagrees with pred_components - tp_objects")
    if counts["total_pixels"] <= 0 or counts["num_images"] <= 0:
        raise ValueError("covered records require positive total_pixels and num_images")
    if counts["fp_pixels"] > counts["total_pixels"]:
        raise ValueError("fp_pixels cannot exceed total_pixels")
    expected_pd = (
        counts["tp_objects"] / counts["gt_objects"]
        if counts["gt_objects"]
        else 0.0
    )
    if not math.isclose(float(record["pd"]), expected_pd, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("pd disagrees with raw tp_objects/gt_objects counts")
    expected_pixel_fa = counts["fp_pixels"] / counts["total_pixels"]
    if not math.isclose(
        float(record["fa_pixel"]), expected_pixel_fa, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("fa_pixel disagrees with raw fp_pixels/total_pixels counts")
    expected_component_fa = counts["fp_components"] / (
        counts["total_pixels"] / 1_000_000.0
    )
    if not math.isclose(
        float(record["fa_component_mp"]),
        expected_component_fa,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "fa_component_mp disagrees with raw fp_components/total_pixels counts"
        )


def _load_adapter_output(
    value: str | Path | Mapping[str, Any],
) -> tuple[Mapping[str, Any], str | None]:
    if isinstance(value, Mapping):
        return value, None
    path = Path(value).expanduser().resolve()
    return _read_json_mapping(path, "adapter output"), path.name


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label.title()} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label.title()} JSON must be an object")
    return payload


def _image_ids(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_nonempty_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate values")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_bool(payload: Mapping[str, Any], name: str) -> bool:
    if name not in payload or not isinstance(payload[name], bool):
        raise TypeError(f"{name} must be a boolean")
    return bool(payload[name])


def _finite_probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite probability") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _sha256_value(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 string")
    result = value.lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return result


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _safe_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-output",
        action="append",
        required=True,
        help="Online-adapter JSON; repeat together with --score-manifest to aggregate",
    )
    parser.add_argument(
        "--score-manifest",
        action="append",
        required=True,
        help="Cryptographically bound score manifest; repeat in matching order",
    )
    parser.add_argument(
        "--calibrator-checkpoint",
        action="append",
        required=True,
        help="Actual calibrator checkpoint; repeat in matching order",
    )
    parser.add_argument(
        "--label-manifest",
        action="append",
        required=True,
        help=(
            "Independent score-bound label manifest; repeat in matching order. "
            "Rejected records do not open this artifact."
        ),
    )
    parser.add_argument("--matching-rule", choices=("overlap", "centroid"))
    parser.add_argument("--centroid-distance", type=float)
    parser.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if len(args.adapter_output) != len(args.score_manifest):
        raise ValueError(
            "--adapter-output and --score-manifest must be repeated the same number of times"
        )
    if len(args.adapter_output) != len(args.calibrator_checkpoint):
        raise ValueError(
            "--adapter-output and --calibrator-checkpoint must be repeated the same number of times"
        )
    if len(args.adapter_output) != len(args.label_manifest):
        raise ValueError(
            "--adapter-output and --label-manifest must be repeated the same number of times"
        )
    evaluations = [
        evaluate_adapter_output(
            adapter_path,
            manifest_path,
            calibrator_checkpoint=calibrator_path,
            label_manifest=label_path,
            matching_rule=args.matching_rule,
            centroid_distance=args.centroid_distance,
        )
        for adapter_path, manifest_path, calibrator_path, label_path in zip(
            args.adapter_output,
            args.score_manifest,
            args.calibrator_checkpoint,
            args.label_manifest,
        )
    ]
    payload: Mapping[str, Any]
    if len(evaluations) == 1:
        payload = evaluations[0]
    else:
        payload = {
            "schema_version": ADAPTER_SUMMARY_SCHEMA_VERSION,
            "evaluations": evaluations,
            "summary": summarise_adapter_evaluations(evaluations),
        }
    rendered = json.dumps(
        payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    if args.output:
        _write_json_atomic(Path(args.output), payload)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
