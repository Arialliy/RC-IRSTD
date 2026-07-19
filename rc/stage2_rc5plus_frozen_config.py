"""Fail-closed verifier for the unique result-free RC5+ configuration.

The configuration is not a launch authorization.  This module binds its
mathematical, training, validation, checkpoint and deployment declarations to
the live implementation so that a stale three-budget/v7 configuration cannot
be presented as RC5+.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from data_ext.stage2_rc5_atomic_decision_set import (
    METHOD_IDS as FULL_PRELABEL_METHOD_IDS,
    T9_DIAGNOSTIC_SCHEMA,
)
from data_ext.stage2_rc5plus_atomic_full_decision_set import (
    DECISION_SET_SCHEMA as FULL_DECISION_SET_SCHEMA,
)
from data_ext.stage2_rc5plus_atomic_learned_decision_set import (
    DECISION_SET_SCHEMA as LEARNED_DECISION_SET_SCHEMA,
    METHOD_IDS as ATOMIC_METHOD_IDS,
)
from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_AXIS_TRANSFORM,
    BUDGET_INTERPOLATION,
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
)
from model.budget_conditioned_residual_transport_calibrator import (
    RESIDUAL_TRANSPORT_DIRECT_MODEL_ID,
    RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID,
    RESIDUAL_TRANSPORT_NO_ANCHOR_MODEL_ID,
    RESIDUAL_TRANSPORT_SCHEMA,
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
)
from model.endpoint_aware_threshold import THRESHOLD_REPRESENTATION_SCHEMA
from rc.build_stage2_rc5_context import BUNDLE_CAPABILITY_SCHEMA
from rc.stage2_calibrator_checkpoint_v8 import (
    CHECKPOINT_SCHEMA,
    EXPECTED_PARAMETER_COUNTS,
    METHODS,
)
from rc.stage2_context_tail_anchor_v2 import CONTEXT_TAIL_ANCHOR_V2_SCHEMA
from rc.stage2_rc5_feature_mask import (
    FEATURE_MASK_APPLICATION,
    FEATURE_MASK_SCHEMA,
    FEATURE_VARIANT_ACTIVE_INDICES,
    build_stage2_rc5_feature_mask,
)
from rc.stage2_rc5plus_calibrator_generation_v3 import (
    GENERATION_COMMIT_SCHEMA,
    GENERATION_MANIFEST_SCHEMA,
    RESUME_STATE_SCHEMA,
    RUN_COMMIT_SCHEMA,
)
from rc.stage2_rc5plus_cyclic_anchor_overlay import OVERLAY_SCHEMA
from rc.stage2_rc5plus_cyclic_training_view import RC5PLUS_TRAINING_VIEW_SCHEMA
from rc.stage2_rc5plus_infer_and_seal import DECISION_SCHEMA, TRANSCRIPT_SCHEMA
from rc.stage2_rc5plus_no_anchor_infer_and_seal import (
    DECISION_SCHEMA as NO_ANCHOR_DECISION_SCHEMA,
    TRANSCRIPT_SCHEMA as NO_ANCHOR_TRANSCRIPT_SCHEMA,
)
from rc.stage2_rc5plus_source_validation_view import (
    PRIMARY_SELECTION_BUDGET,
    PRIMARY_SELECTION_INDEX,
    RC5PLUS_SELECTION_GEOMETRY,
    RC5PLUS_SOURCE_VALIDATION_SCHEMA,
    RC5PLUS_VARIABLE_QUERY_SANITY_SCHEMA,
    SELECTION_RANK,
)
from rc.stage2_rc5plus_training_core import (
    RC5PLUS_MAX_EXACT_BRACKET_ROWS,
    RC5PLUS_METHODS,
    RC5PLUS_TRAINING_CORE_SCHEMA,
)
from rc.train_stage2_rc5plus_cyclic import RC5PLUS_CYCLIC_TRAINER_SCHEMA


CONFIG_SCHEMA = "rc-irstd.aaai27-stage2-crossfit-rc5plus-config.v1"
DEFAULT_CONFIG_PATH = Path("configs/aaai27_stage2_crossfit_rc5plus_v1.json")
SELECTION_RECORD_SCHEMA = "rc-irstd.calibrator-source-selection-record.v3"
_CAPABILITY = object()
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_status",
        "contains_observed_results",
        "execution_authorized",
        "official_test_accessed",
        "model_family",
        "context_feature_dim",
        "budget_contract",
        "threshold_contract",
        "model",
        "feature_mask_contract",
        "source_training_contract",
        "optimizer",
        "loss",
        "source_validation_contract",
        "checkpoint_contract",
        "deployment_contract",
        "ablation_contract",
        "performance_success_gate",
        "novelty_success_gate",
    }
)
_OPTIMIZER_FIELDS = frozenset(
    {
        "name", "learning_rate", "weight_decay", "betas", "epsilon",
        "amsgrad", "scheduler", "batch_size", "max_epochs",
        "early_stopping_patience", "gradient_clip_norm", "num_workers",
        "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic",
        "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
        "float32_matmul_precision", "amp", "dataloader_shuffle", "drop_last",
        "custom_verified_epoch_sampler", "data_iteration",
        "generation_dataloader_rng_field_usage", "process_rank", "world_size",
    }
)
_LOSS_FIELDS = frozenset(
    {
        "training_core_schema", "trainer_schema", "coordinate_huber_delta",
        "lambda_violation", "lambda_utility", "lambda_oracle",
        "lambda_smoothness", "lambda_coverage", "risk_epsilon",
        "exactness_scope", "certified_risk_claim",
    }
)


class Stage2RC5PlusFrozenConfigError(ValueError):
    """The RC5+ configuration is stale, mutable or semantically inconsistent."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2RC5PlusFrozenConfigError(
            "RC5+ configuration is not finite canonical JSON"
        ) from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage2RC5PlusFrozenConfigError(f"{name} must be a mapping")
    return value


def _closed_mapping(
    value: Any, expected_fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    result = _mapping(value, name)
    if set(result) != expected_fields:
        raise Stage2RC5PlusFrozenConfigError(f"{name} field closure mismatch")
    return result


def _exact(value: Any, expected: Any, name: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise Stage2RC5PlusFrozenConfigError(
            f"{name} differs from the live RC5+ contract"
        )


def _exact_false(value: Any, name: str) -> None:
    if type(value) is not bool or value is not False:
        raise Stage2RC5PlusFrozenConfigError(f"{name} must be exact false")


def _exact_true(value: Any, name: str) -> None:
    if type(value) is not bool or value is not True:
        raise Stage2RC5PlusFrozenConfigError(f"{name} must be exact true")


def _parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusFrozenConfig:
    payload: Mapping[str, Any]
    canonical_sha256: str
    source_path: Path | None
    source_bytes_sha256: str | None
    _capability: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("VerifiedStage2RC5PlusFrozenConfig is verifier-issued only")


def _issue(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
    source_bytes_sha256: str | None,
) -> VerifiedStage2RC5PlusFrozenConfig:
    result = object.__new__(VerifiedStage2RC5PlusFrozenConfig)
    for name, value in {
        "payload": _freeze(payload),
        "canonical_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "source_path": source_path,
        "source_bytes_sha256": source_bytes_sha256,
        "_capability": _CAPABILITY,
    }.items():
        object.__setattr__(result, name, value)
    return result


def verify_stage2_rc5plus_frozen_config_payload(
    value: Any,
    *,
    source_path: Path | None = None,
    source_bytes_sha256: str | None = None,
) -> VerifiedStage2RC5PlusFrozenConfig:
    payload = _mapping(value, "configuration")
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise Stage2RC5PlusFrozenConfigError(
            "RC5+ configuration top-level field closure mismatch"
        )
    _exact(payload["schema_version"], CONFIG_SCHEMA, "schema_version")
    _exact(
        payload["artifact_status"],
        "RESULT_FREE_DEVELOPMENT_CANDIDATE_NOT_EXECUTION_AUTHORIZED",
        "artifact_status",
    )
    _exact_false(payload["contains_observed_results"], "contains_observed_results")
    _exact_false(payload["execution_authorized"], "execution_authorized")
    _exact_false(payload["official_test_accessed"], "official_test_accessed")
    _exact(
        payload["model_family"],
        "RC5PLUS_BUDGET_CONDITIONED_ANCHOR_RESIDUAL_TRANSPORT",
        "model_family",
    )
    _exact(payload["context_feature_dim"], 93, "context_feature_dim")

    budget = _mapping(payload["budget_contract"], "budget_contract")
    _exact(
        budget["grid_exact_rationals"],
        [list(row) for row in BUDGET_KNOT_RATIONALS],
        "budget_contract.grid_exact_rationals",
    )
    _exact(
        budget["primary_grid_indices"],
        list(PRIMARY_BUDGET_KNOT_INDICES),
        "budget_contract.primary_grid_indices",
    )
    _exact(
        budget["claim_bearing_primary"],
        list(PRIMARY_SELECTION_BUDGET),
        "budget_contract.claim_bearing_primary",
    )
    _exact(budget["budget_axis_transform"], BUDGET_AXIS_TRANSFORM, "budget axis")
    _exact(budget["interpolation"], BUDGET_INTERPOLATION, "budget interpolation")
    _exact_false(budget["float_budget_authority"], "float_budget_authority")
    _exact_true(
        budget["float_product_for_count_forbidden"],
        "float_product_for_count_forbidden",
    )
    _exact_true(
        budget["distinct_request_log_coordinate_required"],
        "distinct_request_log_coordinate_required",
    )

    threshold = _mapping(payload["threshold_contract"], "threshold_contract")
    _exact(
        threshold["representation_schema"],
        THRESHOLD_REPRESENTATION_SCHEMA,
        "threshold representation",
    )
    _exact(threshold["semantics"], "prediction = probability > threshold", "threshold semantics")
    _exact_true(threshold["upper_endpoint_exact"], "upper_endpoint_exact")
    _exact_true(threshold["endpoint_suffix_required"], "endpoint_suffix_required")

    model_contract = _mapping(payload["model"], "model")
    _exact(model_contract["schema_version"], RESIDUAL_TRANSPORT_SCHEMA, "model schema")
    _exact(model_contract["hidden_dims"], [32], "hidden_dims")
    _exact(model_contract["dropout"], 0.1, "dropout")
    _exact(model_contract["minimum_residual_increment"], 1e-6, "minimum residual increment")
    _exact_true(model_contract["same_budget_anchor_required"], "same_budget_anchor_required")
    _exact_true(
        model_contract["three_point_anchor_interpolation_forbidden"],
        "three_point_anchor_interpolation_forbidden",
    )
    _exact_false(model_contract["reject_head"], "model.reject_head")
    _exact_false(
        model_contract["missing_episode_fallback"], "model.missing_episode_fallback"
    )
    methods = _mapping(model_contract["methods"], "model.methods")
    if tuple(methods) != METHODS or METHODS != RC5PLUS_METHODS:
        raise Stage2RC5PlusFrozenConfigError("method identities/order mismatch")
    direct = BudgetConditionedDirectResidualTransportCalibrator(
        93, (32,), 0.1, 1e-6
    )
    monotone = BudgetConditionedMonotoneResidualTransportCalibrator(
        93, (32,), 0.1, 1e-6
    )
    live_models = {
        "T6_PLUS": direct,
        "T7_PLUS": monotone,
        "T8_PLUS": monotone,
    }
    live_classes = {
        "T6_PLUS": "BudgetConditionedDirectResidualTransportCalibrator",
        "T7_PLUS": "BudgetConditionedMonotoneResidualTransportCalibrator",
        "T8_PLUS": "BudgetConditionedMonotoneResidualTransportCalibrator",
    }
    live_ids = {
        "T6_PLUS": RESIDUAL_TRANSPORT_DIRECT_MODEL_ID,
        "T7_PLUS": RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID,
        "T8_PLUS": RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID,
    }
    for method in METHODS:
        row = _mapping(methods[method], f"model.methods.{method}")
        _exact(row["class"], live_classes[method], f"{method}.class")
        _exact(row["model_id"], live_ids[method], f"{method}.model_id")
        _exact(
            row["structural_monotonicity"],
            method != "T6_PLUS",
            f"{method}.structural_monotonicity",
        )
        count = _parameter_count(live_models[method])
        if count != EXPECTED_PARAMETER_COUNTS[method]:
            raise Stage2RC5PlusFrozenConfigError(
                f"live {method} parameter count is not checkpoint-v8 frozen"
            )
        _exact(
            row["expected_trainable_parameters"], count, f"{method}.parameter_count"
        )
    _exact(
        model_contract["equal_capacity_contrast"], list(METHODS), "equal capacity"
    )
    ablations = _mapping(model_contract["ablation_methods"], "model.ablation_methods")
    if tuple(ablations) != ("T8_PLUS_NO_ANCHOR",):
        raise Stage2RC5PlusFrozenConfigError("no-anchor ablation identity mismatch")
    no_anchor = _mapping(
        ablations["T8_PLUS_NO_ANCHOR"], "model.ablation_methods.T8_PLUS_NO_ANCHOR"
    )
    no_anchor_model = BudgetConditionedMonotoneNoTargetAnchorCalibrator(
        93, (32,), 0.1, 1e-6
    )
    _exact(
        no_anchor["class"],
        "BudgetConditionedMonotoneNoTargetAnchorCalibrator",
        "no-anchor class",
    )
    _exact(
        no_anchor["model_id"],
        RESIDUAL_TRANSPORT_NO_ANCHOR_MODEL_ID,
        "no-anchor model id",
    )
    _exact_false(no_anchor["target_anchor_accessed"], "no-anchor target access")
    _exact_true(no_anchor["structural_monotonicity"], "no-anchor monotonicity")
    _exact(
        no_anchor["expected_trainable_parameters"],
        _parameter_count(no_anchor_model),
        "no-anchor parameter count",
    )
    _exact(no_anchor["checkpoint_schema"], CHECKPOINT_SCHEMA, "no-anchor checkpoint")
    _exact_false(no_anchor["anchor_overlay_required"], "no-anchor overlay")

    mask_contract = _mapping(payload["feature_mask_contract"], "feature mask")
    variants = tuple(FEATURE_VARIANT_ACTIVE_INDICES)
    _exact(mask_contract["schema_version"], FEATURE_MASK_SCHEMA, "feature mask schema")
    _exact(mask_contract["variants_in_order"], list(variants), "feature mask variants")
    counts = [len(build_stage2_rc5_feature_mask(name).active_indices) for name in variants]
    _exact(mask_contract["active_counts"], counts, "feature mask counts")
    _exact(mask_contract["application"], FEATURE_MASK_APPLICATION, "feature mask application")
    for field in (
        "required_in_training",
        "required_in_source_validation",
        "required_in_checkpoint",
        "required_in_inference",
    ):
        _exact_true(mask_contract[field], f"feature_mask_contract.{field}")

    source = _mapping(payload["source_training_contract"], "source training")
    _exact(source["training_view_schema"], RC5PLUS_TRAINING_VIEW_SCHEMA, "training view schema")
    _exact(source["anchor_overlay_schema"], OVERLAY_SCHEMA, "anchor overlay schema")
    _exact(source["maximum_device_bracket_rows_per_episode"], RC5PLUS_MAX_EXACT_BRACKET_ROWS, "maximum bracket rows")
    _exact_true(source["four_roles_required"], "four_roles_required")
    _exact_true(source["per_image_exact_curve_bank"], "per_image_exact_curve_bank")
    _exact_false(source["aggregate_episode_curve_materialization"], "aggregate curve")
    _exact_true(source["labels_source_only"], "labels_source_only")
    _exact_true(source["query_features_forbidden"], "query_features_forbidden")

    optimizer = _closed_mapping(
        payload["optimizer"], _OPTIMIZER_FIELDS, "optimizer"
    )
    for field, expected in {
        "name": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "scheduler": "none",
        "batch_size": 16,
        "max_epochs": 100,
        "early_stopping_patience": 20,
        "gradient_clip_norm": 5.0,
        "num_workers": 0,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "amp": False,
        "dataloader_shuffle": False,
        "drop_last": False,
        "custom_verified_epoch_sampler": True,
        "data_iteration": "custom_loop_verified_sampler_no_dataloader",
        "generation_dataloader_rng_field_usage": (
            "consumed_custom_loop_generator_state"
        ),
        "process_rank": 0,
        "world_size": 1,
    }.items():
        _exact(optimizer[field], expected, f"optimizer.{field}")

    loss = _closed_mapping(payload["loss"], _LOSS_FIELDS, "loss")
    _exact(loss["training_core_schema"], RC5PLUS_TRAINING_CORE_SCHEMA, "training core schema")
    _exact(loss["trainer_schema"], RC5PLUS_CYCLIC_TRAINER_SCHEMA, "trainer schema")
    for field, expected in {
        "coordinate_huber_delta": 1.0,
        "lambda_violation": 4.0,
        "lambda_utility": 1.0,
        "lambda_oracle": 0.1,
        "lambda_smoothness": 0.01,
        "lambda_coverage": 0.0,
        "risk_epsilon": 1e-12,
        "exactness_scope": (
            "verified_uncapped_event_set_exact_integer_budget_and_adjacent_"
            "prediction_brackets_only"
        ),
    }.items():
        _exact(loss[field], expected, f"loss.{field}")
    _exact_false(loss["certified_risk_claim"], "certified_risk_claim")

    validation = _mapping(payload["source_validation_contract"], "source validation")
    _exact(validation["selection_view_schema"], RC5PLUS_SOURCE_VALIDATION_SCHEMA, "selection view schema")
    _exact(validation["selection_record_schema"], SELECTION_RECORD_SCHEMA, "selection record schema")
    _exact(validation["selection_geometry"], RC5PLUS_SELECTION_GEOMETRY, "selection geometry")
    _exact(validation["selection_budget"], list(PRIMARY_SELECTION_BUDGET), "selection budget")
    _exact(validation["selection_grid_index"], PRIMARY_SELECTION_INDEX, "selection index")
    _exact(validation["rank"], list(SELECTION_RANK), "selection rank")
    _exact_false(validation["nonprimary_budget_epoch_rescue"], "nonprimary budget rescue")
    _exact(validation["variable_query_sanity_schema"], RC5PLUS_VARIABLE_QUERY_SANITY_SCHEMA, "variable-query schema")
    _exact_true(validation["variable_query_sanity_excluded_from_epoch_ranking"], "variable-query ranking exclusion")
    _exact_false(validation["outer_target_accessed"], "outer target accessed")

    checkpoint = _mapping(payload["checkpoint_contract"], "checkpoint")
    _exact(checkpoint["schema_version"], CHECKPOINT_SCHEMA, "checkpoint schema")
    for field in (
        "deployment_state_only",
        "training_state_forbidden",
        "feature_mask_bound",
        "training_view_identity_bound",
        "anchor_v2_required",
        "resume_state_separate_from_deployment_checkpoint",
        "optimizer_and_all_rng_states_bound",
        "source_only_primary_epoch_selection",
        "immutable_generation_commit_last",
        "exact_interruption_resume_next_step_required",
    ):
        _exact_true(checkpoint[field], f"checkpoint_contract.{field}")
    _exact(
        checkpoint["resume_state_schema"],
        RESUME_STATE_SCHEMA,
        "resume-state schema",
    )
    _exact(
        checkpoint["generation_manifest_schema"],
        GENERATION_MANIFEST_SCHEMA,
        "generation-manifest schema",
    )
    _exact(
        checkpoint["generation_commit_schema"],
        GENERATION_COMMIT_SCHEMA,
        "generation-commit schema",
    )
    _exact(
        checkpoint["run_commit_schema"],
        RUN_COMMIT_SCHEMA,
        "run-commit schema",
    )
    for field in ("reject_head", "missing_episode_fallback", "official_test_accessed"):
        _exact_false(checkpoint[field], f"checkpoint_contract.{field}")

    deployment = _mapping(payload["deployment_contract"], "deployment")
    _exact(deployment["producer_bundle_schema"], BUNDLE_CAPABILITY_SCHEMA, "producer bundle schema")
    _exact(deployment["anchor_v2_schema"], CONTEXT_TAIL_ANCHOR_V2_SCHEMA, "anchor-v2 schema")
    _exact(deployment["inference_transcript_schema"], TRANSCRIPT_SCHEMA, "inference transcript schema")
    _exact(deployment["threshold_decision_schema"], DECISION_SCHEMA, "decision schema")
    _exact(
        deployment["no_anchor_inference_transcript_schema"],
        NO_ANCHOR_TRANSCRIPT_SCHEMA,
        "no-anchor inference transcript schema",
    )
    _exact(
        deployment["no_anchor_threshold_decision_schema"],
        NO_ANCHOR_DECISION_SCHEMA,
        "no-anchor decision schema",
    )
    _exact(
        deployment["learned_atomic_set_schema"],
        LEARNED_DECISION_SET_SCHEMA,
        "learned atomic set schema",
    )
    _exact(
        deployment["full_atomic_prelabel_set_schema"],
        FULL_DECISION_SET_SCHEMA,
        "full atomic prelabel set schema",
    )
    _exact(deployment["learned_methods_in_order"], list(ATOMIC_METHOD_IDS), "atomic method order")
    _exact(
        deployment["full_atomic_prelabel_methods_in_order"],
        list(FULL_PRELABEL_METHOD_IDS),
        "full atomic prelabel method order",
    )
    _exact(
        deployment["t9_diagnostic_schema"],
        T9_DIAGNOSTIC_SCHEMA,
        "postlabel T9 diagnostic schema",
    )
    _exact_false(
        deployment["t9_included_prelabel"],
        "deployment_contract.t9_included_prelabel",
    )
    _exact(
        deployment["t9_policy"],
        "separate_postlabel_oracle_diagnostic_only",
        "deployment_contract.t9_policy",
    )
    for field in (
        "caller_feature_injection",
        "caller_curve_injection",
        "caller_anchor_injection",
        "caller_threshold_injection",
        "caller_reject_or_fallback",
        "labels_accessed_predecision",
        "query_accessed_predecision",
    ):
        _exact_false(deployment[field], f"deployment_contract.{field}")
    _exact_true(deployment["commit_last"], "deployment_contract.commit_last")

    ablation = _mapping(payload["ablation_contract"], "ablation")
    _exact(ablation["claim_bearing_method"], "T8_PLUS", "claim-bearing method")
    _exact_false(ablation["failed_innovation_may_be_silently_deleted"], "silent innovation deletion")
    _exact_true(ablation["holm_within_preregistered_family_required"], "Holm correction")
    _exact(
        ablation["required_mechanism_macro_BSR_point_strict_min"],
        0.0,
        "mechanism BSR point gate",
    )
    _exact(
        ablation["required_mechanism_macro_Pd_point_min"],
        -0.02,
        "mechanism Pd point gate",
    )
    _exact_true(
        ablation[
            "holm_adjusted_significance_required_for_claimed_contributions"
        ],
        "claimed-contribution Holm significance",
    )

    gate = _mapping(payload["performance_success_gate"], "performance gate")
    _exact(gate["comparison"], "T8_PLUS_minus_T4", "performance comparison")
    _exact(gate["primary_budget"], [1, 100_000], "performance primary budget")
    _exact(gate["macro_domain_delta_BSR_point_min"], 0.05, "BSR point gate")
    _exact(gate["macro_domain_delta_BSR_paired_95CI_lower_strict_min"], 0.0, "BSR CI gate")
    _exact(gate["macro_domain_delta_Pd_point_min"], -0.02, "Pd point gate")
    _exact(gate["macro_domain_delta_Pd_paired_95CI_lower_min"], -0.02, "Pd CI gate")
    _exact_false(gate["secondary_metric_rescue"], "secondary metric rescue")
    _exact_true(gate["fourth_independent_domain_required"], "fourth domain")
    _exact_true(gate["independent_confirmatory_one_look_required"], "confirmatory one-look")

    novelty = _mapping(payload["novelty_success_gate"], "novelty gate")
    _exact(
        novelty["minimum_strict_idea_review_score"],
        4.0,
        "minimum strict novelty score",
    )
    _exact_false(
        novelty["fatal_direct_prior_allowed"], "fatal direct prior allowed"
    )
    _exact_false(
        novelty["unsupported_first_claim_allowed"],
        "unsupported first claim allowed",
    )
    for field in (
        "updated_search_required",
        "closest_prior_difference_table_required",
        "safe_claim_boundary_required",
        "design_success_requires_gate_pass",
    ):
        _exact_true(novelty[field], f"novelty_success_gate.{field}")
    return _issue(
        payload,
        source_path=source_path,
        source_bytes_sha256=source_bytes_sha256,
    )


def verify_stage2_rc5plus_frozen_config_file(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> VerifiedStage2RC5PlusFrozenConfig:
    source = Path(path)
    try:
        data = source.read_bytes()
        value = json.loads(data)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Stage2RC5PlusFrozenConfigError(
            "RC5+ configuration file cannot be read as strict JSON"
        ) from error
    return verify_stage2_rc5plus_frozen_config_payload(
        value,
        source_path=source.resolve(),
        source_bytes_sha256=hashlib.sha256(data).hexdigest(),
    )


def assert_verified_stage2_rc5plus_frozen_config(
    value: object,
) -> VerifiedStage2RC5PlusFrozenConfig:
    if (
        type(value) is not VerifiedStage2RC5PlusFrozenConfig
        or getattr(value, "_capability", None) is not _CAPABILITY
    ):
        raise TypeError("a verifier-issued RC5+ frozen configuration is required")
    replayed = verify_stage2_rc5plus_frozen_config_payload(_plain(value.payload))
    if replayed.canonical_sha256 != value.canonical_sha256:
        raise Stage2RC5PlusFrozenConfigError("retained RC5+ configuration changed")
    if value.source_path is not None:
        current = verify_stage2_rc5plus_frozen_config_file(value.source_path)
        if (
            current.source_bytes_sha256 != value.source_bytes_sha256
            or current.canonical_sha256 != value.canonical_sha256
        ):
            raise Stage2RC5PlusFrozenConfigError(
                "RC5+ configuration source changed after verification"
            )
    return value


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG_PATH",
    "SELECTION_RECORD_SCHEMA",
    "Stage2RC5PlusFrozenConfigError",
    "VerifiedStage2RC5PlusFrozenConfig",
    "assert_verified_stage2_rc5plus_frozen_config",
    "canonical_json_bytes",
    "verify_stage2_rc5plus_frozen_config_file",
    "verify_stage2_rc5plus_frozen_config_payload",
]
