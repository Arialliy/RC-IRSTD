"""Strict result-free verifier for the frozen RC5 Stage-2 config.

The verifier is intentionally separate from the checkpoint-v6/v2 config path.
It requires an externally supplied SHA-256, rejects duplicate/non-finite JSON,
enforces recursive field closure and exact scalar types, and replays the live
endpoint-aware representation contract before granting a capability object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from model.endpoint_aware_pixel_calibrator import (
    ANCHOR_COORDINATE_CONTRACT,
    ANCHOR_MIX_INITIAL_WEIGHT,
    ANCHOR_MIX_PARAMETERIZATION,
    ANCHOR_MIX_RULE,
    T4_ANCHOR_SOURCE,
)
from model.endpoint_aware_threshold import representation_contract
from rc.stage2_calibrator_checkpoint_v7 import (
    ARTIFACT_KIND as CALIBRATOR_DEPLOYMENT_ARTIFACT_KIND,
)
from rc.stage2_cyclic_training_geometry import (
    CONTEXT_SIZE as CYCLIC_TRAINING_CONTEXT_SIZE,
    EPISODE_COUNT_RULE as CYCLIC_TRAINING_EPISODE_COUNT_RULE,
    INDEX_RULE as CYCLIC_TRAINING_INDEX_RULE,
    MINIMUM_ROLE_RECORDS as CYCLIC_TRAINING_MINIMUM_ROLE_RECORDS,
    QUERY_SIZE as CYCLIC_TRAINING_QUERY_SIZE,
    ROLE_SCOPE as CYCLIC_TRAINING_ROLE_SCOPE,
    SCHEMA_VERSION as CYCLIC_TRAINING_SCHEMA,
)
from rc.stage2_domain_balanced_cyclic_sampler import (
    ALGORITHM_ID as DOMAIN_BALANCED_SAMPLER_ALGORITHM,
    SCHEMA_VERSION as DOMAIN_BALANCED_SAMPLER_SCHEMA,
)
from rc.stage2_variable_query_geometry import (
    CONSTRUCTION,
    CONTEXT_SIZE,
    MINIMUM_QUERY_SIZE,
    MINIMUM_WINDOW_SIZE,
    QUERY_SIZE_POLICY,
    SCHEMA_VERSION as VARIABLE_QUERY_GEOMETRY_SCHEMA,
    WINDOW_COUNT_RULE,
)


CONFIG_SCHEMA_VERSION = "rc-irstd.aaai27-stage2-crossfit-config.v3"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VERIFIED_CAPABILITY = object()


class Stage2RC5ConfigContractError(ValueError):
    """The RC5 configuration or its external binding is not exact."""


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5Config:
    """Immutable verified config capability backed by canonical JSON bytes."""

    path: Path
    sha256: str
    canonical_payload: bytes
    _capability: object

    @property
    def payload(self) -> dict[str, Any]:
        if getattr(self, "_capability", None) is not _VERIFIED_CAPABILITY:
            raise RuntimeError("unverified RC5 config capability")
        value = json.loads(self.canonical_payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("verified RC5 config capability is corrupt")
        return value


def frozen_stage2_rc5_config() -> dict[str, Any]:
    """Return a fresh canonical copy of the complete result-free freeze."""

    threshold_representation = representation_contract()
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "artifact_status": "RESULT_FREE_FROZEN_CONFIGURATION",
        "contains_observed_results": False,
        "official_test_accessed": False,
        "context_feature_dim": 93,
        "development_geometry": {
            "schema_version": VARIABLE_QUERY_GEOMETRY_SCHEMA,
            "context_size": CONTEXT_SIZE,
            "minimum_query_size": MINIMUM_QUERY_SIZE,
            "minimum_window_size": MINIMUM_WINDOW_SIZE,
            "window_count_rule": WINDOW_COUNT_RULE,
            "query_size_policy": QUERY_SIZE_POLICY,
            "construction": CONSTRUCTION,
            "all_records_consumed": True,
            "per_window_query_size": "dynamic_manifest_bound",
        },
        "source_training_episode_contract": {
            "schema_version": CYCLIC_TRAINING_SCHEMA,
            "scope": CYCLIC_TRAINING_ROLE_SCOPE,
            "required_verified_role": "oof_holdout_stage2_fit",
            "minimum_ordered_role_record_count": (
                CYCLIC_TRAINING_MINIMUM_ROLE_RECORDS
            ),
            "episode_count_rule": CYCLIC_TRAINING_EPISODE_COUNT_RULE,
            "cyclic_start_domain": "every_integer_in_[0,N)",
            "context_size": CYCLIC_TRAINING_CONTEXT_SIZE,
            "query_size": CYCLIC_TRAINING_QUERY_SIZE,
            "index_rule": CYCLIC_TRAINING_INDEX_RULE,
            "within_episode_context_query_disjoint": True,
            "context_frequency_per_record": CYCLIC_TRAINING_CONTEXT_SIZE,
            "query_frequency_per_record": CYCLIC_TRAINING_QUERY_SIZE,
            "forbidden_roles": [
                "source_diagnostic_validation",
                "outer_target_diagnostic_development",
            ],
            "source_validation_selection_geometry": (
                "source_validation_cyclic_selection_view_c14_q28_all_n_starts"
            ),
            "source_validation_sanity_geometry": (
                "mandatory_variable_query_all_records_consumed_once"
            ),
            "outer_geometry": "mandatory_variable_query_all_records_consumed_once",
            "three_geometries_interchangeable": False,
            "pooled_cyclic_episode_count_by_source_domain": {
                "NUAA-SIRST": 170,
                "NUDT-SIRST": 509,
                "IRSTD-1K": 638,
            },
            "raw_total_by_outer_fold": {
                "outer_leave_nuaa_sirst": 1147,
                "outer_leave_nudt_sirst": 808,
                "outer_leave_irstd_1k": 679,
            },
            "sampler": {
                "schema_version": DOMAIN_BALANCED_SAMPLER_SCHEMA,
                "algorithm_id": DOMAIN_BALANCED_SAMPLER_ALGORITHM,
                "name": (
                    "equal_source_domain_without_replacement_rotating_subset"
                ),
                "implementation_status": (
                    "LIVE_IMPLEMENTED_AND_TRAINER_INTEGRATED_S2_I0"
                ),
                "source_domain_count": 2,
                "draws_per_domain_per_epoch": (
                    "min(total_episodes_across_the_two_source_domains)"
                ),
                "epoch_size_rule": (
                    "2*min(total_episodes_across_the_two_source_domains)"
                ),
                "without_replacement_within_domain_epoch": True,
                "rotating_subset_across_epochs": True,
                "ordered_unit": (
                    "one_episode_per_source_domain_per_domain_pair"
                ),
                "frozen_even_batch_size": 16,
                "exact_batch_domain_split": "8_per_source_domain",
                "epoch_tail_domain_balanced": True,
                "domain_balance_guarantee": "sampler_exact_equal_domain",
                "seed_source": "verified_training_manifest_only",
                "python_builtin_hash_forbidden": True,
                "manual_seed_override_forbidden": True,
                "ordered_selection_digest": (
                    "sha256-canonical-json-domain-balanced-"
                    "epoch-selection-v1"
                ),
                "dataloader_shuffle_forbidden": True,
                "trainer_consumes_ordered_selection": True,
                "resume_order_byte_replay_required": True,
            },
            "epoch_size_by_outer_fold": {
                "outer_leave_nuaa_sirst": 1018,
                "outer_leave_nudt_sirst": 340,
                "outer_leave_irstd_1k": 340,
            },
        },
        "pixel_budget_grid": [1e-4, 1e-5, 1e-6],
        "pixel_budget_exact_rationals": [
            [1, 10_000],
            [1, 100_000],
            [1, 1_000_000],
        ],
        "budget_integerization_contract": {
            "count_formula": "(numerator * pixel_count) // denominator",
            "applies_to": [
                "T4_analytic_anchor_exact_order_statistic",
                "oracle_feasible_false_positive_count",
                "exact_curve_replay_budget_count",
            ],
            "float_product_for_count_forbidden": True,
            "fraction_from_float_forbidden": True,
            "fa_pixel_denominator": "all_native_resolution_query_pixels",
            "background_pixels_role": "conservative_estimability_only",
        },
        "primary_pixel_budget": 1e-5,
        "threshold_semantics": "prediction = probability > threshold",
        "threshold_representation": threshold_representation,
        "model": {
            "hidden_dims": [32],
            "activation": "GELU",
            "dropout": 0.1,
            "raw_coordinate_bounds_hex": {
                "minimum": threshold_representation["raw_coordinate_min_hex"],
                "maximum": threshold_representation["raw_coordinate_max_hex"],
            },
            "minimum_raw_coordinate_gap": 0.001,
            "reject_head": False,
            "missing_episode_fallback": False,
            "analytic_anchor": {
                "source": T4_ANCHOR_SOURCE,
                "coordinate_contract": ANCHOR_COORDINATE_CONTRACT,
                "budget_order": "pixel_budget_grid_order",
                "shared_across_methods": ["T6", "T7", "T8"],
            },
            "global_convex_mix": {
                "rule": ANCHOR_MIX_RULE,
                "parameterization": ANCHOR_MIX_PARAMETERIZATION,
                "initial_weight": ANCHOR_MIX_INITIAL_WEIGHT,
                "mix_stage": "before_hard_canonicalization_then_decode",
                "global_across_budgets_and_examples": True,
                "one_scalar_per_method": True,
            },
            "methods": {
                "T6": {
                    "class": "DirectEndpointAwarePixelCalibrator",
                    "objective": "endpoint_aware_coordinate_huber_only",
                    "structural_monotonicity": False,
                    "learned_curve": "direct",
                    "uses_shared_analytic_anchor": True,
                    "expected_trainable_parameters": 3108,
                },
                "T7": {
                    "class": "MonotoneEndpointAwarePixelCalibrator",
                    "objective": "endpoint_aware_coordinate_huber_only",
                    "structural_monotonicity": True,
                    "learned_curve": "monotone",
                    "uses_shared_analytic_anchor": True,
                    "expected_trainable_parameters": 3141,
                },
                "T8": {
                    "class": "MonotoneEndpointAwarePixelCalibrator",
                    "objective": (
                        "verified_global_exact_event_curve_piecewise_linear_"
                        "risk_surrogate_plus_oracle_coordinate_huber"
                    ),
                    "structural_monotonicity": True,
                    "learned_curve": "monotone",
                    "uses_shared_analytic_anchor": True,
                    "expected_trainable_parameters": 3141,
                },
            },
        },
        "ablation_contract": {
            "schema_version": (
                "rc-irstd.stage2-preregistered-ablation-contract.v1"
            ),
            "artifact_status": "RESULT_FREE_PREREGISTERED_ABLATIONS",
            "contains_observed_results": False,
            "claim_bearing_primary_method": "T8",
            "runtime_method_flag_forbidden": True,
            "checkpoint_v7_main_method_allowlist": ["T6", "T7", "T8"],
            "anchor_variants": {
                "T4": {
                    "identity": "anchor_only",
                    "implementation": "T4_exact_order_statistic",
                    "analytic_anchor": True,
                    "learned_curve": False,
                    "objective": "not_applicable",
                    "expected_trainable_parameters": 0,
                    "artifact_role": "analytic_baseline",
                    "claim_bearing": False,
                },
                "T8_NO_ANCHOR": {
                    "identity": "learned_only",
                    "class": (
                        "LearnedOnlyMonotoneEndpointAwarePixelCalibrator"
                    ),
                    "analytic_anchor": False,
                    "learned_curve": True,
                    "objective": (
                        "verified_global_exact_event_curve_piecewise_linear_"
                        "risk_surrogate_plus_oracle_coordinate_huber"
                    ),
                    "expected_trainable_parameters": 3140,
                    "artifact_role": "risk_aligned_ablation_only",
                    "claim_bearing": False,
                    "checkpoint_v7_supported": False,
                },
                "T8": {
                    "identity": "combined_anchor_plus_learned",
                    "class": "MonotoneEndpointAwarePixelCalibrator",
                    "analytic_anchor": True,
                    "learned_curve": True,
                    "objective": (
                        "verified_global_exact_event_curve_piecewise_linear_"
                        "risk_surrogate_plus_oracle_coordinate_huber"
                    ),
                    "expected_trainable_parameters": 3141,
                    "artifact_role": "claim_bearing_primary_method",
                    "claim_bearing": True,
                    "checkpoint_v7_supported": True,
                },
            },
            "mechanism_comparison_identities": {
                "end_to_end_learned_correction": {
                    "contrast": "T8_minus_T4",
                    "question": "combined_learned_correction_value",
                },
                "structural_monotonicity": {
                    "contrast": "T7_minus_T6",
                    "question": "monotone_structure_value",
                    "held_fixed": [
                        "T4_analytic_anchor",
                        "endpoint_aware_coordinate_huber_objective",
                        "full_93D_features",
                    ],
                },
                "risk_aligned_objective": {
                    "contrast": "T8_minus_T7",
                    "question": "query_risk_aligned_objective_value",
                    "held_fixed": [
                        "T4_analytic_anchor",
                        "monotone_93_to_32_to_4_architecture",
                        "full_93D_features",
                    ],
                },
                "analytic_anchor": {
                    "contrast": "T8_minus_T8_NO_ANCHOR",
                    "question": "analytic_anchor_value",
                    "held_fixed": [
                        "verified_global_exact_event_curve_piecewise_linear_"
                        "risk_surrogate_plus_oracle_coordinate_huber",
                        "monotone_93_to_32_to_4_learned_branch",
                        "full_93D_features",
                    ],
                    "transparent_parameter_difference": 1,
                },
            },
            "feature_comparison_identities": {
                "reference": "C3_T8_full_features_0_92",
                "fixed_except_feature_slice": [
                    "T8_model_identity",
                    "T8_loss",
                    "T4_analytic_anchor",
                    "outer_fold_seed_window_query_identity",
                ],
                "contrasts": [
                    {
                        "contrast": "T8_minus_C4",
                        "variant": "C4",
                        "feature_indices": "0-38",
                        "feature_set": "score_only",
                    },
                    {
                        "contrast": "T8_minus_C5",
                        "variant": "C5",
                        "feature_indices": "0-78",
                        "feature_set": "score_plus_peak",
                    },
                    {
                        "contrast": "T8_minus_C6",
                        "variant": "C6",
                        "feature_indices": "0-86",
                        "feature_set": "score_plus_peak_plus_gray_no_source_distance",
                    },
                ],
            },
            "stage1_by_stage2_comparison_identities": {
                "stage1_levels": ["D0", "D1", "D2", "D3"],
                "stage2_levels": ["T4", "T8"],
                "cell_identity": (
                    "D{stage1}_x_{stage2}_same_outer_fold_seed_window_query"
                ),
                "stage1_contrasts_at_t8": [
                    "D3_minus_D0",
                    "D3_minus_D1",
                    "D3_minus_D2",
                ],
                "stage2_contrasts_within_each_stage1": [
                    "D0_T8_minus_D0_T4",
                    "D1_T8_minus_D1_T4",
                    "D2_T8_minus_D2_T4",
                    "D3_T8_minus_D3_T4",
                ],
                "interaction_contrasts": [
                    "(D3_T8_minus_D3_T4)_minus_(D0_T8_minus_D0_T4)",
                    "(D3_T8_minus_D3_T4)_minus_(D1_T8_minus_D1_T4)",
                    "(D3_T8_minus_D3_T4)_minus_(D2_T8_minus_D2_T4)",
                ],
                "stage1_segmentation_and_stage2_operating_point_reported_separately": True,
            },
            "claim_policy": {
                "one_comparison_one_mechanism_question": True,
                "primary_gate_cannot_substitute_for_ablation": True,
                "unsupported_mechanism_claim_must_be_deleted_or_narrowed": True,
            },
        },
        "optimizer": {
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
            "amp": False,
            "shuffle_training": False,
            "custom_verified_epoch_sampler": True,
            "drop_last": False,
        },
        "loss": {
            "name": "method_routed_endpoint_aware_v1",
            "T6_T7_objective": (
                "valid_oracle_EATC_coordinate_huber_only"
            ),
            "T8_objective": (
                "verified_global_exact_event_curve_piecewise_linear_"
                "differentiable_risk_surrogate_plus_oracle_EATC_"
                "coordinate_huber"
            ),
            "exactness_scope": (
                "verified_uncapped_event_set_and_adjacent_prediction_"
                "brackets_only"
            ),
            "exact_discrete_risk_claim": False,
            "coordinate_huber_delta": 1.0,
            "lambda_violation": 4.0,
            "lambda_utility": 1.0,
            "lambda_oracle": 0.1,
            "lambda_smoothness": 0.01,
            "lambda_coverage": 0.0,
            "coverage_term_policy": (
                "removed_under_verified_global_exact_event_curve"
            ),
            "verified_global_exact_event_curve": True,
            "risk_interpolation": (
                "piecewise_linear_differentiable_surrogate_between_"
                "adjacent_verified_events"
            ),
            "cyclic_curve_provider_contract": {
                "per_image_exact_curve_bank": True,
                "aggregate_curve_materialization": False,
                "episode_curve_composition": (
                    "live_compose_from_per_image_exact_curves"
                ),
                "implementation_status": (
                    "LIVE_PROVIDER_BOUND_AND_AGGREGATE_ABSENT_S2_I0"
                ),
            },
            "risk_epsilon": 1e-12,
        },
        "checkpoint_selection": {
            "record_schema_version": (
                "rc-irstd.calibrator-source-selection-record.v2"
            ),
            "selection_geometry": (
                "source_validation_cyclic_selection_view_c14_q28_all_n_starts"
            ),
            "source_domain_weighting": "equal_one_half",
            "within_domain_bsr": "equal_exhaustive_cyclic_start_mean",
            "within_domain_log_excess": "equal_exhaustive_cyclic_start_mean",
            "within_domain_pd": (
                "pooled_tp_divided_by_pooled_gt_across_all_cyclic_starts"
            ),
            "source_variable_query_sanity_excluded_from_epoch_ranking": True,
            "cyclic_starts_claimed_independent": False,
            "cyclic_start_confidence_interval_reported": False,
            "rank": [
                "macro_source_BSR_max",
                "macro_source_LogExcess_min",
                "macro_source_Pd_max",
                "earlier_epoch_on_exact_tie",
            ],
            "outer_target_accessed": False,
        },
        "collection_contract": {
            "schema_version": (
                "rc-irstd.stage2-source-cyclic-training-collection.v1"
            ),
            "commit_schema_version": (
                "rc-irstd.stage2-source-cyclic-training-collection-commit.v1"
            ),
            "training_geometry_schema_version": CYCLIC_TRAINING_SCHEMA,
            "source_selection_geometry_schema_version": (
                "rc-irstd.stage2-source-validation-cyclic-selection-view.v1"
            ),
            "source_sanity_and_outer_geometry_schema_version": (
                VARIABLE_QUERY_GEOMETRY_SCHEMA
            ),
            "three_geometries_interchangeable": False,
            "required_bundle_members": [
                "npy_mmap_arrays",
                "identity_only_jsonl",
                "manifest",
                "commit_last",
            ],
            "external_sha256_required_for_every_member": True,
            "statistics_config_external_sha256_required": True,
            "statistics_config_shared_object_train_validation": True,
            "train_role": "oof_holdout_stage2_fit",
            "validation_role": "source_diagnostic_validation_detector_full_fit_only",
            "both_source_domains_required": True,
            "outer_target_absent": True,
            "per_image_exact_curve_materialized_once": True,
            "aggregate_cyclic_curve_materialized": False,
            "cyclic_episode_jsonl_payload": "C14_Q28_identity_references_only",
            "score_authority": "VerifiedStage2RC5ScoreBundleV2_replayed",
            "run_complete_identity_bound": True,
            "episode_weighting": "verified_equal_source_domain_sampler",
            "standardizer_fit_scope": "training_contexts_only",
            "standardizer_dtype": "float64",
            "standardizer_scale_floor": 1e-8,
        },
        "checkpoint_contract": {
            "schema_version": "rc-irstd.calibrator.v7",
            "artifact_kind": CALIBRATOR_DEPLOYMENT_ARTIFACT_KIND,
            "deployment_state_only": True,
            "training_state_fields_forbidden": [
                "optimizer",
                "epoch",
                "rank",
                "history",
                "python_numpy_torch_cuda_dataloader_rng",
            ],
            "generation_commit_schema_version": (
                "rc-irstd.calibrator-generation-commit.v2"
            ),
            "run_commit_schema_version": "rc-irstd.calibrator-run-commit.v2",
            "serialization": "torch_tensors_and_primitives_weights_only",
            "immutable_epoch_generations": True,
            "commit_published_last": True,
            "resume_requires_external_generation_commit_sha256": True,
            "threshold_representation_schema": threshold_representation[
                "schema_version"
            ],
            "reject_head": False,
            "official_test_accessed": False,
        },
        "seed_contract": {
            "training_manifest_schema_version": (
                "rc-irstd.stage2-seed-derivation-manifest.v1"
            ),
            "bootstrap_factor_manifest_schema_version": (
                "rc-irstd.stage2-bootstrap-factor-seed-manifest.v1"
            ),
            "algorithm_id": "sha256_domain_separated_seed_v1",
            "base_seeds": [42, 123, 3407],
            "method_roles": {
                "T6": "baseline_t6_direct_mlp::not_applicable",
                "T7": "baseline_t7_monotone_oracle::not_applicable",
                "T8": "stage2_calibrator_t8::not_applicable",
            },
            "bootstrap_factor_roots": (
                "separate_seed_factor_and_window_query_factor_per_domain"
            ),
            "python_builtin_hash_forbidden": True,
            "manual_seed_override_forbidden": True,
        },
        "bootstrap_contract": {
            "schema_version": "rc-irstd.stage2-crossed-paired-bootstrap.v1",
            "protocol_id": (
                "outer_fixed_seed_x_window_query_crossed_paired_bootstrap_v1"
            ),
            "comparison": "T8_minus_T4",
            "resamples": 10_000,
            "confidence_interval": {
                "confidence_level": 0.95,
                "method": "two_sided_percentile_hyndman_fan_type_7",
                "quantiles": [0.025, 0.975],
            },
            "domain_resampling": "none_fixed_equal_one_third",
            "seed_factor": "resample_three_training_seeds_with_replacement",
            "window_factor": (
                "resample_frozen_window_count_with_replacement_once_per_domain_replicate"
            ),
            "query_factor": (
                "resample_actual_query_size_with_replacement_within_selected_window"
            ),
            "factor_relation": "seed_crossed_with_window_query_hierarchy",
            "shared_window_query_draw_across_selected_seed_slots": True,
            "paired_methods": ["T8", "T4"],
            "paired_indices_byte_identical": True,
            "method_id_in_draw_preimage": False,
            "selected_seed_in_window_query_preimage": False,
        },
        "estimability_contract": {
            "primary_budget": 1e-5,
            "expected_background_false_positives_formula": (
                "primary_budget * total_background_pixels"
            ),
            "minimum_expected_background_false_positives": 20.0,
            "comparison_operator": ">=",
            "background_pixels_used_only_for_estimability": True,
            "primary_fa_pixel_denominator": "all_native_resolution_query_pixels",
            "inestimable_primary_cell_policy": "primary_gate_false_no_imputation",
        },
        "primary_gate": {
            "comparison": "T8_minus_T4",
            "confidence_interval_source": "crossed_paired_bootstrap",
            "delta_macro_bsr": {
                "point_estimate": {"operator": ">=", "value": 0.05},
                "confidence_interval_lower": {"operator": ">", "value": 0.0},
            },
            "delta_macro_pd": {
                "point_estimate": {"operator": ">=", "value": -0.02},
                "confidence_interval_lower": {"operator": ">=", "value": -0.02},
            },
            "missing_or_nonfinite_policy": "gate_false_no_imputation",
        },
    }


def _assert_exact(observed: Any, expected: Any, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, Mapping):
            raise Stage2RC5ConfigContractError(f"{name} must be a mapping")
        observed_keys = frozenset(observed)
        expected_keys = frozenset(expected)
        if observed_keys != expected_keys:
            missing = sorted(expected_keys - observed_keys, key=repr)
            extra = sorted(observed_keys - expected_keys, key=repr)
            raise Stage2RC5ConfigContractError(
                f"{name} field closure mismatch: missing={missing}, extra={extra}"
            )
        for key, expected_value in expected.items():
            _assert_exact(observed[key], expected_value, f"{name}.{key}")
        return
    if isinstance(expected, list):
        if type(observed) is not list or len(observed) != len(expected):
            raise Stage2RC5ConfigContractError(f"{name} list mismatch")
        for index, (observed_value, expected_value) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _assert_exact(observed_value, expected_value, f"{name}[{index}]")
        return
    if type(observed) is not type(expected):
        raise Stage2RC5ConfigContractError(f"{name} exact type mismatch")
    if isinstance(expected, float):
        if not math.isfinite(observed) or observed != expected:
            raise Stage2RC5ConfigContractError(f"{name} exact value mismatch")
        return
    if observed != expected:
        raise Stage2RC5ConfigContractError(f"{name} exact value mismatch")


def validate_stage2_rc5_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory payload against the complete RC5 freeze."""

    if not isinstance(payload, Mapping):
        raise TypeError("RC5 Stage2 config must be a mapping")
    expected = frozen_stage2_rc5_config()
    _assert_exact(payload, expected, "config")
    return json.loads(
        json.dumps(
            expected,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2RC5ConfigContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise Stage2RC5ConfigContractError(f"non-finite JSON number: {value}")


def _parse_config_bytes(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5ConfigContractError(f"invalid RC5 config JSON: {error}") from error
    if not isinstance(payload, dict):
        raise Stage2RC5ConfigContractError("RC5 config JSON must contain an object")
    return payload


def _external_sha256(value: Any) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise Stage2RC5ConfigContractError(
            "expected_sha256 must be exact lowercase SHA-256"
        )
    return value


def _read_stable_regular_file(path: str | Path) -> tuple[Path, bytes]:
    expanded = Path(path).expanduser()
    candidate = expanded if expanded.is_absolute() else expanded.absolute()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise Stage2RC5ConfigContractError(f"RC5 config is unavailable: {error}") from error
    if resolved != candidate:
        raise Stage2RC5ConfigContractError(
            "RC5 config must use its canonical absolute non-symlink path"
        )

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise Stage2RC5ConfigContractError("O_NOFOLLOW is required for RC5 config")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise Stage2RC5ConfigContractError(f"RC5 config open failed: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2RC5ConfigContractError(
                "RC5 config must be a non-symlink regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(candidate, follow_symlinks=False)
    except Stage2RC5ConfigContractError:
        raise
    except OSError as error:
        raise Stage2RC5ConfigContractError(f"RC5 config read failed: {error}") from error
    finally:
        os.close(descriptor)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(after) != identity(path_after):
        raise Stage2RC5ConfigContractError("RC5 config changed while read")
    return candidate, b"".join(chunks)


def _verified_capability(
    *, path: Path, sha256: str, canonical_payload: bytes
) -> VerifiedStage2RC5Config:
    value = object.__new__(VerifiedStage2RC5Config)
    object.__setattr__(value, "path", path)
    object.__setattr__(value, "sha256", sha256)
    object.__setattr__(value, "canonical_payload", canonical_payload)
    object.__setattr__(value, "_capability", _VERIFIED_CAPABILITY)
    return value


def verify_stage2_rc5_config(
    path: str | Path, expected_sha256: str
) -> VerifiedStage2RC5Config:
    """Verify external bytes, SHA-256, JSON closure, and the live EATC contract."""

    expected_digest = _external_sha256(expected_sha256)
    verified_path, data = _read_stable_regular_file(path)
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise Stage2RC5ConfigContractError("RC5 config external SHA-256 mismatch")
    payload = _parse_config_bytes(data)
    canonical = validate_stage2_rc5_config(payload)
    canonical_bytes = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(data).hexdigest() != actual_digest:
        raise Stage2RC5ConfigContractError("RC5 config bytes changed after validation")
    return _verified_capability(
        path=verified_path,
        sha256=actual_digest,
        canonical_payload=canonical_bytes,
    )


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "Stage2RC5ConfigContractError",
    "VerifiedStage2RC5Config",
    "frozen_stage2_rc5_config",
    "validate_stage2_rc5_config",
    "verify_stage2_rc5_config",
]
