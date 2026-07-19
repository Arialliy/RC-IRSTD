from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

import rc.stage2_rc5_config as rc5
from model.endpoint_aware_pixel_calibrator import (
    ANCHOR_COORDINATE_CONTRACT,
    ANCHOR_MIX_INITIAL_WEIGHT,
    ANCHOR_MIX_PARAMETERIZATION,
    ANCHOR_MIX_RULE,
    T4_ANCHOR_SOURCE,
)
from model.endpoint_aware_threshold import representation_contract
from rc.stage2_variable_query_geometry import (
    CONSTRUCTION,
    QUERY_SIZE_POLICY,
    SCHEMA_VERSION as VARIABLE_QUERY_GEOMETRY_SCHEMA,
    WINDOW_COUNT_RULE,
)
from rc.train_stage2_rc5_cyclic import (
    build_rc5_training_execution_spec_from_verified_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/aaai27_stage2_crossfit_rc5_v3.json"
OLD_V2_PATH = REPOSITORY_ROOT / "configs/aaai27_stage2_crossfit_v2.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _mapping_paths(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if isinstance(value, dict):
        yield path
        for key, child in value.items():
            yield from _mapping_paths(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _mapping_paths(child, (*path, index))


def _list_paths(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _list_paths(child, (*path, key))
    elif isinstance(value, list):
        yield path
        for index, child in enumerate(value):
            yield from _list_paths(child, (*path, index))


def _leaf_paths(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[Any, ...]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaf_paths(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaf_paths(child, (*path, index))
    else:
        yield path


def _parent_at(value: Any, path: tuple[Any, ...]) -> tuple[Any, Any]:
    current = value
    for key in path[:-1]:
        current = current[key]
    return current, path[-1]


def _value_at(value: Any, path: tuple[Any, ...]) -> Any:
    current = value
    for key in path:
        current = current[key]
    return current


def _mutated_scalar(value: Any) -> Any:
    if type(value) is bool:
        return int(value)
    if type(value) is int:
        return str(value)
    if type(value) is float:
        return value + 0.125
    if type(value) is str:
        return value + "__mutated"
    raise AssertionError(f"unexpected config leaf: {value!r}")


def test_repository_config_verifies_with_external_sha256() -> None:
    digest = _sha256(CONFIG_PATH)
    verified = rc5.verify_stage2_rc5_config(CONFIG_PATH, digest)
    assert verified.path == CONFIG_PATH.absolute()
    assert verified.sha256 == digest
    assert verified.payload == _load()
    assert verified.payload == rc5.frozen_stage2_rc5_config()


def test_verified_repository_config_is_the_production_trainer_authority() -> None:
    digest = _sha256(CONFIG_PATH)
    verified = rc5.verify_stage2_rc5_config(CONFIG_PATH, digest)
    spec = build_rc5_training_execution_spec_from_verified_config(verified)

    assert spec.artifact_scope == "production"
    assert spec.training_contract_sha256 == digest
    assert spec.payload["optimizer"] == verified.payload["optimizer"]
    assert spec.payload["model"]["context_feature_dim"] == 93
    assert spec.payload["data_iteration"] == (
        "custom_loop_verified_sampler_no_dataloader"
    )
    assert spec.payload["generation_dataloader_rng_field_usage"] == (
        "consumed_custom_loop_generator_state"
    )
    with pytest.raises(TypeError, match="VerifiedStage2RC5Config"):
        build_rc5_training_execution_spec_from_verified_config(
            rc5.frozen_stage2_rc5_config()
        )


def test_python_expected_config_is_exactly_the_repository_json() -> None:
    assert rc5.frozen_stage2_rc5_config() == _load()


def test_preregistered_ablation_identities_are_explicit_and_result_free() -> None:
    contract = _load()["ablation_contract"]
    assert contract["schema_version"] == (
        "rc-irstd.stage2-preregistered-ablation-contract.v1"
    )
    assert contract["contains_observed_results"] is False
    assert contract["claim_bearing_primary_method"] == "T8"
    assert contract["runtime_method_flag_forbidden"] is True
    assert contract["checkpoint_v7_main_method_allowlist"] == ["T6", "T7", "T8"]

    variants = contract["anchor_variants"]
    assert variants["T4"]["identity"] == "anchor_only"
    assert variants["T4"]["expected_trainable_parameters"] == 0
    assert variants["T8_NO_ANCHOR"] == {
        "identity": "learned_only",
        "class": "LearnedOnlyMonotoneEndpointAwarePixelCalibrator",
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
    }
    assert variants["T8"]["identity"] == "combined_anchor_plus_learned"
    assert variants["T8"]["expected_trainable_parameters"] == 3141
    assert variants["T8"]["claim_bearing"] is True

    mechanism = contract["mechanism_comparison_identities"]
    assert mechanism["structural_monotonicity"]["contrast"] == "T7_minus_T6"
    assert mechanism["risk_aligned_objective"]["contrast"] == "T8_minus_T7"
    assert mechanism["analytic_anchor"]["contrast"] == "T8_minus_T8_NO_ANCHOR"
    assert mechanism["analytic_anchor"]["transparent_parameter_difference"] == 1
    assert [
        row["contrast"]
        for row in contract["feature_comparison_identities"]["contrasts"]
    ] == ["T8_minus_C4", "T8_minus_C5", "T8_minus_C6"]

    factorial = contract["stage1_by_stage2_comparison_identities"]
    assert factorial["stage1_levels"] == ["D0", "D1", "D2", "D3"]
    assert factorial["stage2_levels"] == ["T4", "T8"]
    assert factorial["stage1_contrasts_at_t8"] == [
        "D3_minus_D0",
        "D3_minus_D1",
        "D3_minus_D2",
    ]
    assert len(factorial["interaction_contrasts"]) == 3


def test_core_rc5_freeze_is_explicit() -> None:
    payload = _load()
    assert payload["schema_version"] == rc5.CONFIG_SCHEMA_VERSION
    assert payload["contains_observed_results"] is False
    assert payload["official_test_accessed"] is False
    assert payload["context_feature_dim"] == 93
    assert payload["pixel_budget_grid"] == [1e-4, 1e-5, 1e-6]
    assert payload["pixel_budget_exact_rationals"] == [
        [1, 10_000],
        [1, 100_000],
        [1, 1_000_000],
    ]
    integerization = payload["budget_integerization_contract"]
    assert integerization == {
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
    }
    assert payload["primary_pixel_budget"] == 1e-5

    geometry = payload["development_geometry"]
    assert geometry == {
        "schema_version": VARIABLE_QUERY_GEOMETRY_SCHEMA,
        "context_size": 14,
        "minimum_query_size": 28,
        "minimum_window_size": 42,
        "window_count_rule": WINDOW_COUNT_RULE,
        "query_size_policy": QUERY_SIZE_POLICY,
        "construction": CONSTRUCTION,
        "all_records_consumed": True,
        "per_window_query_size": "dynamic_manifest_bound",
    }
    assert payload["threshold_representation"] == representation_contract()
    assert payload["model"]["hidden_dims"] == [32]
    assert payload["model"]["activation"] == "GELU"
    assert payload["model"]["dropout"] == 0.1
    assert payload["model"]["minimum_raw_coordinate_gap"] == 0.001
    assert payload["model"]["raw_coordinate_bounds_hex"] == {
        "minimum": representation_contract()["raw_coordinate_min_hex"],
        "maximum": representation_contract()["raw_coordinate_max_hex"],
    }
    assert payload["model"]["analytic_anchor"] == {
        "source": T4_ANCHOR_SOURCE,
        "coordinate_contract": ANCHOR_COORDINATE_CONTRACT,
        "budget_order": "pixel_budget_grid_order",
        "shared_across_methods": ["T6", "T7", "T8"],
    }
    assert payload["model"]["global_convex_mix"] == {
        "rule": ANCHOR_MIX_RULE,
        "parameterization": ANCHOR_MIX_PARAMETERIZATION,
        "initial_weight": ANCHOR_MIX_INITIAL_WEIGHT,
        "mix_stage": "before_hard_canonicalization_then_decode",
        "global_across_budgets_and_examples": True,
        "one_scalar_per_method": True,
    }
    assert {
        method: row["learned_curve"]
        for method, row in payload["model"]["methods"].items()
    } == {"T6": "direct", "T7": "monotone", "T8": "monotone"}
    assert all(
        row["uses_shared_analytic_anchor"] is True
        for row in payload["model"]["methods"].values()
    )
    assert {
        method: row["expected_trainable_parameters"]
        for method, row in payload["model"]["methods"].items()
    } == {"T6": 3108, "T7": 3141, "T8": 3141}


def test_source_training_cyclic_geometry_and_sampler_are_separate_and_frozen() -> None:
    contract = _load()["source_training_episode_contract"]
    assert contract["schema_version"] == (
        "rc-irstd.stage2-source-cyclic-training-geometry.v1"
    )
    assert contract["scope"] == "source_oof_training_only"
    assert contract["required_verified_role"] == "oof_holdout_stage2_fit"
    assert contract["minimum_ordered_role_record_count"] == 42
    assert contract["episode_count_rule"] == "ordered_role_record_count"
    assert contract["cyclic_start_domain"] == "every_integer_in_[0,N)"
    assert (contract["context_size"], contract["query_size"]) == (14, 28)
    assert contract["index_rule"] == (
        "(cyclic_start + local_offset) % ordered_role_record_count"
    )
    assert contract["within_episode_context_query_disjoint"] is True
    assert contract["context_frequency_per_record"] == 14
    assert contract["query_frequency_per_record"] == 28
    assert contract["forbidden_roles"] == [
        "source_diagnostic_validation",
        "outer_target_diagnostic_development",
    ]
    assert contract["source_validation_selection_geometry"] == (
        "source_validation_cyclic_selection_view_c14_q28_all_n_starts"
    )
    assert contract["source_validation_sanity_geometry"] == (
        "mandatory_variable_query_all_records_consumed_once"
    )
    assert contract["outer_geometry"] == (
        "mandatory_variable_query_all_records_consumed_once"
    )
    assert contract["three_geometries_interchangeable"] is False
    assert contract["pooled_cyclic_episode_count_by_source_domain"] == {
        "NUAA-SIRST": 170,
        "NUDT-SIRST": 509,
        "IRSTD-1K": 638,
    }
    assert contract["raw_total_by_outer_fold"] == {
        "outer_leave_nuaa_sirst": 1147,
        "outer_leave_nudt_sirst": 808,
        "outer_leave_irstd_1k": 679,
    }
    sampler = contract["sampler"]
    assert sampler["schema_version"] == (
        "rc-irstd.stage2-domain-balanced-cyclic-epoch-sampler.v1"
    )
    assert sampler["algorithm_id"] == (
        "sha256_fixed_permutation_rotating_slice_domain_pairs_v1"
    )
    assert sampler["name"] == (
        "equal_source_domain_without_replacement_rotating_subset"
    )
    assert sampler["implementation_status"] == (
        "LIVE_IMPLEMENTED_AND_TRAINER_INTEGRATED_S2_I0"
    )
    assert sampler["frozen_even_batch_size"] == 16
    assert sampler["exact_batch_domain_split"] == "8_per_source_domain"
    assert sampler["epoch_tail_domain_balanced"] is True
    assert sampler["domain_balance_guarantee"] == "sampler_exact_equal_domain"
    assert sampler["seed_source"] == "verified_training_manifest_only"
    assert sampler["python_builtin_hash_forbidden"] is True
    assert sampler["manual_seed_override_forbidden"] is True
    assert sampler["ordered_selection_digest"] == (
        "sha256-canonical-json-domain-balanced-epoch-selection-v1"
    )
    assert sampler["dataloader_shuffle_forbidden"] is True
    assert sampler["trainer_consumes_ordered_selection"] is True
    assert sampler["resume_order_byte_replay_required"] is True
    assert contract["epoch_size_by_outer_fold"] == {
        "outer_leave_nuaa_sirst": 1018,
        "outer_leave_nudt_sirst": 340,
        "outer_leave_irstd_1k": 340,
    }


def test_training_checkpoint_bootstrap_and_gate_freeze() -> None:
    payload = _load()
    assert payload["optimizer"] == {
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
    }
    loss = payload["loss"]
    assert loss["name"] == "method_routed_endpoint_aware_v1"
    assert loss["T6_T7_objective"] == (
        "valid_oracle_EATC_coordinate_huber_only"
    )
    assert loss["T8_objective"] == (
        "verified_global_exact_event_curve_piecewise_linear_"
        "differentiable_risk_surrogate_plus_oracle_EATC_coordinate_huber"
    )
    assert loss["exactness_scope"] == (
        "verified_uncapped_event_set_and_adjacent_prediction_brackets_only"
    )
    assert loss["exact_discrete_risk_claim"] is False
    assert loss["lambda_coverage"] == 0.0
    assert loss["coverage_term_policy"] == (
        "removed_under_verified_global_exact_event_curve"
    )
    assert loss["verified_global_exact_event_curve"] is True
    assert loss["risk_interpolation"] == (
        "piecewise_linear_differentiable_surrogate_between_"
        "adjacent_verified_events"
    )
    assert loss["cyclic_curve_provider_contract"] == {
        "per_image_exact_curve_bank": True,
        "aggregate_curve_materialization": False,
        "episode_curve_composition": (
            "live_compose_from_per_image_exact_curves"
        ),
        "implementation_status": "LIVE_PROVIDER_BOUND_AND_AGGREGATE_ABSENT_S2_I0",
    }
    selection = payload["checkpoint_selection"]
    assert selection["record_schema_version"].endswith(".v2")
    assert selection["selection_geometry"] == (
        "source_validation_cyclic_selection_view_c14_q28_all_n_starts"
    )
    assert selection["within_domain_bsr"] == "equal_exhaustive_cyclic_start_mean"
    assert selection["within_domain_log_excess"] == (
        "equal_exhaustive_cyclic_start_mean"
    )
    assert selection["within_domain_pd"] == (
        "pooled_tp_divided_by_pooled_gt_across_all_cyclic_starts"
    )
    assert selection["source_variable_query_sanity_excluded_from_epoch_ranking"] is True
    assert selection["cyclic_starts_claimed_independent"] is False
    assert selection["cyclic_start_confidence_interval_reported"] is False

    checkpoint = payload["checkpoint_contract"]
    assert checkpoint["schema_version"] == "rc-irstd.calibrator.v7"
    assert checkpoint["artifact_kind"] == (
        "immutable_endpoint_aware_deployment_state"
    )
    assert checkpoint["deployment_state_only"] is True
    assert checkpoint["training_state_fields_forbidden"] == [
        "optimizer",
        "epoch",
        "rank",
        "history",
        "python_numpy_torch_cuda_dataloader_rng",
    ]
    assert checkpoint["generation_commit_schema_version"].endswith(".v2")
    assert checkpoint["run_commit_schema_version"].endswith(".v2")

    bootstrap = payload["bootstrap_contract"]
    assert bootstrap["factor_relation"] == "seed_crossed_with_window_query_hierarchy"
    assert bootstrap["resamples"] == 10_000
    assert bootstrap["confidence_interval"]["method"].endswith("type_7")
    assert bootstrap["shared_window_query_draw_across_selected_seed_slots"] is True
    assert bootstrap["paired_indices_byte_identical"] is True
    assert bootstrap["selected_seed_in_window_query_preimage"] is False

    assert payload["estimability_contract"][
        "minimum_expected_background_false_positives"
    ] == 20.0
    assert payload["estimability_contract"][
        "background_pixels_used_only_for_estimability"
    ] is True
    assert payload["estimability_contract"]["primary_fa_pixel_denominator"] == (
        "all_native_resolution_query_pixels"
    )
    gate = payload["primary_gate"]
    assert gate["delta_macro_bsr"]["point_estimate"] == {
        "operator": ">=",
        "value": 0.05,
    }
    assert gate["delta_macro_bsr"]["confidence_interval_lower"] == {
        "operator": ">",
        "value": 0.0,
    }
    assert gate["delta_macro_pd"]["point_estimate"] == {
        "operator": ">=",
        "value": -0.02,
    }
    assert gate["delta_macro_pd"]["confidence_interval_lower"] == {
        "operator": ">=",
        "value": -0.02,
    }


def test_old_v2_is_rejected_in_memory_and_by_file_verifier() -> None:
    old = _load(OLD_V2_PATH)
    with pytest.raises(rc5.Stage2RC5ConfigContractError):
        rc5.validate_stage2_rc5_config(old)
    with pytest.raises(rc5.Stage2RC5ConfigContractError):
        rc5.verify_stage2_rc5_config(OLD_V2_PATH, _sha256(OLD_V2_PATH))


def test_wrong_or_noncanonical_external_sha256_is_rejected() -> None:
    digest = _sha256(CONFIG_PATH)
    with pytest.raises(rc5.Stage2RC5ConfigContractError, match="SHA-256 mismatch"):
        rc5.verify_stage2_rc5_config(CONFIG_PATH, "0" * 64)
    for invalid in (None, True, digest.upper(), digest[:-1], "g" * 64):
        with pytest.raises(rc5.Stage2RC5ConfigContractError, match="lowercase"):
            rc5.verify_stage2_rc5_config(CONFIG_PATH, invalid)  # type: ignore[arg-type]


def test_every_scalar_leaf_mutation_is_rejected() -> None:
    payload = _load()
    paths = tuple(_leaf_paths(payload))
    assert len(paths) > 100
    for path in paths:
        mutated = copy.deepcopy(payload)
        parent, key = _parent_at(mutated, path)
        parent[key] = _mutated_scalar(parent[key])
        with pytest.raises(rc5.Stage2RC5ConfigContractError):
            rc5.validate_stage2_rc5_config(mutated)


def test_every_mapping_rejects_missing_and_extra_fields() -> None:
    payload = _load()
    paths = tuple(_mapping_paths(payload))
    assert len(paths) > 20
    for path in paths:
        original_mapping = _value_at(payload, path)
        key = next(iter(original_mapping))

        missing = copy.deepcopy(payload)
        target = _value_at(missing, path)
        target.pop(key)
        with pytest.raises(rc5.Stage2RC5ConfigContractError, match="closure"):
            rc5.validate_stage2_rc5_config(missing)

        extra = copy.deepcopy(payload)
        target = _value_at(extra, path)
        target["__unexpected_field__"] = None
        with pytest.raises(rc5.Stage2RC5ConfigContractError, match="closure"):
            rc5.validate_stage2_rc5_config(extra)


def test_every_list_rejects_cardinality_or_order_mutation() -> None:
    payload = _load()
    paths = tuple(_list_paths(payload))
    assert len(paths) >= 8
    for path in paths:
        shortened = copy.deepcopy(payload)
        target = _value_at(shortened, path)
        assert target
        target.pop()
        with pytest.raises(rc5.Stage2RC5ConfigContractError, match="list"):
            rc5.validate_stage2_rc5_config(shortened)

        extended = copy.deepcopy(payload)
        target = _value_at(extended, path)
        target.append(copy.deepcopy(target[-1]))
        with pytest.raises(rc5.Stage2RC5ConfigContractError, match="list"):
            rc5.validate_stage2_rc5_config(extended)

        original = _value_at(payload, path)
        if len(original) > 1 and original[0] != original[-1]:
            reordered = copy.deepcopy(payload)
            target = _value_at(reordered, path)
            target.reverse()
            with pytest.raises(rc5.Stage2RC5ConfigContractError):
                rc5.validate_stage2_rc5_config(reordered)


def test_live_representation_contract_is_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _load()
    changed = representation_contract()
    changed["canonicalization"] = "changed_capability"
    monkeypatch.setattr(rc5, "representation_contract", lambda: changed)
    with pytest.raises(rc5.Stage2RC5ConfigContractError):
        rc5.validate_stage2_rc5_config(payload)


def test_live_anchor_contract_is_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _load()
    monkeypatch.setattr(rc5, "ANCHOR_MIX_RULE", "changed_anchor_mix_rule")
    with pytest.raises(rc5.Stage2RC5ConfigContractError):
        rc5.validate_stage2_rc5_config(payload)


def test_verified_capability_cannot_be_constructed_or_forged() -> None:
    empty = rc5.VerifiedStage2RC5Config()
    with pytest.raises(RuntimeError, match="unverified"):
        _ = empty.payload

    with pytest.raises(TypeError):
        rc5.VerifiedStage2RC5Config(
            path=CONFIG_PATH,
            sha256="0" * 64,
            canonical_payload=b"{}",
        )

    forged = object.__new__(rc5.VerifiedStage2RC5Config)
    object.__setattr__(forged, "path", CONFIG_PATH)
    object.__setattr__(forged, "sha256", "0" * 64)
    object.__setattr__(forged, "canonical_payload", b"{}")
    object.__setattr__(forged, "_capability", object())
    with pytest.raises(RuntimeError, match="unverified"):
        _ = forged.payload


def test_file_verifier_rejects_byte_mutation_duplicate_keys_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    payload = _load()
    mutated = copy.deepcopy(payload)
    mutated["primary_pixel_budget"] = 1e-4
    mutated_path = tmp_path / "mutated.json"
    mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(rc5.Stage2RC5ConfigContractError):
        rc5.verify_stage2_rc5_config(mutated_path, _sha256(mutated_path))

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        '{"schema_version":"a","schema_version":"b"}', encoding="utf-8"
    )
    with pytest.raises(rc5.Stage2RC5ConfigContractError, match="duplicate"):
        rc5.verify_stage2_rc5_config(duplicate_path, _sha256(duplicate_path))

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(rc5.Stage2RC5ConfigContractError, match="non-finite"):
        rc5.verify_stage2_rc5_config(nonfinite_path, _sha256(nonfinite_path))


def test_file_verifier_rejects_symlink(tmp_path: Path) -> None:
    link = tmp_path / "config-link.json"
    link.symlink_to(CONFIG_PATH)
    with pytest.raises(rc5.Stage2RC5ConfigContractError, match="non-symlink"):
        rc5.verify_stage2_rc5_config(link, _sha256(CONFIG_PATH))


def test_file_verifier_rejects_fd_identity_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = _sha256(CONFIG_PATH)
    real_fstat = rc5.os.fstat
    call_count = 0

    def unstable_fstat(descriptor: int) -> Any:
        nonlocal call_count
        observed = real_fstat(descriptor)
        call_count += 1
        if call_count != 2:
            return observed
        return SimpleNamespace(
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_mode=observed.st_mode,
            st_size=observed.st_size + 1,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
        )

    monkeypatch.setattr(rc5.os, "fstat", unstable_fstat)
    with pytest.raises(rc5.Stage2RC5ConfigContractError, match="changed while read"):
        rc5.verify_stage2_rc5_config(CONFIG_PATH, digest)
