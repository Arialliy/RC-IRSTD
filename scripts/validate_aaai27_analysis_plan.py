"""Fail-closed validation for the strict AAAI-27 analysis-plan contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_ext.split_utils import read_split_entries, sample_id_from_entry
from scripts.validate_stage1_pilot_matrix import (
    validate_matrix as validate_stage1_pilot_matrix,
)


SCHEMA_VERSION = "rc-irstd.aaai27-analysis-plan.v1"
SPLIT_SCHEMA_VERSION = "rc-irstd.aaai27-official-train-splits.v2"
REQUIRED_STAGE1_VARIANTS = {"D0", "D1", "D2", "D3"}
REQUIRED_THRESHOLD_BASELINES = {f"T{index}" for index in range(10)}
REQUIRED_CALIBRATOR_ABLATIONS = {f"C{index}" for index in range(7)}
FORBIDDEN_PLACEHOLDERS = ("tbd", "todo", "fill_me", "placeholder")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_strict_json(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


def _assert_no_placeholders(value: Any, location: str = "$") -> None:
    if value is None:
        raise ValueError(f"null is forbidden in a frozen plan: {location}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in frozen plan: {location}")
    if isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in FORBIDDEN_PLACEHOLDERS):
            raise ValueError(f"placeholder text is forbidden at {location}: {value!r}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_placeholders(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_placeholders(item, f"{location}[{index}]")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be an object")
    return value


def _require_keys(value: Mapping[str, Any], keys: set[str], location: str) -> None:
    missing = keys.difference(value)
    if missing:
        raise ValueError(f"{location} is missing required keys: {sorted(missing)}")


def _resolve_under_root(root: Path, relative: str, *, location: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"{location} must be repository-relative: {relative}")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{location} escapes repository root: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{location} does not exist: {resolved}")
    return resolved


def _ids_from_file(path: Path) -> list[str]:
    result = [sample_id_from_entry(value) for value in read_split_entries(path)]
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate canonical IDs in {path}")
    return result


def _window_ids(payload: Mapping[str, Any], expected_role: str) -> tuple[set[str], int]:
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"unsupported split-window schema for {expected_role}")
    if payload.get("role") != expected_role:
        raise ValueError(f"split-window role mismatch: expected {expected_role}")
    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError(f"{expected_role} must contain at least one window")
    ids: set[str] = set()
    for index, window_value in enumerate(windows):
        window = _require_mapping(window_value, f"{expected_role}.windows[{index}]")
        if window.get("protocol") != "iid" or window.get(
            "temporal_causality_claimed"
        ) is not False:
            raise ValueError(f"{expected_role} window must be non-causal IID")
        context = [str(value) for value in window.get("context_image_ids", [])]
        query = [str(value) for value in window.get("query_image_ids", [])]
        if not context or not query or set(context).intersection(query):
            raise ValueError(f"invalid context/query IDs in {expected_role} window {index}")
        block = set(context).union(query)
        if ids.intersection(block):
            raise ValueError(f"overlapping IID blocks in {expected_role}")
        ids.update(block)
    return ids, len(windows)


def validate_split_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    development_domains: Sequence[str],
    expected_context_size: int,
    expected_query_size: int,
) -> dict[str, int]:
    manifest = _require_mapping(load_strict_json(manifest_path), "split_manifest")
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported frozen split manifest schema")
    role = _require_mapping(manifest.get("role_contract"), "split_manifest.role_contract")
    deprecated_role_fields = {
        "outer_target_official_train_used",
        "outer_target_official_train_allowed_in_same_outer_fold",
    }
    present_deprecated_fields = deprecated_role_fields.intersection(role)
    if present_deprecated_fields:
        raise ValueError(
            "split manifest contains deprecated ambiguous role fields: "
            f"{sorted(present_deprecated_fields)}"
        )
    if role.get("official_test_emitted") is not False:
        raise ValueError("frozen split manifest may not emit official test IDs")
    if role.get("same_fold_domain_roles_are_mutually_exclusive") is not True:
        raise ValueError("same-fold detector/pseudo-target roles must be exclusive")
    if role.get("official_test_labels_read_for_quarantine") is not False:
        raise ValueError("quarantine may not read official-test labels")
    if role.get("outer_target_official_train_used_for_detector_fit") is not False:
        raise ValueError("outer-target official train may not enter detector fitting")
    if (
        role.get(
            "outer_target_detector_diagnostic_used_for_development_evaluation"
        )
        is not True
    ):
        raise ValueError(
            "outer-target detector_diagnostic must be the development evaluation role"
        )
    if role.get("outer_target_diagnostic_selects_checkpoint") is not False:
        raise ValueError("outer-target diagnostics may not select checkpoints")
    quarantine_manifest = _require_mapping(
        manifest.get("development_quarantine"),
        "split_manifest.development_quarantine",
    )
    if quarantine_manifest.get("status") != "applied_before_random_partitioning":
        raise ValueError("development quarantine was not applied before partitioning")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, list):
        raise TypeError("split_manifest.datasets must be an array")
    names = [str(item.get("dataset_name")) for item in datasets]
    if set(names) != set(development_domains) or len(names) != len(set(names)):
        raise ValueError("split manifest domains do not exactly match development domains")

    split_root = manifest_path.parent
    meta_train_counts: dict[str, int] = {}
    for raw_dataset in datasets:
        dataset = _require_mapping(raw_dataset, "split_manifest.dataset")
        name = str(dataset["dataset_name"])
        if dataset.get("dataset_type") != "iid_images":
            raise ValueError(f"{name} must be frozen as iid_images")
        train_split = _resolve_under_root(
            repository_root,
            str(dataset["official_train_split"]),
            location=f"{name}.official_train_split",
        )
        test_split = _resolve_under_root(
            repository_root,
            str(dataset["official_test_split"]),
            location=f"{name}.official_test_split",
        )
        if sha256_file(train_split) != dataset["official_train_split_sha256"]:
            raise ValueError(f"{name} official train split SHA-256 drift")
        if sha256_file(test_split) != dataset["official_test_split_sha256"]:
            raise ValueError(f"{name} official test split SHA-256 drift")
        official_train = set(_ids_from_file(train_split))
        official_test = set(_ids_from_file(test_split))
        if official_train.intersection(official_test):
            raise ValueError(f"{name} official train/test IDs overlap")
        if len(official_train) != int(dataset["official_train_count"]):
            raise ValueError(f"{name} official train count drift")

        quarantine = _require_mapping(
            dataset.get("development_quarantine"),
            f"{name}.development_quarantine",
        )
        effective_path = (split_root / str(
            quarantine["effective_development_train_file"]
        )).resolve()
        quarantined_path = (split_root / str(
            quarantine["quarantined_file"]
        )).resolve()
        for path, expected_hash in (
            (effective_path, quarantine["effective_development_train_sha256"]),
            (quarantined_path, quarantine["quarantined_sha256"]),
        ):
            try:
                path.relative_to(split_root.resolve())
            except ValueError as error:
                raise ValueError(f"{name} quarantine artifact escapes split root") from error
            if sha256_file(path) != expected_hash:
                raise ValueError(f"{name} quarantine artifact SHA-256 drift")
        effective_train = set(_ids_from_file(effective_path))
        quarantined = (
            set(_ids_from_file(quarantined_path))
            if quarantined_path.stat().st_size
            else set()
        )
        if (
            effective_train.intersection(quarantined)
            or effective_train.union(quarantined) != official_train
        ):
            raise ValueError(
                f"{name} effective development/quarantine is not an official-train partition"
            )
        if effective_train.intersection(official_test) or quarantined.intersection(
            official_test
        ):
            raise ValueError(f"{name} quarantine partition contains official-test IDs")
        if len(effective_train) != int(
            quarantine["effective_development_train_count"]
        ) or len(quarantined) != int(quarantine["quarantined_count"]):
            raise ValueError(f"{name} quarantine counts drift")

        detector = _require_mapping(dataset.get("detector"), f"{name}.detector")
        fit_path = (split_root / str(detector["fit_file"])).resolve()
        diagnostic_path = (split_root / str(detector["diagnostic_file"])).resolve()
        for path in (fit_path, diagnostic_path):
            try:
                path.relative_to(split_root.resolve())
            except ValueError as error:
                raise ValueError(f"{name} detector split escapes manifest root") from error
        fit = set(_ids_from_file(fit_path))
        diagnostic = set(_ids_from_file(diagnostic_path))
        if sha256_file(fit_path) != detector["fit_sha256"]:
            raise ValueError(f"{name} detector-fit SHA-256 drift")
        if sha256_file(diagnostic_path) != detector["diagnostic_sha256"]:
            raise ValueError(f"{name} detector-diagnostic SHA-256 drift")
        if fit.intersection(diagnostic) or fit.union(diagnostic) != effective_train:
            raise ValueError(
                f"{name} detector split is not a disjoint effective-train cover"
            )
        if fit.union(diagnostic).intersection(official_test):
            raise ValueError(f"{name} detector split contains official test IDs")

        meta = _require_mapping(dataset.get("meta"), f"{name}.meta")
        if int(meta["context_size"]) != expected_context_size or int(
            meta["query_size"]
        ) != expected_query_size:
            raise ValueError(f"{name} meta context/query size drift")
        if meta.get("protocol") != "iid_non_overlapping_blocks" or meta.get(
            "temporal_causality_claimed"
        ) is not False:
            raise ValueError(f"{name} meta protocol must remain non-causal IID")
        train_window_path = (split_root / str(meta["train_file"])).resolve()
        val_window_path = (split_root / str(meta["validation_file"])).resolve()
        unused_path = (split_root / str(meta["unused_file"])).resolve()
        for path, expected_hash in (
            (train_window_path, meta["train_sha256"]),
            (val_window_path, meta["validation_sha256"]),
            (unused_path, meta["unused_sha256"]),
        ):
            if sha256_file(path) != expected_hash:
                raise ValueError(f"{name} frozen meta artifact SHA-256 drift: {path}")
        meta_train_ids, train_count = _window_ids(
            _require_mapping(load_strict_json(train_window_path), "meta_train"),
            "meta_train",
        )
        meta_val_ids, val_count = _window_ids(
            _require_mapping(load_strict_json(val_window_path), "meta_validation"),
            "meta_validation",
        )
        unused = set(_ids_from_file(unused_path)) if unused_path.stat().st_size else set()
        if meta_train_ids.intersection(meta_val_ids):
            raise ValueError(f"{name} meta train/validation image IDs overlap")
        emitted = meta_train_ids.union(meta_val_ids).union(unused)
        if emitted != effective_train or emitted.intersection(official_test):
            raise ValueError(
                f"{name} meta roles are not an effective official-train-only cover"
            )
        if train_count != int(meta["train_window_count"]) or val_count != int(
            meta["validation_window_count"]
        ):
            raise ValueError(f"{name} meta window count drift")
        meta_train_counts[name] = train_count
    return meta_train_counts


def _git_output(repository_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _validate_near_duplicate_chain(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
    contracts: Mapping[str, Path],
) -> dict[str, int]:
    near = _require_mapping(plan["near_duplicate_contract"], "near_duplicate_contract")
    if near.get("status") != "passed_with_development_quarantine":
        raise ValueError("near-duplicate contract is not resolved by quarantine")

    source = _require_mapping(
        load_strict_json(contracts["near_duplicate_source_audit"]),
        "near_duplicate_source_audit",
    )
    if (
        source.get("status") != "review_required"
        or source.get("image_only") is not True
        or source.get("labels_scores_checkpoints_or_metrics_read") is not False
    ):
        raise ValueError("source near-duplicate audit has an invalid scope/status")
    algorithm = _require_mapping(source.get("algorithm"), "source_audit.algorithm")
    if int(algorithm["phash_hamming_distance_max"]) != int(
        near["candidate_hamming_distance_max"]
    ) or float(algorithm["confirmation_cosine_min"]) != float(
        near["confirmation_cosine_min"]
    ):
        raise ValueError("near-duplicate algorithm differs from frozen thresholds")
    pairs = source.get("confirmed_near_duplicate_pairs")
    if not isinstance(pairs, list) or len(pairs) != int(
        near["source_confirmed_pair_count"]
    ):
        raise ValueError("source near-duplicate pair count drift")
    pair_ids = [str(pair.get("candidate_id")) for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("source near-duplicate candidate IDs are not unique")
    expected_exclusions: dict[str, set[str]] = {}
    cross_dataset_count = 0
    for pair in pairs:
        left = _require_mapping(pair.get("left"), "source_pair.left")
        right = _require_mapping(pair.get("right"), "source_pair.right")
        if left["dataset_name"] != right["dataset_name"]:
            cross_dataset_count += 1
            continue
        endpoints = {str(left["split_role"]): left, str(right["split_role"]): right}
        if set(endpoints) != {"official_train", "official_test"}:
            raise ValueError("source pair is not an official train/test candidate")
        train = endpoints["official_train"]
        expected_exclusions.setdefault(str(train["dataset_name"]), set()).add(
            str(train["image_id"])
        )
    if cross_dataset_count != int(near["cross_dataset_confirmed_pair_count"]):
        raise ValueError("cross-dataset near-duplicate count drift")

    quarantine = _require_mapping(
        load_strict_json(contracts["near_duplicate_quarantine"]),
        "near_duplicate_quarantine",
    )
    if quarantine.get("schema_version") != (
        "rc-irstd.aaai27-near-duplicate-quarantine.v1"
    ) or quarantine.get("status") != "resolved_by_development_quarantine":
        raise ValueError("unsupported or unresolved quarantine contract")
    source_binding = _require_mapping(
        quarantine.get("source_audit"), "quarantine.source_audit"
    )
    if (
        source_binding.get("path")
        != contracts["near_duplicate_source_audit"].relative_to(
            repository_root
        ).as_posix()
        or source_binding.get("sha256")
        != sha256_file(contracts["near_duplicate_source_audit"])
    ):
        raise ValueError("quarantine does not bind the frozen source audit")
    decisions = quarantine.get("candidate_decisions")
    if not isinstance(decisions, list) or {
        str(item.get("candidate_id")) for item in decisions
    } != set(pair_ids):
        raise ValueError("quarantine decisions do not exactly cover source candidates")
    decision_exclusions: dict[str, set[str]] = {}
    for item in decisions:
        if (
            item.get("final_decision") != "same_scene_related"
            or item.get("action")
            != "exclude_official_train_member_from_all_development_roles"
        ):
            raise ValueError("unsupported near-duplicate candidate resolution")
        decision_exclusions.setdefault(str(item["dataset_name"]), set()).add(
            str(item["official_train_image_id"])
        )
    if decision_exclusions != expected_exclusions:
        raise ValueError("quarantine exclusions differ from source train endpoints")
    if sum(map(len, decision_exclusions.values())) != int(
        near["unique_quarantined_official_train_id_count"]
    ):
        raise ValueError("unique quarantine count drift")

    split_manifest = _require_mapping(
        load_strict_json(contracts["official_train_split_manifest"]),
        "official_train_split_manifest",
    )
    manifest_quarantine = _require_mapping(
        split_manifest.get("development_quarantine"),
        "split_manifest.development_quarantine",
    )
    if (
        manifest_quarantine.get("config_sha256")
        != sha256_file(contracts["near_duplicate_quarantine"])
        or manifest_quarantine.get("source_audit_sha256")
        != sha256_file(contracts["near_duplicate_source_audit"])
    ):
        raise ValueError("split manifest does not bind quarantine evidence")

    effective = _require_mapping(
        load_strict_json(contracts["near_duplicate_effective_audit"]),
        "near_duplicate_effective_audit",
    )
    if (
        effective.get("status") != "passed"
        or effective.get("near_duplicate_contract_passed") is not True
        or int(effective.get("confirmed_near_duplicate_pair_count", -1)) != 0
    ):
        raise ValueError("effective development/test near-duplicate audit did not pass")
    effective_inputs = effective.get("inputs")
    if not isinstance(effective_inputs, list):
        raise TypeError("effective near-duplicate audit inputs must be an array")
    observed_inputs = {
        (str(item["dataset_name"]), str(item["split_role"])): (
            str(item["split_file"]),
            str(item["split_sha256"]),
        )
        for item in effective_inputs
    }
    expected_inputs: dict[tuple[str, str], tuple[str, str]] = {}
    split_root = contracts["official_train_split_manifest"].parent
    for dataset in split_manifest["datasets"]:
        name = str(dataset["dataset_name"])
        q = _require_mapping(dataset["development_quarantine"], f"{name}.quarantine")
        effective_path = (
            split_root / str(q["effective_development_train_file"])
        ).resolve()
        expected_inputs[(name, "development_train")] = (
            effective_path.relative_to(repository_root).as_posix(),
            str(q["effective_development_train_sha256"]),
        )
        expected_inputs[(name, "official_test")] = (
            str(dataset["official_test_split"]),
            str(dataset["official_test_split_sha256"]),
        )
    if observed_inputs != expected_inputs:
        raise ValueError("effective near-duplicate audit inputs differ from frozen splits")

    dataset_contract = _require_mapping(
        load_strict_json(contracts["dataset_contract"]), "dataset_contract"
    )
    if dataset_contract.get("schema_version") != (
        "rc-irstd.aaai27-dataset-contract.v1"
    ):
        raise ValueError("unsupported dataset contract schema")
    dataset_split_binding = _require_mapping(
        dataset_contract.get("split_manifest"), "dataset_contract.split_manifest"
    )
    if dataset_split_binding.get("sha256") != sha256_file(
        contracts["official_train_split_manifest"]
    ):
        raise ValueError("dataset contract does not bind frozen split manifest")
    misc111 = _require_mapping(
        dataset_contract.get("required_special_case"),
        "dataset_contract.required_special_case",
    )
    if (
        misc111.get("dataset_name") != "NUAA-SIRST"
        or misc111.get("image_id") != "Misc_111"
        or misc111.get("alignment_applied") is not True
        or misc111.get("mask_alignment") != "nearest"
    ):
        raise ValueError("NUAA Misc_111 alignment contract is not frozen")

    independence = _require_mapping(
        load_strict_json(contracts["domain_independence_decision"]),
        "domain_independence_decision",
    )
    if independence.get("status") != "passed_for_stage1_development_only":
        raise ValueError("domain independence decision does not authorize Stage 1")
    return {
        "source_confirmed_pair_count": len(pairs),
        "quarantined_train_id_count": sum(map(len, decision_exclusions.values())),
        "effective_confirmed_pair_count": 0,
    }


def _dynamic_blockers(
    plan: Mapping[str, Any],
    *,
    repository_root: Path,
    plan_path: Path,
    meta_train_counts: Mapping[str, int],
) -> tuple[set[str], set[str]]:
    runtime: set[str] = set()
    scientific: set[str] = set()
    if _git_output(repository_root, "status", "--porcelain=v1").stdout.strip():
        runtime.add("WORKTREE_NOT_CLEAN")
    tracked = _git_output(
        repository_root,
        "ls-files",
        "--error-unmatch",
        str(plan_path.relative_to(repository_root)),
    )
    if tracked.returncode != 0:
        runtime.add("ANALYSIS_PLAN_NOT_GIT_TRACKED")

    stage1 = _require_mapping(plan["stage1_contract"], "stage1_contract")
    variants = _require_mapping(stage1["variants"], "stage1_contract.variants")
    if any(
        variant.get("strict_variant_identity_implemented") is not True
        for variant in variants.values()
    ):
        runtime.add("D0_D3_PAIRED_RUNNER_NOT_VALIDATED")

    stage2 = _require_mapping(plan["stage2_contract"], "stage2_contract")
    baselines = _require_mapping(
        stage2["threshold_baselines"], "stage2_contract.threshold_baselines"
    )
    ablations = _require_mapping(
        stage2["calibrator_ablations"], "stage2_contract.calibrator_ablations"
    )
    if any(
        baselines[f"T{index}"].get("strict_runner_validated") is not True
        for index in range(9)
    ) or any(
        ablations[f"C{index}"].get("strict_runner_validated") is not True
        for index in range(7)
    ):
        scientific.add("STAGE2_BASELINE_RUNNERS_INCOMPLETE")

    domains = _require_mapping(plan["domains"], "domains")
    if int(domains["current_independent_domains"]) < int(
        domains["minimum_independent_domains"]
    ):
        scientific.add("FOURTH_INDEPENDENT_DOMAIN_ABSENT")
    if not domains.get("confirmatory"):
        scientific.add("EXTERNAL_UNSEEN_TARGETS_ABSENT")
    minimum_windows = int(plan["internal_splits"]["minimum_meta_train_windows_per_pseudo_target"])
    if meta_train_counts.get("NUAA-SIRST", 0) < minimum_windows:
        scientific.add("NUAA_META_TRAIN_WINDOWS_BELOW_MINIMUM")
    return scientific, runtime


def validate_plan(plan_path: Path, repository_root: Path) -> dict[str, Any]:
    plan = _require_mapping(load_strict_json(plan_path), "analysis_plan")
    _assert_no_placeholders(plan)
    _require_keys(
        plan,
        {
            "schema_version",
            "plan_status",
            "contains_observed_results",
            "authority",
            "authorization",
            "blocker_codes",
            "hash_contracts",
            "domains",
            "data_roles",
            "internal_splits",
            "stage1_contract",
            "stage2_contract",
            "statistics",
            "near_duplicate_contract",
            "gates",
        },
        "analysis_plan",
    )
    if plan["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported analysis-plan schema")
    if plan["plan_status"] not in {
        "frozen_stage1_protocol_pending_release",
        "frozen_stage1_pilot_authorized",
    }:
        raise ValueError("unsupported Stage-1 analysis-plan status")
    if plan["contains_observed_results"] is not False:
        raise ValueError("analysis plan must not contain observed model results")
    authorization = _require_mapping(plan["authorization"], "authorization")
    for key in (
        "stage2_real_data_training",
        "official_test_model_evaluation",
        "paper_performance_claims",
    ):
        if authorization.get(key) is not False:
            raise ValueError(f"three-domain frozen plan must keep {key}=false")
    authorized_plan = plan["plan_status"] == "frozen_stage1_pilot_authorized"
    if authorization.get("stage1_development_comparisons") is not authorized_plan:
        raise ValueError("Stage-1 authorization differs from plan status")
    if authorization.get("gate_minus_1") is not authorized_plan:
        raise ValueError("Gate -1 authorization differs from plan status")

    data_roles = _require_mapping(plan["data_roles"], "data_roles")
    deprecated_role_fields = {
        "outer_target_official_train_used",
        "outer_target_official_train_allowed_in_same_outer_fold",
    }
    present_deprecated_fields = deprecated_role_fields.intersection(data_roles)
    if present_deprecated_fields:
        raise ValueError(
            "analysis plan contains deprecated ambiguous role fields: "
            f"{sorted(present_deprecated_fields)}"
        )
    if data_roles.get("outer_target_official_train_used_for_detector_fit") is not False:
        raise ValueError("outer-target official train may not enter detector fitting")
    if (
        data_roles.get(
            "outer_target_detector_diagnostic_used_for_development_evaluation"
        )
        is not True
    ):
        raise ValueError(
            "outer-target detector_diagnostic must be enabled for development evaluation"
        )
    if data_roles.get("outer_target_diagnostic_selects_checkpoint") is not False:
        raise ValueError("outer-target diagnostics may not select checkpoints")

    contracts = _require_mapping(plan["hash_contracts"], "hash_contracts")
    resolved_contracts: dict[str, Path] = {}
    for name, raw_contract in contracts.items():
        contract = _require_mapping(raw_contract, f"hash_contracts.{name}")
        path = _resolve_under_root(
            repository_root,
            str(contract["path"]),
            location=f"hash_contracts.{name}.path",
        )
        if sha256_file(path) != contract["sha256"]:
            raise ValueError(f"hash contract drift: {name}")
        resolved_contracts[name] = path
    required_contracts = {
        "stage1_config",
        "stage1_pilot_matrix",
        "stage2_config",
        "official_train_split_manifest",
        "near_duplicate_source_audit",
        "near_duplicate_quarantine",
        "near_duplicate_effective_audit",
        "dataset_contract",
        "domain_independence_decision",
    }
    if set(resolved_contracts) != required_contracts:
        raise ValueError(
            "analysis-plan hash contracts differ from the required frozen set"
        )

    domains = _require_mapping(plan["domains"], "domains")
    development_domains = [str(value) for value in domains["development"]]
    if len(development_domains) != len(set(development_domains)):
        raise ValueError("development domains must be unique")
    if domains.get("confirmatory") != []:
        raise ValueError("confirmatory domain must remain unassigned before acquisition")
    if domains.get("dataset_type") != "iid_images" or domains.get(
        "temporal_or_causal_claim"
    ) is not False:
        raise ValueError("current domains must be treated as non-causal IID images")

    internal = _require_mapping(plan["internal_splits"], "internal_splits")
    meta_train_counts = validate_split_manifest(
        resolved_contracts["official_train_split_manifest"],
        repository_root=repository_root,
        development_domains=development_domains,
        expected_context_size=int(internal["iid_context_size"]),
        expected_query_size=int(internal["iid_query_size"]),
    )
    near_duplicate_summary = _validate_near_duplicate_chain(
        plan,
        repository_root=repository_root,
        contracts=resolved_contracts,
    )

    stage1 = _require_mapping(plan["stage1_contract"], "stage1_contract")
    variants = _require_mapping(stage1["variants"], "stage1_contract.variants")
    if set(variants) != REQUIRED_STAGE1_VARIANTS:
        raise ValueError("Stage-1 plan must define exactly D0-D3")
    stage1_config = _require_mapping(
        load_strict_json(resolved_contracts["stage1_config"]), "stage1_config"
    )
    registry = _require_mapping(
        stage1_config.get("stage1_variant_registry"),
        "stage1_config.stage1_variant_registry",
    )
    if set(registry) != REQUIRED_STAGE1_VARIANTS:
        raise ValueError("strict Stage-1 config must define exactly D0-D3")
    for variant_id in sorted(REQUIRED_STAGE1_VARIANTS):
        planned = _require_mapping(
            variants[variant_id], f"stage1_contract.variants.{variant_id}"
        )
        configured = _require_mapping(
            registry[variant_id], f"stage1_config.stage1_variant_registry.{variant_id}"
        )
        for field in (
            "risk_objective",
            "lambda_margin",
            "trainable_tail",
            "stop_gradient_tail",
        ):
            if planned[field] != configured[field]:
                raise ValueError(f"{variant_id} {field} differs from strict config")
        if int(planned["exclusion_radius"]) != int(
            stage1_config["gt_exclusion_radius"]
        ):
            raise ValueError(f"{variant_id} exclusion radius differs from strict config")
        if planned.get("strict_variant_identity_implemented") is not True:
            raise ValueError(f"{variant_id} strict identity is not implemented")
    d3 = _require_mapping(variants["D3"], "stage1_contract.variants.D3")
    if float(d3["lambda_margin"]) != float(stage1_config["lambda_margin"]):
        raise ValueError("D3 lambda_margin differs from strict Stage-1 config")
    if int(d3["exclusion_radius"]) != int(stage1_config["gt_exclusion_radius"]):
        raise ValueError("D3 exclusion radius differs from strict Stage-1 config")
    if stage1["seeds"] != plan["statistics"]["training_seeds"]:
        raise ValueError("Stage-1 and statistical seed contracts differ")
    pilot_matrix_report = validate_stage1_pilot_matrix(
        resolved_contracts["stage1_pilot_matrix"],
        repository_root,
        plan_path=plan_path,
    )
    if pilot_matrix_report["run_count"] != 8:
        raise ValueError("Stage-1 pilot matrix did not validate exactly eight runs")

    stage2 = _require_mapping(plan["stage2_contract"], "stage2_contract")
    baselines = _require_mapping(
        stage2["threshold_baselines"], "stage2_contract.threshold_baselines"
    )
    if set(baselines) != REQUIRED_THRESHOLD_BASELINES:
        raise ValueError("Stage-2 plan must define exactly T0-T9")
    for baseline_id in (f"T{index}" for index in range(9)):
        if baselines[baseline_id].get("reject_supported") is not False:
            raise ValueError(f"{baseline_id} violates the No-Reject contract")
    if baselines["T9"].get("selection_or_gate_input") is not False:
        raise ValueError("target oracle T9 cannot enter selection or gates")
    ablations = _require_mapping(
        stage2["calibrator_ablations"], "stage2_contract.calibrator_ablations"
    )
    if set(ablations) != REQUIRED_CALIBRATOR_ABLATIONS:
        raise ValueError("Stage-2 plan must define exactly C0-C6")
    if ablations["C3"].get("main_method") is not True:
        raise ValueError("C3 must remain the main calibrator ablation")
    grid = [float(value) for value in stage2["pixel_budget_grid_descending"]]
    if any(left <= right for left, right in zip(grid, grid[1:])):
        raise ValueError("Stage-2 budget grid must be strictly descending")
    primary_budget = float(stage2["primary_pixel_budget"])
    if primary_budget not in grid:
        raise ValueError("primary pixel budget is absent from the frozen grid")
    stage2_config = _require_mapping(
        load_strict_json(resolved_contracts["stage2_config"]), "stage2_config"
    )
    if grid != [float(value) for value in stage2_config["pixel_budget_grid"]]:
        raise ValueError("analysis-plan budget grid differs from strict Stage-2 config")
    if stage2.get("reject_head") is not False or stage2_config.get("reject_head") is not False:
        raise ValueError("strict Stage-2 path must remain No-Reject")

    gates = _require_mapping(plan["gates"], "gates")
    if set(gates) != {"G_minus_1", "G1", "G2", "G3", "G4", "G5", "G6"}:
        raise ValueError("analysis plan must define exactly G_minus_1 and G1-G6")
    for gate_id, gate in gates.items():
        expected = authorized_plan if gate_id == "G_minus_1" else False
        if gate.get("current") is not expected:
            raise ValueError(f"{gate_id} current state differs from frozen plan status")

    scientific_blockers, runtime_blockers = _dynamic_blockers(
        plan,
        repository_root=repository_root,
        plan_path=plan_path,
        meta_train_counts=meta_train_counts,
    )
    declared_blockers = set(str(value) for value in plan["blocker_codes"])
    if declared_blockers != scientific_blockers:
        raise ValueError(
            "declared blocker_codes differ from scientific blockers: "
            f"declared={sorted(declared_blockers)}, mechanical={sorted(scientific_blockers)}"
        )
    gate_minus_1 = authorized_plan and not runtime_blockers
    return {
        "artifact_type": "rc_irstd_aaai27_analysis_plan_audit",
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "plan_status": plan["plan_status"],
        "plan_sha256": sha256_file(plan_path),
        "contains_observed_results": False,
        "gate_minus_1": gate_minus_1,
        "development_domains": development_domains,
        "meta_train_window_counts": dict(meta_train_counts),
        "blocker_codes": sorted(scientific_blockers | runtime_blockers),
        "scientific_blocker_codes": sorted(scientific_blockers),
        "runtime_repository_state_blockers": sorted(runtime_blockers),
        "near_duplicate_summary": near_duplicate_summary,
        "stage1_pilot_matrix_summary": {
            "sha256": pilot_matrix_report["matrix_sha256"],
            "run_count": pilot_matrix_report["run_count"],
            "phase_count": pilot_matrix_report["phase_count"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", default="configs/aaai27_analysis_plan.json"
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--require-gate-minus-one", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.repository_root).expanduser().resolve()
    plan_path = Path(args.plan).expanduser()
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    plan_path = plan_path.resolve()
    report = validate_plan(plan_path, root)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_gate_minus_one and report["gate_minus_1"] is not True:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
