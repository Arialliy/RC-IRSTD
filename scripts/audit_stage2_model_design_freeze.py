#!/usr/bin/env python3
"""Result-free S2_I0 auditor for the frozen RC-IRSTD model design.

This auditor intentionally does not accept dataset, checkpoint, GPU, metric or
official-test arguments.  It runs only a frozen synthetic/fault-injection test
suite, validates the model/config/governance contracts, hashes the reviewed
source surface, and publishes an immutable result-free PASS/HOLD bundle.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    # Direct ``python scripts/...py`` execution otherwise exposes only the
    # scripts directory and cannot import the repository model packages.
    sys.path.insert(0, str(REPOSITORY_ROOT))
SCHEMA = "rc-irstd.stage2-model-design-freeze-audit.v2"
COMMIT_SCHEMA = "rc-irstd.stage2-model-design-freeze-audit-commit.v2"
EXPECTED_G1_SHA = "6348cbfa96c0c4db7a6d594eba8d012d4bfd6e06cf67815e9445c941f5a63eef"
EXPECTED_B3_SHA = "d55b29dfb891710cf114d708b4c9f0c63d8d440d5efb107b81faf6b6e34bd1f6"
EXPECTED_MATERIALIZATION_INDEX_SHA = "b52a7938a13df78b8157a39fed02695ff9268bfffbbf7c665c10c1f66fe52d94"
EXPECTED_MATERIALIZATION_AUDIT_SHA = "657defc76e22723fd6dc0e82c49649f1584b50d53bbabb6066dc901030abed2d"
EXPECTED_RUN_CONTRACT_INDEX_SHA = "125bc0e2573bb4740cdf17ab4a266ce137b0a3323e5caed4e98fd2d75d14599f"
EXPECTED_B1_INTEGRATION_SHA = "4d4ce52653e872ffa4f3f71b9475edb03b934b27d0cb4d6c914d63c92b0131d6"
EXPECTED_SEED_MANIFEST_SHA = "4f426ea44e09b4f086092a8a41d5d0cff156b20b2bb1433a2ba3bed5c987604b"
EXPECTED_B2_HOLD_SHA = "c1f68f340b1226ddb5ad00fe3c7988b5bd3735380d976aee6093e90446330623"
EXPECTED_B2_AMENDMENT_SHA = "cc15832de4f85abfae84c4d49a5ac098cff253d0fecfa885d0d7735d3ef5aea6"
EXPECTED_B4_AMENDMENT_SHA = "ff1e575703318214f17d261fc583bd70744c125f53fbd31118a33a6de57e64a4"
AUTHORITATIVE_STAGE2_CONFIG = "configs/aaai27_stage2_crossfit_v2.json"
EXPECTED_STAGE2_CONFIG_SHA = "dd5e49c9633612e52c00091cfcb2543b48f5fd3f0d7fc5690f297ec0e7d9d963"
FINAL_AUTHORITY_FILES = (
    (
        "model_design_document",
        "RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md",
        "49f83fd0389f7aa0406b161ca46f0343bd5aa8e5a991861d77b313c910589144",
    ),
    (
        "preregistered_experiment_document",
        "RC-IRSTD_AAAI27_预注册实验矩阵与结果表模板_20260717.md",
        "bdcf8826d1f72fd1bf8416ba809f5dac68c052f834bf255ed1efe021e4518ccf",
    ),
    ("authoritative_stage2_config", AUTHORITATIVE_STAGE2_CONFIG, EXPECTED_STAGE2_CONFIG_SHA),
)
EXPECTED_W12_PREOPEN_INDEX_SHA = "8646f21423eb25731d621726edc63741a8594d9df62c235916f0096e99342383"

W12_PREOPEN_ARTIFACTS = (
    (
        "w12_preopen_index",
        "outputs/stage2_protocol/RC4_STAGE2_W12_THREE_DOMAIN_PREOPEN_PLAN_INDEX_20260717.json",
        EXPECTED_W12_PREOPEN_INDEX_SHA,
    ),
    (
        "w12_nuaa_metadata",
        "outputs/stage2_protocol/RC4_STAGE2_W12_NUAA_SIRST_OFFICIAL_SPLIT_METADATA_20260717.json",
        "ab3c17edd6ac9e4c36e0a73107904307bc04353e63f690e83f3610273e7358fb",
    ),
    (
        "w12_nuaa_plan",
        "outputs/stage2_protocol/RC4_STAGE2_W12_NUAA_SIRST_PREOPEN_PLAN_20260717.json",
        "bbd7863ef273e10375e1635d11562281002b67edf7c46071b8a89ec63942ad9c",
    ),
    (
        "w12_nudt_metadata",
        "outputs/stage2_protocol/RC4_STAGE2_W12_NUDT_SIRST_OFFICIAL_SPLIT_METADATA_20260717.json",
        "a19c7338c14923e3b4675d14019bb461d2d22c9889d5f53993d9f55682252d1b",
    ),
    (
        "w12_nudt_plan",
        "outputs/stage2_protocol/RC4_STAGE2_W12_NUDT_SIRST_PREOPEN_PLAN_20260717.json",
        "3c82dcaa256d58ff01e5b14dbe61fe5695bcafb554a82f32afd77b29c83ad08b",
    ),
    (
        "w12_irstd_metadata",
        "outputs/stage2_protocol/RC4_STAGE2_W12_IRSTD_1K_OFFICIAL_SPLIT_METADATA_20260717.json",
        "9e8592780dffcf4dc1f3877820dd39d026975c47c847ec7558262a9d2a0af430",
    ),
    (
        "w12_irstd_plan",
        "outputs/stage2_protocol/RC4_STAGE2_W12_IRSTD_1K_PREOPEN_PLAN_20260717.json",
        "fae67183e4b6aa7f5692b0491329934e81e327fb1123142f7b3fc8a3cf888ffd",
    ),
)

GOVERNANCE = (
    (
        "outputs/gate_evidence/G1_DEVELOPMENT_GATE_RESULT_RC4_20260716.json",
        EXPECTED_G1_SHA,
    ),
    (
        "outputs/stage2_protocol/RC4_STAGE2_B3_MODEL_DATA_AND_DEPENDENCY_RESOLUTION_AUTHORIZATION_20260717.json",
        EXPECTED_B3_SHA,
    ),
)

FROZEN_PREREQUISITES = (
    (
        "materialization_index_53",
        "outputs/stage2_manifests/rc4_k2_c14q28_20260716/materialization_index.json",
        EXPECTED_MATERIALIZATION_INDEX_SHA,
    ),
    (
        "materialization_independent_audit",
        "outputs/stage2_protocol/RC4_STAGE2_K2_C14Q28_INDEPENDENT_AUDIT_20260716.json",
        EXPECTED_MATERIALIZATION_AUDIT_SHA,
    ),
    (
        "detector_run_contract_index_27",
        "outputs/stage2_detector_contracts/rc4_k2_c14q28_w01_20260716/run_contract_index.json",
        EXPECTED_RUN_CONTRACT_INDEX_SHA,
    ),
    (
        "b1_contract_spine_integration_pass",
        "outputs/stage2_protocol/RC4_STAGE2_B1_CONTRACT_SPINE_INTEGRATION_PASS_20260716.json",
        EXPECTED_B1_INTEGRATION_SHA,
    ),
    (
        "stage2_seed_manifest_63",
        "outputs/stage2_protocol/RC4_STAGE2_SEED_DERIVATION_MANIFEST_V1_20260716.json",
        EXPECTED_SEED_MANIFEST_SHA,
    ),
    (
        "b2_prior_full_merge_hold",
        "outputs/stage2_protocol/RC4_STAGE2_B2_SAFE_CORE_AND_CAUSAL_PERFORMANCE_HOLD_20260717.json",
        EXPECTED_B2_HOLD_SHA,
    ),
    (
        "b2_result_free_implementation_amendment",
        "outputs/stage2_protocol/RC4_STAGE2_B2_EPISODE_REFERENCE_AUTHORIZATION_AMENDMENT_20260717.json",
        EXPECTED_B2_AMENDMENT_SHA,
    ),
    *W12_PREOPEN_ARTIFACTS,
)

SOURCE_FILES = (
    "configs/aaai27_stage2_crossfit_v2.json",
    "data_ext/stage2_label_attachment.py",
    "data_ext/stage2_role_contract.py",
    "data_ext/stage2_score_manifest.py",
    "data_ext/stage2_threshold_decision.py",
    "data_ext/mask_alignment.py",
    "rc_irstd/data/dataset.py",
    "rc_irstd/data/mask_alignment.py",
    "evaluation/export_stage2_labels.py",
    "evaluation/export_stage2_development_scores.py",
    "evaluation/stage2_crossfit_replay.py",
    "evaluation/stage2_paired_bootstrap.py",
    "evaluation/stage2_source_threshold_pool.py",
    "evaluation/stage2_threshold_family.py",
    "evaluation/stage2_threshold_sweep.py",
    "losses/calibrator_risk.py",
    "losses/hard_target_loss.py",
    "losses/local_peak_cvar.py",
    "losses/sls.py",
    "losses/smooth_worst_domain.py",
    "losses/target_background_margin.py",
    "model/direct_no_reject_pixel_calibrator.py",
    "model/MSHNet.py",
    "model/monotone_pixel_calibrator.py",
    "rc/build_stage2_crossfit_episodes.py",
    "rc/build_stage2_source_reference.py",
    "rc/stage2_crossfit_dataset.py",
    "rc/stage2_crossfit_schema.py",
    "rc/stage2_deployment.py",
    "rc/train_stage2_crossfit_calibrator.py",
    "scripts/freeze_stage2_confirmatory_plan.py",
    "scripts/train_multisource_tail.py",
    "scripts/materialize_stage2_confirmatory_identity.py",
    "scripts/materialize_stage2_detector_run_contracts.py",
    "scripts/orchestrate_stage2_crossfit.py",
    "outputs/audit_tools/audit_stage2_i0.py",
    "outputs/audit_tools/audit_stage2_development_completion.py",
    "scripts/audit_stage2_model_design_freeze.py",
    "tests/stage2_crossfit_fixtures.py",
    "tests/test_stage2_calibrator_v6.py",
    "tests/test_stage2_confirmatory_seal.py",
    "tests/test_stage2_crossfit_dataset_replay.py",
    "tests/test_stage2_crossfit_episode_v5.py",
    "tests/test_stage2_deployment_v2.py",
    "tests/test_stage2_detector_run_contract.py",
    "tests/test_stage2_label_curve_contract.py",
    "tests/test_stage2_model_design_auditor.py",
    "tests/test_stage2_orchestrator_contract.py",
    "tests/test_stage2_paired_bootstrap.py",
    "tests/test_stage2_score_manifest_v4.py",
    "tests/test_stage2_source_reference_contract.py",
    "tests/test_stage2_source_threshold_pool.py",
    "tests/test_stage2_scoped_release.py",
    "tests/test_stage2_threshold_family.py",
    "tests/test_deployment_and_calibration_units.py",
    "tests/test_source_reference_streaming.py",
    "tests/test_risk_losses.py",
    "scripts/freeze_stage2_scoped_release.py",
    "RC-IRSTD_AAAI27_完整模型设计冻结与实验执行计划_20260717.md",
    "RC-IRSTD_AAAI27_预注册实验矩阵与结果表模板_20260717.md",
)

WORK_PACKAGE_REQUIRED_NODES: Mapping[str, tuple[str, ...]] = {
    "W06": (
        "tests/test_stage2_crossfit_episode_v5.py::test_collection_completeness_exact_frozen_matrix",
        "tests/test_stage2_crossfit_episode_v5.py::test_context_bundle_public_verifier_atomic_and_shared_bridge",
        "tests/test_stage2_crossfit_episode_v5.py::test_statistics_config_public_verifier_is_strict",
    ),
    "W07": (
        "tests/test_stage2_crossfit_dataset_replay.py::test_standardizer_exact_floor_and_training_scope",
        "tests/test_stage2_crossfit_dataset_replay.py::test_public_verified_decision_replay_recomputes_four_identity_hashes",
        "tests/test_stage2_crossfit_dataset_replay.py::test_million_event_collate_is_ragged_zero_copy_and_brackets_are_small",
    ),
    "W08": (
        "tests/test_stage2_calibrator_v6.py::test_frozen_config_builds_exact_models",
        "tests/test_stage2_calibrator_v6.py::test_real_lane_a_collate_routes_all_three_objectives",
        "tests/test_stage2_calibrator_v6.py::test_compact_curve_loss_and_gradient_equal_full_piecewise_curve",
        "tests/test_stage2_calibrator_v6.py::test_interrupted_resume_matches_uninterrupted_next_step",
    ),
    "W09": (
        "tests/test_stage2_threshold_family.py::test_source_safe_rank_is_exact_pd_then_fp_then_larger_threshold",
        "tests/test_stage2_threshold_family.py::test_atomic_t0_t8_bundle_is_publicly_verified_and_unforgeable",
        "tests/test_stage2_threshold_family.py::test_t9_is_postlabel_only_and_not_a_prelabel_decision",
        "tests/test_stage2_label_curve_contract.py::test_outer_target_role_opens_only_after_complete_bound_t0_t8_seal",
    ),
    "W10": (
        "tests/test_stage2_deployment_v2.py::test_partition_is_exact_first_14_and_complete_unmodified_suffix",
        "tests/test_stage2_deployment_v2.py::test_sealed_decision_binds_complete_suffix_curve_and_external_sha",
        "tests/test_stage2_deployment_v2.py::test_protocol_requires_both_w12_provenance_hashes",
    ),
    "W11": (
        "tests/test_stage2_paired_bootstrap.py::test_three_stateless_v2_preimages_match_manual_sha256",
        "tests/test_stage2_paired_bootstrap.py::test_10000_manifest_has_exact_three_level_method_agnostic_indices",
        "tests/test_stage2_paired_bootstrap.py::test_public_evaluator_rehashes_files_replays_all_draws_and_reports_estimand",
    ),
    "W12": (
        "tests/test_stage2_confirmatory_seal.py::test_pre_open_plan_exact_schema_and_result_free_semantics",
        "tests/test_stage2_confirmatory_seal.py::test_pre_open_cli_never_stats_or_opens_referenced_split",
        "tests/test_stage2_confirmatory_seal.py::test_post_go_requires_exact_go_before_split_open",
        "tests/test_stage2_confirmatory_seal.py::test_synthetic_post_go_identity_is_exact_first14_complete_suffix_and_label_free",
    ),
    "W13": (
        "tests/test_stage2_orchestrator_contract.py::test_dry_run_is_hash_complete_and_data_free",
        "tests/test_stage2_orchestrator_contract.py::test_real_mode_requires_verified_s2_i0",
        "tests/test_stage2_orchestrator_contract.py::test_gpu_mapping_and_no_official_phase_are_frozen",
    ),
}

LEGACY_AND_TRANSITIVE_TEST_FILES = (
    "tests/test_stage2_detector_run_contract.py",
    "tests/test_stage2_label_curve_contract.py",
    "tests/test_stage2_model_design_auditor.py",
    "tests/test_stage2_score_manifest_v4.py",
    "tests/test_stage2_source_reference_contract.py",
    "tests/test_stage2_source_threshold_pool.py",
    "tests/test_no_reject_v5_integration.py",
    "tests/test_two_stage_no_reject.py",
    "tests/test_calibrator_risk_no_reject.py",
    "tests/test_monotone_pixel_calibrator.py",
    "tests/test_deployment_and_calibration_units.py",
    "tests/test_source_reference_streaming.py",
    "tests/test_risk_losses.py",
    "tests/test_stage2_scoped_release.py",
)

PYTEST_FILES = tuple(
    dict.fromkeys(
        [
            node.split("::", 1)[0]
            for nodes in WORK_PACKAGE_REQUIRED_NODES.values()
            for node in nodes
        ]
        + list(LEGACY_AND_TRANSITIVE_TEST_FILES)
    )
)

OFFICIAL_SENTINEL_REQUIRED_NODES = (
    "tests/test_stage2_confirmatory_seal.py::test_pre_open_cli_never_stats_or_opens_referenced_split",
    "tests/test_stage2_score_manifest_v4.py::test_exporter_is_split_independent_and_sentinel_safe",
    "tests/test_stage2_label_curve_contract.py::test_label_v2_official_test_guard_is_exact_boolean",
    "tests/test_stage2_detector_run_contract.py::test_stage2_trainer_path_never_calls_legacy_test_split_or_dataset_identity",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(before) != identity(path_after):
            raise RuntimeError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, expected_sha256: str | None = None) -> Mapping[str, Any]:
    pairs: list[tuple[str, Any]] = []

    def hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        pairs.clear()
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    data = path.read_bytes()
    if expected_sha256 is not None and _sha256_bytes(data) != expected_sha256:
        raise ValueError(f"external SHA-256 mismatch while loading {path}")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=hook)
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _check(condition: bool, name: str, detail: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}


def _sidecar_matches(path: Path, digest: str) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        return False
    expected = f"{digest}  {path.name}\n".encode("ascii")
    return sidecar.read_bytes() == expected


def _validate_governance() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    checks: list[dict[str, Any]] = []
    bindings: list[dict[str, str]] = []
    for relative, expected in GOVERNANCE:
        path = REPOSITORY_ROOT / relative
        exists = path.is_file() and not path.is_symlink()
        checks.append(_check(exists, f"governance_exists::{relative}", exists))
        if not exists:
            continue
        observed = _sha256_file(path)
        checks.append(_check(observed == expected, f"governance_sha::{relative}", observed))
        bindings.append({"path": relative, "sha256": observed})
    if len(bindings) != len(GOVERNANCE):
        return checks, bindings
    g1 = _load_json(REPOSITORY_ROOT / GOVERNANCE[0][0], GOVERNANCE[0][1])
    checks.append(
        _check(
            g1.get("status") == "PASS",
            "g1_development_gate_pass",
            g1.get("status"),
        )
    )
    checks.append(
        _check(g1.get("criteria", {}).get("official_test_absent") is True, "g1_official_test_sealed", g1.get("criteria", {}).get("official_test_absent"))
    )
    b3 = _load_json(REPOSITORY_ROOT / GOVERNANCE[1][0], GOVERNANCE[1][1])
    checks.extend(
        [
            _check(
                b3.get("artifact_status")
                == "RESULT_FREE_FROZEN_IMPLEMENTATION_AUTHORIZATION",
                "b3_result_free_implementation_authorization",
                b3.get("artifact_status"),
            ),
            _check(
                all(
                    b3.get(field) is False
                    for field in (
                        "contains_observed_results",
                        "execution_authorized",
                        "training_authorized",
                        "real_data_authorized",
                        "gpu_authorized",
                        "official_test_authorized",
                    )
                ),
                "b3_no_execution_or_result_authority",
                {
                    field: b3.get(field)
                    for field in (
                        "contains_observed_results",
                        "execution_authorized",
                        "training_authorized",
                        "real_data_authorized",
                        "gpu_authorized",
                        "official_test_authorized",
                    )
                },
            ),
        ]
    )
    return checks, bindings


def _validate_frozen_prerequisites() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    checks: list[dict[str, Any]] = []
    bindings: list[dict[str, str]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for name, relative, expected in FROZEN_PREREQUISITES:
        path = REPOSITORY_ROOT / relative
        exists = path.is_file() and not path.is_symlink()
        checks.append(_check(exists, f"prerequisite_exists::{name}", exists))
        if not exists:
            continue
        observed = _sha256_file(path)
        checks.append(_check(observed == expected, f"prerequisite_sha::{name}", observed))
        if name.startswith("w12_"):
            checks.append(
                _check(
                    _sidecar_matches(path, observed),
                    f"prerequisite_sidecar::{name}",
                    path.with_name(path.name + ".sha256")
                    .relative_to(REPOSITORY_ROOT)
                    .as_posix(),
                )
            )
        bindings.append({"name": name, "path": relative, "sha256": observed})
        if observed == expected:
            payloads[name] = _load_json(path, expected)

    materialization = payloads.get("materialization_index_53", {})
    checks.append(
        _check(
            materialization.get("artifact_count_excluding_this_index") == 52
            and len(materialization.get("artifacts_excluding_this_index", {})) == 52,
            "materialization_index_exact_53_json_files",
            materialization.get("artifact_count_excluding_this_index"),
        )
    )
    materialization_audit = payloads.get("materialization_independent_audit", {})
    inventory = materialization_audit.get("artifact_inventory", {})
    aggregate = materialization_audit.get("aggregate_counts", {})
    checks.append(
        _check(
            materialization_audit.get("decision")
            == "PASS_MANIFEST_INTEGRITY_STAGE2_EXECUTION_HOLD"
            and inventory.get("expected_equals_actual") is True
            and inventory.get("json_file_count") == 53
            and all(
                aggregate.get(field) == 0
                for field in (
                    "canonical_id_cross_partition_overlap_count",
                    "original_image_sha256_cross_partition_overlap_count",
                    "near_duplicate_cluster_or_sentinel_cross_partition_overlap_count",
                    "exclusion_group_cross_partition_overlap_count",
                )
            ),
            "materialization_independent_audit_pass",
            {
                "decision": materialization_audit.get("decision"),
                "json_file_count": inventory.get("json_file_count"),
            },
        )
    )
    run_index = payloads.get("detector_run_contract_index_27", {})
    checks.append(
        _check(
            run_index.get("artifact_status") == "DEVELOPMENT_ONLY_RESULT_FREE"
            and len(run_index.get("contracts", [])) == 27,
            "detector_run_contract_index_exact_27",
            len(run_index.get("contracts", [])),
        )
    )
    b1 = payloads.get("b1_contract_spine_integration_pass", {})
    w01 = b1.get("w01_materialization", {})
    checks.append(
        _check(
            b1.get("status") == "PASS"
            and b1.get("contains_observed_results") is False
            and w01.get("run_contract_count") == 27
            and w01.get("oof_run_count") == 18
            and w01.get("full_fit_run_count") == 9
            and b1.get("decision", {}).get("b1_merge_gate") == "PASS",
            "b1_contract_spine_integration_exact_pass",
            {
                "status": b1.get("status"),
                "run_contract_count": w01.get("run_contract_count"),
                "b1_merge_gate": b1.get("decision", {}).get("b1_merge_gate"),
            },
        )
    )
    seed = payloads.get("stage2_seed_manifest_63", {})
    seed_rows = seed.get("derived_seed_table", [])
    seed_count = sum(
        len(row.get("derived_seeds_by_role", {}))
        for row in seed_rows
        if isinstance(row, Mapping)
    )
    checks.append(
        _check(
            seed.get("artifact_status") == "VALUES_RESOLVED_EXECUTION_HOLD"
            and seed.get("execution_authorized") is False
            and seed.get("contains_observed_results") is False
            and len(seed_rows) == 9
            and seed_count == 63,
            "seed_manifest_exact_63_result_free_hold",
            {"row_count": len(seed_rows), "derived_seed_count": seed_count},
        )
    )
    b2_hold = payloads.get("b2_prior_full_merge_hold", {})
    checks.append(
        _check(
            b2_hold.get("status")
            == "PASS_SAFE_CORE_WITH_EXPLICIT_CAUSAL_AND_SCALABILITY_HOLD"
            and b2_hold.get("contains_observed_results") is False
            and b2_hold.get("decision", {}).get("b2_safe_core") == "PASS"
            and b2_hold.get("decision", {}).get("b2_full_merge_gate") == "HOLD",
            "b2_prior_hold_exact_and_pending_explicit_resolution",
            {
                "status": b2_hold.get("status"),
                "b2_full_merge_gate": b2_hold.get("decision", {}).get(
                    "b2_full_merge_gate"
                ),
            },
        )
    )
    b2_amendment = payloads.get("b2_result_free_implementation_amendment", {})
    checks.append(
        _check(
            b2_amendment.get("artifact_status")
            == "RESULT_FREE_FROZEN_IMPLEMENTATION_AUTHORIZATION"
            and b2_amendment.get("contains_observed_results") is False
            and all(
                b2_amendment.get(field) is False
                for field in (
                    "execution_authorized",
                    "training_authorized",
                    "gpu_authorized",
                    "official_test_authorized",
                )
            ),
            "b2_implementation_amendment_exact_result_free",
            b2_amendment.get("artifact_status"),
        )
    )
    w12_index = payloads.get("w12_preopen_index", {})
    w12_rows = w12_index.get("datasets", [])
    expected_w12_rows = {
        "nuaa-sirst": {
            "metadata_path": W12_PREOPEN_ARTIFACTS[1][1],
            "metadata_sha256": W12_PREOPEN_ARTIFACTS[1][2],
            "plan_path": W12_PREOPEN_ARTIFACTS[2][1],
            "plan_sha256": W12_PREOPEN_ARTIFACTS[2][2],
        },
        "nudt-sirst": {
            "metadata_path": W12_PREOPEN_ARTIFACTS[3][1],
            "metadata_sha256": W12_PREOPEN_ARTIFACTS[3][2],
            "plan_path": W12_PREOPEN_ARTIFACTS[4][1],
            "plan_sha256": W12_PREOPEN_ARTIFACTS[4][2],
        },
        "irstd-1k": {
            "metadata_path": W12_PREOPEN_ARTIFACTS[5][1],
            "metadata_sha256": W12_PREOPEN_ARTIFACTS[5][2],
            "plan_path": W12_PREOPEN_ARTIFACTS[6][1],
            "plan_sha256": W12_PREOPEN_ARTIFACTS[6][2],
        },
    }
    observed_w12_rows = {
        row.get("dataset"): {
            key: row.get(key)
            for key in ("metadata_path", "metadata_sha256", "plan_path", "plan_sha256")
        }
        for row in w12_rows
        if isinstance(row, Mapping)
    }
    checks.append(
        _check(
            w12_index.get("artifact_status") == "RESULT_FREE_PREOPEN_PLANS_FROZEN"
            and w12_index.get("contains_observed_results") is False
            and w12_index.get("execution_authorized") is False
            and w12_index.get("target_dataset_count") == 3
            and observed_w12_rows == expected_w12_rows,
            "w12_three_domain_preopen_index_exact_result_free",
            {
                "dataset_count": len(observed_w12_rows),
                "datasets": sorted(observed_w12_rows),
            },
        )
    )
    for name, _, _ in W12_PREOPEN_ARTIFACTS[1:]:
        payload = payloads.get(name, {})
        if name.endswith("metadata"):
            condition = (
                set(payload)
                == {
                    "split_expected_record_count",
                    "split_expected_sha256",
                    "split_repository_relative_path",
                }
                and isinstance(payload.get("split_expected_record_count"), int)
                and payload.get("split_expected_record_count", 0) > 14
                and isinstance(payload.get("split_expected_sha256"), str)
                and len(payload.get("split_expected_sha256", "")) == 64
            )
        else:
            condition = (
                payload.get("schema_version") == "rc-irstd.stage2-confirmatory-plan.v1"
                and payload.get("execution_authorized") is False
                and payload.get("context_rule") == "first_C_in_frozen_split_order"
                and payload.get("context_size") == 14
                and payload.get("query_rule") == "all_remaining_suffix"
                and all(
                    payload.get(field) is False
                    for field in (
                        "official_test_accessed",
                        "official_test_ids_materialized",
                        "official_test_images_opened",
                        "official_test_labels_opened",
                        "official_test_masks_opened",
                    )
                )
            )
        checks.append(
            _check(
                condition,
                f"w12_artifact_result_free::{name}",
                {"field_names": sorted(payload)},
            )
        )
    return checks, bindings


def _validate_config() -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = REPOSITORY_ROOT / AUTHORITATIVE_STAGE2_CONFIG
    config = _load_json(path)
    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    loss = config.get("loss", {})
    selection = config.get("checkpoint_selection", {})
    checks = [
        _check(_sha256_file(path) == EXPECTED_STAGE2_CONFIG_SHA, "config_external_frozen_sha", _sha256_file(path)),
        _check(config.get("contains_observed_results") is False, "config_result_free", False),
        _check(config.get("context_feature_dim") == 93, "config_context_dim", 93),
        _check(config.get("pixel_budget_grid") == [1e-4, 1e-5, 1e-6], "config_budget_grid", config.get("pixel_budget_grid")),
        _check(model.get("hidden_dims") == [32], "config_hidden_dims", model.get("hidden_dims")),
        _check(model.get("dropout") == 0.1, "config_dropout", model.get("dropout")),
        _check(model.get("min_logit") == -10.0 and model.get("max_logit") == 18.0, "config_logit_bounds", [model.get("min_logit"), model.get("max_logit")]),
        _check(model.get("minimum_logit_gap") == 0.001, "config_minimum_gap", model.get("minimum_logit_gap")),
        _check(model.get("reject_head") is False and model.get("missing_episode_fallback") is False, "config_no_reject_no_fallback", False),
        _check(optimizer.get("name") == "AdamW" and optimizer.get("batch_size") == 16 and optimizer.get("amp") is False and optimizer.get("num_workers") == 0, "config_optimizer_core", dict(optimizer)),
        _check(loss == {"oracle_huber_delta": 1.0, "lambda_violation": 4.0, "lambda_utility": 1.0, "lambda_oracle": 0.1, "lambda_smoothness": 0.01, "lambda_coverage": 4.0, "risk_epsilon": 1e-12}, "config_loss_exact", dict(loss)),
        _check(selection.get("rank") == ["macro_source_BSR_max", "macro_source_LogExcess_min", "macro_source_Pd_max", "earlier_epoch_on_exact_tie"] and selection.get("outer_target_accessed") is False, "config_checkpoint_rank", dict(selection)),
    ]
    return checks, {"path": path.relative_to(REPOSITORY_ROOT).as_posix(), "sha256": _sha256_file(path)}


def _validate_final_authority_files() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    checks: list[dict[str, Any]] = []
    bindings: list[dict[str, str]] = []
    for name, relative, expected in FINAL_AUTHORITY_FILES:
        path = REPOSITORY_ROOT / relative
        exists = path.is_file() and not path.is_symlink()
        checks.append(_check(exists, f"final_authority_exists::{name}", exists))
        if not exists:
            continue
        observed = _sha256_file(path)
        checks.append(_check(observed == expected, f"final_authority_sha::{name}", observed))
        bindings.append({"name": name, "path": relative, "sha256": observed})
    checks.append(
        _check(
            AUTHORITATIVE_STAGE2_CONFIG != "configs/aaai27_analysis_plan.json",
            "legacy_analysis_plan_rejected_as_stage2_authority",
            AUTHORITATIVE_STAGE2_CONFIG,
        )
    )
    return checks, bindings


def _validate_models() -> list[dict[str, Any]]:
    import torch
    from model.direct_no_reject_pixel_calibrator import DirectNoRejectPixelCalibrator
    from model.monotone_pixel_calibrator import MonotoneNoRejectPixelRiskCalibrator

    budgets = [1e-4, 1e-5, 1e-6]
    direct = DirectNoRejectPixelCalibrator(93, budgets, hidden_dims=[32], dropout=0.1)
    monotone = MonotoneNoRejectPixelRiskCalibrator(
        93,
        budgets,
        hidden_dims=[32],
        dropout=0.1,
        min_logit=-10.0,
        max_logit=18.0,
        minimum_logit_gap=0.001,
    )
    direct_count = sum(parameter.numel() for parameter in direct.parameters() if parameter.requires_grad)
    monotone_count = sum(parameter.numel() for parameter in monotone.parameters() if parameter.requires_grad)
    generator = torch.Generator(device="cpu").manual_seed(20260717)
    features = torch.randn((257, 93), generator=generator, dtype=torch.float32)
    with torch.no_grad():
        monotone_logits = monotone(features).grid_logits
        direct.threshold_head.weight.zero_()
        direct.threshold_head.bias.copy_(torch.tensor([2.0, -2.0, 0.0]))
        direct_logits = direct(features).grid_logits
    monotone_ok = bool(torch.all(monotone_logits[:, 1:] > monotone_logits[:, :-1]).item())
    direct_unsorted = bool(torch.all(direct_logits[:, 0] > direct_logits[:, 1]).item())
    return [
        _check(direct_count == 3107, "T6_parameter_count", direct_count),
        _check(monotone_count == 3140, "T7_parameter_count", monotone_count),
        _check(monotone_count == 3140, "T8_parameter_count", monotone_count),
        _check(monotone_ok, "T7_T8_structural_monotonicity", monotone_ok),
        _check(direct_unsorted and direct.structural_monotonicity is False, "T6_not_silently_sorted", direct_unsorted),
        _check(not torch.cuda.is_initialized(), "cuda_not_initialized_by_model_audit", torch.cuda.is_initialized()),
    ]


def _validate_sources() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    checks: list[dict[str, Any]] = []
    bindings: list[dict[str, str]] = []
    for relative in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        exists = path.is_file() and not path.is_symlink()
        checks.append(_check(exists, f"source_exists::{relative}", exists))
        if not exists:
            continue
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                parsed = True
            except (SyntaxError, UnicodeError):
                parsed = False
            checks.append(_check(parsed, f"source_ast::{relative}", parsed))
        bindings.append({"path": relative, "sha256": _sha256_file(path)})
    return checks, bindings


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        return len(bytes.fromhex(value)) == 32
    except ValueError:
        return False


def _validate_scoped_release(
    release_dir: str | Path,
    expected_sha256: Mapping[str, str],
    current_sources: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    directory = Path(release_dir).expanduser()
    directory = directory if directory.is_absolute() else REPOSITORY_ROOT / directory
    directory = directory.absolute()
    valid_directory = (
        REPOSITORY_ROOT in directory.parents
        and directory.is_dir()
        and not directory.is_symlink()
    )
    checks.append(
        _check(valid_directory, "scoped_release_directory_is_real_and_in_repository", str(directory))
    )
    binding: dict[str, Any] = {
        "directory": (
            directory.relative_to(REPOSITORY_ROOT).as_posix()
            if REPOSITORY_ROOT in directory.parents
            else str(directory)
        )
    }
    logical_files = {
        "manifest": "SCOPED_RELEASE_MANIFEST.json",
        "archive": "RC-IRSTD_STAGE2_MODEL_DESIGN_SCOPED_RELEASE.tar",
        "environment": "ENVIRONMENT.json",
        "commit": "COMMIT.json",
    }
    if not valid_directory:
        for logical in logical_files:
            checks.append(_check(False, f"scoped_release_file::{logical}", "directory invalid"))
        return checks, binding

    payloads: dict[str, Mapping[str, Any]] = {}
    observed_sha: dict[str, str] = {}
    for logical, name in logical_files.items():
        path = directory / name
        expected = expected_sha256.get(logical)
        expected_valid = _is_sha256(expected)
        checks.append(_check(expected_valid, f"scoped_release_external_sha_format::{logical}", expected))
        exists = path.is_file() and not path.is_symlink()
        checks.append(_check(exists, f"scoped_release_file::{logical}", name))
        if not exists:
            continue
        observed = _sha256_file(path)
        observed_sha[logical] = observed
        checks.append(
            _check(
                expected_valid and observed == expected,
                f"scoped_release_external_sha::{logical}",
                observed,
            )
        )
        checks.append(
            _check(
                _sidecar_matches(path, observed),
                f"scoped_release_external_sidecar::{logical}",
                name + ".sha256",
            )
        )
        if logical in {"manifest", "environment", "commit"} and observed == expected:
            payloads[logical] = _load_json(path, expected)
    binding["files"] = {
        logical: {"path": name, "sha256": observed_sha.get(logical)}
        for logical, name in logical_files.items()
    }

    manifest = payloads.get("manifest", {})
    source_rows = manifest.get("source_members", [])
    source_map = {
        row.get("path"): row.get("sha256")
        for row in source_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("path"), str)
        and _is_sha256(row.get("sha256"))
    }
    unique_sources = (
        isinstance(source_rows, list)
        and len(source_map) == len(source_rows)
        and manifest.get("source_member_count") == len(source_rows)
    )
    checks.append(
        _check(unique_sources, "scoped_release_unique_source_inventory", len(source_map))
    )
    unsafe_paths = [
        path
        for path in source_map
        if path.startswith("datasets/")
        or (
            path.startswith("audits/aaai27/")
            and path != "audits/aaai27/near_duplicates_effective_splits_v2.json"
        )
        or (
            path.startswith("splits/aaai27_v2/")
            and path != "splits/aaai27_v2/manifest.json"
        )
    ]
    checks.append(
        _check(not unsafe_paths, "scoped_release_no_official_id_or_dataset_members", unsafe_paths)
    )
    missing_or_changed_sources = [
        row.get("path")
        for row in current_sources
        if source_map.get(row.get("path")) != row.get("sha256")
    ]
    checks.append(
        _check(
            not missing_or_changed_sources,
            "scoped_release_binds_current_s2_source_surface",
            missing_or_changed_sources,
        )
    )
    closure = manifest.get("index_and_sidecar_closure_verification", {})
    archive_verification = manifest.get("archive_verification", {})
    config_authority = manifest.get("stage2_config_authority", {})
    expected_authority = {
        relative: expected for _, relative, expected in FINAL_AUTHORITY_FILES
    }
    observed_authority = {
        row.get("path"): row.get("sha256")
        for row in manifest.get("final_authority_bindings", [])
        if isinstance(row, Mapping)
    }
    manifest_without_self = dict(manifest)
    reported_content_sha = manifest_without_self.pop("manifest_content_sha256", None)
    checks.extend(
        [
            _check(
                manifest.get("artifact_status") == "SCOPED_SOURCE_RELEASE_COMPLETE"
                and manifest.get("contains_observed_stage2_results") is False
                and manifest.get("execution_authorized") is False
                and manifest.get("frozen_development_metadata_hashed") is True
                and manifest.get("official_test_id_artifacts_in_allowlist") is False
                and manifest.get("worktree_clean_claimed") is False
                and manifest.get("git_diff_and_status_scope")
                == "verified_source_member_allowlist_only"
                and manifest.get("release_scope_complete") is True,
                "scoped_release_manifest_contract",
                manifest.get("artifact_status"),
            ),
            _check(
                reported_content_sha == _sha256_bytes(_canonical_bytes(manifest_without_self)),
                "scoped_release_manifest_internal_content_sha",
                reported_content_sha,
            ),
            _check(
                manifest.get("archive", {}).get("sha256") == expected_sha256.get("archive")
                and manifest.get("external_environment_lock", {}).get("sha256")
                == expected_sha256.get("environment")
                and manifest.get("b4_amendment", {}).get("sha256")
                == EXPECTED_B4_AMENDMENT_SHA,
                "scoped_release_manifest_external_bindings",
                {
                    "archive": manifest.get("archive", {}).get("sha256"),
                    "environment": manifest.get("external_environment_lock", {}).get(
                        "sha256"
                    ),
                },
            ),
            _check(
                observed_authority == expected_authority
                and config_authority.get("authoritative_path")
                == AUTHORITATIVE_STAGE2_CONFIG
                and config_authority.get("authoritative_sha256")
                == EXPECTED_STAGE2_CONFIG_SHA
                and config_authority.get("legacy_paths_must_be_rejected_by_stage2_launch")
                is True
                and "configs/aaai27_analysis_plan.json"
                in config_authority.get("legacy_non_authoritative_paths", []),
                "scoped_release_final_authority_and_legacy_rejection_exact",
                dict(config_authority),
            ),
            _check(
                closure.get("materialization_leaf_count") == 52
                and closure.get("run_contract_count") == 27
                and closure.get("selection_contract_count") == 54
                and closure.get("selection_id_list_count") == 54
                and closure.get("w12_dataset_count") == 3,
                "scoped_release_index_closure_exact",
                dict(closure),
            ),
            _check(
                archive_verification.get("fresh_extraction_verified") is True
                and archive_verification.get("verified_member_count")
                == len(source_rows) + len(manifest.get("generated_members", [])),
                "scoped_release_fresh_extraction_verified",
                dict(archive_verification),
            ),
        ]
    )
    environment = payloads.get("environment", {})
    checks.append(
        _check(
            environment.get("schema_version")
            == "rc-irstd.stage2-scoped-release-environment.v1"
            and environment.get("execution_authorized") is False,
            "scoped_release_environment_contract",
            environment.get("schema_version"),
        )
    )
    commit = payloads.get("commit", {})
    checks.append(
        _check(
            commit.get("artifact_status") == "COMMITTED_COMPLETE"
            and commit.get("publication_complete") is True
            and commit.get("execution_authorized") is False
            and commit.get("manifest", {}).get("sha256")
            == expected_sha256.get("manifest")
            and commit.get("archive", {}).get("sha256")
            == expected_sha256.get("archive")
            and commit.get("environment_lock", {}).get("sha256")
            == expected_sha256.get("environment")
            and commit.get("b4_amendment_sha256") == EXPECTED_B4_AMENDMENT_SHA,
            "scoped_release_commit_binds_all_external_artifacts",
            commit.get("artifact_status"),
        )
    )
    return checks, binding


def _run_command(command: Sequence[str], *, timeout: int) -> tuple[int, bytes]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_ADDOPTS": "-p no:cacheprovider",
        }
    )
    completed = subprocess.run(
        list(command),
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return int(completed.returncode), bytes(completed.stdout)


def _parse_collected_nodeids(output: bytes) -> frozenset[str]:
    nodeids: set[str] = set()
    for raw_line in output.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if "::" in line and line.startswith("tests/") and " " not in line:
            nodeids.add(line)
    return frozenset(nodeids)


def _node_is_collected(required: str, collected: frozenset[str]) -> bool:
    return required in collected or any(node.startswith(required + "[") for node in collected)


def _coverage_checks(collected: frozenset[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for package, required_nodes in WORK_PACKAGE_REQUIRED_NODES.items():
        missing = [node for node in required_nodes if not _node_is_collected(node, collected)]
        checks.append(
            _check(
                not missing,
                f"fixed_suite_required_nodes::{package}",
                {"required_count": len(required_nodes), "missing": missing},
            )
        )
    missing_sentinels = [
        node
        for node in OFFICIAL_SENTINEL_REQUIRED_NODES
        if not _node_is_collected(node, collected)
    ]
    checks.append(
        _check(
            not missing_sentinels,
            "fixed_suite_official_access_sentinels_collected",
            {
                "required_count": len(OFFICIAL_SENTINEL_REQUIRED_NODES),
                "missing": missing_sentinels,
            },
        )
    )
    return checks


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    return {
        "tests": len(cases),
        "passed": len(cases) - failures - errors - skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _run_tests() -> tuple[list[dict[str, Any]], bytes, dict[str, Any]]:
    collect_command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        *PYTEST_FILES,
    ]
    collect_code, collect_output = _run_command(collect_command, timeout=600)
    collected = _parse_collected_nodeids(collect_output)
    checks = [
        _check(collect_code == 0, "frozen_stage2_pytest_collection_exit", collect_code)
    ]
    checks.extend(_coverage_checks(collected))
    coverage_pass = all(item["status"] == "PASS" for item in checks)
    allowlist_contract = {
        "work_package_required_nodes": WORK_PACKAGE_REQUIRED_NODES,
        "official_sentinel_required_nodes": OFFICIAL_SENTINEL_REQUIRED_NODES,
        "legacy_and_transitive_test_files": LEGACY_AND_TRANSITIVE_TEST_FILES,
    }
    metadata: dict[str, Any] = {
        "suite_scope": "fixed repository synthetic/fault-injection allowlist",
        "allowlist_contract_sha256": _sha256_bytes(_canonical_bytes(allowlist_contract)),
        "test_files": list(PYTEST_FILES),
        "collection_exit_code": collect_code,
        "collected_test_count": len(collected),
        "collection_stdout_sha256": _sha256_bytes(collect_output),
        "auditor_received_no_dataset_arguments": True,
        "auditor_received_no_checkpoint_arguments": True,
        "auditor_received_no_official_test_arguments": True,
        "cuda_visible_devices_in_subprocess": "",
        "system_level_file_access_instrumented": False,
    }
    if not coverage_pass:
        metadata.update(
            {
                "execution_skipped": True,
                "execution_skip_reason": "collection_or_required_node_coverage_failed",
                "exit_code": None,
                "junit_counts": None,
            }
        )
        return checks, b"== collection ==\n" + collect_output, metadata

    with tempfile.TemporaryDirectory(prefix="rc-irstd-s2-i0-junit-") as raw_tmp:
        junit = Path(raw_tmp) / "pytest.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "xfail_strict=true",
            f"--junitxml={junit}",
            *PYTEST_FILES,
        ]
        code, output = _run_command(command, timeout=1800)
        counts = (
            _junit_counts(junit)
            if junit.is_file()
            else {"tests": 0, "passed": 0, "failures": 0, "errors": 1, "skipped": 0}
        )
    checks.extend(
        [
            _check(code == 0, "frozen_stage2_pytest_exit", code),
            _check(counts["passed"] >= 100, "frozen_stage2_pytest_count_floor", counts),
            _check(counts["failures"] == 0, "frozen_stage2_pytest_zero_failures", counts),
            _check(counts["errors"] == 0, "frozen_stage2_pytest_zero_errors", counts),
            _check(counts["skipped"] == 0, "frozen_stage2_pytest_zero_skips", counts),
        ]
    )
    metadata.update(
        {
            "execution_skipped": False,
            "command": [
                item if not item.startswith("--junitxml=") else "--junitxml=<temporary>"
                for item in command
            ],
            "exit_code": code,
            "junit_counts": counts,
            "stdout_sha256": _sha256_bytes(output),
        }
    )
    combined = b"== collection ==\n" + collect_output + b"\n== execution ==\n" + output
    return checks, combined, metadata


def _git_diff_check() -> dict[str, Any]:
    code, output = _run_command(["git", "diff", "--check", "--", "."], timeout=120)
    return _check(
        code == 0,
        "repository_scoped_git_diff_whitespace_check_not_clean_claim",
        output.decode("utf-8", errors="replace"),
    )


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required for atomic publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    target_parent = os.open(
        target.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        result = renameat2(
            -100,
            os.fsencode(source),
            target_parent,
            os.fsencode(target.name),
            1,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), target)
            raise OSError(error, os.strerror(error), target)
    finally:
        os.close(target_parent)


def _sidecar(path: Path, digest: str) -> None:
    _write_exclusive(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode("ascii"))


def audit(
    output_dir: str | Path,
    *,
    scoped_release_dir: str | Path,
    scoped_release_manifest_sha256: str,
    scoped_release_archive_sha256: str,
    scoped_release_environment_sha256: str,
    scoped_release_commit_sha256: str,
) -> tuple[Path, str, str, str]:
    output = Path(output_dir).expanduser()
    output = output if output.is_absolute() else REPOSITORY_ROOT / output
    output = output.absolute()
    if REPOSITORY_ROOT not in output.parents or output == REPOSITORY_ROOT:
        raise ValueError("output directory must be below repository root")
    if os.path.lexists(output) or not output.parent.is_dir() or output.parent.is_symlink():
        raise FileExistsError("output must be a new directory below a real parent")

    checks, governance = _validate_governance()
    prerequisite_checks, prerequisite_bindings = _validate_frozen_prerequisites()
    checks.extend(prerequisite_checks)
    authority_checks, authority_bindings = _validate_final_authority_files()
    checks.extend(authority_checks)
    config_checks, config_binding = _validate_config()
    checks.extend(config_checks)
    checks.extend(_validate_models())
    source_checks, sources = _validate_sources()
    checks.extend(source_checks)
    release_checks, release_binding = _validate_scoped_release(
        scoped_release_dir,
        {
            "manifest": scoped_release_manifest_sha256,
            "archive": scoped_release_archive_sha256,
            "environment": scoped_release_environment_sha256,
            "commit": scoped_release_commit_sha256,
        },
        sources,
    )
    checks.extend(release_checks)
    test_checks, test_output, test_metadata = _run_tests()
    checks.extend(test_checks)
    checks.append(_git_diff_check())
    b2_full_merge_resolved = all(item["status"] == "PASS" for item in checks)
    checks.append(
        _check(
            b2_full_merge_resolved,
            "b2_full_merge_explicitly_resolved_by_complete_s2_synthetic_audit",
            {
                "prior_b2_hold_sha256": EXPECTED_B2_HOLD_SHA,
                "resolution_basis": "all S2_I0 prerequisites, B4 bindings, W06-W13 required nodes, zero-skip suite and diff check",
            },
        )
    )
    failed = [item["name"] for item in checks if item["status"] != "PASS"]
    status = "PASS" if not failed else "HOLD"
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "artifact_type": "rc_irstd_stage2_complete_model_design_freeze_audit",
        "artifact_status": status,
        "gate_id": "S2_I0",
        "result_free": True,
        "result_free_scope": (
            "No new Stage-2 observed performance results are produced or consumed. "
            "The prior frozen observed Stage-1 G1 decision is bound only as a "
            "prerequisite by SHA-256."
        ),
        "contains_new_stage2_observed_results": False,
        "new_stage2_observed_results": None,
        "prior_frozen_stage1_observed_gate_bound": True,
        "real_data_execution_authorized": False,
        "gpu_execution_authorized": False,
        "official_test_execution_authorized": False,
        "auditor_received_no_dataset_arguments": True,
        "auditor_received_no_checkpoint_arguments": True,
        "auditor_received_no_official_test_arguments": True,
        "system_level_file_access_instrumented": False,
        "shared_worktree_clean_assessed": False,
        "git_tag_created_or_required": False,
        "execution_authorized": False,
        "model_identity": "RC-IRSTD_TwoStage_NoReject_T8",
        "stage1_backbone": "MSHNet",
        "stage2_primary_method": "T8_risk_aligned_monotone_no_reject_calibrator",
        "config_binding": config_binding,
        "final_authority_bindings": authority_bindings,
        "stage2_config_override_supported": False,
        "governance_bindings": governance,
        "frozen_prerequisite_bindings": prerequisite_bindings,
        "scoped_release_binding": release_binding,
        "source_bindings": sources,
        "synthetic_test_evidence": test_metadata,
        "b2_full_merge_resolution": {
            "prior_status": "HOLD",
            "prior_artifact_sha256": EXPECTED_B2_HOLD_SHA,
            "resolved_status": "PASS" if b2_full_merge_resolved else "HOLD",
            "resolved_only_by_this_complete_audit": True,
        },
        "checks": checks,
        "failed_checks": failed,
        "next_action": (
            "STOP_MODEL_DESIGN_AND_AWAIT_SEPARATE_EXPERIMENT_LAUNCH_AUTHORIZATION"
            if status == "PASS"
            else "REPAIR_FAILED_S2_I0_CHECKS_WITHOUT_OPENING_REAL_OR_OFFICIAL_DATA"
        ),
        "claim_boundary": (
            "S2_I0 PASS means the complete model is implemented and ready for the "
            "preregistered development experiment; it is not empirical or AAAI success."
        ),
    }
    report["report_content_sha256_algorithm"] = "sha256-canonical-json-without-self-field-v1"
    report["report_content_sha256"] = _sha256_bytes(_canonical_bytes(report))

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        log_path = staging / "pytest.log"
        _write_exclusive(log_path, test_output)
        log_sha = _sha256_file(log_path)
        _sidecar(log_path, log_sha)
        report_path = staging / "S2_I0_REPORT.json"
        report_bytes = _pretty_bytes(report)
        _write_exclusive(report_path, report_bytes)
        report_sha = _sha256_file(report_path)
        _sidecar(report_path, report_sha)
        commit = {
            "schema_version": COMMIT_SCHEMA,
            "artifact_status": status,
            "publication_complete": True,
            "contains_new_stage2_observed_results": False,
            "real_data_execution_authorized": False,
            "gpu_execution_authorized": False,
            "official_test_execution_authorized": False,
            "system_level_file_access_instrumented": False,
            "execution_authorized": False,
            "report": {"path": report_path.name, "sha256": report_sha},
            "pytest_log": {"path": log_path.name, "sha256": log_sha},
        }
        commit_path = staging / "COMMIT.json"
        commit_bytes = _pretty_bytes(commit)
        _write_exclusive(commit_path, commit_bytes)
        commit_sha = _sha256_file(commit_path)
        _sidecar(commit_path, commit_sha)
        directory = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        _rename_noreplace(staging, output)
        parent = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return output / "S2_I0_REPORT.json", report_sha, commit_sha, status
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scoped-release-dir", required=True)
    parser.add_argument("--scoped-release-manifest-sha256", required=True)
    parser.add_argument("--scoped-release-archive-sha256", required=True)
    parser.add_argument("--scoped-release-environment-sha256", required=True)
    parser.add_argument("--scoped-release-commit-sha256", required=True)
    arguments = parser.parse_args()
    path, report_sha, commit_sha, status = audit(
        arguments.output_dir,
        scoped_release_dir=arguments.scoped_release_dir,
        scoped_release_manifest_sha256=arguments.scoped_release_manifest_sha256,
        scoped_release_archive_sha256=arguments.scoped_release_archive_sha256,
        scoped_release_environment_sha256=arguments.scoped_release_environment_sha256,
        scoped_release_commit_sha256=arguments.scoped_release_commit_sha256,
    )
    print(
        json.dumps(
            {
                "status": status,
                "report": str(path),
                "report_sha256": report_sha,
                "commit_sha256": commit_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
