#!/usr/bin/env python3
"""Fixed 17-category, result-free RC5+ implementation-design auditor.

The auditor accepts no dataset, checkpoint, metric, GPU, or official-test
argument.  It executes a fixed synthetic/fault-injection suite, replays the
sole RC5+ configuration against live code, compiles and hashes the reviewed
source surface, and publishes a commit-last PASS/HOLD bundle.  PASS means only
that the result-free implementation design is frozen; it is never performance,
novelty, launch, or paper-acceptance evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import xml.etree.ElementTree as ET


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_ext.stage2_rc5plus_atomic_full_decision_set import (
    DECISION_SET_SCHEMA as FULL_ATOMIC_SCHEMA,
    publish_stage2_rc5plus_atomic_full_decision_set,
)
from rc.stage2_rc5plus_frozen_config import (
    verify_stage2_rc5plus_frozen_config_file,
)
from rc.stage2_rc5plus_infer_and_seal import infer_and_seal_stage2_rc5plus
from rc.stage2_rc5plus_no_anchor_infer_and_seal import (
    infer_and_seal_stage2_rc5plus_no_anchor,
)


REPORT_SCHEMA = "rc-irstd.stage2-rc5plus-model-design-freeze-audit.v1"
COMMIT_SCHEMA = (
    "rc-irstd.stage2-rc5plus-model-design-freeze-audit-commit.v1"
)
GATE_ID = "S2_I0_RC5PLUS_IMPLEMENTATION"
REPORT_FILENAME = "RC5PLUS_S2_I0_IMPLEMENTATION_AUDIT.json"
PYTEST_LOG_FILENAME = "RC5PLUS_S2_I0_PYTEST.log"
JUNIT_FILENAME = "RC5PLUS_S2_I0_PYTEST.xml"
COMMIT_FILENAME = "RC5PLUS_S2_I0_IMPLEMENTATION_AUDIT.commit.json"
SELF_HASH_ALGORITHM = "sha256-canonical-json-with-self-field-omitted-v1"


TEST_GATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "G01_unique_frozen_config_live_replay",
        ("tests/test_stage2_rc5plus_frozen_config.py",),
    ),
    (
        "G02_exact_rational_budget_function_and_transport_math",
        (
            "tests/test_budget_conditioned_endpoint_calibrator.py",
            "tests/test_budget_conditioned_residual_transport_calibrator.py",
        ),
    ),
    (
        "G03_same_map_same_budget_anchor_v2",
        ("tests/test_stage2_rc5plus_context_anchor_v2.py",),
    ),
    (
        "G04_nine_budget_exact_curve_and_oracle_geometry",
        ("tests/test_stage2_compositional_curve_provider.py",),
    ),
    (
        "G05_four_role_all_start_cyclic_training_geometry",
        (
            "tests/test_stage2_rc5plus_cyclic_anchor_overlay.py",
            "tests/test_stage2_rc5plus_cyclic_training_view.py",
        ),
    ),
    (
        "G06_equal_capacity_method_routing_and_exact_risk_core",
        ("tests/test_stage2_rc5plus_training_core.py",),
    ),
    (
        "G07_trainer_sampler_standardizer_and_feature_masks",
        (
            "tests/test_train_stage2_rc5plus_cyclic.py",
            "tests/test_train_stage2_rc5plus_generation_runner.py",
        ),
    ),
    (
        "G08_primary_only_source_selection_and_variable_q_sanity",
        ("tests/test_stage2_rc5plus_source_validation_view.py",),
    ),
    (
        "G09_checkpoint_v8_weights_only_identity_and_replay",
        (
            "tests/test_stage2_calibrator_checkpoint_v8.py",
            "tests/test_stage2_rc5plus_calibrator_generation_v3.py",
            "tests/test_train_stage2_rc5plus_generation_runner.py::test_interrupted_resume_matches_uninterrupted_generation_v3",
        ),
    ),
    (
        "G10_checkpoint_context_anchor_threshold_sealed_chain",
        ("tests/test_stage2_rc5plus_infer_and_seal.py",),
    ),
    (
        "G11_no_anchor_equal_capacity_ablation_complete_route",
        (
            "tests/test_budget_conditioned_residual_transport_calibrator.py::test_no_target_anchor_ablation_is_capacity_matched_and_has_no_anchor_api",
            "tests/test_stage2_rc5plus_training_core.py::test_t8plus_no_anchor_uses_exact_risk_but_forbids_anchor_input",
            "tests/test_stage2_rc5plus_source_validation_view.py::test_no_anchor_validation_is_bitwise_invariant_to_anchor_values",
            "tests/test_stage2_calibrator_checkpoint_v8.py::test_v8_no_anchor_ablation_round_trip_forbids_anchor_overlay",
            "tests/test_stage2_rc5plus_infer_and_seal.py::test_no_anchor_seal_has_no_anchor_authority_and_replays_byte_exactly",
            "tests/test_stage2_rc5plus_infer_and_seal.py::test_no_anchor_seal_rejects_anchored_checkpoint_and_resigned_tamper",
            "tests/test_stage2_rc5plus_infer_and_seal.py::test_no_anchor_seal_capability_is_unforgeable",
        ),
    ),
    (
        "G12_atomic_learned_t6plus_t8plus_prelabel_authority",
        (
            "tests/test_stage2_rc5plus_infer_and_seal.py::test_atomic_learned_set_binds_all_three_routes_before_labels",
            "tests/test_stage2_rc5plus_infer_and_seal.py::test_atomic_learned_set_rejects_missing_or_reordered_method_routes",
            "tests/test_stage2_rc5plus_infer_and_seal.py::test_atomic_payload_resigned_row_tampering_fails_material_replay",
        ),
    ),
    (
        "G13_production_capability_full_t0_t8_atomic_e2e",
        (
            "tests/test_stage2_rc5_context_producer_e2e.py::test_rc5plus_full_atomic_actual_capabilities_bind_t0_t8_before_labels",
        ),
    ),
    (
        "G14_upstream_tamper_and_commit_last_fault_zero_authority",
        (
            "tests/test_stage2_rc5_context_producer_e2e.py::test_rc5plus_full_atomic_upstream_tamper_never_calls_label_resolver",
            "tests/test_stage2_rc5_context_producer_e2e.py::test_rc5plus_full_atomic_commit_last_fault_leaves_no_authority",
        ),
    ),
    (
        "G15_t9_strictly_separate_postlabel_oracle_diagnostic",
        (
            "tests/test_stage2_rc5_atomic_decision_set.py::test_atomic_t0_t8_round_trip_is_dynamic_q_and_t9_free",
            "tests/test_stage2_rc5_atomic_decision_set.py::test_t9_has_separate_postlabel_schema_and_cannot_be_prelabel_authority",
        ),
    ),
)


SOURCE_FILES = (
    "configs/aaai27_stage2_crossfit_rc5plus_v1.json",
    "RC-IRSTD_AAAI27_RC5plus_预算条件化单调曲线候选设计_20260718.md",
    "model/budget_conditioned_endpoint_calibrator.py",
    "model/budget_conditioned_residual_transport_calibrator.py",
    "data_ext/stage2_rc5_atomic_decision_set.py",
    "data_ext/stage2_rc5plus_atomic_learned_decision_set.py",
    "data_ext/stage2_rc5plus_atomic_full_decision_set.py",
    "rc/stage2_calibrator_checkpoint_v8.py",
    "rc/stage2_rc5plus_context_anchor_v2.py",
    "rc/stage2_rc5plus_calibrator_generation_v3.py",
    "rc/stage2_rc5plus_cyclic_anchor_overlay.py",
    "rc/stage2_rc5plus_cyclic_training_view.py",
    "rc/stage2_rc5plus_frozen_config.py",
    "rc/stage2_rc5plus_infer_and_seal.py",
    "rc/stage2_rc5plus_no_anchor_infer_and_seal.py",
    "rc/stage2_rc5plus_source_validation_view.py",
    "rc/stage2_rc5plus_training_core.py",
    "rc/train_stage2_rc5plus_cyclic.py",
    "scripts/audit_stage2_rc5plus_model_design_freeze.py",
    "tests/test_budget_conditioned_endpoint_calibrator.py",
    "tests/test_budget_conditioned_residual_transport_calibrator.py",
    "tests/test_stage2_calibrator_checkpoint_v8.py",
    "tests/test_stage2_compositional_curve_provider.py",
    "tests/test_stage2_rc5_atomic_decision_set.py",
    "tests/test_stage2_rc5_context_producer_e2e.py",
    "tests/test_stage2_rc5plus_context_anchor_v2.py",
    "tests/test_stage2_rc5plus_calibrator_generation_v3.py",
    "tests/test_stage2_rc5plus_cyclic_anchor_overlay.py",
    "tests/test_stage2_rc5plus_cyclic_training_view.py",
    "tests/test_stage2_rc5plus_frozen_config.py",
    "tests/test_stage2_rc5plus_infer_and_seal.py",
    "tests/test_stage2_rc5plus_source_validation_view.py",
    "tests/test_stage2_rc5plus_training_core.py",
    "tests/test_train_stage2_rc5plus_cyclic.py",
    "tests/test_train_stage2_rc5plus_generation_runner.py",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _plain(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_bytes(path: Path) -> bytes:
    before = path.stat(follow_symlinks=False)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"source is not a direct regular file: {path}")
    data = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise RuntimeError(f"source changed while reading: {path}")
    return data


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {key: item for key, item in value.items() if key != field}
        )
    )


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "detail": _plain(detail),
    }


def _test_run_selectors() -> tuple[str, ...]:
    selectors = [
        selector
        for _name, gate_selectors in TEST_GATES
        for selector in gate_selectors
    ]
    whole_files = {selector for selector in selectors if "::" not in selector}
    selected: list[str] = []
    for selector in selectors:
        test_file = selector.split("::", 1)[0]
        if "::" in selector and test_file in whole_files:
            continue
        if selector not in selected:
            selected.append(selector)
    return tuple(selected)


def _case_matches(selector: str, case: ET.Element) -> bool:
    test_file, separator, function = selector.partition("::")
    expected_classname = test_file.removesuffix(".py").replace("/", ".")
    if case.attrib.get("classname") != expected_classname:
        return False
    if not separator:
        return True
    name = case.attrib.get("name", "")
    return name == function or name.startswith(function + "[")


def _case_passed(case: ET.Element) -> bool:
    return not any(
        case.find(tag) is not None for tag in ("failure", "error", "skipped")
    )


def _run_pytest(temp_directory: Path) -> tuple[bytes, bytes, int]:
    log_path = temp_directory / PYTEST_LOG_FILENAME
    junit_path = temp_directory / JUNIT_FILENAME
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *(_test_run_selectors()),
        f"--junitxml={junit_path}",
        f"--basetemp={temp_directory / 'pytest'}",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONHASHSEED"] = "0"
    with log_path.open("wb") as stream:
        process = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=900,
        )
        stream.flush()
        os.fsync(stream.fileno())
    return (
        _stable_bytes(log_path),
        _stable_bytes(junit_path),
        process.returncode,
    )


def _test_checks(junit_bytes: bytes) -> tuple[list[dict[str, Any]], int]:
    root = ET.fromstring(junit_bytes)
    cases = list(root.iter("testcase"))
    checks: list[dict[str, Any]] = []
    for name, selectors in TEST_GATES:
        matched = [
            case
            for case in cases
            if any(_case_matches(selector, case) for selector in selectors)
        ]
        passed = bool(matched) and all(_case_passed(case) for case in matched)
        checks.append(
            _check(
                name,
                passed,
                {
                    "fixed_selectors": list(selectors),
                    "matched_testcases": len(matched),
                    "failed_or_skipped_testcases": sum(
                        not _case_passed(case) for case in matched
                    ),
                },
            )
        )
    return checks, len(cases)


def _source_bindings_and_compile() -> tuple[list[dict[str, str]], bool]:
    bindings: list[dict[str, str]] = []
    compiled = True
    for relative in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        data = _stable_bytes(path)
        bindings.append({"path": relative, "sha256": _sha256_bytes(data)})
        if path.suffix == ".py":
            try:
                compile(data, str(path), "exec", dont_inherit=True)
            except (SyntaxError, ValueError, TypeError):
                compiled = False
    return bindings, compiled


def _causal_and_result_free_check(config: Mapping[str, Any]) -> dict[str, Any]:
    main_signature = tuple(inspect.signature(infer_and_seal_stage2_rc5plus).parameters)
    no_anchor_signature = tuple(
        inspect.signature(infer_and_seal_stage2_rc5plus_no_anchor).parameters
    )
    full_signature = tuple(
        inspect.signature(
            publish_stage2_rc5plus_atomic_full_decision_set
        ).parameters
    )
    forbidden = {
        "features",
        "curve",
        "threshold",
        "labels",
        "query",
        "alpha",
    }
    no_anchor_forbidden = forbidden | {"anchor", "anchor_v2"}
    deployment = config["deployment_contract"]
    passed = (
        main_signature == ("checkpoint", "producer_bundle", "anchor_v2")
        and no_anchor_signature == ("checkpoint", "producer_bundle")
        and not forbidden.intersection(main_signature)
        and not no_anchor_forbidden.intersection(no_anchor_signature)
        and "labels" not in full_signature
        and deployment["full_atomic_prelabel_set_schema"]
        == FULL_ATOMIC_SCHEMA
        and deployment["t9_included_prelabel"] is False
        and config["contains_observed_results"] is False
        and config["official_test_accessed"] is False
        and config["execution_authorized"] is False
    )
    return _check(
        "G16_result_free_public_api_and_no_injection_boundary",
        passed,
        {
            "main_infer_parameters": list(main_signature),
            "no_anchor_infer_parameters": list(no_anchor_signature),
            "full_atomic_parameters": list(full_signature),
            "contains_observed_results": False,
            "official_test_accessed": False,
            "execution_authorized": False,
        },
    )


def _success_contract_check(
    config: Mapping[str, Any], compiled: bool
) -> dict[str, Any]:
    performance = config["performance_success_gate"]
    ablation = config["ablation_contract"]
    novelty = config["novelty_success_gate"]
    passed = (
        compiled
        and performance["comparison"] == "T8_PLUS_minus_T4"
        and performance["primary_budget"] == [1, 100_000]
        and performance["macro_domain_delta_BSR_point_min"] == 0.05
        and performance[
            "macro_domain_delta_BSR_paired_95CI_lower_strict_min"
        ]
        == 0.0
        and performance["macro_domain_delta_Pd_point_min"] == -0.02
        and performance["macro_domain_delta_Pd_paired_95CI_lower_min"]
        == -0.02
        and performance["secondary_metric_rescue"] is False
        and performance["second_backbone"] == "DNANet"
        and performance["fourth_independent_domain_required"] is True
        and performance["independent_confirmatory_one_look_required"] is True
        and ablation["required_mechanism_macro_BSR_point_strict_min"]
        == 0.0
        and ablation["required_mechanism_macro_Pd_point_min"] == -0.02
        and novelty["minimum_strict_idea_review_score"] == 4.0
        and novelty["fatal_direct_prior_allowed"] is False
        and novelty["design_success_requires_gate_pass"] is True
    )
    return _check(
        "G17_explicit_performance_mechanism_novelty_success_contract",
        passed,
        {
            "source_surface_compiled": compiled,
            "main_comparison": performance["comparison"],
            "primary_budget": performance["primary_budget"],
            "bsr_point_min": performance["macro_domain_delta_BSR_point_min"],
            "bsr_ci_lower_strict_min": performance[
                "macro_domain_delta_BSR_paired_95CI_lower_strict_min"
            ],
            "pd_point_min": performance["macro_domain_delta_Pd_point_min"],
            "pd_ci_lower_min": performance[
                "macro_domain_delta_Pd_paired_95CI_lower_min"
            ],
            "mechanism_bsr_point_strict_min": ablation[
                "required_mechanism_macro_BSR_point_strict_min"
            ],
            "minimum_strict_novelty_score": novelty[
                "minimum_strict_idea_review_score"
            ],
            "performance_evaluated_by_this_audit": False,
            "novelty_evaluated_by_this_audit": False,
        },
    )


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(
    output_directory: Path,
    report_bytes: bytes,
    log_bytes: bytes,
    junit_bytes: bytes,
    commit_bytes: bytes,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=False)
    staging = Path(
        tempfile.mkdtemp(prefix=".rc5plus-s2-i0-", dir=output_directory)
    )
    names_and_bytes = (
        (REPORT_FILENAME, report_bytes),
        (PYTEST_LOG_FILENAME, log_bytes),
        (JUNIT_FILENAME, junit_bytes),
        (COMMIT_FILENAME, commit_bytes),
    )
    published: list[Path] = []
    try:
        for name, data in names_and_bytes:
            _write_exclusive(staging / name, data)
        _fsync_directory(staging)
        for name, _data in names_and_bytes:
            destination = output_directory / name
            os.link(staging / name, destination, follow_symlinks=False)
            published.append(destination)
            _fsync_directory(output_directory)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(output_directory)
        raise
    finally:
        for name, _data in reversed(names_and_bytes):
            try:
                (staging / name).unlink()
            except FileNotFoundError:
                pass
        staging.rmdir()


def audit(output_directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verified_config = verify_stage2_rc5plus_frozen_config_file()
    config = _plain(verified_config.payload)
    source_bindings, compiled = _source_bindings_and_compile()
    with tempfile.TemporaryDirectory(
        prefix="rc5plus-s2-i0-", dir=REPOSITORY_ROOT / ".tmp"
    ) as temporary:
        log_bytes, junit_bytes, pytest_returncode = _run_pytest(Path(temporary))
    checks, test_count = _test_checks(junit_bytes)
    checks.append(_causal_and_result_free_check(config))
    checks.append(_success_contract_check(config, compiled))
    if len(checks) != 17:
        raise RuntimeError("RC5+ audit category count is not exactly 17")
    failed = [row["name"] for row in checks if row["status"] != "PASS"]
    status = "PASS" if not failed and pytest_returncode == 0 else "HOLD"
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "artifact_status": status,
        "gate_id": GATE_ID,
        "result_free": True,
        "contains_observed_model_results": False,
        "observed_model_results": None,
        "official_test_accessed": False,
        "gpu_used": False,
        "execution_authorized": False,
        "implementation_design_gate_passed": status == "PASS",
        "complete_model_design_success_claimed": False,
        "performance_evaluated": False,
        "performance_gate_passed": None,
        "novelty_evaluated": False,
        "novelty_gate_passed": None,
        "config_binding": {
            "path": str(verified_config.source_path.relative_to(REPOSITORY_ROOT)),
            "source_bytes_sha256": verified_config.source_bytes_sha256,
            "canonical_sha256": verified_config.canonical_sha256,
        },
        "source_bindings": source_bindings,
        "pytest_evidence": {
            "fixed_selector_count": len(_test_run_selectors()),
            "testcase_count": test_count,
            "returncode": pytest_returncode,
            "log_sha256": _sha256_bytes(log_bytes),
            "junit_sha256": _sha256_bytes(junit_bytes),
        },
        "checks": checks,
        "passed_check_count": 17 - len(failed),
        "required_check_count": 17,
        "failed_checks": failed,
        "next_action": (
            "RUN_UPDATED_NOVELTY_REVIEW_BEFORE_SEPARATE_HASH_BOUND_EXPERIMENT_LAUNCH"
            if status == "PASS"
            else "FIX_FAILED_RESULT_FREE_IMPLEMENTATION_GATES_AND_RERUN"
        ),
        "claim_boundary": (
            "PASS proves only result-free RC5+ implementation-design integrity. "
            "It does not prove novelty, metric performance, AAAI sufficiency, "
            "or authorize any development/official-test experiment."
        ),
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    report["report_identity_sha256"] = _self_hash(
        report, "report_identity_sha256"
    )
    report_bytes = _pretty_bytes(report)
    commit: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "artifact_status": "committed_result_free_pass" if status == "PASS" else "committed_result_free_hold",
        "publication_order": "report_log_junit_then_commit_last",
        "report_filename": REPORT_FILENAME,
        "report_sha256": _sha256_bytes(report_bytes),
        "report_identity_sha256": report["report_identity_sha256"],
        "pytest_log_filename": PYTEST_LOG_FILENAME,
        "pytest_log_sha256": _sha256_bytes(log_bytes),
        "junit_filename": JUNIT_FILENAME,
        "junit_sha256": _sha256_bytes(junit_bytes),
        "required_check_count": 17,
        "passed_check_count": report["passed_check_count"],
        "result_free": True,
        "execution_authorized": False,
        "self_hash_algorithm": SELF_HASH_ALGORITHM,
    }
    commit["commit_identity_sha256"] = _self_hash(
        commit, "commit_identity_sha256"
    )
    _publish(
        output_directory,
        report_bytes,
        log_bytes,
        junit_bytes,
        _pretty_bytes(commit),
    )
    return report, commit


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed result-free 17-category RC5+ design audit."
    )
    parser.add_argument("--output-directory", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    output = arguments.output_directory
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    report, commit = audit(output.resolve())
    print(
        json.dumps(
            {
                "artifact_status": report["artifact_status"],
                "passed_check_count": report["passed_check_count"],
                "required_check_count": report["required_check_count"],
                "report_identity_sha256": report["report_identity_sha256"],
                "commit_identity_sha256": commit["commit_identity_sha256"],
                "output_directory": str(output),
                "execution_authorized": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["artifact_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
