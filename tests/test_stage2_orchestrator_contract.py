from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Barrier, Lock
import time
from types import SimpleNamespace

import pytest

from outputs.audit_tools.audit_stage2_development_completion import (
    DEVELOPMENT_CELL_RESULT_SCHEMA,
    DEVELOPMENT_RESULT_INDEX_SCHEMA,
    audit_stage2_development_completion,
    verify_stage2_development_completion,
)
from outputs.audit_tools.audit_stage2_i0 import (
    AUTHORITATIVE_STAGE2_CONFIG,
    AUTHORITATIVE_STAGE2_CONFIG_SHA256,
    LEGACY_STAGE2_CONFIG,
    S2_I0_ARTIFACT_TYPE,
    S2_I0_REPORT_SCHEMA,
    SOURCE_MODEL_AUDIT_SCHEMA,
    SOURCE_MODEL_COMMIT_SCHEMA,
    W13_SOURCE_FILES,
    audit_stage2_i0,
    verify_stage2_i0_report,
)
from scripts.orchestrate_stage2_crossfit import (
    AUTHORITATIVE_STAGE2_CONFIG_NAME,
    AUTHORITATIVE_STAGE2_CONFIG_PATH,
    AUTHORITATIVE_STAGE2_CONFIG_SHA256,
    DEVELOPMENT_LAUNCH_AUTHORIZATION_SCHEMA,
    EXECUTION_COMMAND_SPEC_SCHEMA,
    FIXED_BASE_SEEDS,
    FIXED_GPU_BY_OUTER_FOLD,
    FIXED_OUTER_FOLDS,
    FIXED_OUTER_TARGETS,
    LEGACY_STAGE2_ANALYSIS_PLAN_PATH,
    PRIMARY_METHODS,
    RETRY_AUTHORIZATION_SCHEMA,
    Stage2OrchestratorContractError,
    canonical_json_sha256,
    make_stage2_execution_job,
    materialize_stage2_execution_matrix_from_spec,
    orchestrate_stage2_crossfit,
    publish_stage2_execution_matrix,
    verify_stage2_execution_matrix,
    verify_stage2_retry_authorization,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _write_sha_sidecar(path: Path, digest: str) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )


def _source_audit_v2_workspace(
    root: Path,
    *,
    audit_schema: str = SOURCE_MODEL_AUDIT_SCHEMA,
    commit_schema: str = SOURCE_MODEL_COMMIT_SCHEMA,
    config_path: str = AUTHORITATIVE_STAGE2_CONFIG,
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    config_bytes = (repository / AUTHORITATIVE_STAGE2_CONFIG).read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == AUTHORITATIVE_STAGE2_CONFIG_SHA256
    bound_config = root / config_path
    bound_config.parent.mkdir(parents=True, exist_ok=True)
    bound_config.write_bytes(config_bytes)

    release_dir = root / "release" / "stage2-scoped"
    release_dir.mkdir(parents=True)
    release_names = {
        "manifest": "SCOPED_RELEASE_MANIFEST.json",
        "archive": "RC-IRSTD_STAGE2_MODEL_DESIGN_SCOPED_RELEASE.tar",
        "environment": "ENVIRONMENT.json",
        "commit": "COMMIT.json",
    }
    release_files: dict[str, dict[str, str]] = {}
    for logical, name in release_names.items():
        artifact = release_dir / name
        artifact.write_bytes(f"synthetic B4 {logical}\n".encode("utf-8"))
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        _write_sha_sidecar(artifact, digest)
        release_files[logical] = {"path": name, "sha256": digest}

    audit_payload: dict[str, object] = {
        "schema_version": audit_schema,
        "artifact_type": "rc_irstd_stage2_complete_model_design_freeze_audit",
        "artifact_status": "PASS",
        "gate_id": "S2_I0",
        "result_free": True,
        "result_free_scope": "synthetic source-only closure",
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
        "stage2_primary_method": (
            "T8_risk_aligned_monotone_no_reject_calibrator"
        ),
        "config_binding": {
            "path": config_path,
            "sha256": AUTHORITATIVE_STAGE2_CONFIG_SHA256,
        },
        "final_authority_bindings": [],
        "stage2_config_override_supported": False,
        "governance_bindings": [],
        "frozen_prerequisite_bindings": [],
        "scoped_release_binding": {
            "directory": release_dir.relative_to(root).as_posix(),
            "files": release_files,
        },
        "source_bindings": [
            {
                "path": config_path,
                "sha256": AUTHORITATIVE_STAGE2_CONFIG_SHA256,
            }
        ],
        "synthetic_test_evidence": {},
        "b2_full_merge_resolution": {
            "prior_status": "HOLD",
            "prior_artifact_sha256": _sha("prior-b2-hold"),
            "resolved_status": "PASS",
            "resolved_only_by_this_complete_audit": True,
        },
        "checks": [
            {"name": "synthetic_source_audit", "status": "PASS", "detail": True}
        ],
        "failed_checks": [],
        "next_action": (
            "STOP_MODEL_DESIGN_AND_AWAIT_SEPARATE_EXPERIMENT_LAUNCH_AUTHORIZATION"
        ),
        "claim_boundary": "engineering closure only",
        "report_content_sha256_algorithm": (
            "sha256-canonical-json-without-self-field-v1"
        ),
    }
    audit_payload["report_content_sha256"] = canonical_json_sha256(audit_payload)
    source_dir = root / "source-audit"
    source_dir.mkdir()
    report_path = source_dir / "S2_I0_REPORT.json"
    report_sha = _write_json(report_path, audit_payload)
    _write_sha_sidecar(report_path, report_sha)

    log_path = source_dir / "pytest.log"
    log_path.write_text("synthetic tests passed\n", encoding="utf-8")
    log_sha = hashlib.sha256(log_path.read_bytes()).hexdigest()
    commit_payload: dict[str, object] = {
        "schema_version": commit_schema,
        "artifact_status": "PASS",
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
    commit_path = source_dir / "COMMIT.json"
    commit_sha = _write_json(commit_path, commit_payload)
    _write_sha_sidecar(commit_path, commit_sha)
    return {
        "report_path": report_path,
        "report_sha": report_sha,
        "commit_path": commit_path,
        "commit_sha": commit_sha,
        "release_dir": release_dir,
    }


def _authoritative_stage2_config_binding() -> dict[str, str]:
    return {
        "name": AUTHORITATIVE_STAGE2_CONFIG_NAME,
        "path": AUTHORITATIVE_STAGE2_CONFIG_PATH,
        "sha256": AUTHORITATIVE_STAGE2_CONFIG_SHA256,
    }


def _install_authoritative_stage2_config(root: Path) -> None:
    source = Path(__file__).resolve().parents[1] / AUTHORITATIVE_STAGE2_CONFIG_PATH
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == AUTHORITATIVE_STAGE2_CONFIG_SHA256
    target = root / AUTHORITATIVE_STAGE2_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _jobs(input_path: str) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    input_sha = _sha("shared-input")
    for outer in FIXED_OUTER_FOLDS:
        for seed in FIXED_BASE_SEEDS:
            for method in PRIMARY_METHODS:
                jobs.append(
                    make_stage2_execution_job(
                        outer_fold_id=outer,
                        base_seed=seed,
                        method_id=method,
                        argv=[
                            "python",
                            "-m",
                            "synthetic.stage2_cell",
                            "--config",
                            AUTHORITATIVE_STAGE2_CONFIG_PATH,
                            "--config-sha256",
                            AUTHORITATIVE_STAGE2_CONFIG_SHA256,
                            "--method",
                            method,
                            "--outer-fold-id",
                            outer,
                            "--base-seed",
                            str(seed),
                        ],
                        input_bindings=[
                            _authoritative_stage2_config_binding(),
                            {
                                "name": "shared_input",
                                "path": input_path,
                                "sha256": input_sha,
                            }
                        ],
                        output_dir=f"runs/{outer}/s{seed}/{method.lower()}",
                    )
                )
    return jobs


def _matrix(root: Path, *, input_path: str = "absent/development-input.bin"):
    path, digest = publish_stage2_execution_matrix(
        _jobs(input_path), root / "matrix.json", repository_root=root
    )
    return path, digest


def _command_spec(input_path: str) -> dict[str, object]:
    records = [
        {
            "outer_fold_id": job["outer_fold_id"],
            "base_seed": job["base_seed"],
            "method_id": job["method_id"],
            "argv": job["argv"],
            "input_bindings": job["input_bindings"],
            "output_dir": job["output_dir"],
        }
        for job in _jobs(input_path)
    ]
    spec: dict[str, object] = {
        "schema_version": EXECUTION_COMMAND_SPEC_SCHEMA,
        "artifact_type": "rc_irstd_stage2_execution_command_spec",
        "artifact_status": "RESULT_FREE_FROZEN_INPUTS",
        "development_only": True,
        "official_test_accessed": False,
        "official_phase": None,
        "contains_observed_results": False,
        "records": records,
        "spec_content_sha256_algorithm": (
            "sha256-canonical-json-without-self-field-v1"
        ),
    }
    spec["spec_content_sha256"] = canonical_json_sha256(spec)
    return spec


def _i0(root: Path, *, status: str = "PASS") -> tuple[Path, str]:
    bound_payloads = {
        "source/model-report.json": b"source-model-report",
        "source/COMMIT.json": b"source-model-commit",
        **{path: path.encode("utf-8") for path in W13_SOURCE_FILES},
    }
    for relative, data in bound_payloads.items():
        bound_path = root / relative
        bound_path.parent.mkdir(parents=True, exist_ok=True)
        bound_path.write_bytes(data)
    checks = [
        {
            "name": "synthetic_i0",
            "status": "PASS" if status == "PASS" else "FAIL",
            "detail": "synthetic",
        }
    ]
    payload: dict[str, object] = {
        "schema_version": S2_I0_REPORT_SCHEMA,
        "artifact_type": S2_I0_ARTIFACT_TYPE,
        "artifact_status": status,
        "gate_id": "S2_I0",
        "result_free": True,
        "contains_observed_model_results": False,
        "observed_model_results": None,
        "development_data_accessed": False,
        "official_test_accessed": False,
        "gpu_used": False,
        "execution_authorized": False,
        "source_model_design_audit": {
            "path": "source/model-report.json",
            "sha256": hashlib.sha256(bound_payloads["source/model-report.json"]).hexdigest(),
        },
        "source_model_design_commit": {
            "path": "source/COMMIT.json",
            "sha256": hashlib.sha256(bound_payloads["source/COMMIT.json"]).hexdigest(),
        },
        "w13_source_bindings": [
            {"path": path, "sha256": hashlib.sha256(bound_payloads[path]).hexdigest()}
            for path in W13_SOURCE_FILES
        ],
        "checks": checks,
        "failed_checks": [] if status == "PASS" else ["synthetic_i0"],
        "next_action": (
            "STOP_MODEL_DESIGN_AND_AWAIT_SEPARATE_DEVELOPMENT_LAUNCH_AUTHORIZATION"
            if status == "PASS"
            else "REPAIR"
        ),
        "claim_boundary": "engineering closure only",
        "report_content_sha256_algorithm": (
            "sha256-canonical-json-without-self-field-v1"
        ),
    }
    payload["report_content_sha256"] = canonical_json_sha256(payload)
    path = root / "i0.json"
    return path, _write_json(path, payload)


def _launch(
    root: Path, *, matrix_sha: str, i0_sha: str
) -> tuple[Path, str]:
    payload: dict[str, object] = {
        "schema_version": DEVELOPMENT_LAUNCH_AUTHORIZATION_SCHEMA,
        "artifact_type": "rc_irstd_stage2_development_launch_authorization",
        "artifact_status": "PASS",
        "development_only": True,
        "official_test_accessed": False,
        "contains_observed_results": False,
        "execution_authorized": True,
        "authorized_phase": "S2_DGO_PRIMARY",
        "authorized_methods": ["T4", "T8"],
        "allowed_physical_gpus": [0, 1, 2],
        "s2_i0_report": {"path": "i0.json", "sha256": i0_sha},
        "execution_matrix": {"path": "matrix.json", "sha256": matrix_sha},
    }
    path = root / "launch.json"
    return path, _write_json(path, payload)


def test_dry_run_is_hash_complete_and_data_free(
    tmp_path: Path,
) -> None:
    matrix_path, matrix_sha = _matrix(tmp_path)
    result = orchestrate_stage2_crossfit(
        mode="dry-run",
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha,
        repository_root=tmp_path,
    )
    assert result["data_paths_opened"] == 0
    assert result["gpu_jobs_started"] == 0
    assert result["official_test_accessed"] is False
    assert len(result["commands"]) == 18
    records = [json.loads(line) for line in result["commands"]]
    assert [record["environment"]["CUDA_VISIBLE_DEVICES"] for record in records] == [
        str(FIXED_GPU_BY_OUTER_FOLD[outer])
        for outer in FIXED_OUTER_FOLDS
        for _seed in FIXED_BASE_SEEDS
        for _method in PRIMARY_METHODS
    ]
    assert all(
        [
            binding
            for binding in record["input_bindings"]
            if binding["name"] == AUTHORITATIVE_STAGE2_CONFIG_NAME
        ]
        == [_authoritative_stage2_config_binding()]
        for record in records
    )
    assert all(
        any(
            binding["name"] == "shared_input"
            and binding["sha256"] == _sha("shared-input")
            for binding in record["input_bindings"]
        )
        for record in records
    )
    assert all(len(record["command_sha256"]) == 64 for record in records)
    assert not (tmp_path / "absent" / "development-input.bin").exists()
    assert not (tmp_path / AUTHORITATIVE_STAGE2_CONFIG_PATH).exists()


def test_external_sha_command_spec_materializes_exact_18_job_matrix_without_data(
    tmp_path: Path,
) -> None:
    spec = _command_spec("absent/final-cell-input.bin")
    spec_path = tmp_path / "command-spec.json"
    spec_sha = _write_json(spec_path, spec)
    matrix_path, matrix_sha = materialize_stage2_execution_matrix_from_spec(
        spec_path,
        spec_sha,
        tmp_path / "materialized-matrix.json",
        repository_root=tmp_path,
    )
    matrix = verify_stage2_execution_matrix(
        matrix_path, matrix_sha, repository_root=tmp_path
    )
    assert len(matrix.payload["jobs"]) == 18
    assert all(
        [
            binding
            for binding in job["input_bindings"]
            if binding["name"] == AUTHORITATIVE_STAGE2_CONFIG_NAME
        ]
        == [_authoritative_stage2_config_binding()]
        for job in matrix.payload["jobs"]
    )
    assert not (tmp_path / "absent" / "final-cell-input.bin").exists()
    assert not (tmp_path / AUTHORITATIVE_STAGE2_CONFIG_PATH).exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "duplicate",
        "legacy_path",
        "different_sha",
        "extra_legacy",
        "extra_old_config",
    ),
)
def test_command_spec_rejects_non_authoritative_stage2_config_bindings(
    tmp_path: Path, mutation: str
) -> None:
    spec = _command_spec("absent/unopened-cell-input.bin")
    records = spec["records"]
    assert isinstance(records, list)
    first = records[0]
    bindings = first["input_bindings"]
    config = next(
        binding
        for binding in bindings
        if binding["name"] == AUTHORITATIVE_STAGE2_CONFIG_NAME
    )
    if mutation == "missing":
        first["input_bindings"] = [
            binding
            for binding in bindings
            if binding["name"] != AUTHORITATIVE_STAGE2_CONFIG_NAME
        ]
    elif mutation == "duplicate":
        bindings.append(dict(config))
    elif mutation == "legacy_path":
        config["path"] = LEGACY_STAGE2_ANALYSIS_PLAN_PATH
    elif mutation == "different_sha":
        config["sha256"] = _sha("obsolete-stage2-config")
    elif mutation == "extra_legacy":
        bindings.append(
            {
                "name": "analysis_plan",
                "path": LEGACY_STAGE2_ANALYSIS_PLAN_PATH,
                "sha256": _sha("legacy-analysis-plan"),
            }
        )
    else:
        bindings.append(
            {
                "name": "legacy_config",
                "path": "configs/aaai27_stage2_crossfit_v1.json",
                "sha256": _sha("legacy-stage2-config"),
            }
        )
    spec["spec_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in spec.items()
            if key != "spec_content_sha256"
        }
    )
    spec_path = tmp_path / f"bad-binding-{mutation}.json"
    spec_sha = _write_json(spec_path, spec)
    with pytest.raises(Stage2OrchestratorContractError):
        materialize_stage2_execution_matrix_from_spec(
            spec_path,
            spec_sha,
            tmp_path / f"bad-binding-{mutation}-matrix.json",
            repository_root=tmp_path,
        )
    assert not (tmp_path / f"bad-binding-{mutation}-matrix.json").exists()
    assert not (tmp_path / "absent" / "unopened-cell-input.bin").exists()


@pytest.mark.parametrize(
    "mutation",
    ("legacy_path", "different_sha", "missing", "extra_legacy_argument"),
)
def test_command_spec_rejects_non_authoritative_stage2_config_arguments(
    tmp_path: Path, mutation: str
) -> None:
    spec = _command_spec("absent/unopened-command-input.bin")
    records = spec["records"]
    assert isinstance(records, list)
    argv = records[0]["argv"]
    config_index = argv.index("--config")
    sha_index = argv.index("--config-sha256")
    if mutation == "legacy_path":
        argv[config_index + 1] = LEGACY_STAGE2_ANALYSIS_PLAN_PATH
    elif mutation == "different_sha":
        argv[sha_index + 1] = _sha("obsolete-command-config")
    elif mutation == "missing":
        del argv[config_index : config_index + 2]
    else:
        argv.extend(["--analysis-plan", LEGACY_STAGE2_ANALYSIS_PLAN_PATH])
    spec["spec_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in spec.items()
            if key != "spec_content_sha256"
        }
    )
    spec_path = tmp_path / f"bad-argv-{mutation}.json"
    spec_sha = _write_json(spec_path, spec)
    with pytest.raises(Stage2OrchestratorContractError):
        materialize_stage2_execution_matrix_from_spec(
            spec_path,
            spec_sha,
            tmp_path / f"bad-argv-{mutation}-matrix.json",
            repository_root=tmp_path,
        )


@pytest.mark.parametrize("mutation", ("legacy_path", "different_sha"))
def test_verified_materialized_matrix_rejects_config_authority_drift(
    tmp_path: Path, mutation: str
) -> None:
    matrix_path, _ = _matrix(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    job = payload["jobs"][0]
    config = next(
        binding
        for binding in job["input_bindings"]
        if binding["name"] == AUTHORITATIVE_STAGE2_CONFIG_NAME
    )
    if mutation == "legacy_path":
        config["path"] = LEGACY_STAGE2_ANALYSIS_PLAN_PATH
    else:
        config["sha256"] = _sha("drifted-materialized-config")
    job["input_identity_sha256"] = canonical_json_sha256(job["input_bindings"])
    job["command_sha256"] = canonical_json_sha256(
        {
            "argv": job["argv"],
            "environment": job["environment"],
            "input_bindings": job["input_bindings"],
            "output_dir": job["output_dir"],
        }
    )
    payload["matrix_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "matrix_content_sha256"
        }
    )
    tampered_sha = _write_json(matrix_path, payload)
    with pytest.raises(Stage2OrchestratorContractError):
        verify_stage2_execution_matrix(
            matrix_path, tampered_sha, repository_root=tmp_path
        )


def test_gpu_mapping_and_no_official_phase_are_frozen(
    tmp_path: Path,
) -> None:
    matrix_path, _ = _matrix(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["jobs"][0]["gpu_id"] = 2
    payload["jobs"][0]["environment"] = {"CUDA_VISIBLE_DEVICES": "2"}
    payload["matrix_content_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "matrix_content_sha256"}
    )
    bad_sha = _write_json(matrix_path, payload)
    with pytest.raises(Stage2OrchestratorContractError, match="GPU"):
        verify_stage2_execution_matrix(
            matrix_path, bad_sha, repository_root=tmp_path
        )

    matrix_path.unlink()
    matrix_path.with_name(matrix_path.name + ".sha256").unlink()
    matrix_path, _ = _matrix(tmp_path)
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    payload["official_phase"] = {"phase": "confirmatory"}
    payload["matrix_content_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "matrix_content_sha256"}
    )
    bad_sha = _write_json(matrix_path, payload)
    with pytest.raises(Stage2OrchestratorContractError, match="no official phase"):
        verify_stage2_execution_matrix(
            matrix_path, bad_sha, repository_root=tmp_path
        )


def test_real_mode_requires_verified_s2_i0(
    tmp_path: Path,
) -> None:
    _install_authoritative_stage2_config(tmp_path)
    input_path = tmp_path / "metadata" / "shared-input.bin"
    input_path.parent.mkdir()
    input_path.write_bytes(b"shared-input")
    matrix_path, matrix_sha = _matrix(
        tmp_path, input_path="metadata/shared-input.bin"
    )
    with pytest.raises(Stage2OrchestratorContractError, match="requires external PASS"):
        orchestrate_stage2_crossfit(
            mode="real",
            matrix_path=matrix_path,
            matrix_sha256=matrix_sha,
            repository_root=tmp_path,
        )
    i0_path, i0_sha = _i0(tmp_path)
    launch_path, launch_sha = _launch(
        tmp_path, matrix_sha=matrix_sha, i0_sha=i0_sha
    )
    calls: list[tuple[list[str], str]] = []

    def runner(argv, *, cwd, env, check):
        assert cwd == tmp_path
        assert check is False
        calls.append((list(argv), env["CUDA_VISIBLE_DEVICES"]))
        return SimpleNamespace(returncode=0)

    first_job_id = (
        f"s2_dgo__{FIXED_OUTER_FOLDS[0]}__s{FIXED_BASE_SEEDS[0]}__t4"
    )
    result = orchestrate_stage2_crossfit(
        mode="real",
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha,
        s2_i0_report=i0_path,
        s2_i0_report_sha256=i0_sha,
        launch_authorization=launch_path,
        launch_authorization_sha256=launch_sha,
        job_ids=[first_job_id],
        repository_root=tmp_path,
        runner=runner,
    )
    assert len(result["completed"]) == 1
    assert calls[0][1] == "0"
    assert result["official_phase_present"] is False
    with pytest.raises(Stage2OrchestratorContractError, match="SHA-256"):
        orchestrate_stage2_crossfit(
            mode="real",
            matrix_path=matrix_path,
            matrix_sha256=matrix_sha,
            s2_i0_report=i0_path,
            s2_i0_report_sha256=_sha("wrong"),
            launch_authorization=launch_path,
            launch_authorization_sha256=launch_sha,
            job_ids=[first_job_id],
            repository_root=tmp_path,
            runner=runner,
        )


def test_real_scheduler_runs_three_outer_queues_concurrently_but_one_per_gpu(
    tmp_path: Path,
) -> None:
    _install_authoritative_stage2_config(tmp_path)
    input_path = tmp_path / "metadata" / "shared-input.bin"
    input_path.parent.mkdir()
    input_path.write_bytes(b"shared-input")
    matrix_path, matrix_sha = _matrix(
        tmp_path, input_path="metadata/shared-input.bin"
    )
    i0_path, i0_sha = _i0(tmp_path)
    launch_path, launch_sha = _launch(
        tmp_path, matrix_sha=matrix_sha, i0_sha=i0_sha
    )
    first_wave = Barrier(3)
    lock = Lock()
    active_by_gpu = {0: 0, 1: 0, 2: 0}
    maximum_by_gpu = {0: 0, 1: 0, 2: 0}
    maximum_total = 0
    first_seen: set[int] = set()

    def runner(argv, *, cwd, env, check):
        nonlocal maximum_total
        gpu = int(env["CUDA_VISIBLE_DEVICES"])
        with lock:
            active_by_gpu[gpu] += 1
            maximum_by_gpu[gpu] = max(maximum_by_gpu[gpu], active_by_gpu[gpu])
            maximum_total = max(maximum_total, sum(active_by_gpu.values()))
            is_first = gpu not in first_seen
            first_seen.add(gpu)
        if is_first:
            first_wave.wait(timeout=5)
        time.sleep(0.002)
        with lock:
            active_by_gpu[gpu] -= 1
        return SimpleNamespace(returncode=0)

    result = orchestrate_stage2_crossfit(
        mode="real",
        matrix_path=matrix_path,
        matrix_sha256=matrix_sha,
        s2_i0_report=i0_path,
        s2_i0_report_sha256=i0_sha,
        launch_authorization=launch_path,
        launch_authorization_sha256=launch_sha,
        repository_root=tmp_path,
        runner=runner,
    )
    assert len(result["completed"]) == 18
    assert first_seen == {0, 1, 2}
    assert maximum_total >= 3
    assert maximum_by_gpu == {0: 1, 1: 1, 2: 1}


def test_retry_is_pre_result_infrastructure_or_implementation_only(
    tmp_path: Path,
) -> None:
    matrix_path, matrix_sha = _matrix(tmp_path)
    matrix = verify_stage2_execution_matrix(
        matrix_path, matrix_sha, repository_root=tmp_path
    )
    job = matrix.payload["jobs"][0]
    payload: dict[str, object] = {
        "schema_version": RETRY_AUTHORIZATION_SCHEMA,
        "artifact_type": "rc_irstd_stage2_retry_authorization",
        "artifact_status": "PASS",
        "development_only": True,
        "official_test_accessed": False,
        "execution_authorized": True,
        "failure_class": "METRIC_UNDERPERFORMANCE",
        "pre_result_failure": True,
        "observed_result_opened": False,
        "metrics_opened": False,
        "kind": "RETRY",
        "matrix_sha256": matrix_sha,
        "job_id": job["job_id"],
        "command_sha256": job["command_sha256"],
        "input_identity_sha256": job["input_identity_sha256"],
        "attempt": 2,
        "resume_binding": None,
        "resume_arguments": [],
    }
    path = tmp_path / "retry.json"
    digest = _write_json(path, payload)
    with pytest.raises(Stage2OrchestratorContractError, match="limited"):
        verify_stage2_retry_authorization(
            path,
            digest,
            matrix=matrix,
            job_id=job["job_id"],
            repository_root=tmp_path,
        )
    payload["failure_class"] = "INFRASTRUCTURE_FAILURE"
    digest = _write_json(path, payload)
    verified = verify_stage2_retry_authorization(
        path,
        digest,
        matrix=matrix,
        job_id=job["job_id"],
        repository_root=tmp_path,
    )
    assert verified["input_identity_sha256"] == job["input_identity_sha256"]
    payload["command_sha256"] = _sha("changed-command")
    digest = _write_json(path, payload)
    with pytest.raises(Stage2OrchestratorContractError, match="changed"):
        verify_stage2_retry_authorization(
            path,
            digest,
            matrix=matrix,
            job_id=job["job_id"],
            repository_root=tmp_path,
        )


def test_i0_wrapper_binds_w13_source_surface_without_data_access(
    tmp_path: Path,
) -> None:
    for relative in W13_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source {relative}\n", encoding="utf-8")
    source = _source_audit_v2_workspace(tmp_path)
    output, output_sha = audit_stage2_i0(
        source_model_report=source["report_path"],
        source_model_report_sha256=source["report_sha"],
        source_model_commit=source["commit_path"],
        source_model_commit_sha256=source["commit_sha"],
        output_path=tmp_path / "final-i0.json",
        repository_root=tmp_path,
    )
    verified = verify_stage2_i0_report(
        output, output_sha, repository_root=tmp_path
    )
    assert verified.payload["artifact_status"] == "PASS"
    assert verified.payload["development_data_accessed"] is False
    assert verified.payload["execution_authorized"] is False


@pytest.mark.parametrize(
    ("audit_schema", "commit_schema", "config_path", "message"),
    (
        (
            "rc-irstd.stage2-model-design-freeze-audit.v1",
            SOURCE_MODEL_COMMIT_SCHEMA,
            AUTHORITATIVE_STAGE2_CONFIG,
            "not S2_I0 PASS",
        ),
        (
            SOURCE_MODEL_AUDIT_SCHEMA,
            "rc-irstd.stage2-model-design-freeze-audit-commit.v1",
            AUTHORITATIVE_STAGE2_CONFIG,
            "commit is not PASS",
        ),
        (
            SOURCE_MODEL_AUDIT_SCHEMA,
            SOURCE_MODEL_COMMIT_SCHEMA,
            LEGACY_STAGE2_CONFIG,
            "authoritative v2 config",
        ),
    ),
)
def test_i0_wrapper_rejects_v1_source_contracts_and_legacy_config(
    tmp_path: Path,
    audit_schema: str,
    commit_schema: str,
    config_path: str,
    message: str,
) -> None:
    for relative in W13_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source {relative}\n", encoding="utf-8")
    source = _source_audit_v2_workspace(
        tmp_path,
        audit_schema=audit_schema,
        commit_schema=commit_schema,
        config_path=config_path,
    )
    with pytest.raises(Stage2OrchestratorContractError, match=message):
        audit_stage2_i0(
            source_model_report=source["report_path"],
            source_model_report_sha256=source["report_sha"],
            source_model_commit=source["commit_path"],
            source_model_commit_sha256=source["commit_sha"],
            output_path=tmp_path / "rejected-i0.json",
            repository_root=tmp_path,
        )


def test_i0_wrapper_rehashes_all_four_b4_scoped_release_files(
    tmp_path: Path,
) -> None:
    for relative in W13_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic source {relative}\n", encoding="utf-8")
    source = _source_audit_v2_workspace(tmp_path)
    manifest = source["release_dir"] / "SCOPED_RELEASE_MANIFEST.json"
    manifest.write_bytes(b"mutated after source audit publication\n")
    with pytest.raises(Stage2OrchestratorContractError, match="SHA-256 mismatch"):
        audit_stage2_i0(
            source_model_report=source["report_path"],
            source_model_report_sha256=source["report_sha"],
            source_model_commit=source["commit_path"],
            source_model_commit_sha256=source["commit_sha"],
            output_path=tmp_path / "rejected-b4-i0.json",
            repository_root=tmp_path,
        )


def _completion_workspace(root: Path) -> dict[str, object]:
    matrix_path, matrix_sha = _matrix(root)
    i0_path, i0_sha = _i0(root)
    records: list[dict[str, object]] = []
    for outer in FIXED_OUTER_FOLDS:
        for seed in FIXED_BASE_SEEDS:
            windows = [
                f"{outer}__s{seed}__w{index}"
                for index in range(
                    1 if outer == "outer_leave_nuaa_sirst" else 3
                )
            ]
            pair_identities = {
                "execution_matrix_sha256": matrix_sha,
                "s2_i0_report_sha256": i0_sha,
                "shared_pair_identity_sha256": _sha(f"{outer}:{seed}:pair"),
                "context_identity_sha256": _sha(f"{outer}:{seed}:context"),
                "query_identity_sha256": _sha(f"{outer}:{seed}:query"),
                "detector_score_identity_sha256": _sha(
                    f"{outer}:{seed}:detector"
                ),
                "budget_grid_sha256": _sha("budget-grid"),
            }
            for method in PRIMARY_METHODS:
                cell: dict[str, object] = {
                    "schema_version": DEVELOPMENT_CELL_RESULT_SCHEMA,
                    "artifact_type": "rc_irstd_stage2_development_cell_result",
                    "artifact_status": "COMPLETE",
                    "development_only": True,
                    "official_test_accessed": False,
                    "outer_fold_id": outer,
                    "outer_target_domain": FIXED_OUTER_TARGETS[outer],
                    "base_seed": seed,
                    "method_id": method,
                    "primary_budget": 1e-5,
                    "estimable": True,
                    "missing": False,
                    "imputed": False,
                    "t9_used": False,
                    "window_ids": windows,
                    "metrics": {
                        "bsr": 0.5 if method == "T4" else 0.6,
                        "pd": 0.7,
                        "log_excess": 0.1,
                    },
                    "t8_monotonicity_violation_count": 0,
                    "identities": pair_identities,
                    "result_content_sha256_algorithm": (
                        "sha256-canonical-json-without-self-field-v1"
                    ),
                }
                cell["result_content_sha256"] = canonical_json_sha256(cell)
                relative = f"results/{outer}/s{seed}/{method.lower()}.json"
                path = root / relative
                digest = _write_json(path, cell)
                records.append(
                    {
                        "outer_fold_id": outer,
                        "outer_target_domain": FIXED_OUTER_TARGETS[outer],
                        "base_seed": seed,
                        "method_id": method,
                        "artifact": {"path": relative, "sha256": digest},
                    }
                )
    index: dict[str, object] = {
        "schema_version": DEVELOPMENT_RESULT_INDEX_SCHEMA,
        "artifact_type": "rc_irstd_stage2_development_result_index",
        "artifact_status": "COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "official_phase": None,
        "contains_missing_values": False,
        "contains_imputed_values": False,
        "t9_present": False,
        "execution_matrix": {"path": "matrix.json", "sha256": matrix_sha},
        "s2_i0_report": {"path": "i0.json", "sha256": i0_sha},
        "records": records,
        "index_content_sha256_algorithm": (
            "sha256-canonical-json-without-self-field-v1"
        ),
    }
    index["index_content_sha256"] = canonical_json_sha256(index)
    index_path = root / "result-index.json"
    index_sha = _write_json(index_path, index)
    return {
        "matrix_path": matrix_path,
        "matrix_sha": matrix_sha,
        "i0_path": i0_path,
        "i0_sha": i0_sha,
        "index_path": index_path,
        "index_sha": index_sha,
    }


def test_completion_auditor_requires_exact_nine_t4_t8_pairs_and_identity_closure(
    tmp_path: Path,
) -> None:
    workspace = _completion_workspace(tmp_path)
    output, digest = audit_stage2_development_completion(
        result_index=workspace["index_path"],
        result_index_sha256=workspace["index_sha"],
        execution_matrix=workspace["matrix_path"],
        execution_matrix_sha256=workspace["matrix_sha"],
        s2_i0_report=workspace["i0_path"],
        s2_i0_report_sha256=workspace["i0_sha"],
        output_path=tmp_path / "completion.json",
        repository_root=tmp_path,
    )
    verified = verify_stage2_development_completion(
        output, digest, repository_root=tmp_path
    )
    assert verified.payload["domain_seed_cell_count"] == 9
    assert verified.payload["method_result_count"] == 18
    assert verified.payload["s2_dgo_decision"] == (
        "NOT_EVALUATED_BY_COMPLETION_AUDITOR"
    )
    assert verified.payload["official_phase_authorized"] is False


@pytest.mark.parametrize("mutation", ("missing", "imputed", "t9", "identity"))
def test_completion_rejects_missing_imputed_t9_and_pair_identity_drift(
    tmp_path: Path, mutation: str
) -> None:
    workspace = _completion_workspace(tmp_path)
    index_path = workspace["index_path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record_index = 1 if mutation == "identity" else 0
    record = index["records"][record_index]
    cell_path = tmp_path / record["artifact"]["path"]
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    if mutation == "identity":
        cell["identities"]["query_identity_sha256"] = _sha("mutated-query")
    else:
        cell[mutation if mutation != "t9" else "t9_used"] = True
    cell["result_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in cell.items()
            if key != "result_content_sha256"
        }
    )
    record["artifact"]["sha256"] = _write_json(cell_path, cell)
    if mutation == "missing":
        index["contains_missing_values"] = True
    elif mutation == "imputed":
        index["contains_imputed_values"] = True
    elif mutation == "t9":
        index["t9_present"] = True
    index["index_content_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in index.items()
            if key != "index_content_sha256"
        }
    )
    index_sha = _write_json(index_path, index)
    with pytest.raises(Stage2OrchestratorContractError):
        audit_stage2_development_completion(
            result_index=index_path,
            result_index_sha256=index_sha,
            execution_matrix=workspace["matrix_path"],
            execution_matrix_sha256=workspace["matrix_sha"],
            s2_i0_report=workspace["i0_path"],
            s2_i0_report_sha256=workspace["i0_sha"],
            output_path=tmp_path / f"completion-{mutation}.json",
            repository_root=tmp_path,
        )
