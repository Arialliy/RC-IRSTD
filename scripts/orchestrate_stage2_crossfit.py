#!/usr/bin/env python3
"""Fail-closed Stage-2 development orchestrator.

Dry-run consumes only one externally hash-bound metadata matrix and emits
canonical JSON command records.  It never resolves any referenced data path.
Real execution additionally requires externally hash-bound PASS S2_I0 and
development-launch authorization artifacts.  There is deliberately no
official/confirmatory phase in this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from threading import Event
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_MATRIX_SCHEMA = "rc-irstd.stage2-execution-matrix.v1"
EXECUTION_COMMAND_SPEC_SCHEMA = "rc-irstd.stage2-execution-command-spec.v1"
DEVELOPMENT_LAUNCH_AUTHORIZATION_SCHEMA = (
    "rc-irstd.stage2-development-launch-authorization.v1"
)
RETRY_AUTHORIZATION_SCHEMA = "rc-irstd.stage2-retry-authorization.v1"
MATRIX_ARTIFACT_TYPE = "rc_irstd_stage2_execution_matrix"
FIXED_BASE_SEEDS = (42, 123, 3407)
FIXED_OUTER_FOLDS = (
    "outer_leave_nuaa_sirst",
    "outer_leave_nudt_sirst",
    "outer_leave_irstd_1k",
)
FIXED_OUTER_TARGETS = {
    "outer_leave_nuaa_sirst": "NUAA-SIRST",
    "outer_leave_nudt_sirst": "NUDT-SIRST",
    "outer_leave_irstd_1k": "IRSTD-1K",
}
FIXED_GPU_BY_OUTER_FOLD = {
    "outer_leave_nuaa_sirst": 0,
    "outer_leave_nudt_sirst": 1,
    "outer_leave_irstd_1k": 2,
}
PRIMARY_METHODS = ("T4", "T8")
AUTHORITATIVE_STAGE2_CONFIG_NAME = "stage2_config"
AUTHORITATIVE_STAGE2_CONFIG_PATH = "configs/aaai27_stage2_crossfit_v2.json"
AUTHORITATIVE_STAGE2_CONFIG_SHA256 = (
    "dd5e49c9633612e52c00091cfcb2543b48f5fd3f0d7fc5690f297ec0e7d9d963"
)
LEGACY_STAGE2_ANALYSIS_PLAN_PATH = "configs/aaai27_analysis_plan.json"
ALLOWED_FAILURE_CLASSES = frozenset(
    {"IMPLEMENTATION_FAILURE", "INFRASTRUCTURE_FAILURE"}
)
_SHA256_HEX = frozenset("0123456789abcdef")
_VERIFIED_MATRIX_CAPABILITY = object()


class Stage2OrchestratorContractError(RuntimeError):
    """Raised before any job starts when an orchestration contract is invalid."""


@dataclass(frozen=True)
class VerifiedStage2ExecutionMatrix:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    _capability: object


def assert_verified_execution_matrix(
    value: Any,
) -> VerifiedStage2ExecutionMatrix:
    if (
        not isinstance(value, VerifiedStage2ExecutionMatrix)
        or value._capability is not _VERIFIED_MATRIX_CAPABILITY
    ):
        raise TypeError("a verifier-created Stage-2 execution matrix is required")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2OrchestratorContractError(
            "value is not canonical finite JSON"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise Stage2OrchestratorContractError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


def _strict_bool(value: Any, expected: bool, name: str) -> None:
    if type(value) is not bool or value is not expected:
        raise Stage2OrchestratorContractError(
            f"{name} must be exact JSON {str(expected).lower()}"
        )


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage2OrchestratorContractError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise Stage2OrchestratorContractError(
            f"{name} must be at least {minimum}"
        )
    return value


def _exact_keys(
    value: Any, expected: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise Stage2OrchestratorContractError(
            f"{name} keys mismatch: {observed}"
        )
    return value


def _parse_json_bytes(data: bytes, name: str) -> Mapping[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2OrchestratorContractError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise Stage2OrchestratorContractError(
            f"{name} contains non-finite JSON constant {value}"
        )

    try:
        parsed = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2OrchestratorContractError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(parsed, Mapping):
        raise Stage2OrchestratorContractError(f"{name} root must be an object")
    return parsed


def _repository_root(value: str | Path | None) -> Path:
    raw = Path(value) if value is not None else REPOSITORY_ROOT
    if not raw.is_absolute() or raw.is_symlink():
        raise Stage2OrchestratorContractError(
            "repository_root must be absolute and non-symlink"
        )
    resolved = raw.resolve(strict=True)
    if resolved != raw or not resolved.is_dir():
        raise Stage2OrchestratorContractError(
            "repository_root must be a canonical directory"
        )
    return resolved


def _input_file(value: str | Path, root: Path, name: str) -> Path:
    raw = Path(value)
    if ".." in raw.parts:
        raise Stage2OrchestratorContractError(f"{name} may not contain '..'")
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise Stage2OrchestratorContractError(f"{name} may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2OrchestratorContractError(f"{name} does not exist") from error
    if (
        resolved != candidate.absolute()
        or not resolved.is_file()
        or not resolved.is_relative_to(root)
    ):
        raise Stage2OrchestratorContractError(
            f"{name} must be a canonical repository file"
        )
    cursor = resolved.parent
    while cursor != root:
        if cursor.is_symlink():
            raise Stage2OrchestratorContractError(
                f"{name} has a symlink ancestor"
            )
        cursor = cursor.parent
    return resolved


def _stable_file_bytes(path: Path, name: str) -> tuple[bytes, str]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2OrchestratorContractError(f"{name} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    signature_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    signature_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if signature_before != signature_after:
        raise Stage2OrchestratorContractError(f"{name} changed while read")
    data = b"".join(chunks)
    return data, hashlib.sha256(data).hexdigest()


def _relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Stage2OrchestratorContractError(f"{name} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise Stage2OrchestratorContractError(
            f"{name} must be one normalized repository-relative path"
        )
    lowered_parts = tuple(part.lower() for part in path.parts)
    if any(
        part in {"test", "official_test", "official-test", "confirmatory"}
        or ("official" in part and "test" in part)
        for part in lowered_parts
    ):
        raise Stage2OrchestratorContractError(
            f"{name} may not identify official/confirmatory data"
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise Stage2OrchestratorContractError(f"{name} must be non-empty text")
    return value


_BINDING_KEYS = frozenset({"name", "path", "sha256"})
_JOB_KEYS = frozenset(
    {
        "job_id",
        "phase",
        "outer_fold_id",
        "outer_target_domain",
        "base_seed",
        "method_id",
        "gpu_id",
        "environment",
        "argv",
        "input_bindings",
        "input_identity_sha256",
        "command_sha256",
        "output_dir",
        "attempt",
        "resume",
    }
)
_MATRIX_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "official_phase",
        "s2_dgo_status",
        "contains_observed_results",
        "base_seeds",
        "outer_folds",
        "methods",
        "gpu_mapping",
        "jobs",
        "matrix_content_sha256_algorithm",
        "matrix_content_sha256",
    }
)
_SPEC_RECORD_KEYS = frozenset(
    {
        "outer_fold_id",
        "base_seed",
        "method_id",
        "argv",
        "input_bindings",
        "output_dir",
    }
)
_SPEC_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "official_phase",
        "contains_observed_results",
        "records",
        "spec_content_sha256_algorithm",
        "spec_content_sha256",
    }
)


def _validate_binding(value: Any, name: str) -> dict[str, str]:
    item = _exact_keys(value, _BINDING_KEYS, name)
    return {
        "name": _text(item["name"], f"{name}.name"),
        "path": _relative_path(item["path"], f"{name}.path"),
        "sha256": _strict_sha(item["sha256"], f"{name}.sha256"),
    }


def _require_authoritative_stage2_config_binding(
    bindings: Sequence[Mapping[str, str]], name: str
) -> None:
    """Require one non-overridable Stage-2 v2 configuration binding.

    The old analysis plan remains valid historical Stage-1 evidence, but it is
    never an execution authority for this Stage-2 orchestrator.  Rejecting any
    additional config-shaped binding also prevents a command from presenting
    the v2 binding while silently consuming a second configuration.
    """

    authoritative = [
        binding
        for binding in bindings
        if binding["name"] == AUTHORITATIVE_STAGE2_CONFIG_NAME
    ]
    if len(authoritative) != 1:
        raise Stage2OrchestratorContractError(
            f"{name} must contain exactly one {AUTHORITATIVE_STAGE2_CONFIG_NAME!r} binding"
        )
    expected = {
        "name": AUTHORITATIVE_STAGE2_CONFIG_NAME,
        "path": AUTHORITATIVE_STAGE2_CONFIG_PATH,
        "sha256": AUTHORITATIVE_STAGE2_CONFIG_SHA256,
    }
    if dict(authoritative[0]) != expected:
        raise Stage2OrchestratorContractError(
            f"{name}.{AUTHORITATIVE_STAGE2_CONFIG_NAME} must bind the exact "
            "authoritative Stage-2 v2 path and SHA-256"
        )
    for binding in bindings:
        if binding["path"] == LEGACY_STAGE2_ANALYSIS_PLAN_PATH:
            raise Stage2OrchestratorContractError(
                f"{name} may not bind the legacy Stage-2 analysis plan"
            )
        if binding["name"] != AUTHORITATIVE_STAGE2_CONFIG_NAME and (
            "config" in binding["name"].lower()
            or Path(binding["path"]).parts[:1] == ("configs",)
        ):
            raise Stage2OrchestratorContractError(
                f"{name} may not contain an additional configuration binding"
            )


def _require_authoritative_stage2_config_arguments(
    argv: Sequence[str], name: str
) -> None:
    if any(
        token == LEGACY_STAGE2_ANALYSIS_PLAN_PATH
        or token.endswith("=" + LEGACY_STAGE2_ANALYSIS_PLAN_PATH)
        for token in argv
    ):
        raise Stage2OrchestratorContractError(
            f"{name} may not reference the legacy Stage-2 analysis plan"
        )
    if _argument_value(argv, "--config") != AUTHORITATIVE_STAGE2_CONFIG_PATH:
        raise Stage2OrchestratorContractError(
            f"{name} must bind --config to the authoritative Stage-2 v2 path"
        )
    if (
        _argument_value(argv, "--config-sha256")
        != AUTHORITATIVE_STAGE2_CONFIG_SHA256
    ):
        raise Stage2OrchestratorContractError(
            f"{name} must bind --config-sha256 to the authoritative Stage-2 v2 SHA-256"
        )


def _argument_value(argv: Sequence[str], flag: str) -> str | None:
    matches: list[str] = []
    for index, token in enumerate(argv):
        if token == flag:
            if index + 1 >= len(argv):
                raise Stage2OrchestratorContractError(f"{flag} lacks a value")
            matches.append(argv[index + 1])
        elif token.startswith(flag + "="):
            matches.append(token.split("=", 1)[1])
    if len(matches) > 1:
        raise Stage2OrchestratorContractError(f"{flag} is repeated")
    return matches[0] if matches else None


def _validate_job(
    raw: Any,
    *,
    expected_outer_fold: str,
    expected_seed: int,
    expected_method: str,
) -> dict[str, Any]:
    job = dict(_exact_keys(raw, _JOB_KEYS, "execution job"))
    outer = _text(job["outer_fold_id"], "job.outer_fold_id")
    seed = _strict_int(job["base_seed"], "job.base_seed")
    method = _text(job["method_id"], "job.method_id")
    if (outer, seed, method) != (
        expected_outer_fold,
        expected_seed,
        expected_method,
    ):
        raise Stage2OrchestratorContractError("execution job order/identity mismatch")
    if job["phase"] != "S2_DGO_PRIMARY":
        raise Stage2OrchestratorContractError(
            "pre-GO execution matrix may contain only S2_DGO_PRIMARY"
        )
    if job["outer_target_domain"] != FIXED_OUTER_TARGETS[outer]:
        raise Stage2OrchestratorContractError("job outer target mismatch")
    gpu_id = _strict_int(job["gpu_id"], "job.gpu_id")
    if gpu_id != FIXED_GPU_BY_OUTER_FOLD[outer] or gpu_id not in {0, 1, 2}:
        raise Stage2OrchestratorContractError("job violates fixed GPU 0/1/2 mapping")
    environment = job["environment"]
    if environment != {"CUDA_VISIBLE_DEVICES": str(gpu_id)}:
        raise Stage2OrchestratorContractError(
            "job environment must contain only the fixed CUDA_VISIBLE_DEVICES"
        )
    argv = job["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(token, str) or not token or "\x00" in token for token in argv)
    ):
        raise Stage2OrchestratorContractError("job.argv must be non-empty string tokens")
    lowered = tuple(token.lower() for token in argv)
    if any(
        token in {"--official", "--official-test", "--confirmatory"}
        or "official_test" in token
        or "official-test" in token
        for token in lowered
    ):
        raise Stage2OrchestratorContractError(
            "official/confirmatory command arguments are forbidden before S2_DGO GO"
        )
    if any(token == "--resume" or token.startswith("--resume=") or token == "--resume-commit" or token.startswith("--resume-commit=") for token in argv):
        raise Stage2OrchestratorContractError(
            "base execution matrix may not contain resume arguments"
        )
    _require_authoritative_stage2_config_arguments(argv, "job.argv")
    if _argument_value(argv, "--method") != method:
        raise Stage2OrchestratorContractError(
            "job command must bind its exact --method"
        )
    raw_bindings = job["input_bindings"]
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise Stage2OrchestratorContractError(
            "job requires at least one external input binding"
        )
    bindings = [
        _validate_binding(item, f"job.input_bindings[{index}]")
        for index, item in enumerate(raw_bindings)
    ]
    _require_authoritative_stage2_config_binding(bindings, "job.input_bindings")
    names = [item["name"] for item in bindings]
    paths = [item["path"] for item in bindings]
    if names != sorted(names) or len(names) != len(set(names)) or len(paths) != len(set(paths)):
        raise Stage2OrchestratorContractError(
            "job input bindings must be name-sorted and unique"
        )
    input_identity = canonical_json_sha256(bindings)
    if job["input_identity_sha256"] != input_identity:
        raise Stage2OrchestratorContractError("job input identity mismatch")
    output_dir = _relative_path(job["output_dir"], "job.output_dir")
    expected_job_id = f"s2_dgo__{outer}__s{seed}__{method.lower()}"
    if job["job_id"] != expected_job_id:
        raise Stage2OrchestratorContractError("job_id mismatch")
    _strict_int(job["attempt"], "job.attempt", minimum=1)
    if job["attempt"] != 1 or job["resume"] is not None:
        raise Stage2OrchestratorContractError(
            "base matrix must be attempt one with resume=null"
        )
    command_projection = {
        "argv": list(argv),
        "environment": dict(environment),
        "input_bindings": bindings,
        "output_dir": output_dir,
    }
    if job["command_sha256"] != canonical_json_sha256(command_projection):
        raise Stage2OrchestratorContractError("job command SHA-256 mismatch")
    return job


def _validate_matrix_payload(payload: Mapping[str, Any]) -> None:
    _exact_keys(payload, _MATRIX_KEYS, "execution matrix")
    if (
        payload["schema_version"] != EXECUTION_MATRIX_SCHEMA
        or payload["artifact_type"] != MATRIX_ARTIFACT_TYPE
        or payload["artifact_status"] != "FROZEN_DEVELOPMENT_ONLY"
    ):
        raise Stage2OrchestratorContractError("execution matrix identity mismatch")
    _strict_bool(payload["development_only"], True, "matrix.development_only")
    _strict_bool(
        payload["official_test_accessed"],
        False,
        "matrix.official_test_accessed",
    )
    _strict_bool(
        payload["contains_observed_results"],
        False,
        "matrix.contains_observed_results",
    )
    if payload["official_phase"] is not None or payload["s2_dgo_status"] != "NOT_RUN":
        raise Stage2OrchestratorContractError(
            "pre-GO matrix must have no official phase and S2_DGO NOT_RUN"
        )
    if payload["base_seeds"] != list(FIXED_BASE_SEEDS):
        raise Stage2OrchestratorContractError("matrix base seeds mismatch")
    if payload["outer_folds"] != list(FIXED_OUTER_FOLDS):
        raise Stage2OrchestratorContractError("matrix outer folds mismatch")
    if payload["methods"] != list(PRIMARY_METHODS):
        raise Stage2OrchestratorContractError(
            "matrix must contain only primary T4/T8"
        )
    if payload["gpu_mapping"] != FIXED_GPU_BY_OUTER_FOLD:
        raise Stage2OrchestratorContractError("matrix GPU mapping mismatch")
    jobs = payload["jobs"]
    if not isinstance(jobs, list) or len(jobs) != 18:
        raise Stage2OrchestratorContractError(
            "matrix requires exactly 3 domains x 3 seeds x T4/T8"
        )
    expected = [
        (outer, seed, method)
        for outer in FIXED_OUTER_FOLDS
        for seed in FIXED_BASE_SEEDS
        for method in PRIMARY_METHODS
    ]
    validated = [
        _validate_job(
            job,
            expected_outer_fold=outer,
            expected_seed=seed,
            expected_method=method,
        )
        for job, (outer, seed, method) in zip(jobs, expected, strict=True)
    ]
    output_dirs = [item["output_dir"] for item in validated]
    if len(output_dirs) != len(set(output_dirs)):
        raise Stage2OrchestratorContractError("matrix output directories are not unique")
    if (
        payload["matrix_content_sha256_algorithm"]
        != "sha256-canonical-json-without-self-field-v1"
    ):
        raise Stage2OrchestratorContractError("matrix content algorithm mismatch")
    projection = dict(payload)
    digest = projection.pop("matrix_content_sha256")
    _strict_sha(digest, "matrix.matrix_content_sha256")
    if digest != canonical_json_sha256(projection):
        raise Stage2OrchestratorContractError("matrix content SHA-256 mismatch")


def make_stage2_execution_job(
    *,
    outer_fold_id: str,
    base_seed: int,
    method_id: str,
    argv: Sequence[str],
    input_bindings: Sequence[Mapping[str, str]],
    output_dir: str,
) -> dict[str, Any]:
    if outer_fold_id not in FIXED_OUTER_FOLDS:
        raise Stage2OrchestratorContractError("unknown outer fold")
    if base_seed not in FIXED_BASE_SEEDS or method_id not in PRIMARY_METHODS:
        raise Stage2OrchestratorContractError("job seed/method is outside the frozen grid")
    bindings = [
        _validate_binding(dict(item), f"input_bindings[{index}]")
        for index, item in enumerate(input_bindings)
    ]
    bindings.sort(key=lambda item: item["name"])
    _require_authoritative_stage2_config_binding(bindings, "input_bindings")
    _require_authoritative_stage2_config_arguments(argv, "argv")
    gpu_id = FIXED_GPU_BY_OUTER_FOLD[outer_fold_id]
    command_projection = {
        "argv": list(argv),
        "environment": {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        "input_bindings": bindings,
        "output_dir": output_dir,
    }
    return {
        "job_id": f"s2_dgo__{outer_fold_id}__s{base_seed}__{method_id.lower()}",
        "phase": "S2_DGO_PRIMARY",
        "outer_fold_id": outer_fold_id,
        "outer_target_domain": FIXED_OUTER_TARGETS[outer_fold_id],
        "base_seed": base_seed,
        "method_id": method_id,
        "gpu_id": gpu_id,
        "environment": {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        "argv": list(argv),
        "input_bindings": bindings,
        "input_identity_sha256": canonical_json_sha256(bindings),
        "command_sha256": canonical_json_sha256(command_projection),
        "output_dir": output_dir,
        "attempt": 1,
        "resume": None,
    }


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_stage2_execution_matrix(
    jobs: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[Path, str]:
    root = _repository_root(repository_root)
    raw = Path(output_path)
    path = raw if raw.is_absolute() else root / raw
    if ".." in raw.parts or path.is_symlink() or not path.parent.is_dir():
        raise Stage2OrchestratorContractError("invalid matrix output path")
    path = path.absolute()
    if not path.is_relative_to(root) or os.path.lexists(path):
        raise Stage2OrchestratorContractError(
            "matrix output must be a new repository file"
        )
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_MATRIX_SCHEMA,
        "artifact_type": MATRIX_ARTIFACT_TYPE,
        "artifact_status": "FROZEN_DEVELOPMENT_ONLY",
        "development_only": True,
        "official_test_accessed": False,
        "official_phase": None,
        "s2_dgo_status": "NOT_RUN",
        "contains_observed_results": False,
        "base_seeds": list(FIXED_BASE_SEEDS),
        "outer_folds": list(FIXED_OUTER_FOLDS),
        "methods": list(PRIMARY_METHODS),
        "gpu_mapping": dict(FIXED_GPU_BY_OUTER_FOLD),
        "jobs": [dict(job) for job in jobs],
        "matrix_content_sha256_algorithm": (
            "sha256-canonical-json-without-self-field-v1"
        ),
    }
    payload["matrix_content_sha256"] = canonical_json_sha256(payload)
    _validate_matrix_payload(payload)
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
    _write_exclusive(path, data)
    digest = hashlib.sha256(data).hexdigest()
    _write_exclusive(
        path.with_name(path.name + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return path, digest


def materialize_stage2_execution_matrix_from_spec(
    command_spec: str | Path,
    command_spec_sha256: str,
    output_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[Path, str]:
    """Materialize 18 exact jobs from one external-SHA metadata-only spec.

    Referenced input paths are intentionally not resolved here.  This keeps
    result-free materialization and subsequent dry-run at zero data access;
    real mode re-hashes every binding immediately before and after execution.
    """

    root = _repository_root(repository_root)
    spec_path = _input_file(command_spec, root, "execution command spec")
    spec_data, spec_digest = _stable_file_bytes(
        spec_path, "execution command spec"
    )
    if spec_digest != _strict_sha(
        command_spec_sha256, "execution command spec SHA-256"
    ):
        raise Stage2OrchestratorContractError(
            "execution command spec differs from external SHA-256"
        )
    payload = _parse_json_bytes(spec_data, "execution command spec")
    _exact_keys(payload, _SPEC_KEYS, "execution command spec")
    if (
        payload["schema_version"] != EXECUTION_COMMAND_SPEC_SCHEMA
        or payload["artifact_type"]
        != "rc_irstd_stage2_execution_command_spec"
        or payload["artifact_status"] != "RESULT_FREE_FROZEN_INPUTS"
    ):
        raise Stage2OrchestratorContractError(
            "execution command spec identity mismatch"
        )
    for key, expected in (
        ("development_only", True),
        ("official_test_accessed", False),
        ("contains_observed_results", False),
    ):
        _strict_bool(payload[key], expected, f"command spec.{key}")
    if payload["official_phase"] is not None:
        raise Stage2OrchestratorContractError(
            "execution command spec may not contain an official phase"
        )
    records = payload["records"]
    if not isinstance(records, list) or len(records) != 18:
        raise Stage2OrchestratorContractError(
            "command spec requires exactly 18 ordered T4/T8 records"
        )
    expected_grid = [
        (outer, seed, method)
        for outer in FIXED_OUTER_FOLDS
        for seed in FIXED_BASE_SEEDS
        for method in PRIMARY_METHODS
    ]
    jobs: list[dict[str, Any]] = []
    for index, (raw, (outer, seed, method)) in enumerate(
        zip(records, expected_grid, strict=True)
    ):
        record = _exact_keys(raw, _SPEC_RECORD_KEYS, f"spec.records[{index}]")
        if (
            record["outer_fold_id"],
            record["base_seed"],
            record["method_id"],
        ) != (outer, seed, method):
            raise Stage2OrchestratorContractError(
                "command spec record order/identity mismatch"
            )
        jobs.append(
            make_stage2_execution_job(
                outer_fold_id=outer,
                base_seed=seed,
                method_id=method,
                argv=record["argv"],
                input_bindings=record["input_bindings"],
                output_dir=record["output_dir"],
            )
        )
    if (
        payload["spec_content_sha256_algorithm"]
        != "sha256-canonical-json-without-self-field-v1"
    ):
        raise Stage2OrchestratorContractError(
            "execution command spec content algorithm mismatch"
        )
    projection = dict(payload)
    content_digest = projection.pop("spec_content_sha256")
    if content_digest != canonical_json_sha256(projection):
        raise Stage2OrchestratorContractError(
            "execution command spec content SHA-256 mismatch"
        )
    spec_after, digest_after = _stable_file_bytes(
        spec_path, "execution command spec"
    )
    if spec_after != spec_data or digest_after != spec_digest:
        raise Stage2OrchestratorContractError(
            "execution command spec changed during materialization"
        )
    return publish_stage2_execution_matrix(
        jobs, output_path, repository_root=root
    )


def verify_stage2_execution_matrix(
    path: str | Path,
    expected_sha256: str,
    *,
    repository_root: str | Path | None = None,
) -> VerifiedStage2ExecutionMatrix:
    root = _repository_root(repository_root)
    expected = _strict_sha(expected_sha256, "expected matrix SHA-256")
    resolved = _input_file(path, root, "execution matrix")
    data, observed = _stable_file_bytes(resolved, "execution matrix")
    if observed != expected:
        raise Stage2OrchestratorContractError(
            "execution matrix SHA-256 differs from external expectation"
        )
    payload = _parse_json_bytes(data, "execution matrix")
    _validate_matrix_payload(payload)
    data_after, observed_after = _stable_file_bytes(resolved, "execution matrix")
    if data_after != data or observed_after != observed:
        raise Stage2OrchestratorContractError(
            "execution matrix changed during verification"
        )
    return VerifiedStage2ExecutionMatrix(
        path=resolved,
        sha256=observed,
        payload=MappingProxyType(dict(payload)),
        _capability=_VERIFIED_MATRIX_CAPABILITY,
    )


def render_stage2_dry_run(
    matrix: VerifiedStage2ExecutionMatrix,
    *,
    job_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    verified = assert_verified_execution_matrix(matrix)
    selected = set(job_ids) if job_ids is not None else None
    known = {str(job["job_id"]) for job in verified.payload["jobs"]}
    if selected is not None and (not selected or not selected <= known):
        raise Stage2OrchestratorContractError("dry-run job selection is invalid")
    records: list[str] = []
    for job in verified.payload["jobs"]:
        if selected is not None and job["job_id"] not in selected:
            continue
        records.append(
            _canonical_bytes(
                {
                    "argv": job["argv"],
                    "command_sha256": job["command_sha256"],
                    "environment": job["environment"],
                    "input_bindings": job["input_bindings"],
                    "input_identity_sha256": job["input_identity_sha256"],
                    "job_id": job["job_id"],
                    "output_dir": job["output_dir"],
                }
            ).decode("utf-8")
        )
    return tuple(records)


_LAUNCH_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "contains_observed_results",
        "execution_authorized",
        "authorized_phase",
        "authorized_methods",
        "allowed_physical_gpus",
        "s2_i0_report",
        "execution_matrix",
    }
)


def verify_stage2_development_launch_authorization(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_i0_sha256: str,
    expected_matrix_sha256: str,
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    root = _repository_root(repository_root)
    resolved = _input_file(path, root, "development launch authorization")
    data, digest = _stable_file_bytes(resolved, "development launch authorization")
    if digest != _strict_sha(expected_sha256, "launch authorization SHA-256"):
        raise Stage2OrchestratorContractError(
            "launch authorization SHA-256 mismatch"
        )
    payload = _parse_json_bytes(data, "development launch authorization")
    _exact_keys(payload, _LAUNCH_KEYS, "development launch authorization")
    if (
        payload["schema_version"] != DEVELOPMENT_LAUNCH_AUTHORIZATION_SCHEMA
        or payload["artifact_type"]
        != "rc_irstd_stage2_development_launch_authorization"
        or payload["artifact_status"] != "PASS"
    ):
        raise Stage2OrchestratorContractError(
            "development launch authorization identity mismatch"
        )
    for key, expected in (
        ("development_only", True),
        ("official_test_accessed", False),
        ("contains_observed_results", False),
        ("execution_authorized", True),
    ):
        _strict_bool(payload[key], expected, f"launch.{key}")
    if (
        payload["authorized_phase"] != "S2_DGO_PRIMARY"
        or payload["authorized_methods"] != list(PRIMARY_METHODS)
        or payload["allowed_physical_gpus"] != [0, 1, 2]
    ):
        raise Stage2OrchestratorContractError("launch authorization scope mismatch")
    for key, expected_digest in (
        ("s2_i0_report", expected_i0_sha256),
        ("execution_matrix", expected_matrix_sha256),
    ):
        binding = _exact_keys(
            payload[key], frozenset({"path", "sha256"}), f"launch.{key}"
        )
        _relative_path(binding["path"], f"launch.{key}.path")
        if binding["sha256"] != _strict_sha(
            expected_digest, f"expected {key} SHA-256"
        ):
            raise Stage2OrchestratorContractError(
                f"launch authorization {key} binding mismatch"
            )
    return MappingProxyType(dict(payload))


_RETRY_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "execution_authorized",
        "failure_class",
        "pre_result_failure",
        "observed_result_opened",
        "metrics_opened",
        "kind",
        "matrix_sha256",
        "job_id",
        "command_sha256",
        "input_identity_sha256",
        "attempt",
        "resume_binding",
        "resume_arguments",
    }
)


def verify_stage2_retry_authorization(
    path: str | Path,
    expected_sha256: str,
    *,
    matrix: VerifiedStage2ExecutionMatrix,
    job_id: str,
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    verified = assert_verified_execution_matrix(matrix)
    root = _repository_root(repository_root)
    resolved = _input_file(path, root, "retry authorization")
    data, digest = _stable_file_bytes(resolved, "retry authorization")
    if digest != _strict_sha(expected_sha256, "retry authorization SHA-256"):
        raise Stage2OrchestratorContractError("retry authorization SHA-256 mismatch")
    payload = _parse_json_bytes(data, "retry authorization")
    _exact_keys(payload, _RETRY_KEYS, "retry authorization")
    if (
        payload["schema_version"] != RETRY_AUTHORIZATION_SCHEMA
        or payload["artifact_type"] != "rc_irstd_stage2_retry_authorization"
        or payload["artifact_status"] != "PASS"
    ):
        raise Stage2OrchestratorContractError("retry authorization identity mismatch")
    for key, expected in (
        ("development_only", True),
        ("official_test_accessed", False),
        ("execution_authorized", True),
        ("pre_result_failure", True),
        ("observed_result_opened", False),
        ("metrics_opened", False),
    ):
        _strict_bool(payload[key], expected, f"retry.{key}")
    if payload["failure_class"] not in ALLOWED_FAILURE_CLASSES:
        raise Stage2OrchestratorContractError(
            "retry is limited to implementation/infrastructure failure"
        )
    jobs = {str(job["job_id"]): job for job in verified.payload["jobs"]}
    if job_id not in jobs or payload["job_id"] != job_id:
        raise Stage2OrchestratorContractError("retry job binding mismatch")
    job = jobs[job_id]
    if (
        payload["matrix_sha256"] != verified.sha256
        or payload["command_sha256"] != job["command_sha256"]
        or payload["input_identity_sha256"] != job["input_identity_sha256"]
    ):
        raise Stage2OrchestratorContractError(
            "retry changed frozen matrix/command/input hashes"
        )
    _strict_int(payload["attempt"], "retry.attempt", minimum=2)
    kind = payload["kind"]
    arguments = payload["resume_arguments"]
    if kind == "RETRY":
        if payload["resume_binding"] is not None or arguments != []:
            raise Stage2OrchestratorContractError(
                "plain retry may not add resume state"
            )
    elif kind == "RESUME":
        binding = _validate_binding(payload["resume_binding"], "retry.resume_binding")
        expected_arguments = [
            "--resume-commit",
            binding["path"],
            "--resume-commit-sha256",
            binding["sha256"],
        ]
        if arguments != expected_arguments:
            raise Stage2OrchestratorContractError(
                "resume arguments must bind the exact external commit SHA-256"
            )
    else:
        raise Stage2OrchestratorContractError("retry kind must be RETRY or RESUME")
    return MappingProxyType(dict(payload))


def _verify_runtime_binding(
    binding: Mapping[str, Any], root: Path
) -> tuple[Path, str]:
    path = _input_file(binding["path"], root, f"runtime input {binding['name']}")
    _, observed = _stable_file_bytes(path, f"runtime input {binding['name']}")
    if observed != binding["sha256"]:
        raise Stage2OrchestratorContractError(
            f"runtime input SHA-256 mismatch: {binding['name']}"
        )
    return path, observed


def orchestrate_stage2_crossfit(
    *,
    mode: str,
    matrix_path: str | Path,
    matrix_sha256: str,
    s2_i0_report: str | Path | None = None,
    s2_i0_report_sha256: str | None = None,
    launch_authorization: str | Path | None = None,
    launch_authorization_sha256: str | None = None,
    retry_authorization: str | Path | None = None,
    retry_authorization_sha256: str | None = None,
    job_ids: Sequence[str] | None = None,
    repository_root: str | Path | None = None,
    runner: Callable[..., Any] | None = None,
) -> Mapping[str, Any]:
    root = _repository_root(repository_root)
    matrix = verify_stage2_execution_matrix(
        matrix_path, matrix_sha256, repository_root=root
    )
    if mode == "dry-run":
        if any(
            value is not None
            for value in (
                s2_i0_report,
                s2_i0_report_sha256,
                launch_authorization,
                launch_authorization_sha256,
                retry_authorization,
                retry_authorization_sha256,
            )
        ):
            raise Stage2OrchestratorContractError(
                "dry-run accepts no real-execution authorization artifacts"
            )
        commands = render_stage2_dry_run(matrix, job_ids=job_ids)
        return MappingProxyType(
            {
                "mode": "dry-run",
                "matrix_sha256": matrix.sha256,
                "commands": commands,
                "data_paths_opened": 0,
                "official_test_accessed": False,
                "gpu_jobs_started": 0,
            }
        )
    if mode != "real":
        raise Stage2OrchestratorContractError("mode must be dry-run or real")
    if (
        s2_i0_report is None
        or s2_i0_report_sha256 is None
        or launch_authorization is None
        or launch_authorization_sha256 is None
    ):
        raise Stage2OrchestratorContractError(
            "real mode requires external PASS S2_I0 and launch hashes"
        )
    from outputs.audit_tools.audit_stage2_i0 import verify_stage2_i0_report

    verified_i0 = verify_stage2_i0_report(
        s2_i0_report,
        s2_i0_report_sha256,
        require_pass=True,
        repository_root=root,
    )
    verify_stage2_development_launch_authorization(
        launch_authorization,
        launch_authorization_sha256,
        expected_i0_sha256=verified_i0.sha256,
        expected_matrix_sha256=matrix.sha256,
        repository_root=root,
    )
    jobs_by_id = {str(job["job_id"]): job for job in matrix.payload["jobs"]}
    selected_ids = list(job_ids) if job_ids is not None else list(jobs_by_id)
    if (
        not selected_ids
        or len(selected_ids) != len(set(selected_ids))
        or any(job_id not in jobs_by_id for job_id in selected_ids)
    ):
        raise Stage2OrchestratorContractError("real job selection is invalid")
    retry_payload: Mapping[str, Any] | None = None
    if (retry_authorization is None) != (retry_authorization_sha256 is None):
        raise Stage2OrchestratorContractError(
            "retry authorization path and external SHA-256 are jointly required"
        )
    if retry_authorization is not None:
        if len(selected_ids) != 1:
            raise Stage2OrchestratorContractError(
                "one retry authorization may cover exactly one selected job"
            )
        retry_payload = verify_stage2_retry_authorization(
            retry_authorization,
            retry_authorization_sha256,
            matrix=matrix,
            job_id=selected_ids[0],
            repository_root=root,
        )

    run = runner if runner is not None else subprocess.run

    def run_one(job_id: str) -> dict[str, Any]:
        job = jobs_by_id[job_id]
        before = {
            binding["name"]: _verify_runtime_binding(binding, root)[1]
            for binding in job["input_bindings"]
        }
        argv = list(job["argv"])
        if retry_payload is not None and retry_payload["kind"] == "RESUME":
            resume_binding = _validate_binding(
                retry_payload["resume_binding"], "retry.resume_binding"
            )
            _verify_runtime_binding(resume_binding, root)
            argv.extend(retry_payload["resume_arguments"])
        output = root / job["output_dir"]
        if os.path.lexists(output) and retry_payload is None:
            raise Stage2OrchestratorContractError(
                f"fresh job output already exists: {job['output_dir']}"
            )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu_id"])
        result = run(
            argv,
            cwd=root,
            env=environment,
            check=False,
        )
        returncode = int(getattr(result, "returncode", 0))
        after = {
            binding["name"]: _verify_runtime_binding(binding, root)[1]
            for binding in job["input_bindings"]
        }
        if before != after:
            raise Stage2OrchestratorContractError(
                f"job input changed during execution: {job_id}"
            )
        if returncode != 0:
            raise Stage2OrchestratorContractError(
                f"job failed without automatic retry: {job_id} exit={returncode}"
            )
        return {
            "job_id": job_id,
            "gpu_id": job["gpu_id"],
            "command_sha256": job["command_sha256"],
            "input_identity_sha256": job["input_identity_sha256"],
            "returncode": returncode,
        }

    if len(selected_ids) == 1:
        completed = [run_one(selected_ids[0])]
    else:
        queues = {
            gpu_id: [
                job_id
                for job_id in selected_ids
                if jobs_by_id[job_id]["gpu_id"] == gpu_id
            ]
            for gpu_id in (0, 1, 2)
        }
        queues = {gpu_id: queue for gpu_id, queue in queues.items() if queue}
        stop_unstarted = Event()

        def run_gpu_queue(queue: Sequence[str]) -> list[dict[str, Any]]:
            queue_results: list[dict[str, Any]] = []
            for queued_job_id in queue:
                if stop_unstarted.is_set():
                    break
                try:
                    queue_results.append(run_one(queued_job_id))
                except BaseException:
                    stop_unstarted.set()
                    raise
            return queue_results

        completed_by_id: dict[str, dict[str, Any]] = {}
        first_failure: BaseException | None = None
        with ThreadPoolExecutor(
            max_workers=len(queues), thread_name_prefix="stage2-gpu"
        ) as pool:
            futures = [pool.submit(run_gpu_queue, queue) for queue in queues.values()]
            for future in futures:
                try:
                    for item in future.result():
                        completed_by_id[item["job_id"]] = item
                except BaseException as error:
                    stop_unstarted.set()
                    if first_failure is None:
                        first_failure = error
        if first_failure is not None:
            raise first_failure
        if set(completed_by_id) != set(selected_ids):
            raise Stage2OrchestratorContractError(
                "execution was cancelled before all selected jobs started"
            )
        completed = [completed_by_id[job_id] for job_id in selected_ids]
    return MappingProxyType(
        {
            "mode": "real",
            "matrix_sha256": matrix.sha256,
            "s2_i0_report_sha256": verified_i0.sha256,
            "completed": tuple(completed),
            "official_test_accessed": False,
            "official_phase_present": False,
        }
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("materialize", "dry-run", "real"), required=True
    )
    parser.add_argument("--matrix")
    parser.add_argument("--matrix-sha256")
    parser.add_argument("--command-spec")
    parser.add_argument("--command-spec-sha256")
    parser.add_argument("--output-matrix")
    parser.add_argument("--s2-i0-report")
    parser.add_argument("--s2-i0-report-sha256")
    parser.add_argument("--launch-authorization")
    parser.add_argument("--launch-authorization-sha256")
    parser.add_argument("--retry-authorization")
    parser.add_argument("--retry-authorization-sha256")
    parser.add_argument("--job-id", action="append")
    arguments = parser.parse_args(argv)
    if arguments.mode == "materialize":
        if any(
            value is None
            for value in (
                arguments.command_spec,
                arguments.command_spec_sha256,
                arguments.output_matrix,
            )
        ):
            parser.error(
                "materialize requires --command-spec, --command-spec-sha256 "
                "and --output-matrix"
            )
        if arguments.matrix is not None or arguments.matrix_sha256 is not None:
            parser.error("materialize does not consume an existing matrix")
    elif arguments.matrix is None or arguments.matrix_sha256 is None:
        parser.error("dry-run/real require --matrix and --matrix-sha256")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.mode == "materialize":
        path, digest = materialize_stage2_execution_matrix_from_spec(
            arguments.command_spec,
            arguments.command_spec_sha256,
            arguments.output_matrix,
        )
        print(
            json.dumps(
                {
                    "mode": "materialize",
                    "matrix": str(path),
                    "matrix_sha256": digest,
                    "job_count": 18,
                    "data_paths_opened": 0,
                    "official_test_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    result = orchestrate_stage2_crossfit(
        mode=arguments.mode,
        matrix_path=arguments.matrix,
        matrix_sha256=arguments.matrix_sha256,
        s2_i0_report=arguments.s2_i0_report,
        s2_i0_report_sha256=arguments.s2_i0_report_sha256,
        launch_authorization=arguments.launch_authorization,
        launch_authorization_sha256=arguments.launch_authorization_sha256,
        retry_authorization=arguments.retry_authorization,
        retry_authorization_sha256=arguments.retry_authorization_sha256,
        job_ids=arguments.job_id,
    )
    if arguments.mode == "dry-run":
        for command in result["commands"]:
            print(command)
    else:
        print(
            json.dumps(
                {
                    "mode": "real",
                    "matrix_sha256": result["matrix_sha256"],
                    "s2_i0_report_sha256": result["s2_i0_report_sha256"],
                    "completed_job_count": len(result["completed"]),
                    "official_test_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


__all__ = [
    "ALLOWED_FAILURE_CLASSES",
    "AUTHORITATIVE_STAGE2_CONFIG_NAME",
    "AUTHORITATIVE_STAGE2_CONFIG_PATH",
    "AUTHORITATIVE_STAGE2_CONFIG_SHA256",
    "DEVELOPMENT_LAUNCH_AUTHORIZATION_SCHEMA",
    "EXECUTION_COMMAND_SPEC_SCHEMA",
    "EXECUTION_MATRIX_SCHEMA",
    "FIXED_BASE_SEEDS",
    "FIXED_GPU_BY_OUTER_FOLD",
    "FIXED_OUTER_FOLDS",
    "FIXED_OUTER_TARGETS",
    "LEGACY_STAGE2_ANALYSIS_PLAN_PATH",
    "PRIMARY_METHODS",
    "RETRY_AUTHORIZATION_SCHEMA",
    "Stage2OrchestratorContractError",
    "VerifiedStage2ExecutionMatrix",
    "assert_verified_execution_matrix",
    "canonical_json_sha256",
    "make_stage2_execution_job",
    "materialize_stage2_execution_matrix_from_spec",
    "orchestrate_stage2_crossfit",
    "publish_stage2_execution_matrix",
    "render_stage2_dry_run",
    "verify_stage2_development_launch_authorization",
    "verify_stage2_execution_matrix",
    "verify_stage2_retry_authorization",
]


if __name__ == "__main__":
    raise SystemExit(main())
