"""Additive, fail-closed Stage-2 detector RUN_COMPLETE-v2 evidence.

The result-free detector run contract and the runtime contract deliberately do
not claim that training finished.  This module adds that missing boundary
without changing either RC4-era v1 authority.  A RUN_COMPLETE-v2 capability is
issued only after replaying the verified run/runtime closure and proving that
the fixed-last epoch exists in the metrics log, the resumable checkpoint, and
the restricted inference checkpoint.

The adjacent ``RUN_COMPLETE.json.sha256`` file is the commit-last marker.  A
bare JSON file is therefore incomplete and never grants a capability.  Metric
values are checked for structural continuity but are not copied into the
completion artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import torch

from data_ext.dataset_identity import sha256_file
from data_ext.stage2_role_contract import (
    verify_stage2_run_contract_sidecar,
)


RUN_COMPLETE_SCHEMA_V2 = "rc-irstd.stage2-detector-run-complete.v2"
RUN_COMPLETE_ARTIFACT_TYPE = "rc_irstd_stage2_detector_run_complete"
RUN_COMPLETE_STATUS = "RUN_COMPLETE_FIXED_LAST_VERIFIED"
RUN_COMPLETE_NAME = "RUN_COMPLETE.json"
RUN_COMPLETE_SIDECAR_NAME = "RUN_COMPLETE.json.sha256"
FIXED_LAST_POLICY = "fixed_last_no_test_or_target_validation"
SEALED_PROTOCOL_SCOPE = "stage2_development_detector_official_test_sealed"
FULL_CHECKPOINT_FORMAT = "rc-irstd.detector.v2"
RESTRICTED_CHECKPOINT_FORMAT = "rc-irstd.detector-inference.v1"

EXTERNAL_HASH_KEYS = frozenset(
    {
        "run_contract_sha256",
        "runtime_contract_sha256",
        "run_config_sha256",
        "environment_sha256",
        "release_artifact_sha256",
        "release_source_archive_sha256",
        "metrics_sha256",
        "checkpoint_last_sha256",
        "restricted_inference_checkpoint_sha256",
        "weights_last_sha256",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CAPABILITY_TOKEN = object()


class Stage2DetectorRunCompleteV2Error(ValueError):
    """A Stage-2 detector completion claim failed closed."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


class VerifiedStage2DetectorRunCompleteV2:
    """Recursively immutable, public-verifier-issued completion capability."""

    __slots__ = (
        "artifact_path",
        "sha256",
        "run_dir",
        "run_contract_path",
        "payload",
        "external_hashes",
        "_verified_run_contract",
        "_verified_runtime_closure",
        "_capability",
    )

    def __init__(
        self,
        *,
        artifact_path: str,
        sha256: str,
        run_dir: str,
        run_contract_path: str,
        payload: Mapping[str, Any],
        external_hashes: Mapping[str, Any],
        verified_run_contract: Mapping[str, Any],
        verified_runtime_closure: Mapping[str, Any],
        _capability: object | None = None,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedStage2DetectorRunCompleteV2 is verifier-issued only"
            )
        for name, value in (
            ("artifact_path", artifact_path),
            ("sha256", sha256),
            ("run_dir", run_dir),
            ("run_contract_path", run_contract_path),
        ):
            object.__setattr__(self, name, str(value))
        object.__setattr__(self, "payload", _deep_freeze(payload))
        object.__setattr__(self, "external_hashes", _deep_freeze(external_hashes))
        object.__setattr__(
            self, "_verified_run_contract", _deep_freeze(verified_run_contract)
        )
        object.__setattr__(
            self, "_verified_runtime_closure", _deep_freeze(verified_runtime_closure)
        )
        object.__setattr__(self, "_capability", _CAPABILITY_TOKEN)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("verified RUN_COMPLETE-v2 capability is immutable")


def assert_verified_stage2_detector_run_complete_v2(
    value: Any,
) -> VerifiedStage2DetectorRunCompleteV2:
    if (
        type(value) is not VerifiedStage2DetectorRunCompleteV2
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
    ):
        raise TypeError(
            "a verifier-issued VerifiedStage2DetectorRunCompleteV2 is required"
        )
    return value


def _sha256(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Stage2DetectorRunCompleteV2Error(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _external_hashes(value: Mapping[str, Any]) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or set(value) != EXTERNAL_HASH_KEYS:
        raise Stage2DetectorRunCompleteV2Error(
            "external_hashes must have the exact RUN_COMPLETE-v2 key set"
        )
    result: dict[str, str | None] = {}
    for key in sorted(EXTERNAL_HASH_KEYS):
        result[key] = _sha256(
            value[key], key, optional=(key == "weights_last_sha256")
        )
    return result


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Stage2DetectorRunCompleteV2Error(
                f"{label} contains a symlink component: {cursor}"
            )


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    _reject_symlink_components(path, label)
    if not path.exists() or not stat.S_ISDIR(path.lstat().st_mode):
        raise FileNotFoundError(f"{label} is not a real directory: {path}")
    return path.resolve(strict=True)


def _regular_file(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    if not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    return path.resolve(strict=True)


def _run_file(run_dir: Path, name: str, label: str) -> Path:
    path = _regular_file(run_dir / name, label)
    if path.parent != run_dir:
        raise Stage2DetectorRunCompleteV2Error(f"{label} escapes run directory")
    return path


def _stable_bytes(path: Path, label: str) -> tuple[bytes, str]:
    before = sha256_file(path)
    payload = path.read_bytes()
    after = sha256_file(path)
    if before != after or hashlib.sha256(payload).hexdigest() != before:
        raise RuntimeError(f"{label} changed while read")
    return payload, before


def _json_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    raw, digest = _stable_bytes(path, label)
    try:
        text = raw.decode("utf-8", errors="strict")

        def reject_constant(value: str) -> None:
            raise Stage2DetectorRunCompleteV2Error(
                f"{label} contains non-finite JSON constant {value}"
            )

        payload = json.loads(text, parse_constant=reject_constant)
    except UnicodeDecodeError as error:
        raise Stage2DetectorRunCompleteV2Error(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise Stage2DetectorRunCompleteV2Error(f"{label} is invalid JSON") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    _reject_nonfinite(payload, label)
    return payload, digest


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Stage2DetectorRunCompleteV2Error(f"{label} contains non-finite value")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{label}[{index}]")


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_false(value: Any, label: str) -> None:
    if type(value) is not bool or value is not False:
        raise TypeError(f"{label} must be exact false")


def _verify_sha256sum_sidecar(
    artifact: Path,
    sidecar: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, str]:
    artifact = _regular_file(artifact, label)
    sidecar = _regular_file(sidecar, f"{label} SHA-256 sidecar")
    _, actual = _stable_bytes(artifact, label)
    if actual != expected_sha256:
        raise Stage2DetectorRunCompleteV2Error(f"{label} external SHA-256 mismatch")
    sidecar_bytes, _ = _stable_bytes(sidecar, f"{label} SHA-256 sidecar")
    expected = f"{actual}  {artifact.name}\n".encode("utf-8")
    if sidecar_bytes != expected:
        raise Stage2DetectorRunCompleteV2Error(f"{label} SHA-256 sidecar mismatch")
    return {"path": artifact.name, "sha256": actual, "sidecar": sidecar.name}


def _plain(value: Any) -> Any:
    return _deep_thaw(value)


def _identity(run: Mapping[str, Any]) -> dict[str, Any]:
    sources = run.get("source_domains")
    if not isinstance(sources, list) or len(sources) != 2 or len(set(sources)) != 2:
        raise Stage2DetectorRunCompleteV2Error("run source_domains must be exact two")
    target = run.get("outer_target_domain")
    if target in sources:
        raise Stage2DetectorRunCompleteV2Error(
            "outer target appears in detector source selection"
        )
    return {
        "run_id": run["run_id"],
        "outer_fold_id": run["outer_fold_id"],
        "outer_target_domain": target,
        "source_domains": list(sources),
        "base_seed": run["base_seed"],
        "derived_seed": run["derived_seed"],
        "detector_role": run["detector_role"],
        "oof_fold_index": run["oof_fold_index"],
    }


def _verify_config(
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    run_sha256: str,
    environment: Mapping[str, Any],
) -> int:
    target_epochs = _exact_int(config.get("epochs"), "config.epochs", minimum=1)
    expected = {
        "seed": run["derived_seed"],
        "source_names": run["source_domains"],
        "outer_fold_id": run["outer_fold_id"],
        "outer_target": run["outer_target_domain"],
        "held_out_domains": [run["outer_target_domain"]],
        "stage2_detector_role": run["detector_role"],
        "stage2_oof_fold_index": run["oof_fold_index"],
        "checkpoint_selection": FIXED_LAST_POLICY,
        "protocol_scope": SEALED_PROTOCOL_SCOPE,
        "risk_objective": run["training"]["risk_objective"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise Stage2DetectorRunCompleteV2Error(
                f"run config identity/policy mismatch: {key}"
            )
    if config.get("execution_fingerprint") != environment:
        raise Stage2DetectorRunCompleteV2Error(
            "run config/environment provenance mismatch"
        )
    if config.get("engineering_smoke") not in (None, False):
        raise Stage2DetectorRunCompleteV2Error(
            "engineering-smoke run cannot acquire RUN_COMPLETE-v2"
        )
    if config.get("aaai27_pilot") not in (None, False):
        raise Stage2DetectorRunCompleteV2Error(
            "Stage1 pilot semantics cannot acquire Stage2 RUN_COMPLETE-v2"
        )
    input_run = config.get("stage2_input_run_contract")
    if not isinstance(input_run, Mapping):
        raise Stage2DetectorRunCompleteV2Error(
            "run config lacks Stage2 input-run binding"
        )
    if input_run.get("sha256") != run_sha256:
        raise Stage2DetectorRunCompleteV2Error(
            "run config input-run SHA-256 mismatch"
        )
    if input_run.get("run_id") != run["run_id"]:
        raise Stage2DetectorRunCompleteV2Error("run config run_id mismatch")
    _exact_false(
        input_run.get("official_test_accessed"),
        "config.stage2_input_run_contract.official_test_accessed",
    )
    if input_run.get("bindings") != run.get("bindings"):
        raise Stage2DetectorRunCompleteV2Error(
            "run config input-run provenance bindings mismatch"
        )
    return target_epochs


def _metrics_rows(
    metrics_path: Path,
    metrics_sha256: str,
    target_epochs: int,
    run: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    binding = _verify_sha256sum_sidecar(
        metrics_path,
        metrics_path.with_suffix(metrics_path.suffix + ".sha256"),
        metrics_sha256,
        "Stage2 metrics",
    )
    raw, _ = _stable_bytes(metrics_path, "Stage2 metrics")
    if not raw or not raw.endswith(b"\n"):
        raise Stage2DetectorRunCompleteV2Error(
            "metrics.jsonl must be non-empty and newline-terminated"
        )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise Stage2DetectorRunCompleteV2Error("metrics.jsonl is not UTF-8") from error
    lines = text.splitlines()
    if len(lines) != target_epochs or any(not line for line in lines):
        raise Stage2DetectorRunCompleteV2Error(
            "metrics.jsonl is missing, truncated, blank, or has extra epochs"
        )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    Stage2DetectorRunCompleteV2Error(
                        f"metrics[{index}] contains non-finite constant {value}"
                    )
                ),
            )
        except json.JSONDecodeError as error:
            raise Stage2DetectorRunCompleteV2Error(
                f"metrics[{index}] is invalid JSON"
            ) from error
        if not isinstance(row, dict):
            raise TypeError(f"metrics[{index}] must be a JSON object")
        _reject_nonfinite(row, f"metrics[{index}]")
        if _exact_int(row.get("epoch"), f"metrics[{index}].epoch") != index:
            raise Stage2DetectorRunCompleteV2Error(
                "metrics epochs must be the exact continuous range 0..epochs-1"
            )
        if row.get("checkpoint_selection") != FIXED_LAST_POLICY:
            raise Stage2DetectorRunCompleteV2Error(
                f"metrics[{index}] is not fixed-last policy"
            )
        if row.get("protocol_scope") != SEALED_PROTOCOL_SCOPE:
            raise Stage2DetectorRunCompleteV2Error(
                f"metrics[{index}] is not official-test sealed"
            )
        if row.get("risk_objective") != run["training"]["risk_objective"]:
            raise Stage2DetectorRunCompleteV2Error(
                f"metrics[{index}] detector objective mismatch"
            )
        if row.get("stage1_variant") != run["training"]["stage1_variant"]:
            raise Stage2DetectorRunCompleteV2Error(
                f"metrics[{index}] Stage1 variant mismatch"
            )
        rows.append(row)
    return rows, binding


def _stable_torch_load(path: Path, *, weights_only: bool, label: str) -> Mapping[str, Any]:
    before = sha256_file(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=weights_only)
    except Exception as error:
        mode = "weights_only=True" if weights_only else "training-checkpoint loader"
        raise Stage2DetectorRunCompleteV2Error(
            f"{label} cannot be loaded with {mode}"
        ) from error
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"{label} changed while loaded")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must contain a mapping")
    return payload


def _state_dict(value: Any, label: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} must be a non-empty tensor mapping")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} must map non-empty strings to tensors")
        if tensor.layout != torch.strided:
            raise TypeError(f"{label}[{key!r}] must be a strided tensor")
        result[key] = tensor.detach().cpu().contiguous()
    return result


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(b"rc-irstd.detector-state-dict.v1\0")
    for key in sorted(state):
        tensor = state[key]
        key_bytes = key.encode("utf-8")
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        data = _tensor_bytes(tensor)
        for item in (key_bytes, metadata, data):
            digest.update(len(item).to_bytes(8, "big"))
            digest.update(item)
    return digest.hexdigest()


def _assert_same_state_dict(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    if set(left) != set(right):
        raise Stage2DetectorRunCompleteV2Error(f"{label} tensor keys mismatch")
    for key in sorted(left):
        a, b = left[key], right[key]
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape) or not torch.equal(a, b):
            raise Stage2DetectorRunCompleteV2Error(
                f"{label} tensor mismatch: {key}"
            )


def _verify_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    run_sha256: str,
    config_sha256: str,
    runtime_closure: Mapping[str, Any],
    expected_epoch: int,
    expected_format: str,
    label: str,
) -> dict[str, torch.Tensor]:
    if checkpoint.get("format_version") != expected_format:
        raise Stage2DetectorRunCompleteV2Error(f"{label} format mismatch")
    if _exact_int(checkpoint.get("epoch"), f"{label}.epoch") != expected_epoch:
        raise Stage2DetectorRunCompleteV2Error(f"{label} fixed-last epoch mismatch")
    expected = {
        "seed": run["derived_seed"],
        "source_names": run["source_domains"],
        "outer_fold_id": run["outer_fold_id"],
        "outer_target": run["outer_target_domain"],
        "held_out_domains": [run["outer_target_domain"]],
        "detector_role": run["detector_role"],
        "oof_fold_index": run["oof_fold_index"],
        "checkpoint_selection": FIXED_LAST_POLICY,
        "run_contract_sha256": run_sha256,
        "run_config_sha256": config_sha256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise Stage2DetectorRunCompleteV2Error(
                f"{label} identity/provenance mismatch: {key}"
            )
    _exact_false(checkpoint.get("official_test_accessed"), f"{label}.official_test_accessed")
    if checkpoint.get("stage2_runtime_artifacts") != runtime_closure:
        raise Stage2DetectorRunCompleteV2Error(
            f"{label} runtime provenance closure mismatch"
        )
    return _state_dict(checkpoint.get("state_dict"), f"{label}.state_dict")


def _verify_inputs(
    *,
    run_dir: str | Path,
    run_contract_path: str | Path,
    verified_run_contract: Mapping[str, Any],
    verified_runtime_closure: Mapping[str, Any],
    external_hashes: Mapping[str, Any],
) -> tuple[
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, str | None],
    int,
]:
    hashes = _external_hashes(external_hashes)
    root = _directory(run_dir, "Stage2 detector run directory")
    contract_path = _regular_file(
        Path(run_contract_path).expanduser(), "Stage2 detector run contract"
    )
    replayed_run, run_sha = verify_stage2_run_contract_sidecar(contract_path)
    if run_sha != hashes["run_contract_sha256"]:
        raise Stage2DetectorRunCompleteV2Error(
            "run-contract external SHA-256 mismatch"
        )
    if not isinstance(verified_run_contract, Mapping) or _plain(
        verified_run_contract
    ) != replayed_run:
        raise Stage2DetectorRunCompleteV2Error(
            "supplied verified run contract differs from public-verifier replay"
        )

    if not isinstance(verified_runtime_closure, Mapping):
        raise TypeError("verified_runtime_closure must be a mapping")
    supplied_closure = _plain(verified_runtime_closure)
    required_closure = {
        "input_run_contract",
        "run_config",
        "environment_artifact",
        "runtime_contract",
        "release_artifact",
    }
    if set(supplied_closure) != required_closure:
        raise Stage2DetectorRunCompleteV2Error(
            "verified runtime closure has the wrong key set"
        )
    direct_hashes = {
        "input_run_contract": "run_contract_sha256",
        "run_config": "run_config_sha256",
        "environment_artifact": "environment_sha256",
        "runtime_contract": "runtime_contract_sha256",
    }
    for closure_name, hash_name in direct_hashes.items():
        binding = supplied_closure.get(closure_name)
        if not isinstance(binding, Mapping) or binding.get("sha256") != hashes[hash_name]:
            raise Stage2DetectorRunCompleteV2Error(
                f"runtime closure/external hash mismatch: {closure_name}"
            )
    release = supplied_closure.get("release_artifact")
    if not isinstance(release, Mapping) or release.get("sha256") != hashes[
        "release_artifact_sha256"
    ]:
        raise Stage2DetectorRunCompleteV2Error(
            "runtime release-artifact external hash mismatch"
        )
    archive = release.get("source_archive")
    if not isinstance(archive, Mapping) or archive.get("sha256") != hashes[
        "release_source_archive_sha256"
    ]:
        raise Stage2DetectorRunCompleteV2Error(
            "runtime release source-archive external hash mismatch"
        )

    config_path = _run_file(root, "config.json", "Stage2 run config")
    environment_path = _run_file(root, "environment.json", "Stage2 environment")
    config, config_sha = _json_object(config_path, "Stage2 run config")
    environment, environment_sha = _json_object(
        environment_path, "Stage2 environment"
    )
    if config_sha != hashes["run_config_sha256"]:
        raise Stage2DetectorRunCompleteV2Error("run config external hash mismatch")
    if environment_sha != hashes["environment_sha256"]:
        raise Stage2DetectorRunCompleteV2Error("environment external hash mismatch")

    # Reuse the legacy runtime verifier as an input authority; RUN_COMPLETE-v2
    # adds completion evidence but never replaces or weakens runtime-contract v1.
    from scripts.train_multisource_tail import verify_stage2_runtime_artifacts

    replayed_runtime = verify_stage2_runtime_artifacts(
        root,
        replayed_run,
        run_sha,
        config_sha,
        environment,
        input_run_contract_path=contract_path,
    )
    if replayed_runtime != supplied_closure:
        raise Stage2DetectorRunCompleteV2Error(
            "supplied verified runtime closure differs from verifier replay"
        )
    target_epochs = _verify_config(config, replayed_run, run_sha, environment)
    return (
        root,
        contract_path,
        replayed_run,
        replayed_runtime,
        environment,
        hashes,
        target_epochs,
    )


def _build_expected_payload(
    *,
    run_dir: str | Path,
    run_contract_path: str | Path,
    verified_run_contract: Mapping[str, Any],
    verified_runtime_closure: Mapping[str, Any],
    external_hashes: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path, dict[str, Any], dict[str, Any], dict[str, str | None]]:
    (
        root,
        contract_path,
        run,
        runtime,
        environment,
        hashes,
        target_epochs,
    ) = _verify_inputs(
        run_dir=run_dir,
        run_contract_path=run_contract_path,
        verified_run_contract=verified_run_contract,
        verified_runtime_closure=verified_runtime_closure,
        external_hashes=external_hashes,
    )
    expected_epoch = target_epochs - 1

    metrics_path = _run_file(root, "metrics.jsonl", "Stage2 metrics")
    metrics, metrics_binding = _metrics_rows(
        metrics_path,
        str(hashes["metrics_sha256"]),
        target_epochs,
        run,
    )

    checkpoint_path = _run_file(root, "checkpoint_last.pt", "Stage2 training checkpoint")
    checkpoint_binding = _verify_sha256sum_sidecar(
        checkpoint_path,
        _run_file(root, "checkpoint_sha256.txt", "Stage2 training checkpoint sidecar"),
        str(hashes["checkpoint_last_sha256"]),
        "Stage2 training checkpoint",
    )
    full_checkpoint = _stable_torch_load(
        checkpoint_path, weights_only=False, label="Stage2 training checkpoint"
    )
    full_state = _verify_checkpoint_identity(
        full_checkpoint,
        run=run,
        run_sha256=str(hashes["run_contract_sha256"]),
        config_sha256=str(hashes["run_config_sha256"]),
        runtime_closure=runtime,
        expected_epoch=expected_epoch,
        expected_format=FULL_CHECKPOINT_FORMAT,
        label="training checkpoint",
    )
    if full_checkpoint.get("detector_source_domains") != run["source_domains"]:
        raise Stage2DetectorRunCompleteV2Error(
            "training checkpoint detector source-domain identity mismatch"
        )
    if full_checkpoint.get("protocol_scope") != SEALED_PROTOCOL_SCOPE:
        raise Stage2DetectorRunCompleteV2Error(
            "training checkpoint is not official-test sealed"
        )
    if full_checkpoint.get("risk_objective") != run["training"]["risk_objective"]:
        raise Stage2DetectorRunCompleteV2Error(
            "training checkpoint detector objective mismatch"
        )
    if full_checkpoint.get("execution_fingerprint") != environment:
        raise Stage2DetectorRunCompleteV2Error(
            "training checkpoint environment provenance mismatch"
        )
    training_args = full_checkpoint.get("training_args")
    if not isinstance(training_args, Mapping) or training_args.get("epochs") != target_epochs:
        raise Stage2DetectorRunCompleteV2Error(
            "training checkpoint target epochs mismatch"
        )
    if full_checkpoint.get("epoch_metrics") != metrics[-1]:
        raise Stage2DetectorRunCompleteV2Error(
            "training checkpoint final metrics row mismatch"
        )

    restricted_path = _run_file(
        root,
        "stage2_inference_checkpoint.pt",
        "restricted Stage2 inference checkpoint",
    )
    restricted_binding = _verify_sha256sum_sidecar(
        restricted_path,
        _run_file(
            root,
            "stage2_inference_checkpoint.pt.sha256",
            "restricted Stage2 inference checkpoint sidecar",
        ),
        str(hashes["restricted_inference_checkpoint_sha256"]),
        "restricted Stage2 inference checkpoint",
    )
    restricted = _stable_torch_load(
        restricted_path,
        weights_only=True,
        label="restricted Stage2 inference checkpoint",
    )
    restricted_state = _verify_checkpoint_identity(
        restricted,
        run=run,
        run_sha256=str(hashes["run_contract_sha256"]),
        config_sha256=str(hashes["run_config_sha256"]),
        runtime_closure=runtime,
        expected_epoch=expected_epoch,
        expected_format=RESTRICTED_CHECKPOINT_FORMAT,
        label="restricted inference checkpoint",
    )
    if "optimizer" in restricted or "rng_state" in restricted:
        raise Stage2DetectorRunCompleteV2Error(
            "restricted inference checkpoint contains training-only state"
        )
    _assert_same_state_dict(
        full_state,
        restricted_state,
        "training/restricted fixed-last state_dict",
    )
    state_digest = _state_dict_digest(restricted_state)

    weights_binding: dict[str, Any] | None = None
    weights_sha = hashes["weights_last_sha256"]
    if weights_sha is not None:
        weights_path = _run_file(root, "weights_last.pt", "optional weights_last")
        _, actual_weights_sha = _stable_bytes(weights_path, "optional weights_last")
        if actual_weights_sha != weights_sha:
            raise Stage2DetectorRunCompleteV2Error(
                "optional weights_last external SHA-256 mismatch"
            )
        weights_payload = _stable_torch_load(
            weights_path, weights_only=True, label="optional weights_last"
        )
        weights_state = _state_dict(weights_payload, "optional weights_last")
        _assert_same_state_dict(
            restricted_state, weights_state, "restricted/optional weights_last state_dict"
        )
        weights_binding = {
            "path": "weights_last.pt",
            "sha256": weights_sha,
            "integrity_only": True,
            "downstream_authority": False,
        }

    run_binding = runtime["input_run_contract"]
    payload: dict[str, Any] = {
        "schema_version": RUN_COMPLETE_SCHEMA_V2,
        "artifact_type": RUN_COMPLETE_ARTIFACT_TYPE,
        "artifact_status": RUN_COMPLETE_STATUS,
        "development_only": True,
        "claim_bearing": False,
        "official_test_accessed": False,
        "observed_results": None,
        "checkpoint_selection": FIXED_LAST_POLICY,
        "completion_policy": "config_epochs_equals_continuous_metrics_and_both_fixed_last_checkpoints_v2",
        "run_identity": _identity(run),
        "target_epochs": target_epochs,
        "completed_epoch": expected_epoch,
        "metrics_epoch_range": {
            "first": 0,
            "last": expected_epoch,
            "count": target_epochs,
            "continuous": True,
        },
        "state_dict_content_sha256": state_digest,
        "bindings": {
            "input_run_contract": dict(run_binding),
            "runtime_contract": dict(runtime["runtime_contract"]),
            "run_config": dict(runtime["run_config"]),
            "environment_artifact": dict(runtime["environment_artifact"]),
            "release_artifact": _plain(runtime["release_artifact"]),
            "metrics": metrics_binding,
            "checkpoint_last": checkpoint_binding,
            "restricted_inference_checkpoint": restricted_binding,
            "weights_last": weights_binding,
        },
        "invariants": {
            "fixed_last_only": True,
            "metrics_values_embedded": False,
            "metrics_contiguous_from_zero": True,
            "restricted_checkpoint_weights_only_loaded": True,
            "full_and_restricted_state_dict_equal": True,
            "outer_target_excluded_from_detector_sources": True,
            "official_test_selection_forbidden": True,
            "runtime_config_environment_release_replayed": True,
            "weights_last_is_downstream_authority": False,
            "commit_last_marker": RUN_COMPLETE_SIDECAR_NAME,
        },
    }
    return payload, root, contract_path, run, runtime, hashes


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_stage2_detector_run_complete_v2(
    *,
    run_dir: str | Path,
    run_contract_path: str | Path,
    verified_run_contract: Mapping[str, Any],
    verified_runtime_closure: Mapping[str, Any],
    external_hashes: Mapping[str, Any],
) -> VerifiedStage2DetectorRunCompleteV2:
    """Verify a completed fixed-last run and publish the commit-last artifact."""

    payload, root, contract_path, run, runtime, hashes = _build_expected_payload(
        run_dir=run_dir,
        run_contract_path=run_contract_path,
        verified_run_contract=verified_run_contract,
        verified_runtime_closure=verified_runtime_closure,
        external_hashes=external_hashes,
    )
    artifact = root / RUN_COMPLETE_NAME
    sidecar = root / RUN_COMPLETE_SIDECAR_NAME
    _reject_symlink_components(artifact, "RUN_COMPLETE-v2 artifact")
    _reject_symlink_components(sidecar, "RUN_COMPLETE-v2 commit-last sidecar")
    canonical = _canonical_json(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    sidecar_bytes = f"{digest}  {RUN_COMPLETE_NAME}\n".encode("utf-8")

    if sidecar.exists() and not artifact.exists():
        raise Stage2DetectorRunCompleteV2Error(
            "orphan RUN_COMPLETE-v2 sidecar exists without artifact"
        )
    if artifact.exists():
        existing = _regular_file(artifact, "existing RUN_COMPLETE-v2 artifact")
        existing_bytes, _ = _stable_bytes(existing, "existing RUN_COMPLETE-v2 artifact")
        if existing_bytes != canonical:
            raise Stage2DetectorRunCompleteV2Error(
                "existing RUN_COMPLETE-v2 artifact differs; immutable overwrite refused"
            )
    else:
        _atomic_write(artifact, canonical)

    # Replay all mutable inputs immediately before publishing the only
    # authoritative commit marker.  A process or concurrent writer may have
    # changed metrics/checkpoints after the first payload construction.
    precommit_payload, precommit_root, precommit_contract, _, _, _ = (
        _build_expected_payload(
            run_dir=root,
            run_contract_path=contract_path,
            verified_run_contract=run,
            verified_runtime_closure=runtime,
            external_hashes=hashes,
        )
    )
    existing_bytes, _ = _stable_bytes(
        artifact, "precommit RUN_COMPLETE-v2 artifact"
    )
    if (
        precommit_payload != payload
        or precommit_root != root
        or precommit_contract != contract_path
        or existing_bytes != canonical
    ):
        raise Stage2DetectorRunCompleteV2Error(
            "RUN_COMPLETE-v2 inputs changed before commit"
        )

    # The sidecar is intentionally the last publication operation.  If a
    # process stops before this replace, the bare JSON file remains unusable.
    created_sidecar = False
    if sidecar.exists():
        existing_sidecar = _regular_file(sidecar, "existing RUN_COMPLETE-v2 sidecar")
        existing_bytes, _ = _stable_bytes(
            existing_sidecar, "existing RUN_COMPLETE-v2 sidecar"
        )
        if existing_bytes != sidecar_bytes:
            raise Stage2DetectorRunCompleteV2Error(
                "existing RUN_COMPLETE-v2 sidecar differs; immutable overwrite refused"
            )
    else:
        _atomic_write(sidecar, sidecar_bytes)
        created_sidecar = True

    try:
        return verify_stage2_detector_run_complete_v2(
            artifact,
            digest,
            run_dir=root,
            run_contract_path=contract_path,
            verified_run_contract=run,
            verified_runtime_closure=runtime,
            external_hashes=hashes,
        )
    except BaseException:
        # A final verifier failure must not leave a marker created by this
        # call. Never remove a pre-existing commit, and remove only exact
        # bytes at the exact non-symlink path we wrote.
        if created_sidecar and sidecar.exists() and not sidecar.is_symlink():
            try:
                current, _ = _stable_bytes(
                    sidecar, "failed RUN_COMPLETE-v2 commit-last sidecar"
                )
                if current == sidecar_bytes:
                    sidecar.unlink()
            except (OSError, RuntimeError, Stage2DetectorRunCompleteV2Error):
                pass
        raise


def verify_stage2_detector_run_complete_v2(
    artifact_path: str | Path,
    expected_sha256: str,
    *,
    run_dir: str | Path,
    run_contract_path: str | Path,
    verified_run_contract: Mapping[str, Any],
    verified_runtime_closure: Mapping[str, Any],
    external_hashes: Mapping[str, Any],
) -> VerifiedStage2DetectorRunCompleteV2:
    """Replay every bound artifact before issuing a completion capability."""

    expected_sha = str(_sha256(expected_sha256, "RUN_COMPLETE-v2 SHA-256"))
    root = _directory(run_dir, "Stage2 detector run directory")
    artifact = _regular_file(Path(artifact_path).expanduser(), "RUN_COMPLETE-v2 artifact")
    if artifact.parent != root or artifact.name != RUN_COMPLETE_NAME:
        raise Stage2DetectorRunCompleteV2Error(
            "RUN_COMPLETE-v2 artifact must be the canonical run-directory member"
        )
    sidecar = _run_file(root, RUN_COMPLETE_SIDECAR_NAME, "RUN_COMPLETE-v2 commit-last sidecar")
    binding = _verify_sha256sum_sidecar(
        artifact, sidecar, expected_sha, "RUN_COMPLETE-v2 artifact"
    )
    if binding["path"] != RUN_COMPLETE_NAME:
        raise RuntimeError("internal RUN_COMPLETE-v2 binding mismatch")
    raw, actual_sha = _stable_bytes(artifact, "RUN_COMPLETE-v2 artifact")
    payload, parsed_sha = _json_object(artifact, "RUN_COMPLETE-v2 artifact")
    if parsed_sha != actual_sha or raw != _canonical_json(payload):
        raise Stage2DetectorRunCompleteV2Error(
            "RUN_COMPLETE-v2 artifact is not canonical JSON"
        )

    expected_payload, replayed_root, contract_path, run, runtime, hashes = (
        _build_expected_payload(
            run_dir=root,
            run_contract_path=run_contract_path,
            verified_run_contract=verified_run_contract,
            verified_runtime_closure=verified_runtime_closure,
            external_hashes=external_hashes,
        )
    )
    if replayed_root != root or payload != expected_payload:
        raise Stage2DetectorRunCompleteV2Error(
            "RUN_COMPLETE-v2 payload differs from current verified run state"
        )
    return VerifiedStage2DetectorRunCompleteV2(
        artifact_path=str(artifact),
        sha256=actual_sha,
        run_dir=str(root),
        run_contract_path=str(contract_path),
        payload=payload,
        external_hashes=hashes,
        verified_run_contract=run,
        verified_runtime_closure=runtime,
        _capability=_CAPABILITY_TOKEN,
    )


def assert_stage2_run_complete_for_score_export_v2(
    capability: Any,
    *,
    run_contract_path: str | Path,
    run_contract_sha256: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
) -> VerifiedStage2DetectorRunCompleteV2:
    """Replay and bind RUN_COMPLETE-v2 to one restricted score export."""

    value = assert_verified_stage2_detector_run_complete_v2(capability)
    replay = verify_stage2_detector_run_complete_v2(
        value.artifact_path,
        value.sha256,
        run_dir=value.run_dir,
        run_contract_path=value.run_contract_path,
        verified_run_contract=_plain(value._verified_run_contract),
        verified_runtime_closure=_plain(value._verified_runtime_closure),
        external_hashes=_plain(value.external_hashes),
    )
    supplied_run = _regular_file(
        Path(run_contract_path).expanduser(), "score-export run contract"
    )
    bound_run = _regular_file(
        Path(value.run_contract_path), "RUN_COMPLETE-v2 run contract"
    )
    if supplied_run != bound_run or run_contract_sha256 != value.external_hashes[
        "run_contract_sha256"
    ]:
        raise Stage2DetectorRunCompleteV2Error(
            "score export run contract is not the RUN_COMPLETE-v2 run"
        )
    supplied_checkpoint = _regular_file(
        Path(checkpoint_path).expanduser(), "score-export restricted checkpoint"
    )
    expected_checkpoint = _run_file(
        Path(value.run_dir),
        "stage2_inference_checkpoint.pt",
        "RUN_COMPLETE-v2 restricted checkpoint",
    )
    if (
        supplied_checkpoint != expected_checkpoint
        or checkpoint_sha256
        != value.external_hashes["restricted_inference_checkpoint_sha256"]
    ):
        raise Stage2DetectorRunCompleteV2Error(
            "score export checkpoint is not the RUN_COMPLETE-v2 restricted checkpoint"
        )
    if replay.sha256 != value.sha256 or _plain(replay.payload) != _plain(value.payload):
        raise RuntimeError("RUN_COMPLETE-v2 capability replay drift")
    return value


__all__ = [
    "EXTERNAL_HASH_KEYS",
    "RUN_COMPLETE_ARTIFACT_TYPE",
    "RUN_COMPLETE_NAME",
    "RUN_COMPLETE_SCHEMA_V2",
    "RUN_COMPLETE_SIDECAR_NAME",
    "RUN_COMPLETE_STATUS",
    "Stage2DetectorRunCompleteV2Error",
    "VerifiedStage2DetectorRunCompleteV2",
    "assert_stage2_run_complete_for_score_export_v2",
    "assert_verified_stage2_detector_run_complete_v2",
    "publish_stage2_detector_run_complete_v2",
    "verify_stage2_detector_run_complete_v2",
]
