"""Immutable RC5 calibrator training generations and completed-run commits.

Checkpoint-v7 is deliberately deployment-only.  This module owns the separate
resumable state required by S2_I0: optimizer state, epoch/rank/history and every
RNG stream.  Each epoch is published as an immutable directory.  A generation
commit is written last and must be supplied with an external SHA-256 before a
resume capability can be issued.  A run commit is likewise immutable and its
selected generation is recomputed from the frozen source-only ranking rule.

No outer-target score, label, mask or metric is an admissible input.  The
module only reads the files inside a generation/run bundle and the caller-
supplied checkpoint-v7 bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import tempfile
from types import MappingProxyType
from typing import Any

import torch

from rc.stage2_calibrator_checkpoint_v7 import (
    METHODS,
    VerifiedCalibratorCheckpointV7,
    tensor_tree_content_sha256,
    verify_calibrator_checkpoint_v7_bytes,
)


GENERATION_MANIFEST_SCHEMA = "rc-irstd.calibrator-generation-manifest.v2"
GENERATION_COMMIT_SCHEMA = "rc-irstd.calibrator-generation-commit.v2"
RUN_COMMIT_SCHEMA = "rc-irstd.calibrator-run-commit.v2"
RESUME_STATE_SCHEMA = "rc-irstd.calibrator-resume-state.v2"

GENERATION_MANIFEST_ARTIFACT = "rc_irstd_calibrator_generation_manifest"
GENERATION_COMMIT_ARTIFACT = "rc_irstd_calibrator_generation_commit"
RUN_COMMIT_ARTIFACT = "rc_irstd_calibrator_run_commit"

RESUME_FILENAME = "resume_state.pt"
DEPLOYMENT_FILENAME = "deployment_checkpoint_v7.pt"
MANIFEST_FILENAME = "generation_manifest.json"
COMMIT_FILENAME = "GENERATION_COMMIT.json"
RUN_COMMIT_FILENAME = "RUN_COMMIT.json"

SELECTION_RANK = (
    "macro_source_BSR_max",
    "macro_source_LogExcess_min",
    "macro_source_Pd_max",
    "earlier_epoch_on_exact_tie",
)
INPUT_BINDING_NAMES = (
    "rc5_config",
    "training_collection",
    "validation_collection",
    "statistics_config",
    "source_reference",
    "per_image_curve_bank",
    "detector_run_complete_set",
    "seed_manifest",
    "source_release",
)

_SHA_CHARS = frozenset("0123456789abcdef")
_GENERATION_TOKEN = object()
_RUN_TOKEN = object()

_RESUME_FIELDS = frozenset(
    {
        "format_version",
        "method",
        "run_id",
        "outer_fold_id",
        "outer_target_domain",
        "base_seed",
        "derived_seed",
        "epoch",
        "process_rank",
        "world_size",
        "training_contract_sha256",
        "input_identity_sha256",
        "model_state_dict",
        "optimizer_state_dict",
        "history",
        "selection_record",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "dataloader_rng_state",
        "official_test_accessed",
        "outer_target_accessed",
        "query_labels_accessed",
    }
)


class Stage2CalibratorGenerationV2Error(ValueError):
    """An RC5 training-generation or completed-run contract failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2CalibratorGenerationV2Error(
            f"value is not finite canonical JSON: {error}"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise Stage2CalibratorGenerationV2Error(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise Stage2CalibratorGenerationV2Error(
            f"{name} must be non-empty trimmed text"
        )
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2CalibratorGenerationV2Error(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _false(value: Any, name: str) -> None:
    if type(value) is not bool or value is not False:
        raise Stage2CalibratorGenerationV2Error(f"{name} must be exact false")


def _exact_fields(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise Stage2CalibratorGenerationV2Error(
            f"{name} fields mismatch; missing={sorted(fields-actual)}, "
            f"extra={sorted(actual-fields)}"
        )
    return value


def _float_hex(value: Any, name: str) -> float:
    if type(value) is not str:
        raise Stage2CalibratorGenerationV2Error(
            f"{name} must be canonical float.hex text"
        )
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise Stage2CalibratorGenerationV2Error(f"{name} is invalid") from error
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise Stage2CalibratorGenerationV2Error(
            f"{name} must canonically encode finite binary64"
        )
    return parsed


def _relative_path(value: Any, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage2CalibratorGenerationV2Error(
            f"{name} must be a canonical relative POSIX path"
        )
    return path.as_posix()


def _binding(value: Any, name: str) -> dict[str, str]:
    row = _exact_fields(value, frozenset({"path", "sha256"}), name)
    return {
        "path": _relative_path(row["path"], f"{name}.path"),
        "sha256": _sha(row["sha256"], f"{name}.sha256"),
    }


def normalize_input_bindings(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(
        sorted(INPUT_BINDING_NAMES)
    ):
        raise Stage2CalibratorGenerationV2Error(
            "input_bindings must contain the exact frozen RC5 binding set"
        )
    return {
        name: _binding(value[name], f"input_bindings.{name}")
        for name in INPUT_BINDING_NAMES
    }


def input_identity_sha256(value: Any) -> str:
    return canonical_json_sha256(normalize_input_bindings(value))


def build_selection_record(
    *,
    macro_source_bsr: float,
    macro_source_log_excess: float,
    macro_source_pd: float,
) -> dict[str, Any]:
    values = {
        "macro_source_bsr_hex": float(macro_source_bsr).hex(),
        "macro_source_log_excess_hex": float(macro_source_log_excess).hex(),
        "macro_source_pd_hex": float(macro_source_pd).hex(),
    }
    for key, raw in values.items():
        parsed = _float_hex(raw, key)
        if key != "macro_source_log_excess_hex" and not 0.0 <= parsed <= 1.0:
            raise Stage2CalibratorGenerationV2Error(
                f"{key} must lie in [0,1]"
            )
        if key == "macro_source_log_excess_hex" and parsed < 0.0:
            raise Stage2CalibratorGenerationV2Error(
                "macro_source_log_excess must be nonnegative"
            )
    return {
        "schema_version": "rc-irstd.calibrator-source-selection-record.v2",
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
        "rank": list(SELECTION_RANK),
        **values,
        "outer_target_accessed": False,
    }


def _selection_record(value: Any) -> dict[str, Any]:
    fields = frozenset(
        {
            "schema_version",
            "selection_geometry",
            "source_domain_weighting",
            "within_domain_bsr",
            "within_domain_log_excess",
            "within_domain_pd",
            "source_variable_query_sanity_excluded_from_epoch_ranking",
            "cyclic_starts_claimed_independent",
            "cyclic_start_confidence_interval_reported",
            "rank",
            "macro_source_bsr_hex",
            "macro_source_log_excess_hex",
            "macro_source_pd_hex",
            "outer_target_accessed",
        }
    )
    row = dict(_exact_fields(value, fields, "selection_record"))
    if row != build_selection_record(
        macro_source_bsr=_float_hex(
            row["macro_source_bsr_hex"], "selection_record.macro_source_bsr_hex"
        ),
        macro_source_log_excess=_float_hex(
            row["macro_source_log_excess_hex"],
            "selection_record.macro_source_log_excess_hex",
        ),
        macro_source_pd=_float_hex(
            row["macro_source_pd_hex"], "selection_record.macro_source_pd_hex"
        ),
    ):
        raise Stage2CalibratorGenerationV2Error(
            "selection_record contract drifted"
        )
    return row


def _safe_tree(value: Any, name: str) -> None:
    """Accept only the tensor/primitives subset supported by weights_only."""

    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise Stage2CalibratorGenerationV2Error(f"{name} contains non-finite float")
        return
    if isinstance(value, torch.Tensor):
        if (
            value.layout != torch.strided
            or value.is_quantized
            or value.device.type == "meta"
        ):
            raise Stage2CalibratorGenerationV2Error(
                f"{name} contains unsupported tensor"
            )
        tensor = value.detach().cpu()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise Stage2CalibratorGenerationV2Error(
                f"{name} contains non-finite tensor"
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _safe_tree(item, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) not in {str, int}:
                raise Stage2CalibratorGenerationV2Error(
                    f"{name} has unsupported mapping key"
                )
            _safe_tree(item, f"{name}[{key!r}]")
        return
    raise Stage2CalibratorGenerationV2Error(
        f"{name} contains unsupported type {type(value).__name__}"
    )


def _semantic_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"rc-irstd.safe-tensor-primitive-tree.v1\0")

    def visit(item: Any) -> None:
        if item is None:
            digest.update(b"N")
        elif type(item) is bool:
            digest.update(b"B1" if item else b"B0")
        elif type(item) is int:
            raw = str(item).encode("ascii")
            digest.update(b"I" + struct.pack(">Q", len(raw)) + raw)
        elif type(item) is float:
            raw = item.hex().encode("ascii")
            digest.update(b"F" + struct.pack(">Q", len(raw)) + raw)
        elif type(item) is str:
            raw = item.encode("utf-8")
            digest.update(b"S" + struct.pack(">Q", len(raw)) + raw)
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            metadata = canonical_json_bytes(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            )
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            digest.update(b"T" + struct.pack(">Q", len(metadata)) + metadata)
            digest.update(struct.pack(">Q", len(raw)) + raw)
        elif isinstance(item, (list, tuple)):
            digest.update(b"L" + struct.pack(">Q", len(item)))
            for child in item:
                visit(child)
        elif isinstance(item, Mapping):
            keys = sorted(item, key=lambda key: (type(key).__name__, str(key)))
            digest.update(b"M" + struct.pack(">Q", len(keys)))
            for key in keys:
                visit(key)
                visit(item[key])
        else:  # pragma: no cover - guarded by _safe_tree
            raise TypeError(type(item).__name__)

    _safe_tree(value, "semantic_tree")
    visit(value)
    return digest.hexdigest()


def _rng_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise Stage2CalibratorGenerationV2Error(f"{name} must be a tensor")
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() == 0:
        raise Stage2CalibratorGenerationV2Error(
            f"{name} must be non-empty one-dimensional CPU uint8"
        )
    return tensor


def build_resume_state_v2(
    *,
    method: str,
    run_id: str,
    outer_fold_id: str,
    outer_target_domain: str,
    base_seed: int,
    derived_seed: int,
    epoch: int,
    process_rank: int,
    world_size: int,
    training_contract_sha256: str,
    input_bindings: Mapping[str, Any],
    model_state_dict: Mapping[str, torch.Tensor],
    optimizer_state_dict: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    selection_record: Mapping[str, Any],
    python_rng_state: Mapping[str, Any],
    numpy_rng_state: Mapping[str, Any],
    torch_cpu_rng_state: torch.Tensor,
    torch_cuda_rng_states: Sequence[torch.Tensor],
    dataloader_rng_state: torch.Tensor,
) -> dict[str, Any]:
    if method not in METHODS:
        raise Stage2CalibratorGenerationV2Error("method must be T6, T7 or T8")
    bindings = normalize_input_bindings(input_bindings)
    state = {
        "format_version": RESUME_STATE_SCHEMA,
        "method": method,
        "run_id": _text(run_id, "run_id"),
        "outer_fold_id": _text(outer_fold_id, "outer_fold_id"),
        "outer_target_domain": _text(outer_target_domain, "outer_target_domain"),
        "base_seed": _integer(base_seed, "base_seed"),
        "derived_seed": _integer(derived_seed, "derived_seed", 1),
        "epoch": _integer(epoch, "epoch"),
        "process_rank": _integer(process_rank, "process_rank"),
        "world_size": _integer(world_size, "world_size", 1),
        "training_contract_sha256": _sha(
            training_contract_sha256, "training_contract_sha256"
        ),
        "input_identity_sha256": canonical_json_sha256(bindings),
        "model_state_dict": {
            str(key): value.detach().cpu().contiguous().clone()
            for key, value in model_state_dict.items()
        },
        "optimizer_state_dict": dict(optimizer_state_dict),
        "history": [dict(row) for row in history],
        "selection_record": _selection_record(selection_record),
        "python_rng_state": dict(python_rng_state),
        "numpy_rng_state": dict(numpy_rng_state),
        "torch_cpu_rng_state": _rng_tensor(
            torch_cpu_rng_state, "torch_cpu_rng_state"
        ).clone(),
        "torch_cuda_rng_states": [
            _rng_tensor(item, f"torch_cuda_rng_states[{index}]").clone()
            for index, item in enumerate(torch_cuda_rng_states)
        ],
        "dataloader_rng_state": _rng_tensor(
            dataloader_rng_state, "dataloader_rng_state"
        ).clone(),
        "official_test_accessed": False,
        "outer_target_accessed": False,
        "query_labels_accessed": False,
    }
    _verify_resume_state(state, expected_input_identity=state["input_identity_sha256"])
    return state


def _verify_resume_state(
    value: Any, *, expected_input_identity: str
) -> Mapping[str, Any]:
    state = _exact_fields(value, _RESUME_FIELDS, "resume_state")
    if state["format_version"] != RESUME_STATE_SCHEMA or state["method"] not in METHODS:
        raise Stage2CalibratorGenerationV2Error("resume_state identity mismatch")
    for field in ("run_id", "outer_fold_id", "outer_target_domain"):
        _text(state[field], f"resume_state.{field}")
    _integer(state["base_seed"], "resume_state.base_seed")
    _integer(state["derived_seed"], "resume_state.derived_seed", 1)
    _integer(state["epoch"], "resume_state.epoch")
    rank = _integer(state["process_rank"], "resume_state.process_rank")
    world = _integer(state["world_size"], "resume_state.world_size", 1)
    if rank >= world:
        raise Stage2CalibratorGenerationV2Error("process_rank must be < world_size")
    _sha(state["training_contract_sha256"], "resume_state.training_contract_sha256")
    if _sha(state["input_identity_sha256"], "resume_state.input_identity_sha256") != _sha(
        expected_input_identity, "expected_input_identity"
    ):
        raise Stage2CalibratorGenerationV2Error("resume_state input identity mismatch")
    if not isinstance(state["model_state_dict"], Mapping) or not state["model_state_dict"]:
        raise Stage2CalibratorGenerationV2Error("model_state_dict must be non-empty")
    if any(type(key) is not str or not key for key in state["model_state_dict"]):
        raise Stage2CalibratorGenerationV2Error("model_state_dict keys are invalid")
    if not isinstance(state["optimizer_state_dict"], Mapping):
        raise Stage2CalibratorGenerationV2Error("optimizer_state_dict must be a mapping")
    if not isinstance(state["history"], list):
        raise Stage2CalibratorGenerationV2Error("history must be a list")
    if len(state["history"]) != state["epoch"] + 1:
        raise Stage2CalibratorGenerationV2Error(
            "history must contain exactly one record for every completed epoch"
        )
    for index, row in enumerate(state["history"]):
        if not isinstance(row, Mapping) or row.get("epoch") != index:
            raise Stage2CalibratorGenerationV2Error(
                "history epoch order must be contiguous from zero"
            )
    _selection_record(state["selection_record"])
    if not isinstance(state["python_rng_state"], Mapping) or not isinstance(
        state["numpy_rng_state"], Mapping
    ):
        raise Stage2CalibratorGenerationV2Error(
            "Python and NumPy RNG states must be explicit mappings"
        )
    _rng_tensor(state["torch_cpu_rng_state"], "resume_state.torch_cpu_rng_state")
    if not isinstance(state["torch_cuda_rng_states"], list):
        raise Stage2CalibratorGenerationV2Error(
            "torch_cuda_rng_states must be a list (possibly empty on CPU)"
        )
    for index, item in enumerate(state["torch_cuda_rng_states"]):
        _rng_tensor(item, f"resume_state.torch_cuda_rng_states[{index}]")
    _rng_tensor(state["dataloader_rng_state"], "resume_state.dataloader_rng_state")
    for key in (
        "official_test_accessed",
        "outer_target_accessed",
        "query_labels_accessed",
    ):
        _false(state[key], f"resume_state.{key}")
    _safe_tree(state, "resume_state")
    return state


def _root(path: str | Path, *, create: bool) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise Stage2CalibratorGenerationV2Error("run root must not be a symlink")
    if create:
        raw.mkdir(parents=True, exist_ok=True)
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise Stage2CalibratorGenerationV2Error("run root must be a directory")
    return root


def _direct_file(path: Path, parent: Path, name: str) -> Path:
    try:
        relative = path.relative_to(parent)
    except ValueError as error:
        raise Stage2CalibratorGenerationV2Error(f"{name} escapes bundle root") from error
    current = parent
    for part in relative.parts:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise Stage2CalibratorGenerationV2Error(
                f"{name} contains a symlink component"
            )
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise Stage2CalibratorGenerationV2Error(f"{name} is not a regular file")
    return path


def _stable_bytes(path: Path, parent: Path, name: str) -> tuple[bytes, str]:
    candidate = _direct_file(path, parent, name)
    descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if identity(before) != identity(after):
        raise RuntimeError(f"{name} changed during read")
    return b"".join(chunks), digest.hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _torch_bytes(value: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _torch_load_safe(data: bytes, name: str) -> Any:
    import io

    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage2CalibratorGenerationV2Error(
            f"{name} is not weights-only loadable: {error}"
        ) from error


@dataclass(frozen=True, init=False)
class VerifiedCalibratorGenerationV2:
    path: Path
    commit_sha256: str
    manifest: Mapping[str, Any]
    resume_state: Mapping[str, Any]
    deployment_checkpoint: VerifiedCalibratorCheckpointV7
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedCalibratorGenerationV2 is verifier-issued only")


def _generation_capability(
    *,
    path: Path,
    commit_sha256: str,
    manifest: Mapping[str, Any],
    resume_state: Mapping[str, Any],
    deployment: VerifiedCalibratorCheckpointV7,
) -> VerifiedCalibratorGenerationV2:
    value = object.__new__(VerifiedCalibratorGenerationV2)
    object.__setattr__(value, "path", path)
    object.__setattr__(value, "commit_sha256", commit_sha256)
    object.__setattr__(value, "manifest", MappingProxyType(dict(manifest)))
    object.__setattr__(value, "resume_state", MappingProxyType(dict(resume_state)))
    object.__setattr__(value, "deployment_checkpoint", deployment)
    object.__setattr__(value, "_capability", _GENERATION_TOKEN)
    return value


def assert_verified_calibrator_generation_v2(
    value: Any,
) -> VerifiedCalibratorGenerationV2:
    if (
        type(value) is not VerifiedCalibratorGenerationV2
        or getattr(value, "_capability", None) is not _GENERATION_TOKEN
    ):
        raise TypeError("a verifier-issued generation-v2 capability is required")
    return value


def publish_calibrator_generation_v2(
    run_root: str | Path,
    *,
    resume_state: Mapping[str, Any],
    deployment_checkpoint_bytes: bytes,
    input_bindings: Mapping[str, Any],
) -> VerifiedCalibratorGenerationV2:
    """Publish one immutable epoch generation and verify it from disk."""

    root = _root(run_root, create=True)
    bindings = normalize_input_bindings(input_bindings)
    identity = canonical_json_sha256(bindings)
    state = dict(_verify_resume_state(resume_state, expected_input_identity=identity))
    if type(deployment_checkpoint_bytes) is not bytes or not deployment_checkpoint_bytes:
        raise Stage2CalibratorGenerationV2Error(
            "deployment_checkpoint_bytes must be non-empty bytes"
        )
    deployment = verify_calibrator_checkpoint_v7_bytes(
        deployment_checkpoint_bytes,
        expected_training_contract_sha256=state["training_contract_sha256"],
    )
    if tensor_tree_content_sha256(state["model_state_dict"]) != deployment.payload()[
        "model_state_content_sha256"
    ]:
        raise Stage2CalibratorGenerationV2Error(
            "resume and deployment model-state content digests differ"
        )
    if deployment.method != state["method"]:
        raise Stage2CalibratorGenerationV2Error(
            "deployment and resume-state methods differ"
        )

    generation_name = f"generation_e{state['epoch']:06d}_r{state['process_rank']:04d}"
    final = root / generation_name
    if final.exists() or final.is_symlink():
        raise FileExistsError("immutable generation already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_name}.staging-", dir=root))
    try:
        state_bytes = _torch_bytes(state)
        replay = _torch_load_safe(state_bytes, "new resume state")
        _verify_resume_state(replay, expected_input_identity=identity)
        semantic_sha = _semantic_tree_sha256(replay)
        _write_new(staging / RESUME_FILENAME, state_bytes)
        _write_new(staging / DEPLOYMENT_FILENAME, deployment_checkpoint_bytes)
        manifest: dict[str, Any] = {
            "schema_version": GENERATION_MANIFEST_SCHEMA,
            "artifact_type": GENERATION_MANIFEST_ARTIFACT,
            "artifact_status": "IMMUTABLE_GENERATION_COMPLETE",
            "method": state["method"],
            "run_id": state["run_id"],
            "outer_fold_id": state["outer_fold_id"],
            "outer_target_domain": state["outer_target_domain"],
            "base_seed": state["base_seed"],
            "derived_seed": state["derived_seed"],
            "epoch": state["epoch"],
            "process_rank": state["process_rank"],
            "world_size": state["world_size"],
            "training_contract_sha256": state["training_contract_sha256"],
            "input_bindings": bindings,
            "input_identity_sha256": identity,
            "resume_state": {
                "path": RESUME_FILENAME,
                "sha256": hashlib.sha256(state_bytes).hexdigest(),
                "semantic_sha256": semantic_sha,
                "serialization": "torch_tensors_and_primitives_weights_only",
            },
            "deployment_checkpoint": {
                "path": DEPLOYMENT_FILENAME,
                "sha256": hashlib.sha256(deployment_checkpoint_bytes).hexdigest(),
                "schema_version": "rc-irstd.calibrator.v7",
            },
            "selection_record": state["selection_record"],
            "checkpoint_selection_rank": list(SELECTION_RANK),
            "official_test_accessed": False,
            "outer_target_accessed": False,
            "query_labels_accessed": False,
            "manifest_identity_sha256": "",
        }
        projection = dict(manifest)
        projection.pop("manifest_identity_sha256")
        manifest["manifest_identity_sha256"] = canonical_json_sha256(projection)
        manifest_bytes = canonical_json_bytes(manifest)
        _write_new(staging / MANIFEST_FILENAME, manifest_bytes)
        commit: dict[str, Any] = {
            "schema_version": GENERATION_COMMIT_SCHEMA,
            "artifact_type": GENERATION_COMMIT_ARTIFACT,
            "artifact_status": "COMMITTED_LAST",
            "generation_name": generation_name,
            "manifest": {
                "path": MANIFEST_FILENAME,
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            },
            "resume_state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "deployment_checkpoint_sha256": hashlib.sha256(
                deployment_checkpoint_bytes
            ).hexdigest(),
            "input_identity_sha256": identity,
            "commit_identity_sha256": "",
        }
        projection = dict(commit)
        projection.pop("commit_identity_sha256")
        commit["commit_identity_sha256"] = canonical_json_sha256(projection)
        commit_bytes = canonical_json_bytes(commit)
        _write_new(staging / COMMIT_FILENAME, commit_bytes)
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.rename(staging, final)
        staging = Path()
        parent_descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return verify_calibrator_generation_v2(
            final,
            hashlib.sha256(commit_bytes).hexdigest(),
        )
    finally:
        if staging != Path() and staging.exists() and staging.parent == root:
            shutil.rmtree(staging)


def verify_calibrator_generation_v2(
    path: str | Path,
    expected_commit_sha256: str,
) -> VerifiedCalibratorGenerationV2:
    """Verify one generation; external commit SHA is mandatory for resume."""

    generation = Path(path).expanduser()
    if generation.is_symlink():
        raise Stage2CalibratorGenerationV2Error("generation must not be a symlink")
    generation = generation.resolve(strict=True)
    if not generation.is_dir() or generation.name.startswith("."):
        raise Stage2CalibratorGenerationV2Error("invalid generation directory")
    expected = _sha(expected_commit_sha256, "expected_commit_sha256")
    commit_bytes, commit_sha = _stable_bytes(
        generation / COMMIT_FILENAME, generation, "generation commit"
    )
    if commit_sha != expected:
        raise Stage2CalibratorGenerationV2Error(
            "generation external commit SHA-256 mismatch"
        )
    try:
        commit = json.loads(commit_bytes)
    except json.JSONDecodeError as error:
        raise Stage2CalibratorGenerationV2Error("generation commit is not JSON") from error
    if canonical_json_bytes(commit) != commit_bytes:
        raise Stage2CalibratorGenerationV2Error("generation commit is not canonical JSON")
    commit_fields = frozenset(
        {
            "schema_version",
            "artifact_type",
            "artifact_status",
            "generation_name",
            "manifest",
            "resume_state_sha256",
            "deployment_checkpoint_sha256",
            "input_identity_sha256",
            "commit_identity_sha256",
        }
    )
    _exact_fields(commit, commit_fields, "generation commit")
    if (
        commit["schema_version"] != GENERATION_COMMIT_SCHEMA
        or commit["artifact_type"] != GENERATION_COMMIT_ARTIFACT
        or commit["artifact_status"] != "COMMITTED_LAST"
        or commit["generation_name"] != generation.name
    ):
        raise Stage2CalibratorGenerationV2Error("generation commit identity mismatch")
    projection = dict(commit)
    declared_identity = projection.pop("commit_identity_sha256")
    if _sha(declared_identity, "commit_identity_sha256") != canonical_json_sha256(
        projection
    ):
        raise Stage2CalibratorGenerationV2Error("generation commit self-hash mismatch")
    manifest_binding = _binding(commit["manifest"], "generation commit.manifest")
    if manifest_binding["path"] != MANIFEST_FILENAME:
        raise Stage2CalibratorGenerationV2Error("generation manifest path mismatch")
    manifest_bytes, manifest_sha = _stable_bytes(
        generation / MANIFEST_FILENAME, generation, "generation manifest"
    )
    if manifest_sha != manifest_binding["sha256"]:
        raise Stage2CalibratorGenerationV2Error("generation manifest SHA mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise Stage2CalibratorGenerationV2Error("generation manifest is not JSON") from error
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise Stage2CalibratorGenerationV2Error("generation manifest is not canonical JSON")
    manifest_fields = frozenset(
        {
            "schema_version",
            "artifact_type",
            "artifact_status",
            "method",
            "run_id",
            "outer_fold_id",
            "outer_target_domain",
            "base_seed",
            "derived_seed",
            "epoch",
            "process_rank",
            "world_size",
            "training_contract_sha256",
            "input_bindings",
            "input_identity_sha256",
            "resume_state",
            "deployment_checkpoint",
            "selection_record",
            "checkpoint_selection_rank",
            "official_test_accessed",
            "outer_target_accessed",
            "query_labels_accessed",
            "manifest_identity_sha256",
        }
    )
    _exact_fields(manifest, manifest_fields, "generation manifest")
    if (
        manifest["schema_version"] != GENERATION_MANIFEST_SCHEMA
        or manifest["artifact_type"] != GENERATION_MANIFEST_ARTIFACT
        or manifest["artifact_status"] != "IMMUTABLE_GENERATION_COMPLETE"
        or manifest["checkpoint_selection_rank"] != list(SELECTION_RANK)
    ):
        raise Stage2CalibratorGenerationV2Error("generation manifest identity mismatch")
    projection = dict(manifest)
    declared_identity = projection.pop("manifest_identity_sha256")
    if _sha(declared_identity, "manifest_identity_sha256") != canonical_json_sha256(
        projection
    ):
        raise Stage2CalibratorGenerationV2Error("generation manifest self-hash mismatch")
    bindings = normalize_input_bindings(manifest["input_bindings"])
    identity = canonical_json_sha256(bindings)
    if manifest["input_identity_sha256"] != identity or commit[
        "input_identity_sha256"
    ] != identity:
        raise Stage2CalibratorGenerationV2Error("generation input identity mismatch")
    for key in ("official_test_accessed", "outer_target_accessed", "query_labels_accessed"):
        _false(manifest[key], f"generation manifest.{key}")
    _selection_record(manifest["selection_record"])

    resume_binding = _exact_fields(
        manifest["resume_state"],
        frozenset({"path", "sha256", "semantic_sha256", "serialization"}),
        "generation manifest.resume_state",
    )
    if (
        resume_binding["path"] != RESUME_FILENAME
        or resume_binding["serialization"]
        != "torch_tensors_and_primitives_weights_only"
    ):
        raise Stage2CalibratorGenerationV2Error("resume-state binding mismatch")
    state_bytes, state_sha = _stable_bytes(
        generation / RESUME_FILENAME, generation, "resume state"
    )
    if state_sha != _sha(resume_binding["sha256"], "resume_state.sha256") or state_sha != _sha(
        commit["resume_state_sha256"], "commit.resume_state_sha256"
    ):
        raise Stage2CalibratorGenerationV2Error("resume-state SHA mismatch")
    state = _torch_load_safe(state_bytes, "resume state")
    _verify_resume_state(state, expected_input_identity=identity)
    if _semantic_tree_sha256(state) != _sha(
        resume_binding["semantic_sha256"], "resume_state.semantic_sha256"
    ):
        raise Stage2CalibratorGenerationV2Error("resume-state semantic hash mismatch")

    deployment_binding = _exact_fields(
        manifest["deployment_checkpoint"],
        frozenset({"path", "sha256", "schema_version"}),
        "generation manifest.deployment_checkpoint",
    )
    if (
        deployment_binding["path"] != DEPLOYMENT_FILENAME
        or deployment_binding["schema_version"] != "rc-irstd.calibrator.v7"
    ):
        raise Stage2CalibratorGenerationV2Error("deployment binding mismatch")
    deployment_bytes, deployment_sha = _stable_bytes(
        generation / DEPLOYMENT_FILENAME, generation, "deployment checkpoint"
    )
    if deployment_sha != _sha(
        deployment_binding["sha256"], "deployment_checkpoint.sha256"
    ) or deployment_sha != _sha(
        commit["deployment_checkpoint_sha256"],
        "commit.deployment_checkpoint_sha256",
    ):
        raise Stage2CalibratorGenerationV2Error("deployment checkpoint SHA mismatch")
    deployment = verify_calibrator_checkpoint_v7_bytes(
        deployment_bytes,
        expected_training_contract_sha256=manifest["training_contract_sha256"],
    )
    if tensor_tree_content_sha256(state["model_state_dict"]) != deployment.payload()[
        "model_state_content_sha256"
    ]:
        raise Stage2CalibratorGenerationV2Error(
            "resume and deployment model-state content digests differ"
        )
    identity_fields = (
        "method",
        "run_id",
        "outer_fold_id",
        "outer_target_domain",
        "base_seed",
        "derived_seed",
        "epoch",
        "process_rank",
        "world_size",
        "training_contract_sha256",
        "input_identity_sha256",
        "selection_record",
    )
    for field in identity_fields:
        if state[field] != manifest[field]:
            raise Stage2CalibratorGenerationV2Error(
                f"resume/manifest identity mismatch: {field}"
            )
    if deployment.method != manifest["method"]:
        raise Stage2CalibratorGenerationV2Error("deployment method mismatch")
    return _generation_capability(
        path=generation,
        commit_sha256=commit_sha,
        manifest=manifest,
        resume_state=state,
        deployment=deployment,
    )


def _selection_key(generation: VerifiedCalibratorGenerationV2) -> tuple[float, float, float, int]:
    row = generation.manifest["selection_record"]
    return (
        -_float_hex(row["macro_source_bsr_hex"], "macro_source_bsr_hex"),
        _float_hex(row["macro_source_log_excess_hex"], "macro_source_log_excess_hex"),
        -_float_hex(row["macro_source_pd_hex"], "macro_source_pd_hex"),
        int(generation.manifest["epoch"]),
    )


@dataclass(frozen=True, init=False)
class VerifiedCalibratorRunV2:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    generations: tuple[VerifiedCalibratorGenerationV2, ...]
    selected_generation: VerifiedCalibratorGenerationV2
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedCalibratorRunV2 is verifier-issued only")


def _run_capability(
    *, path: Path, sha256: str, payload: Mapping[str, Any], generations: Sequence[VerifiedCalibratorGenerationV2]
) -> VerifiedCalibratorRunV2:
    value = object.__new__(VerifiedCalibratorRunV2)
    ordered = tuple(generations)
    selected_epoch = payload["selected_generation"]["epoch"]
    selected = next(item for item in ordered if item.manifest["epoch"] == selected_epoch)
    object.__setattr__(value, "path", path)
    object.__setattr__(value, "sha256", sha256)
    object.__setattr__(value, "payload", MappingProxyType(dict(payload)))
    object.__setattr__(value, "generations", ordered)
    object.__setattr__(value, "selected_generation", selected)
    object.__setattr__(value, "_capability", _RUN_TOKEN)
    return value


def publish_calibrator_run_v2(
    run_root: str | Path,
    generations: Sequence[VerifiedCalibratorGenerationV2],
) -> VerifiedCalibratorRunV2:
    """Commit a completed run after recomputing its unique source-only winner."""

    root = _root(run_root, create=False)
    ordered = tuple(assert_verified_calibrator_generation_v2(item) for item in generations)
    if not ordered:
        raise Stage2CalibratorGenerationV2Error("a completed run needs generations")
    if any(item.path.parent != root for item in ordered):
        raise Stage2CalibratorGenerationV2Error("generation is outside run_root")
    ordered = tuple(sorted(ordered, key=lambda item: item.manifest["epoch"]))
    if [item.manifest["epoch"] for item in ordered] != list(range(len(ordered))):
        raise Stage2CalibratorGenerationV2Error(
            "completed run generations must cover contiguous epochs from zero"
        )
    first = ordered[0].manifest
    identity_fields = (
        "method",
        "run_id",
        "outer_fold_id",
        "outer_target_domain",
        "base_seed",
        "derived_seed",
        "training_contract_sha256",
        "input_identity_sha256",
        "input_bindings",
    )
    for generation in ordered[1:]:
        for field in identity_fields:
            if generation.manifest[field] != first[field]:
                raise Stage2CalibratorGenerationV2Error(
                    f"generation run identity mismatch: {field}"
                )
    selected = min(ordered, key=_selection_key)
    inventory = [
        {
            "epoch": item.manifest["epoch"],
            "path": item.path.name + "/" + COMMIT_FILENAME,
            "commit_sha256": item.commit_sha256,
            "manifest_identity_sha256": item.manifest[
                "manifest_identity_sha256"
            ],
            "deployment_checkpoint_sha256": item.manifest[
                "deployment_checkpoint"
            ]["sha256"],
            "selection_record": item.manifest["selection_record"],
        }
        for item in ordered
    ]
    payload: dict[str, Any] = {
        "schema_version": RUN_COMMIT_SCHEMA,
        "artifact_type": RUN_COMMIT_ARTIFACT,
        "artifact_status": "RUN_COMPLETE_SOURCE_ONLY_SELECTION",
        "method": first["method"],
        "run_id": first["run_id"],
        "outer_fold_id": first["outer_fold_id"],
        "outer_target_domain": first["outer_target_domain"],
        "base_seed": first["base_seed"],
        "derived_seed": first["derived_seed"],
        "training_contract_sha256": first["training_contract_sha256"],
        "input_bindings": first["input_bindings"],
        "input_identity_sha256": first["input_identity_sha256"],
        "generation_count": len(ordered),
        "generation_inventory": inventory,
        "checkpoint_selection_rank": list(SELECTION_RANK),
        "selected_generation": {
            "epoch": selected.manifest["epoch"],
            "commit_sha256": selected.commit_sha256,
            "deployment_checkpoint_sha256": selected.manifest[
                "deployment_checkpoint"
            ]["sha256"],
        },
        "official_test_accessed": False,
        "outer_target_accessed": False,
        "query_labels_accessed": False,
        "run_identity_sha256": "",
    }
    projection = dict(payload)
    projection.pop("run_identity_sha256")
    payload["run_identity_sha256"] = canonical_json_sha256(projection)
    data = canonical_json_bytes(payload)
    target = root / RUN_COMMIT_FILENAME
    if target.exists() or target.is_symlink():
        raise FileExistsError("immutable run commit already exists")
    _write_new(target, data)
    return verify_calibrator_run_v2(root, hashlib.sha256(data).hexdigest())


def verify_calibrator_run_v2(
    run_root: str | Path, expected_run_commit_sha256: str
) -> VerifiedCalibratorRunV2:
    root = _root(run_root, create=False)
    data, digest = _stable_bytes(root / RUN_COMMIT_FILENAME, root, "run commit")
    if digest != _sha(expected_run_commit_sha256, "expected_run_commit_sha256"):
        raise Stage2CalibratorGenerationV2Error("run commit external SHA mismatch")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise Stage2CalibratorGenerationV2Error("run commit is not JSON") from error
    if canonical_json_bytes(payload) != data:
        raise Stage2CalibratorGenerationV2Error("run commit is not canonical JSON")
    fields = frozenset(
        {
            "schema_version",
            "artifact_type",
            "artifact_status",
            "method",
            "run_id",
            "outer_fold_id",
            "outer_target_domain",
            "base_seed",
            "derived_seed",
            "training_contract_sha256",
            "input_bindings",
            "input_identity_sha256",
            "generation_count",
            "generation_inventory",
            "checkpoint_selection_rank",
            "selected_generation",
            "official_test_accessed",
            "outer_target_accessed",
            "query_labels_accessed",
            "run_identity_sha256",
        }
    )
    _exact_fields(payload, fields, "run commit")
    if (
        payload["schema_version"] != RUN_COMMIT_SCHEMA
        or payload["artifact_type"] != RUN_COMMIT_ARTIFACT
        or payload["artifact_status"] != "RUN_COMPLETE_SOURCE_ONLY_SELECTION"
        or payload["method"] not in METHODS
        or payload["checkpoint_selection_rank"] != list(SELECTION_RANK)
    ):
        raise Stage2CalibratorGenerationV2Error("run commit identity mismatch")
    for key in ("official_test_accessed", "outer_target_accessed", "query_labels_accessed"):
        _false(payload[key], f"run commit.{key}")
    projection = dict(payload)
    declared_identity = projection.pop("run_identity_sha256")
    if _sha(declared_identity, "run_identity_sha256") != canonical_json_sha256(
        projection
    ):
        raise Stage2CalibratorGenerationV2Error("run commit self-hash mismatch")
    bindings = normalize_input_bindings(payload["input_bindings"])
    if canonical_json_sha256(bindings) != payload["input_identity_sha256"]:
        raise Stage2CalibratorGenerationV2Error("run input identity mismatch")
    inventory = payload["generation_inventory"]
    count = _integer(payload["generation_count"], "generation_count", 1)
    if not isinstance(inventory, list) or len(inventory) != count:
        raise Stage2CalibratorGenerationV2Error("generation inventory count mismatch")
    generations: list[VerifiedCalibratorGenerationV2] = []
    expected_epochs = list(range(count))
    if [row.get("epoch") for row in inventory if isinstance(row, Mapping)] != expected_epochs:
        raise Stage2CalibratorGenerationV2Error("generation epochs are not contiguous")
    inventory_fields = frozenset(
        {
            "epoch",
            "path",
            "commit_sha256",
            "manifest_identity_sha256",
            "deployment_checkpoint_sha256",
            "selection_record",
        }
    )
    for index, raw in enumerate(inventory):
        row = _exact_fields(raw, inventory_fields, f"generation_inventory[{index}]")
        expected_path = f"generation_e{index:06d}_r0000/{COMMIT_FILENAME}"
        if row["path"] != expected_path:
            raise Stage2CalibratorGenerationV2Error(
                "run-v2 currently requires single-process rank-zero generations"
            )
        generation = verify_calibrator_generation_v2(
            root / PurePosixPath(row["path"]).parent,
            _sha(row["commit_sha256"], f"generation_inventory[{index}].commit_sha256"),
        )
        if (
            generation.manifest["manifest_identity_sha256"]
            != row["manifest_identity_sha256"]
            or generation.manifest["deployment_checkpoint"]["sha256"]
            != row["deployment_checkpoint_sha256"]
            or generation.manifest["selection_record"] != row["selection_record"]
        ):
            raise Stage2CalibratorGenerationV2Error("run generation inventory drift")
        for field in (
            "method",
            "run_id",
            "outer_fold_id",
            "outer_target_domain",
            "base_seed",
            "derived_seed",
            "training_contract_sha256",
            "input_identity_sha256",
        ):
            if generation.manifest[field] != payload[field]:
                raise Stage2CalibratorGenerationV2Error(
                    f"run/generation identity mismatch: {field}"
                )
        generations.append(generation)
    selected = min(generations, key=_selection_key)
    selected_fields = _exact_fields(
        payload["selected_generation"],
        frozenset({"epoch", "commit_sha256", "deployment_checkpoint_sha256"}),
        "selected_generation",
    )
    expected_selected = {
        "epoch": selected.manifest["epoch"],
        "commit_sha256": selected.commit_sha256,
        "deployment_checkpoint_sha256": selected.manifest[
            "deployment_checkpoint"
        ]["sha256"],
    }
    if dict(selected_fields) != expected_selected:
        raise Stage2CalibratorGenerationV2Error(
            "selected generation is not the recomputed source-only winner"
        )
    return _run_capability(
        path=root / RUN_COMMIT_FILENAME,
        sha256=digest,
        payload=payload,
        generations=generations,
    )


__all__ = [
    "COMMIT_FILENAME",
    "GENERATION_COMMIT_SCHEMA",
    "GENERATION_MANIFEST_SCHEMA",
    "INPUT_BINDING_NAMES",
    "RESUME_STATE_SCHEMA",
    "RUN_COMMIT_FILENAME",
    "RUN_COMMIT_SCHEMA",
    "SELECTION_RANK",
    "Stage2CalibratorGenerationV2Error",
    "VerifiedCalibratorGenerationV2",
    "VerifiedCalibratorRunV2",
    "assert_verified_calibrator_generation_v2",
    "build_resume_state_v2",
    "build_selection_record",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "input_identity_sha256",
    "normalize_input_bindings",
    "publish_calibrator_generation_v2",
    "publish_calibrator_run_v2",
    "verify_calibrator_generation_v2",
    "verify_calibrator_run_v2",
]
