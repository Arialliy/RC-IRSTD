"""Immutable checkpoint-v8 training generations and completed RC5+ runs.

Checkpoint-v8 is deployment-only.  This module keeps resumable optimizer,
history, and RNG state in a separate weights-only member, binds it byte-for-
byte to the deployment checkpoint model state, and publishes the generation
commit last.  A completed run recomputes the frozen source-only primary-budget
selection rank; no non-primary budget, outer target, query label, or official
test value is an admissible authority.
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
import tempfile
from types import MappingProxyType
from typing import Any

import torch

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from rc.stage2_calibrator_checkpoint_v8 import (
    CHECKPOINT_SCHEMA,
    SUPPORTED_METHODS,
    VerifiedCalibratorCheckpointV8,
    tensor_tree_content_sha256,
    verify_calibrator_checkpoint_v8_bytes,
)
from rc.stage2_calibrator_generation_v2 import (
    _exact_fields,
    _false,
    _float_hex,
    _integer,
    _relative_path,
    _rng_tensor,
    _root,
    _safe_tree,
    _semantic_tree_sha256,
    _sha,
    _stable_bytes,
    _text,
    _torch_bytes,
    _torch_load_safe,
    _write_new,
    canonical_json_bytes,
    canonical_json_sha256,
)
from rc.stage2_rc5plus_source_validation_view import (
    PRIMARY_SELECTION_BUDGET,
    PRIMARY_SELECTION_INDEX,
    RC5PLUS_SELECTION_GEOMETRY,
    SELECTION_RANK,
)


GENERATION_MANIFEST_SCHEMA = "rc-irstd.calibrator-generation-manifest.v3"
GENERATION_COMMIT_SCHEMA = "rc-irstd.calibrator-generation-commit.v3"
RUN_COMMIT_SCHEMA = "rc-irstd.calibrator-run-commit.v3"
RESUME_STATE_SCHEMA = "rc-irstd.calibrator-resume-state.v3"
SELECTION_RECORD_SCHEMA = "rc-irstd.calibrator-source-selection-record.v3"

GENERATION_MANIFEST_ARTIFACT = "rc_irstd_rc5plus_calibrator_generation_manifest"
GENERATION_COMMIT_ARTIFACT = "rc_irstd_rc5plus_calibrator_generation_commit"
RUN_COMMIT_ARTIFACT = "rc_irstd_rc5plus_calibrator_run_commit"

RESUME_FILENAME = "resume_state_v3.pt"
DEPLOYMENT_FILENAME = "deployment_checkpoint_v8.pt"
MANIFEST_FILENAME = "generation_manifest_v3.json"
COMMIT_FILENAME = "GENERATION_COMMIT_V3.json"
RUN_COMMIT_FILENAME = "RUN_COMMIT_V3.json"

INPUT_BINDING_NAMES = (
    "rc5plus_config",
    "training_view",
    "source_validation_view",
    "feature_mask",
    "standardizer",
    "source_reference",
    "per_image_curve_bank",
    "detector_run_complete_set",
    "seed_manifest",
    "source_release",
)

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
        "training_view_identity_sha256",
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


class Stage2RC5PlusCalibratorGenerationV3Error(ValueError):
    """An RC5+ generation/resume/run identity failed closed."""


def _binding(value: Any, name: str) -> dict[str, str]:
    row = _exact_fields(value, frozenset({"path", "sha256"}), name)
    return {
        "path": _relative_path(row["path"], f"{name}.path"),
        "sha256": _sha(row["sha256"], f"{name}.sha256"),
    }


def normalize_input_bindings_v3(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(INPUT_BINDING_NAMES):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "input_bindings key closure must equal the frozen RC5+ binding set"
        )
    return {
        name: _binding(value[name], f"input_bindings.{name}")
        for name in INPUT_BINDING_NAMES
    }


def input_identity_sha256_v3(value: Any) -> str:
    return canonical_json_sha256(normalize_input_bindings_v3(value))


def _canonical_float(value: Any, name: str, *, lower: float = 0.0) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < lower:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            f"{name} must be finite and >= {lower}"
        )
    return numeric


def build_selection_record_v3(
    *,
    macro_source_bsr: float,
    macro_source_log_excess: float,
    macro_source_pd: float,
) -> dict[str, Any]:
    bsr = _canonical_float(macro_source_bsr, "macro_source_bsr")
    excess = _canonical_float(
        macro_source_log_excess, "macro_source_log_excess"
    )
    pd = _canonical_float(macro_source_pd, "macro_source_pd")
    if bsr > 1.0 or pd > 1.0:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "macro source BSR and Pd must lie in [0,1]"
        )
    return {
        "schema_version": SELECTION_RECORD_SCHEMA,
        "selection_geometry": RC5PLUS_SELECTION_GEOMETRY,
        "grid_budget_rationals": [
            {"numerator": numerator, "denominator": denominator}
            for numerator, denominator in BUDGET_KNOT_RATIONALS
        ],
        "primary_selection_budget": {
            "numerator": PRIMARY_SELECTION_BUDGET[0],
            "denominator": PRIMARY_SELECTION_BUDGET[1],
            "grid_index": PRIMARY_SELECTION_INDEX,
        },
        "nonprimary_budgets_can_rescue_epoch_selection": False,
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
        "macro_source_bsr_hex": bsr.hex(),
        "macro_source_log_excess_hex": excess.hex(),
        "macro_source_pd_hex": pd.hex(),
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }


def _selection_record_v3(value: Any) -> dict[str, Any]:
    fields = frozenset(build_selection_record_v3(
        macro_source_bsr=0.0,
        macro_source_log_excess=0.0,
        macro_source_pd=0.0,
    ))
    row = dict(_exact_fields(value, fields, "selection_record_v3"))
    expected = build_selection_record_v3(
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
    )
    if row != expected:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "selection_record_v3 contract drifted"
        )
    return row


def build_resume_state_v3(
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
    training_view_identity_sha256: str,
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
    if method not in SUPPORTED_METHODS:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "method must be one checkpoint-v8 RC5+ method"
        )
    bindings = normalize_input_bindings_v3(input_bindings)
    state = {
        "format_version": RESUME_STATE_SCHEMA,
        "method": method,
        "run_id": _text(run_id, "run_id"),
        "outer_fold_id": _text(outer_fold_id, "outer_fold_id"),
        "outer_target_domain": _text(
            outer_target_domain, "outer_target_domain"
        ),
        "base_seed": _integer(base_seed, "base_seed"),
        "derived_seed": _integer(derived_seed, "derived_seed", 1),
        "epoch": _integer(epoch, "epoch"),
        "process_rank": _integer(process_rank, "process_rank"),
        "world_size": _integer(world_size, "world_size", 1),
        "training_contract_sha256": _sha(
            training_contract_sha256, "training_contract_sha256"
        ),
        "training_view_identity_sha256": _sha(
            training_view_identity_sha256,
            "training_view_identity_sha256",
        ),
        "input_identity_sha256": canonical_json_sha256(bindings),
        "model_state_dict": {
            str(key): tensor.detach().cpu().contiguous().clone()
            for key, tensor in model_state_dict.items()
        },
        "optimizer_state_dict": dict(optimizer_state_dict),
        "history": [dict(row) for row in history],
        "selection_record": _selection_record_v3(selection_record),
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
    _verify_resume_state_v3(
        state, expected_input_identity=state["input_identity_sha256"]
    )
    return state


def _verify_resume_state_v3(
    value: Any, *, expected_input_identity: str
) -> Mapping[str, Any]:
    state = _exact_fields(value, _RESUME_FIELDS, "resume_state_v3")
    if (
        state["format_version"] != RESUME_STATE_SCHEMA
        or state["method"] not in SUPPORTED_METHODS
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "resume_state_v3 identity mismatch"
        )
    for field in ("run_id", "outer_fold_id", "outer_target_domain"):
        _text(state[field], f"resume_state.{field}")
    _integer(state["base_seed"], "resume_state.base_seed")
    _integer(state["derived_seed"], "resume_state.derived_seed", 1)
    _integer(state["epoch"], "resume_state.epoch")
    rank = _integer(state["process_rank"], "resume_state.process_rank")
    world = _integer(state["world_size"], "resume_state.world_size", 1)
    if rank >= world:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "process_rank must be smaller than world_size"
        )
    for field in (
        "training_contract_sha256",
        "training_view_identity_sha256",
        "input_identity_sha256",
    ):
        _sha(state[field], f"resume_state.{field}")
    if state["input_identity_sha256"] != _sha(
        expected_input_identity, "expected_input_identity"
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "resume state input identity mismatch"
        )
    if not isinstance(state["model_state_dict"], Mapping) or not state[
        "model_state_dict"
    ]:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "model_state_dict must be nonempty"
        )
    if any(
        type(key) is not str or not key for key in state["model_state_dict"]
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "model_state_dict keys are invalid"
        )
    if not isinstance(state["optimizer_state_dict"], Mapping):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "optimizer_state_dict must be a mapping"
        )
    if not isinstance(state["history"], list) or len(state["history"]) != (
        state["epoch"] + 1
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "history must cover every completed epoch from zero"
        )
    if any(
        not isinstance(row, Mapping) or row.get("epoch") != index
        for index, row in enumerate(state["history"])
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "history epoch order is not contiguous"
        )
    _selection_record_v3(state["selection_record"])
    if not isinstance(state["python_rng_state"], Mapping) or not isinstance(
        state["numpy_rng_state"], Mapping
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "Python and NumPy RNG states must be explicit mappings"
        )
    _rng_tensor(state["torch_cpu_rng_state"], "torch_cpu_rng_state")
    if not isinstance(state["torch_cuda_rng_states"], list):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "torch_cuda_rng_states must be a list"
        )
    for index, tensor in enumerate(state["torch_cuda_rng_states"]):
        _rng_tensor(tensor, f"torch_cuda_rng_states[{index}]")
    _rng_tensor(state["dataloader_rng_state"], "dataloader_rng_state")
    for field in (
        "official_test_accessed",
        "outer_target_accessed",
        "query_labels_accessed",
    ):
        _false(state[field], f"resume_state.{field}")
    _safe_tree(state, "resume_state_v3")
    return state


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, init=False)
class VerifiedRC5PlusCalibratorGenerationV3:
    path: Path
    commit_sha256: str
    manifest: Mapping[str, Any]
    resume_state: Mapping[str, Any]
    deployment_checkpoint: VerifiedCalibratorCheckpointV8
    _capability: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError(
            "VerifiedRC5PlusCalibratorGenerationV3 is verifier-issued only"
        )


def _issue_generation(
    *,
    path: Path,
    commit_sha256: str,
    manifest: Mapping[str, Any],
    resume_state: Mapping[str, Any],
    deployment: VerifiedCalibratorCheckpointV8,
) -> VerifiedRC5PlusCalibratorGenerationV3:
    value = object.__new__(VerifiedRC5PlusCalibratorGenerationV3)
    for field, item in {
        "path": path,
        "commit_sha256": commit_sha256,
        "manifest": MappingProxyType(dict(manifest)),
        "resume_state": MappingProxyType(dict(resume_state)),
        "deployment_checkpoint": deployment,
        "_capability": _GENERATION_TOKEN,
    }.items():
        object.__setattr__(value, field, item)
    return value


def assert_verified_rc5plus_calibrator_generation_v3(
    value: object,
) -> VerifiedRC5PlusCalibratorGenerationV3:
    if (
        type(value) is not VerifiedRC5PlusCalibratorGenerationV3
        or getattr(value, "_capability", None) is not _GENERATION_TOKEN
    ):
        raise TypeError("a verifier-issued RC5+ generation-v3 is required")
    return value


def publish_rc5plus_calibrator_generation_v3(
    run_root: str | Path,
    *,
    resume_state: Mapping[str, Any],
    deployment_checkpoint_bytes: bytes,
    input_bindings: Mapping[str, Any],
) -> VerifiedRC5PlusCalibratorGenerationV3:
    root = _root(run_root, create=True)
    bindings = normalize_input_bindings_v3(input_bindings)
    input_identity = canonical_json_sha256(bindings)
    state = dict(
        _verify_resume_state_v3(
            resume_state, expected_input_identity=input_identity
        )
    )
    deployment = verify_calibrator_checkpoint_v8_bytes(
        deployment_checkpoint_bytes,
        expected_method=state["method"],
        expected_training_contract_sha256=state[
            "training_contract_sha256"
        ],
        expected_training_view_identity_sha256=state[
            "training_view_identity_sha256"
        ],
    )
    if tensor_tree_content_sha256(state["model_state_dict"]) != (
        deployment.payload()["model_state_content_sha256"]
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "resume and checkpoint-v8 model-state digests differ"
        )
    generation_name = (
        f"generation_v3_e{state['epoch']:06d}_r{state['process_rank']:04d}"
    )
    final = root / generation_name
    if final.exists() or final.is_symlink():
        raise FileExistsError("immutable RC5+ generation already exists")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{generation_name}.staging-", dir=root)
    )
    try:
        state_bytes = _torch_bytes(state)
        replayed_state = _torch_load_safe(state_bytes, "RC5+ resume state")
        _verify_resume_state_v3(
            replayed_state, expected_input_identity=input_identity
        )
        state_semantic_sha = _semantic_tree_sha256(replayed_state)
        _write_new(staging / RESUME_FILENAME, state_bytes)
        _write_new(
            staging / DEPLOYMENT_FILENAME, deployment_checkpoint_bytes
        )
        manifest: dict[str, Any] = {
            "schema_version": GENERATION_MANIFEST_SCHEMA,
            "artifact_type": GENERATION_MANIFEST_ARTIFACT,
            "artifact_status": "IMMUTABLE_RC5PLUS_GENERATION_COMPLETE",
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
            "training_view_identity_sha256": state[
                "training_view_identity_sha256"
            ],
            "input_bindings": bindings,
            "input_identity_sha256": input_identity,
            "resume_state": {
                "path": RESUME_FILENAME,
                "sha256": hashlib.sha256(state_bytes).hexdigest(),
                "semantic_sha256": state_semantic_sha,
                "serialization": "torch_tensors_and_primitives_weights_only",
            },
            "deployment_checkpoint": {
                "path": DEPLOYMENT_FILENAME,
                "sha256": deployment.sha256,
                "schema_version": CHECKPOINT_SCHEMA,
            },
            "selection_record": state["selection_record"],
            "checkpoint_selection_rank": list(SELECTION_RANK),
            "selection_budget": {
                "numerator": PRIMARY_SELECTION_BUDGET[0],
                "denominator": PRIMARY_SELECTION_BUDGET[1],
                "grid_index": PRIMARY_SELECTION_INDEX,
            },
            "nonprimary_budget_epoch_rescue": False,
            "official_test_accessed": False,
            "outer_target_accessed": False,
            "query_labels_accessed": False,
            "manifest_identity_sha256": "",
        }
        manifest_projection = dict(manifest)
        manifest_projection.pop("manifest_identity_sha256")
        manifest["manifest_identity_sha256"] = canonical_json_sha256(
            manifest_projection
        )
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
            "deployment_checkpoint_sha256": deployment.sha256,
            "input_identity_sha256": input_identity,
            "commit_identity_sha256": "",
        }
        commit_projection = dict(commit)
        commit_projection.pop("commit_identity_sha256")
        commit["commit_identity_sha256"] = canonical_json_sha256(
            commit_projection
        )
        commit_bytes = canonical_json_bytes(commit)
        _write_new(staging / COMMIT_FILENAME, commit_bytes)
        _fsync_directory(staging)
        os.rename(staging, final)
        staging = Path()
        _fsync_directory(root)
        return verify_rc5plus_calibrator_generation_v3(
            final, hashlib.sha256(commit_bytes).hexdigest()
        )
    finally:
        if staging != Path() and staging.exists() and staging.parent == root:
            shutil.rmtree(staging)


def verify_rc5plus_calibrator_generation_v3(
    path: str | Path,
    expected_commit_sha256: str,
) -> VerifiedRC5PlusCalibratorGenerationV3:
    generation = Path(path).expanduser()
    if generation.is_symlink():
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 must not be a symlink"
        )
    generation = generation.resolve(strict=True)
    if not generation.is_dir() or generation.name.startswith("."):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "invalid generation-v3 directory"
        )
    commit_bytes, commit_sha = _stable_bytes(
        generation / COMMIT_FILENAME, generation, "generation-v3 commit"
    )
    if commit_sha != _sha(
        expected_commit_sha256, "expected generation-v3 commit SHA"
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 external commit SHA mismatch"
        )
    try:
        commit = json.loads(commit_bytes)
    except json.JSONDecodeError as error:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 commit is not JSON"
        ) from error
    if canonical_json_bytes(commit) != commit_bytes:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 commit is not canonical JSON"
        )
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
    _exact_fields(commit, commit_fields, "generation-v3 commit")
    if (
        commit["schema_version"] != GENERATION_COMMIT_SCHEMA
        or commit["artifact_type"] != GENERATION_COMMIT_ARTIFACT
        or commit["artifact_status"] != "COMMITTED_LAST"
        or commit["generation_name"] != generation.name
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 commit identity mismatch"
        )
    commit_projection = dict(commit)
    declared_commit_identity = commit_projection.pop(
        "commit_identity_sha256"
    )
    if _sha(
        declared_commit_identity, "commit_identity_sha256"
    ) != canonical_json_sha256(commit_projection):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 commit self-hash mismatch"
        )
    manifest_binding = _binding(commit["manifest"], "commit.manifest")
    if manifest_binding["path"] != MANIFEST_FILENAME:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 manifest path mismatch"
        )
    manifest_bytes, manifest_sha = _stable_bytes(
        generation / MANIFEST_FILENAME,
        generation,
        "generation-v3 manifest",
    )
    if manifest_sha != manifest_binding["sha256"]:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 manifest SHA mismatch"
        )
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 manifest is not JSON"
        ) from error
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 manifest is not canonical JSON"
        )
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
            "training_view_identity_sha256",
            "input_bindings",
            "input_identity_sha256",
            "resume_state",
            "deployment_checkpoint",
            "selection_record",
            "checkpoint_selection_rank",
            "selection_budget",
            "nonprimary_budget_epoch_rescue",
            "official_test_accessed",
            "outer_target_accessed",
            "query_labels_accessed",
            "manifest_identity_sha256",
        }
    )
    _exact_fields(manifest, manifest_fields, "generation-v3 manifest")
    if (
        manifest["schema_version"] != GENERATION_MANIFEST_SCHEMA
        or manifest["artifact_type"] != GENERATION_MANIFEST_ARTIFACT
        or manifest["artifact_status"]
        != "IMMUTABLE_RC5PLUS_GENERATION_COMPLETE"
        or manifest["method"] not in SUPPORTED_METHODS
        or manifest["checkpoint_selection_rank"] != list(SELECTION_RANK)
        or manifest["selection_budget"]
        != {
            "numerator": PRIMARY_SELECTION_BUDGET[0],
            "denominator": PRIMARY_SELECTION_BUDGET[1],
            "grid_index": PRIMARY_SELECTION_INDEX,
        }
        or manifest["nonprimary_budget_epoch_rescue"] is not False
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 manifest identity mismatch"
        )
    manifest_projection = dict(manifest)
    declared_manifest_identity = manifest_projection.pop(
        "manifest_identity_sha256"
    )
    if _sha(
        declared_manifest_identity, "manifest_identity_sha256"
    ) != canonical_json_sha256(manifest_projection):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 manifest self-hash mismatch"
        )
    bindings = normalize_input_bindings_v3(manifest["input_bindings"])
    input_identity = canonical_json_sha256(bindings)
    if (
        manifest["input_identity_sha256"] != input_identity
        or commit["input_identity_sha256"] != input_identity
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 input identity mismatch"
        )
    for field in (
        "official_test_accessed",
        "outer_target_accessed",
        "query_labels_accessed",
    ):
        _false(manifest[field], f"manifest.{field}")
    _selection_record_v3(manifest["selection_record"])

    resume_binding = _exact_fields(
        manifest["resume_state"],
        frozenset({"path", "sha256", "semantic_sha256", "serialization"}),
        "manifest.resume_state",
    )
    if (
        resume_binding["path"] != RESUME_FILENAME
        or resume_binding["serialization"]
        != "torch_tensors_and_primitives_weights_only"
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 resume binding mismatch"
        )
    state_bytes, state_sha = _stable_bytes(
        generation / RESUME_FILENAME, generation, "resume state v3"
    )
    if (
        state_sha != _sha(resume_binding["sha256"], "resume_state.sha256")
        or state_sha
        != _sha(commit["resume_state_sha256"], "commit.resume_state_sha256")
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 resume-state SHA mismatch"
        )
    state = _torch_load_safe(state_bytes, "resume state v3")
    _verify_resume_state_v3(state, expected_input_identity=input_identity)
    if _semantic_tree_sha256(state) != _sha(
        resume_binding["semantic_sha256"], "resume_state.semantic_sha256"
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 resume semantic hash mismatch"
        )

    checkpoint_binding = _exact_fields(
        manifest["deployment_checkpoint"],
        frozenset({"path", "sha256", "schema_version"}),
        "manifest.deployment_checkpoint",
    )
    if (
        checkpoint_binding["path"] != DEPLOYMENT_FILENAME
        or checkpoint_binding["schema_version"] != CHECKPOINT_SCHEMA
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 checkpoint binding mismatch"
        )
    checkpoint_bytes, checkpoint_sha = _stable_bytes(
        generation / DEPLOYMENT_FILENAME,
        generation,
        "deployment checkpoint-v8",
    )
    if (
        checkpoint_sha
        != _sha(checkpoint_binding["sha256"], "checkpoint.sha256")
        or checkpoint_sha
        != _sha(
            commit["deployment_checkpoint_sha256"],
            "commit.deployment_checkpoint_sha256",
        )
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 checkpoint SHA mismatch"
        )
    deployment = verify_calibrator_checkpoint_v8_bytes(
        checkpoint_bytes,
        checkpoint_sha,
        expected_method=manifest["method"],
        expected_training_contract_sha256=manifest[
            "training_contract_sha256"
        ],
        expected_training_view_identity_sha256=manifest[
            "training_view_identity_sha256"
        ],
    )
    if tensor_tree_content_sha256(state["model_state_dict"]) != (
        deployment.payload()["model_state_content_sha256"]
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 resume/checkpoint model-state mismatch"
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
        "training_view_identity_sha256",
        "input_identity_sha256",
        "selection_record",
    )
    if any(state[field] != manifest[field] for field in identity_fields):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "generation-v3 resume/manifest identity mismatch"
        )
    return _issue_generation(
        path=generation,
        commit_sha256=commit_sha,
        manifest=manifest,
        resume_state=state,
        deployment=deployment,
    )


def _selection_key(
    generation: VerifiedRC5PlusCalibratorGenerationV3,
) -> tuple[float, float, float, int]:
    selection = generation.manifest["selection_record"]
    return (
        -_float_hex(selection["macro_source_bsr_hex"], "macro_source_bsr"),
        _float_hex(
            selection["macro_source_log_excess_hex"],
            "macro_source_log_excess",
        ),
        -_float_hex(selection["macro_source_pd_hex"], "macro_source_pd"),
        int(generation.manifest["epoch"]),
    )


@dataclass(frozen=True, init=False)
class VerifiedRC5PlusCalibratorRunV3:
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    generations: tuple[VerifiedRC5PlusCalibratorGenerationV3, ...]
    selected_generation: VerifiedRC5PlusCalibratorGenerationV3
    _capability: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("VerifiedRC5PlusCalibratorRunV3 is verifier-issued only")


def _issue_run(
    *,
    path: Path,
    sha256: str,
    payload: Mapping[str, Any],
    generations: Sequence[VerifiedRC5PlusCalibratorGenerationV3],
) -> VerifiedRC5PlusCalibratorRunV3:
    ordered = tuple(generations)
    selected_epoch = payload["selected_generation"]["epoch"]
    selected = next(
        item for item in ordered if item.manifest["epoch"] == selected_epoch
    )
    value = object.__new__(VerifiedRC5PlusCalibratorRunV3)
    for field, item in {
        "path": path,
        "sha256": sha256,
        "payload": MappingProxyType(dict(payload)),
        "generations": ordered,
        "selected_generation": selected,
        "_capability": _RUN_TOKEN,
    }.items():
        object.__setattr__(value, field, item)
    return value


def publish_rc5plus_calibrator_run_v3(
    run_root: str | Path,
    generations: Sequence[VerifiedRC5PlusCalibratorGenerationV3],
) -> VerifiedRC5PlusCalibratorRunV3:
    root = _root(run_root, create=False)
    ordered = tuple(
        assert_verified_rc5plus_calibrator_generation_v3(item)
        for item in generations
    )
    if not ordered or any(item.path.parent != root for item in ordered):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 needs in-root verifier-issued generations"
        )
    ordered = tuple(sorted(ordered, key=lambda item: item.manifest["epoch"]))
    if [item.manifest["epoch"] for item in ordered] != list(
        range(len(ordered))
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 generations must be contiguous from epoch zero"
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
        "training_view_identity_sha256",
        "input_identity_sha256",
        "input_bindings",
    )
    for generation in ordered[1:]:
        if any(
            generation.manifest[field] != first[field]
            for field in identity_fields
        ):
            raise Stage2RC5PlusCalibratorGenerationV3Error(
                "run-v3 generation identity mismatch"
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
        "artifact_status": "RC5PLUS_RUN_COMPLETE_SOURCE_ONLY_SELECTION",
        "method": first["method"],
        "run_id": first["run_id"],
        "outer_fold_id": first["outer_fold_id"],
        "outer_target_domain": first["outer_target_domain"],
        "base_seed": first["base_seed"],
        "derived_seed": first["derived_seed"],
        "training_contract_sha256": first["training_contract_sha256"],
        "training_view_identity_sha256": first[
            "training_view_identity_sha256"
        ],
        "input_bindings": first["input_bindings"],
        "input_identity_sha256": first["input_identity_sha256"],
        "generation_count": len(ordered),
        "generation_inventory": inventory,
        "checkpoint_selection_rank": list(SELECTION_RANK),
        "selection_budget": {
            "numerator": PRIMARY_SELECTION_BUDGET[0],
            "denominator": PRIMARY_SELECTION_BUDGET[1],
            "grid_index": PRIMARY_SELECTION_INDEX,
        },
        "nonprimary_budget_epoch_rescue": False,
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
        raise FileExistsError("immutable run-v3 commit already exists")
    _write_new(target, data)
    _fsync_directory(root)
    return verify_rc5plus_calibrator_run_v3(
        root, hashlib.sha256(data).hexdigest()
    )


def verify_rc5plus_calibrator_run_v3(
    run_root: str | Path,
    expected_run_commit_sha256: str,
) -> VerifiedRC5PlusCalibratorRunV3:
    root = _root(run_root, create=False)
    data, digest = _stable_bytes(root / RUN_COMMIT_FILENAME, root, "run-v3")
    if digest != _sha(expected_run_commit_sha256, "expected run-v3 SHA"):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 external SHA mismatch"
        )
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 commit is not JSON"
        ) from error
    if canonical_json_bytes(payload) != data:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 commit is not canonical JSON"
        )
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
            "training_view_identity_sha256",
            "input_bindings",
            "input_identity_sha256",
            "generation_count",
            "generation_inventory",
            "checkpoint_selection_rank",
            "selection_budget",
            "nonprimary_budget_epoch_rescue",
            "selected_generation",
            "official_test_accessed",
            "outer_target_accessed",
            "query_labels_accessed",
            "run_identity_sha256",
        }
    )
    _exact_fields(payload, fields, "run-v3 commit")
    if (
        payload["schema_version"] != RUN_COMMIT_SCHEMA
        or payload["artifact_type"] != RUN_COMMIT_ARTIFACT
        or payload["artifact_status"]
        != "RC5PLUS_RUN_COMPLETE_SOURCE_ONLY_SELECTION"
        or payload["method"] not in SUPPORTED_METHODS
        or payload["checkpoint_selection_rank"] != list(SELECTION_RANK)
        or payload["selection_budget"]
        != {
            "numerator": PRIMARY_SELECTION_BUDGET[0],
            "denominator": PRIMARY_SELECTION_BUDGET[1],
            "grid_index": PRIMARY_SELECTION_INDEX,
        }
        or payload["nonprimary_budget_epoch_rescue"] is not False
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 identity mismatch"
        )
    for field in (
        "official_test_accessed",
        "outer_target_accessed",
        "query_labels_accessed",
    ):
        _false(payload[field], f"run-v3.{field}")
    projection = dict(payload)
    declared_identity = projection.pop("run_identity_sha256")
    if _sha(declared_identity, "run_identity_sha256") != (
        canonical_json_sha256(projection)
    ):
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 self-hash mismatch"
        )
    bindings = normalize_input_bindings_v3(payload["input_bindings"])
    if canonical_json_sha256(bindings) != payload["input_identity_sha256"]:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 input identity mismatch"
        )
    inventory = payload["generation_inventory"]
    count = _integer(payload["generation_count"], "generation_count", 1)
    if not isinstance(inventory, list) or len(inventory) != count:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 generation inventory count mismatch"
        )
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
    generations: list[VerifiedRC5PlusCalibratorGenerationV3] = []
    for index, raw in enumerate(inventory):
        row = _exact_fields(
            raw, inventory_fields, f"generation_inventory[{index}]"
        )
        if row["epoch"] != index or row["path"] != (
            f"generation_v3_e{index:06d}_r0000/{COMMIT_FILENAME}"
        ):
            raise Stage2RC5PlusCalibratorGenerationV3Error(
                "run-v3 requires contiguous rank-zero generations"
            )
        generation = verify_rc5plus_calibrator_generation_v3(
            root / PurePosixPath(row["path"]).parent,
            _sha(row["commit_sha256"], "inventory commit SHA"),
        )
        if (
            generation.manifest["manifest_identity_sha256"]
            != row["manifest_identity_sha256"]
            or generation.manifest["deployment_checkpoint"]["sha256"]
            != row["deployment_checkpoint_sha256"]
            or generation.manifest["selection_record"]
            != row["selection_record"]
        ):
            raise Stage2RC5PlusCalibratorGenerationV3Error(
                "run-v3 generation inventory drifted"
            )
        for field in (
            "method",
            "run_id",
            "outer_fold_id",
            "outer_target_domain",
            "base_seed",
            "derived_seed",
            "training_contract_sha256",
            "training_view_identity_sha256",
            "input_identity_sha256",
        ):
            if generation.manifest[field] != payload[field]:
                raise Stage2RC5PlusCalibratorGenerationV3Error(
                    f"run-v3 generation identity mismatch: {field}"
                )
        generations.append(generation)
    selected = min(generations, key=_selection_key)
    expected_selected = {
        "epoch": selected.manifest["epoch"],
        "commit_sha256": selected.commit_sha256,
        "deployment_checkpoint_sha256": selected.manifest[
            "deployment_checkpoint"
        ]["sha256"],
    }
    if payload["selected_generation"] != expected_selected:
        raise Stage2RC5PlusCalibratorGenerationV3Error(
            "run-v3 selected generation is not the primary-budget winner"
        )
    return _issue_run(
        path=root / RUN_COMMIT_FILENAME,
        sha256=digest,
        payload=payload,
        generations=generations,
    )


__all__ = [
    "COMMIT_FILENAME",
    "DEPLOYMENT_FILENAME",
    "GENERATION_COMMIT_SCHEMA",
    "GENERATION_MANIFEST_SCHEMA",
    "INPUT_BINDING_NAMES",
    "MANIFEST_FILENAME",
    "RESUME_FILENAME",
    "RESUME_STATE_SCHEMA",
    "RUN_COMMIT_FILENAME",
    "RUN_COMMIT_SCHEMA",
    "SELECTION_RECORD_SCHEMA",
    "Stage2RC5PlusCalibratorGenerationV3Error",
    "VerifiedRC5PlusCalibratorGenerationV3",
    "VerifiedRC5PlusCalibratorRunV3",
    "assert_verified_rc5plus_calibrator_generation_v3",
    "build_resume_state_v3",
    "build_selection_record_v3",
    "input_identity_sha256_v3",
    "normalize_input_bindings_v3",
    "publish_rc5plus_calibrator_generation_v3",
    "publish_rc5plus_calibrator_run_v3",
    "verify_rc5plus_calibrator_generation_v3",
    "verify_rc5plus_calibrator_run_v3",
]
