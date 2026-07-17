"""Export hash-bound, label-free Stage2 development scores.

The exporter consumes only an explicit Stage2 selection contract.  It does
not discover dataset splits, enumerate masks, attach labels, or inspect any
evaluation metric.  Every selected image is exported in contract order with
both a float64 sigmoid probability map and a float64 raw-logit diagnostic map
restored to native resolution.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TVF

from data_ext.dataset_meta import build_spatial_transform, restore_tensor_to_original
from data_ext.eval_dataset import IMAGENET_MEAN, IMAGENET_STD
from data_ext.stage2_score_manifest import (
    BASE_SEEDS,
    BINDING_NAMES,
    OUTER_FOLD_TARGETS,
    STAGE2_DEVELOPMENT_ROLES,
    STAGE2_DOMAINS,
    STAGE2_SCORE_ARTIFACT_TYPE,
    STAGE2_SCORE_MANIFEST_SCHEMA,
    STAGE2_SCORE_RECORDS_ALGORITHM,
    STRICT_THRESHOLD_SEMANTICS,
    _selection_records as _project_selection_records,
    _verify_stage2_score_npz,
    stage2_score_records_sha256,
    verify_stage2_score_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _BoundFile:
    name: str
    path: Path
    sha256: str
    identity: tuple[int, int, int]


@dataclass(frozen=True)
class _PreparedImage:
    record: Mapping[str, Any]
    bound_file: _BoundFile
    original_hw: tuple[int, int]
    output_name: str


@dataclass(frozen=True)
class _OutputBundle:
    final_root: Path
    staging_root: Path
    lock_path: Path
    score_names: tuple[str, ...]
    staging_identity: tuple[int, int]
    lock_identity: tuple[int, int]


def export_stage2_development_scores(
    selection_contract: str | Path,
    run_contract: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    selection_contract_sha256: str,
    run_contract_sha256: str,
    checkpoint_sha256: str,
    role: str,
    device: str = "cuda",
    repository_root: str | Path | None = None,
    model_factory: Callable[[Mapping[str, Any], torch.device], torch.nn.Module]
    | None = None,
) -> dict[str, Any]:
    """Run label-free inference for one exact Stage2 development selection.

    ``model_factory`` is a dependency-injection seam for synthetic contract
    tests.  Production callers omit it and receive the repository detector.
    It cannot change selection, identity, hashing, geometry or output checks.
    """

    root = _repository_root(repository_root)
    role = _role(role)
    selection_path, selection_payload = _load_expected_json(
        selection_contract,
        selection_contract_sha256,
        root,
        "selection contract",
    )
    run_path, run_payload = _load_expected_json(
        run_contract,
        run_contract_sha256,
        root,
        "run contract",
    )
    selection_sha = _sha256(selection_contract_sha256, "selection_contract_sha256")
    run_sha = _sha256(run_contract_sha256, "run_contract_sha256")
    weight_path = _existing_path(checkpoint, root, "checkpoint")
    weight_sha = _sha256(checkpoint_sha256, "checkpoint_sha256")
    if _hash_file(weight_path) != weight_sha:
        raise ValueError("checkpoint SHA-256 mismatch")

    _strict_development_contract(selection_payload, "selection contract")
    _strict_development_contract(run_payload, "run contract")
    records = _validate_contract_identity(
        selection_payload,
        run_payload,
        role=role,
        selection_path=_repo_path(selection_path, root),
        selection_sha256=selection_sha,
    )
    run_bindings = _run_input_bindings(run_payload, root)
    materialization_bindings = _materialization_artifact_bindings(run_payload, root)
    prepared_images = _preflight_selected_images(records, root)

    checkpoint_before = _hash_file(weight_path)
    checkpoint_payload = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True,
    )
    if _hash_file(weight_path) != checkpoint_before:
        raise RuntimeError("checkpoint changed while it was loaded")
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    runtime_bindings = _checkpoint_runtime_bindings(
        checkpoint_payload,
        checkpoint_path=weight_path,
        run_payload=run_payload,
        run_path=run_path,
        run_sha256=run_sha,
        root=root,
    )
    _verify_checkpoint_identity(
        checkpoint_payload,
        run_payload=run_payload,
        run_sha256=run_sha,
        checkpoint_sha256=weight_sha,
        runtime_config_sha256=runtime_bindings["runtime_config"]["sha256"],
    )

    input_hw, resize_mode = _inference_geometry(checkpoint_payload)
    bindings = {
        "selection_contract": {
            "path": _repo_path(selection_path, root),
            "sha256": selection_sha,
        },
        "run_contract": {
            "path": _repo_path(run_path, root),
            "sha256": run_sha,
        },
        "checkpoint": {
            "path": _repo_path(weight_path, root),
            "sha256": weight_sha,
        },
        "detector_config": run_bindings["detector_config"],
        "runtime_config": runtime_bindings["runtime_config"],
        "seed_manifest": run_bindings["seed_manifest"],
        "materialization_index": run_bindings["materialization_index"],
        "release_artifact": run_bindings["release_artifact"],
        "environment_artifact": runtime_bindings["environment_artifact"],
        "runtime_contract": runtime_bindings["runtime_contract"],
    }
    if tuple(bindings) != BINDING_NAMES:
        raise RuntimeError("internal Stage2 binding order/closure mismatch")
    bound_files = _snapshot_bound_files(
        bindings,
        materialization_bindings,
        root,
    )
    _verify_prepared_images(prepared_images, root)
    bundle = _prepare_output_bundle(
        output_dir,
        root,
        tuple(prepared.output_name for prepared in prepared_images),
    )
    marker = bundle.staging_root / ".export_incomplete"
    published_identity: tuple[int, int] | None = None
    try:
        _write_text_atomic(
            marker,
            "Stage2 development score export is incomplete; do not consume.\n",
        )
        selected_device = _select_device(device)
        model = (
            _default_model_factory(checkpoint_payload, selected_device)
            if model_factory is None
            else model_factory(checkpoint_payload, selected_device)
        )
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model_factory must return torch.nn.Module")
        model.to(selected_device)
        model.eval()

        # A model factory is allowed only to construct the detector.  Recheck
        # every byte-bound input and selected image immediately before the
        # first forward call so construction cannot invalidate preflight.
        _verify_bound_files(bound_files)
        _verify_prepared_images(prepared_images, root)
        manifest_records = _export_records(
            prepared_images,
            source_domain=_source_domain(selection_payload),
            root=root,
            staging_root=bundle.staging_root,
            published_root=bundle.final_root,
            input_hw=input_hw,
            resize_mode=resize_mode,
            model=model,
            device=selected_device,
        )
        manifest = _build_manifest(
            selection_payload,
            run_payload,
            role=role,
            input_hw=input_hw,
            resize_mode=resize_mode,
            bindings=bindings,
            records=manifest_records,
        )
        staged_manifest_path = bundle.staging_root / "manifest.json"
        _write_json_atomic(staged_manifest_path, manifest)

        _verify_bound_files(bound_files)
        _verify_prepared_images(prepared_images, root)
        manifest_sha = _hash_file(staged_manifest_path)
        _write_text_atomic(
            staged_manifest_path.with_name(staged_manifest_path.name + ".sha256"),
            f"{manifest_sha}  {staged_manifest_path.name}\n",
        )
        _preflight_staged_bundle(
            bundle,
            manifest,
            manifest_sha,
            manifest_records,
        )
        marker.unlink()
        _fsync_directory(bundle.staging_root)
        published_identity = _atomic_publish_directory_no_replace(
            bundle.staging_root,
            bundle.final_root,
        )
        manifest_path = bundle.final_root / "manifest.json"
        verify_stage2_score_manifest(
            manifest_path,
            manifest_sha,
            role,
            repository_root=root,
        )
        _verify_bound_files(bound_files)
        _verify_prepared_images(prepared_images, root)
        _verify_published_bundle(
            bundle,
            published_identity,
            manifest_sha,
            manifest_records,
        )
        return manifest
    except BaseException:
        if published_identity is None:
            _remove_owned_directory(
                bundle.staging_root,
                bundle.staging_identity,
            )
        else:
            _remove_owned_directory(
                bundle.final_root,
                published_identity,
            )
        raise
    finally:
        _remove_owned_directory(
            bundle.staging_root,
            bundle.staging_identity,
        )
        _release_owned_lock(bundle.lock_path, bundle.lock_identity)


def _strict_development_contract(payload: Mapping[str, Any], name: str) -> None:
    guards = payload.get("guardrails")
    if "development_only" in payload:
        if type(payload["development_only"]) is not bool or payload[
            "development_only"
        ] is not True:
            raise TypeError(f"{name} development_only must be exact true")
    elif not isinstance(guards, Mapping) or type(guards.get("development_only")) is not bool or guards.get("development_only") is not True:
        raise TypeError(f"{name} guardrails.development_only must be exact true")
    if "official_test_accessed" in payload:
        if type(payload["official_test_accessed"]) is not bool or payload[
            "official_test_accessed"
        ] is not False:
            raise TypeError(f"{name} official_test_accessed must be exact false")
    else:
        if not isinstance(guards, Mapping):
            raise TypeError(f"{name} must provide official-test guardrails")
        for field in (
            "official_test_split_files_opened",
            "official_test_ids_materialized",
            "official_test_images_opened",
        ):
            if type(guards.get(field)) is not bool or guards.get(field) is not False:
                raise TypeError(f"{name} guardrails.{field} must be exact false")
    if "execution_authorized" in payload and (
        type(payload["execution_authorized"]) is not bool
        or payload["execution_authorized"] is not False
    ):
        raise TypeError(f"{name} execution_authorized must be exact false")
    if "observed_results" in payload and payload["observed_results"] is not None:
        raise ValueError(f"{name} observed_results must be exactly null")


def _validate_contract_identity(
    selection: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    role: str,
    selection_path: str,
    selection_sha256: str,
) -> list[Mapping[str, Any]]:
    outer_fold = _string(run.get("outer_fold_id"), "run.outer_fold_id")
    if outer_fold not in OUTER_FOLD_TARGETS:
        raise ValueError("selection outer_fold_id is not frozen")
    outer_target = _first(run, "outer_target", "outer_target_domain")
    if outer_target != OUTER_FOLD_TARGETS[outer_fold]:
        raise ValueError("selection outer-fold/target identity mismatch")
    source_domain = _source_domain(selection)
    if source_domain not in STAGE2_DOMAINS:
        raise ValueError("selection source_domain is not frozen")
    raw_sources = run.get("source_domains")
    if (
        not isinstance(raw_sources, list)
        or len(raw_sources) != 2
        or len(set(raw_sources)) != 2
        or any(source not in STAGE2_DOMAINS for source in raw_sources)
    ):
        raise ValueError("run.source_domains must be two unique frozen domains")
    if outer_target in raw_sources:
        raise ValueError("run detector sources include the outer target")
    base_seed = _exact_int(run.get("base_seed"), "run.base_seed", 0)
    if base_seed not in BASE_SEEDS:
        raise ValueError("selection base_seed is not frozen")
    _exact_int(run.get("derived_seed"), "run.derived_seed", 0)
    detector_role = _string(run.get("detector_role"), "run.detector_role")
    expected_detector_role = (
        "detector_oof"
        if role in {"oof_train_source_reference", "oof_holdout_stage2_fit"}
        else "detector_full_fit"
    )
    if detector_role != expected_detector_role:
        raise ValueError("score role and selection detector_role disagree")
    oof_index = run.get("oof_fold_index")
    if detector_role == "detector_oof":
        if _exact_int(oof_index, "selection.oof_fold_index", 0) not in {0, 1}:
            raise ValueError("OOF fold index must be 0 or 1")
    elif oof_index is not None:
        raise ValueError("full-fit selection oof_fold_index must be null")
    if role == "outer_target_diagnostic_development":
        if source_domain != outer_target:
            raise ValueError("outer-target score selection must be the outer target")
        if source_domain in raw_sources:
            raise ValueError("outer target appears in detector source domains")
    elif source_domain == outer_target:
        raise ValueError("source/OOF score selection includes the outer target")
    elif source_domain not in raw_sources:
        raise ValueError("source score selection is absent from detector sources")

    for field, aliases in {
        "outer_fold_id": ("outer_fold_id",),
        "outer_target": ("outer_target", "outer_target_domain"),
        "base_seed": ("base_seed",),
        "derived_seed": ("derived_seed",),
        "detector_role": ("detector_role",),
        "oof_fold_index": ("oof_fold_index",),
    }.items():
        selected_value = next(
            (selection[name] for name in aliases if name in selection), _MISSING
        )
        run_value = _first(run, *aliases)
        if selected_value is not _MISSING and selected_value != run_value:
            raise ValueError(f"selection/run {field} identity mismatch")

    selection_type = selection.get("artifact_type")
    if selection_type == "rc_irstd_stage2_detector_selection":
        if role not in {
            "oof_train_source_reference",
            "fullfit_detector_fit_source_reference",
        }:
            raise ValueError("training selection cannot feed fit/diagnostic scores")
        run_selections = run.get("selection_contracts")
        if not isinstance(run_selections, list) or len(run_selections) != 2:
            raise ValueError("run contract must contain exactly two selections")
        matches = sum(
            1
            for item in run_selections
            if isinstance(item, Mapping)
            and item.get("path") == selection_path
            and item.get("sha256") == selection_sha256
        )
        if matches != 1:
            raise ValueError("training selection is not bound exactly once by run")
    else:
        run_bindings = run.get("bindings")
        if not isinstance(run_bindings, Mapping):
            raise TypeError("run.bindings must be an object")
        materialized = run_bindings.get("materialization_artifacts_sha256")
        if not isinstance(materialized, Mapping) or materialized.get(
            selection_path
        ) != selection_sha256:
            raise ValueError("role selection is not bound by materialization index")

    raw_records = _project_selection_records(
        selection,
        role=role,
        oof_fold_index=oof_index,
    )
    result: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"selection.records[{index}] must be an object")
        required = {
            "canonical_id",
            "image_id",
            "original_image_path",
            "original_image_sha256",
            "exclusion_group_id",
            "near_duplicate_cluster_id_or_unique_sentinel",
            "source_role_record_index",
        }
        missing = required.difference(raw)
        if missing:
            raise KeyError(f"selection.records[{index}] missing {sorted(missing)}")
        canonical_id = _string(raw["canonical_id"], "canonical_id")
        image_path = _relative_path(raw["original_image_path"], "original_image_path")
        _sha256(raw["original_image_sha256"], "original_image_sha256")
        _string(raw["image_id"], "image_id")
        _string(raw["exclusion_group_id"], "exclusion_group_id")
        _string(
            raw["near_duplicate_cluster_id_or_unique_sentinel"],
            "near_duplicate_cluster_id_or_unique_sentinel",
        )
        _exact_int(raw["source_role_record_index"], "source_role_record_index", 0)
        if canonical_id in seen_ids or image_path in seen_paths:
            raise ValueError("selection contains duplicate canonical IDs or image paths")
        seen_ids.add(canonical_id)
        seen_paths.add(image_path)
        result.append(raw)
    return result


def _run_input_bindings(
    run: Mapping[str, Any],
    root: Path,
) -> dict[str, dict[str, str]]:
    raw_bindings = run.get("bindings")
    if not isinstance(raw_bindings, Mapping):
        raise TypeError("run.bindings must be an object")
    result: dict[str, dict[str, str]] = {}
    for name in (
        "detector_config",
        "seed_manifest",
        "materialization_index",
        "release_artifact",
    ):
        raw = raw_bindings.get(name)
        if not isinstance(raw, Mapping):
            raise TypeError(f"run.bindings.{name} must be an object")
        relative = _relative_path(raw.get("path"), f"run.bindings.{name}.path")
        digest = _sha256(raw.get("sha256"), f"run.bindings.{name}.sha256")
        artifact = _repository_file(root, relative, name)
        if _hash_file(artifact) != digest:
            raise ValueError(f"run binding {name} SHA-256 mismatch")
        result[name] = {"path": relative, "sha256": digest}
    return result


def _materialization_artifact_bindings(
    run: Mapping[str, Any],
    root: Path,
) -> dict[str, str]:
    raw_bindings = run.get("bindings")
    if not isinstance(raw_bindings, Mapping):
        raise TypeError("run.bindings must be an object")
    raw_artifacts = raw_bindings.get("materialization_artifacts_sha256")
    if not isinstance(raw_artifacts, Mapping):
        raise TypeError(
            "run.bindings.materialization_artifacts_sha256 must be an object"
        )
    result: dict[str, str] = {}
    for index, (raw_path, raw_sha) in enumerate(raw_artifacts.items()):
        relative = _relative_path(
            raw_path,
            f"materialization_artifacts_sha256[{index}].path",
        )
        digest = _sha256(
            raw_sha,
            f"materialization_artifacts_sha256[{index}].sha256",
        )
        artifact = _repository_file(root, relative, "materialization artifact")
        before = _hash_file(artifact)
        if before != digest or _hash_file(artifact) != before:
            raise ValueError(
                f"materialization artifact SHA-256 mismatch: {relative}"
            )
        result[relative] = digest
    return result


def _preflight_selected_images(
    records: Sequence[Mapping[str, Any]],
    root: Path,
) -> tuple[_PreparedImage, ...]:
    prepared: list[_PreparedImage] = []
    used_names: set[str] = set()
    for index, record in enumerate(records):
        relative = _relative_path(
            record["original_image_path"],
            f"selection.records[{index}].original_image_path",
        )
        image_path = _repository_file(root, relative, "selected image")
        digest = _sha256(
            record["original_image_sha256"],
            f"selection.records[{index}].original_image_sha256",
        )
        bound = _snapshot_bound_file(f"selected_image[{index}]", image_path, digest)
        with Image.open(image_path) as image:
            original_hw = (int(image.height), int(image.width))
            image.verify()
        _verify_bound_file(bound)
        output_name = _safe_score_name(record["canonical_id"])
        if output_name in used_names:
            raise ValueError(f"duplicate score output name: {output_name}")
        used_names.add(output_name)
        prepared.append(
            _PreparedImage(
                record=record,
                bound_file=bound,
                original_hw=original_hw,
                output_name=output_name,
            )
        )
    return tuple(prepared)


def _snapshot_bound_files(
    bindings: Mapping[str, Mapping[str, str]],
    materialization_bindings: Mapping[str, str],
    root: Path,
) -> tuple[_BoundFile, ...]:
    snapshots: list[_BoundFile] = []
    by_path: dict[Path, _BoundFile] = {}
    candidates = [
        (name, binding["path"], binding["sha256"])
        for name, binding in bindings.items()
    ]
    candidates.extend(
        (
            f"materialization_artifact[{index}]",
            relative,
            digest,
        )
        for index, (relative, digest) in enumerate(materialization_bindings.items())
    )
    for name, relative, digest in candidates:
        artifact = _repository_file(root, relative, name)
        existing = by_path.get(artifact)
        if existing is not None:
            if existing.sha256 != digest:
                raise ValueError(
                    f"one bound path has conflicting SHA-256 values: {relative}"
                )
            continue
        snapshot = _snapshot_bound_file(name, artifact, digest)
        by_path[artifact] = snapshot
        snapshots.append(snapshot)
    return tuple(snapshots)


def _snapshot_bound_file(name: str, path: Path, digest: str) -> _BoundFile:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    identity = (int(before.st_dev), int(before.st_ino), int(before.st_size))
    if _hash_file(path) != digest:
        raise ValueError(f"{name} SHA-256 mismatch")
    after = path.stat(follow_symlinks=False)
    if (int(after.st_dev), int(after.st_ino), int(after.st_size)) != identity:
        raise RuntimeError(f"{name} changed while snapshotted")
    return _BoundFile(name=name, path=path, sha256=digest, identity=identity)


def _verify_bound_file(bound: _BoundFile) -> None:
    before = bound.path.stat(follow_symlinks=False)
    identity = (int(before.st_dev), int(before.st_ino), int(before.st_size))
    if not stat.S_ISREG(before.st_mode) or identity != bound.identity:
        raise RuntimeError(f"bound file identity changed: {bound.name}")
    if _hash_file(bound.path) != bound.sha256:
        raise RuntimeError(f"bound file bytes changed: {bound.name}")
    after = bound.path.stat(follow_symlinks=False)
    if (int(after.st_dev), int(after.st_ino), int(after.st_size)) != bound.identity:
        raise RuntimeError(f"bound file changed while rehashed: {bound.name}")


def _verify_bound_files(bound_files: Sequence[_BoundFile]) -> None:
    for bound in bound_files:
        _verify_bound_file(bound)


def _verify_prepared_images(
    prepared_images: Sequence[_PreparedImage],
    root: Path,
) -> None:
    for index, prepared in enumerate(prepared_images):
        current = _repository_file(
            root,
            prepared.record["original_image_path"],
            f"selected image[{index}]",
        )
        if current != prepared.bound_file.path:
            raise RuntimeError(f"selected image path identity changed at index {index}")
        _verify_bound_file(prepared.bound_file)
        with Image.open(current) as image:
            current_hw = (int(image.height), int(image.width))
            image.verify()
        if current_hw != prepared.original_hw:
            raise RuntimeError(f"selected image geometry changed at index {index}")
        _verify_bound_file(prepared.bound_file)


def _checkpoint_runtime_bindings(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    run_payload: Mapping[str, Any],
    run_path: Path,
    run_sha256: str,
    root: Path,
) -> dict[str, dict[str, str]]:
    runtime = checkpoint.get("stage2_runtime_artifacts")
    if not isinstance(runtime, Mapping):
        raise ValueError("checkpoint is missing stage2_runtime_artifacts")
    result: dict[str, dict[str, str]] = {}
    runtime_names = {
        "runtime_config": "run_config",
        "environment_artifact": "environment_artifact",
        "runtime_contract": "runtime_contract",
    }
    local_bindings: dict[str, Mapping[str, Any]] = {}
    for manifest_name, checkpoint_name in runtime_names.items():
        raw = runtime.get(checkpoint_name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"checkpoint is missing {checkpoint_name} binding")
        raw_path = _relative_path(raw.get("path"), f"{checkpoint_name}.path")
        digest = _sha256(raw.get("sha256"), f"{checkpoint_name}.sha256")
        candidate = checkpoint_path.parent.joinpath(*PurePosixPath(raw_path).parts)
        artifact = _existing_path(candidate, root, checkpoint_name)
        if _hash_file(artifact) != digest:
            raise ValueError(f"{checkpoint_name} SHA-256 mismatch")
        result[manifest_name] = {
            "path": _repo_path(artifact, root),
            "sha256": digest,
        }
        local_bindings[checkpoint_name] = raw

    runtime_contract_path = _repository_file(
        root, result["runtime_contract"]["path"], "runtime contract"
    )
    runtime_before = _hash_file(runtime_contract_path)
    with runtime_contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if _hash_file(runtime_contract_path) != runtime_before:
        raise RuntimeError("runtime contract changed while read")
    if not isinstance(contract, Mapping):
        raise TypeError("runtime contract must contain a JSON object")
    if contract.get("schema_version") != "rc-irstd.stage2-detector-runtime-contract.v1":
        raise ValueError("runtime contract schema mismatch")
    if type(contract.get("development_only")) is not bool or contract.get(
        "development_only"
    ) is not True:
        raise TypeError("runtime contract development_only must be exact true")
    if type(contract.get("official_test_accessed")) is not bool or contract.get(
        "official_test_accessed"
    ) is not False:
        raise TypeError("runtime contract official_test_accessed must be exact false")
    if contract.get("observed_results") is not None:
        raise ValueError("runtime contract observed_results must be exactly null")
    input_run = contract.get("input_run_contract")
    if not isinstance(input_run, Mapping) or input_run.get("path") != _repo_path(
        run_path, root
    ) or input_run.get("sha256") != run_sha256:
        raise ValueError("runtime contract input-run binding mismatch")
    for field, local_name in (
        ("run_config", "run_config"),
        ("environment_artifact", "environment_artifact"),
    ):
        declared = contract.get(field)
        local = local_bindings[local_name]
        if not isinstance(declared, Mapping):
            raise TypeError(f"runtime contract {field} must be an object")
        if declared.get("path") != local.get("path") or declared.get(
            "sha256"
        ) != local.get("sha256"):
            raise ValueError(f"runtime contract {field} binding mismatch")
    for contract_field, run_fields in (
        ("outer_fold_id", ("outer_fold_id",)),
        ("outer_target_domain", ("outer_target_domain", "outer_target")),
        ("detector_role", ("detector_role",)),
        ("oof_fold_index", ("oof_fold_index",)),
        ("base_seed", ("base_seed",)),
        ("derived_seed", ("derived_seed",)),
    ):
        if contract.get(contract_field) != _first(run_payload, *run_fields):
            raise ValueError(f"runtime contract {contract_field} identity mismatch")
    return result


def _verify_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    *,
    run_payload: Mapping[str, Any],
    run_sha256: str,
    checkpoint_sha256: str,
    runtime_config_sha256: str,
) -> None:
    if checkpoint.get("format_version") != "rc-irstd.detector-inference.v1":
        raise ValueError("checkpoint is not the restricted Stage2 inference schema")
    if type(checkpoint.get("official_test_accessed")) is not bool or checkpoint.get(
        "official_test_accessed"
    ) is not False:
        raise TypeError("checkpoint official_test_accessed must be exact false")
    if checkpoint.get("run_contract_sha256") != run_sha256:
        raise ValueError("checkpoint run_contract_sha256 mismatch")
    if checkpoint.get("run_config_sha256") != runtime_config_sha256:
        raise ValueError("checkpoint run_config_sha256 mismatch")
    for checkpoint_fields, run_fields, label in (
        (("outer_fold_id",), ("outer_fold_id",), "outer_fold_id"),
        (("outer_target", "outer_target_domain"), ("outer_target", "outer_target_domain"), "outer_target"),
        (("seed", "derived_seed"), ("derived_seed",), "derived_seed"),
        (("detector_role",), ("detector_role",), "detector_role"),
        (("oof_fold_index",), ("oof_fold_index",), "oof_fold_index"),
    ):
        checkpoint_value = _first(checkpoint, *checkpoint_fields)
        run_value = _first(run_payload, *run_fields)
        if checkpoint_value != run_value:
            raise ValueError(f"checkpoint/run {label} mismatch")
    if checkpoint.get("checkpoint_selection") != "fixed_last_no_test_or_target_validation":
        raise ValueError("checkpoint is not fixed-last")
    sources = checkpoint.get("source_names")
    if not isinstance(sources, list) or sources != run_payload.get("source_domains"):
        raise ValueError("checkpoint source_names differ from run source_domains")
    held_out = checkpoint.get("held_out_domains")
    outer_target = _first(run_payload, "outer_target_domain", "outer_target")
    if not isinstance(held_out, list) or outer_target not in held_out or set(
        held_out
    ).intersection(sources):
        raise ValueError("checkpoint held-out/source-domain identity mismatch")
    recorded_sha = checkpoint.get("checkpoint_sha256")
    if recorded_sha is not None and recorded_sha != checkpoint_sha256:
        raise ValueError("checkpoint self-declared SHA-256 mismatch")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("checkpoint state_dict must be non-empty tensors")
    _exact_int(checkpoint.get("epoch"), "checkpoint.epoch", 0)


def _inference_geometry(checkpoint: Mapping[str, Any]) -> tuple[tuple[int, int], str]:
    geometry = checkpoint.get("inference_geometry")
    if not isinstance(geometry, Mapping) or set(geometry) != {
        "input_hw",
        "resize_mode",
    }:
        raise ValueError("checkpoint requires exact inference_geometry")
    raw_hw = geometry["input_hw"]
    if not isinstance(raw_hw, (list, tuple)) or len(raw_hw) != 2:
        raise TypeError("inference_geometry.input_hw must be [H, W]")
    geometry_hw = (
        _exact_int(raw_hw[0], "inference_geometry.input_hw[0]", 1),
        _exact_int(raw_hw[1], "inference_geometry.input_hw[1]", 1),
    )
    geometry_mode = geometry["resize_mode"]
    training = checkpoint.get("training_args")
    if not isinstance(training, Mapping):
        raise ValueError("checkpoint training_args is required for inference geometry")
    raw_size = training.get("base_size", training.get("input_size", 256))
    if type(raw_size) is int:
        input_hw = (raw_size, raw_size)
    elif isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
        input_hw = (
            _exact_int(raw_size[0], "training_args.base_size[0]", 1),
            _exact_int(raw_size[1], "training_args.base_size[1]", 1),
        )
    else:
        raise TypeError("training_args base_size must be an integer or [H, W]")
    if any(value <= 0 or value % 16 for value in input_hw):
        raise ValueError("detector input dimensions must be positive multiples of 16")
    resize_mode = training.get("resize_mode", "resize")
    if resize_mode not in {"resize", "letterbox"}:
        raise ValueError("training_args resize_mode must be resize or letterbox")
    if geometry_hw != input_hw or geometry_mode != resize_mode:
        raise ValueError("inference_geometry disagrees with training_args")
    return input_hw, resize_mode


def _default_model_factory(
    checkpoint: Mapping[str, Any], device: torch.device
) -> torch.nn.Module:
    from model.MSHNet import MSHNet

    state: object = checkpoint
    for field in ("state_dict", "model_state_dict", "model_state", "net"):
        if field in checkpoint:
            state = checkpoint[field]
            break
    if not isinstance(state, Mapping) or not state:
        raise TypeError("checkpoint does not contain a non-empty model state mapping")
    normalised: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("checkpoint model state contains non-tensor values")
        rendered = str(key)
        normalised[rendered[7:] if rendered.startswith("module.") else rendered] = value
    model = MSHNet(3)
    model.load_state_dict(normalised, strict=True)
    return model.to(device)


def _export_records(
    prepared_images: Sequence[_PreparedImage],
    *,
    source_domain: str,
    root: Path,
    staging_root: Path,
    published_root: Path,
    input_hw: tuple[int, int],
    resize_mode: str,
    model: torch.nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, prepared in enumerate(prepared_images):
            selected = prepared.record
            image_path = _repository_file(
                root, selected["original_image_path"], "selected image"
            )
            if image_path != prepared.bound_file.path:
                raise RuntimeError(f"selected image path changed at index {index}")
            _verify_bound_file(prepared.bound_file)
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            original_hw = (int(image.height), int(image.width))
            if original_hw != prepared.original_hw:
                raise RuntimeError(f"selected image geometry changed at index {index}")
            _verify_bound_file(prepared.bound_file)
            transform = build_spatial_transform(original_hw, input_hw, resize_mode)
            transformed = _apply_transform(image, transform)
            image_tensor = TVF.normalize(
                TVF.to_tensor(transformed),
                tuple(float(x) for x in IMAGENET_MEAN),
                tuple(float(x) for x in IMAGENET_STD),
            ).unsqueeze(0).to(device)
            output = model(image_tensor, True)
            logits = _extract_logits(output)
            if logits.ndim != 4 or tuple(logits.shape[:2]) != (1, 1):
                raise ValueError(
                    f"detector logits must be [1,1,H,W], got {tuple(logits.shape)}"
                )
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("detector logits contain NaN/Inf")
            raw_logit = restore_tensor_to_original(
                logits[0, 0].to(torch.float64), transform, mode="bilinear"
            )
            probability = restore_tensor_to_original(
                torch.sigmoid(logits[0, 0].to(torch.float64)),
                transform,
                mode="bilinear",
            )
            raw_array = raw_logit.detach().cpu().numpy().astype(np.float64, copy=False)
            prob_array = (
                probability.clamp(0.0, 1.0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            if raw_array.shape != original_hw or prob_array.shape != original_hw:
                raise RuntimeError("native-resolution restoration failed")
            if not np.isfinite(raw_array).all() or not np.isfinite(prob_array).all():
                raise ValueError("restored score arrays contain NaN/Inf")

            _verify_bound_file(prepared.bound_file)
            output_name = prepared.output_name
            score_path = staging_root / output_name
            _write_npz_atomic(
                score_path,
                prob=prob_array,
                raw_logit=raw_array,
                canonical_id=np.asarray(selected["canonical_id"]),
                image_id=np.asarray(selected["image_id"]),
                source_domain=np.asarray(source_domain),
                original_hw=np.asarray(transform.original_hw, dtype=np.int64),
                input_hw=np.asarray(transform.input_hw, dtype=np.int64),
                resized_hw=np.asarray(transform.resized_hw, dtype=np.int64),
                padding_ltrb=np.asarray(transform.padding_ltrb, dtype=np.int64),
                resize_mode=np.asarray(transform.mode),
            )
            score_sha = _hash_file(score_path)
            record = {
                "record_index": index,
                "canonical_id": selected["canonical_id"],
                "image_id": selected["image_id"],
                "source_domain": source_domain,
                "original_image_path": selected["original_image_path"],
                "original_image_sha256": selected["original_image_sha256"],
                "exclusion_group_id": selected["exclusion_group_id"],
                "near_duplicate_cluster_id_or_unique_sentinel": selected[
                    "near_duplicate_cluster_id_or_unique_sentinel"
                ],
                "source_role_record_index": selected["source_role_record_index"],
                "score_file": _future_repo_path(
                    published_root / output_name,
                    root,
                ),
                "score_file_sha256": score_sha,
                "original_hw": list(transform.original_hw),
                "input_hw": list(transform.input_hw),
                "resized_hw": list(transform.resized_hw),
                "padding_ltrb": list(transform.padding_ltrb),
                "resize_mode": transform.mode,
            }
            _verify_stage2_score_npz(score_path, record)
            result.append(record)
    return result


def _build_manifest(
    selection: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    role: str,
    input_hw: tuple[int, int],
    resize_mode: str,
    bindings: Mapping[str, Mapping[str, str]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": STAGE2_SCORE_MANIFEST_SCHEMA,
        "artifact_type": STAGE2_SCORE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "development_only": True,
        "execution_scope": "stage2_development",
        "official_test_accessed": False,
        "labels_embedded": False,
        "native_resolution": True,
        "restored_to_original_hw": True,
        "path_anchor": "repository_root",
        "role": role,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "score_type": "sigmoid_probability",
        "score_dtype": "float64",
        "sigmoid_compute_dtype": "float64",
        "raw_logits_exported": True,
        "raw_logit_dtype": "float64",
        "raw_logit_space": (
            "native_original_hw_spatially_aligned_restored_model_logit"
        ),
        "probability_space": (
            "native_original_hw_float64_sigmoid_then_spatial_restore"
        ),
        "outer_fold_id": run["outer_fold_id"],
        "outer_target": _first(run, "outer_target", "outer_target_domain"),
        "source_domain": _source_domain(selection),
        "base_seed": run["base_seed"],
        "derived_seed": run["derived_seed"],
        "detector_role": run["detector_role"],
        "oof_fold_index": run["oof_fold_index"],
        "input_hw": list(input_hw),
        "resize_mode": resize_mode,
        "bindings": dict(bindings),
        "num_images": len(records),
        "records_content_sha256_algorithm": STAGE2_SCORE_RECORDS_ALGORITHM,
        "records_content_sha256": stage2_score_records_sha256(records),
        "records": records,
    }


def _apply_transform(image: Image.Image, transform: Any) -> Image.Image:
    resized_h, resized_w = transform.resized_hw
    result = image.resize((resized_w, resized_h), resample=_BILINEAR)
    if transform.mode == "letterbox":
        result = ImageOps.expand(result, border=transform.padding_ltrb, fill=(0, 0, 0))
    if result.size != (transform.input_hw[1], transform.input_hw[0]):
        raise RuntimeError("image transform did not produce detector input geometry")
    return result


def _extract_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[-1], torch.Tensor):
        return output[-1]
    raise TypeError("model output does not contain a logits tensor")


def _prepare_output_bundle(
    value: str | Path,
    root: Path,
    score_names: tuple[str, ...],
) -> _OutputBundle:
    if not score_names or len(score_names) != len(set(score_names)):
        raise ValueError("score output names must be non-empty and unique")
    reserved_names = {"manifest.json", "manifest.json.sha256", ".export_incomplete"}
    for name in score_names:
        if (
            PurePosixPath(name).name != name
            or not name.endswith(".npz")
            or name in reserved_names
        ):
            raise ValueError(f"invalid score output name: {name!r}")

    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("output_dir must be inside repository_root") from error
    rendered = _relative_path(relative.as_posix(), "output_dir")
    candidate = root.joinpath(*PurePosixPath(rendered).parts)
    _ensure_output_parent(candidate.parent, root)
    if os.path.lexists(candidate):
        raise FileExistsError("Stage2 output_dir must be completely unoccupied")

    lock_path = candidate.parent / f".{candidate.name}.stage2-export.lock"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise FileExistsError("Stage2 output bundle lock is already occupied") from error
    opened_lock_stat = os.fstat(lock_fd)
    lock_identity = (
        int(opened_lock_stat.st_dev),
        int(opened_lock_stat.st_ino),
    )
    try:
        os.write(lock_fd, b"rc-irstd.stage2-score-export-lock.v1\n")
        os.fsync(lock_fd)
    except BaseException:
        os.close(lock_fd)
        _release_owned_lock(lock_path, lock_identity)
        raise
    else:
        os.close(lock_fd)
    lock_stat = lock_path.lstat()
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or (int(lock_stat.st_dev), int(lock_stat.st_ino)) != lock_identity
    ):
        raise RuntimeError("Stage2 output lock identity changed during reservation")
    staging_root: Path | None = None
    staging_identity: tuple[int, int] | None = None
    try:
        if os.path.lexists(candidate):
            raise FileExistsError("Stage2 output target changed during reservation")
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{candidate.name}.stage2-staging-",
                dir=candidate.parent,
            )
        )
        staging_stat = staging_root.lstat()
        if not stat.S_ISDIR(staging_stat.st_mode) or staging_root.is_symlink():
            raise RuntimeError("Stage2 staging target is not a private directory")
        staging_identity = (int(staging_stat.st_dev), int(staging_stat.st_ino))
        _fsync_directory(candidate.parent)
    except BaseException:
        if staging_root is not None and staging_identity is not None:
            _remove_owned_directory(staging_root, staging_identity)
        _release_owned_lock(lock_path, lock_identity)
        raise
    assert staging_root is not None and staging_identity is not None
    return _OutputBundle(
        final_root=candidate,
        staging_root=staging_root,
        lock_path=lock_path,
        score_names=score_names,
        staging_identity=staging_identity,
        lock_identity=lock_identity,
    )


def _ensure_output_parent(parent: Path, root: Path) -> None:
    try:
        relative = parent.relative_to(root)
    except ValueError as error:
        raise ValueError("output parent escapes repository_root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current):
            current_stat = current.lstat()
            if stat.S_ISLNK(current_stat.st_mode):
                raise ValueError("output parent contains a symlink component")
            if not stat.S_ISDIR(current_stat.st_mode):
                raise NotADirectoryError(f"output parent is not a directory: {current}")
        else:
            current.mkdir(mode=0o700)
    if parent.resolve(strict=True) != parent:
        raise ValueError("output parent is not a canonical non-symlink path")


def _preflight_staged_bundle(
    bundle: _OutputBundle,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    staging_stat = bundle.staging_root.lstat()
    if (
        not stat.S_ISDIR(staging_stat.st_mode)
        or (int(staging_stat.st_dev), int(staging_stat.st_ino))
        != bundle.staging_identity
    ):
        raise RuntimeError("Stage2 staging directory identity changed")
    expected = set(bundle.score_names) | {
        "manifest.json",
        "manifest.json.sha256",
        ".export_incomplete",
    }
    entries = list(bundle.staging_root.iterdir())
    if len(entries) != len(expected) or {path.name for path in entries} != expected:
        raise RuntimeError("Stage2 staged bundle file table mismatch")
    for path in entries:
        entry_stat = path.lstat()
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
            raise RuntimeError(f"Stage2 staged artifact is not regular: {path.name}")

    if len(records) != len(bundle.score_names):
        raise RuntimeError("Stage2 staged score/record count mismatch")
    for name, record in zip(bundle.score_names, records):
        score_path = bundle.staging_root / name
        before = _hash_file(score_path)
        if before != record["score_file_sha256"]:
            raise RuntimeError(f"staged score SHA-256 mismatch: {name}")
        _verify_stage2_score_npz(score_path, record)
        if _hash_file(score_path) != before:
            raise RuntimeError(f"staged score changed during preflight: {name}")
        _fsync_file(score_path)

    manifest_path = bundle.staging_root / "manifest.json"
    before = _hash_file(manifest_path)
    if before != manifest_sha256:
        raise RuntimeError("staged manifest SHA-256 mismatch")
    with manifest_path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if parsed != manifest or _hash_file(manifest_path) != before:
        raise RuntimeError("staged manifest content changed during preflight")
    _fsync_file(manifest_path)

    sidecar_path = bundle.staging_root / "manifest.json.sha256"
    sidecar_before = _hash_file(sidecar_path)
    sidecar_text = sidecar_path.read_text(encoding="utf-8")
    if sidecar_text != f"{manifest_sha256}  manifest.json\n":
        raise RuntimeError("staged manifest sidecar content mismatch")
    if _hash_file(sidecar_path) != sidecar_before:
        raise RuntimeError("staged manifest sidecar changed during preflight")
    _fsync_file(sidecar_path)

    marker_path = bundle.staging_root / ".export_incomplete"
    marker_before = _hash_file(marker_path)
    if marker_path.read_text(encoding="utf-8") != (
        "Stage2 development score export is incomplete; do not consume.\n"
    ):
        raise RuntimeError("staged incomplete marker content mismatch")
    if _hash_file(marker_path) != marker_before:
        raise RuntimeError("staged incomplete marker changed during preflight")


def _verify_published_bundle(
    bundle: _OutputBundle,
    published_identity: tuple[int, int],
    manifest_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    published = bundle.final_root.lstat()
    if (
        stat.S_ISLNK(published.st_mode)
        or not stat.S_ISDIR(published.st_mode)
        or (int(published.st_dev), int(published.st_ino)) != published_identity
    ):
        raise RuntimeError("published Stage2 bundle directory identity changed")
    expected = set(bundle.score_names) | {
        "manifest.json",
        "manifest.json.sha256",
    }
    entries = list(bundle.final_root.iterdir())
    if len(entries) != len(expected) or {path.name for path in entries} != expected:
        raise RuntimeError("published Stage2 bundle file table mismatch")
    for entry in entries:
        entry_stat = entry.lstat()
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISREG(entry_stat.st_mode):
            raise RuntimeError(
                f"published Stage2 artifact is not regular: {entry.name}"
            )
    for name, record in zip(bundle.score_names, records):
        score = bundle.final_root / name
        before = _hash_file(score)
        if before != record["score_file_sha256"]:
            raise RuntimeError(f"published score SHA-256 mismatch: {name}")
        _verify_stage2_score_npz(score, record)
        if _hash_file(score) != before:
            raise RuntimeError(f"published score changed during verification: {name}")
    manifest_path = bundle.final_root / "manifest.json"
    if _hash_file(manifest_path) != manifest_sha256:
        raise RuntimeError("published manifest SHA-256 mismatch")
    sidecar_path = bundle.final_root / "manifest.json.sha256"
    sidecar_before = _hash_file(sidecar_path)
    if sidecar_path.read_text(encoding="utf-8") != (
        f"{manifest_sha256}  manifest.json\n"
    ):
        raise RuntimeError("published manifest sidecar mismatch")
    if _hash_file(sidecar_path) != sidecar_before:
        raise RuntimeError("published manifest sidecar changed during verification")


def _atomic_publish_directory_no_replace(
    staging_root: Path,
    final_root: Path,
) -> tuple[int, int]:
    source_stat = staging_root.lstat()
    if not stat.S_ISDIR(source_stat.st_mode) or staging_root.is_symlink():
        raise RuntimeError("Stage2 staging directory is not publishable")
    source_identity = (int(source_stat.st_dev), int(source_stat.st_ino))
    if os.path.lexists(final_root):
        raise FileExistsError("Stage2 output target was occupied before publish")

    try:
        _rename_path_no_replace(staging_root, final_root)
    except FileExistsError as error:
        raise FileExistsError(
            "Stage2 output target was occupied during publish"
        ) from error
    published_stat = final_root.lstat()
    published_identity = (int(published_stat.st_dev), int(published_stat.st_ino))
    if (
        not stat.S_ISDIR(published_stat.st_mode)
        or final_root.is_symlink()
        or published_identity != source_identity
    ):
        raise RuntimeError("published Stage2 bundle identity mismatch")
    _fsync_directory(final_root.parent)
    return published_identity


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    if not os.path.lexists(path):
        return
    current = path.lstat()
    current_identity = (int(current.st_dev), int(current.st_ino))
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or current_identity != identity
    ):
        raise RuntimeError(f"refusing to remove non-owned directory: {path}")
    shutil.rmtree(path)
    _fsync_directory(path.parent)


def _release_owned_lock(path: Path, identity: tuple[int, int]) -> None:
    if not os.path.lexists(path):
        return
    current = path.lstat()
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (int(current.st_dev), int(current.st_ino)) != identity
    ):
        raise RuntimeError("Stage2 output lock identity changed")
    path.unlink()
    _fsync_directory(path.parent)


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode):
            raise RuntimeError(f"cannot fsync non-regular artifact: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary, descriptor, identity = _open_exclusive_temporary(path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_exclusive_temporary(temporary, path, identity)
    except BaseException:
        _remove_owned_file(temporary, identity)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_bytes_atomic(path, encoded)


def _write_text_atomic(path: Path, text: str) -> None:
    _write_bytes_atomic(path, text.encode("utf-8"))


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary, descriptor, identity = _open_exclusive_temporary(path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_exclusive_temporary(temporary, path, identity)
    except BaseException:
        _remove_owned_file(temporary, identity)
        raise


def _open_exclusive_temporary(path: Path) -> tuple[Path, int, tuple[int, int]]:
    if os.path.lexists(path):
        raise FileExistsError(f"bundle target is already occupied: {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        os.close(descriptor)
        raise RuntimeError("exclusive temporary target is not a regular file")
    return (
        temporary,
        descriptor,
        (int(opened.st_dev), int(opened.st_ino)),
    )


def _commit_exclusive_temporary(
    temporary: Path,
    path: Path,
    identity: tuple[int, int],
) -> None:
    _rename_path_no_replace(temporary, path)
    published = path.lstat()
    if (
        not stat.S_ISREG(published.st_mode)
        or (int(published.st_dev), int(published.st_ino)) != identity
    ):
        raise RuntimeError(f"atomic bundle target identity mismatch: {path.name}")
    _fsync_directory(path.parent)


def _remove_owned_file(path: Path, identity: tuple[int, int]) -> None:
    if not os.path.lexists(path):
        return
    current = path.lstat()
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (int(current.st_dev), int(current.st_ino)) != identity
    ):
        raise RuntimeError(f"refusing to remove non-owned file: {path}")
    path.unlink()
    _fsync_directory(path.parent)


def _rename_path_no_replace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(f"bundle target became occupied: {target}")
    raise OSError(error_number, os.strerror(error_number), str(target))


def _load_expected_json(
    path: str | Path,
    expected_sha256: object,
    root: Path,
    name: str,
) -> tuple[Path, Mapping[str, Any]]:
    resolved = _existing_path(path, root, name)
    digest = _sha256(expected_sha256, f"{name} SHA-256")
    before = _hash_file(resolved)
    if before != digest:
        raise ValueError(f"{name} SHA-256 mismatch")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if _hash_file(resolved) != before:
        raise RuntimeError(f"{name} changed while read")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return resolved, payload


def _repository_root(value: str | Path | None) -> Path:
    root = REPOSITORY_ROOT if value is None else Path(value).expanduser()
    if root.is_symlink():
        raise ValueError("repository_root must not be a symlink")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError("repository_root must be a directory")
    return root


def _existing_path(value: str | Path, root: Path, name: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside repository_root") from error
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} contains a symlink component")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository_root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved


def _repository_file(root: Path, value: object, name: str) -> Path:
    relative = _relative_path(value, name)
    return _existing_path(root.joinpath(*PurePosixPath(relative).parts), root, name)


def _repo_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path escapes repository_root") from error
    rendered = relative.as_posix()
    return _relative_path(rendered, "repository path")


def _future_repo_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("future artifact path escapes repository_root") from error
    return _relative_path(relative.as_posix(), "future repository path")


def _relative_path(value: object, name: str) -> str:
    result = _string(value, name)
    if "\\" in result:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(result)
    if path.is_absolute() or result.startswith("~") or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name} must be a canonical repository-relative path")
    if path.as_posix() != result:
        raise ValueError(f"{name} must be a canonical POSIX path")
    for raw_part in path.parts:
        part = raw_part.lower()
        stem = PurePosixPath(part).stem
        if (
            part in {"test", "official_test", "official-test"}
            or stem.startswith("test_")
            or stem.startswith("official_test")
        ):
            raise ValueError(f"{name} names an official-test-like path")
    return result


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase SHA-256 string")
    if value != value.lower() or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} is not a lowercase SHA-256")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string")
    return value


def _exact_int(value: object, name: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact JSON integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _first(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise KeyError(f"missing one of required fields: {names}")


def _source_domain(selection: Mapping[str, Any]) -> str:
    return _string(
        selection.get("source_domain", selection.get("domain")),
        "selection.source_domain/domain",
    )


def _role(value: object) -> str:
    role = _string(value, "role")
    if role not in STAGE2_DEVELOPMENT_ROLES:
        raise ValueError("role is not a frozen Stage2 development role")
    return role


def _select_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value != "cuda":
        raise ValueError("device must be exact 'cpu' or 'cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda")


def _safe_score_name(canonical_id: object) -> str:
    value = _string(canonical_id, "canonical_id")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    stem = "".join(character if character.isalnum() else "_" for character in value)
    stem = stem.strip("_")[:80] or "sample"
    return f"{stem}.{digest}.npz"


_MISSING = object()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Stage2 development-only native-resolution scores"
    )
    parser.add_argument("--selection-contract", required=True)
    parser.add_argument("--selection-contract-sha256", required=True)
    parser.add_argument("--run-contract", required=True)
    parser.add_argument("--run-contract-sha256", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--role", required=True, choices=STAGE2_DEVELOPMENT_ROLES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = export_stage2_development_scores(
        args.selection_contract,
        args.run_contract,
        args.weight_path,
        args.output_dir,
        selection_contract_sha256=args.selection_contract_sha256,
        run_contract_sha256=args.run_contract_sha256,
        checkpoint_sha256=args.weight_sha256,
        role=args.role,
        device=args.device,
        repository_root=args.repository_root,
    )
    manifest_path = Path(args.output_dir).expanduser().resolve() / "manifest.json"
    print(
        json.dumps(
            {
                "status": "PASS_DEVELOPMENT_EXPORT",
                "role": manifest["role"],
                "num_images": manifest["num_images"],
                "manifest": str(manifest_path),
                "manifest_sha256": _hash_file(manifest_path),
                "official_test_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - Pillow compatibility
    _BILINEAR = Image.BILINEAR


if __name__ == "__main__":
    raise SystemExit(main())
