"""Fail-closed Stage2 development score-manifest v4 verification.

This module is deliberately separate from :mod:`score_manifest_artifacts`.
The legacy verifier continues to accept only the historical score-manifest
v2/v3 formats; Stage2 sample-OOF artifacts must opt into this verifier and its
strict development-only contract.

All paths stored in a v4 manifest are POSIX, repository-relative paths.  The
verifier rejects absolute paths, traversal, duplicate paths and symlinked
components, hashes every consumed artifact before and after use, and never
discovers an official split.  The selected development records come only from
the hash-bound selection contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

STAGE2_SCORE_MANIFEST_SCHEMA = "rc-irstd.score-manifest.v4"
STAGE2_SCORE_ARTIFACT_TYPE = "rc_irstd_stage2_development_score_export"
STAGE2_SCORE_RECORDS_ALGORITHM = "sha256-canonical-json-ordered-score-records-v1"
STRICT_THRESHOLD_SEMANTICS = "prediction = probability > threshold"

OOF_TRAIN_SOURCE_REFERENCE = "oof_train_source_reference"
OOF_HOLDOUT_STAGE2_FIT = "oof_holdout_stage2_fit"
FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE = "fullfit_detector_fit_source_reference"
SOURCE_DIAGNOSTIC_VALIDATION = "source_diagnostic_validation"
OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT = "outer_target_diagnostic_development"

STAGE2_DEVELOPMENT_ROLES = (
    OOF_TRAIN_SOURCE_REFERENCE,
    OOF_HOLDOUT_STAGE2_FIT,
    FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    SOURCE_DIAGNOSTIC_VALIDATION,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
)

STAGE2_DOMAINS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
OUTER_FOLD_TARGETS = {
    "outer_leave_nuaa_sirst": "NUAA-SIRST",
    "outer_leave_nudt_sirst": "NUDT-SIRST",
    "outer_leave_irstd_1k": "IRSTD-1K",
}
BASE_SEEDS = (42, 123, 3407)

BINDING_NAMES = (
    "selection_contract",
    "run_contract",
    "checkpoint",
    "detector_config",
    "runtime_config",
    "seed_manifest",
    "materialization_index",
    "release_artifact",
    "environment_artifact",
    "runtime_contract",
)

MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "execution_scope",
        "official_test_accessed",
        "labels_embedded",
        "native_resolution",
        "restored_to_original_hw",
        "path_anchor",
        "role",
        "threshold_semantics",
        "score_type",
        "score_dtype",
        "sigmoid_compute_dtype",
        "raw_logits_exported",
        "raw_logit_dtype",
        "raw_logit_space",
        "probability_space",
        "outer_fold_id",
        "outer_target",
        "source_domain",
        "base_seed",
        "derived_seed",
        "detector_role",
        "oof_fold_index",
        "input_hw",
        "resize_mode",
        "bindings",
        "num_images",
        "records_content_sha256_algorithm",
        "records_content_sha256",
        "records",
    }
)

RECORD_FIELDS = frozenset(
    {
        "record_index",
        "canonical_id",
        "image_id",
        "source_domain",
        "original_image_path",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
        "score_file",
        "score_file_sha256",
        "original_hw",
        "input_hw",
        "resized_hw",
        "padding_ltrb",
        "resize_mode",
    }
)

NPZ_FIELD_ORDER = (
    "prob",
    "raw_logit",
    "canonical_id",
    "image_id",
    "source_domain",
    "original_hw",
    "input_hw",
    "resized_hw",
    "padding_ltrb",
    "resize_mode",
)
NPZ_FIELDS = frozenset(NPZ_FIELD_ORDER)
NPZ_ZIP_MEMBER_ORDER = tuple(f"{field}.npy" for field in NPZ_FIELD_ORDER)


@dataclass(frozen=True)
class VerifiedStage2ScoreItem:
    """One identity- and byte-verified native-resolution score artifact."""

    record_index: int
    canonical_id: str
    image_id: str
    source_domain: str
    record: Mapping[str, Any]
    score_path: Path
    image_path: Path
    original_hw: tuple[int, int]


@dataclass(frozen=True)
class VerifiedStage2ScoreManifest:
    """A complete, development-only Stage2 score-manifest v4."""

    path: Path
    repository_root: Path
    payload: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    items: tuple[VerifiedStage2ScoreItem, ...]
    role: str
    manifest_sha256: str
    records_content_sha256: str
    bindings: Mapping[str, Mapping[str, str]]


def stage2_score_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact ordered record mappings using canonical JSON."""

    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Stage2 score records must be a non-empty ordered list")
    canonical_records: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a JSON object")
        canonical_records.append(record)
    encoded = json.dumps(
        canonical_records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    _update_frame(digest, STAGE2_SCORE_RECORDS_ALGORITHM)
    digest.update(encoded)
    return digest.hexdigest()


def verify_stage2_score_manifest(
    path: str | Path,
    expected_sha256: str,
    required_role: str,
    *,
    repository_root: str | Path | None = None,
) -> VerifiedStage2ScoreManifest:
    """Verify a score-manifest v4 and every bound development artifact.

    ``repository_root`` is optional for the production repository and exists
    primarily so isolated synthetic tests can use their own temporary root.
    It never comes from the untrusted manifest.
    """

    root = _repository_root(repository_root)
    role = _required_role(required_role)
    manifest_path = _existing_direct_path(path, root, "score manifest")
    if os.path.lexists(manifest_path.parent / ".export_incomplete"):
        raise RuntimeError(
            f"Stage2 score export is incomplete and unsafe: {manifest_path.parent}"
        )
    expected_digest = _sha256_value(expected_sha256, "expected_sha256")
    manifest_before = _sha256_file_stable(manifest_path)
    if manifest_before != expected_digest:
        raise ValueError("Stage2 score manifest SHA-256 does not match expected_sha256")
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if os.path.lexists(sidecar):
        sidecar = _existing_direct_path(sidecar, root, "manifest SHA-256 sidecar")
        sidecar_before = _sha256_file_stable(sidecar)
        text = sidecar.read_text(encoding="utf-8")
        if text != f"{manifest_before}  {manifest_path.name}\n":
            raise ValueError("manifest SHA-256 sidecar content mismatch")
        if _sha256_file_stable(sidecar) != sidecar_before:
            raise RuntimeError("manifest SHA-256 sidecar changed while verified")
    payload = _read_json_file(manifest_path, "score manifest")
    if _sha256_file_stable(manifest_path) != manifest_before:
        raise RuntimeError("Stage2 score manifest changed while being verified")
    if not isinstance(payload, Mapping):
        raise TypeError("Stage2 score manifest must contain a JSON object")
    _exact_keys(payload, MANIFEST_FIELDS, "score manifest")
    _verify_top_level_contract(payload, role)

    bindings = _verify_bindings(payload["bindings"], root)
    selection = _read_bound_json(bindings["selection_contract"], root)
    run_contract = _read_bound_json(bindings["run_contract"], root)
    if not isinstance(selection, Mapping):
        raise TypeError("selection contract must contain a JSON object")
    if not isinstance(run_contract, Mapping):
        raise TypeError("run contract must contain a JSON object")

    _verify_identity_against_contracts(
        payload,
        selection=selection,
        run_contract=run_contract,
        selection_binding=bindings["selection_contract"],
    )
    _verify_provenance_closure(
        payload,
        run_contract=run_contract,
        bindings=bindings,
        root=root,
    )
    selected_records = _selection_records(
        selection,
        role=role,
        oof_fold_index=payload["oof_fold_index"],
    )
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("score manifest records must be a non-empty ordered list")
    if len(raw_records) != _exact_int(payload["num_images"], "num_images", minimum=1):
        raise ValueError("num_images does not equal the number of score records")
    if len(raw_records) != len(selected_records):
        raise ValueError("score manifest must contain one record per selected ID")

    records: list[Mapping[str, Any]] = []
    seen_canonical: set[str] = set()
    seen_score_paths: set[str] = set()
    seen_source_indices: set[int] = set()
    items: list[VerifiedStage2ScoreItem] = []
    for index, (raw_record, selected_record) in enumerate(
        zip(raw_records, selected_records)
    ):
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"records[{index}] must be a JSON object")
        _exact_keys(raw_record, RECORD_FIELDS, f"records[{index}]")
        record = dict(raw_record)
        _verify_record_metadata(
            record,
            selected_record=selected_record,
            payload=payload,
            index=index,
        )
        canonical_id = record["canonical_id"]
        score_file = record["score_file"]
        source_index = record["source_role_record_index"]
        if canonical_id in seen_canonical:
            raise ValueError(f"duplicate canonical_id: {canonical_id!r}")
        if score_file in seen_score_paths:
            raise ValueError(f"duplicate score_file: {score_file!r}")
        if source_index in seen_source_indices:
            raise ValueError(
                f"duplicate source_role_record_index: {source_index!r}"
            )
        seen_canonical.add(canonical_id)
        seen_score_paths.add(score_file)
        seen_source_indices.add(source_index)
        records.append(record)
        items.append(_verify_score_record(record, root))

    if payload["records_content_sha256_algorithm"] != STAGE2_SCORE_RECORDS_ALGORITHM:
        raise ValueError("records_content_sha256_algorithm mismatch")
    calculated_records_sha = stage2_score_records_sha256(records)
    if _sha256_value(
        payload["records_content_sha256"], "records_content_sha256"
    ) != calculated_records_sha:
        raise ValueError("records_content_sha256 does not bind the ordered records")

    # Re-hash every external input after all consuming operations.  This also
    # covers the checkpoint/config/seed/materialization/release/environment
    # artifacts that are byte-bound but not parsed here.
    for name, binding in bindings.items():
        artifact = _resolve_repository_file(root, binding["path"], name)
        if _sha256_file_stable(artifact) != binding["sha256"]:
            raise RuntimeError(f"bound artifact changed while verified: {name}")
    if _sha256_file_stable(manifest_path) != manifest_before:
        raise RuntimeError("Stage2 score manifest changed while being verified")

    return VerifiedStage2ScoreManifest(
        path=manifest_path,
        repository_root=root,
        payload=payload,
        records=tuple(records),
        items=tuple(items),
        role=role,
        manifest_sha256=manifest_before,
        records_content_sha256=calculated_records_sha,
        bindings=bindings,
    )


def _verify_top_level_contract(payload: Mapping[str, Any], role: str) -> None:
    exact_values = {
        "schema_version": STAGE2_SCORE_MANIFEST_SCHEMA,
        "artifact_type": STAGE2_SCORE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "execution_scope": "stage2_development",
        "path_anchor": "repository_root",
        "role": role,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "score_type": "sigmoid_probability",
        "score_dtype": "float64",
        "sigmoid_compute_dtype": "float64",
        "raw_logit_dtype": "float64",
        "raw_logit_space": (
            "native_original_hw_spatially_aligned_restored_model_logit"
        ),
        "probability_space": (
            "native_original_hw_float64_sigmoid_then_spatial_restore"
        ),
    }
    for field, expected in exact_values.items():
        if payload[field] != expected:
            raise ValueError(f"{field} must equal {expected!r}")
    for field, expected in {
        "development_only": True,
        "official_test_accessed": False,
        "labels_embedded": False,
        "native_resolution": True,
        "restored_to_original_hw": True,
        "raw_logits_exported": True,
    }.items():
        _exact_bool(payload[field], field, expected)

    outer_fold = _nonempty_string(payload["outer_fold_id"], "outer_fold_id")
    if outer_fold not in OUTER_FOLD_TARGETS:
        raise ValueError(f"unknown outer_fold_id: {outer_fold!r}")
    outer_target = _domain(payload["outer_target"], "outer_target")
    if OUTER_FOLD_TARGETS[outer_fold] != outer_target:
        raise ValueError("outer_fold_id and outer_target are inconsistent")
    source_domain = _domain(payload["source_domain"], "source_domain")
    base_seed = _exact_int(payload["base_seed"], "base_seed", minimum=0)
    if base_seed not in BASE_SEEDS:
        raise ValueError(f"base_seed must be one of {BASE_SEEDS}")
    _exact_int(payload["derived_seed"], "derived_seed", minimum=0)
    _parse_hw(payload["input_hw"], "input_hw")
    if payload["resize_mode"] not in {"resize", "letterbox"}:
        raise ValueError("resize_mode must be 'resize' or 'letterbox'")

    expected_detector = (
        "detector_oof"
        if role in {OOF_TRAIN_SOURCE_REFERENCE, OOF_HOLDOUT_STAGE2_FIT}
        else "detector_full_fit"
    )
    if payload["detector_role"] != expected_detector:
        raise ValueError(
            f"role {role!r} requires detector_role={expected_detector!r}"
        )
    if expected_detector == "detector_oof":
        oof_index = _exact_int(payload["oof_fold_index"], "oof_fold_index", minimum=0)
        if oof_index not in {0, 1}:
            raise ValueError("detector_oof oof_fold_index must be 0 or 1")
    elif payload["oof_fold_index"] is not None:
        raise ValueError("detector_full_fit oof_fold_index must be exactly null")

    if role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT:
        if source_domain != outer_target:
            raise ValueError("outer-target role source_domain must equal outer_target")
    elif source_domain == outer_target:
        raise ValueError("source development/reference roles must exclude outer_target")


def _verify_bindings(
    value: object,
    root: Path,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping):
        raise TypeError("bindings must be a JSON object")
    _exact_keys(value, frozenset(BINDING_NAMES), "bindings")
    result: dict[str, dict[str, str]] = {}
    seen_paths: set[str] = set()
    for name in BINDING_NAMES:
        raw = value[name]
        if not isinstance(raw, Mapping):
            raise TypeError(f"bindings.{name} must be a JSON object")
        _exact_keys(raw, frozenset({"path", "sha256"}), f"bindings.{name}")
        relative = _relative_repository_path(raw["path"], f"bindings.{name}.path")
        digest = _sha256_value(raw["sha256"], f"bindings.{name}.sha256")
        if relative in seen_paths:
            raise ValueError(f"duplicate bound artifact path: {relative!r}")
        seen_paths.add(relative)
        artifact = _resolve_repository_file(root, relative, name)
        if _sha256_file_stable(artifact) != digest:
            raise ValueError(f"bindings.{name}.sha256 mismatch")
        result[name] = {"path": relative, "sha256": digest}
    return result


def _verify_identity_against_contracts(
    payload: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    run_contract: Mapping[str, Any],
    selection_binding: Mapping[str, str],
) -> None:
    _verify_development_selection_source(selection)
    _exact_bool(
        run_contract.get("development_only"),
        "run contract development_only",
        True,
    )
    _exact_bool(
        run_contract.get("official_test_accessed"),
        "run contract official_test_accessed",
        False,
    )
    raw_sources = run_contract.get("source_domains")
    if (
        not isinstance(raw_sources, list)
        or len(raw_sources) != 2
        or len(set(raw_sources)) != 2
        or any(source not in STAGE2_DOMAINS for source in raw_sources)
    ):
        raise ValueError("run contract source_domains must be two unique frozen domains")
    if payload["outer_target"] in raw_sources:
        raise ValueError("run contract detector sources include the outer target")
    if payload["role"] == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT:
        if payload["source_domain"] in raw_sources:
            raise ValueError("outer-target diagnostic domain appears in detector sources")
    elif payload["source_domain"] not in raw_sources:
        raise ValueError("source role domain is absent from detector source_domains")
    for manifest_field, contract_fields in {
        "outer_fold_id": ("outer_fold_id",),
        "outer_target": ("outer_target", "outer_target_domain"),
        "source_domain": ("source_domain",),
        "base_seed": ("base_seed",),
        "derived_seed": ("derived_seed",),
        "detector_role": ("detector_role",),
        "oof_fold_index": ("oof_fold_index",),
    }.items():
        expected = payload[manifest_field]
        selection_value = _first_present(selection, contract_fields)
        if selection_value is not _MISSING and selection_value != expected:
            raise ValueError(
                f"selection contract {manifest_field} disagrees with score manifest"
            )
        run_value = _first_present(run_contract, contract_fields)
        # A run contract spans two source-domain selections, so source_domain
        # may intentionally live only on each selection contract.
        if (
            manifest_field != "source_domain"
            and run_value is not _MISSING
            and run_value != expected
        ):
            raise ValueError(
                f"run contract {manifest_field} disagrees with score manifest"
            )

    role = payload["role"]
    selection_type = selection.get("artifact_type")
    if selection_type == "rc_irstd_stage2_detector_selection":
        if role not in {
            OOF_TRAIN_SOURCE_REFERENCE,
            FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
        }:
            raise ValueError(
                "detector-training selections may only feed source-reference roles"
            )
        run_selections = run_contract.get("selection_contracts")
        if not isinstance(run_selections, list) or len(run_selections) != 2:
            raise ValueError("run contract must bind exactly two selection_contracts")
        matches = 0
        for index, raw in enumerate(run_selections):
            if not isinstance(raw, Mapping):
                raise TypeError(f"run selection_contracts[{index}] must be an object")
            if (
                raw.get("path") == selection_binding["path"]
                and raw.get("sha256") == selection_binding["sha256"]
            ):
                matches += 1
        if matches != 1:
            raise ValueError(
                "selected training contract must occur exactly once in run selections"
            )
    else:
        if role not in {
            OOF_HOLDOUT_STAGE2_FIT,
            SOURCE_DIAGNOSTIC_VALIDATION,
            OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
        }:
            raise ValueError("materialized selections cannot feed reference roles")
        run_bindings = run_contract.get("bindings")
        if not isinstance(run_bindings, Mapping):
            raise TypeError("run contract bindings must be an object")
        materialized = run_bindings.get("materialization_artifacts_sha256")
        if not isinstance(materialized, Mapping):
            raise TypeError(
                "run contract must bind materialization_artifacts_sha256"
            )
        if materialized.get(selection_binding["path"]) != selection_binding["sha256"]:
            raise ValueError(
                "role selection is not hash-bound by run materialization artifacts"
            )


def _selection_records(
    selection: Mapping[str, Any],
    *,
    role: str,
    oof_fold_index: object,
) -> list[Mapping[str, Any]]:
    """Project one of the three frozen selection-source forms."""

    artifact_type = selection.get("artifact_type")
    if artifact_type == "rc_irstd_stage2_detector_selection":
        if role not in {
            OOF_TRAIN_SOURCE_REFERENCE,
            FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
        }:
            raise ValueError("training selection is invalid for the requested role")
        records = selection.get("records")
        declared_field = "record_count"
    elif artifact_type == "rc_irstd_stage2_detector_fit_group_assignment":
        if role != OOF_HOLDOUT_STAGE2_FIT:
            raise ValueError("K=2 assignment is valid only for OOF holdout scores")
        fold = _exact_int(oof_fold_index, "oof_fold_index", minimum=0)
        raw_records = selection.get("records")
        if not isinstance(raw_records, list):
            raise TypeError("assignment records must be a list")
        records = [
            record
            for record in raw_records
            if isinstance(record, Mapping) and record.get("oof_fold_index") == fold
        ]
        declared_field = None
        declared_counts = selection.get("fold_counts")
        if (
            not isinstance(declared_counts, Mapping)
            or declared_counts.get(str(fold)) != len(records)
        ):
            raise ValueError("assignment fold_counts disagrees with selected records")
    elif artifact_type == "rc_irstd_stage2_role_pure_episode_windows":
        if role not in {
            SOURCE_DIAGNOSTIC_VALIDATION,
            OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
        }:
            raise ValueError("window selection is invalid for the requested role")
        if selection.get("episode_role") != role:
            raise ValueError("window episode_role differs from requested score role")
        windows = selection.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("window selection requires non-empty windows")
        flattened: list[Mapping[str, Any]] = []
        for window_index, window in enumerate(windows):
            if not isinstance(window, Mapping):
                raise TypeError(f"windows[{window_index}] must be an object")
            for partition in ("context_records", "query_records"):
                partition_records = window.get(partition)
                if not isinstance(partition_records, list):
                    raise TypeError(
                        f"windows[{window_index}].{partition} must be a list"
                    )
                flattened.extend(partition_records)
        records = flattened
        declared_field = "window_record_count"
    else:
        raise ValueError("unsupported Stage2 score selection artifact_type")

    if not isinstance(records, list) or not records:
        raise ValueError("selection records must be a non-empty ordered list")
    if declared_field is not None:
        declared = _exact_int(
            selection.get(declared_field),
            f"selection {declared_field}",
            minimum=1,
        )
        if declared != len(records):
            raise ValueError(f"selection {declared_field} does not match records")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    last_source_index = -1
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"selection records[{index}] must be an object")
        canonical_id = _nonempty_string(
            record.get("canonical_id"), f"selection records[{index}].canonical_id"
        )
        source_index = _exact_int(
            record.get("source_role_record_index"),
            f"selection records[{index}].source_role_record_index",
            minimum=0,
        )
        if canonical_id in seen:
            raise ValueError("selection projection contains duplicate canonical IDs")
        if source_index <= last_source_index:
            raise ValueError(
                "selection projection is not in strict source-role record order"
            )
        seen.add(canonical_id)
        last_source_index = source_index
        result.append(record)
    return result


def _verify_development_selection_source(selection: Mapping[str, Any]) -> None:
    """Require exact zero-access guards on every accepted selection form."""

    artifact_type = selection.get("artifact_type")
    if artifact_type == "rc_irstd_stage2_detector_selection":
        _exact_bool(
            selection.get("development_only"),
            "selection development_only",
            True,
        )
        _exact_bool(
            selection.get("official_test_accessed"),
            "selection official_test_accessed",
            False,
        )
        _exact_bool(
            selection.get("execution_authorized"),
            "selection execution_authorized",
            False,
        )
        if selection.get("observed_results") is not None:
            raise ValueError("selection observed_results must be exactly null")
        return
    guards = selection.get("guardrails")
    if not isinstance(guards, Mapping):
        raise TypeError("materialized selection guardrails must be an object")
    for field, expected in {
        "development_only": True,
        "official_test_split_files_opened": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "mask_or_label_files_opened": False,
        "predictions_scores_checkpoints_or_metrics_opened": False,
    }.items():
        _exact_bool(guards.get(field), f"selection guardrails.{field}", expected)
    _exact_bool(
        selection.get("execution_authorized"),
        "selection execution_authorized",
        False,
    )
    if selection.get("observed_results") is not None:
        raise ValueError("selection observed_results must be exactly null")


def _verify_provenance_closure(
    payload: Mapping[str, Any],
    *,
    run_contract: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, str]],
    root: Path,
) -> None:
    """Replay checkpoint/runtime and all direct run-input hash identities."""

    run_bindings = run_contract.get("bindings")
    if not isinstance(run_bindings, Mapping):
        raise TypeError("run contract bindings must be an object")
    for name in (
        "detector_config",
        "seed_manifest",
        "materialization_index",
        "release_artifact",
    ):
        declared = run_bindings.get(name)
        if not isinstance(declared, Mapping):
            raise TypeError(f"run contract bindings.{name} must be an object")
        if declared.get("path") != bindings[name]["path"] or declared.get(
            "sha256"
        ) != bindings[name]["sha256"]:
            raise ValueError(f"manifest/run {name} binding mismatch")

    checkpoint_path = _resolve_repository_file(
        root, bindings["checkpoint"]["path"], "checkpoint"
    )
    checkpoint_before = _sha256_file_stable(checkpoint_path)
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if _sha256_file_stable(checkpoint_path) != checkpoint_before:
        raise RuntimeError("checkpoint changed while restricted-loaded")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("restricted checkpoint must contain a mapping")
    if checkpoint.get("format_version") != "rc-irstd.detector-inference.v1":
        raise ValueError("checkpoint format_version mismatch")
    _exact_bool(
        checkpoint.get("official_test_accessed"),
        "checkpoint official_test_accessed",
        False,
    )
    if checkpoint.get("run_contract_sha256") != bindings["run_contract"]["sha256"]:
        raise ValueError("checkpoint run-contract SHA-256 mismatch")
    if checkpoint.get("run_config_sha256") != bindings["runtime_config"]["sha256"]:
        raise ValueError("checkpoint runtime-config SHA-256 mismatch")
    for checkpoint_field, manifest_field in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target", "outer_target"),
        ("seed", "derived_seed"),
        ("detector_role", "detector_role"),
        ("oof_fold_index", "oof_fold_index"),
    ):
        if checkpoint.get(checkpoint_field) != payload[manifest_field]:
            raise ValueError(f"checkpoint {checkpoint_field} identity mismatch")
    if checkpoint.get("checkpoint_selection") != "fixed_last_no_test_or_target_validation":
        raise ValueError("checkpoint selection is not fixed-last")
    if checkpoint.get("source_names") != run_contract.get("source_domains"):
        raise ValueError("checkpoint source_names mismatch")
    held_out = checkpoint.get("held_out_domains")
    if (
        not isinstance(held_out, list)
        or payload["outer_target"] not in held_out
        or set(held_out).intersection(checkpoint["source_names"])
    ):
        raise ValueError("checkpoint held-out/source identity mismatch")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("restricted checkpoint state_dict must be non-empty tensors")
    _exact_int(checkpoint.get("epoch"), "checkpoint epoch", minimum=0)
    geometry = checkpoint.get("inference_geometry")
    if not isinstance(geometry, Mapping) or set(geometry) != {
        "input_hw",
        "resize_mode",
    }:
        raise ValueError("checkpoint inference_geometry fields mismatch")
    if geometry.get("input_hw") != payload["input_hw"] or geometry.get(
        "resize_mode"
    ) != payload["resize_mode"]:
        raise ValueError("checkpoint/manifest inference geometry mismatch")

    runtime = checkpoint.get("stage2_runtime_artifacts")
    if not isinstance(runtime, Mapping):
        raise TypeError("checkpoint stage2_runtime_artifacts must be an object")
    runtime_names = {
        "run_config": "runtime_config",
        "environment_artifact": "environment_artifact",
        "runtime_contract": "runtime_contract",
    }
    checkpoint_parent = checkpoint_path.parent
    for checkpoint_name, manifest_name in runtime_names.items():
        raw = runtime.get(checkpoint_name)
        if not isinstance(raw, Mapping):
            raise TypeError(f"checkpoint {checkpoint_name} binding must be an object")
        local_path = _relative_repository_path(
            raw.get("path"), f"checkpoint {checkpoint_name}.path"
        )
        candidate = checkpoint_parent.joinpath(*PurePosixPath(local_path).parts)
        resolved = _existing_direct_path(candidate, root, checkpoint_name)
        expected = bindings[manifest_name]
        if _sha256_value(
            raw.get("sha256"), f"checkpoint {checkpoint_name}.sha256"
        ) != expected["sha256"]:
            raise ValueError(f"checkpoint/manifest {checkpoint_name} hash mismatch")
        if resolved != _resolve_repository_file(
            root, expected["path"], manifest_name
        ):
            raise ValueError(f"checkpoint/manifest {checkpoint_name} path mismatch")

    runtime_path = _resolve_repository_file(
        root, bindings["runtime_contract"]["path"], "runtime contract"
    )
    runtime_before = _sha256_file_stable(runtime_path)
    contract = _read_json_file(runtime_path, "runtime contract")
    if _sha256_file_stable(runtime_path) != runtime_before:
        raise RuntimeError("runtime contract changed while verified")
    if not isinstance(contract, Mapping):
        raise TypeError("runtime contract must contain a JSON object")
    if contract.get("schema_version") != "rc-irstd.stage2-detector-runtime-contract.v1":
        raise ValueError("runtime contract schema mismatch")
    _exact_bool(
        contract.get("development_only"),
        "runtime contract development_only",
        True,
    )
    _exact_bool(
        contract.get("official_test_accessed"),
        "runtime contract official_test_accessed",
        False,
    )
    if contract.get("observed_results") is not None:
        raise ValueError("runtime contract observed_results must be null")
    input_run = contract.get("input_run_contract")
    if not isinstance(input_run, Mapping) or input_run.get("path") != bindings[
        "run_contract"
    ]["path"] or input_run.get("sha256") != bindings["run_contract"]["sha256"]:
        raise ValueError("runtime contract input-run binding mismatch")
    for contract_name, manifest_name in (
        ("run_config", "runtime_config"),
        ("environment_artifact", "environment_artifact"),
    ):
        raw = contract.get(contract_name)
        checkpoint_raw = runtime[contract_name]
        if not isinstance(raw, Mapping) or raw.get("path") != checkpoint_raw.get(
            "path"
        ) or raw.get("sha256") != bindings[manifest_name]["sha256"]:
            raise ValueError(f"runtime contract {contract_name} binding mismatch")
    for contract_field, manifest_field in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target_domain", "outer_target"),
        ("detector_role", "detector_role"),
        ("oof_fold_index", "oof_fold_index"),
        ("base_seed", "base_seed"),
        ("derived_seed", "derived_seed"),
    ):
        if contract.get(contract_field) != payload[manifest_field]:
            raise ValueError(f"runtime contract {contract_field} identity mismatch")


def _verify_record_metadata(
    record: Mapping[str, Any],
    *,
    selected_record: Mapping[str, Any],
    payload: Mapping[str, Any],
    index: int,
) -> None:
    if _exact_int(record["record_index"], f"records[{index}].record_index", minimum=0) != index:
        raise ValueError("record_index must equal the exact manifest order index")
    if record["source_domain"] != payload["source_domain"]:
        raise ValueError(f"records[{index}].source_domain mismatch")
    if record["input_hw"] != payload["input_hw"]:
        raise ValueError(f"records[{index}].input_hw mismatch")
    if record["resize_mode"] != payload["resize_mode"]:
        raise ValueError(f"records[{index}].resize_mode mismatch")
    _parse_hw(record["original_hw"], f"records[{index}].original_hw")
    _parse_hw(record["input_hw"], f"records[{index}].input_hw")
    _parse_hw(record["resized_hw"], f"records[{index}].resized_hw")
    _parse_padding(record["padding_ltrb"], f"records[{index}].padding_ltrb")
    _verify_geometry(record, index)

    for field in (
        "canonical_id",
        "image_id",
        "original_image_path",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
    ):
        if field not in selected_record:
            raise KeyError(f"selection records[{index}] is missing {field}")
        if record[field] != selected_record[field]:
            raise ValueError(
                f"records[{index}].{field} differs from the selected record"
            )
    _nonempty_string(record["canonical_id"], f"records[{index}].canonical_id")
    _nonempty_string(record["image_id"], f"records[{index}].image_id")
    _relative_repository_path(
        record["original_image_path"], f"records[{index}].original_image_path"
    )
    _sha256_value(
        record["original_image_sha256"],
        f"records[{index}].original_image_sha256",
    )
    _nonempty_string(
        record["exclusion_group_id"], f"records[{index}].exclusion_group_id"
    )
    _nonempty_string(
        record["near_duplicate_cluster_id_or_unique_sentinel"],
        f"records[{index}].near_duplicate_cluster_id_or_unique_sentinel",
    )
    _exact_int(
        record["source_role_record_index"],
        f"records[{index}].source_role_record_index",
        minimum=0,
    )
    _relative_repository_path(record["score_file"], f"records[{index}].score_file")
    _sha256_value(
        record["score_file_sha256"], f"records[{index}].score_file_sha256"
    )


def _verify_geometry(record: Mapping[str, Any], index: int) -> None:
    original_h, original_w = _parse_hw(
        record["original_hw"], f"records[{index}].original_hw"
    )
    input_h, input_w = _parse_hw(
        record["input_hw"], f"records[{index}].input_hw"
    )
    resized_h, resized_w = _parse_hw(
        record["resized_hw"], f"records[{index}].resized_hw"
    )
    left, top, right, bottom = _parse_padding(
        record["padding_ltrb"], f"records[{index}].padding_ltrb"
    )
    if resized_h + top + bottom != input_h or resized_w + left + right != input_w:
        raise ValueError(f"records[{index}] resize/padding geometry is inconsistent")
    if record["resize_mode"] == "resize":
        if (resized_h, resized_w) != (input_h, input_w) or any(
            (left, top, right, bottom)
        ):
            raise ValueError(f"records[{index}] resize geometry must have zero padding")
    elif record["resize_mode"] == "letterbox":
        scale = min(input_h / original_h, input_w / original_w)
        expected_h = min(input_h, max(1, int(round(original_h * scale))))
        expected_w = min(input_w, max(1, int(round(original_w * scale))))
        remaining_h = input_h - expected_h
        remaining_w = input_w - expected_w
        expected_padding = (
            remaining_w // 2,
            remaining_h // 2,
            remaining_w - remaining_w // 2,
            remaining_h - remaining_h // 2,
        )
        if (resized_h, resized_w) != (expected_h, expected_w) or (
            left,
            top,
            right,
            bottom,
        ) != expected_padding:
            raise ValueError(f"records[{index}] letterbox geometry mismatch")
    else:
        raise ValueError(f"records[{index}].resize_mode is unsupported")


def _verify_score_record(
    record: Mapping[str, Any],
    root: Path,
) -> VerifiedStage2ScoreItem:
    index = int(record["record_index"])
    score_path = _resolve_repository_file(root, record["score_file"], "score_file")
    image_path = _resolve_repository_file(
        root, record["original_image_path"], "original_image_path"
    )
    score_before = _sha256_file_stable(score_path)
    if score_before != record["score_file_sha256"]:
        raise ValueError(f"score-file SHA-256 mismatch for records[{index}]")
    image_before = _sha256_file_stable(image_path)
    if image_before != record["original_image_sha256"]:
        raise ValueError(f"image SHA-256 mismatch for records[{index}]")

    _verify_stage2_score_npz(score_path, record)

    if _sha256_file_stable(score_path) != score_before:
        raise RuntimeError(f"score file changed while verified: {score_path}")
    with Image.open(image_path) as image:
        image_hw = (int(image.height), int(image.width))
        image.verify()
    if image_hw != tuple(record["original_hw"]):
        raise ValueError(f"original image shape mismatch for records[{index}]")
    if _sha256_file_stable(image_path) != image_before:
        raise RuntimeError(f"image changed while verified: {image_path}")

    return VerifiedStage2ScoreItem(
        record_index=index,
        canonical_id=record["canonical_id"],
        image_id=record["image_id"],
        source_domain=record["source_domain"],
        record=record,
        score_path=score_path,
        image_path=image_path,
        original_hw=tuple(record["original_hw"]),
    )


def _verify_stage2_score_npz(
    score_path: Path,
    record: Mapping[str, Any],
) -> None:
    """Verify one physical NPZ against an already validated manifest record."""

    index = int(record["record_index"])
    score_before = _sha256_file_stable(score_path)
    if score_before != record["score_file_sha256"]:
        raise ValueError(f"score-file SHA-256 mismatch for records[{index}]")
    with zipfile.ZipFile(score_path, mode="r") as archive:
        members = tuple(info.filename for info in archive.infolist())
        if len(members) != len(set(members)):
            raise ValueError(f"duplicate ZIP member in score NPZ for records[{index}]")
        if members != NPZ_ZIP_MEMBER_ORDER:
            raise ValueError(
                f"score NPZ member order/count mismatch for records[{index}]"
            )

    with np.load(score_path, allow_pickle=False) as arrays:
        fields = tuple(arrays.files)
        if len(fields) != len(set(fields)):
            raise ValueError(f"duplicate NPZ field for records[{index}]")
        if fields != NPZ_FIELD_ORDER:
            missing = sorted(NPZ_FIELDS.difference(arrays.files))
            extra = sorted(set(arrays.files).difference(NPZ_FIELDS))
            raise ValueError(
                f"score NPZ field order/count mismatch for records[{index}]: "
                f"missing={missing}, extra={extra}"
            )
        probability = np.asarray(arrays["prob"])
        raw_logit = np.asarray(arrays["raw_logit"])
        original_hw = _parse_hw(arrays["original_hw"], "NPZ original_hw")
        if original_hw != tuple(record["original_hw"]):
            raise ValueError(f"NPZ original_hw mismatch for records[{index}]")
        if probability.shape != original_hw or raw_logit.shape != original_hw:
            raise ValueError(f"NPZ arrays are not native resolution for records[{index}]")
        if probability.dtype != np.dtype("float64"):
            raise TypeError(f"NPZ prob must be exact float64 for records[{index}]")
        if raw_logit.dtype != np.dtype("float64"):
            raise TypeError(f"NPZ raw_logit must be exact float64 for records[{index}]")
        if not np.isfinite(probability).all() or not np.isfinite(raw_logit).all():
            raise ValueError(f"NPZ arrays contain NaN/Inf for records[{index}]")
        if probability.size and (
            float(probability.min()) < 0.0 or float(probability.max()) > 1.0
        ):
            raise ValueError(f"NPZ prob is outside [0, 1] for records[{index}]")
        for field in ("canonical_id", "image_id", "source_domain", "resize_mode"):
            if _npz_string(arrays[field], field) != record[field]:
                raise ValueError(f"NPZ {field} mismatch for records[{index}]")
        for field, length in (
            ("input_hw", 2),
            ("resized_hw", 2),
            ("padding_ltrb", 4),
        ):
            values = _npz_int_tuple(arrays[field], field, length)
            if values != tuple(record[field]):
                raise ValueError(f"NPZ {field} mismatch for records[{index}]")

    if _sha256_file_stable(score_path) != score_before:
        raise RuntimeError(f"score file changed while verified: {score_path}")


def _read_bound_json(binding: Mapping[str, str], root: Path) -> Any:
    path = _resolve_repository_file(root, binding["path"], "bound JSON")
    before = _sha256_file_stable(path)
    if before != binding["sha256"]:
        raise ValueError(f"bound JSON SHA-256 mismatch: {binding['path']}")
    payload = _read_json_file(path, binding["path"])
    if _sha256_file_stable(path) != before:
        raise RuntimeError(f"bound JSON changed while read: {binding['path']}")
    return payload


def _read_json_file(path: Path, name: str) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _repository_root(value: str | Path | None) -> Path:
    root = REPOSITORY_ROOT if value is None else Path(value).expanduser()
    if root.is_symlink():
        raise ValueError("repository_root must not be a symlink")
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository_root is not a directory: {root}")
    return root


def _existing_direct_path(value: str | Path, root: Path, name: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside repository_root") from error
    _reject_symlink_components(candidate, root, name)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository_root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved


def _relative_repository_path(value: object, name: str) -> str:
    rendered = _nonempty_string(value, name)
    if "\\" in rendered:
        raise ValueError(f"{name} must use POSIX separators")
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or rendered.startswith("~"):
        raise ValueError(f"{name} must be repository-relative")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{name} contains path traversal or non-canonical components")
    if pure.as_posix() != rendered:
        raise ValueError(f"{name} must be a canonical POSIX path")
    _reject_official_test_like_path(pure, name)
    return rendered


def _reject_official_test_like_path(path: PurePosixPath, name: str) -> None:
    for raw_part in path.parts:
        part = raw_part.lower()
        stem = PurePosixPath(part).stem
        if (
            part in {"test", "official_test", "official-test"}
            or stem.startswith("test_")
            or stem.startswith("official_test")
        ):
            raise ValueError(f"{name} names an official-test-like path")


def _resolve_repository_file(root: Path, value: object, name: str) -> Path:
    relative = _relative_repository_path(value, name)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    _reject_symlink_components(candidate, root, name)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {candidate}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository_root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {resolved}")
    return resolved


def _reject_symlink_components(candidate: Path, root: Path, name: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository_root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} contains a symlinked path component")


def _sha256_file_stable(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_value(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase 64-character SHA-256 string")
    if value != value.lower() or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
    return value


def _required_role(value: object) -> str:
    role = _nonempty_string(value, "required_role")
    if role not in STAGE2_DEVELOPMENT_ROLES:
        raise ValueError(
            "required_role must be one of " + ", ".join(STAGE2_DEVELOPMENT_ROLES)
        )
    return role


def _domain(value: object, name: str) -> str:
    domain = _nonempty_string(value, name)
    if domain not in STAGE2_DOMAINS:
        raise ValueError(f"{name} must be one of {STAGE2_DOMAINS}")
    return domain


def _exact_bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise TypeError(f"{name} must be the exact JSON boolean {str(expected).lower()}")


def _exact_int(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact JSON integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty, whitespace-trimmed string")
    return value


def _parse_hw(value: object, name: str) -> tuple[int, int]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size != 2 or value.dtype.kind not in {"i", "u"}:
            raise TypeError(f"{name} must be an integer [height, width] vector")
        result = tuple(int(item) for item in value.tolist())
        if any(item <= 0 for item in result):
            raise ValueError(f"{name} values must be positive")
        return result  # type: ignore[return-value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{name} must be [height, width]")
    return (
        _exact_int(value[0], f"{name}[0]", minimum=1),
        _exact_int(value[1], f"{name}[1]", minimum=1),
    )


def _parse_padding(value: object, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise TypeError(f"{name} must be [left, top, right, bottom]")
    return tuple(
        _exact_int(item, f"{name}[{index}]", minimum=0)
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _npz_string(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"NPZ {name} must be a true 0-D scalar")
    result = array.item()
    if not isinstance(result, str):
        raise TypeError(f"NPZ {name} must be a string")
    return result


def _npz_int_tuple(value: object, name: str, length: int) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size != length or array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"NPZ {name} must be an integer vector of length {length}")
    result = tuple(int(item) for item in array.tolist())
    if any(item < 0 for item in result):
        raise ValueError(f"NPZ {name} contains negative values")
    return result


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _update_frame(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big", signed=False))
    digest.update(encoded)


_MISSING = object()


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> object:
    for name in names:
        if name in mapping:
            return mapping[name]
    return _MISSING


__all__ = [
    "BASE_SEEDS",
    "BINDING_NAMES",
    "FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE",
    "OOF_HOLDOUT_STAGE2_FIT",
    "OOF_TRAIN_SOURCE_REFERENCE",
    "OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT",
    "SOURCE_DIAGNOSTIC_VALIDATION",
    "STAGE2_DEVELOPMENT_ROLES",
    "STAGE2_SCORE_MANIFEST_SCHEMA",
    "STAGE2_SCORE_RECORDS_ALGORITHM",
    "STRICT_THRESHOLD_SEMANTICS",
    "VerifiedStage2ScoreItem",
    "VerifiedStage2ScoreManifest",
    "stage2_score_records_sha256",
    "verify_stage2_score_manifest",
]
