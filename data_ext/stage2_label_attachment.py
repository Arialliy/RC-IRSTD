"""Fail-closed Stage2 per-window label attachment v2.

This module is intentionally additive.  It does not alter or broaden the
legacy label-v1 consumer.  A v2 attachment is development-only, binds one
explicit C14/Q28 window and one score-manifest v4, and contains labels for the
28 ordered query records only.  Context mask paths are never constructed,
resolved, statted or opened by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from data_ext.stage2_score_manifest import (
    BINDING_NAMES,
    OOF_HOLDOUT_STAGE2_FIT,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    SOURCE_DIAGNOSTIC_VALIDATION,
    STRICT_THRESHOLD_SEMANTICS,
    VerifiedStage2ScoreItem,
    VerifiedStage2ScoreManifest,
    verify_stage2_score_manifest,
)
from data_ext.stage2_threshold_decision import (
    PIXEL_BUDGET_GRID,
    VerifiedStage2ThresholdDecisionSet,
    verify_stage2_threshold_decision_set,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

STAGE2_LABEL_SCHEMA = "rc-irstd.label-attachment.v2"
STAGE2_LABEL_ARTIFACT_TYPE = "rc_irstd_stage2_query_label_attachment"
STAGE2_LABEL_CONTENT_ALGORITHM = (
    "sha256-canonical-json-ordered-stage2-query-label-records-v2"
)
STAGE2_QUERY_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-ordered-stage2-query-identity-v1"
)
STAGE2_WINDOW_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-selected-stage2-c14q28-window-v1"
)

QUERY_BEARING_ROLES = (
    OOF_HOLDOUT_STAGE2_FIT,
    SOURCE_DIAGNOSTIC_VALIDATION,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
)
WINDOW_EPISODE_ROLE = {
    OOF_HOLDOUT_STAGE2_FIT: "stage2_oof_fit",
    SOURCE_DIAGNOSTIC_VALIDATION: SOURCE_DIAGNOSTIC_VALIDATION,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT: OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
}

B2_AUTHORIZATION_BINDING = {
    "path": (
        "outputs/stage2_protocol/"
        "RC4_STAGE2_B2_EPISODE_REFERENCE_AUTHORIZATION_AMENDMENT_20260717.json"
    ),
    "sha256": "cc15832de4f85abfae84c4d49a5ac098cff253d0fecfa885d0d7735d3ef5aea6",
}
B1_PASS_BINDING = {
    "path": (
        "outputs/stage2_protocol/"
        "RC4_STAGE2_B1_CONTRACT_SPINE_INTEGRATION_PASS_20260716.json"
    ),
    "sha256": "4d4ce52653e872ffa4f3f71b9475edb03b934b27d0cb4d6c914d63c92b0131d6",
}
SEMANTICS_BINDING = {
    "path": (
        "outputs/stage2_protocol/"
        "RC4_STAGE2_PRE_G1_RESULT_FREE_ANALYSIS_PLAN_AMENDMENT_SEMANTICS_V1_20260716.json"
    ),
    "sha256": "c60e087116f98a3e59772792e16be389cc2961180b7a9c5de930e2b9cd9abef7",
}
GOVERNANCE_BINDINGS = {
    "b2_authorization": B2_AUTHORIZATION_BINDING,
    "b1_integration_pass": B1_PASS_BINDING,
    "stage2_semantics": SEMANTICS_BINDING,
}

WINDOW_SCHEMA = "rc-irstd.stage2-role-pure-c14q28-windows.v1"
WINDOW_ARTIFACT_TYPE = "rc_irstd_stage2_role_pure_episode_windows"
WINDOW_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "execution_authorized",
        "observed_results",
        "outer_fold_id",
        "outer_target_domain",
        "domain",
        "source_role",
        "episode_role",
        "oof_fold_index",
        "geometry",
        "role_purity",
        "ordered_role_record_count",
        "complete_window_count",
        "window_record_count",
        "unused_suffix",
        "role_binding",
        "bound_inputs",
        "guardrails",
        "windows",
    }
)
WINDOW_ENTRY_FIELDS = frozenset(
    {"window_index", "window_id", "context_records", "query_records"}
)
BASE_WINDOW_RECORD_FIELDS = frozenset(
    {
        "canonical_id",
        "image_id",
        "original_image_path",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
        "source_role",
        "outer_fold_id",
        "episode_role",
    }
)
GUARDRAIL_FIELDS = frozenset(
    {
        "development_only",
        "result_free",
        "execution_authorized",
        "official_test_split_files_opened",
        "official_test_ids_materialized",
        "official_test_images_opened",
        "mask_or_label_files_opened",
        "predictions_scores_checkpoints_or_metrics_opened",
        "original_training_images_opened_only_for_sha256",
    }
)

LABEL_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "execution_scope",
        "official_test_accessed",
        "labels_embedded_in_scores",
        "native_resolution",
        "path_anchor",
        "role",
        "window_id",
        "outer_fold_id",
        "outer_target",
        "source_domain",
        "base_seed",
        "derived_seed",
        "detector_role",
        "oof_fold_index",
        "threshold_semantics",
        "query_size",
        "decision_seal_binding",
        "window_binding",
        "score_manifest_binding",
        "score_bindings",
        "governance_bindings",
        "ordered_query_identity_sha256_algorithm",
        "ordered_query_identity_sha256",
        "labels_content_sha256_algorithm",
        "labels_content_sha256",
        "labels",
    }
)
WINDOW_BINDING_FIELDS = frozenset(
    {"path", "sha256", "window_id", "window_identity_sha256"}
)
SCORE_MANIFEST_BINDING_FIELDS = frozenset(
    {"path", "sha256", "records_content_sha256"}
)
DECISION_SEAL_BINDING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "decision_set_content_sha256",
        "context_package_sha256",
        "context_package_commit_sha256",
        "statistics_config_path",
        "statistics_config_sha256",
    }
)
LABEL_RECORD_FIELDS = frozenset(
    {
        "record_index",
        "window_id",
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
        "label_file",
        "label_file_sha256",
        "source_mask_path",
        "source_mask_file_sha256",
        "source_mask_original_hw",
        "alignment_provenance",
    }
)
ALIGNMENT_FIELDS = frozenset(
    {
        "policy",
        "policy_module_path",
        "policy_module_sha256",
        "interpolation",
        "operation",
        "source_mask_original_hw",
        "target_image_hw",
        "aspect_ratio_relative_error",
        "aspect_tolerance",
        "mask_aligned_to_image_geometry",
        "silent_crop_used",
        "bilinear_resize_used",
        "nuaa_misc_111_policy_applied",
    }
)
LABEL_NPZ_FIELDS = frozenset(
    {
        "mask",
        "canonical_id",
        "image_id",
        "source_domain",
        "original_hw",
        "source_mask_original_hw",
        "alignment_operation",
        "interpolation",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class VerifiedStage2Window:
    """One exact selected C14/Q28 window, verified without label access."""

    path: Path
    payload: Mapping[str, Any]
    window: Mapping[str, Any]
    context_records: tuple[Mapping[str, Any], ...]
    query_records: tuple[Mapping[str, Any], ...]
    role: str
    window_id: str
    manifest_sha256: str
    window_identity_sha256: str


@dataclass(frozen=True)
class VerifiedStage2LabelItem:
    record_index: int
    canonical_id: str
    image_id: str
    record: Mapping[str, Any]
    label_path: Path
    label_file_sha256: str
    original_hw: tuple[int, int]
    score_item: VerifiedStage2ScoreItem


@dataclass(frozen=True)
class VerifiedStage2LabelAttachment:
    path: Path
    repository_root: Path
    payload: Mapping[str, Any]
    score_manifest: VerifiedStage2ScoreManifest
    window: VerifiedStage2Window
    records: tuple[Mapping[str, Any], ...]
    items: tuple[VerifiedStage2LabelItem, ...]
    manifest_sha256: str
    labels_content_sha256: str
    ordered_query_identity_sha256: str


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of closed canonical JSON (NaN is always rejected)."""

    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stage2_ordered_query_identity(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project the exact identity boundary used by downstream Stage2."""

    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"query_records[{index}] must be an object")
        result.append(
            {
                "record_index": index,
                "canonical_id": _nonempty(record.get("canonical_id"), "canonical_id"),
                "image_id": _nonempty(record.get("image_id"), "image_id"),
                "original_image_sha256": _sha256(
                    record.get("original_image_sha256"), "original_image_sha256"
                ),
                "exclusion_group_id": _nonempty(
                    record.get("exclusion_group_id"), "exclusion_group_id"
                ),
                "near_duplicate_cluster_id_or_unique_sentinel": _nonempty(
                    record.get("near_duplicate_cluster_id_or_unique_sentinel"),
                    "near_duplicate_cluster_id_or_unique_sentinel",
                ),
                "source_role_record_index": _exact_int(
                    record.get("source_role_record_index"),
                    "source_role_record_index",
                    minimum=0,
                ),
            }
        )
    return result


def stage2_label_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    if not isinstance(records, (list, tuple)) or len(records) != 28:
        raise ValueError("Stage2 label records must be an ordered Q28 sequence")
    return canonical_json_sha256(list(records))


def verify_stage2_window_contract(
    path: str | Path,
    expected_sha256: str,
    window_id: str,
    expected_role: str,
    *,
    repository_root: str | Path | None = None,
) -> VerifiedStage2Window:
    """Verify one selected window before any mask path can be constructed."""

    root = _repository_root(repository_root)
    role = _role(expected_role)
    requested_window_id = _nonempty(window_id, "window_id")
    window_path = _direct_input_file(path, root, "window manifest")
    expected_digest = _sha256(expected_sha256, "window_manifest_sha256")
    payload, digest = _read_json_stable(window_path, "window manifest")
    if digest != expected_digest:
        raise ValueError("window manifest SHA-256 mismatch")
    if not isinstance(payload, Mapping):
        raise TypeError("window manifest must contain an object")
    _exact_keys(payload, WINDOW_FIELDS, "window manifest")
    if payload["schema_version"] != WINDOW_SCHEMA:
        raise ValueError("unsupported Stage2 window schema")
    if payload["artifact_type"] != WINDOW_ARTIFACT_TYPE:
        raise ValueError("window artifact_type mismatch")
    if payload["artifact_status"] != "DEVELOPMENT_ONLY_RESULT_FREE":
        raise ValueError("window artifact_status mismatch")
    _exact_bool(payload["execution_authorized"], "execution_authorized", False)
    if payload["observed_results"] is not None:
        raise ValueError("window observed_results must be exactly null")
    _verify_guardrails(payload["guardrails"])

    if payload["episode_role"] != WINDOW_EPISODE_ROLE[role]:
        raise ValueError("window episode_role does not match expected role")
    source_role = _nonempty(payload["source_role"], "source_role")
    expected_source_role = (
        "detector_fit" if role == OOF_HOLDOUT_STAGE2_FIT else "detector_diagnostic"
    )
    if source_role != expected_source_role:
        raise ValueError("window source_role does not match expected role")
    geometry = payload["geometry"]
    if not isinstance(geometry, Mapping) or dict(geometry) != {
        "block_size": 42,
        "construction": (
            "ordered_non_overlapping_contiguous_blocks_context_first_query_second"
        ),
        "context_size": 14,
        "query_size": 28,
    }:
        raise ValueError("window geometry must be the frozen C14/Q28 contract")
    _verify_role_purity(payload["role_purity"], role)

    _verify_binding_file(payload["role_binding"], root, "role_binding")
    _verify_binding_file(payload["unused_suffix"], root, "unused_suffix")
    bound_inputs = payload["bound_inputs"]
    if not isinstance(bound_inputs, Mapping) or set(bound_inputs) != {
        "image_only_near_duplicate_audit",
        "k2_geometry_prefreeze_audit",
        "official_train_derived_split_manifest",
    }:
        raise ValueError("window bound_inputs closure mismatch")
    for name, binding in bound_inputs.items():
        _verify_binding_file(binding, root, f"bound_inputs.{name}")

    windows = payload["windows"]
    complete_count = _exact_int(
        payload["complete_window_count"], "complete_window_count", minimum=1
    )
    if not isinstance(windows, list) or len(windows) != complete_count:
        raise ValueError("window list/count mismatch")
    if _exact_int(payload["window_record_count"], "window_record_count", minimum=42) != 42 * len(windows):
        raise ValueError("window_record_count mismatch")
    _exact_int(payload["ordered_role_record_count"], "ordered_role_record_count", minimum=42)

    selected: Mapping[str, Any] | None = None
    all_ids: set[str] = set()
    last_source_index = -1
    for index, raw_window in enumerate(windows):
        if not isinstance(raw_window, Mapping):
            raise TypeError(f"windows[{index}] must be an object")
        _exact_keys(raw_window, WINDOW_ENTRY_FIELDS, f"windows[{index}]")
        if _exact_int(raw_window["window_index"], f"windows[{index}].window_index", minimum=0) != index:
            raise ValueError("window_index must be contiguous and ordered")
        current_id = _nonempty(raw_window["window_id"], f"windows[{index}].window_id")
        if current_id in all_ids:
            raise ValueError("duplicate window_id")
        all_ids.add(current_id)
        context = _window_records(
            raw_window["context_records"],
            expected_count=14,
            episode_role="context",
            payload=payload,
            role=role,
            name=f"windows[{index}].context_records",
        )
        query = _window_records(
            raw_window["query_records"],
            expected_count=28,
            episode_role="query",
            payload=payload,
            role=role,
            name=f"windows[{index}].query_records",
        )
        for boundary in (
            "canonical_id",
            "original_image_sha256",
            "near_duplicate_cluster_id_or_unique_sentinel",
            "exclusion_group_id",
        ):
            if {str(row[boundary]) for row in context}.intersection(
                str(row[boundary]) for row in query
            ):
                raise ValueError(f"context/query overlap at {boundary}")
        for record in context + query:
            current_source_index = int(record["source_role_record_index"])
            if current_source_index <= last_source_index:
                raise ValueError("window records are not in strict source-role order")
            last_source_index = current_source_index
        if current_id == requested_window_id:
            selected = raw_window
    if selected is None:
        raise KeyError(f"window_id is absent from manifest: {requested_window_id!r}")

    context_records = tuple(selected["context_records"])
    query_records = tuple(selected["query_records"])
    window_identity_sha = canonical_json_sha256(selected)
    if _hash_file(window_path) != digest:
        raise RuntimeError("window manifest changed while verified")
    return VerifiedStage2Window(
        path=window_path,
        payload=payload,
        window=selected,
        context_records=context_records,
        query_records=query_records,
        role=role,
        window_id=requested_window_id,
        manifest_sha256=digest,
        window_identity_sha256=window_identity_sha,
    )


def verify_stage2_label_attachment(
    score_manifest: str | Path,
    label_manifest: str | Path,
    expected_role: str,
    *,
    score_manifest_sha256: str,
    label_manifest_sha256: str,
    window_manifest: str | Path,
    window_manifest_sha256: str,
    window_id: str,
    repository_root: str | Path | None = None,
    bundle_root_override: str | Path | None = None,
    _owned_publication_lock: tuple[str | Path, tuple[int, int]] | None = None,
) -> VerifiedStage2LabelAttachment:
    """Verify score-v4, one exact window, and exactly its 28 query labels.

    ``bundle_root_override`` is a producer-only seam used to validate a
    private same-parent staging directory whose manifest already records the
    future repository-relative final paths.  Public consumers omit it.
    """

    root = _repository_root(repository_root)
    role = _role(expected_role)

    # Ordering is security-significant: window identities and score records
    # are resolved before any label path from the label manifest is touched.
    window = verify_stage2_window_contract(
        window_manifest,
        window_manifest_sha256,
        window_id,
        role,
        repository_root=root,
    )
    score = verify_stage2_score_manifest(
        score_manifest,
        _sha256(score_manifest_sha256, "score_manifest_sha256"),
        role,
        repository_root=root,
    )
    _bind_window_to_score(window, score)

    manifest_path = _direct_input_file(label_manifest, root, "label manifest", allow_override=bundle_root_override)
    _verify_bundle_publication_lock(
        manifest_path.parent,
        _owned_publication_lock,
    )
    expected_label_sha = _sha256(label_manifest_sha256, "label_manifest_sha256")
    payload, manifest_sha = _read_json_stable(manifest_path, "label manifest")
    if manifest_sha != expected_label_sha:
        raise ValueError("label manifest SHA-256 mismatch")
    if not isinstance(payload, Mapping):
        raise TypeError("label manifest must contain an object")
    _exact_keys(payload, LABEL_MANIFEST_FIELDS, "label manifest")
    _verify_label_top_level(payload, score, window, role, root)

    raw_records = payload["labels"]
    if not isinstance(raw_records, list) or len(raw_records) != 28:
        raise ValueError("label manifest must contain exactly 28 query labels")
    query_identity = stage2_ordered_query_identity(window.query_records)
    query_identity_sha = canonical_json_sha256(query_identity)
    if payload["ordered_query_identity_sha256_algorithm"] != STAGE2_QUERY_IDENTITY_ALGORITHM:
        raise ValueError("ordered query identity algorithm mismatch")
    if _sha256(payload["ordered_query_identity_sha256"], "ordered_query_identity_sha256") != query_identity_sha:
        raise ValueError("ordered query identity SHA-256 mismatch")

    score_by_canonical = {item.canonical_id: item for item in score.items}
    records: list[Mapping[str, Any]] = []
    items: list[VerifiedStage2LabelItem] = []
    seen_paths: set[str] = set()
    for index, (raw, query_record) in enumerate(zip(raw_records, window.query_records, strict=True)):
        if not isinstance(raw, Mapping):
            raise TypeError(f"labels[{index}] must be an object")
        _exact_keys(raw, LABEL_RECORD_FIELDS, f"labels[{index}]")
        record = dict(raw)
        score_item = score_by_canonical.get(str(query_record["canonical_id"]))
        if score_item is None:
            raise ValueError("query identity is absent from score manifest")
        _verify_label_record_metadata(record, query_record, score_item, window, index, root)
        label_relative = _relative_repository_path(record["label_file"], f"labels[{index}].label_file")
        if label_relative in seen_paths:
            raise ValueError("duplicate label_file path")
        seen_paths.add(label_relative)
        label_path = _resolve_bundle_artifact(
            root,
            label_relative,
            bundle_root_override=bundle_root_override,
            manifest_path=manifest_path,
            name=f"labels[{index}].label_file",
        )
        declared_sha = _sha256(record["label_file_sha256"], f"labels[{index}].label_file_sha256")
        if _hash_file_stable(label_path) != declared_sha:
            raise ValueError("label NPZ SHA-256 mismatch")
        original_hw = _parse_hw(record["original_hw"], f"labels[{index}].original_hw")
        _verify_label_npz(label_path, record, original_hw)
        if _hash_file_stable(label_path) != declared_sha:
            raise RuntimeError("label NPZ changed while verified")
        records.append(record)
        items.append(
            VerifiedStage2LabelItem(
                record_index=index,
                canonical_id=str(record["canonical_id"]),
                image_id=str(record["image_id"]),
                record=record,
                label_path=label_path,
                label_file_sha256=declared_sha,
                original_hw=original_hw,
                score_item=score_item,
            )
        )
    if payload["labels_content_sha256_algorithm"] != STAGE2_LABEL_CONTENT_ALGORITHM:
        raise ValueError("label content algorithm mismatch")
    records_sha = stage2_label_records_sha256(records)
    if _sha256(payload["labels_content_sha256"], "labels_content_sha256") != records_sha:
        raise ValueError("ordered label content SHA-256 mismatch")
    if _hash_file(manifest_path) != manifest_sha:
        raise RuntimeError("label manifest changed while verified")
    _verify_governance(root)
    return VerifiedStage2LabelAttachment(
        path=manifest_path,
        repository_root=root,
        payload=payload,
        score_manifest=score,
        window=window,
        records=tuple(records),
        items=tuple(items),
        manifest_sha256=manifest_sha,
        labels_content_sha256=records_sha,
        ordered_query_identity_sha256=query_identity_sha,
    )


def _verify_bundle_publication_lock(
    bundle_root: Path,
    owned_lock: tuple[str | Path, tuple[int, int]] | None,
) -> None:
    """Reject an in-flight public bundle unless the producer proves ownership."""

    lock = bundle_root.parent / f".{bundle_root.name}.lock"
    if not os.path.lexists(lock):
        return
    if owned_lock is None:
        raise RuntimeError(f"bundle publication lock is active: {lock}")
    allowed_path = Path(owned_lock[0]).expanduser().absolute()
    allowed_identity = owned_lock[1]
    if allowed_path != lock.absolute():
        raise RuntimeError("bundle publication lock authorization path mismatch")
    info = lock.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != allowed_identity:
        raise RuntimeError("bundle publication lock authorization inode mismatch")


def load_stage2_label_mask(item: VerifiedStage2LabelItem) -> np.ndarray:
    """Reload a verified binary label with immutable-byte revalidation."""

    if _hash_file_stable(item.label_path) != item.label_file_sha256:
        raise RuntimeError("label NPZ changed after attachment verification")
    with np.load(item.label_path, allow_pickle=False) as payload:
        mask = np.asarray(payload["mask"], dtype=np.uint8)
    if mask.shape != item.original_hw or not np.isin(mask, (0, 1)).all():
        raise RuntimeError("verified label NPZ contract changed")
    if _hash_file_stable(item.label_path) != item.label_file_sha256:
        raise RuntimeError("label NPZ changed while reloaded")
    return mask


def governance_bindings_payload(repository_root: str | Path | None = None) -> dict[str, dict[str, str]]:
    root = _repository_root(repository_root)
    _verify_governance(root)
    return {name: dict(binding) for name, binding in GOVERNANCE_BINDINGS.items()}


def _verify_label_top_level(
    payload: Mapping[str, Any],
    score: VerifiedStage2ScoreManifest,
    window: VerifiedStage2Window,
    role: str,
    root: Path,
) -> None:
    if role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT:
        _verify_decision_seal_binding(
            payload["decision_seal_binding"], score=score, window=window, root=root
        )
    elif payload["decision_seal_binding"] is not None:
        raise ValueError("source/OOF decision_seal_binding must be exactly null")
    exact = {
        "schema_version": STAGE2_LABEL_SCHEMA,
        "artifact_type": STAGE2_LABEL_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "execution_scope": "stage2_development_query_labels",
        "path_anchor": "repository_root",
        "role": role,
        "window_id": window.window_id,
        "outer_fold_id": score.payload["outer_fold_id"],
        "outer_target": score.payload["outer_target"],
        "source_domain": score.payload["source_domain"],
        "base_seed": score.payload["base_seed"],
        "derived_seed": score.payload["derived_seed"],
        "detector_role": score.payload["detector_role"],
        "oof_fold_index": score.payload["oof_fold_index"],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "query_size": 28,
    }
    for field, expected in exact.items():
        if payload[field] != expected:
            raise ValueError(f"label manifest {field} mismatch")
    for field, expected in {
        "development_only": True,
        "official_test_accessed": False,
        "labels_embedded_in_scores": False,
        "native_resolution": True,
    }.items():
        _exact_bool(payload[field], field, expected)

    window_binding = payload["window_binding"]
    if not isinstance(window_binding, Mapping):
        raise TypeError("window_binding must be an object")
    _exact_keys(window_binding, WINDOW_BINDING_FIELDS, "window_binding")
    expected_window_binding = {
        "path": _repo_relative(window.path, root),
        "sha256": window.manifest_sha256,
        "window_id": window.window_id,
        "window_identity_sha256": window.window_identity_sha256,
    }
    if dict(window_binding) != expected_window_binding:
        raise ValueError("window binding mismatch")

    score_binding = payload["score_manifest_binding"]
    if not isinstance(score_binding, Mapping):
        raise TypeError("score_manifest_binding must be an object")
    _exact_keys(score_binding, SCORE_MANIFEST_BINDING_FIELDS, "score_manifest_binding")
    expected_score_binding = {
        "path": _repo_relative(score.path, root),
        "sha256": score.manifest_sha256,
        "records_content_sha256": score.records_content_sha256,
    }
    if dict(score_binding) != expected_score_binding:
        raise ValueError("score-manifest binding mismatch")
    if not isinstance(payload["score_bindings"], Mapping) or set(payload["score_bindings"]) != set(BINDING_NAMES):
        raise ValueError("score-v4 ten-binding closure mismatch")
    if dict(payload["score_bindings"]) != dict(score.bindings):
        raise ValueError("score-v4 ten bindings are not preserved byte-for-byte")
    governance = payload["governance_bindings"]
    if not isinstance(governance, Mapping) or dict(governance) != governance_bindings_payload(root):
        raise ValueError("governance binding closure mismatch")


def _verify_decision_seal_binding(
    value: Any,
    *,
    score: VerifiedStage2ScoreManifest,
    window: VerifiedStage2Window,
    root: Path,
) -> VerifiedStage2ThresholdDecisionSet:
    """Re-establish the causal T0--T8 seal before any label NPZ is touched."""

    if not isinstance(value, Mapping):
        raise RuntimeError(
            "outer-target label attachment requires a verified complete T0-T8 "
            "decision_seal_binding before any label path operation"
        )
    _exact_keys(value, DECISION_SEAL_BINDING_FIELDS, "decision_seal_binding")
    relative = _relative_repository_path(value["path"], "decision_seal_binding.path")
    external_sha = _sha256(value["sha256"], "decision_seal_binding.sha256")
    content_sha = _sha256(
        value["decision_set_content_sha256"],
        "decision_seal_binding.decision_set_content_sha256",
    )
    context_sha = _sha256(
        value["context_package_sha256"],
        "decision_seal_binding.context_package_sha256",
    )
    context_commit_sha = _sha256(
        value["context_package_commit_sha256"],
        "decision_seal_binding.context_package_commit_sha256",
    )
    statistics_config_relative = _relative_repository_path(
        value["statistics_config_path"],
        "decision_seal_binding.statistics_config_path",
    )
    statistics_config_sha = _sha256(
        value["statistics_config_sha256"],
        "decision_seal_binding.statistics_config_sha256",
    )
    verified = verify_stage2_threshold_decision_set(
        relative,
        external_sha,
        expected_context_package_sha256=context_sha,
        expected_context_commit_sha256=context_commit_sha,
        expected_window_id=window.window_id,
        expected_outer_fold_id=str(score.payload["outer_fold_id"]),
        expected_base_seed=int(score.payload["base_seed"]),
        expected_derived_seed=int(score.payload["derived_seed"]),
        expected_detector_checkpoint_sha256=str(
            score.bindings["checkpoint"]["sha256"]
        ),
        expected_budget_grid=PIXEL_BUDGET_GRID,
        repository_root=root,
    )
    shared = verified.payload["shared_bindings"]
    # Import only after the outer-only decision bundle has passed.  The Lane-A
    # verifier opens and deterministically replays the actual context package
    # and commit; SHA strings carried by the decision JSON are not authority.
    from rc.stage2_crossfit_schema import (
        verify_stage2_context_package,
        verify_stage2_statistics_config,
    )

    statistics_config_path = _repository_file(
        root,
        statistics_config_relative,
        "decision_seal_binding.statistics_config",
    )
    verified_statistics_config = verify_stage2_statistics_config(
        statistics_config_path,
        statistics_config_sha,
        repository_root=root,
    )

    declared_context_path = str(shared["context_package"]["path"])
    pure_context_path = PurePosixPath(declared_context_path)
    if (
        pure_context_path.is_absolute()
        or pure_context_path.as_posix() != declared_context_path
        or any(part in {"", ".", ".."} for part in pure_context_path.parts)
    ):
        raise ValueError("decision_seal_binding context path is not canonical repository-relative")
    try:
        context = verify_stage2_context_package(
            root.joinpath(*pure_context_path.parts),
            shared["context_package"]["sha256"],
            shared["context_package_commit"]["sha256"],
            statistics_config=verified_statistics_config,
            repository_root=root,
        )
    except FileNotFoundError as error:
        raise ValueError("decision_seal_binding context package does not exist") from error
    expected_query_identity = canonical_json_sha256(
        stage2_ordered_query_identity(window.query_records)
    )
    context_payload = context.payload
    detector_identity = context_payload["detector_identity"]
    exact = (
        (_repo_relative(verified.path, root), relative, "path"),
        (verified.manifest_sha256, external_sha, "sha256"),
        (
            verified.payload["decision_set_content_sha256"],
            content_sha,
            "decision_set_content_sha256",
        ),
        (
            verified.payload["outer_target_domain"],
            score.payload["outer_target"],
            "outer_target_domain",
        ),
        (
            shared["window_identity_sha256"],
            window.window_identity_sha256,
            "window_identity_sha256",
        ),
        (
            shared["ordered_query_identity_sha256"],
            expected_query_identity,
            "ordered_query_identity_sha256",
        ),
        (
            shared["score_manifest_sha256"],
            score.manifest_sha256,
            "score_manifest_sha256",
        ),
        (
            shared["score_records_content_sha256"],
            score.records_content_sha256,
            "score_records_content_sha256",
        ),
        (
            _repo_relative(context.path, root),
            shared["context_package"]["path"],
            "context_package.path",
        ),
        (
            context.context_sha256,
            shared["context_package"]["sha256"],
            "context_package.sha256",
        ),
        (
            _repo_relative(context.commit_path, root),
            shared["context_package_commit"]["path"],
            "context_package_commit.path",
        ),
        (
            context.commit_sha256,
            shared["context_package_commit"]["sha256"],
            "context_package_commit.sha256",
        ),
        (context_payload["expected_role"], window.role, "context.expected_role"),
        (context_payload["outer_fold_id"], score.payload["outer_fold_id"], "context.outer_fold_id"),
        (context_payload["outer_target"], score.payload["outer_target"], "context.outer_target"),
        (context_payload["source_domain"], score.payload["source_domain"], "context.source_domain"),
        (context_payload["base_seed"], score.payload["base_seed"], "context.base_seed"),
        (context_payload["derived_seed"], score.payload["derived_seed"], "context.derived_seed"),
        (
            context_payload["source_ordered_query_identity_sha256"],
            expected_query_identity,
            "context.source_ordered_query_identity_sha256",
        ),
        (
            context_payload["window_binding"]["window_id"],
            window.window_id,
            "context.window_id",
        ),
        (
            context_payload["window_binding"]["path"],
            _repo_relative(window.path, root),
            "context.window_path",
        ),
        (
            context_payload["window_binding"]["sha256"],
            window.manifest_sha256,
            "context.window_sha256",
        ),
        (
            context_payload["window_binding"]["window_identity_sha256"],
            window.window_identity_sha256,
            "context.window_identity_sha256",
        ),
        (
            context_payload["score_manifest_binding"]["sha256"],
            score.manifest_sha256,
            "context.score_manifest_sha256",
        ),
        (
            context_payload["score_manifest_binding"]["path"],
            _repo_relative(score.path, root),
            "context.score_manifest_path",
        ),
        (
            context_payload["score_manifest_binding"]["role"],
            score.role,
            "context.score_manifest_role",
        ),
        (
            context_payload["score_manifest_binding"]["records_content_sha256"],
            score.records_content_sha256,
            "context.score_records_content_sha256",
        ),
        (
            detector_identity["checkpoint_sha256"],
            score.bindings["checkpoint"]["sha256"],
            "context.detector_checkpoint_sha256",
        ),
        (detector_identity["outer_fold_id"], score.payload["outer_fold_id"], "context.detector_outer_fold_id"),
        (detector_identity["outer_target"], score.payload["outer_target"], "context.detector_outer_target"),
        (detector_identity["base_seed"], score.payload["base_seed"], "context.detector_base_seed"),
        (detector_identity["derived_seed"], score.payload["derived_seed"], "context.detector_derived_seed"),
        (detector_identity["detector_role"], score.payload["detector_role"], "context.detector_role"),
        (detector_identity["oof_fold_index"], score.payload["oof_fold_index"], "context.detector_oof_fold_index"),
        (
            context_payload["statistics_config_binding"]["path"],
            statistics_config_relative,
            "context.statistics_config_path",
        ),
        (
            context_payload["statistics_config_binding"]["sha256"],
            statistics_config_sha,
            "context.statistics_config_sha256",
        ),
    )
    for observed, expected, name in exact:
        if observed != expected:
            raise ValueError(f"decision_seal_binding {name} mismatch")
    if dict(context_payload["score_bindings"]) != dict(score.bindings):
        raise ValueError("decision_seal_binding context.score_bindings mismatch")
    if context.window.window_identity_sha256 != window.window_identity_sha256:
        raise ValueError("decision_seal_binding verified context window capability mismatch")
    if context.score_manifest.manifest_sha256 != score.manifest_sha256:
        raise ValueError("decision_seal_binding verified context score capability mismatch")
    return verified


def _bind_window_to_score(
    window: VerifiedStage2Window,
    score: VerifiedStage2ScoreManifest,
) -> None:
    payload = window.payload
    score_payload = score.payload
    pairs = (
        (payload["outer_fold_id"], score_payload["outer_fold_id"], "outer_fold_id"),
        (payload["outer_target_domain"], score_payload["outer_target"], "outer_target"),
        (payload["domain"], score_payload["source_domain"], "source_domain"),
        (payload["oof_fold_index"], score_payload["oof_fold_index"], "oof_fold_index"),
    )
    for left, right, name in pairs:
        if left != right:
            raise ValueError(f"window/score {name} mismatch")
    score_by_canonical = {item.canonical_id: item for item in score.items}
    if len(score_by_canonical) != len(score.items):
        raise ValueError("score manifest contains duplicate canonical IDs")
    for index, query in enumerate(window.query_records):
        score_item = score_by_canonical.get(str(query["canonical_id"]))
        if score_item is None:
            raise ValueError(f"query[{index}] is absent from score manifest")
        record = score_item.record
        for field in (
            "canonical_id",
            "image_id",
            "original_image_path",
            "original_image_sha256",
            "exclusion_group_id",
            "near_duplicate_cluster_id_or_unique_sentinel",
            "source_role_record_index",
        ):
            if record[field] != query[field]:
                raise ValueError(f"query/score identity mismatch at {field}")


def _verify_label_record_metadata(
    record: Mapping[str, Any],
    query: Mapping[str, Any],
    score_item: VerifiedStage2ScoreItem,
    window: VerifiedStage2Window,
    index: int,
    root: Path,
) -> None:
    if _exact_int(record["record_index"], "record_index", minimum=0) != index:
        raise ValueError("label record_index mismatch")
    expected = {
        "window_id": window.window_id,
        "canonical_id": query["canonical_id"],
        "image_id": query["image_id"],
        "source_domain": window.payload["domain"],
        "original_image_path": query["original_image_path"],
        "original_image_sha256": query["original_image_sha256"],
        "exclusion_group_id": query["exclusion_group_id"],
        "near_duplicate_cluster_id_or_unique_sentinel": query[
            "near_duplicate_cluster_id_or_unique_sentinel"
        ],
        "source_role_record_index": query["source_role_record_index"],
        "score_file": score_item.record["score_file"],
        "score_file_sha256": score_item.record["score_file_sha256"],
        "original_hw": score_item.record["original_hw"],
    }
    for field, value in expected.items():
        if record[field] != value:
            raise ValueError(f"label record {field} mismatch")
    _relative_repository_path(record["source_mask_path"], "source_mask_path")
    _sha256(record["source_mask_file_sha256"], "source_mask_file_sha256")
    source_hw = _parse_hw(record["source_mask_original_hw"], "source_mask_original_hw")
    source_mask_path = _repository_file(root, record["source_mask_path"], "source_mask_path")
    source_mask_sha = _sha256(record["source_mask_file_sha256"], "source_mask_file_sha256")
    if _hash_file_stable(source_mask_path) != source_mask_sha:
        raise ValueError("source mask SHA-256 mismatch")
    with Image.open(source_mask_path) as source_mask:
        observed_source_hw = (int(source_mask.height), int(source_mask.width))
        source_mask.verify()
    if observed_source_hw != source_hw:
        raise ValueError("source mask geometry provenance mismatch")
    if _hash_file_stable(source_mask_path) != source_mask_sha:
        raise RuntimeError("source mask changed while verified")
    original_hw = _parse_hw(record["original_hw"], "original_hw")
    alignment = record["alignment_provenance"]
    if not isinstance(alignment, Mapping):
        raise TypeError("alignment_provenance must be an object")
    _exact_keys(alignment, ALIGNMENT_FIELDS, "alignment_provenance")
    if alignment["policy"] != "BasicIRSTD-compatible-mask-to-image-nearest-v1":
        raise ValueError("alignment policy mismatch")
    if alignment["interpolation"] != "nearest_neighbor":
        raise ValueError("mask interpolation must be nearest_neighbor")
    if alignment["operation"] not in {"identity", "resize_mask_to_image_geometry"}:
        raise ValueError("unsupported alignment operation")
    if _parse_hw(alignment["source_mask_original_hw"], "alignment source hw") != source_hw:
        raise ValueError("alignment source geometry mismatch")
    if _parse_hw(alignment["target_image_hw"], "alignment target hw") != original_hw:
        raise ValueError("alignment target geometry mismatch")
    for field, expected_bool in {
        "mask_aligned_to_image_geometry": True,
        "silent_crop_used": False,
        "bilinear_resize_used": False,
    }.items():
        _exact_bool(alignment[field], field, expected_bool)
    _exact_bool(
        alignment["nuaa_misc_111_policy_applied"],
        "nuaa_misc_111_policy_applied",
        window.payload["domain"] == "NUAA-SIRST" and str(query["image_id"]) == "Misc_111",
    )
    module_path = _relative_repository_path(alignment["policy_module_path"], "policy_module_path")
    if module_path != "data_ext/mask_alignment.py":
        raise ValueError("alignment policy module path mismatch")
    policy_file = _repository_file(REPOSITORY_ROOT, module_path, "alignment policy module")
    if _hash_file_stable(policy_file) != _sha256(alignment["policy_module_sha256"], "policy_module_sha256"):
        raise ValueError("alignment policy module SHA-256 mismatch")
    aspect_error = _finite_float(alignment["aspect_ratio_relative_error"], "aspect ratio error", minimum=0.0)
    tolerance = _finite_float(alignment["aspect_tolerance"], "aspect tolerance", minimum=0.0)
    if tolerance != 0.01 or aspect_error > tolerance:
        raise ValueError("alignment aspect-ratio tolerance contract mismatch")
    if (source_hw == original_hw) != (alignment["operation"] == "identity"):
        raise ValueError("alignment operation does not match source/target geometry")


def _verify_label_npz(path: Path, record: Mapping[str, Any], original_hw: tuple[int, int]) -> None:
    with np.load(path, allow_pickle=False) as payload:
        if frozenset(payload.files) != LABEL_NPZ_FIELDS:
            raise ValueError("label NPZ field closure mismatch")
        mask = np.asarray(payload["mask"])
        if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != original_hw:
            raise ValueError("label NPZ mask dtype/geometry mismatch")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("label NPZ mask must be binary")
        scalar_expected = {
            "canonical_id": record["canonical_id"],
            "image_id": record["image_id"],
            "source_domain": record["source_domain"],
            "alignment_operation": record["alignment_provenance"]["operation"],
            "interpolation": "nearest_neighbor",
        }
        for field, expected in scalar_expected.items():
            value = str(np.asarray(payload[field]).reshape(()).item())
            if value != expected:
                raise ValueError(f"label NPZ {field} mismatch")
        if _parse_hw(payload["original_hw"], "NPZ original_hw") != original_hw:
            raise ValueError("label NPZ original_hw mismatch")
        if _parse_hw(payload["source_mask_original_hw"], "NPZ source mask hw") != _parse_hw(record["source_mask_original_hw"], "record source mask hw"):
            raise ValueError("label NPZ source-mask geometry mismatch")


def _window_records(
    value: object,
    *,
    expected_count: int,
    episode_role: str,
    payload: Mapping[str, Any],
    role: str,
    name: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} records")
    expected_fields = set(BASE_WINDOW_RECORD_FIELDS)
    if role == OOF_HOLDOUT_STAGE2_FIT:
        expected_fields.add("oof_fold_index")
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{name}[{index}] must be an object")
        _exact_keys(raw, frozenset(expected_fields), f"{name}[{index}]")
        if raw["episode_role"] != episode_role:
            raise ValueError(f"{name}[{index}] episode_role mismatch")
        if raw["outer_fold_id"] != payload["outer_fold_id"]:
            raise ValueError(f"{name}[{index}] outer_fold_id mismatch")
        if raw["source_role"] != payload["source_role"]:
            raise ValueError(f"{name}[{index}] source_role mismatch")
        if role == OOF_HOLDOUT_STAGE2_FIT and raw["oof_fold_index"] != payload["oof_fold_index"]:
            raise ValueError(f"{name}[{index}] oof_fold_index mismatch")
        canonical = _nonempty(raw["canonical_id"], "canonical_id")
        if canonical in seen:
            raise ValueError(f"duplicate canonical ID inside {name}")
        seen.add(canonical)
        _nonempty(raw["image_id"], "image_id")
        _relative_repository_path(raw["original_image_path"], "original_image_path")
        _sha256(raw["original_image_sha256"], "original_image_sha256")
        _nonempty(raw["exclusion_group_id"], "exclusion_group_id")
        _nonempty(raw["near_duplicate_cluster_id_or_unique_sentinel"], "near duplicate")
        _exact_int(raw["source_role_record_index"], "source_role_record_index", minimum=0)
        result.append(raw)
    return tuple(result)


def _verify_guardrails(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("window guardrails must be an object")
    _exact_keys(value, GUARDRAIL_FIELDS, "window guardrails")
    expected = {
        "development_only": True,
        "result_free": True,
        "execution_authorized": False,
        "official_test_split_files_opened": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "mask_or_label_files_opened": False,
        "predictions_scores_checkpoints_or_metrics_opened": False,
        "original_training_images_opened_only_for_sha256": True,
    }
    for field, expected_value in expected.items():
        _exact_bool(value[field], f"guardrails.{field}", expected_value)


def _verify_role_purity(value: object, role: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "allowed_source_role",
        "mixed_roles_allowed",
        "single_source_domain_per_window",
        "single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding",
    }:
        raise ValueError("window role_purity closure mismatch")
    expected_source = "detector_fit" if role == OOF_HOLDOUT_STAGE2_FIT else "detector_diagnostic"
    if value["allowed_source_role"] != expected_source:
        raise ValueError("role_purity allowed source mismatch")
    _exact_bool(value["mixed_roles_allowed"], "mixed_roles_allowed", False)
    _exact_bool(value["single_source_domain_per_window"], "single_source_domain_per_window", True)
    _exact_bool(
        value["single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding"],
        "single_oof_checkpoint_identity",
        role == OOF_HOLDOUT_STAGE2_FIT,
    )


def _verify_binding_file(value: object, root: Path, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} must be an exact path/SHA binding")
    relative = _relative_repository_path(value["path"], f"{name}.path")
    digest = _sha256(value["sha256"], f"{name}.sha256")
    if _contains_official_test(relative):
        raise ValueError(f"{name} may not reference official test")
    artifact = _repository_file(root, relative, name)
    if _hash_file_stable(artifact) != digest:
        raise ValueError(f"{name} SHA-256 mismatch")


def _verify_governance(root: Path) -> None:
    # Governance artifacts live with this implementation even when a test
    # injects an isolated synthetic data root.
    governance_root = REPOSITORY_ROOT
    payloads: dict[str, Mapping[str, Any]] = {}
    for name, binding in GOVERNANCE_BINDINGS.items():
        path = _repository_file(governance_root, binding["path"], name)
        if _hash_file_stable(path) != binding["sha256"]:
            raise RuntimeError(f"frozen governance artifact drift: {name}")
        payload, _ = _read_json_stable(path, name)
        if not isinstance(payload, Mapping):
            raise TypeError(f"governance artifact must be an object: {name}")
        payloads[name] = payload
    if payloads["b2_authorization"].get("artifact_status") != "RESULT_FREE_FROZEN_IMPLEMENTATION_AUTHORIZATION":
        raise RuntimeError("B2 authorization status is not frozen")
    authorization = payloads["b2_authorization"].get("authorization")
    if not isinstance(authorization, Mapping) or "W05" not in authorization.get("authorized_work_items", []):
        raise RuntimeError("B2 authorization does not include W05")
    for field in (
        "real_data_execution_authorized",
        "claim_bearing_training_authorized",
        "official_test_access_authorized",
    ):
        _exact_bool(authorization.get(field), f"B2 authorization.{field}", False)
    if payloads["b1_integration_pass"].get("status") != "PASS":
        raise RuntimeError("B1 integration prerequisite is not PASS")


def _resolve_bundle_artifact(
    root: Path,
    relative: str,
    *,
    bundle_root_override: str | Path | None,
    manifest_path: Path,
    name: str,
) -> Path:
    final_path = root.joinpath(*PurePosixPath(relative).parts)
    if bundle_root_override is None:
        artifact = _repository_file(root, relative, name)
        if artifact.parent != manifest_path.parent:
            raise ValueError("label_file is not a direct member of its atomic bundle")
        return artifact
    override = Path(bundle_root_override).expanduser().absolute()
    _assert_direct_directory(override, root, "bundle_root_override")
    # Every label member must be a direct member of the final bundle.  Map
    # only that filename into the private staging directory.
    final_parent = final_path.parent
    # The private staging directory and future final bundle must be siblings;
    # this is the same-parent prerequisite for an atomic directory rename.
    if final_parent.parent != override.parent or final_parent == override:
        raise ValueError("label_file is outside the future same-parent bundle")
    candidate = override / PurePosixPath(relative).name
    return _direct_existing_file(candidate, root, name, allow_override=override)


def _read_json_stable(path: Path, name: str) -> tuple[Any, str]:
    digest = _hash_file_stable(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if _hash_file_stable(path) != digest:
        raise RuntimeError(f"{name} changed while read")
    return value, digest


def _direct_input_file(
    value: str | Path,
    root: Path,
    name: str,
    *,
    allow_override: str | Path | None = None,
) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    return _direct_existing_file(candidate, root, name, allow_override=allow_override)


def _direct_existing_file(
    candidate: Path,
    root: Path,
    name: str,
    *,
    allow_override: str | Path | None = None,
) -> Path:
    candidate = candidate.absolute()
    allowed_roots = [root]
    if allow_override is not None:
        allowed_roots.append(Path(allow_override).expanduser().absolute())
    if not any(candidate == allowed or allowed in candidate.parents for allowed in allowed_roots):
        raise ValueError(f"{name} must remain inside an allowed repository tree")
    _assert_no_symlink_components(candidate, allowed_roots, name)
    info = candidate.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    return candidate


def _repository_file(root: Path, relative: str, name: str) -> Path:
    normal = _relative_repository_path(relative, f"{name}.path")
    return _direct_existing_file(root.joinpath(*PurePosixPath(normal).parts), root, name)


def _assert_direct_directory(path: Path, root: Path, name: str) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"{name} must remain under repository root")
    _assert_no_symlink_components(path, [root], name)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} must be a directory")


def _assert_no_symlink_components(path: Path, roots: Sequence[Path], name: str) -> None:
    containing = next((root for root in roots if path == root or root in path.parents), None)
    if containing is None:
        raise ValueError(f"{name} is outside allowed roots")
    current = containing
    if current.is_symlink():
        raise ValueError(f"{name} root may not be a symlink")
    relative_parts = path.relative_to(containing).parts
    for part in relative_parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{name} contains symlink component: {current}")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file_stable(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"artifact must be a regular file: {path}")
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    digest = _hash_file(path)
    after = path.stat(follow_symlinks=False)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise RuntimeError(f"artifact changed while hashed: {path}")
    return digest


def _repo_relative(path: Path, root: Path, *, allow_override: Path | None = None) -> str:
    absolute = path.absolute()
    if allow_override is not None and (absolute == allow_override or allow_override in absolute.parents):
        # Producer staging manifests record future final paths; callers should
        # never use this branch for arbitrary consumer paths.
        return absolute.relative_to(root).as_posix() if root in absolute.parents else absolute.name
    if absolute != root and root not in absolute.parents:
        raise ValueError("path is outside repository root")
    return absolute.relative_to(root).as_posix()


def _relative_repository_path(value: object, field: str) -> str:
    rendered = _nonempty(value, field)
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or rendered != pure.as_posix() or not pure.parts:
        raise ValueError(f"{field} must be a canonical POSIX repository-relative path")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{field} contains traversal or non-canonical components")
    if _contains_official_test(rendered):
        raise ValueError(f"{field} may not reference official test")
    return rendered


def _contains_official_test(value: str) -> bool:
    lowered = value.lower().replace("-", "_")
    return "official_test" in lowered or "officialtest" in lowered


def _repository_root(value: str | Path | None) -> Path:
    root = (REPOSITORY_ROOT if value is None else Path(value).expanduser()).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository_root must be a real non-symlink directory")
    return root


def _role(value: object) -> str:
    role = _nonempty(value, "expected_role")
    if role not in QUERY_BEARING_ROLES:
        raise ValueError(f"role must be one of {QUERY_BEARING_ROLES}")
    return role


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} field closure mismatch: missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _exact_bool(value: object, field: str, expected: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field} must be an exact JSON boolean")
    if value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")


def _exact_int(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _finite_float(value: object, field: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and >= {minimum}")
    return result


def _parse_hw(value: object, field: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.size != 2:
        raise ValueError(f"{field} must contain [height, width]")
    raw = array.reshape(-1).tolist()
    if any(type(item) not in (int, np.int32, np.int64) for item in raw):
        # JSON values arrive as int; NPZ values as NumPy integers.
        if any(isinstance(item, bool) or int(item) != item for item in raw):
            raise TypeError(f"{field} must contain exact integers")
    result = (int(raw[0]), int(raw[1]))
    if min(result) <= 0:
        raise ValueError(f"{field} dimensions must be positive")
    return result


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value
