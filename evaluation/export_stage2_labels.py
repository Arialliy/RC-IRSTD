"""Export one atomic Stage2 Q28 label-v2 and exact-curve-v3 bundle.

The producer consumes one explicitly SHA-bound C14/Q28 ``window_id`` and one
score-manifest v4.  It validates the complete window and ordered query
identity before resolving the first mask path.  Only those 28 query masks are
resolved or opened; context labels and context mask paths do not exist in the
producer control flow.
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
import platform
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from data_ext.dataset_meta import safe_output_stem
from data_ext.mask_alignment import (
    DEFAULT_ASPECT_TOLERANCE,
    align_mask_to_image,
    aspect_ratio_relative_error,
)
from data_ext.split_utils import IMAGE_EXTENSIONS, sample_id_from_entry
from data_ext.stage2_label_attachment import (
    GOVERNANCE_BINDINGS,
    STAGE2_LABEL_ARTIFACT_TYPE,
    STAGE2_LABEL_CONTENT_ALGORITHM,
    STAGE2_LABEL_SCHEMA,
    STAGE2_QUERY_IDENTITY_ALGORITHM,
    _bind_window_to_score,
    governance_bindings_payload,
    stage2_label_records_sha256,
    stage2_ordered_query_identity,
    verify_stage2_label_attachment,
    verify_stage2_window_contract,
)
from data_ext.stage2_score_manifest import (
    BINDING_NAMES,
    STRICT_THRESHOLD_SEMANTICS,
    verify_stage2_score_manifest,
)
from data_ext.stage2_threshold_decision import (
    PIXEL_BUDGET_GRID,
    VerifiedStage2ThresholdDecisionSet,
    verify_stage2_threshold_decision_set,
)
from evaluation.stage2_threshold_sweep import (
    build_stage2_query_curve,
    verify_stage2_query_curve_artifacts,
    write_stage2_query_curve_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA = "rc-irstd.stage2-label-curve-bundle-audit.v1"
ALIGNMENT_POLICY = "BasicIRSTD-compatible-mask-to-image-nearest-v1"
ALIGNMENT_MODULE_PATH = "data_ext/mask_alignment.py"


@dataclass(frozen=True)
class _BoundMask:
    record_index: int
    query_record: Mapping[str, Any]
    score_record: Mapping[str, Any]
    path: Path
    sha256: str
    identity: tuple[int, int, int, int]


@dataclass(frozen=True)
class _Publication:
    final_root: Path
    staging_root: Path
    staging_identity: tuple[int, int]
    lock_path: Path
    lock_identity: tuple[int, int]


def export_stage2_labels(
    *,
    score_manifest: str | Path,
    score_manifest_sha256: str,
    window_manifest: str | Path,
    window_manifest_sha256: str,
    window_id: str,
    expected_role: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    mask_folder: str = "masks",
    sealed_decision_set: str | Path | None = None,
    sealed_decision_set_sha256: str | None = None,
    statistics_config: str | Path | None = None,
    statistics_config_sha256: str | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Publish labels, exact curve, audit and sidecars as one directory."""

    root = _repository_root(repository_root)

    # Security ordering: resolve the exact window and ordered query identities
    # first, then score-v4 identity/native geometry; only afterwards may a
    # query mask path be constructed.
    window = verify_stage2_window_contract(
        window_manifest,
        window_manifest_sha256,
        window_id,
        expected_role,
        repository_root=root,
    )
    score = verify_stage2_score_manifest(
        score_manifest,
        score_manifest_sha256,
        expected_role,
        repository_root=root,
    )
    _bind_window_to_score(window, score)
    governance = governance_bindings_payload(root)
    if tuple(score.bindings) != BINDING_NAMES:
        raise RuntimeError("score-v4 ten-binding closure changed")
    decision_seal_binding = _enforce_prelabel_decision_gate(
        window.role,
        sealed_decision_set,
        sealed_decision_set_sha256,
        statistics_config,
        statistics_config_sha256,
        score=score,
        window=window,
        root=root,
    )

    dataset_root = _input_directory(dataset_dir, root, "dataset_dir")
    if dataset_root.name != score.payload["source_domain"]:
        raise ValueError("dataset_dir basename differs from score source_domain")
    if not isinstance(mask_folder, str) or not mask_folder or PurePosixPath(mask_folder).name != mask_folder:
        raise ValueError("mask_folder must be one direct folder name")
    if mask_folder.lower().replace("-", "_") in {"official_test", "officialtest"}:
        raise ValueError("mask_folder may not reference official test")
    bound_masks = _preflight_query_masks(
        window.query_records,
        score,
        dataset_root,
        mask_folder,
        root,
    )
    if len(bound_masks) != 28:
        raise RuntimeError("producer preflight did not bind exactly Q28 masks")

    publication = _prepare_publication(output_dir, root)
    marker = publication.staging_root / ".bundle_incomplete"
    final_identity: tuple[int, int] | None = None
    rename_started = False
    try:
        _write_text_exclusive(marker, "Stage2 label/curve bundle is incomplete.\n")
        records = _materialize_labels(
            bound_masks,
            staging_root=publication.staging_root,
            final_root=publication.final_root,
            root=root,
            source_domain=str(score.payload["source_domain"]),
            window_id=window.window_id,
        )
        _verify_bound_masks(bound_masks)
        query_identity_sha = _canonical_json_sha256(
            stage2_ordered_query_identity(window.query_records)
        )
        label_manifest = {
            "schema_version": STAGE2_LABEL_SCHEMA,
            "artifact_type": STAGE2_LABEL_ARTIFACT_TYPE,
            "artifact_status": "DEVELOPMENT_ONLY",
            "development_only": True,
            "execution_scope": "stage2_development_query_labels",
            "official_test_accessed": False,
            "labels_embedded_in_scores": False,
            "native_resolution": True,
            "path_anchor": "repository_root",
            "role": expected_role,
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
            "decision_seal_binding": decision_seal_binding,
            "window_binding": {
                "path": _repo_relative(window.path, root),
                "sha256": window.manifest_sha256,
                "window_id": window.window_id,
                "window_identity_sha256": window.window_identity_sha256,
            },
            "score_manifest_binding": {
                "path": _repo_relative(score.path, root),
                "sha256": score.manifest_sha256,
                "records_content_sha256": score.records_content_sha256,
            },
            "score_bindings": dict(score.bindings),
            "governance_bindings": governance,
            "ordered_query_identity_sha256_algorithm": STAGE2_QUERY_IDENTITY_ALGORITHM,
            "ordered_query_identity_sha256": query_identity_sha,
            "labels_content_sha256_algorithm": STAGE2_LABEL_CONTENT_ALGORITHM,
            "labels_content_sha256": stage2_label_records_sha256(records),
            "labels": records,
        }
        label_manifest_path = publication.staging_root / "label-manifest.json"
        _write_json_exclusive(label_manifest_path, label_manifest)
        label_manifest_sha = _hash_file_stable(label_manifest_path)
        attachment = verify_stage2_label_attachment(
            score.path,
            label_manifest_path,
            expected_role,
            score_manifest_sha256=score.manifest_sha256,
            label_manifest_sha256=label_manifest_sha,
            window_manifest=window.path,
            window_manifest_sha256=window.manifest_sha256,
            window_id=window.window_id,
            repository_root=root,
            bundle_root_override=publication.staging_root,
        )
        curve = build_stage2_query_curve(
            window.path,
            score.path,
            label_manifest_path,
            window_manifest_sha256=window.manifest_sha256,
            score_manifest_sha256=score.manifest_sha256,
            label_manifest_sha256=label_manifest_sha,
            window_id=window.window_id,
            expected_role=expected_role,
            repository_root=root,
            bundle_root_override=publication.staging_root,
        )
        curve_manifest, curve_hashes = write_stage2_query_curve_artifacts(
            curve,
            staging_root=publication.staging_root,
            final_root=publication.final_root,
            repository_root=root,
        )
        base_hashes = {
            path.name: _hash_file_stable(path)
            for path in sorted(publication.staging_root.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != marker.name
        }
        audit = _build_audit(
            expected_role=expected_role,
            window=window,
            score=score,
            attachment=attachment,
            curve=curve,
            governance=governance,
            base_hashes=base_hashes,
            bound_masks=bound_masks,
        )
        audit_path = publication.staging_root / "audit.json"
        _write_json_exclusive(audit_path, audit)
        base_hashes[audit_path.name] = _hash_file_stable(audit_path)
        _write_sidecars(publication.staging_root, base_hashes)
        marker.unlink()
        _fsync_directory(publication.staging_root)
        _preflight_bundle(
            publication,
            base_hashes,
            score=score,
            window=window,
            expected_role=expected_role,
            label_manifest_sha=label_manifest_sha,
            curve_hashes=curve_hashes,
        )
        _verify_bound_masks(bound_masks)
        rename_started = True
        final_identity = _rename_directory_no_replace(
            publication.staging_root,
            publication.final_root,
            publication.staging_identity,
        )
        _assert_owned_directory(
            publication.final_root,
            final_identity,
            "published bundle",
        )
        _fsync_directory(publication.final_root.parent)
        published_label = publication.final_root / "label-manifest.json"
        published_curve = publication.final_root / "query-curve.csv"
        published_curve_manifest = publication.final_root / "curve-manifest.json"
        published_attachment = verify_stage2_label_attachment(
            score.path,
            published_label,
            expected_role,
            score_manifest_sha256=score.manifest_sha256,
            label_manifest_sha256=label_manifest_sha,
            window_manifest=window.path,
            window_manifest_sha256=window.manifest_sha256,
            window_id=window.window_id,
            repository_root=root,
            _owned_publication_lock=(
                publication.lock_path,
                publication.lock_identity,
            ),
        )
        verify_stage2_query_curve_artifacts(
            published_curve,
            published_curve_manifest,
            curve_sha256=curve_hashes["curve_sha256"],
            curve_manifest_sha256=curve_hashes["curve_manifest_sha256"],
            attachment=published_attachment,
            repository_root=root,
            _owned_publication_lock=(
                publication.lock_path,
                publication.lock_identity,
            ),
        )
        _verify_sidecars(publication.final_root, base_hashes)
        _verify_bound_masks(bound_masks)
        result = {
            "bundle_path": _repo_relative(publication.final_root, root),
            "window_id": window.window_id,
            "role": expected_role,
            "query_labels": 28,
            "curve_operating_points": len(curve.rows),
            "label_manifest_sha256": label_manifest_sha,
            "curve_sha256": curve_hashes["curve_sha256"],
            "curve_manifest_sha256": curve_hashes["curve_manifest_sha256"],
            "audit_sha256": base_hashes["audit.json"],
            "official_test_accessed": False,
        }
    except BaseException as primary:
        cleanup_errors = _cleanup_publication(
            publication,
            include_final=rename_started,
        )
        if cleanup_errors:
            raise BaseExceptionGroup(
                "Stage2 bundle transaction failed and cleanup also reported errors",
                [primary, *cleanup_errors],
            ) from primary
        raise

    # A successful API return is permitted only after staging and the owned
    # lock are gone.  If finalization itself cannot complete, roll the already
    # published directory back before surfacing the failure.
    finalize_errors = _cleanup_publication(publication, include_final=False)
    if finalize_errors:
        rollback_errors = _cleanup_publication(publication, include_final=True)
        primary = RuntimeError("Stage2 bundle finalization failed; publication rolled back")
        raise BaseExceptionGroup(
            "Stage2 bundle finalization/rollback failure",
            [primary, *finalize_errors, *rollback_errors],
        ) from primary
    return result


def _preflight_query_masks(
    query_records: Sequence[Mapping[str, Any]],
    score: Any,
    dataset_root: Path,
    mask_folder: str,
    root: Path,
) -> tuple[_BoundMask, ...]:
    if len(query_records) != 28:
        raise ValueError("query preflight requires exactly 28 identities")
    score_by_canonical = {item.canonical_id: item for item in score.items}
    result: list[_BoundMask] = []
    seen: set[Path] = set()
    for index, query in enumerate(query_records):
        score_item = score_by_canonical[str(query["canonical_id"])]
        image_relative = str(query["original_image_path"])
        expected_prefix = _repo_relative(dataset_root, root) + "/images/"
        if not image_relative.startswith(expected_prefix):
            raise ValueError("query image path is outside supplied dataset_dir/images")
        # This is the first mask-path operation in the call graph.  The exact
        # window, ordered Q28 identities and score-v4 were all verified above.
        mask_path = _resolve_query_mask_direct(
            dataset_root,
            mask_folder,
            str(query["image_id"]),
            root,
        )
        mask_path = _input_file(mask_path, root, f"query mask[{index}]")
        if dataset_root not in mask_path.parents:
            raise ValueError("query mask escaped dataset_dir")
        accepted_stems = {
            str(query["image_id"]),
            f"{query['image_id']}_pixels0",
        }
        if mask_path.stem not in accepted_stems:
            raise ValueError("query mask resolver returned a different identity")
        if mask_path in seen:
            raise ValueError("two query identities resolved to one mask")
        seen.add(mask_path)
        info = mask_path.stat(follow_symlinks=False)
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        digest = _hash_file_stable(mask_path)
        result.append(
            _BoundMask(
                record_index=index,
                query_record=query,
                score_record=score_item.record,
                path=mask_path,
                sha256=digest,
                identity=identity,
            )
        )
    return tuple(result)




def _resolve_query_mask_direct(
    dataset_root: Path,
    mask_folder: str,
    image_id: str,
    repository_root: Path,
) -> Path:
    """Resolve query-specific direct names without scanning other masks."""

    sample_id = sample_id_from_entry(image_id)
    relative = PurePosixPath(sample_id)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("query image_id is not a canonical relative identity")
    folder = dataset_root / mask_folder
    _assert_real_directory(folder, repository_root, "query mask folder")
    stems = [relative.name]
    if relative.name.endswith("_pixels0"):
        stems.append(relative.name[: -len("_pixels0")])
    else:
        stems.append(f"{relative.name}_pixels0")
    candidates: list[Path] = []
    for stem in stems:
        relative_stem = Path(*relative.parts[:-1], stem)
        for extension in IMAGE_EXTENSIONS:
            candidates.append(folder / relative_stem.with_suffix(extension))
            candidates.append(folder / relative_stem.with_suffix(extension.upper()))
    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        # lexists/lstat touches names derived only from this verified query.
        if not os.path.lexists(candidate):
            continue
        existing.append(candidate)
    if not existing:
        raise FileNotFoundError(
            f"No direct query mask candidate for {image_id!r} under {folder}"
        )
    if len(existing) != 1:
        rendered = [path.relative_to(folder).as_posix() for path in existing]
        raise ValueError(
            f"Ambiguous direct query mask candidates for {image_id!r}: {rendered}"
        )
    return _input_file(existing[0], repository_root, "query mask")

def _materialize_labels(
    masks: Sequence[_BoundMask],
    *,
    staging_root: Path,
    final_root: Path,
    root: Path,
    source_domain: str,
    window_id: str,
) -> list[dict[str, Any]]:
    policy_path = REPOSITORY_ROOT / ALIGNMENT_MODULE_PATH
    policy_sha = _hash_file_stable(policy_path)
    names: set[str] = set()
    records: list[dict[str, Any]] = []
    for bound in masks:
        _verify_bound_mask(bound)
        target_hw = tuple(int(value) for value in bound.score_record["original_hw"])
        with Image.open(bound.path) as handle:
            source = handle.convert("L")
        source_hw = (int(source.height), int(source.width))
        aspect_error = aspect_ratio_relative_error(
            (target_hw[1], target_hw[0]),
            source.size,
        )
        aligned = align_mask_to_image(
            source,
            (target_hw[1], target_hw[0]),
            str(bound.query_record["image_id"]),
            aspect_tolerance=DEFAULT_ASPECT_TOLERANCE,
        )
        mask = (np.asarray(aligned, dtype=np.uint8) > 0).astype(np.uint8)
        if mask.shape != target_hw:
            raise RuntimeError("nearest mask alignment did not reach native image geometry")
        _verify_bound_mask(bound)
        operation = "identity" if source_hw == target_hw else "resize_mask_to_image_geometry"
        output_name = f"{bound.record_index:02d}-{safe_output_stem(str(bound.query_record['image_id']))}.label.npz"
        if output_name in names:
            raise RuntimeError("duplicate label output filename")
        names.add(output_name)
        output_path = staging_root / output_name
        _write_npz_exclusive(
            output_path,
            mask=mask,
            canonical_id=np.asarray(bound.query_record["canonical_id"]),
            image_id=np.asarray(bound.query_record["image_id"]),
            source_domain=np.asarray(source_domain),
            original_hw=np.asarray(target_hw, dtype=np.int64),
            source_mask_original_hw=np.asarray(source_hw, dtype=np.int64),
            alignment_operation=np.asarray(operation),
            interpolation=np.asarray("nearest_neighbor"),
        )
        label_sha = _hash_file_stable(output_path)
        is_misc = source_domain == "NUAA-SIRST" and str(bound.query_record["image_id"]) == "Misc_111"
        alignment = {
            "policy": ALIGNMENT_POLICY,
            "policy_module_path": ALIGNMENT_MODULE_PATH,
            "policy_module_sha256": policy_sha,
            "interpolation": "nearest_neighbor",
            "operation": operation,
            "source_mask_original_hw": list(source_hw),
            "target_image_hw": list(target_hw),
            "aspect_ratio_relative_error": float(aspect_error),
            "aspect_tolerance": float(DEFAULT_ASPECT_TOLERANCE),
            "mask_aligned_to_image_geometry": True,
            "silent_crop_used": False,
            "bilinear_resize_used": False,
            "nuaa_misc_111_policy_applied": bool(is_misc),
        }
        records.append(
            {
                "record_index": bound.record_index,
                "window_id": window_id,
                "canonical_id": bound.query_record["canonical_id"],
                "image_id": bound.query_record["image_id"],
                "source_domain": source_domain,
                "original_image_path": bound.query_record["original_image_path"],
                "original_image_sha256": bound.query_record["original_image_sha256"],
                "exclusion_group_id": bound.query_record["exclusion_group_id"],
                "near_duplicate_cluster_id_or_unique_sentinel": bound.query_record[
                    "near_duplicate_cluster_id_or_unique_sentinel"
                ],
                "source_role_record_index": bound.query_record["source_role_record_index"],
                "score_file": bound.score_record["score_file"],
                "score_file_sha256": bound.score_record["score_file_sha256"],
                "original_hw": list(target_hw),
                "label_file": _repo_relative(final_root / output_name, root),
                "label_file_sha256": label_sha,
                "source_mask_path": _repo_relative(bound.path, root),
                "source_mask_file_sha256": bound.sha256,
                "source_mask_original_hw": list(source_hw),
                "alignment_provenance": alignment,
            }
        )
    return records


def _build_audit(
    *,
    expected_role: str,
    window: Any,
    score: Any,
    attachment: Any,
    curve: Any,
    governance: Mapping[str, Any],
    base_hashes: Mapping[str, str],
    bound_masks: Sequence[_BoundMask],
) -> dict[str, Any]:
    resized = sum(
        list(item.record["source_mask_original_hw"]) != list(item.record["original_hw"])
        for item in attachment.items
    )
    misc = [
        item.image_id
        for item in attachment.items
        if item.record["alignment_provenance"]["nuaa_misc_111_policy_applied"]
    ]
    return {
        "schema_version": AUDIT_SCHEMA,
        "artifact_type": "rc_irstd_stage2_label_curve_bundle_audit",
        "artifact_status": "SYNTHETIC_OR_FUTURE_AUTHORIZED_DEVELOPMENT_ONLY",
        "development_only": True,
        "official_test_accessed": False,
        "training_performed": False,
        "gpu_used": False,
        "role": expected_role,
        "window_id": window.window_id,
        "window_manifest_sha256": window.manifest_sha256,
        "window_identity_sha256": window.window_identity_sha256,
        "score_manifest_sha256": score.manifest_sha256,
        "score_records_content_sha256": score.records_content_sha256,
        "score_binding_names": list(BINDING_NAMES),
        "score_bindings": dict(score.bindings),
        "decision_seal_binding": attachment.payload["decision_seal_binding"],
        "governance_bindings": dict(governance),
        "label_manifest_sha256": attachment.manifest_sha256,
        "labels_content_sha256": attachment.labels_content_sha256,
        "ordered_query_identity_sha256": attachment.ordered_query_identity_sha256,
        "query_mask_paths_resolved": len(bound_masks),
        "query_labels_loaded": len(bound_masks),
        "context_mask_paths_resolved_statted_or_opened": 0,
        "context_labels_loaded": 0,
        "native_mask_alignment_count": resized,
        "nuaa_misc_111_policy_records": misc,
        "alignment_policy": ALIGNMENT_POLICY,
        "alignment_interpolation": "nearest_neighbor",
        "silent_crop_used": False,
        "bilinear_mask_resize_used": False,
        "curve_schema": "rc-irstd.stage2-query-curve.v3",
        "curve_operating_points": len(curve.rows),
        "unique_float64_query_events": curve.unique_event_count,
        "event_threshold_cap": None,
        "exact_sweep_implementation": (
            "descending_event_group_incremental_8_connected_dsu_v1"
        ),
        "exact_sweep_legacy_match_components_calls": 0,
        "curve_rows_storage": (
            "ten_read_only_contiguous_float64_or_int64_numpy_columns"
        ),
        "curve_rows_storage_bytes": curve.rows.storage_nbytes,
        "curve_rows_sha256": curve.rows_sha256,
        "exact_sweep_runtime_environment": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "prediction_semantics": STRICT_THRESHOLD_SEMANTICS,
        "fa_denominator": "all_native_resolution_query_pixels",
        "object_matching": "8_connected_maximum_cardinality_one_to_one_overlap",
        "member_sha256_before_audit_and_sidecars": dict(sorted(base_hashes.items())),
    }


def _enforce_prelabel_decision_gate(
    role: str,
    sealed_decision_set: str | Path | None,
    sealed_decision_set_sha256: str | None,
    statistics_config: str | Path | None,
    statistics_config_sha256: str | None,
    *,
    score: Any,
    window: Any,
    root: Path,
) -> dict[str, str] | None:
    supplied = any(
        value is not None
        for value in (
            sealed_decision_set,
            sealed_decision_set_sha256,
            statistics_config,
            statistics_config_sha256,
        )
    )
    if role != "outer_target_diagnostic_development":
        if supplied:
            raise ValueError(
                "source/OOF label roles require sealed_decision_set, its SHA, "
                "statistics_config and its SHA to be null"
            )
        return None
    if any(
        value is None
        for value in (
            sealed_decision_set,
            sealed_decision_set_sha256,
            statistics_config,
            statistics_config_sha256,
        )
    ):
        raise RuntimeError(
            "outer-target query labels remain hard-HOLD until a complete externally "
            "SHA-bound T0-T8 decision set and externally SHA-bound statistics config "
            "are supplied; no mask path has been resolved, statted or opened"
        )

    verified = verify_stage2_threshold_decision_set(
        sealed_decision_set,
        sealed_decision_set_sha256,
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
    verified_config_path, verified_config_sha = _bind_verified_decision_set_to_inputs(
        verified,
        score=score,
        window=window,
        root=root,
        statistics_config=statistics_config,
        statistics_config_sha256=statistics_config_sha256,
    )
    shared = verified.payload["shared_bindings"]
    return {
        "path": _repo_relative(verified.path, root),
        "sha256": verified.manifest_sha256,
        "decision_set_content_sha256": str(
            verified.payload["decision_set_content_sha256"]
        ),
        "context_package_sha256": str(shared["context_package"]["sha256"]),
        "context_package_commit_sha256": str(
            shared["context_package_commit"]["sha256"]
        ),
        "statistics_config_path": _repo_relative(verified_config_path, root),
        "statistics_config_sha256": verified_config_sha,
    }


def _bind_verified_decision_set_to_inputs(
    verified: VerifiedStage2ThresholdDecisionSet,
    *,
    score: Any,
    window: Any,
    root: Path,
    statistics_config: str | Path,
    statistics_config_sha256: str,
) -> tuple[Path, str]:
    """Close decision-set identities that the public verifier cannot infer."""

    shared = verified.payload["shared_bindings"]
    # Delayed import is deliberate: source/OOF paths do not acquire the Lane-A
    # dependency, while an outer gate must verify the actual context bytes and
    # commit rather than trust self-consistent SHA strings in nine JSON files.
    from rc.stage2_crossfit_schema import (
        verify_stage2_context_package,
        verify_stage2_statistics_config,
    )

    raw_config_path = Path(statistics_config).expanduser()
    config_path = _input_file(
        raw_config_path if raw_config_path.is_absolute() else root / raw_config_path,
        root,
        "statistics_config",
    )
    config_sha = str(statistics_config_sha256)
    verified_statistics_config = verify_stage2_statistics_config(
        config_path,
        config_sha,
        repository_root=root,
    )

    declared_context_path = str(shared["context_package"]["path"])
    pure_context_path = PurePosixPath(declared_context_path)
    if (
        pure_context_path.is_absolute()
        or pure_context_path.as_posix() != declared_context_path
        or any(part in {"", ".", ".."} for part in pure_context_path.parts)
    ):
        raise ValueError("decision-set context package path is not canonical repository-relative")
    try:
        context = verify_stage2_context_package(
            root.joinpath(*pure_context_path.parts),
            shared["context_package"]["sha256"],
            shared["context_package_commit"]["sha256"],
            statistics_config=verified_statistics_config,
            repository_root=root,
        )
    except FileNotFoundError as error:
        raise ValueError("decision-set context package does not exist") from error
    expected_query_identity = _canonical_json_sha256(
        stage2_ordered_query_identity(window.query_records)
    )
    context_payload = context.payload
    detector_identity = context_payload["detector_identity"]
    exact = (
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
            _repo_relative(config_path, root),
            "context.statistics_config_path",
        ),
        (
            context_payload["statistics_config_binding"]["sha256"],
            config_sha,
            "context.statistics_config_sha256",
        ),
    )
    for observed, expected, name in exact:
        if observed != expected:
            raise ValueError(f"decision-set {name} binding mismatch")
    if dict(context_payload["score_bindings"]) != dict(score.bindings):
        raise ValueError("decision-set context.score_bindings binding mismatch")
    if context.window.window_identity_sha256 != window.window_identity_sha256:
        raise ValueError("decision-set verified context window capability mismatch")
    if context.score_manifest.manifest_sha256 != score.manifest_sha256:
        raise ValueError("decision-set verified context score capability mismatch")
    return config_path, config_sha


def _preflight_bundle(
    publication: _Publication,
    base_hashes: Mapping[str, str],
    *,
    score: Any,
    window: Any,
    expected_role: str,
    label_manifest_sha: str,
    curve_hashes: Mapping[str, str],
) -> None:
    expected = set(base_hashes) | {f"{name}.sha256" for name in base_hashes}
    actual = {path.name for path in publication.staging_root.iterdir()}
    if actual != expected:
        raise RuntimeError(f"staged bundle inventory mismatch: {sorted(actual ^ expected)}")
    _verify_sidecars(publication.staging_root, base_hashes)
    attachment = verify_stage2_label_attachment(
        score.path,
        publication.staging_root / "label-manifest.json",
        expected_role,
        score_manifest_sha256=score.manifest_sha256,
        label_manifest_sha256=label_manifest_sha,
        window_manifest=window.path,
        window_manifest_sha256=window.manifest_sha256,
        window_id=window.window_id,
        repository_root=score.repository_root,
        bundle_root_override=publication.staging_root,
    )
    verify_stage2_query_curve_artifacts(
        publication.staging_root / "query-curve.csv",
        publication.staging_root / "curve-manifest.json",
        curve_sha256=curve_hashes["curve_sha256"],
        curve_manifest_sha256=curve_hashes["curve_manifest_sha256"],
        attachment=attachment,
        repository_root=score.repository_root,
        bundle_root_override=publication.staging_root,
    )


def _prepare_publication(value: str | Path, root: Path) -> _Publication:
    final = Path(value).expanduser().absolute()
    if root not in final.parents:
        raise ValueError("output_dir must be a repository subdirectory")
    if os.path.lexists(final):
        raise FileExistsError("output_dir already exists; overwrite is forbidden")
    parent = final.parent
    _assert_real_directory(parent, root, "output parent")
    lock = parent / f".{final.name}.lock"
    descriptor = -1
    lock_identity: tuple[int, int] | None = None
    staging: Path | None = None
    staging_identity: tuple[int, int] | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock, flags, 0o600)
        lock_info = os.fstat(descriptor)
        if not stat.S_ISREG(lock_info.st_mode):
            raise RuntimeError("new publication lock is not a regular file")
        lock_identity = (lock_info.st_dev, lock_info.st_ino)
        _write_fd_all(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _assert_owned_regular_file(lock, lock_identity, "publication lock")
        staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=parent))
        staging_stat = staging.stat(follow_symlinks=False)
        if not stat.S_ISDIR(staging_stat.st_mode):
            raise RuntimeError("new staging path is not a directory")
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        _fsync_directory(parent)
        return _Publication(
            final_root=final,
            staging_root=staging,
            staging_identity=staging_identity,
            lock_path=lock,
            lock_identity=lock_identity,
        )
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as error:
                cleanup_errors.append(error)
        if staging is not None and staging_identity is not None:
            cleanup_errors.extend(
                _retry_cleanup(
                    lambda: _remove_owned_directory(staging, staging_identity),
                    "remove failed staging directory",
                )
            )
        if lock_identity is not None:
            cleanup_errors.extend(
                _retry_cleanup(
                    lambda: _release_lock(lock, lock_identity),
                    "release failed publication lock",
                )
            )
        cleanup_errors.extend(
            _retry_cleanup(
                lambda: _fsync_directory(parent),
                "fsync output parent after failed setup",
            )
        )
        if cleanup_errors:
            raise BaseExceptionGroup(
                "Stage2 publication setup and cleanup failed",
                [primary, *cleanup_errors],
            ) from primary
        raise


def _rename_directory_no_replace(
    source: Path,
    target: Path,
    expected_identity: tuple[int, int],
) -> tuple[int, int]:
    if os.path.lexists(target):
        raise FileExistsError("target bundle appeared before publication")
    _assert_owned_directory(source, expected_identity, "staging directory")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("target bundle already exists")
        raise OSError(error, os.strerror(error), str(target))
    # rename preserves the inode.  Returning the already authenticated source
    # identity avoids an unsafe post-rename stat gap; the caller subsequently
    # authenticates the target while already holding this rollback identity.
    return expected_identity


def _write_npz_exclusive(path: Path, **arrays: np.ndarray) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(
        path,
        (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )


def _write_text_exclusive(path: Path, value: str) -> None:
    _write_bytes_exclusive(path, value.encode("utf-8"))


def _write_bytes_exclusive(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_sidecars(root: Path, hashes: Mapping[str, str]) -> None:
    for name, digest in sorted(hashes.items()):
        _write_text_exclusive(root / f"{name}.sha256", f"{digest}  {name}\n")


def _verify_sidecars(root: Path, hashes: Mapping[str, str]) -> None:
    for name, expected in hashes.items():
        artifact = root / name
        sidecar = root / f"{name}.sha256"
        if _hash_file_stable(artifact) != expected:
            raise RuntimeError(f"bundle member hash mismatch: {name}")
        text = _read_text_stable_regular(sidecar)
        if text != f"{expected}  {name}\n":
            raise RuntimeError(f"bundle sidecar mismatch: {name}")


def _read_text_stable_regular(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"bundle text member must be a regular file: {path.name}")
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    value = path.read_text(encoding="utf-8")
    after = path.stat(follow_symlinks=False)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise RuntimeError(f"bundle text member changed while read: {path.name}")
    return value


def _verify_bound_mask(bound: _BoundMask) -> None:
    info = bound.path.stat(follow_symlinks=False)
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    if identity != bound.identity or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("query mask identity changed")
    if _hash_file_stable(bound.path) != bound.sha256:
        raise RuntimeError("query mask bytes changed")


def _verify_bound_masks(values: Sequence[_BoundMask]) -> None:
    for value in values:
        _verify_bound_mask(value)


def _write_fd_all(descriptor: int, value: bytes) -> None:
    offset = 0
    while offset < len(value):
        written = os.write(descriptor, value[offset:])
        if written <= 0:
            raise OSError("short write while creating publication lock")
        offset += written


def _assert_owned_directory(
    path: Path,
    identity: tuple[int, int],
    name: str,
) -> None:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError(f"{name} inode/type changed")


def _assert_owned_regular_file(
    path: Path,
    identity: tuple[int, int],
    name: str,
) -> None:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError(f"{name} inode/type changed")


def _retry_cleanup(
    action: Any,
    description: str,
    *,
    attempts: int = 3,
) -> list[BaseException]:
    last: BaseException | None = None
    for _ in range(attempts):
        try:
            action()
            return []
        except BaseException as error:
            last = error
    assert last is not None
    return [RuntimeError(f"{description}: {last}")]


def _cleanup_publication(
    publication: _Publication,
    *,
    include_final: bool,
) -> list[BaseException]:
    """Best-effort exhaustive cleanup without deleting an unowned inode."""

    errors: list[BaseException] = []
    if include_final:
        errors.extend(
            _retry_cleanup(
                lambda: _remove_owned_directory(
                    publication.final_root,
                    publication.staging_identity,
                ),
                "remove owned final bundle",
            )
        )
    errors.extend(
        _retry_cleanup(
            lambda: _remove_owned_directory(
                publication.staging_root,
                publication.staging_identity,
            ),
            "remove owned staging bundle",
        )
    )
    errors.extend(
        _retry_cleanup(
            lambda: _release_lock(
                publication.lock_path,
                publication.lock_identity,
            ),
            "release owned publication lock",
        )
    )
    # Persist either success finalization (staging/lock removal) or rollback
    # (final/staging/lock removal) in the same parent directory.
    errors.extend(
        _retry_cleanup(
            lambda: _fsync_directory(publication.final_root.parent),
            "fsync publication parent after cleanup",
        )
    )
    return errors


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> bool:
    if not os.path.lexists(path):
        return False
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError(f"refusing to remove unowned directory: {path}")
    shutil.rmtree(path)
    return True


def _release_lock(path: Path, identity: tuple[int, int]) -> bool:
    if not os.path.lexists(path):
        return False
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RuntimeError("refusing to remove unowned export lock")
    path.unlink()
    return True


def _input_directory(value: str | Path, root: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    _assert_real_directory(path, root, name)
    return path


def _input_file(value: str | Path, root: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    if root not in path.parents:
        raise ValueError(f"{name} must remain under repository root")
    _assert_no_symlink(path, root, name)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{name} must be a regular file")
    return path


def _assert_real_directory(path: Path, root: Path, name: str) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"{name} must remain under repository root")
    _assert_no_symlink(path, root, name)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} must be a real directory")


def _assert_no_symlink(path: Path, root: Path, name: str) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{name} contains a symlink component")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file_stable(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("artifact must be a regular file")
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    digest = _hash_file(path)
    after = path.stat(follow_symlinks=False)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise RuntimeError("artifact changed while hashed")
    return digest


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_relative(path: Path, root: Path) -> str:
    absolute = path.absolute()
    if absolute != root and root not in absolute.parents:
        raise ValueError("path is outside repository root")
    relative = absolute.relative_to(root).as_posix()
    lowered = relative.lower().replace("-", "_")
    if "official_test" in lowered or "officialtest" in lowered:
        raise ValueError("artifact path may not reference official test")
    return relative


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("fsync target is not a directory")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _repository_root(value: str | Path | None) -> Path:
    root = (REPOSITORY_ROOT if value is None else Path(value).expanduser()).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository_root must be a real directory")
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--score-manifest-sha256", required=True)
    parser.add_argument("--window-manifest", required=True)
    parser.add_argument("--window-manifest-sha256", required=True)
    parser.add_argument("--window-id", required=True)
    parser.add_argument("--expected-role", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask-folder", default="masks")
    parser.add_argument("--sealed-decision-set")
    parser.add_argument("--sealed-decision-set-sha256")
    parser.add_argument("--statistics-config")
    parser.add_argument("--statistics-config-sha256")
    parser.add_argument("--repository-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = export_stage2_labels(
        score_manifest=args.score_manifest,
        score_manifest_sha256=args.score_manifest_sha256,
        window_manifest=args.window_manifest,
        window_manifest_sha256=args.window_manifest_sha256,
        window_id=args.window_id,
        expected_role=args.expected_role,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        mask_folder=args.mask_folder,
        sealed_decision_set=args.sealed_decision_set,
        sealed_decision_set_sha256=args.sealed_decision_set_sha256,
        statistics_config=args.statistics_config,
        statistics_config_sha256=args.statistics_config_sha256,
        repository_root=args.repository_root,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
