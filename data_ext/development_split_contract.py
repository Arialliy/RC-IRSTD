"""Replay frozen official-train-derived detector diagnostic partitions.

The Stage-1 Gate is a development-only evaluation.  Its selected IDs come
from ``detector_diagnostic.txt`` in the frozen split manifest, never from an
arbitrary file that merely happens to contain the same bytes.  This module
validates that manifest and its complete detector partition without opening
or resolving any mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .dataset_identity import (
    ORDERED_SAMPLE_IDS_ALGORITHM,
    ordered_sample_ids_sha256,
    sha256_file,
)
from .split_utils import read_split_entries, sample_id_from_entry


DETECTOR_DIAGNOSTIC_ROLE = "detector_diagnostic"
DEVELOPMENT_SPLIT_CONTRACT_SCHEMA_VERSION = 2
DEVELOPMENT_MANIFEST_SCHEMA_VERSION = (
    "rc-irstd.aaai27-official-train-splits.v2"
)
DEVELOPMENT_MANIFEST_ARTIFACT_TYPE = "official_train_derived_role_splits"
DEVELOPMENT_PARTITION_SCOPE = (
    "official_train_derived_development_diagnostic"
)


@dataclass(frozen=True)
class FrozenIdPartition:
    """One content-addressed ordered ID file from the frozen manifest."""

    path: Path
    file_sha256: str
    sample_ids: tuple[str, ...]
    ids_sha256: str


@dataclass(frozen=True)
class VerifiedDetectorDiagnosticPartition:
    """A fully replayed detector fit/diagnostic/quarantine partition."""

    manifest_path: Path
    manifest_sha256: str
    dataset_name: str
    dataset_root: Path
    official_train_path: Path
    official_test_path: Path
    partitions: Mapping[str, FrozenIdPartition]


def verify_detector_diagnostic_partition(
    manifest_path: str | Path,
    *,
    dataset_name: str,
    dataset_root: str | Path,
    selected_split_file: str | Path,
    official_train_split: str | Path,
    official_test_split: str | Path,
    expected_manifest_sha256: str | None = None,
) -> VerifiedDetectorDiagnosticPartition:
    """Verify one dataset's frozen development partition, fail closed.

    Only split text and image-independent manifest metadata are read here.
    In particular, this function never enumerates, resolves, or opens masks.
    """

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(
            f"derived split manifest does not exist: {manifest}"
        )
    manifest_sha_before = sha256_file(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest)
    if manifest_sha != manifest_sha_before:
        raise RuntimeError(f"derived split manifest changed while read: {manifest}")
    if expected_manifest_sha256 is not None and manifest_sha != _sha256(
        expected_manifest_sha256,
        "expected derived split manifest SHA-256",
    ):
        raise ValueError("derived split manifest SHA-256 does not match the frozen value")
    if not isinstance(payload, Mapping):
        raise TypeError("derived split manifest must contain a JSON object")
    if payload.get("schema_version") != DEVELOPMENT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "detector_diagnostic requires the frozen official-train split "
            f"manifest schema {DEVELOPMENT_MANIFEST_SCHEMA_VERSION!r}"
        )
    if payload.get("artifact_type") != DEVELOPMENT_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("derived split manifest artifact_type mismatch")

    role_contract = payload.get("role_contract")
    if not isinstance(role_contract, Mapping):
        raise TypeError("derived split manifest role_contract must be a mapping")
    deprecated_role_fields = {
        "outer_target_official_train_used",
        "outer_target_official_train_allowed_in_same_outer_fold",
    }
    present_deprecated_fields = deprecated_role_fields.intersection(role_contract)
    if present_deprecated_fields:
        raise ValueError(
            "derived split manifest contains deprecated ambiguous role fields: "
            f"{sorted(present_deprecated_fields)}"
        )
    exact_role_fields = {
        "official_test_emitted": False,
        "official_test_labels_read_for_quarantine": False,
        "outer_target_official_train_used_for_detector_fit": False,
        "outer_target_detector_diagnostic_used_for_development_evaluation": True,
        "outer_target_diagnostic_selects_checkpoint": False,
        "detector_checkpoint_selection": "fixed_last",
        "detector_diagnostic_used_for_checkpoint_selection": False,
    }
    for field, expected in exact_role_fields.items():
        if role_contract.get(field) != expected:
            raise ValueError(
                f"derived split manifest role_contract.{field} must be "
                f"exactly {expected!r}"
            )

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("derived split manifest requires a non-empty datasets list")
    summaries: dict[str, Mapping[str, Any]] = {}
    for index, raw_summary in enumerate(raw_datasets):
        if not isinstance(raw_summary, Mapping):
            raise TypeError(f"derived split datasets[{index}] must be a mapping")
        name = _nonempty(raw_summary.get("dataset_name"), f"datasets[{index}].dataset_name")
        if name in summaries:
            raise ValueError(f"duplicate dataset_name in derived split manifest: {name!r}")
        summaries[name] = raw_summary
    requested_name = _nonempty(dataset_name, "dataset_name")
    if requested_name not in summaries:
        raise ValueError(
            f"dataset {requested_name!r} is absent from the derived split manifest"
        )
    summary = summaries[requested_name]

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    declared_root = _relative_path(
        summary.get("dataset_root"), "derived dataset_root"
    )
    repository_root = _infer_anchor(root, declared_root)
    if (repository_root / declared_root).resolve() != root:
        raise ValueError(
            "dataset root does not exactly match the path declared by the "
            "frozen split manifest"
        )

    official_train = Path(official_train_split).expanduser().resolve()
    official_test = Path(official_test_split).expanduser().resolve()
    declared_train = _resolve_within(
        repository_root,
        summary.get("official_train_split"),
        "official_train_split",
    )
    declared_test = _resolve_within(
        repository_root,
        summary.get("official_test_split"),
        "official_test_split",
    )
    if official_train != declared_train or official_test != declared_test:
        raise ValueError(
            "concrete official train/test split paths do not match the frozen "
            "derived split manifest"
        )

    train_ids = _verify_official_file(
        official_train,
        declared_sha=summary.get("official_train_split_sha256"),
        declared_count=summary.get("official_train_count"),
        field="official_train",
    )
    test_ids = _verify_official_file(
        official_test,
        declared_sha=summary.get("official_test_split_sha256"),
        declared_count=summary.get("official_test_count"),
        field="official_test",
    )
    if set(train_ids).intersection(test_ids):
        raise ValueError("official train/test IDs overlap in derived split replay")
    if _exact_nonnegative_integer(
        summary.get("official_train_test_id_overlap_count"),
        "official_train_test_id_overlap_count",
    ) != 0:
        raise ValueError(
            "derived split manifest must declare zero official train/test ID overlap"
        )

    detector = summary.get("detector")
    quarantine = summary.get("development_quarantine")
    if not isinstance(detector, Mapping):
        raise TypeError("derived split dataset detector record must be a mapping")
    if not isinstance(quarantine, Mapping):
        raise TypeError(
            "detector_diagnostic requires a v2 development_quarantine record"
        )
    if quarantine.get("partition_of_official_train") is not True:
        raise ValueError(
            "development_quarantine.partition_of_official_train must be exactly true"
        )

    specifications = {
        "effective_development_train": (
            quarantine,
            "effective_development_train_file",
            "effective_development_train_sha256",
            "effective_development_train_count",
            False,
        ),
        "detector_fit": (
            detector,
            "fit_file",
            "fit_sha256",
            "fit_count",
            False,
        ),
        DETECTOR_DIAGNOSTIC_ROLE: (
            detector,
            "diagnostic_file",
            "diagnostic_sha256",
            "diagnostic_count",
            False,
        ),
        "quarantined_official_train_ids": (
            quarantine,
            "quarantined_file",
            "quarantined_sha256",
            "quarantined_count",
            True,
        ),
    }
    partitions: dict[str, FrozenIdPartition] = {}
    for role, (record, file_field, sha_field, count_field, allow_empty) in specifications.items():
        path = _resolve_within(
            manifest.parent,
            record.get(file_field),
            f"{role}.{file_field}",
        )
        declared_sha = _sha256(record.get(sha_field), f"{role}.{sha_field}")
        if not path.is_file():
            raise FileNotFoundError(f"frozen {role} split does not exist: {path}")
        before = sha256_file(path)
        if before != declared_sha:
            raise ValueError(f"frozen {role} split SHA-256 mismatch")
        ids = _read_ids(path, allow_empty=allow_empty)
        after = sha256_file(path)
        if after != before:
            raise RuntimeError(f"frozen {role} split changed while read: {path}")
        declared_count = _exact_nonnegative_integer(
            record.get(count_field), f"{role}.{count_field}"
        )
        if len(ids) != declared_count:
            raise ValueError(f"frozen {role} split count mismatch")
        partitions[role] = FrozenIdPartition(
            path=path,
            file_sha256=before,
            sample_ids=tuple(ids),
            ids_sha256=_ordered_ids_sha256_allow_empty(ids),
        )

    selected = Path(selected_split_file).expanduser().resolve()
    declared_diagnostic = partitions[DETECTOR_DIAGNOSTIC_ROLE].path
    if selected != declared_diagnostic:
        raise ValueError(
            "split_role='detector_diagnostic' requires the exact diagnostic "
            f"path declared by the frozen manifest ({declared_diagnostic}); "
            f"the selected path is {selected}"
        )

    train_set = set(train_ids)
    test_set = set(test_ids)
    effective_set = set(partitions["effective_development_train"].sample_ids)
    fit_set = set(partitions["detector_fit"].sample_ids)
    diagnostic_set = set(partitions[DETECTOR_DIAGNOSTIC_ROLE].sample_ids)
    quarantine_set = set(
        partitions["quarantined_official_train_ids"].sample_ids
    )
    if effective_set.intersection(quarantine_set) or (
        effective_set | quarantine_set
    ) != train_set:
        raise ValueError(
            "effective development and quarantine are not an exact disjoint "
            "partition of official train"
        )
    if fit_set.intersection(diagnostic_set) or (
        fit_set | diagnostic_set
    ) != effective_set:
        raise ValueError(
            "detector_fit and detector_diagnostic are not an exact disjoint "
            "partition of effective development train"
        )
    if diagnostic_set.intersection(test_set):
        raise ValueError("detector_diagnostic overlaps official test")
    if diagnostic_set.intersection(quarantine_set):
        raise ValueError("detector_diagnostic overlaps development quarantine")

    return VerifiedDetectorDiagnosticPartition(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        dataset_name=requested_name,
        dataset_root=root,
        official_train_path=official_train,
        official_test_path=official_test,
        partitions=partitions,
    )


def serialise_development_partition_contract(
    verified: VerifiedDetectorDiagnosticPartition,
    *,
    path_anchor: str | Path,
) -> dict[str, object]:
    """Return the canonical JSON fields embedded in split-contract v2."""

    anchor = Path(path_anchor).expanduser().resolve()
    return {
        "partition_scope": DEVELOPMENT_PARTITION_SCOPE,
        "official_test_artifact": False,
        "final_evaluation_eligible": False,
        "development_only": True,
        "official_test_labels_read": False,
        "derived_split_manifest_file": _portable_path(
            verified.manifest_path, anchor
        ),
        "derived_split_manifest_sha256": verified.manifest_sha256,
        "derived_split_manifest_schema_version": (
            DEVELOPMENT_MANIFEST_SCHEMA_VERSION
        ),
        "derived_dataset_name": verified.dataset_name,
        "derived_partitions": {
            role: {
                "split_file": _portable_path(partition.path, anchor),
                "split_sha256": partition.file_sha256,
                "num_images": len(partition.sample_ids),
                "ids_sha256": partition.ids_sha256,
            }
            for role, partition in verified.partitions.items()
        },
        "partition_audit": {
            "official_train_equals_effective_plus_quarantine": True,
            "effective_equals_fit_plus_diagnostic": True,
            "detector_fit_diagnostic_disjoint": True,
            "diagnostic_official_test_disjoint": True,
            "diagnostic_quarantine_disjoint": True,
        },
    }


def _verify_official_file(
    path: Path,
    *,
    declared_sha: object,
    declared_count: object,
    field: str,
) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{field} split does not exist: {path}")
    expected_sha = _sha256(declared_sha, f"{field}_split_sha256")
    before = sha256_file(path)
    if before != expected_sha:
        raise ValueError(f"{field} split SHA-256 mismatch")
    ids = [sample_id_from_entry(entry) for entry in read_split_entries(path)]
    if len(ids) != _exact_nonnegative_integer(declared_count, f"{field}_count"):
        raise ValueError(f"{field} split count mismatch")
    if sha256_file(path) != before:
        raise RuntimeError(f"{field} split changed while read: {path}")
    return ids


def _read_ids(path: Path, *, allow_empty: bool) -> list[str]:
    if allow_empty and not path.read_text(encoding="utf-8-sig").strip():
        return []
    return [sample_id_from_entry(entry) for entry in read_split_entries(path)]


def _ordered_ids_sha256_allow_empty(sample_ids: list[str]) -> str:
    if sample_ids:
        return ordered_sample_ids_sha256(sample_ids)
    digest = hashlib.sha256()
    _update_frame(digest, ORDERED_SAMPLE_IDS_ALGORITHM)
    return digest.hexdigest()


def _infer_anchor(actual: Path, declared_relative: Path) -> Path:
    if len(declared_relative.parts) > len(actual.parts):
        raise ValueError("declared dataset_root cannot resolve to the dataset root")
    anchor = actual
    for _ in declared_relative.parts:
        anchor = anchor.parent
    return anchor


def _resolve_within(root: Path, value: object, field: str) -> Path:
    relative = _relative_path(value, field)
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{field} escapes its declared path anchor")
    return resolved


def _relative_path(value: object, field: str) -> Path:
    rendered = _nonempty(value, field)
    path = Path(rendered).expanduser()
    if (
        path.is_absolute()
        or not path.parts
        or path == Path(".")
        or any(part == ".." for part in path.parts)
    ):
        raise ValueError(f"{field} must be a repository-relative path")
    return path


def _portable_path(path: Path, anchor: Path) -> str:
    import os

    return Path(os.path.relpath(path.resolve(), start=anchor)).as_posix()


def _nonempty(value: object, field: str) -> str:
    rendered = "" if value is None else str(value).strip()
    if not rendered:
        raise ValueError(f"{field} must be non-empty")
    return rendered


def _sha256(value: object, field: str) -> str:
    rendered = "" if value is None else str(value).strip().lower()
    if len(rendered) != 64 or any(c not in "0123456789abcdef" for c in rendered):
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    return rendered


def _exact_nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer") from exc
    if value != integer or integer < 0:
        raise ValueError(f"{field} must be a non-negative exact integer")
    return integer


def _update_frame(digest: "hashlib._Hash", value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
