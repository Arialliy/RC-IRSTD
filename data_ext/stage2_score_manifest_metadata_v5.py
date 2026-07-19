"""Metadata-only Stage-2 score-manifest capability for RC5 pre-label use.

The existing v4 verifier intentionally promotes every score and original
image member to a byte-verified capability.  That is the right boundary for
post-decision evaluation, but it is too broad for pre-label context
construction because it would open query members before the decision seal.

This additive verifier replays the complete v4 manifest, selection, run,
checkpoint, runtime and ten-binding metadata contract while deliberately not
opening any path named by a record's ``score_file`` or
``original_image_path``.  It validates their canonical paths, declared
digests, identities, order and geometry metadata only.  A downstream RC5
context producer may promote exactly the fourteen context members; query
members remain unopened until a later full-content verifier is authorized.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from data_ext import stage2_score_manifest as _v4


SCHEMA_VERSION = "rc-irstd.stage2-score-manifest-metadata-capability.v5"
CAPABILITY_CONTRACT = MappingProxyType(
    {
        "manifest_metadata_verified": True,
        "upstream_bindings_verified": True,
        "record_identity_and_geometry_verified": True,
        "member_content_verified": False,
        "record_score_files_opened": False,
        "record_original_images_opened": False,
    }
)

_CAPABILITY_TOKEN = object()


class Stage2ScoreManifestMetadataV5Error(ValueError):
    """A metadata-only score manifest failed its closed v5 contract."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class Stage2ScoreManifestMetadataItemV5:
    """One validated record descriptor whose member bytes remain unopened."""

    record_index: int
    canonical_id: str
    image_id: str
    source_domain: str
    record: Mapping[str, Any]
    score_path: Path
    image_path: Path
    original_hw: tuple[int, int]
    member_content_verified: bool = False


@dataclass(frozen=True, init=False)
class VerifiedStage2ScoreManifestMetadataV5:
    """Verifier-issued, recursively immutable metadata-only capability."""

    path: Path
    repository_root: Path
    payload: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    items: tuple[Stage2ScoreManifestMetadataItemV5, ...]
    role: str
    manifest_sha256: str
    records_content_sha256: str
    bindings: Mapping[str, Mapping[str, str]]
    capability_schema: str
    capability_contract: Mapping[str, bool]
    member_content_verified: bool
    _capability: object

    def __init__(
        self,
        *,
        path: Path | None = None,
        repository_root: Path | None = None,
        payload: Mapping[str, Any] | None = None,
        records: tuple[Mapping[str, Any], ...] | None = None,
        items: tuple[Stage2ScoreManifestMetadataItemV5, ...] | None = None,
        role: str | None = None,
        manifest_sha256: str | None = None,
        records_content_sha256: str | None = None,
        bindings: Mapping[str, Mapping[str, str]] | None = None,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedStage2ScoreManifestMetadataV5 is verifier-issued only"
            )
        required = (
            path,
            repository_root,
            payload,
            records,
            items,
            role,
            manifest_sha256,
            records_content_sha256,
            bindings,
        )
        if any(value is None for value in required):
            raise RuntimeError("metadata capability construction is incomplete")
        frozen_payload = _freeze(payload)
        frozen_records = tuple(_freeze(record) for record in records or ())
        frozen_items = tuple(
            Stage2ScoreManifestMetadataItemV5(
                record_index=item.record_index,
                canonical_id=item.canonical_id,
                image_id=item.image_id,
                source_domain=item.source_domain,
                record=frozen_records[item.record_index],
                score_path=item.score_path,
                image_path=item.image_path,
                original_hw=tuple(item.original_hw),
                member_content_verified=False,
            )
            for item in items or ()
        )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "payload", frozen_payload)
        object.__setattr__(self, "records", frozen_records)
        object.__setattr__(self, "items", frozen_items)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(
            self, "records_content_sha256", records_content_sha256
        )
        object.__setattr__(self, "bindings", _freeze(bindings))
        object.__setattr__(self, "capability_schema", SCHEMA_VERSION)
        object.__setattr__(self, "capability_contract", CAPABILITY_CONTRACT)
        object.__setattr__(self, "member_content_verified", False)
        object.__setattr__(self, "_capability", _CAPABILITY_TOKEN)


def assert_verified_stage2_score_manifest_metadata_v5(
    value: Any,
) -> VerifiedStage2ScoreManifestMetadataV5:
    if (
        type(value) is not VerifiedStage2ScoreManifestMetadataV5
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
        or value.member_content_verified is not False
        or dict(value.capability_contract) != dict(CAPABILITY_CONTRACT)
    ):
        raise TypeError(
            "a verifier-issued metadata-only Stage-2 score capability is required"
        )
    return value


def _logical_member_path(root: Path, value: object, name: str) -> Path:
    """Return a lexical in-root path without touching the named member."""

    relative = _v4._relative_repository_path(value, name)
    return root.joinpath(*PurePosixPath(relative).parts)


def verify_stage2_score_manifest_metadata_v5(
    path: str | Path,
    expected_sha256: str,
    required_role: str,
    *,
    repository_root: str | Path | None = None,
) -> VerifiedStage2ScoreManifestMetadataV5:
    """Verify complete score-manifest metadata without opening record members.

    Upstream artifacts named by the ten manifest bindings are fully verified,
    including restricted checkpoint/runtime provenance replay.  In contrast,
    record ``score_file`` and ``original_image_path`` values are treated as
    canonical, SHA-declared descriptors only: they are neither resolved nor
    opened here.
    """

    try:
        root = _v4._repository_root(repository_root)
        role = _v4._required_role(required_role)
        manifest_path = _v4._existing_direct_path(
            path, root, "score manifest"
        )
        if os.path.lexists(manifest_path.parent / ".export_incomplete"):
            raise RuntimeError(
                "Stage2 score export is incomplete and unsafe: "
                f"{manifest_path.parent}"
            )
        expected = _v4._sha256_value(expected_sha256, "expected_sha256")
        manifest_before = _v4._sha256_file_stable(manifest_path)
        if manifest_before != expected:
            raise ValueError(
                "Stage2 score manifest SHA-256 does not match expected_sha256"
            )

        sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
        if os.path.lexists(sidecar):
            sidecar_path = _v4._existing_direct_path(
                sidecar, root, "manifest SHA-256 sidecar"
            )
            sidecar_before = _v4._sha256_file_stable(sidecar_path)
            if sidecar_path.read_text(encoding="utf-8") != (
                f"{manifest_before}  {manifest_path.name}\n"
            ):
                raise ValueError("manifest SHA-256 sidecar content mismatch")
            if _v4._sha256_file_stable(sidecar_path) != sidecar_before:
                raise RuntimeError(
                    "manifest SHA-256 sidecar changed while verified"
                )

        payload = _v4._read_json_file(manifest_path, "score manifest")
        if _v4._sha256_file_stable(manifest_path) != manifest_before:
            raise RuntimeError(
                "Stage2 score manifest changed while being verified"
            )
        if not isinstance(payload, Mapping):
            raise TypeError("Stage2 score manifest must contain a JSON object")
        _v4._exact_keys(payload, _v4.MANIFEST_FIELDS, "score manifest")
        _v4._verify_top_level_contract(payload, role)

        bindings = _v4._verify_bindings(payload["bindings"], root)
        selection = _v4._read_bound_json(
            bindings["selection_contract"], root
        )
        run_contract = _v4._read_bound_json(bindings["run_contract"], root)
        if not isinstance(selection, Mapping):
            raise TypeError("selection contract must contain a JSON object")
        if not isinstance(run_contract, Mapping):
            raise TypeError("run contract must contain a JSON object")
        _v4._verify_identity_against_contracts(
            payload,
            selection=selection,
            run_contract=run_contract,
            selection_binding=bindings["selection_contract"],
        )
        _v4._verify_provenance_closure(
            payload, run_contract=run_contract, bindings=bindings, root=root
        )

        selected_records = _v4._selection_records(
            selection,
            role=role,
            oof_fold_index=payload["oof_fold_index"],
        )
        raw_records = payload["records"]
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(
                "score manifest records must be a non-empty ordered list"
            )
        if len(raw_records) != _v4._exact_int(
            payload["num_images"], "num_images", minimum=1
        ):
            raise ValueError(
                "num_images does not equal the number of score records"
            )
        if len(raw_records) != len(selected_records):
            raise ValueError(
                "score manifest must contain one record per selected ID"
            )

        records: list[Mapping[str, Any]] = []
        items: list[Stage2ScoreManifestMetadataItemV5] = []
        seen_canonical: set[str] = set()
        seen_score_paths: set[str] = set()
        seen_source_indices: set[int] = set()
        for index, (raw_record, selected_record) in enumerate(
            zip(raw_records, selected_records, strict=True)
        ):
            if not isinstance(raw_record, Mapping):
                raise TypeError(f"records[{index}] must be a JSON object")
            _v4._exact_keys(
                raw_record, _v4.RECORD_FIELDS, f"records[{index}]"
            )
            record = dict(raw_record)
            _v4._verify_record_metadata(
                record,
                selected_record=selected_record,
                payload=payload,
                index=index,
            )
            canonical_id = str(record["canonical_id"])
            score_file = str(record["score_file"])
            source_index = int(record["source_role_record_index"])
            if canonical_id in seen_canonical:
                raise ValueError(f"duplicate canonical_id: {canonical_id!r}")
            if score_file in seen_score_paths:
                raise ValueError(f"duplicate score_file: {score_file!r}")
            if source_index in seen_source_indices:
                raise ValueError(
                    "duplicate source_role_record_index: "
                    f"{source_index!r}"
                )
            seen_canonical.add(canonical_id)
            seen_score_paths.add(score_file)
            seen_source_indices.add(source_index)
            records.append(record)
            items.append(
                Stage2ScoreManifestMetadataItemV5(
                    record_index=index,
                    canonical_id=canonical_id,
                    image_id=str(record["image_id"]),
                    source_domain=str(record["source_domain"]),
                    record=record,
                    score_path=_logical_member_path(
                        root,
                        record["score_file"],
                        f"records[{index}].score_file",
                    ),
                    image_path=_logical_member_path(
                        root,
                        record["original_image_path"],
                        f"records[{index}].original_image_path",
                    ),
                    original_hw=tuple(record["original_hw"]),
                    member_content_verified=False,
                )
            )

        if payload["records_content_sha256_algorithm"] != (
            _v4.STAGE2_SCORE_RECORDS_ALGORITHM
        ):
            raise ValueError("records_content_sha256_algorithm mismatch")
        records_sha = _v4.stage2_score_records_sha256(records)
        if _v4._sha256_value(
            payload["records_content_sha256"], "records_content_sha256"
        ) != records_sha:
            raise ValueError(
                "records_content_sha256 does not bind the ordered records"
            )

        # Only upstream binding members are rechecked.  Record member paths
        # are intentionally absent from this loop and remain unopened.
        for name, binding in bindings.items():
            artifact = _v4._resolve_repository_file(
                root, binding["path"], name
            )
            if _v4._sha256_file_stable(artifact) != binding["sha256"]:
                raise RuntimeError(
                    f"bound artifact changed while verified: {name}"
                )
        if _v4._sha256_file_stable(manifest_path) != manifest_before:
            raise RuntimeError(
                "Stage2 score manifest changed while being verified"
            )

        return VerifiedStage2ScoreManifestMetadataV5(
            path=manifest_path,
            repository_root=root,
            payload=payload,
            records=tuple(records),
            items=tuple(items),
            role=role,
            manifest_sha256=manifest_before,
            records_content_sha256=records_sha,
            bindings=bindings,
            _capability=_CAPABILITY_TOKEN,
        )
    except Stage2ScoreManifestMetadataV5Error:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2ScoreManifestMetadataV5Error(str(error)) from error


__all__ = [
    "CAPABILITY_CONTRACT",
    "SCHEMA_VERSION",
    "Stage2ScoreManifestMetadataItemV5",
    "Stage2ScoreManifestMetadataV5Error",
    "VerifiedStage2ScoreManifestMetadataV5",
    "assert_verified_stage2_score_manifest_metadata_v5",
    "verify_stage2_score_manifest_metadata_v5",
]
