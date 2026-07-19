"""Persistent RC5 authority for a variable-Q Stage-2 source reference.

The legacy source-reference bundle is scientifically useful but predates
RUN_COMPLETE-v2. This additive layer closes the missing causal edge:

    two replayed RC5 score bundles -> one replayed source reference ->
    exact variable-Q consumer set -> commit-last RC5 attestation.

Neither a legacy VerifiedStage2SourceReference nor the variable-query-v2
projection is RC5 authority. Public RC5 consumers must require the verifier-
issued capability from this module and replay it at the point of use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any

from data_ext.stage2_rc5_score_bundle_v2 import (
    BUNDLE_CAPABILITY_SCHEMA as SCORE_BUNDLE_CAPABILITY_SCHEMA,
    VerifiedStage2RC5ScoreBundleV2,
    assert_verified_stage2_rc5_score_bundle_v2,
    replay_verified_stage2_rc5_score_bundle_v2,
)
from rc.build_stage2_source_reference import verify_stage2_source_reference
from rc.stage2_source_reference_variable_query_v2 import (
    CAPABILITY_SCHEMA as VARIABLE_QUERY_V2_CAPABILITY_SCHEMA,
    VerifiedStage2SourceReferenceVariableQueryV2,
    assert_verified_stage2_source_reference_variable_query_v2,
    verify_stage2_source_reference_variable_query_v2,
)


ATTESTATION_SCHEMA = "rc-irstd.stage2-rc5-source-reference-attestation.v3"
ATTESTATION_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_source_reference_attestation"
ATTESTATION_STATUS = "RC5_SOURCE_REFERENCE_ATTESTED_COMMIT"
ATTESTATION_FILENAME = "RC5_SOURCE_REFERENCE_ATTESTATION.json"
COMMIT_FILENAME = "RC5_SOURCE_REFERENCE_ATTESTATION.json.sha256"
CAPABILITY_SCHEMA = "rc-irstd.stage2-rc5-source-reference-capability.v3"

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_CAPABILITY_TOKEN = object()


class Stage2RC5SourceReferenceV3Error(ValueError):
    """The RC5 source-reference causal closure failed closed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise Stage2RC5SourceReferenceV3Error(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation is not canonical JSON"
        ) from error


def _identity_sha(value: Mapping[str, Any]) -> str:
    projection = {
        key: item
        for key, item in value.items()
        if key != "attestation_identity_sha256"
    }
    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2RC5SourceReferenceV3Error(
                f"duplicate attestation JSON key: {key!r}"
            )
        result[key] = value
    return result


def _parse_json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                Stage2RC5SourceReferenceV3Error(
                    f"non-finite JSON constant is forbidden: {raw}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation is invalid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict) or _canonical_json_bytes(value) != data:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation bytes are not canonical"
        )
    return value


def _reject_symlink_components(path: Path, name: str) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Stage2RC5SourceReferenceV3Error(
                f"{name} contains a symlink component: {cursor}"
            )


def _root(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    _reject_symlink_components(raw, "repository_root")
    if not raw.exists() or not stat.S_ISDIR(raw.lstat().st_mode):
        raise FileNotFoundError("repository_root is not a direct directory")
    return raw.resolve(strict=True)


def _direct_file(path: Path, root: Path, name: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    _reject_symlink_components(candidate, name)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise Stage2RC5SourceReferenceV3Error(
            f"{name} is outside repository_root"
        ) from error
    if not candidate.exists() or not stat.S_ISREG(candidate.lstat().st_mode):
        raise FileNotFoundError(f"{name} is not a direct regular file")
    return candidate.resolve(strict=True)


def _relative_file(path: Path, root: Path, name: str) -> str:
    resolved = _direct_file(path, root, name)
    relative = resolved.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise Stage2RC5SourceReferenceV3Error(
            f"{name} path is not canonical"
        )
    return relative


def _stable_bytes(path: Path, root: Path, name: str) -> tuple[bytes, str]:
    resolved = _direct_file(path, root, name)
    before = resolved.stat(follow_symlinks=False)
    data = resolved.read_bytes()
    after = resolved.stat(follow_symlinks=False)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or len(data) != after.st_size:
        raise RuntimeError(f"{name} changed while read")
    return data, hashlib.sha256(data).hexdigest()


def _output_directory(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    _reject_symlink_components(candidate, "attestation output directory")
    if not candidate.exists() or not stat.S_ISDIR(candidate.lstat().st_mode):
        raise FileNotFoundError(
            "attestation output directory must already be a direct directory"
        )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Stage2RC5SourceReferenceV3Error(
            "attestation output directory is outside repository_root"
        ) from error
    return resolved


def _atomic_write_new(path: Path, data: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"immutable attestation member exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replay_base(
    value: VerifiedStage2SourceReferenceVariableQueryV2,
    root: Path,
) -> VerifiedStage2SourceReferenceVariableQueryV2:
    supplied = assert_verified_stage2_source_reference_variable_query_v2(value)
    base = verify_stage2_source_reference(
        supplied.path,
        supplied.npz_sha256,
        supplied.audit_sha256,
        statistics_config=supplied.statistics_config,
        repository_root=root,
    )
    replayed = verify_stage2_source_reference_variable_query_v2(
        base, repository_root=root
    )

    def projection(
        item: VerifiedStage2SourceReferenceVariableQueryV2,
    ) -> dict[str, Any]:
        bundle = item.source_reference_bundle
        return {
            "path": str(item.path),
            "npz_sha256": item.npz_sha256,
            "audit_path": str(item.audit_path),
            "audit_sha256": item.audit_sha256,
            "domains": list(bundle.domains),
            "centers": _plain(bundle.centers),
            "scale": _plain(bundle.scale),
            "source_reference": bundle.source_reference.to_dict(),
            "statistics_config": bundle.statistics_config.to_dict(),
            "stage2_contract": _plain(bundle.stage2_contract),
            "source_contract": _plain(bundle.source_contract),
            "detector_identity": _plain(bundle.detector_identity),
            "checkpoint_binding": _plain(bundle.checkpoint_binding),
            "reference_role": bundle.reference_role,
            "consumer_bindings": _plain(item.consumer_bindings),
            "consumer_windows": [
                {
                    "path": str(window.path),
                    "repository_root": str(window.repository_root),
                    "manifest_sha256": window.manifest_sha256,
                    "payload": _plain(window.payload),
                }
                for window in item.consumer_windows
            ],
            "audit": _plain(bundle.audit),
        }

    if projection(replayed) != projection(supplied):
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference v2 capability state differs from full public replay"
        )
    return replayed


def _score_entry(
    bundle: VerifiedStage2RC5ScoreBundleV2,
    *,
    root: Path,
    base: VerifiedStage2SourceReferenceVariableQueryV2,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = bundle.score_manifest_metadata
    manifest_path = _relative_file(metadata.path, root, "source score manifest")
    actual_summary = {
        "path": manifest_path,
        "sha256": metadata.manifest_sha256,
        "source_domain": metadata.payload["source_domain"],
        "records_content_sha256": metadata.records_content_sha256,
        "record_count": len(metadata.records),
        "selection_contract": _plain(metadata.bindings["selection_contract"]),
    }
    if actual_summary != _plain(expected):
        raise Stage2RC5SourceReferenceV3Error(
            "RC5 score bundle differs from base source-score binding"
        )
    if metadata.role != base.reference_role:
        raise Stage2RC5SourceReferenceV3Error(
            "RC5 source score role differs from base reference role"
        )
    if _plain(metadata.bindings["checkpoint"]) != _plain(base.checkpoint_binding):
        raise Stage2RC5SourceReferenceV3Error(
            "RC5 source score checkpoint differs from source reference"
        )
    run_binding = base.stage2_contract["bindings"]["run_contract"]
    if _plain(metadata.bindings["run_contract"]) != _plain(run_binding):
        raise Stage2RC5SourceReferenceV3Error(
            "RC5 source score run contract differs from source reference"
        )
    attestation = bundle.attestation
    restricted = attestation["restricted_checkpoint"]
    if {
        "path": restricted["path"],
        "sha256": restricted["sha256"],
    } != _plain(base.checkpoint_binding):
        raise Stage2RC5SourceReferenceV3Error(
            "RC5 attestation restricted checkpoint differs from source reference"
        )
    detector = base.detector_identity
    for detector_field, score_field in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target", "outer_target"),
        ("base_seed", "base_seed"),
        ("derived_seed", "derived_seed"),
        ("detector_role", "detector_role"),
        ("oof_fold_index", "oof_fold_index"),
    ):
        if detector[detector_field] != metadata.payload[score_field]:
            raise Stage2RC5SourceReferenceV3Error(
                f"source-reference/score detector identity mismatch: {detector_field}"
            )
    return {
        "source_domain": metadata.payload["source_domain"],
        "score_attestation": {
            "path": _relative_file(
                bundle.attestation_path,
                root,
                "source score attestation",
            ),
            "sha256": bundle.attestation_sha256,
            "capability_schema": SCORE_BUNDLE_CAPABILITY_SCHEMA,
        },
        "score_manifest": actual_summary,
        "run_complete": {
            "path": _relative_file(
                Path(bundle.run_complete.artifact_path),
                root,
                "source RUN_COMPLETE-v2",
            ),
            "sha256": bundle.run_complete.sha256,
            "identity_sha256": attestation["run_complete"]["identity"][
                "identity_sha256"
            ],
        },
        "restricted_checkpoint": {
            "path": restricted["path"],
            "sha256": restricted["sha256"],
        },
    }


def _expected_payload(
    *,
    source_reference: VerifiedStage2SourceReferenceVariableQueryV2,
    score_bundles: Sequence[VerifiedStage2RC5ScoreBundleV2],
    root: Path,
) -> tuple[
    dict[str, Any],
    VerifiedStage2SourceReferenceVariableQueryV2,
    tuple[VerifiedStage2RC5ScoreBundleV2, ...],
]:
    base = _replay_base(source_reference, root)
    if (
        isinstance(score_bundles, (str, bytes))
        or not isinstance(score_bundles, Sequence)
        or len(score_bundles) != 2
    ):
        raise Stage2RC5SourceReferenceV3Error(
            "exactly two RC5 source score bundles are required"
        )
    replayed = tuple(
        replay_verified_stage2_rc5_score_bundle_v2(
            assert_verified_stage2_rc5_score_bundle_v2(item)
        )
        for item in score_bundles
    )
    if any(
        item.score_manifest_metadata.repository_root != root
        for item in replayed
    ):
        raise Stage2RC5SourceReferenceV3Error(
            "source score bundles do not share repository_root"
        )
    expected_rows = base.stage2_contract["bindings"][
        "source_score_manifests"
    ]
    if not isinstance(expected_rows, (tuple, list)) or len(expected_rows) != 2:
        raise Stage2RC5SourceReferenceV3Error(
            "base source reference does not bind exactly two score manifests"
        )
    by_path = {
        item.score_manifest_metadata.path.relative_to(root).as_posix(): item
        for item in replayed
    }
    if len(by_path) != 2 or set(by_path) != {
        str(row["path"]) for row in expected_rows
    }:
        raise Stage2RC5SourceReferenceV3Error(
            "RC5 source score bundle path coverage is not exact"
        )
    ordered = tuple(by_path[str(row["path"])] for row in expected_rows)
    score_rows = [
        _score_entry(bundle, root=root, base=base, expected=expected)
        for bundle, expected in zip(ordered, expected_rows, strict=True)
    ]
    run_identities = {
        (
            row["run_complete"]["path"],
            row["run_complete"]["sha256"],
            row["run_complete"]["identity_sha256"],
        )
        for row in score_rows
    }
    if len(run_identities) != 1:
        raise Stage2RC5SourceReferenceV3Error(
            "two source score bundles do not share one RUN_COMPLETE authority"
        )
    statistics_binding = base.stage2_contract["bindings"][
        "statistics_config"
    ]
    payload: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA,
        "artifact_type": ATTESTATION_ARTIFACT_TYPE,
        "artifact_status": ATTESTATION_STATUS,
        "development_only": True,
        "official_test_accessed": False,
        "observed_results": None,
        "source_reference": {
            "npz_path": _relative_file(
                base.path, root, "source-reference NPZ"
            ),
            "npz_sha256": base.npz_sha256,
            "audit_path": _relative_file(
                base.audit_path, root, "source-reference audit"
            ),
            "audit_sha256": base.audit_sha256,
            "variable_query_capability_schema": (
                VARIABLE_QUERY_V2_CAPABILITY_SCHEMA
            ),
            "reference_role": base.reference_role,
            "detector_identity": _plain(base.detector_identity),
            "checkpoint_binding": _plain(base.checkpoint_binding),
        },
        "source_score_bundles": score_rows,
        "consumer_windows": _plain(base.consumer_bindings),
        "statistics_config": _plain(statistics_binding),
        "guardrails": {
            "source_domain_count": 2,
            "same_run_complete_required": True,
            "outer_target_score_bundle_present": False,
            "source_score_member_content_verified_by_base_replay": True,
            "query_labels_accessed": False,
            "official_test_accessed": False,
        },
        "causal_edge": (
            "two_RC5_score_bundles->replayed_source_reference->"
            "variable_query_v2_consumers->RC5_source_reference_v3"
        ),
        "commit_last_marker": COMMIT_FILENAME,
        "attestation_identity_sha256": "",
    }
    payload["attestation_identity_sha256"] = _identity_sha(payload)
    return payload, base, ordered


class VerifiedStage2RC5SourceReferenceV3:
    """Immutable verifier-issued RC5 source-reference authority."""

    __slots__ = (
        "attestation_path",
        "attestation_sha256",
        "attestation",
        "source_reference_v2",
        "score_bundles",
        "repository_root",
        "capability_schema",
        "_capability",
    )

    def __init__(
        self,
        *,
        attestation_path: Path,
        attestation_sha256: str,
        attestation: Mapping[str, Any],
        source_reference_v2: VerifiedStage2SourceReferenceVariableQueryV2,
        score_bundles: tuple[VerifiedStage2RC5ScoreBundleV2, ...],
        repository_root: Path,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedStage2RC5SourceReferenceV3 is verifier-issued only"
            )
        object.__setattr__(self, "attestation_path", attestation_path)
        object.__setattr__(self, "attestation_sha256", attestation_sha256)
        object.__setattr__(self, "attestation", _freeze(attestation))
        object.__setattr__(
            self,
            "source_reference_v2",
            assert_verified_stage2_source_reference_variable_query_v2(
                source_reference_v2
            ),
        )
        object.__setattr__(
            self,
            "score_bundles",
            tuple(
                assert_verified_stage2_rc5_score_bundle_v2(item)
                for item in score_bundles
            ),
        )
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "capability_schema", CAPABILITY_SCHEMA)
        object.__setattr__(self, "_capability", _CAPABILITY_TOKEN)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("verified RC5 source-reference v3 is immutable")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.source_reference_v2, name)


def assert_verified_stage2_rc5_source_reference_v3(
    value: Any,
) -> VerifiedStage2RC5SourceReferenceV3:
    if (
        type(value) is not VerifiedStage2RC5SourceReferenceV3
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
        or value.capability_schema != CAPABILITY_SCHEMA
        or len(value.score_bundles) != 2
    ):
        raise TypeError(
            "a verifier-issued VerifiedStage2RC5SourceReferenceV3 is required"
        )
    assert_verified_stage2_source_reference_variable_query_v2(
        value.source_reference_v2
    )
    for item in value.score_bundles:
        assert_verified_stage2_rc5_score_bundle_v2(item)
    return value


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "observed_results",
        "source_reference",
        "source_score_bundles",
        "consumer_windows",
        "statistics_config",
        "guardrails",
        "causal_edge",
        "commit_last_marker",
        "attestation_identity_sha256",
    }
)


def _validate_payload(value: Mapping[str, Any]) -> None:
    if set(value) != _TOP_LEVEL_FIELDS:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation key closure mismatch"
        )
    if (
        value["schema_version"] != ATTESTATION_SCHEMA
        or value["artifact_type"] != ATTESTATION_ARTIFACT_TYPE
        or value["artifact_status"] != ATTESTATION_STATUS
        or value["development_only"] is not True
        or value["official_test_accessed"] is not False
        or value["observed_results"] is not None
        or value["commit_last_marker"] != COMMIT_FILENAME
        or value["attestation_identity_sha256"] != _identity_sha(value)
    ):
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation top-level contract drifted"
        )
    _sha(
        value["attestation_identity_sha256"],
        "attestation_identity_sha256",
    )
    rows = value["source_score_bundles"]
    if not isinstance(rows, list) or len(rows) != 2:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation must contain two score bundles"
        )
    if len({row.get("source_domain") for row in rows if isinstance(row, Mapping)}) != 2:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation source domains are not unique"
        )
    guardrails = value["guardrails"]
    expected_guardrails = {
        "source_domain_count": 2,
        "same_run_complete_required": True,
        "outer_target_score_bundle_present": False,
        "source_score_member_content_verified_by_base_replay": True,
        "query_labels_accessed": False,
        "official_test_accessed": False,
    }
    if guardrails != expected_guardrails:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference attestation guardrails drifted"
        )


def _issue(
    *,
    attestation_path: Path,
    attestation_sha256: str,
    payload: Mapping[str, Any],
    source_reference: VerifiedStage2SourceReferenceVariableQueryV2,
    score_bundles: tuple[VerifiedStage2RC5ScoreBundleV2, ...],
    root: Path,
) -> VerifiedStage2RC5SourceReferenceV3:
    return VerifiedStage2RC5SourceReferenceV3(
        attestation_path=attestation_path,
        attestation_sha256=attestation_sha256,
        attestation=payload,
        source_reference_v2=source_reference,
        score_bundles=score_bundles,
        repository_root=root,
        _capability=_CAPABILITY_TOKEN,
    )


def verify_stage2_rc5_source_reference_v3(
    attestation_path: str | Path,
    expected_attestation_sha256: str,
    *,
    source_reference: VerifiedStage2SourceReferenceVariableQueryV2,
    score_bundles: Sequence[VerifiedStage2RC5ScoreBundleV2],
    repository_root: str | Path,
) -> VerifiedStage2RC5SourceReferenceV3:
    """Verify commit first, then replay every transitive source dependency."""

    root = _root(repository_root)
    raw_path = Path(attestation_path).expanduser()
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    commit_path = _direct_file(
        raw_path.with_name(COMMIT_FILENAME),
        root,
        "source-reference v3 commit-last sidecar",
    )
    expected_sha = _sha(
        expected_attestation_sha256,
        "expected_attestation_sha256",
    )
    commit_bytes, _ = _stable_bytes(
        commit_path,
        root,
        "source-reference v3 commit-last sidecar",
    )
    expected_commit = (
        f"{expected_sha}  {ATTESTATION_FILENAME}\n".encode("utf-8")
    )
    if commit_bytes != expected_commit:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference v3 commit-last sidecar mismatch"
        )
    path = _direct_file(
        raw_path, root, "source-reference v3 attestation"
    )
    if (
        path.name != ATTESTATION_FILENAME
        or commit_path.parent != path.parent
    ):
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference v3 canonical layout mismatch"
        )
    data, actual_sha = _stable_bytes(
        path, root, "source-reference v3 attestation"
    )
    if actual_sha != expected_sha:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference v3 external SHA-256 mismatch"
        )
    payload = _parse_json(data)
    _validate_payload(payload)
    expected, replayed_base, replayed_bundles = _expected_payload(
        source_reference=source_reference,
        score_bundles=score_bundles,
        root=root,
    )
    if payload != expected or data != _canonical_json_bytes(expected):
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference v3 differs from current-state causal replay"
        )
    commit_after, _ = _stable_bytes(
        commit_path,
        root,
        "source-reference v3 commit-last sidecar",
    )
    data_after, sha_after = _stable_bytes(
        path, root, "source-reference v3 attestation"
    )
    if (
        commit_after != commit_bytes
        or data_after != data
        or sha_after != actual_sha
    ):
        raise RuntimeError(
            "source-reference v3 changed during full verification"
        )
    return _issue(
        attestation_path=path,
        attestation_sha256=actual_sha,
        payload=expected,
        source_reference=replayed_base,
        score_bundles=replayed_bundles,
        root=root,
    )


def publish_stage2_rc5_source_reference_v3(
    *,
    source_reference: VerifiedStage2SourceReferenceVariableQueryV2,
    score_bundles: Sequence[VerifiedStage2RC5ScoreBundleV2],
    output_directory: str | Path,
    repository_root: str | Path,
) -> VerifiedStage2RC5SourceReferenceV3:
    """Publish one immutable attestation and its commit-last sidecar."""

    root = _root(repository_root)
    output = _output_directory(output_directory, root)
    attestation_path = output / ATTESTATION_FILENAME
    commit_path = output / COMMIT_FILENAME
    if os.path.lexists(attestation_path) or os.path.lexists(commit_path):
        raise FileExistsError(
            "immutable source-reference v3 publication already exists"
        )
    payload, replayed_base, replayed_bundles = _expected_payload(
        source_reference=source_reference,
        score_bundles=score_bundles,
        root=root,
    )
    _validate_payload(payload)
    data = _canonical_json_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    commit = f"{digest}  {ATTESTATION_FILENAME}\n".encode("utf-8")
    _atomic_write_new(attestation_path, data)
    precommit, _, _ = _expected_payload(
        source_reference=replayed_base,
        score_bundles=replayed_bundles,
        root=root,
    )
    current, current_sha = _stable_bytes(
        attestation_path,
        root,
        "precommit source-reference v3 attestation",
    )
    if precommit != payload or current != data or current_sha != digest:
        raise Stage2RC5SourceReferenceV3Error(
            "source-reference inputs changed before commit"
        )
    _atomic_write_new(commit_path, commit)
    try:
        return verify_stage2_rc5_source_reference_v3(
            attestation_path,
            digest,
            source_reference=replayed_base,
            score_bundles=replayed_bundles,
            repository_root=root,
        )
    except BaseException:
        if (
            commit_path.exists()
            and not commit_path.is_symlink()
            and stat.S_ISREG(commit_path.lstat().st_mode)
        ):
            try:
                observed, _ = _stable_bytes(
                    commit_path,
                    root,
                    "failed source-reference v3 commit",
                )
                if observed == commit:
                    commit_path.unlink()
            except (OSError, RuntimeError, Stage2RC5SourceReferenceV3Error):
                pass
        raise


def replay_verified_stage2_rc5_source_reference_v3(
    value: Any,
) -> VerifiedStage2RC5SourceReferenceV3:
    """Revalidate a stored v3 capability at each downstream boundary."""

    supplied = assert_verified_stage2_rc5_source_reference_v3(value)
    return verify_stage2_rc5_source_reference_v3(
        supplied.attestation_path,
        supplied.attestation_sha256,
        source_reference=supplied.source_reference_v2,
        score_bundles=supplied.score_bundles,
        repository_root=supplied.repository_root,
    )


__all__ = [
    "ATTESTATION_ARTIFACT_TYPE",
    "ATTESTATION_FILENAME",
    "ATTESTATION_SCHEMA",
    "ATTESTATION_STATUS",
    "CAPABILITY_SCHEMA",
    "COMMIT_FILENAME",
    "Stage2RC5SourceReferenceV3Error",
    "VerifiedStage2RC5SourceReferenceV3",
    "assert_verified_stage2_rc5_source_reference_v3",
    "publish_stage2_rc5_source_reference_v3",
    "replay_verified_stage2_rc5_source_reference_v3",
    "verify_stage2_rc5_source_reference_v3",
]
