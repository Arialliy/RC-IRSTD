"""Persistent RUN_COMPLETE -> score-manifest attestation for RC5.

The v4 score-manifest schema predates detector RUN_COMPLETE-v2.  Checking a
completion capability immediately before inference is necessary but not
sufficient: without a persistent edge, a later context producer cannot prove
which completed run authorized the score bytes.  This additive attestation
binds the unchanged v4 manifest to RUN_COMPLETE-v2 and exposes one verifier-
issued bundle containing both the label-blind metadata-v5 capability and the
replayed detector-completion capability.

``RC5_SCORE_ATTESTATION.json.sha256`` is the commit-last marker.  A v4 export,
or even a bare attestation JSON file, is not an authoritative RC5 score bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

from data_ext.stage2_detector_run_complete_v2 import (
    RUN_COMPLETE_SCHEMA_V2,
    VerifiedStage2DetectorRunCompleteV2,
    assert_stage2_run_complete_for_score_export_v2,
    assert_verified_stage2_detector_run_complete_v2,
)
from data_ext.stage2_score_manifest_metadata_v5 import (
    SCHEMA_VERSION as METADATA_V5_SCHEMA,
    VerifiedStage2ScoreManifestMetadataV5,
    assert_verified_stage2_score_manifest_metadata_v5,
    verify_stage2_score_manifest_metadata_v5,
)


ATTESTATION_SCHEMA_V2 = "rc-irstd.stage2-rc5-score-attestation.v2"
ATTESTATION_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_score_attestation"
ATTESTATION_STATUS = "RC5_SCORE_EXPORT_ATTESTED_COMMIT"
ATTESTATION_NAME = "RC5_SCORE_ATTESTATION.json"
ATTESTATION_SIDECAR_NAME = "RC5_SCORE_ATTESTATION.json.sha256"
BUNDLE_CAPABILITY_SCHEMA = "rc-irstd.stage2-rc5-score-bundle-capability.v2"

CAPABILITY_CONTRACT = MappingProxyType(
    {
        "run_complete_current_state_replayed": True,
        "score_manifest_metadata_v5_replayed": True,
        "score_manifest_member_content_verified": False,
        "score_record_files_opened": False,
        "score_original_images_opened": False,
        "attestation_commit_last_verified": True,
        "restricted_checkpoint_is_score_authority": True,
        "weights_last_is_score_authority": False,
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CAPABILITY_TOKEN = object()


class Stage2RC5ScoreBundleV2Error(ValueError):
    """An RC5 RUN_COMPLETE-to-score attestation failed closed."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class VerifiedStage2RC5ScoreBundleV2:
    """Immutable, public-verifier-issued RUN_COMPLETE -> score capability."""

    __slots__ = (
        "attestation_path",
        "attestation_sha256",
        "attestation",
        "score_manifest_metadata",
        "run_complete",
        "capability_schema",
        "capability_contract",
        "_capability",
    )

    def __init__(
        self,
        *,
        attestation_path: Path,
        attestation_sha256: str,
        attestation: Mapping[str, Any],
        score_manifest_metadata: VerifiedStage2ScoreManifestMetadataV5,
        run_complete: VerifiedStage2DetectorRunCompleteV2,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError("VerifiedStage2RC5ScoreBundleV2 is verifier-issued only")
        metadata = assert_verified_stage2_score_manifest_metadata_v5(
            score_manifest_metadata
        )
        complete = assert_verified_stage2_detector_run_complete_v2(run_complete)
        object.__setattr__(self, "attestation_path", attestation_path)
        object.__setattr__(self, "attestation_sha256", attestation_sha256)
        object.__setattr__(self, "attestation", _freeze(attestation))
        object.__setattr__(self, "score_manifest_metadata", metadata)
        object.__setattr__(self, "run_complete", complete)
        object.__setattr__(self, "capability_schema", BUNDLE_CAPABILITY_SCHEMA)
        object.__setattr__(self, "capability_contract", CAPABILITY_CONTRACT)
        object.__setattr__(self, "_capability", _CAPABILITY_TOKEN)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("verified RC5 score bundle is immutable")


def assert_verified_stage2_rc5_score_bundle_v2(
    value: Any,
) -> VerifiedStage2RC5ScoreBundleV2:
    if (
        type(value) is not VerifiedStage2RC5ScoreBundleV2
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
        or getattr(value, "capability_schema", None) != BUNDLE_CAPABILITY_SCHEMA
        or dict(getattr(value, "capability_contract", {}))
        != dict(CAPABILITY_CONTRACT)
    ):
        raise TypeError(
            "a verifier-issued VerifiedStage2RC5ScoreBundleV2 is required"
        )
    assert_verified_stage2_score_manifest_metadata_v5(
        value.score_manifest_metadata
    )
    assert_verified_stage2_detector_run_complete_v2(value.run_complete)
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise Stage2RC5ScoreBundleV2Error(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Stage2RC5ScoreBundleV2Error(
                f"{label} contains a symlink component: {cursor}"
            )


def _regular_file(path: Path, label: str) -> Path:
    _reject_symlink_components(path, label)
    if not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    return path.resolve(strict=True)


def _stable_bytes(path: Path, label: str) -> tuple[bytes, str]:
    path = _regular_file(path, label)
    before = _file_sha256(path)
    payload = path.read_bytes()
    after = _file_sha256(path)
    if before != after or hashlib.sha256(payload).hexdigest() != before:
        raise RuntimeError(f"{label} changed while read")
    return payload, before


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes, str]:
    raw, digest = _stable_bytes(path, label)
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                Stage2RC5ScoreBundleV2Error(
                    f"{label} contains non-finite JSON constant {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5ScoreBundleV2Error(f"{label} is not canonical JSON") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload, raw, digest


def _repo_relative(path: Path, root: Path, label: str) -> str:
    path = _regular_file(path, label)
    try:
        relative = path.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise Stage2RC5ScoreBundleV2Error(
            f"{label} is outside repository_root"
        ) from error
    rendered = PurePosixPath(relative)
    if (
        rendered.is_absolute()
        or rendered.as_posix() != relative
        or any(part in {"", ".", ".."} for part in rendered.parts)
    ):
        raise Stage2RC5ScoreBundleV2Error(f"{label} path is not canonical")
    return relative


def _resolve_repo_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage2RC5ScoreBundleV2Error(f"{label} path is not canonical")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.as_posix() != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Stage2RC5ScoreBundleV2Error(f"{label} path is not canonical")
    return _regular_file(root.joinpath(*relative.parts), label)


def _identity_digest(identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(b"rc-irstd.stage2-rc5-run-complete-identity.v1\0")
    digest.update(_canonical_json(identity))
    return digest.hexdigest()


def _replay_inputs(
    score_manifest_metadata: VerifiedStage2ScoreManifestMetadataV5,
    run_complete: VerifiedStage2DetectorRunCompleteV2,
) -> tuple[
    VerifiedStage2ScoreManifestMetadataV5,
    VerifiedStage2DetectorRunCompleteV2,
]:
    supplied_metadata = assert_verified_stage2_score_manifest_metadata_v5(
        score_manifest_metadata
    )
    supplied_complete = assert_verified_stage2_detector_run_complete_v2(run_complete)
    metadata = verify_stage2_score_manifest_metadata_v5(
        supplied_metadata.path,
        supplied_metadata.manifest_sha256,
        supplied_metadata.role,
        repository_root=supplied_metadata.repository_root,
    )
    if (
        metadata.manifest_sha256 != supplied_metadata.manifest_sha256
        or metadata.records_content_sha256
        != supplied_metadata.records_content_sha256
        or _thaw(metadata.payload) != _thaw(supplied_metadata.payload)
    ):
        raise Stage2RC5ScoreBundleV2Error(
            "supplied metadata-v5 capability differs from public-verifier replay"
        )

    root = metadata.repository_root.resolve(strict=True)
    run_binding = metadata.bindings["run_contract"]
    checkpoint_binding = metadata.bindings["checkpoint"]
    assert_stage2_run_complete_for_score_export_v2(
        supplied_complete,
        run_contract_path=_resolve_repo_file(
            root, run_binding["path"], "score-manifest run contract"
        ),
        run_contract_sha256=run_binding["sha256"],
        checkpoint_path=_resolve_repo_file(
            root, checkpoint_binding["path"], "score-manifest checkpoint"
        ),
        checkpoint_sha256=checkpoint_binding["sha256"],
    )
    return metadata, supplied_complete


def _expected_attestation(
    score_manifest_metadata: VerifiedStage2ScoreManifestMetadataV5,
    run_complete: VerifiedStage2DetectorRunCompleteV2,
) -> tuple[
    dict[str, Any],
    VerifiedStage2ScoreManifestMetadataV5,
    VerifiedStage2DetectorRunCompleteV2,
]:
    metadata, complete = _replay_inputs(score_manifest_metadata, run_complete)
    root = metadata.repository_root.resolve(strict=True)
    complete_payload = _thaw(complete.payload)
    manifest_payload = _thaw(metadata.payload)
    run_identity = complete_payload["run_identity"]
    run_complete_identity = {
        "schema_version": RUN_COMPLETE_SCHEMA_V2,
        "run_identity": run_identity,
        "target_epochs": complete_payload["target_epochs"],
        "completed_epoch": complete_payload["completed_epoch"],
        "state_dict_content_sha256": complete_payload[
            "state_dict_content_sha256"
        ],
        "restricted_inference_checkpoint_sha256": complete.external_hashes[
            "restricted_inference_checkpoint_sha256"
        ],
    }
    run_complete_identity["identity_sha256"] = _identity_digest(
        run_complete_identity
    )
    score_identity = {
        "role": metadata.role,
        "outer_fold_id": manifest_payload["outer_fold_id"],
        "outer_target": manifest_payload["outer_target"],
        "source_domain": manifest_payload["source_domain"],
        "base_seed": manifest_payload["base_seed"],
        "derived_seed": manifest_payload["derived_seed"],
        "detector_role": manifest_payload["detector_role"],
        "oof_fold_index": manifest_payload["oof_fold_index"],
        "input_hw": manifest_payload["input_hw"],
        "resize_mode": manifest_payload["resize_mode"],
    }
    for manifest_key, run_key in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target", "outer_target_domain"),
        ("base_seed", "base_seed"),
        ("derived_seed", "derived_seed"),
        ("detector_role", "detector_role"),
        ("oof_fold_index", "oof_fold_index"),
    ):
        if score_identity[manifest_key] != run_identity[run_key]:
            raise Stage2RC5ScoreBundleV2Error(
                f"score/RUN_COMPLETE identity mismatch: {manifest_key}"
            )

    payload = {
        "schema_version": ATTESTATION_SCHEMA_V2,
        "artifact_type": ATTESTATION_ARTIFACT_TYPE,
        "artifact_status": ATTESTATION_STATUS,
        "development_only": True,
        "official_test_accessed": False,
        "labels_embedded": False,
        "observed_results": None,
        "member_content_verified": False,
        "score_manifest": {
            "path": _repo_relative(metadata.path, root, "score manifest"),
            "sha256": metadata.manifest_sha256,
            "records_content_sha256": metadata.records_content_sha256,
            "records_content_sha256_algorithm": manifest_payload[
                "records_content_sha256_algorithm"
            ],
            "record_count": len(metadata.records),
            "metadata_capability_schema": METADATA_V5_SCHEMA,
        },
        "run_complete": {
            "path": _repo_relative(
                Path(complete.artifact_path), root, "RUN_COMPLETE-v2 artifact"
            ),
            "sha256": complete.sha256,
            "identity": run_complete_identity,
        },
        "restricted_checkpoint": {
            "path": metadata.bindings["checkpoint"]["path"],
            "sha256": metadata.bindings["checkpoint"]["sha256"],
            "authority": "restricted_inference_checkpoint_only",
            "weights_last_authority": False,
        },
        "run_contract": {
            "path": metadata.bindings["run_contract"]["path"],
            "sha256": metadata.bindings["run_contract"]["sha256"],
            "run_id": run_identity["run_id"],
        },
        "selection_contract": {
            "path": metadata.bindings["selection_contract"]["path"],
            "sha256": metadata.bindings["selection_contract"]["sha256"],
            "role": metadata.role,
            "source_domain": manifest_payload["source_domain"],
        },
        "score_identity": score_identity,
        "causal_edge": (
            "RUN_COMPLETE-v2->restricted-checkpoint->v4-score-manifest->"
            "metadata-v5->RC5-score-attestation-v2"
        ),
        "capability_contract": dict(CAPABILITY_CONTRACT),
        "commit_last_marker": ATTESTATION_SIDECAR_NAME,
    }
    return payload, metadata, complete


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


def publish_stage2_rc5_score_attestation_v2(
    score_manifest_metadata: VerifiedStage2ScoreManifestMetadataV5,
    run_complete: VerifiedStage2DetectorRunCompleteV2,
) -> VerifiedStage2RC5ScoreBundleV2:
    """Publish an attestation only after replaying both upstream capabilities."""

    payload, metadata, complete = _expected_attestation(
        score_manifest_metadata, run_complete
    )
    directory = metadata.path.parent
    attestation_path = directory / ATTESTATION_NAME
    sidecar_path = directory / ATTESTATION_SIDECAR_NAME
    _reject_symlink_components(attestation_path, "RC5 score attestation")
    _reject_symlink_components(sidecar_path, "RC5 score attestation commit")
    canonical = _canonical_json(payload)
    digest = hashlib.sha256(canonical).hexdigest()
    sidecar_bytes = f"{digest}  {ATTESTATION_NAME}\n".encode("utf-8")

    if sidecar_path.exists() and not attestation_path.exists():
        raise Stage2RC5ScoreBundleV2Error(
            "orphan RC5 score-attestation commit exists"
        )
    if attestation_path.exists():
        existing, _ = _stable_bytes(attestation_path, "existing RC5 score attestation")
        if existing != canonical:
            raise Stage2RC5ScoreBundleV2Error(
                "immutable RC5 score-attestation overwrite refused"
            )
    else:
        _atomic_write(attestation_path, canonical)

    # Replay current state immediately before the only authoritative commit.
    precommit_payload, _, _ = _expected_attestation(metadata, complete)
    existing, _ = _stable_bytes(attestation_path, "precommit RC5 score attestation")
    if precommit_payload != payload or existing != canonical:
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score-attestation inputs changed before commit"
        )

    created_sidecar = False
    if sidecar_path.exists():
        existing_sidecar, _ = _stable_bytes(
            sidecar_path, "existing RC5 score-attestation commit"
        )
        if existing_sidecar != sidecar_bytes:
            raise Stage2RC5ScoreBundleV2Error(
                "immutable RC5 score-attestation commit overwrite refused"
            )
    else:
        _atomic_write(sidecar_path, sidecar_bytes)
        created_sidecar = True
    try:
        return verify_stage2_rc5_score_bundle_v2(
            attestation_path,
            digest,
            run_complete=complete,
            repository_root=metadata.repository_root,
        )
    except BaseException:
        # A verifier failure after our commit write must not leave an
        # authoritative marker.  Remove only the exact sidecar we created.
        if created_sidecar and sidecar_path.exists() and not sidecar_path.is_symlink():
            try:
                current, _ = _stable_bytes(
                    sidecar_path, "failed RC5 score-attestation commit"
                )
                if current == sidecar_bytes:
                    sidecar_path.unlink()
            except (OSError, RuntimeError, Stage2RC5ScoreBundleV2Error):
                pass
        raise


def verify_stage2_rc5_score_bundle_v2(
    attestation_path: str | Path,
    expected_sha256: str,
    *,
    run_complete: VerifiedStage2DetectorRunCompleteV2,
    repository_root: str | Path,
) -> VerifiedStage2RC5ScoreBundleV2:
    """Replay RUN_COMPLETE and metadata-v5 before issuing the RC5 bundle."""

    root = Path(repository_root).expanduser()
    _reject_symlink_components(root, "repository_root")
    if not root.exists() or not stat.S_ISDIR(root.lstat().st_mode):
        raise FileNotFoundError("repository_root is not a real directory")
    root = root.resolve(strict=True)
    path = _regular_file(Path(attestation_path).expanduser(), "RC5 score attestation")
    if path.name != ATTESTATION_NAME or path.parent.parent == path.parent:
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score attestation has no canonical export-directory location"
        )
    sidecar = _regular_file(
        path.with_name(ATTESTATION_SIDECAR_NAME),
        "RC5 score-attestation commit-last sidecar",
    )
    expected = _sha256(expected_sha256, "RC5 score-attestation SHA-256")
    raw, actual = _stable_bytes(path, "RC5 score attestation")
    if actual != expected:
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score-attestation external SHA-256 mismatch"
        )
    sidecar_raw, _ = _stable_bytes(
        sidecar, "RC5 score-attestation commit-last sidecar"
    )
    if sidecar_raw != f"{actual}  {ATTESTATION_NAME}\n".encode("utf-8"):
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score-attestation commit-last sidecar mismatch"
        )
    payload, parsed_raw, parsed_sha = _json_object(path, "RC5 score attestation")
    if parsed_raw != raw or parsed_sha != actual or raw != _canonical_json(payload):
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score attestation is not canonical JSON"
        )
    if set(payload) != {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "labels_embedded",
        "observed_results",
        "member_content_verified",
        "score_manifest",
        "run_complete",
        "restricted_checkpoint",
        "run_contract",
        "selection_contract",
        "score_identity",
        "causal_edge",
        "capability_contract",
        "commit_last_marker",
    }:
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score-attestation key set mismatch"
        )
    if (
        payload["schema_version"] != ATTESTATION_SCHEMA_V2
        or payload["artifact_type"] != ATTESTATION_ARTIFACT_TYPE
        or payload["artifact_status"] != ATTESTATION_STATUS
        or payload["development_only"] is not True
        or payload["official_test_accessed"] is not False
        or payload["labels_embedded"] is not False
        or payload["observed_results"] is not None
        or payload["member_content_verified"] is not False
        or payload["commit_last_marker"] != ATTESTATION_SIDECAR_NAME
        or payload["capability_contract"] != dict(CAPABILITY_CONTRACT)
    ):
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score-attestation top-level contract mismatch"
        )
    manifest_binding = payload.get("score_manifest")
    if not isinstance(manifest_binding, Mapping):
        raise TypeError("score_manifest attestation binding must be an object")
    selection_binding = payload.get("selection_contract")
    if not isinstance(selection_binding, Mapping):
        raise TypeError("selection_contract attestation binding must be an object")
    required_role = selection_binding.get("role")
    if not isinstance(required_role, str):
        raise TypeError("selection_contract attestation role must be a string")
    manifest_path = _resolve_repo_file(
        root, manifest_binding.get("path"), "attested score manifest"
    )
    metadata = verify_stage2_score_manifest_metadata_v5(
        manifest_path,
        _sha256(manifest_binding.get("sha256"), "score manifest SHA-256"),
        required_role,
        repository_root=root,
    )
    expected_payload, replayed_metadata, replayed_complete = _expected_attestation(
        metadata,
        assert_verified_stage2_detector_run_complete_v2(run_complete),
    )
    if path.parent != replayed_metadata.path.parent or payload != expected_payload:
        raise Stage2RC5ScoreBundleV2Error(
            "RC5 score attestation differs from current-state replay"
        )
    return VerifiedStage2RC5ScoreBundleV2(
        attestation_path=path,
        attestation_sha256=actual,
        attestation=payload,
        score_manifest_metadata=replayed_metadata,
        run_complete=replayed_complete,
        _capability=_CAPABILITY_TOKEN,
    )


def replay_verified_stage2_rc5_score_bundle_v2(
    value: Any,
) -> VerifiedStage2RC5ScoreBundleV2:
    """Revalidate a stored capability at a later producer boundary."""

    bundle = assert_verified_stage2_rc5_score_bundle_v2(value)
    return verify_stage2_rc5_score_bundle_v2(
        bundle.attestation_path,
        bundle.attestation_sha256,
        run_complete=bundle.run_complete,
        repository_root=bundle.score_manifest_metadata.repository_root,
    )


__all__ = [
    "ATTESTATION_ARTIFACT_TYPE",
    "ATTESTATION_NAME",
    "ATTESTATION_SCHEMA_V2",
    "ATTESTATION_SIDECAR_NAME",
    "ATTESTATION_STATUS",
    "BUNDLE_CAPABILITY_SCHEMA",
    "CAPABILITY_CONTRACT",
    "Stage2RC5ScoreBundleV2Error",
    "VerifiedStage2RC5ScoreBundleV2",
    "assert_verified_stage2_rc5_score_bundle_v2",
    "publish_stage2_rc5_score_attestation_v2",
    "replay_verified_stage2_rc5_score_bundle_v2",
    "verify_stage2_rc5_score_bundle_v2",
]
