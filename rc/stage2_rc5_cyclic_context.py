"""Commit-last, source-only cyclic RC5 context collections.

This module is the production authority for the training/selection C14/Q28
geometry.  It intentionally does not accept ``VerifiedStage2VariableQueryWindow``:
the mandatory variable-Q geometry remains the deployment/sanity authority and
cannot represent the N overlapping cyclic starts.

The builder fresh-replays a score-bundle v2 and source-reference v3, reads
only the fourteen context score/image members for each deterministic start,
and publishes one float32[93] feature plus one exact T4 anchor per start.  The
public verifier recomputes every feature and anchor before issuing a capability.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any

import numpy as np

from data_ext.stage2_rc5_score_bundle_v2 import (
    VerifiedStage2RC5ScoreBundleV2,
    assert_verified_stage2_rc5_score_bundle_v2,
    replay_verified_stage2_rc5_score_bundle_v2,
)
from model.endpoint_aware_threshold import (
    EndpointAwareThresholdError,
    decode_coordinate_numpy,
)
from rc.build_stage2_rc5_context import (
    _read_context_grayscale,
    _read_context_score_probability,
)
from rc.domain_statistics import FEATURE_DIM, extract_unlabeled_statistics
from rc.schema import StatisticsConfig
from rc.stage2_context_tail_anchor import (
    build_context_tail_anchor,
    verify_context_tail_anchor,
)
from rc.stage2_crossfit_schema import verify_stage2_statistics_config
from rc.stage2_crossfit_schema_v6 import (
    OOF_HOLDOUT_STAGE2_FIT,
    ROLE_TO_EPISODE,
    SOURCE_DIAGNOSTIC_VALIDATION,
    full_identity_sha256,
)
from rc.stage2_cyclic_training_geometry import (
    CONTEXT_SIZE,
    QUERY_SIZE,
    build_stage2_cyclic_training_geometry,
)
from rc.stage2_domain_balanced_cyclic_sampler import OUTER_TARGETS
from rc.stage2_rc5_source_reference_v3 import (
    VerifiedStage2RC5SourceReferenceV3,
    assert_verified_stage2_rc5_source_reference_v3,
    replay_verified_stage2_rc5_source_reference_v3,
)


COLLECTION_SCHEMA = "rc-irstd.stage2-rc5-cyclic-context-collection.v1"
EPISODE_SCHEMA = "rc-irstd.stage2-rc5-cyclic-context-episode.v1"
COMMIT_SCHEMA = "rc-irstd.stage2-rc5-cyclic-context-collection-commit.v1"
CAPABILITY_SCHEMA = "rc-irstd.stage2-rc5-cyclic-context-capability.v1"
MANIFEST_FILENAME = "cyclic_context_manifest.json"
EPISODES_FILENAME = "cyclic_context_episodes.jsonl"
FEATURES_FILENAME = "cyclic_context_features.npy"
ANCHORS_FILENAME = "cyclic_context_anchors.npy"
COMMIT_FILENAME = "CYCLIC_CONTEXT_COMMIT.json"
PUBLICATION_ORDER = (
    "features_then_anchors_then_identity_episodes_then_manifest_then_commit_last"
)
SUPPORTED_ROLES = frozenset(
    {OOF_HOLDOUT_STAGE2_FIT, SOURCE_DIAGNOSTIC_VALIDATION}
)
FOUR_BOUNDARY_FIELDS = (
    "canonical_id",
    "original_image_sha256",
    "near_duplicate_cluster_id_or_unique_sentinel",
    "exclusion_group_id",
)

_SHA256_HEX = frozenset("0123456789abcdef")
_CAPABILITY = object()


class Stage2RC5CyclicContextError(ValueError):
    """A cyclic-context causal edge or persistent artifact failed closed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    try:
        data = json.dumps(
            _plain(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2RC5CyclicContextError(
            f"non-canonical JSON value: {error}"
        ) from error
    return data + (b"\n" if newline else b"")


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise Stage2RC5CyclicContextError(f"{name} must be lowercase SHA-256")
    return value


def _strict_json(data: bytes, name: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise Stage2RC5CyclicContextError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise Stage2RC5CyclicContextError(
            f"{name} contains non-finite JSON number {value}"
        )

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5CyclicContextError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict) or _canonical(value, newline=True) != data:
        raise Stage2RC5CyclicContextError(f"{name} is not canonical JSON")
    return value


def _repo_relative(path: Path, root: Path, name: str) -> str:
    try:
        relative = path.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise Stage2RC5CyclicContextError(
            f"{name} must be an existing file inside repository_root"
        ) from error
    return relative.as_posix()


def _reject_symlink_components(path: Path, root: Path, name: str) -> None:
    try:
        relative = Path(os.path.abspath(os.fspath(path))).relative_to(root)
    except ValueError as error:
        raise Stage2RC5CyclicContextError(
            f"{name} escapes repository_root"
        ) from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if stat.S_ISLNK(current.stat(follow_symlinks=False).st_mode):
                raise Stage2RC5CyclicContextError(
                    f"{name} contains a symlink component"
                )


def _stable_bytes(path: Path, root: Path, name: str) -> bytes:
    _reject_symlink_components(path, root, name)
    if not path.exists() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise Stage2RC5CyclicContextError(f"{name} is not a direct regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    identity = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_size,
        row.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(current):
        raise RuntimeError(f"{name} changed during stable read")
    return b"".join(chunks)


@dataclass(frozen=True)
class _Authority:
    root: Path
    score_bundle: VerifiedStage2RC5ScoreBundleV2
    source_reference: VerifiedStage2RC5SourceReferenceV3
    statistics_config: StatisticsConfig
    statistics_config_path: Path
    statistics_config_sha256: str
    geometry: Mapping[str, Any]
    input_bindings: Mapping[str, Any]


def _prepare_authority(
    *,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    source_reference: VerifiedStage2RC5SourceReferenceV3,
    statistics_config: StatisticsConfig,
    statistics_config_path: str | Path,
    statistics_config_sha256: str,
    repository_root: str | Path | None,
) -> _Authority:
    score = replay_verified_stage2_rc5_score_bundle_v2(
        assert_verified_stage2_rc5_score_bundle_v2(score_bundle)
    )
    source = replay_verified_stage2_rc5_source_reference_v3(
        assert_verified_stage2_rc5_source_reference_v3(source_reference)
    )
    metadata = score.score_manifest_metadata
    raw_root = metadata.repository_root if repository_root is None else Path(repository_root)
    try:
        root = raw_root.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2RC5CyclicContextError("repository_root does not exist") from error
    if not root.is_dir() or metadata.repository_root != root or source.repository_root != root:
        raise Stage2RC5CyclicContextError(
            "score/source-reference do not share repository_root"
        )
    role = metadata.role
    payload = metadata.payload
    if role not in SUPPORTED_ROLES:
        raise Stage2RC5CyclicContextError(
            "cyclic contexts accept only source OOF-fit or source validation roles"
        )
    outer_fold = str(payload["outer_fold_id"])
    if outer_fold not in OUTER_TARGETS or payload["outer_target"] != OUTER_TARGETS[outer_fold]:
        raise Stage2RC5CyclicContextError("score outer fold/target mismatch")
    if payload["source_domain"] == payload["outer_target"]:
        raise Stage2RC5CyclicContextError("cyclic contexts are source-only")
    if len(metadata.records) < CONTEXT_SIZE + QUERY_SIZE:
        raise Stage2RC5CyclicContextError("cyclic role requires at least 42 records")
    for index, (record, item) in enumerate(zip(metadata.records, metadata.items, strict=True)):
        if record["record_index"] != index or item.record_index != index:
            raise Stage2RC5CyclicContextError("score records are not exact manifest order")
        if item.record != record:
            raise Stage2RC5CyclicContextError("score item/record capability mismatch")
    for field in ("canonical_id", "original_image_sha256"):
        values = [str(record[field]) for record in metadata.records]
        if len(values) != len(set(values)):
            raise Stage2RC5CyclicContextError(f"score role repeats {field}")

    source_rows = source.attestation["source_score_bundles"]
    if len(source_rows) != 2:
        raise Stage2RC5CyclicContextError(
            "source-reference v3 must bind exactly two source score bundles"
        )
    shared_complete = _plain(source_rows[0]["run_complete"])
    if any(_plain(row["run_complete"]) != shared_complete for row in source_rows[1:]):
        raise Stage2RC5CyclicContextError(
            "source-reference score bundles do not share RUN_COMPLETE"
        )
    query_complete = {
        "path": score.attestation["run_complete"]["path"],
        "sha256": score.run_complete.sha256,
        "identity_sha256": score.attestation["run_complete"]["identity"][
            "identity_sha256"
        ],
    }
    if query_complete != shared_complete:
        raise Stage2RC5CyclicContextError(
            "cyclic query and source-reference do not share RUN_COMPLETE"
        )
    detector = source.detector_identity
    for detector_field, score_field in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target", "outer_target"),
        ("base_seed", "base_seed"),
        ("derived_seed", "derived_seed"),
        ("detector_role", "detector_role"),
        ("oof_fold_index", "oof_fold_index"),
    ):
        if detector[detector_field] != payload[score_field]:
            raise Stage2RC5CyclicContextError(
                f"source-reference detector {detector_field} mismatch"
            )
    if dict(source.checkpoint_binding) != dict(metadata.bindings["checkpoint"]):
        raise Stage2RC5CyclicContextError(
            "source-reference/query checkpoint binding mismatch"
        )

    config_sha = _sha(statistics_config_sha256, "statistics_config_sha256")
    config_path = Path(statistics_config_path).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    verified_config = verify_stage2_statistics_config(
        config_path, config_sha, repository_root=root
    )
    if not isinstance(statistics_config, StatisticsConfig) or verified_config != statistics_config:
        raise Stage2RC5CyclicContextError(
            "statistics-config capability/value mismatch"
        )
    if source.statistics_config != verified_config:
        raise Stage2RC5CyclicContextError(
            "source-reference statistics config mismatch"
        )
    expected_statistics = {
        "path": _repo_relative(config_path, root, "statistics config"),
        "sha256": config_sha,
    }
    if dict(source.stage2_contract["bindings"]["statistics_config"]) != expected_statistics:
        raise Stage2RC5CyclicContextError(
            "source-reference statistics binding mismatch"
        )

    geometry = build_stage2_cyclic_training_geometry(len(metadata.records))
    source_identity = source.attestation["attestation_identity_sha256"]
    bindings = {
        "score_bundle": {
            "path": _repo_relative(score.attestation_path, root, "score attestation"),
            "sha256": score.attestation_sha256,
            "score_manifest_sha256": metadata.manifest_sha256,
            "score_records_content_sha256": metadata.records_content_sha256,
            "role": role,
            "run_complete_artifact_sha256": score.run_complete.sha256,
            "run_complete_identity_sha256": query_complete["identity_sha256"],
            "restricted_checkpoint_sha256": score.attestation[
                "restricted_checkpoint"
            ]["sha256"],
        },
        "source_reference_v3": {
            "path": _repo_relative(
                source.attestation_path, root, "source-reference v3 attestation"
            ),
            "sha256": source.attestation_sha256,
            "identity_sha256": source_identity,
            "source_score_attestation_sha256": sorted(
                str(row["score_attestation"]["sha256"]) for row in source_rows
            ),
        },
        "statistics_config": expected_statistics,
        "seed_manifest": {
            "path": metadata.bindings["seed_manifest"]["path"],
            "sha256": metadata.bindings["seed_manifest"]["sha256"],
        },
    }
    return _Authority(
        root=root,
        score_bundle=score,
        source_reference=source,
        statistics_config=verified_config,
        statistics_config_path=config_path.resolve(strict=True),
        statistics_config_sha256=config_sha,
        geometry=MappingProxyType(geometry),
        input_bindings=MappingProxyType(bindings),
    )


def _identity_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_id": str(record["canonical_id"]),
        "image_id": str(record["image_id"]),
        "original_image_sha256": str(record["original_image_sha256"]),
        "exclusion_group_id": str(record["exclusion_group_id"]),
        "near_duplicate_cluster_id_or_unique_sentinel": str(
            record["near_duplicate_cluster_id_or_unique_sentinel"]
        ),
        "source_role_record_index": int(record["source_role_record_index"]),
    }


def _materialize_all_starts(
    authority: _Authority,
) -> tuple[np.ndarray, np.ndarray, tuple[Mapping[str, Any], ...]]:
    metadata = authority.score_bundle.score_manifest_metadata
    records = metadata.records
    items = metadata.items
    count = len(records)
    geometry = authority.geometry

    def load(index: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            _read_context_score_probability(items[index], authority.root),
            _read_context_grayscale(items[index], authority.root),
        )

    prefix_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    live: deque[tuple[int, np.ndarray, np.ndarray]] = deque()
    for index in range(CONTEXT_SIZE):
        probability, grayscale = load(index)
        live.append((index, probability, grayscale))
        if index < CONTEXT_SIZE - 1:
            prefix_cache[index] = (probability, grayscale)

    features = np.empty((count, FEATURE_DIM), dtype=np.float32)
    anchors = np.empty((count, 3), dtype=np.float64)
    episodes: list[Mapping[str, Any]] = []
    for start, geometry_row in enumerate(geometry["episodes"]):
        context_indices = tuple(int(item) for item in geometry_row["context_indices"])
        query_indices = tuple(int(item) for item in geometry_row["query_indices"])
        if tuple(item[0] for item in live) != context_indices:
            raise RuntimeError("cyclic sliding context drifted from geometry replay")
        context_records = tuple(_identity_projection(records[index]) for index in context_indices)
        query_records = tuple(_identity_projection(records[index]) for index in query_indices)
        for field in FOUR_BOUNDARY_FIELDS:
            if {row[field] for row in context_records}.intersection(
                row[field] for row in query_records
            ):
                raise Stage2RC5CyclicContextError(
                    f"cyclic start {start} overlaps at identity boundary {field}"
                )
        probabilities = tuple(item[1] for item in live)
        grayscales = tuple(item[2] for item in live)
        statistics = extract_unlabeled_statistics(
            probabilities,
            grayscales,
            source_reference=authority.source_reference.source_reference,
            statistics_config=authority.statistics_config,
        )
        feature = np.asarray(statistics.vector, dtype=np.float32)
        if feature.shape != (FEATURE_DIM,) or not np.isfinite(feature).all():
            raise Stage2RC5CyclicContextError(
                f"cyclic start {start} emitted an invalid 93D context"
            )
        context_identity = full_identity_sha256(context_records)
        query_identity = full_identity_sha256(query_records)
        anchor_payload = build_context_tail_anchor(
            context_probability_maps=probabilities,
            context_identity_sha256=context_identity,
        )
        anchor = verify_context_tail_anchor(
            anchor_payload,
            context_probability_maps=probabilities,
            expected_context_identity_sha256=context_identity,
        )
        anchor_row = np.asarray(anchor.coordinates, dtype=np.float64)
        if anchor_row.shape != (3,) or not np.isfinite(anchor_row).all():
            raise Stage2RC5CyclicContextError(
                f"cyclic start {start} emitted an invalid T4 anchor"
            )
        features[start] = feature
        anchors[start] = anchor_row
        episode = {
            "schema_version": EPISODE_SCHEMA,
            "cyclic_start": start,
            "window_id": (
                f"cyclic::{metadata.payload['outer_fold_id']}::"
                f"{metadata.payload['source_domain']}::{metadata.role}::{start:06d}"
            ),
            "context_indices": list(context_indices),
            "query_indices": list(query_indices),
            "ordered_context_original_image_sha256": [
                str(records[index]["original_image_sha256"])
                for index in context_indices
            ],
            "ordered_query_original_image_sha256": [
                str(records[index]["original_image_sha256"])
                for index in query_indices
            ],
            "context_full_identity_sha256": context_identity,
            "query_full_identity_sha256": query_identity,
            "context_feature_vector_sha256": hashlib.sha256(
                feature.astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest(),
            "anchor_identity_sha256": anchor.payload["anchor_identity_sha256"],
            "anchor_coordinates_sha256": hashlib.sha256(
                anchor_row.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest(),
            "context_score_member_count": CONTEXT_SIZE,
            "context_image_member_count": CONTEXT_SIZE,
            "query_score_member_count": 0,
            "query_image_member_count": 0,
        }
        episode["episode_identity_sha256"] = _json_sha(episode)
        episodes.append(MappingProxyType(episode))

        if start + 1 < count:
            live.popleft()
            incoming = (start + CONTEXT_SIZE) % count
            if incoming in prefix_cache:
                probability, grayscale = prefix_cache[incoming]
            else:
                probability, grayscale = load(incoming)
            live.append((incoming, probability, grayscale))

    try:
        decode_coordinate_numpy(anchors)
    except EndpointAwareThresholdError as error:
        raise Stage2RC5CyclicContextError(
            "cyclic T4 anchors are not canonical EATC-v2"
        ) from error
    if np.any(anchors[:, 1:] < anchors[:, :-1]):
        raise Stage2RC5CyclicContextError("cyclic T4 anchors decreased")
    features.setflags(write=False)
    anchors.setflags(write=False)
    return features, anchors, tuple(episodes)


def _member_binding(path: Path, data: bytes, *, dtype: str | None = None,
                    shape: Sequence[int] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    if dtype is not None:
        row["dtype"] = dtype
    if shape is not None:
        row["shape"] = [int(item) for item in shape]
    return row


def _manifest_payload(
    authority: _Authority,
    *,
    features_binding: Mapping[str, Any],
    anchors_binding: Mapping[str, Any],
    episodes_binding: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = authority.score_bundle.score_manifest_metadata
    records = metadata.records
    boundaries = {
        field: sorted({str(record[field]) for record in records})
        for field in FOUR_BOUNDARY_FIELDS
    }
    payload: dict[str, Any] = {
        "schema_version": COLLECTION_SCHEMA,
        "artifact_type": "rc_irstd_stage2_rc5_cyclic_context_collection",
        "artifact_status": "DEVELOPMENT_ONLY_PRELABEL_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "observed_results": None,
        "artifact_scope": "production",
        "geometry_authority": "source_cyclic_c14_q28_all_n_starts",
        "variable_query_window_accepted": False,
        "deployment_inference_authority": False,
        "atomic_decision_authority": False,
        "outer_fold_id": metadata.payload["outer_fold_id"],
        "outer_target": metadata.payload["outer_target"],
        "source_domain": metadata.payload["source_domain"],
        "score_role": metadata.role,
        "episode_role": ROLE_TO_EPISODE[metadata.role],
        "oof_fold_index": metadata.payload["oof_fold_index"],
        "record_count": len(records),
        "cyclic_start_count": len(records),
        "context_size": CONTEXT_SIZE,
        "query_size": QUERY_SIZE,
        "cyclic_geometry_sha256": _json_sha(authority.geometry),
        "ordered_original_image_sha256": [
            str(record["original_image_sha256"]) for record in records
        ],
        "identity_boundary_values": boundaries,
        "input_bindings": _plain(authority.input_bindings),
        "members": {
            "context_features": dict(features_binding),
            "anchor_coordinates": dict(anchors_binding),
            "episodes": dict(episodes_binding),
        },
        "access_audit": {
            "unique_context_score_members_opened": len(records),
            "unique_context_image_members_opened": len(records),
            "query_score_members_opened": 0,
            "query_image_members_opened": 0,
            "context_labels_accessed": False,
            "query_labels_accessed": False,
            "observed_results_accessed": False,
            "sliding_live_map_bound": 2 * CONTEXT_SIZE - 1,
        },
        "publication_order": PUBLICATION_ORDER,
        "manifest_identity_sha256": "",
    }
    projection = dict(payload)
    projection.pop("manifest_identity_sha256")
    payload["manifest_identity_sha256"] = _json_sha(projection)
    return payload


def _commit_payload(
    *, manifest_sha256: str, manifest_identity_sha256: str,
    authority: _Authority,
) -> dict[str, Any]:
    return {
        "schema_version": COMMIT_SCHEMA,
        "artifact_type": "rc_irstd_stage2_rc5_cyclic_context_commit",
        "artifact_status": "COMMITTED",
        "publication_order": PUBLICATION_ORDER,
        "manifest": {"path": MANIFEST_FILENAME, "sha256": manifest_sha256},
        "collection_identity_sha256": manifest_identity_sha256,
        "score_attestation_sha256": authority.score_bundle.attestation_sha256,
        "source_reference_v3_attestation_sha256": (
            authority.source_reference.attestation_sha256
        ),
        "statistics_config_sha256": authority.statistics_config_sha256,
        "official_test_accessed": False,
    }


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5CyclicContextCollection:
    path: Path
    commit_sha256: str
    manifest: Mapping[str, Any]
    score_bundle: VerifiedStage2RC5ScoreBundleV2
    source_reference: VerifiedStage2RC5SourceReferenceV3
    statistics_config: StatisticsConfig
    statistics_config_path: Path
    statistics_config_sha256: str
    context_features: np.ndarray
    anchor_coordinates: np.ndarray
    episodes: tuple[Mapping[str, Any], ...]
    boundary_values: Mapping[str, frozenset[str]]
    capability_schema: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("cyclic context collections are public-verifier-issued only")


def assert_verified_stage2_rc5_cyclic_context_collection(
    value: object,
) -> VerifiedStage2RC5CyclicContextCollection:
    if (
        type(value) is not VerifiedStage2RC5CyclicContextCollection
        or getattr(value, "_capability", None) is not _CAPABILITY
        or value.capability_schema != CAPABILITY_SCHEMA
    ):
        raise TypeError(
            "a verifier-issued Stage2 RC5 cyclic context collection is required"
        )
    assert_verified_stage2_rc5_score_bundle_v2(value.score_bundle)
    assert_verified_stage2_rc5_source_reference_v3(value.source_reference)
    if value.context_features.dtype != np.float32 or value.context_features.ndim != 2 \
            or value.context_features.shape[1] != FEATURE_DIM:
        raise TypeError("cyclic context capability feature matrix is invalid")
    if value.anchor_coordinates.dtype != np.float64 or \
            value.anchor_coordinates.shape != (value.context_features.shape[0], 3):
        raise TypeError("cyclic context capability anchor matrix is invalid")
    return value


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _output_directory(value: str | Path, root: Path) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise Stage2RC5CyclicContextError(
            "output_directory escapes repository_root"
        ) from error
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError("immutable cyclic context output already exists")
    # Reject every already-existing lexical prefix before mkdir can follow a
    # symlink and create anything outside the repository.
    _reject_symlink_components(candidate.parent, root, "output parent")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(candidate.parent, root, "output parent")
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
        resolved_parent.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise Stage2RC5CyclicContextError(
            "output parent resolves outside repository_root"
        ) from error
    if not resolved_parent.is_dir():
        raise NotADirectoryError("cyclic context output parent is not a directory")
    os.mkdir(candidate, 0o755)
    return candidate


def build_and_publish_stage2_rc5_cyclic_context_collection(
    *,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    source_reference: VerifiedStage2RC5SourceReferenceV3,
    statistics_config: StatisticsConfig,
    statistics_config_path: str | Path,
    statistics_config_sha256: str,
    output_directory: str | Path,
    repository_root: str | Path | None = None,
) -> VerifiedStage2RC5CyclicContextCollection:
    """Publish all deterministic source-only cyclic starts, commit last."""

    authority = _prepare_authority(
        score_bundle=score_bundle,
        source_reference=source_reference,
        statistics_config=statistics_config,
        statistics_config_path=statistics_config_path,
        statistics_config_sha256=statistics_config_sha256,
        repository_root=repository_root,
    )
    features, anchors, episodes = _materialize_all_starts(authority)
    output = _output_directory(output_directory, authority.root)
    feature_path = output / FEATURES_FILENAME
    anchor_path = output / ANCHORS_FILENAME
    episode_path = output / EPISODES_FILENAME
    manifest_path = output / MANIFEST_FILENAME
    commit_path = output / COMMIT_FILENAME
    feature_bytes = _npy_bytes(features)
    anchor_bytes = _npy_bytes(anchors)
    episode_bytes = b"".join(
        _canonical(row, newline=True) for row in episodes
    )
    features_binding = _member_binding(
        feature_path,
        feature_bytes,
        dtype="float32",
        shape=features.shape,
    )
    anchors_binding = _member_binding(
        anchor_path,
        anchor_bytes,
        dtype="float64",
        shape=anchors.shape,
    )
    episodes_binding = {
        **_member_binding(episode_path, episode_bytes),
        "row_count": len(episodes),
        "payload_policy": "identity_only_no_features_no_score_values_no_labels",
    }
    manifest = _manifest_payload(
        authority,
        features_binding=features_binding,
        anchors_binding=anchors_binding,
        episodes_binding=episodes_binding,
    )
    manifest_bytes = _canonical(manifest, newline=True)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    commit = _commit_payload(
        manifest_sha256=manifest_sha,
        manifest_identity_sha256=manifest["manifest_identity_sha256"],
        authority=authority,
    )
    commit_bytes = _canonical(commit, newline=True)
    commit_sha = hashlib.sha256(commit_bytes).hexdigest()
    wrote_commit = False
    try:
        for path, data in (
            (feature_path, feature_bytes),
            (anchor_path, anchor_bytes),
            (episode_path, episode_bytes),
            (manifest_path, manifest_bytes),
        ):
            _write_exclusive(path, data)

        # Replay all causal authorities immediately before the only marker
        # that can make the collection authoritative.
        precommit = _prepare_authority(
            score_bundle=authority.score_bundle,
            source_reference=authority.source_reference,
            statistics_config=authority.statistics_config,
            statistics_config_path=authority.statistics_config_path,
            statistics_config_sha256=authority.statistics_config_sha256,
            repository_root=authority.root,
        )
        if (
            _plain(precommit.input_bindings) != _plain(authority.input_bindings)
            or _plain(precommit.geometry) != _plain(authority.geometry)
        ):
            raise Stage2RC5CyclicContextError(
                "cyclic context authorities changed before commit"
            )
        _write_exclusive(commit_path, commit_bytes)
        wrote_commit = True
        return verify_stage2_rc5_cyclic_context_collection(
            commit_path,
            commit_sha,
            score_bundle=precommit.score_bundle,
            source_reference=precommit.source_reference,
            statistics_config=precommit.statistics_config,
            statistics_config_path=precommit.statistics_config_path,
            statistics_config_sha256=precommit.statistics_config_sha256,
            repository_root=precommit.root,
        )
    except BaseException:
        if wrote_commit and commit_path.exists() and not commit_path.is_symlink():
            try:
                if _stable_bytes(commit_path, authority.root, "failed commit") == commit_bytes:
                    commit_path.unlink()
            except (OSError, RuntimeError, Stage2RC5CyclicContextError):
                pass
        raise


def _resolved_commit_path(
    value: str | Path, root: Path,
) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise Stage2RC5CyclicContextError("commit path escapes repository_root") from error
    if candidate.name != COMMIT_FILENAME:
        raise Stage2RC5CyclicContextError("cyclic context commit filename mismatch")
    return candidate


def _load_member_array(
    root: Path,
    collection_path: Path,
    binding: Mapping[str, Any],
    *,
    expected_name: str,
    expected_dtype: np.dtype[Any],
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, bytes]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path", "sha256", "dtype", "shape"
    }:
        raise Stage2RC5CyclicContextError(
            f"{expected_name} member binding closure mismatch"
        )
    if binding["path"] != expected_name or binding["dtype"] != str(expected_dtype) \
            or binding["shape"] != list(expected_shape):
        raise Stage2RC5CyclicContextError(
            f"{expected_name} member dtype/shape/path mismatch"
        )
    path = collection_path / expected_name
    data = _stable_bytes(path, root, expected_name)
    if hashlib.sha256(data).hexdigest() != _sha(
        binding["sha256"], f"{expected_name}.sha256"
    ):
        raise Stage2RC5CyclicContextError(f"{expected_name} SHA-256 mismatch")
    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise Stage2RC5CyclicContextError(
            f"invalid {expected_name} NPY: {error}"
        ) from error
    if loaded.dtype != expected_dtype or loaded.shape != expected_shape:
        raise Stage2RC5CyclicContextError(
            f"{expected_name} loaded dtype/shape mismatch"
        )
    value = np.array(loaded, dtype=expected_dtype, order="C", copy=True)
    value.setflags(write=False)
    return value, data


def verify_stage2_rc5_cyclic_context_collection(
    commit_path: str | Path,
    expected_commit_sha256: str,
    *,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    source_reference: VerifiedStage2RC5SourceReferenceV3,
    statistics_config: StatisticsConfig,
    statistics_config_path: str | Path,
    statistics_config_sha256: str,
    repository_root: str | Path | None = None,
) -> VerifiedStage2RC5CyclicContextCollection:
    """Fresh-replay every upstream and every cyclic 93D/T4 material row."""

    authority = _prepare_authority(
        score_bundle=score_bundle,
        source_reference=source_reference,
        statistics_config=statistics_config,
        statistics_config_path=statistics_config_path,
        statistics_config_sha256=statistics_config_sha256,
        repository_root=repository_root,
    )
    expected = _sha(expected_commit_sha256, "expected_commit_sha256")
    commit_file = _resolved_commit_path(commit_path, authority.root)
    commit_bytes = _stable_bytes(commit_file, authority.root, "cyclic context commit")
    if hashlib.sha256(commit_bytes).hexdigest() != expected:
        raise Stage2RC5CyclicContextError("cyclic context external commit SHA mismatch")
    commit = _strict_json(commit_bytes, "cyclic context commit")
    collection_path = commit_file.parent
    expected_files = {
        FEATURES_FILENAME,
        ANCHORS_FILENAME,
        EPISODES_FILENAME,
        MANIFEST_FILENAME,
        COMMIT_FILENAME,
    }
    if {item.name for item in collection_path.iterdir()} != expected_files:
        raise Stage2RC5CyclicContextError("cyclic context file closure mismatch")
    manifest_ref = commit.get("manifest")
    if not isinstance(manifest_ref, Mapping) or set(manifest_ref) != {"path", "sha256"} \
            or manifest_ref["path"] != MANIFEST_FILENAME:
        raise Stage2RC5CyclicContextError("cyclic context manifest binding mismatch")
    manifest_path = collection_path / MANIFEST_FILENAME
    manifest_bytes = _stable_bytes(manifest_path, authority.root, "cyclic context manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != _sha(
        manifest_ref["sha256"], "manifest.sha256"
    ):
        raise Stage2RC5CyclicContextError("cyclic context manifest SHA mismatch")
    manifest = _strict_json(manifest_bytes, "cyclic context manifest")
    count = len(authority.score_bundle.score_manifest_metadata.records)
    if manifest.get("record_count") != count or manifest.get("cyclic_start_count") != count:
        raise Stage2RC5CyclicContextError("cyclic context cardinality mismatch")
    members = manifest.get("members")
    if not isinstance(members, Mapping) or set(members) != {
        "context_features", "anchor_coordinates", "episodes"
    }:
        raise Stage2RC5CyclicContextError("cyclic context member closure mismatch")
    feature_mmap, feature_bytes = _load_member_array(
        authority.root,
        collection_path,
        members["context_features"],
        expected_name=FEATURES_FILENAME,
        expected_dtype=np.dtype("float32"),
        expected_shape=(count, FEATURE_DIM),
    )
    anchor_mmap, anchor_bytes = _load_member_array(
        authority.root,
        collection_path,
        members["anchor_coordinates"],
        expected_name=ANCHORS_FILENAME,
        expected_dtype=np.dtype("float64"),
        expected_shape=(count, 3),
    )
    episode_binding = members["episodes"]
    if not isinstance(episode_binding, Mapping) or set(episode_binding) != {
        "path", "sha256", "row_count", "payload_policy"
    } or episode_binding["path"] != EPISODES_FILENAME or \
            episode_binding["row_count"] != count or episode_binding[
                "payload_policy"
            ] != "identity_only_no_features_no_score_values_no_labels":
        raise Stage2RC5CyclicContextError("cyclic context episode binding mismatch")
    episode_path = collection_path / EPISODES_FILENAME
    episode_bytes = _stable_bytes(episode_path, authority.root, "cyclic context episodes")
    if hashlib.sha256(episode_bytes).hexdigest() != _sha(
        episode_binding["sha256"], "episodes.sha256"
    ):
        raise Stage2RC5CyclicContextError("cyclic context episodes SHA mismatch")
    lines = episode_bytes.splitlines(keepends=True)
    if len(lines) != count:
        raise Stage2RC5CyclicContextError("cyclic context episode row count mismatch")
    episodes = tuple(_strict_json(line, f"episode[{index}]")
                     for index, line in enumerate(lines))

    expected_features, expected_anchors, expected_episodes = _materialize_all_starts(
        authority
    )
    if not np.array_equal(np.asarray(feature_mmap), expected_features):
        raise Stage2RC5CyclicContextError(
            "persisted cyclic contexts differ from full score/image replay"
        )
    if not np.array_equal(np.asarray(anchor_mmap), expected_anchors):
        raise Stage2RC5CyclicContextError(
            "persisted cyclic anchors differ from exact T4 replay"
        )
    if [_plain(row) for row in episodes] != [
        _plain(row) for row in expected_episodes
    ]:
        raise Stage2RC5CyclicContextError(
            "persisted cyclic episode identities differ from geometry replay"
        )

    expected_manifest = _manifest_payload(
        authority,
        features_binding=_member_binding(
            collection_path / FEATURES_FILENAME,
            feature_bytes,
            dtype="float32",
            shape=(count, FEATURE_DIM),
        ),
        anchors_binding=_member_binding(
            collection_path / ANCHORS_FILENAME,
            anchor_bytes,
            dtype="float64",
            shape=(count, 3),
        ),
        episodes_binding={
            **_member_binding(collection_path / EPISODES_FILENAME, episode_bytes),
            "row_count": count,
            "payload_policy": "identity_only_no_features_no_score_values_no_labels",
        },
    )
    if manifest != expected_manifest:
        raise Stage2RC5CyclicContextError(
            "cyclic context manifest differs from current causal replay"
        )
    expected_commit = _commit_payload(
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_identity_sha256=manifest["manifest_identity_sha256"],
        authority=authority,
    )
    if commit != expected_commit:
        raise Stage2RC5CyclicContextError(
            "cyclic context commit differs from current causal replay"
        )
    boundaries = MappingProxyType({
        field: frozenset(str(item) for item in manifest["identity_boundary_values"][field])
        for field in FOUR_BOUNDARY_FIELDS
    })
    value = object.__new__(VerifiedStage2RC5CyclicContextCollection)
    for name, item in {
        "path": collection_path,
        "commit_sha256": expected,
        "manifest": MappingProxyType(manifest),
        "score_bundle": authority.score_bundle,
        "source_reference": authority.source_reference,
        "statistics_config": authority.statistics_config,
        "statistics_config_path": authority.statistics_config_path,
        "statistics_config_sha256": authority.statistics_config_sha256,
        "context_features": feature_mmap,
        "anchor_coordinates": anchor_mmap,
        "episodes": tuple(MappingProxyType(row) for row in episodes),
        "boundary_values": boundaries,
        "capability_schema": CAPABILITY_SCHEMA,
        "_capability": _CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return assert_verified_stage2_rc5_cyclic_context_collection(value)


def replay_verified_stage2_rc5_cyclic_context_collection(
    value: VerifiedStage2RC5CyclicContextCollection,
) -> VerifiedStage2RC5CyclicContextCollection:
    verified = assert_verified_stage2_rc5_cyclic_context_collection(value)
    return verify_stage2_rc5_cyclic_context_collection(
        verified.path / COMMIT_FILENAME,
        verified.commit_sha256,
        score_bundle=verified.score_bundle,
        source_reference=verified.source_reference,
        statistics_config=verified.statistics_config,
        statistics_config_path=verified.statistics_config_path,
        statistics_config_sha256=verified.statistics_config_sha256,
        repository_root=verified.score_bundle.score_manifest_metadata.repository_root,
    )


__all__ = [
    "ANCHORS_FILENAME",
    "CAPABILITY_SCHEMA",
    "COLLECTION_SCHEMA",
    "COMMIT_FILENAME",
    "COMMIT_SCHEMA",
    "EPISODES_FILENAME",
    "EPISODE_SCHEMA",
    "FEATURES_FILENAME",
    "MANIFEST_FILENAME",
    "Stage2RC5CyclicContextError",
    "VerifiedStage2RC5CyclicContextCollection",
    "assert_verified_stage2_rc5_cyclic_context_collection",
    "build_and_publish_stage2_rc5_cyclic_context_collection",
    "replay_verified_stage2_rc5_cyclic_context_collection",
    "verify_stage2_rc5_cyclic_context_collection",
]
