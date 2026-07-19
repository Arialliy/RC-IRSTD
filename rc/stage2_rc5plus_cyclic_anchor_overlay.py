"""Commit-last nine-budget anchor overlay for verified RC5 cyclic contexts.

The frozen RC5 cyclic-context artifact contains 93D features and three T4
anchors.  RC5+ must not interpolate those three anchors.  This overlay binds a
freshly replayed base capability, reopens only the same fourteen unlabelled
context score maps for every cyclic start, and directly recomputes all nine
exact-rational same-budget order statistics with anchor-v2.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.endpoint_aware_threshold import (
    EndpointAwareThresholdError,
    decode_coordinate_numpy,
)
from rc.build_stage2_rc5_context import _read_context_score_probability
from rc.stage2_context_tail_anchor_v2 import (
    build_context_tail_anchor_v2,
    verify_context_tail_anchor_v2,
)
from rc.stage2_cyclic_training_geometry import CONTEXT_SIZE
from rc.stage2_rc5_cyclic_context import (
    COMMIT_FILENAME as BASE_COMMIT_FILENAME,
    _canonical,
    _json_sha,
    _load_member_array,
    _member_binding,
    _npy_bytes,
    _output_directory,
    _resolved_commit_path,
    _sha,
    _stable_bytes,
    _strict_json,
    _write_exclusive,
    assert_verified_stage2_rc5_cyclic_context_collection,
    replay_verified_stage2_rc5_cyclic_context_collection,
)


OVERLAY_SCHEMA = "rc-irstd.stage2-rc5plus-cyclic-anchor-overlay.v1"
ROW_SCHEMA = "rc-irstd.stage2-rc5plus-cyclic-anchor-row.v1"
COMMIT_SCHEMA = "rc-irstd.stage2-rc5plus-cyclic-anchor-overlay-commit.v1"
CAPABILITY_SCHEMA = "rc-irstd.stage2-rc5plus-cyclic-anchor-capability.v1"
ANCHORS_FILENAME = "rc5plus_cyclic_anchor_coordinates.npy"
ROWS_FILENAME = "rc5plus_cyclic_anchor_rows.jsonl"
MANIFEST_FILENAME = "rc5plus_cyclic_anchor_manifest.json"
COMMIT_FILENAME = "RC5PLUS_CYCLIC_ANCHOR_COMMIT.json"
PUBLICATION_ORDER = "anchors_then_identity_rows_then_manifest_then_commit_last"
_CAPABILITY = object()


class Stage2RC5PlusCyclicAnchorOverlayError(ValueError):
    """The base binding, exact anchor replay or immutable overlay failed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _base_binding(base: Any) -> dict[str, str]:
    root = base.score_bundle.score_manifest_metadata.repository_root
    try:
        relative = (base.path / BASE_COMMIT_FILENAME).resolve(strict=True).relative_to(
            root
        )
    except (FileNotFoundError, ValueError) as error:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "base cyclic-context commit is outside repository_root"
        ) from error
    return {
        "path": relative.as_posix(),
        "sha256": base.commit_sha256,
        "collection_identity_sha256": str(
            base.manifest["manifest_identity_sha256"]
        ),
    }


def _materialize(
    base: Any,
) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
    metadata = base.score_bundle.score_manifest_metadata
    items = metadata.items
    count = len(items)
    root = metadata.repository_root

    def load(index: int) -> np.ndarray:
        return _read_context_score_probability(items[index], root)

    prefix_cache: dict[int, np.ndarray] = {}
    live: deque[tuple[int, np.ndarray]] = deque()
    for index in range(CONTEXT_SIZE):
        probability = load(index)
        live.append((index, probability))
        if index < CONTEXT_SIZE - 1:
            prefix_cache[index] = probability

    anchors = np.empty((count, len(BUDGET_KNOT_RATIONALS)), dtype=np.float64)
    rows: list[Mapping[str, Any]] = []
    for start, base_episode in enumerate(base.episodes):
        context_indices = tuple(int(item) for item in base_episode["context_indices"])
        if tuple(item[0] for item in live) != context_indices:
            raise Stage2RC5PlusCyclicAnchorOverlayError(
                "sliding context order differs from verified base geometry"
            )
        probabilities = tuple(item[1] for item in live)
        context_identity = str(base_episode["context_full_identity_sha256"])
        payload = build_context_tail_anchor_v2(
            context_probability_maps=probabilities,
            context_identity_sha256=context_identity,
        )
        verified = verify_context_tail_anchor_v2(
            payload,
            context_probability_maps=probabilities,
            expected_context_identity_sha256=context_identity,
        )
        coordinates = np.asarray(verified.grid_coordinates, dtype=np.float64)
        if coordinates.shape != (len(BUDGET_KNOT_RATIONALS),):
            raise RuntimeError("anchor-v2 emitted a non-nine-point grid")
        anchors[start] = coordinates
        row = {
            "schema_version": ROW_SCHEMA,
            "cyclic_start": start,
            "base_episode_identity_sha256": str(
                base_episode["episode_identity_sha256"]
            ),
            "context_full_identity_sha256": context_identity,
            "anchor_v2_identity_sha256": str(
                verified.payload["anchor_identity_sha256"]
            ),
            "anchor_coordinates_sha256": hashlib.sha256(
                coordinates.astype("<f8", copy=False).tobytes(order="C")
            ).hexdigest(),
            "grid_budget_rationals": [
                [numerator, denominator]
                for numerator, denominator in BUDGET_KNOT_RATIONALS
            ],
            "context_labels_accessed": False,
            "query_scores_accessed": False,
            "query_labels_accessed": False,
            "anchor_interpolation_used": False,
        }
        row["row_identity_sha256"] = _json_sha(row)
        rows.append(MappingProxyType(row))
        if start + 1 < count:
            live.popleft()
            incoming = (start + CONTEXT_SIZE) % count
            probability = prefix_cache.get(incoming)
            if probability is None:
                probability = load(incoming)
            live.append((incoming, probability))

    try:
        decode_coordinate_numpy(anchors)
    except EndpointAwareThresholdError as error:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ cyclic anchors are not canonical EATC-v2"
        ) from error
    if np.any(anchors[:, 1:] < anchors[:, :-1]):
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ cyclic anchors decreased from loose to strict budget"
        )
    anchors.setflags(write=False)
    return anchors, tuple(rows)


def _manifest(
    base: Any,
    *,
    anchor_binding: Mapping[str, Any],
    rows_binding: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = base.score_bundle.score_manifest_metadata
    payload = {
        "schema_version": OVERLAY_SCHEMA,
        "artifact_type": "rc_irstd_stage2_rc5plus_cyclic_anchor_overlay",
        "artifact_status": "DEVELOPMENT_ONLY_PRELABEL_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "observed_results": None,
        "outer_fold_id": metadata.payload["outer_fold_id"],
        "outer_target": metadata.payload["outer_target"],
        "source_domain": metadata.payload["source_domain"],
        "score_role": metadata.role,
        "oof_fold_index": metadata.payload["oof_fold_index"],
        "cyclic_start_count": len(base.episodes),
        "context_size": CONTEXT_SIZE,
        "base_cyclic_context": _base_binding(base),
        "grid_budget_rationals": [
            [numerator, denominator]
            for numerator, denominator in BUDGET_KNOT_RATIONALS
        ],
        "anchor_source": (
            "direct_same_budget_unlabelled_context_order_statistic_v2"
        ),
        "anchor_interpolation_used": False,
        "members": {
            "anchor_coordinates": dict(anchor_binding),
            "identity_rows": dict(rows_binding),
        },
        "access_audit": {
            "unique_context_score_members_opened": len(base.episodes),
            "query_score_members_opened": 0,
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


def _commit(manifest_bytes: bytes, manifest: Mapping[str, Any], base: Any) -> dict[str, Any]:
    return {
        "schema_version": COMMIT_SCHEMA,
        "artifact_type": "rc_irstd_stage2_rc5plus_cyclic_anchor_commit",
        "artifact_status": "COMMITTED",
        "publication_order": PUBLICATION_ORDER,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "overlay_identity_sha256": str(manifest["manifest_identity_sha256"]),
        "base_cyclic_context_commit_sha256": base.commit_sha256,
        "official_test_accessed": False,
    }


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5PlusCyclicAnchorOverlay:
    path: Path
    commit_sha256: str
    manifest: Mapping[str, Any]
    base_collection: Any
    anchor_coordinates: np.ndarray
    rows: tuple[Mapping[str, Any], ...]
    capability_schema: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("RC5+ cyclic anchor overlays are verifier-issued only")


def assert_verified_stage2_rc5plus_cyclic_anchor_overlay(
    value: object,
) -> VerifiedStage2RC5PlusCyclicAnchorOverlay:
    if (
        type(value) is not VerifiedStage2RC5PlusCyclicAnchorOverlay
        or getattr(value, "_capability", None) is not _CAPABILITY
        or value.capability_schema != CAPABILITY_SCHEMA
    ):
        raise TypeError("a verifier-issued RC5+ cyclic anchor overlay is required")
    assert_verified_stage2_rc5_cyclic_context_collection(value.base_collection)
    expected = (
        len(value.base_collection.episodes),
        len(BUDGET_KNOT_RATIONALS),
    )
    if value.anchor_coordinates.dtype != np.float64 or value.anchor_coordinates.shape != expected:
        raise TypeError("RC5+ cyclic anchor capability matrix is invalid")
    return value


def build_and_publish_stage2_rc5plus_cyclic_anchor_overlay(
    *,
    base_collection: Any,
    output_directory: str | Path,
) -> VerifiedStage2RC5PlusCyclicAnchorOverlay:
    """Fresh-replay a base cyclic context and publish its nine-point overlay."""

    base = replay_verified_stage2_rc5_cyclic_context_collection(
        assert_verified_stage2_rc5_cyclic_context_collection(base_collection)
    )
    anchors, rows = _materialize(base)
    root = base.score_bundle.score_manifest_metadata.repository_root
    output = _output_directory(output_directory, root)
    anchor_bytes = _npy_bytes(anchors)
    rows_bytes = b"".join(_canonical(row, newline=True) for row in rows)
    anchor_binding = _member_binding(
        output / ANCHORS_FILENAME,
        anchor_bytes,
        dtype="float64",
        shape=anchors.shape,
    )
    rows_binding = {
        **_member_binding(output / ROWS_FILENAME, rows_bytes),
        "row_count": len(rows),
        "payload_policy": "identity_and_guardrails_only_no_score_values_no_labels",
    }
    manifest = _manifest(
        base,
        anchor_binding=anchor_binding,
        rows_binding=rows_binding,
    )
    manifest_bytes = _canonical(manifest, newline=True)
    commit = _commit(manifest_bytes, manifest, base)
    commit_bytes = _canonical(commit, newline=True)
    for name, data in (
        (ANCHORS_FILENAME, anchor_bytes),
        (ROWS_FILENAME, rows_bytes),
        (MANIFEST_FILENAME, manifest_bytes),
        (COMMIT_FILENAME, commit_bytes),
    ):
        _write_exclusive(output / name, data)
    descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return verify_stage2_rc5plus_cyclic_anchor_overlay(
        output / COMMIT_FILENAME,
        hashlib.sha256(commit_bytes).hexdigest(),
        base_collection=base,
    )


def verify_stage2_rc5plus_cyclic_anchor_overlay(
    commit_path: str | Path,
    expected_commit_sha256: str,
    *,
    base_collection: Any,
) -> VerifiedStage2RC5PlusCyclicAnchorOverlay:
    """Hash, bind and freshly recompute every nine-budget anchor row."""

    base = replay_verified_stage2_rc5_cyclic_context_collection(
        assert_verified_stage2_rc5_cyclic_context_collection(base_collection)
    )
    root = base.score_bundle.score_manifest_metadata.repository_root
    expected = _sha(expected_commit_sha256, "expected_commit_sha256")
    commit_file = _resolved_commit_path(commit_path, root)
    commit_bytes = _stable_bytes(commit_file, root, "RC5+ anchor overlay commit")
    if hashlib.sha256(commit_bytes).hexdigest() != expected:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor overlay external commit SHA mismatch"
        )
    commit = _strict_json(commit_bytes, "RC5+ anchor overlay commit")
    directory = commit_file.parent
    required = {
        ANCHORS_FILENAME,
        ROWS_FILENAME,
        MANIFEST_FILENAME,
        COMMIT_FILENAME,
    }
    if {item.name for item in directory.iterdir()} != required:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor overlay file closure mismatch"
        )
    manifest_ref = commit.get("manifest")
    if (
        not isinstance(manifest_ref, Mapping)
        or set(manifest_ref) != {"path", "sha256"}
        or manifest_ref["path"] != MANIFEST_FILENAME
    ):
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor overlay manifest binding is invalid"
        )
    manifest_bytes = _stable_bytes(
        directory / MANIFEST_FILENAME, root, "RC5+ anchor overlay manifest"
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != _sha(
        manifest_ref["sha256"], "manifest.sha256"
    ):
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor overlay manifest SHA mismatch"
        )
    manifest = _strict_json(manifest_bytes, "RC5+ anchor overlay manifest")
    count = len(base.episodes)
    members = manifest.get("members")
    if not isinstance(members, Mapping) or set(members) != {
        "anchor_coordinates",
        "identity_rows",
    }:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor overlay member closure mismatch"
        )
    anchors, anchor_bytes = _load_member_array(
        root,
        directory,
        members["anchor_coordinates"],
        expected_name=ANCHORS_FILENAME,
        expected_dtype=np.dtype("float64"),
        expected_shape=(count, len(BUDGET_KNOT_RATIONALS)),
    )
    rows_binding = members["identity_rows"]
    if (
        not isinstance(rows_binding, Mapping)
        or set(rows_binding)
        != {"path", "sha256", "row_count", "payload_policy"}
        or rows_binding["path"] != ROWS_FILENAME
        or rows_binding["row_count"] != count
        or rows_binding["payload_policy"]
        != "identity_and_guardrails_only_no_score_values_no_labels"
    ):
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor identity-row binding mismatch"
        )
    rows_bytes = _stable_bytes(directory / ROWS_FILENAME, root, "RC5+ anchor rows")
    if hashlib.sha256(rows_bytes).hexdigest() != _sha(
        rows_binding["sha256"], "identity_rows.sha256"
    ):
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor identity-row SHA mismatch"
        )
    lines = rows_bytes.splitlines(keepends=True)
    if len(lines) != count:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor identity-row cardinality mismatch"
        )
    rows = tuple(
        _strict_json(line, f"RC5+ anchor row[{index}]")
        for index, line in enumerate(lines)
    )
    expected_anchors, expected_rows = _materialize(base)
    if not np.array_equal(np.asarray(anchors), expected_anchors):
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "persisted RC5+ anchors differ from fresh score-map replay"
        )
    if [_plain(row) for row in rows] != [_plain(row) for row in expected_rows]:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "persisted RC5+ anchor identities differ from fresh replay"
        )
    expected_manifest = _manifest(
        base,
        anchor_binding=_member_binding(
            directory / ANCHORS_FILENAME,
            anchor_bytes,
            dtype="float64",
            shape=(count, len(BUDGET_KNOT_RATIONALS)),
        ),
        rows_binding={
            **_member_binding(directory / ROWS_FILENAME, rows_bytes),
            "row_count": count,
            "payload_policy": "identity_and_guardrails_only_no_score_values_no_labels",
        },
    )
    if manifest != expected_manifest:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor overlay manifest differs from causal replay"
        )
    expected_commit = _commit(manifest_bytes, manifest, base)
    if commit != expected_commit:
        raise Stage2RC5PlusCyclicAnchorOverlayError(
            "RC5+ anchor overlay commit differs from causal replay"
        )
    value = object.__new__(VerifiedStage2RC5PlusCyclicAnchorOverlay)
    for name, item in {
        "path": directory,
        "commit_sha256": expected,
        "manifest": MappingProxyType(manifest),
        "base_collection": base,
        "anchor_coordinates": anchors,
        "rows": tuple(MappingProxyType(row) for row in rows),
        "capability_schema": CAPABILITY_SCHEMA,
        "_capability": _CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return assert_verified_stage2_rc5plus_cyclic_anchor_overlay(value)


def replay_verified_stage2_rc5plus_cyclic_anchor_overlay(
    value: VerifiedStage2RC5PlusCyclicAnchorOverlay,
) -> VerifiedStage2RC5PlusCyclicAnchorOverlay:
    verified = assert_verified_stage2_rc5plus_cyclic_anchor_overlay(value)
    return verify_stage2_rc5plus_cyclic_anchor_overlay(
        verified.path / COMMIT_FILENAME,
        verified.commit_sha256,
        base_collection=verified.base_collection,
    )


__all__ = [
    "ANCHORS_FILENAME",
    "CAPABILITY_SCHEMA",
    "COMMIT_FILENAME",
    "COMMIT_SCHEMA",
    "MANIFEST_FILENAME",
    "OVERLAY_SCHEMA",
    "ROWS_FILENAME",
    "ROW_SCHEMA",
    "Stage2RC5PlusCyclicAnchorOverlayError",
    "VerifiedStage2RC5PlusCyclicAnchorOverlay",
    "assert_verified_stage2_rc5plus_cyclic_anchor_overlay",
    "build_and_publish_stage2_rc5plus_cyclic_anchor_overlay",
    "replay_verified_stage2_rc5plus_cyclic_anchor_overlay",
    "verify_stage2_rc5plus_cyclic_anchor_overlay",
]
