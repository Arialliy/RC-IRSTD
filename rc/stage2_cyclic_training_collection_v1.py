"""Commit-last, mmap-backed, source-only RC5 cyclic training collection.

One C14-derived 93D context/anchor is stored per cyclic start and one exact
event curve per image. Episode JSONL contains only C14/Q28 identity references;
aggregate curves are never serialized. Verification needs an external commit
SHA. The synthetic role builder is deliberately test-only; production roles
must be issued after replaying ``VerifiedStage2RC5ScoreBundleV2``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np

from model.endpoint_aware_threshold import EndpointAwareThresholdError, decode_coordinate_numpy
from rc.stage2_compositional_curve_provider import (
    CURVE_BANK_ID_ALGORITHM,
    PER_IMAGE_CURVE_BANK_SCHEMA,
    PerImageExactEventCurve,
    assert_per_image_exact_event_curve,
    build_compositional_exact_curve_provider_from_verified_curves,
    build_per_image_exact_event_curve,
)
from rc.stage2_cyclic_training_geometry import build_stage2_cyclic_training_geometry
from rc.stage2_domain_balanced_cyclic_sampler import DOMAIN_ORDER, OUTER_TARGETS


COLLECTION_SCHEMA = "rc-irstd.stage2-source-cyclic-training-collection.v1"
COMMIT_SCHEMA = "rc-irstd.stage2-source-cyclic-training-collection-commit.v1"
ROLE_SCHEMA = "rc-irstd.stage2-source-cyclic-role-material.v1"
EPISODE_SCHEMA = "rc-irstd.stage2-source-cyclic-training-episode-ref.v1"
EXTRACTOR_CONTRACT = (
    "extract_unlabeled_statistics_same_source_reference_and_statistics_config_"
    "as_deployment_v1"
)
REQUIRED_ROLE = "oof_holdout_stage2_fit"
MANIFEST_FILENAME = "manifest.json"
EPISODES_FILENAME = "episodes.jsonl"
COMMIT_FILENAME = "COLLECTION_COMMIT.json"
ARRAY_FILENAMES = MappingProxyType({
    "context_features": "context_features.npy",
    "anchor_coordinates": "anchor_coordinates.npy",
    "context_material_sha256": "context_material_sha256.npy",
    "curve_image_identity_sha256": "curve_image_identity_sha256.npy",
    "curve_offsets": "curve_offsets.npy",
    "curve_thresholds": "curve_thresholds.npy",
    "curve_false_positive_pixels": "curve_false_positive_pixels.npy",
    "curve_matched_objects": "curve_matched_objects.npy",
    "curve_total_native_pixels": "curve_total_native_pixels.npy",
    "curve_ground_truth_objects": "curve_ground_truth_objects.npy",
    "curve_content_sha256": "curve_content_sha256.npy",
})
_SHA_CHARS = frozenset("0123456789abcdef")
_ROLE_CAPABILITY = object()
_COLLECTION_CAPABILITY = object()


class Stage2CyclicTrainingCollectionError(ValueError):
    """A cyclic collection, exact-curve bank, or source boundary failed."""


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in _SHA_CHARS for char in value
    ):
        raise Stage2CyclicTrainingCollectionError(f"{name} must be lowercase SHA-256")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2CyclicTrainingCollectionError("non-canonical JSON value") from error


def _json_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _matrix(value: Any, dtype: np.dtype[Any], width: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.ndim != 2:
        raise TypeError(f"{name} must be explicit {dtype}[N,{width}] NumPy")
    if value.shape[0] < 42 or value.shape[1] != width:
        raise Stage2CyclicTrainingCollectionError(f"{name} shape is invalid")
    if not np.isfinite(value).all():
        raise Stage2CyclicTrainingCollectionError(f"{name} contains NaN/Inf")
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _context_material_sha256(*, role_identity_sha256: str, cyclic_start: int,
                             context_identities: Sequence[str], feature: np.ndarray,
                             anchor: np.ndarray) -> str:
    metadata = {
        "schema_version": "rc-irstd.stage2-cyclic-context-material-digest.v1",
        "role_identity_sha256": _sha(role_identity_sha256, "role identity"),
        "cyclic_start": int(cyclic_start),
        "ordered_context_image_identity_sha256": [
            _sha(item, "context identity") for item in context_identities
        ],
        "feature_dtype": "float32", "feature_shape": [93],
        "anchor_dtype": "float64", "anchor_shape": [3],
    }
    digest = hashlib.sha256()
    digest.update(b"rc-irstd.stage2-cyclic-context-material-digest.v1\0")
    for raw in (canonical_json_bytes(metadata),
                np.asarray(feature, dtype="<f4").tobytes(order="C"),
                np.asarray(anchor, dtype="<f8").tobytes(order="C")):
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class VerifiedCyclicSourceRoleMaterial:
    outer_fold_id: str
    outer_target: str
    source_domain: str
    oof_fold: int
    artifact_scope: str
    image_identities: tuple[str, ...]
    context_features: np.ndarray
    anchor_coordinates: np.ndarray
    curves: tuple[PerImageExactEventCurve, ...]
    context_material_sha256: tuple[str, ...]
    upstream_bindings: Mapping[str, str]
    role_identity_sha256: str
    boundary_values: Mapping[str, frozenset[str]]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedCyclicSourceRoleMaterial is verifier-issued only")


def _issue_role_material(*, outer_fold_id: str, source_domain: str, oof_fold: int,
                         image_identities: Sequence[str], context_features: np.ndarray,
                         anchor_coordinates: np.ndarray,
                         per_image_curves: Sequence[PerImageExactEventCurve],
                         upstream_bindings: Mapping[str, str], artifact_scope: str,
                         identity_boundary_records: Sequence[Mapping[str, Any]] | None = None,
                         ) -> VerifiedCyclicSourceRoleMaterial:
    if outer_fold_id not in OUTER_TARGETS:
        raise Stage2CyclicTrainingCollectionError("unsupported outer fold")
    target = OUTER_TARGETS[outer_fold_id]
    sources = tuple(domain for domain in DOMAIN_ORDER if domain != target)
    if source_domain not in sources:
        raise Stage2CyclicTrainingCollectionError("training role is not source-only")
    if type(oof_fold) is not int or oof_fold not in {0, 1}:
        raise Stage2CyclicTrainingCollectionError("oof_fold must be exact 0 or 1")
    if artifact_scope not in {"production", "synthetic_cpu_contract_test"}:
        raise Stage2CyclicTrainingCollectionError("invalid artifact scope")
    identities = tuple(_sha(item, f"image_identities[{index}]")
                       for index, item in enumerate(image_identities))
    if len(identities) < 42 or len(set(identities)) != len(identities):
        raise Stage2CyclicTrainingCollectionError("role needs >=42 unique images")
    features = _matrix(context_features, np.dtype("float32"), 93, "context_features")
    anchors = _matrix(anchor_coordinates, np.dtype("float64"), 3, "anchor_coordinates")
    if features.shape[0] != len(identities) or anchors.shape[0] != len(identities):
        raise Stage2CyclicTrainingCollectionError("one context/anchor per cyclic start required")
    try:
        decode_coordinate_numpy(anchors)
    except EndpointAwareThresholdError as error:
        raise Stage2CyclicTrainingCollectionError("anchors are not canonical EATC-v2") from error
    if np.any(anchors[:, 1:] < anchors[:, :-1]):
        raise Stage2CyclicTrainingCollectionError("anchors decreased")
    if isinstance(per_image_curves, (str, bytes)) or not isinstance(per_image_curves, Sequence):
        raise TypeError("per_image_curves must be ordered")
    curves = tuple(assert_per_image_exact_event_curve(item) for item in per_image_curves)
    if len(curves) != len(identities) or tuple(
        item.image_identity_sha256 for item in curves
    ) != identities:
        raise Stage2CyclicTrainingCollectionError("curve/image order mismatch")
    required = {"score_attestation_sha256", "score_manifest_metadata_sha256",
                "score_records_content_sha256", "run_complete_identity_sha256",
                "run_complete_artifact_sha256",
                "cyclic_context_collection_sha256",
                "seed_manifest_sha256",
                "statistics_config_sha256", "source_reference_sha256",
                "source_release_sha256"}
    if not isinstance(upstream_bindings, Mapping) or set(upstream_bindings) != required:
        raise Stage2CyclicTrainingCollectionError("upstream binding closure mismatch")
    bindings = {key: _sha(item, f"upstream.{key}")
                for key, item in upstream_bindings.items()}
    geometry = build_stage2_cyclic_training_geometry(len(identities))
    boundary_fields = ("canonical_id", "original_image_sha256",
                       "near_duplicate_cluster_id_or_unique_sentinel",
                       "exclusion_group_id")
    if identity_boundary_records is None:
        boundaries = {field: frozenset(identities) for field in boundary_fields}
    else:
        if len(identity_boundary_records) != len(identities):
            raise Stage2CyclicTrainingCollectionError("identity boundary cardinality mismatch")
        boundaries = {}
        for field in boundary_fields:
            values = []
            for index, record in enumerate(identity_boundary_records):
                if not isinstance(record, Mapping) or field not in record:
                    raise Stage2CyclicTrainingCollectionError(
                        f"identity boundary record {index} misses {field}")
                values.append(str(record[field]))
            boundaries[field] = frozenset(values)
        if tuple(str(record["original_image_sha256"])
                 for record in identity_boundary_records) != identities:
            raise Stage2CyclicTrainingCollectionError(
                "boundary original-image identities differ from curve order")
    projection = {
        "schema_version": ROLE_SCHEMA, "outer_fold_id": outer_fold_id,
        "outer_target": target, "source_domain": source_domain, "oof_fold": oof_fold,
        "required_role": REQUIRED_ROLE, "artifact_scope": artifact_scope,
        "extractor_contract": EXTRACTOR_CONTRACT,
        "ordered_image_identity_sha256": list(identities),
        "context_feature_matrix_sha256": hashlib.sha256(features.astype("<f4", copy=False).tobytes()).hexdigest(),
        "anchor_matrix_sha256": hashlib.sha256(anchors.astype("<f8", copy=False).tobytes()).hexdigest(),
        "curve_bindings": [{"image_identity_sha256": item.image_identity_sha256,
                            "curve_content_sha256": item.content_sha256} for item in curves],
        "upstream_bindings": bindings, "geometry_sha256": _json_sha(geometry),
        "identity_boundary_values": {
            field: sorted(values) for field, values in boundaries.items()
        },
        "outer_target_records_present": False, "official_test_accessed": False,
    }
    role_identity = _json_sha(projection)
    materials = tuple(_context_material_sha256(
        role_identity_sha256=role_identity, cyclic_start=start,
        context_identities=[identities[index] for index in row["context_indices"]],
        feature=features[start], anchor=anchors[start])
        for start, row in enumerate(geometry["episodes"]))
    value = object.__new__(VerifiedCyclicSourceRoleMaterial)
    fields = {"outer_fold_id": outer_fold_id, "outer_target": target,
              "source_domain": source_domain, "oof_fold": oof_fold,
              "artifact_scope": artifact_scope, "image_identities": identities,
              "context_features": features, "anchor_coordinates": anchors,
              "curves": curves, "context_material_sha256": materials,
              "upstream_bindings": MappingProxyType(bindings),
              "role_identity_sha256": role_identity,
              "boundary_values": MappingProxyType(boundaries),
              "_capability": _ROLE_CAPABILITY}
    for name, item in fields.items(): object.__setattr__(value, name, item)
    return value


def build_synthetic_cyclic_source_role_material(**kwargs: Any) -> VerifiedCyclicSourceRoleMaterial:
    """Issue an unmistakably test-only role for synthetic CPU E2E."""
    if "artifact_scope" in kwargs:
        raise TypeError("artifact_scope is fixed")
    return _issue_role_material(**kwargs, artifact_scope="synthetic_cpu_contract_test")


def build_cyclic_source_role_material_from_verified_context_collection(
    *,
    cyclic_context_collection: Any,
    per_image_curves: Sequence[PerImageExactEventCurve],
    source_release_sha256: str,
) -> VerifiedCyclicSourceRoleMaterial:
    """Production entry: replay one all-start cyclic-context collection.

    Variable-Q context bundles and bare score metadata are intentionally not
    accepted.  The new capability is the only constructible authority for the
    overlapping C14/Q28 training geometry.
    """
    from rc.stage2_rc5_cyclic_context import (
        assert_verified_stage2_rc5_cyclic_context_collection,
        replay_verified_stage2_rc5_cyclic_context_collection,
    )

    cyclic = replay_verified_stage2_rc5_cyclic_context_collection(
        assert_verified_stage2_rc5_cyclic_context_collection(
            cyclic_context_collection
        )
    )
    replayed_score = cyclic.score_bundle
    metadata = replayed_score.score_manifest_metadata
    payload = metadata.payload
    if metadata.role != REQUIRED_ROLE:
        raise Stage2CyclicTrainingCollectionError(
            "production cyclic training requires OOF holdout Stage-2 fit scores"
        )
    outer_fold = str(payload["outer_fold_id"])
    source_domain = str(payload["source_domain"])
    oof_fold = payload["oof_fold_index"]
    if type(oof_fold) is not int or oof_fold not in {0, 1}:
        raise Stage2CyclicTrainingCollectionError("score bundle has no exact OOF fold")
    identities = tuple(str(record["original_image_sha256"])
                       for record in metadata.records)
    if cyclic.context_features.shape != (len(identities), 93) or \
            cyclic.anchor_coordinates.shape != (len(identities), 3):
        raise Stage2CyclicTrainingCollectionError(
            "cyclic context collection cardinality differs from score role"
        )
    geometry = build_stage2_cyclic_training_geometry(len(identities))
    if cyclic.manifest["cyclic_geometry_sha256"] != _json_sha(geometry):
        raise Stage2CyclicTrainingCollectionError(
            "cyclic context geometry identity differs from training geometry"
        )
    for start, row in enumerate(cyclic.episodes):
        expected = geometry["episodes"][start]
        if tuple(row["context_indices"]) != tuple(expected["context_indices"]) or \
                tuple(row["query_indices"]) != tuple(expected["query_indices"]):
            raise Stage2CyclicTrainingCollectionError(
                "cyclic context episode differs from exact geometry replay"
            )
    attestation = replayed_score.attestation
    run_identity = attestation["run_complete"]["identity"]
    context_inputs = cyclic.manifest["input_bindings"]
    return _issue_role_material(
        outer_fold_id=outer_fold,
        source_domain=source_domain,
        oof_fold=oof_fold,
        image_identities=identities,
        context_features=np.asarray(cyclic.context_features, dtype=np.float32),
        anchor_coordinates=np.asarray(cyclic.anchor_coordinates, dtype=np.float64),
        per_image_curves=per_image_curves,
        identity_boundary_records=metadata.records,
        upstream_bindings={
            "score_attestation_sha256": replayed_score.attestation_sha256,
            "score_manifest_metadata_sha256": metadata.manifest_sha256,
            "score_records_content_sha256": metadata.records_content_sha256,
            "run_complete_identity_sha256": str(run_identity["identity_sha256"]),
            "run_complete_artifact_sha256": replayed_score.run_complete.sha256,
            "cyclic_context_collection_sha256": cyclic.commit_sha256,
            "seed_manifest_sha256": str(
                metadata.bindings["seed_manifest"]["sha256"]
            ),
            "statistics_config_sha256": str(
                context_inputs["statistics_config"]["sha256"]
            ),
            "source_reference_sha256": str(
                context_inputs["source_reference_v3"]["sha256"]
            ),
            "source_release_sha256": _sha(source_release_sha256, "source_release_sha256"),
        },
        artifact_scope="production",
    )


def build_cyclic_source_role_material_from_verified_bundles(
    **kwargs: Any,
) -> VerifiedCyclicSourceRoleMaterial:
    """Compatibility name; still accepts only a cyclic-context capability."""
    return build_cyclic_source_role_material_from_verified_context_collection(
        **kwargs
    )


def assert_verified_cyclic_source_role_material(value: object) -> VerifiedCyclicSourceRoleMaterial:
    if type(value) is not VerifiedCyclicSourceRoleMaterial or getattr(
        value, "_capability", None
    ) is not _ROLE_CAPABILITY:
        raise TypeError("a verifier-issued cyclic source role is required")
    return value


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    except Exception:
        try: os.close(descriptor)
        except OSError: pass
        raise


def _write_npy_new(path: Path, value: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
    except Exception:
        try: os.close(descriptor)
        except OSError: pass
        raise


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bank_id(bindings: Sequence[Mapping[str, Any]], total_rows: int) -> str:
    return _json_sha({
        "schema_version": PER_IMAGE_CURVE_BANK_SCHEMA,
        "bank_id_algorithm": CURVE_BANK_ID_ALGORITHM,
        "image_count": len(bindings), "total_curve_rows": int(total_rows),
        "identity_sorted_curve_bindings": sorted(
            bindings, key=lambda row: row["image_identity_sha256"]
        ),
    })


def publish_cyclic_training_collection_v1(
    output_directory: str | Path,
    role_materials: Sequence[VerifiedCyclicSourceRoleMaterial],
) -> "VerifiedCyclicTrainingCollection":
    """Publish exactly two source domains x two OOF folds, commit last."""
    output = Path(output_directory).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError("immutable cyclic collection already exists")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise Stage2CyclicTrainingCollectionError("invalid collection parent")
    roles = tuple(assert_verified_cyclic_source_role_material(item)
                  for item in role_materials)
    if len(roles) != 4 or len({item.outer_fold_id for item in roles}) != 1:
        raise Stage2CyclicTrainingCollectionError("exactly four same-outer roles required")
    outer_fold = roles[0].outer_fold_id
    target = OUTER_TARGETS[outer_fold]
    sources = tuple(domain for domain in DOMAIN_ORDER if domain != target)
    expected = {(domain, fold) for domain in sources for fold in (0, 1)}
    if {(item.source_domain, item.oof_fold) for item in roles} != expected:
        raise Stage2CyclicTrainingCollectionError("source domain/fold coverage incomplete")
    scopes = {item.artifact_scope for item in roles}
    if len(scopes) != 1:
        raise Stage2CyclicTrainingCollectionError("production/synthetic roles cannot mix")
    roles = tuple(sorted(roles, key=lambda item:
                        (DOMAIN_ORDER.index(item.source_domain), item.oof_fold)))
    identities = tuple(identity for role in roles for identity in role.image_identities)
    if len(set(identities)) != len(identities):
        raise Stage2CyclicTrainingCollectionError("OOF role identities overlap")
    curves = tuple(curve for role in roles for curve in role.curves)
    offsets = [0]
    for curve in curves: offsets.append(offsets[-1] + int(curve.thresholds.size))
    arrays = {
        "context_features": np.concatenate([r.context_features for r in roles]).astype("<f4"),
        "anchor_coordinates": np.concatenate([r.anchor_coordinates for r in roles]).astype("<f8"),
        "context_material_sha256": np.asarray(
            [item for role in roles for item in role.context_material_sha256], dtype="<U64"),
        "curve_image_identity_sha256": np.asarray(identities, dtype="<U64"),
        "curve_offsets": np.asarray(offsets, dtype="<i8"),
        "curve_thresholds": np.concatenate([c.thresholds for c in curves]).astype("<f8"),
        "curve_false_positive_pixels": np.concatenate(
            [c.false_positive_pixels for c in curves]).astype("<i8"),
        "curve_matched_objects": np.concatenate([c.matched_objects for c in curves]).astype("<i8"),
        "curve_total_native_pixels": np.asarray(
            [c.total_native_pixels for c in curves], dtype="<i8"),
        "curve_ground_truth_objects": np.asarray(
            [c.ground_truth_objects for c in curves], dtype="<i8"),
        "curve_content_sha256": np.asarray([c.content_sha256 for c in curves], dtype="<U64"),
    }
    curve_bindings = [{
        "image_identity_sha256": curve.image_identity_sha256,
        "curve_content_sha256": curve.content_sha256,
        "curve_row_count": int(curve.thresholds.size),
        "total_native_pixels": curve.total_native_pixels,
        "ground_truth_objects": curve.ground_truth_objects,
    } for curve in curves]
    bank_id = _bank_id(curve_bindings, offsets[-1])
    episodes: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    global_start = 0
    domain_offsets = {domain: 0 for domain in sources}
    for role in roles:
        geometry = build_stage2_cyclic_training_geometry(len(role.image_identities))
        role_start = global_start
        for row in geometry["episodes"]:
            start = int(row["cyclic_start"])
            episodes.append({
                "schema_version": EPISODE_SCHEMA, "outer_fold_id": outer_fold,
                "source_domain": role.source_domain, "oof_fold": role.oof_fold,
                "domain_episode_index": domain_offsets[role.source_domain] + start,
                "role_episode_index": start, "cyclic_start": start,
                "ordered_context_image_identity_sha256": [
                    role.image_identities[index] for index in row["context_indices"]],
                "ordered_query_image_identity_sha256": [
                    role.image_identities[index] for index in row["query_indices"]],
            })
            global_start += 1
        inventory.append({
            "role_identity_sha256": role.role_identity_sha256,
            "source_domain": role.source_domain, "oof_fold": role.oof_fold,
            "required_role": REQUIRED_ROLE, "record_count": len(role.image_identities),
            "episode_start": role_start, "episode_stop": global_start,
            "curve_start": role_start, "curve_stop": global_start,
            "geometry_sha256": _json_sha(geometry),
            "upstream_bindings": dict(role.upstream_bindings),
        })
        domain_offsets[role.source_domain] += len(role.image_identities)
    def binding_set_identity(field: str) -> str:
        return _json_sha({
            "schema_version": "rc-irstd.stage2-cyclic-input-binding-set.v1",
            "field": field,
            "values": sorted({role.upstream_bindings[field] for role in roles}),
        })
    statistics_values = {role.upstream_bindings["statistics_config_sha256"] for role in roles}
    release_values = {role.upstream_bindings["source_release_sha256"] for role in roles}
    if len(statistics_values) != 1 or len(release_values) != 1:
        raise Stage2CyclicTrainingCollectionError(
            "roles must share one statistics config and one source release"
        )
    actual_input_identities = {
        "statistics_config": next(iter(statistics_values)),
        "source_reference": binding_set_identity("source_reference_sha256"),
        "detector_run_complete_set": _json_sha({
            "schema_version": "rc-irstd.stage2-run-complete-binding-set.v1",
            "bindings": sorted(
                ({"attestation_sha256": role.upstream_bindings["score_attestation_sha256"],
                  "run_complete_artifact_sha256": role.upstream_bindings[
                      "run_complete_artifact_sha256"],
                  "run_complete_identity_sha256": role.upstream_bindings[
                      "run_complete_identity_sha256"]} for role in roles),
                key=lambda row: (row["run_complete_identity_sha256"],
                                 row["attestation_sha256"]),
            ),
        }),
        "seed_manifest": binding_set_identity("seed_manifest_sha256"),
        "source_release": next(iter(release_values)),
    }
    boundary_fields = ("canonical_id", "original_image_sha256",
                       "near_duplicate_cluster_id_or_unique_sentinel",
                       "exclusion_group_id")
    identity_boundary_values = {
        field: sorted(set().union(*(role.boundary_values[field] for role in roles)))
        for field in boundary_fields
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        for name, value in arrays.items():
            _write_npy_new(staging / ARRAY_FILENAMES[name], value)
        episode_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in episodes)
        _write_new(staging / EPISODES_FILENAME, episode_bytes)
        members = {name: {
            "path": ARRAY_FILENAMES[name], "sha256": _file_sha(staging / ARRAY_FILENAMES[name]),
            "dtype": str(value.dtype), "shape": list(value.shape),
        } for name, value in arrays.items()}
        manifest = {
            "schema_version": COLLECTION_SCHEMA,
            "artifact_status": "COMPLETE_SOURCE_ONLY_CYCLIC_TRAINING_COLLECTION",
            "artifact_scope": next(iter(scopes)), "outer_fold_id": outer_fold,
            "outer_target": target, "source_domains": list(sources),
            "required_role": REQUIRED_ROLE, "extractor_contract": EXTRACTOR_CONTRACT,
            "episode_geometry": "C14_Q28_every_start_per_OOF_role",
            "validation_geometry_interchangeable": False,
            "episode_count": len(episodes), "image_count": len(curves),
            "total_curve_rows": offsets[-1], "curve_bank_id": bank_id,
            "curve_bank_id_algorithm": CURVE_BANK_ID_ALGORITHM,
            "role_inventory": inventory, "members": members,
            "actual_input_binding_identities": actual_input_identities,
            "identity_boundary_values": identity_boundary_values,
            "episodes": {"path": EPISODES_FILENAME,
                         "sha256": hashlib.sha256(episode_bytes).hexdigest(),
                         "row_count": len(episodes),
                         "payload_policy": "identity_references_only_no_features_no_curves"},
            "aggregate_curve_materialization": False,
            "outer_target_records_present": False, "official_test_accessed": False,
            "query_labels_accessed_by_context_extractor": False,
            "manifest_identity_sha256": "",
        }
        projection = dict(manifest); projection.pop("manifest_identity_sha256")
        manifest["manifest_identity_sha256"] = _json_sha(projection)
        manifest_bytes = canonical_json_bytes(manifest)
        _write_new(staging / MANIFEST_FILENAME, manifest_bytes)
        commit = {"schema_version": COMMIT_SCHEMA, "artifact_status": "COMMITTED_LAST",
                  "manifest": {"path": MANIFEST_FILENAME,
                               "sha256": hashlib.sha256(manifest_bytes).hexdigest()},
                  "curve_bank_id": bank_id, "episode_count": len(episodes),
                  "commit_identity_sha256": ""}
        projection = dict(commit); projection.pop("commit_identity_sha256")
        commit["commit_identity_sha256"] = _json_sha(projection)
        commit_bytes = canonical_json_bytes(commit)
        _write_new(staging / COMMIT_FILENAME, commit_bytes)
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        os.rename(staging, output); staging = Path()
        return verify_cyclic_training_collection_v1(
            output, hashlib.sha256(commit_bytes).hexdigest())
    finally:
        if staging != Path() and staging.exists() and staging.parent == parent:
            shutil.rmtree(staging)


def _stable_regular_bytes(path: Path, parent: Path, name: str) -> bytes:
    if path.parent != parent:
        raise Stage2CyclicTrainingCollectionError(f"{name} escaped collection")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise Stage2CyclicTrainingCollectionError(f"{name} is not a direct regular file")
    data = path.read_bytes(); after = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_size,
                             item.st_mtime_ns, item.st_ctime_ns)
    if identity(before) != identity(after) or len(data) != after.st_size:
        raise Stage2CyclicTrainingCollectionError(f"{name} changed while read")
    return data


def _strict_json(data: bytes, name: str) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2CyclicTrainingCollectionError(f"{name} has duplicate keys")
            result[key] = value
        return result
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2CyclicTrainingCollectionError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise Stage2CyclicTrainingCollectionError(f"{name} is not canonical JSON")
    return value


@dataclass(frozen=True, init=False)
class VerifiedCyclicTrainingCollection:
    path: Path
    commit_sha256: str
    manifest: Mapping[str, Any]
    episodes: tuple[Mapping[str, Any], ...]
    arrays: Mapping[str, np.ndarray]
    curve_bank_id: str
    domain_episode_indices: Mapping[str, tuple[int, ...]]
    _curve_index_by_identity: Mapping[str, int]
    boundary_values: Mapping[str, frozenset[str]]
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedCyclicTrainingCollection is verifier-issued only")

    @property
    def artifact_scope(self) -> str:
        assert_verified_cyclic_training_collection(self)
        return str(self.manifest["artifact_scope"])

    def episode_for_domain(self, source_domain: str,
                           domain_episode_index: int) -> Mapping[str, Any]:
        assert_verified_cyclic_training_collection(self)
        if source_domain not in self.domain_episode_indices:
            raise KeyError("unknown source domain")
        indices = self.domain_episode_indices[source_domain]
        if type(domain_episode_index) is not int or not 0 <= domain_episode_index < len(indices):
            raise IndexError("domain episode index is out of range")
        return self.episodes[indices[domain_episode_index]]

    def feature_anchor_for_episode(self, source_domain: str,
                                   domain_episode_index: int
                                   ) -> tuple[np.ndarray, np.ndarray]:
        row = self.episode_for_domain(source_domain, domain_episode_index)
        index = int(row["global_episode_index"])
        return (self.arrays["context_features"][index],
                self.arrays["anchor_coordinates"][index])

    def provider_for_episode(self, source_domain: str, domain_episode_index: int):
        """Materialize only the live episode's 28 verified image curves."""
        row = self.episode_for_domain(source_domain, domain_episode_index)
        offsets = self.arrays["curve_offsets"]
        curves = []
        for identity in row["ordered_query_image_identity_sha256"]:
            curve_index = self._curve_index_by_identity[str(identity)]
            start, stop = int(offsets[curve_index]), int(offsets[curve_index + 1])
            curve = build_per_image_exact_event_curve(
                image_identity_sha256=str(identity),
                thresholds=np.asarray(self.arrays["curve_thresholds"][start:stop],
                                      dtype=np.float64),
                false_positive_pixels=np.asarray(
                    self.arrays["curve_false_positive_pixels"][start:stop], dtype=np.int64),
                matched_objects=np.asarray(
                    self.arrays["curve_matched_objects"][start:stop], dtype=np.int64),
                total_native_pixels=int(
                    self.arrays["curve_total_native_pixels"][curve_index]),
                ground_truth_objects=int(
                    self.arrays["curve_ground_truth_objects"][curve_index]),
            )
            if curve.content_sha256 != str(self.arrays["curve_content_sha256"][curve_index]):
                raise Stage2CyclicTrainingCollectionError("mmap curve content drifted")
            curves.append(curve)
        return build_compositional_exact_curve_provider_from_verified_curves(
            curve_bank_id=self.curve_bank_id, curves=curves)

    def fit_training_standardizer(self) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(self.arrays["context_features"], dtype=np.float64)
        mean = np.mean(matrix, axis=0, dtype=np.float64)
        scale = np.maximum(np.std(matrix, axis=0, dtype=np.float64, ddof=0), 1e-8)
        return mean, scale


def assert_verified_cyclic_training_collection(value: object) -> VerifiedCyclicTrainingCollection:
    if type(value) is not VerifiedCyclicTrainingCollection or getattr(
        value, "_capability", None
    ) is not _COLLECTION_CAPABILITY:
        raise TypeError("a verifier-issued cyclic training collection is required")
    return value


def verify_cyclic_training_collection_v1(
    directory: str | Path, expected_commit_sha256: str
) -> VerifiedCyclicTrainingCollection:
    """Hash every member, replay every C14/Q28 row, then issue mmap access."""
    expected = _sha(expected_commit_sha256, "expected_commit_sha256")
    raw = Path(directory).expanduser()
    if raw.is_symlink():
        raise Stage2CyclicTrainingCollectionError("collection directory is a symlink")
    root = raw.resolve(strict=True)
    if not root.is_dir():
        raise Stage2CyclicTrainingCollectionError("collection is not a directory")
    commit_bytes = _stable_regular_bytes(root / COMMIT_FILENAME, root, "commit")
    if hashlib.sha256(commit_bytes).hexdigest() != expected:
        raise Stage2CyclicTrainingCollectionError("external commit SHA mismatch")
    commit = _strict_json(commit_bytes, "commit")
    if commit.get("schema_version") != COMMIT_SCHEMA or commit.get(
        "artifact_status") != "COMMITTED_LAST":
        raise Stage2CyclicTrainingCollectionError("commit schema/status mismatch")
    projection = dict(commit); identity = projection.pop("commit_identity_sha256", None)
    if identity != _json_sha(projection):
        raise Stage2CyclicTrainingCollectionError("commit identity mismatch")
    manifest_ref = commit.get("manifest")
    if not isinstance(manifest_ref, Mapping) or set(manifest_ref) != {"path", "sha256"} \
            or manifest_ref["path"] != MANIFEST_FILENAME:
        raise Stage2CyclicTrainingCollectionError("commit manifest binding invalid")
    manifest_bytes = _stable_regular_bytes(root / MANIFEST_FILENAME, root, "manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != _sha(
        manifest_ref["sha256"], "manifest SHA"):
        raise Stage2CyclicTrainingCollectionError("manifest SHA mismatch")
    manifest = _strict_json(manifest_bytes, "manifest")
    if manifest.get("schema_version") != COLLECTION_SCHEMA:
        raise Stage2CyclicTrainingCollectionError("collection schema mismatch")
    projection = dict(manifest); identity = projection.pop("manifest_identity_sha256", None)
    if identity != _json_sha(projection):
        raise Stage2CyclicTrainingCollectionError("manifest identity mismatch")
    fixed = {"required_role": REQUIRED_ROLE, "extractor_contract": EXTRACTOR_CONTRACT,
             "aggregate_curve_materialization": False,
             "outer_target_records_present": False, "official_test_accessed": False,
             "query_labels_accessed_by_context_extractor": False,
             "validation_geometry_interchangeable": False}
    for key, value in fixed.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise Stage2CyclicTrainingCollectionError(f"manifest {key} mismatch")
    inventory_for_bindings = manifest.get("role_inventory")
    if not isinstance(inventory_for_bindings, list) or len(inventory_for_bindings) != 4:
        raise Stage2CyclicTrainingCollectionError("role inventory missing for input bindings")
    def replay_binding_set(field: str) -> str:
        return _json_sha({
            "schema_version": "rc-irstd.stage2-cyclic-input-binding-set.v1",
            "field": field,
            "values": sorted({row["upstream_bindings"][field]
                              for row in inventory_for_bindings}),
        })
    statistics_values = {row["upstream_bindings"]["statistics_config_sha256"]
                         for row in inventory_for_bindings}
    release_values = {row["upstream_bindings"]["source_release_sha256"]
                      for row in inventory_for_bindings}
    replayed_inputs = {
        "statistics_config": next(iter(statistics_values)) if len(statistics_values) == 1 else "",
        "source_reference": replay_binding_set("source_reference_sha256"),
        "detector_run_complete_set": _json_sha({
            "schema_version": "rc-irstd.stage2-run-complete-binding-set.v1",
            "bindings": sorted(
                ({"attestation_sha256": row["upstream_bindings"]["score_attestation_sha256"],
                  "run_complete_artifact_sha256": row["upstream_bindings"][
                      "run_complete_artifact_sha256"],
                  "run_complete_identity_sha256": row["upstream_bindings"][
                      "run_complete_identity_sha256"]} for row in inventory_for_bindings),
                key=lambda row: (row["run_complete_identity_sha256"],
                                 row["attestation_sha256"]),
            ),
        }),
        "seed_manifest": replay_binding_set("seed_manifest_sha256"),
        "source_release": next(iter(release_values)) if len(release_values) == 1 else "",
    }
    if manifest.get("actual_input_binding_identities") != replayed_inputs:
        raise Stage2CyclicTrainingCollectionError("actual input identities mismatch")
    raw_boundaries = manifest.get("identity_boundary_values")
    boundary_fields = {"canonical_id", "original_image_sha256",
                       "near_duplicate_cluster_id_or_unique_sentinel",
                       "exclusion_group_id"}
    if not isinstance(raw_boundaries, Mapping) or set(raw_boundaries) != boundary_fields:
        raise Stage2CyclicTrainingCollectionError("identity boundary closure mismatch")
    boundary_values: dict[str, frozenset[str]] = {}
    for field in boundary_fields:
        values = raw_boundaries[field]
        if not isinstance(values, list) or values != sorted(set(values)) or not all(
            isinstance(item, str) and item for item in values
        ):
            raise Stage2CyclicTrainingCollectionError(f"identity boundary {field} invalid")
        boundary_values[field] = frozenset(values)
    outer_fold = manifest.get("outer_fold_id")
    if outer_fold not in OUTER_TARGETS or manifest.get("outer_target") != OUTER_TARGETS[outer_fold]:
        raise Stage2CyclicTrainingCollectionError("outer fold/target mismatch")
    sources = tuple(domain for domain in DOMAIN_ORDER if domain != OUTER_TARGETS[outer_fold])
    if manifest.get("source_domains") != list(sources):
        raise Stage2CyclicTrainingCollectionError("source domains mismatch")
    members = manifest.get("members")
    if not isinstance(members, Mapping) or set(members) != set(ARRAY_FILENAMES):
        raise Stage2CyclicTrainingCollectionError("array member closure mismatch")
    required_files = {MANIFEST_FILENAME, EPISODES_FILENAME, COMMIT_FILENAME,
                      *ARRAY_FILENAMES.values()}
    if {item.name for item in root.iterdir()} != required_files:
        raise Stage2CyclicTrainingCollectionError("collection file closure mismatch")
    arrays: dict[str, np.ndarray] = {}
    for name, filename in ARRAY_FILENAMES.items():
        row = members[name]
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256", "dtype", "shape"} \
                or row["path"] != filename:
            raise Stage2CyclicTrainingCollectionError(f"member {name} binding invalid")
        data = _stable_regular_bytes(root / filename, root, f"member {name}")
        if hashlib.sha256(data).hexdigest() != _sha(row["sha256"], f"member {name} SHA"):
            raise Stage2CyclicTrainingCollectionError(f"member {name} SHA mismatch")
        value = np.load(root / filename, mmap_mode="r", allow_pickle=False)
        if str(value.dtype) != row["dtype"] or list(value.shape) != row["shape"]:
            raise Stage2CyclicTrainingCollectionError(f"member {name} dtype/shape mismatch")
        arrays[name] = value
    count = manifest.get("episode_count")
    if type(count) is not int or count < 168 or manifest.get("image_count") != count:
        raise Stage2CyclicTrainingCollectionError("episode/image cardinality mismatch")
    if arrays["context_features"].shape != (count, 93) or \
            arrays["context_features"].dtype != np.float32:
        raise Stage2CyclicTrainingCollectionError("context matrix mismatch")
    if arrays["anchor_coordinates"].shape != (count, 3) or \
            arrays["anchor_coordinates"].dtype != np.float64:
        raise Stage2CyclicTrainingCollectionError("anchor matrix mismatch")
    if arrays["context_material_sha256"].shape != (count,) or \
            arrays["curve_image_identity_sha256"].shape != (count,):
        raise Stage2CyclicTrainingCollectionError("identity vector mismatch")
    identities = tuple(str(item) for item in arrays["curve_image_identity_sha256"])
    if len(set(identities)) != count:
        raise Stage2CyclicTrainingCollectionError("curve identities repeat")
    for item in identities: _sha(item, "curve identity")
    offsets = arrays["curve_offsets"]
    if offsets.dtype != np.int64 or offsets.shape != (count + 1,) or \
            int(offsets[0]) != 0 or np.any(np.diff(offsets) < 2):
        raise Stage2CyclicTrainingCollectionError("curve offsets invalid")
    total_rows = int(offsets[-1])
    if manifest.get("total_curve_rows") != total_rows:
        raise Stage2CyclicTrainingCollectionError("total curve rows mismatch")
    for name in ("curve_thresholds", "curve_false_positive_pixels", "curve_matched_objects"):
        if arrays[name].shape != (total_rows,):
            raise Stage2CyclicTrainingCollectionError(f"{name} cardinality mismatch")
    for name in ("curve_total_native_pixels", "curve_ground_truth_objects",
                 "curve_content_sha256"):
        if arrays[name].shape != (count,):
            raise Stage2CyclicTrainingCollectionError(f"{name} cardinality mismatch")
    curve_bindings = []
    for index, image_identity in enumerate(identities):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        curve = build_per_image_exact_event_curve(
            image_identity_sha256=image_identity,
            thresholds=np.asarray(arrays["curve_thresholds"][start:stop], dtype=np.float64),
            false_positive_pixels=np.asarray(
                arrays["curve_false_positive_pixels"][start:stop], dtype=np.int64),
            matched_objects=np.asarray(
                arrays["curve_matched_objects"][start:stop], dtype=np.int64),
            total_native_pixels=int(arrays["curve_total_native_pixels"][index]),
            ground_truth_objects=int(arrays["curve_ground_truth_objects"][index]),
        )
        if curve.content_sha256 != str(arrays["curve_content_sha256"][index]):
            raise Stage2CyclicTrainingCollectionError("per-image curve hash mismatch")
        curve_bindings.append({
            "image_identity_sha256": image_identity,
            "curve_content_sha256": curve.content_sha256,
            "curve_row_count": stop - start,
            "total_native_pixels": curve.total_native_pixels,
            "ground_truth_objects": curve.ground_truth_objects,
        })
    bank_id = _bank_id(curve_bindings, total_rows)
    if bank_id != manifest.get("curve_bank_id") or bank_id != commit.get("curve_bank_id"):
        raise Stage2CyclicTrainingCollectionError("curve bank ID mismatch")
    episode_ref = manifest.get("episodes")
    episode_bytes = _stable_regular_bytes(root / EPISODES_FILENAME, root, "episodes")
    if not isinstance(episode_ref, Mapping) or set(episode_ref) != {
        "path", "sha256", "row_count", "payload_policy"
    } or episode_ref["path"] != EPISODES_FILENAME or episode_ref[
        "payload_policy"] != "identity_references_only_no_features_no_curves" or \
            hashlib.sha256(episode_bytes).hexdigest() != episode_ref["sha256"]:
        raise Stage2CyclicTrainingCollectionError("episodes binding mismatch")
    lines = episode_bytes.splitlines()
    if len(lines) != count or episode_ref["row_count"] != count:
        raise Stage2CyclicTrainingCollectionError("episode row count mismatch")
    rows = [_strict_json(line, f"episode[{index}]") for index, line in enumerate(lines)]
    inventory = manifest.get("role_inventory")
    if not isinstance(inventory, list) or len(inventory) != 4:
        raise Stage2CyclicTrainingCollectionError("role inventory incomplete")
    expected_roles = {(domain, fold) for domain in sources for fold in (0, 1)}
    if {(row.get("source_domain"), row.get("oof_fold"))
            for row in inventory} != expected_roles:
        raise Stage2CyclicTrainingCollectionError("role domain/fold mismatch")
    domain_indices: dict[str, list[int]] = {domain: [] for domain in sources}
    global_seen = 0
    for role in inventory:
        start, stop = role.get("episode_start"), role.get("episode_stop")
        if type(start) is not int or type(stop) is not int or start != global_seen or \
                stop - start != role.get("record_count") or \
                role.get("curve_start") != start or role.get("curve_stop") != stop:
            raise Stage2CyclicTrainingCollectionError("role ranges not contiguous")
        geometry = build_stage2_cyclic_training_geometry(stop - start)
        if role.get("geometry_sha256") != _json_sha(geometry) or \
                role.get("required_role") != REQUIRED_ROLE:
            raise Stage2CyclicTrainingCollectionError("role geometry/role mismatch")
        role_ids = identities[start:stop]
        for local, geometry_row in enumerate(geometry["episodes"]):
            global_index = start + local
            expected_row = {
                "schema_version": EPISODE_SCHEMA, "outer_fold_id": outer_fold,
                "source_domain": role["source_domain"], "oof_fold": role["oof_fold"],
                "domain_episode_index": len(domain_indices[role["source_domain"]]),
                "role_episode_index": local, "cyclic_start": local,
                "ordered_context_image_identity_sha256": [
                    role_ids[index] for index in geometry_row["context_indices"]],
                "ordered_query_image_identity_sha256": [
                    role_ids[index] for index in geometry_row["query_indices"]],
            }
            if rows[global_index] != expected_row:
                raise Stage2CyclicTrainingCollectionError("episode cyclic replay mismatch")
            material = _context_material_sha256(
                role_identity_sha256=role["role_identity_sha256"], cyclic_start=local,
                context_identities=expected_row["ordered_context_image_identity_sha256"],
                feature=arrays["context_features"][global_index],
                anchor=arrays["anchor_coordinates"][global_index])
            if str(arrays["context_material_sha256"][global_index]) != material:
                raise Stage2CyclicTrainingCollectionError("context material hash mismatch")
            rows[global_index]["global_episode_index"] = global_index
            domain_indices[role["source_domain"]].append(global_index)
        global_seen = stop
    if global_seen != count:
        raise Stage2CyclicTrainingCollectionError("role ranges do not cover collection")
    value = object.__new__(VerifiedCyclicTrainingCollection)
    fields = {
        "path": root, "commit_sha256": expected,
        "manifest": MappingProxyType(manifest),
        "episodes": tuple(MappingProxyType(row) for row in rows),
        "arrays": MappingProxyType(arrays), "curve_bank_id": bank_id,
        "domain_episode_indices": MappingProxyType(
            {key: tuple(item) for key, item in domain_indices.items()}),
        "_curve_index_by_identity": MappingProxyType(
            {identity: index for index, identity in enumerate(identities)}),
        "boundary_values": MappingProxyType(boundary_values),
        "_capability": _COLLECTION_CAPABILITY,
    }
    for name, item in fields.items(): object.__setattr__(value, name, item)
    return value


__all__ = [
    "ARRAY_FILENAMES", "COLLECTION_SCHEMA", "COMMIT_FILENAME", "COMMIT_SCHEMA",
    "EPISODES_FILENAME", "EXTRACTOR_CONTRACT", "MANIFEST_FILENAME", "REQUIRED_ROLE",
    "Stage2CyclicTrainingCollectionError", "VerifiedCyclicSourceRoleMaterial",
    "VerifiedCyclicTrainingCollection", "assert_verified_cyclic_source_role_material",
    "assert_verified_cyclic_training_collection",
    "build_synthetic_cyclic_source_role_material",
    "build_cyclic_source_role_material_from_verified_context_collection",
    "build_cyclic_source_role_material_from_verified_bundles",
    "publish_cyclic_training_collection_v1", "verify_cyclic_training_collection_v1",
]
