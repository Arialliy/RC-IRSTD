"""Additive variable-query Stage-2 context, episode, and collection contracts.

This module does not modify or reinterpret the v5 pipeline.  It closes a
result-free v6 payload layer with four properties:

* context packages are label blind, bind C14 plus a geometry-derived dynamic
  query identity, and expose a query-free 93-dimensional inference material;
* episodes recursively validate their embedded context payload and bind
  immutable digest descriptors for a pre-label T4 anchor, an EATC-v2
  oracle/curve artifact, exact rational budgets, and (for outer episodes
  only) a pre-label decision set;
* collections replay every variable-query geometry, consume every ordered
  identity once, and weight windows equally; and
* JSONL, manifest, and commit bytes are canonical, externally SHA-bound, and
  verified in commit-last order before a recursive immutable capability is
  issued.

The episode payload verifier is semantic and digest-binding only: it does not
open the anchor, curve, or decision paths.  The collection bundle verifier
does open and externally SHA-verify its JSONL, manifest, and commit members.
The variable-query adapter accepts only an already I/O-verified window
capability.  No query score, query label, observed result, or budget float is
consumed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
from types import MappingProxyType
from typing import Any

from data_ext.stage2_variable_query_window import (
    VerifiedStage2VariableQueryWindow,
    assert_verified_stage2_variable_query_window,
)
from rc.domain_statistics import FEATURE_NAMES
from rc.stage2_variable_query_geometry import (
    CONTEXT_SIZE,
    SCHEMA_VERSION as VARIABLE_QUERY_GEOMETRY_SCHEMA,
    validate_stage2_variable_query_geometry,
)


CONTEXT_SCHEMA = "rc-irstd.stage2-context-package.v2"
EPISODE_SCHEMA = "rc-irstd.meta-episode.v6"
COLLECTION_SCHEMA = "rc-irstd.meta-episode-collection.v6"
COLLECTION_COMMIT_SCHEMA = "rc-irstd.meta-episode-collection-commit.v2"

CONTEXT_ARTIFACT_TYPE = "rc_irstd_stage2_unlabeled_context_package"
EPISODE_ARTIFACT_TYPE = "rc_irstd_stage2_crossfit_meta_episode"
COLLECTION_ARTIFACT_TYPE = "rc_irstd_stage2_crossfit_episode_collection"
COLLECTION_COMMIT_ARTIFACT_TYPE = (
    "rc_irstd_stage2_crossfit_episode_collection_commit"
)

OOF_HOLDOUT_STAGE2_FIT = "oof_holdout_stage2_fit"
SOURCE_DIAGNOSTIC_VALIDATION = "source_diagnostic_validation"
OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT = (
    "outer_target_diagnostic_development"
)
STAGE2_OOF_FIT = "stage2_oof_fit"

COLLECTION_TRAIN = "stage2_crossfit_training"
COLLECTION_VALIDATION = "stage2_source_checkpoint_validation"
COLLECTION_OUTER = "stage2_outer_target_development_evaluation"

ROLE_TO_EPISODE = MappingProxyType({
    OOF_HOLDOUT_STAGE2_FIT: STAGE2_OOF_FIT,
    SOURCE_DIAGNOSTIC_VALIDATION: SOURCE_DIAGNOSTIC_VALIDATION,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT: (
        OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
    ),
})
ROLE_TO_COLLECTION = MappingProxyType({
    OOF_HOLDOUT_STAGE2_FIT: COLLECTION_TRAIN,
    SOURCE_DIAGNOSTIC_VALIDATION: COLLECTION_VALIDATION,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT: COLLECTION_OUTER,
})

OUTER_TARGETS = MappingProxyType({
    "outer_leave_nuaa_sirst": "NUAA-SIRST",
    "outer_leave_nudt_sirst": "NUDT-SIRST",
    "outer_leave_irstd_1k": "IRSTD-1K",
})
ALL_DOMAINS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
BASE_SEEDS = (42, 123, 3407)

BUDGET_RATIONALS = (
    MappingProxyType({"numerator": 1, "denominator": 10_000}),
    MappingProxyType({"numerator": 1, "denominator": 100_000}),
    MappingProxyType({"numerator": 1, "denominator": 1_000_000}),
)
STRICT_THRESHOLD_SEMANTICS = "prediction = probability > threshold"
EATC_V2_SCHEMA = "rc-irstd.endpoint-aware-piecewise-tail-coordinate.v2"
ANCHOR_SCHEMA = "rc-irstd.stage2-context-tail-anchor.v1"
ORACLE_CURVE_BINDING_SCHEMA = (
    "rc-irstd.stage2-eatc-oracle-curve-binding.v2"
)
PRELABEL_DECISION_BINDING_SCHEMA = (
    "rc-irstd.stage2-prelabel-decision-binding.v1"
)

FEATURE_DIM = 93
FLOAT32_VECTOR_ALGORITHM = "sha256-little-endian-float32-c-order-v1"
FULL_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-stage2-v6-full-identity-v1"
)
BOOTSTRAP_QUERY_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-stage2-v6-query-image-identity-v1"
)
WINDOW_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-stage2-v6-window-identity-v1"
)
RECORD_HASH_ALGORITHM = "sha256-canonical-json-stage2-v6-record-v1"
ORDERED_RECORD_HASH_ALGORITHM = (
    "sha256-canonical-json-ordered-stage2-v6-record-digests-v1"
)
BUDGET_COUNT_FORMULA = "(numerator * total_native_pixels) // denominator"
EPISODE_WEIGHTING = "equal_window"
COMMIT_PUBLICATION_ORDER = "jsonl_then_manifest_then_commit_last"

_SHA256_HEX = frozenset("0123456789abcdef")
_CAPABILITY_TOKEN = object()

_IDENTITY_FIELDS = frozenset(
    {
        "canonical_id",
        "image_id",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
    }
)
_FOUR_BOUNDARY_FIELDS = (
    "canonical_id",
    "original_image_sha256",
    "near_duplicate_cluster_id_or_unique_sentinel",
    "exclusion_group_id",
)
_CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "context_package_id",
        "expected_role",
        "episode_role",
        "outer_fold_id",
        "outer_target",
        "source_domain",
        "base_seed",
        "derived_seed",
        "geometry",
        "window_index",
        "window_id",
        "context_records",
        "query_identity_records",
        "context_full_identity_sha256",
        "query_full_identity_sha256",
        "bootstrap_query_identity_sha256",
        "window_identity_sha256",
        "context_statistics",
        "guardrails",
    }
)
_CONTEXT_STATISTICS_FIELDS = frozenset(
    {
        "feature_names",
        "feature_dim",
        "dtype",
        "values",
        "vector_sha256_algorithm",
        "vector_sha256",
    }
)
_CONTEXT_GUARDRAIL_FIELDS = frozenset(
    {
        "context_labels_accessed",
        "query_scores_accessed",
        "query_labels_accessed",
        "query_consumed_by_inference",
    }
)


class Stage2CrossfitSchemaV6Error(ValueError):
    """A v6 payload, identity, bundle, or capability failed closed."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2CrossfitSchemaV6Error(
            f"non-canonical JSON value: {error}"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_document_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2CrossfitSchemaV6Error(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _nonfinite_guard(value: str) -> None:
    raise Stage2CrossfitSchemaV6Error(f"non-finite JSON number: {value}")


def parse_json_bytes(data: bytes, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_guard,
            parse_constant=_nonfinite_guard,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2CrossfitSchemaV6Error(
            f"invalid {name}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _strict_fields(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise Stage2CrossfitSchemaV6Error(
            f"{name} fields differ; missing={sorted(fields-observed)}, "
            f"extra={sorted(observed-fields)}"
        )
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must be an exact int >= {minimum}"
        )
    return value


def _strict_bool(value: Any, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must be exactly {expected}"
        )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must be nonempty trimmed text"
        )
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA256_HEX
    ):
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must be lowercase SHA-256"
        )
    return value


def _relative_path(value: Any, name: str) -> str:
    raw = _text(value, name)
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or raw != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must be canonical repository-relative POSIX"
        )
    lowered = raw.lower().replace("-", "_")
    if "official_test" in lowered or "officialtest" in lowered:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} may not reference official test"
        )
    return raw


def _identity_record(value: Any, name: str) -> dict[str, Any]:
    row = _strict_fields(value, _IDENTITY_FIELDS, name)
    return {
        "canonical_id": _text(row["canonical_id"], f"{name}.canonical_id"),
        "image_id": _text(row["image_id"], f"{name}.image_id"),
        "original_image_sha256": _sha256(
            row["original_image_sha256"],
            f"{name}.original_image_sha256",
        ),
        "exclusion_group_id": _text(
            row["exclusion_group_id"],
            f"{name}.exclusion_group_id",
        ),
        "near_duplicate_cluster_id_or_unique_sentinel": _text(
            row["near_duplicate_cluster_id_or_unique_sentinel"],
            f"{name}.near_duplicate_cluster_id_or_unique_sentinel",
        ),
        "source_role_record_index": _strict_int(
            row["source_role_record_index"],
            f"{name}.source_role_record_index",
        ),
    }


def full_identity_projection(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("identity records must be a sequence")
    return [
        {
            "ordinal": index,
            **_identity_record(row, f"records[{index}]"),
        }
        for index, row in enumerate(records)
    ]


def full_identity_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(full_identity_projection(records))


def bootstrap_query_identity_projection(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "image_id": _identity_record(
                row, f"records[{index}]"
            )["image_id"],
            "original_image_sha256": _identity_record(
                row, f"records[{index}]"
            )["original_image_sha256"],
        }
        for index, row in enumerate(records)
    ]


def bootstrap_query_identity_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_json_sha256(
        bootstrap_query_identity_projection(records)
    )


def window_identity_sha256(
    window_id: str,
    context_full_identity_sha256: str,
    bootstrap_query_identity_sha256_value: str,
) -> str:
    return canonical_json_sha256(
        {
            "window_id": _text(window_id, "window_id"),
            "context_full_identity_sha256": _sha256(
                context_full_identity_sha256,
                "context_full_identity_sha256",
            ),
            "bootstrap_query_identity_sha256": _sha256(
                bootstrap_query_identity_sha256_value,
                "bootstrap_query_identity_sha256",
            ),
        }
    )


def _assert_four_identity_boundary(
    records: Sequence[Mapping[str, Any]], name: str
) -> None:
    for field in _FOUR_BOUNDARY_FIELDS:
        values = [str(row[field]) for row in records]
        if len(values) != len(set(values)):
            raise Stage2CrossfitSchemaV6Error(
                f"{name} duplicate identity at {field}"
            )
    source_indices = [int(row["source_role_record_index"]) for row in records]
    if len(source_indices) != len(set(source_indices)):
        raise Stage2CrossfitSchemaV6Error(
            f"{name} duplicate source_role_record_index"
        )


def _canonical_float32_vector(
    values: Any, name: str
) -> tuple[tuple[float, ...], bytes]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != FEATURE_DIM
    ):
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must contain exactly {FEATURE_DIM} values"
        )
    parsed: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Stage2CrossfitSchemaV6Error(
                f"{name}[{index}] must be finite numeric"
            )
        number = float(value)
        if not math.isfinite(number):
            raise Stage2CrossfitSchemaV6Error(
                f"{name}[{index}] must be finite"
            )
        parsed.append(number)
    try:
        packed = struct.pack(f"<{FEATURE_DIM}f", *parsed)
    except (OverflowError, struct.error) as error:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} cannot be represented as float32"
        ) from error
    rounded = tuple(struct.unpack(f"<{FEATURE_DIM}f", packed))
    if tuple(parsed) != rounded:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must contain canonical float32 values"
        )
    return rounded, packed


def _canonical_context_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    top = _strict_fields(payload, _CONTEXT_FIELDS, "context-v2")
    exact = {
        "schema_version": CONTEXT_SCHEMA,
        "artifact_type": CONTEXT_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_UNLABELED",
    }
    for field, expected in exact.items():
        if top[field] != expected:
            raise Stage2CrossfitSchemaV6Error(
                f"context-v2 {field} mismatch"
            )
    _strict_bool(top["development_only"], "development_only", True)
    _strict_bool(
        top["official_test_accessed"], "official_test_accessed", False
    )
    expected_role = _text(top["expected_role"], "expected_role")
    if expected_role not in ROLE_TO_EPISODE:
        raise Stage2CrossfitSchemaV6Error("unsupported context role")
    episode_role = ROLE_TO_EPISODE[expected_role]
    if top["episode_role"] != episode_role:
        raise Stage2CrossfitSchemaV6Error("context role mapping mismatch")
    outer_fold = _text(top["outer_fold_id"], "outer_fold_id")
    outer_target = _text(top["outer_target"], "outer_target")
    if OUTER_TARGETS.get(outer_fold) != outer_target:
        raise Stage2CrossfitSchemaV6Error("outer fold/target mismatch")
    source_domain = _text(top["source_domain"], "source_domain")
    if source_domain not in ALL_DOMAINS:
        raise Stage2CrossfitSchemaV6Error("unknown source_domain")
    if expected_role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT:
        if source_domain != outer_target:
            raise Stage2CrossfitSchemaV6Error(
                "outer context must use only the outer target"
            )
    elif source_domain == outer_target:
        raise Stage2CrossfitSchemaV6Error(
            "source context leaks the outer target"
        )
    base_seed = _strict_int(top["base_seed"], "base_seed")
    if base_seed not in BASE_SEEDS:
        raise Stage2CrossfitSchemaV6Error("unsupported base_seed")
    derived_seed = _strict_int(top["derived_seed"], "derived_seed")

    geometry = validate_stage2_variable_query_geometry(top["geometry"])
    if _plain(top["geometry"]) != geometry:
        raise Stage2CrossfitSchemaV6Error(
            "context geometry is not canonical replay"
        )
    window_index = _strict_int(top["window_index"], "window_index")
    if window_index >= geometry["window_count"]:
        raise Stage2CrossfitSchemaV6Error(
            "window_index exceeds geometry"
        )
    window = geometry["windows"][window_index]
    window_id = _text(top["window_id"], "window_id")
    raw_context = top["context_records"]
    raw_query = top["query_identity_records"]
    if (
        not isinstance(raw_context, list)
        or len(raw_context) != CONTEXT_SIZE
        or not isinstance(raw_query, list)
        or len(raw_query) != window["query_size"]
    ):
        raise Stage2CrossfitSchemaV6Error(
            "context-v2 C14/dynamic-Q cardinality mismatch"
        )
    context = [
        _identity_record(row, f"context_records[{index}]")
        for index, row in enumerate(raw_context)
    ]
    query = [
        _identity_record(row, f"query_identity_records[{index}]")
        for index, row in enumerate(raw_query)
    ]
    expected_context_indices = list(
        range(window["context_start"], window["context_stop"])
    )
    expected_query_indices = list(
        range(window["query_start"], window["query_stop"])
    )
    if [row["source_role_record_index"] for row in context] != (
        expected_context_indices
    ):
        raise Stage2CrossfitSchemaV6Error(
            "context indices differ from geometry replay"
        )
    if [row["source_role_record_index"] for row in query] != (
        expected_query_indices
    ):
        raise Stage2CrossfitSchemaV6Error(
            "query indices differ from geometry replay"
        )
    _assert_four_identity_boundary(
        [*context, *query], "context/query window"
    )
    context_sha = full_identity_sha256(context)
    query_sha = full_identity_sha256(query)
    bootstrap_sha = bootstrap_query_identity_sha256(query)
    window_sha = window_identity_sha256(
        window_id, context_sha, bootstrap_sha
    )
    hashes = {
        "context_full_identity_sha256": context_sha,
        "query_full_identity_sha256": query_sha,
        "bootstrap_query_identity_sha256": bootstrap_sha,
        "window_identity_sha256": window_sha,
    }
    for field, expected in hashes.items():
        if top[field] != expected:
            raise Stage2CrossfitSchemaV6Error(
                f"context-v2 {field} mismatch"
            )

    stats = _strict_fields(
        top["context_statistics"],
        _CONTEXT_STATISTICS_FIELDS,
        "context_statistics",
    )
    if stats["feature_names"] != list(FEATURE_NAMES):
        raise Stage2CrossfitSchemaV6Error(
            "context feature names mismatch"
        )
    if _strict_int(stats["feature_dim"], "feature_dim") != FEATURE_DIM:
        raise Stage2CrossfitSchemaV6Error(
            "context feature_dim mismatch"
        )
    if stats["dtype"] != "float32":
        raise Stage2CrossfitSchemaV6Error(
            "context feature dtype mismatch"
        )
    if stats["vector_sha256_algorithm"] != FLOAT32_VECTOR_ALGORITHM:
        raise Stage2CrossfitSchemaV6Error(
            "context vector algorithm mismatch"
        )
    vector, vector_bytes = _canonical_float32_vector(
        stats["values"], "context_statistics.values"
    )
    vector_sha = sha256_bytes(vector_bytes)
    if stats["vector_sha256"] != vector_sha:
        raise Stage2CrossfitSchemaV6Error(
            "context vector SHA-256 mismatch"
        )

    guardrails = _strict_fields(
        top["guardrails"],
        _CONTEXT_GUARDRAIL_FIELDS,
        "context.guardrails",
    )
    for field in _CONTEXT_GUARDRAIL_FIELDS:
        _strict_bool(
            guardrails[field], f"context.guardrails.{field}", False
        )

    identity_preimage = {
        "schema_version": CONTEXT_SCHEMA,
        "expected_role": expected_role,
        "outer_fold_id": outer_fold,
        "source_domain": source_domain,
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "geometry_sha256": canonical_json_sha256(geometry),
        "window_index": window_index,
        "window_identity_sha256": window_sha,
        "context_feature_vector_sha256": vector_sha,
    }
    package_id = canonical_json_sha256(identity_preimage)
    if top["context_package_id"] != package_id:
        raise Stage2CrossfitSchemaV6Error(
            "context_package_id mismatch"
        )
    return {
        "schema_version": CONTEXT_SCHEMA,
        "artifact_type": CONTEXT_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_UNLABELED",
        "development_only": True,
        "official_test_accessed": False,
        "context_package_id": package_id,
        "expected_role": expected_role,
        "episode_role": episode_role,
        "outer_fold_id": outer_fold,
        "outer_target": outer_target,
        "source_domain": source_domain,
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "geometry": geometry,
        "window_index": window_index,
        "window_id": window_id,
        "context_records": context,
        "query_identity_records": query,
        **hashes,
        "context_statistics": {
            "feature_names": list(FEATURE_NAMES),
            "feature_dim": FEATURE_DIM,
            "dtype": "float32",
            "values": list(vector),
            "vector_sha256_algorithm": FLOAT32_VECTOR_ALGORITHM,
            "vector_sha256": vector_sha,
        },
        "guardrails": {
            field: False for field in sorted(_CONTEXT_GUARDRAIL_FIELDS)
        },
    }


@dataclass(frozen=True, init=False)
class VerifiedStage2ContextV2:
    """Immutable context capability, optionally rooted in a verified window."""

    payload: Mapping[str, Any]
    canonical_payload: bytes
    payload_sha256: str
    variable_query_window: VerifiedStage2VariableQueryWindow | None
    _capability: object

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        canonical_payload: bytes,
        payload_sha256: str,
        _capability: object,
        variable_query_window: VerifiedStage2VariableQueryWindow | None = None,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError("VerifiedStage2ContextV2 is verifier-only")
        object.__setattr__(self, "payload", _freeze(payload))
        object.__setattr__(self, "canonical_payload", bytes(canonical_payload))
        object.__setattr__(self, "payload_sha256", payload_sha256)
        if variable_query_window is not None:
            assert_verified_stage2_variable_query_window(
                variable_query_window
            )
        object.__setattr__(
            self, "variable_query_window", variable_query_window
        )
        object.__setattr__(self, "_capability", _capability)


@dataclass(frozen=True, init=False)
class VerifiedContextInferenceMaterialV2:
    """Query-free projection accepted by RC5 inference adapters."""

    context_package_id: str
    context_full_identity_sha256: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    feature_vector_sha256: str
    source_query_consumed: bool
    _capability: object

    def __init__(
        self,
        *,
        context_package_id: str,
        context_full_identity_sha256: str,
        feature_names: tuple[str, ...],
        feature_values: tuple[float, ...],
        feature_vector_sha256: str,
        _capability: object,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError(
                "VerifiedContextInferenceMaterialV2 is verifier-only"
            )
        object.__setattr__(self, "context_package_id", context_package_id)
        object.__setattr__(
            self,
            "context_full_identity_sha256",
            context_full_identity_sha256,
        )
        object.__setattr__(self, "feature_names", tuple(feature_names))
        object.__setattr__(self, "feature_values", tuple(feature_values))
        object.__setattr__(
            self, "feature_vector_sha256", feature_vector_sha256
        )
        object.__setattr__(self, "source_query_consumed", False)
        object.__setattr__(self, "_capability", _capability)


def verify_context_payload_v2(
    payload: Mapping[str, Any],
) -> VerifiedStage2ContextV2:
    canonical = _canonical_context_payload(payload)
    data = canonical_json_bytes(canonical)
    if canonical_json_bytes(payload) != data:
        raise Stage2CrossfitSchemaV6Error(
            "context payload is not canonical replay"
        )
    return VerifiedStage2ContextV2(
        payload=canonical,
        canonical_payload=data,
        payload_sha256=sha256_bytes(data),
        _capability=_CAPABILITY_TOKEN,
    )


def assert_verified_context_v2(value: object) -> VerifiedStage2ContextV2:
    if (
        not isinstance(value, VerifiedStage2ContextV2)
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
    ):
        raise TypeError("a verifier-issued context-v2 capability is required")
    if value.variable_query_window is not None:
        assert_verified_stage2_variable_query_window(
            value.variable_query_window
        )
    return value


def assert_verified_context_inference_material_v2(
    value: object,
) -> VerifiedContextInferenceMaterialV2:
    if (
        not isinstance(value, VerifiedContextInferenceMaterialV2)
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
    ):
        raise TypeError(
            "verifier-issued context inference material-v2 is required"
        )
    if value.source_query_consumed is not False:
        raise TypeError("context inference material consumed source query")
    return value


def context_inference_material_v2(
    context: VerifiedStage2ContextV2,
) -> VerifiedContextInferenceMaterialV2:
    verified = assert_verified_context_v2(context)
    payload = verified.payload
    stats = payload["context_statistics"]
    return VerifiedContextInferenceMaterialV2(
        context_package_id=str(payload["context_package_id"]),
        context_full_identity_sha256=str(
            payload["context_full_identity_sha256"]
        ),
        feature_names=tuple(str(item) for item in stats["feature_names"]),
        feature_values=tuple(float(item) for item in stats["values"]),
        feature_vector_sha256=str(stats["vector_sha256"]),
        _capability=_CAPABILITY_TOKEN,
    )


def build_context_payload_v2(
    *,
    expected_role: str,
    outer_fold_id: str,
    outer_target: str,
    source_domain: str,
    base_seed: int,
    derived_seed: int,
    geometry: Mapping[str, Any],
    window_index: int,
    window_id: str,
    context_records: Sequence[Mapping[str, Any]],
    query_identity_records: Sequence[Mapping[str, Any]],
    context_feature_values: Sequence[float],
) -> dict[str, Any]:
    canonical_geometry = validate_stage2_variable_query_geometry(geometry)
    context = [
        _identity_record(row, f"context_records[{index}]")
        for index, row in enumerate(context_records)
    ]
    query = [
        _identity_record(row, f"query_identity_records[{index}]")
        for index, row in enumerate(query_identity_records)
    ]
    vector, packed = _canonical_float32_vector(
        context_feature_values, "context_feature_values"
    )
    context_sha = full_identity_sha256(context)
    query_sha = full_identity_sha256(query)
    bootstrap_sha = bootstrap_query_identity_sha256(query)
    window_sha = window_identity_sha256(
        window_id, context_sha, bootstrap_sha
    )
    vector_sha = sha256_bytes(packed)
    package_id = canonical_json_sha256(
        {
            "schema_version": CONTEXT_SCHEMA,
            "expected_role": expected_role,
            "outer_fold_id": outer_fold_id,
            "source_domain": source_domain,
            "base_seed": base_seed,
            "derived_seed": derived_seed,
            "geometry_sha256": canonical_json_sha256(canonical_geometry),
            "window_index": window_index,
            "window_identity_sha256": window_sha,
            "context_feature_vector_sha256": vector_sha,
        }
    )
    payload = {
        "schema_version": CONTEXT_SCHEMA,
        "artifact_type": CONTEXT_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_UNLABELED",
        "development_only": True,
        "official_test_accessed": False,
        "context_package_id": package_id,
        "expected_role": expected_role,
        "episode_role": ROLE_TO_EPISODE.get(expected_role),
        "outer_fold_id": outer_fold_id,
        "outer_target": outer_target,
        "source_domain": source_domain,
        "base_seed": base_seed,
        "derived_seed": derived_seed,
        "geometry": canonical_geometry,
        "window_index": window_index,
        "window_id": window_id,
        "context_records": context,
        "query_identity_records": query,
        "context_full_identity_sha256": context_sha,
        "query_full_identity_sha256": query_sha,
        "bootstrap_query_identity_sha256": bootstrap_sha,
        "window_identity_sha256": window_sha,
        "context_statistics": {
            "feature_names": list(FEATURE_NAMES),
            "feature_dim": FEATURE_DIM,
            "dtype": "float32",
            "values": list(vector),
            "vector_sha256_algorithm": FLOAT32_VECTOR_ALGORITHM,
            "vector_sha256": vector_sha,
        },
        "guardrails": {
            field: False for field in sorted(_CONTEXT_GUARDRAIL_FIELDS)
        },
    }
    return _plain(verify_context_payload_v2(payload).payload)


def context_from_verified_variable_query_window_v2(
    variable_query_window: VerifiedStage2VariableQueryWindow,
    *,
    expected_role: str,
    base_seed: int,
    derived_seed: int,
    window_index: int,
    context_feature_values: Sequence[float],
) -> VerifiedStage2ContextV2:
    """Build a context rooted in an externally verified variable-Q manifest.

    The adapter projects metadata identities only.  In particular, it does
    not expose or consume query scores, query labels, or observed results.
    """

    verified_window = assert_verified_stage2_variable_query_window(
        variable_query_window
    )
    role = _text(expected_role, "expected_role")
    if role not in ROLE_TO_EPISODE:
        raise Stage2CrossfitSchemaV6Error("unsupported context role")
    source = verified_window.payload
    if source["episode_role"] != ROLE_TO_EPISODE[role]:
        raise Stage2CrossfitSchemaV6Error(
            "variable-Q episode_role/context role mismatch"
        )
    index = _strict_int(window_index, "window_index")
    if index >= len(verified_window.windows):
        raise Stage2CrossfitSchemaV6Error(
            "window_index exceeds verified variable-Q windows"
        )
    raw_window = verified_window.windows[index]

    def identity(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            field: record[field]
            for field in _IDENTITY_FIELDS
        }

    payload = build_context_payload_v2(
        expected_role=role,
        outer_fold_id=str(source["outer_fold_id"]),
        outer_target=str(source["outer_target_domain"]),
        source_domain=str(source["domain"]),
        base_seed=base_seed,
        derived_seed=derived_seed,
        geometry=_plain(source["geometry"]),
        window_index=index,
        window_id=str(raw_window["window_id"]),
        context_records=[
            identity(record) for record in raw_window["context_records"]
        ],
        query_identity_records=[
            identity(record) for record in raw_window["query_records"]
        ],
        context_feature_values=context_feature_values,
    )
    pure = verify_context_payload_v2(payload)
    return VerifiedStage2ContextV2(
        payload=pure.payload,
        canonical_payload=pure.canonical_payload,
        payload_sha256=pure.payload_sha256,
        _capability=_CAPABILITY_TOKEN,
        variable_query_window=verified_window,
    )


_CONTEXT_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "payload_sha256",
        "context_package_id",
        "window_identity_sha256",
        "query_full_identity_sha256",
    }
)
_ANCHOR_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "path",
        "sha256",
        "anchor_identity_sha256",
        "anchor_payload_sha256",
        "context_full_identity_sha256",
        "context_probability_content_sha256",
        "context_size",
        "total_context_pixels",
        "budget_rationals",
        "threshold_representation_schema",
        "threshold_semantics",
    }
)
_ORACLE_CURVE_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "curve_path",
        "curve_sha256",
        "manifest_path",
        "manifest_sha256",
        "curve_rows_sha256",
        "oracle_rows_sha256",
        "query_full_identity_sha256",
        "query_size",
        "total_native_pixels",
        "budget_rationals",
        "budget_counts",
        "threshold_representation_schema",
        "threshold_semantics",
        "budget_count_formula",
        "float_budget_counts_forbidden",
    }
)
_BUDGET_FIELDS = frozenset({"numerator", "denominator"})
_BUDGET_COUNT_FIELDS = frozenset(
    {
        "numerator",
        "denominator",
        "allowed_false_positive_pixels",
    }
)
_PRELABEL_DECISION_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "path",
        "sha256",
        "decision_set_content_sha256",
        "context_payload_sha256",
        "window_identity_sha256",
        "query_full_identity_sha256",
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "derived_seed",
    }
)
_SUPERVISION_FIELDS = frozenset(
    {
        "query_label_usage",
        "query_size_from_geometry",
        "episode_weighting",
        "threshold_semantics",
        "budget_count_formula",
        "float_budget_counts_forbidden",
    }
)
_EPISODE_GUARDRAIL_FIELDS = frozenset(
    {
        "context_labels_accessed",
        "query_labels_accessed",
        "decision_made_before_outer_labels",
        "reject_supported",
        "fallback_used",
        "official_test_accessed",
    }
)
_EPISODE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "episode_id",
        "episode_index",
        "collection_role",
        "episode_role",
        "outer_fold_id",
        "outer_target",
        "source_domain",
        "base_seed",
        "derived_seed",
        "episode_weighting",
        "query_size",
        "context_binding",
        "context_payload",
        "anchor_binding",
        "oracle_curve_binding",
        "budget_rationals",
        "threshold_semantics",
        "prelabel_decision_binding",
        "supervision_contract",
        "guardrails",
    }
)


def _budget_rationals(value: Any, name: str) -> list[dict[str, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != len(
        BUDGET_RATIONALS
    ):
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must contain the three frozen rationals"
        )
    result: list[dict[str, int]] = []
    for index, raw in enumerate(value):
        row = _strict_fields(raw, _BUDGET_FIELDS, f"{name}[{index}]")
        result.append(
            {
                "numerator": _strict_int(
                    row["numerator"], f"{name}[{index}].numerator", minimum=1
                ),
                "denominator": _strict_int(
                    row["denominator"],
                    f"{name}[{index}].denominator",
                    minimum=2,
                ),
            }
        )
    if result != [dict(row) for row in BUDGET_RATIONALS]:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} differs from the frozen exact rationals"
        )
    return result


def _canonical_context_binding(
    value: Any, context: VerifiedStage2ContextV2
) -> dict[str, Any]:
    binding = _strict_fields(
        value, _CONTEXT_BINDING_FIELDS, "context_binding"
    )
    expected = {
        "schema_version": CONTEXT_SCHEMA,
        "payload_sha256": context.payload_sha256,
        "context_package_id": context.payload["context_package_id"],
        "window_identity_sha256": context.payload[
            "window_identity_sha256"
        ],
        "query_full_identity_sha256": context.payload[
            "query_full_identity_sha256"
        ],
    }
    if _plain(binding) != expected:
        raise Stage2CrossfitSchemaV6Error(
            "episode context_binding mismatch"
        )
    return expected


def _canonical_anchor_binding(
    value: Any, context: VerifiedStage2ContextV2
) -> dict[str, Any]:
    binding = _strict_fields(
        value, _ANCHOR_BINDING_FIELDS, "anchor_binding"
    )
    if binding["schema_version"] != ANCHOR_SCHEMA:
        raise Stage2CrossfitSchemaV6Error("anchor schema mismatch")
    result = {
        "schema_version": ANCHOR_SCHEMA,
        "path": _relative_path(binding["path"], "anchor.path"),
        "sha256": _sha256(binding["sha256"], "anchor.sha256"),
        "anchor_identity_sha256": _sha256(
            binding["anchor_identity_sha256"],
            "anchor.anchor_identity_sha256",
        ),
        "anchor_payload_sha256": _sha256(
            binding["anchor_payload_sha256"],
            "anchor.anchor_payload_sha256",
        ),
        "context_full_identity_sha256": _sha256(
            binding["context_full_identity_sha256"],
            "anchor.context_full_identity_sha256",
        ),
        "context_probability_content_sha256": _sha256(
            binding["context_probability_content_sha256"],
            "anchor.context_probability_content_sha256",
        ),
        "context_size": _strict_int(
            binding["context_size"], "anchor.context_size", minimum=1
        ),
        "total_context_pixels": _strict_int(
            binding["total_context_pixels"],
            "anchor.total_context_pixels",
            minimum=1,
        ),
        "budget_rationals": _budget_rationals(
            binding["budget_rationals"], "anchor.budget_rationals"
        ),
        "threshold_representation_schema": binding[
            "threshold_representation_schema"
        ],
        "threshold_semantics": binding["threshold_semantics"],
    }
    if result["context_size"] != CONTEXT_SIZE:
        raise Stage2CrossfitSchemaV6Error("anchor context_size mismatch")
    if result["context_full_identity_sha256"] != context.payload[
        "context_full_identity_sha256"
    ]:
        raise Stage2CrossfitSchemaV6Error(
            "anchor/context identity mismatch"
        )
    if result["threshold_representation_schema"] != EATC_V2_SCHEMA:
        raise Stage2CrossfitSchemaV6Error(
            "anchor is not bound to EATC-v2"
        )
    if result["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS:
        raise Stage2CrossfitSchemaV6Error(
            "anchor threshold semantics must be strict >"
        )
    return result


def _canonical_oracle_curve_binding(
    value: Any, context: VerifiedStage2ContextV2
) -> dict[str, Any]:
    binding = _strict_fields(
        value,
        _ORACLE_CURVE_BINDING_FIELDS,
        "oracle_curve_binding",
    )
    if binding["schema_version"] != ORACLE_CURVE_BINDING_SCHEMA:
        raise Stage2CrossfitSchemaV6Error(
            "oracle/curve binding schema mismatch"
        )
    query_size = len(context.payload["query_identity_records"])
    total_pixels = _strict_int(
        binding["total_native_pixels"],
        "oracle_curve.total_native_pixels",
        minimum=1,
    )
    budgets = _budget_rationals(
        binding["budget_rationals"],
        "oracle_curve.budget_rationals",
    )
    raw_counts = binding["budget_counts"]
    if not isinstance(raw_counts, (list, tuple)) or len(raw_counts) != len(
        budgets
    ):
        raise Stage2CrossfitSchemaV6Error(
            "oracle_curve budget_counts cardinality mismatch"
        )
    counts: list[dict[str, int]] = []
    for index, (raw, budget) in enumerate(
        zip(raw_counts, budgets, strict=True)
    ):
        row = _strict_fields(
            raw, _BUDGET_COUNT_FIELDS, f"budget_counts[{index}]"
        )
        parsed = {
            "numerator": _strict_int(
                row["numerator"],
                f"budget_counts[{index}].numerator",
                minimum=1,
            ),
            "denominator": _strict_int(
                row["denominator"],
                f"budget_counts[{index}].denominator",
                minimum=2,
            ),
            "allowed_false_positive_pixels": _strict_int(
                row["allowed_false_positive_pixels"],
                f"budget_counts[{index}].allowed_false_positive_pixels",
            ),
        }
        expected_count = (
            budget["numerator"] * total_pixels
        ) // budget["denominator"]
        if (
            parsed["numerator"] != budget["numerator"]
            or parsed["denominator"] != budget["denominator"]
            or parsed["allowed_false_positive_pixels"] != expected_count
        ):
            raise Stage2CrossfitSchemaV6Error(
                "oracle_curve budget count differs from exact integer replay"
            )
        counts.append(parsed)
    result = {
        "schema_version": ORACLE_CURVE_BINDING_SCHEMA,
        "curve_path": _relative_path(
            binding["curve_path"], "oracle_curve.curve_path"
        ),
        "curve_sha256": _sha256(
            binding["curve_sha256"], "oracle_curve.curve_sha256"
        ),
        "manifest_path": _relative_path(
            binding["manifest_path"], "oracle_curve.manifest_path"
        ),
        "manifest_sha256": _sha256(
            binding["manifest_sha256"],
            "oracle_curve.manifest_sha256",
        ),
        "curve_rows_sha256": _sha256(
            binding["curve_rows_sha256"],
            "oracle_curve.curve_rows_sha256",
        ),
        "oracle_rows_sha256": _sha256(
            binding["oracle_rows_sha256"],
            "oracle_curve.oracle_rows_sha256",
        ),
        "query_full_identity_sha256": _sha256(
            binding["query_full_identity_sha256"],
            "oracle_curve.query_full_identity_sha256",
        ),
        "query_size": _strict_int(
            binding["query_size"],
            "oracle_curve.query_size",
            minimum=1,
        ),
        "total_native_pixels": total_pixels,
        "budget_rationals": budgets,
        "budget_counts": counts,
        "threshold_representation_schema": binding[
            "threshold_representation_schema"
        ],
        "threshold_semantics": binding["threshold_semantics"],
        "budget_count_formula": binding["budget_count_formula"],
        "float_budget_counts_forbidden": binding[
            "float_budget_counts_forbidden"
        ],
    }
    if result["query_size"] != query_size:
        raise Stage2CrossfitSchemaV6Error(
            "oracle_curve query_size differs from geometry"
        )
    if result["query_full_identity_sha256"] != context.payload[
        "query_full_identity_sha256"
    ]:
        raise Stage2CrossfitSchemaV6Error(
            "oracle_curve/query identity mismatch"
        )
    if result["threshold_representation_schema"] != EATC_V2_SCHEMA:
        raise Stage2CrossfitSchemaV6Error(
            "oracle/curve is not bound to EATC-v2"
        )
    if result["threshold_semantics"] != STRICT_THRESHOLD_SEMANTICS:
        raise Stage2CrossfitSchemaV6Error(
            "oracle/curve threshold semantics must be strict >"
        )
    if result["budget_count_formula"] != BUDGET_COUNT_FORMULA:
        raise Stage2CrossfitSchemaV6Error(
            "oracle/curve budget formula mismatch"
        )
    _strict_bool(
        result["float_budget_counts_forbidden"],
        "oracle_curve.float_budget_counts_forbidden",
        True,
    )
    return result


def _canonical_prelabel_decision_binding(
    value: Any,
    context: VerifiedStage2ContextV2,
) -> dict[str, Any] | None:
    role = context.payload["episode_role"]
    if role in {STAGE2_OOF_FIT, SOURCE_DIAGNOSTIC_VALIDATION}:
        if value is not None:
            raise Stage2CrossfitSchemaV6Error(
                "source episode prelabel decision binding must be null"
            )
        return None
    binding = _strict_fields(
        value,
        _PRELABEL_DECISION_BINDING_FIELDS,
        "prelabel_decision_binding",
    )
    result = {
        "schema_version": binding["schema_version"],
        "path": _relative_path(binding["path"], "prelabel.path"),
        "sha256": _sha256(binding["sha256"], "prelabel.sha256"),
        "decision_set_content_sha256": _sha256(
            binding["decision_set_content_sha256"],
            "prelabel.decision_set_content_sha256",
        ),
        "context_payload_sha256": _sha256(
            binding["context_payload_sha256"],
            "prelabel.context_payload_sha256",
        ),
        "window_identity_sha256": _sha256(
            binding["window_identity_sha256"],
            "prelabel.window_identity_sha256",
        ),
        "query_full_identity_sha256": _sha256(
            binding["query_full_identity_sha256"],
            "prelabel.query_full_identity_sha256",
        ),
        "outer_fold_id": _text(
            binding["outer_fold_id"], "prelabel.outer_fold_id"
        ),
        "outer_target": _text(
            binding["outer_target"], "prelabel.outer_target"
        ),
        "base_seed": _strict_int(
            binding["base_seed"], "prelabel.base_seed"
        ),
        "derived_seed": _strict_int(
            binding["derived_seed"], "prelabel.derived_seed"
        ),
    }
    expected = {
        "schema_version": PRELABEL_DECISION_BINDING_SCHEMA,
        "context_payload_sha256": context.payload_sha256,
        "window_identity_sha256": context.payload[
            "window_identity_sha256"
        ],
        "query_full_identity_sha256": context.payload[
            "query_full_identity_sha256"
        ],
        "outer_fold_id": context.payload["outer_fold_id"],
        "outer_target": context.payload["outer_target"],
        "base_seed": context.payload["base_seed"],
        "derived_seed": context.payload["derived_seed"],
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            raise Stage2CrossfitSchemaV6Error(
                f"prelabel decision {field} mismatch"
            )
    return result


def _episode_identity_preimage(
    *,
    episode_index: int,
    context: VerifiedStage2ContextV2,
    anchor_binding: Mapping[str, Any],
    oracle_curve_binding: Mapping[str, Any],
    prelabel_decision_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": EPISODE_SCHEMA,
        "episode_index": episode_index,
        "context_payload_sha256": context.payload_sha256,
        "anchor_payload_sha256": anchor_binding[
            "anchor_payload_sha256"
        ],
        "curve_manifest_sha256": oracle_curve_binding[
            "manifest_sha256"
        ],
        "oracle_rows_sha256": oracle_curve_binding[
            "oracle_rows_sha256"
        ],
        "prelabel_decision_sha256": (
            None
            if prelabel_decision_binding is None
            else prelabel_decision_binding["sha256"]
        ),
    }


def _canonical_episode_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], VerifiedStage2ContextV2]:
    top = _strict_fields(payload, _EPISODE_FIELDS, "episode-v6")
    exact = {
        "schema_version": EPISODE_SCHEMA,
        "artifact_type": EPISODE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_COMPLETE",
        "episode_weighting": EPISODE_WEIGHTING,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
    }
    for field, expected in exact.items():
        if top[field] != expected:
            raise Stage2CrossfitSchemaV6Error(
                f"episode-v6 {field} mismatch"
            )
    _strict_bool(top["development_only"], "development_only", True)
    _strict_bool(
        top["official_test_accessed"], "official_test_accessed", False
    )
    episode_index = _strict_int(top["episode_index"], "episode_index")
    context = verify_context_payload_v2(top["context_payload"])
    cp = context.payload
    context_binding = _canonical_context_binding(
        top["context_binding"], context
    )
    for field in (
        "episode_role",
        "outer_fold_id",
        "outer_target",
        "source_domain",
        "base_seed",
        "derived_seed",
    ):
        if top[field] != cp[field]:
            raise Stage2CrossfitSchemaV6Error(
                f"episode/context {field} mismatch"
            )
    expected_collection = ROLE_TO_COLLECTION[cp["expected_role"]]
    if top["collection_role"] != expected_collection:
        raise Stage2CrossfitSchemaV6Error(
            "episode collection role mismatch"
        )
    query_size = _strict_int(top["query_size"], "query_size", minimum=1)
    if query_size != len(cp["query_identity_records"]):
        raise Stage2CrossfitSchemaV6Error(
            "episode query_size differs from geometry"
        )
    budgets = _budget_rationals(
        top["budget_rationals"], "episode.budget_rationals"
    )
    anchor = _canonical_anchor_binding(top["anchor_binding"], context)
    oracle = _canonical_oracle_curve_binding(
        top["oracle_curve_binding"], context
    )
    decision = _canonical_prelabel_decision_binding(
        top["prelabel_decision_binding"], context
    )
    supervision = _strict_fields(
        top["supervision_contract"],
        _SUPERVISION_FIELDS,
        "supervision_contract",
    )
    expected_label_usage = (
        "outer_target_sealed_prelabel_decision_evaluation_only"
        if cp["episode_role"] == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
        else "stage2_loss_and_exact_curve_replay_only"
    )
    expected_supervision = {
        "query_label_usage": expected_label_usage,
        "query_size_from_geometry": True,
        "episode_weighting": EPISODE_WEIGHTING,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "budget_count_formula": BUDGET_COUNT_FORMULA,
        "float_budget_counts_forbidden": True,
    }
    if _plain(supervision) != expected_supervision:
        raise Stage2CrossfitSchemaV6Error(
            "episode supervision contract mismatch"
        )
    guardrails = _strict_fields(
        top["guardrails"],
        _EPISODE_GUARDRAIL_FIELDS,
        "episode.guardrails",
    )
    expected_guardrails = {
        "context_labels_accessed": False,
        "query_labels_accessed": True,
        "decision_made_before_outer_labels": (
            cp["episode_role"]
            == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
        ),
        "reject_supported": False,
        "fallback_used": False,
        "official_test_accessed": False,
    }
    if _plain(guardrails) != expected_guardrails:
        raise Stage2CrossfitSchemaV6Error(
            "episode guardrail mismatch"
        )
    episode_id = canonical_json_sha256(
        _episode_identity_preimage(
            episode_index=episode_index,
            context=context,
            anchor_binding=anchor,
            oracle_curve_binding=oracle,
            prelabel_decision_binding=decision,
        )
    )
    if top["episode_id"] != episode_id:
        raise Stage2CrossfitSchemaV6Error("episode_id mismatch")
    return (
        {
            "schema_version": EPISODE_SCHEMA,
            "artifact_type": EPISODE_ARTIFACT_TYPE,
            "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_COMPLETE",
            "development_only": True,
            "official_test_accessed": False,
            "episode_id": episode_id,
            "episode_index": episode_index,
            "collection_role": expected_collection,
            "episode_role": cp["episode_role"],
            "outer_fold_id": cp["outer_fold_id"],
            "outer_target": cp["outer_target"],
            "source_domain": cp["source_domain"],
            "base_seed": cp["base_seed"],
            "derived_seed": cp["derived_seed"],
            "episode_weighting": EPISODE_WEIGHTING,
            "query_size": query_size,
            "context_binding": context_binding,
            "context_payload": _plain(cp),
            "anchor_binding": anchor,
            "oracle_curve_binding": oracle,
            "budget_rationals": budgets,
            "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
            "prelabel_decision_binding": decision,
            "supervision_contract": expected_supervision,
            "guardrails": expected_guardrails,
        },
        context,
    )


@dataclass(frozen=True, init=False)
class VerifiedStage2EpisodeV6:
    """Recursively immutable, verifier-issued episode capability."""

    payload: Mapping[str, Any]
    canonical_payload: bytes
    payload_sha256: str
    context: VerifiedStage2ContextV2
    _capability: object

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        canonical_payload: bytes,
        payload_sha256: str,
        context: VerifiedStage2ContextV2,
        _capability: object,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError("VerifiedStage2EpisodeV6 is verifier-only")
        object.__setattr__(self, "payload", _freeze(payload))
        object.__setattr__(self, "canonical_payload", bytes(canonical_payload))
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(
            self, "context", assert_verified_context_v2(context)
        )
        object.__setattr__(self, "_capability", _capability)

    @property
    def episode_id(self) -> str:
        return str(self.payload["episode_id"])


def verify_episode_payload_v6(
    payload: Mapping[str, Any],
) -> VerifiedStage2EpisodeV6:
    """Replay an episode and its context plus immutable artifact bindings.

    This pure verifier validates path/digest descriptors and their semantic
    relationships.  It does not open or transitively verify the bound anchor,
    oracle/curve, or pre-label decision files.
    """

    canonical, context = _canonical_episode_payload(payload)
    data = canonical_json_bytes(canonical)
    if canonical_json_bytes(payload) != data:
        raise Stage2CrossfitSchemaV6Error(
            "episode payload is not canonical replay"
        )
    return VerifiedStage2EpisodeV6(
        payload=canonical,
        canonical_payload=data,
        payload_sha256=sha256_bytes(data),
        context=context,
        _capability=_CAPABILITY_TOKEN,
    )


def assert_verified_episode_v6(value: object) -> VerifiedStage2EpisodeV6:
    if (
        not isinstance(value, VerifiedStage2EpisodeV6)
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
    ):
        raise TypeError("a verifier-issued episode-v6 capability is required")
    assert_verified_context_v2(value.context)
    return value


def make_anchor_binding_v6(
    *,
    context: VerifiedStage2ContextV2,
    path: str,
    sha256: str,
    anchor_identity_sha256: str,
    anchor_payload_sha256: str,
    context_probability_content_sha256: str,
    total_context_pixels: int,
) -> dict[str, Any]:
    verified = assert_verified_context_v2(context)
    raw = {
        "schema_version": ANCHOR_SCHEMA,
        "path": path,
        "sha256": sha256,
        "anchor_identity_sha256": anchor_identity_sha256,
        "anchor_payload_sha256": anchor_payload_sha256,
        "context_full_identity_sha256": verified.payload[
            "context_full_identity_sha256"
        ],
        "context_probability_content_sha256": (
            context_probability_content_sha256
        ),
        "context_size": CONTEXT_SIZE,
        "total_context_pixels": total_context_pixels,
        "budget_rationals": [dict(row) for row in BUDGET_RATIONALS],
        "threshold_representation_schema": EATC_V2_SCHEMA,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
    }
    return _canonical_anchor_binding(raw, verified)


def make_oracle_curve_binding_v2(
    *,
    context: VerifiedStage2ContextV2,
    curve_path: str,
    curve_sha256: str,
    manifest_path: str,
    manifest_sha256: str,
    curve_rows_sha256: str,
    oracle_rows_sha256: str,
    total_native_pixels: int,
) -> dict[str, Any]:
    verified = assert_verified_context_v2(context)
    total = _strict_int(
        total_native_pixels, "total_native_pixels", minimum=1
    )
    counts = [
        {
            "numerator": row["numerator"],
            "denominator": row["denominator"],
            "allowed_false_positive_pixels": (
                row["numerator"] * total
            )
            // row["denominator"],
        }
        for row in BUDGET_RATIONALS
    ]
    raw = {
        "schema_version": ORACLE_CURVE_BINDING_SCHEMA,
        "curve_path": curve_path,
        "curve_sha256": curve_sha256,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "curve_rows_sha256": curve_rows_sha256,
        "oracle_rows_sha256": oracle_rows_sha256,
        "query_full_identity_sha256": verified.payload[
            "query_full_identity_sha256"
        ],
        "query_size": len(verified.payload["query_identity_records"]),
        "total_native_pixels": total,
        "budget_rationals": [dict(row) for row in BUDGET_RATIONALS],
        "budget_counts": counts,
        "threshold_representation_schema": EATC_V2_SCHEMA,
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "budget_count_formula": BUDGET_COUNT_FORMULA,
        "float_budget_counts_forbidden": True,
    }
    return _canonical_oracle_curve_binding(raw, verified)


def make_prelabel_decision_binding_v1(
    *,
    context: VerifiedStage2ContextV2,
    path: str,
    sha256: str,
    decision_set_content_sha256: str,
) -> dict[str, Any]:
    verified = assert_verified_context_v2(context)
    if verified.payload["episode_role"] != (
        OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
    ):
        raise Stage2CrossfitSchemaV6Error(
            "source contexts cannot bind a prelabel decision"
        )
    raw = {
        "schema_version": PRELABEL_DECISION_BINDING_SCHEMA,
        "path": path,
        "sha256": sha256,
        "decision_set_content_sha256": decision_set_content_sha256,
        "context_payload_sha256": verified.payload_sha256,
        "window_identity_sha256": verified.payload[
            "window_identity_sha256"
        ],
        "query_full_identity_sha256": verified.payload[
            "query_full_identity_sha256"
        ],
        "outer_fold_id": verified.payload["outer_fold_id"],
        "outer_target": verified.payload["outer_target"],
        "base_seed": verified.payload["base_seed"],
        "derived_seed": verified.payload["derived_seed"],
    }
    result = _canonical_prelabel_decision_binding(raw, verified)
    assert result is not None
    return result


def build_episode_payload_v6(
    *,
    episode_index: int,
    context: VerifiedStage2ContextV2,
    anchor_binding: Mapping[str, Any],
    oracle_curve_binding: Mapping[str, Any],
    prelabel_decision_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    verified = assert_verified_context_v2(context)
    index = _strict_int(episode_index, "episode_index")
    anchor = _canonical_anchor_binding(anchor_binding, verified)
    oracle = _canonical_oracle_curve_binding(
        oracle_curve_binding, verified
    )
    decision = _canonical_prelabel_decision_binding(
        prelabel_decision_binding, verified
    )
    cp = verified.payload
    context_binding = {
        "schema_version": CONTEXT_SCHEMA,
        "payload_sha256": verified.payload_sha256,
        "context_package_id": cp["context_package_id"],
        "window_identity_sha256": cp["window_identity_sha256"],
        "query_full_identity_sha256": cp[
            "query_full_identity_sha256"
        ],
    }
    expected_label_usage = (
        "outer_target_sealed_prelabel_decision_evaluation_only"
        if cp["episode_role"] == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
        else "stage2_loss_and_exact_curve_replay_only"
    )
    payload = {
        "schema_version": EPISODE_SCHEMA,
        "artifact_type": EPISODE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "episode_id": canonical_json_sha256(
            _episode_identity_preimage(
                episode_index=index,
                context=verified,
                anchor_binding=anchor,
                oracle_curve_binding=oracle,
                prelabel_decision_binding=decision,
            )
        ),
        "episode_index": index,
        "collection_role": ROLE_TO_COLLECTION[cp["expected_role"]],
        "episode_role": cp["episode_role"],
        "outer_fold_id": cp["outer_fold_id"],
        "outer_target": cp["outer_target"],
        "source_domain": cp["source_domain"],
        "base_seed": cp["base_seed"],
        "derived_seed": cp["derived_seed"],
        "episode_weighting": EPISODE_WEIGHTING,
        "query_size": len(cp["query_identity_records"]),
        "context_binding": context_binding,
        "context_payload": _plain(cp),
        "anchor_binding": anchor,
        "oracle_curve_binding": oracle,
        "budget_rationals": [dict(row) for row in BUDGET_RATIONALS],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "prelabel_decision_binding": decision,
        "supervision_contract": {
            "query_label_usage": expected_label_usage,
            "query_size_from_geometry": True,
            "episode_weighting": EPISODE_WEIGHTING,
            "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
            "budget_count_formula": BUDGET_COUNT_FORMULA,
            "float_budget_counts_forbidden": True,
        },
        "guardrails": {
            "context_labels_accessed": False,
            "query_labels_accessed": True,
            "decision_made_before_outer_labels": (
                cp["episode_role"]
                == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
            ),
            "reject_supported": False,
            "fallback_used": False,
            "official_test_accessed": False,
        },
    }
    return _plain(verify_episode_payload_v6(payload).payload)


_MANIFEST_RECORD_FIELDS = frozenset(
    {
        "episode_index",
        "episode_id",
        "window_id",
        "window_index",
        "source_domain",
        "query_size",
        "record_sha256",
    }
)
_DOMAIN_GEOMETRY_FIELDS = frozenset(
    {
        "source_domain",
        "ordered_record_count",
        "window_count",
        "geometry_sha256",
    }
)
_FILE_BINDING_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "path_anchor",
        "collection_role",
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "episode_weighting",
        "episode_count",
        "all_records_consumed_once",
        "external_sha256_required",
        "collection_file",
        "geometry_schema_version",
        "domain_geometries",
        "record_sha256_algorithm",
        "ordered_record_sha256_algorithm",
        "ordered_record_sha256",
        "records",
    }
)
_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "publication_complete",
        "official_test_accessed",
        "path_anchor",
        "publication_order",
        "external_sha256_required_for_every_member",
        "collection_file",
        "collection_manifest",
    }
)


def record_sha256_v6(episode: VerifiedStage2EpisodeV6) -> str:
    verified = assert_verified_episode_v6(episode)
    return verified.payload_sha256


def verify_episode_collection_completeness_v6(
    episodes: Sequence[VerifiedStage2EpisodeV6],
) -> tuple[dict[str, Any], ...]:
    if (
        isinstance(episodes, (str, bytes))
        or not isinstance(episodes, Sequence)
        or not episodes
    ):
        raise Stage2CrossfitSchemaV6Error(
            "episode-v6 collection must be nonempty"
        )
    verified = [assert_verified_episode_v6(item) for item in episodes]
    first = verified[0].payload
    role = str(first["collection_role"])
    outer_fold = str(first["outer_fold_id"])
    outer_target = str(first["outer_target"])
    base_seed = int(first["base_seed"])
    if OUTER_TARGETS.get(outer_fold) != outer_target:
        raise Stage2CrossfitSchemaV6Error(
            "collection outer fold/target mismatch"
        )

    seen_episode_ids: set[str] = set()
    seen_windows: set[str] = set()
    global_identity: dict[str, set[str]] = {
        field: set() for field in _FOUR_BOUNDARY_FIELDS
    }
    by_domain: dict[str, list[VerifiedStage2EpisodeV6]] = {}
    for index, episode in enumerate(verified):
        payload = episode.payload
        if payload["episode_index"] != index:
            raise Stage2CrossfitSchemaV6Error(
                "episode_index must be contiguous in JSONL order"
            )
        if (
            payload["collection_role"] != role
            or payload["outer_fold_id"] != outer_fold
            or payload["outer_target"] != outer_target
            or payload["base_seed"] != base_seed
        ):
            raise Stage2CrossfitSchemaV6Error(
                "collection mixes role/fold/target/base seed"
            )
        if payload["episode_weighting"] != EPISODE_WEIGHTING:
            raise Stage2CrossfitSchemaV6Error(
                "collection requires equal-window episode weighting"
            )
        episode_id = str(payload["episode_id"])
        window_id = str(episode.context.payload["window_id"])
        if episode_id in seen_episode_ids:
            raise Stage2CrossfitSchemaV6Error("duplicate episode_id")
        if window_id in seen_windows:
            raise Stage2CrossfitSchemaV6Error("duplicate window identity")
        seen_episode_ids.add(episode_id)
        seen_windows.add(window_id)
        records = [
            *episode.context.payload["context_records"],
            *episode.context.payload["query_identity_records"],
        ]
        for field in _FOUR_BOUNDARY_FIELDS:
            values = [str(row[field]) for row in records]
            overlap = global_identity[field].intersection(values)
            if overlap:
                raise Stage2CrossfitSchemaV6Error(
                    f"collection duplicate identity at {field}"
                )
            global_identity[field].update(values)
        by_domain.setdefault(str(payload["source_domain"]), []).append(
            episode
        )

    domains = set(by_domain)
    expected_sources = set(ALL_DOMAINS) - {outer_target}
    if role in {COLLECTION_TRAIN, COLLECTION_VALIDATION}:
        if domains != expected_sources:
            raise Stage2CrossfitSchemaV6Error(
                "train/validation require both source domains and no outer target"
            )
    elif role == COLLECTION_OUTER:
        if domains != {outer_target}:
            raise Stage2CrossfitSchemaV6Error(
                "outer collection must contain only the outer target"
            )
    else:
        raise Stage2CrossfitSchemaV6Error(
            "unsupported collection role"
        )

    summaries: list[dict[str, Any]] = []
    for domain in ALL_DOMAINS:
        if domain not in by_domain:
            continue
        domain_episodes = by_domain[domain]
        reference_geometry = _plain(
            domain_episodes[0].context.payload["geometry"]
        )
        geometry = validate_stage2_variable_query_geometry(
            reference_geometry
        )
        windows: dict[int, VerifiedStage2EpisodeV6] = {}
        consumed_indices: list[int] = []
        for episode in domain_episodes:
            cp = episode.context.payload
            if _plain(cp["geometry"]) != geometry:
                raise Stage2CrossfitSchemaV6Error(
                    "collection domain mixes role geometries"
                )
            window_index = int(cp["window_index"])
            if window_index in windows:
                raise Stage2CrossfitSchemaV6Error(
                    "duplicate geometry window_index"
                )
            windows[window_index] = episode
            consumed_indices.extend(
                int(row["source_role_record_index"])
                for row in (
                    *cp["context_records"],
                    *cp["query_identity_records"],
                )
            )
        expected_windows = set(range(geometry["window_count"]))
        if set(windows) != expected_windows:
            raise Stage2CrossfitSchemaV6Error(
                "collection omits one or more geometry windows"
            )
        expected_indices = list(range(geometry["ordered_record_count"]))
        if sorted(consumed_indices) != expected_indices:
            raise Stage2CrossfitSchemaV6Error(
                "collection does not consume every ordered record exactly once"
            )
        summaries.append(
            {
                "source_domain": domain,
                "ordered_record_count": geometry[
                    "ordered_record_count"
                ],
                "window_count": geometry["window_count"],
                "geometry_sha256": canonical_json_sha256(geometry),
            }
        )
    if len(verified) != sum(
        int(summary["window_count"]) for summary in summaries
    ):
        raise Stage2CrossfitSchemaV6Error(
            "collection episode count differs from floor(N/42) replay"
        )
    return tuple(summaries)


def _repo_root(value: str | Path) -> Path:
    root = Path(value).expanduser().absolute()
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise Stage2CrossfitSchemaV6Error(
            "repository_root must be canonical and non-symlink"
        )
    return root


def _assert_no_symlink(path: Path, root: Path, name: str) -> None:
    if path != root and root not in path.parents:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} escapes repository_root"
        )
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise Stage2CrossfitSchemaV6Error(
                f"{name} contains a symlink component"
            )


def _direct_file(value: str | Path, root: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    _assert_no_symlink(path, root, name)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise Stage2CrossfitSchemaV6Error(
            f"{name} must be a regular file"
        )
    return path


def _repo_relative(path: Path, root: Path) -> str:
    absolute = path.expanduser().absolute()
    if absolute != root and root not in absolute.parents:
        raise Stage2CrossfitSchemaV6Error(
            "bundle path is outside repository_root"
        )
    return absolute.relative_to(root).as_posix()


def _stable_read(path: Path, expected_sha256: str, name: str) -> bytes:
    expected = _sha256(expected_sha256, f"{name}.expected_sha256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2CrossfitSchemaV6Error(
                f"{name} is not a regular file"
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(
        current
    ):
        raise RuntimeError(f"{name} changed during verified read")
    observed = digest.hexdigest()
    if observed != expected:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} SHA-256 mismatch: "
            f"observed={observed}, expected={expected}"
        )
    return b"".join(chunks)


def collection_manifest_path_v6(collection_path: Path) -> Path:
    return collection_path.with_name(
        f"{collection_path.stem}.collection.json"
    )


def collection_commit_path_v2(collection_path: Path) -> Path:
    return collection_path.with_name(f"{collection_path.stem}.commit.json")


def collection_jsonl_bytes_v6(
    episodes: Sequence[VerifiedStage2EpisodeV6],
) -> bytes:
    verified = [assert_verified_episode_v6(item) for item in episodes]
    verify_episode_collection_completeness_v6(verified)
    return b"".join(
        episode.canonical_payload + b"\n" for episode in verified
    )


def _manifest_record_rows(
    episodes: Sequence[VerifiedStage2EpisodeV6],
) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": index,
            "episode_id": episode.episode_id,
            "window_id": episode.context.payload["window_id"],
            "window_index": episode.context.payload["window_index"],
            "source_domain": episode.payload["source_domain"],
            "query_size": episode.payload["query_size"],
            "record_sha256": record_sha256_v6(episode),
        }
        for index, episode in enumerate(episodes)
    ]


def build_collection_manifest_payload_v6(
    episodes: Sequence[VerifiedStage2EpisodeV6],
    *,
    collection_path: str | Path,
    collection_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    verified = [assert_verified_episode_v6(item) for item in episodes]
    summaries = verify_episode_collection_completeness_v6(verified)
    root = _repo_root(repository_root)
    path = Path(collection_path).expanduser().absolute()
    collection_relative = _repo_relative(path, root)
    collection_sha = _sha256(
        collection_sha256, "collection_sha256"
    )
    records = _manifest_record_rows(verified)
    first = verified[0].payload
    return {
        "schema_version": COLLECTION_SCHEMA,
        "artifact_type": COLLECTION_ARTIFACT_TYPE,
        "artifact_status": "COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "collection_role": first["collection_role"],
        "outer_fold_id": first["outer_fold_id"],
        "outer_target": first["outer_target"],
        "base_seed": first["base_seed"],
        "episode_weighting": EPISODE_WEIGHTING,
        "episode_count": len(verified),
        "all_records_consumed_once": True,
        "external_sha256_required": True,
        "collection_file": {
            "path": collection_relative,
            "sha256": collection_sha,
        },
        "geometry_schema_version": VARIABLE_QUERY_GEOMETRY_SCHEMA,
        "domain_geometries": list(summaries),
        "record_sha256_algorithm": RECORD_HASH_ALGORITHM,
        "ordered_record_sha256_algorithm": (
            ORDERED_RECORD_HASH_ALGORITHM
        ),
        "ordered_record_sha256": canonical_json_sha256(records),
        "records": records,
    }


def build_collection_commit_payload_v2(
    *,
    collection_path: str | Path,
    collection_sha256: str,
    manifest_path: str | Path,
    manifest_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = _repo_root(repository_root)
    collection = Path(collection_path).expanduser().absolute()
    manifest = Path(manifest_path).expanduser().absolute()
    return {
        "schema_version": COLLECTION_COMMIT_SCHEMA,
        "artifact_type": COLLECTION_COMMIT_ARTIFACT_TYPE,
        "artifact_status": "COMPLETE",
        "publication_complete": True,
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "publication_order": COMMIT_PUBLICATION_ORDER,
        "external_sha256_required_for_every_member": True,
        "collection_file": {
            "path": _repo_relative(collection, root),
            "sha256": _sha256(
                collection_sha256, "collection_sha256"
            ),
        },
        "collection_manifest": {
            "path": _repo_relative(manifest, root),
            "sha256": _sha256(manifest_sha256, "manifest_sha256"),
        },
    }


def _parse_canonical_json_document(
    data: bytes, name: str
) -> Mapping[str, Any]:
    if not data.endswith(b"\n") or data.endswith(b"\n\n") or b"\r" in data:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} requires one LF terminator and no CR"
        )
    payload = parse_json_bytes(data[:-1], name)
    if canonical_json_document_bytes(payload) != data:
        raise Stage2CrossfitSchemaV6Error(
            f"{name} is not canonical JSON bytes"
        )
    return payload


def _parse_canonical_jsonl(data: bytes) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise Stage2CrossfitSchemaV6Error(
            "episode JSONL requires LF termination and no CR"
        )
    lines = data.splitlines()
    if not lines or any(not line for line in lines):
        raise Stage2CrossfitSchemaV6Error(
            "episode JSONL contains an empty line"
        )
    result: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        payload = parse_json_bytes(line, f"episode JSONL line {index}")
        if canonical_json_bytes(payload) != line:
            raise Stage2CrossfitSchemaV6Error(
                "episode JSONL line is not canonical JSON"
            )
        result.append(payload)
    return result


@dataclass(frozen=True, init=False)
class VerifiedStage2CollectionV6(Sequence[VerifiedStage2EpisodeV6]):
    """Verifier-issued immutable collection and nested episode capabilities."""

    path: Path
    manifest_path: Path
    commit_path: Path
    collection_sha256: str
    manifest_sha256: str
    commit_sha256: str
    episodes: tuple[VerifiedStage2EpisodeV6, ...]
    manifest: Mapping[str, Any]
    commit: Mapping[str, Any]
    _capability: object

    def __init__(
        self,
        *,
        path: Path,
        manifest_path: Path,
        commit_path: Path,
        collection_sha256: str,
        manifest_sha256: str,
        commit_sha256: str,
        episodes: Sequence[VerifiedStage2EpisodeV6],
        manifest: Mapping[str, Any],
        commit: Mapping[str, Any],
        _capability: object,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError("VerifiedStage2CollectionV6 is verifier-only")
        frozen_episodes = tuple(
            assert_verified_episode_v6(item) for item in episodes
        )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "commit_path", commit_path)
        object.__setattr__(
            self, "collection_sha256", collection_sha256
        )
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "commit_sha256", commit_sha256)
        object.__setattr__(self, "episodes", frozen_episodes)
        object.__setattr__(self, "manifest", _freeze(manifest))
        object.__setattr__(self, "commit", _freeze(commit))
        object.__setattr__(self, "_capability", _capability)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(
        self, index: int | slice
    ) -> VerifiedStage2EpisodeV6 | tuple[VerifiedStage2EpisodeV6, ...]:
        return self.episodes[index]


def assert_verified_collection_v6(
    value: object,
) -> VerifiedStage2CollectionV6:
    if (
        not isinstance(value, VerifiedStage2CollectionV6)
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
    ):
        raise TypeError(
            "a verifier-issued collection-v6 capability is required"
        )
    for episode in value.episodes:
        assert_verified_episode_v6(episode)
    return value


def _canonical_manifest(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    _strict_fields(payload, _MANIFEST_FIELDS, "collection manifest")
    if _plain(payload) != _plain(expected):
        raise Stage2CrossfitSchemaV6Error(
            "collection manifest differs from recursive replay"
        )
    return _plain(expected)


def _canonical_commit(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    _strict_fields(payload, _COMMIT_FIELDS, "collection commit")
    if _plain(payload) != _plain(expected):
        raise Stage2CrossfitSchemaV6Error(
            "collection commit member binding mismatch"
        )
    return _plain(expected)


def verify_stage2_collection_bundle_v6(
    collection_path: str | Path,
    expected_collection_sha256: str,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    commit_path: str | Path,
    expected_commit_sha256: str,
    *,
    repository_root: str | Path,
) -> VerifiedStage2CollectionV6:
    """Verify collection members and recursively replay nested payloads.

    JSONL, manifest, and commit are opened and externally SHA-verified.  The
    nested episode artifact descriptors remain immutable digest bindings;
    their referenced anchor/curve/decision files are not opened here.
    """

    root = _repo_root(repository_root)
    collection = _direct_file(
        collection_path, root, "episode JSONL"
    )
    manifest = _direct_file(
        manifest_path, root, "episode collection manifest"
    )
    commit = _direct_file(
        commit_path, root, "episode collection commit"
    )
    if manifest != collection_manifest_path_v6(collection):
        raise Stage2CrossfitSchemaV6Error(
            "collection manifest path derivation mismatch"
        )
    if commit != collection_commit_path_v2(collection):
        raise Stage2CrossfitSchemaV6Error(
            "collection commit path derivation mismatch"
        )
    lock = collection.parent / f".{collection.name}.lock"
    if os.path.lexists(lock):
        raise RuntimeError("collection publication lock is present")
    collection_sha = _sha256(
        expected_collection_sha256, "expected_collection_sha256"
    )
    manifest_sha = _sha256(
        expected_manifest_sha256, "expected_manifest_sha256"
    )
    commit_sha = _sha256(
        expected_commit_sha256, "expected_commit_sha256"
    )

    # Commit is deliberately opened last.
    collection_data = _stable_read(
        collection, collection_sha, "episode JSONL"
    )
    manifest_data = _stable_read(
        manifest, manifest_sha, "episode manifest"
    )
    commit_data = _stable_read(commit, commit_sha, "episode commit")

    if commit.stat(follow_symlinks=False).st_mtime_ns < max(
        collection.stat(follow_symlinks=False).st_mtime_ns,
        manifest.stat(follow_symlinks=False).st_mtime_ns,
    ):
        raise Stage2CrossfitSchemaV6Error(
            "collection commit was not published last"
        )
    raw_episodes = _parse_canonical_jsonl(collection_data)
    episodes = tuple(
        verify_episode_payload_v6(payload) for payload in raw_episodes
    )
    verify_episode_collection_completeness_v6(episodes)

    expected_manifest = build_collection_manifest_payload_v6(
        episodes,
        collection_path=collection,
        collection_sha256=collection_sha,
        repository_root=root,
    )
    manifest_payload = _parse_canonical_json_document(
        manifest_data, "episode manifest"
    )
    canonical_manifest = _canonical_manifest(
        manifest_payload, expected_manifest
    )
    expected_commit = build_collection_commit_payload_v2(
        collection_path=collection,
        collection_sha256=collection_sha,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        repository_root=root,
    )
    commit_payload = _parse_canonical_json_document(
        commit_data, "episode commit"
    )
    canonical_commit = _canonical_commit(
        commit_payload, expected_commit
    )
    for path, expected, name in (
        (collection, collection_sha, "episode JSONL"),
        (manifest, manifest_sha, "episode manifest"),
        (commit, commit_sha, "episode commit"),
    ):
        _stable_read(path, expected, f"{name} final recheck")
    return VerifiedStage2CollectionV6(
        path=collection,
        manifest_path=manifest,
        commit_path=commit,
        collection_sha256=collection_sha,
        manifest_sha256=manifest_sha,
        commit_sha256=commit_sha,
        episodes=episodes,
        manifest=canonical_manifest,
        commit=canonical_commit,
        _capability=_CAPABILITY_TOKEN,
    )


__all__ = [
    "ALL_DOMAINS",
    "ANCHOR_SCHEMA",
    "BASE_SEEDS",
    "BOOTSTRAP_QUERY_IDENTITY_ALGORITHM",
    "BUDGET_COUNT_FORMULA",
    "BUDGET_RATIONALS",
    "COLLECTION_COMMIT_SCHEMA",
    "COLLECTION_OUTER",
    "COLLECTION_SCHEMA",
    "COLLECTION_TRAIN",
    "COLLECTION_VALIDATION",
    "COMMIT_PUBLICATION_ORDER",
    "CONTEXT_ARTIFACT_TYPE",
    "CONTEXT_SCHEMA",
    "EATC_V2_SCHEMA",
    "EPISODE_ARTIFACT_TYPE",
    "EPISODE_SCHEMA",
    "EPISODE_WEIGHTING",
    "FULL_IDENTITY_ALGORITHM",
    "OOF_HOLDOUT_STAGE2_FIT",
    "ORACLE_CURVE_BINDING_SCHEMA",
    "OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT",
    "OUTER_TARGETS",
    "PRELABEL_DECISION_BINDING_SCHEMA",
    "RECORD_HASH_ALGORITHM",
    "SOURCE_DIAGNOSTIC_VALIDATION",
    "STAGE2_OOF_FIT",
    "STRICT_THRESHOLD_SEMANTICS",
    "Stage2CrossfitSchemaV6Error",
    "VerifiedContextInferenceMaterialV2",
    "VerifiedStage2CollectionV6",
    "VerifiedStage2ContextV2",
    "VerifiedStage2EpisodeV6",
    "assert_verified_collection_v6",
    "assert_verified_context_inference_material_v2",
    "assert_verified_context_v2",
    "assert_verified_episode_v6",
    "bootstrap_query_identity_projection",
    "bootstrap_query_identity_sha256",
    "build_collection_commit_payload_v2",
    "build_collection_manifest_payload_v6",
    "build_context_payload_v2",
    "build_episode_payload_v6",
    "canonical_json_bytes",
    "canonical_json_document_bytes",
    "canonical_json_sha256",
    "collection_commit_path_v2",
    "collection_jsonl_bytes_v6",
    "collection_manifest_path_v6",
    "context_inference_material_v2",
    "context_from_verified_variable_query_window_v2",
    "full_identity_projection",
    "full_identity_sha256",
    "make_anchor_binding_v6",
    "make_oracle_curve_binding_v2",
    "make_prelabel_decision_binding_v1",
    "record_sha256_v6",
    "sha256_bytes",
    "verify_context_payload_v2",
    "verify_episode_collection_completeness_v6",
    "verify_episode_payload_v6",
    "verify_stage2_collection_bundle_v6",
    "window_identity_sha256",
]
