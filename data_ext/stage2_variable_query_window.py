"""Additive, result-free Stage-2 variable-query window contract (RC5).

This module materializes metadata windows only.  It never opens an image,
mask, label, score, prediction, checkpoint, or metric artifact.  Geometry is
delegated exclusively to :mod:`rc.stage2_variable_query_geometry`: every
window keeps C=14 context records, receives Q_i>=28 query records, and the
frozen ``floor(N/42)`` partition consumes every ordered role record once.

The artifact type deliberately stays compatible with the existing Stage-2
score-manifest selection flattener.  The schema is additive and does not
change or relax the fixed-Q28 v1 consumer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any

from rc.stage2_variable_query_geometry import (
    CONTEXT_SIZE,
    MINIMUM_QUERY_SIZE,
    Stage2VariableQueryGeometryContractError,
    build_stage2_variable_query_geometry,
    validate_stage2_variable_query_geometry,
)


SCHEMA_VERSION = "rc-irstd.stage2-role-pure-variable-query-windows.v2"
ARTIFACT_TYPE = "rc_irstd_stage2_role_pure_episode_windows"
ARTIFACT_STATUS = "DEVELOPMENT_ONLY_RESULT_FREE"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WINDOW_ID_ALGORITHM = (
    "outer-fold-domain-source-role-episode-role-variable-q-index-v2"
)
ORDERED_RECORD_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-ordered-stage2-variable-query-record-identity-v2"
)
BOUND_INPUT_NAMES = frozenset(
    {
        "image_only_near_duplicate_audit",
        "k2_geometry_prefreeze_audit",
        "official_train_derived_split_manifest",
    }
)
IDENTITY_BOUNDARIES = (
    "canonical_id",
    "original_image_sha256",
    "near_duplicate_cluster_id_or_unique_sentinel",
    "exclusion_group_id",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VERIFIER_CAPABILITY = object()
_SENSITIVE_PATH_TOKENS = frozenset(
    {
        "checkpoint",
        "checkpoints",
        "label",
        "labels",
        "mask",
        "masks",
        "metric",
        "metrics",
        "officialtest",
        "prediction",
        "predictions",
        "result",
        "results",
        "score",
        "scores",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
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
_WINDOW_FIELDS = frozenset(
    {
        "window_index",
        "window_id",
        "context_start",
        "context_stop",
        "query_start",
        "query_stop",
        "context_size",
        "query_size",
        "context_records",
        "query_records",
    }
)
_WINDOW_INTEGER_FIELDS = (
    "window_index",
    "context_start",
    "context_stop",
    "query_start",
    "query_stop",
    "context_size",
    "query_size",
)
_RECORD_FIELDS = frozenset(
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
        "oof_fold_index",
    }
)
_RECORD_FIELD_ORDER = (
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
    "oof_fold_index",
)
_ROLE_PURITY_FIELDS = frozenset(
    {
        "allowed_source_role",
        "mixed_roles_allowed",
        "single_source_domain_per_window",
        "single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding",
    }
)
_UNUSED_SUFFIX_FIELDS = frozenset(
    {"record_count", "records", "all_ordered_role_records_consumed_once"}
)
_BINDING_FIELDS = frozenset({"path", "sha256"})
_GUARDRAIL_FIELDS = frozenset(
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
_EXPECTED_GUARDRAILS = {
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


class Stage2VariableQueryWindowContractError(ValueError):
    """A variable-Q window payload violates the additive RC5 contract."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise Stage2VariableQueryWindowContractError(
            f"{name} field closure mismatch: missing={missing}, extra={extra}"
        )


def _nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise Stage2VariableQueryWindowContractError(
            f"{name} must be one exact non-empty string"
        )
    return value


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Stage2VariableQueryWindowContractError(
            f"{name} must be an exact integer >= {minimum}"
        )
    return value


def _exact_bool(value: Any, name: str, expected: bool) -> bool:
    if type(value) is not bool or value is not expected:
        raise Stage2VariableQueryWindowContractError(
            f"{name} must be exactly {str(expected).lower()}"
        )
    return value


def _sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise Stage2VariableQueryWindowContractError(
            f"{name} must be one lowercase SHA-256"
        )
    return value


def _oof_fold_index(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _exact_int(value, name)


def _path_tokens(path: PurePosixPath) -> set[str]:
    tokens: set[str] = set()
    for part in path.parts:
        normalized = re.sub(r"[^a-z0-9]+", "", part.casefold())
        if normalized:
            tokens.add(normalized)
        tokens.update(
            token
            for token in re.split(r"[^a-z0-9]+", part.casefold())
            if token
        )
    return tokens


def _relative_path(value: Any, name: str, *, original_image: bool) -> str:
    raw = _nonempty_string(value, name)
    if "\\" in raw:
        raise Stage2VariableQueryWindowContractError(f"{name} must use POSIX separators")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise Stage2VariableQueryWindowContractError(
            f"{name} must be one canonical repository-relative path"
        )
    tokens = _path_tokens(path)
    if tokens & _SENSITIVE_PATH_TOKENS:
        raise Stage2VariableQueryWindowContractError(
            f"{name} may not reference official-test/label/mask/score/result paths"
        )
    if original_image and "images" not in {part.casefold() for part in path.parts}:
        raise Stage2VariableQueryWindowContractError(
            f"{name} must remain under an images directory"
        )
    return raw


def _binding(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _exact_keys(value, _BINDING_FIELDS, name)
    return {
        "path": _relative_path(value["path"], f"{name}.path", original_image=False),
        "sha256": _sha256(value["sha256"], f"{name}.sha256"),
    }


def _guardrails(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("guardrails must be a mapping")
    _exact_keys(value, _GUARDRAIL_FIELDS, "guardrails")
    for field, expected in _EXPECTED_GUARDRAILS.items():
        _exact_bool(value[field], f"guardrails.{field}", expected)
    return dict(_EXPECTED_GUARDRAILS)


def _role_purity(
    value: Any, *, source_role: str, oof_fold_index: int | None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("role_purity must be a mapping")
    _exact_keys(value, _ROLE_PURITY_FIELDS, "role_purity")
    expected = {
        "allowed_source_role": source_role,
        "mixed_roles_allowed": False,
        "single_source_domain_per_window": True,
        "single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding": (
            oof_fold_index is not None
        ),
    }
    for field, required in expected.items():
        if type(required) is bool:
            _exact_bool(value[field], f"role_purity.{field}", required)
        elif type(value[field]) is not str or value[field] != required:
            raise Stage2VariableQueryWindowContractError(
                f"role_purity.{field} mismatch"
            )
    return expected


def _canonical_record(
    value: Any,
    *,
    expected_index: int,
    domain: str,
    source_role: str,
    outer_fold_id: str,
    episode_role: str,
    oof_fold_index: int | None,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    _exact_keys(value, _RECORD_FIELDS, name)
    image_id = _nonempty_string(value["image_id"], f"{name}.image_id")
    canonical_id = _nonempty_string(value["canonical_id"], f"{name}.canonical_id")
    if canonical_id != f"{domain}::{image_id}":
        raise Stage2VariableQueryWindowContractError(
            f"{name}.canonical_id must equal domain::image_id"
        )
    source_index = _exact_int(
        value["source_role_record_index"], f"{name}.source_role_record_index"
    )
    if source_index != expected_index:
        raise Stage2VariableQueryWindowContractError(
            "window records are missing, duplicated, or out of strict source-role order"
        )
    exact_strings = {
        "source_role": source_role,
        "outer_fold_id": outer_fold_id,
        "episode_role": episode_role,
    }
    for field, expected in exact_strings.items():
        observed = _nonempty_string(value[field], f"{name}.{field}")
        if observed != expected:
            raise Stage2VariableQueryWindowContractError(
                f"{name}.{field} violates role purity"
            )
    observed_oof = _oof_fold_index(value["oof_fold_index"], f"{name}.oof_fold_index")
    if observed_oof != oof_fold_index:
        raise Stage2VariableQueryWindowContractError(
            f"{name}.oof_fold_index violates role purity"
        )
    canonical = {
        "canonical_id": canonical_id,
        "image_id": image_id,
        "original_image_path": _relative_path(
            value["original_image_path"],
            f"{name}.original_image_path",
            original_image=True,
        ),
        "original_image_sha256": _sha256(
            value["original_image_sha256"], f"{name}.original_image_sha256"
        ),
        "exclusion_group_id": _nonempty_string(
            value["exclusion_group_id"], f"{name}.exclusion_group_id"
        ),
        "near_duplicate_cluster_id_or_unique_sentinel": _nonempty_string(
            value["near_duplicate_cluster_id_or_unique_sentinel"],
            f"{name}.near_duplicate_cluster_id_or_unique_sentinel",
        ),
        "source_role_record_index": source_index,
        "source_role": source_role,
        "outer_fold_id": outer_fold_id,
        "episode_role": episode_role,
        "oof_fold_index": oof_fold_index,
    }
    return {field: canonical[field] for field in _RECORD_FIELD_ORDER}


def stage2_variable_query_record_identity(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project the ordered identity boundary used by variable-Q consumers."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be an ordered sequence")
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        result.append(
            {
                "record_index": index,
                "canonical_id": _nonempty_string(
                    record.get("canonical_id"), f"records[{index}].canonical_id"
                ),
                "original_image_sha256": _sha256(
                    record.get("original_image_sha256"),
                    f"records[{index}].original_image_sha256",
                ),
                "near_duplicate_cluster_id_or_unique_sentinel": _nonempty_string(
                    record.get("near_duplicate_cluster_id_or_unique_sentinel"),
                    f"records[{index}].near_duplicate_cluster_id_or_unique_sentinel",
                ),
                "exclusion_group_id": _nonempty_string(
                    record.get("exclusion_group_id"),
                    f"records[{index}].exclusion_group_id",
                ),
                "source_role_record_index": _exact_int(
                    record.get("source_role_record_index"),
                    f"records[{index}].source_role_record_index",
                ),
            }
        )
    return result


def _window_id(
    *,
    outer_fold_id: str,
    domain: str,
    source_role: str,
    episode_role: str,
    window_index: int,
) -> str:
    return (
        f"{outer_fold_id}::{domain}::{source_role}::{episode_role}::"
        f"variable_q_window_{window_index:04d}"
    )


def _reject_duplicate_identities(records: Sequence[Mapping[str, Any]]) -> None:
    seen = {field: set() for field in IDENTITY_BOUNDARIES}
    for index, record in enumerate(records):
        for field in IDENTITY_BOUNDARIES:
            value = record[field]
            if value in seen[field]:
                raise Stage2VariableQueryWindowContractError(
                    f"duplicate {field} across ordered role records at index {index}"
                )
            seen[field].add(value)


def build_stage2_variable_query_window_payload(
    *,
    ordered_role_records: Sequence[Mapping[str, Any]],
    outer_fold_id: str,
    outer_target_domain: str,
    domain: str,
    source_role: str,
    episode_role: str,
    oof_fold_index: int | None,
    role_binding: Mapping[str, Any],
    bound_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a pure result-free v2 payload without reading any external file."""

    if (
        isinstance(ordered_role_records, (str, bytes))
        or not isinstance(ordered_role_records, Sequence)
    ):
        raise TypeError("ordered_role_records must be an ordered sequence")
    fold = _nonempty_string(outer_fold_id, "outer_fold_id")
    target = _nonempty_string(outer_target_domain, "outer_target_domain")
    source_domain = _nonempty_string(domain, "domain")
    source = _nonempty_string(source_role, "source_role")
    episode = _nonempty_string(episode_role, "episode_role")
    oof = _oof_fold_index(oof_fold_index, "oof_fold_index")
    try:
        geometry = build_stage2_variable_query_geometry(len(ordered_role_records))
    except Stage2VariableQueryGeometryContractError as error:
        raise Stage2VariableQueryWindowContractError(
            "ordered_role_records cannot form frozen C14/Qmin28 geometry"
        ) from error

    records = [
        _canonical_record(
            record,
            expected_index=index,
            domain=source_domain,
            source_role=source,
            outer_fold_id=fold,
            episode_role=episode,
            oof_fold_index=oof,
            name=f"ordered_role_records[{index}]",
        )
        for index, record in enumerate(ordered_role_records)
    ]
    _reject_duplicate_identities(records)
    if not isinstance(bound_inputs, Mapping):
        raise TypeError("bound_inputs must be a mapping")
    _exact_keys(bound_inputs, BOUND_INPUT_NAMES, "bound_inputs")
    canonical_bound_inputs = {
        name: _binding(bound_inputs[name], f"bound_inputs.{name}")
        for name in sorted(BOUND_INPUT_NAMES)
    }

    windows: list[dict[str, Any]] = []
    for geometry_window in geometry["windows"]:
        context_start = geometry_window["context_start"]
        context_stop = geometry_window["context_stop"]
        query_start = geometry_window["query_start"]
        query_stop = geometry_window["query_stop"]
        window_index = geometry_window["window_index"]
        windows.append(
            {
                "window_index": window_index,
                "window_id": _window_id(
                    outer_fold_id=fold,
                    domain=source_domain,
                    source_role=source,
                    episode_role=episode,
                    window_index=window_index,
                ),
                "context_start": context_start,
                "context_stop": context_stop,
                "query_start": query_start,
                "query_stop": query_stop,
                "context_size": geometry_window["context_size"],
                "query_size": geometry_window["query_size"],
                "context_records": records[context_start:context_stop],
                "query_records": records[query_start:query_stop],
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_status": ARTIFACT_STATUS,
        "execution_authorized": False,
        "observed_results": None,
        "outer_fold_id": fold,
        "outer_target_domain": target,
        "domain": source_domain,
        "source_role": source,
        "episode_role": episode,
        "oof_fold_index": oof,
        "geometry": geometry,
        "role_purity": {
            "allowed_source_role": source,
            "mixed_roles_allowed": False,
            "single_source_domain_per_window": True,
            "single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding": (
                oof is not None
            ),
        },
        "ordered_role_record_count": len(records),
        "complete_window_count": geometry["window_count"],
        "window_record_count": len(records),
        "unused_suffix": {
            "record_count": 0,
            "records": [],
            "all_ordered_role_records_consumed_once": True,
        },
        "role_binding": _binding(role_binding, "role_binding"),
        "bound_inputs": canonical_bound_inputs,
        "guardrails": dict(_EXPECTED_GUARDRAILS),
        "windows": windows,
    }
    return validate_stage2_variable_query_window_payload(payload)


def validate_stage2_variable_query_window_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a v2 payload completely without performing filesystem access."""

    if not isinstance(payload, Mapping):
        raise TypeError("Stage2 variable-query window payload must be a mapping")
    _exact_keys(payload, _TOP_LEVEL_FIELDS, "window payload")
    exact_strings = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_status": ARTIFACT_STATUS,
    }
    for field, expected in exact_strings.items():
        if type(payload[field]) is not str or payload[field] != expected:
            raise Stage2VariableQueryWindowContractError(
                f"window payload {field} mismatch"
            )
    _exact_bool(payload["execution_authorized"], "execution_authorized", False)
    if payload["observed_results"] is not None:
        raise Stage2VariableQueryWindowContractError(
            "observed_results must be exactly null"
        )

    fold = _nonempty_string(payload["outer_fold_id"], "outer_fold_id")
    target = _nonempty_string(payload["outer_target_domain"], "outer_target_domain")
    domain = _nonempty_string(payload["domain"], "domain")
    source_role = _nonempty_string(payload["source_role"], "source_role")
    episode_role = _nonempty_string(payload["episode_role"], "episode_role")
    oof = _oof_fold_index(payload["oof_fold_index"], "oof_fold_index")
    try:
        geometry = validate_stage2_variable_query_geometry(payload["geometry"])
    except (Stage2VariableQueryGeometryContractError, TypeError) as error:
        raise Stage2VariableQueryWindowContractError(
            "geometry differs from deterministic RC5 replay"
        ) from error
    ordered_count = _exact_int(
        payload["ordered_role_record_count"],
        "ordered_role_record_count",
        minimum=CONTEXT_SIZE + MINIMUM_QUERY_SIZE,
    )
    if geometry["ordered_record_count"] != ordered_count:
        raise Stage2VariableQueryWindowContractError(
            "geometry ordered_record_count differs from the window payload"
        )
    complete_count = _exact_int(
        payload["complete_window_count"], "complete_window_count", minimum=1
    )
    if complete_count != geometry["window_count"]:
        raise Stage2VariableQueryWindowContractError(
            "complete_window_count differs from deterministic geometry"
        )
    window_record_count = _exact_int(
        payload["window_record_count"], "window_record_count", minimum=1
    )
    if window_record_count != ordered_count:
        raise Stage2VariableQueryWindowContractError(
            "window_record_count must consume every ordered role record"
        )
    purity = _role_purity(
        payload["role_purity"], source_role=source_role, oof_fold_index=oof
    )
    guardrails = _guardrails(payload["guardrails"])
    role_binding = _binding(payload["role_binding"], "role_binding")
    raw_bound_inputs = payload["bound_inputs"]
    if not isinstance(raw_bound_inputs, Mapping):
        raise TypeError("bound_inputs must be a mapping")
    _exact_keys(raw_bound_inputs, BOUND_INPUT_NAMES, "bound_inputs")
    bound_inputs = {
        name: _binding(raw_bound_inputs[name], f"bound_inputs.{name}")
        for name in sorted(BOUND_INPUT_NAMES)
    }
    suffix = payload["unused_suffix"]
    if not isinstance(suffix, Mapping):
        raise TypeError("unused_suffix must be a mapping")
    _exact_keys(suffix, _UNUSED_SUFFIX_FIELDS, "unused_suffix")
    if _exact_int(suffix["record_count"], "unused_suffix.record_count") != 0:
        raise Stage2VariableQueryWindowContractError(
            "variable-Q geometry forbids an unused suffix"
        )
    if type(suffix["records"]) is not list or suffix["records"]:
        raise Stage2VariableQueryWindowContractError(
            "unused_suffix.records must be exactly an empty list"
        )
    _exact_bool(
        suffix["all_ordered_role_records_consumed_once"],
        "unused_suffix.all_ordered_role_records_consumed_once",
        True,
    )

    raw_windows = payload["windows"]
    if type(raw_windows) is not list or len(raw_windows) != complete_count:
        raise Stage2VariableQueryWindowContractError(
            "windows must be a list matching complete_window_count"
        )
    canonical_windows: list[dict[str, Any]] = []
    ordered_records: list[dict[str, Any]] = []
    for window_index, (raw_window, geometry_window) in enumerate(
        zip(raw_windows, geometry["windows"], strict=True)
    ):
        if not isinstance(raw_window, Mapping):
            raise TypeError(f"windows[{window_index}] must be a mapping")
        _exact_keys(raw_window, _WINDOW_FIELDS, f"windows[{window_index}]")
        for field in _WINDOW_INTEGER_FIELDS:
            _exact_int(raw_window[field], f"windows[{window_index}].{field}")
            if raw_window[field] != geometry_window[field]:
                raise Stage2VariableQueryWindowContractError(
                    f"windows[{window_index}].{field} differs from geometry replay"
                )
        expected_window_id = _window_id(
            outer_fold_id=fold,
            domain=domain,
            source_role=source_role,
            episode_role=episode_role,
            window_index=window_index,
        )
        if (
            type(raw_window["window_id"]) is not str
            or raw_window["window_id"] != expected_window_id
        ):
            raise Stage2VariableQueryWindowContractError(
                f"windows[{window_index}].window_id differs from deterministic replay"
            )
        raw_context = raw_window["context_records"]
        raw_query = raw_window["query_records"]
        if type(raw_context) is not list or len(raw_context) != geometry_window["context_size"]:
            raise Stage2VariableQueryWindowContractError(
                f"windows[{window_index}].context_records span mismatch"
            )
        if type(raw_query) is not list or len(raw_query) != geometry_window["query_size"]:
            raise Stage2VariableQueryWindowContractError(
                f"windows[{window_index}].query_records/query_size mismatch"
            )
        if len(raw_query) < MINIMUM_QUERY_SIZE:
            raise Stage2VariableQueryWindowContractError(
                f"windows[{window_index}] violates Q_i>=28"
            )

        context: list[dict[str, Any]] = []
        query: list[dict[str, Any]] = []
        for partition_name, raw_partition, output_partition in (
            ("context_records", raw_context, context),
            ("query_records", raw_query, query),
        ):
            for partition_index, raw_record in enumerate(raw_partition):
                expected_index = len(ordered_records)
                record = _canonical_record(
                    raw_record,
                    expected_index=expected_index,
                    domain=domain,
                    source_role=source_role,
                    outer_fold_id=fold,
                    episode_role=episode_role,
                    oof_fold_index=oof,
                    name=(
                        f"windows[{window_index}].{partition_name}"
                        f"[{partition_index}]"
                    ),
                )
                output_partition.append(record)
                ordered_records.append(record)
        canonical_windows.append(
            {
                "window_index": geometry_window["window_index"],
                "window_id": expected_window_id,
                "context_start": geometry_window["context_start"],
                "context_stop": geometry_window["context_stop"],
                "query_start": geometry_window["query_start"],
                "query_stop": geometry_window["query_stop"],
                "context_size": geometry_window["context_size"],
                "query_size": geometry_window["query_size"],
                "context_records": context,
                "query_records": query,
            }
        )
    if len(ordered_records) != ordered_count:
        raise Stage2VariableQueryWindowContractError(
            "windows did not consume every ordered role record exactly once"
        )
    _reject_duplicate_identities(ordered_records)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "artifact_status": ARTIFACT_STATUS,
        "execution_authorized": False,
        "observed_results": None,
        "outer_fold_id": fold,
        "outer_target_domain": target,
        "domain": domain,
        "source_role": source_role,
        "episode_role": episode_role,
        "oof_fold_index": oof,
        "geometry": geometry,
        "role_purity": purity,
        "ordered_role_record_count": ordered_count,
        "complete_window_count": complete_count,
        "window_record_count": window_record_count,
        "unused_suffix": {
            "record_count": 0,
            "records": [],
            "all_ordered_role_records_consumed_once": True,
        },
        "role_binding": role_binding,
        "bound_inputs": bound_inputs,
        "guardrails": guardrails,
        "windows": canonical_windows,
    }


def _repository_root(value: str | Path | None) -> Path:
    raw = REPOSITORY_ROOT if value is None else Path(value)
    try:
        root = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError("repository_root does not exist") from error
    if not root.is_dir():
        raise NotADirectoryError("repository_root must be a directory")
    return root


def _direct_file(
    value: str | Path, root: Path, name: str, *, payload_relative: bool
) -> Path:
    if payload_relative:
        relative = _relative_path(value, f"{name}.path", original_image=False)
        candidate = root / relative
    else:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{name} must be a path")
        raw = Path(value)
        candidate = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative_path = lexical.relative_to(root)
    except ValueError as error:
        raise Stage2VariableQueryWindowContractError(
            f"{name} must stay inside repository_root"
        ) from error
    relative_posix = PurePosixPath(relative_path.as_posix())
    if _path_tokens(relative_posix) & _SENSITIVE_PATH_TOKENS:
        raise Stage2VariableQueryWindowContractError(
            f"{name} may not resolve a label/mask/score/result file"
        )
    current = root
    for component in relative_path.parts:
        current = current / component
        try:
            info = os.lstat(current)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{name} does not exist: {current}") from error
        if stat.S_ISLNK(info.st_mode):
            raise Stage2VariableQueryWindowContractError(
                f"{name} may not use a symlink component"
            )
    if not stat.S_ISREG(os.lstat(lexical).st_mode):
        raise Stage2VariableQueryWindowContractError(
            f"{name} must be a stable regular file"
        )
    return lexical


def _stat_fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _stable_file_bytes(path: Path, name: str) -> bytes:
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise Stage2VariableQueryWindowContractError(f"{name} must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _stat_fingerprint(opened) != _stat_fingerprint(before):
            raise RuntimeError(f"{name} changed while it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.stat(path, follow_symlinks=False)
    if (
        _stat_fingerprint(after_descriptor) != _stat_fingerprint(opened)
        or _stat_fingerprint(after) != _stat_fingerprint(opened)
    ):
        raise RuntimeError(f"{name} changed while read")
    data = b"".join(chunks)
    if len(data) != opened.st_size:
        raise RuntimeError(f"{name} byte count changed while read")
    return data


def _closed_json_object(data: bytes, name: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Stage2VariableQueryWindowContractError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise Stage2VariableQueryWindowContractError(
            f"{name} contains non-finite JSON constant {value}"
        )

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2VariableQueryWindowContractError(
            f"{name} must be strict UTF-8 JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain one JSON object")
    return payload


def _verify_external_binding(
    binding: Mapping[str, str], *, root: Path, name: str
) -> None:
    path = _direct_file(binding["path"], root, name, payload_relative=True)
    digest = hashlib.sha256(_stable_file_bytes(path, name)).hexdigest()
    if digest != binding["sha256"]:
        raise Stage2VariableQueryWindowContractError(f"{name} SHA-256 mismatch")


@dataclass(frozen=True, init=False)
class VerifiedStage2VariableQueryWindow:
    """Verifier-issued immutable capability for one complete v2 manifest."""

    path: Path
    repository_root: Path
    payload: Mapping[str, Any]
    windows: tuple[Mapping[str, Any], ...]
    ordered_records: tuple[Mapping[str, Any], ...]
    manifest_sha256: str
    _capability: object

    def __init__(
        self,
        *,
        path: Path | None = None,
        repository_root: Path | None = None,
        payload: Mapping[str, Any] | None = None,
        manifest_sha256: str | None = None,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _VERIFIER_CAPABILITY:
            raise TypeError(
                "VerifiedStage2VariableQueryWindow is verifier-issued only"
            )
        if path is None or repository_root is None or payload is None or manifest_sha256 is None:
            raise RuntimeError("internal verifier capability construction is incomplete")
        frozen = _freeze(payload)
        windows = tuple(frozen["windows"])
        records = tuple(
            record
            for window in windows
            for partition in ("context_records", "query_records")
            for record in window[partition]
        )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "payload", frozen)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "ordered_records", records)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "_capability", _VERIFIER_CAPABILITY)


def assert_verified_stage2_variable_query_window(
    value: Any,
) -> VerifiedStage2VariableQueryWindow:
    if (
        type(value) is not VerifiedStage2VariableQueryWindow
        or getattr(value, "_capability", None) is not _VERIFIER_CAPABILITY
    ):
        raise TypeError("a verifier-issued variable-Q window capability is required")
    return value


def verify_stage2_variable_query_window(
    path: str | Path,
    expected_sha256: str,
    *,
    repository_root: str | Path | None = None,
) -> VerifiedStage2VariableQueryWindow:
    """Verify a stable non-symlink manifest and all result-free SHA bindings."""

    root = _repository_root(repository_root)
    expected = _sha256(expected_sha256, "expected_sha256")
    manifest_path = _direct_file(path, root, "window manifest", payload_relative=False)
    initial_bytes = _stable_file_bytes(manifest_path, "window manifest")
    initial_digest = hashlib.sha256(initial_bytes).hexdigest()
    if initial_digest != expected:
        raise Stage2VariableQueryWindowContractError(
            "window manifest SHA-256 mismatch"
        )
    payload = validate_stage2_variable_query_window_payload(
        _closed_json_object(initial_bytes, "window manifest")
    )
    _verify_external_binding(payload["role_binding"], root=root, name="role_binding")
    for name in sorted(BOUND_INPUT_NAMES):
        _verify_external_binding(
            payload["bound_inputs"][name],
            root=root,
            name=f"bound_inputs.{name}",
        )
    final_digest = hashlib.sha256(
        _stable_file_bytes(manifest_path, "window manifest final recheck")
    ).hexdigest()
    if final_digest != initial_digest:
        raise RuntimeError("window manifest changed during verification")
    return VerifiedStage2VariableQueryWindow(
        path=manifest_path,
        repository_root=root,
        payload=payload,
        manifest_sha256=initial_digest,
        _capability=_VERIFIER_CAPABILITY,
    )


__all__ = [
    "ARTIFACT_STATUS",
    "ARTIFACT_TYPE",
    "BOUND_INPUT_NAMES",
    "IDENTITY_BOUNDARIES",
    "ORDERED_RECORD_IDENTITY_ALGORITHM",
    "SCHEMA_VERSION",
    "Stage2VariableQueryWindowContractError",
    "VerifiedStage2VariableQueryWindow",
    "WINDOW_ID_ALGORITHM",
    "assert_verified_stage2_variable_query_window",
    "build_stage2_variable_query_window_payload",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "stage2_variable_query_record_identity",
    "validate_stage2_variable_query_window_payload",
    "verify_stage2_variable_query_window",
]
