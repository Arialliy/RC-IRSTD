"""Fail-closed Stage-2 No-Reject deployment protocol (schema v2).

This module is deliberately additive.  It does not alter or accept the legacy
fixed-query-size deployment protocol in :mod:`rc.schema`.  The v2 contract is
the result-free implementation of W10 in the RC4 Stage-2 work breakdown: one
ordered prefix of exactly fourteen unlabeled context records is followed by
the complete, unmodified suffix.  A complete three-budget threshold curve is
sealed before any query label may be attached.

The command line entry point is an artifact sealer, not a model-selection
entry point.  It verifies opaque checkpoint bytes and consumes a synthetic or
future-authorized ``decision_input`` embedded in the score manifest.  It
never opens query scores or labels and never recomputes/reselects a threshold.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import stat
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA_VERSION = "rc-irstd.no-reject-deployment-protocol.v2"
DECISION_SCHEMA_VERSION = "rc-irstd.no-reject-online-decision.v2"
SCORE_IDENTITY_SCHEMA_VERSION = "rc-irstd.stage2-deployment-score-identity.v1"

SOURCE_THAW_SHA256 = (
    "0e4f3e27026d5a2071a2c8f94f84c366d208f3789de17649aad926c64cd6b0b9"
)
WORK_BREAKDOWN_SHA256 = (
    "cc240f97aea6c99dde1e5c537a26c1b22e606b0f499ca495af71d15fa44c9d06"
)
AUTHORIZATION_AMENDMENT_SHA256 = (
    "185b7e4cac7d7a23ca537641575a00c5e64c6a5d0783dc34f999ba402f174845"
)

CONTEXT_SIZE = 14
CONTEXT_RULE = "first_C_in_frozen_split_order"
QUERY_RULE = "all_remaining_suffix"
PARTITION_RULE = "first_C_context_then_all_remaining_suffix_query"
THRESHOLD_SEMANTICS = "prediction = probability > threshold"
PIXEL_BUDGET_GRID = (1e-4, 1e-5, 1e-6)
BASE_SEEDS = (42, 123, 3407)
MAX_DERIVED_SEED = 2_147_483_646
METHOD_IDS = tuple(f"T{index}" for index in range(10))
OUTER_TARGET_DATASETS = {
    "outer_leave_nuaa_sirst": "nuaa-sirst",
    "outer_leave_nudt_sirst": "nudt-sirst",
    "outer_leave_irstd_1k": "irstd-1k",
}

_SHA256_HEX = frozenset("0123456789abcdef")
_ABSTENTION_FIELDS = frozenset(
    {
        "abstain",
        "abstained",
        "abstention",
        "p_min",
        "reject",
        "rejected",
        "reject_cutoff",
        "reject_probability",
        "reject_rule",
        "reject_score",
        "reject_supported",
        "target_reject_cutoff_override_allowed",
    }
)
_QUERY_LABEL_FIELDS = frozenset(
    {
        "ground_truth",
        "ground_truth_path",
        "gt",
        "gt_path",
        "label",
        "label_path",
        "mask",
        "mask_path",
        "target_mask",
    }
)


class Stage2DeploymentContractError(ValueError):
    """Raised when a v2 deployment artifact fails closed."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the single canonical JSON representation used by W10."""

    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise Stage2DeploymentContractError(
            f"payload is not finite canonical JSON: {error}"
        ) from error
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256(value: Any, name: str) -> str:
    """Require a lowercase, exact 64-hex SHA-256 string (booleans rejected)."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or value != value.lower() or not set(value) <= _SHA256_HEX:
        raise Stage2DeploymentContractError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - exact JSON boolean is intentional
        raise TypeError(f"{name} must be an exact JSON boolean")
    return value


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise Stage2DeploymentContractError(f"{name} must be >= {minimum}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise Stage2DeploymentContractError(f"{name} must be finite")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise Stage2DeploymentContractError(
            f"{name} must be non-empty with no surrounding whitespace"
        )
    return value


def _strict_timestamp_utc(value: Any, name: str) -> str:
    timestamp = _nonempty_string(value, name)
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise Stage2DeploymentContractError(
            f"{name} must be an exact UTC second timestamp YYYY-MM-DDTHH:MM:SSZ"
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != timestamp:
        raise Stage2DeploymentContractError(f"{name} is not canonical UTC")
    return timestamp


def _validate_seed_pair(base_seed: Any, derived_seed: Any, *, path: str) -> tuple[int, int]:
    base = _strict_int(base_seed, f"{path}.base_seed")
    if base not in BASE_SEEDS:
        raise Stage2DeploymentContractError(
            f"{path}.base_seed must be one of {list(BASE_SEEDS)}"
        )
    derived = _strict_int(derived_seed, f"{path}.derived_seed", minimum=1)
    if derived > MAX_DERIVED_SEED:
        raise Stage2DeploymentContractError(
            f"{path}.derived_seed must be <= {MAX_DERIVED_SEED}"
        )
    return base, derived


def _validate_outer_target(outer_fold_id: Any, target_dataset: Any, *, path: str) -> tuple[str, str]:
    outer = _nonempty_string(outer_fold_id, f"{path}.outer_fold_id")
    target = _nonempty_string(target_dataset, f"{path}.target_dataset")
    if outer not in OUTER_TARGET_DATASETS:
        raise Stage2DeploymentContractError(f"{path}.outer_fold_id is not frozen")
    if OUTER_TARGET_DATASETS[outer] != target:
        raise Stage2DeploymentContractError(
            f"{path}.target_dataset does not match outer_fold_id"
        )
    return outer, target


def _canonical_repository_root(repository_root: str | Path) -> Path:
    raw = Path(repository_root)
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise Stage2DeploymentContractError(
            "repository_root must be an absolute canonical non-symlink path"
        )
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2DeploymentContractError("repository_root does not exist") from error
    if resolved != raw or not resolved.is_dir():
        raise Stage2DeploymentContractError(
            "repository_root must be an absolute canonical directory"
        )
    return resolved


def _path_within_root(
    path: str | Path, repository_root: Path, *, name: str, must_exist: bool
) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise Stage2DeploymentContractError(
            f"{name} must be an absolute canonical non-symlink path"
        )
    try:
        parent = raw.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2DeploymentContractError(f"{name} parent does not exist") from error
    if parent != raw.parent or not parent.is_dir():
        raise Stage2DeploymentContractError(f"{name} parent is not canonical")
    if not parent.is_relative_to(repository_root):
        raise Stage2DeploymentContractError(f"{name} escapes repository_root")
    if must_exist:
        try:
            resolved = raw.resolve(strict=True)
        except FileNotFoundError as error:
            raise Stage2DeploymentContractError(f"{name} does not exist") from error
        if resolved != raw or not resolved.is_file() or raw.is_symlink():
            raise Stage2DeploymentContractError(
                f"{name} must be a canonical regular file without symlink aliases"
            )
    return raw


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2DeploymentContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise Stage2DeploymentContractError(f"non-finite JSON number is forbidden: {value}")


def _load_json_bytes(data: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2DeploymentContractError(f"invalid UTF-8 JSON in {name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def validate_input_file(path: str | Path, *, name: str) -> Path:
    """Reject relative, traversing, symlinked, and non-regular input paths."""

    raw = Path(path)
    if not raw.is_absolute():
        raise Stage2DeploymentContractError(f"{name} path must be absolute")
    if ".." in raw.parts:
        raise Stage2DeploymentContractError(f"{name} path traversal is forbidden")
    if raw.is_symlink():
        raise Stage2DeploymentContractError(f"{name} path must not be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2DeploymentContractError(f"{name} file does not exist") from error
    if resolved != raw:
        raise Stage2DeploymentContractError(
            f"{name} path must be canonical and contain no symlink aliases"
        )
    if not resolved.is_file():
        raise Stage2DeploymentContractError(f"{name} path must be a regular file")
    return resolved


def validate_output_file(path: str | Path, *, name: str = "output") -> Path:
    """Validate a new absolute output path without following symlink aliases."""

    raw = Path(path)
    if not raw.is_absolute():
        raise Stage2DeploymentContractError(f"{name} path must be absolute")
    if ".." in raw.parts:
        raise Stage2DeploymentContractError(f"{name} path traversal is forbidden")
    if raw.exists() or raw.is_symlink():
        raise Stage2DeploymentContractError(f"{name} path already exists")
    try:
        parent = raw.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2DeploymentContractError(f"{name} parent does not exist") from error
    if parent != raw.parent or not parent.is_dir():
        raise Stage2DeploymentContractError(
            f"{name} parent must be a canonical non-symlink directory"
        )
    return raw


def load_json_exact(
    path: str | Path,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[Mapping[str, Any], Path]:
    checked_path = validate_input_file(path, name=name)
    expected = validate_sha256(expected_sha256, f"{name}_sha256")
    data = checked_path.read_bytes()
    observed = sha256_bytes(data)
    if observed != expected:
        raise Stage2DeploymentContractError(
            f"{name} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return _load_json_bytes(data, name=name), checked_path


def _assert_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    name: str,
) -> None:
    observed = set(payload)
    missing = required - observed
    extra = observed - required
    if missing or extra:
        raise Stage2DeploymentContractError(
            f"{name} fields differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _reject_forbidden_keys(payload: Any, *, path: str = "payload") -> None:
    """Reject abstention metadata at any nesting depth."""

    if isinstance(payload, Mapping):
        forbidden = _ABSTENTION_FIELDS.intersection(payload)
        if forbidden:
            raise Stage2DeploymentContractError(
                f"{path} contains forbidden abstention fields: {sorted(forbidden)}"
            )
        for key, value in payload.items():
            _reject_forbidden_keys(value, path=f"{path}.{key}")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            _reject_forbidden_keys(value, path=f"{path}[{index}]")


def validate_deployment_protocol_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the exact W10 protocol-v2 contract."""

    if type(payload) is not dict:
        raise TypeError("deployment protocol must be an exact JSON object")
    _reject_forbidden_keys(payload, path="protocol")
    required = {
        "schema_version",
        "source_thaw_sha256",
        "work_breakdown_sha256",
        "authorization_amendment_sha256",
        "context_size",
        "context_rule",
        "query_rule",
        "partition_rule",
        "pixel_budget_grid",
        "threshold_semantics",
        "no_reject",
        "context_labels_loaded",
        "decision_sealed_before_query_labels",
        "online_updates_after_decision",
        "context_replacement_allowed",
        "query_truncation_allowed",
        "query_subsampling_allowed",
        "threshold_reselection_after_query_access",
        "config_sha256",
        "split_sha256",
        "context_rule_sha256",
        "geometry_sha256",
        "detector_checkpoint_sha256",
        "pre_open_plan_sha256",
        "confirmatory_identity_sha256",
    }
    _assert_exact_keys(payload, required=required, name="deployment protocol")
    if payload["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise Stage2DeploymentContractError("unsupported deployment protocol schema_version")
    if payload["source_thaw_sha256"] != SOURCE_THAW_SHA256:
        raise Stage2DeploymentContractError("protocol is not bound to the authorized source thaw")
    if payload["work_breakdown_sha256"] != WORK_BREAKDOWN_SHA256:
        raise Stage2DeploymentContractError("protocol is not bound to the W10 work breakdown")
    if payload["authorization_amendment_sha256"] != AUTHORIZATION_AMENDMENT_SHA256:
        raise Stage2DeploymentContractError(
            "protocol is not bound to the integrity authorization amendment"
        )
    context_size = _strict_int(payload["context_size"], "protocol.context_size", minimum=1)
    if context_size != CONTEXT_SIZE:
        raise Stage2DeploymentContractError("protocol.context_size must be exactly 14")
    expected_strings = {
        "context_rule": CONTEXT_RULE,
        "query_rule": QUERY_RULE,
        "partition_rule": PARTITION_RULE,
        "threshold_semantics": THRESHOLD_SEMANTICS,
    }
    for field, expected in expected_strings.items():
        if payload[field] != expected:
            raise Stage2DeploymentContractError(
                f"protocol.{field} must be exactly {expected!r}"
            )
    raw_grid = payload["pixel_budget_grid"]
    if type(raw_grid) is not list:
        raise TypeError("protocol.pixel_budget_grid must be an exact JSON array")
    grid = tuple(
        _finite_float(value, f"protocol.pixel_budget_grid[{index}]")
        for index, value in enumerate(raw_grid)
    )
    if grid != PIXEL_BUDGET_GRID:
        raise Stage2DeploymentContractError(
            f"protocol.pixel_budget_grid must be exactly {PIXEL_BUDGET_GRID}"
        )
    exact_boolean_values = {
        "no_reject": True,
        "context_labels_loaded": False,
        "decision_sealed_before_query_labels": True,
        "online_updates_after_decision": False,
        "context_replacement_allowed": False,
        "query_truncation_allowed": False,
        "query_subsampling_allowed": False,
        "threshold_reselection_after_query_access": False,
    }
    for field, expected in exact_boolean_values.items():
        observed = _strict_bool(payload[field], f"protocol.{field}")
        if observed is not expected:
            raise Stage2DeploymentContractError(
                f"protocol.{field} must be exactly {str(expected).lower()}"
            )
    result = dict(payload)
    result["pixel_budget_grid"] = list(PIXEL_BUDGET_GRID)
    for field in (
        "source_thaw_sha256",
        "work_breakdown_sha256",
        "authorization_amendment_sha256",
        "config_sha256",
        "split_sha256",
        "context_rule_sha256",
        "geometry_sha256",
        "detector_checkpoint_sha256",
        "pre_open_plan_sha256",
        "confirmatory_identity_sha256",
    ):
        result[field] = validate_sha256(payload[field], f"protocol.{field}")
    return result


def partition_first_c_all_remaining(
    records: Sequence[Mapping[str, Any]],
    context_size: int = CONTEXT_SIZE,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Return the exact first-C context and every remaining record.

    There is intentionally no ``query_size`` argument.  Requiring at least one
    suffix record prevents a degenerate decision from being silently accepted.
    The returned tuples contain every input object exactly once and preserve
    object identity and order.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    if isinstance(context_size, bool) or not isinstance(context_size, int):
        raise TypeError("context_size must be an integer")
    if context_size != CONTEXT_SIZE:
        raise Stage2DeploymentContractError("context_size must be exactly 14")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
    if len(records) <= context_size:
        raise Stage2DeploymentContractError(
            "first-C/all_remaining_suffix requires at least one query record"
        )
    context = tuple(records[:context_size])
    query = tuple(records[context_size:])
    if len(context) != CONTEXT_SIZE or len(context) + len(query) != len(records):
        raise AssertionError("partition implementation lost or duplicated records")
    if tuple(context) + tuple(query) != tuple(records):
        raise AssertionError("partition implementation changed record order")
    return context, query


def _record_identity(
    record: Mapping[str, Any],
    *,
    index: int,
    is_context: bool,
) -> dict[str, str]:
    allowed = {
        "image_id",
        "original_image_sha256",
        "score_sha256",
        "score_opened",
    }
    extra = set(record) - allowed
    if extra:
        if _QUERY_LABEL_FIELDS.intersection(extra):
            raise Stage2DeploymentContractError(
                f"records[{index}] contains query-label fields before sealing"
            )
        raise Stage2DeploymentContractError(
            f"records[{index}] contains unsupported fields: {sorted(extra)}"
        )
    required = {"image_id", "original_image_sha256", "score_sha256", "score_opened"}
    missing = required - set(record)
    if missing:
        raise Stage2DeploymentContractError(
            f"records[{index}] is missing fields: {sorted(missing)}"
        )
    image_id = _nonempty_string(record["image_id"], f"records[{index}].image_id")
    original_sha = validate_sha256(
        record["original_image_sha256"], f"records[{index}].original_image_sha256"
    )
    score_opened = _strict_bool(record["score_opened"], f"records[{index}].score_opened")
    score_sha = record["score_sha256"]
    if is_context:
        if score_opened is not True:
            raise Stage2DeploymentContractError("every context score must be opened")
        score_sha = validate_sha256(score_sha, f"records[{index}].score_sha256")
    else:
        if score_opened is not False:
            raise Stage2DeploymentContractError(
                "query scores must remain unopened when the decision is sealed"
            )
        if score_sha is not None:
            raise Stage2DeploymentContractError(
                "query score_sha256 must be null before decision sealing"
            )
    result = {
        "image_id": image_id,
        "original_image_sha256": original_sha,
    }
    if is_context:
        result["score_sha256"] = str(score_sha)
    return result


def _canonical_curve(curve: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    if isinstance(curve, (str, bytes)) or not isinstance(curve, Sequence):
        raise TypeError("threshold_curve must be a sequence")
    if len(curve) != len(PIXEL_BUDGET_GRID):
        raise Stage2DeploymentContractError("threshold_curve must have exactly 3 rows")
    result: list[dict[str, float]] = []
    previous_threshold = -math.inf
    for index, (row, expected_budget) in enumerate(zip(curve, PIXEL_BUDGET_GRID)):
        if not isinstance(row, Mapping):
            raise TypeError(f"threshold_curve[{index}] must be a mapping")
        _assert_exact_keys(
            row,
            required={"pixel_budget", "threshold"},
            name=f"threshold_curve[{index}]",
        )
        budget = _finite_float(row["pixel_budget"], f"threshold_curve[{index}].pixel_budget")
        threshold = _finite_float(row["threshold"], f"threshold_curve[{index}].threshold")
        if budget != expected_budget:
            raise Stage2DeploymentContractError(
                "threshold_curve budget order must be exactly [1e-4,1e-5,1e-6]"
            )
        if not 0.0 <= threshold <= 1.0:
            raise Stage2DeploymentContractError("curve thresholds must lie in [0, 1]")
        if threshold < previous_threshold:
            raise Stage2DeploymentContractError(
                "threshold curve must be nondecreasing as the budget becomes stricter"
            )
        result.append({"pixel_budget": budget, "threshold": threshold})
        previous_threshold = threshold
    return result


def seal_no_reject_curve_decision(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    protocol_artifact_bytes: bytes | None = None,
    records: Sequence[Mapping[str, Any]],
    threshold_curve: Sequence[Mapping[str, Any]],
    score_manifest_sha256: str,
    calibrator_checkpoint_sha256: str,
    outer_fold_id: str,
    target_dataset: str,
    method_id: str,
    base_seed: int,
    derived_seed: int,
    decision_timestamp_utc: str,
    query_labels_attached: bool = False,
    threshold_reselected: bool = False,
    online_update_count: int = 0,
) -> dict[str, Any]:
    """Validate and seal one immutable No-Reject three-budget decision."""

    canonical_protocol = validate_deployment_protocol_v2(protocol)
    canonical_protocol_sha = sha256_bytes(canonical_json_bytes(canonical_protocol))
    supplied_protocol_sha = validate_sha256(protocol_sha256, "protocol_sha256")
    if protocol_artifact_bytes is None:
        if supplied_protocol_sha != canonical_protocol_sha:
            raise Stage2DeploymentContractError(
                "protocol_sha256 does not bind the canonical validated protocol"
            )
    else:
        if not isinstance(protocol_artifact_bytes, bytes):
            raise TypeError("protocol_artifact_bytes must be bytes")
        if sha256_bytes(protocol_artifact_bytes) != supplied_protocol_sha:
            raise Stage2DeploymentContractError("protocol artifact SHA-256 mismatch")
        artifact_protocol = validate_deployment_protocol_v2(
            _load_json_bytes(protocol_artifact_bytes, name="protocol_artifact")
        )
        if canonical_json_bytes(artifact_protocol) != canonical_json_bytes(
            canonical_protocol
        ):
            raise Stage2DeploymentContractError(
                "protocol artifact differs from the validated protocol mapping"
            )
    if _strict_bool(query_labels_attached, "query_labels_attached") is not False:
        raise Stage2DeploymentContractError("query labels must not be attached before sealing")
    if _strict_bool(threshold_reselected, "threshold_reselected") is not False:
        raise Stage2DeploymentContractError("threshold reselection is forbidden")
    if _strict_int(online_update_count, "online_update_count", minimum=0) != 0:
        raise Stage2DeploymentContractError("online threshold updates are forbidden")

    context, query = partition_first_c_all_remaining(
        records, context_size=canonical_protocol["context_size"]
    )
    context_identity = [
        _record_identity(record, index=index, is_context=True)
        for index, record in enumerate(context)
    ]
    query_identity = [
        _record_identity(record, index=CONTEXT_SIZE + index, is_context=False)
        for index, record in enumerate(query)
    ]
    all_ids = [row["image_id"] for row in context_identity + query_identity]
    all_image_shas = [
        row["original_image_sha256"] for row in context_identity + query_identity
    ]
    if len(set(all_ids)) != len(all_ids):
        raise Stage2DeploymentContractError("ordered records contain duplicate image_id values")
    if len(set(all_image_shas)) != len(all_image_shas):
        raise Stage2DeploymentContractError(
            "ordered records contain duplicate original-image SHA-256 values"
        )
    curve = _canonical_curve(threshold_curve)
    curve_sha = sha256_bytes(canonical_json_bytes(curve))
    timestamp = _strict_timestamp_utc(
        decision_timestamp_utc, "decision_timestamp_utc"
    )
    outer, target = _validate_outer_target(
        outer_fold_id, target_dataset, path="decision_input"
    )
    base, derived = _validate_seed_pair(
        base_seed, derived_seed, path="decision_input"
    )
    method = _nonempty_string(method_id, "method_id")
    if method not in METHOD_IDS:
        raise Stage2DeploymentContractError(
            f"method_id must be one of {list(METHOD_IDS)}"
        )
    context_score_shas = [row["score_sha256"] for row in context_identity]
    if len(set(context_score_shas)) != len(context_score_shas):
        raise Stage2DeploymentContractError(
            "ordered context contains duplicate score SHA-256 values"
        )

    result: dict[str, Any] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "authorization_amendment_sha256": AUTHORIZATION_AMENDMENT_SHA256,
        "protocol_sha256": supplied_protocol_sha,
        "validated_protocol_canonical_sha256": canonical_protocol_sha,
        "config_sha256": canonical_protocol["config_sha256"],
        "split_sha256": canonical_protocol["split_sha256"],
        "context_rule_sha256": canonical_protocol["context_rule_sha256"],
        "geometry_sha256": canonical_protocol["geometry_sha256"],
        "detector_checkpoint_sha256": canonical_protocol[
            "detector_checkpoint_sha256"
        ],
        "calibrator_checkpoint_sha256": validate_sha256(
            calibrator_checkpoint_sha256, "calibrator_checkpoint_sha256"
        ),
        "score_manifest_sha256": validate_sha256(
            score_manifest_sha256, "score_manifest_sha256"
        ),
        "pre_open_plan_sha256": canonical_protocol["pre_open_plan_sha256"],
        "confirmatory_identity_sha256": canonical_protocol[
            "confirmatory_identity_sha256"
        ],
        "outer_fold_id": outer,
        "target_dataset": target,
        "method_id": method,
        "base_seed": base,
        "derived_seed": derived,
        "context_size": CONTEXT_SIZE,
        "context_rule": CONTEXT_RULE,
        "query_rule": QUERY_RULE,
        "query_count": len(query_identity),
        "ordered_context_identity": context_identity,
        "ordered_context_identity_sha256": sha256_bytes(
            canonical_json_bytes(context_identity)
        ),
        "ordered_query_identity": query_identity,
        "ordered_query_identity_sha256": sha256_bytes(
            canonical_json_bytes(query_identity)
        ),
        "threshold_semantics": THRESHOLD_SEMANTICS,
        "threshold_curve": curve,
        "threshold_curve_sha256": curve_sha,
        "no_reject": True,
        "decision_sealed": True,
        "query_labels_attached_at_seal": False,
        "query_scores_opened_at_seal": False,
        "threshold_reselected": False,
        "online_update_count": 0,
        "context_replaced": False,
        "query_truncated": False,
        "query_subsampled": False,
        "decision_timestamp_utc": timestamp,
    }
    _reject_forbidden_keys(result, path="decision")
    result["decision_payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def _validate_decision_identity_rows(
    rows: Any, *, is_context: bool
) -> list[dict[str, str]]:
    name = "ordered_context_identity" if is_context else "ordered_query_identity"
    if type(rows) is not list:
        raise TypeError(f"decision.{name} must be an exact JSON array")
    expected_length = CONTEXT_SIZE if is_context else None
    if expected_length is not None and len(rows) != expected_length:
        raise Stage2DeploymentContractError(
            "decision must contain exactly 14 context identities"
        )
    if not is_context and not rows:
        raise Stage2DeploymentContractError(
            "decision must contain the nonempty complete suffix"
        )
    required = {"image_id", "original_image_sha256"}
    if is_context:
        required.add("score_sha256")
    canonical: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        row_name = f"decision.{name}[{index}]"
        if type(row) is not dict:
            raise TypeError(f"{row_name} must be an exact JSON object")
        _assert_exact_keys(row, required=required, name=row_name)
        item = {
            "image_id": _nonempty_string(row["image_id"], f"{row_name}.image_id"),
            "original_image_sha256": validate_sha256(
                row["original_image_sha256"],
                f"{row_name}.original_image_sha256",
            ),
        }
        if is_context:
            item["score_sha256"] = validate_sha256(
                row["score_sha256"], f"{row_name}.score_sha256"
            )
        canonical.append(item)
    return canonical


def _verify_decision_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Perform complete semantic validation after external-byte binding."""

    if type(payload) is not dict:
        raise TypeError("decision must be an exact JSON object")
    _reject_forbidden_keys(payload, path="decision")
    expected_fields = {
        "schema_version",
        "source_thaw_sha256",
        "work_breakdown_sha256",
        "authorization_amendment_sha256",
        "protocol_sha256",
        "validated_protocol_canonical_sha256",
        "config_sha256",
        "split_sha256",
        "context_rule_sha256",
        "geometry_sha256",
        "detector_checkpoint_sha256",
        "pre_open_plan_sha256",
        "confirmatory_identity_sha256",
        "calibrator_checkpoint_sha256",
        "score_manifest_sha256",
        "outer_fold_id",
        "target_dataset",
        "method_id",
        "base_seed",
        "derived_seed",
        "context_size",
        "context_rule",
        "query_rule",
        "query_count",
        "ordered_context_identity",
        "ordered_context_identity_sha256",
        "ordered_query_identity",
        "ordered_query_identity_sha256",
        "threshold_semantics",
        "threshold_curve",
        "threshold_curve_sha256",
        "no_reject",
        "decision_sealed",
        "query_labels_attached_at_seal",
        "query_scores_opened_at_seal",
        "threshold_reselected",
        "online_update_count",
        "context_replaced",
        "query_truncated",
        "query_subsampled",
        "decision_timestamp_utc",
        "decision_payload_sha256",
    }
    _assert_exact_keys(payload, required=expected_fields, name="sealed decision")
    exact_bindings = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "authorization_amendment_sha256": AUTHORIZATION_AMENDMENT_SHA256,
        "context_rule": CONTEXT_RULE,
        "query_rule": QUERY_RULE,
        "threshold_semantics": THRESHOLD_SEMANTICS,
    }
    for field, expected in exact_bindings.items():
        observed = _nonempty_string(payload[field], f"decision.{field}")
        if observed != expected:
            raise Stage2DeploymentContractError(
                f"decision.{field} binding changed"
            )
    for field in (
        "source_thaw_sha256",
        "work_breakdown_sha256",
        "authorization_amendment_sha256",
        "protocol_sha256",
        "validated_protocol_canonical_sha256",
        "config_sha256",
        "split_sha256",
        "context_rule_sha256",
        "geometry_sha256",
        "detector_checkpoint_sha256",
        "pre_open_plan_sha256",
        "confirmatory_identity_sha256",
        "calibrator_checkpoint_sha256",
        "score_manifest_sha256",
        "ordered_context_identity_sha256",
        "ordered_query_identity_sha256",
        "threshold_curve_sha256",
        "decision_payload_sha256",
    ):
        validate_sha256(payload[field], f"decision.{field}")

    outer, target = _validate_outer_target(
        payload["outer_fold_id"], payload["target_dataset"], path="decision"
    )
    method = _nonempty_string(payload["method_id"], "decision.method_id")
    if method not in METHOD_IDS:
        raise Stage2DeploymentContractError(
            f"decision.method_id must be one of {list(METHOD_IDS)}"
        )
    _validate_seed_pair(payload["base_seed"], payload["derived_seed"], path="decision")
    if _strict_int(payload["context_size"], "decision.context_size") != CONTEXT_SIZE:
        raise Stage2DeploymentContractError(
            "decision context_size must be exactly 14"
        )
    query_count = _strict_int(
        payload["query_count"], "decision.query_count", minimum=1
    )
    _strict_timestamp_utc(
        payload["decision_timestamp_utc"], "decision.decision_timestamp_utc"
    )

    booleans = {
        "no_reject": True,
        "decision_sealed": True,
        "query_labels_attached_at_seal": False,
        "query_scores_opened_at_seal": False,
        "threshold_reselected": False,
        "context_replaced": False,
        "query_truncated": False,
        "query_subsampled": False,
    }
    for field, expected in booleans.items():
        if _strict_bool(payload[field], f"decision.{field}") is not expected:
            raise Stage2DeploymentContractError(f"decision.{field} changed")
    if _strict_int(
        payload["online_update_count"], "decision.online_update_count", minimum=0
    ) != 0:
        raise Stage2DeploymentContractError(
            "decision online_update_count must remain zero"
        )

    context_identity = _validate_decision_identity_rows(
        payload["ordered_context_identity"], is_context=True
    )
    query_identity = _validate_decision_identity_rows(
        payload["ordered_query_identity"], is_context=False
    )
    if query_count != len(query_identity):
        raise Stage2DeploymentContractError("decision query_count mismatch")
    all_rows = context_identity + query_identity
    all_ids = [row["image_id"] for row in all_rows]
    all_image_shas = [row["original_image_sha256"] for row in all_rows]
    context_score_shas = [row["score_sha256"] for row in context_identity]
    if len(set(all_ids)) != len(all_ids):
        raise Stage2DeploymentContractError(
            "decision identities contain duplicate image_id values"
        )
    if len(set(all_image_shas)) != len(all_image_shas):
        raise Stage2DeploymentContractError(
            "decision identities contain duplicate original-image SHA-256 values"
        )
    if len(set(context_score_shas)) != len(context_score_shas):
        raise Stage2DeploymentContractError(
            "decision context identities contain duplicate score SHA-256 values"
        )
    if sha256_bytes(canonical_json_bytes(context_identity)) != payload[
        "ordered_context_identity_sha256"
    ]:
        raise Stage2DeploymentContractError(
            "ordered context identity hash mismatch"
        )
    if sha256_bytes(canonical_json_bytes(query_identity)) != payload[
        "ordered_query_identity_sha256"
    ]:
        raise Stage2DeploymentContractError("ordered query identity hash mismatch")

    if type(payload["threshold_curve"]) is not list:
        raise TypeError("decision.threshold_curve must be an exact JSON array")
    if any(type(row) is not dict for row in payload["threshold_curve"]):
        raise TypeError(
            "every decision.threshold_curve row must be an exact JSON object"
        )
    curve = _canonical_curve(payload["threshold_curve"])
    if sha256_bytes(canonical_json_bytes(curve)) != payload[
        "threshold_curve_sha256"
    ]:
        raise Stage2DeploymentContractError("threshold curve hash mismatch")

    without_self_hash = dict(payload)
    supplied_self_hash = without_self_hash.pop("decision_payload_sha256")
    if sha256_bytes(canonical_json_bytes(without_self_hash)) != supplied_self_hash:
        raise Stage2DeploymentContractError("decision payload self-hash mismatch")
    result = dict(payload)
    result["outer_fold_id"] = outer
    result["target_dataset"] = target
    result["ordered_context_identity"] = context_identity
    result["ordered_query_identity"] = query_identity
    result["threshold_curve"] = curve
    return result


def verify_sealed_no_reject_decision(
    payload: Mapping[str, Any],
    expected_artifact_sha256: str,
    *,
    artifact_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Verify a decision against a caller-supplied, trusted artifact digest.

    ``expected_artifact_sha256`` is deliberately mandatory.  Internal identity
    and self hashes are not a trust anchor because an attacker can recompute
    them after mutation.  File consumers should normally call
    :func:`load_verified_sealed_decision`, which hashes the exact file bytes.
    """

    expected = validate_sha256(
        expected_artifact_sha256, "expected_decision_artifact_sha256"
    )
    if artifact_bytes is None:
        observed = sha256_bytes(canonical_json_bytes(payload))
    else:
        if not isinstance(artifact_bytes, bytes):
            raise TypeError("artifact_bytes must be bytes")
        observed = sha256_bytes(artifact_bytes)
        parsed = _load_json_bytes(artifact_bytes, name="sealed decision artifact")
        if canonical_json_bytes(parsed) != canonical_json_bytes(payload):
            raise Stage2DeploymentContractError(
                "decision mapping differs from exact artifact bytes"
            )
    if observed != expected:
        raise Stage2DeploymentContractError(
            "decision artifact SHA-256 mismatch: "
            f"observed={observed}, expected={expected}"
        )
    return _verify_decision_payload(payload)


def _read_descriptor_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def load_verified_sealed_decision(
    path: str | Path,
    expected_sha256: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Load one canonical in-repository decision with two byte-level reads."""

    root = _canonical_repository_root(repository_root)
    checked = _path_within_root(
        path, root, name="sealed decision", must_exist=True
    )
    expected = validate_sha256(expected_sha256, "expected_decision_sha256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(checked, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2DeploymentContractError(
                "sealed decision is not a regular file"
            )
        data_before = _read_descriptor_all(descriptor)
        if sha256_bytes(data_before) != expected:
            raise Stage2DeploymentContractError(
                "sealed decision SHA-256 does not match external expectation"
            )
        payload = _load_json_bytes(data_before, name="sealed decision")
        verified = verify_sealed_no_reject_decision(
            payload, expected, artifact_bytes=data_before
        )
        data_after = _read_descriptor_all(descriptor)
        after = os.fstat(descriptor)
        path_after = os.stat(checked, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        identity_path = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if identity_before != identity_after or identity_before != identity_path:
            raise Stage2DeploymentContractError(
                "sealed decision changed during verified consumption"
            )
        if data_after != data_before or sha256_bytes(data_after) != expected:
            raise Stage2DeploymentContractError(
                "sealed decision bytes changed during verified consumption"
            )
        return verified
    finally:
        os.close(descriptor)


def _pretty_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise Stage2DeploymentContractError(
            f"output is not finite JSON: {error}"
        ) from error


def _transactional_publish_bundle(files: Mapping[Path, bytes]) -> None:
    """Publish an all-new bundle, rolling back inode-matching members."""

    if not files:
        raise Stage2DeploymentContractError("empty output bundle")
    targets = list(files)
    if len(set(targets)) != len(targets):
        raise Stage2DeploymentContractError("duplicate bundle target")
    parent = targets[0].parent
    if any(path.parent != parent for path in targets):
        raise Stage2DeploymentContractError(
            "bundle targets must share one directory"
        )
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise Stage2DeploymentContractError(
            "bundle parent must be canonical"
        )
    for target in targets:
        if not target.is_absolute() or ".." in target.parts:
            raise Stage2DeploymentContractError(
                "bundle target must be absolute canonical"
            )
        try:
            os.lstat(target)
        except FileNotFoundError:
            pass
        else:
            raise Stage2DeploymentContractError(
                f"bundle target already exists or is a symlink: {target.name}"
            )
    staged: list[tuple[Path, Path, tuple[int, int]]] = []
    linked: list[tuple[Path, tuple[int, int]]] = []
    try:
        for target, data in files.items():
            descriptor, temporary_name = tempfile.mkstemp(
                dir=parent, prefix=f".{target.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_stat = os.stat(temporary, follow_symlinks=False)
            staged.append(
                (target, temporary, (temporary_stat.st_dev, temporary_stat.st_ino))
            )
        for target, _, _ in staged:
            try:
                os.lstat(target)
            except FileNotFoundError:
                continue
            raise Stage2DeploymentContractError(
                f"bundle target appeared during staging: {target.name}"
            )
        for target, temporary, identity in staged:
            os.link(temporary, target, follow_symlinks=False)
            linked.append((target, identity))
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for target, identity in reversed(linked):
            try:
                observed = os.stat(target, follow_symlinks=False)
                if (observed.st_dev, observed.st_ino) == identity:
                    os.unlink(target)
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--score-manifest-sha256", required=True)
    parser.add_argument("--calibrator-checkpoint", required=True)
    parser.add_argument("--calibrator-checkpoint-sha256", required=True)
    parser.add_argument("--pre-open-plan", required=True)
    parser.add_argument("--pre-open-plan-sha256", required=True)
    parser.add_argument("--confirmatory-identity", required=True)
    parser.add_argument("--confirmatory-identity-sha256", required=True)
    parser.add_argument("--expected-decision-sha256")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _canonical_repository_root(args.repository_root)
    protocol_input = _path_within_root(
        args.protocol, root, name="protocol", must_exist=True
    )
    manifest_input = _path_within_root(
        args.score_manifest, root, name="score_manifest", must_exist=True
    )
    checkpoint_input = _path_within_root(
        args.calibrator_checkpoint,
        root,
        name="calibrator_checkpoint",
        must_exist=True,
    )
    pre_open_plan_input = _path_within_root(
        args.pre_open_plan, root, name="pre_open_plan", must_exist=True
    )
    confirmatory_identity_input = _path_within_root(
        args.confirmatory_identity,
        root,
        name="confirmatory_identity",
        must_exist=True,
    )
    output = _path_within_root(
        args.output, root, name="output", must_exist=False
    )
    output = validate_output_file(output)
    sidecar = _path_within_root(
        output.with_name(output.name + ".sha256"),
        root,
        name="output sidecar",
        must_exist=False,
    )
    validate_output_file(sidecar, name="output sidecar")

    protocol, protocol_path = load_json_exact(
        protocol_input, args.protocol_sha256, name="protocol"
    )
    canonical_protocol = validate_deployment_protocol_v2(protocol)
    try:
        from scripts.freeze_stage2_confirmatory_plan import (
            load_verified_pre_open_plan,
        )
        from scripts.materialize_stage2_confirmatory_identity import (
            load_verified_confirmatory_identity,
        )

        pre_open_plan = load_verified_pre_open_plan(
            pre_open_plan_input, args.pre_open_plan_sha256, root
        )
        confirmatory_identity = load_verified_confirmatory_identity(
            confirmatory_identity_input, args.confirmatory_identity_sha256, root
        )
    except (TypeError, ValueError) as error:
        raise Stage2DeploymentContractError(
            f"invalid W12 provenance artifact: {error}"
        ) from error
    if canonical_protocol["pre_open_plan_sha256"] != args.pre_open_plan_sha256:
        raise Stage2DeploymentContractError(
            "protocol pre_open_plan_sha256 differs from external expectation"
        )
    if (
        canonical_protocol["confirmatory_identity_sha256"]
        != args.confirmatory_identity_sha256
    ):
        raise Stage2DeploymentContractError(
            "protocol confirmatory_identity_sha256 differs from external expectation"
        )
    if (
        confirmatory_identity["pre_open_plan_sha256"]
        != args.pre_open_plan_sha256
    ):
        raise Stage2DeploymentContractError(
            "confirmatory identity does not replay-bind the supplied pre-open plan"
        )
    if (
        confirmatory_identity["target_dataset"]
        != pre_open_plan["target_dataset"]
    ):
        raise Stage2DeploymentContractError(
            "confirmatory identity target differs from the pre-open plan"
        )
    if (
        confirmatory_identity["split_repository_relative_path"]
        != pre_open_plan["split_repository_relative_path"]
        or confirmatory_identity["split_sha256"]
        != pre_open_plan["split_expected_sha256"]
        or confirmatory_identity["split_record_count"]
        != pre_open_plan["split_expected_record_count"]
        or canonical_protocol["split_sha256"]
        != confirmatory_identity["split_sha256"]
    ):
        raise Stage2DeploymentContractError(
            "protocol/plan/confirmatory split provenance does not replay exactly"
        )
    score_manifest, manifest_path = load_json_exact(
        manifest_input,
        args.score_manifest_sha256,
        name="score_manifest",
    )
    checkpoint_path = validate_input_file(
        checkpoint_input, name="calibrator_checkpoint"
    )
    checkpoint_sha = validate_sha256(
        args.calibrator_checkpoint_sha256, "calibrator_checkpoint_sha256"
    )
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise Stage2DeploymentContractError(
            "calibrator checkpoint SHA-256 mismatch"
        )

    required_manifest = {
        "schema_version",
        "ordered_records",
        "decision_input",
        "official_test_labels_accessed",
        "official_test_query_scores_accessed",
    }
    _assert_exact_keys(
        score_manifest,
        required=required_manifest,
        name="deployment score identity manifest",
    )
    if score_manifest["schema_version"] != SCORE_IDENTITY_SCHEMA_VERSION:
        raise Stage2DeploymentContractError(
            "unsupported score identity schema_version"
        )
    if _strict_bool(
        score_manifest["official_test_labels_accessed"],
        "score_manifest.official_test_labels_accessed",
    ) is not False:
        raise Stage2DeploymentContractError(
            "official-test labels were accessed before sealing"
        )
    if _strict_bool(
        score_manifest["official_test_query_scores_accessed"],
        "score_manifest.official_test_query_scores_accessed",
    ) is not False:
        raise Stage2DeploymentContractError(
            "query scores were accessed before sealing"
        )
    decision_input = score_manifest["decision_input"]
    if type(decision_input) is not dict:
        raise TypeError(
            "score_manifest.decision_input must be an exact JSON object"
        )
    required_input = {
        "threshold_curve",
        "outer_fold_id",
        "target_dataset",
        "method_id",
        "base_seed",
        "derived_seed",
        "decision_timestamp_utc",
        "query_labels_attached",
        "threshold_reselected",
        "online_update_count",
    }
    _assert_exact_keys(
        decision_input, required=required_input, name="decision_input"
    )
    records = score_manifest["ordered_records"]
    if type(records) is not list:
        raise TypeError(
            "score_manifest.ordered_records must be an exact JSON array"
        )
    if decision_input["target_dataset"] != confirmatory_identity["target_dataset"]:
        raise Stage2DeploymentContractError(
            "decision target differs from confirmatory identity"
        )
    identity_rows = (
        confirmatory_identity["ordered_context_identity"]
        + confirmatory_identity["ordered_query_identity"]
    )
    if len(records) != len(identity_rows):
        raise Stage2DeploymentContractError(
            "score manifest record count differs from confirmatory identity"
        )
    for index, (record, identity_row) in enumerate(zip(records, identity_rows)):
        if type(record) is not dict:
            raise TypeError(f"score_manifest.ordered_records[{index}] must be exact object")
        if (
            record.get("image_id") != identity_row["canonical_id"]
            or record.get("original_image_sha256")
            != identity_row["original_image_sha256"]
        ):
            raise Stage2DeploymentContractError(
                f"score manifest record {index} differs from confirmatory identity"
            )
    protocol_bytes = protocol_path.read_bytes()
    if sha256_bytes(protocol_bytes) != validate_sha256(
        args.protocol_sha256, "protocol_sha256"
    ):
        raise Stage2DeploymentContractError(
            "protocol changed during decision sealing"
        )
    decision = seal_no_reject_curve_decision(
        protocol=canonical_protocol,
        protocol_sha256=args.protocol_sha256,
        protocol_artifact_bytes=protocol_bytes,
        records=records,
        threshold_curve=decision_input["threshold_curve"],
        score_manifest_sha256=args.score_manifest_sha256,
        calibrator_checkpoint_sha256=checkpoint_sha,
        outer_fold_id=decision_input["outer_fold_id"],
        target_dataset=decision_input["target_dataset"],
        method_id=decision_input["method_id"],
        base_seed=decision_input["base_seed"],
        derived_seed=decision_input["derived_seed"],
        decision_timestamp_utc=decision_input["decision_timestamp_utc"],
        query_labels_attached=decision_input["query_labels_attached"],
        threshold_reselected=decision_input["threshold_reselected"],
        online_update_count=decision_input["online_update_count"],
    )
    decision_bytes = _pretty_json_bytes(decision)
    decision_sha = sha256_bytes(decision_bytes)
    verify_sealed_no_reject_decision(
        decision, decision_sha, artifact_bytes=decision_bytes
    )
    if args.expected_decision_sha256 is not None:
        expected_output = validate_sha256(
            args.expected_decision_sha256, "expected_decision_sha256"
        )
        if decision_sha != expected_output:
            raise Stage2DeploymentContractError(
                "generated decision differs from externally expected SHA-256"
            )

    if sha256_file(protocol_path) != args.protocol_sha256:
        raise Stage2DeploymentContractError(
            "protocol changed before output publication"
        )
    if sha256_file(manifest_path) != args.score_manifest_sha256:
        raise Stage2DeploymentContractError(
            "score manifest changed before output publication"
        )
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise Stage2DeploymentContractError(
            "calibrator checkpoint changed before output publication"
        )
    if load_verified_pre_open_plan(
        pre_open_plan_input, args.pre_open_plan_sha256, root
    ) != pre_open_plan:
        raise Stage2DeploymentContractError(
            "pre-open plan changed before decision publication"
        )
    if load_verified_confirmatory_identity(
        confirmatory_identity_input, args.confirmatory_identity_sha256, root
    ) != confirmatory_identity:
        raise Stage2DeploymentContractError(
            "confirmatory identity changed before decision publication"
        )
    sidecar_bytes = f"{decision_sha}  {output.name}\n".encode("ascii")
    _transactional_publish_bundle(
        {output: decision_bytes, sidecar: sidecar_bytes}
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
