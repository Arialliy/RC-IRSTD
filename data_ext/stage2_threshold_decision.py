"""Fail-closed Stage-2 threshold-decision bundle contracts.

The outer-target label path imports this module.  A label may be opened only
after a complete, immutable and externally SHA-256-bound T0--T8 bundle has
been verified.  T9 is deliberately a different, post-label diagnostic and is
never accepted by this module.

The verifier performs no score, image, mask or label I/O.  It only opens the
small JSON decision artifacts and their SHA-256 sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DECISION_SCHEMA = "rc-irstd.stage2-threshold-decision.v1"
DECISION_SET_SCHEMA = "rc-irstd.stage2-threshold-decision-set.v1"
DECISION_SET_COMMIT_SCHEMA = "rc-irstd.stage2-threshold-decision-set-commit.v1"
DECISION_ARTIFACT_TYPE = "rc_irstd_stage2_threshold_decision"
DECISION_SET_ARTIFACT_TYPE = "rc_irstd_stage2_threshold_decision_set"
DECISION_SET_COMMIT_ARTIFACT_TYPE = "rc_irstd_stage2_threshold_decision_set_commit"

PRELABEL_METHOD_ORDER = tuple(f"T{index}" for index in range(9))
PIXEL_BUDGET_GRID = (1e-4, 1e-5, 1e-6)
STRICT_THRESHOLD_SEMANTICS = "prediction = probability > threshold"
COMPLETE_OUTCOME = "SEALED_COMPLETE"
T5_MISSING_OUTCOME = "SEALED_MISSING_INSUFFICIENT_TAIL_NO_FALLBACK"

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = object()
_FORBIDDEN_KEY_PARTS = (
    "ground_truth",
    "target_mask",
    "query_mask",
    "query_label",
    "label_path",
    "mask_path",
    "curve_path",
    "reject",
    "abstain",
    "fallback_threshold",
)
_ALLOWED_GUARD_KEYS = frozenset(
    {
        "query_labels_or_masks_opened",
        "reject_supported",
        "fallback_used",
    }
)

_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "query_labels_or_masks_opened",
        "training_performed",
        "gpu_used",
        "method_id",
        "method_name",
        "prelabel_eligible",
        "diagnostic_only",
        "outcome",
        "budget_grid",
        "thresholds",
        "prediction_semantics",
        "reject_supported",
        "fallback_used",
        "outer_fold_id",
        "outer_target_domain",
        "base_seed",
        "derived_seed",
        "shared_bindings",
        "method_binding",
        "method_contract_sha256",
        "decision_content_sha256_algorithm",
        "decision_content_sha256",
    }
)
_SET_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "query_labels_or_masks_opened",
        "training_performed",
        "gpu_used",
        "method_order",
        "budget_grid",
        "outer_fold_id",
        "outer_target_domain",
        "base_seed",
        "derived_seed",
        "shared_bindings",
        "decisions",
        "decision_set_content_sha256_algorithm",
        "decision_set_content_sha256",
    }
)
_SET_DECISION_FIELDS = frozenset(
    {"method_id", "path", "sha256", "outcome", "decision_content_sha256"}
)
_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "decision_set_file",
        "decision_set_sha256",
        "decision_files",
        "inventory_sha256_algorithm",
        "inventory_sha256",
    }
)
_COMMIT_MEMBER_FIELDS = frozenset({"method_id", "path", "sha256"})
_SHARED_BINDING_FIELDS = frozenset(
    {
        "context_package",
        "context_package_commit",
        "window_id",
        "window_identity_sha256",
        "ordered_query_identity_sha256",
        "score_manifest_sha256",
        "score_records_content_sha256",
        "detector_checkpoint_sha256",
        "shared_input_identity_sha256",
    }
)
_FILE_BINDING_FIELDS = frozenset({"path", "sha256"})


class Stage2ThresholdDecisionContractError(ValueError):
    """Raised when a prelabel threshold-decision bundle fails closed."""


@dataclass(frozen=True, init=False)
class VerifiedStage2ThresholdDecision:
    path: Path
    payload: Mapping[str, Any]
    manifest_sha256: str

    def __init__(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
        manifest_sha256: str,
        _token: object,
    ) -> None:
        if _token is not _TOKEN:
            raise TypeError("VerifiedStage2ThresholdDecision is verifier-only")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "payload", MappingProxyType(dict(payload)))
        object.__setattr__(self, "manifest_sha256", manifest_sha256)


@dataclass(frozen=True, init=False)
class VerifiedStage2ThresholdDecisionSet:
    path: Path
    payload: Mapping[str, Any]
    manifest_sha256: str
    decisions: tuple[VerifiedStage2ThresholdDecision, ...]
    decision_by_method: Mapping[str, VerifiedStage2ThresholdDecision]

    def __init__(
        self,
        *,
        path: Path,
        payload: Mapping[str, Any],
        manifest_sha256: str,
        decisions: Sequence[VerifiedStage2ThresholdDecision],
        _token: object,
    ) -> None:
        if _token is not _TOKEN:
            raise TypeError("VerifiedStage2ThresholdDecisionSet is verifier-only")
        ordered = tuple(decisions)
        mapping = {item.payload["method_id"]: item for item in ordered}
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "payload", MappingProxyType(dict(payload)))
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "decisions", ordered)
        object.__setattr__(self, "decision_by_method", MappingProxyType(mapping))


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2ThresholdDecisionContractError(
            f"payload is not finite canonical JSON: {error}"
        ) from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2ThresholdDecisionContractError(f"{path} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"{path} changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _strict_json(path: Path) -> tuple[dict[str, Any], str]:
    before = sha256_file(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise Stage2ThresholdDecisionContractError(f"{path} is not regular")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            payload = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Stage2ThresholdDecisionContractError(f"invalid JSON {path}: {error}") from error
    finally:
        os.close(descriptor)
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"{path} changed while reading")
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload, before


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2ThresholdDecisionContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise Stage2ThresholdDecisionContractError(
            f"{name} must be a lowercase 64-character SHA-256"
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Stage2ThresholdDecisionContractError(f"{name} must be non-empty text")
    return value


def _exact_bool(value: Any, expected: bool, name: str) -> None:
    if type(value) is not bool or value is not expected:  # noqa: E721
        raise Stage2ThresholdDecisionContractError(f"{name} must be {expected}")


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage2ThresholdDecisionContractError(f"{name} must be int >= {minimum}")
    return value


def _exact_keys(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    actual = frozenset(value)
    if actual != fields:
        raise Stage2ThresholdDecisionContractError(
            f"{name} fields mismatch; missing={sorted(fields-actual)}, extra={sorted(actual-fields)}"
        )
    return value


def _budget_grid(value: Any, name: str = "budget_grid") -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise Stage2ThresholdDecisionContractError(f"{name} must contain three values")
    result: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{name}[{index}] must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number <= 0.0:
            raise Stage2ThresholdDecisionContractError(f"{name}[{index}] is invalid")
        result.append(number)
    if tuple(result) != PIXEL_BUDGET_GRID:
        raise Stage2ThresholdDecisionContractError(
            f"{name} must equal {list(PIXEL_BUDGET_GRID)}"
        )
    return tuple(result)


def _thresholds(value: Any, *, outcome: str, method_id: str) -> tuple[float, ...] | None:
    if outcome == T5_MISSING_OUTCOME:
        if method_id != "T5" or value is not None:
            raise Stage2ThresholdDecisionContractError(
                "only missing T5 has null thresholds"
            )
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise Stage2ThresholdDecisionContractError("complete thresholds must have length 3")
    result: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"thresholds[{index}] must be numeric")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise Stage2ThresholdDecisionContractError(
                f"thresholds[{index}] must be finite in [0,1]"
            )
        result.append(number)
    if method_id == "T0" and tuple(result) != (0.5, 0.5, 0.5):
        raise Stage2ThresholdDecisionContractError("T0 must be fixed at 0.5")
    if method_id != "T6" and any(
        right < left for left, right in zip(result, result[1:])
    ):
        raise Stage2ThresholdDecisionContractError(
            f"{method_id} thresholds must not decrease as budgets tighten"
        )
    # The frozen 0.001 gap is in logit space.  It is checked by the model and
    # checkpoint verifier; a probability-only decision cannot reconstruct it.
    return tuple(result)


def _assert_no_forbidden_keys(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered not in _ALLOWED_GUARD_KEYS and any(
                part in lowered for part in _FORBIDDEN_KEY_PARTS
            ):
                raise Stage2ThresholdDecisionContractError(
                    f"forbidden prelabel key at {path}.{key}"
                )
            _assert_no_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, f"{path}[{index}]")


def _validate_relative_member(value: Any, name: str) -> str:
    text = _text(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise Stage2ThresholdDecisionContractError(f"{name} must be one local filename")
    return text


def _canonical_root(repository_root: str | Path | None) -> Path:
    raw = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[1]
    if not raw.is_absolute() or raw.is_symlink():
        raise Stage2ThresholdDecisionContractError("repository_root must be absolute and non-symlink")
    resolved = raw.resolve(strict=True)
    if resolved != raw or not resolved.is_dir():
        raise Stage2ThresholdDecisionContractError("repository_root must be canonical")
    return resolved


def _input_file(value: str | Path, root: Path, name: str) -> Path:
    raw = Path(value)
    path = raw if raw.is_absolute() else root / raw
    if ".." in raw.parts or path.is_symlink():
        raise Stage2ThresholdDecisionContractError(f"{name} may not use symlinks or '..'")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2ThresholdDecisionContractError(f"{name} does not exist") from error
    if resolved != path.absolute() or not resolved.is_file() or not resolved.is_relative_to(root):
        raise Stage2ThresholdDecisionContractError(f"{name} is not a canonical repository file")
    current = resolved.parent
    while current != root:
        info = current.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or current.is_symlink():
            raise Stage2ThresholdDecisionContractError(f"{name} has unsafe ancestor")
        current = current.parent
    return resolved


def _verify_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise Stage2ThresholdDecisionContractError(f"missing regular sidecar for {path.name}")
    expected = f"{digest}  {path.name}\n"
    descriptor = os.open(sidecar, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        data = os.read(descriptor, len(expected.encode("ascii")) + 1)
    finally:
        os.close(descriptor)
    if data != expected.encode("ascii"):
        raise Stage2ThresholdDecisionContractError(f"invalid sidecar for {path.name}")


def _validate_shared_bindings(value: Any) -> dict[str, Any]:
    binding = dict(_exact_keys(value, _SHARED_BINDING_FIELDS, "shared_bindings"))
    for key in ("context_package", "context_package_commit"):
        item = _exact_keys(binding[key], _FILE_BINDING_FIELDS, f"shared_bindings.{key}")
        _text(item["path"], f"shared_bindings.{key}.path")
        _sha(item["sha256"], f"shared_bindings.{key}.sha256")
    _text(binding["window_id"], "shared_bindings.window_id")
    for key in (
        "window_identity_sha256",
        "ordered_query_identity_sha256",
        "score_manifest_sha256",
        "score_records_content_sha256",
        "detector_checkpoint_sha256",
        "shared_input_identity_sha256",
    ):
        _sha(binding[key], f"shared_bindings.{key}")
    expected_identity = canonical_json_sha256(
        {
            "context_package_sha256": binding["context_package"]["sha256"],
            "context_package_commit_sha256": binding["context_package_commit"]["sha256"],
            "window_id": binding["window_id"],
            "window_identity_sha256": binding["window_identity_sha256"],
            "ordered_query_identity_sha256": binding["ordered_query_identity_sha256"],
            "score_manifest_sha256": binding["score_manifest_sha256"],
            "score_records_content_sha256": binding["score_records_content_sha256"],
            "detector_checkpoint_sha256": binding["detector_checkpoint_sha256"],
            "budget_grid": list(PIXEL_BUDGET_GRID),
        }
    )
    if binding["shared_input_identity_sha256"] != expected_identity:
        raise Stage2ThresholdDecisionContractError("shared_input_identity_sha256 mismatch")
    return binding


def _content_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    projection = dict(payload)
    projection.pop(digest_field)
    return canonical_json_sha256(projection)


def _validate_decision(payload: Mapping[str, Any], *, expected_method: str) -> None:
    _exact_keys(payload, _DECISION_FIELDS, f"{expected_method} decision")
    if payload["schema_version"] != DECISION_SCHEMA or payload["artifact_type"] != DECISION_ARTIFACT_TYPE:
        raise Stage2ThresholdDecisionContractError("decision schema/artifact type mismatch")
    for field, expected in (
        ("development_only", True),
        ("official_test_accessed", False),
        ("query_labels_or_masks_opened", False),
        ("training_performed", False),
        ("gpu_used", False),
        ("prelabel_eligible", True),
        ("diagnostic_only", False),
        ("reject_supported", False),
        ("fallback_used", False),
    ):
        _exact_bool(payload[field], expected, field)
    method_id = _text(payload["method_id"], "method_id")
    if method_id != expected_method or method_id not in PRELABEL_METHOD_ORDER:
        raise Stage2ThresholdDecisionContractError("decision method mismatch")
    _text(payload["method_name"], "method_name")
    outcome = _text(payload["outcome"], "outcome")
    allowed = {COMPLETE_OUTCOME, T5_MISSING_OUTCOME} if method_id == "T5" else {COMPLETE_OUTCOME}
    if outcome not in allowed or payload["artifact_status"] != outcome:
        raise Stage2ThresholdDecisionContractError("decision outcome/status mismatch")
    _budget_grid(payload["budget_grid"])
    _thresholds(payload["thresholds"], outcome=outcome, method_id=method_id)
    if payload["prediction_semantics"] != STRICT_THRESHOLD_SEMANTICS:
        raise Stage2ThresholdDecisionContractError("strict threshold semantics mismatch")
    _text(payload["outer_fold_id"], "outer_fold_id")
    _text(payload["outer_target_domain"], "outer_target_domain")
    _exact_int(payload["base_seed"], "base_seed")
    _exact_int(payload["derived_seed"], "derived_seed", minimum=1)
    _validate_shared_bindings(payload["shared_bindings"])
    binding_required = method_id in {"T1", "T2", "T3", "T5", "T6", "T7", "T8"}
    if binding_required != (payload["method_binding"] is not None):
        raise Stage2ThresholdDecisionContractError(
            f"{method_id} method_binding presence violates the frozen contract"
        )
    if payload["method_binding"] is not None:
        item = _exact_keys(payload["method_binding"], _FILE_BINDING_FIELDS, "method_binding")
        _text(item["path"], "method_binding.path")
        _sha(item["sha256"], "method_binding.sha256")
    _sha(payload["method_contract_sha256"], "method_contract_sha256")
    if payload["decision_content_sha256_algorithm"] != "sha256-canonical-json-stage2-threshold-decision-v1":
        raise Stage2ThresholdDecisionContractError("decision content algorithm mismatch")
    digest = _sha(payload["decision_content_sha256"], "decision_content_sha256")
    if digest != _content_digest(payload, "decision_content_sha256"):
        raise Stage2ThresholdDecisionContractError("decision content digest mismatch")
    _assert_no_forbidden_keys(payload)


def verify_stage2_threshold_decision_set(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_context_package_sha256: str | None = None,
    expected_context_commit_sha256: str | None = None,
    expected_window_id: str | None = None,
    expected_outer_fold_id: str | None = None,
    expected_base_seed: int | None = None,
    expected_derived_seed: int | None = None,
    expected_detector_checkpoint_sha256: str | None = None,
    expected_budget_grid: Sequence[float] = PIXEL_BUDGET_GRID,
    repository_root: str | Path | None = None,
    _owned_publication_lock: tuple[str | Path, tuple[int, int]] | None = None,
) -> VerifiedStage2ThresholdDecisionSet:
    """Verify one complete T0--T8 set without touching any label-bearing file."""

    root = _canonical_root(repository_root)
    set_path = _input_file(path, root, "decision set")
    bundle_root = set_path.parent
    _verify_bundle_publication_lock(bundle_root, _owned_publication_lock)
    payload, digest = _strict_json(set_path)
    if digest != _sha(expected_sha256, "expected decision-set SHA-256"):
        raise Stage2ThresholdDecisionContractError("decision-set external SHA-256 mismatch")
    _verify_sidecar(set_path, digest)
    _exact_keys(payload, _SET_FIELDS, "decision set")
    if payload["schema_version"] != DECISION_SET_SCHEMA or payload["artifact_type"] != DECISION_SET_ARTIFACT_TYPE:
        raise Stage2ThresholdDecisionContractError("decision-set schema/artifact type mismatch")
    if payload["artifact_status"] != "SEALED_COMPLETE_T0_T8":
        raise Stage2ThresholdDecisionContractError("decision set is not complete")
    for field, expected in (
        ("development_only", True),
        ("official_test_accessed", False),
        ("query_labels_or_masks_opened", False),
        ("training_performed", False),
        ("gpu_used", False),
    ):
        _exact_bool(payload[field], expected, field)
    if payload["method_order"] != list(PRELABEL_METHOD_ORDER):
        raise Stage2ThresholdDecisionContractError("decision-set method order must be T0--T8")
    if "T9" in payload["method_order"]:
        raise Stage2ThresholdDecisionContractError("T9 is forbidden in a prelabel decision set")
    budget_grid = _budget_grid(payload["budget_grid"])
    if tuple(float(value) for value in expected_budget_grid) != budget_grid:
        raise Stage2ThresholdDecisionContractError("caller budget grid mismatch")
    outer_fold = _text(payload["outer_fold_id"], "outer_fold_id")
    outer_target = _text(payload["outer_target_domain"], "outer_target_domain")
    base_seed = _exact_int(payload["base_seed"], "base_seed")
    derived_seed = _exact_int(payload["derived_seed"], "derived_seed", minimum=1)
    shared = _validate_shared_bindings(payload["shared_bindings"])
    if expected_context_package_sha256 is not None and shared["context_package"]["sha256"] != _sha(expected_context_package_sha256, "expected context package SHA-256"):
        raise Stage2ThresholdDecisionContractError("context package binding mismatch")
    if expected_context_commit_sha256 is not None and shared["context_package_commit"]["sha256"] != _sha(expected_context_commit_sha256, "expected context commit SHA-256"):
        raise Stage2ThresholdDecisionContractError("context commit binding mismatch")
    if expected_window_id is not None and shared["window_id"] != _text(expected_window_id, "expected window_id"):
        raise Stage2ThresholdDecisionContractError("window_id binding mismatch")
    if expected_outer_fold_id is not None and outer_fold != expected_outer_fold_id:
        raise Stage2ThresholdDecisionContractError("outer-fold binding mismatch")
    if expected_base_seed is not None and base_seed != expected_base_seed:
        raise Stage2ThresholdDecisionContractError("base-seed binding mismatch")
    if expected_derived_seed is not None and derived_seed != expected_derived_seed:
        raise Stage2ThresholdDecisionContractError("derived-seed binding mismatch")
    if expected_detector_checkpoint_sha256 is not None and shared["detector_checkpoint_sha256"] != _sha(expected_detector_checkpoint_sha256, "expected detector checkpoint SHA-256"):
        raise Stage2ThresholdDecisionContractError("detector checkpoint binding mismatch")

    raw_members = payload["decisions"]
    if not isinstance(raw_members, list) or len(raw_members) != 9:
        raise Stage2ThresholdDecisionContractError("decision set must contain nine members")
    expected_inventory = {
        "decision-set.json",
        "decision-set.json.sha256",
        "COMMIT.json",
        "COMMIT.json.sha256",
    }
    for method_id in PRELABEL_METHOD_ORDER:
        expected_inventory.add(f"{method_id}.decision.json")
        expected_inventory.add(f"{method_id}.decision.json.sha256")
    actual_inventory = {member.name for member in bundle_root.iterdir()}
    if actual_inventory != expected_inventory:
        raise Stage2ThresholdDecisionContractError(
            "decision bundle inventory mismatch; "
            f"missing={sorted(expected_inventory-actual_inventory)}, "
            f"extra={sorted(actual_inventory-expected_inventory)}"
        )
    decisions: list[VerifiedStage2ThresholdDecision] = []
    commit_projection: list[dict[str, str]] = []
    content_projection: list[dict[str, str]] = []
    for index, method_id in enumerate(PRELABEL_METHOD_ORDER):
        member = _exact_keys(raw_members[index], _SET_DECISION_FIELDS, f"decisions[{index}]")
        if member["method_id"] != method_id:
            raise Stage2ThresholdDecisionContractError("decision member order mismatch")
        filename = _validate_relative_member(member["path"], f"decisions[{index}].path")
        member_path = _input_file(bundle_root / filename, root, f"{method_id} decision")
        member_payload, member_digest = _strict_json(member_path)
        if member_digest != _sha(member["sha256"], f"{method_id}.sha256"):
            raise Stage2ThresholdDecisionContractError(f"{method_id} file SHA-256 mismatch")
        _verify_sidecar(member_path, member_digest)
        _validate_decision(member_payload, expected_method=method_id)
        if member["outcome"] != member_payload["outcome"] or member["decision_content_sha256"] != member_payload["decision_content_sha256"]:
            raise Stage2ThresholdDecisionContractError(f"{method_id} set projection mismatch")
        for field in ("outer_fold_id", "outer_target_domain", "base_seed", "derived_seed", "shared_bindings", "budget_grid"):
            if member_payload[field] != payload[field]:
                raise Stage2ThresholdDecisionContractError(f"{method_id} shared field {field} mismatch")
        decisions.append(
            VerifiedStage2ThresholdDecision(
                path=member_path,
                payload=member_payload,
                manifest_sha256=member_digest,
                _token=_TOKEN,
            )
        )
        commit_projection.append({"method_id": method_id, "path": filename, "sha256": member_digest})
        content_projection.append(dict(member))

    if payload["decision_set_content_sha256_algorithm"] != "sha256-canonical-json-ordered-t0-t8-decision-members-v1":
        raise Stage2ThresholdDecisionContractError("decision-set content algorithm mismatch")
    expected_content = canonical_json_sha256(content_projection)
    if payload["decision_set_content_sha256"] != expected_content:
        raise Stage2ThresholdDecisionContractError("decision-set member content digest mismatch")

    commit_path = _input_file(bundle_root / "COMMIT.json", root, "decision-set commit")
    commit, commit_digest = _strict_json(commit_path)
    _verify_sidecar(commit_path, commit_digest)
    _exact_keys(commit, _COMMIT_FIELDS, "decision-set commit")
    if commit["schema_version"] != DECISION_SET_COMMIT_SCHEMA or commit["artifact_type"] != DECISION_SET_COMMIT_ARTIFACT_TYPE or commit["artifact_status"] != "COMMITTED_COMPLETE_T0_T8":
        raise Stage2ThresholdDecisionContractError("invalid decision-set commit identity")
    _exact_bool(commit["development_only"], True, "commit.development_only")
    _exact_bool(commit["official_test_accessed"], False, "commit.official_test_accessed")
    if commit["decision_set_file"] != set_path.name or commit["decision_set_sha256"] != digest:
        raise Stage2ThresholdDecisionContractError("commit decision-set binding mismatch")
    if not isinstance(commit["decision_files"], list) or len(commit["decision_files"]) != 9:
        raise Stage2ThresholdDecisionContractError("commit decision inventory incomplete")
    normalized_commit: list[dict[str, str]] = []
    for index, raw in enumerate(commit["decision_files"]):
        item = dict(_exact_keys(raw, _COMMIT_MEMBER_FIELDS, f"commit.decision_files[{index}]"))
        _sha(item["sha256"], f"commit.decision_files[{index}].sha256")
        _validate_relative_member(item["path"], f"commit.decision_files[{index}].path")
        normalized_commit.append(item)
    if normalized_commit != commit_projection:
        raise Stage2ThresholdDecisionContractError("commit inventory does not match decisions")
    if commit["inventory_sha256_algorithm"] != "sha256-canonical-json-stage2-decision-bundle-inventory-v1" or commit["inventory_sha256"] != canonical_json_sha256({"decision_set_file": set_path.name, "decision_set_sha256": digest, "decision_files": commit_projection}):
        raise Stage2ThresholdDecisionContractError("commit inventory digest mismatch")
    _assert_no_forbidden_keys(payload)
    return VerifiedStage2ThresholdDecisionSet(
        path=set_path,
        payload=payload,
        manifest_sha256=digest,
        decisions=decisions,
        _token=_TOKEN,
    )


def _verify_bundle_publication_lock(
    bundle_root: Path,
    owned_lock: tuple[str | Path, tuple[int, int]] | None,
) -> None:
    """Reject an in-flight bundle unless its producer proves lock ownership."""

    lock = bundle_root.parent / f".{bundle_root.name}.lock"
    if not os.path.lexists(lock):
        return
    if owned_lock is None:
        raise Stage2ThresholdDecisionContractError(
            "decision bundle publication lock exists"
        )
    allowed_path = Path(owned_lock[0]).expanduser().absolute()
    if allowed_path != lock.absolute():
        raise Stage2ThresholdDecisionContractError(
            "decision bundle publication lock authorization path mismatch"
        )
    info = lock.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(info.st_mode)
        or (info.st_dev, info.st_ino) != owned_lock[1]
    ):
        raise Stage2ThresholdDecisionContractError(
            "decision bundle publication lock authorization inode mismatch"
        )


__all__ = [
    "COMPLETE_OUTCOME",
    "DECISION_SCHEMA",
    "DECISION_SET_SCHEMA",
    "PIXEL_BUDGET_GRID",
    "PRELABEL_METHOD_ORDER",
    "STRICT_THRESHOLD_SEMANTICS",
    "Stage2ThresholdDecisionContractError",
    "T5_MISSING_OUTCOME",
    "VerifiedStage2ThresholdDecision",
    "VerifiedStage2ThresholdDecisionSet",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "verify_stage2_threshold_decision_set",
]
