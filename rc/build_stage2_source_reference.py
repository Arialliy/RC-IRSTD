"""Build checkpoint-specific, query-disjoint Stage2 source references.

This module is additive.  It accepts only the two Stage2 score-manifest-v4
reference roles and deliberately does not broaden the legacy source-reference
builder or verifier.  Every input is externally SHA-256 bound; official-test,
diagnostic and OOF-holdout scores can never become reference records.

The output is a no-replace four-file bundle: one NPZ, one audit JSON and an
adjacent SHA-256 sidecar for each.  Files are fully built and revalidated in a
same-parent private staging directory before transactional publication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image

from data_ext.stage2_score_manifest import (
    FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    OOF_TRAIN_SOURCE_REFERENCE,
    STAGE2_DOMAINS,
    VerifiedStage2ScoreManifest,
    verify_stage2_score_manifest,
)
from .domain_statistics import (
    BASE_FEATURE_DIM,
    extract_unlabeled_statistics,
    load_source_reference,
)
from .schema import SourceReference, StatisticsConfig


STAGE2_SOURCE_REFERENCE_SCHEMA = "rc-irstd.stage2-source-reference.v1"
STAGE2_SOURCE_REFERENCE_AUDIT_SCHEMA = (
    "rc-irstd.stage2-source-reference-audit.v1"
)
STAGE2_SOURCE_REFERENCE_ARTIFACT_TYPE = (
    "rc_irstd_stage2_checkpoint_specific_source_reference"
)
STAGE2_SOURCE_REFERENCE_AUDIT_ARTIFACT_TYPE = (
    "rc_irstd_stage2_checkpoint_specific_source_reference_audit"
)

IDENTITY_BOUNDARY_FIELDS = (
    "canonical_id",
    "original_image_sha256",
    "near_duplicate_cluster_id_or_unique_sentinel",
    "exclusion_group_id",
)

_POLICY_BINDINGS = {
    "b2_authorization": {
        "path": (
            "outputs/stage2_protocol/"
            "RC4_STAGE2_B2_EPISODE_REFERENCE_AUTHORIZATION_AMENDMENT_20260717.json"
        ),
        "sha256": (
            "cc15832de4f85abfae84c4d49a5ac098cff253d0fecfa885d0d7735d3ef5aea6"
        ),
    },
    "b1_integration_pass": {
        "path": (
            "outputs/stage2_protocol/"
            "RC4_STAGE2_B1_CONTRACT_SPINE_INTEGRATION_PASS_20260716.json"
        ),
        "sha256": (
            "4d4ce52653e872ffa4f3f71b9475edb03b934b27d0cb4d6c914d63c92b0131d6"
        ),
    },
    "semantic_amendment": {
        "path": (
            "outputs/stage2_protocol/"
            "RC4_STAGE2_PRE_G1_RESULT_FREE_ANALYSIS_PLAN_AMENDMENT_"
            "SEMANTICS_V1_20260716.json"
        ),
        "sha256": (
            "c60e087116f98a3e59772792e16be389cc2961180b7a9c5de930e2b9cd9abef7"
        ),
    },
}

_WINDOW_TOP_FIELDS = frozenset(
    {
        "artifact_status",
        "artifact_type",
        "bound_inputs",
        "complete_window_count",
        "domain",
        "episode_role",
        "execution_authorized",
        "geometry",
        "guardrails",
        "observed_results",
        "oof_fold_index",
        "ordered_role_record_count",
        "outer_fold_id",
        "outer_target_domain",
        "role_binding",
        "role_purity",
        "schema_version",
        "source_role",
        "unused_suffix",
        "window_record_count",
        "windows",
    }
)
_WINDOW_FIELDS = frozenset(
    {"window_index", "window_id", "context_records", "query_records"}
)
_WINDOW_RECORD_FIELDS = frozenset(
    {
        "canonical_id",
        "episode_role",
        "exclusion_group_id",
        "image_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "original_image_path",
        "original_image_sha256",
        "outer_fold_id",
        "source_role",
        "source_role_record_index",
    }
)
_OOF_WINDOW_RECORD_FIELDS = _WINDOW_RECORD_FIELDS | frozenset({"oof_fold_index"})
_WINDOW_GUARD_FIELDS = frozenset(
    {
        "development_only",
        "execution_authorized",
        "mask_or_label_files_opened",
        "official_test_ids_materialized",
        "official_test_images_opened",
        "official_test_split_files_opened",
        "original_training_images_opened_only_for_sha256",
        "predictions_scores_checkpoints_or_metrics_opened",
        "result_free",
    }
)
_WINDOW_BOUND_INPUT_NAMES = frozenset(
    {
        "image_only_near_duplicate_audit",
        "k2_geometry_prefreeze_audit",
        "official_train_derived_split_manifest",
    }
)
_BINDING_FIELDS = frozenset({"path", "sha256"})
_STATISTICS_CONFIG_FIELDS = frozenset(
    {
        "peak_kernel_size",
        "peak_min_score",
        "probability_histogram_bins",
        "peak_histogram_bins",
        "quantiles",
        "plateau_mode",
        "plateau_atol",
        "grayscale_normalization",
        "quantile_sample_limit",
        "quantile_estimator",
        "algorithm_version",
    }
)
_NPZ_FIELD_ORDER = (
    "schema_version",
    "domains",
    "centers",
    "scale",
    "statistics_config_json",
    "source_contract_json",
    "stage2_contract_json",
)
_NPZ_MEMBER_ORDER = tuple(f"{field}.npy" for field in _NPZ_FIELD_ORDER)
_SCALE_FLOOR = 1e-8
_NO_PENDING_GRAY = object()

_STAGE2_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "development_only",
        "execution_authorized",
        "official_test_accessed",
        "labels_or_masks_opened",
        "reference_role",
        "detector_identity",
        "source_domains",
        "bindings",
        "identity_boundary_audit",
    }
)
_DETECTOR_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "derived_seed",
        "detector_role",
        "oof_fold_index",
        "checkpoint_sha256",
    }
)
_STAGE2_BINDING_FIELDS = frozenset(
    {
        "policy",
        "source_score_manifests",
        "run_contract",
        "checkpoint",
        "statistics_config",
        "consumer_window_manifests",
    }
)
_SOURCE_SCORE_BINDING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "source_domain",
        "records_content_sha256",
        "record_count",
        "selection_contract",
    }
)
_CONSUMER_BINDING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "domain",
        "episode_role",
        "complete_window_count",
        "record_count",
    }
)
_SOURCE_CONTRACT_FIELDS = frozenset(
    {
        "detector_checkpoint_sha",
        "detector_source_domains",
        "outer_fold_id",
        "outer_target",
        "held_out_domains",
        "protocol_scope",
    }
)


@dataclass(frozen=True, init=False)
class VerifiedStage2SourceReference:
    """A publicly consumable, fully closed Stage2 source-reference bundle."""

    path: Path
    npz_sha256: str
    audit_path: Path
    audit_sha256: str
    domains: tuple[str, ...]
    centers: tuple[tuple[float, ...], ...]
    scale: tuple[float, ...]
    source_reference: SourceReference
    statistics_config: StatisticsConfig
    stage2_contract: Mapping[str, Any]
    source_contract: Mapping[str, Any]
    detector_identity: Mapping[str, Any]
    checkpoint_binding: Mapping[str, Any]
    reference_role: str
    consumer_bindings: tuple[Mapping[str, Any], ...]
    audit: Mapping[str, Any]

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "VerifiedStage2SourceReference can only be created by "
            "verify_stage2_source_reference"
        )

    @property
    def npz_sha(self) -> str:
        return self.npz_sha256

    @property
    def audit_sha(self) -> str:
        return self.audit_sha256

    @property
    def detector(self) -> Mapping[str, Any]:
        return self.detector_identity

    @property
    def checkpoint(self) -> Mapping[str, Any]:
        return self.checkpoint_binding


def _make_verified_stage2_source_reference(
    **values: Any,
) -> VerifiedStage2SourceReference:
    expected = frozenset(VerifiedStage2SourceReference.__dataclass_fields__)
    if frozenset(values) != expected:
        raise RuntimeError("internal verified source-reference field mismatch")
    result = object.__new__(VerifiedStage2SourceReference)
    for name in expected:
        object.__setattr__(result, name, values[name])
    return result


class _LockstepPairs:
    """Expose one pair stream as probability and gray iterators without tee."""

    def __init__(self, pairs: Iterator[tuple[np.ndarray, np.ndarray]]) -> None:
        self._pairs = pairs
        self._pending: object | np.ndarray = _NO_PENDING_GRAY
        self._finished = False

    def probabilities(self) -> Iterator[np.ndarray]:
        while True:
            if self._pending is not _NO_PENDING_GRAY:
                raise RuntimeError("probability/grayscale streams are not in lockstep")
            try:
                probability, gray = next(self._pairs)
            except StopIteration:
                self._finished = True
                return
            self._pending = gray
            yield probability
            if self._pending is not _NO_PENDING_GRAY:
                raise RuntimeError("probability/grayscale streams are not in lockstep")

    def grayscale(self) -> Iterator[np.ndarray]:
        while True:
            if self._pending is _NO_PENDING_GRAY:
                if self._finished:
                    return
                raise RuntimeError("grayscale advanced before its probability partner")
            gray = self._pending
            self._pending = _NO_PENDING_GRAY
            assert isinstance(gray, np.ndarray)
            yield gray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = os.stat(path, follow_symlinks=False)
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _sha256_value(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"{name} must be a lowercase 64-character SHA-256")
    if value != value.lower() or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    return value


def _exact_bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise TypeError(
            f"{name} must be the exact JSON boolean {str(expected).lower()}"
        )


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact JSON integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty, whitespace-trimmed string")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _relative_path(value: object, name: str) -> str:
    rendered = _string(value, name)
    if "\\" in rendered:
        raise ValueError(f"{name} must use POSIX separators")
    path = PurePosixPath(rendered)
    if path.is_absolute() or rendered.startswith("~"):
        raise ValueError(f"{name} must be repository-relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} contains traversal/non-canonical components")
    if path.as_posix() != rendered:
        raise ValueError(f"{name} must be a canonical POSIX path")
    for raw_part in path.parts:
        part = raw_part.lower()
        stem = PurePosixPath(part).stem
        if (
            part in {"test", "official_test", "official-test"}
            or stem.startswith("test_")
            or stem.startswith("official_test")
        ):
            raise ValueError(f"{name} names an official-test-like path")
    return rendered


def _root(value: str | Path | None) -> Path:
    candidate = Path(__file__).resolve().parents[1] if value is None else Path(value)
    candidate = candidate.expanduser()
    if candidate.is_symlink():
        raise ValueError("repository_root must not be a symlink")
    result = candidate.resolve(strict=True)
    if not result.is_dir():
        raise FileNotFoundError(f"repository_root is not a directory: {result}")
    return result


def _reject_symlink_components(candidate: Path, root: Path, name: str) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository_root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} contains a symlinked path component")


def _existing_file(value: str | Path, root: Path, name: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} must be inside repository_root") from error
    _reject_symlink_components(candidate, root, name)
    result = candidate.resolve(strict=True)
    try:
        result.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes repository_root") from error
    if not result.is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {result}")
    _relative_path(result.relative_to(root).as_posix(), name)
    return result


def _resolve_binding(root: Path, binding: Mapping[str, Any], name: str) -> Path:
    _exact_keys(binding, _BINDING_FIELDS, name)
    relative = _relative_path(binding["path"], f"{name}.path")
    expected = _sha256_value(binding["sha256"], f"{name}.sha256")
    path = _existing_file(relative, root, name)
    _verify_file_hash(path, expected, name)
    return path


def _verify_file_hash(path: Path, expected: str, name: str) -> None:
    identity = _file_identity(path)
    before = _sha256_file(path)
    if before != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    if _sha256_file(path) != before:
        raise RuntimeError(f"{name} changed while hash-verified")
    if _file_identity(path) != identity:
        raise RuntimeError(f"{name} file identity changed while hash-verified")


def _read_json_stable(path: Path, expected: str, name: str) -> Mapping[str, Any]:
    identity = _file_identity(path)
    before = _sha256_file(path)
    if before != expected:
        raise ValueError(f"{name} SHA-256 mismatch")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if _sha256_file(path) != before:
        raise RuntimeError(f"{name} changed while read")
    if _file_identity(path) != identity:
        raise RuntimeError(f"{name} file identity changed while read")
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _read_text_stable(
    path: Path,
    *,
    encoding: str,
    name: str,
) -> str:
    identity = _file_identity(path)
    before = _sha256_file(path)
    text = path.read_text(encoding=encoding)
    if _sha256_file(path) != before or _file_identity(path) != identity:
        raise RuntimeError(f"{name} changed while read")
    return text


def _binding(path: Path, root: Path, digest: str | None = None) -> dict[str, str]:
    relative = path.relative_to(root).as_posix()
    _relative_path(relative, "artifact path")
    return {"path": relative, "sha256": digest or _sha256_file(path)}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _policy_contract(root: Path) -> tuple[dict[str, dict[str, str]], list[tuple[Path, str]]]:
    bindings: dict[str, dict[str, str]] = {}
    rechecks: list[tuple[Path, str]] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for name, raw in _POLICY_BINDINGS.items():
        path = _existing_file(raw["path"], root, name)
        payloads[name] = _read_json_stable(path, raw["sha256"], name)
        bindings[name] = dict(raw)
        rechecks.append((path, raw["sha256"]))

    b2 = payloads["b2_authorization"]
    if b2.get("schema_version") != "rc-irstd.aaai27-stage2-b2-authorization-amendment.v1":
        raise ValueError("B2 authorization schema mismatch")
    _exact_bool(b2.get("contains_observed_results"), "B2 contains_observed_results", False)
    authorization = b2.get("authorization")
    if not isinstance(authorization, Mapping):
        raise TypeError("B2 authorization.authorization must be an object")
    work_items = authorization.get("authorized_work_items")
    if not isinstance(work_items, list) or "W04" not in work_items:
        raise ValueError("B2 authorization does not include W04")
    for field, expected in {
        "result_free_source_implementation_and_synthetic_tests_authorized": True,
        "claim_bearing_training_authorized": False,
        "real_data_execution_authorized": False,
        "official_test_access_authorized": False,
    }.items():
        _exact_bool(authorization.get(field), f"B2 authorization.{field}", expected)
    bound = b2.get("bound_inputs")
    if not isinstance(bound, Mapping):
        raise TypeError("B2 bound_inputs must be an object")
    if bound.get("b1_integration_pass_sha256") != _POLICY_BINDINGS[
        "b1_integration_pass"
    ]["sha256"] or bound.get("semantic_amendment_sha256") != _POLICY_BINDINGS[
        "semantic_amendment"
    ]["sha256"]:
        raise ValueError("B2 authorization has stale B1/semantics bindings")

    b1 = payloads["b1_integration_pass"]
    if b1.get("status") != "PASS":
        raise ValueError("B1 integration gate is not PASS")
    _exact_bool(b1.get("contains_observed_results"), "B1 contains_observed_results", False)
    b1_scope = b1.get("scope")
    if not isinstance(b1_scope, Mapping):
        raise TypeError("B1 scope must be an object")
    for field in (
        "training_performed",
        "gpu_used",
        "real_data_read",
        "official_test_used",
        "claim_bearing_execution_authorized",
    ):
        _exact_bool(b1_scope.get(field), f"B1 scope.{field}", False)

    semantics = payloads["semantic_amendment"]
    for field in (
        "execution_authorized",
        "training_authorized",
        "official_test_access_authorized",
        "contains_observed_results",
    ):
        _exact_bool(semantics.get(field), f"semantics.{field}", False)
    return bindings, rechecks


def _load_statistics_config(
    path: Path, expected_sha256: str
) -> tuple[StatisticsConfig, Mapping[str, Any]]:
    payload = _read_json_stable(path, expected_sha256, "statistics config")
    _exact_keys(payload, _STATISTICS_CONFIG_FIELDS, "statistics config")
    for name in (
        "peak_kernel_size",
        "probability_histogram_bins",
        "peak_histogram_bins",
        "quantile_sample_limit",
    ):
        _exact_int(payload[name], f"statistics config.{name}", minimum=1)
    for name in ("peak_min_score", "plateau_atol"):
        if type(payload[name]) not in {int, float}:
            raise TypeError(f"statistics config.{name} must be a JSON number")
    quantiles = payload["quantiles"]
    if not isinstance(quantiles, list) or any(
        type(value) not in {int, float} for value in quantiles
    ):
        raise TypeError("statistics config.quantiles must be a list of JSON numbers")
    for name in (
        "plateau_mode",
        "grayscale_normalization",
        "quantile_estimator",
        "algorithm_version",
    ):
        _string(payload[name], f"statistics config.{name}")
    config = StatisticsConfig.from_dict(payload)
    if config.to_dict() != dict(payload):
        raise ValueError("statistics config is not canonical StatisticsConfig v3")
    return config, payload


def _manifest_role(path: Path, expected_sha256: str) -> str:
    payload = _read_json_stable(path, expected_sha256, "score manifest preflight")
    role = _string(payload.get("role"), "score manifest role")
    if role not in {
        OOF_TRAIN_SOURCE_REFERENCE,
        FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    }:
        raise ValueError("score manifest role is forbidden for source references")
    return role


def _verify_reference_manifests(
    manifest_paths: Sequence[Path],
    manifest_sha256s: Sequence[str],
    *,
    root: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> tuple[
    tuple[VerifiedStage2ScoreManifest, ...],
    Mapping[str, Any],
    str,
    list[tuple[Path, str]],
]:
    if len(manifest_paths) != 2 or len(manifest_sha256s) != 2:
        raise ValueError("exactly two score manifests and expected SHA-256 values are required")
    if len(set(manifest_paths)) != 2:
        raise ValueError("the two score manifests must be distinct")
    roles = [
        _manifest_role(path, digest)
        for path, digest in zip(manifest_paths, manifest_sha256s)
    ]
    if len(set(roles)) != 1:
        raise ValueError("both score manifests must use the same reference role")
    role = roles[0]
    verified = tuple(
        verify_stage2_score_manifest(
            path,
            digest,
            role,
            repository_root=root,
        )
        for path, digest in zip(manifest_paths, manifest_sha256s)
    )
    first = verified[0]
    identity_fields = (
        "outer_fold_id",
        "outer_target",
        "base_seed",
        "derived_seed",
        "detector_role",
        "oof_fold_index",
    )
    for current in verified[1:]:
        for field in identity_fields:
            if current.payload[field] != first.payload[field]:
                raise ValueError(f"reference score manifests disagree on {field}")
        for binding_name in first.bindings:
            if binding_name == "selection_contract":
                continue
            if current.bindings[binding_name] != first.bindings[binding_name]:
                raise ValueError(
                    f"reference score manifests disagree on {binding_name} binding"
                )

    domains = tuple(item.payload["source_domain"] for item in verified)
    if len(set(domains)) != 2:
        raise ValueError("reference score manifests must represent two unique domains")
    run_binding = first.bindings["run_contract"]
    run_path = _existing_file(run_binding["path"], root, "run contract")
    run_contract = _read_json_stable(
        run_path, run_binding["sha256"], "run contract"
    )
    raw_sources = run_contract.get("source_domains")
    if (
        not isinstance(raw_sources, list)
        or len(raw_sources) != 2
        or len(set(raw_sources)) != 2
        or any(source not in STAGE2_DOMAINS for source in raw_sources)
    ):
        raise ValueError("run contract must contain exactly two frozen source domains")
    if set(domains) != set(raw_sources):
        raise ValueError("reference manifests do not cover both run source domains")
    if run_contract.get("run_id") is None:
        raise ValueError("run contract is missing run_id")
    _string(run_contract["run_id"], "run contract run_id")

    expected_role = (
        OOF_TRAIN_SOURCE_REFERENCE
        if first.payload["detector_role"] == "detector_oof"
        else FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE
    )
    if role != expected_role:
        raise ValueError("reference role does not match detector OOF/full-fit identity")
    checkpoint_binding = first.bindings["checkpoint"]
    if checkpoint_binding["sha256"] != checkpoint_sha256:
        raise ValueError("score manifests disagree with expected checkpoint SHA-256")
    bound_checkpoint = _existing_file(
        checkpoint_binding["path"], root, "score-bound checkpoint"
    )
    if bound_checkpoint != checkpoint_path:
        raise ValueError("explicit checkpoint path differs from score-manifest binding")

    expected_selection_role = (
        "detector_oof_train"
        if role == OOF_TRAIN_SOURCE_REFERENCE
        else "detector_full_fit_train"
    )
    rechecks: list[tuple[Path, str]] = []
    for manifest, manifest_digest in zip(verified, manifest_sha256s):
        rechecks.append((manifest.path, manifest_digest))
        selection_binding = manifest.bindings["selection_contract"]
        selection_path = _existing_file(
            selection_binding["path"], root, "reference selection"
        )
        selection = _read_json_stable(
            selection_path,
            selection_binding["sha256"],
            "reference selection",
        )
        if selection.get("artifact_type") != "rc_irstd_stage2_detector_selection":
            raise ValueError("reference records must come from detector selections")
        if selection.get("selection_role") != expected_selection_role:
            raise ValueError("reference selection is not the required detector-training role")
        rechecks.append((selection_path, selection_binding["sha256"]))
        for item in manifest.items:
            rechecks.append((item.score_path, item.record["score_file_sha256"]))
            rechecks.append((item.image_path, item.record["original_image_sha256"]))
    for binding_name, binding_value in first.bindings.items():
        if binding_name == "selection_contract":
            continue
        path = _existing_file(binding_value["path"], root, binding_name)
        rechecks.append((path, binding_value["sha256"]))
    return verified, run_contract, role, rechecks


def _window_binding(
    raw: object,
    *,
    name: str,
    root: Path,
    materialization_hashes: Mapping[str, Any] | None,
    rechecks: list[tuple[Path, str]],
) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be an object")
    _exact_keys(raw, _BINDING_FIELDS, name)
    relative = _relative_path(raw["path"], f"{name}.path")
    digest = _sha256_value(raw["sha256"], f"{name}.sha256")
    if materialization_hashes is not None and materialization_hashes.get(relative) != digest:
        raise ValueError(f"{name} is not hash-bound by the detector run contract")
    path = _existing_file(relative, root, name)
    _verify_file_hash(path, digest, name)
    rechecks.append((path, digest))
    return {"path": relative, "sha256": digest}


def _verify_window_manifest(
    path: Path,
    expected_sha256: str,
    *,
    root: Path,
    run_contract: Mapping[str, Any],
    detector_role: str,
    oof_fold_index: int | None,
    rechecks: list[tuple[Path, str]],
) -> dict[str, Any]:
    payload = _read_json_stable(path, expected_sha256, "consumer window manifest")
    _exact_keys(payload, _WINDOW_TOP_FIELDS, "consumer window manifest")
    exact_values = {
        "schema_version": "rc-irstd.stage2-role-pure-c14q28-windows.v1",
        "artifact_type": "rc_irstd_stage2_role_pure_episode_windows",
        "artifact_status": "DEVELOPMENT_ONLY_RESULT_FREE",
        "outer_fold_id": run_contract["outer_fold_id"],
        "outer_target_domain": run_contract["outer_target_domain"],
    }
    for name, expected in exact_values.items():
        if payload[name] != expected:
            raise ValueError(f"consumer window {name} must equal {expected!r}")
    _exact_bool(payload["execution_authorized"], "window execution_authorized", False)
    if payload["observed_results"] is not None:
        raise ValueError("consumer window observed_results must be null")

    guards = payload["guardrails"]
    if not isinstance(guards, Mapping):
        raise TypeError("consumer window guardrails must be an object")
    _exact_keys(guards, _WINDOW_GUARD_FIELDS, "consumer window guardrails")
    expected_guards = {
        "development_only": True,
        "execution_authorized": False,
        "mask_or_label_files_opened": False,
        "official_test_ids_materialized": False,
        "official_test_images_opened": False,
        "official_test_split_files_opened": False,
        "original_training_images_opened_only_for_sha256": True,
        "predictions_scores_checkpoints_or_metrics_opened": False,
        "result_free": True,
    }
    for name, expected in expected_guards.items():
        _exact_bool(guards[name], f"consumer window guardrails.{name}", expected)

    geometry = payload["geometry"]
    if not isinstance(geometry, Mapping) or dict(geometry) != {
        "block_size": 42,
        "construction": (
            "ordered_non_overlapping_contiguous_blocks_context_first_query_second"
        ),
        "context_size": 14,
        "query_size": 28,
    }:
        raise ValueError("consumer window geometry must be exact C14/Q28")

    run_sources = tuple(run_contract["source_domains"])
    domain = _string(payload["domain"], "consumer window domain")
    if detector_role == "detector_oof":
        if payload["episode_role"] != "stage2_oof_fit":
            raise ValueError("OOF reference consumers must be stage2_oof_fit")
        if payload["source_role"] != "detector_fit" or domain not in run_sources:
            raise ValueError("OOF consumer must be a source detector_fit window")
        observed_oof = _exact_int(
            payload["oof_fold_index"], "consumer window oof_fold_index"
        )
        if observed_oof != oof_fold_index:
            raise ValueError("OOF consumer fold differs from the reference detector")
        record_fields = _OOF_WINDOW_RECORD_FIELDS
    else:
        if payload["oof_fold_index"] is not None:
            raise ValueError("full-fit consumer oof_fold_index must be null")
        if payload["source_role"] != "detector_diagnostic":
            raise ValueError("full-fit consumers must use detector_diagnostic records")
        expected_episode = (
            "source_diagnostic_validation"
            if domain in run_sources
            else "outer_target_diagnostic_development"
        )
        if payload["episode_role"] != expected_episode:
            raise ValueError("full-fit consumer domain/episode role mismatch")
        if expected_episode == "outer_target_diagnostic_development" and domain != run_contract[
            "outer_target_domain"
        ]:
            raise ValueError("outer-target consumer domain mismatch")
        record_fields = _WINDOW_RECORD_FIELDS

    purity = payload["role_purity"]
    expected_purity = {
        "allowed_source_role": payload["source_role"],
        "mixed_roles_allowed": False,
        "single_oof_checkpoint_identity_per_fit_window_required_at_future_score_binding": (
            detector_role == "detector_oof"
        ),
        "single_source_domain_per_window": True,
    }
    if not isinstance(purity, Mapping) or set(purity) != set(expected_purity):
        raise ValueError("consumer window role_purity fields mismatch")
    for name, expected in expected_purity.items():
        if type(expected) is bool:
            _exact_bool(purity[name], f"consumer role_purity.{name}", expected)
        elif purity[name] != expected:
            raise ValueError(f"consumer role_purity.{name} mismatch")

    materialization = run_contract.get("bindings", {}).get(
        "materialization_artifacts_sha256"
    )
    if not isinstance(materialization, Mapping):
        raise TypeError("run contract materialization_artifacts_sha256 is missing")
    if materialization.get(path.relative_to(root).as_posix()) != expected_sha256:
        raise ValueError("consumer window is not bound by the detector run contract")
    _window_binding(
        payload["role_binding"],
        name="consumer role_binding",
        root=root,
        materialization_hashes=materialization,
        rechecks=rechecks,
    )
    _window_binding(
        payload["unused_suffix"],
        name="consumer unused_suffix",
        root=root,
        materialization_hashes=materialization,
        rechecks=rechecks,
    )
    bound_inputs = payload["bound_inputs"]
    if not isinstance(bound_inputs, Mapping):
        raise TypeError("consumer window bound_inputs must be an object")
    _exact_keys(bound_inputs, _WINDOW_BOUND_INPUT_NAMES, "consumer bound_inputs")
    for name in sorted(_WINDOW_BOUND_INPUT_NAMES):
        _window_binding(
            bound_inputs[name],
            name=f"consumer bound_inputs.{name}",
            root=root,
            materialization_hashes=None,
            rechecks=rechecks,
        )

    windows = payload["windows"]
    complete_count = _exact_int(
        payload["complete_window_count"], "complete_window_count", minimum=1
    )
    if not isinstance(windows, list) or len(windows) != complete_count:
        raise ValueError("complete_window_count must equal non-empty windows")
    expected_record_count = complete_count * 42
    if _exact_int(payload["window_record_count"], "window_record_count", minimum=1) != expected_record_count:
        raise ValueError("window_record_count must equal complete_window_count*42")
    if _exact_int(payload["ordered_role_record_count"], "ordered_role_record_count", minimum=1) < expected_record_count:
        raise ValueError("ordered_role_record_count is smaller than window records")

    all_records: list[Mapping[str, Any]] = []
    window_ids: set[str] = set()
    for window_index, raw_window in enumerate(windows):
        if not isinstance(raw_window, Mapping):
            raise TypeError(f"windows[{window_index}] must be an object")
        _exact_keys(raw_window, _WINDOW_FIELDS, f"windows[{window_index}]")
        if _exact_int(raw_window["window_index"], f"windows[{window_index}].window_index") != window_index:
            raise ValueError("window_index must equal exact manifest order")
        window_id = _string(raw_window["window_id"], f"windows[{window_index}].window_id")
        if window_id in window_ids:
            raise ValueError("duplicate consumer window_id")
        window_ids.add(window_id)
        context = raw_window["context_records"]
        query = raw_window["query_records"]
        if not isinstance(context, list) or len(context) != 14:
            raise ValueError("every consumer context must contain exactly 14 records")
        if not isinstance(query, list) or len(query) != 28:
            raise ValueError("every consumer query must contain exactly 28 records")
        for partition_name, records in (("context", context), ("query", query)):
            for local_index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    raise TypeError(
                        f"windows[{window_index}].{partition_name}[{local_index}] "
                        "must be an object"
                    )
                _exact_keys(
                    record,
                    record_fields,
                    f"windows[{window_index}].{partition_name}[{local_index}]",
                )
                if record["episode_role"] != partition_name:
                    raise ValueError("consumer record episode_role mismatch")
                if record["outer_fold_id"] != run_contract["outer_fold_id"]:
                    raise ValueError("consumer record outer_fold_id mismatch")
                if record["source_role"] != payload["source_role"]:
                    raise ValueError("consumer record source_role mismatch")
                if record_fields is _OOF_WINDOW_RECORD_FIELDS:
                    record_oof = _exact_int(
                        record["oof_fold_index"],
                        "consumer record oof_fold_index",
                    )
                    if record_oof != oof_fold_index:
                        raise ValueError("consumer record oof_fold_index mismatch")
                canonical = _string(record["canonical_id"], "consumer canonical_id")
                image_id = _string(record["image_id"], "consumer image_id")
                if canonical != f"{domain}::{image_id}":
                    raise ValueError("consumer canonical_id/domain/image_id mismatch")
                _sha256_value(record["original_image_sha256"], "consumer image SHA")
                _relative_path(record["original_image_path"], "consumer image path")
                _string(record["near_duplicate_cluster_id_or_unique_sentinel"], "consumer near-duplicate boundary")
                _string(record["exclusion_group_id"], "consumer exclusion_group_id")
                _exact_int(record["source_role_record_index"], "consumer source_role_record_index")
                all_records.append(record)
        for field in IDENTITY_BOUNDARY_FIELDS:
            if set(item[field] for item in context).intersection(
                item[field] for item in query
            ):
                raise ValueError(f"consumer context/query overlap under {field}")

    source_indices = [int(record["source_role_record_index"]) for record in all_records]
    if source_indices != sorted(source_indices) or len(set(source_indices)) != len(source_indices):
        raise ValueError("consumer records must preserve unique source-role order")
    for field in IDENTITY_BOUNDARY_FIELDS:
        if len({record[field] for record in all_records}) != len(all_records):
            raise ValueError(f"consumer records duplicate the {field} boundary")
    rechecks.append((path, expected_sha256))
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": expected_sha256,
        "domain": domain,
        "episode_role": payload["episode_role"],
        "complete_window_count": complete_count,
        "record_count": len(all_records),
        "records": tuple(all_records),
    }


def _verify_all_consumers(
    paths: Sequence[Path],
    expected_sha256s: Sequence[str],
    *,
    root: Path,
    run_contract: Mapping[str, Any],
    detector_role: str,
    oof_fold_index: int | None,
    rechecks: list[tuple[Path, str]],
) -> tuple[dict[str, Any], ...]:
    expected_count = 2 if detector_role == "detector_oof" else 3
    if len(paths) != expected_count or len(expected_sha256s) != expected_count:
        raise ValueError(
            f"{detector_role} reference requires exactly {expected_count} consumer "
            "window manifests and expected SHA-256 values"
        )
    if len(set(paths)) != expected_count:
        raise ValueError("consumer window manifest paths must be unique")
    consumers = tuple(
        _verify_window_manifest(
            path,
            digest,
            root=root,
            run_contract=run_contract,
            detector_role=detector_role,
            oof_fold_index=oof_fold_index,
            rechecks=rechecks,
        )
        for path, digest in zip(paths, expected_sha256s)
    )
    source_domains = set(run_contract["source_domains"])
    observed = {(item["domain"], item["episode_role"]) for item in consumers}
    if detector_role == "detector_oof":
        expected = {(domain, "stage2_oof_fit") for domain in source_domains}
    else:
        expected = {
            *( (domain, "source_diagnostic_validation") for domain in source_domains ),
            (
                run_contract["outer_target_domain"],
                "outer_target_diagnostic_development",
            ),
        }
    if observed != expected or len(observed) != len(consumers):
        raise ValueError("consumer windows do not form the complete detector consumer set")
    return consumers


def _identity_digest(values: Sequence[str]) -> str:
    encoded = _canonical_json(sorted(values)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_audit(
    manifests: Sequence[VerifiedStage2ScoreManifest],
    consumers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_records = [record for manifest in manifests for record in manifest.records]
    reference_sets = {
        field: {str(record[field]) for record in reference_records}
        for field in IDENTITY_BOUNDARY_FIELDS
    }
    per_consumer: list[dict[str, Any]] = []
    all_consumer_records: list[Mapping[str, Any]] = []
    for consumer in consumers:
        records = list(consumer["records"])
        all_consumer_records.extend(records)
        overlaps = {
            field: len(reference_sets[field].intersection(str(row[field]) for row in records))
            for field in IDENTITY_BOUNDARY_FIELDS
        }
        if any(overlaps.values()):
            raise ValueError(
                f"source-reference/consumer identity overlap: {consumer['path']} {overlaps}"
            )
        per_consumer.append(
            {
                "path": consumer["path"],
                "sha256": consumer["sha256"],
                "domain": consumer["domain"],
                "episode_role": consumer["episode_role"],
                "complete_window_count": consumer["complete_window_count"],
                "record_count": consumer["record_count"],
                "overlap_counts": overlaps,
            }
        )
    aggregate_sets = {
        field: {str(record[field]) for record in all_consumer_records}
        for field in IDENTITY_BOUNDARY_FIELDS
    }
    aggregate_overlaps = {
        field: len(reference_sets[field].intersection(aggregate_sets[field]))
        for field in IDENTITY_BOUNDARY_FIELDS
    }
    if any(aggregate_overlaps.values()):
        raise RuntimeError("internal reference/consumer overlap audit inconsistency")
    return {
        "boundary_fields": list(IDENTITY_BOUNDARY_FIELDS),
        "reference_record_count": len(reference_records),
        "consumer_record_count": len(all_consumer_records),
        "reference_identity_set_sha256": {
            field: _identity_digest(list(reference_sets[field]))
            for field in IDENTITY_BOUNDARY_FIELDS
        },
        "consumer_identity_set_sha256": {
            field: _identity_digest(list(aggregate_sets[field]))
            for field in IDENTITY_BOUNDARY_FIELDS
        },
        "aggregate_overlap_counts": aggregate_overlaps,
        "per_consumer": per_consumer,
        "all_four_boundaries_zero_overlap": True,
    }


def _load_reference_pair(item: Any) -> tuple[np.ndarray, np.ndarray]:
    score_expected = item.record["score_file_sha256"]
    image_expected = item.record["original_image_sha256"]
    score_before = _sha256_file(item.score_path)
    if score_before != score_expected:
        raise ValueError("reference score file SHA-256 drifted before statistics read")
    with np.load(item.score_path, allow_pickle=False) as payload:
        probability = np.asarray(payload["prob"])
    if _sha256_file(item.score_path) != score_before:
        raise RuntimeError("reference score file changed during statistics read")
    image_before = _sha256_file(item.image_path)
    if image_before != image_expected:
        raise ValueError("reference image SHA-256 drifted before statistics read")
    with Image.open(item.image_path) as image:
        gray = np.asarray(image.convert("L"))
    if _sha256_file(item.image_path) != image_before:
        raise RuntimeError("reference image changed during statistics read")
    if probability.dtype != np.dtype("float64") or probability.ndim != 2:
        raise TypeError("reference probability must be a native 2D float64 array")
    if not np.isfinite(probability).all() or (
        probability.size
        and (float(probability.min()) < 0.0 or float(probability.max()) > 1.0)
    ):
        raise ValueError("reference probability contains invalid values")
    if gray.shape != probability.shape:
        raise ValueError("reference image/probability native geometry mismatch")
    return probability, gray


def _compute_centers(
    manifests: Sequence[VerifiedStage2ScoreManifest],
    source_domains: Sequence[str],
    statistics_config: StatisticsConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    by_domain = {manifest.payload["source_domain"]: manifest for manifest in manifests}
    centers: list[np.ndarray] = []
    summaries: dict[str, Any] = {}
    for domain in source_domains:
        manifest = by_domain[domain]

        def pairs() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for item in manifest.items:
                yield _load_reference_pair(item)

        splitter = _LockstepPairs(pairs())
        statistics = extract_unlabeled_statistics(
            splitter.probabilities(),
            splitter.grayscale(),
            statistics_config=statistics_config,
        )
        center = np.asarray(statistics.vector[:BASE_FEATURE_DIM], dtype=np.float32)
        if center.shape != (BASE_FEATURE_DIM,) or not np.isfinite(center).all():
            raise RuntimeError("Stage2 source center has invalid shape/values")
        centers.append(center)
        metadata = dict(statistics.metadata or {})
        summaries[domain] = {
            "record_count": len(manifest.records),
            "num_images": metadata.get("num_images"),
            "num_pixels": metadata.get("num_pixels"),
            "num_peaks": metadata.get("num_peaks"),
            "has_grayscale": metadata.get("has_grayscale"),
        }
    matrix = np.stack(centers, axis=0).astype(np.float32, copy=False)
    scale = matrix.astype(np.float64).std(axis=0)
    scale = np.where(scale < _SCALE_FLOOR, 1.0, scale).astype(np.float32)
    if matrix.shape != (2, BASE_FEATURE_DIM) or scale.shape != (BASE_FEATURE_DIM,):
        raise RuntimeError("Stage2 source-reference shape contract failed")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("Stage2 source-reference scale is invalid")
    return matrix, scale, summaries


def _future_output(value: str | Path, root: Path) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("output must be inside repository_root") from error
    if candidate.suffix != ".npz" or candidate.name.startswith("."):
        raise ValueError("output must be a non-hidden .npz path")
    _reject_symlink_components(candidate.parent, root, "output parent")
    parent = candidate.parent.resolve(strict=True)
    if not parent.is_dir():
        raise FileNotFoundError("output parent must be an existing directory")
    result = parent / candidate.name
    _relative_path(result.relative_to(root).as_posix(), "output")
    return result


def _bundle_paths(output: Path) -> tuple[Path, Path, Path, Path]:
    audit = output.with_suffix(".audit.json")
    return (
        output,
        audit,
        output.with_name(output.name + ".sha256"),
        audit.with_name(audit.name + ".sha256"),
    )


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _scalar_string(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or not isinstance(array.item(), str):
        raise TypeError(f"NPZ {name} must be a 0-D string")
    return str(array.item())


def _verify_staged_npz(
    path: Path,
    *,
    domains: Sequence[str],
    centers: np.ndarray,
    scale: np.ndarray,
    statistics_config: StatisticsConfig,
    source_contract: Mapping[str, Any],
    stage2_contract: Mapping[str, Any],
) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        names = tuple(info.filename for info in archive.infolist())
        if names != _NPZ_MEMBER_ORDER or len(names) != len(set(names)):
            raise ValueError("Stage2 source-reference NPZ ZIP member order mismatch")
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != _NPZ_FIELD_ORDER:
            raise ValueError("Stage2 source-reference NPZ field order mismatch")
        if _scalar_string(payload["schema_version"], "schema_version") != STAGE2_SOURCE_REFERENCE_SCHEMA:
            raise ValueError("Stage2 source-reference NPZ schema mismatch")
        loaded_domains = tuple(str(item) for item in np.asarray(payload["domains"]).tolist())
        if loaded_domains != tuple(domains):
            raise ValueError("Stage2 source-reference NPZ domain order mismatch")
        loaded_centers = np.asarray(payload["centers"])
        loaded_scale = np.asarray(payload["scale"])
        if loaded_centers.dtype != np.dtype("float32") or loaded_scale.dtype != np.dtype("float32"):
            raise TypeError("Stage2 source-reference centers/scale must be float32")
        np.testing.assert_array_equal(loaded_centers, centers)
        np.testing.assert_array_equal(loaded_scale, scale)
        expected_json = {
            "statistics_config_json": statistics_config.to_dict(),
            "source_contract_json": dict(source_contract),
            "stage2_contract_json": dict(stage2_contract),
        }
        for field, expected in expected_json.items():
            text = _scalar_string(payload[field], field)
            if text != _canonical_json(expected) or json.loads(text) != expected:
                raise ValueError(f"Stage2 source-reference NPZ {field} mismatch")
    reference = load_source_reference(path, statistics_config=statistics_config)
    if reference.domains != tuple(domains):
        raise RuntimeError("legacy-compatible source-reference self-check changed domains")


def _audit_fields() -> frozenset[str]:
    return frozenset(
        {
            "schema_version",
            "artifact_type",
            "artifact_status",
            "development_only",
            "execution_authorized",
            "training_authorized",
            "gpu_used",
            "official_test_accessed",
            "labels_or_masks_opened",
            "contains_observed_results",
            "observed_results",
            "path_anchor",
            "reference_role",
            "detector_identity",
            "source_domains",
            "bindings",
            "identity_boundary_audit",
            "statistics",
            "output",
        }
    )


def _verify_audit_payload(payload: Mapping[str, Any]) -> None:
    _exact_keys(payload, _audit_fields(), "source-reference audit")
    exact = {
        "schema_version": STAGE2_SOURCE_REFERENCE_AUDIT_SCHEMA,
        "artifact_type": STAGE2_SOURCE_REFERENCE_AUDIT_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY",
        "path_anchor": "repository_root",
    }
    for name, expected in exact.items():
        if payload[name] != expected:
            raise ValueError(f"source-reference audit {name} mismatch")
    for name, expected in {
        "development_only": True,
        "execution_authorized": False,
        "training_authorized": False,
        "gpu_used": False,
        "official_test_accessed": False,
        "labels_or_masks_opened": False,
        "contains_observed_results": False,
    }.items():
        _exact_bool(payload[name], f"source-reference audit {name}", expected)
    if payload["observed_results"] is not None:
        raise ValueError("source-reference audit observed_results must be null")
    boundary = payload["identity_boundary_audit"]
    if not isinstance(boundary, Mapping):
        raise TypeError("identity_boundary_audit must be an object")
    _exact_bool(
        boundary.get("all_four_boundaries_zero_overlap"),
        "identity_boundary_audit.all_four_boundaries_zero_overlap",
        True,
    )
    overlaps = boundary.get("aggregate_overlap_counts")
    if not isinstance(overlaps, Mapping) or set(overlaps) != set(IDENTITY_BOUNDARY_FIELDS):
        raise ValueError("identity boundary overlap fields mismatch")
    for field in IDENTITY_BOUNDARY_FIELDS:
        if _exact_int(overlaps[field], f"overlap {field}") != 0:
            raise ValueError("source-reference audit contains non-zero overlap")


def _publication_lock_path(output: Path) -> Path:
    return output.parent / f".{output.name}.stage2-source-reference.lock"


def _json_scalar_mapping(value: object, name: str) -> tuple[str, Mapping[str, Any]]:
    text = _scalar_string(value, name)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"NPZ {name} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"NPZ {name} must contain a JSON object")
    if text != _canonical_json(payload):
        raise ValueError(f"NPZ {name} must use canonical JSON")
    return text, payload


def _read_strict_source_reference_npz(
    path: Path,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    StatisticsConfig,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    with zipfile.ZipFile(path, "r") as archive:
        names = tuple(info.filename for info in archive.infolist())
        if names != _NPZ_MEMBER_ORDER or len(names) != len(set(names)):
            raise ValueError("Stage2 source-reference NPZ ZIP member order mismatch")
    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != _NPZ_FIELD_ORDER:
            raise ValueError("Stage2 source-reference NPZ field order mismatch")
        if (
            _scalar_string(payload["schema_version"], "schema_version")
            != STAGE2_SOURCE_REFERENCE_SCHEMA
        ):
            raise ValueError("Stage2 source-reference NPZ schema mismatch")
        raw_domains = np.asarray(payload["domains"])
        if raw_domains.ndim != 1 or raw_domains.dtype.kind != "U":
            raise TypeError("Stage2 source-reference domains must be a 1-D string array")
        domains = tuple(str(item) for item in raw_domains.tolist())
        centers = np.asarray(payload["centers"])
        scale = np.asarray(payload["scale"])
        if centers.dtype != np.dtype("float32") or scale.dtype != np.dtype("float32"):
            raise TypeError("Stage2 source-reference centers/scale must be float32")
        if centers.shape != (2, BASE_FEATURE_DIM):
            raise ValueError("Stage2 source-reference centers shape mismatch")
        if scale.shape != (BASE_FEATURE_DIM,):
            raise ValueError("Stage2 source-reference scale shape mismatch")
        if (
            not np.isfinite(centers).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("Stage2 source-reference centers/scale are invalid")
        _, config_payload = _json_scalar_mapping(
            payload["statistics_config_json"], "statistics_config_json"
        )
        _, source_contract = _json_scalar_mapping(
            payload["source_contract_json"], "source_contract_json"
        )
        _, stage2_contract = _json_scalar_mapping(
            payload["stage2_contract_json"], "stage2_contract_json"
        )
    _exact_keys(config_payload, _STATISTICS_CONFIG_FIELDS, "statistics_config_json")
    statistics_config = StatisticsConfig.from_dict(config_payload)
    if statistics_config.to_dict() != dict(config_payload):
        raise ValueError("embedded statistics config is not canonical")
    _exact_keys(source_contract, _SOURCE_CONTRACT_FIELDS, "source_contract_json")
    return (
        domains,
        centers,
        scale,
        statistics_config,
        source_contract,
        stage2_contract,
    )


def _verify_stage2_contract_payload(payload: Mapping[str, Any]) -> None:
    _exact_keys(payload, _STAGE2_CONTRACT_FIELDS, "stage2_contract_json")
    if payload["schema_version"] != STAGE2_SOURCE_REFERENCE_SCHEMA:
        raise ValueError("stage2_contract_json schema mismatch")
    if payload["artifact_type"] != STAGE2_SOURCE_REFERENCE_ARTIFACT_TYPE:
        raise ValueError("stage2_contract_json artifact_type mismatch")
    for name, expected in {
        "development_only": True,
        "execution_authorized": False,
        "official_test_accessed": False,
        "labels_or_masks_opened": False,
    }.items():
        _exact_bool(payload[name], f"stage2_contract_json.{name}", expected)
    role = payload["reference_role"]
    if role not in {
        OOF_TRAIN_SOURCE_REFERENCE,
        FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    }:
        raise ValueError("stage2_contract_json reference_role is forbidden")
    domains = payload["source_domains"]
    if (
        not isinstance(domains, list)
        or len(domains) != 2
        or len(set(domains)) != 2
        or any(domain not in STAGE2_DOMAINS for domain in domains)
    ):
        raise ValueError("stage2_contract_json requires two unique frozen source domains")
    identity = payload["detector_identity"]
    if not isinstance(identity, Mapping):
        raise TypeError("stage2_contract_json detector_identity must be an object")
    _exact_keys(identity, _DETECTOR_IDENTITY_FIELDS, "detector_identity")
    _string(identity["run_id"], "detector_identity.run_id")
    _string(identity["outer_fold_id"], "detector_identity.outer_fold_id")
    outer_target = _string(identity["outer_target"], "detector_identity.outer_target")
    if outer_target not in STAGE2_DOMAINS or outer_target in domains:
        raise ValueError("detector_identity outer target/source domains mismatch")
    _exact_int(identity["base_seed"], "detector_identity.base_seed")
    _exact_int(identity["derived_seed"], "detector_identity.derived_seed")
    detector_role = identity["detector_role"]
    expected_detector = (
        "detector_oof"
        if role == OOF_TRAIN_SOURCE_REFERENCE
        else "detector_full_fit"
    )
    if detector_role != expected_detector:
        raise ValueError("detector_identity detector/reference role mismatch")
    if detector_role == "detector_oof":
        if _exact_int(identity["oof_fold_index"], "detector_identity.oof_fold_index") not in {
            0,
            1,
        }:
            raise ValueError("detector_identity OOF fold must be 0 or 1")
    elif identity["oof_fold_index"] is not None:
        raise ValueError("full-fit detector_identity oof_fold_index must be null")
    _sha256_value(identity["checkpoint_sha256"], "detector_identity.checkpoint_sha256")
    bindings = payload["bindings"]
    if not isinstance(bindings, Mapping):
        raise TypeError("stage2_contract_json bindings must be an object")
    _exact_keys(bindings, _STAGE2_BINDING_FIELDS, "stage2_contract_json bindings")


def _binding_sequence(
    value: object,
    *,
    fields: frozenset[str],
    name: str,
    expected_count: int,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} entries")
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{name}[{index}] must be an object")
        _exact_keys(raw, fields, f"{name}[{index}]")
        result.append(raw)
    return result


def _verify_closed_stage2_bindings(
    stage2_contract: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[
    StatisticsConfig,
    tuple[VerifiedStage2ScoreManifest, ...],
    tuple[dict[str, Any], ...],
    Mapping[str, Any],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    bindings = stage2_contract["bindings"]
    assert isinstance(bindings, Mapping)
    policy = bindings["policy"]
    if not isinstance(policy, Mapping):
        raise TypeError("stage2 policy bindings must be an object")
    _exact_keys(policy, frozenset(_POLICY_BINDINGS), "stage2 policy bindings")
    policy_bindings, _ = _policy_contract(root)
    if dict(policy) != policy_bindings:
        raise ValueError("stage2 policy bindings differ from frozen policy")

    identity = stage2_contract["detector_identity"]
    assert isinstance(identity, Mapping)
    checkpoint_binding = bindings["checkpoint"]
    if not isinstance(checkpoint_binding, Mapping):
        raise TypeError("stage2 checkpoint binding must be an object")
    checkpoint_path = _resolve_binding(root, checkpoint_binding, "stage2 checkpoint")
    checkpoint_sha = _sha256_value(
        checkpoint_binding["sha256"], "stage2 checkpoint.sha256"
    )
    if checkpoint_sha != identity["checkpoint_sha256"]:
        raise ValueError("stage2 checkpoint/detector identity mismatch")

    source_entries = _binding_sequence(
        bindings["source_score_manifests"],
        fields=_SOURCE_SCORE_BINDING_FIELDS,
        name="source_score_manifests",
        expected_count=2,
    )
    source_paths = tuple(
        _existing_file(entry["path"], root, f"source_score_manifests[{index}]")
        for index, entry in enumerate(source_entries)
    )
    source_hashes = tuple(
        _sha256_value(entry["sha256"], f"source_score_manifests[{index}].sha256")
        for index, entry in enumerate(source_entries)
    )
    manifests, run_contract, role, _ = _verify_reference_manifests(
        source_paths,
        source_hashes,
        root=root,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
    )
    if role != stage2_contract["reference_role"]:
        raise ValueError("source manifests/reference role mismatch")
    expected_source_entries = []
    by_domain = {manifest.payload["source_domain"]: manifest for manifest in manifests}
    for domain in run_contract["source_domains"]:
        manifest = by_domain[domain]
        expected_source_entries.append(
            {
                "path": manifest.path.relative_to(root).as_posix(),
                "sha256": manifest.manifest_sha256,
                "source_domain": domain,
                "records_content_sha256": manifest.records_content_sha256,
                "record_count": len(manifest.records),
                "selection_contract": dict(manifest.bindings["selection_contract"]),
            }
        )
    if source_entries != expected_source_entries:
        raise ValueError("source_score_manifests binding summary mismatch")
    run_binding = bindings["run_contract"]
    if not isinstance(run_binding, Mapping):
        raise TypeError("stage2 run_contract binding must be an object")
    _exact_keys(run_binding, _BINDING_FIELDS, "stage2 run_contract binding")
    if dict(run_binding) != dict(manifests[0].bindings["run_contract"]):
        raise ValueError("stage2 run_contract binding mismatch")

    config_binding = bindings["statistics_config"]
    if not isinstance(config_binding, Mapping):
        raise TypeError("stage2 statistics_config binding must be an object")
    config_path = _resolve_binding(root, config_binding, "stage2 statistics config")
    config_sha = _sha256_value(
        config_binding["sha256"], "stage2 statistics_config.sha256"
    )
    statistics_config, _ = _load_statistics_config(config_path, config_sha)

    detector_role = str(identity["detector_role"])
    expected_consumers = 2 if detector_role == "detector_oof" else 3
    consumer_entries = _binding_sequence(
        bindings["consumer_window_manifests"],
        fields=_CONSUMER_BINDING_FIELDS,
        name="consumer_window_manifests",
        expected_count=expected_consumers,
    )
    consumers = _verify_all_consumers(
        tuple(
            _existing_file(entry["path"], root, f"consumer_window_manifests[{index}]")
            for index, entry in enumerate(consumer_entries)
        ),
        tuple(
            _sha256_value(
                entry["sha256"], f"consumer_window_manifests[{index}].sha256"
            )
            for index, entry in enumerate(consumer_entries)
        ),
        root=root,
        run_contract=run_contract,
        detector_role=detector_role,
        oof_fold_index=identity["oof_fold_index"],
        rechecks=[],
    )
    expected_consumer_entries = [
        {
            key: consumer[key]
            for key in (
                "path",
                "sha256",
                "domain",
                "episode_role",
                "complete_window_count",
                "record_count",
            )
        }
        for consumer in consumers
    ]
    if consumer_entries != expected_consumer_entries:
        raise ValueError("consumer_window_manifests binding summary mismatch")
    boundary = _identity_audit(manifests, consumers)
    centers, scale, summaries = _compute_centers(
        manifests, run_contract["source_domains"], statistics_config
    )
    return (
        statistics_config,
        manifests,
        consumers,
        run_contract,
        {"identity_boundary_audit": boundary, "domain_summaries": summaries},
        centers,
        scale,
    )


def _verify_expected_consumer_window(
    stage2_contract: Mapping[str, Any],
    *,
    root: Path,
    path: str | Path | None,
    sha256: str | None,
    window_id: str | None,
) -> None:
    provided = (path is not None, sha256 is not None, window_id is not None)
    if any(provided) and not all(provided):
        raise ValueError(
            "expected consumer window path, SHA-256 and window_id must be "
            "provided together"
        )
    if not any(provided):
        return
    assert path is not None and sha256 is not None and window_id is not None
    expected_path = _existing_file(path, root, "expected consumer window")
    expected_relative = expected_path.relative_to(root).as_posix()
    expected_sha = _sha256_value(
        sha256, "expected_consumer_window_sha256"
    )
    expected_id = _string(window_id, "expected_consumer_window_id")
    bindings = stage2_contract["bindings"]
    assert isinstance(bindings, Mapping)
    consumers = bindings["consumer_window_manifests"]
    if not isinstance(consumers, list):
        raise TypeError("consumer_window_manifests must be a list")
    matches = [
        binding
        for binding in consumers
        if isinstance(binding, Mapping)
        and binding.get("path") == expected_relative
        and binding.get("sha256") == expected_sha
    ]
    if len(matches) != 1:
        raise ValueError("expected consumer window path/SHA is not uniquely bound")
    payload = _read_json_stable(
        expected_path, expected_sha, "expected consumer window"
    )
    windows = payload.get("windows")
    if not isinstance(windows, list):
        raise TypeError("expected consumer window manifest has no windows list")
    occurrences = sum(
        1
        for raw in windows
        if isinstance(raw, Mapping) and raw.get("window_id") == expected_id
    )
    if occurrences != 1:
        raise ValueError(
            "expected consumer window_id must occur exactly once in its bound manifest"
        )


def _verify_stage2_source_reference_bundle(
    path: Path,
    expected_sha256: str,
    expected_audit_sha256: str,
    *,
    root: Path,
    logical_output_path: Path,
    publication_lock: Path,
    require_lock_present: bool,
    expected_statistics_config: StatisticsConfig,
    expected_consumer_window_path: str | Path | None = None,
    expected_consumer_window_sha256: str | None = None,
    expected_consumer_window_id: str | None = None,
) -> VerifiedStage2SourceReference:
    lock_present = os.path.lexists(publication_lock)
    if lock_present is not require_lock_present:
        state = "present" if lock_present else "absent"
        raise RuntimeError(
            f"source-reference publication lock must be "
            f"{'present' if require_lock_present else 'absent'}; observed {state}"
        )
    bundle_paths = _bundle_paths(path)
    resolved = tuple(
        _existing_file(candidate, root, f"source-reference bundle[{index}]")
        for index, candidate in enumerate(bundle_paths)
    )
    npz_path, audit_path, npz_sidecar, audit_sidecar = resolved
    npz_expected = _sha256_value(expected_sha256, "expected_sha256")
    audit_expected = _sha256_value(
        expected_audit_sha256, "expected_audit_sha256"
    )
    identities = tuple(_file_identity(candidate) for candidate in resolved)
    _verify_file_hash(npz_path, npz_expected, "source-reference NPZ")
    _verify_file_hash(audit_path, audit_expected, "source-reference audit")
    if _read_text_stable(
        npz_sidecar, encoding="ascii", name="source-reference NPZ sidecar"
    ) != f"{npz_expected}  {logical_output_path.name}\n":
        raise ValueError("source-reference NPZ sidecar mismatch")
    if _read_text_stable(
        audit_sidecar, encoding="ascii", name="source-reference audit sidecar"
    ) != f"{audit_expected}  {logical_output_path.with_suffix('.audit.json').name}\n":
        raise ValueError("source-reference audit sidecar mismatch")

    (
        domains,
        centers,
        scale,
        embedded_config,
        source_contract,
        stage2_contract,
    ) = _read_strict_source_reference_npz(npz_path)
    _verify_stage2_contract_payload(stage2_contract)
    (
        bound_config,
        manifests,
        consumers,
        run_contract,
        recomputed,
        recomputed_centers,
        recomputed_scale,
    ) = _verify_closed_stage2_bindings(stage2_contract, root=root)
    if (
        embedded_config != bound_config
        or bound_config != expected_statistics_config
    ):
        raise ValueError("embedded/bound/expected statistics config mismatch")
    _verify_expected_consumer_window(
        stage2_contract,
        root=root,
        path=expected_consumer_window_path,
        sha256=expected_consumer_window_sha256,
        window_id=expected_consumer_window_id,
    )
    if domains != tuple(run_contract["source_domains"]):
        raise ValueError("NPZ/run source-domain order mismatch")
    np.testing.assert_array_equal(centers, recomputed_centers)
    np.testing.assert_array_equal(scale, recomputed_scale)
    if stage2_contract["identity_boundary_audit"] != recomputed[
        "identity_boundary_audit"
    ]:
        raise ValueError("embedded identity boundary audit mismatch")

    identity = stage2_contract["detector_identity"]
    assert isinstance(identity, Mapping)
    expected_source_contract = {
        "detector_checkpoint_sha": identity["checkpoint_sha256"],
        "detector_source_domains": list(domains),
        "outer_fold_id": identity["outer_fold_id"],
        "outer_target": identity["outer_target"],
        "held_out_domains": [identity["outer_target"]],
        "protocol_scope": "multi_source_protocol_candidate",
    }
    if dict(source_contract) != expected_source_contract:
        raise ValueError("source_contract_json/stage2 identity mismatch")

    audit = _read_json_stable(
        audit_path, audit_expected, "source-reference audit"
    )
    _verify_audit_payload(audit)
    for name in (
        "reference_role",
        "detector_identity",
        "source_domains",
        "bindings",
        "identity_boundary_audit",
    ):
        if audit[name] != stage2_contract[name]:
            raise ValueError(f"audit/stage2_contract {name} mismatch")
    expected_statistics = {
        "feature_width": BASE_FEATURE_DIM,
        "centers_shape": [2, BASE_FEATURE_DIM],
        "scale_shape": [BASE_FEATURE_DIM],
        "domain_summaries": recomputed["domain_summaries"],
    }
    if audit["statistics"] != expected_statistics:
        raise ValueError("source-reference audit statistics mismatch")
    expected_output = {
        "source_reference_npz": {
            "path": logical_output_path.relative_to(root).as_posix(),
            "sha256": npz_expected,
        }
    }
    if audit["output"] != expected_output:
        raise ValueError("source-reference audit output binding mismatch")
    reference = load_source_reference(npz_path, statistics_config=bound_config)
    if reference.sha256 != npz_expected:
        raise RuntimeError("public source-reference loader hash mismatch")

    for candidate, identity_before in zip(resolved, identities):
        if _file_identity(candidate) != identity_before:
            raise RuntimeError("source-reference bundle identity changed while verified")
    if os.path.lexists(publication_lock) is not require_lock_present:
        raise RuntimeError("source-reference publication lock changed while verified")
    frozen_stage2 = _deep_freeze(stage2_contract)
    frozen_source = _deep_freeze(source_contract)
    frozen_audit = _deep_freeze(audit)
    frozen_identity = _deep_freeze(identity)
    frozen_checkpoint = _deep_freeze(stage2_contract["bindings"]["checkpoint"])
    frozen_consumers = tuple(
        _deep_freeze(item)
        for item in stage2_contract["bindings"]["consumer_window_manifests"]
    )
    return _make_verified_stage2_source_reference(
        path=npz_path,
        npz_sha256=npz_expected,
        audit_path=audit_path,
        audit_sha256=audit_expected,
        domains=reference.domains,
        centers=reference.centers,
        scale=reference.scale,
        source_reference=reference,
        statistics_config=bound_config,
        stage2_contract=frozen_stage2,
        source_contract=frozen_source,
        detector_identity=frozen_identity,
        checkpoint_binding=frozen_checkpoint,
        reference_role=str(stage2_contract["reference_role"]),
        consumer_bindings=frozen_consumers,
        audit=frozen_audit,
    )


def verify_stage2_source_reference(
    path: str | Path,
    expected_sha256: str,
    expected_audit_sha256: str,
    *,
    statistics_config: StatisticsConfig,
    expected_consumer_window_path: str | Path | None = None,
    expected_consumer_window_sha256: str | None = None,
    expected_consumer_window_id: str | None = None,
    repository_root: str | Path | None = None,
) -> VerifiedStage2SourceReference:
    """Verify and load a complete published Stage2 source-reference bundle.

    The two externally supplied digests are mandatory.  A publication lock,
    missing member, stale sidecar, symlink, byte/identity drift, non-canonical
    contract or stale transitive binding fails closed.
    """

    if not isinstance(statistics_config, StatisticsConfig):
        raise TypeError("statistics_config must be a StatisticsConfig")
    root = _root(repository_root)
    npz_path = _existing_file(path, root, "source-reference NPZ")
    lock_path = _publication_lock_path(npz_path)
    if os.path.lexists(lock_path):
        raise RuntimeError("source-reference publication lock is present")
    return _verify_stage2_source_reference_bundle(
        npz_path,
        expected_sha256,
        expected_audit_sha256,
        root=root,
        logical_output_path=npz_path,
        publication_lock=lock_path,
        require_lock_present=False,
        expected_statistics_config=statistics_config,
        expected_consumer_window_path=expected_consumer_window_path,
        expected_consumer_window_sha256=expected_consumer_window_sha256,
        expected_consumer_window_id=expected_consumer_window_id,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bundle(
    staged_paths: Sequence[Path], final_paths: Sequence[Path]
) -> None:
    published: list[Path] = []
    try:
        for staged, final in zip(staged_paths, final_paths):
            os.link(staged, final, follow_symlinks=False)
            published.append(final)
        _fsync_directory(final_paths[0].parent)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(final_paths[0].parent)
        raise


def build_stage2_source_reference(
    score_manifests: Sequence[str | Path],
    score_manifest_sha256s: Sequence[str],
    checkpoint: str | Path,
    checkpoint_sha256: str,
    statistics_config: str | Path,
    statistics_config_sha256: str,
    consumer_window_manifests: Sequence[str | Path],
    consumer_window_manifest_sha256s: Sequence[str],
    output: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build and publish one strictly hash-bound Stage2 source reference.

    All score and consumer hashes are supplied by the caller.  The output is
    never replaced; any preflight, construction or publication failure leaves
    none of this invocation's four bundle members behind.
    """

    root = _root(repository_root)
    score_paths = tuple(
        _existing_file(value, root, f"score_manifests[{index}]")
        for index, value in enumerate(score_manifests)
    )
    score_hashes = tuple(
        _sha256_value(value, f"score_manifest_sha256s[{index}]")
        for index, value in enumerate(score_manifest_sha256s)
    )
    checkpoint_hash = _sha256_value(checkpoint_sha256, "checkpoint_sha256")
    checkpoint_path = _existing_file(checkpoint, root, "checkpoint")
    _verify_file_hash(checkpoint_path, checkpoint_hash, "checkpoint")
    config_hash = _sha256_value(
        statistics_config_sha256, "statistics_config_sha256"
    )
    config_path = _existing_file(statistics_config, root, "statistics config")
    config, config_payload = _load_statistics_config(config_path, config_hash)
    consumer_paths = tuple(
        _existing_file(value, root, f"consumer_window_manifests[{index}]")
        for index, value in enumerate(consumer_window_manifests)
    )
    consumer_hashes = tuple(
        _sha256_value(value, f"consumer_window_manifest_sha256s[{index}]")
        for index, value in enumerate(consumer_window_manifest_sha256s)
    )
    output_path = _future_output(output, root)
    final_paths = _bundle_paths(output_path)
    if len(set(final_paths)) != 4 or len({path.parent for path in final_paths}) != 1:
        raise RuntimeError("source-reference bundle paths are not unique/same-parent")
    for path in final_paths:
        if os.path.lexists(path):
            raise FileExistsError(f"source-reference bundle target already exists: {path}")

    lock_path = _publication_lock_path(output_path)
    if os.path.lexists(lock_path):
        raise FileExistsError(f"source-reference publication lock exists: {lock_path}")
    lock_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    try:
        os.close(lock_descriptor)
    except BaseException:
        try:
            lock_path.unlink()
        finally:
            raise
    try:
        _fsync_directory(output_path.parent)
    except BaseException:
        try:
            lock_path.unlink()
        finally:
            try:
                _fsync_directory(output_path.parent)
            except BaseException:
                pass
        raise
    staging: Path | None = None
    published = False
    committed = False
    try:
        policy_bindings, rechecks = _policy_contract(root)
        manifests, run_contract, reference_role, manifest_rechecks = (
            _verify_reference_manifests(
                score_paths,
                score_hashes,
                root=root,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
            )
        )
        rechecks.extend(manifest_rechecks)
        detector_role = str(manifests[0].payload["detector_role"])
        oof_index = manifests[0].payload["oof_fold_index"]
        consumers = _verify_all_consumers(
            consumer_paths,
            consumer_hashes,
            root=root,
            run_contract=run_contract,
            detector_role=detector_role,
            oof_fold_index=oof_index,
            rechecks=rechecks,
        )
        boundary_audit = _identity_audit(manifests, consumers)
        centers, scale, domain_summaries = _compute_centers(
            manifests, run_contract["source_domains"], config
        )

        run_binding = manifests[0].bindings["run_contract"]
        source_score_bindings = []
        by_domain = {manifest.payload["source_domain"]: manifest for manifest in manifests}
        for domain in run_contract["source_domains"]:
            manifest = by_domain[domain]
            source_score_bindings.append(
                {
                    "path": manifest.path.relative_to(root).as_posix(),
                    "sha256": manifest.manifest_sha256,
                    "source_domain": domain,
                    "records_content_sha256": manifest.records_content_sha256,
                    "record_count": len(manifest.records),
                    "selection_contract": dict(
                        manifest.bindings["selection_contract"]
                    ),
                }
            )
        consumer_bindings = [
            {
                key: consumer[key]
                for key in (
                    "path",
                    "sha256",
                    "domain",
                    "episode_role",
                    "complete_window_count",
                    "record_count",
                )
            }
            for consumer in consumers
        ]
        bindings: dict[str, Any] = {
            "policy": policy_bindings,
            "source_score_manifests": source_score_bindings,
            "run_contract": dict(run_binding),
            "checkpoint": _binding(checkpoint_path, root, checkpoint_hash),
            "statistics_config": _binding(config_path, root, config_hash),
            "consumer_window_manifests": consumer_bindings,
        }
        detector_identity = {
            "run_id": run_contract["run_id"],
            "outer_fold_id": manifests[0].payload["outer_fold_id"],
            "outer_target": manifests[0].payload["outer_target"],
            "base_seed": manifests[0].payload["base_seed"],
            "derived_seed": manifests[0].payload["derived_seed"],
            "detector_role": detector_role,
            "oof_fold_index": oof_index,
            "checkpoint_sha256": checkpoint_hash,
        }
        stage2_contract = {
            "schema_version": STAGE2_SOURCE_REFERENCE_SCHEMA,
            "artifact_type": STAGE2_SOURCE_REFERENCE_ARTIFACT_TYPE,
            "development_only": True,
            "execution_authorized": False,
            "official_test_accessed": False,
            "labels_or_masks_opened": False,
            "reference_role": reference_role,
            "detector_identity": detector_identity,
            "source_domains": list(run_contract["source_domains"]),
            "bindings": bindings,
            "identity_boundary_audit": boundary_audit,
        }
        source_contract = {
            "detector_checkpoint_sha": checkpoint_hash,
            "detector_source_domains": list(run_contract["source_domains"]),
            "outer_fold_id": manifests[0].payload["outer_fold_id"],
            "outer_target": manifests[0].payload["outer_target"],
            "held_out_domains": [manifests[0].payload["outer_target"]],
            "protocol_scope": "multi_source_protocol_candidate",
        }

        for path, expected in rechecks:
            _verify_file_hash(path, expected, path.relative_to(root).as_posix())
        _verify_file_hash(checkpoint_path, checkpoint_hash, "checkpoint final preflight")
        _verify_file_hash(config_path, config_hash, "statistics config final preflight")

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output_path.stem}.stage2-source-reference-staging-",
                dir=output_path.parent,
            )
        )
        os.chmod(staging, 0o700)
        staged_paths = tuple(staging / path.name for path in final_paths)
        arrays = {
            "schema_version": np.asarray(STAGE2_SOURCE_REFERENCE_SCHEMA),
            "domains": np.asarray(run_contract["source_domains"], dtype=np.str_),
            "centers": centers,
            "scale": scale,
            "statistics_config_json": np.asarray(_canonical_json(config_payload)),
            "source_contract_json": np.asarray(_canonical_json(source_contract)),
            "stage2_contract_json": np.asarray(_canonical_json(stage2_contract)),
        }
        _write_npz_exclusive(staged_paths[0], arrays)
        npz_sha = _sha256_file(staged_paths[0])
        audit_payload: dict[str, Any] = {
            "schema_version": STAGE2_SOURCE_REFERENCE_AUDIT_SCHEMA,
            "artifact_type": STAGE2_SOURCE_REFERENCE_AUDIT_ARTIFACT_TYPE,
            "artifact_status": "DEVELOPMENT_ONLY",
            "development_only": True,
            "execution_authorized": False,
            "training_authorized": False,
            "gpu_used": False,
            "official_test_accessed": False,
            "labels_or_masks_opened": False,
            "contains_observed_results": False,
            "observed_results": None,
            "path_anchor": "repository_root",
            "reference_role": reference_role,
            "detector_identity": detector_identity,
            "source_domains": list(run_contract["source_domains"]),
            "bindings": bindings,
            "identity_boundary_audit": boundary_audit,
            "statistics": {
                "feature_width": BASE_FEATURE_DIM,
                "centers_shape": [2, BASE_FEATURE_DIM],
                "scale_shape": [BASE_FEATURE_DIM],
                "domain_summaries": domain_summaries,
            },
            "output": {
                "source_reference_npz": {
                    "path": output_path.relative_to(root).as_posix(),
                    "sha256": npz_sha,
                }
            },
        }
        _verify_audit_payload(audit_payload)
        audit_bytes = (
            json.dumps(
                audit_payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_exclusive(staged_paths[1], audit_bytes)
        audit_sha = _sha256_file(staged_paths[1])
        _write_exclusive(
            staged_paths[2], f"{npz_sha}  {final_paths[0].name}\n".encode("ascii")
        )
        _write_exclusive(
            staged_paths[3], f"{audit_sha}  {final_paths[1].name}\n".encode("ascii")
        )
        _fsync_directory(staging)

        _verify_staged_npz(
            staged_paths[0],
            domains=run_contract["source_domains"],
            centers=centers,
            scale=scale,
            statistics_config=config,
            source_contract=source_contract,
            stage2_contract=stage2_contract,
        )
        staged_audit = _read_json_stable(
            staged_paths[1], audit_sha, "staged source-reference audit"
        )
        _verify_audit_payload(staged_audit)
        if staged_paths[2].read_text(encoding="ascii") != f"{npz_sha}  {final_paths[0].name}\n":
            raise ValueError("staged NPZ sidecar mismatch")
        if staged_paths[3].read_text(encoding="ascii") != f"{audit_sha}  {final_paths[1].name}\n":
            raise ValueError("staged audit sidecar mismatch")

        for path, expected in rechecks:
            _verify_file_hash(path, expected, path.relative_to(root).as_posix())
        _publish_bundle(staged_paths, final_paths)
        published = True
        _verify_stage2_source_reference_bundle(
            final_paths[0],
            npz_sha,
            audit_sha,
            root=root,
            logical_output_path=output_path,
            publication_lock=lock_path,
            require_lock_present=True,
            expected_statistics_config=config,
        )
        shutil.rmtree(staging)
        staging = None
        # This is the last fallible durability operation.  The lock remains
        # present, so public readers still fail closed if it raises.
        _fsync_directory(output_path.parent)
        # Atomic commit point: after this unlink there are no fallible
        # filesystem operations.  A crash may leave a stale lock, which is
        # fail-closed; it can never expose a partial bundle as complete.
        lock_path.unlink()
        committed = True
        return dict(audit_payload)
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        if published:
            for path in reversed(final_paths):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
        try:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if not committed:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            _fsync_directory(output_path.parent)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            residue = [path.as_posix() for path in (*final_paths, lock_path) if os.path.lexists(path)]
            if staging is not None and staging.exists():
                residue.append(staging.as_posix())
            if residue:
                raise RuntimeError(
                    f"source-reference rollback left residue: {residue}"
                ) from error
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-manifest", action="append", required=True)
    parser.add_argument(
        "--score-manifest-sha256", action="append", required=True
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--statistics-config", required=True)
    parser.add_argument("--statistics-config-sha256", required=True)
    parser.add_argument(
        "--consumer-window-manifest", action="append", required=True
    )
    parser.add_argument(
        "--consumer-window-manifest-sha256", action="append", required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    audit = build_stage2_source_reference(
        args.score_manifest,
        args.score_manifest_sha256,
        args.checkpoint,
        args.expected_checkpoint_sha256,
        args.statistics_config,
        args.statistics_config_sha256,
        args.consumer_window_manifest,
        args.consumer_window_manifest_sha256,
        args.output,
        repository_root=args.repository_root,
    )
    print(
        _canonical_json(
            {
                "schema_version": audit["schema_version"],
                "reference_role": audit["reference_role"],
                "source_domains": audit["source_domains"],
                "official_test_accessed": audit["official_test_accessed"],
                "output": audit["output"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IDENTITY_BOUNDARY_FIELDS",
    "STAGE2_SOURCE_REFERENCE_AUDIT_SCHEMA",
    "STAGE2_SOURCE_REFERENCE_SCHEMA",
    "VerifiedStage2SourceReference",
    "build_arg_parser",
    "build_stage2_source_reference",
    "main",
    "verify_stage2_source_reference",
]
