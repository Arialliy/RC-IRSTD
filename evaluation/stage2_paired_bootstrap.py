"""Fail-closed Stage-2 paired hierarchical bootstrap (semantic v2).

The three outer domains are fixed equal-weight strata.  Within each domain a
replicate draws three training-seed slots, then the frozen number of window
clusters for the selected cell, then exactly 28 query-image positions inside
each selected window.  All draws are stateless SHA-256 functions.  Method IDs,
thresholds, decisions, checkpoints, and results never enter a draw preimage,
so T8 and T4 consume one byte-identical seed/window/query index artifact.

Public evaluation accepts only canonical repository paths plus externally
expected file SHA-256 values.  It re-reads all artifact bytes, validates every
pairing invariant, and replays every deterministic draw before computing any
statistic.  This module is result-free, CPU-only, and never opens a dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence


SOURCE_THAW_SHA256 = (
    "0e4f3e27026d5a2071a2c8f94f84c366d208f3789de17649aad926c64cd6b0b9"
)
WORK_BREAKDOWN_SHA256 = (
    "cc240f97aea6c99dde1e5c537a26c1b22e606b0f499ca495af71d15fa44c9d06"
)
AUTHORIZATION_AMENDMENT_SHA256 = (
    "185b7e4cac7d7a23ca537641575a00c5e64c6a5d0783dc34f999ba402f174845"
)
SEED_MANIFEST_SCHEMA_VERSION = "rc-irstd.stage2-seed-derivation-manifest.v1"
PAIR_MANIFEST_SCHEMA_VERSION = "rc-irstd.stage2-primary-pair-manifest.v2"
IMAGE_COUNTS_SCHEMA_VERSION = "rc-irstd.stage2-image-sufficient-counts.v1"
INDEX_MANIFEST_SCHEMA_VERSION = "rc-irstd.stage2-bootstrap-index-manifest.v1"
REPORT_SCHEMA_VERSION = "rc-irstd.stage2-paired-bootstrap-report.v2"

PROTOCOL_ID = "outer_fixed_seed_window_query_paired_bootstrap_v2"
SEED_INDEX_TAG = "rc-irstd.stage2.bootstrap.seed-index.v2"
WINDOW_INDEX_TAG = "rc-irstd.stage2.bootstrap.window-index.v2"
QUERY_INDEX_TAG = "rc-irstd.stage2.bootstrap.query-index.v2"
BOOTSTRAP_ROLE = "paired_bootstrap_query_images::not_applicable"
THRESHOLD_SEMANTICS = "prediction = probability > threshold"

DOMAIN_ORDER = (
    "outer_leave_nuaa_sirst",
    "outer_leave_nudt_sirst",
    "outer_leave_irstd_1k",
)
TARGET_BY_DOMAIN = {
    "outer_leave_nuaa_sirst": "nuaa-sirst",
    "outer_leave_nudt_sirst": "nudt-sirst",
    "outer_leave_irstd_1k": "irstd-1k",
}
WINDOW_COUNT_BY_DOMAIN = {
    "outer_leave_nuaa_sirst": 1,
    "outer_leave_nudt_sirst": 3,
    "outer_leave_irstd_1k": 3,
}
BASE_SEED_ORDER = (42, 123, 3407)
METHOD_ORDER = ("T8", "T4")
QUERY_IMAGES_PER_WINDOW = 28
PRIMARY_BUDGET = 1e-5
PRIMARY_RESAMPLES = 10_000
CI_QUANTILES = (0.025, 0.975)
_SHA256_HEX = frozenset("0123456789abcdef")


class Stage2BootstrapContractError(ValueError):
    """A Stage-2 bootstrap artifact failed a frozen fail-closed contract."""


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2BootstrapContractError(
            f"payload is not finite canonical JSON: {error}"
        ) from error


def _json_file_bytes(payload: Any) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or value != value.lower() or not set(value) <= _SHA256_HEX:
        raise Stage2BootstrapContractError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:  # noqa: E721 - exact JSON boolean required
        raise TypeError(f"{name} must be an exact JSON boolean")
    return value


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise Stage2BootstrapContractError(f"{name} must be >= {minimum}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise Stage2BootstrapContractError(f"{name} must be finite")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip():
        raise Stage2BootstrapContractError(
            f"{name} must be non-empty with no surrounding whitespace"
        )
    return value


def _assert_exact_keys(
    payload: Mapping[str, Any], *, required: set[str], name: str
) -> None:
    missing = required - set(payload)
    extra = set(payload) - required
    if missing or extra:
        raise Stage2BootstrapContractError(
            f"{name} fields differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2BootstrapContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _nonfinite_guard(value: str) -> None:
    raise Stage2BootstrapContractError(f"non-finite JSON number is forbidden: {value}")


def _parse_json_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_guard,
            parse_constant=_nonfinite_guard,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2BootstrapContractError(f"invalid UTF-8 JSON in {name}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _canonical_repository_root(repository_root: str | Path) -> Path:
    raw = Path(repository_root)
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise Stage2BootstrapContractError(
            "repository_root must be an absolute canonical non-symlink path"
        )
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2BootstrapContractError("repository_root does not exist") from error
    if resolved != raw or not resolved.is_dir():
        raise Stage2BootstrapContractError(
            "repository_root must be an absolute canonical directory"
        )
    return resolved


def _path_within_root(
    path: str | Path, repository_root: Path, *, name: str, must_exist: bool
) -> Path:
    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise Stage2BootstrapContractError(
            f"{name} must be an absolute canonical non-symlink path"
        )
    parent = raw.parent.resolve(strict=True)
    if parent != raw.parent or not parent.is_dir():
        raise Stage2BootstrapContractError(f"{name} parent is not canonical")
    if not parent.is_relative_to(repository_root):
        raise Stage2BootstrapContractError(f"{name} escapes repository_root")
    if must_exist:
        try:
            resolved = raw.resolve(strict=True)
        except FileNotFoundError as error:
            raise Stage2BootstrapContractError(f"{name} does not exist") from error
        if resolved != raw or not resolved.is_file() or raw.is_symlink():
            raise Stage2BootstrapContractError(
                f"{name} must be a canonical regular file without symlink aliases"
            )
    return raw


@dataclass(frozen=True)
class _VerifiedJsonArtifact:
    path: Path
    payload: dict[str, Any]
    data: bytes
    sha256: str


def _load_verified_json(
    path: str | Path,
    expected_sha256: str,
    repository_root: str | Path,
    *,
    name: str,
) -> _VerifiedJsonArtifact:
    root = _canonical_repository_root(repository_root)
    checked = _path_within_root(path, root, name=name, must_exist=True)
    expected = validate_sha256(expected_sha256, f"expected_{name}_sha256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(checked, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2BootstrapContractError(f"{name} is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read()
        after_path = os.stat(checked, follow_symlinks=False)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_size,
        ):
            raise Stage2BootstrapContractError(f"{name} changed during verified read")
    finally:
        os.close(descriptor)
    observed = sha256_bytes(data)
    if observed != expected:
        raise Stage2BootstrapContractError(
            f"{name} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return _VerifiedJsonArtifact(
        path=checked,
        payload=_parse_json_bytes(data, name=name),
        data=data,
        sha256=observed,
    )


def _identity_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "image_id": row["image_id"],
            "original_image_sha256": row["original_image_sha256"],
        }
        for row in rows
    ]
    return sha256_bytes(canonical_json_bytes(identity))


def _validate_count_row(row: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"{name} must be a mapping")
    required = {
        "image_id",
        "original_image_sha256",
        "false_positive_pixels",
        "total_pixels",
        "background_pixels",
        "matched_targets",
        "ground_truth_targets",
    }
    _assert_exact_keys(row, required=required, name=name)
    result = {
        "image_id": _nonempty_string(row["image_id"], f"{name}.image_id"),
        "original_image_sha256": validate_sha256(
            row["original_image_sha256"], f"{name}.original_image_sha256"
        ),
    }
    for field in (
        "false_positive_pixels",
        "total_pixels",
        "background_pixels",
        "matched_targets",
        "ground_truth_targets",
    ):
        result[field] = _strict_int(row[field], f"{name}.{field}")
    if result["total_pixels"] <= 0:
        raise Stage2BootstrapContractError(f"{name}.total_pixels must be positive")
    if result["false_positive_pixels"] > result["total_pixels"]:
        raise Stage2BootstrapContractError(f"{name}.false_positive_pixels exceeds total")
    if result["background_pixels"] > result["total_pixels"]:
        raise Stage2BootstrapContractError(f"{name}.background_pixels exceeds total")
    if result["matched_targets"] > result["ground_truth_targets"]:
        raise Stage2BootstrapContractError(f"{name}.matched_targets exceeds GT")
    return result


def _window_identity(window: Mapping[str, Any]) -> dict[str, str]:
    return {
        "window_id": str(window["window_id"]),
        "window_identity_sha256": str(window["window_identity_sha256"]),
        "context_identity_sha256": str(window["context_identity_sha256"]),
        "ordered_query_identity_sha256": str(
            window["ordered_query_identity_sha256"]
        ),
    }


def _validate_method_window(
    payload: Mapping[str, Any], *, method_id: str, name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    required = {
        "window_id",
        "window_identity_sha256",
        "context_identity_sha256",
        "ordered_query_identity_sha256",
        "decision_sha256",
        "decision_sealed",
        "threshold",
        "threshold_semantics",
        "online_update_count",
        "threshold_reselected",
        "query_counts",
    }
    _assert_exact_keys(payload, required=required, name=name)
    result = dict(payload)
    result["window_id"] = _nonempty_string(payload["window_id"], f"{name}.window_id")
    for field in (
        "window_identity_sha256",
        "context_identity_sha256",
        "ordered_query_identity_sha256",
        "decision_sha256",
    ):
        result[field] = validate_sha256(payload[field], f"{name}.{field}")
    if _strict_bool(payload["decision_sealed"], f"{name}.decision_sealed") is not True:
        raise Stage2BootstrapContractError(f"{name} decision is not sealed")
    result["threshold"] = _finite_float(payload["threshold"], f"{name}.threshold")
    if not 0.0 <= result["threshold"] <= 1.0:
        raise Stage2BootstrapContractError(f"{name}.threshold must lie in [0,1]")
    if payload["threshold_semantics"] != THRESHOLD_SEMANTICS:
        raise Stage2BootstrapContractError(f"{name} threshold semantics mismatch")
    if _strict_int(payload["online_update_count"], f"{name}.online_update_count") != 0:
        raise Stage2BootstrapContractError(f"{name} contains an online update")
    if _strict_bool(payload["threshold_reselected"], f"{name}.threshold_reselected") is not False:
        raise Stage2BootstrapContractError(f"{name} threshold was reselected")
    raw_counts = payload["query_counts"]
    if isinstance(raw_counts, (str, bytes)) or not isinstance(raw_counts, Sequence):
        raise TypeError(f"{name}.query_counts must be a sequence")
    if len(raw_counts) != QUERY_IMAGES_PER_WINDOW:
        raise Stage2BootstrapContractError(f"{name} must contain exactly 28 query images")
    counts = [
        _validate_count_row(row, name=f"{name}.query_counts[{index}]")
        for index, row in enumerate(raw_counts)
    ]
    ids = [row["image_id"] for row in counts]
    shas = [row["original_image_sha256"] for row in counts]
    if len(ids) != len(set(ids)) or len(shas) != len(set(shas)):
        raise Stage2BootstrapContractError(f"{name} has duplicate query identity")
    if _identity_sha(counts) != result["ordered_query_identity_sha256"]:
        raise Stage2BootstrapContractError(f"{name} query identity hash mismatch")
    common_identity = {
        "window_id": result["window_id"],
        "context_identity_sha256": result["context_identity_sha256"],
        "ordered_query_identity_sha256": result["ordered_query_identity_sha256"],
    }
    if sha256_bytes(canonical_json_bytes(common_identity)) != result[
        "window_identity_sha256"
    ]:
        raise Stage2BootstrapContractError(f"{name} window identity hash mismatch")
    result["query_counts"] = counts
    return result


def _validate_method_cell(
    payload: Mapping[str, Any], *, method_id: str, expected_windows: int, name: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    required = {
        "schema_version",
        "method_id",
        "detector_checkpoint_sha256",
        "method_checkpoint_sha256",
        "windows",
    }
    _assert_exact_keys(payload, required=required, name=name)
    if payload["schema_version"] != IMAGE_COUNTS_SCHEMA_VERSION:
        raise Stage2BootstrapContractError(f"{name} schema_version mismatch")
    if payload["method_id"] != method_id:
        raise Stage2BootstrapContractError(f"{name}.method_id mismatch")
    result = dict(payload)
    for field in ("detector_checkpoint_sha256", "method_checkpoint_sha256"):
        result[field] = validate_sha256(payload[field], f"{name}.{field}")
    raw_windows = payload["windows"]
    if isinstance(raw_windows, (str, bytes)) or not isinstance(raw_windows, Sequence):
        raise TypeError(f"{name}.windows must be a sequence")
    if len(raw_windows) != expected_windows:
        raise Stage2BootstrapContractError(
            f"{name} must contain exactly {expected_windows} windows"
        )
    result["windows"] = [
        _validate_method_window(window, method_id=method_id, name=f"{name}.windows[{i}]")
        for i, window in enumerate(raw_windows)
    ]
    return result


def validate_primary_pair_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete 3-domain × 3-seed × frozen-window T8/T4 pair."""

    if not isinstance(payload, Mapping):
        raise TypeError("pair manifest must be a mapping")
    required = {
        "schema_version",
        "source_thaw_sha256",
        "work_breakdown_sha256",
        "authorization_amendment_sha256",
        "comparison",
        "primary_budget",
        "threshold_semantics",
        "domain_weighting",
        "seed_weighting",
        "official_test_used",
        "domains",
    }
    _assert_exact_keys(payload, required=required, name="pair manifest")
    if payload["schema_version"] != PAIR_MANIFEST_SCHEMA_VERSION:
        raise Stage2BootstrapContractError("unsupported pair manifest schema_version")
    exact_bindings = {
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "authorization_amendment_sha256": AUTHORIZATION_AMENDMENT_SHA256,
    }
    for field, expected in exact_bindings.items():
        if validate_sha256(payload[field], f"pair.{field}") != expected:
            raise Stage2BootstrapContractError(f"pair manifest {field} mismatch")
    comparison = payload["comparison"]
    if not isinstance(comparison, Mapping):
        raise TypeError("pair comparison must be a mapping")
    _assert_exact_keys(
        comparison,
        required={"left_method", "right_method", "difference_order"},
        name="pair comparison",
    )
    if dict(comparison) != {
        "left_method": "T8",
        "right_method": "T4",
        "difference_order": "T8_minus_T4",
    }:
        raise Stage2BootstrapContractError("primary comparison must be T8 minus T4")
    if _finite_float(payload["primary_budget"], "primary_budget") != PRIMARY_BUDGET:
        raise Stage2BootstrapContractError("primary_budget must be exactly 1e-5")
    if payload["threshold_semantics"] != THRESHOLD_SEMANTICS:
        raise Stage2BootstrapContractError("threshold semantics mismatch")
    if payload["domain_weighting"] != "fixed_equal_one_third":
        raise Stage2BootstrapContractError("domains must have fixed equal 1/3 weight")
    if payload["seed_weighting"] != "equal_one_third_within_domain":
        raise Stage2BootstrapContractError("seed slots must have equal 1/3 weight")
    if _strict_bool(payload["official_test_used"], "official_test_used") is not False:
        raise Stage2BootstrapContractError("official test must remain sealed")
    raw_domains = payload["domains"]
    if isinstance(raw_domains, (str, bytes)) or not isinstance(raw_domains, Sequence):
        raise TypeError("pair domains must be a sequence")
    if len(raw_domains) != 3:
        raise Stage2BootstrapContractError("pair manifest must contain three domains")

    result_domains: list[dict[str, Any]] = []
    all_decisions: set[str] = set()
    for domain_index, (raw_domain, expected_domain) in enumerate(
        zip(raw_domains, DOMAIN_ORDER)
    ):
        name = f"domains[{domain_index}]"
        if not isinstance(raw_domain, Mapping):
            raise TypeError(f"{name} must be a mapping")
        _assert_exact_keys(
            raw_domain,
            required={"outer_fold_id", "target_dataset", "window_count", "cells"},
            name=name,
        )
        if raw_domain["outer_fold_id"] != expected_domain:
            raise Stage2BootstrapContractError(f"{name} domain order mismatch")
        if raw_domain["target_dataset"] != TARGET_BY_DOMAIN[expected_domain]:
            raise Stage2BootstrapContractError(f"{name} target_dataset mismatch")
        expected_windows = WINDOW_COUNT_BY_DOMAIN[expected_domain]
        if _strict_int(raw_domain["window_count"], f"{name}.window_count", minimum=1) != expected_windows:
            raise Stage2BootstrapContractError(f"{name} window_count mismatch")
        raw_cells = raw_domain["cells"]
        if isinstance(raw_cells, (str, bytes)) or not isinstance(raw_cells, Sequence):
            raise TypeError(f"{name}.cells must be a sequence")
        if len(raw_cells) != 3:
            raise Stage2BootstrapContractError(f"{name} must contain three seed cells")
        cells: list[dict[str, Any]] = []
        for cell_index, (raw_cell, expected_seed) in enumerate(
            zip(raw_cells, BASE_SEED_ORDER)
        ):
            cell_name = f"{name}.cells[{cell_index}]"
            if not isinstance(raw_cell, Mapping):
                raise TypeError(f"{cell_name} must be a mapping")
            _assert_exact_keys(
                raw_cell,
                required={"base_seed", "ordered_window_identity_sha256", "methods"},
                name=cell_name,
            )
            if _strict_int(raw_cell["base_seed"], f"{cell_name}.base_seed") != expected_seed:
                raise Stage2BootstrapContractError(f"{cell_name} seed order mismatch")
            common_windows_sha = validate_sha256(
                raw_cell["ordered_window_identity_sha256"],
                f"{cell_name}.ordered_window_identity_sha256",
            )
            raw_methods = raw_cell["methods"]
            if not isinstance(raw_methods, Mapping) or set(raw_methods) != set(METHOD_ORDER):
                raise Stage2BootstrapContractError(
                    f"{cell_name}.methods must contain exactly T8 and T4"
                )
            methods = {
                method_id: _validate_method_cell(
                    raw_methods[method_id],
                    method_id=method_id,
                    expected_windows=expected_windows,
                    name=f"{cell_name}.methods.{method_id}",
                )
                for method_id in METHOD_ORDER
            }
            left, right = methods["T8"], methods["T4"]
            if left["detector_checkpoint_sha256"] != right[
                "detector_checkpoint_sha256"
            ]:
                raise Stage2BootstrapContractError(f"{cell_name} detector mismatch")
            left_identities = [_window_identity(window) for window in left["windows"]]
            right_identities = [_window_identity(window) for window in right["windows"]]
            if left_identities != right_identities:
                raise Stage2BootstrapContractError(
                    f"{cell_name} T8/T4 window/context/query identities differ"
                )
            if sha256_bytes(canonical_json_bytes(left_identities)) != common_windows_sha:
                raise Stage2BootstrapContractError(f"{cell_name} window-list hash mismatch")
            if len({row["window_id"] for row in left_identities}) != expected_windows:
                raise Stage2BootstrapContractError(f"{cell_name} duplicate window IDs")
            total_background = 0
            total_gt = 0
            for left_window, right_window in zip(left["windows"], right["windows"]):
                left_counts = left_window["query_counts"]
                right_counts = right_window["query_counts"]
                for left_row, right_row in zip(left_counts, right_counts):
                    for field in (
                        "image_id",
                        "original_image_sha256",
                        "total_pixels",
                        "background_pixels",
                        "ground_truth_targets",
                    ):
                        if left_row[field] != right_row[field]:
                            raise Stage2BootstrapContractError(
                                f"{cell_name} T8/T4 paired geometry/count denominator mismatch"
                            )
                    total_background += int(left_row["background_pixels"])
                    total_gt += int(left_row["ground_truth_targets"])
                for method_window in (left_window, right_window):
                    decision = method_window["decision_sha256"]
                    if decision in all_decisions:
                        raise Stage2BootstrapContractError(
                            "decision SHA-256 must identify one unique method/window cell"
                        )
                    all_decisions.add(decision)
            if total_gt <= 0:
                raise Stage2BootstrapContractError(f"{cell_name} has zero GT objects")
            if PRIMARY_BUDGET * total_background < 20.0:
                raise Stage2BootstrapContractError(
                    f"{cell_name} is inestimable at the frozen low-FA endpoint"
                )
            cells.append(
                {
                    "base_seed": expected_seed,
                    "ordered_window_identity_sha256": common_windows_sha,
                    "methods": methods,
                }
            )
        result_domains.append(
            {
                "outer_fold_id": expected_domain,
                "target_dataset": TARGET_BY_DOMAIN[expected_domain],
                "window_count": expected_windows,
                "cells": cells,
            }
        )
    result = dict(payload)
    result["primary_budget"] = PRIMARY_BUDGET
    result["domains"] = result_domains
    return result


def extract_bootstrap_root_seeds(seed_manifest: Mapping[str, Any]) -> dict[str, dict[int, int]]:
    if not isinstance(seed_manifest, Mapping):
        raise TypeError("seed manifest must be a mapping")
    if seed_manifest.get("schema_version") != SEED_MANIFEST_SCHEMA_VERSION:
        raise Stage2BootstrapContractError("unsupported seed manifest schema_version")
    dimensions = seed_manifest.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise TypeError("seed dimensions must be a mapping")
    if dimensions.get("base_seeds") != list(BASE_SEED_ORDER):
        raise Stage2BootstrapContractError("seed manifest base-seed order mismatch")
    if dimensions.get("outer_folds") != list(DOMAIN_ORDER):
        raise Stage2BootstrapContractError("seed manifest domain order mismatch")
    raw_table = seed_manifest.get("derived_seed_table")
    if isinstance(raw_table, (str, bytes)) or not isinstance(raw_table, Sequence):
        raise TypeError("derived_seed_table must be a sequence")
    roots: dict[str, dict[int, int]] = {domain: {} for domain in DOMAIN_ORDER}
    for index, row in enumerate(raw_table):
        if not isinstance(row, Mapping):
            raise TypeError(f"seed row {index} must be a mapping")
        if not {"base_seed", "outer_fold_id", "derived_seeds_by_role"} <= set(row):
            raise Stage2BootstrapContractError(f"seed row {index} is incomplete")
        base_seed = _strict_int(row["base_seed"], f"seed row {index}.base_seed")
        outer_fold = row["outer_fold_id"]
        role_map = row["derived_seeds_by_role"]
        if outer_fold not in roots or base_seed not in BASE_SEED_ORDER:
            continue
        if not isinstance(role_map, Mapping) or BOOTSTRAP_ROLE not in role_map:
            raise Stage2BootstrapContractError(f"seed row {index} lacks bootstrap root")
        root = _strict_int(role_map[BOOTSTRAP_ROLE], f"seed row {index}.root", minimum=1)
        if base_seed in roots[str(outer_fold)]:
            raise Stage2BootstrapContractError("duplicate bootstrap root")
        roots[str(outer_fold)][base_seed] = root
    for domain in DOMAIN_ORDER:
        if set(roots[domain]) != set(BASE_SEED_ORDER):
            raise Stage2BootstrapContractError(f"missing bootstrap roots for {domain}")
    flat = [roots[d][s] for d in DOMAIN_ORDER for s in BASE_SEED_ORDER]
    if len(flat) != len(set(flat)):
        raise Stage2BootstrapContractError("bootstrap roots must be unique")
    return roots


_SEED_TAG_JSON = json.dumps(SEED_INDEX_TAG, ensure_ascii=False).encode("utf-8")
_WINDOW_TAG_JSON = json.dumps(WINDOW_INDEX_TAG, ensure_ascii=False).encode("utf-8")
_QUERY_TAG_JSON = json.dumps(QUERY_INDEX_TAG, ensure_ascii=False).encode("utf-8")


def _u64_mod(encoded_preimage: bytes, population_size: int) -> int:
    size = _strict_int(population_size, "population_size", minimum=1)
    value = int.from_bytes(
        hashlib.sha256(encoded_preimage).digest()[:8], "big", signed=False
    )
    return value % size


def stateless_seed_index(root_seed: int, replicate_index: int, slot_index: int) -> int:
    root = _strict_int(root_seed, "root_seed", minimum=1)
    replicate = _strict_int(replicate_index, "replicate_index")
    slot = _strict_int(slot_index, "slot_index")
    if slot >= 3:
        raise Stage2BootstrapContractError("slot_index must be in [0,2]")
    encoded = (
        b"[" + _SEED_TAG_JSON + b"," + str(root).encode("ascii")
        + b"," + str(replicate).encode("ascii") + b","
        + str(slot).encode("ascii") + b"]"
    )
    return _u64_mod(encoded, 3)


def stateless_window_index(
    selected_cell_root_seed: int,
    replicate_index: int,
    selector_slot_index: int,
    window_draw_slot: int,
    window_count: int,
) -> int:
    root = _strict_int(selected_cell_root_seed, "selected_cell_root_seed", minimum=1)
    replicate = _strict_int(replicate_index, "replicate_index")
    slot = _strict_int(selector_slot_index, "selector_slot_index")
    draw_slot = _strict_int(window_draw_slot, "window_draw_slot")
    count = _strict_int(window_count, "window_count", minimum=1)
    if slot >= 3 or draw_slot >= count:
        raise Stage2BootstrapContractError("window selector slot is out of range")
    encoded = (
        b"[" + _WINDOW_TAG_JSON + b"," + str(root).encode("ascii")
        + b"," + str(replicate).encode("ascii") + b","
        + str(slot).encode("ascii") + b","
        + str(draw_slot).encode("ascii") + b"]"
    )
    return _u64_mod(encoded, count)


def stateless_query_indices(
    selected_cell_root_seed: int,
    replicate_index: int,
    selector_slot_index: int,
    window_draw_slot: int,
    selected_window_id: str,
    query_size: int = QUERY_IMAGES_PER_WINDOW,
) -> tuple[int, ...]:
    root = _strict_int(selected_cell_root_seed, "selected_cell_root_seed", minimum=1)
    replicate = _strict_int(replicate_index, "replicate_index")
    slot = _strict_int(selector_slot_index, "selector_slot_index")
    draw_slot = _strict_int(window_draw_slot, "window_draw_slot")
    window_id = _nonempty_string(selected_window_id, "selected_window_id")
    size = _strict_int(query_size, "query_size", minimum=1)
    if slot >= 3 or size != QUERY_IMAGES_PER_WINDOW:
        raise Stage2BootstrapContractError("v2 query draw requires slot [0,2] and size 28")
    prefix = (
        b"[" + _QUERY_TAG_JSON + b"," + str(root).encode("ascii")
        + b"," + str(replicate).encode("ascii") + b","
        + str(slot).encode("ascii") + b","
        + str(draw_slot).encode("ascii") + b","
        + json.dumps(window_id, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b","
    )
    return tuple(
        _u64_mod(prefix + str(position).encode("ascii") + b"]", size)
        for position in range(size)
    )

def _common_geometry(pair: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "outer_fold_id": domain["outer_fold_id"],
            "target_dataset": domain["target_dataset"],
            "window_count": domain["window_count"],
            "cells": [
                {
                    "base_seed": cell["base_seed"],
                    "ordered_window_identity_sha256": cell[
                        "ordered_window_identity_sha256"
                    ],
                    "windows": [
                        _window_identity(window)
                        for window in cell["methods"]["T8"]["windows"]
                    ],
                }
                for cell in domain["cells"]
            ],
        }
        for domain in pair["domains"]
    ]


def generate_paired_hierarchical_indices(
    pair_manifest: Mapping[str, Any],
    seed_manifest: Mapping[str, Any],
    *,
    seed_manifest_sha256: str,
    resamples: int = PRIMARY_RESAMPLES,
) -> dict[str, Any]:
    pair = validate_primary_pair_manifest(pair_manifest)
    roots = extract_bootstrap_root_seeds(seed_manifest)
    seed_sha = validate_sha256(seed_manifest_sha256, "seed_manifest_sha256")
    if _strict_int(resamples, "resamples", minimum=1) != PRIMARY_RESAMPLES:
        raise Stage2BootstrapContractError("primary bootstrap requires exactly 10000 resamples")
    geometry = _common_geometry(pair)
    replicates: list[dict[str, Any]] = []
    for replicate_index in range(PRIMARY_RESAMPLES):
        domain_draws: list[dict[str, Any]] = []
        for domain, geometry_domain in zip(pair["domains"], geometry):
            outer_fold = domain["outer_fold_id"]
            window_count = int(domain["window_count"])
            seed_slots: list[dict[str, Any]] = []
            for selector_slot_index, selector_base_seed in enumerate(BASE_SEED_ORDER):
                selected_seed_index = stateless_seed_index(
                    roots[outer_fold][selector_base_seed],
                    replicate_index,
                    selector_slot_index,
                )
                selected_base_seed = BASE_SEED_ORDER[selected_seed_index]
                selected_cell = geometry_domain["cells"][selected_seed_index]
                windows: list[dict[str, Any]] = []
                for window_draw_slot in range(window_count):
                    selected_window_index = stateless_window_index(
                        roots[outer_fold][selected_base_seed],
                        replicate_index,
                        selector_slot_index,
                        window_draw_slot,
                        window_count,
                    )
                    selected_window = selected_cell["windows"][selected_window_index]
                    query_indices = stateless_query_indices(
                        roots[outer_fold][selected_base_seed],
                        replicate_index,
                        selector_slot_index,
                        window_draw_slot,
                        selected_window["window_id"],
                    )
                    windows.append(
                        {
                            "window_draw_slot": window_draw_slot,
                            "selected_window_index": selected_window_index,
                            "selected_window_id": selected_window["window_id"],
                            "query_indices": list(query_indices),
                        }
                    )
                seed_slots.append(
                    {
                        "selector_slot_index": selector_slot_index,
                        "selected_seed_index": selected_seed_index,
                        "selected_base_seed": selected_base_seed,
                        "windows": windows,
                    }
                )
            domain_draws.append(
                {"outer_fold_id": outer_fold, "seed_slots": seed_slots}
            )
        replicates.append(
            {"replicate_index": replicate_index, "domains": domain_draws}
        )
    return {
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "authorization_amendment_sha256": AUTHORIZATION_AMENDMENT_SHA256,
        "protocol_id": PROTOCOL_ID,
        "seed_manifest_sha256": seed_sha,
        "pairing_geometry_sha256": sha256_bytes(canonical_json_bytes(geometry)),
        "resample_count": PRIMARY_RESAMPLES,
        "domain_order": list(DOMAIN_ORDER),
        "base_seed_order": list(BASE_SEED_ORDER),
        "window_counts": [WINDOW_COUNT_BY_DOMAIN[d] for d in DOMAIN_ORDER],
        "query_images_per_window": QUERY_IMAGES_PER_WINDOW,
        "method_id_present_in_draw_preimages": False,
        "sealed_fields_present_in_draw_preimages": False,
        "seed_index_preimage_tag": SEED_INDEX_TAG,
        "window_index_preimage_tag": WINDOW_INDEX_TAG,
        "query_index_preimage_tag": QUERY_INDEX_TAG,
        "common_geometry": geometry,
        "replicates": replicates,
    }


@dataclass(frozen=True)
class _VerifiedBootstrapArtifacts:
    pair: dict[str, Any]
    indices: dict[str, Any]
    seed_manifest: dict[str, Any]
    pair_sha256: str
    index_sha256: str
    seed_sha256: str


def _verify_artifacts(
    pair_artifact: _VerifiedJsonArtifact,
    index_artifact: _VerifiedJsonArtifact,
    seed_artifact: _VerifiedJsonArtifact,
) -> _VerifiedBootstrapArtifacts:
    pair = validate_primary_pair_manifest(pair_artifact.payload)
    expected = generate_paired_hierarchical_indices(
        pair,
        seed_artifact.payload,
        seed_manifest_sha256=seed_artifact.sha256,
    )
    if canonical_json_bytes(index_artifact.payload) != canonical_json_bytes(expected):
        raise Stage2BootstrapContractError(
            "bootstrap indices fail complete stateless seed/window/query replay"
        )
    return _VerifiedBootstrapArtifacts(
        pair=pair,
        indices=expected,
        seed_manifest=seed_artifact.payload,
        pair_sha256=pair_artifact.sha256,
        index_sha256=index_artifact.sha256,
        seed_sha256=seed_artifact.sha256,
    )


def verify_paired_hierarchical_indices(
    pair_manifest_path: str | Path,
    expected_pair_sha256: str,
    index_manifest_path: str | Path,
    expected_index_sha256: str,
    seed_manifest_path: str | Path,
    expected_seed_sha256: str,
    *,
    repository_root: str | Path,
) -> _VerifiedBootstrapArtifacts:
    """Verify external bytes and replay all v2 draws before returning a token."""

    pair = _load_verified_json(
        pair_manifest_path, expected_pair_sha256, repository_root, name="pair_manifest"
    )
    indices = _load_verified_json(
        index_manifest_path, expected_index_sha256, repository_root, name="index_manifest"
    )
    seed = _load_verified_json(
        seed_manifest_path, expected_seed_sha256, repository_root, name="seed_manifest"
    )
    return _verify_artifacts(pair, indices, seed)


def indices_for_method(index_manifest: Mapping[str, Any], method_id: str) -> bytes:
    if method_id not in METHOD_ORDER:
        raise Stage2BootstrapContractError("only T8 and T4 consume primary indices")
    if index_manifest.get("schema_version") != INDEX_MANIFEST_SCHEMA_VERSION:
        raise Stage2BootstrapContractError("unsupported index manifest schema")
    return canonical_json_bytes(index_manifest.get("replicates"))


def type7_quantile(values: Sequence[float], probability: float) -> float:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise Stage2BootstrapContractError("type-7 quantile requires nonempty values")
    q = _finite_float(probability, "probability")
    if not 0.0 <= q <= 1.0:
        raise Stage2BootstrapContractError("probability must lie in [0,1]")
    ordered = sorted(_finite_float(value, "quantile value") for value in values)
    h = (len(ordered) - 1) * q
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (h - lower) * (ordered[upper] - ordered[lower])


def _window_sufficient_metrics(
    counts: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> tuple[float, float, int, int]:
    if len(indices) != QUERY_IMAGES_PER_WINDOW:
        raise Stage2BootstrapContractError("every selected window requires 28 query draws")
    fp = total = matched = gt = 0
    for raw_index in indices:
        index = _strict_int(raw_index, "query index")
        if index >= len(counts):
            raise Stage2BootstrapContractError("query index exceeds frozen window")
        row = counts[index]
        fp += int(row["false_positive_pixels"])
        total += int(row["total_pixels"])
        matched += int(row["matched_targets"])
        gt += int(row["ground_truth_targets"])
    if total <= 0:
        raise Stage2BootstrapContractError("zero total pixels makes primary pair missing")
    fa_pixel = fp / total
    satisfied = 1.0 if fa_pixel <= PRIMARY_BUDGET else 0.0
    log_excess = math.log(max(fa_pixel / PRIMARY_BUDGET, 1.0))
    return satisfied, log_excess, matched, gt


def _method_macro_metrics(
    pair: Mapping[str, Any], *, method_id: str, replicate: Mapping[str, Any] | None
) -> tuple[float, float, float]:
    domain_bsr: list[float] = []
    domain_log_excess: list[float] = []
    domain_pd: list[float] = []
    for domain_index, domain in enumerate(pair["domains"]):
        window_count = int(domain["window_count"])
        if replicate is None:
            slots = [
                {
                    "selected_seed_index": seed_index,
                    "windows": [
                        {
                            "selected_window_index": window_index,
                            "query_indices": list(range(QUERY_IMAGES_PER_WINDOW)),
                        }
                        for window_index in range(window_count)
                    ],
                }
                for seed_index in range(3)
            ]
        else:
            replicate_domains = replicate["domains"]
            if len(replicate_domains) != 3:
                raise Stage2BootstrapContractError("replicate domain list incomplete")
            draw_domain = replicate_domains[domain_index]
            if draw_domain["outer_fold_id"] != domain["outer_fold_id"]:
                raise Stage2BootstrapContractError("replicate domain order changed")
            slots = draw_domain["seed_slots"]
            if len(slots) != 3:
                raise Stage2BootstrapContractError("replicate seed slots incomplete")
        seed_bsr: list[float] = []
        seed_log: list[float] = []
        seed_pd: list[float] = []
        for slot in slots:
            selected_seed_index = _strict_int(
                slot["selected_seed_index"], "selected_seed_index"
            )
            if selected_seed_index >= 3:
                raise Stage2BootstrapContractError("selected seed index out of range")
            windows = slot["windows"]
            if len(windows) != window_count:
                raise Stage2BootstrapContractError("window draw count mismatch")
            window_bsr: list[float] = []
            window_log: list[float] = []
            pooled_matched = pooled_gt = 0
            method_windows = domain["cells"][selected_seed_index]["methods"][method_id][
                "windows"
            ]
            for window_draw in windows:
                selected_window_index = _strict_int(
                    window_draw["selected_window_index"], "selected_window_index"
                )
                if selected_window_index >= window_count:
                    raise Stage2BootstrapContractError("selected window index out of range")
                counts = method_windows[selected_window_index]["query_counts"]
                bsr, log_excess, matched, gt = _window_sufficient_metrics(
                    counts, window_draw["query_indices"]
                )
                window_bsr.append(bsr)
                window_log.append(log_excess)
                pooled_matched += matched
                pooled_gt += gt
            if pooled_gt <= 0:
                raise Stage2BootstrapContractError("zero GT replicate makes pair missing")
            seed_bsr.append(sum(window_bsr) / window_count)
            seed_log.append(sum(window_log) / window_count)
            seed_pd.append(pooled_matched / pooled_gt)
        domain_bsr.append(sum(seed_bsr) / 3.0)
        domain_log_excess.append(sum(seed_log) / 3.0)
        domain_pd.append(sum(seed_pd) / 3.0)
    return (
        sum(domain_bsr) / 3.0,
        sum(domain_log_excess) / 3.0,
        sum(domain_pd) / 3.0,
    )


def _evaluate_verified(verified: _VerifiedBootstrapArtifacts) -> dict[str, Any]:
    pair = verified.pair
    indices = verified.indices
    point_t8 = _method_macro_metrics(pair, method_id="T8", replicate=None)
    point_t4 = _method_macro_metrics(pair, method_id="T4", replicate=None)
    bsr_deltas: list[float] = []
    pd_deltas: list[float] = []
    for expected_index, replicate in enumerate(indices["replicates"]):
        if replicate["replicate_index"] != expected_index:
            raise Stage2BootstrapContractError("replicate sequence changed")
        t8 = _method_macro_metrics(pair, method_id="T8", replicate=replicate)
        t4 = _method_macro_metrics(pair, method_id="T4", replicate=replicate)
        bsr_deltas.append(t8[0] - t4[0])
        pd_deltas.append(t8[2] - t4[2])
    decisions = [
        {
            "outer_fold_id": domain["outer_fold_id"],
            "base_seed": cell["base_seed"],
            "window_id": t8_window["window_id"],
            "T8_decision_sha256": t8_window["decision_sha256"],
            "T4_decision_sha256": t4_window["decision_sha256"],
            "T8_threshold": t8_window["threshold"],
            "T4_threshold": t4_window["threshold"],
            "context_identity_sha256": t8_window["context_identity_sha256"],
        }
        for domain in pair["domains"]
        for cell in domain["cells"]
        for t8_window, t4_window in zip(
            cell["methods"]["T8"]["windows"],
            cell["methods"]["T4"]["windows"],
        )
    ]
    replicate_hash = sha256_bytes(
        canonical_json_bytes({"bsr": bsr_deltas, "pd": pd_deltas})
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_thaw_sha256": SOURCE_THAW_SHA256,
        "work_breakdown_sha256": WORK_BREAKDOWN_SHA256,
        "authorization_amendment_sha256": AUTHORIZATION_AMENDMENT_SHA256,
        "protocol_id": PROTOCOL_ID,
        "pair_manifest_sha256": verified.pair_sha256,
        "bootstrap_index_manifest_sha256": verified.index_sha256,
        "seed_manifest_sha256": verified.seed_sha256,
        "comparison": "T8_minus_T4",
        "primary_budget": PRIMARY_BUDGET,
        "resample_count": PRIMARY_RESAMPLES,
        "domain_weighting": "fixed_equal_one_third",
        "seed_weighting": "equal_one_third_within_domain",
        "window_counts": {
            "nuaa-sirst": 1,
            "nudt-sirst": 3,
            "irstd-1k": 3,
        },
        "query_images_per_window": QUERY_IMAGES_PER_WINDOW,
        "point_estimate_uses_every_original_unit_once": True,
        "point_estimate": {
            "T8_macro_bsr": point_t8[0],
            "T4_macro_bsr": point_t4[0],
            "T8_minus_T4_macro_bsr": point_t8[0] - point_t4[0],
            "T8_macro_log_excess": point_t8[1],
            "T4_macro_log_excess": point_t4[1],
            "T8_macro_pd": point_t8[2],
            "T4_macro_pd": point_t4[2],
            "T8_minus_T4_macro_pd": point_t8[2] - point_t4[2],
        },
        "confidence_interval": {
            "confidence_level": 0.95,
            "method": "two_sided_percentile_hyndman_fan_type_7",
            "quantiles": list(CI_QUANTILES),
            "T8_minus_T4_macro_bsr": [
                type7_quantile(bsr_deltas, CI_QUANTILES[0]),
                type7_quantile(bsr_deltas, CI_QUANTILES[1]),
            ],
            "T8_minus_T4_macro_pd": [
                type7_quantile(pd_deltas, CI_QUANTILES[0]),
                type7_quantile(pd_deltas, CI_QUANTILES[1]),
            ],
        },
        "replicate_differences_sha256": replicate_hash,
        "method_agnostic_three_level_indices": True,
        "T8_T4_index_bytes_identical": (
            indices_for_method(indices, "T8") == indices_for_method(indices, "T4")
        ),
        "fixed_decision_bindings_sha256": sha256_bytes(
            canonical_json_bytes(decisions)
        ),
        "sealed_decisions_fixed": True,
        "fa_pixel_denominator": "all_native_resolution_pixels",
        "bsr_aggregation": "equal_window_then_equal_seed_then_equal_domain",
        "pd_aggregation": "pooled_within_seed_then_equal_seed_then_equal_domain",
        "nuaa_one_window_degeneracy": {
            "window_resampling_deterministic": True,
            "between_window_variance_estimable": False,
            "domain_weight_remains_one_third": True,
            "alternate_context_synthesized_or_replaced": False,
        },
        "missing_primary_pair_count": 0,
        "replicate_deletion_count": 0,
        "imputation_used": False,
        "official_test_used": False,
    }


def evaluate_paired_bootstrap(
    pair_manifest_path: str | Path,
    expected_pair_sha256: str,
    index_manifest_path: str | Path,
    expected_index_sha256: str,
    seed_manifest_path: str | Path,
    expected_seed_sha256: str,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Verify external artifact bytes, fully replay v2 indices, then evaluate."""

    verified = verify_paired_hierarchical_indices(
        pair_manifest_path,
        expected_pair_sha256,
        index_manifest_path,
        expected_index_sha256,
        seed_manifest_path,
        expected_seed_sha256,
        repository_root=repository_root,
    )
    return _evaluate_verified(verified)


def _transactional_publish_bundle(files: Mapping[Path, bytes]) -> None:
    """Publish an all-new multi-file bundle; ordinary failure leaves no member."""

    if not files:
        raise Stage2BootstrapContractError("empty output bundle")
    targets = list(files)
    if len(set(targets)) != len(targets):
        raise Stage2BootstrapContractError("duplicate bundle target")
    parent = targets[0].parent
    if any(path.parent != parent for path in targets):
        raise Stage2BootstrapContractError("bundle targets must share one directory")
    if parent.is_symlink() or parent.resolve(strict=True) != parent:
        raise Stage2BootstrapContractError("bundle parent must be canonical")
    for target in targets:
        if not target.is_absolute() or ".." in target.parts:
            raise Stage2BootstrapContractError("bundle target must be absolute canonical")
        try:
            os.lstat(target)
        except FileNotFoundError:
            pass
        else:
            raise Stage2BootstrapContractError(
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
            raise Stage2BootstrapContractError(
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
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--pair-manifest-sha256", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--seed-manifest-sha256", required=True)
    parser.add_argument("--resamples", type=int, default=PRIMARY_RESAMPLES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _canonical_repository_root(args.repository_root)
    pair_artifact = _load_verified_json(
        args.pair_manifest,
        args.pair_manifest_sha256,
        root,
        name="pair_manifest",
    )
    seed_artifact = _load_verified_json(
        args.seed_manifest,
        args.seed_manifest_sha256,
        root,
        name="seed_manifest",
    )
    pair = validate_primary_pair_manifest(pair_artifact.payload)
    indices = generate_paired_hierarchical_indices(
        pair,
        seed_artifact.payload,
        seed_manifest_sha256=seed_artifact.sha256,
        resamples=args.resamples,
    )
    index_data = _json_file_bytes(indices)
    index_artifact = _VerifiedJsonArtifact(
        path=Path("<staged-index>"),
        payload=indices,
        data=index_data,
        sha256=sha256_bytes(index_data),
    )
    verified = _verify_artifacts(pair_artifact, index_artifact, seed_artifact)
    report = _evaluate_verified(verified)
    report_data = _json_file_bytes(report)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute() or output_dir.is_symlink():
        raise Stage2BootstrapContractError("output-dir must be absolute and non-symlink")
    if output_dir.resolve(strict=True) != output_dir or not output_dir.is_relative_to(root):
        raise Stage2BootstrapContractError("output-dir must be canonical inside repository")
    index_path = output_dir / "bootstrap_indices.json"
    report_path = output_dir / "paired_bootstrap_report.json"
    index_sidecar = index_path.with_name(index_path.name + ".sha256")
    report_sidecar = report_path.with_name(report_path.name + ".sha256")
    report_sha = sha256_bytes(report_data)
    bundle = {
        index_path: index_data,
        index_sidecar: f"{index_artifact.sha256}  {index_path.name}\n".encode("utf-8"),
        report_path: report_data,
        report_sidecar: f"{report_sha}  {report_path.name}\n".encode("utf-8"),
    }
    _transactional_publish_bundle(bundle)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
