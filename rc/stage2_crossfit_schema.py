"""Fail-closed Stage-2 context-package and cross-fit episode-v5 contracts.

The module is additive: legacy :mod:`rc.schema` remains the sole owner of
episode-v3/v4.  A Stage-2 context package is deliberately label-blind and is
the only object that may precede an outer-target decision seal.  A complete
v5 episode is produced only after the independently verified W05 label/curve
bundle exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image

from data_ext.stage2_label_attachment import (
    QUERY_BEARING_ROLES,
    VerifiedStage2LabelAttachment,
    VerifiedStage2Window,
    canonical_json_sha256 as w05_canonical_json_sha256,
    stage2_ordered_query_identity,
    verify_stage2_label_attachment,
    verify_stage2_window_contract,
)
from data_ext.stage2_score_manifest import (
    BINDING_NAMES,
    OOF_HOLDOUT_STAGE2_FIT,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    SOURCE_DIAGNOSTIC_VALIDATION,
    STRICT_THRESHOLD_SEMANTICS,
    VerifiedStage2ScoreManifest,
    verify_stage2_score_manifest,
)
from evaluation.stage2_threshold_sweep import (
    Stage2QueryCurve,
    verify_stage2_query_curve_artifacts,
)
from rc.build_stage2_source_reference import (
    VerifiedStage2SourceReference,
    verify_stage2_source_reference,
)
from rc.domain_statistics import FEATURE_DIM, FEATURE_NAMES, extract_unlabeled_statistics
from rc.schema import StatisticsConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CONTEXT_PACKAGE_SCHEMA = "rc-irstd.stage2-context-package.v1"
CONTEXT_PACKAGE_COMMIT_SCHEMA = "rc-irstd.stage2-context-package-commit.v1"
COLLECTION_SPEC_SCHEMA = "rc-irstd.stage2-crossfit-collection-spec.v1"
EPISODE_SCHEMA = "rc-irstd.meta-episode.v5"
COLLECTION_SCHEMA = "rc-irstd.meta-episode-collection.v5"
COLLECTION_COMMIT_SCHEMA = "rc-irstd.meta-episode-collection-commit.v1"

CONTEXT_ARTIFACT_TYPE = "rc_irstd_stage2_unlabeled_context_package"
EPISODE_ARTIFACT_TYPE = "rc_irstd_stage2_crossfit_meta_episode"
COLLECTION_ARTIFACT_TYPE = "rc_irstd_stage2_crossfit_episode_collection"

FULL_IDENTITY_ALGORITHM = "sha256-canonical-json-stage2-v5-full-identity-v1"
BOOTSTRAP_QUERY_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-stage2-bootstrap-query-image-identity-v1"
)
BOOTSTRAP_WINDOW_IDENTITY_ALGORITHM = (
    "sha256-canonical-json-stage2-bootstrap-window-identity-v1"
)
FLOAT32_VECTOR_ALGORITHM = "sha256-little-endian-float32-c-order-v1"
RECORD_HASH_ALGORITHM = "sha256-canonical-json-stage2-v5-record-v1"
ORDERED_RECORD_HASH_ALGORITHM = (
    "sha256-canonical-json-ordered-stage2-v5-record-digests-v1"
)

STAGE2_OOF_FIT = "stage2_oof_fit"
COLLECTION_TRAIN = "stage2_crossfit_training"
COLLECTION_VALIDATION = "stage2_source_checkpoint_validation"
COLLECTION_OUTER = "stage2_outer_target_development_evaluation"

ROLE_TO_EPISODE = {
    OOF_HOLDOUT_STAGE2_FIT: STAGE2_OOF_FIT,
    SOURCE_DIAGNOSTIC_VALIDATION: SOURCE_DIAGNOSTIC_VALIDATION,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT: OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
}
ROLE_TO_COLLECTION = {
    OOF_HOLDOUT_STAGE2_FIT: COLLECTION_TRAIN,
    SOURCE_DIAGNOSTIC_VALIDATION: COLLECTION_VALIDATION,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT: COLLECTION_OUTER,
}

OUTER_TARGETS = {
    "outer_leave_nuaa_sirst": "NUAA-SIRST",
    "outer_leave_nudt_sirst": "NUDT-SIRST",
    "outer_leave_irstd_1k": "IRSTD-1K",
}
ALL_DOMAINS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
BASE_SEEDS = (42, 123, 3407)
EXPECTED_COLLECTION_COUNTS = {
    COLLECTION_TRAIN: {
        "outer_leave_nuaa_sirst": 26,
        "outer_leave_nudt_sirst": 18,
        "outer_leave_irstd_1k": 16,
    },
    COLLECTION_VALIDATION: {
        "outer_leave_nuaa_sirst": 6,
        "outer_leave_nudt_sirst": 4,
        "outer_leave_irstd_1k": 4,
    },
    COLLECTION_OUTER: {
        "outer_leave_nuaa_sirst": 1,
        "outer_leave_nudt_sirst": 3,
        "outer_leave_irstd_1k": 3,
    },
}

GEOMETRY = {
    "block_size": 42,
    "construction": (
        "ordered_non_overlapping_contiguous_blocks_context_first_query_second"
    ),
    "context_size": 14,
    "query_size": 28,
}

B3_AUTHORIZATION_BINDING = {
    "path": (
        "outputs/stage2_protocol/"
        "RC4_STAGE2_B3_MODEL_DATA_AND_DEPENDENCY_RESOLUTION_"
        "AUTHORIZATION_20260717.json"
    ),
    "sha256": "d55b29dfb891710cf114d708b4c9f0c63d8d440d5efb107b81faf6b6e34bd1f6",
}
SEMANTICS_BINDING = {
    "path": (
        "outputs/stage2_protocol/"
        "RC4_STAGE2_PRE_G1_RESULT_FREE_ANALYSIS_PLAN_AMENDMENT_"
        "SEMANTICS_V1_20260716.json"
    ),
    "sha256": "c60e087116f98a3e59772792e16be389cc2961180b7a9c5de930e2b9cd9abef7",
}
GOVERNANCE_BINDINGS = {
    "b3_authorization": B3_AUTHORIZATION_BINDING,
    "stage2_semantics": SEMANTICS_BINDING,
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VERIFIED_CAPABILITY = object()

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
_CONTEXT_BINDING_FIELDS = frozenset(
    {"path", "sha256", "commit_path", "commit_sha256"}
)
_CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "development_only",
        "official_test_accessed",
        "path_anchor",
        "context_package_id",
        "expected_role",
        "episode_role",
        "outer_fold_id",
        "outer_target",
        "source_domain",
        "base_seed",
        "derived_seed",
        "detector_identity",
        "geometry",
        "window_binding",
        "score_manifest_binding",
        "score_bindings",
        "source_reference_binding",
        "statistics_config_binding",
        "extractor_binding",
        "partition_bindings",
        "seed_binding",
        "governance_bindings",
        "context_records",
        "query_identity_records",
        "context_full_identity_sha256",
        "source_query_full_identity_sha256",
        "source_ordered_query_identity_sha256",
        "context_statistics",
        "guardrails",
    }
)
_CONTEXT_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "artifact_status",
        "publication_complete",
        "official_test_accessed",
        "path_anchor",
        "context_package",
        "context_sidecar",
    }
)
_ROW_FIELDS = frozenset(
    {
        "ordinal",
        "partition",
        "score_record_index",
        "canonical_id",
        "image_id",
        "source_domain",
        "original_image_path",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
        "source_role",
        "outer_fold_id",
        "oof_fold_index",
        "score_file",
        "score_file_sha256",
        "original_hw",
        "input_hw",
        "resized_hw",
        "padding_ltrb",
        "resize_mode",
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
        "detector_identity",
        "geometry",
        "context_package_binding",
        "window_binding",
        "score_manifest_binding",
        "score_bindings",
        "source_reference_binding",
        "statistics_config_binding",
        "partition_bindings",
        "seed_binding",
        "governance_bindings",
        "context_records",
        "query_records",
        "context_full_identity_sha256",
        "source_query_full_identity_sha256",
        "source_ordered_query_identity_sha256",
        "context_statistics",
        "decision_seal_binding",
        "label_manifest_binding",
        "curve_binding",
        "supervision_contract",
        "guardrails",
    }
)


class Stage2CrossfitContractError(ValueError):
    """A Stage-2 context/episode artifact failed a frozen contract."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2CrossfitContractError(f"non-canonical JSON value: {error}") from error


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2CrossfitContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _nonfinite_guard(value: str) -> None:
    raise Stage2CrossfitContractError(f"non-finite JSON number: {value}")


def parse_json_bytes(data: bytes, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_guard,
            parse_constant=_nonfinite_guard,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2CrossfitContractError(f"invalid {name}: {error}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _exact_keys(value: Mapping[str, Any], fields: frozenset[str], name: str) -> None:
    if frozenset(value) != fields:
        raise Stage2CrossfitContractError(
            f"{name} field closure mismatch: missing={sorted(fields-set(value))}, "
            f"extra={sorted(set(value)-fields)}"
        )


def _exact_bool(value: object, name: str, expected: bool) -> None:
    if type(value) is not bool:  # noqa: E721 - exact JSON bool
        raise TypeError(f"{name} must be an exact JSON boolean")
    if value is not expected:
        raise Stage2CrossfitContractError(f"{name} must be {expected}")


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise Stage2CrossfitContractError(f"{name} must be >= {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TypeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _binding(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    _exact_keys(value, _BINDING_FIELDS, name)
    return {
        "path": _relative_path(value["path"], f"{name}.path"),
        "sha256": _sha256(value["sha256"], f"{name}.sha256"),
    }


def _relative_path(value: object, name: str) -> str:
    raw = _string(value, name)
    pure = PurePosixPath(raw)
    if pure.is_absolute() or raw != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise Stage2CrossfitContractError(f"{name} must be canonical repository-relative POSIX")
    lowered = raw.lower().replace("-", "_")
    if "official_test" in lowered or "officialtest" in lowered:
        raise Stage2CrossfitContractError(f"{name} may not reference official test")
    return raw


def repository_root(value: str | Path | None = None) -> Path:
    raw = (REPOSITORY_ROOT if value is None else Path(value).expanduser()).absolute()
    if raw.is_symlink() or not raw.is_dir() or raw.resolve(strict=True) != raw:
        raise Stage2CrossfitContractError("repository_root must be canonical and non-symlink")
    return raw


def _assert_no_symlink(path: Path, root: Path, name: str) -> None:
    if path != root and root not in path.parents:
        raise Stage2CrossfitContractError(f"{name} escapes repository root")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise Stage2CrossfitContractError(f"{name} contains a symlink component")


def direct_file(value: str | Path, root: Path, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    _assert_no_symlink(path, root, name)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise Stage2CrossfitContractError(f"{name} must be a regular file")
    return path


def bound_file(relative: object, root: Path, name: str) -> Path:
    normal = _relative_path(relative, name)
    return direct_file(root.joinpath(*PurePosixPath(normal).parts), root, name)


def repo_relative(path: Path, root: Path) -> str:
    absolute = path.absolute()
    if absolute != root and root not in absolute.parents:
        raise Stage2CrossfitContractError("path is outside repository root")
    return absolute.relative_to(root).as_posix()


def stable_read(path: Path, expected_sha256: str, name: str) -> bytes:
    expected = _sha256(expected_sha256, f"{name}.expected_sha256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2CrossfitContractError(f"{name} is not a regular file")
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
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if identity(before) != identity(after) or identity(before) != identity(current):
        raise RuntimeError(f"{name} changed during verified read")
    observed = digest.hexdigest()
    if observed != expected:
        raise Stage2CrossfitContractError(
            f"{name} SHA-256 mismatch: observed={observed}, expected={expected}"
        )
    return b"".join(chunks)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_governance(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, raw in GOVERNANCE_BINDINGS.items():
        path = bound_file(raw["path"], root, f"governance.{name}")
        stable_read(path, raw["sha256"], f"governance.{name}")
        result[name] = dict(raw)
    return result


def verify_stage2_statistics_config(
    path: str | Path,
    expected_sha256: str,
    *,
    repository_root: str | Path | None = None,
) -> StatisticsConfig:
    """Strictly load one externally SHA-bound Stage-2 statistics config."""

    root = globals()["repository_root"](repository_root)
    config_path = direct_file(path, root, "statistics config")
    digest = _sha256(expected_sha256, "statistics config SHA-256")
    payload = parse_json_bytes(
        stable_read(config_path, digest, "statistics config"),
        "statistics config",
    )
    _exact_keys(payload, _STATISTICS_CONFIG_FIELDS, "statistics config")
    try:
        config = StatisticsConfig.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise Stage2CrossfitContractError(
            f"invalid statistics config: {error}"
        ) from error
    if config.to_dict() != dict(payload):
        raise Stage2CrossfitContractError(
            "statistics config is not the exact frozen v3 representation"
        )
    if sha256_file(config_path) != digest:
        raise RuntimeError("statistics config changed after verified read")
    return config


def full_identity_projection(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    projection: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        projection.append(
            {
                "ordinal": index,
                "canonical_id": _string(row.get("canonical_id"), "canonical_id"),
                "image_id": _string(row.get("image_id"), "image_id"),
                "source_domain": _string(row.get("source_domain"), "source_domain"),
                "original_image_sha256": _sha256(
                    row.get("original_image_sha256"), "original_image_sha256"
                ),
                "near_duplicate_cluster_id_or_unique_sentinel": _string(
                    row.get("near_duplicate_cluster_id_or_unique_sentinel"),
                    "near_duplicate_cluster_id_or_unique_sentinel",
                ),
                "exclusion_group_id": _string(
                    row.get("exclusion_group_id"), "exclusion_group_id"
                ),
                "source_role_record_index": _integer(
                    row.get("source_role_record_index"), "source_role_record_index"
                ),
                "score_file_sha256": _sha256(
                    row.get("score_file_sha256"), "score_file_sha256"
                ),
                "original_hw": list(_hw(row.get("original_hw"), "original_hw")),
            }
        )
    return projection


def full_identity_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(full_identity_projection(records))


def bootstrap_query_identity_projection(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "image_id": _string(row.get("image_id"), "image_id"),
            "original_image_sha256": _sha256(
                row.get("original_image_sha256"), "original_image_sha256"
            ),
        }
        for row in records
    ]


def bootstrap_query_identity_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_sha256(bootstrap_query_identity_projection(records))


def bootstrap_window_identity_sha256(
    window_id: str,
    context_identity_sha256: str,
    bootstrap_query_sha256: str,
) -> str:
    return canonical_json_sha256(
        {
            "window_id": _string(window_id, "window_id"),
            "context_identity_sha256": _sha256(
                context_identity_sha256, "context_identity_sha256"
            ),
            "ordered_query_identity_sha256": _sha256(
                bootstrap_query_sha256, "bootstrap_query_sha256"
            ),
        }
    )


def _hw(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{name} must contain [height,width]")
    return (_integer(value[0], f"{name}[0]", minimum=1), _integer(value[1], f"{name}[1]", minimum=1))


def _record_from_score(
    score_record: Mapping[str, Any],
    window_record: Mapping[str, Any],
    *,
    ordinal: int,
    partition: str,
    score_payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "ordinal": ordinal,
        "partition": partition,
        "score_record_index": score_record["record_index"],
        "canonical_id": score_record["canonical_id"],
        "image_id": score_record["image_id"],
        "source_domain": score_record["source_domain"],
        "original_image_path": score_record["original_image_path"],
        "original_image_sha256": score_record["original_image_sha256"],
        "exclusion_group_id": score_record["exclusion_group_id"],
        "near_duplicate_cluster_id_or_unique_sentinel": score_record[
            "near_duplicate_cluster_id_or_unique_sentinel"
        ],
        "source_role_record_index": score_record["source_role_record_index"],
        "source_role": window_record["source_role"],
        "outer_fold_id": window_record["outer_fold_id"],
        "oof_fold_index": score_payload["oof_fold_index"],
        "score_file": score_record["score_file"],
        "score_file_sha256": score_record["score_file_sha256"],
        "original_hw": list(score_record["original_hw"]),
        "input_hw": list(score_record["input_hw"]),
        "resized_hw": list(score_record["resized_hw"]),
        "padding_ltrb": list(score_record["padding_ltrb"]),
        "resize_mode": score_record["resize_mode"],
    }
    _validate_row(result, partition, ordinal)
    for field in (
        "canonical_id",
        "image_id",
        "original_image_path",
        "original_image_sha256",
        "exclusion_group_id",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role_record_index",
    ):
        if score_record[field] != window_record[field]:
            raise Stage2CrossfitContractError(f"window/score identity mismatch at {field}")
    return result


def _validate_row(row: Mapping[str, Any], partition: str, ordinal: int) -> None:
    _exact_keys(row, _ROW_FIELDS, f"{partition}_records[{ordinal}]")
    if _integer(row["ordinal"], "ordinal") != ordinal:
        raise Stage2CrossfitContractError("row ordinal mismatch")
    if row["partition"] != partition:
        raise Stage2CrossfitContractError("row partition mismatch")
    for field in (
        "canonical_id", "image_id", "source_domain", "original_image_path",
        "exclusion_group_id", "near_duplicate_cluster_id_or_unique_sentinel",
        "source_role", "outer_fold_id", "score_file", "resize_mode",
    ):
        _string(row[field], field)
    for field in ("original_image_path", "score_file"):
        _relative_path(row[field], field)
    for field in ("original_image_sha256", "score_file_sha256"):
        _sha256(row[field], field)
    for field in ("score_record_index", "source_role_record_index"):
        _integer(row[field], field)
    if row["oof_fold_index"] is not None:
        value = _integer(row["oof_fold_index"], "oof_fold_index")
        if value not in {0, 1}:
            raise Stage2CrossfitContractError("oof_fold_index must be 0/1/null")
    _hw(row["original_hw"], "original_hw")
    _hw(row["input_hw"], "input_hw")
    _hw(row["resized_hw"], "resized_hw")
    padding = row["padding_ltrb"]
    if not isinstance(padding, (list, tuple)) or len(padding) != 4:
        raise TypeError("padding_ltrb must contain four integers")
    for item in padding:
        _integer(item, "padding_ltrb")


def _assert_four_boundary_disjoint(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], name: str
) -> None:
    for field in (
        "canonical_id",
        "original_image_sha256",
        "near_duplicate_cluster_id_or_unique_sentinel",
        "exclusion_group_id",
    ):
        overlap = {str(row[field]) for row in left} & {str(row[field]) for row in right}
        if overlap:
            raise Stage2CrossfitContractError(f"{name} overlap at {field}")


def _bind_window_score_reference(
    window: VerifiedStage2Window,
    score: VerifiedStage2ScoreManifest,
    reference: VerifiedStage2SourceReference,
) -> None:
    payload = score.payload
    expected = {
        "outer_fold_id": payload["outer_fold_id"],
        "outer_target_domain": payload["outer_target"],
        "domain": payload["source_domain"],
        "oof_fold_index": payload["oof_fold_index"],
    }
    for field, value in expected.items():
        if window.payload[field] != value:
            raise Stage2CrossfitContractError(f"window/score {field} mismatch")
    identity = reference.stage2_contract["detector_identity"]
    for field, score_field in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target", "outer_target"),
        ("base_seed", "base_seed"),
        ("derived_seed", "derived_seed"),
        ("detector_role", "detector_role"),
        ("oof_fold_index", "oof_fold_index"),
    ):
        if identity[field] != payload[score_field]:
            raise Stage2CrossfitContractError(f"source-reference detector {field} mismatch")
    checkpoint = score.bindings["checkpoint"]
    if identity["checkpoint_sha256"] != checkpoint["sha256"]:
        raise Stage2CrossfitContractError("source-reference checkpoint mismatch")
    consumers = reference.stage2_contract["bindings"]["consumer_window_manifests"]
    expected_consumer = {
        "path": repo_relative(window.path, score.repository_root),
        "sha256": window.manifest_sha256,
        "domain": payload["source_domain"],
        "episode_role": ROLE_TO_EPISODE[window.role],
    }
    matches = [
        item for item in consumers
        if all(item.get(key) == value for key, value in expected_consumer.items())
    ]
    if len(matches) != 1:
        raise Stage2CrossfitContractError("source-reference does not bind this consumer window")


def _read_score_probability(item: Any) -> np.ndarray:
    data = stable_read(item.score_path, item.record["score_file_sha256"], "context score")
    with np.load(io.BytesIO(data), allow_pickle=False) as payload:
        probability = np.asarray(payload["prob"])
    if probability.dtype != np.float64 or probability.shape != item.original_hw:
        raise Stage2CrossfitContractError("context score dtype/native geometry mismatch")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise Stage2CrossfitContractError("context score contains invalid probability")
    return probability


def _read_grayscale(item: Any) -> np.ndarray:
    data = stable_read(item.image_path, item.record["original_image_sha256"], "context image")
    with Image.open(io.BytesIO(data)) as image:
        gray = np.asarray(image.convert("L"))
    if gray.shape != item.original_hw:
        raise Stage2CrossfitContractError("context grayscale geometry mismatch")
    return gray


def build_context_payload(
    *,
    window_manifest: str | Path,
    window_manifest_sha256: str,
    window_id: str,
    expected_role: str,
    score_manifest: str | Path,
    score_manifest_sha256: str,
    source_reference: str | Path,
    source_reference_sha256: str,
    source_reference_audit_sha256: str,
    statistics_config: StatisticsConfig,
    repository_root_value: str | Path | None = None,
) -> tuple[dict[str, Any], VerifiedStage2Window, VerifiedStage2ScoreManifest, VerifiedStage2SourceReference]:
    """Build one label-blind C14/Q28 context payload from public verifiers."""

    root = repository_root(repository_root_value)
    role = _string(expected_role, "expected_role")
    if role not in QUERY_BEARING_ROLES:
        raise Stage2CrossfitContractError("unsupported context-package role")
    window = verify_stage2_window_contract(
        window_manifest, window_manifest_sha256, window_id, role, repository_root=root
    )
    score = verify_stage2_score_manifest(
        score_manifest, score_manifest_sha256, role, repository_root=root
    )
    reference = verify_stage2_source_reference(
        source_reference,
        source_reference_sha256,
        source_reference_audit_sha256,
        statistics_config=statistics_config,
        expected_consumer_window_path=window.path,
        expected_consumer_window_sha256=window.manifest_sha256,
        expected_consumer_window_id=window.window_id,
        repository_root=root,
    )
    if reference.statistics_config != statistics_config:
        raise Stage2CrossfitContractError("source-reference statistics config mismatch")
    _bind_window_score_reference(window, score, reference)
    if dict(window.payload["geometry"]) != GEOMETRY:
        raise Stage2CrossfitContractError("context package requires exact C14/Q28 geometry")

    by_id = {item.canonical_id: item for item in score.items}
    if len(by_id) != len(score.items):
        raise Stage2CrossfitContractError("score manifest has duplicate canonical IDs")
    context_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    context_items = []
    for partition, raw_records, output in (
        ("context", window.context_records, context_rows),
        ("query", window.query_records, query_rows),
    ):
        for ordinal, window_record in enumerate(raw_records):
            item = by_id.get(str(window_record["canonical_id"]))
            if item is None:
                raise Stage2CrossfitContractError("window identity absent from score manifest")
            output.append(
                _record_from_score(
                    item.record,
                    window_record,
                    ordinal=ordinal,
                    partition=partition,
                    score_payload=score.payload,
                )
            )
            if partition == "context":
                context_items.append(item)
    if len(context_rows) != 14 or len(query_rows) != 28:
        raise Stage2CrossfitContractError("context package requires exact C14/Q28")
    _assert_four_boundary_disjoint(context_rows, query_rows, "context/query")

    probabilities = [_read_score_probability(item) for item in context_items]
    grayscale = [_read_grayscale(item) for item in context_items]
    statistics = extract_unlabeled_statistics(
        probabilities,
        grayscale,
        source_reference=reference.source_reference,
        statistics_config=reference.statistics_config,
    )
    vector = np.asarray(statistics.vector, dtype=np.float32)
    if vector.shape != (FEATURE_DIM,) or FEATURE_DIM != 93:
        raise RuntimeError("frozen context feature dimension changed")
    vector_sha = hashlib.sha256(vector.astype("<f4", copy=False).tobytes()).hexdigest()
    context_sha = full_identity_sha256(context_rows)
    query_full_sha = full_identity_sha256(query_rows)
    w05_query_records = [
        {
            key: row[key]
            for key in (
                "canonical_id",
                "image_id",
                "original_image_sha256",
                "exclusion_group_id",
                "near_duplicate_cluster_id_or_unique_sentinel",
                "source_role_record_index",
            )
        }
        for row in query_rows
    ]
    source_query_sha = w05_canonical_json_sha256(
        stage2_ordered_query_identity(w05_query_records)
    )
    reference_binding = {
        "path": repo_relative(reference.path, root),
        "sha256": reference.npz_sha256,
        "audit_path": repo_relative(reference.audit_path, root),
        "audit_sha256": reference.audit_sha256,
        "reference_role": reference.stage2_contract["reference_role"],
    }
    stats_binding = dict(reference.stage2_contract["bindings"]["statistics_config"])
    extractor_path = Path(__file__).with_name("domain_statistics.py")
    extractor_binding = {
        "path": repo_relative(extractor_path, root),
        "sha256": sha256_file(extractor_path),
    }
    window_binding = {
        "path": repo_relative(window.path, root),
        "sha256": window.manifest_sha256,
        "window_id": window.window_id,
        "window_identity_sha256": window.window_identity_sha256,
    }
    score_binding = {
        "path": repo_relative(score.path, root),
        "sha256": score.manifest_sha256,
        "records_content_sha256": score.records_content_sha256,
        "role": score.role,
    }
    identity_preimage = {
        "schema_version": CONTEXT_PACKAGE_SCHEMA,
        "window_identity_sha256": window.window_identity_sha256,
        "score_manifest_sha256": score.manifest_sha256,
        "source_reference_sha256": reference.npz_sha256,
        "context_full_identity_sha256": context_sha,
        "source_query_full_identity_sha256": query_full_sha,
    }
    package_id = canonical_json_sha256(identity_preimage)
    payload: dict[str, Any] = {
        "schema_version": CONTEXT_PACKAGE_SCHEMA,
        "artifact_type": CONTEXT_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_UNLABELED",
        "development_only": True,
        "official_test_accessed": False,
        "path_anchor": "repository_root",
        "context_package_id": package_id,
        "expected_role": role,
        "episode_role": ROLE_TO_EPISODE[role],
        "outer_fold_id": score.payload["outer_fold_id"],
        "outer_target": score.payload["outer_target"],
        "source_domain": score.payload["source_domain"],
        "base_seed": score.payload["base_seed"],
        "derived_seed": score.payload["derived_seed"],
        "detector_identity": dict(reference.stage2_contract["detector_identity"]),
        "geometry": dict(GEOMETRY),
        "window_binding": window_binding,
        "score_manifest_binding": score_binding,
        "score_bindings": {name: dict(score.bindings[name]) for name in BINDING_NAMES},
        "source_reference_binding": reference_binding,
        "statistics_config_binding": stats_binding,
        "extractor_binding": extractor_binding,
        "partition_bindings": {
            "window_role_binding": dict(window.payload["role_binding"]),
            "window_unused_suffix": dict(window.payload["unused_suffix"]),
            "window_bound_inputs": {
                name: dict(value) for name, value in window.payload["bound_inputs"].items()
            },
            "score_selection_contract": dict(score.bindings["selection_contract"]),
        },
        "seed_binding": dict(score.bindings["seed_manifest"]),
        "governance_bindings": verify_governance(root),
        "context_records": context_rows,
        "query_identity_records": query_rows,
        "context_full_identity_sha256": context_sha,
        "source_query_full_identity_sha256": query_full_sha,
        "source_ordered_query_identity_sha256": source_query_sha,
        "context_statistics": {
            "feature_names": list(FEATURE_NAMES),
            "feature_dim": FEATURE_DIM,
            "dtype": "float32",
            "values": [float(value) for value in vector],
            "vector_sha256_algorithm": FLOAT32_VECTOR_ALGORITHM,
            "vector_sha256": vector_sha,
            "metadata": dict(statistics.metadata or {}),
        },
        "guardrails": {
            "context_labels_loaded": False,
            "query_labels_loaded": False,
            "mask_or_label_paths_resolved": False,
            "curve_artifacts_accessed": False,
            "official_test_accessed": False,
        },
    }
    validate_context_payload(payload)
    return payload, window, score, reference


def validate_context_payload(payload: Mapping[str, Any]) -> None:
    _exact_keys(payload, _CONTEXT_FIELDS, "context package")
    exact = {
        "schema_version": CONTEXT_PACKAGE_SCHEMA,
        "artifact_type": CONTEXT_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_UNLABELED",
        "path_anchor": "repository_root",
        "geometry": GEOMETRY,
    }
    for field, expected in exact.items():
        if payload[field] != expected:
            raise Stage2CrossfitContractError(f"context package {field} mismatch")
    _exact_bool(payload["development_only"], "development_only", True)
    _exact_bool(payload["official_test_accessed"], "official_test_accessed", False)
    role = payload["expected_role"]
    if role not in ROLE_TO_EPISODE or payload["episode_role"] != ROLE_TO_EPISODE[role]:
        raise Stage2CrossfitContractError("context role mapping mismatch")
    outer = _string(payload["outer_fold_id"], "outer_fold_id")
    if OUTER_TARGETS.get(outer) != payload["outer_target"]:
        raise Stage2CrossfitContractError("outer fold/target mismatch")
    if _integer(payload["base_seed"], "base_seed") not in BASE_SEEDS:
        raise Stage2CrossfitContractError("unsupported base seed")
    _integer(payload["derived_seed"], "derived_seed")
    context = payload["context_records"]
    query = payload["query_identity_records"]
    if not isinstance(context, list) or len(context) != 14:
        raise Stage2CrossfitContractError("context_records must be C14")
    if not isinstance(query, list) or len(query) != 28:
        raise Stage2CrossfitContractError("query_identity_records must be Q28")
    for index, row in enumerate(context):
        if not isinstance(row, Mapping):
            raise TypeError("context row must be an object")
        _validate_row(row, "context", index)
    for index, row in enumerate(query):
        if not isinstance(row, Mapping):
            raise TypeError("query row must be an object")
        _validate_row(row, "query", index)
    _assert_four_boundary_disjoint(context, query, "context/query")
    if payload["context_full_identity_sha256"] != full_identity_sha256(context):
        raise Stage2CrossfitContractError("context identity hash mismatch")
    if payload["source_query_full_identity_sha256"] != full_identity_sha256(query):
        raise Stage2CrossfitContractError("query full identity hash mismatch")
    w05_rows = [
        {key: row[key] for key in (
            "canonical_id", "image_id", "original_image_sha256",
            "exclusion_group_id", "near_duplicate_cluster_id_or_unique_sentinel",
            "source_role_record_index",
        )}
        for row in query
    ]
    expected_query_sha = w05_canonical_json_sha256(stage2_ordered_query_identity(w05_rows))
    if payload["source_ordered_query_identity_sha256"] != expected_query_sha:
        raise Stage2CrossfitContractError("W05 query identity hash mismatch")
    statistics = payload["context_statistics"]
    if not isinstance(statistics, Mapping) or set(statistics) != {
        "feature_names", "feature_dim", "dtype", "values",
        "vector_sha256_algorithm", "vector_sha256", "metadata",
    }:
        raise Stage2CrossfitContractError("context_statistics closure mismatch")
    if statistics["feature_names"] != list(FEATURE_NAMES) or statistics["feature_dim"] != 93:
        raise Stage2CrossfitContractError("context feature schema mismatch")
    if statistics["dtype"] != "float32" or statistics["vector_sha256_algorithm"] != FLOAT32_VECTOR_ALGORITHM:
        raise Stage2CrossfitContractError("context feature dtype/hash algorithm mismatch")
    values = np.asarray(statistics["values"], dtype=np.float32)
    if values.shape != (93,) or not np.isfinite(values).all():
        raise Stage2CrossfitContractError("context feature vector mismatch")
    digest = hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()
    if digest != _sha256(statistics["vector_sha256"], "vector_sha256"):
        raise Stage2CrossfitContractError("context vector SHA-256 mismatch")
    guardrails = payload["guardrails"]
    if not isinstance(guardrails, Mapping) or set(guardrails) != {
        "context_labels_loaded", "query_labels_loaded", "mask_or_label_paths_resolved",
        "curve_artifacts_accessed", "official_test_accessed",
    }:
        raise Stage2CrossfitContractError("context guardrails closure mismatch")
    for field in guardrails:
        _exact_bool(guardrails[field], f"guardrails.{field}", False)


@dataclass(frozen=True)
class VerifiedStage2ContextPackage:
    path: Path
    commit_path: Path
    payload: Mapping[str, Any]
    context_sha256: str
    commit_sha256: str
    window: VerifiedStage2Window
    score_manifest: VerifiedStage2ScoreManifest
    source_reference: VerifiedStage2SourceReference
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_CAPABILITY:
            raise TypeError("VerifiedStage2ContextPackage is verifier-only")


def context_commit_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.commit.json")


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _verify_sidecar(path: Path, artifact: Path, expected: str) -> None:
    data = stable_read(path, sha256_file(path), f"{artifact.name} sidecar")
    expected_bytes = f"{expected}  {artifact.name}\n".encode("ascii")
    if data != expected_bytes:
        raise Stage2CrossfitContractError(f"stale SHA sidecar for {artifact.name}")


def verify_stage2_context_package(
    path: str | Path,
    expected_sha256: str,
    expected_commit_sha256: str,
    *,
    statistics_config: StatisticsConfig,
    repository_root: str | Path | None = None,
) -> VerifiedStage2ContextPackage:
    root = globals()["repository_root"](repository_root)
    package_path = direct_file(path, root, "context package")
    lock = package_path.parent / f".{package_path.name}.lock"
    if os.path.lexists(lock):
        raise RuntimeError("context-package publication lock is present")
    package_sha = _sha256(expected_sha256, "expected context SHA")
    commit_sha = _sha256(expected_commit_sha256, "expected context commit SHA")
    package_data = stable_read(package_path, package_sha, "context package")
    commit = context_commit_path(package_path)
    commit_data = stable_read(commit, commit_sha, "context commit")
    _verify_sidecar(sidecar_path(package_path), package_path, package_sha)
    _verify_sidecar(sidecar_path(commit), commit, commit_sha)
    commit_payload = parse_json_bytes(commit_data, "context commit")
    _exact_keys(commit_payload, _CONTEXT_COMMIT_FIELDS, "context commit")
    if commit_payload["schema_version"] != CONTEXT_PACKAGE_COMMIT_SCHEMA:
        raise Stage2CrossfitContractError("context commit schema mismatch")
    if commit_payload["artifact_type"] != "rc_irstd_stage2_context_package_commit":
        raise Stage2CrossfitContractError("context commit artifact_type mismatch")
    if commit_payload["artifact_status"] != "COMPLETE":
        raise Stage2CrossfitContractError("context commit status mismatch")
    _exact_bool(commit_payload["publication_complete"], "publication_complete", True)
    _exact_bool(commit_payload["official_test_accessed"], "official_test_accessed", False)
    if commit_payload["path_anchor"] != "repository_root":
        raise Stage2CrossfitContractError("context commit path_anchor mismatch")
    expected_binding = {"path": repo_relative(package_path, root), "sha256": package_sha}
    expected_sidecar = {
        "path": repo_relative(sidecar_path(package_path), root),
        "sha256": sha256_file(sidecar_path(package_path)),
    }
    if commit_payload["context_package"] != expected_binding or commit_payload["context_sidecar"] != expected_sidecar:
        raise Stage2CrossfitContractError("context commit member binding mismatch")
    payload = parse_json_bytes(package_data, "context package")
    validate_context_payload(payload)
    rebuilt, window, score, reference = build_context_payload(
        window_manifest=bound_file(payload["window_binding"]["path"], root, "window binding"),
        window_manifest_sha256=payload["window_binding"]["sha256"],
        window_id=payload["window_binding"]["window_id"],
        expected_role=payload["expected_role"],
        score_manifest=bound_file(payload["score_manifest_binding"]["path"], root, "score binding"),
        score_manifest_sha256=payload["score_manifest_binding"]["sha256"],
        source_reference=bound_file(payload["source_reference_binding"]["path"], root, "source reference binding"),
        source_reference_sha256=payload["source_reference_binding"]["sha256"],
        source_reference_audit_sha256=payload["source_reference_binding"]["audit_sha256"],
        statistics_config=statistics_config,
        repository_root_value=root,
    )
    if rebuilt != dict(payload):
        raise Stage2CrossfitContractError("context package differs from full deterministic replay")
    if sha256_file(package_path) != package_sha or sha256_file(commit) != commit_sha:
        raise RuntimeError("context bundle changed after verification")
    return VerifiedStage2ContextPackage(
        path=package_path,
        commit_path=commit,
        payload=MappingProxyType(dict(payload)),
        context_sha256=package_sha,
        commit_sha256=commit_sha,
        window=window,
        score_manifest=score,
        source_reference=reference,
        _capability=_VERIFIED_CAPABILITY,
    )


def assert_verified_context_package(value: object) -> VerifiedStage2ContextPackage:
    if not isinstance(value, VerifiedStage2ContextPackage) or value._capability is not _VERIFIED_CAPABILITY:
        raise TypeError("a public-verifier-produced context package is required")
    return value


def make_stage2_shared_input_bindings(
    context_package: VerifiedStage2ContextPackage,
) -> dict[str, Any]:
    """Project a verified context package into the frozen W09 identity.

    The bridge accepts only the verifier-created capability.  Its constituent
    fields are projected again and W09 recomputes ``shared_input_identity``;
    no caller-supplied or copied identity digest is trusted.
    """

    context = assert_verified_context_package(context_package)
    root = context.score_manifest.repository_root
    source_query = w05_canonical_json_sha256(
        stage2_ordered_query_identity(context.window.query_records)
    )

    # Lazy import prevents a schema-layer import cycle.
    from evaluation.stage2_threshold_family import make_shared_input_bindings

    return make_shared_input_bindings(
        context_package_path=repo_relative(context.path, root),
        context_package_sha256=context.context_sha256,
        context_package_commit_path=repo_relative(context.commit_path, root),
        context_package_commit_sha256=context.commit_sha256,
        window_id=context.window.window_id,
        window_identity_sha256=context.window.window_identity_sha256,
        ordered_query_identity_sha256=source_query,
        score_manifest_sha256=context.score_manifest.manifest_sha256,
        score_records_content_sha256=context.score_manifest.records_content_sha256,
        detector_checkpoint_sha256=context.source_reference.stage2_contract[
            "detector_identity"
        ]["checkpoint_sha256"],
    )


@dataclass(frozen=True)
class Stage2CrossfitEpisode:
    """Structurally valid episode-v5 record; this alone is not verified capability."""

    payload: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Stage2CrossfitEpisode":
        validate_episode_payload(payload)
        return cls(MappingProxyType(dict(payload)))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json_bytes(self.payload).decode("utf-8"))

    @property
    def episode_id(self) -> str:
        return str(self.payload["episode_id"])


def validate_episode_payload(payload: Mapping[str, Any]) -> None:
    _exact_keys(payload, _EPISODE_FIELDS, "episode-v5")
    if payload["schema_version"] != EPISODE_SCHEMA or payload["artifact_type"] != EPISODE_ARTIFACT_TYPE:
        raise Stage2CrossfitContractError("episode-v5 schema/artifact mismatch")
    if payload["artifact_status"] != "DEVELOPMENT_ONLY_VERIFIED_COMPLETE":
        raise Stage2CrossfitContractError("episode-v5 status mismatch")
    _exact_bool(payload["development_only"], "development_only", True)
    _exact_bool(payload["official_test_accessed"], "official_test_accessed", False)
    _integer(payload["episode_index"], "episode_index")
    role = payload["episode_role"]
    if role not in set(ROLE_TO_EPISODE.values()):
        raise Stage2CrossfitContractError("episode_role mismatch")
    expected_collection = {
        value: ROLE_TO_COLLECTION[key] for key, value in ROLE_TO_EPISODE.items()
    }[role]
    if payload["collection_role"] != expected_collection:
        raise Stage2CrossfitContractError("episode collection_role mismatch")
    if payload["geometry"] != GEOMETRY:
        raise Stage2CrossfitContractError("episode geometry mismatch")
    context = payload["context_records"]
    query = payload["query_records"]
    if not isinstance(context, list) or len(context) != 14 or not isinstance(query, list) or len(query) != 28:
        raise Stage2CrossfitContractError("episode must contain C14/Q28")
    for index, row in enumerate(context):
        _validate_row(row, "context", index)
    for index, row in enumerate(query):
        _validate_row(row, "query", index)
    _assert_four_boundary_disjoint(context, query, "episode context/query")
    if payload["context_full_identity_sha256"] != full_identity_sha256(context):
        raise Stage2CrossfitContractError("episode context identity mismatch")
    if payload["source_query_full_identity_sha256"] != full_identity_sha256(query):
        raise Stage2CrossfitContractError("episode query identity mismatch")
    if role in {STAGE2_OOF_FIT, SOURCE_DIAGNOSTIC_VALIDATION}:
        if payload["decision_seal_binding"] is not None:
            raise Stage2CrossfitContractError("source episode decision seal must be null")
    elif not isinstance(payload["decision_seal_binding"], Mapping):
        raise Stage2CrossfitContractError("outer episode requires decision-set seal")
    guardrails = payload["guardrails"]
    if not isinstance(guardrails, Mapping) or set(guardrails) != {
        "context_labels_loaded", "query_labels_loaded", "official_test_accessed",
        "reject_supported", "fallback_used",
    }:
        raise Stage2CrossfitContractError("episode guardrail closure mismatch")
    _exact_bool(guardrails["context_labels_loaded"], "context_labels_loaded", False)
    _exact_bool(guardrails["query_labels_loaded"], "query_labels_loaded", True)
    for field in ("official_test_accessed", "reject_supported", "fallback_used"):
        _exact_bool(guardrails[field], field, False)


def _curve_binding(
    curve_path: Path,
    curve_sha: str,
    manifest_path: Path,
    manifest_sha: str,
    manifest: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    return {
        "path": repo_relative(curve_path, root),
        "sha256": curve_sha,
        "manifest_path": repo_relative(manifest_path, root),
        "manifest_sha256": manifest_sha,
        "rows_sha256": manifest["curve_rows_sha256"],
        "unique_event_count": manifest["num_unique_query_probability_events"],
        "operating_point_count": manifest["num_operating_points"],
        "total_native_pixels": manifest["total_native_pixels"],
        "gt_objects": manifest["gt_objects"],
    }


def build_episode_payload(
    *,
    episode_index: int,
    context_package: VerifiedStage2ContextPackage,
    label_manifest: str | Path,
    label_manifest_sha256: str,
    curve_file: str | Path,
    curve_file_sha256: str,
    curve_manifest: str | Path,
    curve_manifest_sha256: str,
    repository_root_value: str | Path | None = None,
) -> tuple[dict[str, Any], VerifiedStage2LabelAttachment, tuple[Mapping[str, Any], ...]]:
    root = repository_root(repository_root_value)
    context = assert_verified_context_package(context_package)
    cp = context.payload
    attachment = verify_stage2_label_attachment(
        context.score_manifest.path,
        label_manifest,
        cp["expected_role"],
        score_manifest_sha256=context.score_manifest.manifest_sha256,
        label_manifest_sha256=label_manifest_sha256,
        window_manifest=context.window.path,
        window_manifest_sha256=context.window.manifest_sha256,
        window_id=context.window.window_id,
        repository_root=root,
    )
    curve_path = direct_file(curve_file, root, "curve CSV")
    curve_manifest_path = direct_file(curve_manifest, root, "curve manifest")
    curve_payload, rows = verify_stage2_query_curve_artifacts(
        curve_path,
        curve_manifest_path,
        curve_sha256=curve_file_sha256,
        curve_manifest_sha256=curve_manifest_sha256,
        attachment=attachment,
        repository_root=root,
    )
    if attachment.ordered_query_identity_sha256 != cp["source_ordered_query_identity_sha256"]:
        raise Stage2CrossfitContractError("context/label query identity mismatch")
    decision_binding = attachment.payload["decision_seal_binding"]
    role = cp["episode_role"]
    if role in {STAGE2_OOF_FIT, SOURCE_DIAGNOSTIC_VALIDATION} and decision_binding is not None:
        raise Stage2CrossfitContractError("source labels must have null decision seal")
    if role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT and decision_binding is None:
        raise Stage2CrossfitContractError("outer labels require verified T0-T8 decision set")
    context_binding = {
        "path": repo_relative(context.path, root),
        "sha256": context.context_sha256,
        "commit_path": repo_relative(context.commit_path, root),
        "commit_sha256": context.commit_sha256,
    }
    label_path = attachment.path
    label_binding = {
        "path": repo_relative(label_path, root),
        "sha256": attachment.manifest_sha256,
        "labels_content_sha256": attachment.labels_content_sha256,
        "ordered_query_identity_sha256": attachment.ordered_query_identity_sha256,
    }
    episode_index_value = _integer(episode_index, "episode_index")
    identity_preimage = {
        "schema_version": EPISODE_SCHEMA,
        "context_package_sha256": context.context_sha256,
        "label_manifest_sha256": attachment.manifest_sha256,
        "curve_manifest_sha256": curve_manifest_sha256,
        "episode_index": episode_index_value,
    }
    episode_id = canonical_json_sha256(identity_preimage)
    payload = {
        "schema_version": EPISODE_SCHEMA,
        "artifact_type": EPISODE_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_VERIFIED_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "episode_id": episode_id,
        "episode_index": episode_index_value,
        "collection_role": ROLE_TO_COLLECTION[cp["expected_role"]],
        "episode_role": role,
        "outer_fold_id": cp["outer_fold_id"],
        "outer_target": cp["outer_target"],
        "source_domain": cp["source_domain"],
        "base_seed": cp["base_seed"],
        "derived_seed": cp["derived_seed"],
        "detector_identity": cp["detector_identity"],
        "geometry": cp["geometry"],
        "context_package_binding": context_binding,
        "window_binding": cp["window_binding"],
        "score_manifest_binding": cp["score_manifest_binding"],
        "score_bindings": cp["score_bindings"],
        "source_reference_binding": cp["source_reference_binding"],
        "statistics_config_binding": cp["statistics_config_binding"],
        "partition_bindings": cp["partition_bindings"],
        "seed_binding": cp["seed_binding"],
        "governance_bindings": cp["governance_bindings"],
        "context_records": cp["context_records"],
        "query_records": cp["query_identity_records"],
        "context_full_identity_sha256": cp["context_full_identity_sha256"],
        "source_query_full_identity_sha256": cp["source_query_full_identity_sha256"],
        "source_ordered_query_identity_sha256": cp["source_ordered_query_identity_sha256"],
        "context_statistics": cp["context_statistics"],
        "decision_seal_binding": decision_binding,
        "label_manifest_binding": label_binding,
        "curve_binding": _curve_binding(
            curve_path,
            curve_file_sha256,
            curve_manifest_path,
            curve_manifest_sha256,
            curve_payload,
            root,
        ),
        "supervision_contract": {
            "query_label_usage": (
                "outer_target_sealed_decision_evaluation_only"
                if role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT
                else "stage2_loss_and_exact_replay_only"
            ),
            "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
            "query_size": 28,
            "global_exact": True,
            "event_threshold_cap": None,
            "endpoints": [0.0, 1.0],
            "false_alarm_denominator": "all_native_resolution_query_pixels",
            "matching": "8_connected_maximum_cardinality_one_to_one_overlap",
        },
        "guardrails": {
            "context_labels_loaded": False,
            "query_labels_loaded": True,
            "official_test_accessed": False,
            "reject_supported": False,
            "fallback_used": False,
        },
    }
    validate_episode_payload(payload)
    return payload, attachment, rows


def record_sha256(payload: Mapping[str, Any]) -> str:
    validate_episode_payload(payload)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class VerifiedEpisodeArtifacts:
    context: VerifiedStage2ContextPackage
    attachment: VerifiedStage2LabelAttachment
    curve_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class VerifiedStage2EpisodeCollection(Sequence[Stage2CrossfitEpisode]):
    path: Path
    manifest_path: Path
    commit_path: Path
    episodes: tuple[Stage2CrossfitEpisode, ...]
    artifacts: tuple[VerifiedEpisodeArtifacts, ...]
    collection_sha256: str
    manifest_sha256: str
    commit_sha256: str
    manifest: Mapping[str, Any]
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_CAPABILITY:
            raise TypeError("VerifiedStage2EpisodeCollection is verifier-only")
        if len(self.episodes) != len(self.artifacts):
            raise ValueError("verified episode/artifact cardinality mismatch")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Stage2CrossfitEpisode:
        return self.episodes[index]


def assert_verified_episode_collection(value: object) -> VerifiedStage2EpisodeCollection:
    if not isinstance(value, VerifiedStage2EpisodeCollection) or value._capability is not _VERIFIED_CAPABILITY:
        raise TypeError("a complete public-verifier-produced v5 collection is required")
    return value


def collection_manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.collection.json")


def collection_commit_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.commit.json")


def _parse_jsonl(data: bytes) -> list[Mapping[str, Any]]:
    if not data or not data.endswith(b"\n") or b"\r" in data:
        raise Stage2CrossfitContractError("episode JSONL requires LF termination and no CR")
    lines = data.splitlines()
    if not lines or any(not line for line in lines):
        raise Stage2CrossfitContractError("episode JSONL contains an empty line")
    result = []
    for index, line in enumerate(lines):
        payload = parse_json_bytes(line, f"episode JSONL line {index}")
        if canonical_json_bytes(payload) != line:
            raise Stage2CrossfitContractError("episode JSONL line is not canonical JSON")
        result.append(payload)
    return result


def verify_episode_collection_completeness(episodes: Sequence[Stage2CrossfitEpisode]) -> None:
    if not episodes:
        raise Stage2CrossfitContractError("episode collection is empty")
    first = episodes[0].payload
    role = first["collection_role"]
    outer = first["outer_fold_id"]
    seed = first["base_seed"]
    expected = EXPECTED_COLLECTION_COUNTS.get(role, {}).get(outer)
    if expected is None or len(episodes) != expected:
        raise Stage2CrossfitContractError(
            f"collection count mismatch: observed={len(episodes)}, expected={expected}"
        )
    seen_ids: set[str] = set()
    seen_windows: set[tuple[str, str]] = set()
    domains: set[str] = set()
    for index, episode in enumerate(episodes):
        payload = episode.payload
        if payload["episode_index"] != index:
            raise Stage2CrossfitContractError("episode_index must be contiguous")
        if payload["collection_role"] != role or payload["outer_fold_id"] != outer or payload["base_seed"] != seed:
            raise Stage2CrossfitContractError("collection mixes role/fold/base seed")
        if episode.episode_id in seen_ids:
            raise Stage2CrossfitContractError("duplicate episode_id")
        seen_ids.add(episode.episode_id)
        window_key = (
            str(payload["window_binding"]["path"]),
            str(payload["window_binding"]["window_id"]),
        )
        if window_key in seen_windows:
            raise Stage2CrossfitContractError("duplicate selected window")
        seen_windows.add(window_key)
        domains.add(str(payload["source_domain"]))
    target = OUTER_TARGETS[outer]
    if role in {COLLECTION_TRAIN, COLLECTION_VALIDATION}:
        expected_domains = set(ALL_DOMAINS) - {target}
        if domains != expected_domains:
            raise Stage2CrossfitContractError("train/validation must contain both source domains")
    elif domains != {target}:
        raise Stage2CrossfitContractError("outer collection must contain only outer target")


def make_verified_collection(
    *,
    path: Path,
    manifest_path: Path,
    commit_path: Path,
    episodes: Sequence[Stage2CrossfitEpisode],
    artifacts: Sequence[VerifiedEpisodeArtifacts],
    collection_sha256: str,
    manifest_sha256: str,
    commit_sha256: str,
    manifest: Mapping[str, Any],
) -> VerifiedStage2EpisodeCollection:
    verify_episode_collection_completeness(episodes)
    return VerifiedStage2EpisodeCollection(
        path=path,
        manifest_path=manifest_path,
        commit_path=commit_path,
        episodes=tuple(episodes),
        artifacts=tuple(artifacts),
        collection_sha256=collection_sha256,
        manifest_sha256=manifest_sha256,
        commit_sha256=commit_sha256,
        manifest=MappingProxyType(dict(manifest)),
        _capability=_VERIFIED_CAPABILITY,
    )


__all__ = [
    "BASE_SEEDS",
    "BOOTSTRAP_QUERY_IDENTITY_ALGORITHM",
    "BOOTSTRAP_WINDOW_IDENTITY_ALGORITHM",
    "COLLECTION_COMMIT_SCHEMA",
    "COLLECTION_OUTER",
    "COLLECTION_SCHEMA",
    "COLLECTION_SPEC_SCHEMA",
    "COLLECTION_TRAIN",
    "COLLECTION_VALIDATION",
    "CONTEXT_PACKAGE_COMMIT_SCHEMA",
    "CONTEXT_PACKAGE_SCHEMA",
    "EPISODE_SCHEMA",
    "EXPECTED_COLLECTION_COUNTS",
    "FULL_IDENTITY_ALGORITHM",
    "GEOMETRY",
    "Stage2CrossfitContractError",
    "Stage2CrossfitEpisode",
    "VerifiedEpisodeArtifacts",
    "VerifiedStage2ContextPackage",
    "VerifiedStage2EpisodeCollection",
    "assert_verified_context_package",
    "assert_verified_episode_collection",
    "bootstrap_query_identity_projection",
    "bootstrap_query_identity_sha256",
    "bootstrap_window_identity_sha256",
    "build_context_payload",
    "build_episode_payload",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "collection_commit_path",
    "collection_manifest_path",
    "context_commit_path",
    "direct_file",
    "full_identity_projection",
    "full_identity_sha256",
    "make_stage2_shared_input_bindings",
    "make_verified_collection",
    "parse_json_bytes",
    "record_sha256",
    "repo_relative",
    "repository_root",
    "sha256_file",
    "sidecar_path",
    "stable_read",
    "validate_context_payload",
    "validate_episode_payload",
    "verify_episode_collection_completeness",
    "verify_governance",
    "verify_stage2_context_package",
    "verify_stage2_statistics_config",
]
