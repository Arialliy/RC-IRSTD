"""Label-blind RC5 Stage-2 context and exact-rational anchor producer.

The producer consumes a freshly replayed RC5 score bundle and a freshly
replayed RC5 source-reference-v3 authority.  It promotes exactly the fourteen
selected context score/image members to byte-verified content, extracts the
frozen 93-dimensional unlabeled statistic, and derives the T4 EATC anchor from
those same probability maps.  Query identities are bound as manifest
metadata; query score, image and label members are never opened here.

Four canonical JSON members are published without replacement in strict
``context -> anchor -> producer manifest -> commit last`` order.  A bare
context-v2 capability, including one built from arbitrary 93D values, is not
producer authority and cannot pass ``assert_verified_stage2_rc5_context_bundle``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from types import MappingProxyType
from typing import Any
import zipfile

import numpy as np
from PIL import Image

from data_ext import stage2_score_manifest as _score_v4
from data_ext.stage2_score_manifest_metadata_v5 import (
    VerifiedStage2ScoreManifestMetadataV5,
    assert_verified_stage2_score_manifest_metadata_v5,
    verify_stage2_score_manifest_metadata_v5,
)
from data_ext.stage2_rc5_score_bundle_v2 import (
    BUNDLE_CAPABILITY_SCHEMA as SCORE_BUNDLE_CAPABILITY_SCHEMA,
    VerifiedStage2RC5ScoreBundleV2,
    assert_verified_stage2_rc5_score_bundle_v2,
    replay_verified_stage2_rc5_score_bundle_v2,
)
from data_ext.stage2_variable_query_window import (
    VerifiedStage2VariableQueryWindow,
    assert_verified_stage2_variable_query_window,
    verify_stage2_variable_query_window,
)
from rc.domain_statistics import FEATURE_DIM, extract_unlabeled_statistics
from rc.schema import StatisticsConfig
from rc.stage2_context_tail_anchor import (
    VerifiedContextTailAnchor,
    assert_verified_context_tail_anchor,
    build_context_tail_anchor,
    verify_context_tail_anchor,
)
from rc.stage2_crossfit_schema import verify_stage2_statistics_config
from rc.stage2_crossfit_schema_v6 import (
    ROLE_TO_EPISODE,
    VerifiedStage2ContextV2,
    assert_verified_context_v2,
    context_from_verified_variable_query_window_v2,
    verify_context_payload_v2,
)
from rc.stage2_rc5_source_reference_v3 import (
    CAPABILITY_SCHEMA as SOURCE_REFERENCE_CAPABILITY_SCHEMA,
    VerifiedStage2RC5SourceReferenceV3,
    assert_verified_stage2_rc5_source_reference_v3,
    replay_verified_stage2_rc5_source_reference_v3,
)


PRODUCER_MANIFEST_SCHEMA = "rc-irstd.stage2-rc5-context-producer-manifest.v3"
COMMIT_SCHEMA = "rc-irstd.stage2-rc5-context-bundle-commit.v3"
BUNDLE_CAPABILITY_SCHEMA = "rc-irstd.stage2-rc5-context-bundle-capability.v3"
PRODUCER_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_label_blind_context_producer"
COMMIT_ARTIFACT_TYPE = "rc_irstd_stage2_rc5_context_bundle_commit"
PUBLICATION_ORDER = "context_then_anchor_then_producer_manifest_then_commit_last"

CONTEXT_FILENAME = "context.json"
ANCHOR_FILENAME = "context_tail_anchor.json"
PRODUCER_MANIFEST_FILENAME = "producer_manifest.json"
COMMIT_FILENAME = "context_bundle.commit.json"

_CONTEXT_SIZE = 14
_IDENTITY_FIELDS = (
    "canonical_id",
    "image_id",
    "original_image_path",
    "original_image_sha256",
    "exclusion_group_id",
    "near_duplicate_cluster_id_or_unique_sentinel",
    "source_role_record_index",
)
_FOUR_BOUNDARY_FIELDS = (
    "canonical_id",
    "original_image_sha256",
    "near_duplicate_cluster_id_or_unique_sentinel",
    "exclusion_group_id",
)
_CAPABILITY_TOKEN = object()


class Stage2RC5ContextProducerError(ValueError):
    """The RC5 pre-label producer or published bundle failed closed."""


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
        raise Stage2RC5ContextProducerError(
            f"non-canonical producer JSON: {error}"
        ) from error


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_value(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage2RC5ContextProducerError(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def _strict_json_object(data: bytes, name: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {raw}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Stage2RC5ContextProducerError(
            f"{name} is not strict JSON: {error}"
        ) from error
    if not isinstance(value, dict):
        raise Stage2RC5ContextProducerError(f"{name} must contain one object")
    if canonical_json_bytes(value) != data:
        raise Stage2RC5ContextProducerError(
            f"{name} bytes are not the exact canonical representation"
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _root(
    value: str | Path | None,
    window: VerifiedStage2VariableQueryWindow,
    score: VerifiedStage2ScoreManifestMetadataV5,
) -> Path:
    root = window.repository_root if value is None else Path(value).expanduser()
    if root.is_symlink():
        raise Stage2RC5ContextProducerError(
            "repository_root must not be a symlink"
        )
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise Stage2RC5ContextProducerError(
            "repository_root must be an existing directory"
        )
    if window.repository_root != root or score.repository_root != root:
        raise Stage2RC5ContextProducerError(
            "window and score capabilities must share repository_root"
        )
    return root


def _repo_relative(path: Path, root: Path, name: str) -> str:
    if path.is_symlink():
        raise Stage2RC5ContextProducerError(f"{name} must not be a symlink")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise Stage2RC5ContextProducerError(
            f"{name} is outside repository_root"
        ) from error
    rendered = relative.as_posix()
    if any(part in {"", ".", ".."} for part in PurePosixPath(rendered).parts):
        raise Stage2RC5ContextProducerError(f"{name} path is not canonical")
    return rendered


def _direct_existing_file(path: Path, root: Path, name: str) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise Stage2RC5ContextProducerError(
            f"{name} is outside repository_root"
        ) from error
    if not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise Stage2RC5ContextProducerError(
            f"{name} path is empty or contains an unsafe component"
        )
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as error:
            raise Stage2RC5ContextProducerError(
                f"{name} does not exist"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise Stage2RC5ContextProducerError(
                f"{name} contains a symlink component"
            )
    if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise Stage2RC5ContextProducerError(f"{name} is not a regular file")
    return path


def _stable_read_member(
    path: Path, expected_sha256: str, root: Path, name: str
) -> bytes:
    expected = _sha_value(expected_sha256, f"{name}.sha256")
    candidate = _direct_existing_file(path, root, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2RC5ContextProducerError(
                f"{name} is not a regular file"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = candidate.stat(follow_symlinks=False)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(current):
        raise RuntimeError(f"{name} changed during verified read")
    if digest.hexdigest() != expected:
        raise Stage2RC5ContextProducerError(f"{name} SHA-256 mismatch")
    return b"".join(chunks)


def _npz_string(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise Stage2RC5ContextProducerError(f"{name} must be one scalar string")
    result = str(array.item())
    if not result:
        raise Stage2RC5ContextProducerError(f"{name} must not be empty")
    return result


def _npz_int_tuple(value: np.ndarray, length: int, name: str) -> tuple[int, ...]:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in {"i", "u"}:
        raise Stage2RC5ContextProducerError(
            f"{name} must be an integer vector of length {length}"
        )
    return tuple(int(item) for item in array.tolist())


def _read_context_score_probability(
    item: Any, root: Path
) -> np.ndarray:
    """Promote and fully validate one context score member only."""

    record = item.record
    index = item.record_index
    data = _stable_read_member(
        item.score_path,
        record["score_file_sha256"],
        root,
        f"context score records[{index}]",
    )
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            members = tuple(info.filename for info in archive.infolist())
            if len(members) != len(set(members)) or members != (
                _score_v4.NPZ_ZIP_MEMBER_ORDER
            ):
                raise Stage2RC5ContextProducerError(
                    f"context score NPZ member closure mismatch at records[{index}]"
                )
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            if tuple(arrays.files) != _score_v4.NPZ_FIELD_ORDER:
                raise Stage2RC5ContextProducerError(
                    f"context score NPZ field closure mismatch at records[{index}]"
                )
            probability = np.asarray(arrays["prob"])
            raw_logit = np.asarray(arrays["raw_logit"])
            original_hw = _npz_int_tuple(
                arrays["original_hw"], 2, "NPZ original_hw"
            )
            if original_hw != item.original_hw:
                raise Stage2RC5ContextProducerError(
                    f"context score original_hw mismatch at records[{index}]"
                )
            if probability.shape != original_hw or raw_logit.shape != original_hw:
                raise Stage2RC5ContextProducerError(
                    f"context score is not native geometry at records[{index}]"
                )
            if probability.dtype != np.dtype("float64") or raw_logit.dtype != np.dtype(
                "float64"
            ):
                raise Stage2RC5ContextProducerError(
                    f"context score dtype mismatch at records[{index}]"
                )
            if not np.isfinite(probability).all() or not np.isfinite(raw_logit).all():
                raise Stage2RC5ContextProducerError(
                    f"context score contains NaN/Inf at records[{index}]"
                )
            if probability.size and (
                float(probability.min()) < 0.0
                or float(probability.max()) > 1.0
            ):
                raise Stage2RC5ContextProducerError(
                    f"context probability is outside [0,1] at records[{index}]"
                )
            for field in (
                "canonical_id",
                "image_id",
                "source_domain",
                "resize_mode",
            ):
                if _npz_string(arrays[field], f"NPZ {field}") != record[field]:
                    raise Stage2RC5ContextProducerError(
                        f"context score {field} mismatch at records[{index}]"
                    )
            for field, length in (
                ("input_hw", 2),
                ("resized_hw", 2),
                ("padding_ltrb", 4),
            ):
                if _npz_int_tuple(arrays[field], length, f"NPZ {field}") != tuple(
                    record[field]
                ):
                    raise Stage2RC5ContextProducerError(
                        f"context score {field} mismatch at records[{index}]"
                    )
            return probability.copy()
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, Stage2RC5ContextProducerError):
            raise
        raise Stage2RC5ContextProducerError(
            f"invalid context score NPZ at records[{index}]: {error}"
        ) from error


def _read_context_grayscale(item: Any, root: Path) -> np.ndarray:
    """Promote and validate one context original image only."""

    index = item.record_index
    data = _stable_read_member(
        item.image_path,
        item.record["original_image_sha256"],
        root,
        f"context image records[{index}]",
    )
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            gray = np.asarray(image.convert("L"))
    except (OSError, ValueError) as error:
        raise Stage2RC5ContextProducerError(
            f"invalid context image at records[{index}]: {error}"
        ) from error
    if gray.shape != item.original_hw:
        raise Stage2RC5ContextProducerError(
            f"context grayscale geometry mismatch at records[{index}]"
        )
    return gray.copy()


@dataclass(frozen=True)
class _PreparedInputs:
    root: Path
    window: VerifiedStage2VariableQueryWindow
    score_bundle: VerifiedStage2RC5ScoreBundleV2
    score: VerifiedStage2ScoreManifestMetadataV5
    source: VerifiedStage2RC5SourceReferenceV3
    statistics_config: StatisticsConfig
    statistics_config_path: Path
    statistics_config_sha256: str
    window_index: int
    raw_window: Mapping[str, Any]
    context_items: tuple[Any, ...]


def _prepare_inputs(
    *,
    variable_query_window: VerifiedStage2VariableQueryWindow,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    source_reference: VerifiedStage2RC5SourceReferenceV3,
    statistics_config: StatisticsConfig,
    statistics_config_path: str | Path,
    statistics_config_sha256: str,
    window_index: int,
    repository_root: str | Path | None,
) -> _PreparedInputs:
    window = assert_verified_stage2_variable_query_window(
        variable_query_window
    )
    supplied_score_bundle = assert_verified_stage2_rc5_score_bundle_v2(
        score_bundle
    )
    replayed_score_bundle = replay_verified_stage2_rc5_score_bundle_v2(
        supplied_score_bundle
    )
    score = replayed_score_bundle.score_manifest_metadata
    source = replay_verified_stage2_rc5_source_reference_v3(
        assert_verified_stage2_rc5_source_reference_v3(source_reference)
    )
    if not isinstance(statistics_config, StatisticsConfig):
        raise TypeError("statistics_config must be a StatisticsConfig")
    if type(window_index) is not int or window_index < 0:
        raise Stage2RC5ContextProducerError(
            "window_index must be one non-negative exact integer"
        )
    root = _root(repository_root, window, score)
    if source.repository_root != root:
        raise Stage2RC5ContextProducerError(
            "source-reference v3 and query score do not share repository_root"
        )
    source_score_rows = source.attestation["source_score_bundles"]
    if len(source_score_rows) != 2:
        raise Stage2RC5ContextProducerError(
            "source-reference v3 must bind exactly two source score attestations"
        )
    source_run_complete = _plain(source_score_rows[0]["run_complete"])
    if any(
        _plain(row["run_complete"]) != source_run_complete
        for row in source_score_rows[1:]
    ):
        raise Stage2RC5ContextProducerError(
            "source-reference v3 score attestations do not share RUN_COMPLETE"
        )
    query_run_complete = {
        "path": replayed_score_bundle.attestation["run_complete"]["path"],
        "sha256": replayed_score_bundle.run_complete.sha256,
        "identity_sha256": replayed_score_bundle.attestation[
            "run_complete"
        ]["identity"]["identity_sha256"],
    }
    if query_run_complete != source_run_complete:
        raise Stage2RC5ContextProducerError(
            "query and source score attestations do not share one RUN_COMPLETE"
        )
    replayed_window = verify_stage2_variable_query_window(
        window.path,
        window.manifest_sha256,
        repository_root=root,
    )
    if (
        replayed_window.path != window.path
        or replayed_window.manifest_sha256 != window.manifest_sha256
        or canonical_json_bytes(replayed_window.payload)
        != canonical_json_bytes(window.payload)
    ):
        raise Stage2RC5ContextProducerError(
            "variable-Q window capability differs from public replay"
        )
    replayed_score = verify_stage2_score_manifest_metadata_v5(
        score.path,
        score.manifest_sha256,
        score.role,
        repository_root=root,
    )
    if (
        replayed_score.path != score.path
        or replayed_score.manifest_sha256 != score.manifest_sha256
        or replayed_score.records_content_sha256
        != score.records_content_sha256
        or replayed_score.role != score.role
        or canonical_json_bytes(replayed_score.payload)
        != canonical_json_bytes(score.payload)
    ):
        raise Stage2RC5ContextProducerError(
            "score metadata capability differs from public replay"
        )
    if (
        replayed_score.manifest_sha256
        != replayed_score_bundle.score_manifest_metadata.manifest_sha256
        or replayed_score.records_content_sha256
        != replayed_score_bundle.score_manifest_metadata.records_content_sha256
    ):
        raise Stage2RC5ContextProducerError(
            "score bundle/metadata replay identity mismatch"
        )
    window = replayed_window
    score = replayed_score
    _stable_read_member(
        source.path,
        source.npz_sha256,
        root,
        "source-reference NPZ",
    )
    _stable_read_member(
        source.audit_path,
        source.audit_sha256,
        root,
        "source-reference audit",
    )
    for index, consumer in enumerate(source.consumer_windows):
        if consumer.repository_root != root:
            raise Stage2RC5ContextProducerError(
                f"source-reference consumer_windows[{index}] repository root mismatch"
            )
    if window_index >= len(window.windows):
        raise Stage2RC5ContextProducerError(
            "window_index exceeds the verified variable-Q manifest"
        )
    raw_window = window.windows[window_index]

    config_sha = _sha_value(
        statistics_config_sha256, "statistics_config_sha256"
    )
    config_path = Path(statistics_config_path).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    verified_config = verify_stage2_statistics_config(
        config_path, config_sha, repository_root=root
    )
    if verified_config != statistics_config:
        raise Stage2RC5ContextProducerError(
            "statistics-config capability/value mismatch"
        )
    if source.statistics_config != verified_config:
        raise Stage2RC5ContextProducerError(
            "source-reference statistics config mismatch"
        )
    source_config_binding = source.stage2_contract["bindings"][
        "statistics_config"
    ]
    expected_config_binding = {
        "path": _repo_relative(config_path.resolve(strict=True), root, "statistics config"),
        "sha256": config_sha,
    }
    if dict(source_config_binding) != expected_config_binding:
        raise Stage2RC5ContextProducerError(
            "source-reference statistics-config binding mismatch"
        )

    score_payload = score.payload
    window_payload = window.payload
    role = score.role
    if role not in ROLE_TO_EPISODE:
        raise Stage2RC5ContextProducerError(
            "metadata score role is not query-bearing"
        )
    for window_field, score_field in (
        ("outer_fold_id", "outer_fold_id"),
        ("outer_target_domain", "outer_target"),
        ("domain", "source_domain"),
        ("oof_fold_index", "oof_fold_index"),
    ):
        if window_payload[window_field] != score_payload[score_field]:
            raise Stage2RC5ContextProducerError(
                f"window/score {window_field} identity mismatch"
            )
    if window_payload["episode_role"] != ROLE_TO_EPISODE[role]:
        raise Stage2RC5ContextProducerError(
            "window/score episode role mismatch"
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
        if detector[detector_field] != score_payload[score_field]:
            raise Stage2RC5ContextProducerError(
                f"source-reference detector {detector_field} mismatch"
            )
    if dict(source.checkpoint_binding) != dict(score.bindings["checkpoint"]):
        raise Stage2RC5ContextProducerError(
            "source-reference/score checkpoint binding mismatch"
        )
    if detector["checkpoint_sha256"] != score.bindings["checkpoint"]["sha256"]:
        raise Stage2RC5ContextProducerError(
            "detector checkpoint identity mismatch"
        )

    selected_source_windows = [
        candidate
        for candidate in source.consumer_windows
        if candidate.path == window.path
        and candidate.manifest_sha256 == window.manifest_sha256
    ]
    if len(selected_source_windows) != 1:
        raise Stage2RC5ContextProducerError(
            "source-reference does not uniquely bind this variable-Q window"
        )

    if len(window.ordered_records) != len(score.records):
        raise Stage2RC5ContextProducerError(
            "window/score ordered record cardinality mismatch"
        )
    for index, (window_record, score_record) in enumerate(
        zip(window.ordered_records, score.records, strict=True)
    ):
        if score_record["record_index"] != index:
            raise Stage2RC5ContextProducerError(
                "score record indices are not exact manifest order"
            )
        for field in _IDENTITY_FIELDS:
            if window_record[field] != score_record[field]:
                raise Stage2RC5ContextProducerError(
                    f"window/score ordered identity mismatch at {field}"
                )

    by_id = {item.canonical_id: item for item in score.items}
    if len(by_id) != len(score.items):
        raise Stage2RC5ContextProducerError(
            "score metadata contains duplicate canonical IDs"
        )
    context_items = tuple(
        by_id[str(record["canonical_id"])]
        for record in raw_window["context_records"]
    )
    if len(context_items) != _CONTEXT_SIZE:
        raise Stage2RC5ContextProducerError(
            "RC5 context must contain exactly fourteen members"
        )
    query = tuple(raw_window["query_records"])
    if len(query) < 28:
        raise Stage2RC5ContextProducerError(
            "RC5 query must contain at least twenty-eight identities"
        )
    for field in _FOUR_BOUNDARY_FIELDS:
        if {item.record[field] for item in context_items}.intersection(
            record[field] for record in query
        ):
            raise Stage2RC5ContextProducerError(
                f"context/query identity overlap at {field}"
            )

    return _PreparedInputs(
        root=root,
        window=window,
        score_bundle=replayed_score_bundle,
        score=score,
        source=source,
        statistics_config=verified_config,
        statistics_config_path=config_path.resolve(strict=True),
        statistics_config_sha256=config_sha,
        window_index=window_index,
        raw_window=raw_window,
        context_items=context_items,
    )


def _produce_context_and_anchor(
    prepared: _PreparedInputs,
) -> tuple[VerifiedStage2ContextV2, VerifiedContextTailAnchor, tuple[np.ndarray, ...]]:
    probabilities = tuple(
        _read_context_score_probability(item, prepared.root)
        for item in prepared.context_items
    )
    grayscale = tuple(
        _read_context_grayscale(item, prepared.root)
        for item in prepared.context_items
    )
    if len(probabilities) != _CONTEXT_SIZE or len(grayscale) != _CONTEXT_SIZE:
        raise RuntimeError("internal context member cardinality changed")
    statistics = extract_unlabeled_statistics(
        probabilities,
        grayscale,
        source_reference=prepared.source.source_reference,
        statistics_config=prepared.statistics_config,
    )
    vector = np.asarray(statistics.vector, dtype=np.float32)
    if vector.shape != (FEATURE_DIM,) or FEATURE_DIM != 93:
        raise RuntimeError("frozen context feature dimension changed")
    if not np.isfinite(vector).all():
        raise Stage2RC5ContextProducerError(
            "context feature vector contains NaN/Inf"
        )
    context = context_from_verified_variable_query_window_v2(
        prepared.window,
        expected_role=prepared.score.role,
        base_seed=int(prepared.score.payload["base_seed"]),
        derived_seed=int(prepared.score.payload["derived_seed"]),
        window_index=prepared.window_index,
        context_feature_values=tuple(
            float(value) for value in vector.tolist()
        ),
    )
    anchor_payload = build_context_tail_anchor(
        context_probability_maps=probabilities,
        context_identity_sha256=str(
            context.payload["context_full_identity_sha256"]
        ),
    )
    anchor = verify_context_tail_anchor(
        anchor_payload,
        context_probability_maps=probabilities,
        expected_context_identity_sha256=str(
            context.payload["context_full_identity_sha256"]
        ),
    )
    return context, anchor, probabilities


def _input_bindings(prepared: _PreparedInputs) -> dict[str, Any]:
    source = prepared.source
    root = prepared.root
    score_bundle = prepared.score_bundle
    score_attestation = score_bundle.attestation
    source_attestation = source.attestation
    source_score_rows = source_attestation["source_score_bundles"]
    shared_run_complete = _plain(source_score_rows[0]["run_complete"])
    return {
        "variable_query_window": {
            "path": _repo_relative(prepared.window.path, root, "variable-Q window"),
            "sha256": prepared.window.manifest_sha256,
            "schema_version": prepared.window.payload["schema_version"],
            "window_id": prepared.raw_window["window_id"],
        },
        "score_manifest_metadata": {
            "path": _repo_relative(prepared.score.path, root, "score manifest"),
            "sha256": prepared.score.manifest_sha256,
            "records_content_sha256": prepared.score.records_content_sha256,
            "role": prepared.score.role,
            "capability_schema": prepared.score.capability_schema,
            "member_content_verified": False,
        },
        "score_bundle": {
            "path": _repo_relative(
                score_bundle.attestation_path,
                root,
                "RC5 score-bundle attestation",
            ),
            "sha256": score_bundle.attestation_sha256,
            "capability_schema": SCORE_BUNDLE_CAPABILITY_SCHEMA,
            "attestation_schema": score_attestation["schema_version"],
            "run_complete_path": score_attestation["run_complete"]["path"],
            "run_complete_sha256": score_bundle.run_complete.sha256,
            "run_complete_identity_sha256": score_attestation[
                "run_complete"
            ]["identity"]["identity_sha256"],
            "restricted_checkpoint_sha256": score_attestation[
                "restricted_checkpoint"
            ]["sha256"],
            "member_content_verified": False,
            "current_state_replayed": True,
        },
        "source_reference": {
            "attestation": {
                "path": _repo_relative(
                    source.attestation_path,
                    root,
                    "RC5 source-reference v3 attestation",
                ),
                "sha256": source.attestation_sha256,
                "schema_version": source_attestation["schema_version"],
                "capability_schema": SOURCE_REFERENCE_CAPABILITY_SCHEMA,
            },
            "base_reference": {
                "npz_path": _repo_relative(
                    source.path, root, "source reference"
                ),
                "npz_sha256": source.npz_sha256,
                "audit_path": _repo_relative(
                    source.audit_path, root, "source-reference audit"
                ),
                "audit_sha256": source.audit_sha256,
                "variable_query_capability_schema": source_attestation[
                    "source_reference"
                ]["variable_query_capability_schema"],
            },
            "reference_role": source.reference_role,
            "source_score_attestations": [
                {
                    "source_domain": row["source_domain"],
                    "path": row["score_attestation"]["path"],
                    "sha256": row["score_attestation"]["sha256"],
                    "capability_schema": row["score_attestation"][
                        "capability_schema"
                    ],
                }
                for row in source_score_rows
            ],
            "shared_run_complete": shared_run_complete,
            "restricted_checkpoint": _plain(
                source_score_rows[0]["restricted_checkpoint"]
            ),
            "current_state_replayed": True,
            "mixed_consumer_schemas_allowed": False,
        },
        "statistics_config": {
            "path": _repo_relative(
                prepared.statistics_config_path, root, "statistics config"
            ),
            "sha256": prepared.statistics_config_sha256,
        },
    }


def _producer_manifest(
    prepared: _PreparedInputs,
    context: VerifiedStage2ContextV2,
    anchor: VerifiedContextTailAnchor,
    *,
    context_path: Path,
    context_sha256: str,
    anchor_path: Path,
    anchor_sha256: str,
) -> dict[str, Any]:
    context = assert_verified_context_v2(context)
    anchor = assert_verified_context_tail_anchor(anchor)
    payload: dict[str, Any] = {
        "schema_version": PRODUCER_MANIFEST_SCHEMA,
        "artifact_type": PRODUCER_ARTIFACT_TYPE,
        "artifact_status": "DEVELOPMENT_ONLY_PRELABEL_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "producer_authority": "verified-stage2-rc5-label-blind-producer-v3",
        "expected_role": prepared.score.role,
        "episode_role": prepared.window.payload["episode_role"],
        "outer_fold_id": prepared.score.payload["outer_fold_id"],
        "outer_target": prepared.score.payload["outer_target"],
        "source_domain": prepared.score.payload["source_domain"],
        "base_seed": prepared.score.payload["base_seed"],
        "derived_seed": prepared.score.payload["derived_seed"],
        "detector_identity": _plain(prepared.source.detector_identity),
        "checkpoint_binding": _plain(prepared.source.checkpoint_binding),
        "geometry": _plain(prepared.window.payload["geometry"]),
        "window_index": prepared.window_index,
        "window_id": prepared.raw_window["window_id"],
        "context_size": _CONTEXT_SIZE,
        "query_size": prepared.raw_window["query_size"],
        "inputs": _input_bindings(prepared),
        "outputs": {
            "context": {
                "path": _repo_relative(context_path, prepared.root, "context output"),
                "sha256": context_sha256,
                "context_package_id": context.payload["context_package_id"],
                "context_full_identity_sha256": context.payload[
                    "context_full_identity_sha256"
                ],
                "query_full_identity_sha256": context.payload[
                    "query_full_identity_sha256"
                ],
                "window_identity_sha256": context.payload[
                    "window_identity_sha256"
                ],
                "feature_vector_sha256": context.payload[
                    "context_statistics"
                ]["vector_sha256"],
            },
            "anchor": {
                "path": _repo_relative(anchor_path, prepared.root, "anchor output"),
                "sha256": anchor_sha256,
                "anchor_identity_sha256": anchor.payload[
                    "anchor_identity_sha256"
                ],
                "context_identity_sha256": anchor.payload[
                    "context_identity_sha256"
                ],
                "context_probability_content_sha256": anchor.payload[
                    "context_probability_content_sha256"
                ],
            },
        },
        "access_audit": {
            "context_score_member_open_count": _CONTEXT_SIZE,
            "context_image_member_open_count": _CONTEXT_SIZE,
            "query_score_member_open_count": 0,
            "query_image_member_open_count": 0,
            "context_labels_accessed": False,
            "query_labels_accessed": False,
            "observed_results_accessed": False,
            "only_context_member_bytes_promoted": True,
        },
    }
    payload["producer_identity_sha256"] = _sha_bytes(canonical_json_bytes(payload))
    return payload


def _commit_payload(
    *,
    root: Path,
    context_path: Path,
    context_sha256: str,
    anchor_path: Path,
    anchor_sha256: str,
    producer_manifest_path: Path,
    producer_manifest_sha256: str,
    producer_identity_sha256: str,
) -> dict[str, Any]:
    members = {
        "context": {
            "path": _repo_relative(context_path, root, "context output"),
            "sha256": context_sha256,
        },
        "anchor": {
            "path": _repo_relative(anchor_path, root, "anchor output"),
            "sha256": anchor_sha256,
        },
        "producer_manifest": {
            "path": _repo_relative(
                producer_manifest_path, root, "producer manifest output"
            ),
            "sha256": producer_manifest_sha256,
        },
    }
    bundle_identity = _sha_bytes(
        canonical_json_bytes(
            {
                "schema_version": COMMIT_SCHEMA,
                "publication_order": PUBLICATION_ORDER,
                "producer_identity_sha256": producer_identity_sha256,
                "members": members,
            }
        )
    )
    payload: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "artifact_type": COMMIT_ARTIFACT_TYPE,
        "artifact_status": "complete",
        "development_only": True,
        "official_test_accessed": False,
        "publication_order": PUBLICATION_ORDER,
        "producer_identity_sha256": producer_identity_sha256,
        "bundle_identity_sha256": bundle_identity,
        "members": members,
    }
    payload["commit_identity_sha256"] = _sha_bytes(canonical_json_bytes(payload))
    return payload


def _bundle_paths(output_directory: Path) -> tuple[Path, Path, Path, Path]:
    return (
        output_directory / CONTEXT_FILENAME,
        output_directory / ANCHOR_FILENAME,
        output_directory / PRODUCER_MANIFEST_FILENAME,
        output_directory / COMMIT_FILENAME,
    )


def _output_directory(
    value: str | Path, root: Path
) -> Path:
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise Stage2RC5ContextProducerError(
            "output_directory must be inside repository_root"
        ) from error
    relative = directory.relative_to(root)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise Stage2RC5ContextProducerError(
            "output_directory contains an unsafe path component"
        )
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as error:
            raise Stage2RC5ContextProducerError(
                "output_directory does not exist"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise Stage2RC5ContextProducerError(
                "output_directory contains a symlink component"
            )
        if not stat.S_ISDIR(info.st_mode):
            raise Stage2RC5ContextProducerError(
                "output_directory contains a non-directory component"
            )
    directory = directory.resolve(strict=True)
    if not directory.is_dir():
        raise Stage2RC5ContextProducerError(
            "output_directory must be an existing directory"
        )
    return directory


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_commit_last(
    staged: Sequence[Path], final: Sequence[Path]
) -> None:
    if len(staged) != 4 or len(final) != 4:
        raise RuntimeError("RC5 publication requires exactly four members")
    published: list[Path] = []
    try:
        for source, destination in zip(staged, final, strict=True):
            os.link(source, destination, follow_symlinks=False)
            published.append(destination)
            _fsync_directory(destination.parent)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(final[0].parent)
        raise


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5ContextBundle:
    """Verifier-only, recursively immutable pre-label producer authority."""

    context: VerifiedStage2ContextV2
    anchor: VerifiedContextTailAnchor
    variable_query_window: VerifiedStage2VariableQueryWindow
    score_bundle: VerifiedStage2RC5ScoreBundleV2
    score_manifest_metadata: VerifiedStage2ScoreManifestMetadataV5
    source_reference: VerifiedStage2RC5SourceReferenceV3
    statistics_config: StatisticsConfig
    producer_manifest: Mapping[str, Any]
    commit: Mapping[str, Any]
    context_path: Path
    anchor_path: Path
    producer_manifest_path: Path
    commit_path: Path
    context_sha256: str
    anchor_sha256: str
    producer_manifest_sha256: str
    commit_sha256: str
    bundle_identity_sha256: str
    capability_schema: str
    _capability: object

    def __init__(
        self,
        *,
        context: VerifiedStage2ContextV2 | None = None,
        anchor: VerifiedContextTailAnchor | None = None,
        variable_query_window: VerifiedStage2VariableQueryWindow | None = None,
        score_bundle: VerifiedStage2RC5ScoreBundleV2 | None = None,
        score_manifest_metadata: VerifiedStage2ScoreManifestMetadataV5 | None = None,
        source_reference: VerifiedStage2RC5SourceReferenceV3 | None = None,
        statistics_config: StatisticsConfig | None = None,
        producer_manifest: Mapping[str, Any] | None = None,
        commit: Mapping[str, Any] | None = None,
        context_path: Path | None = None,
        anchor_path: Path | None = None,
        producer_manifest_path: Path | None = None,
        commit_path: Path | None = None,
        context_sha256: str | None = None,
        anchor_sha256: str | None = None,
        producer_manifest_sha256: str | None = None,
        commit_sha256: str | None = None,
        _capability: object | None = None,
    ) -> None:
        if _capability is not _CAPABILITY_TOKEN:
            raise TypeError("VerifiedStage2RC5ContextBundle is verifier-issued only")
        required = (
            context,
            anchor,
            variable_query_window,
            score_bundle,
            score_manifest_metadata,
            source_reference,
            statistics_config,
            producer_manifest,
            commit,
            context_path,
            anchor_path,
            producer_manifest_path,
            commit_path,
            context_sha256,
            anchor_sha256,
            producer_manifest_sha256,
            commit_sha256,
        )
        if any(value is None for value in required):
            raise RuntimeError("RC5 bundle capability construction is incomplete")
        assert context is not None and anchor is not None
        assert variable_query_window is not None
        assert score_bundle is not None
        assert score_manifest_metadata is not None and source_reference is not None
        assert statistics_config is not None and commit is not None
        object.__setattr__(self, "context", assert_verified_context_v2(context))
        object.__setattr__(
            self, "anchor", assert_verified_context_tail_anchor(anchor)
        )
        object.__setattr__(
            self,
            "variable_query_window",
            assert_verified_stage2_variable_query_window(variable_query_window),
        )
        object.__setattr__(
            self,
            "score_bundle",
            assert_verified_stage2_rc5_score_bundle_v2(score_bundle),
        )
        object.__setattr__(
            self,
            "score_manifest_metadata",
            assert_verified_stage2_score_manifest_metadata_v5(
                score_manifest_metadata
            ),
        )
        if (
            self.score_bundle.score_manifest_metadata.manifest_sha256
            != self.score_manifest_metadata.manifest_sha256
            or self.score_bundle.score_manifest_metadata.records_content_sha256
            != self.score_manifest_metadata.records_content_sha256
        ):
            raise Stage2RC5ContextProducerError(
                "context capability score bundle/metadata mismatch"
            )
        object.__setattr__(
            self,
            "source_reference",
            assert_verified_stage2_rc5_source_reference_v3(source_reference),
        )
        object.__setattr__(self, "statistics_config", statistics_config)
        object.__setattr__(self, "producer_manifest", _freeze(producer_manifest))
        object.__setattr__(self, "commit", _freeze(commit))
        object.__setattr__(self, "context_path", context_path)
        object.__setattr__(self, "anchor_path", anchor_path)
        object.__setattr__(self, "producer_manifest_path", producer_manifest_path)
        object.__setattr__(self, "commit_path", commit_path)
        object.__setattr__(self, "context_sha256", context_sha256)
        object.__setattr__(self, "anchor_sha256", anchor_sha256)
        object.__setattr__(
            self, "producer_manifest_sha256", producer_manifest_sha256
        )
        object.__setattr__(self, "commit_sha256", commit_sha256)
        object.__setattr__(
            self,
            "bundle_identity_sha256",
            str(commit["bundle_identity_sha256"]),
        )
        object.__setattr__(self, "capability_schema", BUNDLE_CAPABILITY_SCHEMA)
        object.__setattr__(self, "_capability", _CAPABILITY_TOKEN)


def assert_verified_stage2_rc5_context_bundle(
    value: Any,
) -> VerifiedStage2RC5ContextBundle:
    if (
        type(value) is not VerifiedStage2RC5ContextBundle
        or getattr(value, "_capability", None) is not _CAPABILITY_TOKEN
        or value.capability_schema != BUNDLE_CAPABILITY_SCHEMA
    ):
        raise TypeError(
            "a verifier-issued Stage-2 RC5 context producer bundle is required"
        )
    assert_verified_context_v2(value.context)
    assert_verified_context_tail_anchor(value.anchor)
    assert_verified_stage2_variable_query_window(value.variable_query_window)
    assert_verified_stage2_rc5_score_bundle_v2(value.score_bundle)
    assert_verified_stage2_score_manifest_metadata_v5(
        value.score_manifest_metadata
    )
    if (
        value.score_bundle.score_manifest_metadata.manifest_sha256
        != value.score_manifest_metadata.manifest_sha256
        or value.score_bundle.score_manifest_metadata.records_content_sha256
        != value.score_manifest_metadata.records_content_sha256
    ):
        raise Stage2RC5ContextProducerError(
            "context capability score bundle/metadata mismatch"
        )
    assert_verified_stage2_rc5_source_reference_v3(value.source_reference)
    return value


def _issue_bundle(
    *,
    prepared: _PreparedInputs,
    context: VerifiedStage2ContextV2,
    anchor: VerifiedContextTailAnchor,
    producer_manifest: Mapping[str, Any],
    commit: Mapping[str, Any],
    paths: tuple[Path, Path, Path, Path],
    hashes: tuple[str, str, str, str],
) -> VerifiedStage2RC5ContextBundle:
    return VerifiedStage2RC5ContextBundle(
        context=context,
        anchor=anchor,
        variable_query_window=prepared.window,
        score_bundle=prepared.score_bundle,
        score_manifest_metadata=prepared.score,
        source_reference=prepared.source,
        statistics_config=prepared.statistics_config,
        producer_manifest=producer_manifest,
        commit=commit,
        context_path=paths[0],
        anchor_path=paths[1],
        producer_manifest_path=paths[2],
        commit_path=paths[3],
        context_sha256=hashes[0],
        anchor_sha256=hashes[1],
        producer_manifest_sha256=hashes[2],
        commit_sha256=hashes[3],
        _capability=_CAPABILITY_TOKEN,
    )


def build_and_publish_stage2_rc5_context_bundle(
    *,
    variable_query_window: VerifiedStage2VariableQueryWindow,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    source_reference: VerifiedStage2RC5SourceReferenceV3,
    statistics_config: StatisticsConfig,
    statistics_config_path: str | Path,
    statistics_config_sha256: str,
    window_index: int,
    output_directory: str | Path,
    repository_root: str | Path | None = None,
) -> VerifiedStage2RC5ContextBundle:
    """Build and atomically publish one label-blind RC5 context bundle."""

    prepared = _prepare_inputs(
        variable_query_window=variable_query_window,
        score_bundle=score_bundle,
        source_reference=source_reference,
        statistics_config=statistics_config,
        statistics_config_path=statistics_config_path,
        statistics_config_sha256=statistics_config_sha256,
        window_index=window_index,
        repository_root=repository_root,
    )
    output = _output_directory(output_directory, prepared.root)
    final_paths = _bundle_paths(output)
    for path in final_paths:
        if os.path.lexists(path):
            raise FileExistsError(f"RC5 context bundle target exists: {path}")

    context, anchor, _ = _produce_context_and_anchor(prepared)
    context_bytes = bytes(context.canonical_payload)
    anchor_bytes = canonical_json_bytes(anchor.payload)
    context_sha = _sha_bytes(context_bytes)
    anchor_sha = _sha_bytes(anchor_bytes)
    producer_manifest = _producer_manifest(
        prepared,
        context,
        anchor,
        context_path=final_paths[0],
        context_sha256=context_sha,
        anchor_path=final_paths[1],
        anchor_sha256=anchor_sha,
    )
    producer_bytes = canonical_json_bytes(producer_manifest)
    producer_sha = _sha_bytes(producer_bytes)
    commit = _commit_payload(
        root=prepared.root,
        context_path=final_paths[0],
        context_sha256=context_sha,
        anchor_path=final_paths[1],
        anchor_sha256=anchor_sha,
        producer_manifest_path=final_paths[2],
        producer_manifest_sha256=producer_sha,
        producer_identity_sha256=producer_manifest[
            "producer_identity_sha256"
        ],
    )
    commit_bytes = canonical_json_bytes(commit)
    commit_sha = _sha_bytes(commit_bytes)
    contents = (context_bytes, anchor_bytes, producer_bytes, commit_bytes)

    staging = Path(tempfile.mkdtemp(prefix=".rc5-context-staging-", dir=output))
    staged_paths = _bundle_paths(staging)
    published = False
    try:
        for path, data in zip(staged_paths, contents, strict=True):
            _write_exclusive(path, data)
        _fsync_directory(staging)
        _publish_commit_last(staged_paths, final_paths)
        published = True
        observed = tuple(
            _sha_bytes(
                _stable_read_member(path, digest, prepared.root, f"published member {index}")
            )
            for index, (path, digest) in enumerate(
                zip(final_paths, (context_sha, anchor_sha, producer_sha, commit_sha), strict=True)
            )
        )
        if observed != (context_sha, anchor_sha, producer_sha, commit_sha):
            raise RuntimeError("published RC5 bundle digest replay failed")
        return _issue_bundle(
            prepared=prepared,
            context=context,
            anchor=anchor,
            producer_manifest=producer_manifest,
            commit=commit,
            paths=final_paths,
            hashes=(context_sha, anchor_sha, producer_sha, commit_sha),
        )
    except BaseException:
        if published:
            for path in reversed(final_paths):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            _fsync_directory(output)
        raise
    finally:
        for path in staged_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        try:
            staging.rmdir()
        except FileNotFoundError:
            pass


def verify_stage2_rc5_context_bundle(
    commit_path: str | Path,
    expected_commit_sha256: str,
    *,
    variable_query_window: VerifiedStage2VariableQueryWindow,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    source_reference: VerifiedStage2RC5SourceReferenceV3,
    statistics_config: StatisticsConfig,
    statistics_config_path: str | Path,
    statistics_config_sha256: str,
    repository_root: str | Path | None = None,
) -> VerifiedStage2RC5ContextBundle:
    """Reverify a published bundle and replay its fourteen context members."""

    window = assert_verified_stage2_variable_query_window(variable_query_window)
    replayed_score_bundle = replay_verified_stage2_rc5_score_bundle_v2(
        assert_verified_stage2_rc5_score_bundle_v2(score_bundle)
    )
    score = replayed_score_bundle.score_manifest_metadata
    root = _root(repository_root, window, score)
    commit_candidate = Path(commit_path).expanduser()
    if not commit_candidate.is_absolute():
        commit_candidate = root / commit_candidate
    commit_candidate = _direct_existing_file(
        commit_candidate, root, "RC5 bundle commit"
    )
    commit_sha = _sha_value(expected_commit_sha256, "expected_commit_sha256")
    commit_bytes = _stable_read_member(
        commit_candidate, commit_sha, root, "RC5 bundle commit"
    )
    commit = _strict_json_object(commit_bytes, "RC5 bundle commit")
    members = commit.get("members")
    if not isinstance(members, Mapping) or set(members) != {
        "context",
        "anchor",
        "producer_manifest",
    }:
        raise Stage2RC5ContextProducerError("commit member closure mismatch")

    paths: list[Path] = []
    hashes: list[str] = []
    for name in ("context", "anchor", "producer_manifest"):
        binding = members[name]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise Stage2RC5ContextProducerError(
                f"commit {name} binding closure mismatch"
            )
        relative = _score_v4._relative_repository_path(
            binding["path"], f"commit {name}.path"
        )
        path = _direct_existing_file(
            root.joinpath(*PurePosixPath(relative).parts), root, f"commit {name}"
        )
        paths.append(path)
        hashes.append(_sha_value(binding["sha256"], f"commit {name}.sha256"))
    if commit_candidate != paths[0].parent / COMMIT_FILENAME:
        raise Stage2RC5ContextProducerError(
            "commit path/name does not match the canonical bundle layout"
        )
    if tuple(path.name for path in paths) != (
        CONTEXT_FILENAME,
        ANCHOR_FILENAME,
        PRODUCER_MANIFEST_FILENAME,
    ) or len({path.parent for path in paths}) != 1:
        raise Stage2RC5ContextProducerError(
            "commit members do not use the canonical same-directory layout"
        )

    context_bytes, anchor_bytes, producer_bytes = tuple(
        _stable_read_member(path, digest, root, f"published {name}")
        for name, path, digest in zip(
            ("context", "anchor", "producer manifest"),
            paths,
            hashes,
            strict=True,
        )
    )
    context_payload = _strict_json_object(context_bytes, "published context")
    anchor_payload = _strict_json_object(anchor_bytes, "published anchor")
    producer_manifest = _strict_json_object(
        producer_bytes, "published producer manifest"
    )
    window_index = context_payload.get("window_index")
    if type(window_index) is not int:
        raise Stage2RC5ContextProducerError(
            "published context window_index is invalid"
        )
    prepared = _prepare_inputs(
        variable_query_window=window,
        score_bundle=replayed_score_bundle,
        source_reference=source_reference,
        statistics_config=statistics_config,
        statistics_config_path=statistics_config_path,
        statistics_config_sha256=statistics_config_sha256,
        window_index=window_index,
        repository_root=root,
    )
    expected_context, expected_anchor, _ = _produce_context_and_anchor(prepared)
    if expected_context.canonical_payload != context_bytes:
        raise Stage2RC5ContextProducerError(
            "published context differs from label-blind producer replay"
        )
    # Also force the standalone v6 semantic verifier over the disk payload.
    pure_context = verify_context_payload_v2(context_payload)
    if pure_context.payload_sha256 != expected_context.payload_sha256:
        raise Stage2RC5ContextProducerError(
            "published context semantic digest mismatch"
        )
    if canonical_json_bytes(expected_anchor.payload) != anchor_bytes:
        raise Stage2RC5ContextProducerError(
            "published anchor differs from exact-rational producer replay"
        )
    expected_manifest = _producer_manifest(
        prepared,
        expected_context,
        expected_anchor,
        context_path=paths[0],
        context_sha256=hashes[0],
        anchor_path=paths[1],
        anchor_sha256=hashes[1],
    )
    if canonical_json_bytes(expected_manifest) != producer_bytes:
        raise Stage2RC5ContextProducerError(
            "published producer manifest differs from transitive replay"
        )
    expected_commit = _commit_payload(
        root=root,
        context_path=paths[0],
        context_sha256=hashes[0],
        anchor_path=paths[1],
        anchor_sha256=hashes[1],
        producer_manifest_path=paths[2],
        producer_manifest_sha256=hashes[2],
        producer_identity_sha256=expected_manifest[
            "producer_identity_sha256"
        ],
    )
    if canonical_json_bytes(expected_commit) != commit_bytes:
        raise Stage2RC5ContextProducerError(
            "published commit differs from commit-last replay"
        )
    final_paths = (paths[0], paths[1], paths[2], commit_candidate)
    final_hashes = (hashes[0], hashes[1], hashes[2], commit_sha)
    return _issue_bundle(
        prepared=prepared,
        context=expected_context,
        anchor=expected_anchor,
        producer_manifest=expected_manifest,
        commit=expected_commit,
        paths=final_paths,
        hashes=final_hashes,
    )


def replay_verified_stage2_rc5_context_bundle(
    value: Any,
) -> VerifiedStage2RC5ContextBundle:
    """Freshly replay one stored producer capability from its bound files.

    The returned object is always newly issued by the public verifier.  A
    caller-retained capability whose in-memory fields differ from that replay
    is rejected instead of being silently normalized.
    """

    supplied = assert_verified_stage2_rc5_context_bundle(value)
    inputs = supplied.producer_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise Stage2RC5ContextProducerError(
            "producer bundle lacks replay input bindings"
        )
    statistics = inputs.get("statistics_config")
    if (
        not isinstance(statistics, Mapping)
        or set(statistics) != {"path", "sha256"}
    ):
        raise Stage2RC5ContextProducerError(
            "producer statistics-config binding closure drifted"
        )
    root = supplied.score_manifest_metadata.repository_root
    replayed = verify_stage2_rc5_context_bundle(
        supplied.commit_path,
        supplied.commit_sha256,
        variable_query_window=supplied.variable_query_window,
        score_bundle=supplied.score_bundle,
        source_reference=supplied.source_reference,
        statistics_config=supplied.statistics_config,
        statistics_config_path=statistics["path"],
        statistics_config_sha256=statistics["sha256"],
        repository_root=root,
    )
    scalar_fields = (
        "context_sha256",
        "anchor_sha256",
        "producer_manifest_sha256",
        "commit_sha256",
        "bundle_identity_sha256",
        "capability_schema",
    )
    path_fields = (
        "context_path",
        "anchor_path",
        "producer_manifest_path",
        "commit_path",
    )
    if any(
        getattr(replayed, field) != getattr(supplied, field)
        for field in (*scalar_fields, *path_fields)
    ):
        raise Stage2RC5ContextProducerError(
            "producer bundle identity differs from full current-state replay"
        )
    material_pairs = (
        (
            canonical_json_bytes(replayed.producer_manifest),
            canonical_json_bytes(supplied.producer_manifest),
        ),
        (
            canonical_json_bytes(replayed.commit),
            canonical_json_bytes(supplied.commit),
        ),
        (replayed.context.canonical_payload, supplied.context.canonical_payload),
        (
            canonical_json_bytes(replayed.anchor.payload),
            canonical_json_bytes(supplied.anchor.payload),
        ),
        (
            canonical_json_bytes(replayed.variable_query_window.payload),
            canonical_json_bytes(supplied.variable_query_window.payload),
        ),
        (
            canonical_json_bytes(replayed.score_bundle.attestation),
            canonical_json_bytes(supplied.score_bundle.attestation),
        ),
        (
            canonical_json_bytes(replayed.score_manifest_metadata.payload),
            canonical_json_bytes(supplied.score_manifest_metadata.payload),
        ),
        (
            canonical_json_bytes(replayed.source_reference.attestation),
            canonical_json_bytes(supplied.source_reference.attestation),
        ),
    )
    if any(current != retained for current, retained in material_pairs):
        raise Stage2RC5ContextProducerError(
            "producer bundle material differs from full current-state replay"
        )
    upstream_scalar_pairs = (
        (
            replayed.variable_query_window.path,
            supplied.variable_query_window.path,
        ),
        (
            replayed.variable_query_window.manifest_sha256,
            supplied.variable_query_window.manifest_sha256,
        ),
        (
            replayed.score_bundle.attestation_path,
            supplied.score_bundle.attestation_path,
        ),
        (
            replayed.score_bundle.attestation_sha256,
            supplied.score_bundle.attestation_sha256,
        ),
        (
            replayed.score_bundle.run_complete.artifact_path,
            supplied.score_bundle.run_complete.artifact_path,
        ),
        (
            replayed.score_bundle.run_complete.sha256,
            supplied.score_bundle.run_complete.sha256,
        ),
        (
            replayed.score_manifest_metadata.path,
            supplied.score_manifest_metadata.path,
        ),
        (
            replayed.score_manifest_metadata.manifest_sha256,
            supplied.score_manifest_metadata.manifest_sha256,
        ),
        (
            replayed.score_manifest_metadata.records_content_sha256,
            supplied.score_manifest_metadata.records_content_sha256,
        ),
        (
            replayed.source_reference.attestation_path,
            supplied.source_reference.attestation_path,
        ),
        (
            replayed.source_reference.attestation_sha256,
            supplied.source_reference.attestation_sha256,
        ),
        (
            replayed.source_reference.repository_root,
            supplied.source_reference.repository_root,
        ),
        (
            replayed.score_manifest_metadata.repository_root,
            supplied.score_manifest_metadata.repository_root,
        ),
        (replayed.statistics_config, supplied.statistics_config),
    )
    if any(current != retained for current, retained in upstream_scalar_pairs):
        raise Stage2RC5ContextProducerError(
            "producer upstream capability differs from full current-state replay"
        )
    return replayed


__all__ = [
    "ANCHOR_FILENAME",
    "BUNDLE_CAPABILITY_SCHEMA",
    "COMMIT_FILENAME",
    "COMMIT_SCHEMA",
    "CONTEXT_FILENAME",
    "PRODUCER_MANIFEST_FILENAME",
    "PRODUCER_MANIFEST_SCHEMA",
    "PUBLICATION_ORDER",
    "Stage2RC5ContextProducerError",
    "VerifiedStage2RC5ContextBundle",
    "assert_verified_stage2_rc5_context_bundle",
    "build_and_publish_stage2_rc5_context_bundle",
    "canonical_json_bytes",
    "replay_verified_stage2_rc5_context_bundle",
    "verify_stage2_rc5_context_bundle",
]
