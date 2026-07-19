"""Persistent source-only exact-curve authority for RC5 Stage-2.

One bank covers every record of exactly one Stage-2 source score role.  It
fresh-replays the RC5 score attestation, promotes metadata-v5 through the full
score-manifest-v4 verifier, resolves only source masks, materialises one
aligned label NPZ and one exact event curve per image, and publishes a
commit-last bundle.  Public verification repeats score+mask->label->curve
materialisation and never treats the persisted curve arrays as a trust root.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image

from data_ext.dataset_meta import safe_output_stem
from data_ext.mask_alignment import (
    DEFAULT_ASPECT_TOLERANCE,
    align_mask_to_image,
    aspect_ratio_relative_error,
)
from data_ext.split_utils import IMAGE_EXTENSIONS, sample_id_from_entry
from data_ext.stage2_rc5_score_bundle_v2 import (
    VerifiedStage2RC5ScoreBundleV2,
    assert_verified_stage2_rc5_score_bundle_v2,
    replay_verified_stage2_rc5_score_bundle_v2,
)
from data_ext.stage2_score_manifest import (
    FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
    NPZ_FIELD_ORDER,
    NPZ_ZIP_MEMBER_ORDER,
    OOF_HOLDOUT_STAGE2_FIT,
    OOF_TRAIN_SOURCE_REFERENCE,
    OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT,
    SOURCE_DIAGNOSTIC_VALIDATION,
    STRICT_THRESHOLD_SEMANTICS,
    VerifiedStage2ScoreManifest,
    verify_stage2_score_manifest,
)
from evaluation.component_matching import prepare_target
from evaluation.stage2_threshold_sweep import (
    STAGE2_MATCHING_CONTRACT,
    STAGE2_THRESHOLD_ALGORITHM,
    _build_incremental_exact_sweep,
)
from rc.stage2_compositional_curve_provider import (
    PerImageExactEventCurve,
    PerImageExactEventCurveBank,
    assert_per_image_exact_event_curve,
    assert_per_image_exact_event_curve_bank,
    build_per_image_exact_event_curve,
    build_per_image_exact_event_curve_bank,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_SCHEMA = "rc-irstd.stage2-rc5-source-exact-curve-bank.v1"
COMMIT_SCHEMA = "rc-irstd.stage2-rc5-source-exact-curve-bank-commit.v1"
CAPABILITY_SCHEMA = "rc-irstd.stage2-rc5-source-exact-curve-bank-capability.v1"
MANIFEST_FILENAME = "source_exact_curve_manifest.json"
COMMIT_FILENAME = "SOURCE_EXACT_CURVE_BANK_COMMIT.json"

OFFSETS_FILENAME = "curve_offsets.npy"
THRESHOLDS_FILENAME = "curve_thresholds.npy"
FP_FILENAME = "curve_false_positive_pixels.npy"
TP_FILENAME = "curve_matched_objects.npy"

ARRAY_FILENAMES = MappingProxyType(
    {
        "offsets": OFFSETS_FILENAME,
        "thresholds": THRESHOLDS_FILENAME,
        "false_positive_pixels": FP_FILENAME,
        "matched_objects": TP_FILENAME,
    }
)

SOURCE_ROLES = frozenset(
    {
        OOF_TRAIN_SOURCE_REFERENCE,
        OOF_HOLDOUT_STAGE2_FIT,
        FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE,
        SOURCE_DIAGNOSTIC_VALIDATION,
    }
)
OOF_ROLES = frozenset(
    {OOF_TRAIN_SOURCE_REFERENCE, OOF_HOLDOUT_STAGE2_FIT}
)
FULLFIT_ROLES = frozenset(
    {FULLFIT_DETECTOR_FIT_SOURCE_REFERENCE, SOURCE_DIAGNOSTIC_VALIDATION}
)

LABEL_NPZ_FIELD_ORDER = (
    "mask",
    "canonical_id",
    "image_id",
    "source_domain",
    "original_hw",
    "source_mask_original_hw",
    "alignment_operation",
    "interpolation",
)
ALIGNMENT_POLICY = "BasicIRSTD-compatible-mask-to-image-nearest-v1"
ALIGNMENT_MODULE_PATH = "data_ext/mask_alignment.py"
RECORDS_ALGORITHM = (
    "sha256-canonical-json-ordered-rc5-source-label-curve-records-v1"
)
ORDERED_IDENTITIES_ALGORITHM = (
    "sha256-canonical-json-ordered-original-image-sha256-v1"
)
CAPABILITY_STATE_ALGORITHM = (
    "sha256-rc5-source-exact-curve-capability-state-v1"
)

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_CAPABILITY = object()


class Stage2RC5SourceExactCurveBankV1Error(ValueError):
    """The source-only score->label->curve authority failed closed."""


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


def _canonical(value: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(
        _plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return data + (b"\n" if newline else b"")


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    projection = _plain(value)
    projection[field] = ""
    return _json_sha(projection)


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_json(data: bytes, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise Stage2RC5SourceExactCurveBankV1Error(
                    f"{name} has duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} contains non-finite JSON value {value}"
        )

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"invalid {name}: {error}"
        ) from error
    if not isinstance(value, dict) or _canonical(value, newline=True) != data:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} is not canonical JSON"
        )
    return value


def _repository_root(value: str | Path | None) -> Path:
    raw = REPOSITORY_ROOT if value is None else Path(value).expanduser()
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if absolute.is_symlink():
        raise Stage2RC5SourceExactCurveBankV1Error(
            "repository_root may not be a symlink"
        )
    try:
        info = absolute.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "repository_root does not exist"
        ) from error
    if not stat.S_ISDIR(info.st_mode):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "repository_root is not a directory"
        )
    return absolute.resolve(strict=True)


def _reject_symlink_components(path: Path, root: Path, name: str) -> None:
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} escapes repository_root"
        ) from error
    current = root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current):
            info = current.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise Stage2RC5SourceExactCurveBankV1Error(
                    f"{name} contains a symlink component"
                )


def _repo_relative(path: Path, root: Path, name: str) -> str:
    candidate = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(candidate, root, name)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} escapes repository_root"
        ) from error
    return relative.as_posix()


def _relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} must be a nonempty path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} is not canonical repository-relative POSIX"
        )
    lowered = value.lower().replace("-", "_")
    if "official_test" in lowered or "officialtest" in lowered:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} may not reference official test"
        )
    return value


def _stable_bytes(
    path: Path,
    root: Path,
    name: str,
    *,
    expected_sha256: str | None = None,
) -> bytes:
    candidate = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(candidate, root, name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"cannot open direct {name}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise Stage2RC5SourceExactCurveBankV1Error(
                f"{name} is not a regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = candidate.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise RuntimeError(f"{name} disappeared during stable read") from error

    def identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            int(info.st_dev),
            int(info.st_ino),
            int(info.st_mode),
            int(info.st_size),
            int(info.st_mtime_ns),
            int(info.st_ctime_ns),
        )

    if identity(before) != identity(after) or identity(before) != identity(current):
        raise RuntimeError(f"{name} changed during stable read")
    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != _sha(
        expected_sha256, f"{name}.expected_sha256"
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} SHA-256 mismatch"
        )
    return data


def _input_directory(value: str | Path, root: Path, name: str) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    _reject_symlink_components(candidate, root, name)
    try:
        info = candidate.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} does not exist"
        ) from error
    if not stat.S_ISDIR(info.st_mode):
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} is not a real directory"
        )
    return candidate


def _output_directory(value: str | Path, root: Path) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "output_directory escapes repository_root"
        ) from error
    if os.path.lexists(candidate):
        raise FileExistsError("immutable source exact-curve bank already exists")
    _reject_symlink_components(candidate.parent, root, "output parent")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(candidate.parent, root, "output parent")
    resolved = candidate.parent.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "output parent resolves outside repository_root"
        ) from error
    os.mkdir(candidate, 0o755)
    return candidate


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _owned_readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _npz_scalar(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} must be a zero-dimensional scalar"
        )
    return str(array.item())


def _decode_score_probability(data: bytes, record: Mapping[str, Any]) -> np.ndarray:
    try:
        with zipfile.ZipFile(io.BytesIO(data), mode="r") as archive:
            members = tuple(item.filename for item in archive.infolist())
            if len(members) != len(set(members)) or members != NPZ_ZIP_MEMBER_ORDER:
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "score NPZ ZIP member closure/order mismatch"
                )
        with np.load(io.BytesIO(data), allow_pickle=False) as payload:
            if tuple(payload.files) != NPZ_FIELD_ORDER:
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "score NPZ field closure/order mismatch"
                )
            probability = np.array(payload["prob"], dtype=np.float64, copy=True)
            raw_logit = np.asarray(payload["raw_logit"])
            original_hw = tuple(int(item) for item in np.asarray(
                payload["original_hw"], dtype=np.int64
            ).reshape(-1))
            if original_hw != tuple(record["original_hw"]):
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "score NPZ original_hw mismatch"
                )
            if probability.dtype != np.float64 or probability.shape != original_hw:
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "score probability is not native float64"
                )
            if raw_logit.dtype != np.float64 or raw_logit.shape != original_hw:
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "score raw_logit is not native float64"
                )
            for field in ("canonical_id", "image_id", "source_domain", "resize_mode"):
                if _npz_scalar(payload[field], f"score.{field}") != str(record[field]):
                    raise Stage2RC5SourceExactCurveBankV1Error(
                        f"score NPZ {field} mismatch"
                    )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        if isinstance(error, Stage2RC5SourceExactCurveBankV1Error):
            raise
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"invalid score NPZ: {error}"
        ) from error
    if not np.isfinite(probability).all() or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "score probability contains invalid values"
        )
    probability.setflags(write=False)
    return probability


def _resolve_mask_direct(
    dataset: Path,
    mask_folder: str,
    image_id: str,
    root: Path,
) -> Path:
    if (
        not isinstance(mask_folder, str)
        or not mask_folder
        or PurePosixPath(mask_folder).name != mask_folder
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "mask_folder must be one direct folder name"
        )
    lowered = mask_folder.lower().replace("-", "_")
    if lowered in {"official_test", "officialtest"}:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "mask_folder may not reference official test"
        )
    sample = sample_id_from_entry(image_id)
    pure = PurePosixPath(sample)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "image_id is not a canonical relative identity"
        )
    folder = dataset / mask_folder
    _reject_symlink_components(folder, root, "source mask folder")
    if not folder.is_dir():
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source mask folder is not a real directory"
        )
    stems = [pure.name]
    if pure.name.endswith("_pixels0"):
        stems.append(pure.name[: -len("_pixels0")])
    else:
        stems.append(f"{pure.name}_pixels0")
    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem in stems:
        relative_stem = Path(*pure.parts[:-1], stem)
        for extension in IMAGE_EXTENSIONS:
            for suffix in (extension, extension.upper()):
                candidate = folder / relative_stem.with_suffix(suffix)
                if candidate not in seen:
                    seen.add(candidate)
                    if os.path.lexists(candidate):
                        candidates.append(candidate)
    if len(candidates) != 1:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source mask direct resolution requires exactly one candidate"
        )
    _reject_symlink_components(candidates[0], root, "source mask")
    info = candidates[0].stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source mask is not a direct regular file"
        )
    return candidates[0]


def _decode_source_mask(data: bytes, name: str) -> tuple[Image.Image, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(data)) as handle:
            source = handle.convert("L")
            source.load()
    except (OSError, ValueError) as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"invalid source mask {name}: {error}"
        ) from error
    return source, (int(source.height), int(source.width))


def _encode_label_npz(
    *,
    mask: np.ndarray,
    record: Mapping[str, Any],
    source_hw: tuple[int, int],
    operation: str,
) -> bytes:
    values = {
        "mask": np.asarray(mask, dtype=np.uint8),
        "canonical_id": np.asarray(record["canonical_id"]),
        "image_id": np.asarray(record["image_id"]),
        "source_domain": np.asarray(record["source_domain"]),
        "original_hw": np.asarray(record["original_hw"], dtype=np.int64),
        "source_mask_original_hw": np.asarray(source_hw, dtype=np.int64),
        "alignment_operation": np.asarray(operation),
        "interpolation": np.asarray("nearest_neighbor"),
    }
    if tuple(values) != LABEL_NPZ_FIELD_ORDER:
        raise RuntimeError("label NPZ field order drifted")
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def _decode_label_npz(
    data: bytes,
    *,
    record: Mapping[str, Any],
    source_hw: tuple[int, int],
    operation: str,
) -> np.ndarray:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as payload:
            if tuple(payload.files) != LABEL_NPZ_FIELD_ORDER:
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "label NPZ field closure/order mismatch"
                )
            mask = np.array(payload["mask"], dtype=np.uint8, copy=True)
            if mask.dtype != np.uint8 or mask.shape != tuple(record["original_hw"]):
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "label NPZ mask geometry mismatch"
                )
            if not np.isin(mask, (0, 1)).all():
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "label NPZ mask is not binary"
                )
            expected = {
                "canonical_id": str(record["canonical_id"]),
                "image_id": str(record["image_id"]),
                "source_domain": str(record["source_domain"]),
                "alignment_operation": operation,
                "interpolation": "nearest_neighbor",
            }
            for field, value in expected.items():
                if _npz_scalar(payload[field], f"label.{field}") != value:
                    raise Stage2RC5SourceExactCurveBankV1Error(
                        f"label NPZ {field} mismatch"
                    )
            observed_hw = tuple(int(item) for item in np.asarray(
                payload["original_hw"], dtype=np.int64
            ).reshape(-1))
            observed_source = tuple(int(item) for item in np.asarray(
                payload["source_mask_original_hw"], dtype=np.int64
            ).reshape(-1))
            if observed_hw != tuple(record["original_hw"]) or observed_source != source_hw:
                raise Stage2RC5SourceExactCurveBankV1Error(
                    "label NPZ geometry provenance mismatch"
                )
    except (OSError, ValueError) as error:
        if isinstance(error, Stage2RC5SourceExactCurveBankV1Error):
            raise
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"invalid label NPZ: {error}"
        ) from error
    mask.setflags(write=False)
    return mask


@dataclass(frozen=True)
class _Authority:
    root: Path
    score_bundle: VerifiedStage2RC5ScoreBundleV2
    score_manifest: VerifiedStage2ScoreManifest
    role: str
    source_domain: str
    payload: Mapping[str, Any]


def _prepare_authority(
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    repository_root: str | Path | None,
) -> _Authority:
    root = _repository_root(repository_root)
    supplied = assert_verified_stage2_rc5_score_bundle_v2(score_bundle)
    replayed = replay_verified_stage2_rc5_score_bundle_v2(supplied)
    metadata = replayed.score_manifest_metadata
    role = str(metadata.role)

    # Security-significant ordering: reject outer/unknown roles before any
    # caller-provided dataset or mask path can enter the call graph.
    if role not in SOURCE_ROLES:
        if role == OUTER_TARGET_DIAGNOSTIC_DEVELOPMENT:
            raise Stage2RC5SourceExactCurveBankV1Error(
                "outer-target scores can never construct a source curve bank"
            )
        raise Stage2RC5SourceExactCurveBankV1Error(
            "score role is not one of the four source-only roles"
        )
    payload = metadata.payload
    detector_role = str(payload["detector_role"])
    fold = payload["oof_fold_index"]
    if role in OOF_ROLES:
        if detector_role != "detector_oof" or type(fold) is not int or fold not in {0, 1}:
            raise Stage2RC5SourceExactCurveBankV1Error(
                "OOF source role requires detector_oof and fold 0/1"
            )
    elif role in FULLFIT_ROLES and (
        detector_role != "detector_full_fit" or fold is not None
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "full-fit source role requires detector_full_fit and null fold"
        )
    if str(payload["source_domain"]) == str(payload["outer_target"]):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source curve bank domain equals the outer target"
        )
    if Path(metadata.repository_root).resolve(strict=True) != root:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "score bundle repository_root mismatch"
        )

    full = verify_stage2_score_manifest(
        metadata.path,
        metadata.manifest_sha256,
        role,
        repository_root=root,
    )
    if (
        full.path != metadata.path
        or full.manifest_sha256 != metadata.manifest_sha256
        or full.records_content_sha256 != metadata.records_content_sha256
        or _plain(full.payload) != _plain(metadata.payload)
        or _plain(full.records) != _plain(metadata.records)
        or _plain(full.bindings) != _plain(metadata.bindings)
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "full score-v4 replay differs from RC5 metadata-v5 authority"
        )
    return _Authority(
        root=root,
        score_bundle=replayed,
        score_manifest=full,
        role=role,
        source_domain=str(payload["source_domain"]),
        payload=payload,
    )


@dataclass(frozen=True)
class _Materialized:
    dataset_directory: Path
    mask_folder: str
    records: tuple[Mapping[str, Any], ...]
    label_files: tuple[tuple[str, bytes], ...]
    curves: tuple[PerImageExactEventCurve, ...]
    curve_bank: PerImageExactEventCurveBank
    offsets: np.ndarray
    thresholds: np.ndarray
    false_positive_pixels: np.ndarray
    matched_objects: np.ndarray


def _materialize(
    authority: _Authority,
    dataset_directory: str | Path,
    mask_folder: str,
    *,
    published_directory: Path | None = None,
) -> _Materialized:
    root = authority.root
    dataset = _input_directory(dataset_directory, root, "dataset_directory")
    if dataset.name != authority.source_domain:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "dataset_directory basename differs from score source_domain"
        )
    dataset_relative = _repo_relative(dataset, root, "dataset_directory")
    expected_image_prefix = f"{dataset_relative}/images/"
    policy_path = root / ALIGNMENT_MODULE_PATH
    policy_data = _stable_bytes(policy_path, root, "mask alignment policy")
    policy_sha = hashlib.sha256(policy_data).hexdigest()

    score_by_index = {
        int(item.record_index): item for item in authority.score_manifest.items
    }
    records: list[Mapping[str, Any]] = []
    labels: list[tuple[str, bytes]] = []
    curves: list[PerImageExactEventCurve] = []
    offsets = [0]
    for index, score_record in enumerate(authority.score_manifest.records):
        item = score_by_index.get(index)
        if item is None or int(score_record["record_index"]) != index:
            raise Stage2RC5SourceExactCurveBankV1Error(
                "score item/record order is not exact"
            )
        if not str(score_record["original_image_path"]).startswith(
            expected_image_prefix
        ):
            raise Stage2RC5SourceExactCurveBankV1Error(
                "score original image is outside dataset_directory/images"
            )
        score_data = _stable_bytes(
            item.score_path,
            root,
            f"score[{index}]",
            expected_sha256=str(score_record["score_file_sha256"]),
        )
        probability = _decode_score_probability(score_data, score_record)

        mask_path = _resolve_mask_direct(
            dataset,
            mask_folder,
            str(score_record["image_id"]),
            root,
        )
        mask_data = _stable_bytes(mask_path, root, f"source mask[{index}]")
        mask_sha = hashlib.sha256(mask_data).hexdigest()
        source_image, source_hw = _decode_source_mask(
            mask_data, str(score_record["image_id"])
        )
        target_hw = tuple(int(value) for value in score_record["original_hw"])
        aspect_error = aspect_ratio_relative_error(
            (target_hw[1], target_hw[0]), source_image.size
        )
        aligned_image = align_mask_to_image(
            source_image,
            (target_hw[1], target_hw[0]),
            str(score_record["image_id"]),
            aspect_tolerance=DEFAULT_ASPECT_TOLERANCE,
        )
        aligned = (np.asarray(aligned_image, dtype=np.uint8) > 0).astype(np.uint8)
        if aligned.shape != target_hw:
            raise Stage2RC5SourceExactCurveBankV1Error(
                "aligned source mask does not match native score geometry"
            )
        operation = "identity" if source_hw == target_hw else "resize_mask_to_image_geometry"
        label_name = (
            f"{index:06d}-{safe_output_stem(str(score_record['image_id']))}.label.npz"
        )
        if published_directory is None:
            label_data = _encode_label_npz(
                mask=aligned,
                record=score_record,
                source_hw=source_hw,
                operation=operation,
            )
        else:
            label_data = _stable_bytes(
                published_directory / label_name,
                root,
                f"published label[{index}]",
            )
        decoded_label = _decode_label_npz(
            label_data,
            record=score_record,
            source_hw=source_hw,
            operation=operation,
        )
        if not np.array_equal(decoded_label, aligned):
            raise Stage2RC5SourceExactCurveBankV1Error(
                "published aligned label differs from source-mask replay"
            )

        sweep = _build_incremental_exact_sweep(
            [probability], [prepare_target(decoded_label)]
        )
        curve = build_per_image_exact_event_curve(
            image_identity_sha256=str(score_record["original_image_sha256"]),
            thresholds=np.asarray(sweep.thresholds, dtype=np.float64),
            false_positive_pixels=np.asarray(
                sweep.rows.column("fp_pixels"), dtype=np.int64
            ),
            matched_objects=np.asarray(
                sweep.rows.column("tp_objects"), dtype=np.int64
            ),
            total_native_pixels=int(sweep.total_pixels),
            ground_truth_objects=int(sweep.gt_objects),
        )
        curves.append(curve)
        offsets.append(offsets[-1] + int(curve.thresholds.size))
        alignment = {
            "policy": ALIGNMENT_POLICY,
            "policy_module_path": ALIGNMENT_MODULE_PATH,
            "policy_module_sha256": policy_sha,
            "interpolation": "nearest_neighbor",
            "operation": operation,
            "source_mask_original_hw": list(source_hw),
            "target_image_hw": list(target_hw),
            "aspect_ratio_relative_error": float(aspect_error),
            "aspect_tolerance": float(DEFAULT_ASPECT_TOLERANCE),
            "mask_aligned_to_image_geometry": True,
            "silent_crop_used": False,
            "bilinear_resize_used": False,
            "nuaa_misc_111_policy_applied": bool(
                authority.source_domain == "NUAA-SIRST"
                and str(score_record["image_id"]) == "Misc_111"
            ),
        }
        records.append(
            {
                "record_index": index,
                "canonical_id": score_record["canonical_id"],
                "image_id": score_record["image_id"],
                "source_domain": authority.source_domain,
                "original_image_path": score_record["original_image_path"],
                "original_image_sha256": score_record["original_image_sha256"],
                "exclusion_group_id": score_record["exclusion_group_id"],
                "near_duplicate_cluster_id_or_unique_sentinel": score_record[
                    "near_duplicate_cluster_id_or_unique_sentinel"
                ],
                "source_role_record_index": score_record[
                    "source_role_record_index"
                ],
                "score_file": score_record["score_file"],
                "score_file_sha256": score_record["score_file_sha256"],
                "original_hw": list(target_hw),
                "source_mask_path": _repo_relative(
                    mask_path, root, f"source mask[{index}]"
                ),
                "source_mask_file_sha256": mask_sha,
                "source_mask_original_hw": list(source_hw),
                "label_file": label_name,
                "label_file_sha256": hashlib.sha256(label_data).hexdigest(),
                "alignment_provenance": alignment,
                "image_identity_sha256": curve.image_identity_sha256,
                "curve_row_start": offsets[-2],
                "curve_row_stop": offsets[-1],
                "curve_row_count": int(curve.thresholds.size),
                "curve_content_sha256": curve.content_sha256,
                "total_native_pixels": curve.total_native_pixels,
                "ground_truth_objects": curve.ground_truth_objects,
            }
        )
        labels.append((label_name, label_data))

    verified_curves = tuple(curves)
    bank = build_per_image_exact_event_curve_bank(verified_curves)
    threshold = _owned_readonly(
        np.concatenate([curve.thresholds for curve in verified_curves]), np.float64
    )
    fp = _owned_readonly(
        np.concatenate(
            [curve.false_positive_pixels for curve in verified_curves]
        ),
        np.int64,
    )
    tp = _owned_readonly(
        np.concatenate([curve.matched_objects for curve in verified_curves]),
        np.int64,
    )
    offset_array = _owned_readonly(np.asarray(offsets), np.int64)
    return _Materialized(
        dataset_directory=dataset,
        mask_folder=mask_folder,
        records=tuple(records),
        label_files=tuple(labels),
        curves=verified_curves,
        curve_bank=bank,
        offsets=offset_array,
        thresholds=threshold,
        false_positive_pixels=fp,
        matched_objects=tp,
    )


def _member_binding(name: str, data: bytes, value: np.ndarray) -> dict[str, Any]:
    return {
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }


def _score_authority(authority: _Authority) -> dict[str, Any]:
    score = authority.score_bundle
    metadata = score.score_manifest_metadata
    attestation = _plain(score.attestation)
    run_identity = attestation["run_complete"]["identity"]
    return {
        "score_attestation_path": _repo_relative(
            score.attestation_path, authority.root, "score attestation"
        ),
        "score_attestation_sha256": score.attestation_sha256,
        "score_manifest_path": _repo_relative(
            metadata.path, authority.root, "score manifest"
        ),
        "score_manifest_sha256": metadata.manifest_sha256,
        "score_records_content_sha256": metadata.records_content_sha256,
        "run_complete_path": _repo_relative(
            Path(score.run_complete.artifact_path),
            authority.root,
            "RUN_COMPLETE",
        ),
        "run_complete_artifact_sha256": score.run_complete.sha256,
        "run_complete_identity_sha256": str(run_identity["identity_sha256"]),
        "restricted_checkpoint": _plain(attestation["restricted_checkpoint"]),
        "full_score_v4_member_content_verified": True,
    }


def _manifest_payload(
    authority: _Authority,
    material: _Materialized,
    *,
    array_bytes: Mapping[str, bytes],
) -> dict[str, Any]:
    payload = authority.payload
    identities = [curve.image_identity_sha256 for curve in material.curves]
    records = [_plain(record) for record in material.records]
    boundaries = {
        field: sorted({str(record[field]) for record in records})
        for field in (
            "canonical_id",
            "original_image_sha256",
            "near_duplicate_cluster_id_or_unique_sentinel",
            "exclusion_group_id",
        )
    }
    arrays = {
        "offsets": material.offsets,
        "thresholds": material.thresholds,
        "false_positive_pixels": material.false_positive_pixels,
        "matched_objects": material.matched_objects,
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "artifact_type": "rc_irstd_stage2_rc5_source_exact_curve_bank",
        "artifact_status": "SOURCE_ONLY_EXACT_CURVE_BANK_COMPLETE",
        "development_only": True,
        "official_test_accessed": False,
        "outer_target_present": False,
        "role": authority.role,
        "outer_fold_id": payload["outer_fold_id"],
        "outer_target": payload["outer_target"],
        "source_domain": authority.source_domain,
        "base_seed": payload["base_seed"],
        "derived_seed": payload["derived_seed"],
        "detector_role": payload["detector_role"],
        "oof_fold_index": payload["oof_fold_index"],
        "threshold_semantics": STRICT_THRESHOLD_SEMANTICS,
        "threshold_algorithm": STAGE2_THRESHOLD_ALGORITHM,
        "matching_contract": STAGE2_MATCHING_CONTRACT,
        "dataset_directory": _repo_relative(
            material.dataset_directory,
            authority.root,
            "dataset_directory",
        ),
        "mask_folder": material.mask_folder,
        "score_authority": _score_authority(authority),
        "record_count": len(records),
        "total_curve_rows": int(material.thresholds.size),
        "curve_bank_id": material.curve_bank.bank_id,
        "records_content_sha256_algorithm": RECORDS_ALGORITHM,
        "records_content_sha256": _json_sha(
            {"algorithm": RECORDS_ALGORITHM, "records": records}
        ),
        "ordered_image_identity_sha256_algorithm": ORDERED_IDENTITIES_ALGORITHM,
        "ordered_image_identity_sha256": _json_sha(
            {"algorithm": ORDERED_IDENTITIES_ALGORITHM, "identities": identities}
        ),
        "identity_boundary_values": boundaries,
        "records": records,
        "members": {
            name: _member_binding(ARRAY_FILENAMES[name], array_bytes[name], value)
            for name, value in arrays.items()
        },
        "guardrails": {
            "all_score_records_covered": True,
            "labels_recomputed_from_source_masks": True,
            "curves_recomputed_from_score_and_label_bytes": True,
            "aggregate_curve_materialized": False,
            "caller_supplied_curve_arrays_accepted": False,
            "outer_target_scores_or_labels_accessed": False,
        },
        "manifest_identity_sha256": "",
    }
    manifest["manifest_identity_sha256"] = _self_hash(
        manifest, "manifest_identity_sha256"
    )
    return manifest


def _commit_payload(authority: _Authority, manifest: Mapping[str, Any], manifest_data: bytes) -> dict[str, Any]:
    commit: dict[str, Any] = {
        "schema_version": COMMIT_SCHEMA,
        "artifact_type": "rc_irstd_stage2_rc5_source_exact_curve_bank_commit",
        "artifact_status": "COMMITTED_LAST",
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_data).hexdigest(),
            "identity_sha256": manifest["manifest_identity_sha256"],
        },
        "score_attestation_sha256": authority.score_bundle.attestation_sha256,
        "role": authority.role,
        "source_domain": authority.source_domain,
        "record_count": manifest["record_count"],
        "curve_bank_id": manifest["curve_bank_id"],
        "commit_identity_sha256": "",
    }
    commit["commit_identity_sha256"] = _self_hash(
        commit, "commit_identity_sha256"
    )
    return commit


def _load_npy_member(
    directory: Path,
    root: Path,
    binding: Mapping[str, Any],
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, bytes]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path", "sha256", "dtype", "shape"
    }:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} member binding closure mismatch"
        )
    filename = ARRAY_FILENAMES[name]
    if (
        binding["path"] != filename
        or binding["dtype"] != str(dtype)
        or binding["shape"] != list(shape)
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"{name} member path/dtype/shape mismatch"
        )
    data = _stable_bytes(
        directory / filename,
        root,
        f"curve member {name}",
        expected_sha256=str(binding["sha256"]),
    )
    try:
        loaded = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"invalid {name} NPY: {error}"
        ) from error
    if not isinstance(loaded, np.ndarray) or loaded.dtype != dtype or loaded.shape != shape:
        raise Stage2RC5SourceExactCurveBankV1Error(
            f"loaded {name} dtype/shape mismatch"
        )
    return _owned_readonly(loaded, dtype), data


def _verify_members_and_recompute(
    directory: Path,
    authority: _Authority,
    manifest: Mapping[str, Any],
) -> tuple[_Materialized, dict[str, np.ndarray], dict[str, bytes], dict[str, Any]]:
    material = _materialize(
        authority,
        authority.root / _relative_path(
            manifest.get("dataset_directory"), "dataset_directory"
        ),
        str(manifest.get("mask_folder")),
        published_directory=directory,
    )
    count = len(material.records)
    total_rows = int(material.thresholds.size)
    members = manifest.get("members")
    if not isinstance(members, Mapping) or set(members) != set(ARRAY_FILENAMES):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "curve array member closure mismatch"
        )
    expected_arrays = {
        "offsets": material.offsets,
        "thresholds": material.thresholds,
        "false_positive_pixels": material.false_positive_pixels,
        "matched_objects": material.matched_objects,
    }
    shapes = {
        "offsets": (count + 1,),
        "thresholds": (total_rows,),
        "false_positive_pixels": (total_rows,),
        "matched_objects": (total_rows,),
    }
    dtypes = {
        "offsets": np.dtype("int64"),
        "thresholds": np.dtype("float64"),
        "false_positive_pixels": np.dtype("int64"),
        "matched_objects": np.dtype("int64"),
    }
    loaded: dict[str, np.ndarray] = {}
    raw: dict[str, bytes] = {}
    for name in ARRAY_FILENAMES:
        loaded[name], raw[name] = _load_npy_member(
            directory,
            authority.root,
            members[name],
            name=name,
            dtype=dtypes[name],
            shape=shapes[name],
        )
        if not np.array_equal(loaded[name], expected_arrays[name]):
            raise Stage2RC5SourceExactCurveBankV1Error(
                f"persisted {name} differs from score+mask exact replay"
            )
    expected_manifest = _manifest_payload(
        authority, material, array_bytes=raw
    )
    if _plain(manifest) != expected_manifest:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source exact-curve manifest differs from causal replay"
        )
    return material, loaded, raw, expected_manifest


def _capability_state(
    *,
    commit_sha256: str,
    manifest: Mapping[str, Any],
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    arrays: Mapping[str, np.ndarray],
    curve_bank: PerImageExactEventCurveBank,
) -> str:
    return _json_sha(
        {
            "algorithm": CAPABILITY_STATE_ALGORITHM,
            "commit_sha256": commit_sha256,
            "manifest_identity_sha256": manifest["manifest_identity_sha256"],
            "score_attestation_sha256": score_bundle.attestation_sha256,
            "curve_bank_id": curve_bank.bank_id,
            "arrays": {
                name: hashlib.sha256(
                    np.ascontiguousarray(value).tobytes(order="C")
                ).hexdigest()
                for name, value in arrays.items()
            },
        }
    )


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5SourceExactCurveBankV1:
    path: Path
    commit_path: Path
    commit_sha256: str
    manifest: Mapping[str, Any]
    score_bundle: VerifiedStage2RC5ScoreBundleV2
    role: str
    source_domain: str
    ordered_image_identities: tuple[str, ...]
    curves_in_record_order: tuple[PerImageExactEventCurve, ...]
    curve_bank: PerImageExactEventCurveBank
    curve_offsets: np.ndarray
    curve_thresholds: np.ndarray
    curve_false_positive_pixels: np.ndarray
    curve_matched_objects: np.ndarray
    boundary_values: Mapping[str, frozenset[str]]
    capability_state_sha256: str
    capability_schema: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("source exact-curve banks are public-verifier-issued only")


def assert_verified_stage2_rc5_source_exact_curve_bank_v1(
    value: object,
) -> VerifiedStage2RC5SourceExactCurveBankV1:
    if (
        type(value) is not VerifiedStage2RC5SourceExactCurveBankV1
        or getattr(value, "_capability", None) is not _CAPABILITY
        or getattr(value, "capability_schema", None) != CAPABILITY_SCHEMA
    ):
        raise TypeError(
            "a verifier-issued VerifiedStage2RC5SourceExactCurveBankV1 is required"
        )
    assert_verified_stage2_rc5_score_bundle_v2(value.score_bundle)
    bank = assert_per_image_exact_event_curve_bank(value.curve_bank)
    if value.curves_in_record_order != tuple(
        bank.curve_for_identity(identity)
        for identity in value.ordered_image_identities
    ):
        raise TypeError("source exact-curve bank ordered curve index drifted")
    arrays = {
        "offsets": value.curve_offsets,
        "thresholds": value.curve_thresholds,
        "false_positive_pixels": value.curve_false_positive_pixels,
        "matched_objects": value.curve_matched_objects,
    }
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.flags.writeable or not array.flags.owndata:
            raise TypeError(f"source exact-curve capability {name} is not owned/read-only")
    state = _capability_state(
        commit_sha256=value.commit_sha256,
        manifest=value.manifest,
        score_bundle=value.score_bundle,
        arrays=arrays,
        curve_bank=bank,
    )
    if state != value.capability_state_sha256:
        raise TypeError("source exact-curve capability retained-token state drifted")
    return value


def _issue(
    *,
    directory: Path,
    commit_path: Path,
    commit_sha256: str,
    manifest: Mapping[str, Any],
    authority: _Authority,
    material: _Materialized,
    arrays: Mapping[str, np.ndarray],
) -> VerifiedStage2RC5SourceExactCurveBankV1:
    bank = assert_per_image_exact_event_curve_bank(material.curve_bank)
    identities = tuple(curve.image_identity_sha256 for curve in material.curves)
    boundaries = MappingProxyType(
        {
            field: frozenset(str(item) for item in values)
            for field, values in manifest["identity_boundary_values"].items()
        }
    )
    state = _capability_state(
        commit_sha256=commit_sha256,
        manifest=manifest,
        score_bundle=authority.score_bundle,
        arrays=arrays,
        curve_bank=bank,
    )
    value = object.__new__(VerifiedStage2RC5SourceExactCurveBankV1)
    fields = {
        "path": directory,
        "commit_path": commit_path,
        "commit_sha256": commit_sha256,
        "manifest": _freeze(manifest),
        "score_bundle": authority.score_bundle,
        "role": authority.role,
        "source_domain": authority.source_domain,
        "ordered_image_identities": identities,
        "curves_in_record_order": material.curves,
        "curve_bank": bank,
        "curve_offsets": arrays["offsets"],
        "curve_thresholds": arrays["thresholds"],
        "curve_false_positive_pixels": arrays["false_positive_pixels"],
        "curve_matched_objects": arrays["matched_objects"],
        "boundary_values": boundaries,
        "capability_state_sha256": state,
        "capability_schema": CAPABILITY_SCHEMA,
        "_capability": _CAPABILITY,
    }
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    return assert_verified_stage2_rc5_source_exact_curve_bank_v1(value)


def build_and_publish_stage2_rc5_source_exact_curve_bank_v1(
    *,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    dataset_directory: str | Path,
    output_directory: str | Path,
    mask_folder: str = "masks",
    repository_root: str | Path | None = None,
) -> VerifiedStage2RC5SourceExactCurveBankV1:
    """Materialise every source record and publish COMMIT last."""

    authority = _prepare_authority(score_bundle, repository_root)
    material = _materialize(authority, dataset_directory, mask_folder)
    output = _output_directory(output_directory, authority.root)
    arrays = {
        "offsets": material.offsets,
        "thresholds": material.thresholds,
        "false_positive_pixels": material.false_positive_pixels,
        "matched_objects": material.matched_objects,
    }
    array_bytes = {name: _npy_bytes(value) for name, value in arrays.items()}
    manifest = _manifest_payload(authority, material, array_bytes=array_bytes)
    manifest_data = _canonical(manifest, newline=True)
    commit = _commit_payload(authority, manifest, manifest_data)
    commit_data = _canonical(commit, newline=True)
    commit_sha = hashlib.sha256(commit_data).hexdigest()
    commit_path = output / COMMIT_FILENAME
    wrote_commit = False
    try:
        for name, data in material.label_files:
            _write_exclusive(output / name, data)
        for name, data in array_bytes.items():
            _write_exclusive(output / ARRAY_FILENAMES[name], data)
        _write_exclusive(output / MANIFEST_FILENAME, manifest_data)
        _fsync_directory(output)

        # Re-open every authority and recompute every curve immediately before
        # the sole file that makes this directory authoritative.
        precommit = _prepare_authority(authority.score_bundle, authority.root)
        precommit_manifest = _strict_json(
            _stable_bytes(
                output / MANIFEST_FILENAME,
                authority.root,
                "precommit manifest",
            ),
            "precommit manifest",
        )
        _verify_members_and_recompute(output, precommit, precommit_manifest)
        _write_exclusive(commit_path, commit_data)
        wrote_commit = True
        _fsync_directory(output)
        return verify_stage2_rc5_source_exact_curve_bank_v1(
            commit_path,
            commit_sha,
            score_bundle=precommit.score_bundle,
            repository_root=precommit.root,
        )
    except BaseException:
        if wrote_commit and commit_path.exists() and not commit_path.is_symlink():
            try:
                if _stable_bytes(
                    commit_path, authority.root, "failed bank commit"
                ) == commit_data:
                    commit_path.unlink()
            except (OSError, RuntimeError, Stage2RC5SourceExactCurveBankV1Error):
                pass
        raise


def _commit_path(value: str | Path, root: Path) -> Path:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    _reject_symlink_components(candidate, root, "bank commit")
    if candidate.name != COMMIT_FILENAME:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source exact-curve bank commit filename mismatch"
        )
    return candidate


def verify_stage2_rc5_source_exact_curve_bank_v1(
    commit_path: str | Path,
    expected_commit_sha256: str,
    *,
    score_bundle: VerifiedStage2RC5ScoreBundleV2,
    repository_root: str | Path | None = None,
) -> VerifiedStage2RC5SourceExactCurveBankV1:
    """Fresh-replay score+mask->label->curve and issue an owned capability."""

    authority = _prepare_authority(score_bundle, repository_root)
    expected = _sha(expected_commit_sha256, "expected_commit_sha256")
    commit_file = _commit_path(commit_path, authority.root)
    commit_data = _stable_bytes(commit_file, authority.root, "bank commit")
    if hashlib.sha256(commit_data).hexdigest() != expected:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source exact-curve bank external commit SHA mismatch"
        )
    commit = _strict_json(commit_data, "bank commit")
    directory = commit_file.parent
    manifest_ref = commit.get("manifest")
    if not isinstance(manifest_ref, Mapping) or set(manifest_ref) != {
        "path", "sha256", "identity_sha256"
    } or manifest_ref["path"] != MANIFEST_FILENAME:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "bank commit manifest binding mismatch"
        )
    manifest_data = _stable_bytes(
        directory / MANIFEST_FILENAME,
        authority.root,
        "source exact-curve manifest",
        expected_sha256=str(manifest_ref["sha256"]),
    )
    manifest = _strict_json(manifest_data, "source exact-curve manifest")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("manifest_identity_sha256")
        != _self_hash(manifest, "manifest_identity_sha256")
        or manifest_ref["identity_sha256"]
        != manifest.get("manifest_identity_sha256")
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source exact-curve manifest identity mismatch"
        )
    expected_commit = _commit_payload(authority, manifest, manifest_data)
    if commit != expected_commit:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "bank commit differs from current score authority"
        )
    if commit.get("commit_identity_sha256") != _self_hash(
        commit, "commit_identity_sha256"
    ):
        raise Stage2RC5SourceExactCurveBankV1Error(
            "bank commit identity mismatch"
        )

    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise Stage2RC5SourceExactCurveBankV1Error("manifest records are absent")
    label_names = {
        _relative_path(record.get("label_file"), f"records[{index}].label_file")
        for index, record in enumerate(raw_records)
        if isinstance(record, Mapping)
    }
    expected_files = {
        MANIFEST_FILENAME,
        COMMIT_FILENAME,
        *ARRAY_FILENAMES.values(),
        *label_names,
    }
    actual_files = {item.name for item in directory.iterdir()}
    if actual_files != expected_files:
        raise Stage2RC5SourceExactCurveBankV1Error(
            "source exact-curve bank file closure mismatch"
        )
    material, arrays, _raw, _expected_manifest = _verify_members_and_recompute(
        directory, authority, manifest
    )
    return _issue(
        directory=directory,
        commit_path=commit_file,
        commit_sha256=expected,
        manifest=manifest,
        authority=authority,
        material=material,
        arrays=arrays,
    )


def replay_verified_stage2_rc5_source_exact_curve_bank_v1(
    value: VerifiedStage2RC5SourceExactCurveBankV1,
) -> VerifiedStage2RC5SourceExactCurveBankV1:
    """Reject retained-token drift, then replay every persistent authority."""

    verified = assert_verified_stage2_rc5_source_exact_curve_bank_v1(value)
    return verify_stage2_rc5_source_exact_curve_bank_v1(
        verified.commit_path,
        verified.commit_sha256,
        score_bundle=verified.score_bundle,
        repository_root=verified.score_bundle.score_manifest_metadata.repository_root,
    )


__all__ = [
    "ARRAY_FILENAMES",
    "CAPABILITY_SCHEMA",
    "COMMIT_FILENAME",
    "COMMIT_SCHEMA",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "SOURCE_ROLES",
    "Stage2RC5SourceExactCurveBankV1Error",
    "VerifiedStage2RC5SourceExactCurveBankV1",
    "assert_verified_stage2_rc5_source_exact_curve_bank_v1",
    "build_and_publish_stage2_rc5_source_exact_curve_bank_v1",
    "replay_verified_stage2_rc5_source_exact_curve_bank_v1",
    "verify_stage2_rc5_source_exact_curve_bank_v1",
]
