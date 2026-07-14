"""Content-addressed dataset and score-export identity helpers.

The protocol must distinguish a real imaging dataset from a user-provided
logical domain label.  The dataset digest below is therefore rooted in the
relative paths and bytes under ``images/``; the absolute dataset path,
directory inode and timestamps are deliberately excluded.  Renaming or
copying a dataset directory consequently preserves its identity.  Labels are
bound separately by the selected split/artifacts and are intentionally not
read while establishing detector-source provenance.

All sequence hashes use length-prefixed UTF-8 fields.  This prevents ambiguous
concatenations (for example, ``["ab", "c"]`` versus ``["a", "bc"]``) and
makes the algorithms straightforward to reimplement outside Python.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .split_utils import read_split_entries, sample_id_from_entry


DATASET_IDENTITY_ALGORITHM = "sha256-relative-path-content-v1"
IMAGE_CONTENT_LEAF_ALGORITHM = "sha256-image-file-bytes-leaf-multiset-v1"
IMAGE_CONTENT_LEAF_SET_ALGORITHM = (
    "sha256-length-prefixed-sorted-image-content-leaves-v1"
)
ORDERED_SAMPLE_IDS_ALGORITHM = "sha256-length-prefixed-sample-ids-v1"
TRAINING_ARTIFACT_ALGORITHM = (
    "sha256-length-prefixed-ordered-sample-image-mask-content-v1"
)
SPLIT_IMAGE_ARTIFACT_ALGORITHM = (
    "sha256-length-prefixed-ordered-sample-image-content-v1"
)
SCORE_MANIFEST_CONTENT_ALGORITHM = (
    "sha256-length-prefixed-image-score-gray-v1"
)
DATASET_RECORD_SCHEMA_VERSION = 2
DATASET_IDENTITY_FOLDERS = ("images",)
_HASH_BUFFER_BYTES = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one file, detecting changes during the read.

    File size, timestamps and inode/device are compared before and after the
    streaming read.  These values are *not* included in the digest, so copies
    remain equivalent; they are used only to fail closed if a producer mutates
    an input while its identity is being computed.
    """

    resolved = Path(path).expanduser().resolve()
    before = _stat_signature(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_BUFFER_BYTES), b""):
            digest.update(chunk)
    after = _stat_signature(resolved)
    if after != before:
        raise RuntimeError(f"File changed while hashing: {resolved}")
    return digest.hexdigest()


def dataset_identity(
    dataset_dir: str | Path,
    *,
    folders: Sequence[str] = DATASET_IDENTITY_FOLDERS,
) -> dict[str, object]:
    """Fingerprint dataset image content independently of its root path.

    Every regular file below the requested folders contributes its folder-
    relative POSIX path, byte count and content SHA-256.  Files are streamed
    once in sorted relative-path order.  A second directory/stat snapshot
    catches additions, removals and in-place changes during the scan rather
    than emitting a mixed, non-atomic identity.
    """

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    normalised_folders = _normalise_folders(folders)
    files_before = _dataset_file_snapshot(root, normalised_folders)
    if not files_before:
        raise ValueError(
            f"No regular files found under dataset folders {normalised_folders}: {root}"
        )

    digest = hashlib.sha256()
    _update_frame(digest, DATASET_IDENTITY_ALGORITHM)
    total_bytes = 0
    image_content_leaves: list[str] = []
    for relative_path, (path, stat_signature) in files_before.items():
        content_sha256 = sha256_file(path)
        # Catch a file that changed after sha256_file returned but before the
        # whole dataset snapshot was completed.
        if _stat_signature(path) != stat_signature:
            raise RuntimeError(f"Dataset file changed while hashing: {path}")
        size = stat_signature[0]
        _update_frame(digest, relative_path)
        _update_frame(digest, str(size))
        _update_frame(digest, content_sha256)
        total_bytes += size
        image_content_leaves.append(content_sha256)

    files_after = _dataset_file_snapshot(root, normalised_folders)
    before_signatures = {
        relative: signature for relative, (_, signature) in files_before.items()
    }
    after_signatures = {
        relative: signature for relative, (_, signature) in files_after.items()
    }
    if after_signatures != before_signatures:
        raise RuntimeError(f"Dataset changed while hashing: {root}")

    sorted_leaves = sorted(image_content_leaves)
    leaf_set_digest = hashlib.sha256()
    _update_frame(leaf_set_digest, IMAGE_CONTENT_LEAF_SET_ALGORITHM)
    for content_sha256 in sorted_leaves:
        _update_frame(leaf_set_digest, content_sha256)

    return {
        "dataset_identity_algorithm": DATASET_IDENTITY_ALGORITHM,
        "dataset_identity_sha256": digest.hexdigest(),
        "dataset_num_files": len(files_before),
        "dataset_num_bytes": total_bytes,
        "dataset_identity_folders": list(normalised_folders),
        # The dataset identity above intentionally includes relative paths.
        # This second, path-independent multiset is the contamination guard:
        # copying/renaming a file cannot hide shared image bytes.
        "image_content_leaf_algorithm": IMAGE_CONTENT_LEAF_ALGORITHM,
        "image_content_sha256_leaves": sorted_leaves,
        "image_content_leaf_set_algorithm": IMAGE_CONTENT_LEAF_SET_ALGORITHM,
        "image_content_leaf_set_sha256": leaf_set_digest.hexdigest(),
    }


def ordered_sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    """Hash an ordered sample-ID sequence with unambiguous framing."""

    digest = hashlib.sha256()
    _update_frame(digest, ORDERED_SAMPLE_IDS_ALGORITHM)
    count = 0
    for raw_sample_id in sample_ids:
        sample_id = str(raw_sample_id).replace("\\", "/").strip()
        if not sample_id:
            raise ValueError("ordered sample IDs must be non-empty strings")
        _update_frame(digest, sample_id)
        count += 1
    if count == 0:
        raise ValueError("ordered sample IDs must not be empty")
    return digest.hexdigest()


def build_dataset_record(
    dataset_dir: str | Path,
    split_file: str | Path,
    sample_ids: Sequence[str],
    *,
    source_name: str | None = None,
    training_artifacts: Sequence[tuple[str | Path, str | Path]] | None = None,
) -> dict[str, object]:
    """Build the checkpoint/manifest record for one concrete dataset split.

    ``training_artifacts`` is required for detector sources and must align
    one-to-one with the selected split order.  Only those concrete image/mask
    paths are opened; the helper never enumerates a mask directory.  Target
    score exports omit it and therefore remain label-free.
    """

    ids = [str(value).replace("\\", "/").strip() for value in sample_ids]
    if not ids or any(not value for value in ids):
        raise ValueError("sample_ids must contain non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids contain duplicates")
    split_path = Path(split_file).expanduser().resolve()
    split_sha256_before = sha256_file(split_path)
    split_ids = [
        sample_id_from_entry(entry)
        for entry in read_split_entries(split_path)
    ]
    if ids != split_ids:
        raise ValueError(
            "sample_ids do not exactly match the selected split's ordered IDs"
        )
    identity = dataset_identity(dataset_dir)
    split_image_digest = hashlib.sha256()
    _update_frame(split_image_digest, SPLIT_IMAGE_ARTIFACT_ALGORITHM)
    split_image_items: list[dict[str, str]] = []
    # Resolve only split-selected images.  This binds each ordered ID to raw
    # image bytes while preserving the label-free target-record path.
    from .split_utils import resolve_sample_file

    for sample_id, entry in zip(ids, read_split_entries(split_path)):
        image_path = resolve_sample_file(
            dataset_dir,
            "images",
            entry,
            kind="image",
        )
        image_sha256 = sha256_file(image_path)
        _update_frame(split_image_digest, sample_id)
        _update_frame(split_image_digest, image_sha256)
        split_image_items.append(
            {"sample_id": sample_id, "image_sha256": image_sha256}
        )
    split_sha256_after = sha256_file(split_path)
    if split_sha256_after != split_sha256_before:
        raise RuntimeError(
            f"Split file changed while building dataset record: {split_path}"
        )
    record: dict[str, object] = {
        "record_schema_version": DATASET_RECORD_SCHEMA_VERSION,
        **identity,
        "split_sha256": split_sha256_after,
        "ordered_sample_ids_algorithm": ORDERED_SAMPLE_IDS_ALGORITHM,
        "ordered_sample_ids_sha256": ordered_sample_ids_sha256(ids),
        "num_samples": len(ids),
        "split_image_artifact_algorithm": SPLIT_IMAGE_ARTIFACT_ALGORITHM,
        "split_image_artifact_sha256": split_image_digest.hexdigest(),
        "split_image_artifact_items": split_image_items,
    }
    if source_name is not None:
        logical_name = str(source_name).strip()
        if not logical_name:
            raise ValueError("source_name must be non-empty when supplied")
        record["source_name"] = logical_name
        if training_artifacts is None:
            raise ValueError(
                "detector source records require selected training_artifacts"
            )
    if training_artifacts is not None:
        if len(training_artifacts) != len(ids):
            raise ValueError(
                "training_artifacts must align one-to-one with ordered sample IDs"
            )
        artifact_digest = hashlib.sha256()
        _update_frame(artifact_digest, TRAINING_ARTIFACT_ALGORITHM)
        artifact_items: list[dict[str, str]] = []
        for index, (sample_id, paths) in enumerate(zip(ids, training_artifacts)):
            if not isinstance(paths, (tuple, list)) or len(paths) != 2:
                raise TypeError(
                    f"training_artifacts[{index}] must be an (image, mask) pair"
                )
            image_path = Path(paths[0]).expanduser().resolve()
            mask_path = Path(paths[1]).expanduser().resolve()
            image_sha256 = sha256_file(image_path)
            mask_sha256 = sha256_file(mask_path)
            _update_frame(artifact_digest, sample_id)
            _update_frame(artifact_digest, image_sha256)
            _update_frame(artifact_digest, mask_sha256)
            artifact_items.append(
                {
                    "sample_id": sample_id,
                    "image_sha256": image_sha256,
                    "mask_sha256": mask_sha256,
                }
            )
        record.update(
            {
                "training_artifact_algorithm": TRAINING_ARTIFACT_ALGORITHM,
                "training_artifact_sha256": artifact_digest.hexdigest(),
                "training_artifact_num_samples": len(artifact_items),
                "training_artifact_items": artifact_items,
            }
        )
    validate_dataset_record(
        record,
        require_source_name=source_name is not None,
        require_training_artifact=source_name is not None,
    )
    return record


def validate_dataset_record(
    raw_record: object,
    *,
    require_source_name: bool = False,
    require_training_artifact: bool | None = None,
) -> dict[str, object]:
    """Validate and normalise an externally loaded dataset record."""

    if not isinstance(raw_record, Mapping):
        raise TypeError("dataset identity record must be a mapping")
    required = {
        "record_schema_version",
        "dataset_identity_algorithm",
        "dataset_identity_sha256",
        "dataset_num_files",
        "dataset_num_bytes",
        "dataset_identity_folders",
        "image_content_leaf_algorithm",
        "image_content_sha256_leaves",
        "image_content_leaf_set_algorithm",
        "image_content_leaf_set_sha256",
        "split_sha256",
        "ordered_sample_ids_algorithm",
        "ordered_sample_ids_sha256",
        "num_samples",
        "split_image_artifact_algorithm",
        "split_image_artifact_sha256",
        "split_image_artifact_items",
    }
    if require_source_name:
        required.add("source_name")
    if require_training_artifact is None:
        require_training_artifact = require_source_name
    if require_training_artifact:
        required.update(
            {
                "training_artifact_algorithm",
                "training_artifact_sha256",
                "training_artifact_num_samples",
                "training_artifact_items",
            }
        )
    missing = required.difference(raw_record)
    if missing:
        raise ValueError(f"dataset identity record is missing fields: {sorted(missing)}")
    if int(raw_record["record_schema_version"]) != DATASET_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported dataset identity record schema version")
    if raw_record["dataset_identity_algorithm"] != DATASET_IDENTITY_ALGORITHM:
        raise ValueError("unsupported dataset identity algorithm")
    if raw_record["ordered_sample_ids_algorithm"] != ORDERED_SAMPLE_IDS_ALGORITHM:
        raise ValueError("unsupported ordered sample-ID algorithm")
    _require_sha256(raw_record["dataset_identity_sha256"], "dataset_identity_sha256")
    if raw_record["image_content_leaf_algorithm"] != IMAGE_CONTENT_LEAF_ALGORITHM:
        raise ValueError("unsupported image-content leaf algorithm")
    if (
        raw_record["image_content_leaf_set_algorithm"]
        != IMAGE_CONTENT_LEAF_SET_ALGORITHM
    ):
        raise ValueError("unsupported image-content leaf-set algorithm")
    _require_sha256(
        raw_record["image_content_leaf_set_sha256"],
        "image_content_leaf_set_sha256",
    )
    _require_sha256(raw_record["split_sha256"], "split_sha256")
    _require_sha256(
        raw_record["ordered_sample_ids_sha256"],
        "ordered_sample_ids_sha256",
    )
    num_files = _positive_int(raw_record["dataset_num_files"], "dataset_num_files")
    num_bytes = _nonnegative_int(raw_record["dataset_num_bytes"], "dataset_num_bytes")
    num_samples = _positive_int(raw_record["num_samples"], "num_samples")
    folders = raw_record["dataset_identity_folders"]
    if not isinstance(folders, (list, tuple)):
        raise TypeError("dataset_identity_folders must be an ordered list")
    normalised_folders = list(_normalise_folders([str(value) for value in folders]))
    if tuple(normalised_folders) != DATASET_IDENTITY_FOLDERS:
        raise ValueError(
            "dataset identity record folders must be exactly ['images'] for "
            "this schema"
        )
    raw_leaves = raw_record["image_content_sha256_leaves"]
    if not isinstance(raw_leaves, (list, tuple)) or not raw_leaves:
        raise ValueError("image_content_sha256_leaves must be a non-empty list")
    leaves = [
        _require_sha256(value, f"image_content_sha256_leaves[{index}]")
        for index, value in enumerate(raw_leaves)
    ]
    if leaves != sorted(leaves):
        raise ValueError("image_content_sha256_leaves must be sorted")
    if len(leaves) != num_files:
        raise ValueError(
            "image_content_sha256_leaves must contain one leaf per image file"
        )
    leaf_set_digest = hashlib.sha256()
    _update_frame(leaf_set_digest, IMAGE_CONTENT_LEAF_SET_ALGORITHM)
    for leaf in leaves:
        _update_frame(leaf_set_digest, leaf)
    if leaf_set_digest.hexdigest() != str(
        raw_record["image_content_leaf_set_sha256"]
    ).lower():
        raise ValueError("image-content leaf-set SHA-256 mismatch")
    if raw_record["split_image_artifact_algorithm"] != SPLIT_IMAGE_ARTIFACT_ALGORITHM:
        raise ValueError("unsupported split-image artifact algorithm")
    split_image_sha = _require_sha256(
        raw_record["split_image_artifact_sha256"],
        "split_image_artifact_sha256",
    )
    raw_split_image_items = raw_record["split_image_artifact_items"]
    if not isinstance(raw_split_image_items, (list, tuple)) or len(
        raw_split_image_items
    ) != num_samples:
        raise ValueError(
            "split_image_artifact_items must align with selected split count"
        )
    split_image_digest = hashlib.sha256()
    _update_frame(split_image_digest, SPLIT_IMAGE_ARTIFACT_ALGORITHM)
    split_image_items: list[dict[str, str]] = []
    for index, raw_item in enumerate(raw_split_image_items):
        if not isinstance(raw_item, Mapping):
            raise TypeError(f"split_image_artifact_items[{index}] must be a mapping")
        sample_id = str(raw_item.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("split-image artifact sample IDs must be non-empty")
        image_sha = _require_sha256(
            raw_item.get("image_sha256"),
            f"split_image_artifact_items[{index}].image_sha256",
        )
        _update_frame(split_image_digest, sample_id)
        _update_frame(split_image_digest, image_sha)
        split_image_items.append(
            {"sample_id": sample_id, "image_sha256": image_sha}
        )
    if split_image_digest.hexdigest() != split_image_sha:
        raise ValueError("split_image_artifact_sha256 does not match its ordered items")
    if ordered_sample_ids_sha256(
        [item["sample_id"] for item in split_image_items]
    ) != str(raw_record["ordered_sample_ids_sha256"]).lower():
        raise ValueError("split-image artifact order does not match selected split IDs")
    split_image_multiplicity = Counter(
        item["image_sha256"] for item in split_image_items
    )
    leaf_multiplicity = Counter(leaves)
    if any(
        count > leaf_multiplicity.get(digest, 0)
        for digest, count in split_image_multiplicity.items()
    ):
        raise ValueError(
            "split-image artifact SHA-256 values are absent from or exceed the "
            "dataset image-content leaf multiset"
        )
    source_name: str | None = None
    if "source_name" in raw_record:
        source_name = str(raw_record["source_name"]).strip()
        if not source_name:
            raise ValueError("source_name must be non-empty")

    result: dict[str, object] = {
        "record_schema_version": DATASET_RECORD_SCHEMA_VERSION,
        "dataset_identity_algorithm": DATASET_IDENTITY_ALGORITHM,
        "dataset_identity_sha256": str(raw_record["dataset_identity_sha256"]).lower(),
        "dataset_num_files": num_files,
        "dataset_num_bytes": num_bytes,
        "dataset_identity_folders": normalised_folders,
        "image_content_leaf_algorithm": IMAGE_CONTENT_LEAF_ALGORITHM,
        "image_content_sha256_leaves": leaves,
        "image_content_leaf_set_algorithm": IMAGE_CONTENT_LEAF_SET_ALGORITHM,
        "image_content_leaf_set_sha256": str(
            raw_record["image_content_leaf_set_sha256"]
        ).lower(),
        "split_sha256": str(raw_record["split_sha256"]).lower(),
        "ordered_sample_ids_algorithm": ORDERED_SAMPLE_IDS_ALGORITHM,
        "ordered_sample_ids_sha256": str(
            raw_record["ordered_sample_ids_sha256"]
        ).lower(),
        "num_samples": num_samples,
        "split_image_artifact_algorithm": SPLIT_IMAGE_ARTIFACT_ALGORITHM,
        "split_image_artifact_sha256": split_image_sha,
        "split_image_artifact_items": split_image_items,
    }
    if source_name is not None:
        result["source_name"] = source_name
    if require_training_artifact or "training_artifact_sha256" in raw_record:
        if raw_record.get("training_artifact_algorithm") != TRAINING_ARTIFACT_ALGORITHM:
            raise ValueError("unsupported training-artifact algorithm")
        training_sha = _require_sha256(
            raw_record.get("training_artifact_sha256"),
            "training_artifact_sha256",
        )
        training_count = _positive_int(
            raw_record.get("training_artifact_num_samples"),
            "training_artifact_num_samples",
        )
        raw_items = raw_record.get("training_artifact_items")
        if not isinstance(raw_items, (list, tuple)) or len(raw_items) != training_count:
            raise ValueError(
                "training_artifact_items must align with training_artifact_num_samples"
            )
        if training_count != num_samples:
            raise ValueError(
                "training artifact sample count must match the selected split"
            )
        artifact_digest = hashlib.sha256()
        _update_frame(artifact_digest, TRAINING_ARTIFACT_ALGORITHM)
        training_items: list[dict[str, str]] = []
        seen_training_ids: set[str] = set()
        for index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, Mapping):
                raise TypeError(f"training_artifact_items[{index}] must be a mapping")
            sample_id = str(raw_item.get("sample_id", "")).strip()
            if not sample_id or sample_id in seen_training_ids:
                raise ValueError(
                    "training artifact sample IDs must be non-empty and unique"
                )
            seen_training_ids.add(sample_id)
            image_sha = _require_sha256(
                raw_item.get("image_sha256"),
                f"training_artifact_items[{index}].image_sha256",
            )
            mask_sha = _require_sha256(
                raw_item.get("mask_sha256"),
                f"training_artifact_items[{index}].mask_sha256",
            )
            _update_frame(artifact_digest, sample_id)
            _update_frame(artifact_digest, image_sha)
            _update_frame(artifact_digest, mask_sha)
            training_items.append(
                {
                    "sample_id": sample_id,
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                }
            )
        if artifact_digest.hexdigest() != training_sha:
            raise ValueError("training_artifact_sha256 does not match its ordered items")
        if ordered_sample_ids_sha256(
            [item["sample_id"] for item in training_items]
        ) != str(raw_record["ordered_sample_ids_sha256"]).lower():
            raise ValueError(
                "training artifact item order does not match selected split IDs"
            )
        training_image_multiplicity = Counter(
            item["image_sha256"] for item in training_items
        )
        outside_or_overused = {
            digest: count
            for digest, count in training_image_multiplicity.items()
            if count > leaf_multiplicity.get(digest, 0)
        }
        if outside_or_overused:
            raise ValueError(
                "training artifact image SHA-256 values are absent from or exceed "
                "the dataset image-content leaf multiset"
            )
        if [item["image_sha256"] for item in training_items] != [
            item["image_sha256"] for item in split_image_items
        ]:
            raise ValueError(
                "training artifact images do not match the ordered split-image artifact"
            )
        result.update(
            {
                "training_artifact_algorithm": TRAINING_ARTIFACT_ALGORITHM,
                "training_artifact_sha256": training_sha,
                "training_artifact_num_samples": training_count,
                "training_artifact_items": training_items,
            }
        )
    return result


def score_manifest_content_sha256(items: Sequence[Mapping[str, object]]) -> str:
    """Aggregate ordered score-export item identities into one digest.

    Only the protocol-fixed identity triplet contributes: ``image_id``,
    ``score_file_sha256`` and ``gray_file_sha256``.  Paths and metadata may be
    relocated without changing this content identity.
    """

    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("score manifest items must be a non-empty ordered list")
    digest = hashlib.sha256()
    _update_frame(digest, SCORE_MANIFEST_CONTENT_ALGORITHM)
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"score manifest item {index} must be a mapping")
        missing = {
            "image_id",
            "score_file_sha256",
            "gray_file_sha256",
        }.difference(item)
        if missing:
            raise ValueError(
                f"score manifest item {index} is missing fields: {sorted(missing)}"
            )
        image_id = str(item["image_id"]).strip()
        if not image_id:
            raise ValueError(f"score manifest item {index} has an empty image_id")
        if image_id in seen:
            raise ValueError(f"duplicate score manifest image_id: {image_id!r}")
        seen.add(image_id)
        score_sha = _require_sha256(
            item["score_file_sha256"],
            f"items[{index}].score_file_sha256",
        )
        gray_sha = _require_sha256(
            item["gray_file_sha256"],
            f"items[{index}].gray_file_sha256",
        )
        _update_frame(digest, image_id)
        _update_frame(digest, score_sha)
        _update_frame(digest, gray_sha)
    return digest.hexdigest()


def _normalise_folders(folders: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for raw_folder in folders:
        folder = str(raw_folder).replace("\\", "/").strip("/").strip()
        if not folder or folder in {".", ".."} or "/" in folder:
            raise ValueError(
                "dataset identity folders must be simple non-empty directory names"
            )
        if folder in result:
            raise ValueError(f"duplicate dataset identity folder: {folder!r}")
        result.append(folder)
    if not result:
        raise ValueError("at least one dataset identity folder is required")
    return tuple(result)


def _dataset_file_snapshot(
    root: Path,
    folders: Sequence[str],
) -> dict[str, tuple[Path, tuple[int, int, int, int, int]]]:
    snapshot: dict[str, tuple[Path, tuple[int, int, int, int, int]]] = {}
    for folder in folders:
        folder_root = root / folder
        if not folder_root.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder_root}")
        for path in folder_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in snapshot:
                raise RuntimeError(f"duplicate dataset relative path: {relative}")
            snapshot[relative] = (path.resolve(), _stat_signature(path.resolve()))
    return dict(sorted(snapshot.items(), key=lambda item: item[0].encode("utf-8")))


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(f"Expected a regular file: {path}")
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_dev),
        int(stat.st_ino),
    )


def _update_frame(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _require_sha256(value: object, field: str) -> str:
    rendered = str(value).lower()
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{field} must be a lowercase/uppercase SHA-256 hex digest")
    return rendered


def _positive_int(value: object, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must be an integer") from error
    if result < 0 or result != value:
        raise ValueError(f"{field} must be a non-negative integer")
    return result
