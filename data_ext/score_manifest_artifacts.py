"""Fail-closed verification for exported native-resolution score manifests.

The exporter records two identities for every sample: the compressed score
artifact and the original image bytes.  Consumers use this module to validate
those identities, the ordered manifest aggregate, and the native-resolution
NPZ contract before loading any selected score map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .dataset_identity import (
    ORDERED_SAMPLE_IDS_ALGORITHM,
    SCORE_MANIFEST_CONTENT_ALGORITHM,
    SPLIT_IMAGE_ARTIFACT_ALGORITHM,
    ordered_sample_ids_sha256,
    score_manifest_content_sha256,
    sha256_file,
    validate_dataset_record,
)
from .split_utils import read_split_entries, resolve_sample_file, sample_id_from_entry


SIGMOID_SCORE_TYPE = "sigmoid_probability"
STRICT_THRESHOLD_SEMANTICS = "prediction = probability > threshold"
SCORE_MANIFEST_SCHEMA_VERSION = 3
SPLIT_CONTRACT_SCHEMA_VERSION = 1
OFFICIAL_SPLIT_ROLES = ("official_train", "official_test")


@dataclass(frozen=True)
class VerifiedScoreItem:
    """Resolved, content-verified paths for one manifest item."""

    manifest_index: int
    image_id: str
    record: Mapping[str, Any]
    score_path: Path
    gray_path: Path
    original_hw: tuple[int, int]


@dataclass(frozen=True)
class VerifiedScoreManifest:
    """A score manifest whose aggregate and selected artifacts were checked."""

    path: Path
    payload: Mapping[str, Any]
    items: tuple[Mapping[str, Any], ...]
    selected_items: tuple[VerifiedScoreItem, ...]
    content_sha256: str
    manifest_sha256: str
    split_contract: Mapping[str, Any] | None
    split_role: str | None
    legacy_final_evaluation_only: bool


def verify_score_manifest_artifacts(
    manifest_path: str | Path,
    *,
    image_ids: Sequence[str] | None = None,
    require_mask: bool = False,
    require_native_contract: bool = True,
    verify_artifact_bytes: bool = True,
    allow_legacy_combined_diagnostic: bool = False,
    required_split_role: str | None = None,
) -> VerifiedScoreManifest:
    """Verify a score manifest and the selected/relevant sample artifacts.

    The aggregate digest is always checked for the full ordered item list.
    File bytes and native NPZ/image dimensions are checked for all items when
    ``image_ids`` is omitted, or only for the explicitly selected IDs.  Set
    ``verify_artifact_bytes=False`` to validate only the manifest, ordered
    aggregate and item metadata.  Online consumers use that mode to discover
    the query IDs without touching query artifacts, then make a second call
    for the selected context IDs only.

    Calibration episode consumers must pass
    ``required_split_role="official_train"``.  A schema-v2/role-less manifest
    is accepted only when that gate is omitted, preserving historical final
    test evaluation without allowing it into new calibration episodes.
    """

    if require_mask and not allow_legacy_combined_diagnostic:
        raise ValueError(
            "embedded score-NPZ masks are forbidden for verified artifacts; "
            "attach labels through a separate label manifest. Set "
            "allow_legacy_combined_diagnostic=True only for non-claim-bearing diagnostics."
        )
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"score manifest does not exist: {path}")
    incomplete_marker = path.parent / ".export_incomplete"
    if incomplete_marker.exists():
        raise RuntimeError(
            f"score export is incomplete and unsafe to consume: {path.parent}"
        )

    manifest_sha256_before = sha256_file(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"score manifest must contain a JSON object: {path}")
    manifest_sha256 = sha256_file(path)
    if manifest_sha256 != manifest_sha256_before:
        raise RuntimeError(f"score manifest changed while being verified: {path}")

    raw_items = payload.get("items", payload.get("records"))
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(
            f"score manifest requires a non-empty items/records list: {path}"
        )
    items: tuple[Mapping[str, Any], ...] = tuple(
        _require_item_mapping(item, index, path)
        for index, item in enumerate(raw_items)
    )
    _verify_num_images(payload, len(items), path)

    calculated_content_sha256 = score_manifest_content_sha256(items)
    algorithm = payload.get("content_sha256_algorithm")
    if algorithm != SCORE_MANIFEST_CONTENT_ALGORITHM:
        raise ValueError(
            "score manifest content_sha256_algorithm mismatch: "
            f"{algorithm!r} != {SCORE_MANIFEST_CONTENT_ALGORITHM!r}"
        )
    declared_content_sha256 = _sha256_value(
        payload.get("content_sha256"), "score manifest content_sha256"
    )
    if declared_content_sha256 != calculated_content_sha256:
        raise ValueError(
            "score manifest content_sha256 does not match its ordered item identities"
        )

    if require_native_contract:
        _verify_native_manifest_contract(
            payload,
            allow_legacy_combined_diagnostic=allow_legacy_combined_diagnostic,
        )

    by_id: dict[str, tuple[int, Mapping[str, Any]]] = {}
    score_files: set[str] = set()
    ordered_ids: list[str] = []
    for index, item in enumerate(items):
        image_id = _nonempty_string(item.get("image_id"), f"items[{index}].image_id")
        if image_id in by_id:
            raise ValueError(f"duplicate score manifest image_id: {image_id!r}")
        score_value = _score_file_value(item, index)
        rendered_score = str(score_value)
        if rendered_score in score_files:
            raise ValueError(
                f"duplicate score-map path in score manifest: {rendered_score!r}"
            )
        score_files.add(rendered_score)
        _explicit_relative_image_value(item, index)
        _parse_hw(item.get("original_hw"), f"items[{index}].original_hw")
        by_id[image_id] = (index, item)
        ordered_ids.append(image_id)

    split_contract = validate_score_split_contract(
        payload,
        items=items,
        manifest_root=path.parent,
        required_split_role=required_split_role,
    )

    if not verify_artifact_bytes:
        if require_mask:
            raise ValueError(
                "require_mask=True is incompatible with verify_artifact_bytes=False"
            )
        selected_ids: list[str] = []
    elif image_ids is None:
        selected_ids = ordered_ids
    else:
        selected_ids = [
            _nonempty_string(value, "selected image_id") for value in image_ids
        ]
        if not selected_ids:
            raise ValueError("selected image IDs must not be empty")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selected image IDs contain duplicates")
        missing = [image_id for image_id in selected_ids if image_id not in by_id]
        if missing:
            raise KeyError(
                f"selected image IDs are absent from score manifest: {missing}"
            )

    selected: list[VerifiedScoreItem] = []
    for image_id in selected_ids:
        index, item = by_id[image_id]
        selected.append(
            _verify_selected_item(
                path,
                payload,
                item,
                index=index,
                image_id=image_id,
                require_mask=require_mask,
                allow_legacy_combined_diagnostic=allow_legacy_combined_diagnostic,
            )
        )

    return VerifiedScoreManifest(
        path=path,
        payload=payload,
        items=items,
        selected_items=tuple(selected),
        content_sha256=calculated_content_sha256,
        manifest_sha256=manifest_sha256,
        split_contract=split_contract,
        split_role=(
            None if split_contract is None else str(split_contract["role"])
        ),
        legacy_final_evaluation_only=split_contract is None,
    )


def validate_score_split_contract(
    payload: Mapping[str, Any],
    *,
    items: Sequence[Mapping[str, Any]],
    manifest_root: str | Path | None = None,
    required_split_role: str | None = None,
) -> dict[str, Any] | None:
    """Validate and normalise a replayable official train/test contract.

    Score-manifest schema v3 requires this contract.  Earlier role-less
    manifests remain readable only when ``required_split_role`` is omitted;
    they are final-test-evaluation-only compatibility artifacts.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("score manifest payload must be a mapping")
    if required_split_role is not None:
        required_split_role = str(required_split_role).strip()
        if required_split_role not in OFFICIAL_SPLIT_ROLES:
            raise ValueError(
                "required_split_role must be one of "
                + ", ".join(OFFICIAL_SPLIT_ROLES)
            )
    raw_schema = payload.get("schema_version")
    if raw_schema is None:
        if "split_contract" in payload:
            raise ValueError(
                "a score manifest declaring split_contract requires schema_version=3"
            )
        if required_split_role is not None:
            raise ValueError(
                "legacy/role-less score manifests are final-test-evaluation-only "
                f"and cannot satisfy required_split_role={required_split_role!r}"
            )
        return None
    schema_version = _exact_integer(
        raw_schema,
        "score manifest schema_version",
        minimum=1,
    )
    if schema_version != SCORE_MANIFEST_SCHEMA_VERSION:
        if "split_contract" in payload:
            raise ValueError(
                "only score manifest schema_version=3 may declare split_contract"
            )
        if required_split_role is not None:
            raise ValueError(
                "legacy/role-less score manifests are final-test-evaluation-only "
                f"and cannot satisfy required_split_role={required_split_role!r}"
            )
        return None

    raw_contract = payload.get("split_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("score manifest schema_version=3 requires split_contract")
    contract = dict(raw_contract)
    contract_schema = _exact_integer(
        contract.get("schema_version"),
        "split_contract.schema_version",
        minimum=1,
    )
    if contract_schema != SPLIT_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported score split_contract schema_version")
    role = _nonempty_string(contract.get("role"), "split_contract.role")
    if role not in OFFICIAL_SPLIT_ROLES:
        raise ValueError(
            "split_contract.role must be 'official_train' or 'official_test'"
        )
    if required_split_role is not None and role != required_split_role:
        raise ValueError(
            f"score manifest split role {role!r} cannot satisfy required "
            f"role {required_split_role!r}"
        )
    if contract.get("ordered_sample_ids_algorithm") != ORDERED_SAMPLE_IDS_ALGORITHM:
        raise ValueError("split_contract ordered_sample_ids_algorithm mismatch")
    if contract.get("split_image_artifact_algorithm") != SPLIT_IMAGE_ARTIFACT_ALGORITHM:
        raise ValueError("split_contract split_image_artifact_algorithm mismatch")

    if manifest_root is None:
        raise ValueError(
            "manifest_root is required to replay schema-v3 split files and image bytes"
        )
    root = Path(manifest_root).expanduser().resolve()
    dataset_dir = _relative_path_value(
        payload.get("dataset_dir"),
        "score manifest dataset_dir",
    )
    dataset_root = _resolve_path(root, dataset_dir)
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"score manifest dataset_dir does not exist: {dataset_root}"
        )
    role_records = {
        official_role: _validate_official_split_record(
            contract,
            official_role,
            manifest_root=root,
            dataset_root=dataset_root,
        )
        for official_role in OFFICIAL_SPLIT_ROLES
    }
    train = role_records["official_train"]
    test = role_records["official_test"]
    id_overlap = sorted(set(train["sample_ids"]) & set(test["sample_ids"]))
    content_overlap = sorted(
        set(train["image_sha256s"]) & set(test["image_sha256s"])
    )
    declared_id_count = _exact_integer(
        contract.get("train_test_id_overlap_count"),
        "split_contract.train_test_id_overlap_count",
        minimum=0,
    )
    declared_content_count = _exact_integer(
        contract.get("train_test_image_content_overlap_count"),
        "split_contract.train_test_image_content_overlap_count",
        minimum=0,
    )
    declared_ids = _string_list(
        contract.get("train_test_id_overlap_ids"),
        "split_contract.train_test_id_overlap_ids",
    )
    declared_content = _sha256_list(
        contract.get("train_test_image_content_overlap_sha256_leaves"),
        "split_contract.train_test_image_content_overlap_sha256_leaves",
    )
    if declared_ids != id_overlap or declared_id_count != len(id_overlap):
        raise ValueError("split_contract train/test sample-ID overlap audit mismatch")
    if (
        declared_content != content_overlap
        or declared_content_count != len(content_overlap)
    ):
        raise ValueError("split_contract train/test image-content overlap audit mismatch")
    if id_overlap or content_overlap:
        raise ValueError(
            "score manifest official train/test splits are not ID/content disjoint"
        )
    if contract.get("disjointness_verified") is not True:
        raise ValueError("split_contract disjointness_verified must be exactly true")

    selected = role_records[role]
    selected_fields = {
        "selected_split_file": selected["split_file"],
        "selected_split_sha256": selected["split_sha256"],
        "selected_num_images": selected["num_images"],
        "selected_ids_sha256": selected["ids_sha256"],
    }
    for field, expected in selected_fields.items():
        if contract.get(field) != expected:
            raise ValueError(
                f"split_contract {field} does not match its declared {role} record"
            )
    if payload.get("split_file") != selected["split_file"]:
        raise ValueError(
            "score manifest split_file does not match split_contract selected split"
        )

    selected_ids = [str(item["sample_id"]) for item in selected["items"]]
    selected_hashes = [str(item["image_sha256"]) for item in selected["items"]]
    manifest_ids = [
        _nonempty_string(item.get("image_id"), f"items[{index}].image_id")
        for index, item in enumerate(items)
    ]
    manifest_hashes = [
        _sha256_value(
            item.get("gray_file_sha256"),
            f"items[{index}].gray_file_sha256",
        )
        for index, item in enumerate(items)
    ]
    if manifest_ids != selected_ids:
        raise ValueError(
            "score manifest item IDs do not exactly match the selected official split"
        )
    if manifest_hashes != selected_hashes:
        raise ValueError(
            "score manifest image hashes do not exactly match the selected official split"
        )

    raw_target_record = payload.get("target_dataset_record")
    if raw_target_record is None:
        raise KeyError("score manifest schema_version=3 is missing target_dataset_record")
    target_record = validate_dataset_record(
        raw_target_record,
        require_source_name=False,
        require_training_artifact=False,
    )
    target_expected = {
        "split_sha256": selected["split_sha256"],
        "num_samples": selected["num_images"],
        "ordered_sample_ids_sha256": selected["ids_sha256"],
        "split_image_artifact_sha256": selected["split_image_artifact_sha256"],
        "split_image_artifact_items": selected["items"],
    }
    for field, expected in target_expected.items():
        if target_record[field] != expected:
            raise ValueError(
                f"target_dataset_record {field} disagrees with split_contract"
            )

    # Consumers may copy this result into episode provenance without retaining
    # attacker-controlled extra fields from the raw JSON mapping.
    normalised: dict[str, Any] = {
        "schema_version": SPLIT_CONTRACT_SCHEMA_VERSION,
        "role": role,
        "ordered_sample_ids_algorithm": ORDERED_SAMPLE_IDS_ALGORITHM,
        "split_image_artifact_algorithm": SPLIT_IMAGE_ARTIFACT_ALGORITHM,
        **selected_fields,
        "train_test_id_overlap_count": 0,
        "train_test_id_overlap_ids": [],
        "train_test_image_content_overlap_count": 0,
        "train_test_image_content_overlap_sha256_leaves": [],
        "disjointness_verified": True,
    }
    for official_role, record in role_records.items():
        normalised.update(
            {
                f"{official_role}_split_file": record["split_file"],
                f"{official_role}_split_sha256": record["split_sha256"],
                f"{official_role}_num_images": record["num_images"],
                f"{official_role}_ids_sha256": record["ids_sha256"],
                f"{official_role}_split_image_artifact_sha256": record[
                    "split_image_artifact_sha256"
                ],
                f"{official_role}_split_image_artifact_items": record["items"],
            }
        )
    return normalised


def _validate_official_split_record(
    contract: Mapping[str, Any],
    role: str,
    *,
    manifest_root: Path | None,
    dataset_root: Path | None,
) -> dict[str, Any]:
    prefix = f"{role}_"
    split_file = _relative_path_value(
        contract.get(prefix + "split_file"),
        f"split_contract.{prefix}split_file",
    )
    split_sha256 = _sha256_value(
        contract.get(prefix + "split_sha256"),
        f"split_contract.{prefix}split_sha256",
    )
    num_images = _exact_integer(
        contract.get(prefix + "num_images"),
        f"split_contract.{prefix}num_images",
        minimum=1,
    )
    ids_sha256 = _sha256_value(
        contract.get(prefix + "ids_sha256"),
        f"split_contract.{prefix}ids_sha256",
    )
    artifact_sha256 = _sha256_value(
        contract.get(prefix + "split_image_artifact_sha256"),
        f"split_contract.{prefix}split_image_artifact_sha256",
    )
    raw_items = contract.get(prefix + "split_image_artifact_items")
    if not isinstance(raw_items, (list, tuple)) or len(raw_items) != num_images:
        raise ValueError(
            f"split_contract {prefix}split_image_artifact_items must contain "
            f"exactly {num_images} records"
        )
    items: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    artifact_digest = hashlib.sha256()
    _update_frame(artifact_digest, SPLIT_IMAGE_ARTIFACT_ALGORITHM)
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise TypeError(
                f"split_contract {prefix}split_image_artifact_items[{index}] "
                "must be a mapping"
            )
        sample_id = _nonempty_string(
            raw_item.get("sample_id"),
            f"split_contract.{prefix}items[{index}].sample_id",
        )
        if sample_id in seen_ids:
            raise ValueError(f"split_contract {role} contains duplicate sample IDs")
        seen_ids.add(sample_id)
        image_sha256 = _sha256_value(
            raw_item.get("image_sha256"),
            f"split_contract.{prefix}items[{index}].image_sha256",
        )
        _update_frame(artifact_digest, sample_id)
        _update_frame(artifact_digest, image_sha256)
        items.append({"sample_id": sample_id, "image_sha256": image_sha256})
    sample_ids = [item["sample_id"] for item in items]
    if ordered_sample_ids_sha256(sample_ids) != ids_sha256:
        raise ValueError(f"split_contract {role} ordered sample-ID SHA-256 mismatch")
    if artifact_digest.hexdigest() != artifact_sha256:
        raise ValueError(f"split_contract {role} split-image artifact SHA-256 mismatch")

    if manifest_root is not None:
        path = _resolve_path(manifest_root, split_file)
        if not path.is_file():
            raise FileNotFoundError(
                f"score split_contract references missing {role} split file: {path}"
            )
        before = sha256_file(path)
        if before != split_sha256:
            raise ValueError(f"score split_contract {role} split-file SHA-256 mismatch")
        entries = read_split_entries(path)
        actual_ids = [sample_id_from_entry(entry) for entry in entries]
        after = sha256_file(path)
        if after != before:
            raise RuntimeError(f"official split file changed while verified: {path}")
        if actual_ids != sample_ids:
            raise ValueError(
                f"score split_contract {role} IDs do not match the frozen split file"
            )
        if dataset_root is None:
            raise RuntimeError("dataset_root is required for split-content replay")
        actual_hashes = []
        for entry in entries:
            image_path = resolve_sample_file(
                dataset_root,
                "images",
                entry,
                kind="image",
            )
            actual_hashes.append(sha256_file(image_path))
        embedded_hashes = [item["image_sha256"] for item in items]
        if actual_hashes != embedded_hashes:
            raise ValueError(
                f"score split_contract {role} image hashes do not match the "
                "actual official split images"
            )

    return {
        "split_file": split_file,
        "split_sha256": split_sha256,
        "num_images": num_images,
        "ids_sha256": ids_sha256,
        "split_image_artifact_sha256": artifact_sha256,
        "items": items,
        "sample_ids": sample_ids,
        "image_sha256s": [item["image_sha256"] for item in items],
    }


def _verify_selected_item(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    index: int,
    image_id: str,
    require_mask: bool,
    allow_legacy_combined_diagnostic: bool,
) -> VerifiedScoreItem:
    root = manifest_path.parent
    score_path = _resolve_path(root, _score_file_value(item, index))
    gray_value = _explicit_relative_image_value(item, index)
    gray_path = _resolve_path(root, gray_value)
    if not score_path.is_file():
        raise FileNotFoundError(
            f"score manifest item {image_id!r} references missing score file: {score_path}"
        )
    if not gray_path.is_file():
        raise FileNotFoundError(
            f"score manifest item {image_id!r} references missing image file: {gray_path}"
        )

    declared_score_sha = _sha256_value(
        item.get("score_file_sha256"),
        f"items[{index}].score_file_sha256",
    )
    actual_score_sha = sha256_file(score_path)
    if actual_score_sha != declared_score_sha:
        raise ValueError(
            f"score-file SHA-256 mismatch for manifest item {image_id!r}: {score_path}"
        )
    declared_gray_sha = _sha256_value(
        item.get("gray_file_sha256"),
        f"items[{index}].gray_file_sha256",
    )
    actual_gray_sha = sha256_file(gray_path)
    if actual_gray_sha != declared_gray_sha:
        raise ValueError(
            f"original-image SHA-256 mismatch for manifest item {image_id!r}: {gray_path}"
        )

    item_hw = _parse_hw(item.get("original_hw"), f"items[{index}].original_hw")
    with np.load(score_path, allow_pickle=False) as score_payload:
        if "mask" in score_payload and not allow_legacy_combined_diagnostic:
            raise ValueError(
                "score NPZ embeds a mask and is ineligible for verified/main-protocol "
                f"use: {score_path}"
            )
        if "prob" not in score_payload:
            raise KeyError(f"score NPZ is missing 'prob': {score_path}")
        probability = np.asarray(score_payload["prob"])
        if probability.ndim != 2:
            raise ValueError(
                f"score NPZ prob must be native-resolution 2D for {image_id!r}, "
                f"got {probability.shape}"
            )
        if not np.issubdtype(probability.dtype, np.number):
            raise TypeError(f"score NPZ prob must be numeric: {score_path}")
        if not np.isfinite(probability).all():
            raise ValueError(f"score NPZ prob contains NaN/Inf: {score_path}")
        if probability.size and (
            float(probability.min()) < 0.0 or float(probability.max()) > 1.0
        ):
            raise ValueError(f"score NPZ prob is outside [0, 1]: {score_path}")
        declared_dtype = manifest.get("score_dtype")
        if declared_dtype is not None and str(probability.dtype) != str(declared_dtype):
            raise ValueError(
                f"score NPZ prob dtype disagrees with manifest for {image_id!r}: "
                f"{probability.dtype} != {declared_dtype}"
            )
        if "image_id" not in score_payload:
            raise KeyError(f"score NPZ is missing 'image_id': {score_path}")
        stored_id = _npz_scalar_string(score_payload["image_id"], "image_id", score_path)
        if stored_id != image_id:
            raise ValueError(
                f"score NPZ image_id mismatch: {stored_id!r} != {image_id!r}"
            )
        if "original_hw" not in score_payload:
            raise KeyError(f"score NPZ is missing 'original_hw': {score_path}")
        npz_hw = _parse_hw(score_payload["original_hw"], "score NPZ original_hw")
        if npz_hw != item_hw:
            raise ValueError(
                f"score NPZ/item original_hw mismatch for {image_id!r}: "
                f"{npz_hw} != {item_hw}"
            )
        if probability.shape != npz_hw:
            raise ValueError(
                f"score NPZ prob shape/original_hw mismatch for {image_id!r}: "
                f"{probability.shape} != {npz_hw}"
            )
        if require_mask:
            if "mask" not in score_payload:
                raise KeyError(f"score NPZ is missing 'mask': {score_path}")
            mask = np.asarray(score_payload["mask"])
            if mask.ndim != 2 or mask.shape != npz_hw:
                raise ValueError(
                    f"score NPZ mask shape/original_hw mismatch for {image_id!r}: "
                    f"{mask.shape} != {npz_hw}"
                )
        if "dataset_name" in score_payload and "target_dataset" in manifest:
            stored_dataset = _npz_scalar_string(
                score_payload["dataset_name"], "dataset_name", score_path
            )
            if stored_dataset != str(manifest["target_dataset"]):
                raise ValueError(
                    f"score NPZ dataset_name mismatch for {image_id!r}: "
                    f"{stored_dataset!r} != {manifest['target_dataset']!r}"
                )

    with Image.open(gray_path) as image:
        image_hw = (int(image.height), int(image.width))
    if image_hw != item_hw:
        raise ValueError(
            f"original image shape/original_hw mismatch for {image_id!r}: "
            f"{image_hw} != {item_hw}"
        )

    return VerifiedScoreItem(
        manifest_index=index,
        image_id=image_id,
        record=item,
        score_path=score_path,
        gray_path=gray_path,
        original_hw=item_hw,
    )


def _verify_native_manifest_contract(
    payload: Mapping[str, Any],
    *,
    allow_legacy_combined_diagnostic: bool,
) -> None:
    if payload.get("path_anchor") != "manifest_directory":
        raise ValueError(
            "score manifest path_anchor must equal 'manifest_directory'"
        )
    if payload.get("score_type") != SIGMOID_SCORE_TYPE:
        raise ValueError(
            "score manifest score_type must equal "
            f"{SIGMOID_SCORE_TYPE!r}"
        )
    if payload.get("restored_to_original_hw") is not True:
        raise ValueError("score manifest restored_to_original_hw must be exactly true")
    if payload.get("threshold_semantics") != STRICT_THRESHOLD_SEMANTICS:
        raise ValueError(
            "score manifest threshold_semantics must equal "
            f"{STRICT_THRESHOLD_SEMANTICS!r}"
        )
    if payload.get("extreme_tail_precision_verified") is True:
        if payload.get("score_dtype") != "float64":
            raise ValueError(
                "extreme-tail precision requires score_dtype='float64'"
            )
        if payload.get("sigmoid_compute_dtype") != "float64":
            raise ValueError(
                "extreme-tail precision requires sigmoid_compute_dtype='float64'"
            )
    if allow_legacy_combined_diagnostic:
        return
    if payload.get("schema_version") not in {2, SCORE_MANIFEST_SCHEMA_VERSION}:
        raise ValueError(
            "verified label-free score manifest schema_version must equal 2 or 3"
        )
    if payload.get("artifact_type") != "label_free_score_export":
        raise ValueError(
            "verified score manifest artifact_type must be 'label_free_score_export'"
        )
    if payload.get("labels_embedded") is not False:
        raise ValueError("verified score manifest labels_embedded must be exactly false")


def _verify_num_images(
    payload: Mapping[str, Any], item_count: int, path: Path
) -> None:
    if "num_images" not in payload:
        raise KeyError(f"score manifest is missing num_images: {path}")
    raw_value = payload["num_images"]
    if isinstance(raw_value, bool):
        raise TypeError("score manifest num_images must be an integer")
    try:
        value = int(raw_value)
        numeric = float(raw_value)
    except (TypeError, ValueError) as error:
        raise TypeError("score manifest num_images must be an integer") from error
    if not np.isfinite(numeric) or numeric != float(value) or value != item_count:
        raise ValueError(f"score manifest num_images disagrees with items: {path}")


def _require_item_mapping(
    item: object, index: int, path: Path
) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        raise TypeError(f"score manifest item {index} must be a mapping: {path}")
    return item


def _score_file_value(item: Mapping[str, Any], index: int) -> str:
    value = item.get("file", item.get("prob_path", item.get("score_path")))
    rendered = _nonempty_string(value, f"items[{index}].file")
    if Path(rendered).expanduser().is_absolute():
        raise ValueError(
            f"items[{index}].file must be relative to the manifest directory"
        )
    return rendered


def _explicit_relative_image_value(item: Mapping[str, Any], index: int) -> str:
    if "image_path" not in item:
        raise KeyError(f"items[{index}] is missing explicit image_path")
    value = _nonempty_string(item["image_path"], f"items[{index}].image_path")
    if Path(value).expanduser().is_absolute():
        raise ValueError(
            f"items[{index}].image_path must be relative to the manifest directory"
        )
    return value


def _resolve_path(root: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _parse_hw(value: object, name: str) -> tuple[int, int]:
    if value is None:
        raise KeyError(f"{name} is missing")
    array = np.asarray(value)
    if array.size != 2:
        raise ValueError(f"{name} must contain exactly [height, width]")
    flat = array.reshape(-1)
    result: list[int] = []
    for raw in flat:
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError(f"{name} values must be positive integers")
        try:
            integer = int(raw)
            numeric = float(raw)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} values must be positive integers") from error
        if not np.isfinite(numeric) or numeric != float(integer) or integer <= 0:
            raise ValueError(f"{name} values must be positive integers")
        result.append(integer)
    return result[0], result[1]


def _npz_scalar_string(value: object, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"score NPZ {name} must be scalar: {path}")
    return str(array.reshape(()).item())


def _nonempty_string(value: object, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _relative_path_value(value: object, name: str) -> str:
    rendered = _nonempty_string(value, name)
    if Path(rendered).expanduser().is_absolute():
        raise ValueError(f"{name} must be relative to the manifest directory")
    return rendered


def _exact_integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if not np.isfinite(numeric) or numeric != float(result) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an ordered list")
    result = [
        _nonempty_string(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    if result != sorted(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _sha256_list(value: object, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an ordered list")
    result = [
        _sha256_value(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    if result != sorted(set(result)):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _update_frame(digest: Any, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _sha256_value(value: object, name: str) -> str:
    rendered = "" if value is None else str(value).lower()
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return rendered
