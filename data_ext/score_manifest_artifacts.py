"""Fail-closed verification for exported native-resolution score manifests.

The exporter records two identities for every sample: the compressed score
artifact and the original image bytes.  Consumers use this module to validate
those identities, the ordered manifest aggregate, and the native-resolution
NPZ contract before loading any selected score map.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .dataset_identity import (
    SCORE_MANIFEST_CONTENT_ALGORITHM,
    score_manifest_content_sha256,
    sha256_file,
)


SIGMOID_SCORE_TYPE = "sigmoid_probability"
STRICT_THRESHOLD_SEMANTICS = "prediction = probability > threshold"


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


def verify_score_manifest_artifacts(
    manifest_path: str | Path,
    *,
    image_ids: Sequence[str] | None = None,
    require_mask: bool = False,
    require_native_contract: bool = True,
    verify_artifact_bytes: bool = True,
    allow_legacy_combined_diagnostic: bool = False,
) -> VerifiedScoreManifest:
    """Verify a score manifest and the selected/relevant sample artifacts.

    The aggregate digest is always checked for the full ordered item list.
    File bytes and native NPZ/image dimensions are checked for all items when
    ``image_ids`` is omitted, or only for the explicitly selected IDs.  Set
    ``verify_artifact_bytes=False`` to validate only the manifest, ordered
    aggregate and item metadata.  Online consumers use that mode to discover
    the query IDs without touching query artifacts, then make a second call
    for the selected context IDs only.
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
    )


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
    if allow_legacy_combined_diagnostic:
        return
    if payload.get("schema_version") != 2:
        raise ValueError("verified label-free score manifest schema_version must equal 2")
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


def _sha256_value(value: object, name: str) -> str:
    rendered = "" if value is None else str(value).lower()
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    return rendered
