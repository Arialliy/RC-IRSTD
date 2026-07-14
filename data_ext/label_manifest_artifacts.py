"""Independent, score-bound ground-truth label attachment artifacts.

Score exports are intentionally label-free.  Offline evaluation attaches
labels through this second manifest, whose ordered items are bound to one
exact score-manifest file and content identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_identity import sha256_file
from .score_manifest_artifacts import (
    VerifiedScoreManifest,
    verify_score_manifest_artifacts,
)


LABEL_MANIFEST_SCHEMA_VERSION = 1
LABEL_MANIFEST_ARTIFACT_TYPE = "score_bound_label_attachment"
LABEL_MANIFEST_CONTENT_ALGORITHM = (
    "sha256-length-prefixed-image-label-source-image-original-hw-v1"
)


@dataclass(frozen=True)
class VerifiedLabelItem:
    manifest_index: int
    image_id: str
    record: Mapping[str, Any]
    label_path: Path
    label_file_sha256: str
    original_hw: tuple[int, int]


@dataclass(frozen=True)
class VerifiedLabelAttachment:
    path: Path
    payload: Mapping[str, Any]
    score_manifest: VerifiedScoreManifest
    items: tuple[Mapping[str, Any], ...]
    selected_items: tuple[VerifiedLabelItem, ...]
    content_sha256: str
    manifest_sha256: str


def label_manifest_content_sha256(
    items: Sequence[Mapping[str, object]],
) -> str:
    """Hash the complete ordered label attachment item sequence."""

    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("label manifest items must be a non-empty ordered list")
    digest = hashlib.sha256()
    _update_frame(digest, LABEL_MANIFEST_CONTENT_ALGORITHM)
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"label manifest item {index} must be a mapping")
        image_id = _nonempty(item.get("image_id"), f"items[{index}].image_id")
        if image_id in seen:
            raise ValueError(f"duplicate label manifest image_id: {image_id!r}")
        seen.add(image_id)
        label_sha = _sha256(
            item.get("label_file_sha256"),
            f"items[{index}].label_file_sha256",
        )
        source_image_sha = _sha256(
            item.get("source_image_file_sha256"),
            f"items[{index}].source_image_file_sha256",
        )
        original_hw = _parse_hw(item.get("original_hw"), f"items[{index}].original_hw")
        _update_frame(digest, image_id)
        _update_frame(digest, label_sha)
        _update_frame(digest, source_image_sha)
        _update_frame(digest, str(original_hw[0]))
        _update_frame(digest, str(original_hw[1]))
    return digest.hexdigest()


def verify_label_attachment(
    score_manifest_path: str | Path,
    label_manifest_path: str | Path,
    *,
    image_ids: Sequence[str] | None = None,
    verify_artifact_bytes: bool = True,
) -> VerifiedLabelAttachment:
    """Verify score artifacts, their separate label attachment, and binding.

    This is the single entry point for offline labelled consumers.  The score
    side is always verified under the strict label-free contract; combined
    legacy score/mask NPZ files are not accepted here.
    """

    score = verify_score_manifest_artifacts(
        score_manifest_path,
        image_ids=image_ids,
        verify_artifact_bytes=verify_artifact_bytes,
        allow_legacy_combined_diagnostic=False,
    )
    path = Path(label_manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"label manifest does not exist: {path}")
    marker = path.parent / ".label_export_incomplete"
    if marker.exists():
        raise RuntimeError(f"label export is incomplete and unsafe to consume: {path.parent}")
    _require_disjoint_artifact_trees(score.path.parent, path.parent)

    manifest_sha_before = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("label manifest must contain a JSON object")
    manifest_sha = sha256_file(path)
    if manifest_sha != manifest_sha_before:
        raise RuntimeError(f"label manifest changed while being verified: {path}")
    if payload.get("schema_version") != LABEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported label manifest schema_version")
    if payload.get("artifact_type") != LABEL_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("label manifest artifact_type mismatch")
    if payload.get("path_anchor") != "manifest_directory":
        raise ValueError("label manifest path_anchor must equal 'manifest_directory'")
    if payload.get("labels_embedded_in_scores") is not False:
        raise ValueError("label manifest must declare labels_embedded_in_scores=false")
    if score.split_role == "detector_diagnostic":
        expected_development_scope = {
            "score_split_role": "detector_diagnostic",
            "score_partition_scope": (
                "official_train_derived_development_diagnostic"
            ),
            "official_test_artifact": False,
            "final_evaluation_eligible": False,
            "development_only": True,
            "claim_bearing_final_evaluation": False,
        }
        for field, expected in expected_development_scope.items():
            if payload.get(field) != expected:
                raise ValueError(
                    f"development label manifest {field} must be exactly "
                    f"{expected!r}"
                )

    raw_score_ref = _nonempty(
        payload.get("score_manifest_file"), "score_manifest_file"
    )
    score_ref = Path(raw_score_ref).expanduser()
    if score_ref.is_absolute():
        raise ValueError("score_manifest_file must be relative to label manifest")
    if (path.parent / score_ref).resolve() != score.path:
        raise ValueError("label manifest is bound to a different score manifest path")
    if _sha256(payload.get("score_manifest_sha256"), "score_manifest_sha256") != score.manifest_sha256:
        raise ValueError("label/score manifest SHA-256 binding mismatch")
    if _sha256(
        payload.get("score_manifest_content_sha256"),
        "score_manifest_content_sha256",
    ) != score.content_sha256:
        raise ValueError("label/score ordered content binding mismatch")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("label manifest requires a non-empty items list")
    items: tuple[Mapping[str, Any], ...] = tuple(
        _mapping(item, index) for index, item in enumerate(raw_items)
    )
    if int(payload.get("num_images", -1)) != len(items):
        raise ValueError("label manifest num_images disagrees with items")
    if payload.get("content_sha256_algorithm") != LABEL_MANIFEST_CONTENT_ALGORITHM:
        raise ValueError("label manifest content_sha256_algorithm mismatch")
    content_sha = label_manifest_content_sha256(items)
    if _sha256(payload.get("content_sha256"), "content_sha256") != content_sha:
        raise ValueError("label manifest ordered content SHA-256 mismatch")

    score_by_id = {
        str(item["image_id"]): item for item in score.items
    }
    label_ids = [
        _nonempty(item.get("image_id"), f"items[{index}].image_id")
        for index, item in enumerate(items)
    ]
    score_ids = [str(item["image_id"]) for item in score.items]
    if label_ids != score_ids:
        raise ValueError(
            "label manifest items must exactly match score manifest IDs and order"
        )
    if image_ids is None:
        selected_ids = label_ids if verify_artifact_bytes else []
    else:
        selected_ids = [str(value).strip() for value in image_ids]
        if not selected_ids or any(not value for value in selected_ids):
            raise ValueError("selected label image IDs must be non-empty")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("selected label image IDs contain duplicates")
        missing = [value for value in selected_ids if value not in score_by_id]
        if missing:
            raise KeyError(f"selected label IDs are absent from manifest: {missing}")
        if not verify_artifact_bytes:
            selected_ids = []

    by_id = {str(item["image_id"]): (index, item) for index, item in enumerate(items)}
    selected: list[VerifiedLabelItem] = []
    for image_id in selected_ids:
        index, item = by_id[image_id]
        score_hw = _parse_hw(score_by_id[image_id].get("original_hw"), "score original_hw")
        label_hw = _parse_hw(item.get("original_hw"), f"items[{index}].original_hw")
        if label_hw != score_hw:
            raise ValueError(f"label/score original_hw mismatch for {image_id!r}")
        if _sha256(
            item.get("source_image_file_sha256"),
            f"items[{index}].source_image_file_sha256",
        ) != _sha256(
            score_by_id[image_id].get("gray_file_sha256"),
            f"score item {image_id!r} gray_file_sha256",
        ):
            raise ValueError(
                f"label source-image/score gray SHA-256 mismatch for {image_id!r}"
            )
        file_value = _relative_file(item.get("file"), f"items[{index}].file")
        label_path = (path.parent / file_value).resolve()
        if label_path.parent != path.parent:
            raise ValueError("label NPZ files must be directly under label manifest directory")
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label NPZ for {image_id!r}: {label_path}")
        declared_sha = _sha256(
            item.get("label_file_sha256"),
            f"items[{index}].label_file_sha256",
        )
        if sha256_file(label_path) != declared_sha:
            raise ValueError(f"label-file SHA-256 mismatch for {image_id!r}")
        with np.load(label_path, allow_pickle=False) as attachment:
            forbidden = {"prob", "score", "logits"}.intersection(attachment.files)
            if forbidden:
                raise ValueError(
                    f"label attachment contains score arrays {sorted(forbidden)}: {label_path}"
                )
            if "mask" not in attachment or "image_id" not in attachment:
                raise KeyError(f"label NPZ requires mask and image_id: {label_path}")
            mask = np.asarray(attachment["mask"])
            if mask.ndim != 2 or tuple(mask.shape) != label_hw:
                raise ValueError(f"label NPZ mask/original_hw mismatch for {image_id!r}")
            if not np.isin(mask, (0, 1)).all():
                raise ValueError(f"label NPZ mask is not binary for {image_id!r}")
            stored_id = str(np.asarray(attachment["image_id"]).reshape(()).item())
            if stored_id != image_id:
                raise ValueError(f"label NPZ image_id mismatch for {image_id!r}")
            if "original_hw" not in attachment or _parse_hw(
                attachment["original_hw"], "label NPZ original_hw"
            ) != label_hw:
                raise ValueError(f"label NPZ original_hw mismatch for {image_id!r}")
        selected.append(
            VerifiedLabelItem(
                manifest_index=index,
                image_id=image_id,
                record=item,
                label_path=label_path,
                label_file_sha256=declared_sha,
                original_hw=label_hw,
            )
        )
    return VerifiedLabelAttachment(
        path=path,
        payload=payload,
        score_manifest=score,
        items=items,
        selected_items=tuple(selected),
        content_sha256=content_sha,
        manifest_sha256=manifest_sha,
    )


def load_label_mask(item: VerifiedLabelItem) -> np.ndarray:
    """Reload one verified binary mask, rechecking its immutable file digest."""

    if sha256_file(item.label_path) != item.label_file_sha256:
        raise ValueError(f"label file changed after verification: {item.label_path}")
    with np.load(item.label_path, allow_pickle=False) as payload:
        mask = np.asarray(payload["mask"], dtype=np.uint8)
    if mask.shape != item.original_hw or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"verified label contract changed: {item.label_path}")
    return mask


def _require_disjoint_artifact_trees(score_root: Path, label_root: Path) -> None:
    score_root = score_root.resolve()
    label_root = label_root.resolve()
    if (
        score_root == label_root
        or score_root in label_root.parents
        or label_root in score_root.parents
    ):
        raise ValueError(
            "score and label artifacts must occupy disjoint directory trees"
        )


def _mapping(value: object, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"label manifest item {index} must be a mapping")
    return value


def _relative_file(value: object, field: str) -> Path:
    rendered = _nonempty(value, field)
    path = Path(rendered).expanduser()
    if path.is_absolute() or len(path.parts) != 1 or path.name in {".", ".."}:
        raise ValueError(f"{field} must be one relative filename")
    return path


def _parse_hw(value: object, field: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.size != 2:
        raise ValueError(f"{field} must contain [height, width]")
    values = tuple(int(raw) for raw in array.reshape(-1))
    if any(value <= 0 for value in values):
        raise ValueError(f"{field} values must be positive")
    return values[0], values[1]


def _nonempty(value: object, field: str) -> str:
    rendered = "" if value is None else str(value).strip()
    if not rendered:
        raise ValueError(f"{field} must be non-empty")
    return rendered


def _sha256(value: object, field: str) -> str:
    rendered = "" if value is None else str(value).lower()
    if len(rendered) != 64 or any(c not in "0123456789abcdef" for c in rendered):
        raise ValueError(f"{field} must be a SHA-256 hexadecimal digest")
    return rendered


def _update_frame(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big", signed=False))
    digest.update(encoded)
