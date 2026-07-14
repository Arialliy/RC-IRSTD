"""Attach offline labels to a strict label-free score manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from data_ext.dataset_identity import sha256_file
from data_ext.dataset_meta import safe_output_stem
from data_ext.label_manifest_artifacts import (
    LABEL_MANIFEST_ARTIFACT_TYPE,
    LABEL_MANIFEST_CONTENT_ALGORITHM,
    LABEL_MANIFEST_SCHEMA_VERSION,
    _require_disjoint_artifact_trees,
    label_manifest_content_sha256,
    verify_label_attachment,
)
from data_ext.mask_alignment import (
    DEFAULT_ASPECT_TOLERANCE,
    align_mask_to_image,
    aspect_ratio_relative_error,
)
from data_ext.score_manifest_artifacts import verify_score_manifest_artifacts
from data_ext.split_utils import resolve_sample_file


def export_label_maps(
    *,
    dataset_dir: str | Path,
    score_manifest: str | Path,
    output_dir: str | Path,
    mask_folder: str = "masks",
    overwrite: bool = False,
) -> dict[str, object]:
    """Create an independent label attachment aligned to native score geometry."""

    score = verify_score_manifest_artifacts(score_manifest)
    dataset_root = Path(dataset_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset_root}")
    declared_target = str(score.payload.get("target_dataset", "")).strip()
    if not declared_target or dataset_root.name != declared_target:
        raise ValueError(
            "dataset directory name must match score manifest target_dataset: "
            f"{dataset_root.name!r} != {declared_target!r}"
        )
    output_root = Path(output_dir).expanduser().resolve()
    _require_disjoint_artifact_trees(score.path.parent, output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / ".label_export_incomplete"
    if marker.exists() and not overwrite:
        raise RuntimeError("a previous label export is incomplete; inspect or use --overwrite")
    existing = list(output_root.glob("*.npz"))
    if existing and not overwrite:
        raise FileExistsError(
            f"label output contains {len(existing)} NPZ files; pass --overwrite"
        )
    marker.write_text("Label attachment export in progress.\n", encoding="utf-8")

    items: list[dict[str, object]] = []
    names: set[str] = set()
    for score_item in score.items:
        image_id = str(score_item["image_id"])
        target_h, target_w = (int(value) for value in score_item["original_hw"])
        image_path = resolve_sample_file(
            dataset_root,
            "images",
            image_id,
            kind="image",
        )
        source_image_sha = sha256_file(image_path)
        if source_image_sha != str(score_item["gray_file_sha256"]).lower():
            raise ValueError(
                "dataset image bytes do not match the score manifest for "
                f"{image_id!r}; refusing to attach labels from the wrong data root"
            )
        with Image.open(image_path) as image_file:
            source_image_size = image_file.size
        if source_image_size != (target_w, target_h):
            raise ValueError(
                "score manifest original_hw does not match the source image for "
                f"{image_id!r}: manifest={(target_w, target_h)} "
                f"image={source_image_size}"
            )
        mask_path = resolve_sample_file(
            dataset_root,
            mask_folder,
            image_id,
            kind="mask",
        )
        source_mask_sha = sha256_file(mask_path)
        with Image.open(mask_path) as mask_file:
            mask_image = mask_file.convert("L")
        source_hw = (mask_image.height, mask_image.width)
        aspect_error = aspect_ratio_relative_error(
            (target_w, target_h),
            mask_image.size,
        )
        mask_image = align_mask_to_image(
            mask_image,
            (target_w, target_h),
            image_id,
        )
        mask = (np.asarray(mask_image, dtype=np.uint8) > 0).astype(np.uint8)
        output_name = f"{safe_output_stem(image_id)}.label.npz"
        if output_name in names:
            raise RuntimeError(f"duplicate label output filename: {output_name}")
        names.add(output_name)
        output_path = output_root / output_name
        _write_npz_atomic(
            output_path,
            mask=mask,
            image_id=np.asarray(image_id),
            original_hw=np.asarray((target_h, target_w), dtype=np.int32),
            source_mask_original_hw=np.asarray(source_hw, dtype=np.int32),
        )
        items.append(
            {
                "image_id": image_id,
                "file": output_name,
                "label_file_sha256": sha256_file(output_path),
                "source_image_file_sha256": source_image_sha,
                "original_hw": [target_h, target_w],
                "source_mask_file_sha256": source_mask_sha,
                "source_mask_original_hw": list(source_hw),
                "source_mask_resized_to_score_hw": source_hw != (target_h, target_w),
                "mask_alignment_aspect_ratio_relative_error": aspect_error,
                "mask_alignment_aspect_tolerance": DEFAULT_ASPECT_TOLERANCE,
            }
        )

    if sha256_file(score.path) != score.manifest_sha256:
        raise RuntimeError("score manifest changed during label attachment export")
    payload: dict[str, object] = {
        "schema_version": LABEL_MANIFEST_SCHEMA_VERSION,
        "artifact_type": LABEL_MANIFEST_ARTIFACT_TYPE,
        "path_anchor": "manifest_directory",
        "score_manifest_file": Path(
            os.path.relpath(score.path, start=output_root)
        ).as_posix(),
        "score_manifest_sha256": score.manifest_sha256,
        "score_manifest_content_sha256": score.content_sha256,
        "target_dataset": declared_target,
        "score_split_role": score.split_role,
        "score_partition_scope": score.payload.get("partition_scope"),
        "official_test_artifact": score.split_role == "official_test",
        "final_evaluation_eligible": score.split_role == "official_test",
        "development_only": score.split_role != "official_test",
        "claim_bearing_final_evaluation": False,
        "labels_embedded_in_scores": False,
        "alignment_rule": (
            "binary mask; nearest-neighbor to score original_hw only when "
            "image/mask aspect-ratio relative error <= 0.01; otherwise fail closed"
        ),
        "num_images": len(items),
        "content_sha256_algorithm": LABEL_MANIFEST_CONTENT_ALGORITHM,
        "content_sha256": label_manifest_content_sha256(items),
        "items": items,
    }
    _write_json_atomic(output_root / "label-manifest.json", payload)
    marker.unlink()
    # Self-verification prevents a producer bug from emitting a seemingly
    # complete but unusable attachment.
    verify_label_attachment(score.path, output_root / "label-manifest.json")
    return payload


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mask-folder", default="masks")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = export_label_maps(
        dataset_dir=args.dataset_dir,
        score_manifest=args.score_manifest,
        output_dir=args.output_dir,
        mask_folder=args.mask_folder,
        overwrite=args.overwrite,
    )
    print(
        f"Exported {payload['num_images']} score-bound labels to "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
