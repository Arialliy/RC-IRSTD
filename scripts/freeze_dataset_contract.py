"""Freeze ordered image/mask byte and geometry contracts for AAAI-27 data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from data_ext.mask_alignment import (
    DEFAULT_ASPECT_TOLERANCE,
    aspect_ratio_relative_error,
)
from data_ext.split_utils import (
    read_split_entries,
    resolve_image_and_mask,
    resolve_split_file,
    sample_id_from_entry,
)


SCHEMA_VERSION = "rc-irstd.aaai27-dataset-contract.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"dataset artifact is outside repository: {path}") from error


def _ordered_content_sha(items: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in items:
        rendered = json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(rendered).to_bytes(8, "big"))
        digest.update(rendered)
    return digest.hexdigest()


def _split_contract(
    dataset_name: str,
    dataset_root: Path,
    role: str,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    split = resolve_split_file(dataset_root, split=role)
    entries = read_split_entries(split)
    items: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for entry in entries:
        image_id = sample_id_from_entry(entry)
        image_path, mask_path = resolve_image_and_mask(dataset_root, entry)
        with Image.open(image_path) as image:
            image_size = tuple(int(value) for value in image.size)
        with Image.open(mask_path) as mask:
            mask_size = tuple(int(value) for value in mask.size)
        mismatch = image_size != mask_size
        relative_error = aspect_ratio_relative_error(image_size, mask_size)
        if mismatch and relative_error > DEFAULT_ASPECT_TOLERANCE:
            raise ValueError(
                f"unsafe image/mask geometry for {dataset_name}/{image_id}: "
                f"image={image_size}, mask={mask_size}, error={relative_error}"
            )
        item = {
            "image_id": image_id,
            "image_path": _portable(image_path, repository_root),
            "image_sha256": sha256_file(image_path),
            "image_hw": [image_size[1], image_size[0]],
            "mask_path": _portable(mask_path, repository_root),
            "mask_sha256": sha256_file(mask_path),
            "original_mask_hw": [mask_size[1], mask_size[0]],
            "alignment_applied": mismatch,
            "mask_alignment": "nearest" if mismatch else "identity",
            "aspect_ratio_relative_error": relative_error,
        }
        items.append(item)
        if mismatch:
            mismatches.append(item)
    ids = [str(item["image_id"]) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {dataset_name} official_{role}")
    return {
        "role": f"official_{role}",
        "split_file": _portable(split, repository_root),
        "split_sha256": sha256_file(split),
        "count": len(items),
        "ordered_item_content_algorithm": (
            "sha256-length-prefixed-canonical-image-mask-records-v1"
        ),
        "ordered_item_content_sha256": _ordered_content_sha(items),
        "items": items,
        "geometry_mismatch_count": len(mismatches),
        "geometry_mismatches": mismatches,
    }


def build_contract(
    dataset_specs: Sequence[tuple[str, Path]],
    *,
    repository_root: Path,
    split_manifest: Path,
) -> dict[str, Any]:
    names = [name for name, _ in dataset_specs]
    if not names or len(names) != len(set(names)):
        raise ValueError("dataset names must be non-empty and unique")
    split_manifest_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    if split_manifest_payload.get("schema_version") != (
        "rc-irstd.aaai27-official-train-splits.v2"
    ):
        raise ValueError("dataset contract requires the quarantined v2 split manifest")
    datasets: list[dict[str, Any]] = []
    misc111: Mapping[str, Any] | None = None
    for name, root in dataset_specs:
        train = _split_contract(name, root, "train", repository_root=repository_root)
        test = _split_contract(name, root, "test", repository_root=repository_root)
        train_ids = {str(item["image_id"]) for item in train["items"]}
        test_ids = {str(item["image_id"]) for item in test["items"]}
        if train_ids.intersection(test_ids):
            raise ValueError(f"official train/test IDs overlap for {name}")
        for role in (train, test):
            for item in role["geometry_mismatches"]:
                if name == "NUAA-SIRST" and item["image_id"] == "Misc_111":
                    misc111 = item
        datasets.append(
            {
                "dataset_name": name,
                "dataset_root": _portable(root, repository_root),
                "official_train": train,
                "official_test": test,
            }
        )
    if misc111 is None:
        raise ValueError("NUAA-SIRST Misc_111 mismatch was not found and frozen")
    if (
        misc111.get("alignment_applied") is not True
        or misc111.get("mask_alignment") != "nearest"
    ):
        raise ValueError("Misc_111 does not use the required NEAREST alignment")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "aaai27_ordered_dataset_byte_and_geometry_contract",
        "split_manifest": {
            "path": _portable(split_manifest, repository_root),
            "sha256": sha256_file(split_manifest),
        },
        "data_policy": {
            "official_train_and_test_only": True,
            "official_test_labels_used_for_model_or_hyperparameter_selection": False,
            "original_resolution_evaluation": True,
            "mask_binarization_after_alignment": "greater_than_zero",
            "mismatch_alignment": "PIL_NEAREST_before_other_transforms",
            "maximum_aspect_ratio_relative_error": DEFAULT_ASPECT_TOLERANCE,
        },
        "datasets": datasets,
        "required_special_case": {
            "dataset_name": "NUAA-SIRST",
            **dict(misc111),
        },
    }


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid dataset: {value}")
    return name.strip(), path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute and byte-compare an existing frozen contract.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.repository_root).expanduser().resolve()
    split_manifest = Path(args.split_manifest).expanduser()
    output = Path(args.output).expanduser()
    if not split_manifest.is_absolute():
        split_manifest = root / split_manifest
    if not output.is_absolute():
        output = root / output
    if output.exists() and not args.check:
        raise FileExistsError(f"refusing to overwrite dataset contract: {output}")
    payload = build_contract(
        args.dataset,
        repository_root=root,
        split_manifest=split_manifest.resolve(),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(output)
        if output.read_text(encoding="utf-8") != rendered:
            raise ValueError("frozen dataset contract differs from current bytes/geometry")
        action = "verified"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        action = "written"
    print(
        json.dumps(
            {
                "status": "PASS",
                "action": action,
                "output": _portable(output, root),
                "sha256": sha256_file(output),
                "dataset_count": len(payload["datasets"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
