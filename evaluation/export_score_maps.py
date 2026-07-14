"""Export native-resolution sigmoid score maps from an MSHNet checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_ext.dataset_meta import (
    image_meta_from_batch,
    restore_tensor_to_original,
    safe_output_stem,
)
from data_ext.dataset_identity import (
    SCORE_MANIFEST_CONTENT_ALGORITHM,
    build_dataset_record,
    score_manifest_content_sha256,
    sha256_file,
    validate_dataset_record,
)
from data_ext.inference_dataset import IRSTDInferenceDataset
from model.MSHNet import MSHNet


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
def select_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if device != "auto":
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalise_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Accept repository weights, checkpoints and DataParallel prefixes."""

    state_dict = checkpoint
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "net" in checkpoint:
            state_dict = checkpoint["net"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict mapping")
    if not state_dict:
        raise ValueError("Checkpoint state_dict is empty")
    normalised = {
        (key[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }
    if not all(isinstance(value, torch.Tensor) for value in normalised.values()):
        raise TypeError("Checkpoint mapping contains non-tensor model parameters")
    return normalised


def load_model(
    weight_path: str | Path,
    device: torch.device,
    *,
    checkpoint: object | None = None,
) -> torch.nn.Module:
    model = MSHNet(3)
    if checkpoint is None:
        checkpoint = torch.load(
            Path(weight_path).expanduser().resolve(),
            map_location=device,
        )
    model.load_state_dict(normalise_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()
    return model


def checkpoint_provenance(checkpoint: object) -> dict[str, object]:
    """Extract detector-fold metadata without trusting a CLI label.

    Historical paper weights contain tensors only.  They remain usable for
    diagnostic reproduction but are explicitly marked unverified; RC protocol
    artifacts must use a training checkpoint carrying source-domain metadata.
    """

    if not isinstance(checkpoint, Mapping):
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": "checkpoint_is_not_a_metadata_mapping",
        }
    raw_sources = checkpoint.get(
        "detector_source_domains",
        checkpoint.get("source_names"),
    )
    if raw_sources is None:
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": "missing_detector_source_domains",
        }
    if isinstance(raw_sources, (str, bytes)):
        sources = [str(raw_sources)]
    else:
        sources = [str(value) for value in raw_sources]
    if not sources or any(not value for value in sources) or len(set(sources)) != len(sources):
        raise ValueError("checkpoint detector source domains are empty or duplicated")
    raw_held_out = checkpoint.get("held_out_domains", [])
    if not isinstance(raw_held_out, (list, tuple)):
        raise TypeError("checkpoint held_out_domains must be an ordered list")
    held_out = [str(value) for value in raw_held_out]
    if len(set(held_out)) != len(held_out):
        raise ValueError("checkpoint held_out_domains contain duplicates")
    overlap = sorted(set(sources) & set(held_out))
    if overlap:
        raise ValueError(
            "checkpoint source/held-out domain metadata overlap: " + ", ".join(overlap)
        )
    outer_target = checkpoint.get("outer_target")
    if outer_target is not None and str(outer_target) in sources:
        raise ValueError("checkpoint outer_target occurs in detector source domains")
    common: dict[str, object] = {
        "detector_source_domains": sources,
        "held_out_domains": held_out,
        "outer_fold_id": checkpoint.get("outer_fold_id"),
        "outer_target": outer_target,
        "checkpoint_selection": checkpoint.get("checkpoint_selection"),
        "protocol_scope": checkpoint.get("protocol_scope"),
        "training_seed": checkpoint.get("seed"),
    }
    raw_records = checkpoint.get("detector_source_records")
    if raw_records is None:
        # Keep historical metadata visible for diagnostics, but old
        # source-name-only checkpoints are not eligible for the main protocol.
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": "missing_detector_source_records",
            **common,
        }
    if not isinstance(raw_records, (list, tuple)):
        raise TypeError("checkpoint detector_source_records must be an ordered list")
    if len(raw_records) != len(sources):
        raise ValueError(
            "checkpoint detector_source_records must align one-to-one with "
            "detector_source_domains"
        )
    # Schema-v1 records lack per-image content leaves and the ordered
    # image+mask training artifact.  They may be exported diagnostically but
    # can never establish main-protocol target exclusion.
    if any(
        not isinstance(record, Mapping)
        or int(record.get("record_schema_version", -1)) != 2
        for record in raw_records
    ):
        return {
            "provenance_level": "legacy_unverified",
            "legacy_reason": "detector_source_records_precede_content_leaf_schema_v2",
            **common,
        }
    records = [
        validate_dataset_record(
            record,
            require_source_name=True,
            require_training_artifact=True,
        )
        for record in raw_records
    ]
    record_names = [str(record["source_name"]) for record in records]
    if record_names != sources:
        raise ValueError(
            "checkpoint detector source record order/names do not match "
            "detector_source_domains"
        )
    identities = [str(record["dataset_identity_sha256"]) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "checkpoint detector source records contain duplicate dataset identities"
        )
    dataset_sizes = checkpoint.get("dataset_sizes")
    if dataset_sizes is not None:
        if not isinstance(dataset_sizes, Mapping):
            raise TypeError("checkpoint dataset_sizes must be a mapping")
        for record in records:
            name = str(record["source_name"])
            if name in dataset_sizes and int(dataset_sizes[name]) != int(
                record["num_samples"]
            ):
                raise ValueError(
                    f"checkpoint dataset size disagrees with source record {name!r}"
                )
    if common["checkpoint_selection"] != "fixed_last_no_test_or_target_validation":
        raise ValueError(
            "checkpoint_verified requires checkpoint_selection="
            "'fixed_last_no_test_or_target_validation'"
        )
    if common["protocol_scope"] not in {
        "multi_source_protocol_candidate",
        "single_source_inner_smoke_not_main_result",
    }:
        raise ValueError(
            "checkpoint_verified requires a known detector protocol_scope"
        )
    expected_scope = (
        "single_source_inner_smoke_not_main_result"
        if len(sources) == 1
        else "multi_source_protocol_candidate"
    )
    if common["protocol_scope"] != expected_scope:
        raise ValueError(
            "checkpoint detector source count is inconsistent with protocol_scope"
        )
    return {
        "provenance_level": "checkpoint_verified",
        **common,
        "detector_source_records": records,
    }


def extract_logits(model_output: object) -> torch.Tensor:
    """Extract final logits from the current ``(aux, logits)`` model output."""

    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, (tuple, list)) and model_output:
        candidate = model_output[-1]
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError("Model output does not contain a logits tensor")


def export_score_maps(
    *,
    dataset_dir: str | Path,
    weight_path: str | Path,
    output_dir: str | Path,
    base_size: int | Sequence[int] = 256,
    resize_mode: str = "resize",
    split: str = "test",
    split_file: str | Path | None = None,
    source_dataset: str | None = None,
    device: str = "auto",
    num_workers: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run inference and write one native-resolution ``.npz`` per image."""

    input_hw = (
        (int(base_size), int(base_size))
        if isinstance(base_size, (int, np.integer))
        else tuple(int(value) for value in base_size)
    )
    if len(input_hw) != 2 or any(value <= 0 or value % 16 != 0 for value in input_hw):
        raise ValueError(
            "MSHNet input height and width must be positive multiples of 16"
        )

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    incomplete_marker = output_root / ".export_incomplete"
    if incomplete_marker.exists() and not overwrite:
        raise RuntimeError(
            f"A previous score export did not finish under {output_root}; "
            "rerun with --overwrite after checking the directory."
        )
    existing = list(output_root.glob("*.npz"))
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory already contains {len(existing)} score maps: "
            f"{output_root}. Pass overwrite=True to replace matching files."
        )
    incomplete_marker.write_text(
        "Score-map export is in progress. Do not sweep this directory.\n",
        encoding="utf-8",
    )

    dataset = IRSTDInferenceDataset(
        dataset_dir,
        base_size=base_size,
        resize_mode=resize_mode,
        split=split,
        split_file=split_file,
    )
    target_dataset_record = build_dataset_record(
        dataset.root,
        dataset.split_file,
        [sample[0] for sample in dataset.samples],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu" and torch.cuda.is_available()),
    )
    selected_device = select_device(device)
    weight = Path(weight_path).expanduser().resolve()
    checkpoint = torch.load(weight, map_location=selected_device)
    detector_provenance = checkpoint_provenance(checkpoint)
    detector_provenance["target_exclusion_verified"] = False
    if detector_provenance["provenance_level"] == "checkpoint_verified":
        source_domains = set(detector_provenance["detector_source_domains"])
        held_out_domains = set(detector_provenance["held_out_domains"])
        logical_exclusion_verified = (
            dataset.dataset_name in held_out_domains
            and dataset.dataset_name not in source_domains
        )
        target_leaves = set(target_dataset_record["image_content_sha256_leaves"])
        source_union: set[str] = set()
        per_source_collisions: list[dict[str, object]] = []
        for source_record in detector_provenance["detector_source_records"]:
            source_leaves = set(source_record["image_content_sha256_leaves"])
            source_union.update(source_leaves)
            collisions = sorted(target_leaves & source_leaves)
            per_source_collisions.append(
                {
                    "source_name": source_record["source_name"],
                    "source_leaf_count": len(source_leaves),
                    "collision_count": len(collisions),
                    "collision_sha256_leaves": collisions,
                }
            )
        all_collisions = sorted(target_leaves & source_union)
        identity_exclusion_verified = not all_collisions
        detector_provenance["target_identity_collision_audit"] = {
            "comparison": "raw_image_file_sha256_leaf_intersection",
            "target_leaf_count": len(target_leaves),
            "detector_source_union_leaf_count": len(source_union),
            "collision_count": len(all_collisions),
            "collision_sha256_leaves": all_collisions,
            "per_source": per_source_collisions,
        }
        detector_provenance["logical_target_exclusion_verified"] = (
            logical_exclusion_verified
        )
        detector_provenance["target_identity_exclusion_verified"] = (
            identity_exclusion_verified
        )
        detector_provenance["target_exclusion_verified"] = (
            logical_exclusion_verified and identity_exclusion_verified
        )
    model = load_model(weight_path, selected_device, checkpoint=checkpoint)

    items: list[dict[str, object]] = []
    used_output_names: set[str] = set()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Exporting native score maps"):
            if "mask" in batch:
                raise RuntimeError("label-free inference batch unexpectedly contains a mask")
            metadata = image_meta_from_batch(batch["meta"], 0)
            image = batch["image"].to(selected_device, non_blocking=True)
            logits = extract_logits(model(image, True))
            if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 1:
                raise ValueError(f"Expected logits [1, 1, H, W], got {tuple(logits.shape)}")
            probability = torch.sigmoid(logits[0, 0])
            probability = restore_tensor_to_original(
                probability,
                metadata.transform,
                mode="bilinear",
            )
            probability_array = (
                probability.clamp_(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
            )
            if probability_array.shape != metadata.transform.original_hw:
                raise RuntimeError(
                    f"Restored score/image mismatch for {metadata.image_id!r}: "
                    f"{probability_array.shape} vs {metadata.transform.original_hw}"
                )

            output_name = f"{safe_output_stem(metadata.image_id)}.npz"
            if output_name in used_output_names:
                raise RuntimeError(
                    f"Two sample IDs map to the same output filename: {output_name}"
                )
            used_output_names.add(output_name)
            output_path = output_root / output_name
            _write_npz_atomic(
                output_path,
                prob=probability_array,
                image_id=np.asarray(metadata.image_id),
                dataset_name=np.asarray(metadata.dataset_name),
                original_hw=np.asarray(metadata.transform.original_hw, dtype=np.int32),
                input_hw=np.asarray(metadata.transform.input_hw, dtype=np.int32),
                resized_hw=np.asarray(metadata.transform.resized_hw, dtype=np.int32),
                padding_ltrb=np.asarray(
                    metadata.transform.padding_ltrb,
                    dtype=np.int32,
                ),
                resize_mode=np.asarray(metadata.transform.mode),
            )
            score_file_sha256 = sha256_file(output_path)
            gray_file_sha256 = sha256_file(metadata.image_path)
            items.append(
                {
                    "image_id": metadata.image_id,
                    "file": output_name,
                    "score_file_sha256": score_file_sha256,
                    "image_path": _portable_path(metadata.image_path, output_root),
                    "gray_file_sha256": gray_file_sha256,
                    "original_hw": list(metadata.transform.original_hw),
                }
            )

    manifest: dict[str, object] = {
        "schema_version": 2,
        "artifact_type": "label_free_score_export",
        "source_dataset": source_dataset,
        "source_dataset_assertion": source_dataset,
        "target_dataset": dataset.dataset_name,
        "target_dataset_record": target_dataset_record,
        "path_anchor": "manifest_directory",
        "dataset_dir": _portable_path(dataset.root, output_root),
        "split_file": _portable_path(dataset.split_file, output_root),
        "weight_path": _portable_path(weight, output_root),
        "weight_sha256": sha256_file(weight),
        "detector_provenance": detector_provenance,
        "input_hw": list(dataset.input_hw),
        "resize_mode": resize_mode,
        "restored_to_original_hw": True,
        "score_type": "sigmoid_probability",
        "threshold_semantics": "prediction = probability > threshold",
        "labels_embedded": False,
        "label_contract": "external_label_attachment_manifest_required_offline",
        "num_images": len(items),
        "content_sha256_algorithm": SCORE_MANIFEST_CONTENT_ALGORITHM,
        "content_sha256": score_manifest_content_sha256(items),
        "items": items,
    }
    # Duplicate the protocol-critical fields at the top level so every
    # downstream stage can hard-validate them without interpreting a free-form
    # nested provenance payload.  The full payload remains for audit detail.
    for field in (
        "provenance_level",
        "detector_source_domains",
        "held_out_domains",
        "outer_fold_id",
        "outer_target",
        "checkpoint_selection",
        "protocol_scope",
        "detector_source_records",
        "logical_target_exclusion_verified",
        "target_identity_exclusion_verified",
        "target_identity_collision_audit",
        "target_exclusion_verified",
    ):
        if field in detector_provenance:
            manifest[field] = detector_provenance[field]
    _write_json_atomic(output_root / "manifest.json", manifest)
    incomplete_marker.unlink()
    return manifest


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


def _portable_path(path: str | Path, base: str | Path | None = None) -> str:
    """Render paths relative to the repository for portable manifests."""

    resolved = Path(path).expanduser().resolve()
    anchor = Path(base or REPOSITORY_ROOT).expanduser().resolve()
    return Path(os.path.relpath(resolved, start=anchor)).as_posix()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--weight-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Optional non-square input size; overrides --base-size",
    )
    parser.add_argument("--resize-mode", choices=("resize", "letterbox"), default="resize")
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-file")
    parser.add_argument("--source-dataset")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    base_size: int | Sequence[int] = args.input_size or args.base_size
    manifest = export_score_maps(
        dataset_dir=args.dataset_dir,
        weight_path=args.weight_path,
        output_dir=args.output_dir,
        base_size=base_size,
        resize_mode=args.resize_mode,
        split=args.split,
        split_file=args.split_file,
        source_dataset=args.source_dataset,
        device=args.device,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    print(
        f"Exported {manifest['num_images']} native-resolution score maps to "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
