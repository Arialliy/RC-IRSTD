"""Audit IRSTD train/test data and nested-LODO claim eligibility.

This command is read-only.  It resolves the exact official splits, checks ID
and image-byte separation, validates image/mask geometry with the same policy
used by evaluation, and emits a machine-readable protocol assessment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from data_ext.dataset_identity import sha256_file
from data_ext.mask_alignment import (
    DEFAULT_ASPECT_TOLERANCE,
    align_mask_to_image,
    aspect_ratio_relative_error,
)
from data_ext.split_utils import (
    read_split_entries,
    resolve_image_and_mask,
    resolve_split_file,
    sample_id_from_entry,
)


def audit_dataset(dataset_dir: str | Path) -> dict[str, object]:
    root = Path(dataset_dir).expanduser().resolve()
    train_path = resolve_split_file(root, "train")
    test_path = resolve_split_file(root, "test")
    split_entries = {
        "train": read_split_entries(train_path),
        "test": read_split_entries(test_path),
    }
    split_ids = {
        role: [sample_id_from_entry(entry) for entry in entries]
        for role, entries in split_entries.items()
    }
    id_overlap = sorted(set(split_ids["train"]) & set(split_ids["test"]))

    image_hashes: dict[str, dict[str, list[str]]] = {"train": {}, "test": {}}
    geometry_mismatches: list[dict[str, object]] = []
    rejected_geometry: list[str] = []
    for role, entries in split_entries.items():
        for entry in entries:
            image_id = sample_id_from_entry(entry)
            image_path, mask_path = resolve_image_and_mask(root, entry)
            digest = sha256_file(image_path)
            image_hashes[role].setdefault(digest, []).append(image_id)
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L")
            if image.size != mask.size:
                error = aspect_ratio_relative_error(image.size, mask.size)
                accepted = True
                try:
                    align_mask_to_image(mask, image, image_id)
                except ValueError as exc:
                    accepted = False
                    rejected_geometry.append(str(exc))
                geometry_mismatches.append(
                    {
                        "split": role,
                        "image_id": image_id,
                        "image_wh": list(image.size),
                        "mask_wh": list(mask.size),
                        "aspect_ratio_relative_error": error,
                        "nearest_alignment_accepted": accepted,
                    }
                )

    content_overlap_hashes = sorted(
        set(image_hashes["train"]) & set(image_hashes["test"])
    )
    content_overlap = [
        {
            "image_sha256": digest,
            "train_ids": image_hashes["train"][digest],
            "test_ids": image_hashes["test"][digest],
        }
        for digest in content_overlap_hashes
    ]
    within_split_content_duplicates = {
        role: [
            {"image_sha256": digest, "image_ids": ids}
            for digest, ids in sorted(image_hashes[role].items())
            if len(ids) > 1
        ]
        for role in ("train", "test")
    }
    within_split_duplicate_count = sum(
        len(groups) for groups in within_split_content_duplicates.values()
    )
    content_index = [
        {
            "image_sha256": digest,
            "split": role,
            "image_ids": ids,
        }
        for role in ("train", "test")
        for digest, ids in sorted(image_hashes[role].items())
    ]
    return {
        "dataset_name": root.name,
        "dataset_root": str(root),
        "train_split": str(train_path),
        "train_split_sha256": sha256_file(train_path),
        "test_split": str(test_path),
        "test_split_sha256": sha256_file(test_path),
        "num_train": len(split_ids["train"]),
        "num_test": len(split_ids["test"]),
        "train_test_id_overlap_count": len(id_overlap),
        "train_test_id_overlap": id_overlap,
        "train_test_image_content_overlap_count": len(content_overlap),
        "train_test_image_content_overlap": content_overlap,
        "within_split_image_content_duplicate_group_count": (
            within_split_duplicate_count
        ),
        "within_split_image_content_duplicates": within_split_content_duplicates,
        # Private build_report input.  The final report retains only actual
        # cross-dataset collision groups, not thousands of non-colliding rows.
        "_image_content_sha256_index": content_index,
        "mask_alignment_aspect_tolerance": DEFAULT_ASPECT_TOLERANCE,
        "geometry_mismatch_count": len(geometry_mismatches),
        "geometry_mismatches": geometry_mismatches,
        "rejected_geometry_count": len(rejected_geometry),
        "rejected_geometry": rejected_geometry,
        "split_contract_passed": not id_overlap
        and not content_overlap
        and not within_split_duplicate_count
        and not rejected_geometry,
    }


def assess_nested_protocol(
    dataset_names: Sequence[str],
    *,
    outer_target: str | None,
    pseudo_target: str | None,
    minimum_detector_sources: int = 2,
) -> dict[str, object]:
    names = list(dataset_names)
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    excluded = [value for value in (outer_target, pseudo_target) if value]
    if len(excluded) != len(set(excluded)):
        raise ValueError("outer_target and pseudo_target must be distinct")
    unknown = sorted(set(excluded) - set(names))
    if unknown:
        raise ValueError(f"held-out domains are not among audited datasets: {unknown}")
    sources = [name for name in names if name not in set(excluded)]
    eligible = len(sources) >= int(minimum_detector_sources)
    return {
        "outer_target": outer_target,
        "pseudo_target": pseudo_target,
        "detector_sources": sources,
        "num_detector_sources": len(sources),
        "minimum_detector_sources": int(minimum_detector_sources),
        "claim_bearing_nested_lodo_eligible": eligible,
        "protocol_scope": (
            "multi_source_protocol_candidate"
            if eligible
            else "single_source_inner_smoke_not_main_result"
        ),
        "reason": (
            None
            if eligible
            else "fewer than two detector sources remain after outer/pseudo holdout"
        ),
    }


def build_report(
    dataset_dirs: Sequence[str | Path],
    *,
    outer_target: str | None = None,
    pseudo_target: str | None = None,
    near_duplicate_audit: str | Path | None = None,
) -> dict[str, object]:
    datasets = [audit_dataset(path) for path in dataset_dirs]
    cross_index: dict[str, list[dict[str, object]]] = {}
    for dataset in datasets:
        for item in dataset["_image_content_sha256_index"]:
            cross_index.setdefault(str(item["image_sha256"]), []).append(
                {
                    "dataset_name": dataset["dataset_name"],
                    "split": item["split"],
                    "image_ids": item["image_ids"],
                }
            )
    cross_dataset_exact_duplicates = [
        {
            "image_sha256": digest,
            "occurrences": occurrences,
        }
        for digest, occurrences in sorted(cross_index.items())
        if len({str(item["dataset_name"]) for item in occurrences}) > 1
    ]
    for dataset in datasets:
        dataset.pop("_image_content_sha256_index", None)
    protocol = assess_nested_protocol(
        [str(item["dataset_name"]) for item in datasets],
        outer_target=outer_target,
        pseudo_target=pseudo_target,
    )
    per_dataset_passed = all(
        bool(item["split_contract_passed"]) for item in datasets
    )
    if near_duplicate_audit is None:
        near_duplicate_record: dict[str, object] = {
            "status": "not_run",
            "claim_boundary": (
                "a registered perceptual near-duplicate method is required before "
                "admitting a fourth domain or claim-bearing outer evaluation"
            ),
        }
        near_duplicate_passed = True
    else:
        near_path = Path(near_duplicate_audit).expanduser().resolve()
        near_payload = json.loads(near_path.read_text(encoding="utf-8"))
        if not isinstance(near_payload, Mapping):
            raise TypeError("near-duplicate audit root must be an object")
        near_duplicate_passed = (
            near_payload.get("status") == "passed"
            and near_payload.get("near_duplicate_contract_passed") is True
            and int(near_payload.get("confirmed_near_duplicate_pair_count", -1)) == 0
            and near_payload.get("image_only") is True
            and near_payload.get("labels_scores_checkpoints_or_metrics_read") is False
        )
        input_names = {
            str(item["dataset_name"])
            for item in near_payload.get("inputs", [])
            if isinstance(item, Mapping)
        }
        dataset_names = {str(item["dataset_name"]) for item in datasets}
        if input_names != dataset_names:
            raise ValueError(
                "near-duplicate audit datasets differ from protocol audit datasets"
            )
        near_duplicate_record = {
            "status": "passed" if near_duplicate_passed else "failed",
            "path": str(near_path),
            "sha256": sha256_file(near_path),
            "confirmed_near_duplicate_pair_count": int(
                near_payload.get("confirmed_near_duplicate_pair_count", -1)
            ),
            "image_only": near_payload.get("image_only"),
            "claim_scope": "effective_development_train_vs_official_test",
        }
    return {
        "artifact_type": "rc_irstd_aaai_protocol_audit",
        "schema_version": "1.0",
        "read_only_audit": True,
        "datasets": datasets,
        "cross_dataset_exact_image_duplicate_group_count": len(
            cross_dataset_exact_duplicates
        ),
        "cross_dataset_exact_image_duplicates": cross_dataset_exact_duplicates,
        "cross_dataset_exact_duplicate_contract_passed": not bool(
            cross_dataset_exact_duplicates
        ),
        "near_duplicate_audit": near_duplicate_record,
        "all_split_contracts_passed": (
            per_dataset_passed
            and not cross_dataset_exact_duplicates
            and near_duplicate_passed
        ),
        "nested_protocol": protocol,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dirs", nargs="+", required=True)
    parser.add_argument("--outer-target", default=None)
    parser.add_argument("--pseudo-target", default=None)
    parser.add_argument("--near-duplicate-audit", default=None)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = build_report(
        args.dataset_dirs,
        outer_target=args.outer_target,
        pseudo_target=args.pseudo_target,
        near_duplicate_audit=args.near_duplicate_audit,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        _write_json_atomic(Path(args.output).expanduser(), report)
    print(rendered)
    return 0 if report["all_split_contracts_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
