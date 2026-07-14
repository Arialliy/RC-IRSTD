"""Freeze deterministic AAAI-27 development splits from official train only.

The public NUAA-SIRST, NUDT-SIRST and IRSTD-1K releases used by this
repository are unordered image benchmarks rather than temporal sequences.
This tool therefore creates two explicit, role-dependent contracts:

* an 80/20 detector fit/diagnostic partition; and
* non-overlapping IID context/query blocks for future meta-calibration.

Detector and meta IDs may overlap *across different nested-fold roles*.  A
domain can never be a detector source and a pseudo-target in the same fold.
Official test IDs are read only to prove disjointness and are never emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data_ext.split_utils import (
    read_split_entries,
    resolve_split_file,
    sample_id_from_entry,
)


SCHEMA_VERSION_V1 = "rc-irstd.aaai27-official-train-splits.v1"
SCHEMA_VERSION_V2 = "rc-irstd.aaai27-official-train-splits.v2"
QUARANTINE_SCHEMA_VERSION = "rc-irstd.aaai27-near-duplicate-quarantine.v1"
# Backward-compatible public constant for callers that create unquarantined
# fixture splits. Claim-bearing development splits use v2.
SCHEMA_VERSION = SCHEMA_VERSION_V1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def text_bytes(values: Iterable[str]) -> bytes:
    return ("".join(f"{value}\n" for value in values)).encode("utf-8")


def _derived_seed(base_seed: int, dataset_name: str, purpose: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}::{dataset_name}::{purpose}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _validate_fraction(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result


def _sample_ids(split_file: Path) -> list[str]:
    result = [sample_id_from_entry(item) for item in read_split_entries(split_file)]
    if len(result) != len(set(result)):
        raise ValueError(f"split resolves to duplicate sample IDs: {split_file}")
    return result


def _load_quarantine(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[Mapping[str, Any], dict[str, set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("quarantine config root must be an object")
    if payload.get("schema_version") != QUARANTINE_SCHEMA_VERSION:
        raise ValueError("unsupported near-duplicate quarantine schema")
    if payload.get("status") != "resolved_by_development_quarantine":
        raise ValueError("quarantine config is not resolved")
    policy = payload.get("decision_policy")
    if not isinstance(policy, Mapping):
        raise TypeError("quarantine decision_policy must be an object")
    if any(
        policy.get(key) is not False
        for key in (
            "official_test_labels_read",
            "raw_data_modified",
            "official_split_files_modified",
        )
    ):
        raise ValueError("quarantine violates the image-only non-mutating policy")

    for evidence_name, evidence in (
        ("source_audit", payload.get("source_audit")),
        ("visual_review", payload.get("visual_review")),
    ):
        if not isinstance(evidence, Mapping):
            raise TypeError(f"quarantine {evidence_name} must be an object")
        relative = Path(str(evidence.get("path") or evidence.get("preview_path")))
        if relative.is_absolute():
            raise ValueError(f"quarantine {evidence_name} path must be relative")
        resolved = (repository_root / relative).resolve()
        try:
            resolved.relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValueError(
                f"quarantine {evidence_name} path escapes repository"
            ) from error
        expected_hash = str(
            evidence.get("sha256") or evidence.get("preview_sha256")
        )
        if not resolved.is_file() or sha256_file(resolved) != expected_hash:
            raise ValueError(f"quarantine {evidence_name} evidence hash drift")

    decisions = payload.get("candidate_decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("quarantine must record every candidate decision")
    candidate_ids = [str(item.get("candidate_id")) for item in decisions]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("quarantine candidate decisions are not unique")
    if len(decisions) != int(payload["source_audit"]["confirmed_pair_count"]):
        raise ValueError("quarantine decision count differs from source audit")
    if any(
        item.get("final_decision") != "same_scene_related"
        or item.get("action")
        != "exclude_official_train_member_from_all_development_roles"
        for item in decisions
    ):
        raise ValueError("quarantine contains an unsupported candidate decision")

    exclusions: dict[str, set[str]] = {}
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise TypeError("quarantine datasets must be an array")
    for raw in datasets:
        if not isinstance(raw, Mapping):
            raise TypeError("quarantine dataset entry must be an object")
        name = str(raw["dataset_name"])
        if name in exclusions:
            raise ValueError(f"duplicate quarantine dataset: {name}")
        values = [str(value) for value in raw["excluded_official_train_ids"]]
        if len(values) != len(set(values)) or len(values) != int(raw["excluded_count"]):
            raise ValueError(f"invalid quarantine ID list for {name}")
        exclusions[name] = set(values)
    decision_exclusions: dict[str, set[str]] = {}
    for item in decisions:
        decision_exclusions.setdefault(str(item["dataset_name"]), set()).add(
            str(item["official_train_image_id"])
        )
    if exclusions != decision_exclusions:
        raise ValueError("quarantine dataset exclusions differ from pair decisions")
    if sum(len(values) for values in exclusions.values()) != int(
        payload["total_excluded_official_train_ids"]
    ):
        raise ValueError("quarantine total excluded count is inconsistent")
    return payload, exclusions


def detector_partition(
    image_ids: Sequence[str],
    *,
    diagnostic_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Return stable fit/diagnostic IDs while retaining official split order."""

    fraction = _validate_fraction(diagnostic_fraction, "diagnostic_fraction")
    if len(image_ids) < 2:
        raise ValueError("detector partition requires at least two images")
    diagnostic_count = max(
        1,
        min(len(image_ids) - 1, int(round(len(image_ids) * fraction))),
    )
    permutation = np.random.default_rng(seed).permutation(len(image_ids))
    diagnostic_indices = {int(value) for value in permutation[:diagnostic_count]}
    fit = [value for index, value in enumerate(image_ids) if index not in diagnostic_indices]
    diagnostic = [
        value for index, value in enumerate(image_ids) if index in diagnostic_indices
    ]
    if set(fit).intersection(diagnostic) or set(fit).union(diagnostic) != set(image_ids):
        raise RuntimeError("detector partition is not an exact disjoint cover")
    return fit, diagnostic


def iid_meta_windows(
    image_ids: Sequence[str],
    *,
    context_size: int,
    query_size: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Create non-overlapping IID blocks and a group-level train/val split."""

    fraction = _validate_fraction(validation_fraction, "validation_fraction")
    if context_size <= 0 or query_size <= 0:
        raise ValueError("context_size and query_size must be positive")
    block_size = int(context_size + query_size)
    block_count = len(image_ids) // block_size
    if block_count < 2:
        raise ValueError(
            "meta split requires at least two non-overlapping context/query blocks; "
            f"got {len(image_ids)} images for block size {block_size}"
        )

    permutation = np.random.default_rng(seed).permutation(len(image_ids)).tolist()
    blocks: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for block_index in range(block_count):
        start = block_index * block_size
        indices = [int(value) for value in permutation[start : start + block_size]]
        if used_indices.intersection(indices):
            raise RuntimeError("IID meta blocks overlap")
        used_indices.update(indices)
        context_indices = indices[:context_size]
        query_indices = indices[context_size:]
        blocks.append(
            {
                "window_id": f"iid_block_{block_index:06d}",
                "protocol": "iid",
                "temporal_causality_claimed": False,
                "context_image_ids": [image_ids[index] for index in context_indices],
                "query_image_ids": [image_ids[index] for index in query_indices],
            }
        )

    validation_count = max(
        1,
        min(block_count - 1, int(round(block_count * fraction))),
    )
    validation_permutation = np.random.default_rng(seed + 1).permutation(block_count)
    validation_indices = {
        int(value) for value in validation_permutation[:validation_count]
    }
    train = [block for index, block in enumerate(blocks) if index not in validation_indices]
    validation = [
        block for index, block in enumerate(blocks) if index in validation_indices
    ]
    unused = [
        image_ids[index]
        for index in range(len(image_ids))
        if index not in used_indices
    ]

    train_ids = {
        value
        for block in train
        for key in ("context_image_ids", "query_image_ids")
        for value in block[key]
    }
    validation_ids = {
        value
        for block in validation
        for key in ("context_image_ids", "query_image_ids")
        for value in block[key]
    }
    if train_ids.intersection(validation_ids):
        raise RuntimeError("meta train/validation images overlap")
    if train_ids.union(validation_ids).intersection(unused):
        raise RuntimeError("unused meta IDs overlap emitted windows")
    if train_ids.union(validation_ids).union(unused) != set(image_ids):
        raise RuntimeError("meta windows and unused IDs do not cover official train")
    return train, validation, unused


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path


def parse_dataset_spec(value: str) -> DatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("--dataset must use non-empty NAME=PATH")
    root = Path(raw_path).expanduser().resolve()
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"dataset directory does not exist: {root}")
    return DatasetSpec(name=name, root=root)


def _portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_files(
    datasets: Sequence[DatasetSpec],
    *,
    repository_root: Path,
    output_dir: Path,
    seed: int,
    detector_diagnostic_fraction: float,
    meta_validation_fraction: float,
    context_size: int,
    query_size: int,
    quarantine_config: Path | None = None,
) -> dict[Path, bytes]:
    if not datasets:
        raise ValueError("at least one dataset is required")
    names = [item.name for item in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"dataset names must be unique: {names}")

    quarantine_payload: Mapping[str, Any] | None = None
    exclusions: dict[str, set[str]] = {}
    if quarantine_config is not None:
        quarantine_config = quarantine_config.resolve()
        quarantine_payload, exclusions = _load_quarantine(
            quarantine_config,
            repository_root=repository_root,
        )
        unknown_datasets = set(exclusions).difference(names)
        if unknown_datasets:
            raise ValueError(
                f"quarantine contains datasets absent from split request: {sorted(unknown_datasets)}"
            )
    schema_version = (
        SCHEMA_VERSION_V2 if quarantine_payload is not None else SCHEMA_VERSION_V1
    )

    files: dict[Path, bytes] = {}
    dataset_summaries: list[dict[str, Any]] = []
    for spec in datasets:
        train_split = resolve_split_file(spec.root, split="train")
        test_split = resolve_split_file(spec.root, split="test")
        train_ids = _sample_ids(train_split)
        test_ids = _sample_ids(test_split)
        overlap = sorted(set(train_ids).intersection(test_ids))
        if overlap:
            raise ValueError(
                f"official train/test IDs overlap for {spec.name}: {overlap[:10]}"
            )
        quarantined = exclusions.get(spec.name, set())
        unknown_ids = quarantined.difference(train_ids)
        if unknown_ids:
            raise ValueError(
                f"quarantine IDs are absent from {spec.name} official train: "
                f"{sorted(unknown_ids)[:10]}"
            )
        if quarantined.intersection(test_ids):
            raise ValueError(f"{spec.name} quarantine unexpectedly contains test IDs")
        effective_train_ids = [
            image_id for image_id in train_ids if image_id not in quarantined
        ]
        quarantined_ids = [
            image_id for image_id in train_ids if image_id in quarantined
        ]
        if (
            set(effective_train_ids).intersection(quarantined_ids)
            or set(effective_train_ids).union(quarantined_ids) != set(train_ids)
        ):
            raise RuntimeError("effective development train is not a quarantine partition")

        fit_seed = _derived_seed(seed, spec.name, "detector")
        meta_seed = _derived_seed(seed, spec.name, "meta")
        fit, diagnostic = detector_partition(
            effective_train_ids,
            diagnostic_fraction=detector_diagnostic_fraction,
            seed=fit_seed,
        )
        meta_train, meta_validation, meta_unused = iid_meta_windows(
            effective_train_ids,
            context_size=context_size,
            query_size=query_size,
            validation_fraction=meta_validation_fraction,
            seed=meta_seed,
        )

        slug = spec.name.lower().replace("_", "-")
        relative_files = {
            "detector_fit": Path(slug) / "detector_fit.txt",
            "detector_diagnostic": Path(slug) / "detector_diagnostic.txt",
            "meta_train": Path(slug) / "meta_train_windows.json",
            "meta_validation": Path(slug) / "meta_validation_windows.json",
            "meta_unused": Path(slug) / "meta_unused_ids.txt",
        }
        if quarantine_payload is not None:
            relative_files.update(
                {
                    "effective_development_train": Path(slug)
                    / "effective_development_train.txt",
                    "quarantined_official_train": Path(slug)
                    / "quarantined_official_train_ids.txt",
                }
            )
        payloads = {
            "detector_fit": text_bytes(fit),
            "detector_diagnostic": text_bytes(diagnostic),
            "meta_train": json_bytes(
                {
                    "schema_version": schema_version,
                    "dataset_name": spec.name,
                    "role": "meta_train",
                    "windows": meta_train,
                }
            ),
            "meta_validation": json_bytes(
                {
                    "schema_version": schema_version,
                    "dataset_name": spec.name,
                    "role": "meta_validation",
                    "windows": meta_validation,
                }
            ),
            "meta_unused": text_bytes(meta_unused),
        }
        if quarantine_payload is not None:
            payloads.update(
                {
                    "effective_development_train": text_bytes(effective_train_ids),
                    "quarantined_official_train": text_bytes(quarantined_ids),
                }
            )
        for key, relative in relative_files.items():
            files[output_dir / relative] = payloads[key]

        dataset_summary: dict[str, Any] = {
                "dataset_name": spec.name,
                "dataset_type": "iid_images",
                "dataset_root": _portable_path(spec.root, repository_root),
                "official_train_split": _portable_path(train_split, repository_root),
                "official_train_split_sha256": sha256_file(train_split),
                "official_train_count": len(train_ids),
                "official_test_split": _portable_path(test_split, repository_root),
                "official_test_split_sha256": sha256_file(test_split),
                "official_test_count": len(test_ids),
                "official_train_test_id_overlap_count": 0,
                "derived_seeds": {
                    "detector": fit_seed,
                    "meta": meta_seed,
                },
                "detector": {
                    "fit_count": len(fit),
                    "diagnostic_count": len(diagnostic),
                    "fit_file": relative_files["detector_fit"].as_posix(),
                    "fit_sha256": sha256_bytes(payloads["detector_fit"]),
                    "diagnostic_file": relative_files[
                        "detector_diagnostic"
                    ].as_posix(),
                    "diagnostic_sha256": sha256_bytes(
                        payloads["detector_diagnostic"]
                    ),
                },
                "meta": {
                    "protocol": "iid_non_overlapping_blocks",
                    "temporal_causality_claimed": False,
                    "context_size": context_size,
                    "query_size": query_size,
                    "train_window_count": len(meta_train),
                    "validation_window_count": len(meta_validation),
                    "unused_image_count": len(meta_unused),
                    "train_file": relative_files["meta_train"].as_posix(),
                    "train_sha256": sha256_bytes(payloads["meta_train"]),
                    "validation_file": relative_files[
                        "meta_validation"
                    ].as_posix(),
                    "validation_sha256": sha256_bytes(
                        payloads["meta_validation"]
                    ),
                    "unused_file": relative_files["meta_unused"].as_posix(),
                    "unused_sha256": sha256_bytes(payloads["meta_unused"]),
                },
            }
        if quarantine_payload is not None:
            dataset_summary["development_quarantine"] = {
                "quarantined_count": len(quarantined_ids),
                "quarantined_file": relative_files[
                    "quarantined_official_train"
                ].as_posix(),
                "quarantined_sha256": sha256_bytes(
                    payloads["quarantined_official_train"]
                ),
                "effective_development_train_count": len(effective_train_ids),
                "effective_development_train_file": relative_files[
                    "effective_development_train"
                ].as_posix(),
                "effective_development_train_sha256": sha256_bytes(
                    payloads["effective_development_train"]
                ),
                "partition_of_official_train": True,
            }
        dataset_summaries.append(dataset_summary)

    manifest = {
        "schema_version": schema_version,
        "artifact_type": "official_train_derived_role_splits",
        "base_seed": int(seed),
        "detector_diagnostic_fraction": float(detector_diagnostic_fraction),
        "meta_validation_fraction": float(meta_validation_fraction),
        "role_contract": {
            "official_test_emitted": False,
            "official_test_labels_read_for_quarantine": False,
            "outer_target_official_train_used": False,
            "same_fold_domain_roles_are_mutually_exclusive": True,
            "cross_fold_role_reuse": (
                "allowed only because every nested fold trains a new detector and a "
                "domain is either detector_source or pseudo_target, never both"
            ),
            "detector_checkpoint_selection": "fixed_last",
            "detector_diagnostic_used_for_checkpoint_selection": False,
        },
        "datasets": dataset_summaries,
    }
    if quarantine_payload is not None and quarantine_config is not None:
        manifest["development_quarantine"] = {
            "status": "applied_before_random_partitioning",
            "config_path": _portable_path(quarantine_config, repository_root),
            "config_sha256": sha256_file(quarantine_config),
            "source_audit_path": quarantine_payload["source_audit"]["path"],
            "source_audit_sha256": quarantine_payload["source_audit"]["sha256"],
            "raw_data_modified": False,
            "official_split_files_modified": False,
        }
    files[output_dir / "manifest.json"] = json_bytes(manifest)
    return files


def write_files(files: Mapping[Path, bytes]) -> None:
    output_roots = {path.parents[1] for path in files if path.name != "manifest.json"}
    if not output_roots:
        output_roots = {next(iter(files)).parent}
    output_dir = next(iter(files)).parent
    if (output_dir / "manifest.json") not in files:
        # Nested file happened to be first; find the manifest parent explicitly.
        output_dir = next(path.parent for path in files if path.name == "manifest.json")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty frozen split directory: {output_dir}"
        )
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def check_files(files: Mapping[Path, bytes]) -> None:
    expected_paths = set(files)
    output_dir = next(path.parent for path in files if path.name == "manifest.json")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"frozen split directory does not exist: {output_dir}")
    actual_paths = {path for path in output_dir.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise ValueError(f"frozen split file set differs; missing={missing}, extra={extra}")
    mismatches = [
        str(path)
        for path, expected in files.items()
        if path.read_bytes() != expected
    ]
    if mismatches:
        raise ValueError(f"frozen split contents are not reproducible: {mismatches}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=parse_dataset_spec, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector-diagnostic-fraction", type=float, default=0.20)
    parser.add_argument("--meta-validation-fraction", type=float, default=0.20)
    parser.add_argument("--context-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument(
        "--quarantine-config",
        help=(
            "Optional repository-local reviewed near-duplicate quarantine. "
            "When present, emit the v2 effective-development contract."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repository_root = Path(args.repository_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = repository_root / output_dir
    output_dir = output_dir.resolve()
    quarantine_config: Path | None = None
    if args.quarantine_config:
        quarantine_config = Path(args.quarantine_config).expanduser()
        if not quarantine_config.is_absolute():
            quarantine_config = repository_root / quarantine_config
        quarantine_config = quarantine_config.resolve()
        if not quarantine_config.is_file():
            raise FileNotFoundError(quarantine_config)
    files = build_files(
        args.dataset,
        repository_root=repository_root,
        output_dir=output_dir,
        seed=args.seed,
        detector_diagnostic_fraction=args.detector_diagnostic_fraction,
        meta_validation_fraction=args.meta_validation_fraction,
        context_size=args.context_size,
        query_size=args.query_size,
        quarantine_config=quarantine_config,
    )
    if args.write:
        write_files(files)
        action = "written"
    else:
        check_files(files)
        action = "verified"
    manifest_path = output_dir / "manifest.json"
    print(
        json.dumps(
            {
                "status": "PASS",
                "action": action,
                "manifest": _portable_path(manifest_path, repository_root),
                "manifest_sha256": sha256_file(manifest_path),
                "file_count": len(files),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
