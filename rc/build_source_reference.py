"""Build an auditable source-domain statistics reference from score manifests.

Only unlabeled sigmoid probability maps and, when consistently available,
original grayscale images are consumed.  Ground-truth masks may coexist in an
exported NPZ, but this module never reads them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .domain_statistics import (
    BASE_FEATURE_DIM,
    extract_unlabeled_statistics,
    load_probability_and_grayscale,
    load_source_reference,
)
from .schema import SourceReference, StatisticsConfig


_CHECKPOINT_SHA_FIELDS = (
    "weight_sha256",
    "detector_checkpoint_sha",
    "detector_weight_sha256",
    "checkpoint_sha256",
)
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_SCALE_FLOOR = 1e-8


def _read_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"score manifest must contain a JSON object: {path}")
    return payload


def _nonempty_string(value: Any, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _string_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of domain names")
    result = tuple(_nonempty_string(item, name) for item in value)
    if not result and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique domain names")
    return result


def _checkpoint_sha(payload: Mapping[str, Any]) -> str:
    values = [str(payload[field]).lower() for field in _CHECKPOINT_SHA_FIELDS if field in payload]
    if not values:
        raise KeyError(
            "score manifest is missing detector checkpoint SHA-256 "
            f"({', '.join(_CHECKPOINT_SHA_FIELDS)})"
        )
    if len(set(values)) != 1:
        raise ValueError("score manifest contains conflicting detector checkpoint SHA fields")
    value = values[0]
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("detector checkpoint SHA must be lowercase 64-character hexadecimal")
    return value


def _manifest_contract_value(payload: Mapping[str, Any], key: str) -> Any:
    """Read a top-level contract field and audit its nested duplicate."""

    nested = payload.get("detector_provenance")
    nested_present = isinstance(nested, Mapping) and key in nested
    nested_value = nested[key] if nested_present else None
    top_present = key in payload
    if top_present and nested_present and payload[key] != nested_value:
        raise ValueError(f"score manifest has conflicting top-level/nested {key}")
    return payload[key] if top_present else nested_value


def _manifest_items(payload: Mapping[str, Any], path: Path) -> tuple[Mapping[str, Any], ...]:
    raw_items = payload.get("items", payload.get("records"))
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"score manifest requires a non-empty items/records list: {path}")
    items: list[Mapping[str, Any]] = []
    image_ids: list[str] = []
    score_files: list[str] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            raise TypeError(f"score manifest item {index} must be a mapping: {path}")
        image_id = _nonempty_string(item.get("image_id", ""), f"items[{index}].image_id")
        score_file = item.get("file", item.get("prob_path", item.get("score_path")))
        if score_file is None:
            raise KeyError(f"score manifest item {image_id!r} has no score-map path")
        image_ids.append(image_id)
        score_files.append(str(score_file))
        items.append(item)
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"score manifest image IDs must be unique: {path}")
    if len(set(score_files)) != len(score_files):
        raise ValueError(f"score manifest score-map paths must be unique: {path}")
    if int(payload.get("num_images", len(items))) != len(items):
        raise ValueError(f"score manifest num_images disagrees with items: {path}")
    return tuple(items)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _dataset_image_index(payload: Mapping[str, Any], manifest_path: Path) -> Mapping[str, Path]:
    dataset_value = payload.get("dataset_dir")
    if dataset_value in (None, ""):
        return {}
    dataset_root = _resolve_path(manifest_path.parent, str(dataset_value))
    image_root = dataset_root / "images"
    if not image_root.is_dir():
        return {}
    result: dict[str, Path] = {}
    for candidate in image_root.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if candidate.stem in result:
            raise ValueError(
                f"multiple grayscale images share ID {candidate.stem!r} under {image_root}"
            )
        result[candidate.stem] = candidate
    return result


def _score_path(item: Mapping[str, Any], manifest_path: Path) -> Path:
    value = item.get("file", item.get("prob_path", item.get("score_path")))
    if value is None:
        raise KeyError(f"manifest item {item.get('image_id')!r} has no score-map path")
    path = _resolve_path(manifest_path.parent, str(value))
    if not path.is_file():
        raise FileNotFoundError(f"score map does not exist: {path}")
    return path


def _explicit_gray_path(item: Mapping[str, Any], manifest_path: Path) -> Path | None:
    # Deliberately do not inspect mask_path or any other label-bearing field.
    value = item.get("gray_path", item.get("image_path"))
    if value in (None, ""):
        return None
    path = _resolve_path(manifest_path.parent, str(value))
    if not path.is_file():
        raise FileNotFoundError(f"grayscale image does not exist: {path}")
    return path


def _load_domain_inputs(
    payload: Mapping[str, Any],
    manifest_path: Path,
    items: Sequence[Mapping[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    image_index = _dataset_image_index(payload, manifest_path)
    probabilities: list[np.ndarray] = []
    grayscale_images: list[np.ndarray | None] = []
    for item in items:
        image_id = str(item["image_id"])
        probability_path = _score_path(item, manifest_path)
        grayscale_path = _explicit_gray_path(item, manifest_path)
        if grayscale_path is None:
            grayscale_path = image_index.get(image_id)
        probability, grayscale = load_probability_and_grayscale(
            probability_path,
            grayscale_path,
        )
        if probability.size == 0 or not np.isfinite(probability).all():
            raise ValueError(f"score map contains non-finite probabilities: {probability_path}")
        if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
            raise ValueError(f"score map is outside probability range [0, 1]: {probability_path}")
        if grayscale is not None and grayscale.shape != probability.shape:
            raise ValueError(
                f"grayscale/score-map shape mismatch for {image_id!r}: "
                f"{grayscale.shape} != {probability.shape}"
            )
        with np.load(probability_path, allow_pickle=False) as score_payload:
            if "image_id" in score_payload:
                stored_id = str(np.asarray(score_payload["image_id"]).item())
                if stored_id != image_id:
                    raise ValueError(
                        f"score-map image_id mismatch: {stored_id!r} != {image_id!r}"
                    )
        probabilities.append(probability)
        grayscale_images.append(grayscale)
    availability = [value is not None for value in grayscale_images]
    if any(availability) and not all(availability):
        raise ValueError(
            f"grayscale availability must be all-or-none within domain "
            f"{payload.get('target_dataset')!r}"
        )
    return probabilities, (
        [np.asarray(value) for value in grayscale_images if value is not None]
        if all(availability)
        else None
    )


def _optional_contract_tuple(
    payload: Mapping[str, Any],
    key: str,
) -> tuple[str, ...] | None:
    value = _manifest_contract_value(payload, key)
    if value is None:
        return None
    return _string_tuple(value, key, allow_empty=True)


def _optional_contract_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = _manifest_contract_value(payload, key)
    if value is None:
        return None
    return _nonempty_string(value, key)


def _load_persisted_source_contract(path: Path) -> Mapping[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if "source_contract_json" not in payload:
            raise KeyError("source reference NPZ is missing source_contract_json")
        contract_text = str(np.asarray(payload["source_contract_json"]).item())
    contract = json.loads(contract_text)
    if not isinstance(contract, Mapping):
        raise ValueError("source_contract_json must contain a JSON object")
    return contract


def _atomic_save_reference(
    path: Path,
    *,
    domains: Sequence[str],
    centers: np.ndarray,
    scale: np.ndarray,
    statistics_config: StatisticsConfig,
    source_contract: Mapping[str, Any],
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"source reference already exists: {path}; pass --overwrite")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                domains=np.asarray(tuple(domains), dtype=np.str_),
                centers=np.asarray(centers, dtype=np.float32),
                scale=np.asarray(scale, dtype=np.float32),
                statistics_config_json=np.asarray(
                    json.dumps(
                        statistics_config.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
                source_contract_json=np.asarray(
                    json.dumps(
                        source_contract,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_source_reference(
    score_manifests: Sequence[str | Path],
    output: str | Path,
    *,
    statistics_config: StatisticsConfig,
    domains: Sequence[str] | None = None,
    overwrite: bool = False,
) -> SourceReference:
    """Build one fold-specific source reference and verify its persisted SHA."""

    manifest_paths = tuple(Path(value).expanduser().resolve() for value in score_manifests)
    if not manifest_paths:
        raise ValueError("at least one --score-manifest is required")
    if len(set(manifest_paths)) != len(manifest_paths):
        raise ValueError("score manifests must be unique")

    manifests: list[Mapping[str, Any]] = []
    targets: list[str] = []
    item_sets: list[tuple[Mapping[str, Any], ...]] = []
    checkpoint_sha: str | None = None
    source_contract: tuple[str, ...] | None = None
    fold_contract: dict[str, Any] | None = None
    for path in manifest_paths:
        if not path.is_file():
            raise FileNotFoundError(f"score manifest does not exist: {path}")
        if (path.parent / ".export_incomplete").exists():
            raise RuntimeError(f"score export is incomplete and unsafe to consume: {path.parent}")
        payload = _read_json_mapping(path)
        target = _nonempty_string(payload.get("target_dataset", ""), "target_dataset")
        score_type = payload.get("score_type")
        if score_type is not None and str(score_type) != "sigmoid_probability":
            raise ValueError(f"unsupported score_type {score_type!r}; expected sigmoid_probability")
        current_sha = _checkpoint_sha(payload)
        raw_sources = _manifest_contract_value(payload, "detector_source_domains")
        if raw_sources is None:
            raise KeyError("score manifest is missing detector_source_domains")
        current_sources = _string_tuple(raw_sources, "detector_source_domains")
        if target not in current_sources:
            raise ValueError(
                f"source-manifest target_dataset {target!r} is not in "
                "detector_source_domains"
            )
        target_exclusion = _manifest_contract_value(payload, "target_exclusion_verified")
        if target_exclusion is not None:
            if not isinstance(target_exclusion, bool):
                raise TypeError("target_exclusion_verified must be boolean when present")
            if target_exclusion:
                raise ValueError(
                    "a source-domain manifest cannot claim target_exclusion_verified=true"
                )
        if checkpoint_sha is None:
            checkpoint_sha = current_sha
            source_contract = current_sources
        elif current_sha != checkpoint_sha:
            raise ValueError("all score manifests must use the same detector checkpoint")
        elif current_sources != source_contract:
            raise ValueError("all score manifests must use the same detector source contract")

        current_fold_contract = {
            "outer_fold_id": _optional_contract_string(payload, "outer_fold_id"),
            "outer_target": _optional_contract_string(payload, "outer_target"),
            "held_out_domains": list(
                _optional_contract_tuple(payload, "held_out_domains") or ()
            ),
            "protocol_scope": _optional_contract_string(payload, "protocol_scope"),
        }
        if fold_contract is None:
            fold_contract = current_fold_contract
        elif current_fold_contract != fold_contract:
            raise ValueError(
                "all score manifests must use the same outer/held-out/protocol contract"
            )

        manifests.append(payload)
        targets.append(target)
        item_sets.append(_manifest_items(payload, path))

    if len(set(targets)) != len(targets):
        raise ValueError("score-manifest target_dataset values must be unique")
    if domains is None:
        selected_domains = tuple(targets)
    else:
        if len(domains) != len(manifest_paths):
            raise ValueError("--domain and --score-manifest must be repeated the same number of times")
        selected_domains = tuple(_nonempty_string(value, "domain") for value in domains)
        mismatches = [
            f"{target!r}!={domain!r}"
            for target, domain in zip(targets, selected_domains)
            if target != domain
        ]
        if mismatches:
            raise ValueError(
                "each --domain must match its score manifest target_dataset: "
                + ", ".join(mismatches)
            )
    if len(set(selected_domains)) != len(selected_domains):
        raise ValueError("source domains must be unique")
    assert source_contract is not None
    assert checkpoint_sha is not None
    if (
        set(targets) != set(source_contract)
        or set(selected_domains) != set(source_contract)
        or len(selected_domains) != len(source_contract)
    ):
        raise ValueError(
            "source manifests/domains must cover the detector_source_domains contract exactly"
        )

    assert fold_contract is not None
    held_out = set(fold_contract["held_out_domains"])
    outer_target_value = fold_contract["outer_target"]
    forbidden = held_out | ({str(outer_target_value)} if outer_target_value not in (None, "") else set())
    leaked = sorted((set(targets) | set(selected_domains) | set(source_contract)) & forbidden)
    if leaked:
        raise ValueError("source reference contains held-out/outer-target domains: " + ", ".join(leaked))

    centers_by_domain: dict[str, np.ndarray] = {}
    grayscale_availability: list[bool] = []
    for domain, payload, path, items in zip(
        selected_domains, manifests, manifest_paths, item_sets
    ):
        probabilities, grayscale_images = _load_domain_inputs(payload, path, items)
        grayscale_availability.append(grayscale_images is not None)
        statistics = extract_unlabeled_statistics(
            probabilities,
            grayscale_images,
            statistics_config=statistics_config,
        )
        centers_by_domain[domain] = np.asarray(
            statistics.vector[:BASE_FEATURE_DIM], dtype=np.float64
        )
    if any(grayscale_availability) and not all(grayscale_availability):
        raise ValueError("grayscale availability must be consistent across all source domains")

    # Store rows in the checkpoint contract order, independent of CLI ordering.
    ordered_centers = np.stack(
        [centers_by_domain[domain] for domain in source_contract], axis=0
    )
    scale = ordered_centers.std(axis=0)
    scale = np.where(scale < _SCALE_FLOOR, 1.0, scale)
    persisted_contract = {
        "detector_checkpoint_sha": checkpoint_sha,
        "detector_source_domains": list(source_contract),
        "outer_fold_id": fold_contract["outer_fold_id"],
        "outer_target": fold_contract["outer_target"],
        "held_out_domains": list(fold_contract["held_out_domains"]),
        "protocol_scope": fold_contract["protocol_scope"],
    }
    output_path = Path(output).expanduser().resolve()
    _atomic_save_reference(
        output_path,
        domains=source_contract,
        centers=ordered_centers,
        scale=scale,
        statistics_config=statistics_config,
        source_contract=persisted_contract,
        overwrite=overwrite,
    )
    reference = load_source_reference(
        output_path,
        statistics_config=statistics_config,
    )
    if reference.domains != source_contract:
        raise RuntimeError("source-reference self-check changed the domain contract")
    loaded_contract = dict(_load_persisted_source_contract(output_path))
    if loaded_contract != persisted_contract:
        raise RuntimeError("source-reference self-check changed the detector/fold contract")
    return reference


def _statistics_config_from_args(args: argparse.Namespace) -> StatisticsConfig:
    if args.statistics_config_json is None:
        if args.peak_kernel_size is None or args.peak_min_score is None:
            raise ValueError(
                "provide --statistics-config-json or both --peak-kernel-size and --peak-min-score"
            )
        return StatisticsConfig(
            peak_kernel_size=args.peak_kernel_size,
            peak_min_score=args.peak_min_score,
        )

    raw = str(args.statistics_config_json)
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        payload = _read_json_mapping(candidate.resolve())
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                "--statistics-config-json must be a JSON object or an existing JSON file"
            ) from error
        if not isinstance(payload, Mapping):
            raise ValueError("statistics config JSON must contain an object")
    config = StatisticsConfig.from_dict(payload)
    if args.peak_kernel_size is not None and args.peak_kernel_size != config.peak_kernel_size:
        raise ValueError("--peak-kernel-size disagrees with statistics config JSON")
    if args.peak_min_score is not None and args.peak_min_score != config.peak_min_score:
        raise ValueError("--peak-min-score disagrees with statistics config JSON")
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-manifest",
        action="append",
        required=True,
        help="Source-domain score manifest; repeat once per detector source domain",
    )
    parser.add_argument(
        "--domain",
        action="append",
        help=(
            "Logical detector-source domain paired with each manifest; if omitted, "
            "target_dataset is used"
        ),
    )
    parser.add_argument(
        "--statistics-config-json",
        "--statistics-config",
        dest="statistics_config_json",
        help="Inline StatisticsConfig JSON object or path to a JSON file",
    )
    parser.add_argument("--peak-kernel-size", type=int)
    parser.add_argument("--peak-min-score", type=float)
    parser.add_argument("--output", required=True, help="Output source-reference NPZ")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    statistics_config = _statistics_config_from_args(args)
    reference = build_source_reference(
        args.score_manifest,
        args.output,
        statistics_config=statistics_config,
        domains=args.domain,
        overwrite=args.overwrite,
    )
    persisted_contract = _load_persisted_source_contract(
        Path(args.output).expanduser().resolve()
    )
    print(
        json.dumps(
            {
                "domains": list(reference.domains),
                "feature_dim": len(reference.scale),
                "detector_checkpoint_sha": persisted_contract[
                    "detector_checkpoint_sha"
                ],
                "output": str(Path(args.output).expanduser().resolve()),
                "source_reference_sha256": reference.sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
