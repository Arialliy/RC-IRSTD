"""Causal zero-label adaptation from a context prefix to a later query suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from model.threshold_calibrator import ThresholdCalibrator

from .domain_statistics import (
    extract_unlabeled_statistics,
    load_probability_and_grayscale,
    load_source_reference,
)
from .meta_dataset import FeatureStandardizer
from .schema import BudgetSpec, FoldContract, SCHEMA_VERSION
from .schema import SourceReference, StatisticsConfig


def causal_partition(
    records: Sequence[Mapping[str, Any]],
    *,
    context_size: int,
    query_size: int | None = None,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split an ordered stream at one strict context/query boundary."""

    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if context_size >= len(records):
        raise ValueError("at least one record must remain after the context boundary")
    if query_size is None:
        query_size = len(records) - context_size
    if query_size <= 0 or context_size + query_size > len(records):
        raise ValueError("query_size is outside the available post-context suffix")
    selected = list(records[: context_size + query_size])
    image_ids = [str(record["image_id"]) for record in selected]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("ordered score records must have unique image_id values")
    context = selected[:context_size]
    query = selected[context_size:]
    if {str(item["image_id"]) for item in context}.intersection(
        str(item["image_id"]) for item in query
    ):
        raise RuntimeError("internal context/query overlap")
    return context, query


def _find_grayscale(image_root: Path | None, image_id: str) -> Path | None:
    if image_root is None:
        return None
    matches = sorted(path for path in image_root.glob(f"{image_id}.*") if path.is_file())
    return matches[0] if matches else None


def load_ordered_score_records(
    manifest_or_directory: str | Path,
    *,
    image_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    source = Path(manifest_or_directory).expanduser().resolve()
    if source.is_dir():
        manifest_path = source / "manifest.json"
        if manifest_path.exists():
            source = manifest_path
        else:
            records = [
                {"image_id": path.stem, "prob_path": str(path), "gray_path": None}
                for path in sorted(source.glob("*.npz"))
            ]
            if not records:
                raise FileNotFoundError(f"no score maps found under {source}")
            raise ValueError("audited online adaptation requires a score manifest, not a bare directory")
    with source.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    raw_items = manifest.get("items", manifest.get("records"))
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("score manifest requires a non-empty items/records list")
    if image_dir is not None:
        image_root = Path(image_dir).expanduser().resolve()
    elif manifest.get("dataset_dir"):
        dataset_root = Path(manifest["dataset_dir"]).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = source.parent / dataset_root
        image_root = dataset_root.resolve() / "images"
    else:
        image_root = None
    records = []
    for item in raw_items:
        image_id = str(item["image_id"])
        probability_value = item.get(
            "prob_path", item.get("score_path", item.get("file"))
        )
        if probability_value is None:
            raise KeyError(f"manifest item {image_id!r} lacks file/prob_path/score_path")
        probability_path = Path(probability_value)
        if not probability_path.is_absolute():
            probability_path = source.parent / probability_path
        gray_value = item.get("gray_path", item.get("image_path"))
        gray_path = (
            _find_grayscale(image_root, image_id)
            if gray_value is None
            else Path(gray_value).expanduser()
        )
        if gray_path is not None and not gray_path.is_absolute():
            gray_path = source.parent / gray_path
        records.append(
            {
                "image_id": image_id,
                "prob_path": str(probability_path.resolve()),
                "gray_path": None if gray_path is None else str(gray_path.resolve()),
            }
        )
    return records, manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: str | Path, device: torch.device) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise TypeError("calibrator checkpoint must be a mapping")
    return payload


def _manifest_contract_value(payload: Mapping[str, Any], key: str) -> Any:
    nested = payload.get("detector_provenance")
    nested_present = isinstance(nested, Mapping) and key in nested
    top_present = key in payload
    if top_present and nested_present and payload[key] != nested[key]:
        raise ValueError(f"score manifest has conflicting top-level/nested {key}")
    if top_present:
        return payload[key]
    return nested[key] if nested_present else None


def _deployment_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[FoldContract, SourceReference]:
    reference = SourceReference.from_dict(checkpoint["deployment_source_reference"])
    fold = FoldContract(
        outer_fold_id=str(checkpoint["outer_fold_id"]),
        outer_target=str(checkpoint["outer_target"]),
        detector_source_domains=tuple(
            str(value) for value in checkpoint["deployment_detector_source_domains"]
        ),
        detector_checkpoint_sha=str(checkpoint["deployment_detector_checkpoint_sha"]),
        held_out_domains=tuple(
            str(value) for value in checkpoint["deployment_held_out_domains"]
        ),
        protocol_scope=str(checkpoint["deployment_protocol_scope"]),
    )
    fold.assert_matches_source_reference(reference)
    if fold.protocol_scope != "multi_source_protocol_candidate":
        raise ValueError("calibrator deployment contract is not main-protocol eligible")
    return fold, reference


def load_calibrator_bundle(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[ThresholdCalibrator, FeatureStandardizer, Mapping[str, Any]]:
    checkpoint = _torch_load(checkpoint_path, device)
    required = {
        "model_state_dict",
        "input_dim",
        "hidden_dim",
        "dropout",
        "standardizer",
        "statistics_feature_names",
        "input_feature_names",
        "statistics_config",
        "p_min",
        "outer_fold_id",
        "outer_target",
        "calibration_pseudo_targets",
        "deployment_detector_source_domains",
        "deployment_detector_checkpoint_sha",
        "deployment_held_out_domains",
        "deployment_protocol_scope",
        "deployment_source_reference",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"calibrator checkpoint is missing: {sorted(missing)}")
    model = ThresholdCalibrator(
        int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    standardizer = FeatureStandardizer.from_dict(checkpoint["standardizer"])
    if tuple(checkpoint["input_feature_names"]) != standardizer.feature_names:
        raise ValueError("checkpoint input feature names disagree with standardizer")
    _deployment_contract(checkpoint)
    return model, standardizer, checkpoint


@torch.no_grad()
def adapt_context_to_query(
    *,
    model: ThresholdCalibrator,
    standardizer: FeatureStandardizer,
    checkpoint_metadata: Mapping[str, Any],
    context_records: Sequence[Mapping[str, Any]],
    query_records: Sequence[Mapping[str, Any]],
    budgets: BudgetSpec,
    source_reference: SourceReference,
    score_manifest: Mapping[str, Any],
    score_manifest_sha256: str,
    device: torch.device,
    target_domain: str,
    reject_probability: float | None = None,
    temporal_order_verified: bool = False,
) -> dict[str, Any]:
    """Estimate one threshold using context only, then bind it to later query IDs."""

    context_ids = tuple(str(record["image_id"]) for record in context_records)
    query_ids = tuple(str(record["image_id"]) for record in query_records)
    overlap = set(context_ids).intersection(query_ids)
    if overlap:
        raise ValueError(f"context/query image IDs overlap: {sorted(overlap)}")
    outer_target = str(checkpoint_metadata["outer_target"])
    if target_domain != outer_target:
        raise ValueError("online target_domain must equal checkpoint outer_target")
    deployment_fold, embedded_reference = _deployment_contract(checkpoint_metadata)
    calibration_targets = tuple(
        str(value) for value in checkpoint_metadata["calibration_pseudo_targets"]
    )
    if target_domain in calibration_targets:
        raise ValueError("online target leaked into calibration pseudo-targets")
    manifest_target = str(score_manifest.get("target_dataset", ""))
    if manifest_target != target_domain:
        raise ValueError("score manifest target_dataset must equal online target")
    if str(_manifest_contract_value(score_manifest, "outer_fold_id") or "") != str(
        checkpoint_metadata["outer_fold_id"]
    ):
        raise ValueError("score manifest outer_fold_id differs from calibrator contract")
    if str(_manifest_contract_value(score_manifest, "outer_target") or "") != outer_target:
        raise ValueError("score manifest outer_target differs from calibrator contract")
    manifest_protocol_scope = _manifest_contract_value(score_manifest, "protocol_scope")
    if manifest_protocol_scope != deployment_fold.protocol_scope:
        raise ValueError("score manifest protocol_scope differs from calibrator contract")
    if manifest_protocol_scope != "multi_source_protocol_candidate":
        raise ValueError("online main protocol requires a multi-source detector checkpoint")
    if _manifest_contract_value(score_manifest, "target_exclusion_verified") is not True:
        raise ValueError("score manifest does not verify target-domain exclusion")
    manifest_checkpoint_sha = str(score_manifest.get("weight_sha256", "")).lower()
    expected_checkpoint_sha = str(
        checkpoint_metadata["deployment_detector_checkpoint_sha"]
    ).lower()
    if manifest_checkpoint_sha != expected_checkpoint_sha:
        raise ValueError("score manifest detector checkpoint differs from calibrator contract")
    manifest_source_value = _manifest_contract_value(
        score_manifest, "detector_source_domains"
    )
    if manifest_source_value is None:
        raise ValueError("score manifest must record detector_source_domains")
    manifest_sources = tuple(str(value) for value in manifest_source_value)
    expected_sources = tuple(
        str(value) for value in checkpoint_metadata["deployment_detector_source_domains"]
    )
    if manifest_sources != expected_sources:
        raise ValueError("score manifest detector_source_domains differ from checkpoint")
    if target_domain in manifest_sources:
        raise ValueError("online target must not be a detector source domain")
    manifest_held_out = tuple(
        str(value)
        for value in (
            _manifest_contract_value(score_manifest, "held_out_domains") or ()
        )
    )
    if manifest_held_out != deployment_fold.held_out_domains:
        raise ValueError("score manifest held_out_domains differ from checkpoint")
    if target_domain not in manifest_held_out:
        raise ValueError("online target must occur in detector held_out_domains")
    manifest_items = score_manifest.get("items", score_manifest.get("records"))
    if not isinstance(manifest_items, list):
        raise ValueError("score manifest requires ordered items/records")
    manifest_ids = tuple(str(item["image_id"]) for item in manifest_items)
    expected_prefix = context_ids + query_ids
    if manifest_ids[: len(expected_prefix)] != expected_prefix:
        raise ValueError("context+query IDs must exactly equal the score manifest prefix")
    if source_reference != embedded_reference:
        raise ValueError("online source reference differs from checkpoint deployment reference")
    statistics_config = StatisticsConfig.from_dict(
        checkpoint_metadata["statistics_config"]
    )
    probabilities = []
    grays = []
    for record in context_records:
        probability, grayscale = load_probability_and_grayscale(
            record["prob_path"], record.get("gray_path")
        )
        probabilities.append(probability)
        grays.append(grayscale)
    if any(value is None for value in grays) and not all(value is None for value in grays):
        raise ValueError("context grayscale availability must be all-or-none")
    statistics = extract_unlabeled_statistics(
        probabilities,
        None if all(value is None for value in grays) else grays,
        source_reference=source_reference,
        statistics_config=statistics_config,
    )
    expected_statistics_names = tuple(checkpoint_metadata["statistics_feature_names"])
    if statistics.feature_names != expected_statistics_names:
        raise ValueError("online statistics schema differs from calibrator checkpoint")
    raw_features = np.asarray(
        tuple(float(value) for value in statistics.vector) + budgets.encoded(),
        dtype=np.float64,
    )[None, :]
    normalised = standardizer.transform(raw_features).astype(np.float32)
    features = torch.from_numpy(normalised).to(device)
    threshold, reject_logit = model(features)
    probability = float(torch.sigmoid(reject_logit)[0].cpu())
    cutoff = (
        float(checkpoint_metadata.get("reject_probability", 0.5))
        if reject_probability is None
        else float(reject_probability)
    )
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("reject_probability must lie in [0, 1]")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "causal_online" if temporal_order_verified else "prefix_holdout",
        "protocol": (
            "verified_temporal_prefix" if temporal_order_verified else "manifest_order_prefix_holdout"
        ),
        "temporal_order_verified": bool(temporal_order_verified),
        "target_domain": target_domain,
        "outer_fold_id": str(checkpoint_metadata["outer_fold_id"]),
        "outer_target": outer_target,
        "detector_source_domains": list(expected_sources),
        "detector_checkpoint_sha": expected_checkpoint_sha,
        "held_out_domains": list(deployment_fold.held_out_domains),
        "protocol_scope": deployment_fold.protocol_scope,
        "score_manifest_sha256": score_manifest_sha256,
        "score_manifest_target_dataset": manifest_target,
        "score_manifest_detector_checkpoint_sha": manifest_checkpoint_sha,
        "calibration_pseudo_targets": list(calibration_targets),
        "context_image_ids": list(context_ids),
        "query_image_ids": list(query_ids),
        "causal_boundary": {
            "context_start_index": 0,
            "context_end_index_inclusive": len(context_ids) - 1,
            "query_start_index": len(context_ids),
            "query_end_index_inclusive": len(context_ids) + len(query_ids) - 1,
            "context_last_image_id": context_ids[-1],
            "query_first_image_id": query_ids[0],
        },
        "budgets": budgets.to_dict(),
        "p_min": float(checkpoint_metadata["p_min"]),
        "statistics_config": statistics_config.to_dict(),
        "source_reference": source_reference.to_dict(),
        "threshold": float(threshold[0].cpu()),
        "reject_probability": probability,
        "reject_cutoff": cutoff,
        "reject": probability >= cutoff,
        "statistics_feature_names": list(statistics.feature_names),
        "statistics_metadata": dict(statistics.metadata or {}),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--score-manifest", dest="manifest", required=True)
    parser.add_argument("--calibrator-checkpoint", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--query-size", type=int)
    parser.add_argument("--pixel-budget", type=float)
    parser.add_argument("--component-budget", type=float)
    parser.add_argument("--source-reference")
    parser.add_argument("--image-dir")
    parser.add_argument("--reject-probability", type=float)
    parser.add_argument(
        "--temporal-order-verified",
        action="store_true",
        help="Assert manifest order is a verified temporal stream; otherwise report prefix_holdout",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", required=True)
    return parser


def _select_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    budgets = BudgetSpec.from_optional(args.pixel_budget, args.component_budget)
    records, manifest = load_ordered_score_records(args.manifest, image_dir=args.image_dir)
    context, query = causal_partition(
        records, context_size=args.context_size, query_size=args.query_size
    )
    device = _select_device(args.device)
    model, standardizer, checkpoint = load_calibrator_bundle(
        args.calibrator_checkpoint, device=device
    )
    statistics_config = StatisticsConfig.from_dict(checkpoint["statistics_config"])
    _, embedded_reference = _deployment_contract(checkpoint)
    if args.source_reference is None:
        source_reference = embedded_reference
    else:
        source_reference = load_source_reference(
            args.source_reference,
            statistics_config=statistics_config,
        )
        if source_reference != embedded_reference:
            raise ValueError("provided source reference differs from checkpoint")
    manifest_path = Path(args.manifest).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    result = adapt_context_to_query(
        model=model,
        standardizer=standardizer,
        checkpoint_metadata=checkpoint,
        context_records=context,
        query_records=query,
        budgets=budgets,
        source_reference=source_reference,
        score_manifest=manifest,
        score_manifest_sha256=manifest_sha256,
        device=device,
        target_domain=args.target_domain,
        reject_probability=args.reject_probability,
        temporal_order_verified=args.temporal_order_verified,
    )
    result["score_manifest"] = Path(args.manifest).name
    result["calibrator_checkpoint"] = Path(args.calibrator_checkpoint).name
    result["score_manifest_target"] = manifest.get("target_dataset")
    _write_json_atomic(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
