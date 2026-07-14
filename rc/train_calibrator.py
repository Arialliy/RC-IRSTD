"""Train the RC threshold calibrator with pseudo-target-isolated validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_ext.dataset_identity import sha256_file
from model.threshold_calibrator import ThresholdCalibrator, asymmetric_threshold_loss

from .meta_dataset import (
    FeatureStandardizer,
    RCMetaDataset,
    assert_pseudo_target_isolation,
    assert_verified_provenance,
    load_episodes,
    split_by_pseudo_target,
    validate_episode_collection,
)
from .domain_statistics import load_source_reference
from .schema import FoldContract, RCEpisode, VALID_THRESHOLD_TRANSFORMS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if value != "auto":
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_episode_splits(args: argparse.Namespace) -> tuple[list[RCEpisode], list[RCEpisode]]:
    if args.episodes:
        if args.train_file or args.val_file:
            raise ValueError("use either --episodes or --train-file/--val-file, not both")
        if not args.val_pseudo_target:
            raise ValueError("--episodes requires at least one --val-pseudo-target")
        return split_by_pseudo_target(
            load_episodes(args.episodes), args.val_pseudo_target
        )
    if not args.train_file or not args.val_file:
        raise ValueError("provide --episodes or both --train-file and --val-file")
    if args.val_pseudo_target:
        raise ValueError("--val-pseudo-target is only valid with --episodes")
    train = load_episodes(args.train_file)
    validation = load_episodes(args.val_file)
    assert_pseudo_target_isolation(train, validation)
    return train, validation


def _batch_loss(
    model: ThresholdCalibrator,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    under_weight: float,
    threshold_transform: str,
    reject_weight: float,
    threshold_on_reject: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features = batch["features"].to(device)
    target_threshold = batch["threshold"].to(device)
    target_reject = batch["reject"].to(device)
    prediction, reject_logit = model(features)
    sample_weight = None if threshold_on_reject else (1.0 - target_reject)
    threshold_loss_sum = asymmetric_threshold_loss(
        prediction,
        target_threshold,
        under_weight=under_weight,
        transform=threshold_transform,
        sample_weight=sample_weight,
        reduction="sum",
    )
    threshold_denominator = (
        torch.tensor(float(target_threshold.numel()), device=device)
        if sample_weight is None
        else sample_weight.sum()
    )
    threshold_loss = threshold_loss_sum / threshold_denominator.clamp_min(1.0)
    reject_loss_sum = F.binary_cross_entropy_with_logits(
        reject_logit,
        target_reject,
        reduction="sum",
    )
    reject_denominator = torch.tensor(float(target_reject.numel()), device=device)
    reject_loss = reject_loss_sum / reject_denominator.clamp_min(1.0)
    total = threshold_loss + reject_weight * reject_loss
    return total, {
        "threshold_loss": threshold_loss.detach(),
        "threshold_loss_sum": threshold_loss_sum.detach(),
        "threshold_denominator": threshold_denominator.detach(),
        "reject_loss": reject_loss.detach(),
        "reject_loss_sum": reject_loss_sum.detach(),
        "reject_denominator": reject_denominator.detach(),
        "prediction": prediction.detach(),
        "reject_probability": torch.sigmoid(reject_logit.detach()),
        "target_threshold": target_threshold.detach(),
        "target_reject": target_reject.detach(),
    }


def train_one_epoch(
    model: ThresholdCalibrator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    under_weight: float,
    threshold_transform: str,
    reject_weight: float,
    threshold_on_reject: bool,
) -> dict[str, float]:
    model.train()
    totals = {
        "threshold_loss_sum": 0.0,
        "threshold_denominator": 0.0,
        "reject_loss_sum": 0.0,
        "reject_denominator": 0.0,
    }
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, parts = _batch_loss(
            model,
            batch,
            device=device,
            under_weight=under_weight,
            threshold_transform=threshold_transform,
            reject_weight=reject_weight,
            threshold_on_reject=threshold_on_reject,
        )
        loss.backward()
        optimizer.step()
        for key in totals:
            totals[key] += float(parts[key].cpu())
    threshold_loss = totals["threshold_loss_sum"] / max(
        totals["threshold_denominator"], 1.0
    )
    reject_loss = totals["reject_loss_sum"] / max(
        totals["reject_denominator"], 1.0
    )
    return {
        "loss": threshold_loss + reject_weight * reject_loss,
        "threshold_loss": threshold_loss,
        "reject_loss": reject_loss,
    }


def _prediction_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("prediction records must be non-empty")
    target_reject = np.asarray([record["target_reject"] for record in records], dtype=bool)
    predicted_reject = np.asarray([record["predicted_reject"] for record in records], dtype=bool)
    target_threshold = np.asarray([record["target_threshold"] for record in records], dtype=float)
    prediction = np.asarray([record["prediction"] for record in records], dtype=float)
    oracle_covered = ~target_reject
    tp = int(np.sum(predicted_reject & target_reject))
    fp = int(np.sum(predicted_reject & ~target_reject))
    fn = int(np.sum(~predicted_reject & target_reject))
    metrics: dict[str, Any] = {
        "num_episodes": len(records),
        "coverage": float(np.mean(~predicted_reject)),
        "reject_rate": float(np.mean(predicted_reject)),
        "oracle_coverage": float(np.mean(oracle_covered)),
        "oracle_reject_rate": float(np.mean(target_reject)),
        "reject_accuracy": float(np.mean(predicted_reject == target_reject)),
        "reject_precision": tp / max(tp + fp, 1),
        "reject_recall": tp / max(tp + fn, 1),
        "threshold_mae_all": float(np.mean(np.abs(prediction - target_threshold))),
        "threshold_under_rate_all": float(np.mean(prediction < target_threshold)),
    }
    if oracle_covered.any():
        metrics["threshold_mae_oracle_covered"] = float(
            np.mean(np.abs(prediction[oracle_covered] - target_threshold[oracle_covered]))
        )
        metrics["threshold_under_rate_oracle_covered"] = float(
            np.mean(prediction[oracle_covered] < target_threshold[oracle_covered])
        )
    else:
        metrics["threshold_mae_oracle_covered"] = None
        metrics["threshold_under_rate_oracle_covered"] = None
    return metrics


@torch.no_grad()
def evaluate(
    model: ThresholdCalibrator,
    loader: DataLoader,
    *,
    device: torch.device,
    under_weight: float,
    threshold_transform: str,
    reject_weight: float,
    threshold_on_reject: bool,
    reject_probability: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    threshold_loss_sum = 0.0
    threshold_denominator = 0.0
    reject_loss_sum = 0.0
    reject_denominator = 0.0
    for batch in loader:
        loss, parts = _batch_loss(
            model,
            batch,
            device=device,
            under_weight=under_weight,
            threshold_transform=threshold_transform,
            reject_weight=reject_weight,
            threshold_on_reject=threshold_on_reject,
        )
        count = int(batch["features"].shape[0])
        threshold_loss_sum += float(parts["threshold_loss_sum"].cpu())
        threshold_denominator += float(parts["threshold_denominator"].cpu())
        reject_loss_sum += float(parts["reject_loss_sum"].cpu())
        reject_denominator += float(parts["reject_denominator"].cpu())
        predictions = parts["prediction"].cpu().numpy()
        probabilities = parts["reject_probability"].cpu().numpy()
        thresholds = parts["target_threshold"].cpu().numpy()
        rejects = parts["target_reject"].cpu().numpy()
        for index in range(count):
            records.append(
                {
                    "episode_id": batch["episode_id"][index],
                    "pseudo_target": batch["pseudo_target"][index],
                    "prediction": float(predictions[index]),
                    "target_threshold": float(thresholds[index]),
                    "reject_probability": float(probabilities[index]),
                    "predicted_reject": bool(probabilities[index] >= reject_probability),
                    "target_reject": bool(rejects[index] >= 0.5),
                }
            )
    threshold_loss = threshold_loss_sum / max(threshold_denominator, 1.0)
    reject_loss = reject_loss_sum / max(reject_denominator, 1.0)
    metrics = _prediction_metrics(records)
    metrics.update(
        {
            "loss": threshold_loss + reject_weight * reject_loss,
            "threshold_loss": threshold_loss,
            "reject_loss": reject_loss,
            "threshold_loss_denominator": threshold_denominator,
        }
    )
    per_target = {}
    for target in sorted({record["pseudo_target"] for record in records}):
        per_target[target] = _prediction_metrics(
            [record for record in records if record["pseudo_target"] == target]
        )
    metrics["per_pseudo_target"] = per_target
    return metrics, records


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _episode_input_provenance(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.episodes:
        path = Path(args.episodes).expanduser().resolve()
        sha256 = sha256_file(path)
        return (
            {
                "mode": "combined",
                "combined": {"file": path.name, "sha256": sha256},
            },
            sha256,
        )
    train_path = Path(args.train_file).expanduser().resolve()
    validation_path = Path(args.val_file).expanduser().resolve()
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    digest = hashlib.sha256()
    digest.update(b"train\0")
    digest.update(train_sha.encode("ascii"))
    digest.update(b"\0validation\0")
    digest.update(validation_sha.encode("ascii"))
    return (
        {
            "mode": "split",
            "train": {"file": train_path.name, "sha256": train_sha},
            "validation": {
                "file": validation_path.name,
                "sha256": validation_sha,
            },
            "aggregate_rule": "sha256('train\\0' + train_sha + '\\0validation\\0' + validation_sha)",
        },
        digest.hexdigest(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", help="Combined JSON/JSONL episodes to split by pseudo-target")
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    parser.add_argument("--val-pseudo-target", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployment-detector-checkpoint-sha", required=True)
    parser.add_argument(
        "--deployment-detector-source-domain", action="append", required=True
    )
    parser.add_argument("--deployment-source-reference", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--under-weight", type=float, default=4.0)
    parser.add_argument("--reject-weight", type=float, default=1.0)
    parser.add_argument("--threshold-on-reject", action="store_true")
    parser.add_argument("--threshold-transform", choices=VALID_THRESHOLD_TRANSFORMS)
    parser.add_argument("--reject-probability", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.reject_weight < 0.0:
        raise ValueError("reject-weight must be non-negative")
    if not 0.0 <= args.reject_probability <= 1.0:
        raise ValueError("reject-probability must lie in [0, 1]")
    seed_everything(args.seed)
    episode_collection_provenance, episode_collection_sha256 = (
        _episode_input_provenance(args)
    )
    train_episodes, validation_episodes = resolve_episode_splits(args)
    provenance_after_load, collection_sha_after_load = _episode_input_provenance(args)
    if (
        provenance_after_load != episode_collection_provenance
        or collection_sha_after_load != episode_collection_sha256
    ):
        raise RuntimeError("episode input files changed while loading the collection")
    assert_pseudo_target_isolation(train_episodes, validation_episodes)
    validate_episode_collection(train_episodes + validation_episodes)
    assert_verified_provenance(train_episodes + validation_episodes)
    schema_transform = train_episodes[0].threshold_transform
    if validation_episodes[0].threshold_transform != schema_transform:
        raise ValueError("train and validation threshold transforms differ")
    if args.threshold_transform is not None and args.threshold_transform != schema_transform:
        raise ValueError(
            "--threshold-transform must match the episode schema: "
            f"{args.threshold_transform!r} != {schema_transform!r}"
        )
    threshold_transform = schema_transform
    statistics_config = train_episodes[0].statistics_config
    outer_fold_id = train_episodes[0].outer_fold_id
    outer_target = train_episodes[0].outer_target
    deployment_source_reference = load_source_reference(
        args.deployment_source_reference,
        statistics_config=statistics_config,
    )
    reference_contract = deployment_source_reference.contract
    if reference_contract.outer_fold_id is None or reference_contract.outer_target is None:
        raise ValueError("deployment source reference must record an outer fold/target")
    if reference_contract.protocol_scope is None:
        raise ValueError("deployment source reference must record protocol_scope")
    deployment_fold = FoldContract(
        outer_fold_id=outer_fold_id,
        outer_target=outer_target,
        detector_source_domains=tuple(args.deployment_detector_source_domain),
        detector_checkpoint_sha=args.deployment_detector_checkpoint_sha,
        held_out_domains=reference_contract.held_out_domains,
        protocol_scope=reference_contract.protocol_scope,
    )
    deployment_fold.assert_matches_source_reference(deployment_source_reference)
    if deployment_fold.protocol_scope != "multi_source_protocol_candidate":
        raise ValueError("deployment source reference is not a main-protocol detector")

    # This is deliberately fit before the validation dataset is materialised.
    standardizer = FeatureStandardizer.fit_train(train_episodes)
    train_dataset = RCMetaDataset(train_episodes, standardizer=standardizer)
    validation_dataset = RCMetaDataset(validation_episodes, standardizer=standardizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = select_device(args.device)
    model = ThresholdCalibrator(
        train_dataset.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = math.inf
    best_epoch = -1
    checkpoint_path = output_dir / "calibrator.pt"
    common_checkpoint = {
        "format_version": "rc-irstd.calibrator.v3",
        "input_dim": train_dataset.input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "statistics_feature_names": list(train_episodes[0].feature_names),
        "input_feature_names": list(train_episodes[0].input_feature_names),
        "threshold_transform": threshold_transform,
        "statistics_config": statistics_config.to_dict(),
        "p_min": train_episodes[0].p_min,
        "outer_fold_id": outer_fold_id,
        "outer_target": outer_target,
        "episode_collection_provenance": episode_collection_provenance,
        "episode_collection_sha256": episode_collection_sha256,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "under_weight": args.under_weight,
            "reject_weight": args.reject_weight,
            "threshold_on_reject": bool(args.threshold_on_reject),
            "threshold_transform": threshold_transform,
            "reject_probability": args.reject_probability,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device_requested": args.device,
            "device_resolved": str(device),
        },
        "standardizer": standardizer.to_dict(),
        "train_pseudo_targets": sorted(train_dataset.pseudo_targets),
        "validation_pseudo_targets": sorted(validation_dataset.pseudo_targets),
        "calibration_pseudo_targets": sorted(
            train_dataset.pseudo_targets.union(validation_dataset.pseudo_targets)
        ),
        "deployment_detector_source_domains": list(
            deployment_fold.detector_source_domains
        ),
        "deployment_detector_checkpoint_sha": deployment_fold.detector_checkpoint_sha,
        "deployment_held_out_domains": list(deployment_fold.held_out_domains),
        "deployment_protocol_scope": deployment_fold.protocol_scope,
        "deployment_source_reference": deployment_source_reference.to_dict(),
        "episode_detector_contracts": [
            {
                **episode.fold.to_dict(),
                "source_reference_sha256": episode.source_reference.sha256,
            }
            for episode in sorted(
                train_episodes + validation_episodes, key=lambda value: value.episode_id
            )
        ],
        "reject_probability": args.reject_probability,
    }
    best_records: list[dict[str, Any]] = []
    best_validation_metrics: dict[str, Any] = {}
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            under_weight=args.under_weight,
            threshold_transform=threshold_transform,
            reject_weight=args.reject_weight,
            threshold_on_reject=args.threshold_on_reject,
        )
        validation_metrics, validation_records = evaluate(
            model,
            validation_loader,
            device=device,
            under_weight=args.under_weight,
            threshold_transform=threshold_transform,
            reject_weight=args.reject_weight,
            threshold_on_reject=args.threshold_on_reject,
            reject_probability=args.reject_probability,
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        if float(validation_metrics["loss"]) < best_loss:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            best_records = validation_records
            best_validation_metrics = validation_metrics
            _save_checkpoint_atomic(
                checkpoint_path,
                {
                    **common_checkpoint,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
            )
    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation": best_validation_metrics,
        "num_train_episodes": len(train_dataset),
        "num_validation_episodes": len(validation_dataset),
        "train_pseudo_targets": sorted(train_dataset.pseudo_targets),
        "validation_pseudo_targets": sorted(validation_dataset.pseudo_targets),
        "standardizer_fit": "train_only",
        "threshold_transform": threshold_transform,
        "statistics_config": statistics_config.to_dict(),
        "p_min": train_episodes[0].p_min,
        "outer_fold_id": outer_fold_id,
        "outer_target": outer_target,
        "episode_collection_sha256": episode_collection_sha256,
        "deployment_detector_checkpoint_sha": deployment_fold.detector_checkpoint_sha,
        "deployment_detector_source_domains": list(deployment_fold.detector_source_domains),
        "deployment_held_out_domains": list(deployment_fold.held_out_domains),
        "deployment_protocol_scope": deployment_fold.protocol_scope,
        "deployment_source_reference_sha256": deployment_source_reference.sha256,
        "budget_metrics_status": "not_computed_requires_query_curve_replay",
        "checkpoint": checkpoint_path.name,
    }
    _write_json_atomic(output_dir / "history.json", history)
    _write_json_atomic(output_dir / "validation_predictions.json", best_records)
    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
