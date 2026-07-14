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
import torch.nn as nn
from torch.utils.data import DataLoader

from data_ext.dataset_identity import sha256_file
from model.monotone_pixel_calibrator import MonotonePixelRiskCalibrator
from model.threshold_calibrator import ThresholdCalibrator, asymmetric_threshold_loss

from .meta_dataset import (
    FeatureStandardizer,
    RCMetaDataset,
    RCPixelRiskMetaDataset,
    assert_pseudo_target_isolation,
    assert_verified_provenance,
    load_episodes,
    split_by_pseudo_target,
    validate_episode_collection,
)
from .domain_statistics import load_source_reference
from .schema import (
    DeploymentProtocolContract,
    FoldContract,
    OFFICIAL_TRAIN_SPLIT_ROLE,
    RCEpisode,
    SCHEMA_VERSION,
    VALID_THRESHOLD_TRANSFORMS,
    canonicalize_episode_score_split_contract,
)


DIRECT_CALIBRATOR = "direct"
MONOTONE_PIXEL_CALIBRATOR = "monotone_pixel"
CALIBRATOR_MODELS = (DIRECT_CALIBRATOR, MONOTONE_PIXEL_CALIBRATOR)


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


def audit_official_train_score_provenance(
    train_episodes: Sequence[RCEpisode],
    validation_episodes: Sequence[RCEpisode],
) -> dict[str, Any]:
    """Fail closed before either fitting or pseudo-target model selection.

    Pseudo-target validation is still calibration: its oracle thresholds may
    select the best checkpoint.  Consequently both the optimiser split and
    the pseudo-target validation split must have been exported exclusively
    from each pseudo-target's official training split.  Legacy v3 episodes
    remain deserialisable for diagnostics, but are ineligible here.
    """

    partitions = (
        ("calibrator_train", train_episodes),
        ("pseudo_target_validation", validation_episodes),
    )
    contracts_by_target: dict[str, dict[str, Any]] = {}
    counts_by_target: dict[str, int] = {}
    partition_by_target: dict[str, str] = {}
    total = 0
    for partition, episodes in partitions:
        if not episodes:
            raise ValueError(f"{partition} episode split must be non-empty")
        for episode in episodes:
            total += 1
            if episode.schema_version != SCHEMA_VERSION:
                raise ValueError(
                    "claim-bearing calibrator training requires meta-episode v4 "
                    "official-train provenance; legacy/missing split contract in "
                    f"episode {episode.episode_id!r}"
                )
            raw_contract = episode.provenance.split_contract
            if raw_contract is None:
                raise ValueError(
                    "claim-bearing calibrator training requires an official-train "
                    f"split contract in episode {episode.episode_id!r}"
                )
            contract = canonicalize_episode_score_split_contract(raw_contract)
            if contract["role"] != OFFICIAL_TRAIN_SPLIT_ROLE:
                raise ValueError(
                    "calibrator train/pseudo-target validation episodes must use "
                    "role='official_train'; "
                    f"episode {episode.episode_id!r} declares {contract['role']!r}"
                )
            if contract["selected_split_sha256"] != contract[
                "official_train_split_sha256"
            ]:
                raise ValueError(
                    "official-train episode selected_split_sha256 mismatch: "
                    f"{episode.episode_id!r}"
                )
            if (
                contract["train_test_id_overlap_count"] != 0
                or contract["train_test_image_content_overlap_count"] != 0
                or contract["disjointness_verified"] is not True
            ):
                raise ValueError(
                    "official-train episode does not prove train/test disjointness: "
                    f"{episode.episode_id!r}"
                )

            target = episode.pseudo_target
            prior_partition = partition_by_target.setdefault(target, partition)
            if prior_partition != partition:
                raise ValueError(
                    "one pseudo-target occurs in both calibrator train and validation: "
                    f"{target!r}"
                )
            prior_contract = contracts_by_target.setdefault(target, contract)
            if prior_contract != contract:
                raise ValueError(
                    "episodes for one pseudo-target disagree on their official-train "
                    f"split contract: {target!r}"
                )
            counts_by_target[target] = counts_by_target.get(target, 0) + 1

    target_audit: dict[str, Any] = {}
    for target in sorted(contracts_by_target):
        contract = contracts_by_target[target]
        contract_json = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        target_audit[target] = {
            "partition": partition_by_target[target],
            "num_episodes": counts_by_target[target],
            "split_contract_sha256": hashlib.sha256(contract_json).hexdigest(),
            "split_contract": contract,
        }
    return {
        "schema_version": "rc-irstd.calibrator-official-train-provenance.v1",
        "required_episode_schema": SCHEMA_VERSION,
        "required_score_split_role": OFFICIAL_TRAIN_SPLIT_ROLE,
        "pseudo_target_validation_may_select_best_checkpoint": True,
        "official_test_scores_consumed": False,
        "num_episodes": total,
        "pseudo_targets": target_audit,
    }


def _batch_loss(
    model: nn.Module,
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
    if isinstance(model, MonotonePixelRiskCalibrator):
        pixel_budgets = batch["pixel_budget"].to(device).reshape(-1, 1)
        output = model(features, pixel_budgets=pixel_budgets)
        if (
            output.requested_thresholds is None
            or output.requested_reject_logits is None
        ):
            raise RuntimeError("monotone calibrator did not return requested outputs")
        prediction = output.requested_thresholds[:, 0]
        reject_logit = output.requested_reject_logits[:, 0]
    elif isinstance(model, ThresholdCalibrator):
        prediction, reject_logit = model(features)
    else:
        raise TypeError(f"unsupported calibrator model: {type(model).__name__}")
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
    model: nn.Module,
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
    model: nn.Module,
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


def _freeze_deployment_protocol(
    episodes: Sequence[RCEpisode],
    *,
    reject_cutoff: float,
    matching_rule: str,
    centroid_distance: float,
) -> DeploymentProtocolContract:
    """Derive one immutable target-time protocol from pseudo-target episodes."""

    size_pairs = {
        (len(episode.context_image_ids), len(episode.query_image_ids))
        for episode in episodes
    }
    if len(size_pairs) != 1:
        raise ValueError(
            "claim-bearing calibration requires one pre-specified context/query "
            f"size pair across all episodes, got {sorted(size_pairs)}"
        )
    context_size, query_size = next(iter(size_pairs))
    return DeploymentProtocolContract(
        context_size=context_size,
        query_size=query_size,
        reject_cutoff=reject_cutoff,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
    )


def _validate_monotone_pixel_grid(
    values: Sequence[float] | None,
    episodes: Sequence[RCEpisode],
) -> tuple[float, ...]:
    if values is None:
        raise ValueError(
            "--calibrator-model monotone_pixel requires --pixel-budget-grid"
        )
    grid = tuple(float(value) for value in values)
    # Constructor validation is the canonical ordering/range check.
    probe = MonotonePixelRiskCalibrator(
        context_feature_dim=len(episodes[0].feature_names),
        pixel_budget_grid=grid,
        hidden_dims=(),
        dropout=0.0,
    )
    lower = float(probe.pixel_budget_grid[-1])
    upper = float(probe.pixel_budget_grid[0])
    unsupported = [
        episode.episode_id
        for episode in episodes
        if episode.budgets.active != (True, False)
    ]
    if unsupported:
        raise ValueError(
            "monotone_pixel supports only active pixel budgets; "
            f"unsupported episodes={unsupported}"
        )
    outside = [
        episode.episode_id
        for episode in episodes
        if not lower <= float(episode.budgets.values[0]) <= upper
    ]
    if outside:
        raise ValueError(
            "episode pixel budgets fall outside the frozen monotone grid; "
            f"episodes={outside}"
        )
    return grid


def _monotone_budget_contract(
    grid: Sequence[float],
    *,
    train_episodes: Sequence[RCEpisode],
    validation_episodes: Sequence[RCEpisode],
) -> dict[str, Any]:
    """Audit supervision coverage and bind the frozen interpolation policy."""

    frozen_grid = tuple(float(value) for value in grid)

    def grid_index(value: float) -> int | None:
        for index, candidate in enumerate(frozen_grid):
            if math.isclose(value, candidate, rel_tol=1e-12, abs_tol=0.0):
                return index
        return None

    def audit_split(name: str, episodes: Sequence[RCEpisode]) -> dict[str, Any]:
        counts = [0] * len(frozen_grid)
        grouped: dict[
            tuple[str, tuple[str, ...], tuple[str, ...], str],
            list[RCEpisode],
        ] = {}
        for episode in episodes:
            value = float(episode.budgets.values[0])
            index = grid_index(value)
            if index is not None:
                counts[index] += 1
            key = (
                episode.pseudo_target,
                episode.context_image_ids,
                episode.query_image_ids,
                episode.provenance.curve_file_sha256,
            )
            grouped.setdefault(key, []).append(episode)
        missing = [
            frozen_grid[index] for index, count in enumerate(counts) if count == 0
        ]
        if missing:
            raise ValueError(
                f"{name} episodes do not supervise every frozen pixel-budget "
                f"grid point; missing={missing}"
            )
        checked_multi_budget_groups = 0
        for key, rows in grouped.items():
            by_budget: dict[float, RCEpisode] = {}
            for episode in rows:
                budget = float(episode.budgets.values[0])
                previous = by_budget.get(budget)
                if previous is not None:
                    observed = (
                        episode.oracle_threshold,
                        episode.oracle_pd,
                        episode.reject,
                    )
                    expected = (
                        previous.oracle_threshold,
                        previous.oracle_pd,
                        previous.reject,
                    )
                    if observed != expected:
                        raise ValueError(
                            "conflicting oracle supervision for the same "
                            f"context/query/budget group: key={key}, budget={budget}"
                        )
                by_budget[budget] = episode
            missing_group_grid = [
                value
                for value in frozen_grid
                if not any(
                    math.isclose(value, observed, rel_tol=1e-12, abs_tol=0.0)
                    for observed in by_budget
                )
            ]
            if missing_group_grid:
                raise ValueError(
                    "each context/query curve group must supervise the complete "
                    f"frozen pixel-budget grid; group={key}, "
                    f"missing={missing_group_grid}"
                )
            ordered = [by_budget[value] for value in sorted(by_budget, reverse=True)]
            checked_multi_budget_groups += 1
            thresholds = [float(row.oracle_threshold) for row in ordered]
            rejects = [bool(row.reject) for row in ordered]
            if any(
                strict + 1e-12 < loose
                for loose, strict in zip(thresholds, thresholds[1:])
            ):
                raise ValueError(
                    "pixel-only oracle thresholds decrease as the budget tightens; "
                    f"group={key}, thresholds={thresholds}"
                )
            if any(loose and not strict for loose, strict in zip(rejects, rejects[1:])):
                raise ValueError(
                    "pixel-only reject labels decrease as the budget tightens; "
                    f"group={key}, reject={rejects}"
                )
        return {
            "name": name,
            "num_episodes": len(episodes),
            "grid_counts": counts,
            "all_grid_points_supervised": True,
            "num_context_query_groups": len(grouped),
            "num_multi_budget_groups_checked": checked_multi_budget_groups,
            "duplicate_oracle_conflicts": 0,
        }

    canonical = json.dumps(
        {
            "risk": "fa_pixel",
            "grid": list(frozen_grid),
            "grid_order": "loose_to_strict",
            "interpolation": "piecewise_linear_log10",
            "extrapolation_allowed": False,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema_version": "rc-irstd.monotone-pixel-budget.v1",
        "risk": "fa_pixel",
        "component_budget_supported": False,
        "grid": list(frozen_grid),
        "grid_order": "loose_to_strict",
        "grid_policy_sha256": hashlib.sha256(canonical).hexdigest(),
        "interpolation": "piecewise_linear_log10",
        "extrapolation_allowed": False,
        "curve_compute_dtype": "float64",
        "train_supervision": audit_split("train", train_episodes),
        "validation_supervision": audit_split(
            "validation", validation_episodes
        ),
    }


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
    parser.add_argument(
        "--calibrator-model",
        choices=CALIBRATOR_MODELS,
        default=DIRECT_CALIBRATOR,
        help=(
            "direct is the scalar MLP baseline; monotone_pixel is "
            "the pixel-risk inverse curve and intentionally rejects component budgets."
        ),
    )
    parser.add_argument(
        "--pixel-budget-grid",
        nargs="+",
        type=float,
        help=(
            "Loose-to-strict positive grid required by monotone_pixel, for "
            "example: 1e-4 1e-5 1e-6. Extrapolation is prohibited."
        ),
    )
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
    parser.add_argument(
        "--evaluation-matching-rule",
        choices=("overlap", "centroid"),
        default="overlap",
    )
    parser.add_argument(
        "--evaluation-centroid-distance",
        type=float,
        default=3.0,
    )
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
    official_train_score_provenance = audit_official_train_score_provenance(
        train_episodes,
        validation_episodes,
    )
    monotone_pixel_grid: tuple[float, ...] | None = None
    monotone_budget_contract: dict[str, Any] | None = None
    if args.calibrator_model == MONOTONE_PIXEL_CALIBRATOR:
        monotone_pixel_grid = _validate_monotone_pixel_grid(
            args.pixel_budget_grid,
            train_episodes + validation_episodes,
        )
        monotone_budget_contract = _monotone_budget_contract(
            monotone_pixel_grid,
            train_episodes=train_episodes,
            validation_episodes=validation_episodes,
        )
    elif args.pixel_budget_grid is not None:
        raise ValueError(
            "--pixel-budget-grid is only valid with --calibrator-model monotone_pixel"
        )
    deployment_protocol_contract = _freeze_deployment_protocol(
        train_episodes + validation_episodes,
        reject_cutoff=args.reject_probability,
        matching_rule=args.evaluation_matching_rule,
        centroid_distance=args.evaluation_centroid_distance,
    )
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
    # The monotone model keeps budgets outside the context encoder so its
    # ordering guarantee is architectural rather than learned from a generic
    # budget feature.
    if args.calibrator_model == MONOTONE_PIXEL_CALIBRATOR:
        standardizer = FeatureStandardizer.fit_context_train(train_episodes)
        train_dataset = RCPixelRiskMetaDataset(
            train_episodes, standardizer=standardizer
        )
        validation_dataset = RCPixelRiskMetaDataset(
            validation_episodes, standardizer=standardizer
        )
    else:
        standardizer = FeatureStandardizer.fit_train(train_episodes)
        train_dataset = RCMetaDataset(train_episodes, standardizer=standardizer)
        validation_dataset = RCMetaDataset(
            validation_episodes, standardizer=standardizer
        )
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
    if args.calibrator_model == MONOTONE_PIXEL_CALIBRATOR:
        assert monotone_pixel_grid is not None
        model: nn.Module = MonotonePixelRiskCalibrator(
            context_feature_dim=train_dataset.input_dim,
            pixel_budget_grid=monotone_pixel_grid,
            hidden_dims=(args.hidden_dim, args.hidden_dim),
            dropout=args.dropout,
        ).to(device)
        model_config: dict[str, Any] = model.export_config()
        capability_contract: dict[str, Any] = model.capability_contract()
        format_version = "rc-irstd.calibrator.v4"
        input_feature_names = list(train_episodes[0].feature_names)
    else:
        model = ThresholdCalibrator(
            train_dataset.input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        model_config = {
            "input_dim": train_dataset.input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
        capability_contract = {
            "budget_scope": "pixel_or_component_or_dual_empirical_direct",
            "supports_component_budget": True,
            "supports_reject": True,
            "training_pipeline_integrated": True,
            "budget_monotonicity_guaranteed": False,
            "risk_aligned_query_loss": False,
            "training_objective": "asymmetric_oracle_threshold_plus_reject_bce",
            "risk_guarantee": "empirical_not_certified",
        }
        format_version = "rc-irstd.calibrator.v3"
        input_feature_names = list(train_episodes[0].input_feature_names)
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
        "format_version": format_version,
        "calibrator_model": args.calibrator_model,
        "model_config": model_config,
        "capability_contract": capability_contract,
        "monotone_budget_contract": monotone_budget_contract,
        "input_dim": train_dataset.input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "statistics_feature_names": list(train_episodes[0].feature_names),
        "input_feature_names": input_feature_names,
        "threshold_transform": threshold_transform,
        "statistics_config": statistics_config.to_dict(),
        "p_min": train_episodes[0].p_min,
        "outer_fold_id": outer_fold_id,
        "outer_target": outer_target,
        "episode_collection_provenance": episode_collection_provenance,
        "episode_collection_sha256": episode_collection_sha256,
        "official_train_score_provenance": official_train_score_provenance,
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
            "evaluation_matching_rule": args.evaluation_matching_rule,
            "evaluation_centroid_distance": args.evaluation_centroid_distance,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device_requested": args.device,
            "device_resolved": str(device),
            "calibrator_model": args.calibrator_model,
            "pixel_budget_grid": (
                None
                if monotone_pixel_grid is None
                else list(monotone_pixel_grid)
            ),
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
        "deployment_protocol_contract": deployment_protocol_contract.to_dict(),
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
        "calibrator_model": args.calibrator_model,
        "calibrator_format_version": format_version,
        "capability_contract": capability_contract,
        "monotone_budget_contract": monotone_budget_contract,
        "threshold_transform": threshold_transform,
        "statistics_config": statistics_config.to_dict(),
        "p_min": train_episodes[0].p_min,
        "outer_fold_id": outer_fold_id,
        "outer_target": outer_target,
        "episode_collection_sha256": episode_collection_sha256,
        "official_train_score_provenance": official_train_score_provenance,
        "deployment_detector_checkpoint_sha": deployment_fold.detector_checkpoint_sha,
        "deployment_detector_source_domains": list(deployment_fold.detector_source_domains),
        "deployment_held_out_domains": list(deployment_fold.held_out_domains),
        "deployment_protocol_scope": deployment_fold.protocol_scope,
        "deployment_source_reference_sha256": deployment_source_reference.sha256,
        "deployment_protocol_contract": deployment_protocol_contract.to_dict(),
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
