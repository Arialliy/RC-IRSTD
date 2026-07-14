"""Train the final grouped, query-risk-aligned, no-Reject calibrator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.calibrator_replay import ExactGroupedPixelRiskReplay
from losses.calibrator_risk import (
    calibrator_risk_capability_contract,
    curve_query_risk_aligned_calibrator_loss,
)
from model.monotone_pixel_calibrator import MonotoneNoRejectPixelRiskCalibrator

from .domain_statistics import load_source_reference
from .meta_dataset import (
    FeatureStandardizer,
    RCGroupedPixelRiskMetaDataset,
    assert_pseudo_target_isolation,
    assert_verified_provenance,
    collate_grouped_pixel_risk_batch,
    validate_episode_collection,
)
from .online_adapter import (
    NO_REJECT_CALIBRATOR_FORMAT,
    NO_REJECT_CALIBRATOR_MODEL,
)
from .schema import FoldContract, NoRejectDeploymentProtocolContract
from .train_calibrator import (
    _episode_input_provenance,
    _monotone_budget_contract,
    _save_checkpoint_atomic,
    _validate_monotone_pixel_grid,
    _write_json_atomic,
    audit_official_train_score_provenance,
    resolve_episode_splits,
    seed_everything,
    select_device,
)


def _freeze_no_reject_protocol(
    episodes: Sequence[Any],
    *,
    matching_rule: str,
    centroid_distance: float,
) -> NoRejectDeploymentProtocolContract:
    sizes = {
        (len(episode.context_image_ids), len(episode.query_image_ids))
        for episode in episodes
    }
    if len(sizes) != 1:
        raise ValueError(
            "claim-bearing no-reject calibration requires one frozen "
            f"context/query size pair, got {sorted(sizes)}"
        )
    context_size, query_size = next(iter(sizes))
    return NoRejectDeploymentProtocolContract(
        context_size=context_size,
        query_size=query_size,
        matching_rule=matching_rule,
        centroid_distance=centroid_distance,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }


def _loss_for_batch(
    model: MonotoneNoRejectPixelRiskCalibrator,
    batch: Mapping[str, Any],
    args: argparse.Namespace,
):
    output = model(batch["features"])
    if output.grid_logits.shape != batch["oracle_logits"].shape:
        raise RuntimeError("model/grid supervision shape mismatch")
    loss = curve_query_risk_aligned_calibrator_loss(
        output.grid_logits,
        batch["pixel_budgets"],
        batch["oracle_logits"],
        batch["curve_logits"],
        batch["curve_pixel_risk"],
        batch["curve_pd"],
        batch["curve_valid_mask"],
        batch["curve_exact_lower_logit"],
        batch["curve_global_exact"],
        utility_episode_valid=batch["curve_gt_objects"] > 0,
        lambda_violation=args.lambda_violation,
        lambda_utility=args.lambda_utility,
        lambda_oracle_logit=args.lambda_oracle,
        lambda_curve_smoothness=args.lambda_smoothness,
        lambda_coverage=args.lambda_coverage,
        epsilon=args.risk_epsilon,
        oracle_huber_delta=args.oracle_huber_delta,
    )
    return output, loss


_LOSS_FIELDS = (
    "total",
    "violation",
    "utility",
    "oracle_logit",
    "curve_smoothness",
    "coverage_penalty",
)


def _train_epoch(
    model: MonotoneNoRejectPixelRiskCalibrator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in _LOSS_FIELDS}
    rows = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = _loss_for_batch(model, batch, args)
        loss.total.backward()
        if args.grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        size = int(batch["features"].shape[0])
        rows += size
        for name in totals:
            totals[name] += float(getattr(loss, name).detach().cpu()) * size
    if rows == 0:
        raise RuntimeError("training loader produced no groups")
    return {name: value / rows for name, value in totals.items()}


@torch.no_grad()
def _surrogate_validation(
    model: MonotoneNoRejectPixelRiskCalibrator,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    totals = {name: 0.0 for name in _LOSS_FIELDS}
    rows = 0
    thresholds: list[np.ndarray] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        output, loss = _loss_for_batch(model, batch, args)
        size = int(batch["features"].shape[0])
        rows += size
        for name in totals:
            totals[name] += float(getattr(loss, name).detach().cpu()) * size
        thresholds.append(output.grid_thresholds.detach().cpu().numpy())
    if rows == 0:
        raise RuntimeError("validation loader produced no groups")
    return (
        {name: value / rows for name, value in totals.items()},
        np.concatenate(thresholds, axis=0),
    )


def _provenance_rows(dataset: RCGroupedPixelRiskMetaDataset) -> list[dict[str, Any]]:
    if dataset.query_curves is None:
        raise ValueError("group provenance requires verified query curves")
    return [
        {
            "group_id": group.group_id,
            "pseudo_target": group.pseudo_target,
            "context_image_ids": list(group.context_image_ids),
            "query_image_ids": list(group.query_image_ids),
            "curve_file_sha256": group.representative.provenance.curve_file_sha256,
            "curve_manifest_sha256": (
                group.representative.provenance.curve_manifest_sha256
            ),
            "query_score_manifest_sha256": (
                group.representative.provenance.query_score_manifest_sha256
            ),
            "label_manifest_sha256": (
                group.representative.provenance.label_manifest_sha256
            ),
            "label_manifest_content_sha256": (
                group.representative.provenance.label_manifest_content_sha256
            ),
            "matching_rule": curve.matching_rule,
            "centroid_distance": curve.centroid_distance,
        }
        for group, curve in zip(dataset.groups, dataset.query_curves)
    ]


def _assert_query_matching_contract(
    dataset: RCGroupedPixelRiskMetaDataset,
    protocol: NoRejectDeploymentProtocolContract,
) -> None:
    if dataset.query_curves is None:
        raise ValueError("risk-aligned training requires verified query curves")
    mismatches = [
        dataset.groups[index].group_id
        for index, curve in enumerate(dataset.query_curves)
        if curve.matching_rule != protocol.matching_rule
        or not math.isclose(
            curve.centroid_distance,
            protocol.centroid_distance,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if mismatches:
        raise ValueError(
            "query-curve matching contract differs from the frozen deployment/"
            f"hard-replay contract; groups={mismatches}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes")
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    parser.add_argument("--val-pseudo-target", action="append", default=[])
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deployment-detector-checkpoint-sha", required=True)
    parser.add_argument(
        "--deployment-detector-source-domain", action="append", required=True
    )
    parser.add_argument("--deployment-source-reference", required=True)
    parser.add_argument(
        "--pixel-budget-grid", type=float, nargs="+", required=True
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--min-logit", type=float, default=-10.0)
    parser.add_argument("--max-logit", type=float, default=18.0)
    parser.add_argument("--minimum-logit-gap", type=float, default=1e-3)
    parser.add_argument("--lambda-violation", type=float, default=4.0)
    parser.add_argument("--lambda-utility", type=float, default=1.0)
    parser.add_argument("--lambda-oracle", type=float, default=0.10)
    parser.add_argument("--lambda-smoothness", type=float, default=0.01)
    parser.add_argument("--lambda-coverage", type=float, default=4.0)
    parser.add_argument("--oracle-huber-delta", type=float, default=1.0)
    parser.add_argument("--oracle-logit-eps", type=float, default=1e-12)
    parser.add_argument("--risk-epsilon", type=float, default=1e-12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--evaluation-matching-rule",
        choices=("overlap", "centroid"),
        default="overlap",
    )
    parser.add_argument("--evaluation-centroid-distance", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.hidden_dim <= 0 or not 0.0 <= args.dropout < 1.0:
        raise ValueError("hidden-dim must be positive and dropout must lie in [0, 1)")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("lr must be positive and weight-decay non-negative")
    if args.grad_clip < 0.0 or args.patience < 0:
        raise ValueError("grad-clip and patience must be non-negative")
    for name in (
        "lambda_violation",
        "lambda_utility",
        "lambda_oracle",
        "lambda_smoothness",
        "lambda_coverage",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    seed_everything(args.seed)
    input_provenance, input_sha256 = _episode_input_provenance(args)
    train_episodes, validation_episodes = resolve_episode_splits(args)
    provenance_after, sha_after = _episode_input_provenance(args)
    if provenance_after != input_provenance or sha_after != input_sha256:
        raise RuntimeError("episode input files changed while loading")
    assert_pseudo_target_isolation(train_episodes, validation_episodes)
    all_episodes = train_episodes + validation_episodes
    validate_episode_collection(all_episodes)
    assert_verified_provenance(all_episodes)
    official_train_audit = audit_official_train_score_provenance(
        train_episodes, validation_episodes
    )
    pixel_grid = _validate_monotone_pixel_grid(args.pixel_budget_grid, all_episodes)
    budget_contract = _monotone_budget_contract(
        pixel_grid,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
    )
    budget_contract.update(
        {
            "method_supports_reject": False,
            "grouped_complete_curve_supervision": True,
            "query_supervision": "verified_event_exact_or_global_exact",
            "checkpoint_selection": "exact_native_replay_BSR_LogExcess_Pd",
        }
    )
    protocol = _freeze_no_reject_protocol(
        all_episodes,
        matching_rule=args.evaluation_matching_rule,
        centroid_distance=args.evaluation_centroid_distance,
    )

    statistics_config = train_episodes[0].statistics_config
    outer_fold_id = train_episodes[0].outer_fold_id
    outer_target = train_episodes[0].outer_target
    source_reference = load_source_reference(
        args.deployment_source_reference,
        statistics_config=statistics_config,
    )
    source_contract = source_reference.contract
    if (
        source_contract.outer_fold_id is None
        or source_contract.outer_target is None
        or source_contract.protocol_scope is None
    ):
        raise ValueError("deployment source reference lacks outer-fold provenance")
    deployment_fold = FoldContract(
        outer_fold_id=outer_fold_id,
        outer_target=outer_target,
        detector_source_domains=tuple(args.deployment_detector_source_domain),
        detector_checkpoint_sha=args.deployment_detector_checkpoint_sha,
        held_out_domains=source_contract.held_out_domains,
        protocol_scope=source_contract.protocol_scope,
    )
    deployment_fold.assert_matches_source_reference(source_reference)
    if deployment_fold.protocol_scope != "multi_source_protocol_candidate":
        raise ValueError("deployment source reference is not main-protocol eligible")

    # Fit normalisation before materialising any validation query supervision.
    standardizer = FeatureStandardizer.fit_context_train(train_episodes)
    train_dataset = RCGroupedPixelRiskMetaDataset(
        train_episodes,
        pixel_budget_grid=pixel_grid,
        standardizer=standardizer,
        query_curve_mode="verified_event_exact",
        artifact_root=args.artifact_root,
        oracle_logit_eps=args.oracle_logit_eps,
    )
    validation_dataset = RCGroupedPixelRiskMetaDataset(
        validation_episodes,
        pixel_budget_grid=pixel_grid,
        standardizer=standardizer,
        query_curve_mode="verified_event_exact",
        artifact_root=args.artifact_root,
        oracle_logit_eps=args.oracle_logit_eps,
    )
    _assert_query_matching_contract(train_dataset, protocol)
    _assert_query_matching_contract(validation_dataset, protocol)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        collate_fn=collate_grouped_pixel_risk_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_grouped_pixel_risk_batch,
    )
    hard_replay = ExactGroupedPixelRiskReplay(
        validation_dataset.groups,
        artifact_root=args.artifact_root,
        matching_rule=protocol.matching_rule,
        centroid_distance=protocol.centroid_distance,
    )
    device = select_device(args.device)
    model = MonotoneNoRejectPixelRiskCalibrator(
        context_feature_dim=train_dataset.input_dim,
        pixel_budget_grid=pixel_grid,
        hidden_dims=(args.hidden_dim, args.hidden_dim),
        dropout=args.dropout,
        min_logit=args.min_logit,
        max_logit=args.max_logit,
        minimum_logit_gap=args.minimum_logit_gap,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 4) if args.patience else 5,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "calibrator.pt"
    last_checkpoint_path = output_dir / "calibrator_last.pt"
    capability = model.capability_contract()
    if capability.get("supports_reject") is not False:
        raise RuntimeError("final calibrator capability unexpectedly supports rejection")
    common_checkpoint: dict[str, Any] = {
        "format_version": NO_REJECT_CALIBRATOR_FORMAT,
        "calibrator_model": NO_REJECT_CALIBRATOR_MODEL,
        "model_config": model.export_config(),
        "capability_contract": capability,
        "risk_loss_contract": calibrator_risk_capability_contract(),
        "monotone_budget_contract": budget_contract,
        "input_dim": train_dataset.input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "statistics_feature_names": list(train_episodes[0].feature_names),
        "input_feature_names": list(train_episodes[0].feature_names),
        "threshold_transform": train_episodes[0].threshold_transform,
        "statistics_config": statistics_config.to_dict(),
        "outer_fold_id": outer_fold_id,
        "outer_target": outer_target,
        "episode_collection_provenance": input_provenance,
        "episode_collection_sha256": input_sha256,
        "official_train_score_provenance": official_train_audit,
        "standardizer": standardizer.to_dict(),
        "train_pseudo_targets": sorted(train_dataset.pseudo_targets),
        "validation_pseudo_targets": sorted(validation_dataset.pseudo_targets),
        "calibration_pseudo_targets": sorted(
            train_dataset.pseudo_targets.union(validation_dataset.pseudo_targets)
        ),
        "deployment_detector_source_domains": list(
            deployment_fold.detector_source_domains
        ),
        "deployment_detector_checkpoint_sha": (
            deployment_fold.detector_checkpoint_sha
        ),
        "deployment_held_out_domains": list(deployment_fold.held_out_domains),
        "deployment_protocol_scope": deployment_fold.protocol_scope,
        "deployment_source_reference": source_reference.to_dict(),
        "deployment_protocol_contract": protocol.to_dict(),
        "checkpoint_selection_order": ["BSR", "LogExcess", "Pd"],
        "risk_guarantee": "empirical_meta_calibration_not_certified",
        "reject_head": False,
        "artifact_root_persisted": False,
        "train_group_provenance": _provenance_rows(train_dataset),
        "validation_group_provenance": _provenance_rows(validation_dataset),
        "episode_detector_contracts": [
            {
                **episode.fold.to_dict(),
                "source_reference_sha256": episode.source_reference.sha256,
            }
            for episode in sorted(all_episodes, key=lambda value: value.episode_id)
        ],
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lambda_violation": args.lambda_violation,
            "lambda_utility": args.lambda_utility,
            "lambda_oracle": args.lambda_oracle,
            "lambda_smoothness": args.lambda_smoothness,
            "lambda_coverage": args.lambda_coverage,
            "oracle_huber_delta": args.oracle_huber_delta,
            "oracle_logit_eps": args.oracle_logit_eps,
            "risk_epsilon": args.risk_epsilon,
            "grad_clip": args.grad_clip,
            "patience": args.patience,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "device_requested": args.device,
            "device_resolved": str(device),
            "pixel_budget_grid": list(pixel_grid),
            "query_curve_mode": "verified_event_exact",
            "hard_replay": "native_resolution_every_epoch",
            "threshold_semantics": "prediction = probability > threshold",
            "evaluation_matching_rule": protocol.matching_rule,
            "evaluation_centroid_distance": protocol.centroid_distance,
        },
    }

    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    best_replay: dict[str, Any] = {}
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_metrics = _train_epoch(
            model, train_loader, optimizer, device=device, args=args
        )
        validation_surrogate, predicted_thresholds = _surrogate_validation(
            model, validation_loader, device=device, args=args
        )
        replay = hard_replay.evaluate(
            predicted_thresholds,
            pixel_budget_grid=pixel_grid,
            epsilon=args.risk_epsilon,
        )
        replay_payload = replay.to_dict()
        scheduler.step(replay.log_excess)
        rank = replay.rank_key
        improved = rank > best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            best_replay = replay_payload
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation_surrogate": validation_surrogate,
            "validation_exact_replay": replay_payload,
            "checkpoint_rank": list(rank),
            "is_best": improved,
        }
        history.append(record)
        payload = {
            **common_checkpoint,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_rank": list(best_rank),
            "validation_metrics": replay_payload,
        }
        _save_checkpoint_atomic(last_checkpoint_path, payload)
        if improved:
            _save_checkpoint_atomic(checkpoint_path, payload)
            _write_json_atomic(output_dir / "best_hard_replay.json", replay_payload)
        _write_json_atomic(output_dir / "history.json", history)
        if args.patience and epochs_without_improvement >= args.patience:
            break

    if best_epoch < 0 or not checkpoint_path.is_file():
        raise RuntimeError("training completed without a valid best checkpoint")
    summary = {
        "status": "completed",
        "method": "two_stage_no_reject_monotone_inverse_pixel_risk",
        "format_version": NO_REJECT_CALIBRATOR_FORMAT,
        "best_epoch": best_epoch,
        "best_rank": list(best_rank),
        "best_validation_exact_replay": best_replay,
        "checkpoint_selection_order": ["BSR", "LogExcess", "Pd"],
        "num_train_groups": len(train_dataset),
        "num_validation_groups": len(validation_dataset),
        "pixel_budget_grid": list(pixel_grid),
        "reject_head": False,
        "standardizer_fit": "train_context_only",
        "official_test_scores_consumed": False,
        "checkpoint": checkpoint_path.name,
        "last_checkpoint": last_checkpoint_path.name,
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
