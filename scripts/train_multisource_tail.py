"""Train MSHNet with balanced multi-source tail-risk objectives.

This entry point deliberately uses a fixed-last checkpoint policy.  It never
constructs a target/test loader and never chooses a checkpoint using target or
official-test labels.  Cross-domain evaluation belongs in a separate command.

Run from the repository root:

    python -m scripts.train_multisource_tail --source-dirs ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adagrad
from tqdm import tqdm

from data_ext.balanced_domain_loader import BalancedDomainLoader
from data_ext.dataset_identity import build_dataset_record, sha256_file
from data_ext.split_utils import (
    read_split_entries,
    resolve_split_file,
    sample_id_from_entry,
)
from losses.hard_target_loss import hard_target_miss_loss
from losses.local_peak_cvar import domain_pixel_tail_risks, domain_tail_risks
from losses.schedules import linear_risk_weight
from losses.smooth_worst_domain import smooth_worst_domain
from losses.target_background_margin import domain_target_background_margin_risks
from model.MSHNet import MSHNet
from model.loss import SLSIoULoss
from utils.data import IRSTD_Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced multi-source MSHNet training with tail-risk losses"
    )
    parser.add_argument("--source-dirs", nargs="+", required=True)
    parser.add_argument(
        "--allow-single-source-inner-smoke",
        action="store_true",
        help=(
            "Permit a one-source detector only for a nested-LODO smoke test. "
            "Such a run is marked non-claim-bearing and must not enter main results."
        ),
    )
    parser.add_argument(
        "--source-split-files",
        nargs="+",
        default=None,
        help="Optional train split files aligned one-to-one with --source-dirs.",
    )
    parser.add_argument(
        "--source-names",
        nargs="+",
        default=None,
        help="Optional names aligned with --source-dirs; directory names are used otherwise.",
    )
    parser.add_argument(
        "--outer-fold-id",
        default=None,
        help="Auditable outer-fold identifier for RC detector checkpoints.",
    )
    parser.add_argument(
        "--outer-target",
        default=None,
        help="Final unseen target for this outer fold; must not be a source domain.",
    )
    parser.add_argument(
        "--held-out-domains",
        nargs="*",
        default=None,
        help="All domains excluded from detector training (outer target and inner pseudo-target).",
    )
    parser.add_argument("--batch-per-domain", type=int, default=2)
    parser.add_argument(
        "--epoch-steps",
        type=int,
        default=None,
        help="Defaults to floor(len(longest_domain) / batch_per_domain).",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument(
        "--risk-warmup-epochs",
        type=int,
        default=6,
        help="Number of initial epochs with zero tail/miss risk gradient.",
    )
    parser.add_argument(
        "--risk-ramp-epochs",
        type=int,
        default=10,
        help="Linear ramp length for tail/miss weights after risk warm-up.",
    )
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--risk-objective",
        choices=("separate", "margin"),
        default="separate",
        help=(
            "Risk formulation: the unchanged separate background-tail + hard-miss "
            "baseline, or the shift-invariant target--background logit margin candidate."
        ),
    )
    parser.add_argument(
        "--tail-mode",
        choices=("local-peak", "pixel"),
        default="local-peak",
        help="Use local-peak Tail-CVaR or the pixel-top-k ablation.",
    )
    parser.add_argument("--lambda-tail", type=float, default=0.1)
    parser.add_argument("--lambda-miss", type=float, default=0.1)
    parser.add_argument("--lambda-margin", type=float, default=0.1)
    parser.add_argument(
        "--target-background-margin",
        type=float,
        default=1.0,
        help="Required hard-target minus background-tail separation in logit units.",
    )
    parser.add_argument("--tail-q", type=float, default=0.01)
    parser.add_argument("--miss-q", type=float, default=0.2)
    parser.add_argument("--object-pixel-q", type=float, default=0.25)
    parser.add_argument("--tail-gamma", type=float, default=10.0)
    parser.add_argument("--peak-kernel-size", type=int, default=3)
    parser.add_argument("--peak-min-score", type=float, default=0.05)
    parser.add_argument("--plateau-atol", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-dir", default="repro_runs/rc_tail")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.source_dirs:
        raise ValueError("at least one --source-dir is required")
    if len(args.source_dirs) < 2 and not args.allow_single_source_inner_smoke:
        raise ValueError(
            "multi-source training requires at least two --source-dirs; "
            "use --allow-single-source-inner-smoke only for non-claim-bearing diagnostics"
        )
    if args.source_names is not None and len(args.source_names) != len(args.source_dirs):
        raise ValueError("--source-names must align one-to-one with --source-dirs")
    if args.source_split_files is not None and len(args.source_split_files) != len(args.source_dirs):
        raise ValueError("--source-split-files must align one-to-one with --source-dirs")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.warm_epoch < -1:
        raise ValueError("--warm-epoch must be at least -1")
    if args.risk_warmup_epochs < 0 or args.risk_ramp_epochs < 0:
        raise ValueError("risk warm-up and ramp epochs cannot be negative")
    if args.lambda_tail < 0.0 or args.lambda_miss < 0.0 or args.lambda_margin < 0.0:
        raise ValueError("loss weights must be non-negative")
    if (
        args.target_background_margin < 0.0
        or not np.isfinite(args.target_background_margin)
    ):
        raise ValueError("--target-background-margin must be finite and non-negative")
    if args.grad_clip_norm < 0.0:
        raise ValueError("--grad-clip-norm cannot be negative")
    canonical_sources = [
        str(Path(path).expanduser().resolve()) for path in args.source_dirs
    ]
    if len(set(canonical_sources)) != len(canonical_sources):
        raise ValueError(
            "--source-dirs contains duplicate physical/canonical directories"
        )
    for name in ("base_size", "crop_size"):
        value = int(getattr(args, name))
        if value <= 0 or value % 16 != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive multiple of 16")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def source_names(args: argparse.Namespace) -> List[str]:
    names = args.source_names or [Path(path).resolve().name for path in args.source_dirs]
    if any(not name for name in names):
        raise ValueError("all source names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"source names must be unique, got {names}")
    return names


def build_source_datasets(
    args: argparse.Namespace,
    names: Iterable[str],
) -> Dict[str, IRSTD_Dataset]:
    datasets = {}
    split_files = args.source_split_files or [None] * len(args.source_dirs)
    for name, directory, split_file in zip(names, args.source_dirs, split_files):
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"source dataset directory does not exist: {root}")
        selected_train_split = audited_source_train_split(root, split_file)
        dataset_args = SimpleNamespace(
            dataset_dir=str(root),
            base_size=args.base_size,
            crop_size=args.crop_size,
            split_file=str(selected_train_split),
        )
        # Only the training split is instantiated.  No source test split and
        # no target-domain data can influence checkpoint selection.
        datasets[name] = IRSTD_Dataset(dataset_args, mode="train")
    return datasets


def audited_source_train_split(
    dataset_root: str | Path,
    split_file: str | Path | None = None,
) -> Path:
    """Resolve a source-train split and prove it is disjoint from official test.

    The check is intentionally performed before any training dataset is
    instantiated.  It catches an explicitly supplied test split as well as
    contaminated ``trainval.txt`` mirrors such as the legacy NUDT copy.
    """

    root = Path(dataset_root).expanduser().resolve()
    train_path = resolve_split_file(root, "train", split_file)
    test_path = resolve_split_file(root, "test")
    if train_path == test_path:
        raise ValueError(
            f"source training split resolves to the official test split: {train_path}"
        )
    train_ids = {sample_id_from_entry(item) for item in read_split_entries(train_path)}
    test_ids = {sample_id_from_entry(item) for item in read_split_entries(test_path)}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        preview = ", ".join(overlap[:10])
        suffix = "" if len(overlap) <= 10 else f", ... ({len(overlap)} total)"
        raise ValueError(
            "source training split overlaps the official test split: "
            f"{preview}{suffix}; train={train_path}; test={test_path}"
        )
    return train_path


def build_detector_source_records(
    names: Iterable[str],
    datasets: Dict[str, IRSTD_Dataset],
) -> List[Dict[str, object]]:
    """Bind logical source names to the concrete datasets and train splits.

    Dataset identities are content-addressed and deliberately independent of
    the absolute source directory.  This catches the same physical dataset
    supplied through a copy, symlink or renamed directory, which canonical
    path checks alone cannot detect.
    """

    records: List[Dict[str, object]] = []
    for name in names:
        dataset = datasets[name]
        sample_ids = [sample_id_from_entry(entry) for entry in dataset.names]
        # Resolve exactly the selected train entries.  These helpers try only
        # deterministic per-sample candidates, so provenance generation never
        # enumerates or reads masks outside the selected training split.
        training_artifacts: list[tuple[str, str]] = []
        for entry in dataset.names:
            entry_stem = os.path.splitext(str(entry).strip())[0]
            training_artifacts.append(
                (
                    dataset._resolve_image_path(dataset.imgs_dir, entry_stem),
                    dataset._resolve_mask_path(dataset.label_dir, entry_stem),
                )
            )
        record = build_dataset_record(
            Path(dataset.imgs_dir).resolve().parent,
            dataset.list_dir,
            sample_ids,
            source_name=name,
            training_artifacts=training_artifacts,
        )
        if int(record["num_samples"]) != len(dataset):
            raise RuntimeError(
                f"source record sample count does not match dataset {name!r}"
            )
        records.append(record)

    identities = [str(record["dataset_identity_sha256"]) for record in records]
    if len(set(identities)) != len(identities):
        aliases: Dict[str, List[str]] = {}
        for record in records:
            aliases.setdefault(
                str(record["dataset_identity_sha256"]), []
            ).append(str(record["source_name"]))
        duplicates = [values for values in aliases.values() if len(values) > 1]
        raise ValueError(
            "detector sources contain duplicate dataset content under different "
            f"logical names: {duplicates}"
        )
    # Reject partial aliases too.  Dataset roots and filenames can be changed;
    # raw image-content leaves are the invariant contamination boundary.
    for left_index, left in enumerate(records):
        left_leaves = set(left["image_content_sha256_leaves"])
        for right in records[left_index + 1 :]:
            overlap = sorted(left_leaves & set(right["image_content_sha256_leaves"]))
            if overlap:
                raise ValueError(
                    "detector sources share raw image content under different "
                    f"logical names {left['source_name']!r} and "
                    f"{right['source_name']!r}; collision_count={len(overlap)}"
                )
    return records


def multiscale_sls_loss(
    sls_loss: SLSIoULoss,
    final_logits: torch.Tensor,
    auxiliary_logits: List[torch.Tensor],
    masks: torch.Tensor,
    warm_epoch: int,
    epoch: int,
) -> torch.Tensor:
    predictions = [final_logits] + list(auxiliary_logits)
    losses = []
    for prediction in predictions:
        if prediction.shape[-2:] == masks.shape[-2:]:
            target = masks
        else:
            target = F.adaptive_max_pool2d(masks, prediction.shape[-2:])
        losses.append(sls_loss(prediction, target, warm_epoch, epoch))
    return torch.stack(losses).mean()


def compute_domain_tail_risks(
    args: argparse.Namespace,
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.tail_mode == "pixel":
        return domain_pixel_tail_risks(
            logits,
            masks,
            domain_ids,
            q=args.tail_q,
            return_domain_ids=True,
        )
    return domain_tail_risks(
        logits,
        masks,
        domain_ids,
        q=args.tail_q,
        kernel_size=args.peak_kernel_size,
        min_score=args.peak_min_score,
        plateau_atol=args.plateau_atol,
        return_domain_ids=True,
    )


def compute_domain_margin_risks(
    args: argparse.Namespace,
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Image-first, domain-balanced target--background logit margins."""

    return domain_target_background_margin_risks(
        logits,
        masks,
        domain_ids,
        background_q=args.tail_q,
        target_q=args.miss_q,
        object_pixel_fraction=args.object_pixel_q,
        margin=args.target_background_margin,
        kernel_size=args.peak_kernel_size,
        plateau_atol=args.plateau_atol,
        return_domain_ids=True,
    )


def risk_objective_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Serializable capability record for configs and detector checkpoints."""

    if args.risk_objective == "margin":
        return {
            "name": "target_background_tail_margin",
            "candidate_status": "explicit_nondefault_candidate",
            "score_space": "logit_difference",
            "common_logit_shift_invariant": True,
            "background_summary": "deterministic_local_peak_top_fraction",
            "target_summary": "hard_object_bottom_fraction_of_top_pixel_logits",
            "aggregation": "image_first_then_equal_image_domain_mean_then_normalized_smooth_worst_domain",
            "empty_target_or_background": "graph_connected_zero",
            "background_q": args.tail_q,
            "target_q": args.miss_q,
            "object_pixel_fraction": args.object_pixel_q,
            "margin_logit": args.target_background_margin,
            "lambda_margin": args.lambda_margin,
        }
    return {
        "name": "separate_background_tail_plus_hard_miss",
        "candidate_status": "baseline_default",
        "score_space": "probability",
        "common_logit_shift_invariant": False,
        "tail_mode": args.tail_mode,
        "lambda_tail": args.lambda_tail,
        "lambda_miss": args.lambda_miss,
    }


def _git_state() -> Dict[str, object]:
    result: Dict[str, object] = {"revision": None, "dirty": None}
    safe_directory = f"safe.directory={Path.cwd().resolve()}"
    try:
        revision = subprocess.run(
            ["git", "-c", safe_directory, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-c", safe_directory, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result = {"revision": revision, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def create_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name or time.strftime(
        "MSHNet-tail-%Y-%m-%d-%H-%M-%S-s{}".format(args.seed),
        time.localtime(),
    )
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise ValueError("--run-name must be one safe path component")
    # Keep a user-supplied relative save path relative.  Persisted logs then
    # remain portable when a run directory is copied to another machine.
    run_dir = Path(args.save_dir).expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def save_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    names: List[str],
    detector_source_records: List[Dict[str, object]],
    epoch_metrics: Dict[str, object],
    run_config_sha256: str,
) -> None:
    payload = {
        "state_dict": model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": args.seed,
        "source_names": names,
        "detector_source_domains": names,
        "detector_source_records": detector_source_records,
        "outer_fold_id": args.outer_fold_id,
        "outer_target": args.outer_target,
        "held_out_domains": sorted(set(args.held_out_domains or [])),
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "head_training_schedule": "all_auxiliary_and_fused_heads_from_epoch_zero",
        "risk_objective": args.risk_objective,
        "risk_objective_contract": risk_objective_contract(args),
        "detector_capability_contract": {
            "risk_objective": risk_objective_contract(args),
        },
        "protocol_scope": (
            "single_source_inner_smoke_not_main_result"
            if len(names) == 1
            else "multi_source_protocol_candidate"
        ),
        "epoch_metrics": epoch_metrics,
        "training_args": dict(vars(args)),
        "run_config_sha256": run_config_sha256,
    }
    temporary = run_dir / "checkpoint_last.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, run_dir / "checkpoint_last.pt")


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sls_loss: SLSIoULoss,
    loader: BalancedDomainLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> Dict[str, object]:
    model.train()
    loader.set_epoch(epoch)
    # Always instantiate and supervise every auxiliary head plus the fused
    # final head.  During the SLS warm-up the loss itself reduces to plain
    # soft IoU, so this avoids switching at epoch warm+1 to a fusion layer that
    # has never received a gradient.
    multiscale_forward = True
    risk_weight = linear_risk_weight(
        epoch,
        args.risk_warmup_epochs,
        args.risk_ramp_epochs,
    )
    totals = {
        "loss": 0.0,
        "loss_sls": 0.0,
        "loss_tail": 0.0,
        "loss_miss": 0.0,
        "loss_margin": 0.0,
    }
    domain_risk_sums = {domain_id: 0.0 for domain_id in loader.domain_ids.values()}
    domain_risk_counts = {domain_id: 0 for domain_id in loader.domain_ids.values()}

    progress = tqdm(loader, total=len(loader), desc=f"epoch {epoch:04d}")
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)

        auxiliary_logits, final_logits = model(images, multiscale_forward)
        loss_sls = multiscale_sls_loss(
            sls_loss,
            final_logits,
            auxiliary_logits,
            masks,
            args.warm_epoch,
            epoch,
        )
        graph_zero = final_logits.sum() * 0.0
        if args.risk_objective == "margin":
            per_domain_risks, represented_ids = compute_domain_margin_risks(
                args,
                final_logits,
                masks,
                domain_ids,
            )
            loss_margin = smooth_worst_domain(
                per_domain_risks,
                gamma=args.tail_gamma,
            )
            # Do not silently blend the separate probability objectives into
            # the explicit margin candidate.
            loss_tail = graph_zero
            loss_miss = graph_zero
            loss = loss_sls + risk_weight * args.lambda_margin * loss_margin
        else:
            per_domain_risks, represented_ids = compute_domain_tail_risks(
                args,
                final_logits,
                masks,
                domain_ids,
            )
            loss_tail = smooth_worst_domain(per_domain_risks, gamma=args.tail_gamma)
            loss_miss = hard_target_miss_loss(
                final_logits,
                masks,
                q=args.miss_q,
                object_pixel_fraction=args.object_pixel_q,
            )
            loss_margin = graph_zero
            loss = (
                loss_sls
                + risk_weight * args.lambda_tail * loss_tail
                + risk_weight * args.lambda_miss * loss_miss
            )
        objective_loss = loss_margin if args.risk_objective == "margin" else loss_tail
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at epoch={epoch}, step={step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        values = {
            "loss": float(loss.detach()),
            "loss_sls": float(loss_sls.detach()),
            "loss_tail": float(loss_tail.detach()),
            "loss_miss": float(loss_miss.detach()),
            "loss_margin": float(loss_margin.detach()),
        }
        for key, value in values.items():
            totals[key] += value
        for domain_id, risk in zip(
            represented_ids.detach().cpu().tolist(),
            per_domain_risks.detach().cpu().tolist(),
        ):
            domain_risk_sums[int(domain_id)] += float(risk)
            domain_risk_counts[int(domain_id)] += 1

        progress.set_postfix(
            loss=f"{values['loss']:.4f}",
            risk=f"{float(objective_loss.detach()):.4f}",
        )

    steps = len(loader)
    id_to_name = {domain_id: name for name, domain_id in loader.domain_ids.items()}
    objective_risk_by_domain = {
        id_to_name[domain_id]: domain_risk_sums[domain_id]
        / max(1, domain_risk_counts[domain_id])
        for domain_id in sorted(domain_risk_sums)
    }
    metrics = {
        "epoch": epoch,
        "steps": steps,
        **{key: value / steps for key, value in totals.items()},
        "risk_objective": args.risk_objective,
        "objective_risk_by_domain": objective_risk_by_domain,
        "risk_weight": risk_weight,
        "effective_lambda_tail": (
            risk_weight * args.lambda_tail if args.risk_objective == "separate" else 0.0
        ),
        "effective_lambda_miss": (
            risk_weight * args.lambda_miss if args.risk_objective == "separate" else 0.0
        ),
        "effective_lambda_margin": (
            risk_weight * args.lambda_margin if args.risk_objective == "margin" else 0.0
        ),
        "domain_cycle_counts": dict(loader.last_cycle_counts),
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "head_training_schedule": "all_auxiliary_and_fused_heads_from_epoch_zero",
        "protocol_scope": (
            "single_source_inner_smoke_not_main_result"
            if len(loader.domain_names) == 1
            else "multi_source_protocol_candidate"
        ),
    }
    if args.risk_objective == "margin":
        metrics["margin_risk_by_domain"] = objective_risk_by_domain
    else:
        # Preserve the baseline's established metric name and meaning.
        metrics["tail_risk_by_domain"] = objective_risk_by_domain
    return metrics


def main() -> None:
    args = parse_args()
    _validate_args(args)
    names = source_names(args)
    held_out_domains = set(args.held_out_domains or [])
    if args.outer_target:
        held_out_domains.add(args.outer_target)
    overlap = sorted(set(names) & held_out_domains)
    if overlap:
        raise ValueError(
            "detector source domains overlap held-out domains: " + ", ".join(overlap)
        )
    if bool(args.outer_fold_id) != bool(args.outer_target):
        raise ValueError("--outer-fold-id and --outer-target must be supplied together")
    args.held_out_domains = sorted(held_out_domains)
    seed_everything(args.seed, args.deterministic)
    device = select_device(args.device)

    datasets = build_source_datasets(args, names)
    detector_source_records = build_detector_source_records(names, datasets)
    loader = BalancedDomainLoader(
        datasets,
        args.batch_per_domain,
        epoch_steps=args.epoch_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model: nn.Module = MSHNet(3)
    if args.data_parallel:
        cuda_device_count = torch.cuda.device_count()
        if device.type != "cuda" or cuda_device_count < 2:
            raise RuntimeError("--data-parallel requires at least two CUDA devices")
        if args.batch_per_domain % cuda_device_count != 0:
            raise ValueError(
                "With --data-parallel, --batch-per-domain must be divisible by "
                f"the visible CUDA device count ({cuda_device_count}) so every "
                "replica receives the same domain mixture for BatchNorm"
            )
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = Adagrad(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    sls_loss = SLSIoULoss()
    run_dir = create_run_dir(args)

    config = {
        **vars(args),
        # Runtime dataset construction resolves these paths internally, but
        # the persisted experiment contract retains the user's portable form.
        "source_dirs": [str(Path(path).expanduser()) for path in args.source_dirs],
        "source_names": names,
        "domain_ids": loader.domain_ids,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "detector_source_records": detector_source_records,
        "steps_per_epoch": len(loader),
        "total_batch_size": loader.total_batch_size,
        "loader_seed_rule": "seed + epoch*1000003 + domain_position*10007",
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "head_training_schedule": "all_auxiliary_and_fused_heads_from_epoch_zero",
        "risk_objective": args.risk_objective,
        "risk_objective_contract": risk_objective_contract(args),
        "detector_capability_contract": {
            "risk_objective": risk_objective_contract(args),
        },
        "protocol_scope": (
            "single_source_inner_smoke_not_main_result"
            if len(names) == 1
            else "multi_source_protocol_candidate"
        ),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "command": shlex.join(sys.argv),
        "git": _git_state(),
    }
    write_json(run_dir / "config.json", config)
    run_config_sha256 = sha256_file(run_dir / "config.json")
    # The full content-addressed source records can be hundreds of kilobytes;
    # keep them in config.json without flooding scheduler/stdout logs.
    startup_summary = {
        "run_dir": str(run_dir),
        "config_file": str(run_dir / "config.json"),
        "run_config_sha256": run_config_sha256,
        "source_names": names,
        "dataset_sizes": config["dataset_sizes"],
        "outer_fold_id": args.outer_fold_id,
        "outer_target": args.outer_target,
        "held_out_domains": args.held_out_domains,
        "checkpoint_selection": config["checkpoint_selection"],
        "risk_objective": args.risk_objective,
        "protocol_scope": config["protocol_scope"],
        "device_resolved": str(device),
        "cuda_device_count": config["cuda_device_count"],
        "data_parallel": args.data_parallel,
        "total_batch_size": loader.total_batch_size,
    }
    print(json.dumps(startup_summary, indent=2, sort_keys=True))

    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(args.epochs):
        epoch_metrics = train_one_epoch(
            model,
            optimizer,
            sls_loss,
            loader,
            device,
            args,
            epoch,
        )
        append_jsonl(metrics_path, epoch_metrics)
        save_checkpoint(
            run_dir,
            model,
            optimizer,
            epoch,
            args,
            names,
            detector_source_records,
            epoch_metrics,
            run_config_sha256,
        )
        print(json.dumps(epoch_metrics, sort_keys=True))

    temporary = run_dir / "weights_last.pt.tmp"
    torch.save(model_state_dict(model), temporary)
    os.replace(temporary, run_dir / "weights_last.pt")


if __name__ == "__main__":
    main()
