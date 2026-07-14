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
from losses.hard_target_loss import hard_target_miss_loss
from losses.local_peak_cvar import domain_pixel_tail_risks, domain_tail_risks
from losses.smooth_worst_domain import smooth_worst_domain
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
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--tail-mode",
        choices=("local-peak", "pixel"),
        default="local-peak",
        help="Use local-peak Tail-CVaR or the pixel-top-k ablation.",
    )
    parser.add_argument("--lambda-tail", type=float, default=0.1)
    parser.add_argument("--lambda-miss", type=float, default=0.1)
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
    if args.lambda_tail < 0.0 or args.lambda_miss < 0.0:
        raise ValueError("loss weights must be non-negative")
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
        dataset_args = SimpleNamespace(
            dataset_dir=str(root),
            base_size=args.base_size,
            crop_size=args.crop_size,
            split_file=split_file,
        )
        # Only the training split is instantiated.  No source test split and
        # no target-domain data can influence checkpoint selection.
        datasets[name] = IRSTD_Dataset(dataset_args, mode="train")
    return datasets


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
    epoch_metrics: Dict[str, object],
) -> None:
    payload = {
        "state_dict": model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": args.seed,
        "source_names": names,
        "detector_source_domains": names,
        "outer_fold_id": args.outer_fold_id,
        "outer_target": args.outer_target,
        "held_out_domains": sorted(set(args.held_out_domains or [])),
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "protocol_scope": (
            "single_source_inner_smoke_not_main_result"
            if len(names) == 1
            else "multi_source_protocol_candidate"
        ),
        "epoch_metrics": epoch_metrics,
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
    warm_flag = epoch > args.warm_epoch
    totals = {"loss": 0.0, "loss_sls": 0.0, "loss_tail": 0.0, "loss_miss": 0.0}
    domain_risk_sums = {domain_id: 0.0 for domain_id in loader.domain_ids.values()}
    domain_risk_counts = {domain_id: 0 for domain_id in loader.domain_ids.values()}

    progress = tqdm(loader, total=len(loader), desc=f"epoch {epoch:04d}")
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)

        auxiliary_logits, final_logits = model(images, warm_flag)
        loss_sls = multiscale_sls_loss(
            sls_loss,
            final_logits,
            auxiliary_logits,
            masks,
            args.warm_epoch,
            epoch,
        )
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
        loss = (
            loss_sls
            + args.lambda_tail * loss_tail
            + args.lambda_miss * loss_miss
        )
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
            tail=f"{values['loss_tail']:.4f}",
            miss=f"{values['loss_miss']:.4f}",
        )

    steps = len(loader)
    id_to_name = {domain_id: name for name, domain_id in loader.domain_ids.items()}
    tail_risk_by_domain = {
        id_to_name[domain_id]: domain_risk_sums[domain_id]
        / max(1, domain_risk_counts[domain_id])
        for domain_id in sorted(domain_risk_sums)
    }
    return {
        "epoch": epoch,
        "steps": steps,
        **{key: value / steps for key, value in totals.items()},
        "tail_risk_by_domain": tail_risk_by_domain,
        "domain_cycle_counts": dict(loader.last_cycle_counts),
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "protocol_scope": (
            "single_source_inner_smoke_not_main_result"
            if len(loader.domain_names) == 1
            else "multi_source_protocol_candidate"
        ),
    }


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
        "steps_per_epoch": len(loader),
        "total_batch_size": loader.total_batch_size,
        "loader_seed_rule": "seed + epoch*1000003 + domain_position*10007",
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
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
    print(json.dumps({"run_dir": str(run_dir), **config}, indent=2, sort_keys=True))

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
            epoch_metrics,
        )
        print(json.dumps(epoch_metrics, sort_keys=True))

    temporary = run_dir / "weights_last.pt.tmp"
    torch.save(model_state_dict(model), temporary)
    os.replace(temporary, run_dir / "weights_last.pt")


if __name__ == "__main__":
    main()
