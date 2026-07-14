"""Train MSHNet with balanced multi-source tail-risk objectives.

This entry point deliberately uses a fixed-last checkpoint policy.  It never
constructs a target/test loader and never chooses a checkpoint using target or
official-test labels.  Cross-domain evaluation belongs in a separate command.

Run from the repository root:

    python -m scripts.train_multisource_tail --source-dirs ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Tuple

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
from losses.target_background_margin import (
    DomainTailSeparationOutput,
    domain_tail_separation_loss,
    legacy_image_margin_loss,
)
from model.MSHNet import MSHNet
from losses.sls import SLSIoULoss
from utils.data import IRSTD_Dataset


DETECTOR_CHECKPOINT_FORMAT = "rc-irstd.detector.v2"
RESUME_CONTRACT_VERSION = 1
AAAI27_PILOT_RUN_CONTRACT_VERSION = "rc-irstd.aaai27-stage1-run-contract.v1"
AAAI27_ANALYSIS_PLAN_SCHEMA = "rc-irstd.aaai27-analysis-plan.v1"
AAAI27_PILOT_MATRIX_SCHEMA = "rc-irstd.aaai27-stage1-pilot-matrix.v1"
STAGE1_SLS_LOSS_EPS = 1e-8
_RESUME_MUTABLE_ARGUMENTS = frozenset({"epochs", "resume"})
DOMAIN_MARGIN_OBJECTIVES = frozenset(
    {"margin-background-only", "margin-target-only", "margin"}
)
MARGIN_DIAGNOSTIC_OBJECTIVES = DOMAIN_MARGIN_OBJECTIVES | frozenset(
    {"segmentation-only", "legacy-image-margin"}
)


def stage1_segmentation_loss_implementation() -> Dict[str, object]:
    """Return the shared, auditable SLS implementation identity for D0 and D3."""

    return {
        "qualified_name": f"{SLSIoULoss.__module__}.{SLSIoULoss.__qualname__}",
        "implementation_revision": "empty-mask-safe-epsilon-v1",
        "eps": STAGE1_SLS_LOSS_EPS,
        "multiscale_reduction": "mean_final_plus_four_auxiliary_heads",
        "paired_stage1_variants": ["D0", "D3"],
    }


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
        default=5,
        help="Number of initial epochs with zero auxiliary risk gradient.",
    )
    parser.add_argument(
        "--risk-ramp-epochs",
        type=int,
        default=10,
        help="Linear ramp length for the selected risk objective after warm-up.",
    )
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--risk-objective",
        choices=(
            "segmentation-only",
            "margin-background-only",
            "margin-target-only",
            "margin",
            "separate",
            "legacy-image-margin",
        ),
        default="margin",
        help=(
            "Frozen Stage-1 identities: segmentation-only=D0; margin-background-"
            "only=D1 (target tail stop-gradient); margin-target-only=D2 "
            "(background tail stop-gradient); margin=D3. 'separate' and "
            "'legacy-image-margin' are compatibility baselines outside D0-D3."
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
    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument(
        "--target-background-margin",
        type=float,
        default=1.0,
        help=(
            "Required domain target-lower-tail minus domain background-upper-tail "
            "separation in logit units."
        ),
    )
    parser.add_argument("--tail-q", type=float, default=0.05)
    parser.add_argument("--miss-q", type=float, default=0.25)
    parser.add_argument("--object-pixel-q", type=float, default=0.25)
    parser.add_argument("--tail-gamma", type=float, default=10.0)
    parser.add_argument("--peak-kernel-size", type=int, default=5)
    parser.add_argument(
        "--exclusion-radius",
        type=int,
        default=2,
        help=(
            "GT dilation radius in pixels for background candidates in the final "
            "margin objective."
        ),
    )
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
    parser.add_argument(
        "--engineering-smoke",
        action="store_true",
        help=(
            "Mark every emitted artifact as engineering smoke, never paper "
            "evidence. This changes provenance only, not the optimisation."
        ),
    )
    parser.add_argument(
        "--aaai27-pilot",
        action="store_true",
        help=(
            "Enable the fail-closed AAAI-27 Stage-1 development-pilot contract. "
            "This requires a clean tagged Git release, an exact source archive, "
            "an authorized analysis plan, and a matching frozen pilot-matrix run."
        ),
    )
    parser.add_argument(
        "--analysis-plan",
        default=None,
        metavar="JSON",
        help="Frozen analysis-plan JSON required by --aaai27-pilot.",
    )
    parser.add_argument(
        "--pilot-matrix",
        default=None,
        metavar="JSON",
        help="Frozen Stage-1 pilot-matrix JSON required by --aaai27-pilot.",
    )
    parser.add_argument(
        "--pilot-run-id",
        default=None,
        help="Unique run_id selected from --pilot-matrix.",
    )
    parser.add_argument(
        "--release-tag",
        default=None,
        help="Git tag at HEAD used to create the frozen source archive.",
    )
    parser.add_argument(
        "--source-archive",
        default=None,
        metavar="ZIP",
        help="Exact git-archive ZIP for the tagged release.",
    )
    parser.add_argument(
        "--source-archive-sha256-file",
        default=None,
        metavar="SHA256SUM",
        help=(
            "sha256sum-format checksum file for --source-archive. The archive "
            "is re-hashed and independently compared with git archive output."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT_LAST_PT",
        help=(
            "Resume the same fixed-last run from its checkpoint_last.pt. All "
            "immutable training/data/objective contracts must match; --epochs "
            "is interpreted as the new total epoch count."
        ),
    )
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
    if args.epoch_steps is not None and args.epoch_steps <= 0:
        raise ValueError("--epoch-steps must be positive when supplied")
    if args.engineering_smoke:
        if args.epoch_steps is None:
            raise ValueError("--engineering-smoke requires an explicit --epoch-steps cap")
        if args.epochs > 2 or args.epoch_steps > 10:
            raise ValueError(
                "--engineering-smoke is bounded to at most 2 epochs and 10 steps/epoch"
            )
        if args.risk_warmup_epochs != 0 or args.risk_ramp_epochs != 0:
            raise ValueError(
                "--engineering-smoke requires risk warm-up/ramp of zero so the "
                "risk gradient is exercised"
            )
    pilot_argument_names = (
        "analysis_plan",
        "pilot_matrix",
        "pilot_run_id",
        "release_tag",
        "source_archive",
        "source_archive_sha256_file",
    )
    if bool(getattr(args, "aaai27_pilot", False)):
        if bool(getattr(args, "engineering_smoke", False)):
            raise ValueError("--aaai27-pilot cannot be combined with --engineering-smoke")
        missing = [
            "--" + name.replace("_", "-")
            for name in pilot_argument_names
            if not getattr(args, name, None)
        ]
        if missing:
            raise ValueError(
                "--aaai27-pilot requires explicit provenance arguments: "
                + ", ".join(missing)
            )
    else:
        supplied = [
            "--" + name.replace("_", "-")
            for name in pilot_argument_names
            if getattr(args, name, None) is not None
        ]
        if supplied:
            raise ValueError(
                "pilot provenance arguments require --aaai27-pilot: "
                + ", ".join(supplied)
            )
    if not np.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("--lr must be finite and positive")
    if args.warm_epoch < -1:
        raise ValueError("--warm-epoch must be at least -1")
    if args.risk_warmup_epochs < 0 or args.risk_ramp_epochs < 0:
        raise ValueError("risk warm-up and ramp epochs cannot be negative")
    loss_weights = (args.lambda_tail, args.lambda_miss, args.lambda_margin)
    if any(not np.isfinite(value) or value < 0.0 for value in loss_weights):
        raise ValueError("loss weights must be finite and non-negative")
    if args.risk_objective == "segmentation-only" and args.lambda_margin != 0.0:
        raise ValueError("D0 segmentation-only requires --lambda-margin 0")
    if args.risk_objective in DOMAIN_MARGIN_OBJECTIVES and args.lambda_margin <= 0.0:
        raise ValueError("D1-D3 margin objectives require positive --lambda-margin")
    if (
        args.target_background_margin < 0.0
        or not np.isfinite(args.target_background_margin)
    ):
        raise ValueError("--target-background-margin must be finite and non-negative")
    if args.exclusion_radius < 0:
        raise ValueError("--exclusion-radius must be non-negative")
    if not np.isfinite(args.grad_clip_norm) or args.grad_clip_norm < 0.0:
        raise ValueError("--grad-clip-norm must be finite and non-negative")
    for name in ("tail_q", "miss_q", "object_pixel_q"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and in (0, 1]"
            )
    if not np.isfinite(args.tail_gamma) or args.tail_gamma <= 0.0:
        raise ValueError("--tail-gamma must be finite and positive")
    if not np.isfinite(args.peak_min_score) or not 0.0 <= args.peak_min_score <= 1.0:
        raise ValueError("--peak-min-score must be finite and in [0, 1]")
    if not np.isfinite(args.plateau_atol) or args.plateau_atol < 0.0:
        raise ValueError("--plateau-atol must be finite and non-negative")
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser()
        if resume_path.name != "checkpoint_last.pt":
            raise ValueError("--resume must point to a checkpoint_last.pt artifact")
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
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


def protocol_scope(args: argparse.Namespace, names: Iterable[str]) -> str:
    """Return the explicit evidence scope persisted in every run artifact."""

    names = list(names)
    if bool(getattr(args, "engineering_smoke", False)):
        return "engineering_smoke_not_paper_evidence"
    if len(names) == 1:
        return "single_source_inner_smoke_not_main_result"
    return "multi_source_protocol_candidate"


def validate_fold_identity(args: argparse.Namespace) -> None:
    """Require an explicit outer identity for every non-smoke detector run."""

    if bool(args.outer_fold_id) != bool(args.outer_target):
        raise ValueError("--outer-fold-id and --outer-target must be supplied together")
    if not args.engineering_smoke and not args.outer_fold_id:
        raise ValueError(
            "non-smoke detector training requires --outer-fold-id and --outer-target"
        )


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


def compute_domain_margin_output(
    args: argparse.Namespace,
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
) -> DomainTailSeparationOutput:
    """Final two-tail margin with the hinge applied after domain aggregation."""

    trainable_tail = {
        "margin-background-only": "background",
        "margin-target-only": "target",
        "margin": "both",
    }.get(args.risk_objective, "both")
    return domain_tail_separation_loss(
        logits,
        masks,
        domain_ids,
        margin=args.target_background_margin,
        background_tail_fraction=args.tail_q,
        hard_object_fraction=args.miss_q,
        object_top_fraction=args.object_pixel_q,
        peak_kernel_size=args.peak_kernel_size,
        exclusion_radius=getattr(args, "exclusion_radius", 2),
        worst_gamma=args.tail_gamma,
        plateau_atol=args.plateau_atol,
        trainable_tail=trainable_tail,
    )


def compute_legacy_image_margin_output(
    args: argparse.Namespace,
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
) -> DomainTailSeparationOutput:
    """Legacy ablation with the hinge formed per image before domain averaging."""

    return legacy_image_margin_loss(
        logits,
        masks,
        domain_ids,
        margin=args.target_background_margin,
        background_tail_fraction=args.tail_q,
        hard_object_fraction=args.miss_q,
        object_top_fraction=args.object_pixel_q,
        peak_kernel_size=args.peak_kernel_size,
        worst_gamma=args.tail_gamma,
        plateau_atol=args.plateau_atol,
    )


def compute_domain_margin_risks(
    args: argparse.Namespace,
    logits: torch.Tensor,
    masks: torch.Tensor,
    domain_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return valid domain gaps and IDs for backward-compatible diagnostics."""

    output = compute_domain_margin_output(args, logits, masks, domain_ids)
    valid = output.valid_domain_mask
    return output.domain_gap[valid], output.domain_ids[valid]


def risk_objective_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Serializable capability record for configs and detector checkpoints."""

    if args.risk_objective == "segmentation-only":
        return {
            "name": "multiscale_sls_segmentation_only",
            "stage1_variant": "D0",
            "candidate_status": "primary_detector_baseline",
            "auxiliary_risk_gradient": False,
            "effective_lambda_margin": 0.0,
            "tail_diagnostics_computed_without_gradient": True,
        }
    if args.risk_objective in DOMAIN_MARGIN_OBJECTIVES:
        variant, trainable_tail, detached_tail = {
            "margin-background-only": ("D1", "background", "target"),
            "margin-target-only": ("D2", "target", "background"),
            "margin": ("D3", "both", "none"),
        }[args.risk_objective]
        contract = {
            "name": "domain_target_background_tail_separation",
            "stage1_variant": variant,
            "candidate_status": (
                "final_primary_objective_default_cli"
                if variant == "D3"
                else "single_trainable_tail_ablation"
            ),
            "score_space": "logit_difference",
            "common_logit_shift_invariant": True,
            "trainable_tail": trainable_tail,
            "stop_gradient_tail": detached_tail,
            "forward_hinge_identical_across_D1_D2_D3": True,
            "background_summary": (
                "per_image_deterministic_local_peak_top_fraction_after_gt_dilation"
            ),
            "target_summary": "domain_hard_object_bottom_fraction_of_top_pixel_logits",
            "aggregation": (
                "equal_image_background_domain_mean_and_equal_object_target_tail_"
                "then_domain_hinge_then_normalized_smooth_worst_valid_domain"
            ),
            "hinge_level": "domain_after_two_tail_aggregation",
            "target_free_image": "contributes_background_tail",
            "target_free_domain": "background_diagnostic_only_excluded_from_hinge",
            "plateau_rule": "one_deterministic_8_connected_representative",
            "background_q": args.tail_q,
            "target_q": args.miss_q,
            "object_pixel_fraction": args.object_pixel_q,
            "peak_kernel_size": getattr(args, "peak_kernel_size", 3),
            "gt_exclusion_radius": getattr(args, "exclusion_radius", 2),
            "plateau_atol": getattr(args, "plateau_atol", 0.0),
            "normalized_smooth_max_gamma": getattr(args, "tail_gamma", 10.0),
            "margin_logit": args.target_background_margin,
            "lambda_margin": args.lambda_margin,
        }
        if variant == "D1":
            contract["name"] = "domain_tail_separation_background_trainable_only"
        elif variant == "D2":
            contract["name"] = "domain_tail_separation_target_trainable_only"
        return contract
    if args.risk_objective == "legacy-image-margin":
        return {
            "name": "legacy_image_paired_target_background_margin",
            "candidate_status": "ablation_only_not_final_method",
            "score_space": "logit_difference",
            "common_logit_shift_invariant": True,
            "background_summary": (
                "per_image_deterministic_local_peak_upper_tail_without_gt_dilation"
            ),
            "target_summary": "per_image_hard_object_lower_tail_of_top_pixel_logits",
            "aggregation": (
                "per_image_hinge_then_equal_image_domain_mean_then_normalized_"
                "smooth_worst_domain"
            ),
            "hinge_level": "image_before_domain_aggregation",
            "invalid_image_pair": "graph_connected_zero_included_in_domain_mean",
            "target_free_image": "zero_legacy_margin_risk",
            "plateau_rule": "legacy_local_rank_representative",
            "background_q": args.tail_q,
            "target_q": args.miss_q,
            "object_pixel_fraction": args.object_pixel_q,
            "peak_kernel_size": getattr(args, "peak_kernel_size", 3),
            "gt_exclusion_radius": 0,
            "plateau_atol": getattr(args, "plateau_atol", 0.0),
            "normalized_smooth_max_gamma": getattr(args, "tail_gamma", 10.0),
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


def detector_capability_contract(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "risk_objective": risk_objective_contract(args),
        "segmentation_loss_implementation": (
            stage1_segmentation_loss_implementation()
        ),
        "training_diagnostics": {
            "domain_tail_fields": [
                "background_tail_logit",
                "target_tail_logit",
                "raw_gap_logit",
                "margin_violation",
            ],
            "global_logit_fields": [
                "mean",
                "std",
                "q001",
                "q50",
                "q99",
                "q999",
                "min",
                "max",
                "max_abs",
                "nonfinite_count",
            ],
            "gradient_fields": ["pre_clip_global_l2_norm", "finite"],
            "candidate_object_counts": True,
            "empty_target_image_counts": True,
            "parameter_norm": "post_update_global_l2",
            "learning_rate_and_elapsed_steps": True,
        },
        "resume": {
            "supported": True,
            "checkpoint_format": DETECTOR_CHECKPOINT_FORMAT,
            "selection": "fixed_last_no_test_or_target_validation",
            "immutable_contract_required": True,
        },
    }


def _installed_source_state(package_root: Path) -> Dict[str, object]:
    """Content-address an installed distribution when no Git metadata exists."""

    digest = hashlib.sha256()
    files: list[Path] = []
    for root_name in (
        "data_ext",
        "evaluation",
        "losses",
        "model",
        "rc",
        "rc_irstd",
        "scripts",
        "utils",
    ):
        source_root = package_root / root_name
        if source_root.is_dir():
            files.extend(source_root.rglob("*.py"))
    for name in ("pyproject.toml", "requirements.txt"):
        candidate = package_root / name
        if candidate.is_file():
            files.append(candidate)
    files = sorted({path.resolve() for path in files})
    if not files:
        raise RuntimeError(
            f"cannot content-address installed RC-IRSTD sources under {package_root}"
        )
    for path in files:
        relative = path.relative_to(package_root.resolve()).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "mode": "installed_source_tree_sha256",
        "revision": None,
        "dirty": None,
        "tracked_diff_sha256": None,
        "untracked_manifest_sha256": None,
        "untracked_file_count": None,
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(files),
    }


def _git_state(repository_root: Path | None = None) -> Dict[str, object]:
    result: Dict[str, object] = {
        "revision": None,
        "dirty": None,
        "tracked_diff_sha256": None,
        "untracked_manifest_sha256": None,
        "untracked_file_count": None,
    }
    repository_root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    safe_directory = f"safe.directory={repository_root}"
    try:
        discovered_root = Path(
            subprocess.run(
                ["git", "-c", safe_directory, "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
                cwd=repository_root,
            ).stdout.strip()
        ).resolve()
        if discovered_root != repository_root:
            return _installed_source_state(repository_root)
        revision = subprocess.run(
            ["git", "-c", safe_directory, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-c", safe_directory, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root,
        ).stdout
        tracked_diff = subprocess.run(
            [
                "git",
                "-c",
                safe_directory,
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            cwd=repository_root,
        ).stdout
        untracked_output = subprocess.run(
            [
                "git",
                "-c",
                safe_directory,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            cwd=repository_root,
        ).stdout
        untracked_paths = [item for item in untracked_output.split(b"\0") if item]
        untracked_digest = hashlib.sha256()
        for encoded_path in sorted(untracked_paths):
            path = repository_root / os.fsdecode(encoded_path)
            content_digest = hashlib.sha256()
            if path.is_symlink():
                content_digest.update(b"symlink\0")
                content_digest.update(os.fsencode(os.readlink(path)))
            else:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        content_digest.update(chunk)
            untracked_digest.update(len(encoded_path).to_bytes(8, "big"))
            untracked_digest.update(encoded_path)
            untracked_digest.update(content_digest.digest())
        result = {
            "mode": "git_worktree",
            "revision": revision,
            "dirty": bool(status.strip()),
            "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "untracked_manifest_sha256": untracked_digest.hexdigest(),
            "untracked_file_count": len(untracked_paths),
            "source_tree_sha256": None,
            "source_file_count": None,
        }
    except (OSError, subprocess.CalledProcessError):
        result = _installed_source_state(repository_root)
    return result


def execution_fingerprint() -> Dict[str, object]:
    """Bind continuation to the exact source tree and runtime stack."""

    cuda_devices: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            cuda_devices.append(
                {
                    "logical_index": index,
                    "name": properties.name,
                    "uuid": str(getattr(properties, "uuid", "unavailable")),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(capability[0]), int(capability[1])],
                }
            )
    return {
        "schema_version": 1,
        "git": _git_state(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_device_names": [item["name"] for item in cuda_devices],
        "visible_cuda_devices": cuda_devices,
    }


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
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_text(path: Path, payload: str) -> None:
    """Atomically write a UTF-8 text artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def canonical_training_command() -> str:
    """Return a directly replayable module invocation for this process."""

    return shlex.join(
        [sys.executable, "-m", "scripts.train_multisource_tail", *sys.argv[1:]]
    )


def _contract_path(
    value: str | Path,
    repository_root: Path,
    *,
    label: str,
    require_in_repository: bool,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if require_in_repository:
        try:
            path.relative_to(repository_root)
        except ValueError as error:
            raise ValueError(
                f"{label} must be a tracked file inside {repository_root}: {path}"
            ) from error
    return path


def _portable_contract_path(path: Path, repository_root: Path) -> str:
    try:
        return path.resolve().relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_json_object(path: Path, label: str) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain one JSON object: {path}")
    return payload


def _git_command(
    repository_root: Path,
    arguments: List[str],
    *,
    text: bool = True,
) -> str | bytes:
    command = [
        "git",
        "-c",
        f"safe.directory={repository_root}",
        *arguments,
    ]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=text,
            cwd=repository_root,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", b"" if not text else "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        detail = str(stderr).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Git command failed: {' '.join(command)}{suffix}") from error


def _verify_tracked_head_file(path: Path, repository_root: Path, label: str) -> str:
    relative = path.relative_to(repository_root).as_posix()
    _git_command(repository_root, ["ls-files", "--error-unmatch", "--", relative])
    committed = _git_command(
        repository_root,
        ["show", f"HEAD:{relative}"],
        text=False,
    )
    assert isinstance(committed, bytes)
    committed_sha256 = hashlib.sha256(committed).hexdigest()
    working_sha256 = sha256_file(path)
    if committed_sha256 != working_sha256:
        raise ValueError(
            f"{label} bytes differ from the file frozen at HEAD: {relative}"
        )
    return working_sha256


def _read_single_sha256sum(path: Path, archive: Path) -> str:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(
            "source archive checksum file must contain exactly one non-empty entry"
        )
    match = re.fullmatch(r"([0-9A-Fa-f]{64})[ \t]+\*?(.+)", lines[0])
    if match is None:
        raise ValueError("source archive checksum is not in sha256sum format")
    recorded_name = match.group(2).strip()
    if Path(recorded_name).name != archive.name:
        raise ValueError(
            "source archive checksum names a different file: " + recorded_name
        )
    return match.group(1).lower()


def _validate_authorized_analysis_plan(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != AAAI27_ANALYSIS_PLAN_SCHEMA:
        raise ValueError("unsupported AAAI-27 analysis-plan schema")
    if payload.get("plan_status") != "frozen_stage1_pilot_authorized":
        raise ValueError("analysis plan has not authorized the Stage-1 pilot")
    if payload.get("contains_observed_results") is not False:
        raise ValueError("analysis plan must explicitly contain no observed results")
    authorization = payload.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("analysis plan authorization must be an object")
    required_true = ("gate_minus_1", "stage1_development_comparisons")
    for name in required_true:
        if authorization.get(name) is not True:
            raise ValueError(f"analysis plan has not authorized {name}")
    forbidden_true = ("official_test_model_evaluation", "paper_performance_claims")
    for name in forbidden_true:
        if authorization.get(name) is not False:
            raise ValueError(f"analysis plan must keep {name} disabled")


def _select_pilot_matrix_run(
    payload: Mapping[str, object],
    run_id: str,
) -> Dict[str, object]:
    if payload.get("schema_version") != AAAI27_PILOT_MATRIX_SCHEMA:
        raise ValueError("unsupported AAAI-27 Stage-1 pilot-matrix schema")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("pilot matrix must contain a non-empty runs list")
    matches = [
        item
        for item in runs
        if isinstance(item, dict) and item.get("run_id") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"pilot run_id must occur exactly once in the matrix: {run_id!r}"
        )
    return dict(matches[0])


def _repo_relative_matrix_path(
    value: object,
    repository_root: Path,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pilot-matrix {label} must be a non-empty relative path")
    portable = Path(value)
    if portable.is_absolute():
        raise ValueError(f"pilot-matrix {label} must be repository-root-relative")
    resolved = (repository_root / portable).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as error:
        raise ValueError(f"pilot-matrix {label} escapes the repository") from error
    return resolved


def _validate_pilot_matrix_run_contract(
    args: argparse.Namespace,
    selected_run: Mapping[str, object],
    pilot_matrix: Mapping[str, object],
    analysis_plan: Mapping[str, object],
    repository_root: Path,
) -> None:
    """Prove that the current trainer CLI is the selected frozen matrix row."""

    required = {
        "run_id",
        "phase",
        "experiment_scope",
        "variant",
        "seed",
        "epochs",
        "fixed_last",
        "sources",
        "source_dirs",
        "source_split_files",
        "outer_fold_id",
        "outer_target",
        "held_out",
        "primary_diagnostic_domains",
        "evaluation_diagnostic_domains",
        "evaluation_diagnostic_files",
        "gpu_visible_devices",
        "data_parallel",
        "output_dir",
    }
    missing = sorted(required.difference(selected_run))
    if missing:
        raise ValueError(f"pilot-matrix run is missing fields: {missing}")
    if selected_run["run_id"] != getattr(args, "pilot_run_id"):
        raise ValueError("pilot-matrix run_id differs from --pilot-run-id")
    if selected_run["experiment_scope"] != "single_seed_stage1_gate":
        raise ValueError("pilot-matrix run is outside the single-seed Stage-1 gate")
    if selected_run["fixed_last"] is not True:
        raise ValueError("pilot-matrix run must use fixed-last checkpoint selection")
    if pilot_matrix.get("contains_observed_results") is not False:
        raise ValueError("pilot matrix must explicitly contain no observed results")
    scheduling = pilot_matrix.get("scheduling")
    if not isinstance(scheduling, Mapping) or not isinstance(
        scheduling.get("phases"), list
    ):
        raise ValueError("pilot matrix scheduling contract is incomplete")
    scheduled_matches = []
    for phase in scheduling["phases"]:
        if not isinstance(phase, Mapping):
            raise ValueError("pilot matrix scheduling phase must be an object")
        run_ids = phase.get("concurrent_run_ids")
        if not isinstance(run_ids, list):
            raise ValueError("pilot matrix scheduling phase has no run list")
        if selected_run["run_id"] in run_ids:
            scheduled_matches.append(phase.get("phase_id"))
    if scheduled_matches != [selected_run["phase"]]:
        raise ValueError("pilot run is not uniquely assigned to its scheduling phase")

    variant = selected_run["variant"]
    expected_objective = {"D0": "segmentation-only", "D3": "margin"}.get(variant)
    if expected_objective is None:
        raise ValueError("Stage-1 single-seed pilot matrix may contain only D0 or D3")
    if getattr(args, "risk_objective") != expected_objective:
        raise ValueError(
            f"current risk objective does not match matrix variant {variant}"
        )

    stage1 = analysis_plan.get("stage1_contract")
    if not isinstance(stage1, Mapping):
        raise ValueError("analysis plan has no Stage-1 contract")
    variants = stage1.get("variants")
    common = stage1.get("common_training")
    gpu_protocol = stage1.get("gpu_protocol")
    if not all(isinstance(item, Mapping) for item in (variants, common, gpu_protocol)):
        raise ValueError("analysis plan Stage-1 training contracts are incomplete")
    assert isinstance(variants, Mapping)
    assert isinstance(common, Mapping)
    assert isinstance(gpu_protocol, Mapping)
    variant_contract = variants.get(variant)
    if not isinstance(variant_contract, Mapping):
        raise ValueError(f"analysis plan has no contract for variant {variant}")

    protocol = pilot_matrix.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("pilot matrix protocol must be an object")
    if protocol.get("checkpoint_selection") != (
        "fixed_last_no_test_or_target_validation"
    ) or protocol.get("diagnostics_select_checkpoint") is not False:
        raise ValueError("pilot matrix violates fixed-last/no-diagnostic selection")
    if protocol.get("optimizer") != "Adagrad":
        raise ValueError("pilot matrix optimizer differs from the trainer")
    if protocol.get("epoch_steps_mode") != "full_longest_domain":
        raise ValueError("pilot matrix must use the full longest-domain epoch")
    protocol_argument_fields = {
        "seed": "seed",
        "epochs": "epochs",
        "deterministic": "deterministic",
        "learning_rate": "lr",
        "warm_epoch": "warm_epoch",
        "risk_warmup_epochs": "risk_warmup_epochs",
        "risk_ramp_epochs": "risk_ramp_epochs",
        "base_size": "base_size",
        "crop_size": "crop_size",
        "batch_per_domain": "batch_per_domain",
        "num_workers": "num_workers",
        "tail_mode": "tail_mode",
        "lambda_tail": "lambda_tail",
        "lambda_miss": "lambda_miss",
        "target_background_margin": "target_background_margin",
        "tail_q": "tail_q",
        "miss_q": "miss_q",
        "object_pixel_q": "object_pixel_q",
        "tail_gamma": "tail_gamma",
        "peak_kernel_size": "peak_kernel_size",
        "peak_min_score": "peak_min_score",
        "plateau_atol": "plateau_atol",
        "grad_clip_norm": "grad_clip_norm",
    }
    for protocol_name, argument_name in protocol_argument_fields.items():
        if protocol.get(protocol_name) != getattr(args, argument_name):
            raise ValueError(
                f"current --{argument_name.replace('_', '-')} differs from "
                f"pilot-matrix protocol.{protocol_name}"
            )
    matrix_variants = protocol.get("variants")
    matrix_variant = (
        matrix_variants.get(variant)
        if isinstance(matrix_variants, Mapping)
        else None
    )
    if not isinstance(matrix_variant, Mapping):
        raise ValueError(f"pilot matrix protocol has no {variant} variant")
    for matrix_name, argument_name in (
        ("risk_objective", "risk_objective"),
        ("lambda_margin", "lambda_margin"),
        ("exclusion_radius", "exclusion_radius"),
    ):
        if matrix_variant.get(matrix_name) != getattr(args, argument_name):
            raise ValueError(
                f"current {argument_name} differs from pilot-matrix {variant}"
            )

    exact_values = {
        "seed": selected_run["seed"],
        "epochs": selected_run["epochs"],
        "risk_objective": variant_contract.get("risk_objective"),
        "lambda_margin": variant_contract.get("lambda_margin"),
        "lr": common.get("learning_rate"),
        "warm_epoch": common.get("warm_epoch"),
        "risk_warmup_epochs": common.get("risk_warmup_epochs"),
        "risk_ramp_epochs": common.get("risk_ramp_epochs"),
        "base_size": common.get("base_size"),
        "crop_size": common.get("crop_size"),
        "batch_per_domain": gpu_protocol.get("batch_per_domain"),
        "deterministic": common.get("deterministic"),
        "data_parallel": selected_run["data_parallel"],
    }
    for name, expected in exact_values.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"current --{name.replace('_', '-')}={getattr(args, name)!r} "
                f"differs from frozen matrix/plan value {expected!r}"
            )
    if int(selected_run["epochs"]) != int(stage1.get("single_seed_pilot_epochs", -1)):
        raise ValueError("pilot-matrix epochs differ from the frozen analysis plan")
    if int(selected_run["seed"]) != int(stage1.get("single_seed_pilot_seed", -1)):
        raise ValueError("pilot-matrix seed differs from the frozen analysis plan")
    if getattr(args, "device") != "cuda":
        raise ValueError("AAAI-27 Stage-1 pilot requires explicit --device cuda")
    if getattr(args, "epoch_steps") is not None:
        raise ValueError("AAAI-27 Stage-1 pilot forbids a capped --epoch-steps run")
    if bool(getattr(args, "allow_single_source_inner_smoke", False)):
        raise ValueError("AAAI-27 Stage-1 pilot cannot use a single-source smoke flag")
    if getattr(args, "run_name") != getattr(args, "pilot_run_id"):
        raise ValueError("AAAI-27 pilot requires --run-name to equal --pilot-run-id")
    expected_output_dir = _repo_relative_matrix_path(
        selected_run["output_dir"],
        repository_root,
        label="output_dir",
    )
    actual_output_dir = (
        Path(getattr(args, "save_dir")).expanduser() / str(getattr(args, "run_name"))
    )
    if not actual_output_dir.is_absolute():
        actual_output_dir = repository_root / actual_output_dir
    if actual_output_dir.resolve() != expected_output_dir:
        raise ValueError(
            "final --save-dir/--run-name path differs from pilot-matrix output_dir"
        )
    if getattr(args, "resume") is None:
        if expected_output_dir.exists():
            raise FileExistsError(
                f"fresh pilot output_dir already exists: {expected_output_dir}"
            )
    else:
        resume_dir = Path(getattr(args, "resume")).expanduser().resolve().parent
        if resume_dir != expected_output_dir:
            raise ValueError("pilot resume checkpoint is outside its frozen output_dir")

    names = source_names(args)
    matrix_sources = selected_run["sources"]
    if not isinstance(matrix_sources, list) or names != matrix_sources:
        raise ValueError("current source-name order differs from the pilot matrix")
    matrix_source_dirs = selected_run["source_dirs"]
    if not isinstance(matrix_source_dirs, list) or len(matrix_source_dirs) != len(names):
        raise ValueError("pilot-matrix source_dirs do not align with sources")
    actual_source_dirs: list[Path] = []
    for index, value in enumerate(getattr(args, "source_dirs")):
        actual = Path(value).expanduser()
        if not actual.is_absolute():
            raise ValueError(
                "AAAI-27 pilot --source-dirs must be absolute paths materialized "
                "from the repository-relative matrix"
            )
        actual_source_dirs.append(actual.resolve())
        expected = _repo_relative_matrix_path(
            matrix_source_dirs[index],
            repository_root,
            label=f"source_dirs[{index}]",
        )
        if actual_source_dirs[-1] != expected:
            raise ValueError("current source directory differs from the pilot matrix")

    matrix_split_files = selected_run["source_split_files"]
    actual_split_values = getattr(args, "source_split_files")
    if (
        not isinstance(matrix_split_files, list)
        or not isinstance(actual_split_values, list)
        or len(matrix_split_files) != len(names)
        or len(actual_split_values) != len(names)
    ):
        raise ValueError("pilot split files must align one-to-one with sources")
    for index, value in enumerate(actual_split_values):
        actual = Path(value).expanduser()
        if not actual.is_absolute():
            raise ValueError(
                "AAAI-27 pilot --source-split-files must be absolute paths; "
                "trainer-relative split paths resolve under each dataset root"
            )
        expected = _repo_relative_matrix_path(
            matrix_split_files[index],
            repository_root,
            label=f"source_split_files[{index}]",
        )
        if actual.resolve() != expected:
            raise ValueError("current source split differs from the pilot matrix")
        if expected.name != "detector_fit.txt" or not expected.is_file():
            raise ValueError("Stage-1 pilot sources must use existing detector_fit.txt files")

    scalar_roles = {
        "outer_fold_id": "outer_fold_id",
        "outer_target": "outer_target",
    }
    for argument_name, matrix_name in scalar_roles.items():
        if getattr(args, argument_name) != selected_run[matrix_name]:
            raise ValueError(f"current {argument_name} differs from the pilot matrix")
    if sorted(getattr(args, "held_out_domains") or []) != sorted(
        selected_run["held_out"]
        if isinstance(selected_run["held_out"], list)
        else []
    ):
        raise ValueError("current held-out domains differ from the pilot matrix")

    diagnostic_domains = selected_run["primary_diagnostic_domains"]
    evaluation_domains = selected_run["evaluation_diagnostic_domains"]
    diagnostic_files = selected_run["evaluation_diagnostic_files"]
    if (
        not isinstance(diagnostic_domains, list)
        or not diagnostic_domains
        or not isinstance(evaluation_domains, list)
        or not evaluation_domains
        or not isinstance(diagnostic_files, list)
        or len(evaluation_domains) != len(diagnostic_files)
        or not set(diagnostic_domains).issubset(evaluation_domains)
    ):
        raise ValueError(
            "pilot evaluation domains/files must align and contain the primary domains"
        )
    for index, value in enumerate(diagnostic_files):
        path = _repo_relative_matrix_path(
            value,
            repository_root,
            label=f"evaluation_diagnostic_files[{index}]",
        )
        if path.name != "detector_diagnostic.txt" or not path.is_file():
            raise ValueError(
                "Stage-1 pilot evaluation may reference only existing "
                "detector_diagnostic.txt files"
            )

    visible_devices = selected_run["gpu_visible_devices"]
    if (
        not isinstance(visible_devices, list)
        or not visible_devices
        or any(type(item) is not int or item not in {0, 1, 2} for item in visible_devices)
        or len(set(visible_devices)) != len(visible_devices)
    ):
        raise ValueError("pilot matrix GPU list must contain unique physical IDs 0,1,2")
    visible_environment = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_environment = ",".join(str(item) for item in visible_devices)
    if visible_environment != expected_environment:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES differs from the frozen pilot matrix: "
            f"expected {expected_environment!r}, got {visible_environment!r}"
        )
    if bool(selected_run["data_parallel"]) != (len(visible_devices) > 1):
        raise ValueError("pilot data_parallel mode is inconsistent with visible GPUs")


def validate_aaai27_pilot_release(
    args: argparse.Namespace,
    repository_root: str | Path | None = None,
) -> Dict[str, object]:
    """Validate and bind a clean tagged release before any pilot data is loaded.

    The caller supplies paths, never digests. Every digest is recomputed here.
    The source ZIP is compared both with its checksum manifest and with fresh
    ``git archive`` output for the requested tag, closing the common loophole
    where an unrelated archive is paired with a self-consistent hash.
    """

    if not bool(getattr(args, "aaai27_pilot", False)):
        raise ValueError("AAAI-27 release validation requires --aaai27-pilot")
    repository_root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    discovered_root = Path(
        str(_git_command(repository_root, ["rev-parse", "--show-toplevel"])).strip()
    ).resolve()
    if discovered_root != repository_root:
        raise ValueError(
            f"AAAI-27 pilot must run from the frozen repository root {repository_root}"
        )

    git_state = _git_state(repository_root)
    if git_state.get("mode") != "git_worktree" or not git_state.get("revision"):
        raise ValueError("AAAI-27 pilot requires an auditable Git worktree")
    if git_state.get("dirty") is not False:
        raise ValueError("AAAI-27 pilot requires a clean Git worktree")

    release_tag = str(getattr(args, "release_tag"))
    _git_command(repository_root, ["check-ref-format", f"refs/tags/{release_tag}"])
    head = str(_git_command(repository_root, ["rev-parse", "HEAD"])).strip()
    tagged_commit = str(
        _git_command(repository_root, ["rev-parse", f"refs/tags/{release_tag}^{{commit}}"])
    ).strip()
    if tagged_commit != head:
        raise ValueError(
            f"release tag {release_tag!r} does not resolve to current HEAD"
        )

    analysis_plan_path = _contract_path(
        getattr(args, "analysis_plan"),
        repository_root,
        label="analysis plan",
        require_in_repository=True,
    )
    pilot_matrix_path = _contract_path(
        getattr(args, "pilot_matrix"),
        repository_root,
        label="pilot matrix",
        require_in_repository=True,
    )
    analysis_plan = _load_json_object(analysis_plan_path, "analysis plan")
    pilot_matrix = _load_json_object(pilot_matrix_path, "pilot matrix")
    _validate_authorized_analysis_plan(analysis_plan)
    selected_run = _select_pilot_matrix_run(
        pilot_matrix,
        str(getattr(args, "pilot_run_id")),
    )
    analysis_plan_sha256 = _verify_tracked_head_file(
        analysis_plan_path, repository_root, "analysis plan"
    )
    pilot_matrix_sha256 = _verify_tracked_head_file(
        pilot_matrix_path, repository_root, "pilot matrix"
    )
    hash_contracts = analysis_plan.get("hash_contracts")
    if not isinstance(hash_contracts, Mapping):
        raise ValueError("analysis plan hash_contracts must be an object")
    matrix_hash_contract = hash_contracts.get("stage1_pilot_matrix")
    expected_matrix_path = _portable_contract_path(
        pilot_matrix_path, repository_root
    )
    if (
        not isinstance(matrix_hash_contract, Mapping)
        or matrix_hash_contract.get("path") != expected_matrix_path
        or matrix_hash_contract.get("sha256") != pilot_matrix_sha256
    ):
        raise ValueError(
            "analysis plan does not authorize the selected Stage-1 pilot matrix"
        )
    analysis_binding = pilot_matrix.get("analysis_plan_binding")
    if not isinstance(analysis_binding, Mapping):
        raise ValueError("pilot matrix has no analysis_plan_binding")
    if (
        analysis_binding.get("path")
        != _portable_contract_path(analysis_plan_path, repository_root)
        or analysis_binding.get("schema_version") != AAAI27_ANALYSIS_PLAN_SCHEMA
        or analysis_plan.get("plan_status")
        not in analysis_binding.get("allowed_plan_statuses", [])
    ):
        raise ValueError("pilot matrix analysis-plan identity/status binding mismatch")
    for matrix_name, plan_name in (
        ("stage1_config", "stage1_config"),
        ("split_manifest", "official_train_split_manifest"),
    ):
        if analysis_binding.get(matrix_name) != hash_contracts.get(plan_name):
            raise ValueError(
                f"pilot matrix {matrix_name} differs from the analysis plan"
            )

    # Reuse the complete plan validator rather than duplicating its dataset,
    # split, near-duplicate, D0-D3 and Gate -1 checks in the trainer.
    from scripts.validate_aaai27_analysis_plan import validate_plan

    plan_audit = validate_plan(analysis_plan_path, repository_root)
    if (
        plan_audit.get("status") != "PASS"
        or plan_audit.get("gate_minus_1") is not True
        or plan_audit.get("plan_sha256") != analysis_plan_sha256
    ):
        raise ValueError("complete analysis-plan audit did not pass Gate -1")
    _validate_pilot_matrix_run_contract(
        args,
        selected_run,
        pilot_matrix,
        analysis_plan,
        repository_root,
    )

    archive_path = _contract_path(
        getattr(args, "source_archive"),
        repository_root,
        label="source archive",
        require_in_repository=False,
    )
    checksum_path = _contract_path(
        getattr(args, "source_archive_sha256_file"),
        repository_root,
        label="source archive checksum file",
        require_in_repository=False,
    )
    release_contract = pilot_matrix.get("release_contract")
    if not isinstance(release_contract, Mapping):
        raise ValueError("pilot matrix has no release_contract")
    expected_archive = _repo_relative_matrix_path(
        release_contract.get("source_archive"),
        repository_root,
        label="release_contract.source_archive",
    )
    expected_checksum = _repo_relative_matrix_path(
        release_contract.get("source_archive_sha256_file"),
        repository_root,
        label="release_contract.source_archive_sha256_file",
    )
    if (
        release_contract.get("tag") != release_tag
        or expected_archive != archive_path
        or expected_checksum != checksum_path
    ):
        raise ValueError("CLI release tag/archive paths differ from the pilot matrix")
    recorded_archive_sha256 = _read_single_sha256sum(checksum_path, archive_path)
    archive_sha256 = sha256_file(archive_path)
    if recorded_archive_sha256 != archive_sha256:
        raise ValueError("source archive SHA-256 differs from its checksum file")

    with tempfile.TemporaryDirectory(prefix="rc-irstd-pilot-archive-") as temporary:
        regenerated_archive = Path(temporary) / archive_path.name
        _git_command(
            repository_root,
            [
                "archive",
                "--format=zip",
                f"--output={regenerated_archive}",
                release_tag,
            ],
        )
        regenerated_sha256 = sha256_file(regenerated_archive)
    if regenerated_sha256 != archive_sha256:
        raise ValueError(
            "source archive bytes are not the exact git archive of the release tag"
        )

    return {
        "schema_version": "rc-irstd.aaai27-pilot-release-binding.v1",
        "git": git_state,
        "release_tag": release_tag,
        "release_commit": head,
        "analysis_plan": {
            "path": _portable_contract_path(analysis_plan_path, repository_root),
            "sha256": analysis_plan_sha256,
        },
        "pilot_matrix": {
            "path": _portable_contract_path(pilot_matrix_path, repository_root),
            "sha256": pilot_matrix_sha256,
            "run_id": str(getattr(args, "pilot_run_id")),
            "selected_run": selected_run,
        },
        "source_archive": {
            "path": _portable_contract_path(archive_path, repository_root),
            "sha256": archive_sha256,
            "checksum_file": _portable_contract_path(checksum_path, repository_root),
            "checksum_file_sha256": sha256_file(checksum_path),
            "verification": "checksum_file_and_exact_git_archive_bytes",
        },
    }


def write_aaai27_pilot_run_artifacts(
    run_dir: Path,
    args: argparse.Namespace,
    names: List[str],
    detector_source_records: List[Dict[str, object]],
    run_config_sha256: str,
    release_binding: Mapping[str, object],
    execution_fingerprint_payload: Mapping[str, object],
) -> str:
    """Create the immutable, machine-readable pilot provenance bundle."""

    if not bool(getattr(args, "aaai27_pilot", False)):
        raise ValueError("pilot artifacts require --aaai27-pilot")
    git = release_binding.get("git")
    source_archive = release_binding.get("source_archive")
    pilot_matrix = release_binding.get("pilot_matrix")
    if not all(isinstance(item, Mapping) for item in (git, source_archive, pilot_matrix)):
        raise ValueError("validated release binding is incomplete")
    assert isinstance(git, Mapping)
    assert isinstance(source_archive, Mapping)
    assert isinstance(pilot_matrix, Mapping)
    if git.get("dirty") is not False:
        raise ValueError("pilot run artifacts cannot represent a dirty release")

    source_identities = []
    for record in detector_source_records:
        identity = {
            "source_name": record.get("source_name"),
            "dataset_identity_sha256": record.get("dataset_identity_sha256"),
            "split_sha256": record.get("split_sha256"),
            "ordered_sample_ids_sha256": record.get("ordered_sample_ids_sha256"),
            "split_image_artifact_sha256": record.get(
                "split_image_artifact_sha256"
            ),
            "training_artifact_sha256": record.get("training_artifact_sha256"),
            "num_samples": record.get("num_samples"),
        }
        for hash_name in (
            "dataset_identity_sha256",
            "split_sha256",
            "ordered_sample_ids_sha256",
            "split_image_artifact_sha256",
            "training_artifact_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(identity[hash_name])) is None:
                raise ValueError(
                    f"pilot source record has no valid {hash_name}: "
                    f"{identity['source_name']!r}"
                )
        if not identity["source_name"] or int(identity["num_samples"] or 0) <= 0:
            raise ValueError("pilot source record has no name or positive sample count")
        source_identities.append(identity)
    command = canonical_training_command()
    contract: Dict[str, object] = {
        "schema_version": AAAI27_PILOT_RUN_CONTRACT_VERSION,
        "method": "RC-IRSTD-v5-two-stage-no-reject",
        "stage": 1,
        "pilot_run_id": getattr(args, "pilot_run_id"),
        "evidence_scope": "stage1_development_gate_official_test_sealed",
        "claim_bearing": False,
        "claim_bearing_reason": (
            "development pilot artifact; paper claims require the frozen "
            "post-gate expansion and separate result freeze"
        ),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release": dict(release_binding),
        "command": command,
        "run_config": {
            "path": "config.json",
            "sha256": run_config_sha256,
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "stage1_variant": risk_objective_contract(args).get("stage1_variant"),
            "risk_objective": args.risk_objective,
            "segmentation_loss_implementation": (
                stage1_segmentation_loss_implementation()
            ),
            "source_names": names,
            "source_identities": source_identities,
            "outer_fold_id": args.outer_fold_id,
            "outer_target": args.outer_target,
            "held_out_domains": sorted(set(args.held_out_domains or [])),
            "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        },
        "environment": dict(execution_fingerprint_payload),
        "expected_artifacts": {
            "checkpoint": "checkpoint_last.pt",
            "checkpoint_sha256": "checkpoint_sha256.txt",
            "metrics": "metrics.jsonl",
        },
    }
    write_json(run_dir / "run_contract.json", contract)
    run_contract_sha256 = sha256_file(run_dir / "run_contract.json")
    write_text(run_dir / "command.txt", command + "\n")
    write_text(run_dir / "git_commit.txt", str(release_binding["release_commit"]) + "\n")
    write_text(run_dir / "git_tag.txt", str(release_binding["release_tag"]) + "\n")
    write_text(
        run_dir / "git_status.txt",
        "clean=true\n"
        f"dirty={str(bool(git['dirty'])).lower()}\n"
        f"tracked_diff_sha256={git.get('tracked_diff_sha256')}\n"
        f"untracked_manifest_sha256={git.get('untracked_manifest_sha256')}\n"
        f"untracked_file_count={git.get('untracked_file_count')}\n",
    )
    write_text(
        run_dir / "source_archive_sha256.txt",
        f"{source_archive['sha256']}  {Path(str(source_archive['path'])).name}\n",
    )
    write_text(
        run_dir / "analysis_plan_sha256.txt",
        f"{release_binding['analysis_plan']['sha256']}  "
        f"{release_binding['analysis_plan']['path']}\n",
    )
    write_text(
        run_dir / "pilot_matrix_sha256.txt",
        f"{pilot_matrix['sha256']}  {pilot_matrix['path']}\n",
    )
    write_json(run_dir / "environment.json", dict(execution_fingerprint_payload))
    return run_contract_sha256


def checkpoint_sha256_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name("checkpoint_sha256.txt")


def write_checkpoint_sha256(checkpoint_path: Path) -> str:
    digest = sha256_file(checkpoint_path)
    write_text(
        checkpoint_sha256_path(checkpoint_path),
        f"{digest}  {checkpoint_path.name}\n",
    )
    return digest


def verify_checkpoint_sha256(checkpoint_path: Path, *, required: bool) -> str | None:
    sidecar = checkpoint_sha256_path(checkpoint_path)
    if not sidecar.is_file():
        if required:
            raise FileNotFoundError(
                f"pilot checkpoint is missing SHA-256 sidecar: {sidecar}"
            )
        return None
    recorded = _read_single_sha256sum(sidecar, checkpoint_path)
    actual = sha256_file(checkpoint_path)
    if recorded != actual:
        raise ValueError("checkpoint_last.pt differs from checkpoint_sha256.txt")
    return actual


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def resume_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Return the immutable CLI contract that must survive continuation."""

    values = {
        key: value
        for key, value in vars(args).items()
        if key not in _RESUME_MUTABLE_ARGUMENTS
    }
    if "held_out_domains" in values:
        values["held_out_domains"] = sorted(set(values["held_out_domains"] or []))
    return {
        "schema_version": RESUME_CONTRACT_VERSION,
        "immutable_training_args": values,
        "risk_objective_contract": risk_objective_contract(args),
        "segmentation_loss_implementation": (
            stage1_segmentation_loss_implementation()
        ),
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
    }


def _rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(payload: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"resume checkpoint RNG state is missing: {sorted(missing)}")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(torch.as_tensor(payload["torch_cpu"], device="cpu"))
    if torch.cuda.is_available():
        cuda_states = payload.get("torch_cuda_all")
        if not isinstance(cuda_states, (list, tuple)):
            raise ValueError("CUDA resume requires checkpoint torch_cuda_all RNG states")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "resume checkpoint CUDA RNG device count differs from the visible devices"
            )
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(state, device="cpu") for state in cuda_states]
        )


def _load_model_state_dict(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state_dict, strict=True)


def _last_metrics_epoch(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"resume run is missing metrics.jsonl: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("resume metrics.jsonl is empty")
    payload = json.loads(lines[-1])
    if not isinstance(payload, Mapping) or "epoch" not in payload:
        raise ValueError("last metrics.jsonl record has no epoch")
    return int(payload["epoch"])


def load_resume_checkpoint(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    names: List[str],
    detector_source_records: List[Dict[str, object]],
    device: torch.device,
    pilot_release_binding: Mapping[str, object] | None = None,
) -> Tuple[Path, int, str, Dict[str, object]]:
    """Fail closed unless a fixed-last artifact matches the current run exactly."""

    checkpoint_path = Path(args.resume).expanduser().resolve()
    run_dir = checkpoint_path.parent
    pilot_mode = bool(getattr(args, "aaai27_pilot", False))
    verify_checkpoint_sha256(checkpoint_path, required=pilot_mode)
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"resume run is missing config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("resume config.json must contain an object")
    expected_segmentation_loss = stage1_segmentation_loss_implementation()
    if config.get("segmentation_loss_implementation") != expected_segmentation_loss:
        raise ValueError("resume config segmentation-loss implementation mismatch")
    run_config_sha256 = sha256_file(config_path)
    run_contract_sha256: str | None = None
    if pilot_mode:
        if pilot_release_binding is None:
            raise ValueError("pilot resume requires a validated release binding")
        if config.get("pilot_release_binding") != pilot_release_binding:
            raise ValueError("pilot resume release/plan/matrix binding mismatch")
        run_contract_path = run_dir / "run_contract.json"
        if not run_contract_path.is_file():
            raise FileNotFoundError(
                f"pilot resume is missing run_contract.json: {run_contract_path}"
            )
        run_contract = _load_json_object(run_contract_path, "pilot run contract")
        if run_contract.get("schema_version") != AAAI27_PILOT_RUN_CONTRACT_VERSION:
            raise ValueError("unsupported pilot run-contract schema")
        if run_contract.get("release") != pilot_release_binding:
            raise ValueError("pilot run contract release binding mismatch")
        if run_contract.get("pilot_run_id") != getattr(args, "pilot_run_id"):
            raise ValueError("pilot run contract run_id mismatch")
        run_contract_training = run_contract.get("training")
        if (
            not isinstance(run_contract_training, Mapping)
            or run_contract_training.get("segmentation_loss_implementation")
            != expected_segmentation_loss
        ):
            raise ValueError(
                "pilot run contract segmentation-loss implementation mismatch"
            )
        config_binding = run_contract.get("run_config")
        if (
            not isinstance(config_binding, Mapping)
            or config_binding.get("sha256") != run_config_sha256
        ):
            raise ValueError("pilot run contract config SHA-256 mismatch")
        run_contract_sha256 = sha256_file(run_contract_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("resume checkpoint must contain a mapping")
    required = {
        "format_version",
        "state_dict",
        "optimizer",
        "epoch",
        "source_names",
        "detector_source_records",
        "held_out_domains",
        "outer_fold_id",
        "outer_target",
        "checkpoint_selection",
        "risk_objective_contract",
        "segmentation_loss_implementation",
        "resume_contract",
        "run_config_sha256",
        "rng_state",
        "execution_fingerprint",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"resume checkpoint is missing: {sorted(missing)}")
    if checkpoint["format_version"] != DETECTOR_CHECKPOINT_FORMAT:
        raise ValueError("unsupported detector resume checkpoint format")
    if checkpoint["checkpoint_selection"] != "fixed_last_no_test_or_target_validation":
        raise ValueError("resume checkpoint is not a fixed-last detector checkpoint")
    if str(checkpoint["run_config_sha256"]) != run_config_sha256:
        raise ValueError("resume config.json SHA-256 differs from checkpoint contract")
    expected_resume_contract = resume_contract(args)
    if checkpoint["resume_contract"] != expected_resume_contract:
        raise ValueError("resume immutable training/objective contract mismatch")
    if list(checkpoint["source_names"]) != list(names):
        raise ValueError("resume source-domain order mismatch")
    if checkpoint["detector_source_records"] != detector_source_records:
        raise ValueError("resume detector source records/content hashes mismatch")
    if sorted(checkpoint["held_out_domains"]) != sorted(args.held_out_domains or []):
        raise ValueError("resume held-out-domain contract mismatch")
    if checkpoint["outer_fold_id"] != args.outer_fold_id:
        raise ValueError("resume outer-fold contract mismatch")
    if checkpoint["outer_target"] != args.outer_target:
        raise ValueError("resume outer-target contract mismatch")
    if checkpoint["risk_objective_contract"] != risk_objective_contract(args):
        raise ValueError("resume risk-objective capability contract mismatch")
    if checkpoint["segmentation_loss_implementation"] != expected_segmentation_loss:
        raise ValueError("resume checkpoint segmentation-loss implementation mismatch")
    if checkpoint["execution_fingerprint"] != execution_fingerprint():
        raise ValueError(
            "resume source-tree/runtime execution fingerprint mismatch"
        )
    if pilot_mode and checkpoint.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("pilot checkpoint run-contract SHA-256 mismatch")

    completed_epoch = int(checkpoint["epoch"])
    if completed_epoch < 0:
        raise ValueError("resume checkpoint epoch must be non-negative")
    if _last_metrics_epoch(run_dir / "metrics.jsonl") != completed_epoch:
        raise ValueError("resume metrics.jsonl and checkpoint_last.pt epochs disagree")
    start_epoch = completed_epoch + 1
    if args.epochs <= start_epoch:
        raise ValueError(
            f"--epochs must exceed resumed start epoch {start_epoch}; got {args.epochs}"
        )

    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("resume checkpoint state_dict must be a mapping")
    _load_model_state_dict(model, state_dict)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng_state(checkpoint["rng_state"])
    return run_dir, start_epoch, run_config_sha256, config


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
    execution_fingerprint_payload: Mapping[str, object] | None = None,
    run_contract_sha256: str | None = None,
) -> None:
    payload = {
        "format_version": DETECTOR_CHECKPOINT_FORMAT,
        "state_dict": model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": args.seed,
        "source_names": names,
        "detector_source_domains": names,
        "detector_source_records": detector_source_records,
        "dataset_sizes": {
            str(record["source_name"]): int(record["num_samples"])
            for record in detector_source_records
            if "source_name" in record and "num_samples" in record
        },
        "outer_fold_id": args.outer_fold_id,
        "outer_target": args.outer_target,
        "held_out_domains": sorted(set(args.held_out_domains or [])),
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "head_training_schedule": "all_auxiliary_and_fused_heads_from_epoch_zero",
        "risk_objective": args.risk_objective,
        "segmentation_loss_implementation": (
            stage1_segmentation_loss_implementation()
        ),
        "risk_objective_contract": risk_objective_contract(args),
        "detector_capability_contract": detector_capability_contract(args),
        "protocol_scope": protocol_scope(args, names),
        "epoch_metrics": epoch_metrics,
        "training_args": dict(vars(args)),
        "resume_contract": resume_contract(args),
        "rng_state": _rng_state(),
        "run_config_sha256": run_config_sha256,
        "execution_fingerprint": dict(
            execution_fingerprint_payload or execution_fingerprint()
        ),
    }
    if run_contract_sha256 is not None:
        if re.fullmatch(r"[0-9a-f]{64}", run_contract_sha256) is None:
            raise ValueError("run_contract_sha256 must be a lowercase SHA-256 digest")
        payload["run_contract_sha256"] = run_contract_sha256
    temporary = run_dir / "checkpoint_last.pt.tmp"
    torch.save(payload, temporary)
    checkpoint_path = run_dir / "checkpoint_last.pt"
    os.replace(temporary, checkpoint_path)
    write_checkpoint_sha256(checkpoint_path)


def gradient_norm_and_finite(model: nn.Module) -> Tuple[float, bool]:
    """Measure the pre-clip global L2 gradient norm with one scalar sync."""

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not parameters:
        return 0.0, True
    total_norm = nn.utils.clip_grad_norm_(
        parameters,
        max_norm=float("inf"),
        error_if_nonfinite=False,
    )
    finite = bool(torch.isfinite(total_norm))
    return float(total_norm.detach()), finite


def model_parameters_are_finite(model: nn.Module) -> bool:
    """Check the post-update model with a single device synchronisation."""

    parameters = [
        parameter.detach().reshape(-1)
        for parameter in model.parameters()
        if parameter.numel() > 0
    ]
    if not parameters:
        return True
    return bool(torch.isfinite(torch.cat(parameters)).all())


def model_parameter_l2_norm(model: nn.Module) -> float:
    """Return the post-update global parameter L2 norm."""

    total: torch.Tensor | None = None
    for parameter in model.parameters():
        if parameter.numel() == 0:
            continue
        contribution = parameter.detach().double().square().sum()
        total = contribution if total is None else total + contribution
    return 0.0 if total is None else float(total.sqrt())


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
    totals: Dict[str, float] = {
        "loss": 0.0,
        "loss_sls": 0.0,
        "loss_tail": 0.0,
        "loss_miss": 0.0,
        "loss_margin": 0.0,
    }
    domain_risk_sums = {domain_id: 0.0 for domain_id in loader.domain_ids.values()}
    domain_risk_counts = {domain_id: 0 for domain_id in loader.domain_ids.values()}
    diagnostic_names = (
        "background_tail_logit",
        "target_tail_logit",
        "raw_gap_logit",
        "margin_violation",
        "background_candidate_count_per_image",
        "object_count_per_batch",
    )
    diagnostic_sums = {
        name: {domain_id: 0.0 for domain_id in loader.domain_ids.values()}
        for name in diagnostic_names
    }
    diagnostic_counts = {
        name: {domain_id: 0 for domain_id in loader.domain_ids.values()}
        for name in diagnostic_names
    }
    valid_margin_sums = {
        domain_id: 0.0 for domain_id in loader.domain_ids.values()
    }
    valid_margin_counts = {
        domain_id: 0 for domain_id in loader.domain_ids.values()
    }
    logit_sum = 0.0
    logit_square_sum = 0.0
    logit_count = 0
    logit_min = float("inf")
    logit_max = float("-inf")
    logit_quantile_samples: list[torch.Tensor] = []
    nonfinite_logit_count = 0
    empty_target_image_counts = {
        domain_id: 0 for domain_id in loader.domain_ids.values()
    }
    gradient_norm_sum = 0.0
    gradient_norm_max = 0.0
    gradients_finite = True
    risk_gradient_checked = False
    risk_gradient_norm = 0.0
    risk_gradients_finite = True

    progress = tqdm(loader, total=len(loader), desc=f"epoch {epoch:04d}")
    for step, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        domain_ids = batch["domain_id"].to(device, non_blocking=True)

        auxiliary_logits, final_logits = model(images, multiscale_forward)
        detached_logits = final_logits.detach()
        finite_mask = torch.isfinite(detached_logits)
        step_nonfinite = int((~finite_mask).sum())
        nonfinite_logit_count += step_nonfinite
        if step_nonfinite:
            raise FloatingPointError(
                f"non-finite final logits at epoch={epoch}, step={step}"
            )
        logit_sum += float(detached_logits.sum(dtype=torch.float64))
        logit_square_sum += float(
            detached_logits.square().sum(dtype=torch.float64)
        )
        logit_count += detached_logits.numel()
        logit_min = min(logit_min, float(detached_logits.min()))
        logit_max = max(logit_max, float(detached_logits.max()))
        flattened_logits = detached_logits.reshape(-1)
        sample_count = min(4096, flattened_logits.numel())
        sample_indices = torch.linspace(
            0,
            flattened_logits.numel() - 1,
            steps=sample_count,
            device=flattened_logits.device,
        ).long()
        logit_quantile_samples.append(
            flattened_logits.index_select(0, sample_indices).float().cpu()
        )
        empty_targets = masks.flatten(1).sum(dim=1) <= 0
        for domain_id, empty in zip(
            domain_ids.detach().cpu().tolist(),
            empty_targets.detach().cpu().tolist(),
        ):
            empty_target_image_counts[int(domain_id)] += int(bool(empty))
        loss_sls = multiscale_sls_loss(
            sls_loss,
            final_logits,
            auxiliary_logits,
            masks,
            args.warm_epoch,
            epoch,
        )
        graph_zero = final_logits.sum() * 0.0
        margin_output: DomainTailSeparationOutput | None = None
        if args.risk_objective == "segmentation-only":
            # D0 optimises SLS only. Tail values are retained as detached
            # mechanism diagnostics so D0/D3 can be compared without giving D0
            # an auxiliary-risk gradient.
            with torch.no_grad():
                margin_output = compute_domain_margin_output(
                    args,
                    final_logits,
                    masks,
                    domain_ids,
                )
            objective_mask = margin_output.valid_domain_mask
            per_domain_risks = margin_output.domain_violation[objective_mask]
            represented_ids = margin_output.domain_ids[objective_mask]
            loss_margin = graph_zero
            loss_tail = graph_zero
            loss_miss = graph_zero
            risk_auxiliary_loss = graph_zero
            loss = loss_sls
        elif args.risk_objective in DOMAIN_MARGIN_OBJECTIVES | {
            "legacy-image-margin"
        }:
            if args.risk_objective in DOMAIN_MARGIN_OBJECTIVES:
                margin_output = compute_domain_margin_output(
                    args,
                    final_logits,
                    masks,
                    domain_ids,
                )
                objective_mask = margin_output.valid_domain_mask
            else:
                margin_output = compute_legacy_image_margin_output(
                    args,
                    final_logits,
                    masks,
                    domain_ids,
                )
                # The historical objective averages invalid image pairs as
                # zeros and includes every represented domain in its smooth max.
                objective_mask = torch.ones_like(
                    margin_output.valid_domain_mask, dtype=torch.bool
                )
            per_domain_risks = margin_output.domain_violation[objective_mask]
            represented_ids = margin_output.domain_ids[objective_mask]
            # The output already applies the normalized smooth worst-domain
            # reduction under the selected aggregation contract. Reducing it
            # again here would silently change that objective.
            loss_margin = margin_output.loss
            # Do not blend the separate probability objectives into either
            # logit-margin formulation.
            loss_tail = graph_zero
            loss_miss = graph_zero
            risk_auxiliary_loss = args.lambda_margin * loss_margin
            loss = loss_sls + risk_weight * risk_auxiliary_loss
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
            risk_auxiliary_loss = (
                args.lambda_tail * loss_tail + args.lambda_miss * loss_miss
            )
            loss = loss_sls + risk_weight * risk_auxiliary_loss
        objective_loss = (
            loss_margin
            if args.risk_objective in MARGIN_DIAGNOSTIC_OBJECTIVES
            else loss_tail
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at epoch={epoch}, step={step}")

        optimizer.zero_grad(set_to_none=True)
        if (
            not risk_gradient_checked
            and risk_weight > 0.0
            and args.risk_objective != "segmentation-only"
        ):
            trainable = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            auxiliary_gradients = torch.autograd.grad(
                risk_auxiliary_loss,
                trainable,
                retain_graph=True,
                allow_unused=True,
            )
            squared_norm = final_logits.new_zeros(())
            for gradient in auxiliary_gradients:
                if gradient is not None:
                    squared_norm = squared_norm + gradient.detach().square().sum()
            risk_gradient_norm_tensor = squared_norm.sqrt()
            risk_gradients_finite = bool(torch.isfinite(risk_gradient_norm_tensor))
            if not risk_gradients_finite:
                raise FloatingPointError(
                    f"non-finite auxiliary-risk gradient at epoch={epoch}, step={step}"
                )
            risk_gradient_norm = float(risk_gradient_norm_tensor)
            risk_gradient_checked = True
        loss.backward()
        gradient_norm, gradient_finite = gradient_norm_and_finite(model)
        gradients_finite = gradients_finite and gradient_finite
        if not gradient_finite:
            raise FloatingPointError(
                f"non-finite gradients at epoch={epoch}, step={step}"
            )
        gradient_norm_sum += gradient_norm
        gradient_norm_max = max(gradient_norm_max, gradient_norm)
        if args.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip_norm,
                error_if_nonfinite=True,
            )
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
        if margin_output is not None:
            output_ids = margin_output.domain_ids.detach().cpu().tolist()
            valid_mask = margin_output.valid_domain_mask.detach().cpu().tolist()
            diagnostic_tensors = {
                "background_tail_logit": margin_output.domain_background_tail,
                "target_tail_logit": margin_output.domain_target_tail,
                "raw_gap_logit": margin_output.domain_raw_gap,
                "margin_violation": margin_output.domain_violation,
                "background_candidate_count_per_image": (
                    margin_output.domain_background_candidate_mean
                ),
                "object_count_per_batch": margin_output.domain_object_count,
            }
            for name, tensor in diagnostic_tensors.items():
                values_by_domain = tensor.detach().cpu().tolist()
                for position, (domain_id, value) in enumerate(
                    zip(output_ids, values_by_domain)
                ):
                    # A target-free domain has a real background/candidate/object
                    # diagnostic, but no defined target tail or raw gap.
                    undefined_without_target = name in {
                        "target_tail_logit",
                        "raw_gap_logit",
                    } or (
                        name == "margin_violation"
                        and args.risk_objective
                        in DOMAIN_MARGIN_OBJECTIVES | {"segmentation-only"}
                    )
                    if undefined_without_target and not bool(valid_mask[position]):
                        continue
                    diagnostic_sums[name][int(domain_id)] += float(value)
                    diagnostic_counts[name][int(domain_id)] += 1
            for domain_id, valid in zip(output_ids, valid_mask):
                valid_margin_sums[int(domain_id)] += float(bool(valid))
                valid_margin_counts[int(domain_id)] += 1

        progress.set_postfix(
            loss=f"{values['loss']:.4f}",
            risk=f"{float(objective_loss.detach()):.4f}",
        )

    steps = len(loader)
    parameters_finite = model_parameters_are_finite(model)
    if not parameters_finite:
        raise FloatingPointError(f"non-finite model parameter after epoch={epoch}")
    id_to_name = {domain_id: name for name, domain_id in loader.domain_ids.items()}
    objective_risk_by_domain = {
        id_to_name[domain_id]: domain_risk_sums[domain_id]
        / max(1, domain_risk_counts[domain_id])
        for domain_id in sorted(domain_risk_sums)
    }
    if logit_count <= 0:
        raise RuntimeError("training epoch produced no logits")
    logit_mean = logit_sum / logit_count
    logit_variance = max(logit_square_sum / logit_count - logit_mean**2, 0.0)
    sampled_logits = torch.cat(logit_quantile_samples).double()
    logit_quantiles = torch.quantile(
        sampled_logits,
        torch.tensor([0.001, 0.5, 0.99, 0.999], dtype=torch.float64),
    ).tolist()
    parameter_norm = model_parameter_l2_norm(model)
    current_learning_rates = sorted(
        {float(group["lr"]) for group in optimizer.param_groups}
    )
    metrics = {
        "epoch": epoch,
        "steps": steps,
        "elapsed_steps": (epoch + 1) * steps,
        **{key: value / steps for key, value in totals.items()},
        "loss_total": totals["loss"] / steps,
        "loss_seg": totals["loss_sls"] / steps,
        "loss_tail_sep": totals["loss_margin"] / steps,
        "learning_rate": (
            current_learning_rates[0]
            if len(current_learning_rates) == 1
            else current_learning_rates
        ),
        "risk_objective": args.risk_objective,
        "segmentation_loss_implementation": (
            stage1_segmentation_loss_implementation()
        ),
        "stage1_variant": risk_objective_contract(args).get(
            "stage1_variant", "compatibility_baseline"
        ),
        "objective_risk_by_domain": objective_risk_by_domain,
        "logit_mean": logit_mean,
        "logit_std": float(np.sqrt(logit_variance)),
        "logit_min": logit_min,
        "logit_max": logit_max,
        "logit_q001": float(logit_quantiles[0]),
        "logit_q50": float(logit_quantiles[1]),
        "logit_q99": float(logit_quantiles[2]),
        "logit_q999": float(logit_quantiles[3]),
        "logit_quantile_method": "deterministic_even_4096_per_step_sample",
        "max_abs_logit": max(abs(logit_min), abs(logit_max)),
        "nonfinite_count": nonfinite_logit_count,
        "logits_finite": True,
        "gradient_norm_mean": gradient_norm_sum / steps,
        "gradient_norm_max": gradient_norm_max,
        "gradients_finite": gradients_finite,
        "parameters_finite_after_epoch": parameters_finite,
        "parameter_norm": parameter_norm,
        "gradient_norm_stage": "pre_clip_global_l2",
        "risk_gradient_checked": risk_gradient_checked,
        "risk_gradient_norm_first_active_step": risk_gradient_norm,
        "risk_gradients_finite": risk_gradients_finite,
        "risk_weight": risk_weight,
        "effective_lambda_tail": (
            risk_weight * args.lambda_tail if args.risk_objective == "separate" else 0.0
        ),
        "effective_lambda_miss": (
            risk_weight * args.lambda_miss if args.risk_objective == "separate" else 0.0
        ),
        "effective_lambda_margin": (
            risk_weight * args.lambda_margin
            if args.risk_objective
            in DOMAIN_MARGIN_OBJECTIVES | {"legacy-image-margin"}
            else 0.0
        ),
        "domain_cycle_counts": dict(loader.last_cycle_counts),
        "num_empty_target_images_by_domain": {
            id_to_name[domain_id]: empty_target_image_counts[domain_id]
            for domain_id in sorted(empty_target_image_counts)
        },
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "head_training_schedule": "all_auxiliary_and_fused_heads_from_epoch_zero",
        "protocol_scope": protocol_scope(args, loader.domain_names),
    }
    if bool(getattr(args, "engineering_smoke", False)):
        if args.risk_objective != "segmentation-only" and (
            not risk_gradient_checked or risk_gradient_norm <= 0.0
        ):
            raise RuntimeError(
                "engineering smoke did not exercise a non-zero auxiliary-risk gradient"
            )
    if args.risk_objective in MARGIN_DIAGNOSTIC_OBJECTIVES:
        for name in diagnostic_names:
            metrics[f"{name}_by_domain"] = {
                id_to_name[domain_id]: (
                    diagnostic_sums[name][domain_id]
                    / diagnostic_counts[name][domain_id]
                )
                for domain_id in sorted(diagnostic_sums[name])
                if diagnostic_counts[name][domain_id] > 0
            }
        metrics["valid_margin_batch_fraction_by_domain"] = {
            id_to_name[domain_id]: (
                valid_margin_sums[domain_id] / valid_margin_counts[domain_id]
            )
            for domain_id in sorted(valid_margin_sums)
            if valid_margin_counts[domain_id] > 0
        }
        metrics["margin_diagnostics_contract"] = {
            "raw_gap": "target_tail_logit_minus_background_tail_logit",
            "violation": "relu(margin_logit_minus_raw_gap)",
            "aggregation": (
                "domain_tails_before_hinge"
                if args.risk_objective
                in DOMAIN_MARGIN_OBJECTIVES | {"segmentation-only"}
                else "image_hinge_before_domain_mean"
            ),
            "candidate_count": "mean_deterministic_background_candidates_per_image",
            "object_count": "mean_gt_objects_per_batch",
        }
        metrics["margin_risk_by_domain"] = objective_risk_by_domain
        if args.risk_objective == "legacy-image-margin":
            metrics["legacy_image_margin_risk_by_domain"] = objective_risk_by_domain
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
    validate_fold_identity(args)
    args.held_out_domains = sorted(held_out_domains)
    pilot_release_binding = (
        validate_aaai27_pilot_release(args)
        if bool(getattr(args, "aaai27_pilot", False))
        else None
    )
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
    sls_loss = SLSIoULoss(eps=STAGE1_SLS_LOSS_EPS)

    run_execution_fingerprint = execution_fingerprint()
    current_config = {
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
        "segmentation_loss_implementation": (
            stage1_segmentation_loss_implementation()
        ),
        "risk_objective_contract": risk_objective_contract(args),
        "detector_capability_contract": detector_capability_contract(args),
        "protocol_scope": protocol_scope(args, names),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_device_count": torch.cuda.device_count(),
        "command": canonical_training_command(),
        "git": _git_state(),
        "execution_fingerprint": run_execution_fingerprint,
    }
    if pilot_release_binding is not None:
        current_config["pilot_release_binding"] = pilot_release_binding
    if args.resume is None:
        run_dir = create_run_dir(args)
        config = current_config
        write_json(run_dir / "config.json", config)
        run_config_sha256 = sha256_file(run_dir / "config.json")
        run_contract_sha256 = (
            write_aaai27_pilot_run_artifacts(
                run_dir,
                args,
                names,
                detector_source_records,
                run_config_sha256,
                pilot_release_binding,
                run_execution_fingerprint,
            )
            if pilot_release_binding is not None
            else None
        )
        start_epoch = 0
    else:
        run_dir, start_epoch, run_config_sha256, config = load_resume_checkpoint(
            args,
            model,
            optimizer,
            names,
            detector_source_records,
            device,
            pilot_release_binding,
        )
        run_contract_sha256 = (
            sha256_file(run_dir / "run_contract.json")
            if pilot_release_binding is not None
            else None
        )
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
        "resumed": args.resume is not None,
        "resumed_from": (
            str(Path(args.resume).expanduser().resolve())
            if args.resume is not None
            else None
        ),
        "start_epoch": start_epoch,
        "target_total_epochs": args.epochs,
        "aaai27_pilot": bool(getattr(args, "aaai27_pilot", False)),
        "pilot_run_id": getattr(args, "pilot_run_id", None),
        "run_contract_sha256": run_contract_sha256,
    }
    print(json.dumps(startup_summary, indent=2, sort_keys=True))

    metrics_path = run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
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
            run_execution_fingerprint,
            run_contract_sha256,
        )
        print(json.dumps(epoch_metrics, sort_keys=True))

    temporary = run_dir / "weights_last.pt.tmp"
    torch.save(model_state_dict(model), temporary)
    os.replace(temporary, run_dir / "weights_last.pt")


if __name__ == "__main__":
    main()
