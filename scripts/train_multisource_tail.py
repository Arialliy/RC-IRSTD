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
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
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
from data_ext.stage2_role_contract import (
    REPOSITORY_ROOT as STAGE2_REPOSITORY_ROOT,
    load_stage2_selection,
    verify_stage2_run_contract_sidecar,
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
STAGE2_RUNTIME_CONTRACT_SCHEMA = "rc-irstd.stage2-detector-runtime-contract.v1"
STAGE2_RUNTIME_ARTIFACT_TYPE = "rc_irstd_stage2_detector_runtime_contract"
STAGE2_FIXED_LAST_POLICY = "fixed_last_no_test_or_target_validation"
STAGE2_RUNTIME_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "development_only",
        "official_test_accessed",
        "observed_results",
        "run_id",
        "outer_fold_id",
        "outer_target_domain",
        "detector_role",
        "oof_fold_index",
        "base_seed",
        "derived_seed",
        "checkpoint_selection",
        "input_run_contract",
        "run_config",
        "environment_artifact",
        "release_artifact",
        "expected_artifacts",
    }
)
STAGE2_RUNTIME_ARTIFACT_KEYS = frozenset(
    {
        "input_run_contract",
        "run_config",
        "environment_artifact",
        "runtime_contract",
        "release_artifact",
    }
)
STAGE2_EXPECTED_ARTIFACTS = {
    "training_checkpoint": "checkpoint_last.pt",
    "training_checkpoint_sha256": "checkpoint_sha256.txt",
    "restricted_inference_checkpoint": "stage2_inference_checkpoint.pt",
    "restricted_inference_checkpoint_sha256": (
        "stage2_inference_checkpoint.pt.sha256"
    ),
    "metrics": "metrics.jsonl",
    "metrics_sha256": "metrics.jsonl.sha256",
}
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
    stage2_requested = any(
        argument == "--stage2-run-contract"
        or argument.startswith("--stage2-run-contract=")
        for argument in sys.argv[1:]
    )
    parser = argparse.ArgumentParser(
        description="Balanced multi-source MSHNet training with tail-risk losses"
    )
    # Keep the legacy argparse contract byte-semantically unchanged unless the
    # new explicit Stage2 switch is present.  Stage2 obtains sources solely
    # from its verified run contract.
    parser.add_argument("--source-dirs", nargs="+", required=not stage2_requested)
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
        "--stage2-run-contract",
        default=None,
        metavar="JSON",
        help=(
            "Enable the additive Stage2 two-source detector mode. The exact "
            "contract and adjacent SHA sidecar determine every source record, "
            "outer identity, detector role, and runtime seed."
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
    if getattr(args, "stage2_run_contract", None):
        return "stage2_development_detector_official_test_sealed"
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


def bind_stage2_run_contract_to_args(
    args: argparse.Namespace,
    verified_contract: Mapping[str, object],
    *,
    repository_root: str | Path | None = None,
) -> None:
    """Make a verified Stage2 run contract the sole source-role authority.

    User-supplied source/fold arguments are rejected rather than reconciled.
    This prevents a valid contract from being paired with a different runtime
    data role.  The runtime seed must be supplied explicitly/effectively as the
    already-derived value; it is never silently replaced.
    """

    if not getattr(args, "stage2_run_contract", None):
        raise ValueError("Stage2 argument binding requires --stage2-run-contract")
    incompatible = {
        "--source-dirs": getattr(args, "source_dirs", None),
        "--source-split-files": getattr(args, "source_split_files", None),
        "--source-names": getattr(args, "source_names", None),
        "--outer-fold-id": getattr(args, "outer_fold_id", None),
        "--outer-target": getattr(args, "outer_target", None),
        "--held-out-domains": getattr(args, "held_out_domains", None),
    }
    supplied = [name for name, value in incompatible.items() if value is not None]
    if supplied:
        raise ValueError(
            "Stage2 run contract is the sole source/fold authority; remove: "
            + ", ".join(supplied)
        )
    if bool(getattr(args, "allow_single_source_inner_smoke", False)):
        raise ValueError("Stage2 mode forbids single-source smoke semantics")
    if bool(getattr(args, "engineering_smoke", False)):
        raise ValueError("Stage2 contract mode cannot be relabeled engineering smoke")
    if bool(getattr(args, "aaai27_pilot", False)):
        raise ValueError("Stage2 contract mode cannot be combined with Stage1 pilot mode")
    for name in (
        "analysis_plan",
        "pilot_matrix",
        "pilot_run_id",
        "release_tag",
        "source_archive",
        "source_archive_sha256_file",
    ):
        if getattr(args, name, None) is not None:
            raise ValueError(f"Stage1 pilot argument --{name.replace('_', '-')} is forbidden in Stage2 mode")

    derived_seed = verified_contract.get("derived_seed")
    if type(derived_seed) is not int or args.seed != derived_seed:
        raise ValueError(
            "runtime --seed must equal the contract-derived frozen seed "
            f"{derived_seed!r}; got {args.seed!r}"
        )
    training = verified_contract.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("verified Stage2 contract lacks training identity")
    argument_map = {
        "risk_objective": "risk_objective",
        "tail_mode": "tail_mode",
        "lambda_margin": "lambda_margin",
        "target_background_margin": "target_background_margin",
        "tail_q": "tail_q",
        "miss_q": "miss_q",
        "object_pixel_q": "object_pixel_q",
        "tail_gamma": "tail_gamma",
        "peak_kernel_size": "peak_kernel_size",
        "exclusion_radius": "exclusion_radius",
        "peak_min_score": "peak_min_score",
        "plateau_atol": "plateau_atol",
        "warm_epoch": "warm_epoch",
        "risk_warmup_epochs": "risk_warmup_epochs",
        "risk_ramp_epochs": "risk_ramp_epochs",
        "lr": "lr",
    }
    for contract_name, argument_name in argument_map.items():
        if getattr(args, argument_name) != training.get(contract_name):
            raise ValueError(
                f"Stage2 runtime --{argument_name.replace('_', '-')} differs "
                f"from the frozen D3 contract"
            )

    root = (
        STAGE2_REPOSITORY_ROOT
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    expected_role = (
        "detector_oof_train"
        if verified_contract.get("detector_role") == "detector_oof"
        else "detector_full_fit_train"
    )
    selections: list[dict[str, object]] = []
    for binding in verified_contract["selection_contracts"]:
        selection = load_stage2_selection(
            root / str(binding["path"]),
            str(binding["sha256"]),
            expected_role,
            repository_root=root,
        )
        selections.append(selection)
    args.source_dirs = [
        str((root / str(selection["dataset_root"])).resolve())
        for selection in selections
    ]
    args.source_split_files = [
        str((root / str(selection["id_list"]["path"])).resolve())
        for selection in selections
    ]
    args.source_names = list(verified_contract["source_domains"])
    args.outer_fold_id = str(verified_contract["outer_fold_id"])
    args.outer_target = str(verified_contract["outer_target_domain"])
    args.held_out_domains = [args.outer_target]
    args.stage2_detector_role = str(verified_contract["detector_role"])
    args.stage2_oof_fold_index = verified_contract["oof_fold_index"]


def build_source_datasets_from_stage2_contract(
    args: argparse.Namespace,
    verified_contract: Mapping[str, object],
    *,
    repository_root: str | Path | None = None,
) -> Dict[str, IRSTD_Dataset]:
    """Construct exactly two selected datasets without opening a test split.

    Unlike the legacy path, this function never calls
    :func:`audited_source_train_split`.  It passes the SHA-verified selection ID
    file directly to ``IRSTD_Dataset``; no train/test split discovery occurs.
    """

    root = (
        STAGE2_REPOSITORY_ROOT
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    expected_role = (
        "detector_oof_train"
        if verified_contract.get("detector_role") == "detector_oof"
        else "detector_full_fit_train"
    )
    raw_bindings = verified_contract.get("selection_contracts")
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 2:
        raise ValueError("Stage2 dataset construction requires exactly two selections")
    datasets: Dict[str, IRSTD_Dataset] = {}
    for binding in raw_bindings:
        selection = load_stage2_selection(
            root / str(binding["path"]),
            str(binding["sha256"]),
            expected_role,
            repository_root=root,
        )
        name = str(selection["source_domain"])
        dataset_root = (root / str(selection["dataset_root"])).resolve()
        try:
            dataset_root.relative_to(root)
        except ValueError as error:
            raise ValueError("Stage2 dataset root escapes repository") from error
        if not dataset_root.is_dir() or dataset_root.is_symlink():
            raise FileNotFoundError(f"Stage2 dataset root does not exist: {dataset_root}")
        id_path = (root / str(selection["id_list"]["path"])).resolve()
        dataset_args = SimpleNamespace(
            dataset_dir=str(dataset_root),
            base_size=args.base_size,
            crop_size=args.crop_size,
            split_file=str(id_path),
        )
        dataset = IRSTD_Dataset(dataset_args, mode="train")
        expected_ids = [str(record["image_id"]) for record in selection["records"]]
        actual_ids = [sample_id_from_entry(value) for value in dataset.names]
        if actual_ids != expected_ids or len(dataset) != selection["record_count"]:
            raise ValueError("Stage2 dataset order/count differs from selection contract")
        if name in datasets:
            raise ValueError(f"duplicate Stage2 source domain: {name}")
        datasets[name] = dataset
    if list(datasets) != list(verified_contract["source_domains"]):
        raise ValueError("Stage2 dataset order differs from source_domains")
    return datasets


def build_stage2_detector_source_records(
    names: Iterable[str],
    datasets: Dict[str, IRSTD_Dataset],
    verified_contract: Mapping[str, object],
    *,
    repository_root: str | Path | None = None,
) -> List[Dict[str, object]]:
    """Build source provenance solely from selected manifest identities.

    The legacy builder fingerprints every image in a dataset, which would read
    official-test images.  Stage2 instead binds the exact assignment-derived
    records and selection hashes and touches no image or mask here.
    """

    root = (
        STAGE2_REPOSITORY_ROOT
        if repository_root is None
        else Path(repository_root).expanduser().resolve()
    )
    expected_role = (
        "detector_oof_train"
        if verified_contract.get("detector_role") == "detector_oof"
        else "detector_full_fit_train"
    )
    by_domain: dict[str, Mapping[str, object]] = {}
    for binding in verified_contract["selection_contracts"]:
        selection = load_stage2_selection(
            root / str(binding["path"]),
            str(binding["sha256"]),
            expected_role,
            repository_root=root,
        )
        by_domain[str(selection["source_domain"])] = selection
    records: List[Dict[str, object]] = []
    all_image_hashes: set[str] = set()
    for name in names:
        if name not in by_domain or name not in datasets:
            raise ValueError("Stage2 source provenance/domain mismatch")
        selection = by_domain[name]
        selected_ids = [str(item["image_id"]) for item in selection["records"]]
        dataset_ids = [sample_id_from_entry(value) for value in datasets[name].names]
        if selected_ids != dataset_ids:
            raise ValueError("Stage2 source dataset order changed after construction")
        image_hashes = [
            str(item["original_image_sha256"]) for item in selection["records"]
        ]
        overlap = all_image_hashes.intersection(image_hashes)
        if overlap:
            raise ValueError(
                "Stage2 detector sources share original image SHA-256 values: "
                f"collision_count={len(overlap)}"
            )
        all_image_hashes.update(image_hashes)
        binding = next(
            item
            for item in verified_contract["selection_contracts"]
            if item["source_domain"] == name
        )
        records.append(
            {
                "record_schema_version": "rc-irstd.stage2-detector-source-record.v1",
                "source_name": name,
                "dataset_root": selection["dataset_root"],
                "selection_contract": {
                    "path": binding["path"],
                    "sha256": binding["sha256"],
                    "selection_role": binding["selection_role"],
                },
                "id_list": dict(selection["id_list"]),
                "records_content_sha256": selection["records_content_sha256"],
                "ordered_sample_ids": selected_ids,
                "ordered_original_image_sha256": image_hashes,
                "num_samples": len(selected_ids),
                "official_test_accessed": False,
            }
        )
    return records


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


def write_artifact_sha256(path: Path) -> Dict[str, str]:
    """Write an adjacent sha256sum-format sidecar for a regular artifact."""

    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    write_text(sidecar, f"{digest}  {path.name}\n")
    return {"path": path.name, "sha256": digest, "sidecar": sidecar.name}


def _stage2_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _stage2_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical POSIX relative path")
    return value


def _stage2_reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {cursor}")


def _stage2_run_directory(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    _stage2_reject_symlink_components(raw, "Stage2 run directory")
    if not raw.exists() or not stat.S_ISDIR(raw.lstat().st_mode):
        raise FileNotFoundError(f"Stage2 run directory is not a real directory: {raw}")
    return raw.resolve(strict=True)


def _stage2_regular_file(path: Path, label: str) -> Path:
    _stage2_reject_symlink_components(path, label)
    if not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    return path.resolve(strict=True)


def _stage2_run_file(
    run_dir: str | Path,
    declared_path: object,
    expected_name: str,
    label: str,
) -> Path:
    root = _stage2_run_directory(run_dir)
    relative = _stage2_relative_path(declared_path, f"{label}.path")
    if PurePosixPath(relative).parts != (expected_name,):
        raise ValueError(f"{label}.path must be exactly {expected_name!r}")
    resolved = _stage2_regular_file(root / expected_name, label)
    if resolved.parent != root:
        raise ValueError(f"{label} escapes the canonical Stage2 run directory")
    return resolved


def _stage2_repository_file(value: object, label: str) -> tuple[Path, str]:
    relative = _stage2_relative_path(value, f"{label}.path")
    root = STAGE2_REPOSITORY_ROOT.resolve(strict=True)
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    resolved = _stage2_regular_file(candidate, label)
    try:
        replay = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    if replay != relative:
        raise ValueError(f"{label}.path is not canonical after resolution")
    return resolved, relative


def _stage2_stable_text(path: Path, label: str) -> str:
    before = sha256_file(path)
    payload = path.read_text(encoding="utf-8")
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"{label} changed while read: {path}")
    return payload


def _verify_stage2_sha256sum_sidecar(
    artifact: Path,
    sidecar: Path,
    label: str,
) -> str:
    artifact = _stage2_regular_file(artifact, label)
    sidecar = _stage2_regular_file(sidecar, f"{label} SHA-256 sidecar")
    before = sha256_file(artifact)
    sidecar_text = _stage2_stable_text(sidecar, f"{label} SHA-256 sidecar")
    after = sha256_file(artifact)
    if before != after:
        raise RuntimeError(f"{label} changed while its sidecar was read")
    if sidecar_text != f"{before}  {artifact.name}\n":
        raise ValueError(f"{label} SHA-256 sidecar mismatch")
    return before


def _verify_stage2_adjacent_sidecar(
    run_dir: str | Path,
    artifact_name: str,
    label: str,
) -> Dict[str, str]:
    artifact = _stage2_run_file(run_dir, artifact_name, artifact_name, label)
    sidecar_name = artifact_name + ".sha256"
    sidecar = _stage2_run_file(
        run_dir,
        sidecar_name,
        sidecar_name,
        f"{label} SHA-256 sidecar",
    )
    digest = _verify_stage2_sha256sum_sidecar(artifact, sidecar, label)
    return {"path": artifact_name, "sha256": digest, "sidecar": sidecar_name}


def _stage2_exact_binding(
    value: object,
    label: str,
    *,
    expected_path: str,
    expected_sidecar: str | None,
) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    expected_keys = {"path", "sha256"}
    if expected_sidecar is not None:
        expected_keys.add("sidecar")
    if set(value) != expected_keys:
        raise ValueError(f"{label} keys mismatch")
    if value.get("path") != expected_path:
        raise ValueError(f"{label}.path mismatch")
    if expected_sidecar is not None and value.get("sidecar") != expected_sidecar:
        raise ValueError(f"{label}.sidecar mismatch")
    return {
        "path": expected_path,
        "sha256": _stage2_sha256(value.get("sha256"), f"{label}.sha256"),
        **({"sidecar": expected_sidecar} if expected_sidecar is not None else {}),
    }


def _stage2_input_binding(
    input_run_contract_path: str | Path,
    input_run_contract_sha256: str,
) -> Dict[str, str]:
    raw = Path(input_run_contract_path).expanduser()
    resolved = _stage2_regular_file(raw, "Stage2 input run contract")
    root = STAGE2_REPOSITORY_ROOT.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Stage2 input run contract is outside repository") from error
    declared_sha = _stage2_sha256(
        input_run_contract_sha256, "Stage2 input run-contract SHA-256"
    )
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    actual_sha = _verify_stage2_sha256sum_sidecar(
        resolved, sidecar, "Stage2 input run contract"
    )
    if actual_sha != declared_sha:
        raise ValueError("Stage2 input run-contract SHA-256 mismatch")
    return {"path": relative, "sha256": declared_sha}


def _verify_stage2_release_binding(
    value: object,
    expected: object,
) -> Dict[str, object]:
    if not isinstance(value, Mapping) or not isinstance(expected, Mapping):
        raise TypeError("Stage2 release_artifact must be an object")
    if dict(value) != dict(expected):
        raise ValueError("Stage2 runtime release_artifact binding mismatch")
    release_path, _ = _stage2_repository_file(value.get("path"), "release artifact")
    if sha256_file(release_path) != _stage2_sha256(
        value.get("sha256"), "release_artifact.sha256"
    ):
        raise ValueError("Stage2 release artifact SHA-256 mismatch")
    archive = value.get("source_archive")
    if not isinstance(archive, Mapping) or set(archive) != {"path", "sha256"}:
        raise ValueError("Stage2 release source_archive binding mismatch")
    archive_path, _ = _stage2_repository_file(
        archive.get("path"), "release source archive"
    )
    if sha256_file(archive_path) != _stage2_sha256(
        archive.get("sha256"), "release source_archive.sha256"
    ):
        raise ValueError("Stage2 release source archive SHA-256 mismatch")
    return dict(value)


def write_stage2_runtime_artifacts(
    run_dir: Path,
    args: argparse.Namespace,
    verified_contract: Mapping[str, object],
    input_run_contract_sha256: str,
    run_config_sha256: str,
    execution_fingerprint_payload: Mapping[str, object],
) -> Dict[str, object]:
    """Persist W02 runtime/environment bindings without circular hashes."""

    if not getattr(args, "stage2_run_contract", None):
        raise ValueError("Stage2 runtime artifacts require --stage2-run-contract")
    root = _stage2_run_directory(run_dir)
    input_binding = _stage2_input_binding(
        str(args.stage2_run_contract), input_run_contract_sha256
    )
    config_binding = _verify_stage2_adjacent_sidecar(
        root, "config.json", "Stage2 run config"
    )
    if config_binding["sha256"] != _stage2_sha256(
        run_config_sha256, "Stage2 run-config SHA-256"
    ):
        raise ValueError("Stage2 run-config SHA-256 mismatch")
    environment_path = root / "environment.json"
    write_json(environment_path, dict(execution_fingerprint_payload))
    environment_artifact = write_artifact_sha256(environment_path)
    release_artifact = verified_contract.get("bindings", {}).get("release_artifact")
    if not isinstance(release_artifact, Mapping):
        raise ValueError("Stage2 run contract lacks release_artifact binding")
    runtime_contract: Dict[str, object] = {
        "schema_version": STAGE2_RUNTIME_CONTRACT_SCHEMA,
        "artifact_type": STAGE2_RUNTIME_ARTIFACT_TYPE,
        "development_only": True,
        "official_test_accessed": False,
        "observed_results": None,
        "run_id": verified_contract["run_id"],
        "outer_fold_id": verified_contract["outer_fold_id"],
        "outer_target_domain": verified_contract["outer_target_domain"],
        "detector_role": verified_contract["detector_role"],
        "oof_fold_index": verified_contract["oof_fold_index"],
        "base_seed": verified_contract["base_seed"],
        "derived_seed": verified_contract["derived_seed"],
        "checkpoint_selection": STAGE2_FIXED_LAST_POLICY,
        "input_run_contract": input_binding,
        "run_config": config_binding,
        "environment_artifact": environment_artifact,
        "release_artifact": dict(release_artifact),
        "expected_artifacts": dict(STAGE2_EXPECTED_ARTIFACTS),
    }
    runtime_path = root / "stage2_runtime_contract.json"
    write_json(runtime_path, runtime_contract)
    write_artifact_sha256(runtime_path)
    return verify_stage2_runtime_artifacts(
        root,
        verified_contract,
        input_run_contract_sha256,
        run_config_sha256,
        execution_fingerprint_payload,
        input_run_contract_path=str(args.stage2_run_contract),
    )


def verify_stage2_runtime_artifacts(
    run_dir: Path,
    verified_contract: Mapping[str, object],
    input_run_contract_sha256: str,
    run_config_sha256: str,
    execution_fingerprint_payload: Mapping[str, object],
    *,
    input_run_contract_path: str | Path,
) -> Dict[str, object]:
    """Replay existing runtime artifacts before a Stage2 fixed-last resume."""

    root = _stage2_run_directory(run_dir)
    input_binding = _stage2_input_binding(
        input_run_contract_path, input_run_contract_sha256
    )
    config_binding = _verify_stage2_adjacent_sidecar(
        root, "config.json", "Stage2 run config"
    )
    if config_binding["sha256"] != _stage2_sha256(
        run_config_sha256, "Stage2 run-config SHA-256"
    ):
        raise ValueError("Stage2 run-config SHA-256 mismatch")
    runtime_binding = _verify_stage2_adjacent_sidecar(
        root, "stage2_runtime_contract.json", "Stage2 runtime contract"
    )
    runtime_path = _stage2_run_file(
        root,
        runtime_binding["path"],
        "stage2_runtime_contract.json",
        "Stage2 runtime contract",
    )
    runtime = _load_json_object(runtime_path, "Stage2 runtime contract")
    if set(runtime) != STAGE2_RUNTIME_CONTRACT_KEYS:
        raise ValueError("Stage2 runtime-contract keys mismatch")
    if runtime["schema_version"] != STAGE2_RUNTIME_CONTRACT_SCHEMA:
        raise ValueError("Stage2 runtime-contract schema mismatch")
    if runtime["artifact_type"] != STAGE2_RUNTIME_ARTIFACT_TYPE:
        raise ValueError("Stage2 runtime-contract artifact_type mismatch")
    for key, expected_bool in (
        ("development_only", True),
        ("official_test_accessed", False),
    ):
        if type(runtime[key]) is not bool or runtime[key] is not expected_bool:
            raise TypeError(f"Stage2 runtime-contract {key} must be exact {expected_bool}")
    if runtime["observed_results"] is not None:
        raise ValueError("Stage2 runtime-contract observed_results must be null")
    if runtime["checkpoint_selection"] != STAGE2_FIXED_LAST_POLICY:
        raise ValueError("Stage2 runtime contract is not fixed-last")
    if runtime["expected_artifacts"] != STAGE2_EXPECTED_ARTIFACTS:
        raise ValueError("Stage2 runtime expected_artifacts mismatch")
    declared_input = _stage2_exact_binding(
        runtime["input_run_contract"],
        "Stage2 runtime input_run_contract",
        expected_path=input_binding["path"],
        expected_sidecar=None,
    )
    if declared_input != input_binding:
        raise ValueError("Stage2 runtime input-run binding mismatch")
    declared_config = _stage2_exact_binding(
        runtime["run_config"],
        "Stage2 runtime run_config",
        expected_path="config.json",
        expected_sidecar="config.json.sha256",
    )
    if declared_config != config_binding:
        raise ValueError("Stage2 runtime config binding mismatch")
    environment_binding = _verify_stage2_adjacent_sidecar(
        root, "environment.json", "Stage2 environment artifact"
    )
    declared_environment = _stage2_exact_binding(
        runtime["environment_artifact"],
        "Stage2 runtime environment_artifact",
        expected_path="environment.json",
        expected_sidecar="environment.json.sha256",
    )
    if declared_environment != environment_binding:
        raise ValueError("Stage2 runtime environment binding mismatch")
    environment_path = _stage2_run_file(
        root,
        environment_binding["path"],
        "environment.json",
        "Stage2 environment artifact",
    )
    environment = _load_json_object(environment_path, "Stage2 environment")
    if environment != execution_fingerprint_payload:
        raise ValueError("Stage2 resume environment/source fingerprint mismatch")
    expected = {
        "run_id": verified_contract["run_id"],
        "outer_fold_id": verified_contract["outer_fold_id"],
        "outer_target_domain": verified_contract["outer_target_domain"],
        "detector_role": verified_contract["detector_role"],
        "oof_fold_index": verified_contract["oof_fold_index"],
        "base_seed": verified_contract["base_seed"],
        "derived_seed": verified_contract["derived_seed"],
    }
    for key, value in expected.items():
        if runtime[key] != value:
            raise ValueError(f"Stage2 runtime identity mismatch: {key}")
    bindings = verified_contract.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("verified Stage2 run contract lacks bindings")
    release_artifact = _verify_stage2_release_binding(
        runtime["release_artifact"], bindings.get("release_artifact")
    )
    return {
        "input_run_contract": input_binding,
        "run_config": config_binding,
        "environment_artifact": environment_binding,
        "runtime_contract": runtime_binding,
        "release_artifact": release_artifact,
    }


def verify_stage2_checkpoint_provenance(
    checkpoint: Mapping[str, object],
    run_dir: str | Path,
    verified_contract: Mapping[str, object],
    input_run_contract_sha256: str,
    run_config_sha256: str,
    execution_fingerprint_payload: Mapping[str, object],
    *,
    input_run_contract_path: str | Path,
    expected_runtime_artifacts: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Replay the complete Stage2 checkpoint -> runtime -> disk closure."""

    runtime_artifacts = verify_stage2_runtime_artifacts(
        run_dir,
        verified_contract,
        input_run_contract_sha256,
        run_config_sha256,
        execution_fingerprint_payload,
        input_run_contract_path=input_run_contract_path,
    )
    raw_runtime = checkpoint.get("stage2_runtime_artifacts")
    if not isinstance(raw_runtime, Mapping):
        raise ValueError("Stage2 checkpoint lacks runtime artifact bindings")
    if set(raw_runtime) != STAGE2_RUNTIME_ARTIFACT_KEYS:
        raise ValueError("Stage2 checkpoint runtime-artifact keys mismatch")
    if any(not isinstance(raw_runtime[key], Mapping) for key in raw_runtime):
        raise TypeError("Stage2 checkpoint runtime-artifact bindings must be objects")
    if dict(raw_runtime) != runtime_artifacts:
        raise ValueError("Stage2 checkpoint/runtime/disk provenance divergence")
    if expected_runtime_artifacts is not None:
        if (
            not isinstance(expected_runtime_artifacts, Mapping)
            or set(expected_runtime_artifacts) != STAGE2_RUNTIME_ARTIFACT_KEYS
            or dict(expected_runtime_artifacts) != runtime_artifacts
        ):
            raise ValueError("Stage2 expected runtime-artifact closure mismatch")
    if checkpoint.get("run_contract_sha256") != _stage2_sha256(
        input_run_contract_sha256, "Stage2 input run-contract SHA-256"
    ):
        raise ValueError("Stage2 checkpoint run-contract SHA-256 mismatch")
    if checkpoint.get("run_config_sha256") != _stage2_sha256(
        run_config_sha256, "Stage2 run-config SHA-256"
    ):
        raise ValueError("Stage2 checkpoint run-config SHA-256 mismatch")
    if checkpoint.get("checkpoint_selection") != STAGE2_FIXED_LAST_POLICY:
        raise ValueError("Stage2 checkpoint is not fixed-last")
    if type(checkpoint.get("official_test_accessed")) is not bool or checkpoint.get(
        "official_test_accessed"
    ) is not False:
        raise TypeError("Stage2 checkpoint official_test_accessed must be exact false")
    identity = {
        "outer_fold_id": verified_contract["outer_fold_id"],
        "outer_target": verified_contract["outer_target_domain"],
        "detector_role": verified_contract["detector_role"],
        "oof_fold_index": verified_contract["oof_fold_index"],
    }
    for key, expected in identity.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"Stage2 checkpoint identity mismatch: {key}")
    return runtime_artifacts


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
    before = sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"{label} changed while read: {path}")
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


def serialised_training_args(args: argparse.Namespace) -> Dict[str, object]:
    """Hide the additive Stage2 None default from every legacy artifact."""

    values = dict(vars(args))
    if values.get("stage2_run_contract") is None:
        values.pop("stage2_run_contract", None)
    return values


def resume_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Return the immutable CLI contract that must survive continuation."""

    values = {
        key: value
        for key, value in serialised_training_args(args).items()
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
    before = sha256_file(path)
    text = path.read_text(encoding="utf-8")
    after = sha256_file(path)
    if before != after:
        raise RuntimeError(f"resume metrics.jsonl changed while read: {path}")
    lines = [line for line in text.splitlines() if line.strip()]
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
    stage2_run_contract_sha256: str | None = None,
    stage2_verified_contract: Mapping[str, object] | None = None,
    stage2_execution_fingerprint: Mapping[str, object] | None = None,
    return_stage2_runtime_artifacts: bool = False,
) -> tuple:
    """Fail closed unless a fixed-last artifact matches the current run exactly."""

    pilot_mode = bool(getattr(args, "aaai27_pilot", False))
    stage2_mode = bool(getattr(args, "stage2_run_contract", None))
    raw_checkpoint_path = Path(args.resume).expanduser()
    if stage2_mode:
        run_dir = _stage2_run_directory(raw_checkpoint_path.parent)
        checkpoint_path = _stage2_run_file(
            run_dir,
            raw_checkpoint_path.name,
            "checkpoint_last.pt",
            "Stage2 training checkpoint",
        )
        checkpoint_sidecar = _stage2_run_file(
            run_dir,
            "checkpoint_sha256.txt",
            "checkpoint_sha256.txt",
            "Stage2 training checkpoint SHA-256 sidecar",
        )
        _verify_stage2_sha256sum_sidecar(
            checkpoint_path,
            checkpoint_sidecar,
            "Stage2 training checkpoint",
        )
        config_binding = _verify_stage2_adjacent_sidecar(
            run_dir, "config.json", "Stage2 run config"
        )
        config_path = _stage2_run_file(
            run_dir, "config.json", "config.json", "Stage2 run config"
        )
        config = _load_json_object(config_path, "Stage2 run config")
        run_config_sha256 = config_binding["sha256"]
    else:
        checkpoint_path = raw_checkpoint_path.resolve()
        run_dir = checkpoint_path.parent
        verify_checkpoint_sha256(checkpoint_path, required=pilot_mode)
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"resume run is missing config.json: {config_path}")
        config = _load_json_object(config_path, "resume config")
        run_config_sha256 = sha256_file(config_path)
    expected_segmentation_loss = stage1_segmentation_loss_implementation()
    if config.get("segmentation_loss_implementation") != expected_segmentation_loss:
        raise ValueError("resume config segmentation-loss implementation mismatch")
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

    stage2_runtime_artifacts: Dict[str, object] | None = None
    current_execution_fingerprint = (
        dict(stage2_execution_fingerprint)
        if stage2_execution_fingerprint is not None
        else execution_fingerprint()
    )
    if stage2_mode:
        if stage2_verified_contract is None:
            raise ValueError("Stage2 resume requires the verified input run contract")
        if re.fullmatch(r"[0-9a-f]{64}", str(stage2_run_contract_sha256)) is None:
            raise ValueError("Stage2 resume requires a verified run-contract SHA-256")
        stage2_runtime_artifacts = verify_stage2_runtime_artifacts(
            run_dir,
            stage2_verified_contract,
            str(stage2_run_contract_sha256),
            run_config_sha256,
            current_execution_fingerprint,
            input_run_contract_path=str(args.stage2_run_contract),
        )

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
    if checkpoint["execution_fingerprint"] != current_execution_fingerprint:
        raise ValueError(
            "resume source-tree/runtime execution fingerprint mismatch"
        )
    if pilot_mode and checkpoint.get("run_contract_sha256") != run_contract_sha256:
        raise ValueError("pilot checkpoint run-contract SHA-256 mismatch")
    if stage2_mode:
        assert stage2_verified_contract is not None
        assert stage2_runtime_artifacts is not None
        replayed_runtime = verify_stage2_checkpoint_provenance(
            checkpoint,
            run_dir,
            stage2_verified_contract,
            str(stage2_run_contract_sha256),
            run_config_sha256,
            current_execution_fingerprint,
            input_run_contract_path=str(args.stage2_run_contract),
            expected_runtime_artifacts=stage2_runtime_artifacts,
        )
        if replayed_runtime != stage2_runtime_artifacts:
            raise RuntimeError("Stage2 runtime closure changed during resume preflight")

    completed_epoch = int(checkpoint["epoch"])
    if completed_epoch < 0:
        raise ValueError("resume checkpoint epoch must be non-negative")
    if stage2_mode:
        _verify_stage2_adjacent_sidecar(
            run_dir, "metrics.jsonl", "Stage2 metrics artifact"
        )
        metrics_path = _stage2_run_file(
            run_dir, "metrics.jsonl", "metrics.jsonl", "Stage2 metrics artifact"
        )
    else:
        metrics_path = run_dir / "metrics.jsonl"
    if _last_metrics_epoch(metrics_path) != completed_epoch:
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
    result = (run_dir, start_epoch, run_config_sha256, config)
    if return_stage2_runtime_artifacts:
        if not stage2_mode or stage2_runtime_artifacts is None:
            raise ValueError("Stage2 runtime artifacts were requested outside Stage2 mode")
        return (*result, stage2_runtime_artifacts)
    return result


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
    stage2_runtime_artifacts: Mapping[str, object] | None = None,
    stage2_verified_contract: Mapping[str, object] | None = None,
) -> None:
    checkpoint_execution_fingerprint = dict(
        execution_fingerprint_payload or execution_fingerprint()
    )
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
        "training_args": serialised_training_args(args),
        "resume_contract": resume_contract(args),
        "rng_state": _rng_state(),
        "run_config_sha256": run_config_sha256,
        "execution_fingerprint": checkpoint_execution_fingerprint,
    }
    if run_contract_sha256 is not None:
        if re.fullmatch(r"[0-9a-f]{64}", run_contract_sha256) is None:
            raise ValueError("run_contract_sha256 must be a lowercase SHA-256 digest")
        payload["run_contract_sha256"] = run_contract_sha256
    if stage2_runtime_artifacts is not None:
        if stage2_verified_contract is None:
            raise ValueError("Stage2 checkpoint requires the verified run contract")
        if run_contract_sha256 is None:
            raise ValueError("Stage2 checkpoint requires the input run-contract SHA-256")
        payload.update(
            {
                "detector_role": args.stage2_detector_role,
                "oof_fold_index": args.stage2_oof_fold_index,
                "official_test_accessed": False,
                "stage2_runtime_artifacts": dict(stage2_runtime_artifacts),
            }
        )
        verified_runtime = verify_stage2_checkpoint_provenance(
            payload,
            run_dir,
            stage2_verified_contract,
            run_contract_sha256,
            run_config_sha256,
            checkpoint_execution_fingerprint,
            input_run_contract_path=str(args.stage2_run_contract),
            expected_runtime_artifacts=stage2_runtime_artifacts,
        )
        payload["stage2_runtime_artifacts"] = verified_runtime
    temporary = run_dir / "checkpoint_last.pt.tmp"
    torch.save(payload, temporary)
    checkpoint_path = run_dir / "checkpoint_last.pt"
    os.replace(temporary, checkpoint_path)
    write_checkpoint_sha256(checkpoint_path)


def save_stage2_inference_checkpoint(
    run_dir: Path,
    model: nn.Module,
    epoch: int,
    args: argparse.Namespace,
    names: List[str],
    detector_source_records: List[Dict[str, object]],
    run_config_sha256: str,
    input_run_contract_sha256: str,
    stage2_runtime_artifacts: Mapping[str, object],
    stage2_verified_contract: Mapping[str, object],
    execution_fingerprint_payload: Mapping[str, object],
) -> Dict[str, str]:
    """Write a tensors/primitives-only checkpoint for restricted consumers."""

    if not getattr(args, "stage2_run_contract", None):
        raise ValueError("restricted Stage2 checkpoint requires contract mode")
    root = _stage2_run_directory(run_dir)
    for label, digest in (
        ("run config", run_config_sha256),
        ("input run contract", input_run_contract_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise ValueError(f"invalid {label} SHA-256")
    verified_runtime = verify_stage2_runtime_artifacts(
        root,
        stage2_verified_contract,
        input_run_contract_sha256,
        run_config_sha256,
        execution_fingerprint_payload,
        input_run_contract_path=str(args.stage2_run_contract),
    )
    if dict(stage2_runtime_artifacts) != verified_runtime:
        raise ValueError("restricted Stage2 checkpoint runtime closure mismatch")
    # A plain dict of tensors plus JSON primitives is accepted by the fixed
    # PyTorch 2.9 restricted unpickler.  Optimizer and RNG objects intentionally
    # remain only in checkpoint_last.pt and are never needed by W03 inference.
    payload: Dict[str, object] = {
        "format_version": "rc-irstd.detector-inference.v1",
        "state_dict": {
            str(key): value.detach().cpu()
            for key, value in model_state_dict(model).items()
        },
        "epoch": int(epoch),
        "seed": int(args.seed),
        "source_names": list(names),
        "detector_source_records": detector_source_records,
        "outer_fold_id": args.outer_fold_id,
        "outer_target": args.outer_target,
        "held_out_domains": sorted(set(args.held_out_domains or [])),
        "detector_role": args.stage2_detector_role,
        "oof_fold_index": args.stage2_oof_fold_index,
        "checkpoint_selection": "fixed_last_no_test_or_target_validation",
        "risk_objective_contract": risk_objective_contract(args),
        "segmentation_loss_implementation": stage1_segmentation_loss_implementation(),
        "run_config_sha256": run_config_sha256,
        "run_contract_sha256": input_run_contract_sha256,
        "stage2_runtime_artifacts": verified_runtime,
        "training_args": {
            "base_size": int(args.base_size),
            "crop_size": int(args.crop_size),
            "resize_mode": "resize",
        },
        "inference_geometry": {
            "input_hw": [int(args.base_size), int(args.base_size)],
            "resize_mode": "resize",
        },
        "official_test_accessed": False,
    }
    temporary = root / "stage2_inference_checkpoint.pt.tmp"
    torch.save(payload, temporary)
    checkpoint = root / "stage2_inference_checkpoint.pt"
    os.replace(temporary, checkpoint)
    binding = write_artifact_sha256(checkpoint)
    # Test the actual production bytes through the restricted loader before
    # publishing the binding to downstream exporters.
    published_binding = _verify_stage2_adjacent_sidecar(
        root,
        "stage2_inference_checkpoint.pt",
        "restricted Stage2 inference checkpoint",
    )
    if binding != published_binding:
        raise RuntimeError("restricted Stage2 checkpoint binding changed after publish")
    replay = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(replay, Mapping) or replay.get("format_version") != payload["format_version"]:
        raise RuntimeError("restricted Stage2 inference-checkpoint replay failed")
    if replay.get("run_contract_sha256") != input_run_contract_sha256:
        raise RuntimeError("restricted Stage2 checkpoint provenance mismatch")
    replayed_runtime = verify_stage2_checkpoint_provenance(
        replay,
        root,
        stage2_verified_contract,
        input_run_contract_sha256,
        run_config_sha256,
        execution_fingerprint_payload,
        input_run_contract_path=str(args.stage2_run_contract),
        expected_runtime_artifacts=verified_runtime,
    )
    if replayed_runtime != verified_runtime:
        raise RuntimeError("restricted Stage2 checkpoint provenance replay drift")
    return binding


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
    risk_gradient_probe_count = 0
    risk_gradient_positive_found = False
    risk_gradient_norm_first_positive_step = 0.0
    risk_gradient_first_positive_step: int | None = None

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
        risk_gradient_enabled = (
            risk_weight > 0.0
            and args.risk_objective != "segmentation-only"
        )
        # Preserve the historical first-active-step diagnostic, but do not
        # mistake a hinge-satisfied first batch for evidence that the D3 path
        # has no gradient.  After a zero first probe, inspect only subsequent
        # batches with a strictly positive auxiliary objective until the first
        # strictly positive isolated risk gradient is observed.  autograd.grad
        # does not populate parameter .grad buffers and therefore cannot alter
        # the optimizer update performed below.
        should_probe_risk_gradient = risk_gradient_enabled and (
            not risk_gradient_checked
            or (
                not risk_gradient_positive_found
                and float(risk_auxiliary_loss.detach()) > 0.0
            )
        )
        if should_probe_risk_gradient:
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
            probed_norm = float(risk_gradient_norm_tensor)
            risk_gradient_probe_count += 1
            if not risk_gradient_checked:
                risk_gradient_norm = probed_norm
                risk_gradient_checked = True
            if not risk_gradient_positive_found and probed_norm > 0.0:
                risk_gradient_positive_found = True
                risk_gradient_norm_first_positive_step = probed_norm
                risk_gradient_first_positive_step = step
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
        "risk_gradient_probe_count": risk_gradient_probe_count,
        "risk_gradient_positive_found": risk_gradient_positive_found,
        "risk_gradient_norm_first_positive_step": (
            risk_gradient_norm_first_positive_step
        ),
        "risk_gradient_first_positive_step": risk_gradient_first_positive_step,
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
            not risk_gradient_checked
            or not risk_gradient_positive_found
            or risk_gradient_norm_first_positive_step <= 0.0
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
    stage2_verified_contract: Mapping[str, object] | None = None
    stage2_run_contract_sha256: str | None = None
    if getattr(args, "stage2_run_contract", None):
        stage2_verified_contract, stage2_run_contract_sha256 = (
            verify_stage2_run_contract_sidecar(args.stage2_run_contract)
        )
        bind_stage2_run_contract_to_args(args, stage2_verified_contract)
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

    if stage2_verified_contract is None:
        datasets = build_source_datasets(args, names)
        detector_source_records = build_detector_source_records(names, datasets)
    else:
        datasets = build_source_datasets_from_stage2_contract(
            args, stage2_verified_contract
        )
        detector_source_records = build_stage2_detector_source_records(
            names, datasets, stage2_verified_contract
        )
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
        **serialised_training_args(args),
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
    if stage2_verified_contract is not None:
        current_config["stage2_input_run_contract"] = {
            "path": Path(str(args.stage2_run_contract))
            .expanduser()
            .resolve()
            .relative_to(STAGE2_REPOSITORY_ROOT)
            .as_posix(),
            "sha256": stage2_run_contract_sha256,
            "schema_version": stage2_verified_contract["schema_version"],
            "run_id": stage2_verified_contract["run_id"],
            "bindings": stage2_verified_contract["bindings"],
            "official_test_accessed": False,
        }
    if args.resume is None:
        run_dir = create_run_dir(args)
        config = current_config
        write_json(run_dir / "config.json", config)
        run_config_sha256 = sha256_file(run_dir / "config.json")
        if stage2_verified_contract is not None:
            write_artifact_sha256(run_dir / "config.json")
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
            else stage2_run_contract_sha256
        )
        stage2_runtime_artifacts = (
            write_stage2_runtime_artifacts(
                run_dir,
                args,
                stage2_verified_contract,
                str(stage2_run_contract_sha256),
                run_config_sha256,
                run_execution_fingerprint,
            )
            if stage2_verified_contract is not None
            else None
        )
        start_epoch = 0
    else:
        resume_result = load_resume_checkpoint(
            args,
            model,
            optimizer,
            names,
            detector_source_records,
            device,
            pilot_release_binding,
            stage2_run_contract_sha256,
            stage2_verified_contract,
            run_execution_fingerprint,
            stage2_verified_contract is not None,
        )
        if stage2_verified_contract is not None:
            (
                run_dir,
                start_epoch,
                run_config_sha256,
                config,
                stage2_runtime_artifacts,
            ) = resume_result
        else:
            run_dir, start_epoch, run_config_sha256, config = resume_result
            stage2_runtime_artifacts = None
        run_contract_sha256 = (
            sha256_file(run_dir / "run_contract.json")
            if pilot_release_binding is not None
            else stage2_run_contract_sha256
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
    if stage2_verified_contract is not None:
        startup_summary.update(
            {
                "stage2_run_id": stage2_verified_contract["run_id"],
                "stage2_detector_role": stage2_verified_contract["detector_role"],
                "official_test_accessed": False,
            }
        )
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
        if stage2_verified_contract is not None:
            write_artifact_sha256(metrics_path)
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
            stage2_runtime_artifacts,
            stage2_verified_contract,
        )
        if stage2_verified_contract is not None:
            assert stage2_runtime_artifacts is not None
            save_stage2_inference_checkpoint(
                run_dir,
                model,
                epoch,
                args,
                names,
                detector_source_records,
                run_config_sha256,
                str(stage2_run_contract_sha256),
                stage2_runtime_artifacts,
                stage2_verified_contract,
                run_execution_fingerprint,
            )
        print(json.dumps(epoch_metrics, sort_keys=True))

    temporary = run_dir / "weights_last.pt.tmp"
    torch.save(model_state_dict(model), temporary)
    os.replace(temporary, run_dir / "weights_last.pt")


if __name__ == "__main__":
    main()
