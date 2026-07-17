"""Train the frozen T6/T7/T8 Stage-2 cross-fit calibrators.

This runner is intentionally fail closed.  Public entry points consume only
Lane-A verified collection bundles, externally hash-bound configuration and
seed manifests, and immutable committed resume generations.  A raw JSONL path
is never promoted to training data inside this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from losses.calibrator_risk import curve_query_risk_aligned_calibrator_loss
from model.direct_no_reject_pixel_calibrator import (
    DIRECT_NO_REJECT_MODEL_ID,
    DirectNoRejectPixelCalibrator,
)
from model.monotone_pixel_calibrator import MonotoneNoRejectPixelRiskCalibrator


CONFIG_SCHEMA = "rc-irstd.aaai27-stage2-crossfit-config.v2"
CHECKPOINT_SCHEMA = "rc-irstd.calibrator.v6"
GENERATION_COMMIT_SCHEMA = "rc-irstd.calibrator-generation-commit.v1"
RUN_COMMIT_SCHEMA = "rc-irstd.calibrator-run-commit.v1"
SEED_MANIFEST_SCHEMA = "rc-irstd.stage2-seed-derivation-manifest.v1"
SEED_DOMAIN_TAG = "rc-irstd.stage2.seed.v1"
METHODS = ("T6", "T7", "T8")
OUTER_TARGETS = {
    "outer_leave_nuaa_sirst": "NUAA-SIRST",
    "outer_leave_nudt_sirst": "NUDT-SIRST",
    "outer_leave_irstd_1k": "IRSTD-1K",
}
METHOD_SEED_ROLES = {
    "T6": "baseline_t6_direct_mlp::not_applicable",
    "T7": "baseline_t7_monotone_oracle::not_applicable",
    "T8": "stage2_calibrator_t8::not_applicable",
}
EXPECTED_PARAMETER_COUNTS = {"T6": 3107, "T7": 3140, "T8": 3140}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class Stage2CalibratorContractError(ValueError):
    """Raised when a frozen W08 input or artifact fails closed."""


def _validate_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise Stage2CalibratorContractError(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage2CalibratorContractError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise Stage2CalibratorContractError(f"non-finite JSON value is forbidden: {value}")


def _load_json_bytes(data: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage2CalibratorContractError(f"invalid {name} JSON: {error}") from error
    if type(payload) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    return payload


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2CalibratorContractError(
            f"value is not canonical finite JSON: {error}"
        ) from error


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2CalibratorContractError(
            f"value is not finite JSON: {error}"
        ) from error


def _assert_exact_keys(
    value: Any, expected: Iterable[str], *, name: str
) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact JSON object")
    wanted = frozenset(expected)
    observed = frozenset(value)
    if observed != wanted:
        raise Stage2CalibratorContractError(
            f"{name} keys differ; missing={sorted(wanted-observed)}, "
            f"extra={sorted(observed-wanted)}"
        )
    return value


def _exact(value: Any, expected: Any, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise Stage2CalibratorContractError(f"{name} must equal {expected!r}")


def _verified_regular_file(path: str | Path, expected_sha256: str, name: str) -> bytes:
    expected = _validate_sha256(expected_sha256, f"{name}_sha256")
    raw = Path(path).expanduser()
    if not raw.is_absolute() or ".." in raw.parts or raw.is_symlink():
        raise Stage2CalibratorContractError(
            f"{name} must be an absolute canonical non-symlink path"
        )
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as error:
        raise Stage2CalibratorContractError(f"{name} does not exist") from error
    if resolved != raw or not raw.is_file():
        raise Stage2CalibratorContractError(
            f"{name} must be an absolute canonical regular file"
        )
    descriptor = os.open(raw, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            data = handle.read()
        after = os.fstat(descriptor)
        path_after = os.stat(raw, follow_symlinks=False)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(before) != identity(path_after):
            raise Stage2CalibratorContractError(f"{name} changed while read")
        if sha256_bytes(data) != expected:
            raise Stage2CalibratorContractError(f"{name} SHA-256 mismatch")
        return data
    finally:
        os.close(descriptor)


@dataclass(frozen=True, init=False)
class VerifiedStage2CrossfitConfig:
    path: Path
    sha256: str
    canonical_payload: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "VerifiedStage2CrossfitConfig can only be created by "
            "verify_stage2_crossfit_config"
        )

    @property
    def payload(self) -> dict[str, Any]:
        return _load_json_bytes(self.canonical_payload, "verified config")


def _make_verified_config(
    path: Path, sha256: str, payload: Mapping[str, Any]
) -> VerifiedStage2CrossfitConfig:
    result = object.__new__(VerifiedStage2CrossfitConfig)
    object.__setattr__(result, "path", path)
    object.__setattr__(result, "sha256", sha256)
    object.__setattr__(result, "canonical_payload", _canonical_json_bytes(payload))
    return result


def _validate_frozen_config(payload: Mapping[str, Any]) -> None:
    _assert_exact_keys(
        payload,
        {
            "schema_version",
            "artifact_status",
            "contains_observed_results",
            "context_feature_dim",
            "development_geometry",
            "pixel_budget_grid",
            "primary_pixel_budget",
            "threshold_semantics",
            "model",
            "optimizer",
            "loss",
            "checkpoint_selection",
            "collection_contract",
            "checkpoint_contract",
            "seed_contract",
        },
        name="config",
    )
    _exact(payload["schema_version"], CONFIG_SCHEMA, "config.schema_version")
    _exact(
        payload["artifact_status"],
        "RESULT_FREE_FROZEN_CONFIGURATION",
        "config.artifact_status",
    )
    _exact(payload["contains_observed_results"], False, "contains_observed_results")
    _exact(payload["context_feature_dim"], 93, "context_feature_dim")
    geometry = _assert_exact_keys(
        payload["development_geometry"],
        {"context_size", "query_size", "construction"},
        name="development_geometry",
    )
    _exact(geometry["context_size"], 14, "context_size")
    _exact(geometry["query_size"], 28, "query_size")
    _exact(
        geometry["construction"],
        "ordered_non_overlapping_contiguous_blocks_context_first_query_second",
        "development_geometry.construction",
    )
    _exact(payload["pixel_budget_grid"], [1e-4, 1e-5, 1e-6], "pixel_budget_grid")
    _exact(payload["primary_pixel_budget"], 1e-5, "primary_pixel_budget")
    _exact(
        payload["threshold_semantics"],
        "prediction = probability > threshold",
        "threshold_semantics",
    )

    model = _assert_exact_keys(
        payload["model"],
        {
            "hidden_dims",
            "activation",
            "dropout",
            "min_logit",
            "max_logit",
            "minimum_logit_gap",
            "reject_head",
            "missing_episode_fallback",
            "methods",
        },
        name="model",
    )
    for key, expected in (
        ("hidden_dims", [32]),
        ("activation", "GELU"),
        ("dropout", 0.1),
        ("min_logit", -10.0),
        ("max_logit", 18.0),
        ("minimum_logit_gap", 0.001),
        ("reject_head", False),
        ("missing_episode_fallback", False),
    ):
        _exact(model[key], expected, f"model.{key}")
    methods = _assert_exact_keys(model["methods"], METHODS, name="model.methods")
    expected_methods = {
        "T6": (
            "DirectNoRejectPixelCalibrator",
            "oracle_logit_huber_only",
            False,
            3107,
        ),
        "T7": (
            "MonotoneNoRejectPixelRiskCalibrator",
            "oracle_logit_huber_only",
            True,
            3140,
        ),
        "T8": (
            "MonotoneNoRejectPixelRiskCalibrator",
            "query_risk_aligned_exact_curve",
            True,
            3140,
        ),
    }
    for method, expected in expected_methods.items():
        item = _assert_exact_keys(
            methods[method],
            {
                "class",
                "objective",
                "structural_monotonicity",
                "expected_trainable_parameters",
            },
            name=f"model.methods.{method}",
        )
        for key, wanted in zip(
            (
                "class",
                "objective",
                "structural_monotonicity",
                "expected_trainable_parameters",
            ),
            expected,
        ):
            _exact(item[key], wanted, f"model.methods.{method}.{key}")

    optimizer = _assert_exact_keys(
        payload["optimizer"],
        {
            "name",
            "learning_rate",
            "weight_decay",
            "betas",
            "epsilon",
            "amsgrad",
            "scheduler",
            "batch_size",
            "max_epochs",
            "early_stopping_patience",
            "gradient_clip_norm",
            "num_workers",
            "deterministic_algorithms",
            "amp",
            "shuffle_training",
            "drop_last",
        },
        name="optimizer",
    )
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "scheduler": "none",
        "batch_size": 16,
        "max_epochs": 100,
        "early_stopping_patience": 20,
        "gradient_clip_norm": 5.0,
        "num_workers": 0,
        "deterministic_algorithms": True,
        "amp": False,
        "shuffle_training": True,
        "drop_last": False,
    }
    for key, wanted in expected_optimizer.items():
        _exact(optimizer[key], wanted, f"optimizer.{key}")

    loss = _assert_exact_keys(
        payload["loss"],
        {
            "oracle_huber_delta",
            "lambda_violation",
            "lambda_utility",
            "lambda_oracle",
            "lambda_smoothness",
            "lambda_coverage",
            "risk_epsilon",
        },
        name="loss",
    )
    expected_loss = {
        "oracle_huber_delta": 1.0,
        "lambda_violation": 4.0,
        "lambda_utility": 1.0,
        "lambda_oracle": 0.1,
        "lambda_smoothness": 0.01,
        "lambda_coverage": 4.0,
        "risk_epsilon": 1e-12,
    }
    for key, wanted in expected_loss.items():
        _exact(loss[key], wanted, f"loss.{key}")

    selection = _assert_exact_keys(
        payload["checkpoint_selection"],
        {
            "source_domain_weighting",
            "within_domain_bsr",
            "within_domain_log_excess",
            "within_domain_pd",
            "rank",
            "outer_target_accessed",
        },
        name="checkpoint_selection",
    )
    expected_selection = {
        "source_domain_weighting": "equal_one_half",
        "within_domain_bsr": "equal_mandatory_window_mean",
        "within_domain_log_excess": "equal_mandatory_window_mean",
        "within_domain_pd": "pooled_tp_divided_by_pooled_gt",
        "rank": [
            "macro_source_BSR_max",
            "macro_source_LogExcess_min",
            "macro_source_Pd_max",
            "earlier_epoch_on_exact_tie",
        ],
        "outer_target_accessed": False,
    }
    for key, wanted in expected_selection.items():
        _exact(selection[key], wanted, f"checkpoint_selection.{key}")

    collection = _assert_exact_keys(
        payload["collection_contract"],
        {
            "schema_version",
            "commit_schema_version",
            "required_bundle_members",
            "external_sha256_required_for_every_member",
            "statistics_config_external_sha256_required",
            "statistics_config_shared_object_train_validation",
            "train_role",
            "validation_role",
            "both_source_domains_required",
            "outer_target_absent",
            "standardizer_fit_scope",
            "standardizer_dtype",
            "standardizer_scale_floor",
        },
        name="collection_contract",
    )
    expected_collection = {
        "schema_version": "rc-irstd.meta-episode-collection.v5",
        "commit_schema_version": "rc-irstd.meta-episode-collection-commit.v1",
        "required_bundle_members": ["jsonl", "manifest", "commit"],
        "external_sha256_required_for_every_member": True,
        "statistics_config_external_sha256_required": True,
        "statistics_config_shared_object_train_validation": True,
        "train_role": "stage2_oof_fit_detector_oof_only",
        "validation_role": "source_diagnostic_validation_detector_full_fit_only",
        "both_source_domains_required": True,
        "outer_target_absent": True,
        "standardizer_fit_scope": "training_contexts_only",
        "standardizer_dtype": "float64",
        "standardizer_scale_floor": 1e-8,
    }
    for key, wanted in expected_collection.items():
        _exact(collection[key], wanted, f"collection_contract.{key}")

    checkpoint = _assert_exact_keys(
        payload["checkpoint_contract"],
        {
            "schema_version",
            "serialization",
            "immutable_epoch_generations",
            "commit_published_last",
            "resume_requires_external_generation_commit_sha256",
            "reject_head",
            "official_test_accessed",
        },
        name="checkpoint_contract",
    )
    expected_checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "serialization": "torch_tensors_and_primitives_weights_only",
        "immutable_epoch_generations": True,
        "commit_published_last": True,
        "resume_requires_external_generation_commit_sha256": True,
        "reject_head": False,
        "official_test_accessed": False,
    }
    for key, wanted in expected_checkpoint.items():
        _exact(checkpoint[key], wanted, f"checkpoint_contract.{key}")

    seed = _assert_exact_keys(
        payload["seed_contract"],
        {
            "manifest_schema_version",
            "algorithm_id",
            "base_seeds",
            "method_roles",
            "python_builtin_hash_forbidden",
            "manual_seed_override_forbidden",
        },
        name="seed_contract",
    )
    expected_seed = {
        "manifest_schema_version": SEED_MANIFEST_SCHEMA,
        "algorithm_id": "sha256_domain_separated_seed_v1",
        "base_seeds": [42, 123, 3407],
        "method_roles": METHOD_SEED_ROLES,
        "python_builtin_hash_forbidden": True,
        "manual_seed_override_forbidden": True,
    }
    for key, wanted in expected_seed.items():
        _exact(seed[key], wanted, f"seed_contract.{key}")


def verify_stage2_crossfit_config(
    path: str | Path, expected_sha256: str
) -> VerifiedStage2CrossfitConfig:
    data = _verified_regular_file(path, expected_sha256, "stage2_crossfit_config")
    payload = _load_json_bytes(data, "stage2 crossfit config")
    _validate_frozen_config(payload)
    return _make_verified_config(Path(path), expected_sha256, payload)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_stage2_model(
    method: str, config: VerifiedStage2CrossfitConfig
) -> nn.Module:
    if type(config) is not VerifiedStage2CrossfitConfig:
        raise TypeError("config must be a publicly verified Stage2 config")
    if method not in METHODS:
        raise Stage2CalibratorContractError(f"unsupported method: {method!r}")
    payload = config.payload
    model_config = payload["model"]
    common = {
        "context_feature_dim": payload["context_feature_dim"],
        "pixel_budget_grid": payload["pixel_budget_grid"],
        "hidden_dims": model_config["hidden_dims"],
        "dropout": model_config["dropout"],
    }
    if method == "T6":
        model: nn.Module = DirectNoRejectPixelCalibrator(
            **common,
            min_logit=model_config["min_logit"],
            max_logit=model_config["max_logit"],
        )
    else:
        model = MonotoneNoRejectPixelRiskCalibrator(
            **common,
            min_logit=model_config["min_logit"],
            max_logit=model_config["max_logit"],
            minimum_logit_gap=model_config["minimum_logit_gap"],
        )
    observed = trainable_parameter_count(model)
    if observed != EXPECTED_PARAMETER_COUNTS[method]:
        raise RuntimeError(
            f"{method} trainable parameter count {observed} != "
            f"{EXPECTED_PARAMETER_COUNTS[method]}"
        )
    exported = model.export_config()
    if exported.get("hidden_dims") != [32]:
        raise RuntimeError(f"{method} export_config did not preserve hidden_dims=[32]")
    capability = model.capability_contract()
    if capability.get("supports_reject") is not False:
        raise RuntimeError(f"{method} unexpectedly supports reject")
    return model


def oracle_logit_huber_loss(
    predicted_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    oracle_valid: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if not isinstance(predicted_logits, torch.Tensor) or not predicted_logits.is_floating_point():
        raise TypeError("predicted_logits must be a floating-point tensor")
    if predicted_logits.ndim != 2 or predicted_logits.numel() == 0:
        raise ValueError("predicted_logits must be non-empty with shape [B,J]")
    if not bool(torch.isfinite(predicted_logits).all().item()):
        raise ValueError("predicted_logits must be finite")
    if (
        not isinstance(oracle_logits, torch.Tensor)
        or not oracle_logits.is_floating_point()
        or oracle_logits.shape != predicted_logits.shape
    ):
        raise ValueError("oracle_logits must be floating point and match predicted_logits")
    if (
        not isinstance(oracle_valid, torch.Tensor)
        or oracle_valid.dtype is not torch.bool
        or oracle_valid.shape != predicted_logits.shape
    ):
        raise ValueError("oracle_valid must be bool and match predicted_logits")
    if isinstance(delta, bool) or not math.isfinite(float(delta)) or float(delta) <= 0.0:
        raise ValueError("delta must be finite and positive")
    mask = oracle_valid.to(device=predicted_logits.device)
    values = oracle_logits.to(device=predicted_logits.device, dtype=torch.float64)
    if not bool(mask.any().item()):
        raise Stage2CalibratorContractError(
            "oracle regression batch contains no valid oracle target"
        )
    if not bool(torch.isfinite(values[mask]).all().item()):
        raise ValueError("valid oracle logits must be finite")
    return F.huber_loss(
        predicted_logits.to(dtype=torch.float64)[mask],
        values[mask],
        reduction="mean",
        delta=float(delta),
    )


def checkpoint_rank(metrics: Mapping[str, Any], epoch: int) -> tuple[float, float, float, int]:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    aliases = (
        ("macro_source_BSR", "bsr"),
        ("macro_source_LogExcess", "log_excess"),
        ("macro_source_Pd", "pd"),
    )
    values: list[float] = []
    for preferred, fallback in aliases:
        if preferred in metrics:
            raw = metrics[preferred]
        elif fallback in metrics:
            raw = metrics[fallback]
        else:
            raise Stage2CalibratorContractError(
                f"validation replay is missing {preferred}"
            )
        if isinstance(raw, bool):
            raise TypeError(f"{preferred} must be numeric, not bool")
        value = float(raw)
        if not math.isfinite(value):
            raise Stage2CalibratorContractError(f"{preferred} must be finite")
        values.append(value)
    return values[0], -values[1], values[2], -epoch


def is_better_checkpoint(
    candidate: tuple[float, float, float, int],
    incumbent: tuple[float, float, float, int] | None,
) -> bool:
    if len(candidate) != 4 or any(
        isinstance(value, bool) or not math.isfinite(float(value)) for value in candidate
    ):
        raise ValueError("candidate rank must contain four finite numeric values")
    if incumbent is None:
        return True
    if len(incumbent) != 4 or any(
        isinstance(value, bool) or not math.isfinite(float(value)) for value in incumbent
    ):
        raise ValueError("incumbent rank must contain four finite numeric values")
    return candidate > incumbent


def derive_stage2_seed(
    base_seed: int,
    outer_fold_id: str,
    artifact_role: str,
    oof_marker: str,
) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    for value, name in (
        (outer_fold_id, "outer_fold_id"),
        (artifact_role, "artifact_role"),
        (oof_marker, "oof_marker"),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must be a non-empty canonical string")
        try:
            value.encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError(f"{name} must be ASCII") from error
    preimage = json.dumps(
        [SEED_DOMAIN_TAG, base_seed, outer_fold_id, artifact_role, oof_marker],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    unsigned = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big")
    return 1 + unsigned % 2147483646


@dataclass(frozen=True, init=False)
class VerifiedStage2SeedSelection:
    manifest_path: Path
    manifest_sha256: str
    base_seed: int
    outer_fold_id: str
    method: str
    artifact_role: str
    oof_marker: str
    derived_seed: int
    row_sha256: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "VerifiedStage2SeedSelection can only be created by "
            "verify_stage2_seed_selection"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "base_seed": self.base_seed,
            "outer_fold_id": self.outer_fold_id,
            "method": self.method,
            "artifact_role": self.artifact_role,
            "oof_marker": self.oof_marker,
            "derived_seed": self.derived_seed,
            "row_sha256": self.row_sha256,
        }


def _make_verified_seed_selection(**values: Any) -> VerifiedStage2SeedSelection:
    expected = frozenset(VerifiedStage2SeedSelection.__dataclass_fields__)
    if frozenset(values) != expected:
        raise RuntimeError("internal seed selection field mismatch")
    result = object.__new__(VerifiedStage2SeedSelection)
    for key in expected:
        object.__setattr__(result, key, values[key])
    return result


def _seed_role_parts(mapping_key: str) -> tuple[str, str]:
    if not isinstance(mapping_key, str) or mapping_key.count("::") != 1:
        raise Stage2CalibratorContractError("seed role mapping key is malformed")
    artifact_role, oof_marker = mapping_key.split("::", 1)
    if not artifact_role or not oof_marker:
        raise Stage2CalibratorContractError("seed role mapping key is empty")
    return artifact_role, oof_marker


def verify_stage2_seed_selection(
    manifest_path: str | Path,
    manifest_sha256: str,
    *,
    base_seed: int,
    outer_fold_id: str,
    method: str,
) -> VerifiedStage2SeedSelection:
    if method not in METHODS:
        raise Stage2CalibratorContractError(f"unsupported method: {method!r}")
    if outer_fold_id not in OUTER_TARGETS:
        raise Stage2CalibratorContractError("outer_fold_id is not frozen")
    if isinstance(base_seed, bool) or base_seed not in (42, 123, 3407):
        raise Stage2CalibratorContractError("base_seed must be one of [42,123,3407]")
    data = _verified_regular_file(
        manifest_path, manifest_sha256, "stage2_seed_manifest"
    )
    payload = _load_json_bytes(data, "Stage2 seed manifest")
    if payload.get("schema_version") != SEED_MANIFEST_SCHEMA:
        raise Stage2CalibratorContractError("unsupported seed manifest schema")
    algorithm = payload.get("derivation_algorithm")
    if not isinstance(algorithm, Mapping):
        raise Stage2CalibratorContractError("seed manifest algorithm is missing")
    if (
        algorithm.get("algorithm_id") != "sha256_domain_separated_seed_v1"
        or algorithm.get("domain_tag") != SEED_DOMAIN_TAG
    ):
        raise Stage2CalibratorContractError("seed derivation algorithm changed")
    dimensions = payload.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise Stage2CalibratorContractError("seed manifest dimensions are missing")
    if dimensions.get("base_seeds") != [42, 123, 3407]:
        raise Stage2CalibratorContractError("seed manifest base seed order changed")
    if dimensions.get("outer_folds") != list(OUTER_TARGETS):
        raise Stage2CalibratorContractError("seed manifest outer fold order changed")
    role_rows = dimensions.get("role_order")
    if not isinstance(role_rows, list) or len(role_rows) != 7:
        raise Stage2CalibratorContractError("seed manifest must contain seven roles")
    role_keys: list[str] = []
    for row in role_rows:
        if not isinstance(row, Mapping):
            raise Stage2CalibratorContractError("seed role row must be an object")
        mapping_key = row.get("mapping_key")
        artifact_role, marker = _seed_role_parts(mapping_key)
        if row.get("artifact_role") != artifact_role or row.get("oof_marker") != marker:
            raise Stage2CalibratorContractError("seed role row disagrees with mapping key")
        role_keys.append(mapping_key)
    if len(set(role_keys)) != 7:
        raise Stage2CalibratorContractError("seed role mapping keys are not unique")

    table = payload.get("derived_seed_table")
    if not isinstance(table, list) or len(table) != 9:
        raise Stage2CalibratorContractError("seed table must contain nine rows")
    seen: set[tuple[int, str]] = set()
    selected_value: int | None = None
    selected_key = METHOD_SEED_ROLES[method]
    for row_index, row in enumerate(table):
        if not isinstance(row, Mapping):
            raise Stage2CalibratorContractError("seed table row must be an object")
        row_base = row.get("base_seed")
        seeds = row.get("derived_seeds_by_role")
        if (
            isinstance(row_base, bool)
            or row_base not in (42, 123, 3407)
            or not isinstance(seeds, Mapping)
            or list(seeds) != role_keys
        ):
            raise Stage2CalibratorContractError("seed table row dimensions changed")
        expected_base = (42, 123, 3407)[row_index // 3]
        row_outer = tuple(OUTER_TARGETS)[row_index % 3]
        if row_base != expected_base:
            raise Stage2CalibratorContractError("seed table base-major order changed")
        seen.add((row_base, row_outer))
        for mapping_key in role_keys:
            role, marker = _seed_role_parts(mapping_key)
            observed = seeds[mapping_key]
            if isinstance(observed, bool) or not isinstance(observed, int):
                raise Stage2CalibratorContractError("derived seed must be an integer")
            expected = derive_stage2_seed(row_base, row_outer, role, marker)
            if observed != expected:
                raise Stage2CalibratorContractError(
                    f"seed table mismatch for {row_base}/{row_outer}/{mapping_key}"
                )
        if row_base == base_seed and row_outer == outer_fold_id:
            selected_value = int(seeds[selected_key])
    if len(seen) != 9 or selected_value is None:
        raise Stage2CalibratorContractError("seed selection is incomplete")
    artifact_role, marker = _seed_role_parts(selected_key)
    row_payload = {
        "manifest_sha256": manifest_sha256,
        "base_seed": base_seed,
        "outer_fold_id": outer_fold_id,
        "method": method,
        "artifact_role": artifact_role,
        "oof_marker": marker,
        "derived_seed": selected_value,
    }
    return _make_verified_seed_selection(
        manifest_path=Path(manifest_path),
        manifest_sha256=manifest_sha256,
        base_seed=base_seed,
        outer_fold_id=outer_fold_id,
        method=method,
        artifact_role=artifact_role,
        oof_marker=marker,
        derived_seed=selected_value,
        row_sha256=sha256_bytes(_canonical_json_bytes(row_payload)),
    )


def seed_runtime(seed: int, *, include_cuda: bool = False) -> torch.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed <= 2147483646:
        raise ValueError("runtime seed is outside the frozen interval")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _numpy_rng_to_primitives(state: tuple[Any, ...]) -> dict[str, Any]:
    if len(state) != 5:
        raise RuntimeError("unexpected NumPy RNG state")
    return {
        "algorithm": str(state[0]),
        "keys": np.asarray(state[1], dtype=np.uint32).astype(np.uint64).tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _numpy_rng_from_primitives(state: Mapping[str, Any]) -> tuple[Any, ...]:
    _assert_exact_keys(
        state,
        {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"},
        name="numpy_rng_state",
    )
    return (
        state["algorithm"],
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    )


def _python_rng_to_primitives(state: tuple[Any, ...]) -> dict[str, Any]:
    if len(state) != 3:
        raise RuntimeError("unexpected Python RNG state")
    return {
        "version": int(state[0]),
        "internal": [int(item) for item in state[1]],
        "gauss_next": None if state[2] is None else float(state[2]),
    }


def capture_rng_state(*, include_cuda: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": _python_rng_to_primitives(random.getstate()),
        "numpy": _numpy_rng_to_primitives(np.random.get_state()),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": [],
    }
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("cannot capture requested CUDA RNG state")
        state["torch_cuda_all"] = list(torch.cuda.get_rng_state_all())
    return state


def restore_rng_state(state: Mapping[str, Any], *, include_cuda: bool) -> None:
    _assert_exact_keys(
        state,
        {"python", "numpy", "torch_cpu", "torch_cuda_all"},
        name="rng_state",
    )
    python_state = _assert_exact_keys(
        state["python"], {"version", "internal", "gauss_next"}, name="python_rng_state"
    )
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(item) for item in python_state["internal"]),
            python_state["gauss_next"],
        )
    )
    np.random.set_state(_numpy_rng_from_primitives(state["numpy"]))
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
    cuda_states = state["torch_cuda_all"]
    if include_cuda:
        if not torch.cuda.is_available() or not isinstance(cuda_states, list):
            raise RuntimeError("CUDA resume state is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise Stage2CalibratorContractError("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in cuda_states]
        )
    elif cuda_states != []:
        raise Stage2CalibratorContractError("CPU resume contains unexpected CUDA state")


TRAIN_WINDOW_COUNTS = {
    "outer_leave_nuaa_sirst": 26,
    "outer_leave_nudt_sirst": 18,
    "outer_leave_irstd_1k": 16,
}
VALIDATION_WINDOW_COUNTS = {
    "outer_leave_nuaa_sirst": 6,
    "outer_leave_nudt_sirst": 4,
    "outer_leave_irstd_1k": 4,
}


@dataclass(frozen=True, init=False)
class VerifiedLaneATrainingInputs:
    train: Any
    validation: Any
    statistics_config: Any
    standardizer: Any
    replay_capability: Any
    statistics_config_binding: Mapping[str, Any]
    train_binding: Mapping[str, Any]
    validation_binding: Mapping[str, Any]

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "VerifiedLaneATrainingInputs can only be created by "
            "verify_lane_a_training_inputs"
        )


def _make_verified_lane_a_inputs(**values: Any) -> VerifiedLaneATrainingInputs:
    expected = frozenset(VerifiedLaneATrainingInputs.__dataclass_fields__)
    if frozenset(values) != expected:
        raise RuntimeError("internal Lane-A training-input field mismatch")
    result = object.__new__(VerifiedLaneATrainingInputs)
    for key in expected:
        object.__setattr__(result, key, values[key])
    return result


def _episode_field(episode: Any, name: str) -> Any:
    if isinstance(episode, Mapping):
        if name not in episode:
            raise Stage2CalibratorContractError(f"episode is missing {name}")
        return episode[name]
    payload = getattr(episode, "payload", None)
    if isinstance(payload, Mapping):
        if name not in payload:
            raise Stage2CalibratorContractError(f"episode payload is missing {name}")
        return payload[name]
    if not hasattr(episode, name):
        raise Stage2CalibratorContractError(f"episode is missing {name}")
    return getattr(episode, name)


def _validate_collection_scope(
    collection: Any,
    *,
    expected_role: str,
    outer_fold_id: str,
    base_seed: int,
    expected_count: int,
) -> None:
    episodes = getattr(collection, "episodes", None)
    if not isinstance(episodes, (tuple, list)) or len(episodes) != expected_count:
        raise Stage2CalibratorContractError(
            f"{expected_role} collection must contain exactly {expected_count} episodes"
        )
    target = OUTER_TARGETS[outer_fold_id]
    expected_sources = set(OUTER_TARGETS.values()) - {target}
    observed_sources: set[str] = set()
    for episode in episodes:
        if _episode_field(episode, "outer_fold_id") != outer_fold_id:
            raise Stage2CalibratorContractError("episode outer fold mismatch")
        if _episode_field(episode, "episode_role") != expected_role:
            raise Stage2CalibratorContractError("episode role mismatch")
        if _episode_field(episode, "base_seed") != base_seed:
            raise Stage2CalibratorContractError("episode base seed mismatch")
        source_domain = _episode_field(episode, "source_domain")
        if source_domain == target:
            raise Stage2CalibratorContractError("outer target entered Stage2 training")
        observed_sources.add(source_domain)
        if _episode_field(episode, "official_test_accessed") is not False:
            raise Stage2CalibratorContractError("official test access must be exact false")
    if observed_sources != expected_sources:
        raise Stage2CalibratorContractError(
            "both and only the two source domains are required"
        )


def verify_lane_a_training_inputs(
    *,
    train_collection: str | Path,
    train_collection_sha256: str,
    train_manifest: str | Path,
    train_manifest_sha256: str,
    train_commit: str | Path,
    train_commit_sha256: str,
    validation_collection: str | Path,
    validation_collection_sha256: str,
    validation_manifest: str | Path,
    validation_manifest_sha256: str,
    validation_commit: str | Path,
    validation_commit_sha256: str,
    statistics_config_path: str | Path,
    statistics_config_sha256: str,
    outer_fold_id: str,
    base_seed: int,
    repository_root: str | Path | None = None,
) -> VerifiedLaneATrainingInputs:
    """Promote two complete Lane-A bundles; raw JSONL never crosses this API."""

    if outer_fold_id not in OUTER_TARGETS:
        raise Stage2CalibratorContractError("outer_fold_id is not frozen")
    if isinstance(base_seed, bool) or base_seed not in (42, 123, 3407):
        raise Stage2CalibratorContractError("base_seed is not frozen")
    # Delayed import lets model/config/checkpoint unit tests run while Lane A is
    # being materialized, without weakening the production dependency.
    try:
        from rc.stage2_crossfit_dataset import (
            COLLECTION_TRAIN,
            COLLECTION_VALIDATION,
            SOURCE_DIAGNOSTIC_VALIDATION,
            STAGE2_OOF_FIT,
            assert_stage2_context_standardizer,
            assert_stage2_sample_isolation,
            assert_verified_episode_collection,
            fit_stage2_context_standardizer,
            load_stage2_episodes_v5,
            make_stage2_trainer_replay_capability,
        )
    except ImportError as error:
        raise RuntimeError("the authorized Lane-A verifier is unavailable") from error
    try:
        from rc.stage2_crossfit_schema import verify_stage2_statistics_config
    except ImportError as error:
        raise RuntimeError("the authorized statistics-config verifier is unavailable") from error

    statistics_config = verify_stage2_statistics_config(
        statistics_config_path,
        statistics_config_sha256,
        repository_root=repository_root,
    )

    train = load_stage2_episodes_v5(
        train_collection,
        train_collection_sha256,
        collection_manifest_path=train_manifest,
        collection_manifest_sha256=train_manifest_sha256,
        commit_marker_path=train_commit,
        commit_marker_sha256=train_commit_sha256,
        statistics_config=statistics_config,
        repository_root=repository_root,
    )
    validation = load_stage2_episodes_v5(
        validation_collection,
        validation_collection_sha256,
        collection_manifest_path=validation_manifest,
        collection_manifest_sha256=validation_manifest_sha256,
        commit_marker_path=validation_commit,
        commit_marker_sha256=validation_commit_sha256,
        statistics_config=statistics_config,
        repository_root=repository_root,
    )
    assert_verified_episode_collection(train)
    assert_verified_episode_collection(validation)
    if train.manifest["collection_role"] != COLLECTION_TRAIN:
        raise Stage2CalibratorContractError("train manifest collection_role mismatch")
    if validation.manifest["collection_role"] != COLLECTION_VALIDATION:
        raise Stage2CalibratorContractError(
            "validation manifest collection_role mismatch"
        )
    _validate_collection_scope(
        train,
        expected_role=STAGE2_OOF_FIT,
        outer_fold_id=outer_fold_id,
        base_seed=base_seed,
        expected_count=TRAIN_WINDOW_COUNTS[outer_fold_id],
    )
    _validate_collection_scope(
        validation,
        expected_role=SOURCE_DIAGNOSTIC_VALIDATION,
        outer_fold_id=outer_fold_id,
        base_seed=base_seed,
        expected_count=VALIDATION_WINDOW_COUNTS[outer_fold_id],
    )
    assert_stage2_sample_isolation(train, validation)
    standardizer = fit_stage2_context_standardizer(train)
    assert_stage2_context_standardizer(standardizer)
    if getattr(standardizer, "train_collection_sha256", None) != train_collection_sha256:
        raise Stage2CalibratorContractError("standardizer training binding mismatch")
    replay_capability = make_stage2_trainer_replay_capability(
        train, validation, standardizer
    )
    config_payload = statistics_config.to_dict()
    if type(config_payload) is not dict:
        raise TypeError("verified statistics config must export one exact dict")
    statistics_config_binding = {
        "sha256": _validate_sha256(
            statistics_config_sha256, "statistics_config_sha256"
        ),
        "config": config_payload,
        "external_sha256_verified": True,
    }
    train_binding = {
        "collection_sha256": _validate_sha256(
            train_collection_sha256, "train_collection_sha256"
        ),
        "manifest_sha256": _validate_sha256(
            train_manifest_sha256, "train_manifest_sha256"
        ),
        "commit_sha256": _validate_sha256(train_commit_sha256, "train_commit_sha256"),
        "role": "stage2_oof_fit_detector_oof_only",
        "episode_count": TRAIN_WINDOW_COUNTS[outer_fold_id],
    }
    validation_binding = {
        "collection_sha256": _validate_sha256(
            validation_collection_sha256, "validation_collection_sha256"
        ),
        "manifest_sha256": _validate_sha256(
            validation_manifest_sha256, "validation_manifest_sha256"
        ),
        "commit_sha256": _validate_sha256(
            validation_commit_sha256, "validation_commit_sha256"
        ),
        "role": "source_diagnostic_validation_detector_full_fit_only",
        "episode_count": VALIDATION_WINDOW_COUNTS[outer_fold_id],
    }
    return _make_verified_lane_a_inputs(
        train=train,
        validation=validation,
        statistics_config=statistics_config,
        standardizer=standardizer,
        replay_capability=replay_capability,
        statistics_config_binding=statistics_config_binding,
        train_binding=train_binding,
        validation_binding=validation_binding,
    )


def _assert_canonical_output_parent(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise Stage2CalibratorContractError("output parent must be absolute and canonical")
    if path.resolve(strict=True) != path or not path.is_dir():
        raise Stage2CalibratorContractError("output parent must be a canonical directory")


def _transactional_publish_bundle(
    files: Mapping[Path, bytes], *, commit_last: Path
) -> None:
    """Publish all-new same-parent files with the commit inode linked last."""

    if not files or commit_last not in files:
        raise Stage2CalibratorContractError("transaction requires a final commit member")
    targets = list(files)
    if len(set(targets)) != len(targets):
        raise Stage2CalibratorContractError("duplicate transaction target")
    parent = targets[0].parent
    if any(target.parent != parent for target in targets):
        raise Stage2CalibratorContractError("transaction targets must share one parent")
    _assert_canonical_output_parent(parent)
    if targets[-1] != commit_last:
        raise Stage2CalibratorContractError("commit must be the last transaction member")
    for target in targets:
        if not target.is_absolute() or ".." in target.parts:
            raise Stage2CalibratorContractError("transaction target is not canonical")
        try:
            os.lstat(target)
        except FileNotFoundError:
            continue
        raise Stage2CalibratorContractError(
            f"transaction target already exists: {target.name}"
        )

    staged: list[tuple[Path, Path, tuple[int, int]]] = []
    linked: list[tuple[Path, tuple[int, int]]] = []
    try:
        for target, data in files.items():
            descriptor, temporary_name = tempfile.mkstemp(
                dir=parent, prefix=f".{target.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            stat = os.stat(temporary, follow_symlinks=False)
            staged.append((target, temporary, (stat.st_dev, stat.st_ino)))
        for target, _, _ in staged:
            try:
                os.lstat(target)
            except FileNotFoundError:
                continue
            raise Stage2CalibratorContractError(
                f"transaction target appeared while staging: {target.name}"
            )
        for target, temporary, identity in staged:
            os.link(temporary, target, follow_symlinks=False)
            linked.append((target, identity))
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        for target, identity in reversed(linked):
            try:
                observed = os.stat(target, follow_symlinks=False)
                if (observed.st_dev, observed.st_ino) == identity:
                    os.unlink(target)
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _torch_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


@dataclass(frozen=True, init=False)
class VerifiedCalibratorGeneration:
    commit_path: Path
    commit_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    history_path: Path
    history_sha256: str
    replay_path: Path
    replay_sha256: str
    epoch: int

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "VerifiedCalibratorGeneration can only be created by "
            "verify_calibrator_generation"
        )


def _make_verified_generation(**values: Any) -> VerifiedCalibratorGeneration:
    expected = frozenset(VerifiedCalibratorGeneration.__dataclass_fields__)
    if frozenset(values) != expected:
        raise RuntimeError("internal generation field mismatch")
    result = object.__new__(VerifiedCalibratorGeneration)
    for key in expected:
        object.__setattr__(result, key, values[key])
    return result


def publish_calibrator_generation(
    output_dir: str | Path,
    *,
    epoch: int,
    checkpoint_payload: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    exact_replay: Mapping[str, Any],
) -> VerifiedCalibratorGeneration:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    root = Path(output_dir).expanduser()
    if not root.is_absolute() or ".." in root.parts or root.is_symlink():
        raise Stage2CalibratorContractError("output_dir must be absolute and canonical")
    if not root.exists():
        root.mkdir(mode=0o700, parents=False)
    _assert_canonical_output_parent(root)
    generations = root / "generations"
    if not generations.exists():
        generations.mkdir(mode=0o700)
    _assert_canonical_output_parent(generations)
    generation_dir = generations / f"epoch_{epoch:04d}"
    generation_dir.mkdir(mode=0o700)
    _assert_canonical_output_parent(generation_dir)

    if frozenset(checkpoint_payload) != CHECKPOINT_KEYS:
        raise Stage2CalibratorContractError("generation checkpoint keys are not v6")
    if checkpoint_payload["format_version"] != CHECKPOINT_SCHEMA:
        raise Stage2CalibratorContractError("generation checkpoint schema mismatch")
    if checkpoint_payload["completed_epoch"] != epoch:
        raise Stage2CalibratorContractError("generation checkpoint epoch mismatch")
    checkpoint_bytes = _torch_payload_bytes(checkpoint_payload)
    history_bytes = b"".join(
        _canonical_json_bytes(dict(row)) + b"\n" for row in history
    )
    replay_bytes = _pretty_json_bytes(dict(exact_replay))
    history_sha = sha256_bytes(history_bytes)
    replay_sha = sha256_bytes(replay_bytes)
    if checkpoint_payload["history_sha256"] != history_sha:
        raise Stage2CalibratorContractError("checkpoint/history precommit mismatch")
    if checkpoint_payload["exact_replay_sha256"] != replay_sha:
        raise Stage2CalibratorContractError("checkpoint/replay precommit mismatch")
    member_bytes = {
        "checkpoint.pt": checkpoint_bytes,
        "history.jsonl": history_bytes,
        "exact_replay.json": replay_bytes,
    }
    members = {
        name: {"sha256": sha256_bytes(data), "size_bytes": len(data)}
        for name, data in member_bytes.items()
    }
    commit = {
        "schema_version": GENERATION_COMMIT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "members": members,
        "commit_published_last": True,
        "official_test_accessed": False,
    }
    commit_bytes = _pretty_json_bytes(commit)
    commit_sha = sha256_bytes(commit_bytes)
    files: dict[Path, bytes] = {}
    for name, data in member_bytes.items():
        files[generation_dir / name] = data
        files[generation_dir / f"{name}.sha256"] = (
            f"{members[name]['sha256']}  {name}\n".encode("ascii")
        )
    commit_path = generation_dir / "COMMIT.json"
    files[generation_dir / "COMMIT.json.sha256"] = (
        f"{commit_sha}  COMMIT.json\n".encode("ascii")
    )
    files[commit_path] = commit_bytes
    try:
        _transactional_publish_bundle(files, commit_last=commit_path)
        return verify_calibrator_generation(commit_path, commit_sha)
    except BaseException:
        try:
            generation_dir.rmdir()
        except OSError:
            pass
        raise


def _read_sha_sidecar(path: Path, expected_name: str) -> str:
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise Stage2CalibratorContractError("SHA sidecar path is not canonical")
    if path.resolve(strict=True) != path or not path.is_file():
        raise Stage2CalibratorContractError("SHA sidecar is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            data = handle.read()
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(before) != identity(path_after):
            raise Stage2CalibratorContractError("SHA sidecar changed while read")
    finally:
        os.close(descriptor)
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise Stage2CalibratorContractError("SHA sidecar is not ASCII") from error
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)\n", text)
    if match is None or match.group(2) != expected_name:
        raise Stage2CalibratorContractError(f"malformed SHA sidecar for {expected_name}")
    return match.group(1)


def verify_calibrator_generation(
    commit_path: str | Path, expected_commit_sha256: str
) -> VerifiedCalibratorGeneration:
    commit_bytes = _verified_regular_file(
        commit_path, expected_commit_sha256, "calibrator_generation_commit"
    )
    commit = _load_json_bytes(commit_bytes, "calibrator generation commit")
    _assert_exact_keys(
        commit,
        {
            "schema_version",
            "checkpoint_schema_version",
            "epoch",
            "members",
            "commit_published_last",
            "official_test_accessed",
        },
        name="generation_commit",
    )
    _exact(commit["schema_version"], GENERATION_COMMIT_SCHEMA, "commit schema")
    _exact(commit["checkpoint_schema_version"], CHECKPOINT_SCHEMA, "checkpoint schema")
    _exact(commit["commit_published_last"], True, "commit_published_last")
    _exact(commit["official_test_accessed"], False, "official_test_accessed")
    epoch = commit["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise Stage2CalibratorContractError("commit epoch is invalid")
    members = _assert_exact_keys(
        commit["members"],
        {"checkpoint.pt", "history.jsonl", "exact_replay.json"},
        name="generation members",
    )
    parent = Path(commit_path).parent
    paths: dict[str, Path] = {}
    for name, binding in members.items():
        _assert_exact_keys(binding, {"sha256", "size_bytes"}, name=f"member {name}")
        digest = _validate_sha256(binding["sha256"], f"{name}.sha256")
        size = binding["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise Stage2CalibratorContractError(f"{name} size is invalid")
        member_path = parent / name
        data = _verified_regular_file(member_path, digest, name)
        if len(data) != size:
            raise Stage2CalibratorContractError(f"{name} size mismatch")
        sidecar = parent / f"{name}.sha256"
        if _read_sha_sidecar(sidecar, name) != digest:
            raise Stage2CalibratorContractError(f"{name} sidecar mismatch")
        paths[name] = member_path
    commit_sidecar = parent / "COMMIT.json.sha256"
    if _read_sha_sidecar(commit_sidecar, "COMMIT.json") != expected_commit_sha256:
        raise Stage2CalibratorContractError("commit sidecar mismatch")
    return _make_verified_generation(
        commit_path=Path(commit_path),
        commit_sha256=expected_commit_sha256,
        checkpoint_path=paths["checkpoint.pt"],
        checkpoint_sha256=members["checkpoint.pt"]["sha256"],
        history_path=paths["history.jsonl"],
        history_sha256=members["history.jsonl"]["sha256"],
        replay_path=paths["exact_replay.json"],
        replay_sha256=members["exact_replay.json"]["sha256"],
        epoch=epoch,
    )


def _cpu_primitive_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise Stage2CalibratorContractError("checkpoint contains non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError("checkpoint mappings require string/integer keys")
            result[key] = _cpu_primitive_tree(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_cpu_primitive_tree(item) for item in value]
    raise TypeError(f"unsupported checkpoint value type: {type(value).__name__}")


def standardizer_checkpoint_payload(standardizer: Any) -> dict[str, Any]:
    feature_names = tuple(getattr(standardizer, "feature_names", ()))
    mean = np.asarray(getattr(standardizer, "mean", ()), dtype=np.float64)
    scale = np.asarray(getattr(standardizer, "scale", ()), dtype=np.float64)
    if len(feature_names) != 93 or mean.shape != (93,) or scale.shape != (93,):
        raise Stage2CalibratorContractError("standardizer checkpoint shape mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise Stage2CalibratorContractError("standardizer checkpoint values are invalid")
    return {
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "fit_records_sha256": _validate_sha256(
            getattr(standardizer, "fit_records_sha256", None),
            "standardizer.fit_records_sha256",
        ),
        "fit_manifest_sha256": _validate_sha256(
            getattr(standardizer, "fit_manifest_sha256", None),
            "standardizer.fit_manifest_sha256",
        ),
        "train_collection_sha256": _validate_sha256(
            getattr(standardizer, "train_collection_sha256", None),
            "standardizer.train_collection_sha256",
        ),
        "calculation_dtype": "float64",
        "scale_floor": 1e-8,
        "feature_mask_applied_after_standardization": True,
    }


def collection_transitive_bindings(inputs: VerifiedLaneATrainingInputs) -> dict[str, Any]:
    if type(inputs) is not VerifiedLaneATrainingInputs:
        raise TypeError("verified Lane-A inputs are required")
    rows: list[dict[str, Any]] = []
    for split, collection in (("train", inputs.train), ("validation", inputs.validation)):
        for episode in collection.episodes:
            payload = episode.payload
            rows.append(
                {
                    "split": split,
                    "episode_id": episode.episode_id,
                    "detector_identity": dict(payload["detector_identity"]),
                    "source_reference_binding": dict(payload["source_reference_binding"]),
                }
            )
    return {
        "rows": rows,
        "rows_sha256": sha256_bytes(_canonical_json_bytes(rows)),
    }


CHECKPOINT_KEYS = frozenset(
    {
        "format_version",
        "artifact_kind",
        "method",
        "calibrator_model",
        "model_config",
        "capability_contract",
        "expected_trainable_parameters",
        "model_state_dict",
        "optimizer_state_dict",
        "completed_epoch",
        "next_epoch",
        "best_epoch",
        "best_rank",
        "epochs_without_improvement",
        "training_contract",
        "standardizer",
        "rng_state",
        "data_loader_generator_state",
        "history_sha256",
        "exact_replay_sha256",
        "reject_head",
        "missing_episode_fallback",
        "official_test_accessed",
    }
)


def make_calibrator_checkpoint_v6(
    *,
    method: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    best_epoch: int,
    best_rank: Sequence[float | int],
    epochs_without_improvement: int,
    training_contract: Mapping[str, Any],
    standardizer: Any,
    data_loader_generator: torch.Generator,
    history_sha256: str,
    exact_replay_sha256: str,
    include_cuda_rng: bool,
) -> dict[str, Any]:
    if method not in METHODS:
        raise Stage2CalibratorContractError("checkpoint method is not frozen")
    if (
        isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or completed_epoch < 0
    ):
        raise ValueError("completed_epoch must be non-negative")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 0 <= best_epoch <= completed_epoch:
        raise ValueError("best_epoch is outside completed training")
    if (
        isinstance(epochs_without_improvement, bool)
        or not isinstance(epochs_without_improvement, int)
        or epochs_without_improvement < 0
    ):
        raise ValueError("epochs_without_improvement must be non-negative")
    raw_rank = list(best_rank)
    if len(raw_rank) != 4 or any(
        isinstance(value, bool) or not math.isfinite(float(value)) for value in raw_rank
    ):
        raise ValueError("best_rank must contain four finite values")
    rank = [float(value) if index < 3 else int(value) for index, value in enumerate(raw_rank)]
    if trainable_parameter_count(model) != EXPECTED_PARAMETER_COUNTS[method]:
        raise Stage2CalibratorContractError("checkpoint model parameter count changed")
    capability = model.capability_contract()
    if capability.get("supports_reject") is not False:
        raise Stage2CalibratorContractError("checkpoint model supports reject")
    model_name = (
        DIRECT_NO_REJECT_MODEL_ID
        if method == "T6"
        else "monotone_no_reject_pixel_risk_calibrator"
    )
    payload = {
        "format_version": CHECKPOINT_SCHEMA,
        "artifact_kind": "immutable_epoch_training_state",
        "method": method,
        "calibrator_model": model_name,
        "model_config": model.export_config(),
        "capability_contract": capability,
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNTS[method],
        "model_state_dict": _cpu_primitive_tree(model.state_dict()),
        "optimizer_state_dict": _cpu_primitive_tree(optimizer.state_dict()),
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "best_epoch": best_epoch,
        "best_rank": rank,
        "epochs_without_improvement": epochs_without_improvement,
        "training_contract": _cpu_primitive_tree(training_contract),
        "standardizer": standardizer_checkpoint_payload(standardizer),
        "rng_state": capture_rng_state(include_cuda=include_cuda_rng),
        "data_loader_generator_state": data_loader_generator.get_state().cpu(),
        "history_sha256": _validate_sha256(history_sha256, "history_sha256"),
        "exact_replay_sha256": _validate_sha256(
            exact_replay_sha256, "exact_replay_sha256"
        ),
        "reject_head": False,
        "missing_episode_fallback": False,
        "official_test_accessed": False,
    }
    if frozenset(payload) != CHECKPOINT_KEYS:
        raise RuntimeError("internal checkpoint key mismatch")
    return payload


@dataclass(frozen=True, init=False)
class VerifiedCalibratorCheckpointV6:
    path: Path
    sha256: str
    method: str
    completed_epoch: int
    checkpoint_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "VerifiedCalibratorCheckpointV6 can only be created by "
            "verify_calibrator_checkpoint_v6"
        )

    def payload(self) -> dict[str, Any]:
        value = torch.load(io.BytesIO(self.checkpoint_bytes), map_location="cpu", weights_only=True)
        if type(value) is not dict:
            raise TypeError("verified checkpoint payload changed type")
        return value


def _make_verified_checkpoint(**values: Any) -> VerifiedCalibratorCheckpointV6:
    expected = frozenset(VerifiedCalibratorCheckpointV6.__dataclass_fields__)
    if frozenset(values) != expected:
        raise RuntimeError("internal verified checkpoint field mismatch")
    result = object.__new__(VerifiedCalibratorCheckpointV6)
    for key in expected:
        object.__setattr__(result, key, values[key])
    return result


def _model_from_checkpoint_payload(payload: Mapping[str, Any]) -> nn.Module:
    method = payload["method"]
    config = payload["model_config"]
    if type(config) is not dict:
        raise TypeError("checkpoint model_config must be an exact dict")
    expected_config_keys = (
        {"context_feature_dim", "pixel_budget_grid", "hidden_dims", "dropout", "min_logit", "max_logit"}
        if method == "T6"
        else {
            "context_feature_dim",
            "pixel_budget_grid",
            "hidden_dims",
            "dropout",
            "min_logit",
            "max_logit",
            "minimum_logit_gap",
        }
    )
    _assert_exact_keys(config, expected_config_keys, name="checkpoint model_config")
    if config["context_feature_dim"] != 93 or config["pixel_budget_grid"] != [1e-4, 1e-5, 1e-6]:
        raise Stage2CalibratorContractError("checkpoint model dimensions changed")
    if config["hidden_dims"] != [32] or config["dropout"] != 0.1:
        raise Stage2CalibratorContractError("checkpoint hidden architecture changed")
    if config["min_logit"] != -10.0 or config["max_logit"] != 18.0:
        raise Stage2CalibratorContractError("checkpoint logit bounds changed")
    if method == "T6":
        model: nn.Module = DirectNoRejectPixelCalibrator(**config)
    else:
        if config["minimum_logit_gap"] != 0.001:
            raise Stage2CalibratorContractError("checkpoint minimum logit gap changed")
        model = MonotoneNoRejectPixelRiskCalibrator(**config)
    return model


def verify_calibrator_checkpoint_v6(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_method: str | None = None,
    expected_training_contract: Mapping[str, Any] | None = None,
) -> VerifiedCalibratorCheckpointV6:
    data = _verified_regular_file(path, expected_sha256, "calibrator_v6_checkpoint")
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage2CalibratorContractError(
            "checkpoint is not tensors/primitives-only weights-only data"
        ) from error
    if type(payload) is not dict:
        raise TypeError("checkpoint must contain an exact dict")
    if frozenset(payload) != CHECKPOINT_KEYS:
        raise Stage2CalibratorContractError("checkpoint keys differ from schema v6")
    _exact(payload["format_version"], CHECKPOINT_SCHEMA, "checkpoint format")
    _exact(
        payload["artifact_kind"],
        "immutable_epoch_training_state",
        "checkpoint artifact_kind",
    )
    method = payload["method"]
    if method not in METHODS or (expected_method is not None and method != expected_method):
        raise Stage2CalibratorContractError("checkpoint method mismatch")
    expected_name = (
        DIRECT_NO_REJECT_MODEL_ID
        if method == "T6"
        else "monotone_no_reject_pixel_risk_calibrator"
    )
    _exact(payload["calibrator_model"], expected_name, "calibrator_model")
    _exact(
        payload["expected_trainable_parameters"],
        EXPECTED_PARAMETER_COUNTS[method],
        "expected_trainable_parameters",
    )
    for field in ("reject_head", "missing_episode_fallback", "official_test_accessed"):
        _exact(payload[field], False, f"checkpoint.{field}")
    capability = payload["capability_contract"]
    if not isinstance(capability, Mapping) or capability.get("supports_reject") is not False:
        raise Stage2CalibratorContractError("checkpoint capability supports reject")
    completed = payload["completed_epoch"]
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise Stage2CalibratorContractError("checkpoint completed_epoch is invalid")
    _exact(payload["next_epoch"], completed + 1, "checkpoint.next_epoch")
    best_epoch = payload["best_epoch"]
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or not 0 <= best_epoch <= completed:
        raise Stage2CalibratorContractError("checkpoint best_epoch is invalid")
    best_rank = payload["best_rank"]
    if not isinstance(best_rank, list) or len(best_rank) != 4 or any(
        isinstance(item, bool) or not math.isfinite(float(item)) for item in best_rank
    ):
        raise Stage2CalibratorContractError("checkpoint best_rank is invalid")
    patience = payload["epochs_without_improvement"]
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
        raise Stage2CalibratorContractError("checkpoint patience state is invalid")
    if expected_training_contract is not None and payload["training_contract"] != dict(
        expected_training_contract
    ):
        raise Stage2CalibratorContractError("checkpoint training contract mismatch")
    for field in ("history_sha256", "exact_replay_sha256"):
        _validate_sha256(payload[field], f"checkpoint.{field}")
    model = _model_from_checkpoint_payload(payload)
    if trainable_parameter_count(model) != EXPECTED_PARAMETER_COUNTS[method]:
        raise Stage2CalibratorContractError("checkpoint parameter count mismatch")
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("checkpoint model_state_dict is invalid")
    model.load_state_dict(state, strict=True)
    if model.export_config() != payload["model_config"]:
        raise Stage2CalibratorContractError("checkpoint model replay changed config")
    if model.capability_contract() != capability:
        raise Stage2CalibratorContractError("checkpoint capability replay mismatch")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise TypeError("checkpoint optimizer state is invalid")
    if not isinstance(payload["rng_state"], Mapping):
        raise TypeError("checkpoint RNG state is invalid")
    generator_state = payload["data_loader_generator_state"]
    if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8:
        raise TypeError("checkpoint DataLoader generator state is invalid")
    return _make_verified_checkpoint(
        path=Path(path),
        sha256=expected_sha256,
        method=method,
        completed_epoch=completed,
        checkpoint_bytes=data,
    )


def _parse_history_jsonl(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(data.splitlines()):
        if not line:
            raise Stage2CalibratorContractError("history contains an empty row")
        row = _load_json_bytes(line, f"history row {index}")
        if row.get("epoch") != index:
            raise Stage2CalibratorContractError("history epochs are not contiguous")
        rows.append(row)
    if not rows:
        raise Stage2CalibratorContractError("history must not be empty")
    return rows


def resume_calibrator_generation(
    *,
    commit_path: str | Path,
    commit_sha256: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader_generator: torch.Generator,
    expected_method: str,
    expected_training_contract: Mapping[str, Any],
    include_cuda_rng: bool,
) -> tuple[int, int, tuple[float, float, float, int], int, list[dict[str, Any]]]:
    generation = verify_calibrator_generation(commit_path, commit_sha256)
    checkpoint = verify_calibrator_checkpoint_v6(
        generation.checkpoint_path,
        generation.checkpoint_sha256,
        expected_method=expected_method,
        expected_training_contract=expected_training_contract,
    )
    payload = checkpoint.payload()
    if checkpoint.completed_epoch != generation.epoch:
        raise Stage2CalibratorContractError("generation/checkpoint epoch mismatch")
    if payload["history_sha256"] != generation.history_sha256:
        raise Stage2CalibratorContractError("checkpoint/history binding mismatch")
    if payload["exact_replay_sha256"] != generation.replay_sha256:
        raise Stage2CalibratorContractError("checkpoint/replay binding mismatch")
    history_bytes = _verified_regular_file(
        generation.history_path, generation.history_sha256, "resume_history"
    )
    history = _parse_history_jsonl(history_bytes)
    if history[-1]["epoch"] != checkpoint.completed_epoch:
        raise Stage2CalibratorContractError("history/checkpoint epoch mismatch")
    replay_bytes = _verified_regular_file(
        generation.replay_path, generation.replay_sha256, "resume_exact_replay"
    )
    replay = _load_json_bytes(replay_bytes, "resume exact replay")
    if replay.get("epoch") != checkpoint.completed_epoch:
        raise Stage2CalibratorContractError("replay/checkpoint epoch mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_rng_state(payload["rng_state"], include_cuda=include_cuda_rng)
    data_loader_generator.set_state(payload["data_loader_generator_state"])
    rank = tuple(payload["best_rank"])
    return (
        int(payload["next_epoch"]),
        int(payload["best_epoch"]),
        (float(rank[0]), float(rank[1]), float(rank[2]), int(rank[3])),
        int(payload["epochs_without_improvement"]),
        history,
    )


LOSS_METRIC_NAMES = (
    "total",
    "violation",
    "utility",
    "oracle_logit",
    "curve_smoothness",
    "coverage_penalty",
)


_RAGGED_CURVE_FIELDS = frozenset(
    {
        "curve_thresholds",
        "curve_logits",
        "curve_pixel_risk",
        "curve_pd",
        "curve_fp_pixels",
        "curve_tp_objects",
    }
)


def _move_batch(
    batch: Mapping[str, Any], device: torch.device, *, method: str
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key in _RAGGED_CURVE_FIELDS:
            if method == "T8" and isinstance(value, torch.Tensor):
                raise Stage2CalibratorContractError(
                    "T8 requires ragged CPU exact curves; padded curve tensors are forbidden"
                )
            moved[key] = value
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=False)
        else:
            moved[key] = value
    return moved


def _ragged_curve_size(value: Any, *, field: str) -> int:
    """Return a ragged CPU column length without materialising that column."""

    from rc.stage2_crossfit_dataset import Stage2CurveLogitView

    if isinstance(value, Stage2CurveLogitView):
        if field != "curve_logits":
            raise Stage2CalibratorContractError(
                "Stage2CurveLogitView is valid only for curve_logits"
            )
        size = len(value)
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 1:
            raise Stage2CalibratorContractError(f"{field} must be one ragged CPU vector")
        size = value.numel()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise Stage2CalibratorContractError(f"{field} must be one ragged CPU vector")
        size = value.size
    else:
        raise Stage2CalibratorContractError(
            f"{field} must contain CPU tensors, NumPy arrays, or the Lane-A logit view"
        )
    if size < 2:
        raise Stage2CalibratorContractError(f"{field} requires at least two exact points")
    return int(size)


def _curve_bracket_union(value: Any, queries: np.ndarray) -> np.ndarray:
    """Find exact neighbouring rows without converting the complete curve."""

    from rc.stage2_crossfit_dataset import Stage2CurveLogitView

    if isinstance(value, Stage2CurveLogitView):
        # logit(.) is strictly increasing.  Searching the verified threshold
        # backing column avoids materialising its potentially million-row
        # logit projection.  Only the three live queries are transformed.
        coordinate = value.thresholds
        query_tensor = torch.from_numpy(np.asarray(queries, dtype=np.float64))
        search_queries = torch.sigmoid(query_tensor).numpy()
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.ndim != 1:
            raise Stage2CalibratorContractError(
                "curve_logits must contain ragged CPU vectors"
            )
        coordinate = value.detach().numpy()
        search_queries = np.asarray(queries, dtype=np.float64)
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise Stage2CalibratorContractError(
                "curve_logits must contain ragged CPU vectors"
            )
        coordinate = value
        search_queries = np.asarray(queries, dtype=np.float64)
    else:
        raise Stage2CalibratorContractError(
            "curve_logits must use the Lane-A logit view or a ragged CPU vector"
        )
    size = int(coordinate.size)
    if size < 2:
        raise Stage2CalibratorContractError("ragged curve logits require two points")
    right = np.searchsorted(coordinate, search_queries, side="right")
    right = np.clip(right, 1, size - 1).astype(np.int64, copy=False)
    ordered = np.unique(np.concatenate((right - 1, right)))
    if ordered.size < 2 or ordered.size > 6:
        raise RuntimeError("compact exact-curve bracket cardinality is invalid")
    return ordered


def _select_ragged_curve_values(
    value: Any,
    indices: np.ndarray,
    *,
    field: str,
    expected_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Transfer only selected exact rows from a verified ragged CPU column."""

    from rc.stage2_crossfit_dataset import Stage2CurveLogitView

    if _ragged_curve_size(value, field=field) != expected_size:
        raise Stage2CalibratorContractError("ragged exact curve columns misalign")
    if isinstance(value, Stage2CurveLogitView):
        selected = torch.from_numpy(np.asarray(value[indices], dtype=np.float64))
    elif isinstance(value, np.ndarray):
        selected = torch.from_numpy(np.asarray(value[indices], dtype=np.float64))
    else:
        selected = value.index_select(0, torch.from_numpy(indices)).to(dtype=torch.float64)
    selected = selected.to(device=device, dtype=torch.float64)
    if not bool(torch.isfinite(selected).all().item()):
        raise Stage2CalibratorContractError(f"{field} selected non-finite values")
    return selected


def compact_exact_curve_brackets(
    threshold_logits: torch.Tensor,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather only exact curve segments touched by the current three logits.

    Segment lookup is performed against the complete ragged CPU curve using a
    detached copy of the current logits.  The selected left/right event union
    has at most ``2*J`` points per episode; interpolation remains exactly the
    same piecewise-linear function and gradients flow through the live logits.
    """

    if threshold_logits.ndim != 2 or threshold_logits.shape[1] != 3:
        raise ValueError("threshold_logits must have shape [B,3]")
    if not bool(torch.isfinite(threshold_logits).all().item()):
        raise ValueError("threshold_logits must be finite")
    curve_fields = ("curve_logits", "curve_pixel_risk", "curve_pd")
    values_by_field: dict[str, Sequence[Any]] = {}
    for field in curve_fields:
        values = batch.get(field)
        if isinstance(values, torch.Tensor) or not isinstance(values, (tuple, list)):
            raise Stage2CalibratorContractError(
                f"{field} must be a ragged CPU tuple, never one padded tensor"
            )
        if len(values) != threshold_logits.shape[0]:
            raise Stage2CalibratorContractError(f"{field} batch length mismatch")
        values_by_field[field] = values
    eta = threshold_logits.detach().to(device="cpu", dtype=torch.float64).numpy()
    selected_indices: list[np.ndarray] = []
    curve_sizes: list[int] = []
    for row, raw_curve in enumerate(values_by_field["curve_logits"]):
        curve_sizes.append(_ragged_curve_size(raw_curve, field="curve_logits"))
        selected_indices.append(_curve_bracket_union(raw_curve, eta[row]))
    max_points = max(index.size for index in selected_indices)
    device = threshold_logits.device
    compact: dict[str, torch.Tensor] = {
        field: torch.zeros(
            (threshold_logits.shape[0], max_points),
            dtype=torch.float64,
            device=device,
        )
        for field in curve_fields
    }
    valid = torch.zeros(
        (threshold_logits.shape[0], max_points), dtype=torch.bool, device=device
    )
    for row, indices in enumerate(selected_indices):
        valid[row, : indices.size] = True
        for field in curve_fields:
            raw = values_by_field[field][row]
            selected = _select_ragged_curve_values(
                raw,
                indices,
                field=field,
                expected_size=curve_sizes[row],
                device=device,
            )
            compact[field][row, : indices.size] = selected
    return (
        compact["curve_logits"],
        compact["curve_pixel_risk"],
        compact["curve_pd"],
        valid,
    )


def _batch_loss(
    method: str,
    model: nn.Module,
    batch: Mapping[str, Any],
    loss_config: Mapping[str, Any],
) -> tuple[Any, dict[str, torch.Tensor]]:
    output = model(batch["features"])
    if output.grid_logits.shape != batch["oracle_logits"].shape:
        raise RuntimeError("model output/oracle supervision shape mismatch")
    if method in ("T6", "T7"):
        oracle = oracle_logit_huber_loss(
            output.grid_logits,
            batch["oracle_logits"],
            torch.ones_like(batch["oracle_logits"], dtype=torch.bool),
            delta=loss_config["oracle_huber_delta"],
        )
        zero = oracle * 0.0
        metrics = {
            "total": oracle,
            "violation": zero,
            "utility": zero,
            "oracle_logit": oracle,
            "curve_smoothness": zero,
            "coverage_penalty": zero,
        }
        return output, metrics
    if method != "T8":
        raise Stage2CalibratorContractError("unsupported loss method")
    batch_size = int(batch["features"].shape[0])
    curve_logits, curve_pixel_risk, curve_pd, curve_valid = compact_exact_curve_brackets(
        output.grid_logits, batch
    )
    if not bool(curve_valid[:, 0].all().item()):
        raise Stage2CalibratorContractError("exact curve has no lower endpoint")
    loss = curve_query_risk_aligned_calibrator_loss(
        output.grid_logits,
        batch["pixel_budgets"],
        batch["oracle_logits"],
        curve_logits,
        curve_pixel_risk,
        curve_pd,
        curve_valid,
        curve_logits[:, 0],
        torch.ones(batch_size, dtype=torch.bool, device=output.grid_logits.device),
        oracle_valid=torch.ones_like(batch["oracle_logits"], dtype=torch.bool),
        utility_episode_valid=batch["curve_gt_objects"] > 0,
        lambda_violation=loss_config["lambda_violation"],
        lambda_utility=loss_config["lambda_utility"],
        lambda_oracle_logit=loss_config["lambda_oracle"],
        lambda_curve_smoothness=loss_config["lambda_smoothness"],
        lambda_coverage=loss_config["lambda_coverage"],
        epsilon=loss_config["risk_epsilon"],
        oracle_huber_delta=loss_config["oracle_huber_delta"],
    )
    return output, {name: getattr(loss, name) for name in LOSS_METRIC_NAMES}


def _train_epoch(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    loss_config: Mapping[str, Any],
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in LOSS_METRIC_NAMES}
    count = 0
    for raw in loader:
        batch = _move_batch(raw, device, method=method)
        optimizer.zero_grad(set_to_none=True)
        _, losses = _batch_loss(method, model, batch, loss_config)
        losses["total"].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not bool(torch.isfinite(torch.as_tensor(norm)).item()):
            raise FloatingPointError("non-finite calibrator gradient norm")
        optimizer.step()
        size = int(batch["features"].shape[0])
        count += size
        for name in LOSS_METRIC_NAMES:
            value = float(losses[name].detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite training loss: {name}")
            totals[name] += value * size
    if count == 0:
        raise RuntimeError("training DataLoader produced no rows")
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def _validation_predictions(
    method: str,
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: Mapping[str, Any],
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    totals = {name: 0.0 for name in LOSS_METRIC_NAMES}
    count = 0
    thresholds: list[np.ndarray] = []
    for raw in loader:
        batch = _move_batch(raw, device, method=method)
        output, losses = _batch_loss(method, model, batch, loss_config)
        size = int(batch["features"].shape[0])
        count += size
        for name in LOSS_METRIC_NAMES:
            value = float(losses[name].detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite validation loss: {name}")
            totals[name] += value * size
        thresholds.append(output.grid_thresholds.detach().cpu().numpy())
    if count == 0:
        raise RuntimeError("validation DataLoader produced no rows")
    return (
        {name: value / count for name, value in totals.items()},
        np.concatenate(thresholds, axis=0),
    )


def _curve_row_index(curve_thresholds: np.ndarray, threshold: float) -> int:
    values = np.asarray(curve_thresholds, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0.0):
        raise Stage2CalibratorContractError("exact curve thresholds are not ascending")
    if values[0] != 0.0 or values[-1] != 1.0:
        raise Stage2CalibratorContractError("exact curve endpoints must be 0 and 1")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise Stage2CalibratorContractError("predicted threshold is invalid")
    return max(0, min(values.size - 1, int(np.searchsorted(values, threshold, side="right") - 1)))


def evaluate_source_validation_primary(
    validation: Any,
    groups: Sequence[Any],
    thresholds: np.ndarray,
    replay_capability: Any,
    *,
    selection_budget: float = 1e-5,
) -> dict[str, Any]:
    """Private training replay; it cannot mint a Lane-C sealed decision."""

    if type(selection_budget) is not float or selection_budget != 1e-5:
        raise Stage2CalibratorContractError(
            "checkpoint selection budget must be exact frozen 1e-5"
        )
    from rc.stage2_crossfit_dataset import assert_stage2_trainer_replay_capability

    assert_stage2_trainer_replay_capability(replay_capability, validation)
    eta = np.asarray(thresholds, dtype=np.float64)
    if eta.shape != (len(groups), 3) or not np.isfinite(eta).all() or np.any(
        (eta < 0.0) | (eta > 1.0)
    ):
        raise Stage2CalibratorContractError("validation thresholds must be finite [N,3]")
    budgets = np.asarray([1e-4, 1e-5, 1e-6], dtype=np.float64)
    per_domain: dict[str, dict[str, Any]] = {}
    window_rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        domain = str(group.episode.payload["source_domain"])
        risk_values: list[float] = []
        pd_values: list[float] = []
        fp_values: list[int] = []
        tp_values: list[int] = []
        for budget_index in range(3):
            row_index = _curve_row_index(
                group.curve_thresholds, float(eta[group_index, budget_index])
            )
            risk_values.append(float(group.curve_pixel_risk[row_index]))
            pd_values.append(float(group.curve_pd[row_index]))
            fp_values.append(int(group.curve_fp_pixels[row_index]))
            tp = int(group.curve_tp_objects[row_index])
            gt = int(group.curve_gt_objects)
            if tp < 0 or gt < 0 or tp > gt:
                raise Stage2CalibratorContractError("exact curve TP/GT counts are invalid")
            expected_pd = tp / gt if gt else 0.0
            if not math.isclose(
                pd_values[-1], expected_pd, rel_tol=0.0, abs_tol=1e-15
            ):
                raise Stage2CalibratorContractError("exact curve Pd disagrees with integer TP/GT")
            tp_values.append(tp)
        primary = 1
        row = {
            "episode_id": group.episode.episode_id,
            "window_id": group.episode.payload["window_binding"]["window_id"],
            "source_domain": domain,
            "thresholds": eta[group_index].tolist(),
            "pixel_risk": risk_values,
            "pd": pd_values,
            "fp_pixels": fp_values,
            "tp_objects": tp_values,
            "total_pixels": int(group.curve_total_pixels),
            "gt_objects": int(group.curve_gt_objects),
            "primary_satisfied": bool(risk_values[primary] <= budgets[primary]),
            "primary_log_excess": math.log(
                max(risk_values[primary] / budgets[primary], 1.0)
            ),
        }
        window_rows.append(row)
        aggregate = per_domain.setdefault(
            domain,
            {"satisfied": [], "log_excess": [], "tp": 0, "gt": 0},
        )
        aggregate["satisfied"].append(float(row["primary_satisfied"]))
        aggregate["log_excess"].append(float(row["primary_log_excess"]))
        aggregate["tp"] += tp_values[primary]
        aggregate["gt"] += int(group.curve_gt_objects)
    expected_domains = set(OUTER_TARGETS.values()) - {
        OUTER_TARGETS[str(validation.manifest["outer_fold_id"])]
    }
    if set(per_domain) != expected_domains:
        raise Stage2CalibratorContractError("validation replay source domains are incomplete")
    domain_metrics: dict[str, dict[str, Any]] = {}
    for domain in sorted(per_domain):
        values = per_domain[domain]
        if not values["satisfied"] or values["gt"] <= 0:
            raise Stage2CalibratorContractError("source domain replay is not estimable")
        domain_metrics[domain] = {
            "BSR": float(np.mean(values["satisfied"])),
            "LogExcess": float(np.mean(values["log_excess"])),
            "Pd": float(values["tp"] / values["gt"]),
            "tp_objects": int(values["tp"]),
            "gt_objects": int(values["gt"]),
        }
    metrics = {
        "selection_pixel_budget": 1e-5,
        "selection_budget_index": 1,
        "source_domain_weighting": "equal_one_half",
        "within_domain_BSR_LogExcess": "equal_mandatory_window_mean",
        "within_domain_Pd": "pooled_tp_divided_by_pooled_gt",
        "macro_source_BSR": sum(item["BSR"] for item in domain_metrics.values()) / 2.0,
        "macro_source_LogExcess": sum(
            item["LogExcess"] for item in domain_metrics.values()
        )
        / 2.0,
        "macro_source_Pd": sum(item["Pd"] for item in domain_metrics.values()) / 2.0,
        "domain_metrics": domain_metrics,
        "complete_three_budget_window_records": window_rows,
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }
    for field in ("macro_source_BSR", "macro_source_LogExcess", "macro_source_Pd"):
        if not math.isfinite(float(metrics[field])):
            raise FloatingPointError(f"non-finite exact replay metric: {field}")
    return metrics


def _history_bytes(history: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(dict(row)) + b"\n" for row in history)


def _replay_bytes(replay: Mapping[str, Any]) -> bytes:
    return _pretty_json_bytes(dict(replay))


def runtime_environment_contract(device: torch.device) -> dict[str, Any]:
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device_type": device.type,
        "cuda_runtime_version": str(torch.version.cuda) if device.type == "cuda" else None,
        "deterministic_algorithms": True,
        "amp": False,
        "num_workers": 0,
    }


def source_release_contract(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    relative_paths = (
        "model/direct_no_reject_pixel_calibrator.py",
        "model/monotone_pixel_calibrator.py",
        "losses/calibrator_risk.py",
        "rc/train_stage2_crossfit_calibrator.py",
        "rc/stage2_crossfit_schema.py",
        "rc/stage2_crossfit_dataset.py",
    )
    files: list[dict[str, str]] = []
    for relative in relative_paths:
        path = root / relative
        if path.resolve(strict=True) != path or not path.is_file() or path.is_symlink():
            raise Stage2CalibratorContractError(f"release source is not canonical: {relative}")
        files.append({"repository_relative_path": relative, "sha256": sha256_file(path)})
    return {
        "schema_version": "rc-irstd.calibrator-source-release.v1",
        "files": files,
        "files_sha256": sha256_bytes(_canonical_json_bytes(files)),
    }


def build_training_contract(
    *,
    method: str,
    config: VerifiedStage2CrossfitConfig,
    seed: VerifiedStage2SeedSelection,
    inputs: VerifiedLaneATrainingInputs,
    environment: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    if type(config) is not VerifiedStage2CrossfitConfig:
        raise TypeError("verified config required")
    if type(seed) is not VerifiedStage2SeedSelection:
        raise TypeError("verified seed selection required")
    if type(inputs) is not VerifiedLaneATrainingInputs:
        raise TypeError("verified Lane-A inputs required")
    payload = config.payload
    return {
        "schema_version": "rc-irstd.calibrator-training-contract.v1",
        "method": method,
        "config_sha256": config.sha256,
        "outer_fold_id": seed.outer_fold_id,
        "outer_target": OUTER_TARGETS[seed.outer_fold_id],
        "seed": seed.to_dict(),
        "statistics_config": dict(inputs.statistics_config_binding),
        "train_collection": dict(inputs.train_binding),
        "validation_collection": dict(inputs.validation_binding),
        "collection_transitive_bindings": collection_transitive_bindings(inputs),
        "standardizer_fit_manifest_sha256": _validate_sha256(
            inputs.standardizer.fit_manifest_sha256,
            "standardizer_fit_manifest_sha256",
        ),
        "environment_sha256": sha256_bytes(_canonical_json_bytes(environment)),
        "release_sha256": sha256_bytes(_canonical_json_bytes(release)),
        "optimizer_and_schedule": dict(payload["optimizer"]),
        "loss": dict(payload["loss"]),
        "checkpoint_selection": dict(payload["checkpoint_selection"]),
        "selection_pixel_budget": 1e-5,
        "reject_head": False,
        "missing_episode_fallback": False,
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }


def _generation_binding(generation: VerifiedCalibratorGeneration) -> dict[str, Any]:
    return {
        "epoch": generation.epoch,
        "commit_sha256": generation.commit_sha256,
        "checkpoint_sha256": generation.checkpoint_sha256,
        "history_sha256": generation.history_sha256,
        "exact_replay_sha256": generation.replay_sha256,
    }


def publish_calibrator_run_commit(
    output_dir: str | Path,
    *,
    method: str,
    best: VerifiedCalibratorGeneration,
    last: VerifiedCalibratorGeneration,
    training_contract: Mapping[str, Any],
    environment: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(output_dir).expanduser()
    if not root.exists():
        root.mkdir(mode=0o700, parents=False)
    _assert_canonical_output_parent(root)
    member_payloads = {
        "training_contract.json": _pretty_json_bytes(dict(training_contract)),
        "environment.json": _pretty_json_bytes(dict(environment)),
        "release.json": _pretty_json_bytes(dict(release)),
    }
    members = {
        name: {"sha256": sha256_bytes(data), "size_bytes": len(data)}
        for name, data in member_payloads.items()
    }
    run_commit = {
        "schema_version": RUN_COMMIT_SCHEMA,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA,
        "method": method,
        "best_generation": _generation_binding(best),
        "last_generation": _generation_binding(last),
        "members": members,
        "checkpoint_selection": [
            "macro_source_BSR_max",
            "macro_source_LogExcess_min",
            "macro_source_Pd_max",
            "earlier_epoch_on_exact_tie",
        ],
        "selection_pixel_budget": 1e-5,
        "reject_head": False,
        "official_test_accessed": False,
        "commit_published_last": True,
    }
    commit_bytes = _pretty_json_bytes(run_commit)
    commit_sha = sha256_bytes(commit_bytes)
    files: dict[Path, bytes] = {}
    for name, data in member_payloads.items():
        files[root / name] = data
        files[root / f"{name}.sha256"] = (
            f"{members[name]['sha256']}  {name}\n".encode("ascii")
        )
    commit_path = root / "RUN_COMMIT.json"
    files[root / "RUN_COMMIT.json.sha256"] = (
        f"{commit_sha}  RUN_COMMIT.json\n".encode("ascii")
    )
    files[commit_path] = commit_bytes
    _transactional_publish_bundle(files, commit_last=commit_path)
    _verified_regular_file(commit_path, commit_sha, "calibrator_run_commit")
    return {
        "path": str(commit_path),
        "sha256": commit_sha,
        "best_checkpoint_sha256": best.checkpoint_sha256,
        "last_checkpoint_sha256": last.checkpoint_sha256,
        "training_contract_sha256": members["training_contract.json"]["sha256"],
        "environment_sha256": members["environment.json"]["sha256"],
        "release_sha256": members["release.json"]["sha256"],
    }


def _recover_best_generation(
    resume: VerifiedCalibratorGeneration,
    *,
    best_epoch: int,
    history: Sequence[Mapping[str, Any]],
) -> VerifiedCalibratorGeneration:
    if best_epoch == resume.epoch:
        return resume
    if not 0 <= best_epoch < resume.epoch:
        raise Stage2CalibratorContractError("resume best epoch is invalid")
    best_sha = history[best_epoch].get("generation_commit_sha256")
    _validate_sha256(best_sha, "history best generation commit SHA-256")
    generations = resume.commit_path.parent.parent
    best_path = generations / f"epoch_{best_epoch:04d}" / "COMMIT.json"
    return verify_calibrator_generation(best_path, best_sha)


def train_stage2_crossfit_calibrator(
    *,
    method: str,
    config: VerifiedStage2CrossfitConfig,
    seed: VerifiedStage2SeedSelection,
    inputs: VerifiedLaneATrainingInputs,
    output_dir: str | Path,
    repository_root: str | Path,
    device: torch.device,
    resume_commit: str | Path | None = None,
    resume_commit_sha256: str | None = None,
) -> dict[str, Any]:
    if method not in METHODS or seed.method != method:
        raise Stage2CalibratorContractError("method/seed selection mismatch")
    if type(config) is not VerifiedStage2CrossfitConfig or type(inputs) is not VerifiedLaneATrainingInputs:
        raise TypeError("verified config and Lane-A inputs are required")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if (resume_commit is None) != (resume_commit_sha256 is None):
        raise Stage2CalibratorContractError(
            "resume commit path and external SHA-256 are jointly required"
        )
    output = Path(output_dir).expanduser()
    if not output.is_absolute() or ".." in output.parts:
        raise Stage2CalibratorContractError("output_dir must be an absolute canonical path")
    if output.parent.resolve(strict=True) != output.parent or output.parent.is_symlink():
        raise Stage2CalibratorContractError("output_dir parent must be canonical")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output_dir must be a fresh, absent directory")
    generator = seed_runtime(seed.derived_seed, include_cuda=device.type == "cuda")
    model = build_stage2_model(method, config).to(device)
    optimizer_config = config.payload["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
        amsgrad=optimizer_config["amsgrad"],
        foreach=False,
        fused=False,
    )
    from rc.stage2_crossfit_dataset import (
        Stage2CrossfitDataset,
        collate_stage2_crossfit_batch,
    )

    train_dataset = Stage2CrossfitDataset(inputs.train, inputs.standardizer)
    validation_dataset = Stage2CrossfitDataset(inputs.validation, inputs.standardizer)
    if train_dataset.input_dim != 93 or validation_dataset.input_dim != 93:
        raise Stage2CalibratorContractError("Lane-A dataset feature dimension changed")
    train_loader = DataLoader(
        train_dataset,
        batch_size=optimizer_config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_stage2_crossfit_batch,
        generator=generator,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=optimizer_config["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_stage2_crossfit_batch,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
    )
    environment = runtime_environment_contract(device)
    release = source_release_contract(repository_root)
    contract = build_training_contract(
        method=method,
        config=config,
        seed=seed,
        inputs=inputs,
        environment=environment,
        release=release,
    )
    start_epoch = 0
    best_epoch = -1
    best_rank: tuple[float, float, float, int] | None = None
    without_improvement = 0
    history: list[dict[str, Any]] = []
    best_generation: VerifiedCalibratorGeneration | None = None
    last_generation: VerifiedCalibratorGeneration | None = None
    if resume_commit is not None:
        assert resume_commit_sha256 is not None
        verified_resume = verify_calibrator_generation(
            resume_commit, resume_commit_sha256
        )
        (
            start_epoch,
            best_epoch,
            resumed_rank,
            without_improvement,
            history,
        ) = resume_calibrator_generation(
            commit_path=resume_commit,
            commit_sha256=resume_commit_sha256,
            model=model,
            optimizer=optimizer,
            data_loader_generator=generator,
            expected_method=method,
            expected_training_contract=contract,
            include_cuda_rng=device.type == "cuda",
        )
        best_rank = resumed_rank
        history[-1]["generation_commit_sha256"] = verified_resume.commit_sha256
        last_generation = verified_resume
        best_generation = _recover_best_generation(
            verified_resume, best_epoch=best_epoch, history=history
        )

    loss_config = config.payload["loss"]
    max_epochs = optimizer_config["max_epochs"]
    patience = optimizer_config["early_stopping_patience"]
    # A committed generation at the patience boundary is terminal.  Resuming
    # it may republish a run-level binding, but must never execute one extra
    # optimizer step merely because the previous process stopped before that
    # final run commit was written.
    end_epoch = (
        start_epoch
        if resume_commit is not None and without_improvement >= patience
        else max_epochs
    )
    for epoch in range(start_epoch, end_epoch):
        train_metrics = _train_epoch(
            method,
            model,
            train_loader,
            optimizer,
            device=device,
            loss_config=loss_config,
            gradient_clip_norm=optimizer_config["gradient_clip_norm"],
        )
        validation_surrogate, predicted_thresholds = _validation_predictions(
            method,
            model,
            validation_loader,
            device=device,
            loss_config=loss_config,
        )
        exact_replay = evaluate_source_validation_primary(
            inputs.validation,
            validation_dataset.groups,
            predicted_thresholds,
            inputs.replay_capability,
            selection_budget=1e-5,
        )
        exact_replay["epoch"] = epoch
        rank = checkpoint_rank(exact_replay, epoch)
        improved = is_better_checkpoint(rank, best_rank)
        if improved:
            best_rank = rank
            best_epoch = epoch
            without_improvement = 0
        else:
            without_improvement += 1
        record: dict[str, Any] = {
            "epoch": epoch,
            "method": method,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training": train_metrics,
            "validation_surrogate": validation_surrogate,
            "validation_exact_replay": exact_replay,
            "checkpoint_rank": list(rank),
            "is_best": improved,
            "official_test_accessed": False,
        }
        history.append(record)
        history_sha = sha256_bytes(_history_bytes(history))
        replay_sha = sha256_bytes(_replay_bytes(exact_replay))
        if best_rank is None or best_epoch < 0:
            raise RuntimeError("first finite checkpoint was not selected")
        checkpoint_payload = make_calibrator_checkpoint_v6(
            method=method,
            model=model,
            optimizer=optimizer,
            completed_epoch=epoch,
            best_epoch=best_epoch,
            best_rank=best_rank,
            epochs_without_improvement=without_improvement,
            training_contract=contract,
            standardizer=inputs.standardizer,
            data_loader_generator=generator,
            history_sha256=history_sha,
            exact_replay_sha256=replay_sha,
            include_cuda_rng=device.type == "cuda",
        )
        generation = publish_calibrator_generation(
            output,
            epoch=epoch,
            checkpoint_payload=checkpoint_payload,
            history=history,
            exact_replay=exact_replay,
        )
        record["generation_commit_sha256"] = generation.commit_sha256
        last_generation = generation
        if improved:
            best_generation = generation
        if without_improvement >= patience:
            break
    if best_generation is None or last_generation is None:
        raise RuntimeError("training produced no committed generation")
    return publish_calibrator_run_commit(
        output,
        method=method,
        best=best_generation,
        last=last_generation,
        training_contract=contract,
        environment=environment,
        release=release,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    for split in ("train", "validation"):
        parser.add_argument(f"--{split}-collection", required=True)
        parser.add_argument(f"--{split}-collection-sha256", required=True)
        parser.add_argument(f"--{split}-manifest", required=True)
        parser.add_argument(f"--{split}-manifest-sha256", required=True)
        parser.add_argument(f"--{split}-commit", required=True)
        parser.add_argument(f"--{split}-commit-sha256", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--seed-manifest-sha256", required=True)
    parser.add_argument("--statistics-config", required=True)
    parser.add_argument("--statistics-config-sha256", required=True)
    parser.add_argument("--base-seed", required=True, type=int, choices=(42, 123, 3407))
    parser.add_argument("--outer-fold-id", required=True, choices=tuple(OUTER_TARGETS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume-commit")
    parser.add_argument("--resume-commit-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.repository_root).expanduser().resolve(strict=True)
    config = verify_stage2_crossfit_config(args.config, args.config_sha256)
    seed = verify_stage2_seed_selection(
        args.seed_manifest,
        args.seed_manifest_sha256,
        base_seed=args.base_seed,
        outer_fold_id=args.outer_fold_id,
        method=args.method,
    )
    inputs = verify_lane_a_training_inputs(
        train_collection=args.train_collection,
        train_collection_sha256=args.train_collection_sha256,
        train_manifest=args.train_manifest,
        train_manifest_sha256=args.train_manifest_sha256,
        train_commit=args.train_commit,
        train_commit_sha256=args.train_commit_sha256,
        validation_collection=args.validation_collection,
        validation_collection_sha256=args.validation_collection_sha256,
        validation_manifest=args.validation_manifest,
        validation_manifest_sha256=args.validation_manifest_sha256,
        validation_commit=args.validation_commit,
        validation_commit_sha256=args.validation_commit_sha256,
        statistics_config_path=args.statistics_config,
        statistics_config_sha256=args.statistics_config_sha256,
        outer_fold_id=args.outer_fold_id,
        base_seed=args.base_seed,
        repository_root=root,
    )
    result = train_stage2_crossfit_calibrator(
        method=args.method,
        config=config,
        seed=seed,
        inputs=inputs,
        output_dir=args.output_dir,
        repository_root=root,
        device=torch.device(args.device),
        resume_commit=args.resume_commit,
        resume_commit_sha256=args.resume_commit_sha256,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI contract tests
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "Stage2CalibratorContractError",
    "VerifiedCalibratorCheckpointV6",
    "VerifiedCalibratorGeneration",
    "VerifiedLaneATrainingInputs",
    "VerifiedStage2CrossfitConfig",
    "VerifiedStage2SeedSelection",
    "build_stage2_model",
    "checkpoint_rank",
    "derive_stage2_seed",
    "evaluate_source_validation_primary",
    "is_better_checkpoint",
    "make_calibrator_checkpoint_v6",
    "oracle_logit_huber_loss",
    "publish_calibrator_generation",
    "resume_calibrator_generation",
    "train_stage2_crossfit_calibrator",
    "verify_calibrator_checkpoint_v6",
    "verify_calibrator_generation",
    "verify_lane_a_training_inputs",
    "verify_stage2_crossfit_config",
    "verify_stage2_seed_selection",
]
