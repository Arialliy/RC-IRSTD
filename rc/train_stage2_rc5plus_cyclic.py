"""RC5+ cyclic minibatch and optimization primitives.

This module routes the verifier-issued four-role training view into the
nine-budget T6+/T7+/T8+ loss core.  It is intentionally not yet a checkpoint
or run publication authority; those contracts are admitted separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from model.budget_conditioned_endpoint_calibrator import BUDGET_KNOT_RATIONALS
from model.budget_conditioned_residual_transport_calibrator import (
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from rc.stage2_rc5plus_cyclic_training_view import (
    VerifiedStage2RC5PlusCyclicTrainingView,
    assert_verified_stage2_rc5plus_cyclic_training_view,
)
from rc.stage2_calibrator_checkpoint_v8 import (
    make_calibrator_checkpoint_v8,
    serialize_calibrator_checkpoint_v8,
)
from rc.stage2_domain_balanced_cyclic_sampler import (
    assert_verified_domain_balanced_cyclic_epoch,
    build_domain_balanced_cyclic_epoch,
    verify_domain_balanced_cyclic_epoch,
)
from rc.stage2_rc5plus_calibrator_generation_v3 import (
    VerifiedRC5PlusCalibratorGenerationV3,
    VerifiedRC5PlusCalibratorRunV3,
    build_resume_state_v3,
    input_identity_sha256_v3,
    normalize_input_bindings_v3,
    publish_rc5plus_calibrator_generation_v3,
    publish_rc5plus_calibrator_run_v3,
    verify_rc5plus_calibrator_generation_v3,
)
from rc.stage2_rc5plus_source_validation_view import (
    VerifiedStage2RC5PlusSourceValidationView,
    VerifiedStage2RC5PlusVariableQuerySanityView,
    assert_verified_stage2_rc5plus_source_validation_view,
    assert_verified_stage2_rc5plus_variable_query_sanity_view,
    evaluate_stage2_rc5plus_source_validation_view,
    evaluate_stage2_rc5plus_variable_query_sanity_view,
)
from rc.stage2_rc5_feature_mask import (
    VerifiedStage2RC5FeatureMask,
    apply_stage2_rc5_feature_mask_numpy,
    assert_verified_stage2_rc5_feature_mask,
)
from rc.stage2_rc5plus_training_core import (
    RC5PLUS_LOSS_METRIC_NAMES,
    RC5PLUS_METHODS,
    RC5PLUS_TRAINING_METHODS,
    rc5plus_batch_loss,
)


RC5PLUS_CYCLIC_TRAINER_SCHEMA = "rc-irstd.stage2-rc5plus-cyclic-trainer.v1"
_EXECUTION_SPEC_CAPABILITY = object()


class Stage2RC5PlusCyclicTrainerError(ValueError):
    """A verified view, balanced batch or optimizer step is invalid."""


def build_rc5plus_training_model(
    method: str,
    model_config: Mapping[str, Any],
) -> torch.nn.Module:
    if method not in RC5PLUS_TRAINING_METHODS:
        raise Stage2RC5PlusCyclicTrainerError(
            "method is not a frozen RC5+ training route"
        )
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")
    required = {
        "context_feature_dim",
        "hidden_dims",
        "dropout",
        "minimum_residual_increment",
    }
    if set(model_config) != required:
        raise Stage2RC5PlusCyclicTrainerError(
            "RC5+ model_config field closure mismatch"
        )
    kwargs = {
        "context_feature_dim": model_config["context_feature_dim"],
        "hidden_dims": model_config["hidden_dims"],
        "dropout": model_config["dropout"],
        "minimum_residual_increment": model_config[
            "minimum_residual_increment"
        ],
    }
    if method == "T6_PLUS":
        return BudgetConditionedDirectResidualTransportCalibrator(**kwargs)
    if method == "T8_PLUS_NO_ANCHOR":
        return BudgetConditionedMonotoneNoTargetAnchorCalibrator(**kwargs)
    return BudgetConditionedMonotoneResidualTransportCalibrator(**kwargs)


def _budget_tensors(
    batch_size: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    numerators = torch.tensor(
        [row[0] for row in BUDGET_KNOT_RATIONALS],
        dtype=torch.int64,
        device=device,
    ).reshape(1, -1).repeat(batch_size, 1)
    denominators = torch.tensor(
        [row[1] for row in BUDGET_KNOT_RATIONALS],
        dtype=torch.int64,
        device=device,
    ).reshape(1, -1).repeat(batch_size, 1)
    return numerators, denominators


def collate_rc5plus_cyclic_batch(
    *,
    collection: VerifiedStage2RC5PlusCyclicTrainingView,
    ordered_rows: Sequence[Mapping[str, Any]],
    standardizer_mean: np.ndarray,
    standardizer_scale: np.ndarray,
    feature_mask: VerifiedStage2RC5FeatureMask,
    device: str | torch.device,
    method: str,
) -> dict[str, Any]:
    """Collate one exactly domain-balanced cyclic batch without route leakage."""

    view = assert_verified_stage2_rc5plus_cyclic_training_view(collection)
    verified_feature_mask = assert_verified_stage2_rc5_feature_mask(feature_mask)
    if method not in RC5PLUS_TRAINING_METHODS:
        raise Stage2RC5PlusCyclicTrainerError(
            "method is not a frozen RC5+ training route"
        )
    if (
        isinstance(ordered_rows, (str, bytes))
        or not isinstance(ordered_rows, Sequence)
        or not ordered_rows
        or len(ordered_rows) % 2
    ):
        raise Stage2RC5PlusCyclicTrainerError(
            "ordered_rows must be a nonempty even sequence"
        )
    for value, name in (
        (standardizer_mean, "standardizer_mean"),
        (standardizer_scale, "standardizer_scale"),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != (93,)
            or not np.isfinite(value).all()
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                f"{name} must be finite float64[93]"
            )
    if np.any(standardizer_scale <= 0.0):
        raise Stage2RC5PlusCyclicTrainerError(
            "standardizer_scale must be strictly positive"
        )

    features: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    providers: list[Any] = []
    oracle_coordinates: list[np.ndarray] = []
    domains: list[str] = []
    for row in ordered_rows:
        required_row_fields = {"source_domain", "domain_episode_index"}
        if not isinstance(row, Mapping) or not required_row_fields.issubset(row):
            raise Stage2RC5PlusCyclicTrainerError(
                "sampler row lacks source_domain/domain_episode_index"
            )
        domain = str(row["source_domain"])
        index = row["domain_episode_index"]
        if type(index) is not int:
            raise Stage2RC5PlusCyclicTrainerError(
                "domain_episode_index must be an exact integer"
            )
        if method == "T8_PLUS_NO_ANCHOR":
            feature = view.feature_for_episode(domain, index)
            anchor = None
        else:
            feature, anchor = view.feature_anchor_for_episode(domain, index)
        provider = view.provider_for_episode(domain, index)
        features.append(np.asarray(feature, dtype=np.float32))
        if anchor is not None:
            anchors.append(np.asarray(anchor, dtype=np.float64))
        providers.append(provider)
        domains.append(domain)
        if method in {"T6_PLUS", "T7_PLUS"}:
            oracle_coordinates.append(
                np.asarray(
                    provider.select_exact_oracle_rows_v2().coordinates,
                    dtype=np.float64,
                )
            )
    counts = {domain: domains.count(domain) for domain in set(domains)}
    if len(counts) != 2 or len(set(counts.values())) != 1:
        raise Stage2RC5PlusCyclicTrainerError(
            "every RC5+ minibatch must be exactly balanced across two domains"
        )
    feature_matrix = np.stack(features).astype(np.float64)
    if feature_matrix.shape != (len(ordered_rows), 93):
        raise Stage2RC5PlusCyclicTrainerError(
            "cyclic feature batch must have shape [B,93]"
        )
    anchor_matrix: np.ndarray | None = None
    if method != "T8_PLUS_NO_ANCHOR":
        anchor_matrix = np.stack(anchors).astype(np.float64)
        if anchor_matrix.shape != (
            len(ordered_rows),
            len(BUDGET_KNOT_RATIONALS),
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                "cyclic anchor batch must have shape [B,9]"
            )
    standardized = (
        (feature_matrix - standardizer_mean) / standardizer_scale
    ).astype(np.float32)
    standardized = apply_stage2_rc5_feature_mask_numpy(
        standardized, verified_feature_mask
    )
    resolved_device = torch.device(device)
    numerators, denominators = _budget_tensors(
        len(ordered_rows), device=resolved_device
    )
    batch: dict[str, Any] = {
        "features": torch.from_numpy(standardized).to(device=resolved_device),
        "budget_numerators": numerators,
        "budget_denominators": denominators,
    }
    if anchor_matrix is not None:
        batch["anchor_coordinates"] = torch.from_numpy(anchor_matrix).to(
            device=resolved_device
        )
    if method in {"T6_PLUS", "T7_PLUS"}:
        oracle_matrix = np.stack(oracle_coordinates).astype(np.float64)
        assert anchor_matrix is not None
        if oracle_matrix.shape != anchor_matrix.shape:
            raise Stage2RC5PlusCyclicTrainerError(
                "provider oracle batch must have shape [B,9]"
            )
        batch["oracle_coordinates"] = torch.from_numpy(oracle_matrix).to(
            device=resolved_device
        )
    else:
        batch["compositional_curve_providers"] = tuple(providers)
    return batch


def rc5plus_cyclic_optimization_step(
    *,
    method: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    gradient_clip_norm: float,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Execute one finite deterministic-compatible optimizer step."""

    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if isinstance(gradient_clip_norm, bool):
        raise TypeError("gradient_clip_norm must be a real number")
    clip = float(gradient_clip_norm)
    if not math.isfinite(clip) or clip <= 0.0:
        raise Stage2RC5PlusCyclicTrainerError(
            "gradient_clip_norm must be finite and positive"
        )
    optimizer.zero_grad(set_to_none=True)
    output, losses = rc5plus_batch_loss(
        method=method,
        model=model,
        batch=batch,
        loss_config=loss_config,
    )
    if tuple(losses) != RC5PLUS_LOSS_METRIC_NAMES or not all(
        bool(torch.isfinite(value).item()) for value in losses.values()
    ):
        raise FloatingPointError("RC5+ training losses are non-finite or incomplete")
    losses["total"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not bool(torch.isfinite(gradient_norm).item()):
        raise FloatingPointError("RC5+ gradient norm is non-finite")
    optimizer.step()
    return output, losses


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, init=False)
class VerifiedRC5PlusTrainingExecutionSpec:
    payload: Mapping[str, Any]
    training_contract_sha256: str
    config_source_sha256: str
    artifact_scope: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("RC5+ training execution specs are verifier-issued only")


def _issue_execution_spec(
    payload: Mapping[str, Any],
    *,
    training_contract_sha256: str,
    config_source_sha256: str,
    artifact_scope: str,
) -> VerifiedRC5PlusTrainingExecutionSpec:
    value = object.__new__(VerifiedRC5PlusTrainingExecutionSpec)
    for name, item in {
        "payload": _freeze(payload),
        "training_contract_sha256": training_contract_sha256,
        "config_source_sha256": config_source_sha256,
        "artifact_scope": artifact_scope,
        "_capability": _EXECUTION_SPEC_CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return value


def build_synthetic_rc5plus_training_execution_spec(
    *,
    max_epochs: int = 2,
    early_stopping_patience: int = 20,
) -> VerifiedRC5PlusTrainingExecutionSpec:
    """Issue a short CPU-only spec for resume/publication contract tests."""

    if type(max_epochs) is not int or not 1 <= max_epochs <= 4:
        raise Stage2RC5PlusCyclicTrainerError(
            "synthetic max_epochs must be an exact integer in [1,4]"
        )
    if (
        type(early_stopping_patience) is not int
        or early_stopping_patience < 1
    ):
        raise Stage2RC5PlusCyclicTrainerError(
            "synthetic early-stopping patience must be positive"
        )
    payload = {
        "schema_version": RC5PLUS_CYCLIC_TRAINER_SCHEMA,
        "artifact_scope": "synthetic_cpu_contract_test",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "amsgrad": False,
            "scheduler": "none",
            "batch_size": 16,
            "max_epochs": max_epochs,
            "early_stopping_patience": early_stopping_patience,
            "gradient_clip_norm": 5.0,
            "num_workers": 0,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "amp": False,
            "dataloader_shuffle": False,
            "drop_last": False,
            "custom_verified_epoch_sampler": True,
            "data_iteration": "custom_loop_verified_sampler_no_dataloader",
            "generation_dataloader_rng_field_usage": (
                "consumed_custom_loop_generator_state"
            ),
            "process_rank": 0,
            "world_size": 1,
        },
        "loss": {
            "coordinate_huber_delta": 1.0,
            "lambda_violation": 4.0,
            "lambda_utility": 1.0,
            "lambda_oracle": 0.1,
            "lambda_smoothness": 0.01,
            "lambda_coverage": 0.0,
            "risk_epsilon": 1e-12,
        },
        "model": {
            "context_feature_dim": 93,
            "hidden_dims": [32],
            "dropout": 0.1,
            "minimum_residual_increment": 1e-6,
        },
        "source_selection": (
            "source_validation_cyclic_selection_view_c14_q28_"
            "all_n_starts_nine_budget"
        ),
        "primary_selection_budget": [1, 100000],
        "nonprimary_budget_epoch_rescue": False,
        "source_variable_query_sanity_mandatory": True,
        "source_variable_query_sanity_excluded_from_ranking": True,
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return _issue_execution_spec(
        payload,
        training_contract_sha256=digest,
        config_source_sha256=digest,
        artifact_scope="synthetic_cpu_contract_test",
    )


def build_rc5plus_training_execution_spec_from_verified_config(
    verified_config: Any,
) -> VerifiedRC5PlusTrainingExecutionSpec:
    """Promote only the unique, file-backed and verifier-issued RC5+ freeze."""

    from rc.stage2_rc5plus_frozen_config import (
        assert_verified_stage2_rc5plus_frozen_config,
    )

    config = assert_verified_stage2_rc5plus_frozen_config(verified_config)
    if config.source_path is None or config.source_bytes_sha256 is None:
        raise Stage2RC5PlusCyclicTrainerError(
            "production execution requires a file-backed RC5+ configuration"
        )
    source = config.payload
    loss = source["loss"]
    model = source["model"]
    payload = {
        "schema_version": RC5PLUS_CYCLIC_TRAINER_SCHEMA,
        "artifact_scope": "production",
        "optimizer": _plain(source["optimizer"]),
        "loss": {
            "coordinate_huber_delta": loss["coordinate_huber_delta"],
            "lambda_violation": loss["lambda_violation"],
            "lambda_utility": loss["lambda_utility"],
            "lambda_oracle": loss["lambda_oracle"],
            "lambda_smoothness": loss["lambda_smoothness"],
            "lambda_coverage": loss["lambda_coverage"],
            "risk_epsilon": loss["risk_epsilon"],
        },
        "model": {
            "context_feature_dim": source["context_feature_dim"],
            "hidden_dims": _plain(model["hidden_dims"]),
            "dropout": model["dropout"],
            "minimum_residual_increment": model[
                "minimum_residual_increment"
            ],
        },
        "source_selection": source["source_validation_contract"][
            "selection_geometry"
        ],
        "primary_selection_budget": _plain(
            source["source_validation_contract"]["selection_budget"]
        ),
        "nonprimary_budget_epoch_rescue": source[
            "source_validation_contract"
        ]["nonprimary_budget_epoch_rescue"],
        "source_variable_query_sanity_mandatory": True,
        "source_variable_query_sanity_excluded_from_ranking": source[
            "source_validation_contract"
        ]["variable_query_sanity_excluded_from_epoch_ranking"],
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }
    return _issue_execution_spec(
        payload,
        training_contract_sha256=config.canonical_sha256,
        config_source_sha256=config.source_bytes_sha256,
        artifact_scope="production",
    )


def assert_verified_rc5plus_training_execution_spec(
    value: object,
) -> VerifiedRC5PlusTrainingExecutionSpec:
    if (
        type(value) is not VerifiedRC5PlusTrainingExecutionSpec
        or getattr(value, "_capability", None)
        is not _EXECUTION_SPEC_CAPABILITY
    ):
        raise TypeError("a verifier-issued RC5+ training execution spec is required")
    return value


@dataclass(frozen=True)
class RC5PlusTrainingOutcome:
    generations: tuple[VerifiedRC5PlusCalibratorGenerationV3, ...]
    run: VerifiedRC5PlusCalibratorRunV3 | None
    history: tuple[Mapping[str, Any], ...]
    interrupted_after_epoch: int | None


def rc5plus_standardizer_identity_sha256(
    mean: np.ndarray, scale: np.ndarray
) -> str:
    for value, name in ((mean, "mean"), (scale, "scale")):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != (93,)
            or not np.isfinite(value).all()
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                f"standardizer {name} must be finite float64[93]"
            )
    if np.any(scale <= 0.0):
        raise Stage2RC5PlusCyclicTrainerError(
            "standardizer scale must be strictly positive"
        )
    digest = hashlib.sha256()
    digest.update(b"rc-irstd.stage2-rc5plus-standardizer-input.v1\0")
    for value in (mean, scale):
        raw = value.astype("<f8", copy=False).tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def rc5plus_source_validation_identity_sha256(
    selection_view: VerifiedStage2RC5PlusSourceValidationView,
    sanity_view: VerifiedStage2RC5PlusVariableQuerySanityView,
) -> str:
    selection = assert_verified_stage2_rc5plus_source_validation_view(
        selection_view
    )
    sanity = assert_verified_stage2_rc5plus_variable_query_sanity_view(
        sanity_view
    )
    if (
        selection.base_view.outer_fold_id
        != sanity.base_view.outer_fold_id
        or selection.artifact_scope != sanity.artifact_scope
    ):
        raise Stage2RC5PlusCyclicTrainerError(
            "RC5+ source-validation identities cannot combine"
        )
    payload = {
        "schema_version": (
            "rc-irstd.stage2-rc5plus-source-validation-combined-identity.v1"
        ),
        "outer_fold_id": selection.base_view.outer_fold_id,
        "selection_view_identity_sha256": selection.identity_sha256,
        "variable_query_sanity_view_identity_sha256": sanity.identity_sha256,
        "geometries_interchangeable": False,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _python_rng_state() -> dict[str, Any]:
    version, state, gauss = random.getstate()
    return {
        "version": version,
        "internal_state": list(state),
        "gauss_next": gauss,
    }


def _restore_python_rng(value: Mapping[str, Any]) -> None:
    random.setstate(
        (
            int(value["version"]),
            tuple(value["internal_state"]),
            value["gauss_next"],
        )
    )


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {
        "bit_generator": name,
        "keys": keys.astype(np.uint32).tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian_hex": float(cached).hex(),
    }


def _restore_numpy_rng(value: Mapping[str, Any]) -> None:
    np.random.set_state(
        (
            str(value["bit_generator"]),
            np.asarray(value["keys"], dtype=np.uint32),
            int(value["position"]),
            int(value["has_gauss"]),
            float.fromhex(str(value["cached_gaussian_hex"])),
        )
    )


def _selection_key(
    record: Mapping[str, Any], epoch: int
) -> tuple[float, float, float, int]:
    return (
        -float.fromhex(str(record["macro_source_bsr_hex"])),
        float.fromhex(str(record["macro_source_log_excess_hex"])),
        -float.fromhex(str(record["macro_source_pd_hex"])),
        epoch,
    )


def _validate_full_inputs(
    *,
    collection: Any,
    selection_view: Any,
    sanity_view: Any,
    feature_mask: Any,
    execution_spec: Any,
    device: torch.device,
) -> tuple[
    VerifiedStage2RC5PlusCyclicTrainingView,
    VerifiedStage2RC5PlusSourceValidationView,
    VerifiedStage2RC5PlusVariableQuerySanityView,
    VerifiedStage2RC5FeatureMask,
    VerifiedRC5PlusTrainingExecutionSpec,
]:
    train = assert_verified_stage2_rc5plus_cyclic_training_view(collection)
    selection = assert_verified_stage2_rc5plus_source_validation_view(
        selection_view
    )
    sanity = assert_verified_stage2_rc5plus_variable_query_sanity_view(
        sanity_view
    )
    mask = assert_verified_stage2_rc5_feature_mask(feature_mask)
    spec = assert_verified_rc5plus_training_execution_spec(execution_spec)
    folds = {
        str(train.manifest["outer_fold_id"]),
        str(selection.base_view.outer_fold_id),
        str(sanity.base_view.outer_fold_id),
    }
    if len(folds) != 1:
        raise Stage2RC5PlusCyclicTrainerError(
            "training/selection/sanity outer folds differ"
        )
    scopes = {
        train.artifact_scope,
        selection.artifact_scope,
        sanity.artifact_scope,
        spec.artifact_scope,
    }
    if len(scopes) != 1:
        raise Stage2RC5PlusCyclicTrainerError(
            "production and synthetic capabilities cannot mix"
        )
    if spec.artifact_scope == "synthetic_cpu_contract_test" and device.type != "cpu":
        raise Stage2RC5PlusCyclicTrainerError(
            "synthetic RC5+ execution is CPU-only"
        )
    selection_base = selection.base_view
    sanity_base = sanity.base_view
    for field, training_values in train.boundary_values.items():
        if training_values.intersection(selection_base.boundary_values[field]):
            raise Stage2RC5PlusCyclicTrainerError(
                f"training/selection overlap at identity boundary {field}"
            )
        if training_values.intersection(sanity_base.boundary_values[field]):
            raise Stage2RC5PlusCyclicTrainerError(
                f"training/sanity overlap at identity boundary {field}"
            )
    if spec.artifact_scope == "production":
        if dict(selection_base.boundary_values) != dict(
            sanity_base.boundary_values
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                "production selection and sanity boundaries differ"
            )
        if dict(selection_base.upstream_bindings) != dict(
            sanity_base.upstream_bindings
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                "production selection and sanity upstream bindings differ"
            )
        actual = train.manifest["actual_input_binding_identities"]
        for actual_name, upstream_name in {
            "statistics_config": "statistics_config_sha256",
            "source_reference": "source_reference_sha256",
            "seed_manifest": "seed_manifest_sha256",
            "source_release": "source_release_sha256",
        }.items():
            if actual[actual_name] != selection_base.upstream_bindings[
                upstream_name
            ]:
                raise Stage2RC5PlusCyclicTrainerError(
                    f"training/validation {actual_name} identity differs"
                )
    return train, selection, sanity, mask, spec


def _validate_full_input_bindings(
    *,
    bindings: Mapping[str, Mapping[str, str]],
    train: VerifiedStage2RC5PlusCyclicTrainingView,
    selection: VerifiedStage2RC5PlusSourceValidationView,
    sanity: VerifiedStage2RC5PlusVariableQuerySanityView,
    mask: VerifiedStage2RC5FeatureMask,
    spec: VerifiedRC5PlusTrainingExecutionSpec,
    standardizer_identity: str,
) -> None:
    actual = train.manifest["actual_input_binding_identities"]
    expected = {
        "rc5plus_config": spec.config_source_sha256,
        "training_view": train.view_identity_sha256,
        "source_validation_view": rc5plus_source_validation_identity_sha256(
            selection, sanity
        ),
        "feature_mask": mask.identity_sha256,
        "standardizer": standardizer_identity,
        "source_reference": actual["source_reference"],
        "per_image_curve_bank": train.curve_bank_id,
        "detector_run_complete_set": actual["detector_run_complete_set"],
        "seed_manifest": actual["seed_manifest"],
        "source_release": actual["source_release"],
    }
    for name, digest in expected.items():
        if bindings[name]["sha256"] != digest:
            raise Stage2RC5PlusCyclicTrainerError(
                f"input_bindings.{name} does not match its verified capability"
            )


def _configure_deterministic_runtime(
    optimizer_config: Mapping[str, Any], derived_seed: int
) -> torch.Generator:
    torch.use_deterministic_algorithms(
        bool(optimizer_config["deterministic_algorithms"])
    )
    torch.backends.cudnn.benchmark = bool(
        optimizer_config["cudnn_benchmark"]
    )
    torch.backends.cudnn.deterministic = bool(
        optimizer_config["cudnn_deterministic"]
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(
        optimizer_config["cuda_matmul_allow_tf32"]
    )
    torch.backends.cudnn.allow_tf32 = bool(
        optimizer_config["cudnn_allow_tf32"]
    )
    torch.set_float32_matmul_precision(
        str(optimizer_config["float32_matmul_precision"])
    )
    random.seed(derived_seed)
    np.random.seed(derived_seed % (2**32))
    torch.manual_seed(derived_seed)
    return torch.Generator(device="cpu").manual_seed(derived_seed ^ 0x5A17)


def train_stage2_rc5plus_cyclic(
    *,
    method: str,
    collection: VerifiedStage2RC5PlusCyclicTrainingView,
    selection_view: VerifiedStage2RC5PlusSourceValidationView,
    sanity_view: VerifiedStage2RC5PlusVariableQuerySanityView,
    feature_mask: VerifiedStage2RC5FeatureMask,
    execution_spec: VerifiedRC5PlusTrainingExecutionSpec,
    run_root: str | Path,
    run_id: str,
    base_seed: int,
    derived_seed: int,
    input_bindings: Mapping[str, Any],
    device: str | torch.device = "cpu",
    resume_generation_path: str | Path | None = None,
    resume_generation_commit_sha256: str | None = None,
    synthetic_interrupt_after_epoch: int | None = None,
) -> RC5PlusTrainingOutcome:
    """Run one explicit source-only RC5+ job and commit generation-v3 last."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise Stage2RC5PlusCyclicTrainerError("requested CUDA is unavailable")
    train, selection, sanity, mask, spec = _validate_full_inputs(
        collection=collection,
        selection_view=selection_view,
        sanity_view=sanity_view,
        feature_mask=feature_mask,
        execution_spec=execution_spec,
        device=resolved_device,
    )
    if method not in RC5PLUS_TRAINING_METHODS:
        raise Stage2RC5PlusCyclicTrainerError(
            "method is not a frozen RC5+ training route"
        )
    if (
        type(base_seed) is not int
        or base_seed < 0
        or type(derived_seed) is not int
        or derived_seed < 1
        or type(run_id) is not str
        or not run_id
    ):
        raise Stage2RC5PlusCyclicTrainerError("run/seed identity is invalid")
    mean, scale = train.fit_training_standardizer()
    standardizer_identity = rc5plus_standardizer_identity_sha256(mean, scale)
    bindings = normalize_input_bindings_v3(input_bindings)
    _validate_full_input_bindings(
        bindings=bindings,
        train=train,
        selection=selection,
        sanity=sanity,
        mask=mask,
        spec=spec,
        standardizer_identity=standardizer_identity,
    )
    binding_identity = input_identity_sha256_v3(bindings)
    root = Path(run_root).expanduser()
    optimizer_config = spec.payload["optimizer"]
    loss_config = spec.payload["loss"]
    data_generator = _configure_deterministic_runtime(
        optimizer_config, derived_seed
    )
    model = build_rc5plus_training_model(
        method, spec.payload["model"]
    ).to(device=resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        amsgrad=optimizer_config["amsgrad"],
    )
    history: list[dict[str, Any]] = []
    generations: list[VerifiedRC5PlusCalibratorGenerationV3] = []
    start_epoch = 0
    resume_supplied = (
        resume_generation_path is not None
        or resume_generation_commit_sha256 is not None
    )
    if resume_supplied:
        if (
            resume_generation_path is None
            or resume_generation_commit_sha256 is None
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                "resume path and external commit SHA are both required"
            )
        resumed = verify_rc5plus_calibrator_generation_v3(
            resume_generation_path, resume_generation_commit_sha256
        )
        manifest = resumed.manifest
        state = resumed.resume_state
        expected_identity = {
            "method": method,
            "run_id": run_id,
            "outer_fold_id": train.manifest["outer_fold_id"],
            "outer_target_domain": train.manifest["outer_target"],
            "base_seed": base_seed,
            "derived_seed": derived_seed,
            "training_contract_sha256": spec.training_contract_sha256,
            "training_view_identity_sha256": train.view_identity_sha256,
            "input_identity_sha256": binding_identity,
        }
        for key, value in expected_identity.items():
            if manifest[key] != value:
                raise Stage2RC5PlusCyclicTrainerError(
                    f"resume identity mismatch: {key}"
                )
        root = root.resolve(strict=True)
        if resumed.path.parent != root:
            raise Stage2RC5PlusCyclicTrainerError(
                "resume generation is outside run_root"
            )
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = [dict(row) for row in state["history"]]
        current_epoch = int(state["epoch"])
        if (
            len(history) != current_epoch + 1
            or int(history[-1]["epoch"]) != current_epoch
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                "resume history is not contiguous"
            )
        for epoch in range(current_epoch):
            commit_sha = history[epoch].get("generation_commit_sha256")
            if not isinstance(commit_sha, str):
                raise Stage2RC5PlusCyclicTrainerError(
                    "prior generation SHA is missing from resume history"
                )
            generations.append(
                verify_rc5plus_calibrator_generation_v3(
                    root / f"generation_v3_e{epoch:06d}_r0000",
                    commit_sha,
                )
            )
        generations.append(resumed)
        history[-1]["generation_commit_sha256"] = resumed.commit_sha256
        _restore_python_rng(state["python_rng_state"])
        _restore_numpy_rng(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_cpu_rng_state"])
        if resolved_device.type == "cuda":
            torch.cuda.set_rng_state_all(list(state["torch_cuda_rng_states"]))
        data_generator.set_state(state["dataloader_rng_state"])
        start_epoch = current_epoch + 1
    elif root.exists() or root.is_symlink():
        raise FileExistsError("new immutable RC5+ run_root already exists")

    max_epochs = int(optimizer_config["max_epochs"])
    if synthetic_interrupt_after_epoch is not None and (
        spec.artifact_scope != "synthetic_cpu_contract_test"
        or type(synthetic_interrupt_after_epoch) is not int
        or not 0 <= synthetic_interrupt_after_epoch < max_epochs
    ):
        raise Stage2RC5PlusCyclicTrainerError(
            "synthetic interruption point is invalid"
        )
    episode_counts = {
        domain: len(train.domain_episode_indices[domain])
        for domain in train.manifest["source_domains"]
    }
    for epoch in range(start_epoch, max_epochs):
        custom_loop_rng_token = int(
            torch.randint(
                0,
                2**31,
                (1,),
                generator=data_generator,
                dtype=torch.int64,
            ).item()
        )
        sampler = assert_verified_domain_balanced_cyclic_epoch(
            verify_domain_balanced_cyclic_epoch(
                build_domain_balanced_cyclic_epoch(
                    outer_fold_id=train.manifest["outer_fold_id"],
                    derived_seed=derived_seed,
                    epoch=epoch,
                    episode_counts=episode_counts,
                )
            )
        ).payload
        ordered = sampler["ordered_selection"]
        if len(ordered) != sampler["epoch_size"] or len(ordered) % 2:
            raise Stage2RC5PlusCyclicTrainerError(
                "verified sampler epoch size is invalid"
            )
        totals = {name: 0.0 for name in RC5PLUS_LOSS_METRIC_NAMES}
        model.train()
        for start in range(0, len(ordered), optimizer_config["batch_size"]):
            rows = ordered[start : start + optimizer_config["batch_size"]]
            if len(rows) % 2:
                raise Stage2RC5PlusCyclicTrainerError(
                    "epoch tail broke exact domain pairing"
                )
            batch = collate_rc5plus_cyclic_batch(
                collection=train,
                ordered_rows=rows,
                standardizer_mean=mean,
                standardizer_scale=scale,
                feature_mask=mask,
                device=resolved_device,
                method=method,
            )
            _, losses = rc5plus_cyclic_optimization_step(
                method=method,
                model=model,
                optimizer=optimizer,
                batch=batch,
                loss_config=loss_config,
                gradient_clip_norm=optimizer_config["gradient_clip_norm"],
            )
            for name in RC5PLUS_LOSS_METRIC_NAMES:
                totals[name] += (
                    float(losses[name].detach().cpu().item()) * len(rows)
                )
        selection_metrics = evaluate_stage2_rc5plus_source_validation_view(
            model=model,
            view=selection,
            standardizer_mean=mean,
            standardizer_scale=scale,
            feature_mask=mask,
            device=resolved_device,
            batch_size=optimizer_config["batch_size"],
        )
        sanity_metrics = evaluate_stage2_rc5plus_variable_query_sanity_view(
            model=model,
            view=sanity,
            standardizer_mean=mean,
            standardizer_scale=scale,
            feature_mask=mask,
            device=resolved_device,
            batch_size=optimizer_config["batch_size"],
        )
        if (
            sanity_metrics["excluded_from_epoch_ranking"] is not True
            or sanity_metrics["selection_record_present"] is not False
        ):
            raise Stage2RC5PlusCyclicTrainerError(
                "mandatory variable-Q sanity leaked into epoch ranking"
            )
        history_row = {
            "epoch": epoch,
            "sampler_ordered_selection_sha256": sampler[
                "ordered_selection_sha256"
            ],
            "epoch_size": len(ordered),
            "custom_loop_rng_token": custom_loop_rng_token,
            "data_iteration": optimizer_config["data_iteration"],
            "mean_loss_hex": {
                name: (totals[name] / len(ordered)).hex()
                for name in RC5PLUS_LOSS_METRIC_NAMES
            },
            "selection_record": selection_metrics["selection_record"],
            "selection_geometry": selection_metrics["selection_geometry"],
            "selection_domain_metrics": selection_metrics["domain_metrics"],
            "source_variable_query_sanity": sanity_metrics,
            "feature_mask_variant": mask.variant,
            "feature_mask_identity_sha256": mask.identity_sha256,
            "outer_target_accessed": False,
            "official_test_accessed": False,
            "query_labels_accessed": False,
        }
        history.append(history_row)
        python_state = _python_rng_state()
        numpy_state = _numpy_rng_state()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = (
            [item.clone() for item in torch.cuda.get_rng_state_all()]
            if resolved_device.type == "cuda"
            else []
        )
        data_rng = data_generator.get_state().clone()
        resume_state = build_resume_state_v3(
            method=method,
            run_id=run_id,
            outer_fold_id=train.manifest["outer_fold_id"],
            outer_target_domain=train.manifest["outer_target"],
            base_seed=base_seed,
            derived_seed=derived_seed,
            epoch=epoch,
            process_rank=optimizer_config["process_rank"],
            world_size=optimizer_config["world_size"],
            training_contract_sha256=spec.training_contract_sha256,
            training_view_identity_sha256=train.view_identity_sha256,
            input_bindings=bindings,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            history=history,
            selection_record=selection_metrics["selection_record"],
            python_rng_state=python_state,
            numpy_rng_state=numpy_state,
            torch_cpu_rng_state=cpu_rng,
            torch_cuda_rng_states=cuda_rng,
            dataloader_rng_state=data_rng,
        )
        checkpoint = make_calibrator_checkpoint_v8(
            method=method,
            model=model,
            standardizer_mean=mean,
            standardizer_scale=scale,
            training_contract_sha256=spec.training_contract_sha256,
            training_view_identity_sha256=train.view_identity_sha256,
            feature_mask=mask,
        )
        generation = publish_rc5plus_calibrator_generation_v3(
            root,
            resume_state=resume_state,
            deployment_checkpoint_bytes=serialize_calibrator_checkpoint_v8(
                checkpoint
            ),
            input_bindings=bindings,
        )
        generations.append(generation)
        history[-1]["generation_commit_sha256"] = generation.commit_sha256
        _restore_python_rng(python_state)
        _restore_numpy_rng(numpy_state)
        torch.set_rng_state(cpu_rng)
        if resolved_device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
        data_generator.set_state(data_rng)
        if synthetic_interrupt_after_epoch == epoch:
            return RC5PlusTrainingOutcome(
                generations=tuple(generations),
                run=None,
                history=tuple(
                    MappingProxyType(dict(row)) for row in history
                ),
                interrupted_after_epoch=epoch,
            )
        keys = [
            _selection_key(row["selection_record"], int(row["epoch"]))
            for row in history
        ]
        best_index = min(range(len(keys)), key=keys.__getitem__)
        if (
            epoch - best_index
            >= optimizer_config["early_stopping_patience"]
        ):
            break
    completed = publish_rc5plus_calibrator_run_v3(root, generations)
    return RC5PlusTrainingOutcome(
        generations=tuple(generations),
        run=completed,
        history=tuple(MappingProxyType(dict(row)) for row in history),
        interrupted_after_epoch=None,
    )


__all__ = [
    "RC5PlusTrainingOutcome",
    "RC5PLUS_CYCLIC_TRAINER_SCHEMA",
    "Stage2RC5PlusCyclicTrainerError",
    "VerifiedRC5PlusTrainingExecutionSpec",
    "assert_verified_rc5plus_training_execution_spec",
    "build_rc5plus_training_execution_spec_from_verified_config",
    "build_rc5plus_training_model",
    "build_synthetic_rc5plus_training_execution_spec",
    "collate_rc5plus_cyclic_batch",
    "rc5plus_cyclic_optimization_step",
    "rc5plus_source_validation_identity_sha256",
    "rc5plus_standardizer_identity_sha256",
    "train_stage2_rc5plus_cyclic",
]
