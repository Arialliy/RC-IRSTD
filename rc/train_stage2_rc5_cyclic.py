"""Deterministic RC5 source-only cyclic trainer with generation-v2 resume.

This module defines a runner but starts no training by itself. Production uses
the frozen config; the public synthetic spec is CPU-only and exists solely for
uninterrupted-versus-resume contract tests. The trainer consumes only a
verified domain-balanced sampler capability, composes Q28 curves live, executes
the independent cyclic source-selection view and mandatory variable-Q sanity
each epoch, publishes generation-v2 every epoch, and commits run-v2 last.
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

from model.endpoint_aware_pixel_calibrator import (
    DirectEndpointAwarePixelCalibrator,
    MonotoneEndpointAwarePixelCalibrator,
)
from rc.stage2_calibrator_checkpoint_v7 import (
    make_calibrator_checkpoint_v7,
    serialize_calibrator_checkpoint_v7,
)
from rc.stage2_calibrator_generation_v2 import (
    VerifiedCalibratorGenerationV2,
    VerifiedCalibratorRunV2,
    build_resume_state_v2,
    input_identity_sha256,
    normalize_input_bindings,
    publish_calibrator_generation_v2,
    publish_calibrator_run_v2,
    verify_calibrator_generation_v2,
)
from rc.stage2_cyclic_training_collection_v1 import (
    COMMIT_FILENAME as COLLECTION_COMMIT_FILENAME,
    VerifiedCyclicTrainingCollection,
    assert_verified_cyclic_training_collection,
)
from rc.stage2_domain_balanced_cyclic_sampler import (
    assert_verified_domain_balanced_cyclic_epoch,
    build_domain_balanced_cyclic_epoch,
    verify_domain_balanced_cyclic_epoch,
)
from rc.stage2_rc5_training_core import RC5_LOSS_METRIC_NAMES, rc5_batch_loss
from rc.stage2_source_validation_views import (
    VerifiedSourceValidationCyclicSelectionView,
    VerifiedSourceVariableQuerySanityView,
    assert_verified_source_validation_cyclic_selection_view,
    assert_verified_source_variable_query_sanity_view,
    evaluate_source_validation_cyclic_selection_view,
    evaluate_source_variable_query_sanity_view,
    source_validation_collection_identity_sha256,
)


TRAINER_SCHEMA = "rc-irstd.stage2-rc5-cyclic-trainer.v1"
_SPEC_CAPABILITY = object()


class Stage2RC5CyclicTrainerError(ValueError):
    """A frozen trainer, source-only boundary, or resume check failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True, init=False)
class VerifiedRC5TrainingExecutionSpec:
    payload: Mapping[str, Any]
    training_contract_sha256: str
    artifact_scope: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("training execution spec is verifier-issued only")


def build_synthetic_rc5_training_execution_spec(
    *, max_epochs: int = 2, early_stopping_patience: int = 20,
) -> VerifiedRC5TrainingExecutionSpec:
    """Build a shortened CPU-only spec without minting a production contract."""
    if type(max_epochs) is not int or not 1 <= max_epochs <= 4:
        raise Stage2RC5CyclicTrainerError("synthetic max_epochs must be in [1,4]")
    if type(early_stopping_patience) is not int or early_stopping_patience < 1:
        raise Stage2RC5CyclicTrainerError("patience must be positive")
    payload = {
        "schema_version": TRAINER_SCHEMA,
        "artifact_scope": "synthetic_cpu_contract_test",
        "optimizer": {"name": "AdamW", "learning_rate": 0.001,
                      "weight_decay": 0.0001, "betas": [0.9, 0.999],
                      "epsilon": 1e-8, "amsgrad": False, "scheduler": "none",
                      "batch_size": 16, "max_epochs": max_epochs,
                      "early_stopping_patience": early_stopping_patience,
                      "gradient_clip_norm": 5.0, "num_workers": 0,
                      "deterministic_algorithms": True, "amp": False,
                      "shuffle_training": False, "drop_last": False,
                      "custom_verified_epoch_sampler": True},
        "loss": {"coordinate_huber_delta": 1.0, "lambda_violation": 4.0,
                 "lambda_utility": 1.0, "lambda_oracle": 0.1,
                 "lambda_smoothness": 0.01, "lambda_coverage": 0.0,
                 "risk_epsilon": 1e-12},
        "model": {"context_feature_dim": 93, "pixel_budget_grid": [1e-4, 1e-5, 1e-6],
                  "hidden_dims": [32], "dropout": 0.1,
                  "minimum_raw_coordinate_gap": 0.001},
        "source_selection": "source_validation_cyclic_selection_view_c14_q28_all_n_starts",
        "source_variable_query_sanity_mandatory": True,
        "source_variable_query_sanity_excluded_from_ranking": True,
        "data_iteration": "custom_loop_verified_sampler_no_dataloader",
        "generation_dataloader_rng_field_usage": "consumed_custom_loop_generator_state",
        "outer_target_accessed": False, "official_test_accessed": False,
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    value = object.__new__(VerifiedRC5TrainingExecutionSpec)
    for name, item in {"payload": MappingProxyType(payload),
                       "training_contract_sha256": digest,
                       "artifact_scope": "synthetic_cpu_contract_test",
                       "_capability": _SPEC_CAPABILITY}.items():
        object.__setattr__(value, name, item)
    return value


def build_rc5_training_execution_spec_from_verified_config(
    verified_config: Any,
) -> VerifiedRC5TrainingExecutionSpec:
    """Promote only a verifier-issued, externally SHA-bound RC5 freeze."""
    from rc.stage2_rc5_config import VerifiedStage2RC5Config

    if type(verified_config) is not VerifiedStage2RC5Config:
        raise TypeError("production trainer requires VerifiedStage2RC5Config")
    config = verified_config.payload  # property checks the private capability token
    optimizer = dict(config["optimizer"])
    loss = config["loss"]
    model = config["model"]
    payload = {
        "schema_version": TRAINER_SCHEMA,
        "artifact_scope": "production",
        "optimizer": optimizer,
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
            "context_feature_dim": config["context_feature_dim"],
            "pixel_budget_grid": list(config["pixel_budget_grid"]),
            "hidden_dims": list(model["hidden_dims"]),
            "dropout": model["dropout"],
            "minimum_raw_coordinate_gap": model["minimum_raw_coordinate_gap"],
        },
        "source_selection": (
            "source_validation_cyclic_selection_view_c14_q28_all_n_starts"
        ),
        "source_variable_query_sanity_mandatory": True,
        "source_variable_query_sanity_excluded_from_ranking": True,
        "data_iteration": "custom_loop_verified_sampler_no_dataloader",
        "generation_dataloader_rng_field_usage": "consumed_custom_loop_generator_state",
        "outer_target_accessed": False,
        "official_test_accessed": False,
    }
    value = object.__new__(VerifiedRC5TrainingExecutionSpec)
    for name, item in {
        "payload": MappingProxyType(payload),
        "training_contract_sha256": verified_config.sha256,
        "artifact_scope": "production",
        "_capability": _SPEC_CAPABILITY,
    }.items():
        object.__setattr__(value, name, item)
    return value


def assert_verified_rc5_training_execution_spec(
    value: object,
) -> VerifiedRC5TrainingExecutionSpec:
    if type(value) is not VerifiedRC5TrainingExecutionSpec or getattr(
        value, "_capability", None
    ) is not _SPEC_CAPABILITY:
        raise TypeError("a verified RC5 training execution spec is required")
    return value


@dataclass(frozen=True)
class RC5TrainingOutcome:
    generations: tuple[VerifiedCalibratorGenerationV2, ...]
    run: VerifiedCalibratorRunV2 | None
    history: tuple[Mapping[str, Any], ...]
    interrupted_after_epoch: int | None


def _model(method: str, config: Mapping[str, Any]) -> torch.nn.Module:
    kwargs = {"context_feature_dim": config["context_feature_dim"],
              "pixel_budget_grid": config["pixel_budget_grid"],
              "hidden_dims": config["hidden_dims"], "dropout": config["dropout"]}
    if method == "T6": return DirectEndpointAwarePixelCalibrator(**kwargs)
    if method in {"T7", "T8"}:
        return MonotoneEndpointAwarePixelCalibrator(
            **kwargs, minimum_raw_coordinate_gap=config["minimum_raw_coordinate_gap"])
    raise Stage2RC5CyclicTrainerError("method must be T6, T7 or T8")


def _python_rng() -> dict[str, Any]:
    version, state, gauss = random.getstate()
    return {"version": version, "internal_state": list(state), "gauss_next": gauss}


def _restore_python(value: Mapping[str, Any]) -> None:
    random.setstate((int(value["version"]), tuple(value["internal_state"]),
                     value["gauss_next"]))


def _numpy_rng() -> dict[str, Any]:
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"bit_generator": name, "keys": keys.astype(np.uint32).tolist(),
            "position": int(position), "has_gauss": int(has_gauss),
            "cached_gaussian_hex": float(cached).hex()}


def _restore_numpy(value: Mapping[str, Any]) -> None:
    np.random.set_state((str(value["bit_generator"]),
                         np.asarray(value["keys"], dtype=np.uint32),
                         int(value["position"]), int(value["has_gauss"]),
                         float.fromhex(str(value["cached_gaussian_hex"]))))


def _selection_key(record: Mapping[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return (-float.fromhex(record["macro_source_bsr_hex"]),
            float.fromhex(record["macro_source_log_excess_hex"]),
            -float.fromhex(record["macro_source_pd_hex"]), epoch)


def _collate(
    collection: VerifiedCyclicTrainingCollection,
    ordered_rows: Sequence[Mapping[str, Any]],
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
    method: str,
) -> dict[str, Any]:
    features = []
    anchors = []
    oracles = []
    providers = []
    gt_objects = []
    domains = []
    for row in ordered_rows:
        domain = str(row["source_domain"])
        index = int(row["domain_episode_index"])
        feature, anchor = collection.feature_anchor_for_episode(domain, index)
        provider = collection.provider_for_episode(domain, index)
        oracle = provider.select_exact_oracle_rows()
        features.append(feature); anchors.append(anchor)
        providers.append(provider); oracles.append(oracle.coordinates)
        gt_objects.append(provider.ground_truth_objects); domains.append(domain)
    counts = {domain: domains.count(domain) for domain in set(domains)}
    if len(counts) != 2 or len(set(counts.values())) != 1:
        raise Stage2RC5CyclicTrainerError("every minibatch must be exactly domain balanced")
    raw_features = np.stack(features).astype(np.float64)
    standardized = ((raw_features - mean) / scale).astype(np.float32)
    batch: dict[str, Any] = {
        "features": torch.from_numpy(standardized).to(device=device),
        "anchor_coordinates": torch.from_numpy(np.stack(anchors).astype(np.float64)).to(
            device=device),
        "oracle_coordinates": torch.from_numpy(np.stack(oracles).astype(np.float64)).to(
            device=device),
    }
    if method == "T8":
        batch.update({
            "pixel_budgets": torch.tensor([1e-4, 1e-5, 1e-6], dtype=torch.float64,
                                          device=device).repeat(len(providers), 1),
            "curve_gt_objects": torch.tensor(gt_objects, dtype=torch.int64, device=device),
            "compositional_curve_providers": tuple(providers),
        })
    return batch


def _validate_inputs(
    *, collection: Any, selection_view: Any, sanity_view: Any,
    spec: Any, device: torch.device,
) -> tuple[VerifiedCyclicTrainingCollection,
           VerifiedSourceValidationCyclicSelectionView,
           VerifiedSourceVariableQuerySanityView,
           VerifiedRC5TrainingExecutionSpec]:
    train = assert_verified_cyclic_training_collection(collection)
    selection = assert_verified_source_validation_cyclic_selection_view(selection_view)
    sanity = assert_verified_source_variable_query_sanity_view(sanity_view)
    execution = assert_verified_rc5_training_execution_spec(spec)
    if len({train.manifest["outer_fold_id"], selection.outer_fold_id,
            sanity.outer_fold_id}) != 1:
        raise Stage2RC5CyclicTrainerError("training/validation outer folds differ")
    if len({train.artifact_scope, selection.artifact_scope,
            sanity.artifact_scope, execution.artifact_scope}) != 1:
        raise Stage2RC5CyclicTrainerError("production/synthetic capabilities cannot mix")
    if execution.artifact_scope == "synthetic_cpu_contract_test" and device.type != "cpu":
        raise Stage2RC5CyclicTrainerError("synthetic contract runner is CPU-only")
    for field in train.boundary_values:
        if train.boundary_values[field].intersection(selection.boundary_values[field]) or \
                train.boundary_values[field].intersection(sanity.boundary_values[field]):
            raise Stage2RC5CyclicTrainerError(
                f"training/source-validation overlap at identity boundary {field}")
    if execution.artifact_scope == "production":
        if dict(selection.boundary_values) != dict(sanity.boundary_values):
            raise Stage2RC5CyclicTrainerError(
                "production selection and sanity views are not the same validation roles"
            )
        if dict(selection.upstream_bindings) != dict(sanity.upstream_bindings):
            raise Stage2RC5CyclicTrainerError(
                "production validation views have different score/context attestations"
            )
        if selection.upstream_bindings["statistics_config_sha256"] != \
                train.manifest["actual_input_binding_identities"]["statistics_config"]:
            raise Stage2RC5CyclicTrainerError(
                "training/validation/deployment statistics config differs"
            )
    return train, selection, sanity, execution


def _validate_input_bindings(
    bindings: Mapping[str, Mapping[str, str]],
    train: VerifiedCyclicTrainingCollection,
    selection: VerifiedSourceValidationCyclicSelectionView,
    sanity: VerifiedSourceVariableQuerySanityView,
    spec: VerifiedRC5TrainingExecutionSpec,
) -> None:
    actual = train.manifest["actual_input_binding_identities"]
    expected = {
        "rc5_config": spec.training_contract_sha256,
        "training_collection": train.commit_sha256,
        "validation_collection": source_validation_collection_identity_sha256(
            selection, sanity),
        "statistics_config": actual["statistics_config"],
        "source_reference": actual["source_reference"],
        "per_image_curve_bank": train.curve_bank_id,
        "detector_run_complete_set": actual["detector_run_complete_set"],
        "seed_manifest": actual["seed_manifest"],
        "source_release": actual["source_release"],
    }
    for name, digest in expected.items():
        if bindings[name]["sha256"] != digest:
            raise Stage2RC5CyclicTrainerError(
                f"input_bindings.{name} does not match the verified capability"
            )
    training_path = Path(bindings["training_collection"]["path"])
    if training_path.name != COLLECTION_COMMIT_FILENAME or \
            training_path.parent.name != train.path.name:
        raise Stage2RC5CyclicTrainerError(
            "training_collection path must identify the verified commit"
        )


def train_stage2_rc5_cyclic(
    *,
    method: str,
    collection: VerifiedCyclicTrainingCollection,
    selection_view: VerifiedSourceValidationCyclicSelectionView,
    sanity_view: VerifiedSourceVariableQuerySanityView,
    execution_spec: VerifiedRC5TrainingExecutionSpec,
    run_root: str | Path,
    run_id: str,
    base_seed: int,
    derived_seed: int,
    input_bindings: Mapping[str, Any],
    device: str | torch.device = "cpu",
    resume_generation_path: str | Path | None = None,
    resume_generation_commit_sha256: str | None = None,
    synthetic_interrupt_after_epoch: int | None = None,
) -> RC5TrainingOutcome:
    """Run an explicitly invoked training job; no CLI side effect is defined."""
    resolved_device = torch.device(device)
    train, selection, sanity, spec = _validate_inputs(
        collection=collection, selection_view=selection_view,
        sanity_view=sanity_view, spec=execution_spec, device=resolved_device)
    if method not in {"T6", "T7", "T8"}:
        raise Stage2RC5CyclicTrainerError("method must be T6, T7 or T8")
    if type(base_seed) is not int or base_seed < 0 or type(derived_seed) is not int \
            or derived_seed < 1 or not isinstance(run_id, str) or not run_id:
        raise Stage2RC5CyclicTrainerError("run/seed identity is invalid")
    bindings = normalize_input_bindings(input_bindings)
    _validate_input_bindings(bindings, train, selection, sanity, spec)
    binding_identity = input_identity_sha256(bindings)
    root = Path(run_root).expanduser()
    optimizer_config = spec.payload["optimizer"]
    loss_config = spec.payload["loss"]
    mean, scale = train.fit_training_standardizer()
    torch.use_deterministic_algorithms(True)
    random.seed(derived_seed); np.random.seed(derived_seed % (2**32))
    torch.manual_seed(derived_seed)
    data_generator = torch.Generator(device="cpu").manual_seed(derived_seed ^ 0x5A17)
    model = _model(method, spec.payload["model"]).to(device=resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        betas=tuple(optimizer_config["betas"]), eps=optimizer_config["epsilon"],
        amsgrad=optimizer_config["amsgrad"])
    history: list[dict[str, Any]] = []
    generations: list[VerifiedCalibratorGenerationV2] = []
    start_epoch = 0
    resume_supplied = resume_generation_path is not None or \
        resume_generation_commit_sha256 is not None
    if resume_supplied:
        if resume_generation_path is None or resume_generation_commit_sha256 is None:
            raise Stage2RC5CyclicTrainerError("resume path and external SHA are both required")
        resumed = verify_calibrator_generation_v2(
            resume_generation_path, resume_generation_commit_sha256)
        manifest = resumed.manifest; state = resumed.resume_state
        expected_identity = {
            "method": method, "run_id": run_id,
            "outer_fold_id": train.manifest["outer_fold_id"],
            "outer_target_domain": train.manifest["outer_target"],
            "base_seed": base_seed, "derived_seed": derived_seed,
            "training_contract_sha256": spec.training_contract_sha256,
            "input_identity_sha256": binding_identity,
        }
        for key, value in expected_identity.items():
            if manifest[key] != value:
                raise Stage2RC5CyclicTrainerError(f"resume identity mismatch: {key}")
        if resumed.path.parent != root.resolve(strict=True):
            raise Stage2RC5CyclicTrainerError("resume generation is outside run_root")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = [dict(row) for row in state["history"]]
        current_epoch = int(state["epoch"])
        if len(history) != current_epoch + 1 or history[-1]["epoch"] != current_epoch:
            raise Stage2RC5CyclicTrainerError("resume history is not contiguous")
        for epoch in range(current_epoch):
            commit_sha = history[epoch].get("generation_commit_sha256")
            if not isinstance(commit_sha, str):
                raise Stage2RC5CyclicTrainerError("prior generation SHA missing from history")
            generations.append(verify_calibrator_generation_v2(
                root / f"generation_e{epoch:06d}_r0000", commit_sha))
        generations.append(resumed)
        history[-1]["generation_commit_sha256"] = resumed.commit_sha256
        _restore_python(state["python_rng_state"]); _restore_numpy(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_cpu_rng_state"])
        if resolved_device.type == "cuda":
            torch.cuda.set_rng_state_all(list(state["torch_cuda_rng_states"]))
        data_generator.set_state(state["dataloader_rng_state"])
        start_epoch = current_epoch + 1
    elif root.exists() or root.is_symlink():
        raise FileExistsError("new immutable run_root must not already exist")

    max_epochs = int(optimizer_config["max_epochs"])
    if synthetic_interrupt_after_epoch is not None and (
        spec.artifact_scope != "synthetic_cpu_contract_test" or
        type(synthetic_interrupt_after_epoch) is not int or
        not 0 <= synthetic_interrupt_after_epoch < max_epochs
    ):
        raise Stage2RC5CyclicTrainerError("invalid synthetic interruption point")
    episode_counts = {domain: len(train.domain_episode_indices[domain])
                      for domain in train.manifest["source_domains"]}
    for epoch in range(start_epoch, max_epochs):
        custom_loop_rng_token = int(torch.randint(
            0, 2**31, (1,), generator=data_generator, dtype=torch.int64).item())
        raw_sampler = build_domain_balanced_cyclic_epoch(
            outer_fold_id=train.manifest["outer_fold_id"], derived_seed=derived_seed,
            epoch=epoch, episode_counts=episode_counts)
        verified_sampler = verify_domain_balanced_cyclic_epoch(raw_sampler)
        sampler = assert_verified_domain_balanced_cyclic_epoch(verified_sampler).payload
        ordered = sampler["ordered_selection"]
        if len(ordered) != sampler["epoch_size"] or len(ordered) % 2:
            raise Stage2RC5CyclicTrainerError("verified sampler epoch size is invalid")
        totals = {name: 0.0 for name in RC5_LOSS_METRIC_NAMES}
        model.train()
        for start in range(0, len(ordered), optimizer_config["batch_size"]):
            rows = ordered[start:start + optimizer_config["batch_size"]]
            if len(rows) % 2:
                raise Stage2RC5CyclicTrainerError("epoch tail broke domain pairing")
            batch = _collate(train, rows, mean, scale, resolved_device, method)
            optimizer.zero_grad(set_to_none=True)
            _, losses = rc5_batch_loss(
                method=method, model=model, batch=batch, loss_config=loss_config)
            total = losses["total"]
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError("non-finite RC5 training loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           optimizer_config["gradient_clip_norm"])
            optimizer.step()
            for name in RC5_LOSS_METRIC_NAMES:
                totals[name] += float(losses[name].detach().cpu().item()) * len(rows)
        selection_metrics = evaluate_source_validation_cyclic_selection_view(
            model=model, view=selection, standardizer_mean=mean,
            standardizer_scale=scale, device=resolved_device,
            batch_size=optimizer_config["batch_size"])
        sanity_metrics = evaluate_source_variable_query_sanity_view(
            model=model, view=sanity, standardizer_mean=mean,
            standardizer_scale=scale, device=resolved_device,
            batch_size=optimizer_config["batch_size"])
        if sanity_metrics["excluded_from_epoch_ranking"] is not True:
            raise Stage2RC5CyclicTrainerError("mandatory sanity leaked into ranking")
        history_row = {
            "epoch": epoch,
            "sampler_ordered_selection_sha256": sampler["ordered_selection_sha256"],
            "epoch_size": len(ordered),
            "custom_loop_rng_token": custom_loop_rng_token,
            "data_iteration": "custom_loop_verified_sampler_no_dataloader",
            "mean_loss_hex": {name: (totals[name] / len(ordered)).hex()
                              for name in RC5_LOSS_METRIC_NAMES},
            "selection_record": selection_metrics["selection_record"],
            "selection_geometry": selection_metrics["selection_geometry"],
            "selection_domain_metrics": selection_metrics["domain_metrics"],
            "source_variable_query_sanity": sanity_metrics,
            "outer_target_accessed": False, "official_test_accessed": False,
        }
        history.append(history_row)
        python_state = _python_rng(); numpy_state = _numpy_rng()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = [item.clone() for item in torch.cuda.get_rng_state_all()] \
            if resolved_device.type == "cuda" else []
        loader_rng = data_generator.get_state().clone()
        resume_state = build_resume_state_v2(
            method=method, run_id=run_id,
            outer_fold_id=train.manifest["outer_fold_id"],
            outer_target_domain=train.manifest["outer_target"],
            base_seed=base_seed, derived_seed=derived_seed, epoch=epoch,
            process_rank=0, world_size=1,
            training_contract_sha256=spec.training_contract_sha256,
            input_bindings=bindings, model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(), history=history,
            selection_record=selection_metrics["selection_record"],
            python_rng_state=python_state, numpy_rng_state=numpy_state,
            torch_cpu_rng_state=cpu_rng, torch_cuda_rng_states=cuda_rng,
            dataloader_rng_state=loader_rng)
        checkpoint = make_calibrator_checkpoint_v7(
            method=method, model=model, standardizer_mean=mean,
            standardizer_scale=scale,
            training_contract_sha256=spec.training_contract_sha256)
        generation = publish_calibrator_generation_v2(
            root, resume_state=resume_state,
            deployment_checkpoint_bytes=serialize_calibrator_checkpoint_v7(checkpoint),
            input_bindings=bindings)
        generations.append(generation)
        history[-1]["generation_commit_sha256"] = generation.commit_sha256
        _restore_python(python_state); _restore_numpy(numpy_state)
        torch.set_rng_state(cpu_rng)
        if resolved_device.type == "cuda": torch.cuda.set_rng_state_all(cuda_rng)
        data_generator.set_state(loader_rng)
        if synthetic_interrupt_after_epoch == epoch:
            return RC5TrainingOutcome(tuple(generations), None,
                                      tuple(MappingProxyType(row) for row in history), epoch)
        keys = [_selection_key(row["selection_record"], int(row["epoch"]))
                for row in history]
        best_index = min(range(len(keys)), key=keys.__getitem__)
        if epoch - best_index >= optimizer_config["early_stopping_patience"]:
            break
    completed = publish_calibrator_run_v2(root, generations)
    return RC5TrainingOutcome(tuple(generations), completed,
                              tuple(MappingProxyType(row) for row in history), None)


__all__ = [
    "RC5TrainingOutcome", "Stage2RC5CyclicTrainerError", "TRAINER_SCHEMA",
    "VerifiedRC5TrainingExecutionSpec", "assert_verified_rc5_training_execution_spec",
    "build_synthetic_rc5_training_execution_spec", "train_stage2_rc5_cyclic",
    "build_rc5_training_execution_spec_from_verified_config",
]
