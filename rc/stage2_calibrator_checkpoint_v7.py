"""Strict checkpoint-v7 deployment core for RC5 endpoint-aware calibrators.

This module is intentionally independent from the checkpoint-v6 trainer.  A
v7 checkpoint is an immutable deployment state containing only one of the two
endpoint-aware model classes, its inference standardizer, and the current
endpoint-aware threshold representation.  It is not a resumable training
checkpoint: optimizer state, epoch/rank/history, and Python/NumPy/Torch/CUDA/
DataLoader RNG state are outside this artifact and exact field closure rejects
them.  The public verifiers reconstruct the model, strict-load its state, and
exercise one CPU inference before returning a verifier-created capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from model.endpoint_aware_pixel_calibrator import (
    ANCHOR_COORDINATE_CONTRACT,
    ANCHOR_MIX_INITIAL_WEIGHT,
    ANCHOR_MIX_PARAMETERIZATION,
    ANCHOR_MIX_RULE,
    DIRECT_ENDPOINT_AWARE_MODEL_ID,
    MONOTONE_ENDPOINT_AWARE_MODEL_ID,
    T4_ANCHOR_SOURCE,
    DirectEndpointAwarePixelCalibrator,
    MonotoneEndpointAwarePixelCalibrator,
)
from model.endpoint_aware_threshold import (
    RAW_COORDINATE_MAX,
    RAW_COORDINATE_MIN,
    THRESHOLD_REPRESENTATION_SCHEMA,
    decode_coordinate_torch,
    encode_probability_numpy,
    representation_contract,
)
from rc.domain_statistics import FEATURE_NAMES
from rc.stage2_rc5_feature_mask import (
    FEATURE_MASK_APPLICATION,
    Stage2RC5FeatureMaskError,
    VerifiedStage2RC5FeatureMask,
    apply_stage2_rc5_feature_mask_torch,
    assert_verified_stage2_rc5_feature_mask,
    build_stage2_rc5_feature_mask,
    feature_mask_payload,
    verify_stage2_rc5_feature_mask_payload,
)


CHECKPOINT_SCHEMA = "rc-irstd.calibrator.v7"
STANDARDIZER_SCHEMA = "rc-irstd.calibrator-standardizer.v3"
TENSOR_CONTENT_DIGEST_ALGORITHM = (
    "sha256-sorted-tensor-name-dtype-shape-contiguous-cpu-bytes-v1"
)
FEATURE_SCHEMA_DIGEST_ALGORITHM = "sha256-canonical-json-feature-names-v1"
ARTIFACT_KIND = "immutable_endpoint_aware_deployment_state"
METHODS = ("T6", "T7", "T8")
PIXEL_BUDGET_GRID = [1e-4, 1e-5, 1e-6]
EXPECTED_PARAMETER_COUNTS = {"T6": 3108, "T7": 3141, "T8": 3141}
_SHA_HEX = frozenset("0123456789abcdef")
_TOKEN = object()

_CHECKPOINT_FIELDS = frozenset(
    {
        "format_version",
        "artifact_kind",
        "method",
        "calibrator_model",
        "model_config",
        "capability_contract",
        "representation_contract",
        "expected_trainable_parameters",
        "model_state_dict",
        "model_state_content_sha256",
        "model_state_content_digest_algorithm",
        "standardizer",
        "standardizer_content_sha256",
        "training_contract_sha256",
        "inference_contract",
        "reject_head",
        "missing_episode_fallback",
        "official_test_accessed",
    }
)
_STANDARDIZER_FIELDS = frozenset(
    {
        "schema_version",
        "feature_dim",
        "feature_names",
        "feature_schema_sha256",
        "feature_schema_digest_algorithm",
        "mean",
        "scale",
        "mean_content_sha256",
        "scale_content_sha256",
        "tensor_content_sha256",
        "tensor_content_digest_algorithm",
        "calculation_dtype",
        "scale_floor",
        "model_input_dtype",
        "transformation",
        "feature_mask",
    }
)


class Stage2CalibratorCheckpointV7Error(ValueError):
    """A checkpoint violates the strict endpoint-aware v7 contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage2CalibratorCheckpointV7Error(
            "value is not canonical finite JSON"
        ) from error


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA_HEX
    ):
        raise Stage2CalibratorCheckpointV7Error(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def _exact_keys(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise Stage2CalibratorCheckpointV7Error(
            f"{name} fields mismatch; missing={sorted(fields-actual)}, "
            f"extra={sorted(actual-fields)}"
        )
    return value


def _exact_false(value: Any, name: str) -> None:
    if type(value) is not bool or value is not False:
        raise Stage2CalibratorCheckpointV7Error(f"{name} must be exact false")


def feature_schema_sha256() -> str:
    """Digest the exact ordered 93-dimensional feature registry."""

    if len(FEATURE_NAMES) != 93:
        raise RuntimeError("RC5 feature registry is no longer 93-dimensional")
    return hashlib.sha256(_canonical_json_bytes(list(FEATURE_NAMES))).hexdigest()


def _cpu_contiguous_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (
        value.layout != torch.strided
        or value.is_quantized
        or value.device.type == "meta"
    ):
        raise TypeError(f"{name} must be one dense strided tensor")
    return value.detach().to(device="cpu").contiguous()


def tensor_tree_content_sha256(value: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor mapping independently of ``torch.save`` container bytes.

    Keys are sorted and every record is length-delimited.  Tensor content is
    the native contiguous CPU byte sequence together with its name, dtype and
    shape, so key reordering cannot alter the digest while any tensor semantic
    change does.
    """

    if not isinstance(value, Mapping) or not value:
        raise TypeError("tensor tree must be a non-empty mapping")
    keys = list(value)
    if any(not isinstance(key, str) or not key for key in keys):
        raise TypeError("tensor tree keys must be non-empty strings")
    if len(keys) != len(set(keys)):
        raise ValueError("tensor tree keys must be unique")
    digest = hashlib.sha256()
    digest.update((TENSOR_CONTENT_DIGEST_ALGORITHM + "\0").encode("ascii"))
    for key in sorted(keys):
        tensor = _cpu_contiguous_tensor(value[key], f"tensor_tree[{key!r}]")
        metadata = _canonical_json_bytes(
            {
                "name": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack(">Q", len(metadata)))
        digest.update(metadata)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _single_tensor_content_sha256(name: str, tensor: torch.Tensor) -> str:
    return tensor_tree_content_sha256({name: tensor})


def _canonical_anchor_mix_alpha_hex(value: Any) -> str:
    if not isinstance(value, str):
        raise Stage2CalibratorCheckpointV7Error(
            "anchor_mix_alpha_hex must be a canonical float64 hexadecimal string"
        )
    try:
        alpha = float.fromhex(value)
    except ValueError as error:
        raise Stage2CalibratorCheckpointV7Error(
            "anchor_mix_alpha_hex is not hexadecimal float64"
        ) from error
    if (
        not math.isfinite(alpha)
        or not 0.0 < alpha < 1.0
        or alpha.hex() != value
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "anchor_mix_alpha_hex must canonically encode a finite value in (0,1)"
        )
    return value


def cpu_inference_contract(
    anchor_mix_alpha_hex: str,
    feature_mask: VerifiedStage2RC5FeatureMask | None = None,
) -> dict[str, Any]:
    """Return the exact RC5 deployment semantics for checkpoint-v7."""

    checked_alpha_hex = _canonical_anchor_mix_alpha_hex(anchor_mix_alpha_hex)
    checked_mask = assert_verified_stage2_rc5_feature_mask(
        build_stage2_rc5_feature_mask("C3")
        if feature_mask is None
        else feature_mask
    )
    return {
        "schema_version": "rc-irstd.calibrator-cpu-inference.v1",
        "device": "cpu",
        "model_mode": "eval",
        "autograd_enabled": False,
        "context_feature_dim": 93,
        "standardizer_compute_dtype": "float64",
        "model_input_dtype": "float32",
        "model_input_shape": "[batch,93]",
        "feature_mask_variant": checked_mask.variant,
        "feature_mask_identity_sha256": checked_mask.identity_sha256,
        "feature_mask_application": FEATURE_MASK_APPLICATION,
        "anchor_algorithm": T4_ANCHOR_SOURCE,
        "anchor_input_dtype": "float64",
        "anchor_input_shape": "[batch,3]",
        "anchor_coordinate_contract": ANCHOR_COORDINATE_CONTRACT,
        "anchor_mix_stage": "pre_canonicalization",
        "anchor_mix_rule": ANCHOR_MIX_RULE,
        "anchor_mix_parameterization": ANCHOR_MIX_PARAMETERIZATION,
        "anchor_mix_initial_weight_hex": float(
            ANCHOR_MIX_INITIAL_WEIGHT
        ).hex(),
        "anchor_mix_alpha_hex": checked_alpha_hex,
        "pixel_budget_grid": list(PIXEL_BUDGET_GRID),
        "budget_request": "complete_trained_grid_only",
        "output_dtype": "float64",
        "output_fields": [
            "anchor_coordinates",
            "anchor_mix_weight",
            "grid_learned_raw_coordinates",
            "grid_raw_coordinates",
            "grid_coordinates",
            "grid_thresholds",
        ],
        "threshold_representation_schema": THRESHOLD_REPRESENTATION_SCHEMA,
        "threshold_decode": "endpoint_aware_piecewise_tail_coordinate_v2",
        "threshold_semantics": "prediction = probability > threshold",
        "sigmoid_logit_interpretation_forbidden": True,
        "reject_supported": False,
    }


def _standardizer_tensor(value: Any, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(f"standardizer {name} is not numeric") from error
    if tensor.shape != (93,):
        raise Stage2CalibratorCheckpointV7Error(
            f"standardizer {name} must have shape [93]"
        )
    tensor = tensor.detach().clone().contiguous()
    if not bool(torch.isfinite(tensor).all().item()):
        raise Stage2CalibratorCheckpointV7Error(
            f"standardizer {name} must be finite"
        )
    return tensor


def _feature_mask_tensor(
    feature_mask: VerifiedStage2RC5FeatureMask,
) -> torch.Tensor:
    checked = assert_verified_stage2_rc5_feature_mask(feature_mask)
    return torch.as_tensor(
        checked.boolean_mask.copy(), dtype=torch.bool, device="cpu"
    ).contiguous()


def _make_standardizer(
    mean: Any,
    scale: Any,
    feature_mask: VerifiedStage2RC5FeatureMask,
) -> dict[str, Any]:
    mean_tensor = _standardizer_tensor(mean, "mean")
    scale_tensor = _standardizer_tensor(scale, "scale")
    checked_mask = assert_verified_stage2_rc5_feature_mask(feature_mask)
    if not bool((scale_tensor >= 1e-8).all().item()):
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer scale must be at least the frozen 1e-8 floor"
        )
    tensor_digest = tensor_tree_content_sha256(
        {
            "feature_mask": _feature_mask_tensor(checked_mask),
            "mean": mean_tensor,
            "scale": scale_tensor,
        }
    )
    return {
        "schema_version": STANDARDIZER_SCHEMA,
        "feature_dim": 93,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_sha256": feature_schema_sha256(),
        "feature_schema_digest_algorithm": FEATURE_SCHEMA_DIGEST_ALGORITHM,
        "mean": mean_tensor,
        "scale": scale_tensor,
        "mean_content_sha256": _single_tensor_content_sha256("mean", mean_tensor),
        "scale_content_sha256": _single_tensor_content_sha256("scale", scale_tensor),
        "tensor_content_sha256": tensor_digest,
        "tensor_content_digest_algorithm": TENSOR_CONTENT_DIGEST_ALGORITHM,
        "calculation_dtype": "float64",
        "scale_floor": 1e-8,
        "model_input_dtype": "float32",
        "transformation": (
            "mask_float32(float32((context_float64-mean_float64)/scale_float64))"
        ),
        "feature_mask": feature_mask_payload(checked_mask),
    }


def _verify_standardizer(value: Any) -> Mapping[str, Any]:
    standardizer = _exact_keys(value, _STANDARDIZER_FIELDS, "standardizer")
    if standardizer["schema_version"] != STANDARDIZER_SCHEMA:
        raise Stage2CalibratorCheckpointV7Error("standardizer schema mismatch")
    if standardizer["feature_dim"] != 93:
        raise Stage2CalibratorCheckpointV7Error("standardizer feature_dim mismatch")
    if standardizer["feature_names"] != list(FEATURE_NAMES):
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer feature names/order differ from FEATURE_NAMES"
        )
    if (
        standardizer["feature_schema_sha256"] != feature_schema_sha256()
        or standardizer["feature_schema_digest_algorithm"]
        != FEATURE_SCHEMA_DIGEST_ALGORITHM
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer feature-schema digest mismatch"
        )
    for name in ("mean", "scale"):
        tensor = standardizer[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float64
            or tensor.shape != (93,)
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise Stage2CalibratorCheckpointV7Error(
                f"standardizer {name} must be finite contiguous CPU float64[93]"
            )
    if not bool((standardizer["scale"] >= 1e-8).all().item()):
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer scale violates the frozen 1e-8 floor"
        )
    if standardizer["mean_content_sha256"] != _single_tensor_content_sha256(
        "mean", standardizer["mean"]
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer mean content digest mismatch"
        )
    if standardizer["scale_content_sha256"] != _single_tensor_content_sha256(
        "scale", standardizer["scale"]
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer scale content digest mismatch"
        )
    try:
        feature_mask = verify_stage2_rc5_feature_mask_payload(
            standardizer["feature_mask"]
        )
    except (Stage2RC5FeatureMaskError, TypeError) as error:
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer feature-mask payload mismatch"
        ) from error
    expected_tree_digest = tensor_tree_content_sha256(
        {
            "feature_mask": _feature_mask_tensor(feature_mask),
            "mean": standardizer["mean"],
            "scale": standardizer["scale"],
        }
    )
    if (
        standardizer["tensor_content_sha256"] != expected_tree_digest
        or standardizer["tensor_content_digest_algorithm"]
        != TENSOR_CONTENT_DIGEST_ALGORITHM
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "standardizer tensor content digest mismatch"
        )
    exact_metadata = {
        "calculation_dtype": "float64",
        "scale_floor": 1e-8,
        "model_input_dtype": "float32",
        "transformation": (
            "mask_float32(float32((context_float64-mean_float64)/scale_float64))"
        ),
    }
    for key, expected in exact_metadata.items():
        if standardizer[key] != expected:
            raise Stage2CalibratorCheckpointV7Error(
                f"standardizer {key} mismatch"
            )
    return standardizer


def _trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _expected_model_type(method: str) -> type[nn.Module]:
    if method == "T6":
        return DirectEndpointAwarePixelCalibrator
    if method in {"T7", "T8"}:
        return MonotoneEndpointAwarePixelCalibrator
    raise Stage2CalibratorCheckpointV7Error(f"unsupported v7 method: {method!r}")


def _expected_model_id(method: str) -> str:
    return (
        DIRECT_ENDPOINT_AWARE_MODEL_ID
        if method == "T6"
        else MONOTONE_ENDPOINT_AWARE_MODEL_ID
    )


def _validate_frozen_model_config(method: str, config: Any) -> Mapping[str, Any]:
    direct_fields = {
        "context_feature_dim",
        "pixel_budget_grid",
        "hidden_dims",
        "dropout",
        "raw_coordinate_min_hex",
        "raw_coordinate_max_hex",
        "threshold_representation_schema",
        "anchor_source",
        "anchor_coordinate_contract",
        "anchor_mix_rule",
        "anchor_mix_parameterization",
        "anchor_mix_initial_weight",
    }
    expected_fields = direct_fields | (
        {"minimum_raw_coordinate_gap"} if method in {"T7", "T8"} else set()
    )
    if not isinstance(config, Mapping) or set(config) != expected_fields:
        raise Stage2CalibratorCheckpointV7Error("model_config fields mismatch")
    exact = {
        "context_feature_dim": 93,
        "pixel_budget_grid": list(PIXEL_BUDGET_GRID),
        "hidden_dims": [32],
        "dropout": 0.1,
        "raw_coordinate_min_hex": RAW_COORDINATE_MIN.hex(),
        "raw_coordinate_max_hex": RAW_COORDINATE_MAX.hex(),
        "threshold_representation_schema": THRESHOLD_REPRESENTATION_SCHEMA,
        "anchor_source": T4_ANCHOR_SOURCE,
        "anchor_coordinate_contract": ANCHOR_COORDINATE_CONTRACT,
        "anchor_mix_rule": ANCHOR_MIX_RULE,
        "anchor_mix_parameterization": ANCHOR_MIX_PARAMETERIZATION,
        "anchor_mix_initial_weight": ANCHOR_MIX_INITIAL_WEIGHT,
    }
    if method in {"T7", "T8"}:
        exact["minimum_raw_coordinate_gap"] = 0.001
    if dict(config) != exact:
        raise Stage2CalibratorCheckpointV7Error(
            "model_config differs from the frozen endpoint-aware architecture"
        )
    return config


def _reconstruct_model(method: str, config: Any) -> nn.Module:
    checked = _validate_frozen_model_config(method, config)
    common = {
        "context_feature_dim": checked["context_feature_dim"],
        "pixel_budget_grid": checked["pixel_budget_grid"],
        "hidden_dims": checked["hidden_dims"],
        "dropout": checked["dropout"],
    }
    if method == "T6":
        model: nn.Module = DirectEndpointAwarePixelCalibrator(**common)
    else:
        model = MonotoneEndpointAwarePixelCalibrator(
            **common,
            minimum_raw_coordinate_gap=checked["minimum_raw_coordinate_gap"],
        )
    if model.export_config() != dict(checked):
        raise Stage2CalibratorCheckpointV7Error(
            "reconstructed model changed its export_config"
        )
    return model


def _cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if not state:
        raise Stage2CalibratorCheckpointV7Error("model state_dict is empty")
    result: dict[str, torch.Tensor] = {}
    for key in sorted(state):
        if not isinstance(key, str) or not isinstance(state[key], torch.Tensor):
            raise TypeError("model state_dict must map strings to tensors")
        result[key] = _cpu_contiguous_tensor(state[key], f"state_dict[{key}]").clone()
    return result


def _verify_finite_state(state: Mapping[str, torch.Tensor]) -> None:
    if not state or any(not isinstance(key, str) or not key for key in state):
        raise TypeError("model_state_dict must be a non-empty string-key mapping")
    for key, value in state.items():
        tensor = _cpu_contiguous_tensor(value, f"model_state_dict[{key!r}]")
        if tensor.device.type != "cpu":
            raise TypeError("model_state_dict tensors must be on CPU")
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                raise Stage2CalibratorCheckpointV7Error(
                    f"model_state_dict[{key!r}] is non-finite"
                )


def _anchor_mix_alpha_hex(model: nn.Module) -> str:
    mix_logit = getattr(model, "anchor_mix_logit", None)
    if (
        not isinstance(mix_logit, torch.Tensor)
        or mix_logit.dtype != torch.float64
        or mix_logit.shape != torch.Size([])
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "model anchor_mix_logit must be one scalar float64 tensor"
        )
    value = mix_logit.detach().to(device="cpu")
    if not bool(torch.isfinite(value).item()):
        raise Stage2CalibratorCheckpointV7Error(
            "model anchor_mix_logit must be finite"
        )
    alpha = torch.sigmoid(value)
    return _canonical_anchor_mix_alpha_hex(float(alpha.item()).hex())


def _smoke_anchor_coordinates() -> torch.Tensor:
    coordinates = torch.from_numpy(
        encode_probability_numpy([0.25, 0.75, 1.0])
    ).reshape(1, 3)
    if coordinates.dtype != torch.float64:
        raise RuntimeError("endpoint-aware encoder no longer returns float64")
    return coordinates.contiguous()


def _assert_weights_only_tree(value: Any, name: str = "checkpoint") -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise TypeError(f"{name} tensor must be on CPU")
        return
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise Stage2CalibratorCheckpointV7Error(f"{name} contains non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} mapping keys must be strings")
            _assert_weights_only_tree(item, f"{name}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_weights_only_tree(item, f"{name}[{index}]")
        return
    raise TypeError(f"{name} contains unsupported value type {type(value).__name__}")


def _exercise_cpu_inference(
    model: nn.Module,
    method: str,
    expected_alpha_hex: str,
    feature_mask: VerifiedStage2RC5FeatureMask,
) -> None:
    model = model.to(device="cpu").eval()
    anchor_coordinates = _smoke_anchor_coordinates()
    probe = torch.linspace(-1.0, 1.0, 93, dtype=torch.float32).reshape(1, 93)
    masked_probe = apply_stage2_rc5_feature_mask_torch(probe, feature_mask)
    with torch.inference_mode():
        output = model(
            masked_probe,
            anchor_coordinates=anchor_coordinates,
        )
    if (
        output.pixel_budget_grid.device.type != "cpu"
        or output.pixel_budget_grid.dtype != torch.float64
        or output.pixel_budget_grid.detach().cpu().tolist() != PIXEL_BUDGET_GRID
    ):
        raise Stage2CalibratorCheckpointV7Error("model output budget grid mismatch")
    for name in (
        "anchor_coordinates",
        "grid_learned_raw_coordinates",
        "grid_raw_coordinates",
        "grid_coordinates",
        "grid_thresholds",
    ):
        tensor = getattr(output, name, None)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float64
            or tensor.shape != (1, 3)
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise Stage2CalibratorCheckpointV7Error(
                f"CPU inference produced invalid {name}"
            )
    if output.anchor_coordinates.dtype != torch.float64 or not torch.equal(
        output.anchor_coordinates, anchor_coordinates
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "CPU inference changed the supplied anchor coordinates"
        )
    mix_weight = getattr(output, "anchor_mix_weight", None)
    if (
        not isinstance(mix_weight, torch.Tensor)
        or mix_weight.device.type != "cpu"
        or mix_weight.dtype != torch.float64
        or mix_weight.shape != torch.Size([])
        or not bool(torch.isfinite(mix_weight).item())
        or float(mix_weight.item()).hex()
        != _canonical_anchor_mix_alpha_hex(expected_alpha_hex)
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "CPU inference anchor mix weight differs from checkpoint state"
        )
    expected_raw = (
        (1.0 - mix_weight) * output.anchor_coordinates
        + mix_weight * output.grid_learned_raw_coordinates
    )
    if not torch.equal(expected_raw, output.grid_raw_coordinates):
        raise Stage2CalibratorCheckpointV7Error(
            "CPU inference did not mix the T4 anchor before canonicalization"
        )
    decoded = decode_coordinate_torch(output.grid_coordinates)
    if not torch.equal(decoded, output.grid_thresholds):
        raise Stage2CalibratorCheckpointV7Error(
            "thresholds do not exactly decode from endpoint-aware coordinates"
        )
    if method in {"T7", "T8"}:
        if not bool(
            (output.grid_raw_coordinates[:, 1:]
             > output.grid_raw_coordinates[:, :-1]).all().item()
        ):
            raise Stage2CalibratorCheckpointV7Error(
                "monotone v7 raw coordinates are not strictly increasing"
            )
        if not bool(
            (output.grid_coordinates[:, 1:]
             >= output.grid_coordinates[:, :-1]).all().item()
        ):
            raise Stage2CalibratorCheckpointV7Error(
                "monotone v7 canonical coordinates decrease"
            )


def _verify_payload(payload: Any) -> nn.Module:
    checkpoint = _exact_keys(payload, _CHECKPOINT_FIELDS, "checkpoint-v7")
    if checkpoint["format_version"] != CHECKPOINT_SCHEMA:
        raise Stage2CalibratorCheckpointV7Error(
            "checkpoint is not schema rc-irstd.calibrator.v7"
        )
    if checkpoint["artifact_kind"] != ARTIFACT_KIND:
        raise Stage2CalibratorCheckpointV7Error("checkpoint artifact_kind mismatch")
    method = checkpoint["method"]
    if method not in METHODS:
        raise Stage2CalibratorCheckpointV7Error("checkpoint method mismatch")
    if checkpoint["calibrator_model"] != _expected_model_id(method):
        raise Stage2CalibratorCheckpointV7Error("calibrator_model mismatch")
    if checkpoint["representation_contract"] != representation_contract():
        raise Stage2CalibratorCheckpointV7Error(
            "endpoint-aware representation_contract mismatch"
        )
    for field in ("reject_head", "missing_episode_fallback", "official_test_accessed"):
        _exact_false(checkpoint[field], f"checkpoint.{field}")
    _sha256(checkpoint["training_contract_sha256"], "training_contract_sha256")

    standardizer = _verify_standardizer(checkpoint["standardizer"])
    feature_mask = verify_stage2_rc5_feature_mask_payload(
        standardizer["feature_mask"]
    )
    if checkpoint["standardizer_content_sha256"] != standardizer[
        "tensor_content_sha256"
    ]:
        raise Stage2CalibratorCheckpointV7Error(
            "checkpoint/standardizer content digest mismatch"
        )

    model = _reconstruct_model(method, checkpoint["model_config"])
    expected_parameters = EXPECTED_PARAMETER_COUNTS[method]
    if (
        checkpoint["expected_trainable_parameters"] != expected_parameters
        or _trainable_parameter_count(model) != expected_parameters
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "checkpoint trainable parameter count mismatch"
        )
    capability = checkpoint["capability_contract"]
    if not isinstance(capability, Mapping):
        raise Stage2CalibratorCheckpointV7Error(
            "endpoint-aware model capability contract must be a mapping"
        )
    if capability != model.capability_contract():
        raise Stage2CalibratorCheckpointV7Error(
            "endpoint-aware model capability contract mismatch"
        )
    capability_representation = capability.get(
        "threshold_representation"
    )
    if capability_representation != representation_contract():
        raise Stage2CalibratorCheckpointV7Error(
            "capability representation contract mismatch"
        )

    state = checkpoint["model_state_dict"]
    if not isinstance(state, Mapping) or any(
        not isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise TypeError("model_state_dict must map strings to tensors")
    _verify_finite_state(state)
    observed_state_digest = tensor_tree_content_sha256(state)
    if (
        checkpoint["model_state_content_sha256"] != observed_state_digest
        or checkpoint["model_state_content_digest_algorithm"]
        != TENSOR_CONTENT_DIGEST_ALGORITHM
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "model_state_dict content digest mismatch"
        )
    try:
        model.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2CalibratorCheckpointV7Error(
            "strict endpoint-aware model state replay failed"
        ) from error
    if tensor_tree_content_sha256(model.state_dict()) != observed_state_digest:
        raise Stage2CalibratorCheckpointV7Error(
            "strict model load changed state content"
        )
    anchor_mix_alpha_hex = _anchor_mix_alpha_hex(model)
    if checkpoint["inference_contract"] != cpu_inference_contract(
        anchor_mix_alpha_hex, feature_mask
    ):
        raise Stage2CalibratorCheckpointV7Error("CPU inference contract mismatch")
    _exercise_cpu_inference(model, method, anchor_mix_alpha_hex, feature_mask)
    return model.to(device="cpu").eval()


def make_calibrator_checkpoint_v7(
    *,
    method: str,
    model: nn.Module,
    standardizer_mean: Any,
    standardizer_scale: Any,
    training_contract_sha256: str,
    feature_mask: VerifiedStage2RC5FeatureMask | None = None,
) -> dict[str, Any]:
    """Build one strict tensors/primitives-only deployment checkpoint."""

    if method not in METHODS:
        raise Stage2CalibratorCheckpointV7Error(f"unsupported v7 method: {method!r}")
    expected_type = _expected_model_type(method)
    if type(model) is not expected_type:
        raise TypeError(
            f"{method} requires exact {expected_type.__name__}, "
            f"got {type(model).__name__}"
        )
    state = _cpu_model_state(model)
    anchor_mix_alpha_hex = _anchor_mix_alpha_hex(model)
    checked_mask = assert_verified_stage2_rc5_feature_mask(
        build_stage2_rc5_feature_mask("C3")
        if feature_mask is None
        else feature_mask
    )
    standardizer = _make_standardizer(
        standardizer_mean, standardizer_scale, checked_mask
    )
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "method": method,
        "calibrator_model": _expected_model_id(method),
        "model_config": dict(model.export_config()),
        "capability_contract": dict(model.capability_contract()),
        "representation_contract": representation_contract(),
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNTS[method],
        "model_state_dict": state,
        "model_state_content_sha256": tensor_tree_content_sha256(state),
        "model_state_content_digest_algorithm": TENSOR_CONTENT_DIGEST_ALGORITHM,
        "standardizer": standardizer,
        "standardizer_content_sha256": standardizer["tensor_content_sha256"],
        "training_contract_sha256": _sha256(
            training_contract_sha256, "training_contract_sha256"
        ),
        "inference_contract": cpu_inference_contract(
            anchor_mix_alpha_hex, checked_mask
        ),
        "reject_head": False,
        "missing_episode_fallback": False,
        "official_test_accessed": False,
    }
    _assert_weights_only_tree(payload)
    _verify_payload(payload)
    return payload


def serialize_calibrator_checkpoint_v7(payload: Mapping[str, Any]) -> bytes:
    """Serialize and restricted-load replay one validated v7 payload."""

    _assert_weights_only_tree(payload)
    _verify_payload(payload)
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    data = stream.getvalue()
    try:
        replay = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage2CalibratorCheckpointV7Error(
            "serialized checkpoint is not weights-only loadable"
        ) from error
    _verify_payload(replay)
    return data


@dataclass(frozen=True, init=False)
class VerifiedCalibratorCheckpointV7:
    sha256: str
    method: str
    training_contract_sha256: str
    checkpoint_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "VerifiedCalibratorCheckpointV7 can only be created by a public verifier"
        )

    def payload(self) -> dict[str, Any]:
        value = torch.load(
            io.BytesIO(self.checkpoint_bytes), map_location="cpu", weights_only=True
        )
        _verify_payload(value)
        return value

    def model(self) -> nn.Module:
        return _verify_payload(self.payload())


def _make_verified(
    *, data: bytes, digest: str, payload: Mapping[str, Any]
) -> VerifiedCalibratorCheckpointV7:
    result = object.__new__(VerifiedCalibratorCheckpointV7)
    object.__setattr__(result, "sha256", digest)
    object.__setattr__(result, "method", payload["method"])
    object.__setattr__(
        result, "training_contract_sha256", payload["training_contract_sha256"]
    )
    object.__setattr__(result, "checkpoint_bytes", bytes(data))
    return result


def verify_calibrator_checkpoint_v7_bytes(
    data: bytes,
    expected_sha256: str | None = None,
    *,
    expected_method: str | None = None,
    expected_training_contract_sha256: str | None = None,
) -> VerifiedCalibratorCheckpointV7:
    """Restricted-load and fully replay checkpoint bytes."""

    if not isinstance(data, bytes) or not data:
        raise TypeError("checkpoint data must be non-empty bytes")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != _sha256(
        expected_sha256, "expected checkpoint SHA-256"
    ):
        raise Stage2CalibratorCheckpointV7Error("checkpoint SHA-256 mismatch")
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage2CalibratorCheckpointV7Error(
            "checkpoint is not tensors/primitives-only weights-only data"
        ) from error
    _verify_payload(payload)
    if expected_method is not None and payload["method"] != expected_method:
        raise Stage2CalibratorCheckpointV7Error("expected method mismatch")
    if expected_training_contract_sha256 is not None and payload[
        "training_contract_sha256"
    ] != _sha256(
        expected_training_contract_sha256,
        "expected training contract SHA-256",
    ):
        raise Stage2CalibratorCheckpointV7Error(
            "expected training contract SHA-256 mismatch"
        )
    return _make_verified(data=data, digest=digest, payload=payload)


def _read_regular_file_stable(path: str | Path) -> bytes:
    raw = Path(path).expanduser().absolute()
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Stage2CalibratorCheckpointV7Error(
                "checkpoint path contains a symlink component"
            )
    if not raw.exists() or not stat.S_ISREG(raw.lstat().st_mode):
        raise FileNotFoundError(f"checkpoint is not a regular file: {raw}")
    descriptor = os.open(raw, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            data = handle.read()
        after = os.fstat(descriptor)
        path_after = os.stat(raw, follow_symlinks=False)
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(path_after):
        raise Stage2CalibratorCheckpointV7Error(
            "checkpoint file changed while read"
        )
    return data


def verify_calibrator_checkpoint_v7_file(
    path: str | Path,
    expected_sha256: str,
    *,
    expected_method: str | None = None,
    expected_training_contract_sha256: str | None = None,
) -> VerifiedCalibratorCheckpointV7:
    """Verify a stable non-symlink file against an external SHA-256."""

    data = _read_regular_file_stable(path)
    return verify_calibrator_checkpoint_v7_bytes(
        data,
        expected_sha256,
        expected_method=expected_method,
        expected_training_contract_sha256=expected_training_contract_sha256,
    )


__all__ = [
    "ARTIFACT_KIND",
    "CHECKPOINT_SCHEMA",
    "EXPECTED_PARAMETER_COUNTS",
    "FEATURE_SCHEMA_DIGEST_ALGORITHM",
    "METHODS",
    "PIXEL_BUDGET_GRID",
    "STANDARDIZER_SCHEMA",
    "Stage2CalibratorCheckpointV7Error",
    "TENSOR_CONTENT_DIGEST_ALGORITHM",
    "VerifiedCalibratorCheckpointV7",
    "cpu_inference_contract",
    "feature_schema_sha256",
    "make_calibrator_checkpoint_v7",
    "serialize_calibrator_checkpoint_v7",
    "tensor_tree_content_sha256",
    "verify_calibrator_checkpoint_v7_bytes",
    "verify_calibrator_checkpoint_v7_file",
]
