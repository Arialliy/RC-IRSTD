"""Strict checkpoint-v8 deployment core for RC5+ residual transport.

The artifact is deployment-only and tensors/primitives-only.  It binds one
capacity-matched residual-transport model, the exact nine-rational budget
lattice, the C3 feature mask and standardizer, the four-role RC5+ training-view
identity, and the anchor-v2/no-reject CPU inference contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn

from model.budget_conditioned_endpoint_calibrator import (
    BUDGET_KNOT_RATIONALS,
    PRIMARY_BUDGET_KNOT_INDICES,
)
from model.budget_conditioned_residual_transport_calibrator import (
    RESIDUAL_TRANSPORT_DIRECT_MODEL_ID,
    RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID,
    RESIDUAL_TRANSPORT_NO_ANCHOR_MODEL_ID,
    RESIDUAL_TRANSPORT_RULE,
    BudgetConditionedDirectResidualTransportCalibrator,
    BudgetConditionedMonotoneNoTargetAnchorCalibrator,
    BudgetConditionedMonotoneResidualTransportCalibrator,
)
from model.endpoint_aware_threshold import (
    THRESHOLD_REPRESENTATION_SCHEMA,
    decode_coordinate_torch,
    encode_probability_numpy,
    representation_contract,
)
from rc.stage2_calibrator_checkpoint_v7 import (
    FEATURE_SCHEMA_DIGEST_ALGORITHM,
    STANDARDIZER_SCHEMA,
    TENSOR_CONTENT_DIGEST_ALGORITHM,
    _assert_weights_only_tree,
    _cpu_model_state,
    _make_standardizer,
    _read_regular_file_stable,
    _verify_finite_state,
    _verify_standardizer,
    tensor_tree_content_sha256,
)
from rc.stage2_rc5_feature_mask import (
    FEATURE_MASK_APPLICATION,
    VerifiedStage2RC5FeatureMask,
    apply_stage2_rc5_feature_mask_torch,
    assert_verified_stage2_rc5_feature_mask,
    build_stage2_rc5_feature_mask,
    verify_stage2_rc5_feature_mask_payload,
)


CHECKPOINT_SCHEMA = "rc-irstd.calibrator.v8"
ARTIFACT_KIND = "immutable_budget_conditioned_residual_transport_deployment_state"
METHODS = ("T6_PLUS", "T7_PLUS", "T8_PLUS")
ABLATION_METHODS = ("T8_PLUS_NO_ANCHOR",)
SUPPORTED_METHODS = METHODS + ABLATION_METHODS
EXPECTED_PARAMETER_COUNTS = {method: 3339 for method in SUPPORTED_METHODS}
_SHA_HEX = frozenset("0123456789abcdef")
_FIELDS = frozenset(
    {
        "format_version",
        "artifact_kind",
        "method",
        "calibrator_model",
        "model_config",
        "capability_contract",
        "representation_contract",
        "budget_knot_rationals",
        "primary_budget_knot_indices",
        "expected_trainable_parameters",
        "model_state_dict",
        "model_state_content_sha256",
        "model_state_content_digest_algorithm",
        "standardizer",
        "standardizer_content_sha256",
        "training_contract_sha256",
        "training_view_identity_sha256",
        "inference_contract",
        "reject_head",
        "missing_episode_fallback",
        "anchor_overlay_required",
        "official_test_accessed",
    }
)


class Stage2CalibratorCheckpointV8Error(ValueError):
    """A checkpoint violates the strict RC5+ v8 contract."""


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _SHA_HEX
    ):
        raise Stage2CalibratorCheckpointV8Error(
            f"{name} must be one lowercase SHA-256 digest"
        )
    return value


def _exact_false(value: Any, name: str) -> None:
    if type(value) is not bool or value is not False:
        raise Stage2CalibratorCheckpointV8Error(f"{name} must be exact false")


def _exact_true(value: Any, name: str) -> None:
    if type(value) is not bool or value is not True:
        raise Stage2CalibratorCheckpointV8Error(f"{name} must be exact true")


def _model_type(method: str) -> type[nn.Module]:
    if method == "T6_PLUS":
        return BudgetConditionedDirectResidualTransportCalibrator
    if method in {"T7_PLUS", "T8_PLUS"}:
        return BudgetConditionedMonotoneResidualTransportCalibrator
    if method == "T8_PLUS_NO_ANCHOR":
        return BudgetConditionedMonotoneNoTargetAnchorCalibrator
    raise Stage2CalibratorCheckpointV8Error(f"unsupported v8 method: {method!r}")


def _model_id(method: str) -> str:
    if method == "T6_PLUS":
        return RESIDUAL_TRANSPORT_DIRECT_MODEL_ID
    if method in {"T7_PLUS", "T8_PLUS"}:
        return RESIDUAL_TRANSPORT_MONOTONE_MODEL_ID
    if method == "T8_PLUS_NO_ANCHOR":
        return RESIDUAL_TRANSPORT_NO_ANCHOR_MODEL_ID
    raise Stage2CalibratorCheckpointV8Error(f"unsupported v8 method: {method!r}")


def _new_model(method: str) -> nn.Module:
    return _model_type(method)(
        context_feature_dim=93,
        hidden_dims=(32,),
        dropout=0.1,
        minimum_residual_increment=1e-6,
    )


def _reconstruct_model(method: str, config: Any) -> nn.Module:
    model = _new_model(method)
    if not isinstance(config, Mapping) or dict(config) != model.export_config():
        raise Stage2CalibratorCheckpointV8Error(
            "model_config differs from the frozen RC5+ architecture"
        )
    return model


def _correction_strength_hex(model: nn.Module) -> str:
    value = getattr(model, "correction_strength_logit", None)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float64
        or value.shape != torch.Size([])
        or not bool(torch.isfinite(value).item())
    ):
        raise Stage2CalibratorCheckpointV8Error(
            "correction_strength_logit must be one finite float64 scalar"
        )
    alpha = float(torch.sigmoid(value.detach().to(device="cpu")).item())
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise Stage2CalibratorCheckpointV8Error(
            "correction strength must lie strictly inside (0,1)"
        )
    return alpha.hex()


def cpu_inference_contract_v2(
    *,
    correction_strength_hex: str,
    feature_mask: VerifiedStage2RC5FeatureMask,
    anchor_overlay_required: bool = True,
) -> dict[str, Any]:
    try:
        alpha = float.fromhex(correction_strength_hex)
    except (TypeError, ValueError) as error:
        raise Stage2CalibratorCheckpointV8Error(
            "correction_strength_hex must be canonical float.hex text"
        ) from error
    if (
        not math.isfinite(alpha)
        or not 0.0 < alpha < 1.0
        or alpha.hex() != correction_strength_hex
    ):
        raise Stage2CalibratorCheckpointV8Error(
            "correction_strength_hex must canonically encode (0,1)"
        )
    if type(anchor_overlay_required) is not bool:
        raise TypeError("anchor_overlay_required must be exact bool")
    mask = assert_verified_stage2_rc5_feature_mask(feature_mask)
    output_fields = (
        [
            "anchor_coordinates",
            "correction_strength",
            "anchor_slope",
            "grid_residual",
            "grid_transport_latent",
            "grid_raw_coordinates",
            "grid_coordinates",
            "grid_thresholds",
        ]
        if anchor_overlay_required
        else [
            "correction_strength",
            "context_scale",
            "grid_residual",
            "grid_transport_latent",
            "grid_raw_coordinates",
            "grid_coordinates",
            "grid_thresholds",
        ]
    )
    return {
        "schema_version": "rc-irstd.calibrator-cpu-inference.v2",
        "device": "cpu",
        "model_mode": "eval",
        "autograd_enabled": False,
        "context_feature_dim": 93,
        "standardizer_compute_dtype": "float64",
        "model_input_dtype": "float32",
        "model_input_shape": "[batch,93]",
        "feature_mask_variant": mask.variant,
        "feature_mask_identity_sha256": mask.identity_sha256,
        "feature_mask_application": FEATURE_MASK_APPLICATION,
        "anchor_algorithm": (
            "stage2_context_tail_anchor_v2"
            if anchor_overlay_required
            else "not_applicable_no_target_anchor_ablation"
        ),
        "anchor_input_dtype": "float64" if anchor_overlay_required else "not_applicable",
        "anchor_input_shape": "[batch,9]" if anchor_overlay_required else "not_applicable",
        "anchor_source": (
            "direct_same_budget_unlabelled_context_order_statistic_not_interpolation"
            if anchor_overlay_required
            else "none_target_anchor_forbidden"
        ),
        "anchor_overlay_required": anchor_overlay_required,
        "budget_knot_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
        "primary_budget_knot_indices": list(PRIMARY_BUDGET_KNOT_INDICES),
        "budget_request": "exact_rational_grid_or_in_range_ordered_rationals",
        "residual_transport_rule": RESIDUAL_TRANSPORT_RULE,
        "correction_strength_hex": correction_strength_hex,
        "output_dtype": "float64",
        "output_fields": output_fields,
        "threshold_representation_schema": THRESHOLD_REPRESENTATION_SCHEMA,
        "threshold_decode": "endpoint_aware_piecewise_tail_coordinate_v2",
        "threshold_semantics": "prediction = probability > threshold",
        "float_budget_authority_forbidden": True,
        "caller_threshold_injection_forbidden": True,
        "reject_supported": False,
        "fallback_supported": False,
    }


def _exercise_cpu_inference(
    model: nn.Module,
    method: str,
    correction_strength_hex: str,
    feature_mask: VerifiedStage2RC5FeatureMask,
) -> None:
    model = model.to(device="cpu").eval()
    anchor = torch.from_numpy(
        encode_probability_numpy(np.linspace(0.15, 0.95, 9, dtype=np.float64))
    ).reshape(1, 9)
    probe = torch.linspace(-1.0, 1.0, 93, dtype=torch.float32).reshape(1, 93)
    probe = apply_stage2_rc5_feature_mask_torch(probe, feature_mask)
    with torch.inference_mode():
        if method == "T8_PLUS_NO_ANCHOR":
            output = model(probe)
        else:
            output = model(probe, anchor_coordinates=anchor)
    if method != "T8_PLUS_NO_ANCHOR" and not torch.equal(
        output.anchor_coordinates, anchor
    ):
        raise Stage2CalibratorCheckpointV8Error(
            "CPU inference changed the supplied anchor-v2 coordinates"
        )
    if output.budget_knot_numerators.tolist() != [
        row[0] for row in BUDGET_KNOT_RATIONALS
    ] or output.budget_knot_denominators.tolist() != [
        row[1] for row in BUDGET_KNOT_RATIONALS
    ]:
        raise Stage2CalibratorCheckpointV8Error(
            "CPU inference exact budget lattice mismatch"
        )
    diagnostic_fields = (
        ("grid_anchor_latent",)
        if method != "T8_PLUS_NO_ANCHOR"
        else ()
    ) + (
        "grid_residual",
        "grid_transport_latent",
        "grid_raw_coordinates",
        "grid_coordinates",
        "grid_thresholds",
    )
    for name in diagnostic_fields:
        value = getattr(output, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float64
            or value.shape != (1, 9)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise Stage2CalibratorCheckpointV8Error(
                f"CPU inference produced invalid {name}"
            )
    if float(output.correction_strength.item()).hex() != correction_strength_hex:
        raise Stage2CalibratorCheckpointV8Error(
            "CPU inference correction strength differs from strict state"
        )
    if not torch.equal(
        decode_coordinate_torch(output.grid_coordinates), output.grid_thresholds
    ):
        raise Stage2CalibratorCheckpointV8Error(
            "CPU inference thresholds do not decode from EATC-v2 coordinates"
        )
    if method in {"T7_PLUS", "T8_PLUS", "T8_PLUS_NO_ANCHOR"} and (
        not bool((output.grid_residual[:, 1:] >= output.grid_residual[:, :-1]).all())
        or not bool(
            (output.grid_raw_coordinates[:, 1:] >= output.grid_raw_coordinates[:, :-1]).all()
        )
        or not bool(
            (output.grid_coordinates[:, 1:] >= output.grid_coordinates[:, :-1]).all()
        )
    ):
        raise Stage2CalibratorCheckpointV8Error(
            "monotone RC5+ CPU inference violated structural order"
        )


def _verify_payload(payload: Any) -> nn.Module:
    if not isinstance(payload, Mapping) or set(payload) != _FIELDS:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 exact field closure mismatch"
        )
    if payload["format_version"] != CHECKPOINT_SCHEMA or payload[
        "artifact_kind"
    ] != ARTIFACT_KIND:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 schema/artifact kind mismatch"
        )
    method = payload["method"]
    if method not in SUPPORTED_METHODS or payload["calibrator_model"] != _model_id(method):
        raise Stage2CalibratorCheckpointV8Error("checkpoint-v8 method/model mismatch")
    if payload["representation_contract"] != representation_contract():
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 threshold representation mismatch"
        )
    if payload["budget_knot_rationals"] != [list(row) for row in BUDGET_KNOT_RATIONALS] or payload[
        "primary_budget_knot_indices"
    ] != list(PRIMARY_BUDGET_KNOT_INDICES):
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 exact rational budget lattice mismatch"
        )
    for field in ("reject_head", "missing_episode_fallback", "official_test_accessed"):
        _exact_false(payload[field], field)
    anchor_required = method != "T8_PLUS_NO_ANCHOR"
    if payload["anchor_overlay_required"] is not anchor_required:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 anchor-overlay ablation contract mismatch"
        )
    _sha(payload["training_contract_sha256"], "training_contract_sha256")
    _sha(payload["training_view_identity_sha256"], "training_view_identity_sha256")
    try:
        standardizer = _verify_standardizer(payload["standardizer"])
    except Exception as error:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 standardizer verification failed"
        ) from error
    if payload["standardizer_content_sha256"] != standardizer[
        "tensor_content_sha256"
    ]:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 standardizer digest mismatch"
        )
    feature_mask = verify_stage2_rc5_feature_mask_payload(
        standardizer["feature_mask"]
    )
    model = _reconstruct_model(method, payload["model_config"])
    if payload["capability_contract"] != model.capability_contract():
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 model capability contract mismatch"
        )
    if payload["expected_trainable_parameters"] != EXPECTED_PARAMETER_COUNTS[
        method
    ] or sum(parameter.numel() for parameter in model.parameters()) != 3339:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 trainable parameter count mismatch"
        )
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise Stage2CalibratorCheckpointV8Error("checkpoint-v8 model state is invalid")
    try:
        _verify_finite_state(state)
    except Exception as error:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 model state contains invalid tensors"
        ) from error
    observed = tensor_tree_content_sha256(state)
    if payload["model_state_content_sha256"] != observed or payload[
        "model_state_content_digest_algorithm"
    ] != TENSOR_CONTENT_DIGEST_ALGORITHM:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 model state content digest mismatch"
        )
    try:
        model.load_state_dict(state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 strict residual-transport state replay failed"
        ) from error
    if tensor_tree_content_sha256(model.state_dict()) != observed:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 strict model load changed state content"
        )
    strength = _correction_strength_hex(model)
    expected_contract = cpu_inference_contract_v2(
        correction_strength_hex=strength,
        feature_mask=feature_mask,
        anchor_overlay_required=anchor_required,
    )
    if payload["inference_contract"] != expected_contract:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 CPU inference contract mismatch"
        )
    _exercise_cpu_inference(model, method, strength, feature_mask)
    return model.to(device="cpu").eval()


def make_calibrator_checkpoint_v8(
    *,
    method: str,
    model: nn.Module,
    standardizer_mean: Any,
    standardizer_scale: Any,
    training_contract_sha256: str,
    training_view_identity_sha256: str,
    feature_mask: VerifiedStage2RC5FeatureMask | None = None,
) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise Stage2CalibratorCheckpointV8Error(f"unsupported v8 method: {method!r}")
    expected_type = _model_type(method)
    if type(model) is not expected_type:
        raise TypeError(
            f"{method} requires exact {expected_type.__name__}, got {type(model).__name__}"
        )
    mask = assert_verified_stage2_rc5_feature_mask(
        build_stage2_rc5_feature_mask("C3") if feature_mask is None else feature_mask
    )
    try:
        standardizer = _make_standardizer(
            standardizer_mean, standardizer_scale, mask
        )
    except Exception as error:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 standardizer construction failed"
        ) from error
    state = _cpu_model_state(model)
    strength = _correction_strength_hex(model)
    payload = {
        "format_version": CHECKPOINT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "method": method,
        "calibrator_model": _model_id(method),
        "model_config": dict(model.export_config()),
        "capability_contract": dict(model.capability_contract()),
        "representation_contract": representation_contract(),
        "budget_knot_rationals": [list(row) for row in BUDGET_KNOT_RATIONALS],
        "primary_budget_knot_indices": list(PRIMARY_BUDGET_KNOT_INDICES),
        "expected_trainable_parameters": EXPECTED_PARAMETER_COUNTS[method],
        "model_state_dict": state,
        "model_state_content_sha256": tensor_tree_content_sha256(state),
        "model_state_content_digest_algorithm": TENSOR_CONTENT_DIGEST_ALGORITHM,
        "standardizer": standardizer,
        "standardizer_content_sha256": standardizer["tensor_content_sha256"],
        "training_contract_sha256": _sha(
            training_contract_sha256, "training_contract_sha256"
        ),
        "training_view_identity_sha256": _sha(
            training_view_identity_sha256, "training_view_identity_sha256"
        ),
        "inference_contract": cpu_inference_contract_v2(
            correction_strength_hex=strength,
            feature_mask=mask,
            anchor_overlay_required=(method != "T8_PLUS_NO_ANCHOR"),
        ),
        "reject_head": False,
        "missing_episode_fallback": False,
        "anchor_overlay_required": method != "T8_PLUS_NO_ANCHOR",
        "official_test_accessed": False,
    }
    _assert_weights_only_tree(payload)
    _verify_payload(payload)
    return payload


def serialize_calibrator_checkpoint_v8(payload: Mapping[str, Any]) -> bytes:
    _assert_weights_only_tree(payload)
    _verify_payload(payload)
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    data = stream.getvalue()
    try:
        replay = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage2CalibratorCheckpointV8Error(
            "serialized checkpoint-v8 is not weights-only loadable"
        ) from error
    _verify_payload(replay)
    return data


@dataclass(frozen=True, init=False)
class VerifiedCalibratorCheckpointV8:
    sha256: str
    method: str
    training_contract_sha256: str
    training_view_identity_sha256: str
    checkpoint_bytes: bytes

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("VerifiedCalibratorCheckpointV8 is public-verifier-issued only")

    def payload(self) -> dict[str, Any]:
        value = torch.load(
            io.BytesIO(self.checkpoint_bytes), map_location="cpu", weights_only=True
        )
        _verify_payload(value)
        return value

    def model(self) -> nn.Module:
        return _verify_payload(self.payload())


def verify_calibrator_checkpoint_v8_bytes(
    data: bytes,
    expected_sha256: str | None = None,
    *,
    expected_method: str | None = None,
    expected_training_contract_sha256: str | None = None,
    expected_training_view_identity_sha256: str | None = None,
) -> VerifiedCalibratorCheckpointV8:
    if not isinstance(data, bytes) or not data:
        raise TypeError("checkpoint-v8 data must be nonempty bytes")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != _sha(
        expected_sha256, "expected checkpoint SHA-256"
    ):
        raise Stage2CalibratorCheckpointV8Error("checkpoint-v8 SHA-256 mismatch")
    try:
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 is not tensors/primitives-only weights-only data"
        ) from error
    _verify_payload(payload)
    if expected_method is not None and payload["method"] != expected_method:
        raise Stage2CalibratorCheckpointV8Error("checkpoint-v8 expected method mismatch")
    for expected_value, field in (
        (expected_training_contract_sha256, "training_contract_sha256"),
        (expected_training_view_identity_sha256, "training_view_identity_sha256"),
    ):
        if expected_value is not None and payload[field] != _sha(
            expected_value, f"expected {field}"
        ):
            raise Stage2CalibratorCheckpointV8Error(
                f"checkpoint-v8 expected {field} mismatch"
            )
    result = object.__new__(VerifiedCalibratorCheckpointV8)
    for name, value in {
        "sha256": digest,
        "method": payload["method"],
        "training_contract_sha256": payload["training_contract_sha256"],
        "training_view_identity_sha256": payload[
            "training_view_identity_sha256"
        ],
        "checkpoint_bytes": bytes(data),
    }.items():
        object.__setattr__(result, name, value)
    return result


def verify_calibrator_checkpoint_v8_file(
    path: str | Path,
    expected_sha256: str,
    **kwargs: Any,
) -> VerifiedCalibratorCheckpointV8:
    try:
        data = _read_regular_file_stable(path)
    except Exception as error:
        raise Stage2CalibratorCheckpointV8Error(
            "checkpoint-v8 file is not a stable direct regular file"
        ) from error
    return verify_calibrator_checkpoint_v8_bytes(
        data, expected_sha256, **kwargs
    )


__all__ = [
    "ARTIFACT_KIND",
    "ABLATION_METHODS",
    "CHECKPOINT_SCHEMA",
    "EXPECTED_PARAMETER_COUNTS",
    "FEATURE_SCHEMA_DIGEST_ALGORITHM",
    "METHODS",
    "STANDARDIZER_SCHEMA",
    "SUPPORTED_METHODS",
    "Stage2CalibratorCheckpointV8Error",
    "TENSOR_CONTENT_DIGEST_ALGORITHM",
    "VerifiedCalibratorCheckpointV8",
    "cpu_inference_contract_v2",
    "make_calibrator_checkpoint_v8",
    "serialize_calibrator_checkpoint_v8",
    "tensor_tree_content_sha256",
    "verify_calibrator_checkpoint_v8_bytes",
    "verify_calibrator_checkpoint_v8_file",
]
