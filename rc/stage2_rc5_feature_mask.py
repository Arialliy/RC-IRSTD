"""Frozen RC5 feature-mask identities and exact post-standardization replay.

Feature ablations keep the 93D architecture unchanged.  A mask is applied
only after train-fit float64 standardization and the float32 cast, immediately
before the model call.  Inactive entries become exact positive zero; neither
the standardizer fit nor the raw context artifact is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch


FEATURE_DIM = 93
FEATURE_MASK_SCHEMA = "rc-irstd.stage2-rc5-feature-mask.v1"
FEATURE_MASK_ID_ALGORITHM = (
    "sha256-canonical-json-rc5-feature-mask-without-identity-v1"
)
FEATURE_MASK_APPLICATION = (
    "float64_standardize_then_float32_cast_then_exact_positive_zero_mask_v1"
)

FEATURE_VARIANT_ACTIVE_INDICES = MappingProxyType(
    {
        "C3": tuple(range(93)),
        "C4": tuple(range(39)),
        "C5": tuple(range(79)),
        "C6": tuple(range(87)),
    }
)

_TOKEN = object()
_SHA_CHARS = frozenset("0123456789abcdef")
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "variant",
        "feature_dim",
        "active_indices",
        "inactive_indices",
        "active_count",
        "application",
        "inactive_value_float32_hex",
        "identity_algorithm",
        "identity_sha256",
    }
)


class Stage2RC5FeatureMaskError(ValueError):
    """A feature mask or its application order violated the RC5 freeze."""


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
        raise Stage2RC5FeatureMaskError(
            "feature-mask value is not finite canonical JSON"
        ) from error


def _identity(payload: Mapping[str, Any]) -> str:
    projection = {
        key: value for key, value in payload.items() if key != "identity_sha256"
    }
    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise Stage2RC5FeatureMaskError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, init=False)
class VerifiedStage2RC5FeatureMask:
    variant: str
    active_indices: tuple[int, ...]
    inactive_indices: tuple[int, ...]
    boolean_mask: np.ndarray
    payload: Mapping[str, Any]
    identity_sha256: str
    _capability: object

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("VerifiedStage2RC5FeatureMask is verifier-issued only")


def _issue(payload: Mapping[str, Any]) -> VerifiedStage2RC5FeatureMask:
    variant = str(payload["variant"])
    active = tuple(int(value) for value in payload["active_indices"])
    inactive = tuple(int(value) for value in payload["inactive_indices"])
    mask = np.zeros(FEATURE_DIM, dtype=np.bool_)
    mask[np.asarray(active, dtype=np.int64)] = True
    mask = np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_)
    mask.setflags(write=False)
    frozen_payload = MappingProxyType(
        {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in payload.items()
        }
    )
    result = object.__new__(VerifiedStage2RC5FeatureMask)
    for name, value in {
        "variant": variant,
        "active_indices": active,
        "inactive_indices": inactive,
        "boolean_mask": mask,
        "payload": frozen_payload,
        "identity_sha256": str(payload["identity_sha256"]),
        "_capability": _TOKEN,
    }.items():
        object.__setattr__(result, name, value)
    return result


def build_stage2_rc5_feature_mask(
    variant: str = "C3",
) -> VerifiedStage2RC5FeatureMask:
    """Build one of the four preregistered fixed-width feature masks."""

    if not isinstance(variant, str) or variant not in FEATURE_VARIANT_ACTIVE_INDICES:
        raise Stage2RC5FeatureMaskError(
            f"variant must be one of {tuple(FEATURE_VARIANT_ACTIVE_INDICES)}"
        )
    active = FEATURE_VARIANT_ACTIVE_INDICES[variant]
    active_set = set(active)
    inactive = tuple(index for index in range(FEATURE_DIM) if index not in active_set)
    payload: dict[str, Any] = {
        "schema_version": FEATURE_MASK_SCHEMA,
        "variant": variant,
        "feature_dim": FEATURE_DIM,
        "active_indices": list(active),
        "inactive_indices": list(inactive),
        "active_count": len(active),
        "application": FEATURE_MASK_APPLICATION,
        "inactive_value_float32_hex": float(np.float32(0.0)).hex(),
        "identity_algorithm": FEATURE_MASK_ID_ALGORITHM,
        "identity_sha256": "",
    }
    payload["identity_sha256"] = _identity(payload)
    return _issue(payload)


def verify_stage2_rc5_feature_mask_payload(
    value: Any,
) -> VerifiedStage2RC5FeatureMask:
    """Verify a checkpoint/training payload and issue a fresh capability."""

    if not isinstance(value, Mapping) or set(value) != _PAYLOAD_FIELDS:
        raise Stage2RC5FeatureMaskError("feature-mask payload field closure mismatch")
    payload = {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in value.items()
    }
    if payload["schema_version"] != FEATURE_MASK_SCHEMA:
        raise Stage2RC5FeatureMaskError("feature-mask schema mismatch")
    variant = payload["variant"]
    if variant not in FEATURE_VARIANT_ACTIVE_INDICES:
        raise Stage2RC5FeatureMaskError("feature-mask variant is not preregistered")
    active_raw = payload["active_indices"]
    inactive_raw = payload["inactive_indices"]
    if not isinstance(active_raw, list) or not isinstance(inactive_raw, list):
        raise TypeError("feature-mask indices must be ordered JSON arrays")
    active = tuple(active_raw)
    inactive = tuple(inactive_raw)
    expected_active = FEATURE_VARIANT_ACTIVE_INDICES[str(variant)]
    expected_inactive = tuple(
        index for index in range(FEATURE_DIM) if index not in set(expected_active)
    )
    if (
        any(type(index) is not int for index in active + inactive)
        or active != expected_active
        or inactive != expected_inactive
        or payload["feature_dim"] != FEATURE_DIM
        or payload["active_count"] != len(expected_active)
        or payload["application"] != FEATURE_MASK_APPLICATION
        or payload["inactive_value_float32_hex"] != float(np.float32(0.0)).hex()
        or payload["identity_algorithm"] != FEATURE_MASK_ID_ALGORITHM
    ):
        raise Stage2RC5FeatureMaskError("feature-mask frozen semantics mismatch")
    declared = _sha256(payload["identity_sha256"], "feature-mask identity")
    if declared != _identity(payload):
        raise Stage2RC5FeatureMaskError("feature-mask identity SHA-256 mismatch")
    return _issue(payload)


def assert_verified_stage2_rc5_feature_mask(
    value: object,
) -> VerifiedStage2RC5FeatureMask:
    if (
        type(value) is not VerifiedStage2RC5FeatureMask
        or getattr(value, "_capability", None) is not _TOKEN
    ):
        raise TypeError("a verifier-issued RC5 feature mask is required")
    replayed = verify_stage2_rc5_feature_mask_payload(value.payload)
    if (
        value.identity_sha256 != replayed.identity_sha256
        or value.variant != replayed.variant
        or value.active_indices != replayed.active_indices
        or value.inactive_indices != replayed.inactive_indices
        or not np.array_equal(value.boolean_mask, replayed.boolean_mask)
    ):
        raise TypeError("RC5 feature-mask retained-token state differs from replay")
    return value


def feature_mask_payload(
    value: VerifiedStage2RC5FeatureMask,
) -> dict[str, Any]:
    verified = assert_verified_stage2_rc5_feature_mask(value)
    return {
        key: list(item) if isinstance(item, tuple) else item
        for key, item in verified.payload.items()
    }


def apply_stage2_rc5_feature_mask_numpy(
    standardized_float32: np.ndarray,
    feature_mask: VerifiedStage2RC5FeatureMask,
) -> np.ndarray:
    """Apply the frozen mask to an already-standardized float32 array."""

    verified = assert_verified_stage2_rc5_feature_mask(feature_mask)
    if (
        not isinstance(standardized_float32, np.ndarray)
        or standardized_float32.dtype != np.float32
        or standardized_float32.ndim < 1
        or standardized_float32.shape[-1] != FEATURE_DIM
        or not np.isfinite(standardized_float32).all()
    ):
        raise Stage2RC5FeatureMaskError(
            "feature-mask input must be finite float32[...,93] after standardization"
        )
    result = np.array(standardized_float32, dtype=np.float32, order="C", copy=True)
    if verified.inactive_indices:
        result[..., list(verified.inactive_indices)] = np.float32(0.0)
        inactive = result[..., list(verified.inactive_indices)]
        if np.any(np.signbit(inactive)) or np.any(inactive != np.float32(0.0)):
            raise RuntimeError("inactive NumPy features are not exact positive zero")
    return result


def apply_stage2_rc5_feature_mask_torch(
    standardized_float32: torch.Tensor,
    feature_mask: VerifiedStage2RC5FeatureMask,
) -> torch.Tensor:
    """Apply the same mask on-device without changing dtype or gradients."""

    verified = assert_verified_stage2_rc5_feature_mask(feature_mask)
    if (
        not isinstance(standardized_float32, torch.Tensor)
        or standardized_float32.dtype != torch.float32
        or standardized_float32.ndim < 1
        or standardized_float32.shape[-1] != FEATURE_DIM
        or not bool(torch.isfinite(standardized_float32).all().item())
    ):
        raise Stage2RC5FeatureMaskError(
            "feature-mask tensor must be finite float32[...,93] after standardization"
        )
    mask = torch.as_tensor(
        np.array(verified.boolean_mask, dtype=np.bool_, copy=True),
        dtype=torch.bool,
        device=standardized_float32.device,
    )
    result = torch.where(
        mask,
        standardized_float32,
        torch.zeros((), dtype=torch.float32, device=standardized_float32.device),
    ).contiguous()
    if verified.inactive_indices:
        inactive = result[..., list(verified.inactive_indices)]
        if not bool((inactive == 0).all().item()) or bool(torch.signbit(inactive).any().item()):
            raise RuntimeError("inactive Torch features are not exact positive zero")
    return result


__all__ = [
    "FEATURE_DIM",
    "FEATURE_MASK_APPLICATION",
    "FEATURE_MASK_ID_ALGORITHM",
    "FEATURE_MASK_SCHEMA",
    "FEATURE_VARIANT_ACTIVE_INDICES",
    "Stage2RC5FeatureMaskError",
    "VerifiedStage2RC5FeatureMask",
    "apply_stage2_rc5_feature_mask_numpy",
    "apply_stage2_rc5_feature_mask_torch",
    "assert_verified_stage2_rc5_feature_mask",
    "build_stage2_rc5_feature_mask",
    "feature_mask_payload",
    "verify_stage2_rc5_feature_mask_payload",
]
