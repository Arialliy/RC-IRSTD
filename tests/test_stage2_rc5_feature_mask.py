from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rc.stage2_rc5_feature_mask import (
    FEATURE_DIM,
    FEATURE_VARIANT_ACTIVE_INDICES,
    Stage2RC5FeatureMaskError,
    VerifiedStage2RC5FeatureMask,
    apply_stage2_rc5_feature_mask_numpy,
    apply_stage2_rc5_feature_mask_torch,
    assert_verified_stage2_rc5_feature_mask,
    build_stage2_rc5_feature_mask,
    feature_mask_payload,
    verify_stage2_rc5_feature_mask_payload,
)


@pytest.mark.parametrize(
    ("variant", "active_count"),
    (("C3", 93), ("C4", 39), ("C5", 79), ("C6", 87)),
)
def test_preregistered_masks_have_exact_indices_and_distinct_identity(
    variant: str,
    active_count: int,
) -> None:
    mask = build_stage2_rc5_feature_mask(variant)
    replayed = verify_stage2_rc5_feature_mask_payload(feature_mask_payload(mask))

    assert mask.active_indices == FEATURE_VARIANT_ACTIVE_INDICES[variant]
    assert len(mask.active_indices) == active_count
    assert mask.inactive_indices == tuple(range(active_count, FEATURE_DIM))
    assert mask.boolean_mask.dtype == np.bool_
    assert not mask.boolean_mask.flags.writeable
    assert replayed.identity_sha256 == mask.identity_sha256
    identities = {
        build_stage2_rc5_feature_mask(name).identity_sha256
        for name in FEATURE_VARIANT_ACTIVE_INDICES
    }
    assert len(identities) == 4


def test_numpy_mask_preserves_active_values_and_writes_positive_zero() -> None:
    values = np.linspace(-4.0, 5.0, 2 * 3 * FEATURE_DIM, dtype=np.float32).reshape(
        2, 3, FEATURE_DIM
    )
    original = values.copy()
    mask = build_stage2_rc5_feature_mask("C4")

    result = apply_stage2_rc5_feature_mask_numpy(values, mask)

    np.testing.assert_array_equal(result[..., :39], original[..., :39])
    np.testing.assert_array_equal(result[..., 39:], np.float32(0.0))
    assert not np.signbit(result[..., 39:]).any()
    np.testing.assert_array_equal(values, original)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_torch_mask_preserves_active_gradient_and_zeroes_inactive_gradient() -> None:
    values = torch.linspace(-4.0, 5.0, 2 * FEATURE_DIM, dtype=torch.float32).reshape(
        2, FEATURE_DIM
    )
    values.requires_grad_(True)
    mask = build_stage2_rc5_feature_mask("C5")

    result = apply_stage2_rc5_feature_mask_torch(values, mask)
    result.sum().backward()

    torch.testing.assert_close(result[..., :79], values.detach()[..., :79])
    assert bool((result[..., 79:] == 0).all().item())
    assert not bool(torch.signbit(result[..., 79:]).any().item())
    torch.testing.assert_close(values.grad[..., :79], torch.ones_like(values.grad[..., :79]))
    torch.testing.assert_close(values.grad[..., 79:], torch.zeros_like(values.grad[..., 79:]))


@pytest.mark.parametrize(
    "invalid",
    (
        np.zeros((2, FEATURE_DIM), dtype=np.float64),
        np.zeros((2, FEATURE_DIM - 1), dtype=np.float32),
        np.array([[np.nan] * FEATURE_DIM], dtype=np.float32),
    ),
)
def test_numpy_mask_rejects_inputs_outside_post_standardization_contract(
    invalid: np.ndarray,
) -> None:
    with pytest.raises(Stage2RC5FeatureMaskError, match=r"float32\[\.\.\.,93\]"):
        apply_stage2_rc5_feature_mask_numpy(
            invalid,
            build_stage2_rc5_feature_mask("C6"),
        )


@pytest.mark.parametrize(
    "invalid",
    (
        torch.zeros((2, FEATURE_DIM), dtype=torch.float64),
        torch.zeros((2, FEATURE_DIM - 1), dtype=torch.float32),
        torch.full((1, FEATURE_DIM), torch.inf, dtype=torch.float32),
    ),
)
def test_torch_mask_rejects_inputs_outside_post_standardization_contract(
    invalid: torch.Tensor,
) -> None:
    with pytest.raises(Stage2RC5FeatureMaskError, match=r"float32\[\.\.\.,93\]"):
        apply_stage2_rc5_feature_mask_torch(
            invalid,
            build_stage2_rc5_feature_mask("C6"),
        )


def test_payload_tamper_and_retained_token_tamper_fail_closed() -> None:
    mask = build_stage2_rc5_feature_mask("C4")
    payload = feature_mask_payload(mask)
    tampered = copy.deepcopy(payload)
    tampered["active_count"] = 40
    with pytest.raises(Stage2RC5FeatureMaskError, match="frozen semantics"):
        verify_stage2_rc5_feature_mask_payload(tampered)

    tampered = copy.deepcopy(payload)
    tampered["identity_sha256"] = "0" * 64
    with pytest.raises(Stage2RC5FeatureMaskError, match="identity SHA-256"):
        verify_stage2_rc5_feature_mask_payload(tampered)

    object.__setattr__(mask, "variant", "C6")
    with pytest.raises(TypeError, match="retained-token state"):
        assert_verified_stage2_rc5_feature_mask(mask)


def test_capability_cannot_be_constructed_or_replaced_by_a_payload() -> None:
    with pytest.raises(TypeError, match="verifier-issued"):
        VerifiedStage2RC5FeatureMask()
    with pytest.raises(TypeError, match="verifier-issued"):
        apply_stage2_rc5_feature_mask_numpy(
            np.zeros((1, FEATURE_DIM), dtype=np.float32),
            feature_mask_payload(build_stage2_rc5_feature_mask("C3")),  # type: ignore[arg-type]
        )
