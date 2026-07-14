"""Smooth aggregation of worst-source-domain risks."""

from __future__ import annotations

import math

import torch


def smooth_max(
    values: torch.Tensor,
    gamma: float = 10.0,
    dim: int = 0,
    keepdim: bool = False,
) -> torch.Tensor:
    """Normalized log-sum-exp approximation to a maximum.

    Subtracting ``log(K) / gamma`` makes the result invariant to duplicating
    equal domain risks: ``smooth_max([x, ..., x]) == x``.  This avoids changing
    the loss baseline merely because an experiment uses a different number of
    source domains.
    """

    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    if gamma <= 0.0 or not math.isfinite(gamma):
        raise ValueError(f"gamma must be finite and positive, got {gamma}")
    if values.ndim == 0:
        return values

    dim = dim if dim >= 0 else values.ndim + dim
    if dim < 0 or dim >= values.ndim:
        raise IndexError(f"dim {dim} is out of range for shape {tuple(values.shape)}")
    count = values.shape[dim]
    if count == 0:
        raise ValueError("cannot aggregate an empty domain dimension")

    normalizer = math.log(count)
    result = (
        torch.logsumexp(values * gamma, dim=dim, keepdim=keepdim) - normalizer
    ) / gamma
    return result


def smooth_worst_domain(
    domain_risks: torch.Tensor,
    gamma: float = 10.0,
) -> torch.Tensor:
    """Convenience alias for a one-dimensional vector of domain risks."""

    if domain_risks.ndim != 1:
        raise ValueError(
            f"domain_risks must be one-dimensional, got {tuple(domain_risks.shape)}"
        )
    return smooth_max(domain_risks, gamma=gamma, dim=0)
