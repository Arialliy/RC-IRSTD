from __future__ import annotations

import math

import torch


def upper_cvar(values: torch.Tensor, quantile: float = 0.95) -> torch.Tensor:
    """Mean of the upper ``1-quantile`` fraction, preserving gradients."""
    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must be in [0, 1)")
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return values.sum() * 0.0
    count = max(1, int(math.ceil((1.0 - quantile) * flat.numel())))
    return torch.topk(flat, k=count, largest=True, sorted=False).values.mean()


def smooth_upper_max(group_risks: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """Log-sum-exp upper approximation of the maximum group risk."""
    if group_risks.numel() == 0:
        raise ValueError("group_risks must not be empty")
    if gamma <= 0:
        return group_risks.mean()
    return torch.logsumexp(gamma * group_risks, dim=0) / gamma


def normalized_log_mean_exp(
    group_risks: torch.Tensor,
    gamma: float = 10.0,
) -> torch.Tensor:
    """Normalised log-mean-exp; useful when an upper bound is not required."""
    if group_risks.numel() == 0:
        raise ValueError("group_risks must not be empty")
    if gamma <= 0:
        return group_risks.mean()
    count = torch.as_tensor(
        float(group_risks.numel()),
        dtype=group_risks.dtype,
        device=group_risks.device,
    )
    return (torch.logsumexp(gamma * group_risks, dim=0) - torch.log(count)) / gamma


def smooth_worst_group(group_risks: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """Backward-compatible name for the smooth upper maximum."""
    return smooth_upper_max(group_risks, gamma=gamma)
