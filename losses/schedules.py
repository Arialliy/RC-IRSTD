"""Deterministic schedules for staged risk-aware detector training."""

from __future__ import annotations


def linear_risk_weight(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    """Return a zero-warmup then linear-ramp multiplier in ``[0, 1]``.

    ``warmup_epochs`` counts complete zero-risk epochs.  The first epoch after
    warm-up receives ``1 / ramp_epochs``; a zero-length ramp switches directly
    to one.  Keeping this function pure makes the exact schedule auditable in
    saved configs and unit tests.
    """

    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    ramp_epochs = int(ramp_epochs)
    if epoch < 0:
        raise ValueError("epoch cannot be negative")
    if warmup_epochs < 0 or ramp_epochs < 0:
        raise ValueError("warmup_epochs and ramp_epochs cannot be negative")
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return 1.0
    completed = epoch - warmup_epochs + 1
    return min(max(float(completed) / float(ramp_epochs), 0.0), 1.0)
