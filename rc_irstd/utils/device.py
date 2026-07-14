from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def create_grad_scaler(device: torch.device, enabled: bool):
    """Construct a GradScaler across PyTorch 2.x API variants."""
    use_amp = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler(device.type, enabled=use_amp)
    except (AttributeError, TypeError):  # PyTorch 2.0/2.1 compatibility
        return torch.cuda.amp.GradScaler(enabled=use_amp)
