from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from rc_irstd.models.mshnet import MSHNet
from rc_irstd.models.tiny_detector import TinyUNet
from rc_irstd.utils.io import normalise_state_dict


@dataclass
class DetectorOutput:
    logits: torch.Tensor
    auxiliary_logits: list[torch.Tensor]


class DetectorAdapter(nn.Module):
    """Normalise heterogeneous detector APIs to final and auxiliary logits.

    Forward dispatch is determined once from the signature. This avoids masking
    genuine model-internal ``TypeError`` exceptions, which the previous generic
    try/except implementation could silently reinterpret as an API mismatch.
    """

    def __init__(self, model: nn.Module, name: str) -> None:
        super().__init__()
        self.model = model
        self.name = name
        parameters = inspect.signature(model.forward).parameters
        if "warm_flag" in parameters:
            self.forward_mode = "warm_flag"
        elif "training_tag" in parameters:
            self.forward_mode = "training_tag"
        else:
            self.forward_mode = "plain"

    def forward(self, x: torch.Tensor, training_tag: bool = True) -> DetectorOutput:
        if self.forward_mode == "warm_flag":
            raw = self.model(x, warm_flag=training_tag)
        elif self.forward_mode == "training_tag":
            raw = self.model(x, training_tag=training_tag)
        else:
            raw = self.model(x)

        if isinstance(raw, torch.Tensor):
            return DetectorOutput(raw, [])
        if isinstance(raw, dict):
            final = raw.get("logits", raw.get("pred", raw.get("out")))
            if final is None:
                raise KeyError("Detector dictionary must contain logits, pred or out")
            auxiliary = raw.get("auxiliary_logits", raw.get("aux", []))
            return DetectorOutput(final, list(auxiliary))
        if isinstance(raw, (tuple, list)):
            if len(raw) >= 2 and isinstance(raw[0], (tuple, list)):
                auxiliary, final = raw[0], raw[1]
                return DetectorOutput(final, list(auxiliary))
            tensors = [item for item in raw if isinstance(item, torch.Tensor)]
            if not tensors:
                raise TypeError("Detector returned a tuple without tensors")
            return DetectorOutput(tensors[-1], tensors[:-1])
        raise TypeError(f"Unsupported detector output type: {type(raw).__name__}")


def _external_mshnet(in_channels: int) -> nn.Module:
    try:
        module = importlib.import_module("model.MSHNet")
        constructor = getattr(module, "MSHNet")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "mshnet_external was requested, but model.MSHNet.MSHNet is unavailable. "
            "Use --detector mshnet for the bundled implementation or add the "
            "external MSHNet root to PYTHONPATH."
        ) from exc
    return constructor(in_channels)


def build_detector(
    name: str,
    in_channels: int = 3,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> DetectorAdapter:
    normalised = name.lower().replace("-", "_")
    if normalised in {"tiny", "tiny_unet"}:
        model: nn.Module = TinyUNet(in_channels=in_channels)
    elif normalised in {"mshnet", "mshnet_internal"}:
        model = MSHNet(input_channels=in_channels)
        normalised = "mshnet"
    elif normalised == "mshnet_external":
        model = _external_mshnet(in_channels)
    else:
        raise ValueError(f"Unknown detector '{name}'")

    adapter = DetectorAdapter(model, normalised)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = normalise_state_dict(payload)
        attempts = [
            state,
            {key.removeprefix("model."): value for key, value in state.items()},
            {key.removeprefix("module."): value for key, value in state.items()},
        ]
        last_error: RuntimeError | None = None
        for candidate in attempts:
            try:
                adapter.model.load_state_dict(candidate, strict=strict)
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    return adapter.to(device)


def resize_logits(logits: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if tuple(logits.shape[-2:]) == tuple(target_hw):
        return logits
    return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
