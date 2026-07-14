from __future__ import annotations

"""Self-contained MSHNet implementation used by RC-IRSTD.

The module mirrors the public CVPR 2024 MSHNet parameter names so that official
and common fork checkpoints can be loaded without requiring a second repository.
RC-IRSTD wraps the network through :mod:`rc_irstd.models.detector_adapter` and
keeps the original warm-up API: before ``warm_flag`` is enabled, only the
full-resolution head is used; afterwards four decoder heads are fused.
"""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class DeterministicGlobalMaxPool2d(nn.Module):
    """Global max pool with AdaptiveMaxPool2d(1)-equivalent tie semantics."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).max(dim=-1, keepdim=True).values.unsqueeze(-1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(int(in_planes) // int(ratio), 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # The parameter-free reduction is checkpoint-compatible and avoids the
        # nondeterministic CUDA backward of AdaptiveMaxPool2d.
        self.max_pool = DeterministicGlobalMaxPool2d()
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("SpatialAttention kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class ResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut: nn.Module | None = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return self.relu(out + residual)


@dataclass(frozen=True)
class MSHNetFeatures:
    decoder_0: torch.Tensor
    decoder_1: torch.Tensor
    decoder_2: torch.Tensor
    decoder_3: torch.Tensor
    middle: torch.Tensor


class MSHNet(nn.Module):
    """Multi-Scale Head Network with official checkpoint-compatible names."""

    def __init__(
        self,
        input_channels: int = 3,
        block: type[nn.Module] = ResNet,
        channels: tuple[int, int, int, int, int] = (16, 32, 64, 128, 256),
        blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(channels) != 5 or len(blocks) != 4:
            raise ValueError("channels must contain 5 entries and blocks 4 entries")
        c0, c1, c2, c3, c4 = [int(value) for value in channels]
        b0, b1, b2, b3 = [int(value) for value in blocks]
        self.input_channels = int(input_channels)
        self.channels = (c0, c1, c2, c3, c4)
        self.blocks = (b0, b1, b2, b3)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.conv_init = nn.Conv2d(self.input_channels, c0, 1, 1)
        self.encoder_0 = self._make_layer(c0, c0, block)
        self.encoder_1 = self._make_layer(c0, c1, block, b0)
        self.encoder_2 = self._make_layer(c1, c2, block, b1)
        self.encoder_3 = self._make_layer(c2, c3, block, b2)
        self.middle_layer = self._make_layer(c3, c4, block, b3)
        self.decoder_3 = self._make_layer(c3 + c4, c3, block, b2)
        self.decoder_2 = self._make_layer(c2 + c3, c2, block, b1)
        self.decoder_1 = self._make_layer(c1 + c2, c1, block, b0)
        self.decoder_0 = self._make_layer(c0 + c1, c0, block)
        self.output_0 = nn.Conv2d(c0, 1, 1)
        self.output_1 = nn.Conv2d(c1, 1, 1)
        self.output_2 = nn.Conv2d(c2, 1, 1)
        self.output_3 = nn.Conv2d(c3, 1, 1)
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        block: type[nn.Module],
        block_num: int = 1,
    ) -> nn.Sequential:
        if block_num < 1:
            raise ValueError("block_num must be positive")
        layers: list[nn.Module] = [block(in_channels, out_channels)]
        layers.extend(block(out_channels, out_channels) for _ in range(block_num - 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=True)

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool = True,
        return_feature: bool = False,
    ):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self._resize_like(x_m, x_e3)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self._resize_like(x_d3, x_e2)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self._resize_like(x_d2, x_e1)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self._resize_like(x_d1, x_e0)], dim=1))

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(
                torch.cat(
                    [
                        mask0,
                        self._resize_like(mask1, mask0),
                        self._resize_like(mask2, mask0),
                        self._resize_like(mask3, mask0),
                    ],
                    dim=1,
                )
            )
            auxiliary = [mask0, mask1, mask2, mask3]
        else:
            auxiliary = []
            output = self.output_0(x_d0)

        if return_feature:
            features = MSHNetFeatures(x_d0, x_d1, x_d2, x_d3, x_m)
            return auxiliary, output, features
        return auxiliary, output

    def export_config(self) -> dict[str, object]:
        return {
            "input_channels": self.input_channels,
            "channels": list(self.channels),
            "blocks": list(self.blocks),
        }
