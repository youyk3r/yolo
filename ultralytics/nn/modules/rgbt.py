# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""RGB and infrared dual-branch modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import C2f, SPPF
from .conv import Conv

__all__ = ("DualInputBackbone",)


class DualInputBackbone(nn.Module):
    """Extract multi-scale features from separate RGB and infrared branches.

    The input is a six-channel tensor with RGB in channels 0:3 and infrared data in channels 3:6. Each modality has an
    independent stem through the P3 stage, after which the branches are fused and processed by shared layers.
    """

    def __init__(self, p3_channels: int = 64, p4_channels: int = 128, p5_channels: int = 256):
        """Initialize the dual-branch backbone."""
        super().__init__()
        branch_channels = p3_channels // 2
        if branch_channels < 8 or p3_channels % 2:
            raise ValueError("p3_channels must be an even integer of at least 16")

        self.rgb_stem = nn.Sequential(
            Conv(3, branch_channels // 4, 3, 2),
            Conv(branch_channels // 4, branch_channels // 2, 3, 2),
            C2f(branch_channels // 2, branch_channels // 2, n=1, shortcut=True),
            Conv(branch_channels // 2, branch_channels, 3, 2),
            C2f(branch_channels, branch_channels, n=1, shortcut=True),
        )
        self.ir_stem = nn.Sequential(
            Conv(3, branch_channels // 4, 3, 2),
            Conv(branch_channels // 4, branch_channels // 2, 3, 2),
            C2f(branch_channels // 2, branch_channels // 2, n=1, shortcut=True),
            Conv(branch_channels // 2, branch_channels, 3, 2),
            C2f(branch_channels, branch_channels, n=1, shortcut=True),
        )
        self.fuse = Conv(p3_channels, p3_channels, 1, 1)
        self.p4 = nn.Sequential(
            Conv(p3_channels, p4_channels, 3, 2),
            C2f(p4_channels, p4_channels, n=2, shortcut=True),
        )
        self.p5 = nn.Sequential(
            Conv(p4_channels, p5_channels, 3, 2),
            C2f(p5_channels, p5_channels, n=1, shortcut=True),
            SPPF(p5_channels, p5_channels, 5),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return fused P3, P4, and P5 feature maps."""
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"DualInputBackbone expects [B, 6, H, W], got {tuple(x.shape)}")
        rgb = self.rgb_stem(x[:, :3])
        ir = self.ir_stem(x[:, 3:6])
        p3 = self.fuse(torch.cat((rgb, ir), dim=1))
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]
