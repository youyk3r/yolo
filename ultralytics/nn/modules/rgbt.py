# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""RGB and infrared dual-branch modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import C2f, SPPF
from .conv import Conv

__all__ = ("DualInputBackbone",)


class AFF(nn.Module):
    """Attentional feature fusion for two same-scale modality features."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4):
        """Initialize AFF with local/global attention and a concat projection."""
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.act = nn.Sigmoid()
        self.project = Conv(channels * 2, out_channels, 1, 1)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Weight each modality adaptively, then keep both streams before projection."""
        mixed = rgb + ir
        weights = self.act(self.local_att(mixed) + self.global_att(mixed))
        return self.project(torch.cat((rgb * weights, ir * (1.0 - weights)), dim=1))


class ResidualAFF(nn.Module):
    """Residual AFF-v2 fusion that preserves the baseline concat path."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4, alpha: float = 0.1):
        """Initialize a baseline fusion path plus a learnable AFF residual."""
        super().__init__()
        self.base = Conv(channels * 2, out_channels, 1, 1)
        self.aff = AFF(channels, out_channels, reduction)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse with Cat + 1x1 Conv as the main path and AFF-v2 as residual enhancement."""
        return self.base(torch.cat((rgb, ir), dim=1)) + self.alpha * self.aff(rgb, ir)


class DualInputBackbone(nn.Module):
    """Extract multi-scale features from separate RGB and infrared branches.

    The input is a six-channel tensor with RGB in channels 0:3 and infrared data in channels 3:6. Each modality has an
    independent stem through the P3 stage, after which the branches are fused and processed by shared layers.
    """

    def __init__(
        self,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,
        use_aff: bool = False,
        aff_reduction: int = 4,
        residual_aff: bool = False,
        residual_alpha: float = 0.1,
    ):
        """Initialize the dual-branch backbone."""
        super().__init__()
        # P3 输出由 RGB/IR 两个分支拼接得到，所以每个分支先各自产生一半通道。
        branch_channels = p3_channels // 2
        if branch_channels < 8 or p3_channels % 2:
            raise ValueError("p3_channels must be an even integer of at least 16")

        # RGB 分支：只处理输入张量的前 3 个通道，逐步下采样到 P3/8 尺度。
        self.rgb_stem = self._make_stem(branch_channels)
        # IR 分支：结构和 RGB 分支一致，但参数独立，用来学习红外模态特征。
        self.ir_stem = self._make_stem(branch_channels)
        # 在 P3 尺度融合 RGB/IR：baseline 保留 Cat + 1x1 Conv，AFF 版本增加自适应模态权重。
        if residual_aff:
            self.fuse = ResidualAFF(branch_channels, p3_channels, aff_reduction, residual_alpha)
        elif use_aff:
            self.fuse = AFF(branch_channels, p3_channels, aff_reduction)
        else:
            self.fuse = Conv(p3_channels, p3_channels, 1, 1)
        # 融合后的特征进入共享 backbone，继续生成 P4/16 特征。
        self.p4 = nn.Sequential(
            Conv(p3_channels, p4_channels, 3, 2),
            C2f(p4_channels, p4_channels, n=2, shortcut=True),
        )
        # 继续下采样生成 P5/32 特征，并接 SPPF 增强大感受野。
        self.p5 = nn.Sequential(
            Conv(p4_channels, p5_channels, 3, 2),
            C2f(p5_channels, p5_channels, n=1, shortcut=True),
            SPPF(p5_channels, p5_channels, 5),
        )

    @staticmethod
    def _make_stem(channels: int) -> nn.Sequential:
        """Build one modality branch from 3-channel input to P3/8 features."""
        return nn.Sequential(
            Conv(3, channels // 4, 3, 2),
            Conv(channels // 4, channels // 2, 3, 2),
            C2f(channels // 2, channels // 2, n=1, shortcut=True),
            Conv(channels // 2, channels, 3, 2),
            C2f(channels, channels, n=1, shortcut=True),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return fused P3, P4, and P5 feature maps."""
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"DualInputBackbone expects [B, 6, H, W], got {tuple(x.shape)}")
        # 输入约定为 6 通道：前 3 通道是可见光 RGB，后 3 通道是红外 IR。
        rgb = self.rgb_stem(x[:, :3])
        ir = self.ir_stem(x[:, 3:6])
        # AFF/ResidualAFF 直接输入两个模态；普通 dual baseline 则保持 Cat + 1x1 Conv。
        p3 = self.fuse(rgb, ir) if isinstance(self.fuse, (AFF, ResidualAFF)) else self.fuse(torch.cat((rgb, ir), dim=1))
        # 基于融合特征继续提取 P4/P5，供 YOLO neck/head 做多尺度检测。
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]
