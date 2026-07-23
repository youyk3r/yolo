# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""RGB and infrared dual-branch modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import C2f, SPPF
from .conv import CBAM, Conv

__all__ = ("DualInputBackbone",)


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
        use_cbam: bool = False,
        cbam_kernel_size: int = 7,
    ):
        """Initialize the dual-branch backbone."""
        super().__init__()
        # P3 输出由 RGB/IR 两个分支拼接得到，所以每个分支先各自产生一半通道。
        branch_channels = p3_channels // 2
        if branch_channels < 8 or p3_channels % 2:
            raise ValueError("p3_channels must be an even integer of at least 16")

        # RGB 分支：只处理输入张量的前 3 个通道，逐步下采样到 P3/8 尺度。
        self.rgb_stem = self._make_stem(branch_channels, use_cbam, cbam_kernel_size)
        # IR 分支：结构和 RGB 分支一致，但参数独立，用来学习红外模态特征。
        self.ir_stem = self._make_stem(branch_channels, use_cbam, cbam_kernel_size)
        # 在 P3 尺度把 RGB/IR 特征拼接后，用 1x1 卷积做一次通道混合。
        self.fuse = Conv(p3_channels, p3_channels, 1, 1)
        # 融合后的特征进入共享 backbone，继续生成 P4/16 特征。
        self.p4 = nn.Sequential(
            Conv(p3_channels, p4_channels, 3, 2),
            C2f(p4_channels, p4_channels, n=2, shortcut=True),
            self._make_attention(p4_channels, use_cbam, cbam_kernel_size),
        )
        # 继续下采样生成 P5/32 特征，并接 SPPF 增强大感受野。
        self.p5 = nn.Sequential(
            Conv(p4_channels, p5_channels, 3, 2),
            C2f(p5_channels, p5_channels, n=1, shortcut=True),
            self._make_attention(p5_channels, use_cbam, cbam_kernel_size),
            SPPF(p5_channels, p5_channels, 5),
        )

    @staticmethod
    def _make_attention(channels: int, use_cbam: bool, kernel_size: int) -> nn.Module:
        """Return CBAM when enabled, otherwise an identity layer."""
        return CBAM(channels, kernel_size) if use_cbam else nn.Identity()

    @classmethod
    def _make_stem(cls, channels: int, use_cbam: bool, cbam_kernel_size: int) -> nn.Sequential:
        """Build one modality branch from 3-channel input to P3/8 features."""
        return nn.Sequential(
            Conv(3, channels // 4, 3, 2),
            Conv(channels // 4, channels // 2, 3, 2),
            C2f(channels // 2, channels // 2, n=1, shortcut=True),
            cls._make_attention(channels // 2, use_cbam, cbam_kernel_size),
            Conv(channels // 2, channels, 3, 2),
            C2f(channels, channels, n=1, shortcut=True),
            cls._make_attention(channels, use_cbam, cbam_kernel_size),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return fused P3, P4, and P5 feature maps."""
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"DualInputBackbone expects [B, 6, H, W], got {tuple(x.shape)}")
        # 输入约定为 6 通道：前 3 通道是可见光 RGB，后 3 通道是红外 IR。
        rgb = self.rgb_stem(x[:, :3])
        ir = self.ir_stem(x[:, 3:6])
        # 在通道维度拼接两个模态的 P3 特征，得到融合后的 P3。
        p3 = self.fuse(torch.cat((rgb, ir), dim=1))
        # 基于融合特征继续提取 P4/P5，供 YOLO neck/head 做多尺度检测。
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]
