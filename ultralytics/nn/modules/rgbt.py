# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""RGB and infrared dual-branch modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import C2f, SPPF
from .conv import Conv

__all__ = (
    "DualInputBackbone",
    "MultiScaleDualInputBackbone",
    "ResCGAFFP2Backbone",
    "ResCGAFFP2RDLEBackbone",
)


def _make_modality_stem(channels: int) -> nn.Sequential:
    """Build one modality branch from a three-channel input to P3/8 features."""
    return nn.Sequential(
        Conv(3, channels // 4, 3, 2),
        Conv(channels // 4, channels // 2, 3, 2),
        C2f(channels // 2, channels // 2, n=1, shortcut=True),
        Conv(channels // 2, channels, 3, 2),
        C2f(channels, channels, n=1, shortcut=True),
    )


def _make_modality_stage(in_channels: int, out_channels: int, repeats: int) -> nn.Sequential:
    """Downsample and refine one modality at the next feature scale."""
    return nn.Sequential(
        Conv(in_channels, out_channels, 3, 2),
        C2f(out_channels, out_channels, n=repeats, shortcut=True),
    )


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


class ConcatGatedAFF(nn.Module):
    """Concat-gated AFF that predicts modality weights from RGB/IR contrast."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4):
        """Initialize concat-gated AFF with local/global attention and a concat projection."""
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        gate_channels = channels * 2
        self.local_att = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.act = nn.Sigmoid()
        self.project = Conv(gate_channels, out_channels, 1, 1)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Predict modality weights from concatenated RGB/IR features, then project both streams."""
        mixed = torch.cat((rgb, ir), dim=1)
        weights = self.act(self.local_att(mixed) + self.global_att(mixed))
        return self.project(torch.cat((rgb * weights, ir * (1.0 - weights)), dim=1))


class ResidualConcatGatedAFF(nn.Module):
    """Residual concat-gated AFF fusion that preserves the baseline concat path."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4, alpha: float = 0.1):
        """Initialize a baseline fusion path plus a learnable concat-gated AFF residual."""
        super().__init__()
        self.base = Conv(channels * 2, out_channels, 1, 1)
        self.aff = ConcatGatedAFF(channels, out_channels, reduction)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse with Cat + 1x1 Conv as the main path and concat-gated AFF as residual enhancement."""
        return self.base(torch.cat((rgb, ir), dim=1)) + self.alpha * self.aff(rgb, ir)


class DifferenceAwareAFF(nn.Module):
    """Difference-aware concat-gated AFF for RGB/IR modality fusion."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4, mode: str = "full"):
        """Initialize difference-aware AFF with selectable difference and consistency cues."""
        super().__init__()
        if mode not in {"full", "difference", "consistency"}:
            raise ValueError(f"mode must be 'full', 'difference', or 'consistency', got {mode!r}")

        self.use_difference = mode in {"full", "difference"}
        self.use_consistency = mode in {"full", "consistency"}
        hidden_channels = max(channels // reduction, 8)
        gate_channels = channels * (2 + int(self.use_difference) + int(self.use_consistency))
        self.local_att = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.act = nn.Sigmoid()
        self.project = Conv(channels * 2, out_channels, 1, 1)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse RGB/IR features using modality, difference, and consistency cues."""
        cues = [rgb, ir]
        if self.use_difference:
            cues.append(torch.abs(rgb - ir))
        if self.use_consistency:
            cues.append(rgb * ir)
        mixed = torch.cat(cues, dim=1)
        weights = self.act(self.local_att(mixed) + self.global_att(mixed))
        return self.project(torch.cat((rgb * weights, ir * (1.0 - weights)), dim=1))


class ResidualDifferenceAwareAFF(nn.Module):
    """Residual difference-aware fusion that preserves the baseline concat path."""

    def __init__(
        self, channels: int, out_channels: int, reduction: int = 4, alpha: float = 0.1, mode: str = "full"
    ):
        """Initialize a baseline fusion path plus a learnable difference-aware residual."""
        super().__init__()
        self.base = Conv(channels * 2, out_channels, 1, 1)
        self.aff = DifferenceAwareAFF(channels, out_channels, reduction, mode)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse with Cat + 1x1 Conv as the main path and difference-aware AFF as residual enhancement."""
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
        residual_cgaff: bool = False,
        residual_daff: bool = False,
        da_mode: str = "full",
    ):
        """Initialize the dual-branch backbone."""
        super().__init__()
        # P3 output is concatenated from RGB/IR branches, so each branch first produces half the channels.
        branch_channels = p3_channels // 2
        if branch_channels < 8 or p3_channels % 2:
            raise ValueError("p3_channels must be an even integer of at least 16")

        # RGB/IR stems have the same structure but independent parameters for modality-specific features.
        self.rgb_stem = _make_modality_stem(branch_channels)
        self.ir_stem = _make_modality_stem(branch_channels)

        # Fuse RGB/IR at P3. Residual variants keep Cat + 1x1 Conv as the main path.
        if residual_daff:
            self.fuse = ResidualDifferenceAwareAFF(
                branch_channels, p3_channels, aff_reduction, residual_alpha, da_mode
            )
        elif residual_cgaff:
            self.fuse = ResidualConcatGatedAFF(branch_channels, p3_channels, aff_reduction, residual_alpha)
        elif residual_aff:
            self.fuse = ResidualAFF(branch_channels, p3_channels, aff_reduction, residual_alpha)
        elif use_aff:
            self.fuse = AFF(branch_channels, p3_channels, aff_reduction)
        else:
            self.fuse = Conv(p3_channels, p3_channels, 1, 1)

        # Shared backbone after fusion produces P4/16 and P5/32 features.
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

        # The six-channel input stores RGB in the first three channels and infrared in the last three channels.
        rgb = self.rgb_stem(x[:, :3])
        ir = self.ir_stem(x[:, 3:6])

        # AFF variants receive the two modalities directly; the plain baseline uses Cat + 1x1 Conv.
        p3 = (
            self.fuse(rgb, ir)
            if isinstance(
                self.fuse,
                (
                    AFF,
                    ResidualAFF,
                    ConcatGatedAFF,
                    ResidualConcatGatedAFF,
                    DifferenceAwareAFF,
                    ResidualDifferenceAwareAFF,
                ),
            )
            else self.fuse(torch.cat((rgb, ir), dim=1))
        )

        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]


class ResCGAFFP2Backbone(nn.Module):
    """Preserve P2 features before ResCGAFF fusion at P3 for four-scale detection."""

    def __init__(
        self,
        p2_channels: int = 32,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,
        aff_reduction: int = 4,
        residual_alpha: float = 0.1,
    ):
        """Initialize separate RGB/IR P2-P3 stages followed by the shared P4-P5 backbone."""
        super().__init__()
        output_channels = (p2_channels, p3_channels, p4_channels, p5_channels)
        if any(channels < 16 or channels % 2 for channels in output_channels):
            raise ValueError("P2, P3, P4, and P5 channels must be even integers of at least 16")

        branch_p2 = p2_channels // 2
        branch_p3 = p3_channels // 2
        stem_channels = branch_p2 // 2
        if stem_channels < 8:
            raise ValueError("p2_channels must be at least 32")

        # P2 retains shallow spatial detail from both modalities without competitive attention filtering.
        self.rgb_p2 = nn.Sequential(
            Conv(3, stem_channels, 3, 2),
            Conv(stem_channels, branch_p2, 3, 2),
            C2f(branch_p2, branch_p2, n=1, shortcut=True),
        )
        self.ir_p2 = nn.Sequential(
            Conv(3, stem_channels, 3, 2),
            Conv(stem_channels, branch_p2, 3, 2),
            C2f(branch_p2, branch_p2, n=1, shortcut=True),
        )
        self.fuse_p2 = Conv(p2_channels, p2_channels, 1, 1)

        # P3 keeps the original modality-specific stage and Residual Concat-Gated AFF fusion.
        self.rgb_p3 = _make_modality_stage(branch_p2, branch_p3, repeats=1)
        self.ir_p3 = _make_modality_stage(branch_p2, branch_p3, repeats=1)
        self.fuse_p3 = ResidualConcatGatedAFF(branch_p3, p3_channels, aff_reduction, residual_alpha)

        self.p4 = _make_modality_stage(p3_channels, p4_channels, repeats=2)
        self.p5 = nn.Sequential(
            Conv(p4_channels, p5_channels, 3, 2),
            C2f(p5_channels, p5_channels, n=1, shortcut=True),
            SPPF(p5_channels, p5_channels, 5),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return fused P2, P3, P4, and P5 feature maps."""
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"ResCGAFFP2Backbone expects [B, 6, H, W], got {tuple(x.shape)}")

        rgb_p2 = self.rgb_p2(x[:, :3])
        ir_p2 = self.ir_p2(x[:, 3:6])
        p2 = self._fuse_p2(rgb_p2, ir_p2)

        rgb_p3 = self.rgb_p3(rgb_p2)
        ir_p3 = self.ir_p3(ir_p2)
        p3 = self.fuse_p3(rgb_p3, ir_p3)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p2, p3, p4, p5]

    def _fuse_p2(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse shallow RGB/IR features with the baseline concat projection."""
        return self.fuse_p2(torch.cat((rgb, ir), dim=1))


class ResCGAFFP2RDLEBackbone(ResCGAFFP2Backbone):
    """Add residual difference localization enhancement to the P2 fusion output."""

    def __init__(
        self,
        p2_channels: int = 32,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,
        aff_reduction: int = 4,
        residual_alpha: float = 0.1,
        difference_beta: float = 0.0,
    ):
        """Initialize the P2 baseline and a lightweight normalized difference residual."""
        super().__init__(p2_channels, p3_channels, p4_channels, p5_channels, aff_reduction, residual_alpha)
        branch_channels = p2_channels // 2
        self.rgb_norm = nn.GroupNorm(1, branch_channels, affine=False)
        self.ir_norm = nn.GroupNorm(1, branch_channels, affine=False)
        self.difference = nn.Sequential(
            Conv(branch_channels, branch_channels, 3, 1, g=branch_channels),
            Conv(branch_channels, p2_channels, 1, 1),
        )
        self.beta = nn.Parameter(torch.tensor(difference_beta, dtype=torch.float32))

    def _fuse_p2(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Preserve baseline fusion and add normalized RGB/IR differences as a localization residual."""
        base = super()._fuse_p2(rgb, ir)
        difference = torch.abs(self.rgb_norm(rgb) - self.ir_norm(ir))
        return base + self.beta * self.difference(difference)


class MultiScaleDualInputBackbone(nn.Module):
    """Extract and fuse independent RGB/infrared features at the P3, P4, and P5 scales."""

    def __init__(
        self,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,  
        aff_reduction: int = 4,
        residual_alpha: float = 0.1,
        fusion_pattern: str = "RRR",
    ):
        """Initialize modality-specific stages and the requested P3/P4/P5 residual fusion pattern."""
        super().__init__()
        output_channels = (p3_channels, p4_channels, p5_channels)
        if any(channels < 16 or channels % 8 for channels in output_channels):
            raise ValueError("P3, P4, and P5 channels must be multiples of 8 and at least 16")
        fusion_pattern = fusion_pattern.upper()
        if len(fusion_pattern) != 3 or any(mode not in "RDC" for mode in fusion_pattern):
            raise ValueError(f"fusion_pattern must contain three R/D/C modes for P3/P4/P5, got {fusion_pattern!r}")

        branch_p3 = p3_channels // 2
        branch_p4 = p4_channels * 5 // 8
        branch_p5 = p5_channels * 5 // 8

        # P3 remains half-width; 5/8-width deep branches match the parameter budget of the P3-only baseline.
        self.rgb_stem = _make_modality_stem(branch_p3)
        self.ir_stem = _make_modality_stem(branch_p3)
        self.rgb_p4 = _make_modality_stage(branch_p3, branch_p4, repeats=2)
        self.ir_p4 = _make_modality_stage(branch_p3, branch_p4, repeats=2)
        self.rgb_p5 = _make_modality_stage(branch_p4, branch_p5, repeats=1)
        self.ir_p5 = _make_modality_stage(branch_p4, branch_p5, repeats=1)

        # Each scale selects R=ResCGAFF, D=difference-aware, or C=consistency-aware residual fusion.
        self.fusion_pattern = fusion_pattern
        self.fuse_p3 = self._make_fusion(
            fusion_pattern[0], branch_p3, p3_channels, aff_reduction, residual_alpha
        )
        self.fuse_p4 = self._make_fusion(
            fusion_pattern[1], branch_p4, p4_channels, aff_reduction, residual_alpha
        )
        self.fuse_p5 = self._make_fusion(
            fusion_pattern[2], branch_p5, p5_channels, aff_reduction, residual_alpha
        )
        self.sppf = SPPF(p5_channels, p5_channels, 5)

    @staticmethod
    def _make_fusion(
        mode: str, channels: int, out_channels: int, reduction: int, alpha: float
    ) -> nn.Module:
        """Build one scale's residual fusion module from its compact mode code."""
        if mode == "R":
            return ResidualConcatGatedAFF(channels, out_channels, reduction, alpha)
        da_mode = "difference" if mode == "D" else "consistency"
        return ResidualDifferenceAwareAFF(channels, out_channels, reduction, alpha, da_mode)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return independently fused P3, P4, and P5 feature maps."""
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"MultiScaleDualInputBackbone expects [B, 6, H, W], got {tuple(x.shape)}")

        rgb_p3 = self.rgb_stem(x[:, :3])
        ir_p3 = self.ir_stem(x[:, 3:6])
        p3 = self.fuse_p3(rgb_p3, ir_p3)

        rgb_p4 = self.rgb_p4(rgb_p3)
        ir_p4 = self.ir_p4(ir_p3)
        p4 = self.fuse_p4(rgb_p4, ir_p4)

        rgb_p5 = self.rgb_p5(rgb_p4)
        ir_p5 = self.ir_p5(ir_p4)
        p5 = self.sppf(self.fuse_p5(rgb_p5, ir_p5))
        return [p3, p4, p5]
