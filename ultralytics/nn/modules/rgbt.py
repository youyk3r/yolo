# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""RGB and infrared dual-branch modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import C2f, SPPF
from .conv import CBAM, Conv, CoordinateAttention, DWConv, ECAttention, EMAAttention

__all__ = (
    "BGF",
    "MFB",
    "DualInputBackbone",
    "MultiScaleDualInputBackbone",
    "ResCGAFFP2Backbone",
    "CMAEnhance",
    "ASPPEnhance",
    "NonLocalEnhance",
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

    def __init__(
        self,
        channels: int,
        out_channels: int,
        reduction: int = 4,
        attention_mode: str = "ours_ours",
        weight_mode: str = "complement",
    ):
        """Initialize concat-gated AFF with local/global attention and a concat projection."""
        super().__init__()
        if attention_mode not in {"rfc_rfc", "ours_ours", "rfc_ours", "ours_rfc"}:
            raise ValueError(f"attention_mode must be one of rfc_rfc, ours_ours, rfc_ours, ours_rfc, got {attention_mode!r}")
        if weight_mode not in {"complement", "softmax", "independent", "norm_independent"}:
            raise ValueError(
                f"weight_mode must be one of complement, softmax, independent, norm_independent, got {weight_mode!r}"
            )
        self.attention_mode = attention_mode
        self.weight_mode = weight_mode
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
        self.rfc_channel = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
        )
        self.rfc_spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.modality_local = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, 1, bias=True),
        )
        self.modality_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(gate_channels, hidden_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels * 2, 1, bias=True),
        )
        self.act = nn.Sigmoid()
        self.project = Conv(gate_channels, out_channels, 1, 1)

    def forward(self, Frgb: torch.Tensor, Fir: torch.Tensor) -> torch.Tensor:
        """Fuse feature inputs with ``Fcat=[Frgb,Fir]`` and a learned modality gate.

        The gate follows ``A=sigmoid(A_local(Fcat)+A_global(Fcat))``. The returned feature is
        ``P([Frgb*A,Fir*(1-A)])`` for complementary weights, or the selected two-stream
        weighting variant for other ``weight_mode`` values.
        """
        if Frgb.ndim != 4 or Fir.ndim != 4 or Frgb.shape != Fir.shape:
            raise ValueError(
                "ResCGAFF expects matching [B, C, H, W] Frgb/Fir tensors, "
                f"got {tuple(Frgb.shape)} and {tuple(Fir.shape)}"
            )
        mixed = torch.cat((Frgb, Fir), dim=1)
        if self.weight_mode == "complement":
            if self.attention_mode == "ours_ours":
                weights = self.act(self.local_att(mixed) + self.global_att(mixed))
            else:
                channel_pool = torch.cat((mixed.mean(dim=1, keepdim=True), mixed.amax(dim=1, keepdim=True)), dim=1)
                rfc_spatial = self.rfc_spatial(channel_pool)
                rfc_channel = self.rfc_channel(mixed.mean(dim=(2, 3), keepdim=True)) + self.rfc_channel(
                    mixed.amax(dim=(2, 3), keepdim=True)
                )
                if self.attention_mode == "rfc_rfc":
                    weights = self.act(rfc_channel) * self.act(rfc_spatial)
                elif self.attention_mode == "rfc_ours":
                    weights = self.act(rfc_channel + self.local_att(mixed))
                else:  # ours_rfc
                    weights = self.act(self.global_att(mixed)) * self.act(rfc_spatial)
            rgb_weight, ir_weight = weights, 1.0 - weights
        else:
            logits = self.modality_local(mixed) + self.modality_global(mixed)
            logits = logits.reshape(logits.shape[0], 2, -1, logits.shape[2], logits.shape[3])
            if self.weight_mode == "softmax":
                modality_weights = torch.softmax(logits, dim=1)
                rgb_weight, ir_weight = modality_weights[:, 0], modality_weights[:, 1]
            else:
                rgb_weight, ir_weight = torch.sigmoid(logits[:, 0]), torch.sigmoid(logits[:, 1])
                if self.weight_mode == "norm_independent":
                    total = rgb_weight + ir_weight + 1e-6
                    rgb_weight = 2.0 * rgb_weight / total
                    ir_weight = 2.0 * ir_weight / total
        return self.project(torch.cat((Frgb * rgb_weight, Fir * ir_weight), dim=1))


class ResidualConcatGatedAFF(nn.Module):
    """Residual concat-gated AFF fusion that preserves the baseline concat path."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        reduction: int = 4,
        alpha: float = 0.1,
        attention_mode: str = "ours_ours",
        weight_mode: str = "complement",
        residual_attention: nn.Module | None = None,
    ):
        """Initialize ResCGAFF with optional attention inside its residual enhancement path."""
        super().__init__()
        self.base = Conv(channels * 2, out_channels, 1, 1)
        self.aff = ConcatGatedAFF(channels, out_channels, reduction, attention_mode, weight_mode)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.residual_attention = residual_attention or nn.Identity()

    def forward(self, Frgb: torch.Tensor, Fir: torch.Tensor) -> torch.Tensor:
        """Apply the ResCGAFF equation ``Fout=B([Frgb,Fir])+alpha*P([Frgb*A,Fir*(1-A)])``."""
        if Frgb.ndim != 4 or Fir.ndim != 4 or Frgb.shape != Fir.shape:
            raise ValueError(
                "ResCGAFF expects matching [B, C, H, W] Frgb/Fir tensors, "
                f"got {tuple(Frgb.shape)} and {tuple(Fir.shape)}"
            )
        residual = self.residual_attention(self.aff(Frgb, Fir))
        return self.base(torch.cat((Frgb, Fir), dim=1)) + self.alpha * residual


class PostFusionAttention(nn.Module):
    """Apply attention after a complete two-input fusion, including its residual addition."""

    def __init__(self, fusion: nn.Module, attention: nn.Module):
        """Compose an existing fusion module with output attention."""
        super().__init__()
        self.fusion = fusion
        self.attention = attention

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Refine the fused feature before it feeds the neck and shared backbone."""
        return self.attention(self.fusion(rgb, ir))


class CMAEnhance(nn.Module):
    """Bidirectional cross-modal attention enhancement with a fixed output width."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4):
        """Initialize cross-attention projections for RGB-to-IR and IR-to-RGB updates."""
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.rgb_q = nn.Conv2d(channels, hidden, 1, bias=False)
        self.ir_k = nn.Conv2d(channels, hidden, 1, bias=False)
        self.ir_v = nn.Conv2d(channels, hidden, 1, bias=False)
        self.ir_q = nn.Conv2d(channels, hidden, 1, bias=False)
        self.rgb_k = nn.Conv2d(channels, hidden, 1, bias=False)
        self.rgb_v = nn.Conv2d(channels, hidden, 1, bias=False)
        self.project = Conv(hidden * 2, out_channels, 1, 1)
        self.scale = hidden**-0.5

    def _cross(self, query, key, value):
        """Compute scaled cross-attention over flattened spatial tokens."""
        b, _, h, w = query.shape
        q = query.flatten(2).transpose(1, 2)
        k = key.flatten(2)
        v = value.flatten(2).transpose(1, 2)
        return torch.softmax(q @ k * self.scale, dim=-1) @ v, h, w

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Return projected bidirectional cross-modal attention features."""
        rgb_to_ir, h, w = self._cross(self.rgb_q(rgb), self.ir_k(ir), self.ir_v(ir))
        ir_to_rgb, _, _ = self._cross(self.ir_q(ir), self.rgb_k(rgb), self.rgb_v(rgb))
        enhanced = torch.cat((rgb_to_ir.transpose(1, 2).reshape(rgb.shape[0], -1, h, w),
                              ir_to_rgb.transpose(1, 2).reshape(rgb.shape[0], -1, h, w)), dim=1)
        return self.project(enhanced)


class ASPPEnhance(nn.Module):
    """Apply shared-width atrous context branches independently to both modalities."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4):
        """Initialize three atrous branches and a final projection."""
        super().__init__()
        hidden = max(out_channels // (3 * reduction), 8)
        self.branches = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=dilation, dilation=dilation, bias=False),
                          nn.BatchNorm2d(hidden), nn.SiLU(inplace=True))
            for dilation in (1, 2, 3)
        )
        self.project = Conv(hidden * len(self.branches) * 2, out_channels, 1, 1)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Return concatenated RGB/IR multi-dilation context projected to output width."""
        features = [branch(stream) for stream in (rgb, ir) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class NonLocalEnhance(nn.Module):
    """Model long-range intra-modal dependencies with reduced query/key channels."""

    def __init__(self, channels: int, out_channels: int, reduction: int = 4):
        """Initialize independent RGB and IR non-local projections."""
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.q = nn.Conv2d(channels, hidden, 1, bias=False)
        self.k = nn.Conv2d(channels, hidden, 1, bias=False)
        self.v = nn.Conv2d(channels, hidden, 1, bias=False)
        self.project = Conv(hidden * 2, out_channels, 1, 1)
        self.scale = hidden**-0.5

    def _self(self, stream):
        """Compute reduced non-local self-attention for one modality."""
        b, _, h, w = stream.shape
        q = self.q(stream).flatten(2).transpose(1, 2)
        k = self.k(stream).flatten(2)
        v = self.v(stream).flatten(2).transpose(1, 2)
        return (torch.softmax(q @ k * self.scale, dim=-1) @ v).transpose(1, 2).reshape(b, -1, h, w)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Return projected RGB/IR non-local responses."""
        return self.project(torch.cat((self._self(rgb), self._self(ir)), dim=1))


class DualStreamEnhanceFusion(nn.Module):
    """Fuse two modality streams with a selectable enhancement and residual rule."""

    def __init__(self, channels: int, out_channels: int, module: str = "none", fusion: str = "r0",
                 reduction: int = 4, alpha: float = 0.1, gate_init: float = -2.0):
        """Initialize a shared base path and one explicitly configured residual path."""
        super().__init__()
        module = module.lower()
        fusion = fusion.lower()
        modules = {"cma": CMAEnhance, "aspp": ASPPEnhance, "nonlocal": NonLocalEnhance}
        if module not in ("none", *modules):
            raise ValueError(f"Unknown enhancement module {module!r}")
        if fusion not in {"r0", "r1", "r2", "r3", "r4", "r5"}:
            raise ValueError(f"Unknown residual fusion {fusion!r}")
        if module == "none" and fusion != "r0":
            raise ValueError("An enhancement module is required for residual modes other than r0")
        self.module_name, self.fusion_name = module, fusion
        self.base = Conv(channels * 2, out_channels, 1, 1)
        self.enhance = modules[module](channels, out_channels, reduction) if module != "none" else None
        if fusion == "r3":
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        elif fusion == "r4":
            self.channel_alpha = nn.Parameter(torch.full((1, out_channels, 1, 1), alpha))
        elif fusion == "r5":
            self.gate = nn.Conv2d(out_channels * 2, out_channels, 1)
            nn.init.constant_(self.gate.bias, gate_init)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Return the configured base plus residual enhancement output."""
        base = self.base(torch.cat((rgb, ir), dim=1))
        if self.enhance is None:
            return base
        enhanced = self.enhance(rgb, ir)
        if self.fusion_name == "r1":
            return base + enhanced
        if self.fusion_name == "r2":
            return base + 0.1 * enhanced
        if self.fusion_name == "r3":
            return base + self.alpha * enhanced
        if self.fusion_name == "r4":
            return base + self.channel_alpha * enhanced
        gate = torch.sigmoid(self.gate(torch.cat((base, enhanced), dim=1)))
        return base + gate * enhanced


class BGF(nn.Module):
    """Bilateral guided fusion base branch with sigmoid guidance and cross-modal residual updates."""

    def __init__(self, channels: int, out_channels: int):
        """Initialize BGF for two same-scale modality features."""
        super().__init__()
        self.rgb_conv = Conv(channels, channels, 3, 1)
        self.ir_conv = Conv(channels, channels, 3, 1)
        self.rgb_gate = nn.Sequential(Conv(channels, channels, 3, 1), nn.Sigmoid())
        self.ir_gate = nn.Sequential(Conv(channels, channels, 3, 1), nn.Sigmoid())
        self.rgb_add = Conv(channels, channels, 3, 1)
        self.ir_add = Conv(channels, channels, 3, 1)
        self.project = Conv(channels * 2, out_channels, 3, 1)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse modalities with independent gates, bilateral additions, and final refinement."""
        if rgb.ndim != 4 or ir.ndim != 4 or rgb.shape != ir.shape:
            raise ValueError(f"BGF expects matching [B, C, H, W] tensors, got {tuple(rgb.shape)} and {tuple(ir.shape)}")
        rgb_m = self.rgb_conv(rgb) * self.rgb_gate(rgb)
        ir_m = self.ir_conv(ir) * self.ir_gate(ir)
        return self.project(torch.cat((rgb_m + self.rgb_add(ir), ir_m + self.ir_add(rgb)), dim=1))


class MFB(nn.Module):
    """Multi-modal factorized bilinear fusion base branch."""

    def __init__(self, channels: int, out_channels: int):
        """Initialize factorized cross-modal multiplicative fusion."""
        super().__init__()
        self.rgb_conv = Conv(channels, channels, 3, 1)
        self.ir_conv = Conv(channels, channels, 3, 1)
        self.rgb_factor = Conv(channels, channels, 3, 1)
        self.ir_factor = Conv(channels, channels, 3, 1)
        self.rgb_add = Conv(channels, channels, 3, 1)
        self.ir_add = Conv(channels, channels, 3, 1)
        self.project = Conv(channels * 2, out_channels, 3, 1)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse modalities with factorized elementwise cross-modal interactions and refinement."""
        if rgb.ndim != 4 or ir.ndim != 4 or rgb.shape != ir.shape:
            raise ValueError(f"MFB expects matching [B, C, H, W] tensors, got {tuple(rgb.shape)} and {tuple(ir.shape)}")
        rgb_m = self.rgb_conv(rgb) * self.ir_factor(ir) + self.rgb_add(rgb)
        ir_m = self.ir_conv(ir) * self.rgb_factor(rgb) + self.ir_add(ir)
        return self.project(torch.cat((rgb_m, ir_m), dim=1))


class ESeriesFusion(nn.Module):
    """Select an E-series base branch and optionally add the unchanged current attention branch."""

    def __init__(
        self, channels: int, out_channels: int, base_branch: str = "cat", use_attention: bool = True,
        reduction: int = 4, alpha: float = 0.1,
    ):
        """Initialize one E0-E5 P3 fusion configuration."""
        super().__init__()
        if base_branch == "cat":
            self.base = Conv(channels * 2, out_channels, 1, 1)
        elif base_branch == "bgf":
            self.base = BGF(channels, out_channels)
        elif base_branch == "mfb":
            self.base = MFB(channels, out_channels)
        else:
            raise ValueError(f"base_branch must be cat, bgf, or mfb, got {base_branch!r}")
        self.base_branch = base_branch
        self.use_attention = use_attention
        if use_attention:
            self.attention = ConcatGatedAFF(channels, out_channels, reduction, "ours_ours", "complement")
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Return base fusion, optionally augmented by the current complementary attention branch."""
        base = self.base(torch.cat((rgb, ir), dim=1)) if self.base_branch == "cat" else self.base(rgb, ir)
        return base + self.alpha * self.attention(rgb, ir) if self.use_attention else base


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


class P2DifferenceFusion(nn.Module):
    """Add a configurable RGB/IR difference residual to the P2 concat baseline."""

    def __init__(
        self, channels: int, out_channels: int, reduction: int = 4, alpha: float = 0.1, mode: str = "baseline"
    ):
        """Initialize a P2 difference branch with optional base-feature context and gating."""
        super().__init__()
        if mode not in {"baseline", "difference", "base_difference", "base_difference_gate"}:
            raise ValueError(
                "P2 difference mode must be one of baseline, difference, base_difference, base_difference_gate, "
                f"got {mode!r}"
            )
        self.mode = mode
        self.base = Conv(channels * 2, out_channels, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        if mode != "baseline":
            branch_channels = channels if mode == "difference" else channels + out_channels
            self.difference = nn.Sequential(
                DWConv(branch_channels, branch_channels, 3, 1),
                Conv(branch_channels, out_channels, 1, 1),
            )
        if mode == "base_difference_gate":
            self.gate = nn.Sequential(Conv(out_channels * 2, out_channels, 1, 1), nn.Sigmoid())

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Fuse P2 streams with an optional explicit difference residual."""
        base = self.base(torch.cat((rgb, ir), dim=1))
        if self.mode == "baseline":
            return base
        difference = torch.abs(rgb - ir)
        branch_input = difference if self.mode == "difference" else torch.cat((base, difference), dim=1)
        enhanced = self.difference(branch_input)
        if self.mode == "base_difference_gate":
            enhanced = self.gate(torch.cat((base, enhanced), dim=1)) * enhanced
        return base + self.alpha * enhanced


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
        cgaff_attention: str = "ours_ours",
        cgaff_weight: str = "complement",
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
            self.fuse = ResidualConcatGatedAFF(
                branch_channels, p3_channels, aff_reduction, residual_alpha, cgaff_attention, cgaff_weight
            )
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
    """Fuse P2/P3 as before, then preserve RGB/IR streams for configurable P4/P5 enhancement."""

    def __init__(
        self,
        p2_channels: int = 32,
        p3_channels: int = 64,
        p4_channels: int = 128,
        p5_channels: int = 256,
        aff_reduction: int = 4,
        residual_alpha: float = 0.1,
        base_branch: str = "cat",
        use_attention: bool = True,
        p2_rescgaff: bool = False,
        p3_attention: str = "none",
        p4_enhance: str = "none",
        p4_fusion: str = "r0",
        p5_enhance: str = "none",
        p5_fusion: str = "r0",
        enhance_reduction: int = 4,
        gate_init: float = -2.0,
        p2_mode: str = "baseline",
    ):
        """Initialize separate RGB/IR P2-P3 stages and independent P4/P5 dual streams."""
        super().__init__()
        output_channels = (p2_channels, p3_channels, p4_channels, p5_channels)
        if any(channels < 16 or channels % 2 for channels in output_channels):
            raise ValueError("P2, P3, P4, and P5 channels must be even integers of at least 16")

        branch_p2 = p2_channels // 2
        branch_p3 = p3_channels // 2
        stem_channels = branch_p2 // 2
        if stem_channels < 8:
            raise ValueError("p2_channels must be at least 32")

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
        self.fuse_p2 = (
            ResidualConcatGatedAFF(branch_p2, p2_channels, aff_reduction, residual_alpha)
            if p2_rescgaff
            else (
                Conv(p2_channels, p2_channels, 1, 1)
                if p2_mode == "baseline"
                else P2DifferenceFusion(branch_p2, p2_channels, aff_reduction, residual_alpha, p2_mode)
            )
        )

        self.rgb_p3 = _make_modality_stage(branch_p2, branch_p3, repeats=1)
        self.ir_p3 = _make_modality_stage(branch_p2, branch_p3, repeats=1)
        attention_modules = {"cbam": CBAM, "ema": EMAAttention, "ca": CoordinateAttention, "eca": ECAttention}
        if p3_attention not in ("none", *attention_modules):
            raise ValueError(f"Unknown P3 attention {p3_attention!r}; choose from {tuple(attention_modules)}")
        if base_branch == "cat" and use_attention:
            residual_attention = (
                attention_modules[p3_attention](p3_channels) if p3_attention != "none" else None
            )
            self.fuse_p3 = ResidualConcatGatedAFF(
                branch_p3, p3_channels, aff_reduction, residual_alpha, residual_attention=residual_attention
            )
        else:
            self.fuse_p3 = ESeriesFusion(branch_p3, p3_channels, base_branch, use_attention, aff_reduction, residual_alpha)
            if p3_attention != "none":
                self.fuse_p3 = PostFusionAttention(self.fuse_p3, attention_modules[p3_attention](p3_channels))

        self.rgb_p4 = _make_modality_stage(branch_p3, p4_channels, repeats=2)
        self.ir_p4 = _make_modality_stage(branch_p3, p4_channels, repeats=2)
        self.fuse_p4 = DualStreamEnhanceFusion(
            p4_channels, p4_channels, p4_enhance, p4_fusion, enhance_reduction, residual_alpha, gate_init
        )
        self.rgb_p5 = nn.Sequential(
            Conv(p4_channels, p5_channels, 3, 2),
            C2f(p5_channels, p5_channels, n=1, shortcut=True),
        )
        self.ir_p5 = nn.Sequential(
            Conv(p4_channels, p5_channels, 3, 2),
            C2f(p5_channels, p5_channels, n=1, shortcut=True),
        )
        self.fuse_p5 = DualStreamEnhanceFusion(
            p5_channels, p5_channels, p5_enhance, p5_fusion, enhance_reduction, residual_alpha, gate_init
        )
        self.sppf = SPPF(p5_channels, p5_channels, 5)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return fused P2, P3, P4, and P5 feature maps."""
        if x.ndim != 4 or x.shape[1] != 6:
            raise ValueError(f"ResCGAFFP2Backbone expects [B, 6, H, W], got {tuple(x.shape)}")

        rgb_p2 = self.rgb_p2(x[:, :3])
        ir_p2 = self.ir_p2(x[:, 3:6])
        p2 = (
            self.fuse_p2(rgb_p2, ir_p2)
            if isinstance(self.fuse_p2, (ResidualConcatGatedAFF, P2DifferenceFusion))
            else self.fuse_p2(torch.cat((rgb_p2, ir_p2), dim=1))
        )

        rgb_p3 = self.rgb_p3(rgb_p2)
        ir_p3 = self.ir_p3(ir_p2)
        p3 = self.fuse_p3(rgb_p3, ir_p3)
        rgb_p4 = self.rgb_p4(rgb_p3)
        ir_p4 = self.ir_p4(ir_p3)
        p4 = self.fuse_p4(rgb_p4, ir_p4)
        rgb_p5 = self.rgb_p5(rgb_p4)
        ir_p5 = self.ir_p5(ir_p4)
        p5 = self.sppf(self.fuse_p5(rgb_p5, ir_p5))
        return [p2, p3, p4, p5]


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
        cgaff_attention: str = "ours_ours",
        cgaff_weight: str = "complement",
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
            fusion_pattern[0], branch_p3, p3_channels, aff_reduction, residual_alpha, cgaff_attention, cgaff_weight
        )
        self.fuse_p4 = self._make_fusion(
            fusion_pattern[1], branch_p4, p4_channels, aff_reduction, residual_alpha, cgaff_attention, cgaff_weight
        )
        self.fuse_p5 = self._make_fusion(
            fusion_pattern[2], branch_p5, p5_channels, aff_reduction, residual_alpha, cgaff_attention, cgaff_weight
        )
        self.sppf = SPPF(p5_channels, p5_channels, 5)

    @staticmethod
    def _make_fusion(
        mode: str, channels: int, out_channels: int, reduction: int, alpha: float, attention_mode: str, weight_mode: str
    ) -> nn.Module:
        """Build one scale's residual fusion module from its compact mode code."""
        if mode == "R":
            return ResidualConcatGatedAFF(channels, out_channels, reduction, alpha, attention_mode, weight_mode)
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
