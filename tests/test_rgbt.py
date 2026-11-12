import pytest
import torch

from ultralytics.nn.modules.rgbt import ResidualConcatGatedAFF, ResCGAFFP2Backbone


@pytest.mark.parametrize("fusion", ("r1", "r2", "r3", "r4", "r5"))
def test_rescgaff_residual_fusion(fusion):
    module = ResidualConcatGatedAFF(8, 16, fusion=fusion)
    rgb = torch.randn(2, 8, 10, 10, requires_grad=True)
    ir = torch.randn(2, 8, 10, 10, requires_grad=True)
    output = module(rgb, ir)

    assert output.shape == (2, 16, 10, 10)
    output.mean().backward()
    assert rgb.grad is not None and ir.grad is not None
    if fusion == "r3":
        assert module.alpha.item() == pytest.approx(0.1)
        assert module.alpha.grad is not None
    elif fusion == "r4":
        assert module.channel_alpha.shape == (1, 16, 1, 1)
        assert module.channel_alpha.grad is not None
    elif fusion == "r5":
        assert torch.allclose(module.gate.bias, torch.full_like(module.gate.bias, -2.0))
        assert module.gate.weight.grad is not None


@pytest.mark.parametrize("p2_fusion,p3_fusion", (("r1", "r3"), ("r4", "r3"), ("r5", "r3"), ("r3", "r1"), ("r3", "r4"), ("r3", "r5")))
def test_rescgaff_backbone_fusion_modes(p2_fusion, p3_fusion):
    model = ResCGAFFP2Backbone(
        p2_rescgaff=True,
        p4_enhance="cma",
        p4_fusion="r4",
        p5_enhance="aspp",
        p5_fusion="r3",
        p2_fusion=p2_fusion,
        p3_fusion=p3_fusion,
    )
    outputs = model(torch.randn(1, 6, 64, 64))

    assert [tuple(output.shape) for output in outputs] == [
        (1, 32, 16, 16),
        (1, 64, 8, 8),
        (1, 128, 4, 4),
        (1, 256, 2, 2),
    ]


def test_rescgaff_backbone_legacy_defaults():
    model = ResCGAFFP2Backbone(p2_rescgaff=True)

    assert model.fuse_p2.fusion_name == "r3"
    assert model.fuse_p3.fusion_name == "r3"

