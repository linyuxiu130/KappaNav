from types import SimpleNamespace

import torch
from torch import nn

from kappanav.global_planner.network import GlobePlanner, MaskedSemanticReadout, local_upsample
from kappanav.global_planner.taxonomy import CLASSES


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(16,16)
    def forward(self,tokens,**kwargs):return tokens+self.linear(tokens)*.1


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=16,patch_size=16)
        self.projection=nn.Conv2d(3,16,16,stride=16)
        self.layer=nn.ModuleList(TinyBlock() for _ in range(24))
        self.norm=nn.LayerNorm(16)
    def rope_embeddings(self,rgb):return None
    def embeddings(self,rgb):return self.projection(rgb).flatten(2).transpose(1,2)


def test_native_class_maps_for_any_crop_side_and_resolution():
    torch.manual_seed(0)
    model=GlobePlanner(TinyBackbone(),.15).train()
    for side in (64,160):
        rgb=torch.randn(2,3,side,side)
        logits=model(rgb,torch.tensor([.1,.2]))
        assert logits.shape==(2,len(CLASSES),side,side)
        logits.logsumexp(1).mean().backward()
    assert model.backbone.layer[0].linear.weight.grad is not None


def test_resolution_conditioning_starts_neutral_and_then_changes_the_output():
    torch.manual_seed(1)
    model=GlobePlanner(TinyBackbone(),.15).eval()
    rgb=torch.randn(1,3,64,64)
    with torch.no_grad():
        a=model(rgb,torch.tensor([.08]));b=model(rgb,torch.tensor([.25]))
        torch.testing.assert_close(a,b)
        model.head.resolution.weight.fill_(.5)
        c=model(rgb,torch.tensor([.08]));d=model(rgb,torch.tensor([.25]))
    assert (c-d).abs().max()>0


def test_masked_readout_starts_as_the_plain_classifier():
    torch.manual_seed(42);flat=nn.Conv2d(64,len(CLASSES),1)
    torch.manual_seed(42);readout=MaskedSemanticReadout(64)
    x=torch.randn(2,64,5,9)
    torch.testing.assert_close(readout(x),flat(x),rtol=0,atol=0)


def test_local_readout_preserves_constants():
    logits=torch.full((1,13,3,4),2.5);kernels=torch.randn(1,144,3,4)
    torch.testing.assert_close(local_upsample(logits,kernels),torch.full((1,13,12,16),2.5))
