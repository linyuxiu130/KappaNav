import numpy as np
import torch
import pytest


def test_fine_cells_reach_condition_without_pooling():
    from types import SimpleNamespace
    from kappanav.local_planner.model.types import ConditionFeatures
    from kappanav.local_planner.features.local_features import append_fine_local
    field=torch.randn(1,6,64,64)
    condition=ConditionFeatures(torch.zeros(1,2,8),torch.ones(1,2,dtype=torch.bool),torch.zeros(1,2,3),
        torch.zeros(1,2),torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool),torch.zeros(1,2),torch.zeros(1,1),field)
    visual=torch.randn(3,8,requires_grad=True)
    actor=SimpleNamespace(direct_local=True,policy=SimpleNamespace(condition_encoder=SimpleNamespace(memory_norm=torch.nn.Identity())))
    plan=dict(points=np.arange(9,dtype=np.float32).reshape(3,3),clock=5.,times=np.array([3.5,4.,5.]))
    encoded=append_fine_local(actor,condition,[plan],[visual])
    torch.testing.assert_close(encoded.tokens[0,2:],visual)
    torch.testing.assert_close(encoded.metric_position[0,2:],torch.from_numpy(plan['points']))
    torch.testing.assert_close(encoded.frame_age[0,2:],torch.tensor([1.5,1.,0.]))
    assert encoded.configuration_field is field
    encoded.tokens.sum().backward();assert torch.equal(visual.grad,torch.ones_like(visual))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('lengths',[(0,3,7),(0,0)])
def test_packed_fine_cells_keep_masked_norm_values_and_gradients(device,lengths):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from types import SimpleNamespace
    from kappanav.local_planner.model.types import ConditionFeatures
    from kappanav.local_planner.model.layers import RMSNorm
    from kappanav.local_planner.features.local_features import append_fine_local
    torch.manual_seed(81)
    b=len(lengths);c=8;count=max(lengths)
    norm=RMSNorm(c).to(device)
    actor=SimpleNamespace(policy=SimpleNamespace(condition_encoder=SimpleNamespace(memory_norm=norm)))
    condition=ConditionFeatures(torch.zeros(b,2,c,device=device),torch.ones(b,2,device=device,dtype=torch.bool),
        torch.zeros(b,2,3,device=device),torch.zeros(b,2,device=device),torch.zeros(b,2,device=device),
        torch.zeros(b,2,device=device,dtype=torch.bool),torch.zeros(b,2,device=device),
        torch.zeros(b,1,device=device),torch.zeros(b,6,64,64,device=device))
    visuals=[torch.randn(n,c,device=device,requires_grad=True) for n in lengths]
    plans=[dict(points=np.arange(n*3,dtype=np.float32).reshape(n,3),clock=5.,times=np.arange(n)) for n in lengths]
    valid=torch.arange(count,device=device)[None]<torch.tensor(lengths,device=device)[:,None]
    padded=torch.stack([torch.nn.functional.pad(v,(0,0,0,count-len(v))) for v in visuals])
    expected=torch.cat((condition.tokens,norm(padded)*valid[...,None]),1)
    actual=append_fine_local(actor,condition,plans,visuals)
    torch.testing.assert_close(actual.tokens,expected,rtol=1e-6,atol=1e-6)
    assert torch.equal(actual.token_valid[:,2:],valid)
    probe=torch.randn_like(expected)
    a=torch.autograd.grad(expected,(*visuals,norm.weight),probe)
    b=torch.autograd.grad(actual.tokens,(*visuals,norm.weight),probe)
    for first,second in zip(a,b,strict=True):
        torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-6)


