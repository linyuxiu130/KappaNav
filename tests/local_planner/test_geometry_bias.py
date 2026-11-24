"""CUDA fusion agrees with neural precision math and retains no hidden activation."""
from copy import deepcopy
import pytest
import torch
from torch.nn import functional as F

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),reason='CUDA BF16 required')


def parameters(hidden,heads):
    return (torch.nn.Parameter(torch.randn(hidden,7,device='cuda')*.15),
        torch.nn.Parameter(torch.randn(hidden,device='cuda')*.1),
        torch.nn.Parameter(torch.randn(heads,hidden,device='cuda')*.1),
        torch.nn.Parameter(torch.randn(heads,device='cuda')*.1))


@pytest.mark.parametrize('shape,hidden,heads,geometry_grad,strided,training_tiles',[
    ((1,),8,4,True,False,False),((65,),96,8,True,True,False),((257,),96,8,False,False,False),
    ((16384,),96,8,True,False,False),((2,7,33),96,8,True,False,False),
    ((3,17,67),96,8,False,False,False),((3,17,67),96,8,False,False,True)])
def test_register_bias_matches_rounded_fp32_oracle_and_all_gradients(monkeypatch,shape,hidden,heads,geometry_grad,strided,training_tiles):
    if training_tiles:
        from kappanav.local_planner.training.execution import register_training_bias_kernels
        register_training_bias_kernels()
    from kappanav.local_planner.model.network._geometry_bias import geometric_bias
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    torch.manual_seed(908)
    x=(torch.randn(7,*shape,device='cuda',dtype=torch.bfloat16).movedim(0,-1) if strided else
        torch.randn(*shape,7,device='cuda',dtype=torch.bfloat16)).detach().requires_grad_(geometry_grad)
    weights=parameters(hidden,heads);probe=torch.randn(*shape,heads,device='cuda',dtype=torch.bfloat16)
    values=(x,*weights);actual=geometric_bias(*values)
    if len(shape)==3:assert actual.permute(0,3,1,2).is_contiguous()
    gradients=torch.autograd.grad(actual,[v for v in values if v.requires_grad],probe)
    oracle=[v.detach().float().requires_grad_(v.requires_grad) for v in values]
    xx,w1,b1,w2,b2=oracle
    z=F.linear(xx,w1.to(torch.bfloat16).float(),b1.to(torch.bfloat16).float()).to(torch.bfloat16).float()
    h=F.silu(z).to(torch.bfloat16).float()
    expected=F.linear(h,w2.to(torch.bfloat16).float(),b2.to(torch.bfloat16).float()).to(torch.bfloat16)
    expected_grad=torch.autograd.grad(expected,[v for v in oracle if v.requires_grad],probe)
    error=(actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)
    assert error<.001
    for a,b in zip(gradients,expected_grad,strict=True):
        assert torch.isfinite(a).all()
        error=(a.float()-b.to(torch.bfloat16).float()).norm()/b.float().norm().clamp_min(1e-12)
        assert error<.001
    zero=torch.autograd.grad(geometric_bias(*values),[v for v in values if v.requires_grad],torch.zeros_like(probe))
    assert all(torch.count_nonzero(v)==0 for v in zero)


def test_register_bias_retains_geometry_not_per_pair_hidden_values():
    from kappanav.local_planner.model.network._geometry_bias import geometric_bias
    torch.manual_seed(909);x=torch.randn(1000,7,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    weights=parameters(96,8);saved=[]
    def pack(tensor):saved.append(tuple(tensor.shape));return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        output=geometric_bias(x,*weights)
    assert (1000,96) not in saved and (1000,7) in saved
    output.float().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (x,*weights))


@pytest.mark.parametrize('dropout',[0.,.2])
def test_fused_attention_checkpoint_preserves_values_gradients_and_rng(dropout):
    from kappanav.local_planner.model.network.blocks import ReusableConditionCrossAttention
    from kappanav.local_planner.model.types import ConditionFeatures
    torch.manual_seed(910);base=ReusableConditionCrossAttention(32,4,dropout).cuda().train()
    query=torch.randn(4,7,32,device='cuda',dtype=torch.bfloat16)
    geometry=torch.randn(4,7,17,7,device='cuda');tokens=torch.randn(2,17,32,device='cuda')
    valid=torch.ones(2,17,device='cuda',dtype=torch.bool);valid[:,-3:]=False
    results=[]
    for budget in (0,512<<20):
        layer=deepcopy(base);layer.saved_bias_bytes=budget
        q=query.clone().requires_grad_();g=geometry.clone().requires_grad_();t=tokens.clone().requires_grad_()
        condition=ConditionFeatures(tokens=t,token_valid=valid,metric_position=None,surface_hit=None,frame_age=None,motion_token=None,terminal_goal=None,goal_present=None,configuration_field=None)
        torch.manual_seed(911)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=layer(q,layer.project_condition(condition),g)
        gradients=torch.autograd.grad(out.float().square().sum(),(q,g,t,*layer.parameters()))
        results.append((out,gradients,torch.cuda.get_rng_state()))
    a,b=results;torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    assert torch.equal(a[2],b[2])
    for left,right in zip(a[1],b[1],strict=True):
        torch.testing.assert_close(left,right,rtol=.01,atol=.005)


def test_training_monitoring_and_inference_share_the_fused_forward(monkeypatch):
    from kappanav.local_planner.model.network import _geometry_bias
    from kappanav.local_planner.model.network.blocks import ReusableConditionCrossAttention
    torch.manual_seed(918)
    layer=ReusableConditionCrossAttention(32,4,0.).cuda().eval()
    q=torch.randn(1,4,7,8,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(1,4,17,8,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    pairs=torch.randn(1,7,17,7,device='cuda');valid=torch.ones(1,1,1,17,device='cuda',dtype=torch.bool)
    calls=[];original=_geometry_bias.geometric_bias
    def audited(*args):
        calls.append(torch.is_grad_enabled())
        return original(*args)
    monkeypatch.setattr(_geometry_bias,'geometric_bias',audited)
    with torch.autocast('cuda',dtype=torch.bfloat16):training=layer._attend(q,k,v,pairs,valid)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):monitoring=layer._attend(q,k,v,pairs,valid)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):inference=layer._attend(q,k,v,pairs,valid)
    assert calls==[True,False,False] and not monitoring.requires_grad and not inference.requires_grad
    torch.testing.assert_close(monitoring,training,rtol=0,atol=0)
    torch.testing.assert_close(inference,training,rtol=0,atol=0)


def test_fused_bias_adamw_state_roundtrip_continues_the_same_update(tmp_path):
    from kappanav.local_planner.model.network._geometry_bias import geometric_bias
    torch.manual_seed(913)
    model=torch.nn.Sequential(torch.nn.Linear(7,96),torch.nn.SiLU(),torch.nn.Linear(96,8)).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True)
    x=torch.randn(129,7,device='cuda',dtype=torch.bfloat16)
    def step(module,opt):
        opt.zero_grad(set_to_none=True)
        output=geometric_bias(x,module[0].weight,module[0].bias,module[2].weight,module[2].bias)
        output.float().square().mean().backward();opt.step()
    step(model,optimizer)
    path=tmp_path/'state.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),path)
    step(model,optimizer);expected=deepcopy(model.state_dict())
    restored=deepcopy(model);opt=torch.optim.AdamW(restored.parameters(),lr=1e-4,fused=True)
    saved=torch.load(path,weights_only=True)
    restored.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
    step(restored,opt)
    for name,value in expected.items():
        torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)

@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),reason='CUDA BF16 required')
def test_fused_bias_budget_does_not_checkpoint_retired_hidden_activation(monkeypatch):
    from kappanav.local_planner.model.network import blocks
    from kappanav.local_planner.model.types import ConditionFeatures
    torch.manual_seed(917)
    layer=blocks.ReusableConditionCrossAttention(384,8,0.).cuda().eval()
    layer.saved_bias_bytes=30_000
    query=torch.randn(4,7,384,device='cuda',requires_grad=True)
    geometry=torch.randn(4,7,17,7,device='cuda',requires_grad=True)
    tokens=torch.randn(2,17,384,device='cuda',requires_grad=True)
    condition=ConditionFeatures(tokens=tokens,token_valid=torch.ones(2,17,device='cuda',dtype=torch.bool),metric_position=None,surface_hit=None,frame_age=None,motion_token=None,terminal_goal=None,goal_present=None,configuration_field=None)
    def unexpected_checkpoint(*args,**kwargs):
        raise AssertionError('Fused bias fits the saved-tensor budget; hidden96 is not retained')
    monkeypatch.setattr(blocks,'checkpoint',unexpected_checkpoint)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=layer(query,layer.project_condition(condition),geometry)
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (query,geometry,tokens,*layer.parameters()))


@pytest.mark.parametrize('count,hidden,heads',[(1,8,4),(2053,96,8)])
def test_weight_gradient_reduction_keeps_neural_rounding_and_own_storage(count,hidden,heads):
    from kappanav.local_planner.model.network._geometry_bias import _weight_gradients
    torch.manual_seed(919)
    weights=parameters(hidden,heads)
    partial=torch.randn(count,hidden*(8+heads)+heads,device='cuda')
    if count==1:
        partial[0,:8]=partial.new_tensor([0.,-0.,1.+2.**-8,1.+3*2.**-8,-1.-2.**-8,2.**-133,2.**-134,torch.finfo(torch.float32).max])
    actual=_weight_gradients(partial,*weights)
    expected=partial.sum(0).to(torch.bfloat16).float().split((hidden*7,hidden,heads*hidden,heads))
    assert len({value.untyped_storage().data_ptr() for value in actual})==4
    for value,reference in zip(actual,expected,strict=True):
        torch.testing.assert_close(value.flatten(),reference,rtol=0,atol=0)


def test_fused_attention_saves_packed_geometry_once_for_strided_neural_input():
    from kappanav.local_planner.model.network.blocks import ReusableConditionCrossAttention
    torch.manual_seed(921)
    actual=ReusableConditionCrossAttention(32,4,0.).cuda().eval();reference=deepcopy(actual)
    q=torch.randn(2,4,7,8,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn(2,4,17,8,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    geometry=torch.randn(2,13,17,7,device='cuda',dtype=torch.bfloat16)[:,:7].detach().requires_grad_()
    assert not geometry.is_contiguous()
    valid=torch.ones(2,1,1,17,device='cuda',dtype=torch.bool);valid[...,-3:]=False
    saved=[]
    def pack(tensor):
        if tensor.dtype==torch.bfloat16 and tensor.shape==geometry.shape:saved.append(tensor)
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        first=actual._attend(q,k,v,geometry,valid)
    assert len(saved)==1 and saved[0].is_contiguous()
    other=geometry.detach().float().requires_grad_()
    second=reference._attend(q,k,v,other,valid)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    a=torch.autograd.grad(first.float().square().sum(),(geometry,q,*actual.relative_bias.parameters()))
    b=torch.autograd.grad(second.float().square().sum(),(other,q,*reference.relative_bias.parameters()))
    for x,y in zip(a,b,strict=True):torch.testing.assert_close(x.float(),y.float(),rtol=.001,atol=1e-5)
