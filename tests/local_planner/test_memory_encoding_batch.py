"""Batch independent geometry encodings without changing inputs or gradients."""
from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from torch import nn
from kappanav.local_planner.model.conditioning.transformer import PolicyConditionEncoder
from kappanav.local_planner.features.temporal_geometry import FeatureGeometryDecoder
from kappanav.local_planner.features.history_fusion import encode_memories
from reference_implementations import encode_memory


def test_batch_memory_fields_tokens_and_all_gradients_match_scalar():
    torch.manual_seed(35)
    policy=nn.Module();policy.depth_encoder=SimpleNamespace(max_depth_m=10.,spatial_projection=SimpleNamespace(out_channels=32))
    policy.condition_encoder=PolicyConditionEncoder(observation_frames=4,planning_horizon_m=3.6,model_dim=32)
    policy.condition_encoder.motion_encoder.neural_dtype=torch.float32
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8);head.neural_dtype=torch.float32
    other_policy=deepcopy(policy);other_head=deepcopy(head)
    plans=[]
    for sample in range(2):
        pose=np.eye(4);pose[:2,3]=[sample*.23,-.16]
        poses=np.tile(pose,(4,1,1));poses[:,0,3]-=np.linspace(.6,0,4)
        rng=np.random.default_rng(35+sample);n=9
        plans.append(dict(pixels=np.arange(n,dtype=np.int64),offsets=np.zeros((n,3),np.float32),depths=np.full(n,2.,np.float32),ages=np.linspace(2,0,n).astype(np.float32),kinds=np.arange(n)%2,lengths=np.ones(n,np.int64),frame_ids=np.arange(n)%4,points=rng.uniform(-2.,2.,(n,3)).astype(np.float32),depth=np.full(n,2.,np.float32),
            keys=np.column_stack((np.arange(n),np.zeros(n),np.arange(n)%2)).astype(np.int64),origins=np.zeros((n,3),np.float32),pose=pose,
            selection=dict(valid=np.ones(4,bool),poses=poses,ages=np.linspace(2,0,4).astype(np.float32))))
    visual=[torch.randn(9,32,requires_grad=True) for _ in plans]
    reference_visual=[v.detach().clone().requires_grad_() for v in visual]
    goals=[torch.tensor([2.,1.]),torch.tensor([9.,-4.])];present=[True,False]
    scalar=[encode_memory(policy,head,p,v,g,q) for p,v,g,q in zip(plans,visual,goals,present,strict=True)]
    from concurrent.futures import ThreadPoolExecutor
    from kappanav.local_planner.training.batches import prepare_training_batch
    prepared=deepcopy(plans)
    raw=dict(depths=np.ones((4,47,89),np.float32))
    records=[(dict(history_plan=p),raw,None,None) for p in prepared]
    rng_state=torch.get_rng_state().clone()
    with ThreadPoolExecutor(max_workers=1) as loader:
        loaded=loader.submit(prepare_training_batch,SimpleNamespace(policy=other_policy,head=other_head,memory=None),records,[1,0,1],bucket=512).result()
    assert loaded==[records[1],records[0],records[1]] and torch.equal(rng_state,torch.get_rng_state())
    from kappanav.local_planner.features.history_fusion import _plan_inputs
    assert all('_fusion_inputs' in _plan_inputs(p) for p in prepared)
    from kappanav.local_planner.features.history_fusion import source_fusion_inputs
    for original,copy in zip(plans,prepared,strict=True):
        for a,b in zip(source_fusion_inputs(policy.depth_encoder,original),source_fusion_inputs(other_policy.depth_encoder,copy),strict=True):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(((1,32,3,6),512) in _plan_inputs(p)['_source_sample_inputs'] for p in prepared)
    actual,(field,logits)=encode_memories(other_policy,other_head,prepared,reference_visual,goals,present)
    # Distance-derived channels amplify FP32 rounding by 1/res (sigmoids by <=1/(4*res)).
    res=2*head.horizon/63
    metric_scale=field.new_tensor([1.,res,res,4*res,4*res,1.])[None,:,None,None]
    for name in [f.name for f in fields(actual)]:
        if getattr(actual,name) is None:continue
        expected=torch.cat([getattr(s[0],name) for s in scalar])
        scale=metric_scale if name=='configuration_field' else 1.
        atol=1e-4 if name=='configuration_field' else 2e-5
        torch.testing.assert_close(getattr(actual,name)*scale,expected*scale,rtol=3e-5,atol=atol)
    expected_field=torch.cat([s[1][0] for s in scalar]);expected_logits=torch.cat([s[1][1] for s in scalar])
    torch.testing.assert_close(field*metric_scale,expected_field*metric_scale,rtol=3e-5,atol=1e-4)
    torch.testing.assert_close(logits,expected_logits,rtol=3e-5,atol=2e-5)
    reference_loss=sum(s[0].tokens.square().mean()+s[1][0].square().mean()+s[1][1].square().mean() for s in scalar)/2
    actual_loss=actual.tokens.square().mean()+field.square().mean()+logits.square().mean()
    a=torch.autograd.grad(reference_loss,list(policy.parameters())+list(head.parameters())+visual,allow_unused=True)
    b=torch.autograd.grad(actual_loss,list(other_policy.parameters())+list(other_head.parameters())+reference_visual,allow_unused=True)
    for expected,actual in zip(a,b,strict=True):
        assert (expected is None)==(actual is None)
        if expected is not None:torch.testing.assert_close(actual,expected,rtol=5e-4,atol=3e-5)
    assert not actual_loss.isnan()
    empty=deepcopy(plans[0])
    for name in ('points','depth','keys','origins','frame_ids','pixels','offsets','depths','ages','kinds','lengths'):
        empty[name]=empty[name][:0]
    empty_records=[(dict(history_plan=empty),raw,None,None)]
    assert prepare_training_batch(SimpleNamespace(policy=other_policy,head=other_head,memory=None),empty_records,[0],bucket=512)==empty_records
    assert '_encoding_inputs' in _plan_inputs(empty)


@pytest.mark.parametrize('count',[0,9])
def test_static_segment_raster_preserves_values_and_feature_gradients(count):
    from kappanav.local_planner.model.encoders.configuration import (memory_splat_stencil, static_splat_segments,
                                                                     _StaticMemorySplat)
    from reference_implementations import apply_memory_splat
    torch.manual_seed(731)
    points=torch.rand(1,count,3)*2-1
    stencil=memory_splat_stencil(points,torch.ones(1,count,dtype=torch.bool),3.6,4,0)
    entries,mass,size=stencil
    tokens=torch.randn(1,count,5,requires_grad=True)
    indices=torch.stack([item[0] for item in entries])
    segments=static_splat_segments(stencil)
    actual=_StaticMemorySplat.apply(tokens,*segments[:4],indices,segments[4],size)
    expected,_=apply_memory_splat(tokens,stencil)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
    probe=torch.randn_like(actual)
    torch.testing.assert_close(torch.autograd.grad((actual*probe).sum(),tokens)[0],
        torch.autograd.grad((expected*probe).sum(),tokens)[0],rtol=2e-6,atol=2e-6)
    if count:
        values=tokens.detach().double().requires_grad_()
        def splat(value):return _StaticMemorySplat.apply(value,*segments[:4],indices,segments[4],size)
        assert torch.autograd.gradcheck(splat,(values,))
        assert torch.autograd.gradgradcheck(splat,(values,))


def test_sparse_affine_projection_preserves_dense_head_and_all_gradients():
    from kappanav.local_planner.features.history_fusion import memory_rasters
    torch.manual_seed(133)
    policy=nn.Module();policy.depth_encoder=SimpleNamespace(max_depth_m=10.)
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8);head.neural_dtype=torch.float32
    other=deepcopy(head);rng=np.random.default_rng(133);points=rng.uniform(-1,1,(9,3)).astype(np.float32)
    points[1:3]=points[0]+.001
    plan=dict(points=points,depth=np.full(9,2.,np.float32),keys=np.column_stack((np.arange(9),np.zeros(9),np.arange(9)%2)).astype(np.int64),origins=np.zeros((9,3),np.float32),pose=np.eye(4),selection=dict(valid=np.ones(4,bool),poses=np.tile(np.eye(4),(4,1,1)),ages=np.linspace(2,0,4).astype(np.float32)))
    visual=torch.randn(9,32,requires_grad=True);other_visual=visual.detach().clone().requires_grad_()
    raw,mass,pose,*_=memory_rasters(policy,head,[plan],[visual])
    projected,other_mass,other_pose,*_=memory_rasters(policy,other,[deepcopy(plan)],[other_visual],project=True)
    assert raw.shape[1]==40 and projected.shape[1]==8
    field,logits,configuration=head(raw,pose,mass)
    f,l,c=other(projected,other_pose,other_mass,memory_is_projected=True)
    for actual,expected in ((f,field),(l,logits),(c.tokens,configuration.tokens)):
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-5)
    loss=lambda f,l,c:f.square().mean()+l.square().mean()+c.tokens.square().mean()
    a=torch.autograd.grad(loss(field,logits,configuration),(*head.parameters(),visual))
    b=torch.autograd.grad(loss(f,l,c),(*other.parameters(),other_visual))
    for actual,expected in zip(b,a,strict=True):torch.testing.assert_close(actual,expected,rtol=5e-4,atol=5e-5)


def test_compiled_geometry_repeated_calls_preserve_fields_and_gradients():
    """Training keeps several microbatch graphs alive until one backward."""
    torch.manual_seed(918)
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    head.neural_dtype=torch.float32
    compiled=deepcopy(head)
    compiled.forward=torch.compile(compiled.forward,backend='aot_eager',fullgraph=True,dynamic=True)
    inputs=[torch.randn(n,8,224,224,requires_grad=True) for n in (2,1)]
    other=[value.detach().clone().requires_grad_() for value in inputs]
    expected=[];actual=[]
    for x,y in zip(inputs,other,strict=True):
        pose=torch.eye(4).expand(len(x),-1,-1)
        mass=torch.rand(len(x),1,224,224)
        a=head(x,pose,mass,memory_is_projected=True)
        b=compiled(y,pose,mass,memory_is_projected=True)
        for first,second in zip((a[0],a[1],a[2].tokens),(b[0],b[1],b[2].tokens),strict=True):
            torch.testing.assert_close(first,second,rtol=3e-5,atol=1e-4)
            expected.append(first.square().mean());actual.append(second.square().mean())
        clearance=a[0][:,:1];padded=torch.nn.functional.pad(clearance,(1,1,1,1),mode='replicate')
        res=2*head.horizon/63
        torch.testing.assert_close(a[0][:,1:2],(padded[:,:,1:-1,2:]-padded[:,:,1:-1,:-2])/(2*res))
        torch.testing.assert_close(a[0][:,2:3],(padded[:,:,2:,1:-1]-padded[:,:,:-2,1:-1])/(2*res))
    a=torch.autograd.grad(sum(expected),(*head.parameters(),*inputs))
    b=torch.autograd.grad(sum(actual),(*compiled.parameters(),*other))
    for first,second in zip(a,b,strict=True):torch.testing.assert_close(first,second,rtol=5e-4,atol=5e-5)


def test_compiled_pixels_preserve_fp32_fusion_and_parameter_gradients():
    from kappanav.local_planner.model.encoders.depth import DepthObservationEncoder
    torch.manual_seed(919)
    encoder=DepthObservationEncoder(model_dim=32,frame_tokens_height=8,frame_tokens_width=12,
        dropout=0.,max_depth_m=5.,planning_horizon_m=3.6)
    other=deepcopy(encoder)
    other.encode_pixels=torch.compile(other.encode_pixels,backend='aot_eager',fullgraph=True,dynamic=True)
    expected=[];actual=[];inputs=[];copies=[]
    for size in (17,29):
        visual=torch.randn(size,32,requires_grad=True);copy=visual.detach().clone().requires_grad_()
        geometry=torch.randn(size,6);ages=torch.tensor([[0.],[.5],[1.5]])
        indices=torch.randint(3,(size,))
        a=encoder.output_norm(visual+encoder.geometry_projection(geometry)
            +torch.nn.functional.embedding(indices,encoder.time_projection(ages)))
        b=other.encode_pixels(copy,geometry,ages,indices)
        assert b.dtype==torch.float32
        torch.testing.assert_close(a,b)
        expected.append(a.square().mean());actual.append(b.square().mean());inputs.append(visual);copies.append(copy)
    names=('geometry_projection','time_projection','output_norm')
    params=lambda module:[p for name in names for p in getattr(module,name).parameters()]
    a=torch.autograd.grad(sum(expected),(*params(encoder),*inputs))
    b=torch.autograd.grad(sum(actual),(*params(other),*copies))
    for first,second in zip(a,b,strict=True):torch.testing.assert_close(first,second,rtol=5e-4,atol=5e-5)


def test_dense_context_is_independent_of_fine_history_and_preserves_parameter_shapes():
    from kappanav.local_planner.features.four_frame_history import READ_RADIUS_M
    heads=[FeatureGeometryDecoder(40,3.6,.167584539,width=8,context_radius_m=r) for r in (12.,8.,6.,3.6)]
    assert [head.raster_size for head in heads]==[224,160,128,64]
    assert FeatureGeometryDecoder(40,24*.15,.167584539,width=8,context_radius_m=3.6).raster_size==64
    assert FeatureGeometryDecoder(40,3.6000000000000005,.167584539,width=8,context_radius_m=3.6).raster_size==64
    assert READ_RADIUS_M==12.
    assert all(dict((name,p.shape) for name,p in head.named_parameters())
        ==dict((name,p.shape) for name,p in heads[0].named_parameters()) for head in heads)
    assert not torch.equal(heads[0].metric_position,heads[1].metric_position)


@pytest.mark.parametrize('microbatch',[1,2])
def test_online_replay_condition_path_matches_explicit_reference_and_gradients(microbatch):
    from kappanav.local_planner.features.conditions import encode_conditions, cat_conditions
    from kappanav.local_planner.features.local_features import append_fine_local
    torch.manual_seed(921)
    policy=nn.Module();policy.depth_encoder=SimpleNamespace(max_depth_m=5.)
    policy.condition_encoder=PolicyConditionEncoder(observation_frames=4,planning_horizon_m=3.6,model_dim=32)
    policy.condition_encoder.motion_encoder.neural_dtype=torch.float32
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8);head.neural_dtype=torch.float32
    actor=SimpleNamespace(policy=policy,head=head,config=SimpleNamespace(training=SimpleNamespace(geometry_batch_size=microbatch)))
    other=deepcopy(actor);plans=[]
    rng=np.random.default_rng(921)
    for n in (5,0):
        plans.append(dict(points=rng.uniform(-1,1,(n,3)).astype(np.float32),depth=np.full(n,2.,np.float32),
            keys=np.zeros((n,3),np.int64),origins=np.zeros((n,3),np.float32),pose=np.eye(4),clock=2.,times=np.ones(n),
            selection=dict(valid=np.ones(4,bool),poses=np.tile(np.eye(4),(4,1,1)),ages=np.arange(3,-1,-1).astype(np.float32))))
    visual=[torch.randn(len(p['points']),32,requires_grad=True) for p in plans]
    copied=[v.detach().clone().requires_grad_() for v in visual]
    goals=[torch.tensor([2.,1.]),torch.zeros(2)];present=[True,False]
    scalar=[encode_memory(policy,head,p,v,g,q) for p,v,g,q in zip(plans,visual,goals,present,strict=True)]
    expected=append_fine_local(actor,cat_conditions([c for c,_ in scalar]),plans,visual)
    actual,(field,logits)=encode_conditions(other,deepcopy(plans),copied,goals,present)
    assert torch.equal(field.detach(),actual.configuration_field) and not actual.configuration_field.requires_grad
    res=2*head.horizon/63
    metric_scale=field.new_tensor([1.,res,res,4*res,4*res,1.])[None,:,None,None]
    for f in fields(actual):
        if getattr(actual,f.name) is None:continue
        # Use the same metric-scaled field tolerance as the scalar/batched head oracle.
        scale=metric_scale if f.name=='configuration_field' else 1.
        torch.testing.assert_close(getattr(actual,f.name)*scale,getattr(expected,f.name)*scale,rtol=5e-5,atol=1e-4)
    old=expected.tokens.square().mean()+expected.configuration_field.square().mean()
    new=actual.tokens.square().mean()+actual.configuration_field.square().mean()
    a=torch.autograd.grad(old,(*policy.parameters(),*head.parameters(),*visual),allow_unused=True)
    b=torch.autograd.grad(new,(*other.policy.parameters(),*other.head.parameters(),*copied),allow_unused=True)
    for first,second in zip(a,b,strict=True):
        assert (first is None)==(second is None)
        if first is not None:torch.testing.assert_close(first,second,rtol=5e-4,atol=5e-5)
