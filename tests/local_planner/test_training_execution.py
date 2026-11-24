"""Aligned/compiled training preserves attention, gradients and monitoring."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from kappanav.local_planner.model.network.blocks import ConditionalTrajectoryBlock, _GeometryCast
from kappanav.local_planner.training.bspline_objective import _aligned_condition
from kappanav.local_planner.training.execution import _gradient_forward, register_training_bias_kernels
from kappanav.local_planner.model.types import ConditionFeatures

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def condition():
    n=19
    tokens=torch.randn(2,n,384,device='cuda',requires_grad=True)
    valid=torch.ones(2,n+5,device='cuda',dtype=torch.bool)[:,:n];valid[:,-2:]=False
    return ConditionFeatures(tokens,valid,torch.randn(2,n,3,device='cuda'),
        torch.ones(2,n,device='cuda'),torch.zeros(2,n,device='cuda'),
        torch.zeros(2,n,device='cuda',dtype=torch.bool),torch.ones(2,2,device='cuda'),
        torch.ones(2,1,device='cuda'),torch.zeros(2,6,64,64,device='cuda'))


def test_alignment_preserves_valid_inputs_and_adds_only_masked_keys():
    source=condition();aligned=_aligned_condition(source)
    assert aligned.tokens.shape==(2,32,384)
    assert aligned.token_valid.is_contiguous() and not aligned.token_valid[:,19:].any()
    torch.testing.assert_close(aligned.tokens[:,:19],source.tokens,rtol=0,atol=0)
    assert aligned.configuration_field is source.configuration_field
    assert aligned.terminal_goal is source.terminal_goal
    with torch.no_grad():assert _aligned_condition(source) is source


@pytest.mark.parametrize('recompute', [False, True])
def test_compiled_euler_block_preserves_gradients_rng_and_monitoring(monkeypatch, recompute):
    from kappanav.local_planner.model.network import blocks
    monkeypatch.setattr(blocks,'_BIAS_CHUNK_BYTES',0)
    if torch.cuda.get_device_capability() in ((8,9),(12,0)):register_training_bias_kernels()
    torch.manual_seed(404)
    reference=ConditionalTrajectoryBlock(384,8,0.).cuda().eval()
    if recompute:reference.cross_attention.saved_bias_bytes=0
    candidate=deepcopy(reference)
    source=condition();aligned=_aligned_condition(source)
    query=torch.randn(4,7,384,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    geometry=torch.randn(4,7,19,7,device='cuda',requires_grad=True)
    padded=F.pad(geometry,(0,0,0,13))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        expected=reference(query,reference.project_condition(source),geometry)
    inputs=(query,source.tokens,geometry)
    expected_grad=torch.autograd.grad(expected.float().square().sum(),(*inputs,*reference.parameters()))
    candidate.forward=_gradient_forward(candidate.forward)
    rng=torch.cuda.get_rng_state()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=candidate(query,candidate.project_condition(aligned),_GeometryCast.apply(padded,padded.detach().to(query.dtype)))
    assert torch.equal(rng,torch.cuda.get_rng_state())
    actual_grad=torch.autograd.grad(actual.float().square().sum(),(*inputs,*candidate.parameters()))
    torch.testing.assert_close(actual,expected,rtol=.01,atol=.003)
    q32=query.detach().float().requires_grad_();t32=source.tokens.detach().requires_grad_()
    p32=geometry.detach().requires_grad_();c32=replace(source,tokens=t32)
    out32=reference(q32,reference.project_condition(c32),p32)
    grad32=torch.autograd.grad(out32.square().sum(),(q32,t32,p32,*reference.parameters()))
    for a,b,oracle in zip(actual_grad,expected_grad,grad32,strict=True):
        assert torch.isfinite(a).all()
        # Near-zero softmax-shift bias gradients make coordinate-relative BF16 errors misleading.
        baseline_error=(b.float()-oracle).norm()
        assert (a.float()-oracle).norm()<=2*baseline_error+1e-4*oracle.norm()+1e-5
    for a,b in zip(actual_grad[:len(inputs)],expected_grad[:len(inputs)],strict=True):
        assert (a.float()-b.float()).norm()/b.float().norm()<.01
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        monitored=candidate(query,candidate.project_condition(source),geometry)
        original=reference(query,reference.project_condition(source),geometry)
    torch.testing.assert_close(monitored,original,rtol=0,atol=0)


@pytest.mark.parametrize(('capability','bucket'),[((12,0),512),((8,9),None)])
def test_training_ray_bucket_is_limited_to_sm120(monkeypatch,capability,bucket):
    from types import SimpleNamespace
    from kappanav.local_planner.features import history_fusion
    from kappanav.local_planner.training import execution

    block=SimpleNamespace(cross_attention=SimpleNamespace(saved_bias_bytes=0),forward=lambda *args:None)
    policy=SimpleNamespace(parameters=lambda:iter([torch.nn.Parameter(torch.ones(1))]),
        trajectory_decoder=SimpleNamespace(blocks=[block]),trajectory_evaluator=SimpleNamespace(blocks=[block]))
    actor=SimpleNamespace(policy=policy)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:capability)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(total_memory=80<<30))
    monkeypatch.setattr(execution,'register_training_bias_kernels',lambda:None)
    monkeypatch.setattr(history_fusion,'source_features_batch',history_fusion.source_features_batch)
    from kappanav.local_planner.model.network import blocks
    monkeypatch.setattr(blocks,'_BIAS_CHUNK_BYTES',blocks._BIAS_CHUNK_BYTES)
    settings=execution.configure_training_execution(actor)
    assert settings['source_sampling_bucket']==bucket
    assert history_fusion.source_features_batch.keywords['bucket']==bucket
