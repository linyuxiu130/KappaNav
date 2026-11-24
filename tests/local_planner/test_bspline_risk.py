"""Canonical source-C-space GF agrees with an analytic wall,including gradients."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from kappanav.local_planner.data.bspline_replay import dense_curve, SAMPLES
from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
from kappanav.local_planner.training.bspline_objective import _source_geometry_risk


def test_source_risk_matches_wall_without_inflating_the_footprint_twice():
    signed=np.ones((6,6),np.float32);signed[4:]=-1.
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(signed,np.array([-1.5,-1.5]),.5,'canonical_world_xy'),))
    controls=torch.stack((torch.arange(1,8)*.12,torch.zeros(7)),-1).flatten()
    values=torch.stack((controls,controls*.5))[None].requires_grad_()
    paths,pad=dense_curve(values.flatten(0,1));paths=paths.reshape(1,-1,2).double()
    radius=.167584539;actor=SimpleNamespace(head=SimpleNamespace(radius=radius))
    records=[(dict(view=np.zeros(3),source_grid_index=0,teacher_values=torch.zeros(14)),None,None,SimpleNamespace(source_query=query))]
    actual=_source_geometry_risk(actor,values,paths,pad,records)
    # The wall at x=.5 is already a boundary for the robot centre.
    margin=.5-paths[...,0].reshape(1,2,SAMPLES)
    deficit=((pad.detach().reshape(1,2,1)-margin)/radius).clamp_min(0)
    expected=(4*torch.where(deficit<=1,deficit.square(),2*deficit-1).amax(-1)).mean()
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    a=torch.autograd.grad(actual,values,retain_graph=True)[0]
    b=torch.autograd.grad(expected,values)[0]
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    assert torch.isfinite(a).all() and a[0,0].abs().sum()>0 and not a[0,1].any()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_GF_reuses_boundary_search_with_entry_face_gradients(monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    from kappanav.local_planner.training import bspline_objective as objective
    signed=np.ones((24,24),np.float32);signed[10:13,8:16]=-1
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(signed,np.array([-1.2,-1.2]),.1),))
    controls=torch.stack((torch.arange(1,8)*.19,torch.zeros(7)),-1).flatten()
    values=torch.stack((controls,controls*.5))[None].to(device).requires_grad_()
    paths,pad=dense_curve(values.flatten(0,1));paths=paths.reshape(1,-1,2).double()
    world=paths+paths.new_tensor([-.8,0.])
    x=world[...,0].reshape(1,2,SAMPLES)
    # The wall spans [-.2,.1]; its nearest face flips at x=-.05.
    deficit=((pad.detach().reshape(1,2,1)-(-.2-x))/.167584539).clamp_min(0)
    error=deficit*(x<=-.05)
    expected=(4*torch.where(error<=1,error.square(),2*error-1).amax(-1)).mean()
    actor=SimpleNamespace(head=SimpleNamespace(radius=.167584539))
    records=[(dict(view=[-.8,0.,0.],source_grid_index=0,teacher_values=torch.zeros(14)),None,None,SimpleNamespace(source_query=query))]
    calls=[];original=objective.signed_distance_rows
    def counted(*args):
        calls.append(1);return original(*args)
    monkeypatch.setattr(objective,'signed_distance_rows',counted)
    actual=objective._source_geometry_risk(actor,values,paths,pad,records)
    assert len(calls)==1
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    actual_grad,=torch.autograd.grad(actual,values,retain_graph=True)
    expected_grad,=torch.autograd.grad(expected,values)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-6,atol=1e-6)
    with torch.no_grad():
        validation=objective._source_geometry_risk(actor,values,paths,pad,records)
    assert not validation.requires_grad
    torch.testing.assert_close(validation,actual,rtol=0,atol=0)
