import torch
import pytest
import numpy as np
from scipy.spatial import cKDTree
from kappanav.local_planner.features.temporal_geometry import FeatureGeometryDecoder, geometry_reconstruction_loss


@pytest.mark.parametrize('shape',[(1,1),(3,5),(8,10)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_bilinear_2x_values_and_transpose_match_native_at_edges(shape,dtype):
    from kappanav.local_planner.features.temporal_geometry import _Bilinear2x
    torch.manual_seed(904)
    value=torch.randn(2,3,*shape,dtype=dtype).to(memory_format=torch.channels_last).requires_grad_()
    size=tuple(2*n for n in shape)
    expected=torch.nn.functional.interpolate(value,size=size,mode='bilinear',align_corners=False)
    actual=_Bilinear2x.apply(value,size)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    probe=torch.randn_like(actual)
    actual_grad=torch.autograd.grad(actual,value,probe)[0]
    expected_grad=torch.autograd.grad(expected,value,probe)[0]
    tolerance=1e-12 if dtype==torch.float64 else 1e-6
    torch.testing.assert_close(actual_grad,expected_grad,rtol=tolerance,atol=tolerance)


def test_bilinear_2x_bf16_gradient_accumulates_before_rounding_and_supports_double_backward():
    from kappanav.local_planner.features.temporal_geometry import _Bilinear2x
    torch.manual_seed(905)
    value=torch.randn(2,3,5,7,dtype=torch.bfloat16,requires_grad=True)
    actual=_Bilinear2x.apply(value,(10,14));probe=torch.randn_like(actual)
    gradient=torch.autograd.grad(actual,value,probe)[0]
    precise=value.detach().float().requires_grad_()
    expected=torch.nn.functional.interpolate(precise,size=(10,14),mode='bilinear',align_corners=False)
    expected_grad=torch.autograd.grad(expected,precise,probe.float())[0].to(value.dtype)
    torch.testing.assert_close(gradient,expected_grad,rtol=0,atol=0)
    double=torch.randn(1,1,2,3,dtype=torch.float64,requires_grad=True)
    fn=lambda x:_Bilinear2x.apply(x,(4,6))
    assert torch.autograd.gradcheck(fn,(double,))
    assert torch.autograd.gradgradcheck(fn,(double,))


@pytest.mark.parametrize('clearance,free_target',[(.01,1.),(.05,1.),(.05,0.),(-.05,0.)])
def test_free_evidence_cannot_move_an_already_correct_metric_distance(clearance,free_target):
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    with torch.no_grad():
        head.readout.weight.zero_();head.readout.bias.copy_(torch.tensor([clearance/3.6,8.]))
    field,logits,_=head(torch.zeros(1,40,head.raster_size,head.raster_size),
        torch.eye(4,dtype=torch.float64)[None],torch.zeros(1,1,head.raster_size,head.raster_size))
    target=field.detach().clone();target[:,5]=free_target
    gradient=torch.autograd.grad(geometry_reconstruction_loss(field,logits,target),head.readout.bias)[0]
    torch.testing.assert_close(gradient[0],torch.zeros_like(gradient[0]),atol=1e-6,rtol=0)
    assert gradient[1]*(1-2*free_target)>0


def test_geometry_read_and_direct_supervision_reach_both_outputs_without_hard_gates():
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    with torch.no_grad():
        head.readout.weight.zero_();head.readout.bias.copy_(torch.tensor([.1,-.5]))
    field,logits,configuration=head(torch.zeros(1,40,head.raster_size,head.raster_size),torch.eye(4,dtype=torch.float64)[None],torch.zeros(1,1,head.raster_size,head.raster_size))
    assert logits.shape==(1,2,64,64)
    grad=torch.autograd.grad(field[:,[0,3,5]].sum(),logits,retain_graph=True)[0]
    assert (grad.abs().sum((0,2,3))>0).all()
    torch.testing.assert_close(field[:,3],field[:,4]+(1-field[:,4])*field[:,5],rtol=0,atol=0)
    assert (field[:,3]<=1).all() and (field[:,5]>=0).all()
    target=field.detach().clone();target[:,0]-=.1;target[:,5]=0
    loss=geometry_reconstruction_loss(field,logits,target)
    gradient=torch.autograd.grad(loss,head.readout.bias)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs()>0).all()
    changed=target.clone();changed[:,0]-=.1
    assert geometry_reconstruction_loss(field,logits,changed)>loss


def test_unknown_is_not_supervised_as_free():
    logits=torch.zeros(1,2,64,64,requires_grad=True)
    target=torch.zeros(1,6,64,64);target[:,0]=2.
    field=target.clone();field[:,0]=logits[:,0]*3.6
    loss=geometry_reconstruction_loss(field,logits,target)
    gradient=torch.autograd.grad(loss,logits)[0]
    assert (gradient[:,1]>0).all()  # Gradient descent reduces unknown-space free probability.
    assert (gradient[:,0]<0).all()  # Distance remains independently supervised.


def test_free_supervision_is_stationary_when_consumed_probability_matches_target():
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    with torch.no_grad():
        head.readout.weight.zero_();head.readout.bias.zero_()
    field,logits,_=head(torch.zeros(1,40,head.raster_size,head.raster_size),
        torch.eye(4,dtype=torch.float64)[None],torch.zeros(1,1,head.raster_size,head.raster_size))
    target=field.detach().clone()
    torch.testing.assert_close(target[:,5],torch.full_like(target[:,5],.5))
    loss=geometry_reconstruction_loss(field,logits,target)
    gradient=torch.autograd.grad(loss,logits)[0]
    torch.testing.assert_close(gradient,torch.zeros_like(gradient),atol=1e-7,rtol=0)


def test_free_loss_keeps_finite_gradient_for_saturated_wrong_predictions():
    logits=torch.zeros(2,2,64,64,requires_grad=True)
    with torch.no_grad():logits[0,1]=1000.;logits[1,1]=-1000.
    target=torch.zeros(2,6,64,64);target[1,5]=1.
    field=target.clone();field[:,0]=logits[:,0]
    gradient=torch.autograd.grad(geometry_reconstruction_loss(field,logits,target),logits)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient[0,1]>0).all() and (gradient[1,1]<0).all()
    assert not gradient[:,0].any()


def test_geometry_labels_match_gf_hits_error_and_radius_without_certifying_unknown():
    from kappanav.local_planner.data.observed_geometry import field_from_observed_hits
    horizon=3.6;radius=.167584539;error=.013
    tree=cKDTree([[.137,.219],[5.,0.]])
    field=field_from_observed_hits(tree,error,np.zeros((64,64),bool),horizon,radius)
    axis=np.arange(64)*(2*horizon/63)-horizon
    x,y=np.meshgrid(axis,axis,indexing='xy');points=np.stack((x,y),-1)
    expected=np.linalg.norm(points[:,:,None]-tree.data,axis=-1).min(-1)-error-radius
    np.testing.assert_allclose(field[0],expected,atol=3e-7,rtol=0)
    assert (field[0]>0).any() and not field[5].any()
    torch.testing.assert_close(field[1,:,1:-1],(field[0,:,2:]-field[0,:,:-2])/(4*horizon/63))
    known=field_from_observed_hits(tree,error,np.ones((64,64),bool),horizon,radius)
    assert not known[5][known[0]<=0].any()
    assert (known[5][known[0]>0]==1).all()


def test_unified_tokens_read_pregeometry_features_and_keep_far_memory():
    from reference_implementations import splat_memory
    head=FeatureGeometryDecoder(40,3.6,.167584539,width=8)
    points=torch.tensor([[[11.9,0.,0.]]]);features=torch.randn(1,1,40,requires_grad=True)
    raster,mass=splat_memory(features,points,torch.ones(1,1,dtype=torch.bool),3.6,64,head.raster_padding)
    torch.testing.assert_close(mass.sum(),torch.tensor(1.))
    size=head.raster_size
    field,logits,configuration=head(raster.transpose(1,2).reshape(1,40,size,size),torch.eye(4,dtype=torch.float64)[None],mass.transpose(1,2).reshape(1,1,size,size))
    assert configuration.tokens.shape==(1,256,32)
    assert configuration.observed_fraction.max()>0
    assert configuration.metric_position[...,:2].abs().max()>=12.
    source_gradient,geometry_gradient=torch.autograd.grad(configuration.tokens.square().mean(),(features,head.readout.weight),allow_unused=True)
    assert source_gradient.abs().sum()>0 and geometry_gradient is None


def test_empty_obstacle_observation_has_no_distance_or_distance_gradient_target():
    from kappanav.local_planner.data.observed_geometry import field_from_observed_hits
    target=field_from_observed_hits(cKDTree(np.empty((0,2))),.01,np.ones((64,64),bool),3.6,.167584539)[None]
    assert torch.isfinite(target).all() and not target[:,4].any() and target[:,5].all()
    field=target.clone().requires_grad_();logits=torch.zeros(1,2,64,64,requires_grad=True)
    loss=geometry_reconstruction_loss(field,logits,target,distance_available=[False])
    df,dl=torch.autograd.grad(loss,(field,logits))
    assert torch.isfinite(loss) and not df.any() and (dl[:,1]<0).all()
def test_boundary_broad_phase_matches_exhaustive_distances_and_gradients():
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    rng=np.random.default_rng(819)
    free=rng.random((71,83))>.3
    grid=SourceConfigurationGrid(np.where(free,1.,-1.).astype(np.float32),np.array([-1.2,-.7]),.05)
    points=rng.uniform([.1,.1],[.3,.3],(100,2))
    segments=torch.tensor(grid.boundary_segments);a=segments[:,0];delta=segments[:,1]-a
    world=torch.tensor(points,requires_grad=True)
    fraction=(((world[:,None]-a)*delta).sum(-1)/delta.square().sum(-1)).clamp(0,1)
    expected=(world[:,None]-a-fraction[...,None]*delta).norm(dim=-1).amin(-1)
    actual=SourceConfigurationSpaceQuery((grid,)).signed_distance(world,0).abs()
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-14)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),world)[0],torch.autograd.grad(expected.sum(),world)[0],rtol=0,atol=1e-12)


def test_merged_boundaries_match_cell_edges_with_holes_and_outside_points():
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    free=np.ones((60,70),bool);free[15:35,20:45]=False;free[42:50,10:15]=False
    grid=SourceConfigurationGrid(np.where(free,1.,-1.).astype(np.float32),np.array([-.3,.7]),.05)
    padded=np.pad(free,1)
    vertical=np.argwhere(padded[1:,1:-1]!=padded[:-1,1:-1])
    horizontal=np.argwhere(padded[1:-1,1:]!=padded[1:-1,:-1])
    starts=grid.origin_xy+np.concatenate((vertical,horizontal))*.05
    delta=np.concatenate((np.tile([0.,.05],(len(vertical),1)),np.tile([.05,0.],(len(horizontal),1))))
    assert len(grid.boundary_segments)==12 < len(starts)/20
    world=torch.tensor(np.random.default_rng(217).uniform([-.8,.2],[3.2,4.7],(150,2)),requires_grad=True)
    start=torch.tensor(starts);direction=torch.tensor(delta)
    fraction=(((world[:,None]-start)*direction).sum(-1)/direction.square().sum(-1)).clamp(0,1)
    expected=(world[:,None]-start-fraction[...,None]*direction).norm(dim=-1).amin(-1)
    actual=SourceConfigurationSpaceQuery((grid,)).signed_distance(world,0)
    torch.testing.assert_close(actual.abs(),expected,rtol=0,atol=2e-15)
    torch.testing.assert_close(torch.autograd.grad(actual.abs().sum(),world)[0],
        torch.autograd.grad(expected.sum(),world)[0],rtol=0,atol=1e-12)
    # Zero-distance grid corners and boundary contacts retain closed-cell safety.
    corners=torch.tensor(grid.origin_xy+np.array([[15,20],[35,45],[0,0]])*.05)
    torch.testing.assert_close(SourceConfigurationSpaceQuery((grid,)).signed_distance(corners,0),
        torch.zeros(3,dtype=torch.float64),rtol=0,atol=1e-15)
