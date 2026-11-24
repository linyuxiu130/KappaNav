"""Fused FP64 selection matches a dense oracle, including stable ties and block tails."""
import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA boundary backend')


def dense_nearest(world,a,delta,length2,valid):
    difference=world[:,:,None]-a[None,None]
    fraction=((difference*delta).sum(-1)/length2).clamp(0,1)
    distance=(difference-fraction[...,None]*delta).square().sum(-1)
    return distance.masked_fill(~valid[:,None],torch.inf).argmin(-1)


@pytest.mark.parametrize('segments',[1,127,128,129,257,1025])
def test_boundary_compaction_stable_ties_sparse_rows_and_tail(segments):
    from kappanav.local_planner.data._boundary_cuda import nearest_valid_boundary
    torch.manual_seed(739)
    world=torch.randn(3,65,2,device='cuda',dtype=torch.float64)
    a=torch.zeros(segments,2,device='cuda',dtype=torch.float64)
    delta=torch.zeros_like(a);delta[:,0]=1
    length2=delta.square().sum(-1)
    valid=torch.zeros(3,segments,device='cuda',dtype=torch.bool)
    valid[0,-1]=True
    valid[1,0]=valid[1,-1]=True
    valid[2,segments//2]=valid[2,-1]=True
    expected=dense_nearest(world,a,delta,length2,valid)
    actual=nearest_valid_boundary(world,a,delta,length2,valid)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert actual[0].eq(segments-1).all() and actual[1].eq(0).all()


def test_boundary_compaction_handles_strided_world_and_unaligned_segment_arrays():
    from kappanav.local_planner.data._boundary_cuda import nearest_valid_boundary
    torch.manual_seed(740)
    world=torch.randn(3,2,257,device='cuda',dtype=torch.float64).transpose(1,2)
    segments=torch.randn(261,2,2,device='cuda',dtype=torch.float64)
    a=segments[:,0];delta=segments[:,1]-segments[:,0];length2=delta.square().sum(-1)
    valid=torch.rand(3,261,device='cuda')>.7;valid[:,0]=True
    assert not world.is_contiguous() and not a.is_contiguous()
    torch.testing.assert_close(nearest_valid_boundary(world,a,delta,length2,valid),dense_nearest(world,a,delta,length2,valid),rtol=0,atol=0)
