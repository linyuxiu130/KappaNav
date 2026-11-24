import numpy as np
import pytest
import torch
from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
from kappanav.local_planner.data.boundary_batch import signed_distance_rows


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_mixed_boundary_maps_preserve_order_vertices_oob_and_gradient(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    grids=[]
    for i,cell in enumerate((.05,.1,.3)):
        field=np.ones((27+5*i,31+3*i),np.float32);field[5:15,7:16]=-.1
        grids.append(SourceConfigurationGrid(field,np.array([-1.+.1*i,-1.2]),cell,'canonical_world_xy'))
    query=SourceConfigurationSpaceQuery(tuple(grids));indices=[2,0,2,1,0]
    torch.manual_seed(929);world=torch.randn(5,33,2,device=device,dtype=dtype)*2
    for row,index in enumerate(indices):world[row,:3]=torch.as_tensor(grids[index].boundary_segments[:3,0],device=device,dtype=dtype)
    old=world.detach().requires_grad_();new=world.detach().clone().requires_grad_()
    expected=torch.stack([query.signed_distance(old[row],index) for row,index in enumerate(indices)])
    actual=signed_distance_rows(query,new,indices)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),new)[0],torch.autograd.grad(expected.sum(),old)[0],rtol=0,atol=0)
    # Reuse immutable packing with a different per-row map assignment.
    reverse=indices[::-1]
    repeated=signed_distance_rows(query,world,reverse)
    oracle=torch.stack([query.signed_distance(world[row],index) for row,index in enumerate(reverse)])
    torch.testing.assert_close(repeated,oracle,rtol=0,atol=0)
