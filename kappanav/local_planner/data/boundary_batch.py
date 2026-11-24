"""Exact GF queries across immutable source maps in one GPU batch."""
import numpy as np
import torch
from kappanav.local_planner.data.privileged import _closed_grid_cells
from kappanav.local_planner.features.staging import stage


def signed_distance_rows(query,world,grid_indices):
    if not world.is_cuda:
        result=[None]*len(world)
        for index in dict.fromkeys(grid_indices):
            rows=[i for i,g in enumerate(grid_indices) if g==index]
            distances=query.signed_distance_batch(world[rows],index)
            for i,value in zip(rows,distances,strict=True):result[i]=value
        return torch.stack(result)
    from kappanav.local_planner.data._boundary_cuda import nearest_valid_boundary
    cache_key=(world.device,'all_maps',world.dtype)
    if cache_key not in query._boundaries:
        segments=[grid.boundary_segments for grid in query.grids]
        counts=np.asarray([len(s) for s in segments],dtype=np.int64);width=int(counts.max())
        starts=np.zeros((len(segments),width,2),dtype=np.float64)
        directions=np.zeros_like(starts);lengths=np.ones((len(segments),width),dtype=np.float64)
        for i,s in enumerate(segments):
            starts[i,:len(s)]=s[:,0];directions[i,:len(s)]=s[:,1]-s[:,0]
            lengths[i,:len(s)]=np.square(directions[i,:len(s)]).sum(-1)
        # Match Tensor / Python-scalar division's reciprocal in the input dtype.
        inverse=torch.stack([torch.ones((),dtype=world.dtype)/g.cell_size_m for g in query.grids])
        query._boundaries[cache_key]=tuple(torch.as_tensor(v,device=world.device) for v in (starts,directions,lengths,counts,inverse))
    starts,directions,lengths,counts,inverse=query._boundaries[cache_key]
    ids=stage([torch.as_tensor(grid_indices,dtype=torch.long)],world.device)[0]
    with torch.no_grad():
        a=starts.index_select(0,ids);delta=directions.index_select(0,ids);length2=lengths.index_select(0,ids)
        center=(world.detach().amin(1)+world.detach().amax(1))/2
        reach=(world.detach()-center[:,None]).norm(dim=-1).amax(1)
        fraction=(((center[:,None]-a)*delta).sum(-1)/length2).clamp(0,1)
        squared=(center[:,None]-a-fraction[...,None]*delta).square().sum(-1)
        present=torch.arange(a.shape[1],device=world.device)[None]<counts[ids,None]
        squared=squared.masked_fill(~present,torch.inf)
        upper=squared.amin(-1).sqrt()+2*reach+1e-12
        valid=(squared<=upper[:,None].square())&present
        chosen=nearest_valid_boundary(world.detach(),a,delta,length2,valid)
        selected=chosen[:,:,None].expand(-1,-1,2)
        start=a.gather(1,selected);direction=delta.gather(1,selected);denominator=length2.gather(1,chosen)
        values,origins,_,shapes,offsets,_=query._atlas(world.device)
        coordinates=(world.detach()-origins[ids,None].to(world.dtype))*inverse[ids,None,None]
        cells=_closed_grid_cells(coordinates);shape=shapes[ids,None,None]
        inside=(cells>=0).all(-1)&(cells<shape).all(-1)
        address=offsets[ids,None,None]+cells[...,0]*shape[...,1]+cells[...,1]
        navigable=(inside&(values[torch.where(inside,address,0)]>=0)).all(-1)
    fraction=(((world-start)*direction).sum(-1)/denominator).clamp(0,1)
    distance=(world-start-fraction[...,None]*direction).norm(dim=-1)
    return torch.where(navigable,distance,-distance)

