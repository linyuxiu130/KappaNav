"""Calibrated memory splatting and metric position encoding."""

import math

import torch
from torch import Tensor


def memory_splat_stencil(points,valid,planning_horizon_m,grid_size,padding):
    """Compute immutable calibrated destinations, weights and mass once."""
    batch=points.shape[0]
    normalized = points[..., :2].float() * (1.0 / planning_horizon_m)
    grid = grid_size
    stride = 64 / grid
    size = grid+2*padding
    coordinate = ((normalized + 1.0) * 0.5 * 63 - (stride-1)/2) / stride+padding
    coordinate = coordinate.clamp(0, size - 1)
    inside = valid & (normalized.abs() <= 1.0+padding*stride*2/63).all(dim=-1)
    x0 = coordinate[..., 0].floor().clamp(0, size - 1).long()
    y0 = coordinate[..., 1].floor().clamp(0, size - 1).long()
    x1 = (x0 + 1).clamp_max(size - 1)
    y1 = (y0 + 1).clamp_max(size - 1)
    wx, wy = coordinate[..., 0] - x0, coordinate[..., 1] - y0
    entries=[]
    mass=coordinate.new_zeros(batch,size**2,1)
    for x, y, weight in (
        (x0, y0, (1 - wx) * (1 - wy)),
        (x1, y0, wx * (1 - wy)),
        (x0, y1, (1 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        weight = (weight * inside).float()[..., None]
        index = (y * size + x)[..., None]
        mass.scatter_add_(1, index, weight)
        entries.append((index,weight))
    return entries,mass,size


def static_splat_segments(stencil):
    """Sort immutable CPU contributions once, retaining corner/source order."""
    entries,mass,_=stencil
    indices=torch.cat([index.flatten() for index,weight in entries])
    normalized=torch.stack([weight/mass.gather(1,index).clamp_min(1e-6) for index,weight in entries])
    weights=normalized.flatten()
    order=indices.argsort(stable=True)
    cells,lengths=torch.unique_consecutive(indices[order],return_counts=True)
    sources=torch.arange(entries[0][0].shape[1]).repeat(4)[order]
    return sources,weights[order],cells,lengths,normalized


class _StaticMemorySplat(torch.autograd.Function):
    """Normalized raster with fixed calibration; only features need gradients."""

    @staticmethod
    def forward(ctx,tokens,sources,weights,cells,lengths,corner_indices,corner_weights,size):
        batch,_,dimension=tokens.shape
        raster=tokens.new_zeros(batch*size*size,dimension)
        if lengths.numel():
            contributions=tokens.flatten(0,1).index_select(0,sources)*weights[:,None]
            reduced=torch.segment_reduce(contributions,reduce='sum',lengths=lengths,unsafe=True)
            raster.index_copy_(0,cells,reduced)
        ctx.save_for_backward(corner_indices,corner_weights)
        return raster.reshape(batch,size*size,dimension)

    @staticmethod
    def backward(ctx,gradient):
        indices,weights=ctx.saved_tensors
        result=None
        for index,weight in zip(indices,weights,strict=True):
            value=gradient.gather(1,index.expand(-1,-1,gradient.shape[-1]))*weight
            result=value if result is None else result+value
        return result,None,None,None,None,None,None,None


def metric_grid(
    model_dim: int,
    grid_size: int,
    planning_horizon_m: float,
) -> tuple[Tensor, Tensor]:
    axis = ((torch.arange(grid_size) * 4 + 1.5) / 63 * 2 - 1) * planning_horizon_m
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    position = torch.stack((x, y, torch.zeros_like(x)), dim=-1).reshape(
        1, grid_size**2, 3
    )
    quarter = model_dim // 4
    frequency = torch.exp(
        torch.arange(quarter, dtype=torch.float32)
        * (-math.log(10_000.0) / max(quarter - 1, 1))
    )
    x_phase = math.pi * x[..., None] / planning_horizon_m * frequency
    y_phase = math.pi * y[..., None] / planning_horizon_m * frequency
    encoding = torch.cat(
        (x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()), dim=-1
    ).reshape(1, grid_size**2, model_dim)
    return position, encoding
