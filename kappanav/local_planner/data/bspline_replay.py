"""Observed-route B-spline labels and native joint losses; never timed-control labels."""
from functools import lru_cache

import numpy as np
import torch

from kappanav.local_planner.model.trajectory.basis import bspline_basis_and_first_derivative, _open_uniform_knots

SAMPLES = 257


@lru_cache(maxsize=8)
def dense_basis(device):
    """Fixed basis, held once per device: a per-call host copy would synchronize the stream."""
    basis, _ = bspline_basis_and_first_derivative(8, 3, SAMPLES)
    knots = _open_uniform_knots(8, 3)
    return basis.to(device), (3/(knots[4:11]-knots[1:8])).to(device)


def dense_curve(values):
    """Exact fixed cubic basis; nearest-sample distance bounded by sup|C'|/(2N)."""
    controls = torch.cat((values.new_zeros(len(values),1,2), values.float().reshape(-1,7,2)), 1)
    basis, factors = dense_basis(values.device)
    path = torch.einsum('pc,bcd->bpd', basis, controls)
    derivative_controls = torch.diff(controls,dim=1)*factors[None,:,None]
    pad = derivative_controls.norm(dim=-1).amax(-1)/(2*(SAMPLES-1))
    return path, pad


def resample_route(route, count=64, horizon=3.6):
    lengths = np.linalg.norm(np.diff(route,axis=0),axis=1)
    route = route[np.r_[True,lengths>1e-8]]
    if len(route)<2:return None
    arc = np.r_[0.,np.linalg.norm(np.diff(route,axis=0),axis=1).cumsum()]
    at = np.linspace(0,min(horizon,arc[-1]),count)
    return np.column_stack([np.interp(at,arc,route[:,i]) for i in range(2)])


