"""Batch host inputs into one pinned transfer per dtype.

A copy from pageable host memory synchronizes the stream before it starts, so
per-row copies make the CPU wait for the GPU and the GPU wait for the CPU.
"""
import torch
from torch.utils import _pytree


class Pinned:
    """Host tensors packed into one pinned buffer per dtype, ready for asynchronous transfers.

    Packing may run on a loader thread; `to(device)` then issues one non-blocking copy per buffer and returns
    device views with the original shapes and values.
    """
    __slots__ = ('buffers', 'layout')

    def __init__(self, tensors):
        tensors = list(tensors)
        groups = {}
        for index, tensor in enumerate(tensors):
            groups.setdefault(tensor.dtype, []).append(index)
        self.buffers = {};self.layout = [None]*len(tensors)
        for dtype, indices in groups.items():
            flat = torch.empty(sum(tensors[i].numel() for i in indices), dtype=dtype, pin_memory=torch.cuda.is_available())
            if indices:torch.cat([tensors[i].reshape(-1) for i in indices], out=flat)
            self.buffers[dtype] = flat;offset = 0
            for i in indices:
                self.layout[i] = (dtype, offset, tensors[i].shape);offset += tensors[i].numel()

    def to(self, device):
        moved = {dtype:flat.to(device, non_blocking=True) for dtype, flat in self.buffers.items()}
        return [moved[dtype][offset:offset+shape.numel()].view(shape) for dtype, offset, shape in self.layout]


def stage(tensors, device):
    """Return `tensors` on `device` (same shapes and values), one transfer per dtype."""
    out = list(tensors)
    if torch.device(device).type == 'cpu':
        return out
    host = [i for i, tensor in enumerate(out) if tensor.device.type == 'cpu']
    for i, moved in zip(host, Pinned(out[i] for i in host).to(device), strict=True):
        out[i] = moved
    return out


class HostTree:
    """A nested structure of host inputs whose CPU tensors travel in one pinned transfer per dtype.

    Leaves wrapped in `Keep` stay on the host (for host-side logic); every other CPU tensor is moved.
    """
    __slots__ = ('spec', 'leaves', 'moved', 'pinned')

    def __init__(self, tree):
        self.leaves, self.spec = _pytree.tree_flatten(tree)
        self.moved = [i for i, leaf in enumerate(self.leaves) if isinstance(leaf, torch.Tensor) and leaf.device.type == 'cpu']
        self.pinned = Pinned(self.leaves[i] for i in self.moved)

    def to(self, device):
        leaves = list(self.leaves)
        if torch.device(device).type != 'cpu':
            for i, value in zip(self.moved, self.pinned.to(device), strict=True):leaves[i] = value
        return _pytree.tree_unflatten([leaf.value if isinstance(leaf, Keep) else leaf for leaf in leaves], self.spec)


class Keep:
    """A host-only leaf of a HostTree."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value
