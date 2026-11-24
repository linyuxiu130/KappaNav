"""Surface loss over every supervised pixel.

A single-class label is plain cross-entropy. A set label says only what the pixel is not: the other classes are pushed
down and the set's members raised equally, so their order is left to the sources that know it. (The usual
-log sum p raises each member in proportion to its probability, and the largest takes the whole set: FLAIR's paving
became 92% vehicle that way.) As in cross-entropy, a pixel's gradient sums to zero over the classes. The loss is
normalised by the global (all-rank) count of supervised pixels.
"""
import torch
import torch.distributed as dist
from torch import nn

from kappanav.global_planner.taxonomy import CLASSES


def allowed(labels):
    """Bitmask labels [B,H,W] -> bool [B,K,H,W] of the classes each pixel's set allows."""
    shifts = torch.arange(len(CLASSES), device=labels.device)
    return (labels.unsqueeze(1) >> shifts[None, :, None, None]) & 1 == 1


def global_count(count):
    count = count.to(torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count)
        count /= dist.get_world_size()
    return count


class SurfaceCriterion(nn.Module):
    def forward(self, logits, labels):
        logits, inside = logits.float(), allowed(labels)
        supervised = labels != 0
        # Members keep their values but share one gradient: that of their mean. A single class is its own mean.
        mean = logits.masked_fill(~inside, 0.).sum(1, keepdim=True) / inside.sum(1, keepdim=True).clamp_min(1)
        logits = torch.where(inside, logits.detach() + mean - mean.detach(), logits)
        nll = torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~inside, -torch.inf), 1)
        nll = torch.where(supervised, nll, torch.zeros_like(nll))
        surface = nll.sum() / global_count(supervised.sum()).clamp_min(1.)
        return surface
