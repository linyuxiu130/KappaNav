from dataclasses import replace
import torch
import numpy as np
from kappanav.local_planner.features.staging import stage


def fine_local_inputs(plans,sizes):
    """Host side of append_fine_local for plans contributing `sizes` fused cells."""
    count=max(sizes)
    indices=np.zeros((len(sizes),count),dtype=np.int64)
    offset=1
    for row,size in enumerate(sizes):
        indices[row,:size]=np.arange(offset,offset+size);offset+=size
    def pad(value):return torch.nn.functional.pad(value,(0,0,0,count-len(value)))
    return [torch.stack([pad(torch.as_tensor(p['points'])) for p in plans]),
        torch.stack([torch.nn.functional.pad(torch.as_tensor(p['clock']-p['times'],dtype=torch.float32),
        (0,count-size)) for p,size in zip(plans,sizes,strict=True)]),torch.from_numpy(indices)]


def append_fine_local(actor,condition,plans,visuals,*,host=None):
    """Expose existing four-frame cell features before BEV pooling; add no encoder."""
    device=condition.tokens.device
    if host is None:host=stage(fine_local_inputs(plans,[len(v) for v in visuals]),device)
    normalized=actor.policy.condition_encoder.memory_norm(torch.cat(visuals))
    positions,ages,indices=host
    tokens=torch.nn.functional.embedding(indices,torch.nn.functional.pad(normalized,(0,0,1,0)),padding_idx=0)
    valid=indices!=0
    updates=dict(tokens=tokens,token_valid=valid,
        metric_position=positions,surface_hit=valid.float(),frame_age=ages,motion_token=torch.zeros_like(valid))
    return replace(condition,**{name:torch.cat((getattr(condition,name),value),1) for name,value in updates.items()})
