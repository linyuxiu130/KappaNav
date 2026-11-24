"""Current-weight four-frame encoding shared by data preparation and training."""
from dataclasses import fields
import torch
from kappanav.local_planner.model.types import ConditionFeatures
from kappanav.local_planner.features.history_fusion import fuse_sources_batch, encode_memories, source_groups
from kappanav.local_planner.features.local_features import append_fine_local
from kappanav.local_planner.features.staging import HostTree


def live_conditions(actor,records,feature_maps,*,host=None):
    """Fuse this step's writes together, then bound geometry-network activations."""
    maps=[];widths=[];plans=[r[0]['history_plan'] for r in records]
    for item,raw,*_ in records:
        ids=[int(index) for index,_ in source_groups(item['history_plan'])]
        maps.append({index:feature_maps[item['raw_sha256'],index] for index in ids})
        widths.append({index:raw['depths'][index].shape[-1] for index in ids})
    visuals=fuse_sources_batch(actor.policy,plans,maps,widths,**({} if host is None else dict(host=host['fusion'])))
    return encode_conditions(actor,plans,visuals,[r[0]['goal'] for r in records],
        [r[0]['goal_present'] for r in records],**({} if host is None else dict(host=host['conditions'])))


def condition_inputs(actor,records,*,bucket=None):
    """Host side of live_conditions for replay records, with feature-map shapes known from the image size: the
    backbone and the calibrated pixel sampler both use a stride-16 lattice."""
    from kappanav.local_planner.features.history_fusion import fusion_inputs
    plans=[r[0]['history_plan'] for r in records];shapes=[];widths=[]
    channels=actor.policy.depth_encoder.spatial_projection.out_channels
    for item,raw,*_ in records:
        ids=[int(index) for index,_ in source_groups(item['history_plan'])]
        shapes.append({index:(1,channels,(raw['depths'][index].shape[-2]+15)//16,(raw['depths'][index].shape[-1]+15)//16) for index in ids})
        widths.append({index:raw['depths'][index].shape[-1] for index in ids})
    fusion=fusion_inputs(actor.policy.depth_encoder,plans,shapes,widths,bucket=bucket)
    return dict(fusion=fusion,conditions=encoding_inputs(actor,plans,fusion['sizes'],[r[0]['goal'] for r in records],
        [r[0]['goal_present'] for r in records]))


def encoding_inputs(actor,plans,sizes,goals,goal_present):
    """Host side of encode_conditions for plans contributing `sizes` fused cells."""
    from kappanav.local_planner.features.history_fusion import memory_inputs
    from kappanav.local_planner.features.local_features import fine_local_inputs
    batch_size=actor.config.training.geometry_batch_size
    return dict(chunks=[memory_inputs(actor.policy,actor.head,plans[start:start+batch_size],goals[start:start+batch_size],
        goal_present[start:start+batch_size],max(sizes[start:start+batch_size])) for start in range(0,len(plans),batch_size)],
        fine=fine_local_inputs(plans,sizes))


def encode_conditions(actor,plans,visuals,goals,goal_present,*,host=None):
    """The shared online/replay path from fused cells to complete scene conditions."""
    if host is None:host=HostTree(encoding_inputs(actor,plans,[len(v) for v in visuals],goals,goal_present)).to(visuals[0].device)
    conditions=[];fields=[];logits=[]
    batch_size=actor.config.training.geometry_batch_size
    for chunk,start in zip(host['chunks'],range(0,len(plans),batch_size),strict=True):
        condition,(field,logit)=encode_memories(actor.policy,actor.head,
            plans[start:start+batch_size],visuals[start:start+batch_size],goals[start:start+batch_size],
            goal_present[start:start+batch_size],host=chunk)
        conditions.append(condition);fields.append(field);logits.append(logit)
    condition=append_fine_local(actor,cat_conditions(conditions),plans,visuals,host=host['fine'])
    # LH trains the field itself; the condition carries its detached reading.
    return condition,(torch.cat(fields),torch.cat(logits))

def cat_conditions(conditions):
    if len(conditions)==1:return conditions[0]
    return ConditionFeatures(**{f.name:torch.cat([getattr(c,f.name) for c in conditions],0) for f in fields(ConditionFeatures)})

def select_condition(condition,indices):
    ids=torch.as_tensor(indices,dtype=torch.long).to(condition.tokens.device,non_blocking=True)
    return ConditionFeatures(**{f.name:getattr(condition,f.name).index_select(0,ids) for f in fields(ConditionFeatures)})
