"""Four-frame fusion produces descriptors; one spatial latent feeds geometry and environment tokens."""
from dataclasses import replace
from collections import OrderedDict
import threading
import numpy as np
import torch
from torch.nn import functional as F
from kappanav.local_planner.features.history import inverse_rigid
from kappanav.local_planner.model.encoders.configuration import (memory_splat_stencil, static_splat_segments,
                                                                 _StaticMemorySplat)
from kappanav.local_planner.features.staging import stage, HostTree


_PLAN_INPUTS = OrderedDict()
# Bound CPU-only derived inputs without retaining every visited replay plan.
_PLAN_INPUT_LIMIT = 256
_PLAN_INPUT_LOCK = threading.Lock()


def _plan_inputs(plan):
    """Bound derived CPU inputs instead of growing every replay plan for all epochs.

    Entries hold their plan, so an id is never reused while cached. A training loader prepares the next batch while
    the forward reads the current one, so the cache is locked and must hold both batches (reserve_plan_inputs).
    """
    key = id(plan)
    with _PLAN_INPUT_LOCK:
        entry = _PLAN_INPUTS.get(key)
        if entry is None:
            if len(_PLAN_INPUTS) >= _PLAN_INPUT_LIMIT:
                _PLAN_INPUTS.popitem(last=False)
            entry = _PLAN_INPUTS[key] = (plan, {})
        else:
            _PLAN_INPUTS.move_to_end(key)
        return entry[1]


def reserve_plan_inputs(count):
    """Keep at least `count` plans' derived inputs (e.g. the batch in the forward and the one being prepared)."""
    global _PLAN_INPUT_LIMIT
    with _PLAN_INPUT_LOCK:_PLAN_INPUT_LIMIT = max(_PLAN_INPUT_LIMIT, int(count))


def source_groups(plan):
    """The replay plan is immutable; group its pixels once, in original order."""
    cache=_plan_inputs(plan)
    if '_source_groups' not in cache:
        ids,counts=np.unique(plan['frame_ids'],return_counts=True)
        order=np.argsort(plan['frame_ids'],kind='stable')
        positions=np.split(order,np.cumsum(counts)[:-1])
        cache['_source_groups']=list(zip(ids,positions))
    return cache['_source_groups']


def source_sample_inputs(plan,shapes,widths,*,bucket=None):
    """Immutable pixel indices on CPU; the batch stages them together."""
    groups=source_groups(plan)
    first=shapes[int(groups[0][0])]
    cache=_plan_inputs(plan).setdefault('_source_sample_inputs',{})
    key=(first,bucket)
    if key not in cache:
        buckets={}
        for frame,selected in groups:
            frame=int(frame);size=((len(selected)+bucket-1)//bucket)*bucket if bucket else 1<<(len(selected)-1).bit_length()
            buckets.setdefault((shapes[frame][1:],widths[frame],size),[]).append((frame,selected))
        prepared=[];destinations=[]
        for (_,image_width,size),members in buckets.items():
            pixels=np.zeros((len(members),1,size),dtype=np.int64);valid=[]
            for row,(_,selected) in enumerate(members):
                pixels[row,0,:len(selected)]=plan['pixels'][selected]
                valid.append(row*size+np.arange(len(selected)))
                destinations.append(selected)
            prepared.append(([frame for frame,_ in members],image_width,
                torch.as_tensor(pixels),torch.as_tensor(np.concatenate(valid))))
        cache[key]=(prepared,torch.as_tensor(np.argsort(np.concatenate(destinations))))
    return cache[key]


def source_features_batch(encoder,plans,maps,widths,*,host=None):
    """Batch the original sampler across independent states with matching buckets."""
    if host is not None:raise ValueError('Prepared source inputs belong to the packed training sampler')
    host=[source_sample_inputs(p,{i:tuple(v.shape) for i,v in m.items()},w) for p,m,w in zip(plans,maps,widths,strict=True)]
    device=next(iter(maps[0].values())).device
    staged=iter(stage([value for prepared,order in host for value in
        [part for *_,pixels,valid in prepared for part in (pixels,valid)]+[order]],device))
    inputs=[([(frames,width,next(staged),next(staged)) for frames,width,_,_ in prepared],next(staged)) for prepared,_ in host]
    groups={};pieces=[[None]*len(prepared) for prepared,_ in inputs]
    for row,(prepared,_) in enumerate(inputs):
        for slot,(frames,image_width,pixels,valid) in enumerate(prepared):
            shape=tuple(maps[row][frames[0]].shape[1:])
            groups.setdefault((shape,image_width,pixels.shape[-1]),[]).append((row,slot,frames,pixels,valid))
    for (_,image_width,_),members in groups.items():
        features=torch.cat([maps[row][frame] for row,_,frames,_,_ in members for frame in frames])
        pixels=torch.cat([pixels for _,_,_,pixels,_ in members])
        sampled=encoder.sample_pixel_features(features,pixels,image_width)[:,0]
        for values,(row,slot,_,_,valid) in zip(sampled.split([len(frames) for _,_,frames,_,_ in members]),members,strict=True):
            pieces[row][slot]=values.reshape(-1,features.shape[1]).index_select(0,valid)
    return [torch.cat(parts).index_select(0,order).float() for parts,(_,order) in zip(pieces,inputs,strict=True)]


def fuse_sources(policy,plan,maps,widths):
    """Identical differentiable write fusion for online memory and training replay."""
    return fuse_sources_batch(policy,[plan],[maps],[widths])[0]


def source_fusion_inputs(encoder,plan):
    """Immutable CPU ray descriptors, shared by preparation and live fusion."""
    cache=_plan_inputs(plan).setdefault('_fusion_inputs',{});key=encoder.max_depth_m
    if key not in cache:
        lengths=np.asarray(plan['lengths'])
        if (lengths<0).any() or lengths.sum()!=len(plan['frame_ids']):
            raise ValueError('Ray segment lengths must be nonnegative and cover every source ray')
        tensor=lambda value:torch.as_tensor(value,dtype=torch.float32)
        geometry=torch.cat((tensor(plan['offsets'])*(1.0/encoder.max_depth_m),
            (tensor(plan['depths'])*(1.0/encoder.max_depth_m))[:,None],
            (tensor(plan['depths'])<encoder.max_depth_m)[:,None],tensor(plan['kinds'])[:,None]),-1)
        ages,inverse=np.unique(plan['ages'],return_inverse=True)
        cache[key]=(geometry,tensor(ages)[:,None],torch.as_tensor(inverse),torch.as_tensor(lengths))
    return cache[key]


def fusion_inputs(encoder,plans,shapes=None,widths=None,*,bucket=None):
    """Host side of fuse_sources_batch; with `shapes` (feature-map shapes per plan) also the packed sampler's."""
    sizes=[len(p['lengths']) for p in plans];active=[i for i,size in enumerate(sizes) if size]
    if not active:return dict(sizes=sizes,active=active)
    inputs=[source_fusion_inputs(encoder,plans[index]) for index in active]
    offsets=np.cumsum([0]+[len(values[1]) for values in inputs])
    host=dict(sizes=sizes,active=active,geometry=torch.cat([values[0] for values in inputs]),ages=torch.cat([values[1] for values in inputs]),
        indices=torch.cat([values[2]+offset for values,offset in zip(inputs,offsets[:-1],strict=True)]),lengths=torch.cat([values[3] for values in inputs]))
    if shapes is not None:
        from kappanav.local_planner.features.image_batch import packed_source_inputs
        host['sources']=packed_source_inputs([plans[i] for i in active],[shapes[i] for i in active],[widths[i] for i in active],bucket=bucket)
    return host


def fuse_sources_batch(policy,plans,maps,widths,*,host=None):
    """Fuse independent writes together; segment lengths keep each cell separate."""
    encoder=policy.depth_encoder;device=next(policy.parameters()).device
    if host is None:host=HostTree(fusion_inputs(encoder,plans)).to(device)
    sizes,active=host['sizes'],host['active']
    if not active:return [torch.empty(0,encoder.spatial_projection.out_channels,device=device) for _ in plans]
    geometry,ages,indices,lengths=host['geometry'],host['ages'],host['indices'],host['lengths']
    extra={} if host.get('sources') is None else dict(host=host['sources'])
    visual=torch.cat(source_features_batch(encoder,[plans[i] for i in active],[maps[i] for i in active],[widths[i] for i in active],**extra))
    # FP32 fusion avoids a second BF16 rounding boundary dependent on the number of written cells.
    with torch.autocast(device_type=device.type,enabled=False):
        # Every pixel from one write shares an age; compute its current embedding once.
        tokens=encoder.encode_pixels(visual,geometry,ages,indices)
        fused=torch.segment_reduce(tokens,reduce='mean',lengths=lengths,unsafe=True)
        return fused.split(sizes)


def memory_raster_inputs(policy,head,plan):
    """Build immutable calibration on CPU, without a GPU round trip on cache misses."""
    tensor=torch.as_tensor
    maximum=policy.depth_encoder.max_depth_m
    cache=_plan_inputs(plan).setdefault('_encoding_inputs',{})
    key=(maximum,head.horizon,head.raster_padding)
    if key not in cache:
        points=tensor(plan['points'])[None];valid=torch.ones(points.shape[:2],dtype=torch.bool)
        suffix=torch.cat((points[0]*(1.0/maximum),tensor(plan['depth'])[:,None]*(1.0/maximum),
            tensor(plan['keys'][:,2])[:,None].float(),tensor(plan['origins'])*(1.0/maximum)),-1)
        selection=plan['selection'];transforms=inverse_rigid(plan['pose'])@selection['poses']
        stencil=memory_splat_stencil(points,valid,head.horizon,64,head.raster_padding)
        cache[key]=(suffix,tensor(plan['pose'])[None],tensor(selection['valid'])[None],
            tensor(transforms).float()[None],tensor(selection['ages'])[None],stencil)
    return cache[key]


def raster_inputs(policy,head,plans,count=None):
    """Host side of memory_rasters for `count` padded cells: the most fused visual rows, by default the most
    written cells, len(plan['lengths'])."""
    inputs=[memory_raster_inputs(policy,head,p) for p in plans]
    if count is None:count=max(len(p['lengths']) for p in plans)
    suffix=torch.stack([F.pad(values[0],(0,0,0,count-len(values[0]))) for values in inputs])
    indices=[]
    for slot in range(4):
        cell,_=zip(*(values[-1][0][slot] for values in inputs),strict=True)
        indices.append(torch.cat([F.pad(x,(0,0,0,count-x.shape[1])) for x in cell]))
    corner_indices=torch.stack(indices)
    prepared=[];size=head.raster_size
    for plan,values in zip(plans,inputs,strict=True):
        cache=_plan_inputs(plan).setdefault('_splat_segments',{});key=(head.horizon,head.raster_padding)
        if key not in cache:cache[key]=static_splat_segments(values[-1])
        prepared.append(cache[key])
    corner_weights=torch.stack([torch.cat([F.pad(parts[4][slot],(0,0,0,count-parts[4][slot].shape[1])) for parts in prepared])
        for slot in range(4)])
    sources=torch.cat([parts[0]+row*count for row,parts in enumerate(prepared)])
    weights=torch.cat([parts[1] for parts in prepared])
    cells=torch.cat([parts[2]+row*size*size for row,parts in enumerate(prepared)])
    lengths=torch.cat([parts[3] for parts in prepared])
    return dict(count=count,size=size,tensors=[suffix,corner_indices,corner_weights,sources,weights,cells,lengths,torch.cat([values[-1][1] for values in inputs]),
        *(torch.cat(values) for values in zip(*(x[1:-1] for x in inputs),strict=True))])


def memory_rasters(policy,head,plans,visuals,*,project=False,host=None):
    """Splat independent memories in one batch, padding only their sparse cells."""
    device=visuals[0].device
    if host is None:host=HostTree(raster_inputs(policy,head,plans,max(len(v) for v in visuals))).to(device)
    count,size=host['count'],host['size']
    if max(len(v) for v in visuals)!=count:raise ValueError('Fused cells differ from the plans written cells')
    (suffix,corner_indices,corner_weights,sources,weights,cells,lengths,mass,pose,selection_valid,transforms,ages)=host['tensors']
    visual=torch.stack([F.pad(v,(0,0,0,count-len(v))) for v in visuals])
    raw=torch.cat((visual,suffix),-1)
    if project:raw=head.project_memory(raw)
    raster=_StaticMemorySplat.apply(raw,sources,weights,cells,lengths,corner_indices,corner_weights,size)
    return raster.transpose(1,2).reshape(len(plans),-1,size,size),mass.transpose(1,2).reshape(len(plans),1,size,size),pose,selection_valid,transforms,ages


def memory_inputs(policy,head,plans,goals,goal_present,count=None):
    """Host side of encode_memories."""
    return dict(raster=raster_inputs(policy,head,plans,count),goal=torch.stack([torch.as_tensor(g,dtype=torch.float32).reshape(2) for g in goals]),
        present=torch.as_tensor(goal_present,dtype=torch.bool)[:,None])


def encode_memories(policy,head,plans,visuals,goals,goal_present,*,host=None):
    """Same independent memory encodings, batched through the geometry network."""
    if host is None:host=HostTree(memory_inputs(policy,head,plans,goals,goal_present,max(len(v) for v in visuals))).to(visuals[0].device)
    raster,mass,poses,valid,transforms,ages=memory_rasters(policy,head,plans,visuals,project=True,host=host['raster'])
    field,logits,configuration=head(raster,poses,mass,memory_is_projected=True)
    goal,present=host['goal'],host['present']
    goal=torch.where(present,goal,torch.zeros_like(goal))
    # The field is the geometry estimate fixed by its own supervision (LH) and read by the
    # generator, critic and deployment clearance filter; their losses train tokens, not it.
    condition=policy.condition_encoder.encode_configuration(configuration,field.detach(),goal,valid,transforms,ages)
    return replace(condition,goal_present=present.float()),(field,logits)
