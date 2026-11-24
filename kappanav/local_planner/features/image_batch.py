"""Depth features of a training microbatch: the CNN over every distinct frame, in same-shape chunks."""
import numpy as np
import torch
from itertools import groupby
from kappanav.local_planner.features.history_fusion import source_groups, source_sample_inputs
from kappanav.local_planner.features.staging import HostTree, Keep


def feature_map_inputs(records,cnn_batch_size=64):
    """Host side of step_feature_maps: every CNN chunk's (raw, frame) keys and its gathered depth frames."""
    inputs={}
    for item,raw,_,_ in records:
        key=item['raw_sha256']
        if key not in inputs:
            depths=raw.setdefault('_host_depths',torch.from_numpy(np.asarray(raw['depths']))[:,None])
            inputs[key]=(depths,{})
        inputs[key][1].update((int(index),None) for index,_ in source_groups(item['history_plan']))
    groups={}
    for key,(depth,indices) in inputs.items():
        groups.setdefault(tuple(depth.shape[1:]),[]).extend((key,index) for index in indices)
    chunks=[]
    for keys in groups.values():
        for start in range(0,len(keys),cnn_batch_size):
            chunk=keys[start:start+cnn_batch_size]
            parts=[]
            for key,members in groupby(chunk,key=lambda pair:pair[0]):
                indices=torch.tensor([index for _,index in members])
                parts.append(inputs[key][0].index_select(0,indices))
            chunks.append((Keep(chunk),torch.cat(parts)))
    return chunks


def step_feature_maps(actor,records,cache=None,*,cnn_batch_size=64,host=None):
    """{(raw_sha256, frame index): [1,384,8,14] image features} of every frame the records' four-frame plans read,
    each frame encoded once (from `cache` when given)."""
    device=next(actor.policy.parameters()).device
    if cache is not None:
        keys={(item['raw_sha256'],int(index)) for item,*_ in records for index,_ in source_groups(item['history_plan'])}
        return {key:cache[key].to(device) for key in keys}
    if host is None:host=HostTree(feature_map_inputs(records,cnn_batch_size)).to(device)
    maps={}
    for chunk,depth in host:
        # Depths are converted exactly (e.g. FP16 to FP32) on the device after the transfer.
        features=actor.policy.depth_encoder.image_features(depth.to(dtype=torch.float32))
        maps.update(zip(chunk,features.split(1),strict=True))
    return maps


def packed_source_inputs(plans,shapes,widths,*,bucket=None):
    """Host side of packed_source_features_batch for feature maps of the given shapes."""
    host=[source_sample_inputs(p,s,w,bucket=bucket) for p,s,w in zip(plans,shapes,widths,strict=True)]
    groups={};destinations=[[None]*len(prepared) for prepared,_ in host]
    for row,(prepared,_) in enumerate(host):
        for slot,(frames,image_width,pixels,valid) in enumerate(prepared):
            shape=tuple(shapes[row][frames[0]][1:])
            groups.setdefault((shape,image_width,pixels.shape[-1]),[]).append((row,slot,frames,pixels,valid))
    offset=0;pixel_groups=[]
    for members in groups.values():
        pixel_groups.append(torch.cat([pixels for _,_,_,pixels,_ in members]))
        for row,slot,_,pixels,valid in members:
            destinations[row][slot]=valid+offset
            offset+=pixels.numel()
    order=torch.cat([torch.cat(parts).index_select(0,order) for parts,(_,order) in zip(destinations,host,strict=True)])
    return dict(bucket=bucket,groups=Keep([(shape,image_width,[(row,frames) for row,_,frames,_,_ in members]) for (shape,image_width,_),members in groups.items()]),
        pixels=pixel_groups,order=order,counts=Keep([len(order) for _,order in host]))


def packed_source_features_batch(encoder,plans,maps,widths,*,bucket=None,host=None):
    """Gather all states once, preserving the calibrated sampler and ray order."""
    if host is None or host['bucket']!=bucket or any(tuple(maps[row][frames[0]].shape[1:])!=shape for shape,_,members in host['groups'] for row,frames in members):
        device=next(iter(maps[0].values())).device
        host=HostTree(packed_source_inputs(plans,[{i:tuple(v.shape) for i,v in m.items()} for m in maps],widths,bucket=bucket)).to(device)
    values=[]
    for (_,image_width,members),indices in zip(host['groups'],host['pixels'],strict=True):
        features=torch.cat([maps[row][frame] for row,frames in members for frame in frames])
        values.append(encoder.sample_pixel_features(features,indices,image_width)[:,0].flatten(0,1))
    return torch.cat(values).index_select(0,host['order']).float().split(host['counts'])
