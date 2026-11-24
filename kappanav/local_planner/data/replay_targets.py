"""Fixed targets compiled into the training shards, and the source-map coordinates the route utility (Q's target) reads."""
from collections import defaultdict
import numpy as np
import torch


@torch.no_grad()
def prepare_generator_targets(actor,records):
    """Compile generator targets once while preparing the array snapshot."""
    codec=actor.policy.curve_codec
    qualified=[item for item,*_ in records if item['teacher_values'] is not None]
    for start in range(0,len(qualified),1024):
        chunk=qualified[start:start+1024]
        clean=codec.coordinates_from_values(torch.stack([item['teacher_values'] for item in chunk]))
        for i,item in enumerate(chunk):item['cfm_target']=clean[i:i+1]
    groups=defaultdict(list)
    for item,*_ in records:
        if len(item['trajectory_teacher_values']):groups[len(item['trajectory_teacher_values'])].append(item)
    for rows in groups.values():
        for start in range(0,len(rows),256):
            chunk=rows[start:start+256]
            values=torch.stack([item['trajectory_teacher_values'] for item in chunk])
            target,_=codec.decode_values(values.flatten(0,1))
            target=target.unflatten(0,values.shape[:2])
            for i,item in enumerate(chunk):item['trajectory_target']=target[i]


def critic_sources(rows):
    """Source-map coordinates of rows for RouteUtilityTeacher: source_grid_index [N] int64, source_metadata_cpu
    [N,4] float64 (grid index, view x, y, yaw), source_origin_xy [N,2] and source_yaw_rad [N] float64 (the row's view
    in the source map's world frame), point_goal [N,2] (body frame)."""
    grids=np.asarray([row['source_grid_index'] for row in rows],dtype=np.int64)
    views=np.stack([np.asarray(row['view'],dtype=np.float64) for row in rows])
    return dict(source_grid_index=torch.from_numpy(grids),source_metadata_cpu=torch.from_numpy(np.concatenate((grids[:,None].astype(np.float64),views),1)),
        source_origin_xy=torch.from_numpy(np.ascontiguousarray(views[:,:2])),source_yaw_rad=torch.from_numpy(np.ascontiguousarray(views[:,2])),
        point_goal=torch.stack([row['goal'] for row in rows]))


def critic_source_inputs(item):
    """critic_sources of one row plus its observed target field [1,C,H,W] (the no-goal novelty term reads it)."""
    return dict(critic_sources([item]),observed_field=item['local_target_field'][None])


@torch.no_grad()
def prepare_replay_targets(actor,records):
    """Compile the generator targets and decode each Q-supervised row's expert pool to paths. Q's targets are
    scored online in training (bspline_objective) over these paths and the current and recorded candidates with the
    current utility, so no Q target is stored (the scene compile used to score them for nothing: 11% of a large
    scene's main-process time, one full-scene Dijkstra per relabelled goal)."""
    prepare_generator_targets(actor,records)
    for item,*_ in records:
        if not item['q_supervision_available']:
            continue
        recorded=item['critic_replay_paths']
        if len(item['q_expert_values']):
            expert,_=actor.policy.curve_codec.decode_values(item['q_expert_values'])
        else:
            expert=recorded.new_empty((0,*recorded.shape[1:]))
        if not torch.isfinite(expert).all():
            raise ValueError('Non-finite immutable replay targets')
        item['q_expert_paths']=expert.detach().cpu()
