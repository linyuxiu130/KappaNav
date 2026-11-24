"""Single-row and unbatched reference implementations that the tests hold the batched production paths to.

They are definitions for equivalence tests only; training, data preparation and deployment never call them.
"""
from collections import defaultdict
import math
import torch
from kappanav.local_planner.model.config import CurveNavConfig
from kappanav.local_planner.features.conditions import live_conditions
from kappanav.local_planner.features.obstacle_memory import GRID_SIZE
from kappanav.local_planner.features.history_fusion import source_features_batch, encode_memories, source_groups
from kappanav.local_planner.features.staging import stage
from kappanav.local_planner.model.encoders.configuration import memory_splat_stencil
from kappanav.local_planner.model.encoders.geometry import MetricDepthProjector
from kappanav.common.physical import ROBOT_FOOTPRINT_RADIUS_M, EXTRA_CLEARANCE_M
from kappanav.local_planner.training.bspline_objective import start_critic
from kappanav.local_planner.data.replay_targets import critic_source_inputs


def apply_memory_splat(tokens,stencil):
    """Only learned features are recomputed; accumulation remains FP32."""
    entries,mass,size=stencil
    batch,_,model_dim=tokens.shape
    projected=tokens.float();bev=projected.new_zeros(batch,size**2,model_dim)
    for index,weight in entries:
        bev.scatter_add_(1,index.expand(-1,-1,model_dim),projected*weight)
    return bev/mass.clamp_min(1e-6),mass


CRITIC_PATHS=('q_expert_paths','critic_replay_paths')


def prepare_critic_inputs(item, goal_present, paths, utility, recorded_paths):
    """Prepare an admitted Q row: its expert, current and recorded candidates, all scored by `utility` here."""
    fixed=critic_source_inputs(item)
    expert,recorded=recorded_paths
    qbatch=dict(fixed,goal_present=goal_present[:,0].bool())
    paths=paths.detach();expert_count=len(expert)
    if expert_count:paths=torch.cat((expert,paths))
    return dict(paths=paths, qbatch=qbatch, utility=utility, expert_count=expert_count, recorded=recorded)


def label_critic_inputs(inputs):
    """Score every candidate of a row with the current utility in one batched query per source oracle.

    Stored utility targets would be a cache of one utility definition; scoring expert, current and recorded
    candidates together keeps Q's target defined in exactly one place. Rows are padded with their own current
    candidate to a common bank size; utilities are per path, so padded scores are simply discarded.
    """
    groups=defaultdict(list)
    for prepared in inputs:groups[prepared['utility']].append(prepared)
    for utility,group in groups.items():
        experts=max(p['expert_count'] for p in group);recorded=max(len(p['recorded']) for p in group)
        banks=[]
        for p in group:
            native=p['paths'][p['expert_count']:];pad=lambda tensor,size:torch.cat((tensor,native[:1].expand(size-len(tensor),*native.shape[1:])))
            banks.append(torch.cat((pad(p['paths'][:p['expert_count']],experts),native,pad(p['recorded'].to(native.device),recorded))))
        paths=torch.stack(banks)
        batch={key:torch.cat([p['qbatch'][key] for p in group]) for key in group[0]['qbatch']}
        keys=[key for key in batch if key!='source_metadata_cpu']
        moved=stage([batch[key] for key in keys],paths.device)
        batch.update(zip(keys,moved,strict=True))
        scores=utility(paths,batch).score.detach()
        native_count=group[0]['paths'].shape[0]-group[0]['expert_count']
        for row,p in enumerate(group):
            count=p['expert_count']
            p['targets']=torch.cat((scores[row:row+1,:count],scores[row:row+1,experts:experts+native_count]),-1)
            p['recorded_targets']=scores[row:row+1,experts+native_count:experts+native_count+len(p['recorded'])]


def label_on_stream(inputs,stream):
    # CUDA streams are thread-local; keep labels on their caller's stream.
    if stream is not None:torch.cuda.set_stream(stream)
    label_critic_inputs(inputs)


def batched_critic(actor,condition,items,paths,utilities,*,host=None,fields=None):
    """LQ of the generated bank against current-utility targets of every candidate."""
    return start_critic(actor,condition,items,paths,utilities,host=host,fields=fields).result()


def live_condition(actor,item,raw,*,feature_provider):
    maps={(item['raw_sha256'],int(index)):feature_provider(int(index)) for index,_ in source_groups(item['history_plan'])}
    return live_conditions(actor,[(item,raw)],maps)


def source_features(encoder,plan,maps,widths):
    return source_features_batch(encoder,[plan],[maps],[widths])[0]


def encode_memory(policy,head,plan,visual,goal,goal_present):
    """Online and replay use the same encoder; online supplies one memory."""
    return encode_memories(policy,head,[plan],[visual],[goal],[goal_present])


def splat_memory(tokens,points,valid,planning_horizon_m,grid_size,padding):
    batch,count,_=tokens.shape
    return apply_memory_splat(tokens,memory_splat_stencil(points,valid,planning_horizon_m,grid_size,padding))


def build_evaluation_projector(config: CurveNavConfig) -> MetricDepthProjector:
    """Build the raw-depth diagnostic geometry outside the policy graph."""
    data = config.data
    return MetricDepthProjector(
        token_height=config.depth_encoder.frame_tokens_height,
        token_width=config.depth_encoder.frame_tokens_width,
        max_depth_m=data.max_depth_m,
        planning_horizon_m=data.future_steps * data.expert_waypoint_spacing_m,
        geometry=data.robot_geometry,
    )


def memory_grid_shape(horizon_m, radius_m=ROBOT_FOOTPRINT_RADIUS_M):
    resolution = 2 * horizon_m / (GRID_SIZE - 1)
    padding = math.ceil((radius_m + EXTRA_CLEARANCE_M) / resolution) + 1
    return GRID_SIZE + 2 * padding
