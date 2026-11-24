"""The four-frame local B-spline model and checkpoint I/O."""
import os
import torch
from torch import nn
from kappanav.local_planner.model.config_io import config_from_mapping
from kappanav.local_planner.model.factory import build_policy
from kappanav.local_planner.model.network.policy import generate_candidate_bank, INFERENCE_CANDIDATES
from kappanav.local_planner.features.conditions import encode_conditions
from kappanav.local_planner.model.configuration_space import query_configuration_field
from kappanav.local_planner.features.four_frame_history import FourFrameHistory, HISTORY_CONTRACT
from kappanav.common.physical import DINGO, ROBOT
from kappanav.local_planner.features.temporal_geometry import FeatureGeometryDecoder

FORMAT = 'curvenav_policy'
ACTOR_SEMANTICS = 'native16_euler2_four_frame_baseline'
CANDIDATES = INFERENCE_CANDIDATES
# The perceived configuration field is the C-space of the robot the policy was trained on (the Dingo). A wider robot
# needs that much more clearance: a candidate counts as clear only beyond the difference in size.
CLEARANCE_MARGIN_M = float(os.environ.get('KAPPANAV_CLEARANCE_MARGIN_M', max(0., ROBOT.footprint_radius_m - DINGO.footprint_radius_m)))

class LocalBSplineBundle(nn.Module):
    """The complete perception, G and Q model."""
    def __init__(self,config,policy,head,architecture,provenance):
        super().__init__()
        self.config,self.architecture,self.provenance=config,architecture,provenance
        self.policy,self.head=policy,head

    def encode(self, history, goal, goal_present):
        history.write()
        plan,visual=history.packet()
        return encode_conditions(self,[plan],[visual],[goal],[goal_present])[0]

    def native_bank(self, condition, *, memory=None, source=None):
        return generate_candidate_bank(self.policy, condition, memory=memory, source=source)

    @torch.autocast(device_type='cuda',enabled=False)
    def candidate_batches(self, condition, goal):
        goal=torch.as_tensor(goal,dtype=torch.float32,device=condition.tokens.device).reshape(-1,2)
        bank=self.native_bank(condition)
        scores=self.policy.score_candidate_paths(condition,goal,bank['paths'])
        batch,count=bank['paths'].shape[:2]
        clear=query_configuration_field(condition.configuration_field,bank['paths'].flatten(0,1),
            self.policy.planning_horizon_m).estimated_clearance_m.unflatten(0,(batch,count)).amin(-1)>=CLEARANCE_MARGIN_M
        selected=scores.masked_fill(~clear & clear.any(-1,keepdim=True),float('-inf')).argmax(-1)
        return dict(values=bank['values'],paths=bank['paths'],tangent=bank['tangent'],
            scores=scores,clear=clear,selected_index=selected)

    def candidates(self, condition, goal):
        bank={name:value[0] for name,value in self.candidate_batches(condition,goal).items()}
        bank['selected_index']=int(bank['selected_index'])
        return bank

    @torch.inference_mode()
    def step(self,history,goal,goal_present):
        current=self.encode(history,goal,goal_present)
        return self.candidates(current,current.terminal_goal)

    def new_history(self, *, seed=0):
        return FourFrameHistory(self.policy, seed=seed, **self.architecture['history'])


def create_bundle(config):
    """Initialize the single current graph directly; no checkpoint-dependent construction."""
    policy=build_policy(config)
    head=FeatureGeometryDecoder(config.depth_encoder.model_dim+8,policy.planning_horizon_m,policy.depth_encoder.metric_projector.radius,
        context_radius_m=config.depth_encoder.geometry_context_m)
    architecture=dict(actor_semantics=ACTOR_SEMANTICS,history_contract=HISTORY_CONTRACT,
        precision='bf16_neural_fp32_fusion',direct_fourframe=True,history=dict(capacity=32768))
    architecture['dense_geometry']=dict(context_radius_m=config.depth_encoder.geometry_context_m,raster_size=head.raster_size)
    return LocalBSplineBundle(config,policy.eval(),head.eval(),architecture,dict(initialization='fresh')).eval()


def load_bundle(path, device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    raw=saved['base_config']
    actor=create_bundle(config_from_mapping(raw))
    for name in ('policy','head'):
        getattr(actor,name).load_state_dict(saved[name],strict=True);getattr(actor,name).to(device).eval()
    actor.provenance=dict(saved['provenance'])
    return actor


def save_bundle(path, *, base_config, policy, head, architecture, provenance):
    state=lambda module:{name:value.detach().cpu() for name,value in module.state_dict().items()}
    torch.save(dict(format=FORMAT,base_config=base_config,architecture=architecture,
        provenance=provenance,policy=state(policy),head=state(head)),path)


# Receding-horizon commitment: Q's top field-clear candidates are often near-tied, and a fresh argmax every decision
# would flip between symmetric escapes. The candidate that continues the published plan keeps a small bonus.
COMMITMENT_BONUS=.08
COMMITMENT_RADIUS_M=.3
COMMITMENT_ARC_M=1.


def committed_selection(paths,scores,clear,previous=None):
    """Q argmax among perceived-field-clear candidates (all candidates when none is clear), with COMMITMENT_BONUS
    for the candidate whose first COMMITMENT_ARC_M stays within COMMITMENT_RADIUS_M (mean) of the previous plan.
    paths [C,P,2], scores [C], clear [C] and previous [M,2] are numpy arrays in the current body frame."""
    import numpy as np
    paths=np.asarray(paths,dtype=np.float64);score=np.asarray(scores,dtype=np.float64).copy();clear=np.asarray(clear,dtype=bool)
    eligible=clear if clear.any() else np.ones_like(clear)
    score[~eligible]=-np.inf
    if previous is not None and len(previous)>1:
        previous=np.asarray(previous,dtype=np.float64)
        arc=np.concatenate((np.zeros((len(paths),1)),np.cumsum(np.linalg.norm(np.diff(paths,axis=1),axis=-1),1)),1)
        near=np.linalg.norm(paths[:,:,None]-previous[None,None],axis=-1).min(-1)
        within=arc<=COMMITMENT_ARC_M
        distance=(near*within).sum(1)/np.maximum(within.sum(1),1)
        distance[~eligible]=np.inf
        continuing=int(np.argmin(distance))
        if distance[continuing]<=COMMITMENT_RADIUS_M:score[continuing]+=COMMITMENT_BONUS
    return int(np.argmax(score))


class PlanCommitment:
    """The deployed selection of one episode: committed_selection with the last published plan as its memory.

    Every closed loop (Habitat, Isaac, open-loop replay) selects through one instance per episode,
    so the memory and its frame changes are defined once. `rotation` [2,2] maps body to world and `origin` [2] is
    the body position in the world: a body-frame point p is p @ rotation.T + origin."""

    def __init__(self):
        self.world=None

    def select(self,paths,scores,clear,rotation,origin):
        """(selected index, its world path) for body-frame candidates paths [C,P,2], scores [C], clear [C]."""
        import numpy as np
        paths=np.asarray(paths);rotation=np.asarray(rotation);origin=np.asarray(origin)
        previous=None if self.world is None else (self.world-origin)@rotation
        index=committed_selection(paths,scores,clear,previous)
        self.world=paths[index]@rotation.T+origin
        return index,self.world
