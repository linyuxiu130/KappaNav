"""Four-frame B-spline model: absent-goal semantics, generation and portable state."""
from dataclasses import asdict, replace

import torch
import pytest

from kappanav.local_planner import build_policy
from kappanav.local_planner.model.config import CurveNavConfig, TrajectoryDecoderConfig, DataConfig
from kappanav.local_planner.inference import bundle as bundle_api
from kappanav.local_planner.inference.bundle import LocalBSplineBundle, load_bundle, save_bundle
from kappanav.local_planner.features.temporal_geometry import FeatureGeometryDecoder
from test_policy import tiny_config, make_encoded_condition


def test_absent_goal_and_goal_do_not_change_the_memory_features():
    import numpy as np
    cfg=tiny_config();cfg=replace(cfg,data=replace(cfg.data,observation_frames=6))
    policy=build_policy(cfg).eval()
    plan=dict(points=np.empty((0,3),np.float32),keys=np.empty((0,3),np.int64),depth=np.empty(0,np.float32),
        origins=np.empty((0,3),np.float32),times=np.empty(0,np.float64),clock=0.,pose=np.eye(4),
        selection=dict(valid=np.array([False]*5+[True]),poses=np.tile(np.eye(4),(6,1,1)),ages=np.zeros(6,np.float32)))
    class Memory:
        def write(self):pass
        def packet(self):return plan,torch.empty(0,32)
    head=FeatureGeometryDecoder(40,policy.planning_horizon_m,.167584539,width=8).eval()
    actor=LocalBSplineBundle(cfg,policy,head,{},{});memory=Memory()
    absent=actor.encode(memory,[1.,2.],False);present=actor.encode(memory,[1.,2.],True)
    assert not absent.terminal_goal.any() and not absent.goal_present.any()
    torch.testing.assert_close(present.terminal_goal,torch.tensor([[1.,2.]]))
    torch.testing.assert_close(absent.tokens,present.tokens,rtol=0,atol=0)
    assert present.tokens.shape[1]==261 and present.token_valid.sum()==256


def test_nogoal_native16_masks_q_goal_and_keeps_two_flow_gradients():
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 1)
    encoded = replace(encoded, goal_present=torch.zeros(1, 1))
    actor = LocalBSplineBundle(None, policy, None,
        {}, {})
    first = actor.candidates(encoded, [3., -2.])
    second = actor.candidates(replace(encoded, terminal_goal=torch.tensor([[-9., 8.]])), [-9., 8.])
    assert first['paths'].shape == (16, 64, 2)
    for name in ('values', 'paths', 'scores'):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    assert first['selected_index'] == second['selected_index']
    (first['values'].square().mean()+first['scores'].square().mean()).backward()
    assert policy.trajectory_decoder.velocity_readout.projection.weight.grad.abs().sum() > 0
    assert policy.trajectory_evaluator.readout[2].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('context,size',[(12.,224),(8.,160)])
def test_current_bundle_roundtrip_has_one_graph(tmp_path,context,size):
    cfg=CurveNavConfig(data=DataConfig(observation_frames=4),trajectory_decoder=TrajectoryDecoderConfig(transformer_layers=1))
    cfg=replace(cfg,depth_encoder=replace(cfg.depth_encoder,geometry_context_m=context))
    actor=bundle_api.create_bundle(cfg)
    raw=dict(data=asdict(cfg.data),training=asdict(cfg.training),model=dict(trajectory=asdict(cfg.trajectory),
        depth_encoder=asdict(cfg.depth_encoder),condition_encoder=asdict(cfg.condition_encoder),trajectory_decoder=asdict(cfg.trajectory_decoder)))
    path=tmp_path/'actor.pt'
    save_bundle(path,base_config=raw,policy=actor.policy,head=actor.head,architecture=actor.architecture,provenance=actor.provenance)
    restored=load_bundle(path)
    for name in ('policy','head'):
        for key,value in getattr(actor,name).state_dict().items():
            torch.testing.assert_close(value,getattr(restored,name).state_dict()[key],rtol=0,atol=0)
    assert restored.policy.curve_codec.coordinate_dim==14
    assert restored.policy.condition_encoder.motion_encoder.state_tokens==3
    assert restored.head.stem[0].in_channels==397
    assert not hasattr(restored.policy.condition_encoder,'configuration_encoder')
    assert not hasattr(restored,'reader')
    assert restored.provenance['initialization']=='fresh'
    assert restored.head.raster_size==size
    saved=torch.load(path,weights_only=False)
    saved['head']['metric_position']=saved['head']['metric_position']+1
    torch.save(saved,path)
    saved['head']['metric_position']=actor.head.metric_position.clone()
    saved['base_config']['model']['depth_encoder']['geometry_context_m']=8. if context==12. else 12.
    torch.save(saved,path)


def test_evidence_q_does_not_treat_absent_goal_as_origin():
    from kappanav.local_planner.model.network.bspline_evidence_evaluator import EvidenceTrajectoryEvaluator
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 2)
    encoded = replace(encoded, goal_present=torch.tensor([[0.], [1.]]),
        configuration_field=encoded.configuration_field[:1].expand(2, -1, -1, -1))
    critic = EvidenceTrajectoryEvaluator(encoded.tokens.shape[-1], 4, policy.planning_horizon_m)
    policy.trajectory_evaluator = critic
    path = torch.stack((torch.linspace(0, 1, 64), torch.zeros(64)), -1)
    candidates = path[None, None].expand(2, 1, -1, -1)
    quality = []
    handle = critic.input_embedding[0].register_forward_pre_hook(lambda _, args: quality.append(args[0].detach()))
    try:
        scores = policy.score_candidate_paths(encoded, torch.zeros(2, 2), candidates)
    finally:
        handle.remove()
    assert torch.isfinite(scores).all()
    assert (quality[0][0,:,-1]==0).all()
    assert (quality[0][1,:,-1]==1).all()
    torch.testing.assert_close(quality[0][0,:,:-1],quality[0][1,:,:-1])




def test_two_euler_steps_query_the_actual_noise_and_intermediate_curves(monkeypatch):
    policy = build_policy(tiny_config()).eval()
    encoded = replace(make_encoded_condition(policy, 1), goal_present=torch.zeros(1, 1))
    actor = LocalBSplineBundle(None, policy, None, {}, {})
    decoder = policy.trajectory_decoder
    calls = []
    original = decoder._trajectory_geometry
    def query(path, condition):
        calls.append(path.detach().clone())
        return original(path, condition)
    monkeypatch.setattr(decoder, '_trajectory_geometry', query)
    states = []
    original_velocity = policy._predict_velocity
    def velocity(state, time, condition, memory, step=0.):
        states.append(state.detach().clone())
        return original_velocity(state, time, condition, memory, step)
    monkeypatch.setattr(policy, '_predict_velocity', velocity)
    actor.native_bank(encoded)
    assert len(calls) == 2
    for state, queried in zip(states, calls, strict=True):
        expected, _ = policy.curve_codec.decode(state)
        torch.testing.assert_close(queried, expected, rtol=0, atol=0)
    assert not torch.equal(calls[0], calls[1])
    assert calls[0].unique(dim=0).shape[0] == 16


def test_bundle_roundtrip_and_step_use_the_one_four_frame_graph(tmp_path):
    from kappanav.local_planner.model.config_io import config_to_mapping
    cfg=replace(tiny_config(),data=replace(tiny_config().data,observation_frames=4))
    actor=bundle_api.create_bundle(cfg)
    path=tmp_path/'baseline.pt'
    save_bundle(path,base_config=config_to_mapping(cfg),policy=actor.policy,head=actor.head,architecture=actor.architecture,provenance=actor.provenance)
    assert set(torch.load(path,weights_only=False))=={'format','base_config','architecture','provenance','policy','head'}
    restored=load_bundle(path)
    encoded=make_encoded_condition(restored.policy,1)
    expected=actor.candidates(encoded,encoded.terminal_goal);actual=restored.candidates(encoded,encoded.terminal_goal)
    for key in ('values','paths','scores'):torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert restored.new_history() is not None

def test_selection_skips_candidates_the_field_shows_colliding(monkeypatch):
    from types import SimpleNamespace
    from kappanav.local_planner.inference import bundle as bspline_bundle
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 1)
    actor = LocalBSplineBundle(None, policy, None, {}, {})
    monkeypatch.setattr(policy, 'score_candidate_paths', lambda condition, goal, paths: torch.arange(16.)[None])
    clearance = torch.ones(16, 64)
    monkeypatch.setattr(bspline_bundle, 'query_configuration_field',
        lambda *args, **kwargs: SimpleNamespace(estimated_clearance_m=clearance))
    clearance[15, 10] = -.01   # Q's favourite touches an obstacle at one sample
    assert actor.candidates(encoded, [1., 0.])['selected_index'] == 14
    clearance[:] = -.01        # every candidate does: fall back to Q alone
    assert actor.candidates(encoded, [1., 0.])['selected_index'] == 15


def test_committed_selection_is_plain_argmax_without_a_previous_plan_and_keeps_a_near_tie():
    import numpy as np
    from kappanav.local_planner.inference.bundle import committed_selection, COMMITMENT_BONUS
    s=np.linspace(0,2,32)[:,None]
    left=np.hstack((s*.7,s*.7));right=np.hstack((s*.7,-s*.7));ahead=np.hstack((s,0*s))
    paths=np.stack((left,right,ahead));clear=np.array([True,True,False])
    # No previous plan: argmax over field-clear candidates (the blocked best one is skipped).
    assert committed_selection(paths,np.array([.50,.52,.9]),clear)==1
    # Previous plan went left: a near-tie keeps it, a clearly better right turn wins.
    assert committed_selection(paths,np.array([.50,.52,.9]),clear,previous=left)==0
    assert committed_selection(paths,np.array([.50,.50+COMMITMENT_BONUS+.01,.9]),clear,previous=left)==1
    # Nothing clear: all candidates are eligible, as in candidate_batches.
    assert committed_selection(paths,np.array([.5,.4,.9]),np.zeros(3,bool))==2


def test_plan_commitment_carries_the_published_plan_across_body_frames():
    import numpy as np
    from kappanav.local_planner.inference.bundle import PlanCommitment, committed_selection
    rng = np.random.default_rng(0)
    paths = [np.cumsum(rng.normal(0, .1, (6, 16, 2)), 1) for _ in range(3)]
    scores = [rng.normal(0, .05, 6) for _ in range(3)]
    clear = [rng.random(6) > .2 for _ in range(3)]
    views = [np.array([0., 0., 0.]), np.array([.2, .1, .4]), np.array([.35, .3, 1.1])]
    commitment, previous_world = PlanCommitment(), None
    for p, q, c, view in zip(paths, scores, clear, views):
        cos, sin = np.cos(view[2]), np.sin(view[2]); rotation = np.array([[cos, -sin], [sin, cos]])
        # The inline form the closed loops used before PlanCommitment.
        previous = None if previous_world is None else (previous_world - view[:2]) @ rotation
        expected = committed_selection(p, q, c, previous)
        previous_world = p[expected] @ np.array([[cos, sin], [-sin, cos]]) + view[:2]
        index, world = commitment.select(p, q, c, rotation, view[:2])
        assert index == expected
        np.testing.assert_array_equal(world, previous_world)
