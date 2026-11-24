"""Preserved real data/geometry/availability boundaries for native joint training."""
import numpy as np
import torch
from kappanav.local_planner.data.bspline_replay import dense_curve
from kappanav.local_planner.data.observed_geometry import BoxIndex
from kappanav.local_planner import build_policy
from test_policy import tiny_config


def test_dense_native_curve_bound_and_exact_segment_box_contacts():
    values = torch.zeros(1, 14)
    values[0, 0::2] = torch.linspace(.1, .7, 7)
    path, pad = dense_curve(values)
    assert path.shape == (1, 257, 2)
    assert torch.equal(path[:, 0], torch.zeros(1, 2))
    torch.testing.assert_close(path[:, -1], values.reshape(1, 7, 2)[:, -1], rtol=0, atol=0)
    assert pad.item() > 0
    boxes = np.array([[0., 0., 1., 1.]], dtype=np.float64)
    crossing = np.array([[[-1., .5], [2., .5]]])
    separated = np.array([[[-1., 2.], [2., 2.]]])
    assert BoxIndex(boxes).clearance(crossing, radius=.2)[0] == -.2
    assert BoxIndex(boxes).clearance(separated, radius=.2)[0] == .8

def test_expert_pool_exact_pose_source_and_float32_deduplication():
    from kappanav.local_planner.data.bspline_dataset import attach_q_expert_pools
    def item(value, source='grid', x=0.):
        return dict(teacher_values=torch.full((14,),value),source_grid_sha256=source,
            view=np.array([x,2.,3.],dtype=np.float64),raw_sha256='raw',acquisitions=10,step=0)
    a,b,duplicate,c,d = item(.1),item(.2),item(.1),item(.3,source='other'),item(.4,x=np.nextafter(0.,1.))
    attach_q_expert_pools([(x,None,None,None) for x in (a,b,duplicate,c,d)])
    torch.testing.assert_close(a['q_expert_values'],torch.stack((a['teacher_values'],b['teacher_values'])),rtol=0,atol=0)
    torch.testing.assert_close(b['q_expert_values'],torch.stack((b['teacher_values'],a['teacher_values'])),rtol=0,atol=0)
    assert c['q_expert_values'].shape==d['q_expert_values'].shape==(1,14)

def test_sampler_cycles_every_state_without_history_weight_and_keeps_case_balance():
    from collections import Counter
    from kappanav.local_planner.data.bspline_dataset import case_groups, sample_index
    items=[]
    for trial,history,count in [('D2','old',5),('D2','new',18),('K','other',3)]:
        items.extend((dict(trial_id=trial,raw_sha256=history),{},None,None) for _ in range(count))
    groups=case_groups([r[0]['trial_id'] for r in items],[r[0]['raw_sha256'] for r in items],[False]*len(items),range(len(items)))
    selected=[sample_index(groups,p) for p in range(46)]
    assert selected[::2]==list(range(23))
    assert set(selected)==set(range(len(items)))
    assert Counter(items[i][0]['trial_id'] for i in selected)=={'D2':23,'K':23}


def test_partial_q_pool_may_use_compatible_expert_but_not_for_cfm():
    from kappanav.local_planner.data.bspline_dataset import attach_q_expert_pools
    def item(teacher,source):
        return dict(teacher_values=teacher,source_grid_sha256=source,view=[0.,1.,2.],raw_sha256='raw',step=0,acquisitions=10)
    partial=item(None,'a');expert=item(torch.ones(14),'a');isolated=item(None,'b')
    attach_q_expert_pools([(x,None,None,None) for x in (partial,expert,isolated)])
    assert partial['teacher_values'] is None and partial['q_expert_values'].shape==(1,14)
    assert isolated['teacher_values'] is None and isolated['q_expert_values'].shape==(0,14)

def test_source_utility_availability_uses_real_cells_and_disconnected_start():
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    from kappanav.local_planner.data.bspline_teachers import source_utility_availability
    from kappanav.local_planner.data.goal_distance import GoalDistanceQuery
    grid=np.ones((5,5),dtype=np.float32);grid[1,1]=-.05
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(grid,np.zeros(2),1.,'canonical_world_xy'),))
    distance=GoalDistanceQuery(query)
    defined,evidence=source_utility_availability(distance,np.array([1.5,1.5,0.]),torch.tensor([2.,2.]))
    assert not defined and evidence['start_cell']==[1,1]
    assert source_utility_availability(distance,np.array([1.5,1.5,0.]),torch.tensor([2.,2.]))==(defined,evidence)
    defined,evidence=source_utility_availability(distance,np.array([.5,.5,0.]),torch.tensor([3.,3.]))
    assert defined and np.isfinite(GoalDistanceQuery(query).query(0,np.array(evidence['goal_cell']),np.array([evidence['start_cell']])))[0]
    defined,evidence=source_utility_availability(distance,np.array([.5,.5,0.]),torch.zeros(2))
    assert defined
    defined,evidence=source_utility_availability(distance,np.array([1.5,1.5,0.]),torch.zeros(2))
    assert not defined


def test_component_connectivity_equals_finite_goal_distance():
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    from kappanav.local_planner.data.goal_distance import GoalDistanceQuery
    rng=np.random.default_rng(0)
    for trial in range(20):
        grid=np.where(rng.random((24,24))<.35,-.05,1.).astype(np.float32)
        distance=GoalDistanceQuery(SourceConfigurationSpaceQuery((SourceConfigurationGrid(grid,np.zeros(2),.05,'canonical_world_xy'),)))
        free=np.argwhere(grid>=0);goal=free[rng.integers(len(free))]
        finite=np.isfinite(distance.query(0,goal,free))
        assert np.array_equal(finite,np.array([distance.connected(0,cell,goal) for cell in free]))

def test_generated_teacher_pool_is_the_same_pose_pool_with_the_own_teacher_first():
    from kappanav.local_planner.data.bspline_dataset import attach_trajectory_teacher_pools
    def item(value,raw='a',goal_present=False,goal=0.,plan='p'):
        return dict(raw_sha256=raw,acquisitions=10,history_plan_sha256=plan,view=np.array([1.,2.]),
            goal_present=goal_present,goal=torch.tensor([goal,0.]),
            teacher_values=None if value is None else torch.full((14,),value))
    rows=[item(1.),item(2.),item(3.,raw='b'),item(4.,goal_present=True,goal=3.),item(None),item(5.,plan='different')]
    attach_trajectory_teacher_pools([(row,) for row in rows])
    # Same capture, history and pose share teachers across goal modes and goals; the own teacher comes first.
    assert rows[0]['trajectory_teacher_values'][:,0].tolist()==[1.,2.,4.]
    assert rows[1]['trajectory_teacher_values'][:,0].tolist()==[2.,1.,4.]
    assert rows[3]['trajectory_teacher_values'][:,0].tolist()==[4.,1.,2.]
    # Another capture or another causal history is a different state.
    assert rows[2]['trajectory_teacher_values'][:,0].tolist()==[3.]
    assert rows[5]['trajectory_teacher_values'][:,0].tolist()==[5.]
    # A row whose own teacher failed has no G target.
    assert rows[4]['trajectory_teacher_values'].shape==(0,14)


def test_energy_score_teacher_weights_give_the_own_teacher_half():
    from kappanav.local_planner.training.bspline_objective import teacher_weights, OWN_TEACHER_WEIGHT
    assert teacher_weights(1).tolist()==[1.]
    weights=teacher_weights(5)
    assert abs(float(weights[0])-OWN_TEACHER_WEIGHT)<1e-7 and abs(float(weights.sum())-1)<1e-6


def test_reference_fit_preserves_backwards_tangent_and_endpoint():
    from kappanav.local_planner.model.trajectory.control_points import project_reference_tangent
    codec = build_policy(tiny_config()).curve_codec
    reference = torch.stack((-torch.linspace(0, 2, 64), torch.zeros(64)), -1)[None]
    values, path = project_reference_tangent(codec, reference)
    assert values[0, 0] < 0
    torch.testing.assert_close(path, reference, atol=1e-5, rtol=1e-5)


def test_dataset_entry_lists_split_manifests_and_rows(tmp_path):
    import json
    from kappanav.local_planner.data.bspline_dataset import dataset_manifests
    from test_replay_pack import make_array_replay
    from kappanav.local_planner.data.replay_pack import trajectory_contract
    actor,manifest,_,_=make_array_replay(tmp_path/'data')
    path=tmp_path/'dataset.json';path.write_text(json.dumps(dict(format='curvenav_array_dataset',
        manifests=dict(train=['data/manifest.json'],validation=[]),rows=dict(train=2,validation=0),
        trajectory_contract=trajectory_contract(actor))))
    manifests,rows=dataset_manifests(path)
    assert manifests['train']==[manifest] and rows['train']==2
    # several datasets concatenate in order under one trajectory contract, and refuse another contract
    second=tmp_path/'second.json';second.write_text(path.read_text())
    manifests,rows=dataset_manifests([path,second])
    assert manifests['train']==[manifest,manifest] and rows==dict(train=4,validation=0)
    other=json.loads(path.read_text());other['trajectory_contract']=dict(other='contract');second.write_text(json.dumps(other))
    import pytest
    with pytest.raises(ValueError,match='trajectory contract'):dataset_manifests([path,second])
