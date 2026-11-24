"""Score supervision, exact candidate selection, and inference conditioning."""
import reference_implementations
import numpy as np
import pytest
import torch

from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
from kappanav.local_planner.data.value_targets import RouteUtilityTeacher, candidate_query


def oracle():
    clearance = np.ones((200, 200), dtype=np.float32)
    clearance[110:120, :] = -0.2  # world x in [1,2): an impassable wall
    return SourceConfigurationSpaceQuery((SourceConfigurationGrid(clearance, np.array([-10., -10.]), 0.1),))


def source_batch(batch, device):
    return dict(observed_field=torch.ones(batch,6,64,64,device=device), goal_present=torch.ones(batch, device=device, dtype=torch.bool), source_grid_index=torch.zeros(batch, device=device, dtype=torch.long),
                source_origin_xy=torch.zeros(batch, 2, device=device),
                source_yaw_rad=torch.zeros(batch, device=device),
                point_goal=torch.tensor([[0., 3.]], device=device).expand(batch, -1))








def test_teacher_queries_body_clearance_whole_curve_and_is_batch_independent():
    # Endpoints clear, intermediate segment crosses wall beyond a nominal 0.5 m horizon.
    paths = torch.tensor([[[[0., 0.], [0., 3.]], [[0., 0.], [3., 0.]], [[0., 0.], [30., 0.]]]])
    teacher = RouteUtilityTeacher(oracle(), 0.5)
    truth = teacher(paths, source_batch(1, 'cpu')).clearance_m
    torch.testing.assert_close(truth, torch.tensor([[1., -0.2, -0.5]]))
    for i in range(3):
        alone = teacher(paths[:, i:i+1], source_batch(1, 'cpu')).clearance_m
        torch.testing.assert_close(alone[:, 0], truth[:, i])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_cached_source_metadata_preserves_mixed_goal_labels(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    teacher=RouteUtilityTeacher(oracle(),3.6);batch=source_batch(2,device)
    batch['source_origin_xy']=torch.tensor([[.023,.041],[-.017,.013]],dtype=torch.float64,device=device)
    batch['source_yaw_rad']=torch.tensor([.73,-.27],dtype=torch.float64,device=device)
    batch['goal_present'][1]=False;batch['point_goal'][1]=0
    paths=torch.tensor([[[[0.,0.],[0.,2.]],[[0.,0.],[2.,1.]]]],device=device).expand(2,-1,-1,-1)
    reference=teacher(paths,batch)
    batch['source_metadata_cpu']=torch.cat((batch['source_grid_index'].double().cpu()[:,None],batch['source_origin_xy'].cpu(),batch['source_yaw_rad'].cpu()[:,None]),-1)
    cached=teacher(paths,batch)
    for field in ('score','clearance_m','progress_m','failed'):
        torch.testing.assert_close(getattr(cached,field),getattr(reference,field),rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_grouped_source_labels_keep_each_pose_goal_and_unpadded_candidates(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from kappanav.local_planner.training.bspline_objective import _LABEL_WORKER
    from reference_implementations import label_on_stream as _label_on_stream
    teacher=RouteUtilityTeacher(oracle(),3.6);calls=[]
    def utility(paths,batch):
        calls.append(tuple(paths.shape[:2]));return teacher(paths,batch)
    paths=[torch.tensor([[[0.,0.],[0.,3.]],[[0.,0.],[3.,0.]],[[0.,0.],[0.,-1.]]],device=device),
        torch.tensor([[[0.,0.],[0.,-2.]],[[0.,0.],[30.,0.]]],device=device)]
    prepared=[];expected=[]
    for row,bank in enumerate(paths):
        batch=source_batch(1,device)
        batch['source_origin_xy'][:,0]=row*.05
        batch['goal_present'][:]=row==0
        if row:batch['point_goal'].zero_()
        expected.append(teacher(bank[None],batch).score)
        prepared.append(dict(paths=bank,qbatch=batch,utility=utility,expert_count=1-row,recorded=bank[:0]))
    stream=torch.cuda.current_stream() if device=='cuda' else None
    _LABEL_WORKER.submit(_label_on_stream,prepared,stream).result()
    # Expert and current candidates are scored together, padded to the largest expert count.
    assert calls==[(2,3)]
    for p,target in zip(prepared,expected,strict=True):torch.testing.assert_close(p['targets'],target,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_source_utility_matches_separate_sources(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from reference_implementations import label_critic_inputs
    teachers=[RouteUtilityTeacher(oracle(),3.6),RouteUtilityTeacher(SourceConfigurationSpaceQuery((
        SourceConfigurationGrid(np.ones((100,120),np.float32),np.array([-2.,-3.]),.05),)),3.6)]
    paths=torch.tensor([[[0.,0.],[0.,1.]],[[0.,0.],[1.2,0.]],[[0.,0.],[0.,0.]]],device=device)
    expected=[];batches=[]
    for row,teacher in enumerate(teachers):
        batch=source_batch(1,device)
        batch['goal_present'][:]=row==0
        if row:batch['point_goal'].zero_()
        batches.append(batch);expected.append(teacher(paths[None],batch))
    shared=RouteUtilityTeacher(SourceConfigurationSpaceQuery(tuple(teacher.source_query.grids[0] for teacher in teachers)),3.6)
    prepared=[];calls=[]
    def utility(paths,batch):calls.append(len(paths));return shared(paths,batch)
    for row,batch in enumerate(batches):
        batch=dict(batch,source_grid_index=torch.tensor([row],device=device))
        prepared.append(dict(paths=paths,qbatch=batch,utility=utility,expert_count=0,recorded=paths[:0]))
    label_critic_inputs(prepared)
    assert calls==[2]
    for result,target in zip(prepared,expected,strict=True):
        torch.testing.assert_close(result['targets'],target.score,rtol=1e-6,atol=1e-6)



def test_execution_prefix_scale_matches_executor_and_early_failure_cost():
    from kappanav.local_planner.data.value_targets import EXECUTION_PREFIX_M
    assert EXECUTION_PREFIX_M>0
    teacher=RouteUtilityTeacher(oracle(),3.6)
    scores=teacher(torch.tensor([[[[0.,0.],[3.,0.]]]]),source_batch(1,'cpu'))
    assert scores.failed.item() and torch.isfinite(scores.score).all()


def test_score_is_goal_conditioned_and_selection_returns_exact_candidate():
    from test_policy import tiny_config, make_encoded_condition
    from kappanav.local_planner.model.factory import build_policy
    torch.manual_seed(9)
    policy = build_policy(tiny_config()).eval()
    encoded = make_encoded_condition(policy, 1)
    evaluator = policy.trajectory_evaluator
    memory = evaluator.project_condition_memory(encoded)
    path, _ = policy.curve_codec.decode(torch.randn(1, 14))
    score = evaluator(path, encoded.terminal_goal, encoded, memory)
    other = evaluator(path, -encoded.terminal_goal, encoded, memory)
    assert not torch.equal(score, other)
    from kappanav.local_planner.inference.bundle import LocalBSplineBundle
    prediction = LocalBSplineBundle(None, policy, None, {}, {}).candidates(encoded, encoded.terminal_goal)
    assert prediction["paths"].shape == (16, 64, 2)
    clear = prediction["clear"]
    allowed = prediction["scores"].masked_fill(~clear, float("-inf")) if clear.any() else prediction["scores"]
    assert prediction["selected_index"] == int(allowed.argmax())
    assert policy.inference_source.unique(dim=0).shape[0] == 16


def test_teacher_prefers_progress_to_stopping_and_does_not_credit_wall_crossing():
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    paths = torch.tensor([[[[0., 0.], [0., 0.]], [[0., 0.], [0., 2.]], [[0., 0.], [3., 2.]]]])
    labels = teacher(paths, batch)
    assert labels.score[0, 1] > labels.score[0, 0] > labels.score[0, 2]
    assert labels.clearance_m[0, 2] < 0
    # Ranking is independent of the other trajectories in a batch.
    for i in range(3):
        alone = teacher(paths[:, i:i+1], batch)
        torch.testing.assert_close(alone.score[:, 0], labels.score[:, i])


def test_teacher_ranking_follows_actual_goal_for_identical_safe_candidates():
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    paths = torch.tensor([[[[0., 0.], [0., 2.]], [[0., 0.], [0., -2.]]]])
    forward = teacher(paths, batch)
    backward = teacher(paths, {**batch, 'point_goal': -batch['point_goal']})
    assert forward.score.argmax(1).item() == 0
    assert backward.score.argmax(1).item() == 1
    torch.testing.assert_close(forward.clearance_m, backward.clearance_m)
    # Zero goal means arrival, not exploration: the utility favors stopping.
    stopped = torch.zeros_like(paths[:, :1])
    zero_goal = teacher(torch.cat((stopped, paths), dim=1),
                        {**batch, 'point_goal': torch.zeros_like(batch['point_goal'])})
    assert zero_goal.score.argmax(1).item() == 0


def test_geodesic_progress_rewards_moving_away_from_goal_to_exit_dead_end():
    # Wall separates start and goal except for an opening at world y > 2.
    clear = np.ones((100, 100), np.float32)
    clear[50:53, :70] = -0.1
    query = SourceConfigurationSpaceQuery((SourceConfigurationGrid(clear, np.array([-5., -5.]), .1),))
    teacher = RouteUtilityTeacher(query, 3.6)
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'][:] = torch.tensor([-1., 0.])
    batch['point_goal'][:] = torch.tensor([2., 0.])
    # The robot faces the exit (yaw +90 degrees; the goal is then local +y): the exit leaves straight ahead, away
    # from the goal in Euclidean terms, without a turn on the spot; the other path heads for the goal and the wall.
    batch['source_yaw_rad'] = batch['source_yaw_rad'] + np.pi / 2
    batch['point_goal'][:] = torch.tensor([0., 2.])
    paths = torch.tensor([[[[0., 0.], [1., 0.]], [[0., 0.], [.4, .5]]]])
    labels = teacher(paths, batch)
    assert labels.progress_m[0, 0] > 0
    assert labels.score[0, 0] > labels.score[0, 1]


def test_utility_is_invariant_to_robot_frame_rotation_apart_from_the_turn_on_the_spot():
    from kappanav.local_planner.data.value_targets import in_place_turn_m
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'][:] = torch.tensor([.023, .041])
    paths = torch.tensor([[[[0., 0.], [0., 2.]], [[0., 0.], [3., 1.]]]])
    expected = teacher(paths, batch)
    angle = torch.tensor(.73)
    rotation = torch.stack((torch.stack((angle.cos(), -angle.sin())),
                            torch.stack((angle.sin(), angle.cos()))))
    rotated = dict(batch)
    rotated['source_yaw_rad'] = batch['source_yaw_rad'] + angle
    rotated['point_goal'] = batch['point_goal'] @ rotation.T
    actual = teacher(paths @ rotation.T, rotated)
    # The world geometry is the same; only the heading the robot starts from differs.
    turn = lambda candidates: .5 * in_place_turn_m(candidates) / 3.6
    torch.testing.assert_close(actual.score + turn(paths @ rotation.T), expected.score + turn(paths), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual.progress_m, expected.progress_m)


def test_source_trace_detects_short_corner_crossing_and_preserves_vertices():
    clear = np.ones((20, 20), np.float32)
    clear[1, 1] = -.1
    query = SourceConfigurationSpaceQuery((SourceConfigurationGrid(clear, np.zeros(2), .1),))
    path = torch.tensor([[[0., 0.], [.101, .101]]])
    result = query.query(path, torch.tensor([0]), torch.tensor([[.05, .151]]), torch.tensor([0.]), 3.6)
    assert result.minimum_clearance_m.item() < 0
    assert ((result.grid_cells[0] == torch.tensor([1, 1])).all(-1) & result.active[0]).any()
    points = result.local_path[0, result.active[0]]
    torch.testing.assert_close(points[0], path[0, 0])
    torch.testing.assert_close(points[-1], path[0, -1])
    assert torch.diff(points, dim=0).norm(dim=-1).max() <= .025001


def test_closed_cell_contacts_agree_with_geodesic_connectivity():
    # All side-cell arrangements, both traversal directions, and exact edge contact.
    for side_x, side_y in ((.2, .2), (-.1, .2), (.2, -.1), (-.1, -.1)):
        grid = SourceConfigurationGrid(np.array([[.2, side_y], [side_x, .2]], np.float32), np.zeros(2), 1.)
        query = SourceConfigurationSpaceQuery((grid,))
        teacher = RouteUtilityTeacher(query, 3.6)
        for reverse in (False, True):
            batch = source_batch(1, 'cpu')
            batch['source_origin_xy'][:] = 1.5 if reverse else .5
            batch['point_goal'][:] = torch.tensor([[.1, 0.]])
            path = torch.tensor([[[[0., 0.], [-1., 1.] if reverse else [1., -1.]]]])
            result = teacher(path, batch)
            assert bool(result.clearance_m[0, 0] < 0) == (min(side_x, side_y) < 0)
            assert torch.isfinite(result.score).all()
        np.testing.assert_allclose(grid.query_world(np.array([[1., 1.]])), min(.2, side_x, side_y))
    # A segment lying on an occupied cell edge must also count as contact.
    path = torch.tensor([[[0., 0.], [0., -.5]]])
    result = query.query(path, torch.tensor([0]), torch.tensor([[1., .25]]), torch.tensor([0.]), 3.6)
    assert result.minimum_clearance_m < 0
    assert torch.all(result.clearance_m[result.active] < 0)


def test_teacher_reuses_queried_cells_at_large_origin_boundary():
    clear = np.ones((20, 20), np.float32)
    clear[9, :] = -.05
    query = SourceConfigurationSpaceQuery((SourceConfigurationGrid(clear, np.array([1000.00003, -1000.00003]), .05),))
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'][:] = torch.tensor([[1000.75, -999.75]])
    batch['point_goal'][:] = torch.tensor([[.1, 0.]])
    paths = torch.tensor([[[[0., 0.], [-.25, 0.]], [[0., 0.], [.1, 0.]], [[0., 0.], [0., 0.]]]])
    result, _ = candidate_query(query, paths, batch, 3.6)
    cells = query.point_cells(paths.flatten(0, 1)[:, :1], batch['source_grid_index'].repeat_interleave(3),
                             batch['source_origin_xy'].repeat_interleave(3, 0), batch['source_yaw_rad'].repeat_interleave(3))
    torch.testing.assert_close(result.grid_cells[:, :1], cells)
    labels = RouteUtilityTeacher(query, 3.6)(paths, batch)
    assert torch.isfinite(labels.score).all()
    assert labels.clearance_m[0, 0] < 0
    assert labels.clearance_m[0, 1] > 0


def test_packed_scene_queries_match_separate_maps_with_different_shapes_and_scales():
    grids = (
        SourceConfigurationGrid(np.ones((20, 30), np.float32), np.array([-.5, -.7]), .05),
        SourceConfigurationGrid(np.ones((40, 25), np.float32), np.array([-1., -.8]), .1),
    )
    grids[0].signed_clearance_m[13, :] = -.05
    grids[1].signed_clearance_m[15, :] = -.1
    path = torch.tensor([[[0., 0.], [.6, .1]], [[0., 0.], [1.5, -.3]], [[0., 0.], [0., 0.]]])
    indices = torch.tensor([1, 0, 1])
    origins, yaws = torch.zeros(3, 2), torch.zeros(3)
    result = SourceConfigurationSpaceQuery(grids).query(path, indices, origins, yaws, 3.6)
    for row, index in enumerate(indices.tolist()):
        one = SourceConfigurationSpaceQuery((grids[index],)).query(
            path[row:row+1], torch.zeros(1, dtype=torch.long), origins[row:row+1], yaws[row:row+1], 3.6,
        )
        for name in ("local_path", "grid_cells", "clearance_m", "in_world_bounds"):
            torch.testing.assert_close(getattr(result, name)[row, result.active[row]],
                                       getattr(one, name)[0, one.active[0]], rtol=0, atol=0)


def test_explicit_xy_grid_frame_preserves_asymmetric_obstacle_and_utility(tmp_path):
    # Positive world Y contains a wall; negative Y is free. Mirroring changes Q labels.
    clear = np.ones((30, 30), np.float32)
    clear[8:13, 14:20] = -.1
    path = tmp_path / 'canonical_xy.npz'
    np.savez(path, free=clear > 0, clearance_m=clear, origin_xy=np.array([-1., -1.]),
             cell_size_m=.1, coordinate_frame='canonical_world_xy')
    query = SourceConfigurationSpaceQuery.from_paths((path,))
    assert query.grids[0].coordinate_frame == 'canonical_world_xy'
    origin = torch.tensor([[.05, .05]], dtype=torch.float64)
    local = torch.tensor([[[0., 0.], [0., .8], [.3, -.6]]], dtype=torch.float64)
    for yaw in (0., .73, -1.9):
        angle = torch.tensor([yaw], dtype=torch.float64)
        rotation = torch.tensor([[np.cos(yaw), -np.sin(yaw)],
                                 [np.sin(yaw), np.cos(yaw)]], dtype=torch.float64)
        expected_world = local @ rotation.T + origin[:, None]
        actual = query.grid_coordinates(local, torch.tensor([0]), origin, angle)
        torch.testing.assert_close(actual, (expected_world + 1) / .1)
    paths = torch.tensor([[[[0., 0.], [0., .8]], [[0., 0.], [0., -.5]]]])
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'] = origin
    batch['point_goal'] = torch.tensor([[0., -.5]])
    labels = RouteUtilityTeacher(query, 3.6)(paths, batch)
    assert labels.clearance_m[0, 0] < 0 < labels.clearance_m[0, 1]
    assert labels.score[0, 1] > labels.score[0, 0]
    # Source files without metadata retain the original Habitat XZ convention.
    legacy_path = tmp_path / 'legacy_xz.npz'
    np.savez(legacy_path, free=clear > 0, clearance_m=clear, origin_xy=np.array([-1., -1.]),cell_size_m=.1)
    legacy = SourceConfigurationSpaceQuery.from_paths((legacy_path,))
    mirrored = legacy.grid_coordinates(local, torch.tensor([0]), origin, torch.zeros(1,dtype=torch.float64))
    torch.testing.assert_close(mirrored[..., 1], (origin[:, None, 1] - local[..., 1] + 1) / .1)


def test_exploration_labels_ignore_hidden_goals_reward_motion_and_penalize_loops():
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    batch['goal_present'][:] = False
    paths = torch.tensor([[[[0., 0.], [0., 0.], [0., 0.]],
                           [[0., 0.], [0., 1.], [0., 2.]],
                           [[0., 0.], [0., -1.], [0., -2.]],
                           [[0., 0.], [0., 1.], [0., 0.]],
                           [[0., 0.], [1.5, 0.], [3., 0.]]]])
    labels = teacher(paths, batch)
    # Even a hidden goal outside the world must be completely irrelevant.
    changed = teacher(paths, {**batch, 'point_goal': torch.full((1, 2), 1000.)})
    torch.testing.assert_close(labels.score, changed.score, rtol=0, atol=0)
    assert labels.score[0, 1] > labels.score[0, 0] > labels.score[0, 3]
    assert labels.score[0, 1] > labels.score[0, 4]
    torch.testing.assert_close(labels.score[:, 1], labels.score[:, 2])
    goal_batch = {**batch, 'goal_present': torch.ones(1, dtype=torch.bool), 'point_goal': torch.zeros(1, 2)}
    assert teacher(paths, goal_batch).score.argmax(1).item() == 0


def test_novelty_beyond_first_collision_cannot_raise_exploration_utility():
    teacher=RouteUtilityTeacher(oracle(),3.6)
    batch=source_batch(1,'cpu');batch['goal_present'][:]=False
    path=torch.tensor([[[[0.,0.],[1.5,0.],[3.,0.]]]])
    known=teacher(path,batch)
    behind=batch['observed_field'].clone()
    axis=torch.linspace(-3.6,3.6,64)
    behind[:,3,:,axis>1.5]=0
    unreachable=teacher(path,{**batch,'observed_field':behind})
    torch.testing.assert_close(unreachable.score,known.score,rtol=0,atol=0)
    unknown=teacher(path,{**batch,'observed_field':torch.zeros_like(behind)})
    assert unknown.score>known.score


def test_in_place_turn_follows_the_controller_turn_rule():
    from kappanav.local_planner.data.value_targets import in_place_turn_m
    from kappanav.local_planner.control.mpc import (TURN_ENTER_RAD, TURN_EXIT_RAD, MAX_ANGULAR_SPEED_RAD_S,
                                                    MAX_LINEAR_SPEED_MPS)
    def leaving(angle, length=1.):
        return torch.tensor([[0., 0.], [length*np.cos(angle), length*np.sin(angle)]], dtype=torch.float32)
    below, above, behind = TURN_ENTER_RAD*.95, TURN_ENTER_RAD*1.05, np.pi*.9
    paths = torch.stack((leaving(0.), leaving(below), leaving(-above), leaving(behind), leaving(behind, 0.)))[None]
    turn = in_place_turn_m(paths)[0]
    scale = MAX_LINEAR_SPEED_MPS / MAX_ANGULAR_SPEED_RAD_S
    torch.testing.assert_close(turn, torch.tensor([0., 0., (above-TURN_EXIT_RAD)*scale, (behind-TURN_EXIT_RAD)*scale, 0.], dtype=torch.float32), atol=1e-5, rtol=1e-5)
    # Measured at TURN_LOOKAHEAD_M along the path, as the controller does: a path that bends back later is forward.
    hook = torch.tensor([[[[0., 0.], [.5, 0.], [.5, .5], [0., .5]]]])
    assert in_place_turn_m(hook)[0, 0] == 0


def test_turning_on_the_spot_costs_against_an_equally_safe_forward_path():
    teacher = RouteUtilityTeacher(oracle(), 3.6)
    batch = source_batch(1, 'cpu')
    batch['source_origin_xy'][:] = torch.tensor([.023, .041])
    batch['point_goal'][:] = torch.tensor([0., 3.])
    # The goal is to the left: leaving forward and curving left vs a stub straight back of the same length.
    forward = torch.tensor([[0., 0.], [.3, 0.], [.4, .3]])
    back = torch.tensor([[0., 0.], [-.3, 0.], [-.4, .3]])
    labels = teacher(torch.stack((forward, back))[None], batch)
    assert labels.score[0, 0] > labels.score[0, 1]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('oracles',[1,2])
def test_prepared_label_targets_equal_the_per_row_reference_bit_for_bit(device,oracles):
    """The loader-prepared critic inputs score the same banks as label_critic_inputs and give identical targets,
    candidate weights and source coordinates."""
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from torch.nn.utils.rnn import pad_sequence
    from kappanav.local_planner.features.staging import HostTree
    from kappanav.local_planner.training import bspline_objective as objective
    from kappanav.local_planner.data.replay_targets import critic_source_inputs, critic_sources
    generator=torch.Generator().manual_seed(5)
    teachers=[RouteUtilityTeacher(oracle(),3.6) for _ in range(oracles)]
    walk=lambda count:(torch.randn(count,64,2,generator=generator)*.05).cumsum(1)
    rows=[];utilities=[]
    for index,experts in enumerate([2,0,3,1,0]):
        rows.append(dict(q_supervision_available=True,history_plan=dict(points=range(index+1)),source_grid_index=0,view=np.array([.02*index,-.03*index,.05*index],dtype=np.float32),
            goal=torch.tensor([0.,2.+index*.3]),local_target_field=torch.rand(6,64,64,generator=generator),
            q_expert_paths=walk(experts),critic_replay_paths=walk(4 if index%2 else 3)))
        utilities.append(teachers[index%oracles])
    native=walk(16*len(rows)).reshape(len(rows),16,64,2).to(device)
    present=torch.tensor([True,False,True,True,False],device=device)
    fields=torch.stack([row['local_target_field'] for row in rows]).to(device)
    reference=[]
    for i,row in enumerate(rows):
        reference.append(reference_implementations.prepare_critic_inputs(row,present[i:i+1,None].float(),native[i],utilities[i],
            [row[name].to(device) for name in reference_implementations.CRITIC_PATHS]))
    reference_implementations.label_critic_inputs(reference)
    expected=torch.cat((pad_sequence([p['targets'][0,:p['expert_count']] for p in reference],batch_first=True),
        torch.stack([p['targets'][0,p['expert_count']:] for p in reference]),
        pad_sequence([p['recorded_targets'][0] for p in reference],batch_first=True)),1)
    host=HostTree(objective.critic_inputs(rows,utilities)).to(device)
    shape=(host['experts'].shape[1],host['recorded'].shape[1])
    actual=objective.label_critic_targets(host['labels'],native,present,fields,shape)
    actual=torch.where(host['target_mask'],actual,torch.zeros((),dtype=actual.dtype,device=device))
    assert torch.equal(actual,expected)
    # The candidate weights and source coordinates of the per-row loop.
    for row,weights in zip(rows,host['weights'].cpu(),strict=True):
        count,replay=len(row['q_expert_paths']),len(row['critic_replay_paths']);factor=.75 if count else 1.
        weight=np.zeros(shape[0]+16+shape[1],dtype=np.float32)
        if count:weight[:count]=.25/count
        weight[shape[0]:shape[0]+16]=factor/32;weight[shape[0]+16:shape[0]+16+replay]=factor/(2*replay)
        assert np.array_equal(weights.numpy(),weight)
    # The per-row source coordinates (the formula the batched critic_sources replaced), concatenated.
    sources=[dict(source_grid_index=torch.tensor([row['source_grid_index']],dtype=torch.long),
        source_metadata_cpu=torch.as_tensor(np.r_[row['source_grid_index'],row['view']][None],dtype=torch.float64),
        source_origin_xy=torch.as_tensor(row['view'][:2],dtype=torch.float64)[None],
        source_yaw_rad=torch.tensor([row['view'][2]],dtype=torch.float64),point_goal=row['goal'][None]) for row in rows]
    for key,value in critic_sources(rows).items():
        expected=torch.cat([source[key] for source in sources])
        assert value.dtype==expected.dtype and torch.equal(value,expected)
    single=critic_source_inputs(rows[0])
    assert torch.equal(single['observed_field'],rows[0]['local_target_field'][None])
