import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from kappanav.evaluation.benchmark.protocol import decision_steps
from kappanav.evaluation.benchmark.prepare import benchmark_episodes, visible_clearance_m
from kappanav.evaluation.benchmark.results import write_summary, read_summary, paired
from kappanav.evaluation.benchmark import isaac


def grid_file(tmp_path):
    path=tmp_path/'navigation.npz'
    np.savez(path,free=np.ones((100,100),bool),clearance_m=np.ones((100,100)),origin_xy=(-5.,-5.),cell_size_m=.1)
    return path


def test_full_benchmark_preserves_all_official_pairs_in_file_order(tmp_path):
    path=grid_file(tmp_path)
    pairs=np.random.default_rng(31).normal(size=(100,5))
    episodes=[e for i in range(40) for e in benchmark_episodes(dict(scene_id=f'scene_{i}',navigation=str(path)),pairs,start_clearance_m=.3)]
    assert len(episodes)==4000 and len({e['episode_id'] for e in episodes})==4000
    for scene in range(40):
        for index,e in enumerate(episodes[scene*100:(scene+1)*100]):
            np.testing.assert_array_equal([*e['start'][:2],*e['goal'],e['start'][2]],pairs[index])
            assert e['episode_index']==index and e['goal_budget_s']==122. and not e['collision_ends']
            assert e['arrival_hold_s']==4.
    with pytest.raises(ValueError,match='100 finite'):
        benchmark_episodes(dict(scene_id='bad',navigation=str(path)),pairs[:99],start_clearance_m=.3)
    pairs[3,0]=np.nan
    with pytest.raises(ValueError,match='100 finite'):
        benchmark_episodes(dict(scene_id='bad',navigation=str(path)),pairs,start_clearance_m=.3)


@pytest.mark.dingo
def test_a_moved_start_sees_its_surroundings_and_a_moved_goal_lies_in_the_cspace(tmp_path):
    from kappanav.evaluation.simulation.navigation import save_navigation_grid, NavigationGrid
    from kappanav.common.physical import TURN_CENTRE_FORWARD_OFFSET_M as D
    # the lowest body-band point enters the image 0.52 m ahead of base_link: 0.356 m of contact distance
    margin=visible_clearance_m([[114.2394943,0.,112.],[0.,114.2394943,63.],[0.,0.,1.]]);assert margin==pytest.approx(.356,abs=1e-3)
    free=np.ones((100,100),bool);free[40:60,40:60]=False;free[80,:]=False  # a block, and a wall cutting off x>=3 m
    path=tmp_path/'navigation.npz';save_navigation_grid(path,free,np.array([-5.,-5.]),.1)
    pairs=np.zeros((100,5));pairs[:]=[-4.,-4.,-3.,-3.,0.]
    pairs[0]=[-.15,.05,-3.,-3.,0.]   # start inside the block
    pairs[1]=[-4.,-4.,3.02,.03,0.]   # goal inside the wall: the cell on the start's side, not the far side
    pairs[2]=[-1.05,.05,-3.,-3.,0.]  # start against the block, facing it: in the C-space but no room to turn
    pairs[3]=[-1.15,.05,-3.,-3.,0.]  # room to turn in place: kept, although the block is within the hidden range
    episodes=benchmark_episodes(dict(scene_id='s',navigation=str(path)),pairs,start_clearance_m=margin);navigation=NavigationGrid(path)
    clearance=lambda p:navigation.clearance(np.repeat(np.asarray(p,float)[None],2,0))
    for episode,moved in ((episodes[0],1.3),(episodes[2],.4)):
        start=np.array(episode['start'][:2]);np.testing.assert_allclose(start,[-1.45,.05],atol=1e-9)
        assert clearance(start)>=margin and episode['endpoint_moved_m'][0]==pytest.approx(moved) and episode['start'][2]==0.
    assert episodes[3]['endpoint_moved_m']==[0.,0.] and clearance(np.array([-1.15+D,.05]))>=D
    goal=np.array(episodes[1]['goal']);np.testing.assert_allclose(goal,[2.95,.05],atol=1e-9);assert clearance(goal)>=0
    assert episodes[0]['official']==list(pairs[0]) and episodes[4]['endpoint_moved_m']==[0.,0.]
    assert episodes[0]['spl_distance_m']==pytest.approx(np.linalg.norm(np.array(episodes[0]['start'][:2])-[-3.,-3.]))


@pytest.mark.parametrize('latency,steps',[(.1,1),(.2,2),(.3,3)])
def test_decision_latency_lands_on_the_same_controller_tick(latency,steps):
    assert decision_steps(latency,.1)==steps


@pytest.mark.parametrize('latency',[0.,-.1,.15,float('nan'),float('inf')])
def test_invalid_decision_latency_is_not_silently_rounded(latency):
    with pytest.raises(ValueError):decision_steps(latency,.1)


def result(key,success=True,status=None):
    return dict(episode_id=key,source='benchmark',scene_id=key.split('/')[0],goal=dict(success=success,spl=float(success)),
        goal_collision=False,goal_status=status or ('goal_reached' if success else 'budget_exhausted'),goal_trajectory=None)


def test_shared_summary_keeps_failures_orders_tasks_and_rejects_missing_results(tmp_path):
    rows=[result('scene/goal_001',False),result('scene/goal_000')]
    episodes=[dict(episode_id=r['episode_id']) for r in rows[::-1]]
    summary=write_summary(tmp_path,rows,episodes)
    assert summary['complete'] and summary['groups']['all']['episodes']==2
    assert summary['groups']['benchmark']['success_rate']==.5
    assert [r['episode_id'] for r in summary['rows']]==[e['episode_id'] for e in episodes]
    with pytest.raises(ValueError,match='every requested episode exactly once'):
        write_summary(tmp_path,rows[:1],episodes)
    with pytest.raises(ValueError,match='every requested episode exactly once'):
        write_summary(tmp_path,[rows[0],rows[0]],episodes)


def test_comparison_rejects_smoke_runs_and_different_task_sets(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b';a.mkdir();b.mkdir()
    rows=[result('scene/goal_000'),result('scene/goal_001',False)]
    episodes=[dict(episode_id=r['episode_id']) for r in rows]
    write_summary(a,rows,episodes)
    write_summary(b,[result(r['episode_id']) for r in rows],episodes)
    assert read_summary(a)['groups']['all']['success_rate']==.5
    assert paired(a,b)['success_rate_difference']==-.5 and paired(a,b)['episodes_lost']==1
    write_summary(b,rows[:1],episodes[:1])
    with pytest.raises(ValueError,match='different task sets'):paired(a,b)
    write_summary(b,[result('scene/goal_000',False,'smoke_limit')],episodes[:1])
    with pytest.raises(ValueError,match='Incomplete evaluation'):read_summary(b)


def test_level_body_frame_preserves_the_true_camera_world_pose():
    body=np.eye(4);body[:3,:3]=Rotation.from_euler('zyx',[.8,.12,-.03]).as_matrix();body[:3,3]=[1.,2.,.08]
    mount=np.eye(4);mount[:3,:3]=Rotation.from_euler('y',.17).as_matrix();mount[:3,3]=[.143,0.,.313]
    level=isaac.level_body_pose(body[None])[0]
    camera=np.linalg.inv(level.astype(np.float64))@body@mount
    np.testing.assert_allclose(level@camera,body@mount,atol=1e-7)
    np.testing.assert_array_equal(level[:3,2],[0.,0.,1.])
    np.testing.assert_allclose(level[:3,3],body[:3,3])


@pytest.mark.parametrize('finish', ['goal_reached','budget_exhausted','smoke_limit'])
def test_isaac_episode_uses_batch_one_and_delayed_publication(tmp_path,monkeypatch,finish):
    from kappanav.evaluation.simulation.navigation import NavigationGrid
    camera=SimpleNamespace(data=SimpleNamespace(output={'distance_to_image_plane':torch.ones(1,8,12,1)},
        intrinsic_matrices=torch.tensor([[[8.,0.,6.],[0.,8.,4.],[0.,0.,1.]]])))
    robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor([[1.,0.,0.,0.]]),
        root_pos_w=torch.zeros(1,3),root_lin_vel_w=torch.tensor([[.1,0.,0.]])))
    class Env:
        step_dt=.1;steps=0
        def __init__(self):self.robot,self.camera=robot,camera;self.contact=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(1,1,3)))
        def step(self,command):
            assert not torch.is_inference_mode_enabled()
            self.steps+=1;robot.data.root_pos_w[0,0]=self.steps*.01
            return finish if self.steps==6 and finish!='smoke_limit' else 'running'
    class History:
        def update(self,*args):self.selection={'indices':[0,0,0,0]}
    class Actor:
        def new_history(self):return History()
        def step(self,history,goal,present):
            assert present
            path=torch.stack((torch.linspace(0.,1.,64),torch.zeros(64)),-1).repeat(16,1,1)
            return dict(paths=path,values=torch.zeros(16,8),scores=torch.zeros(16),clear=torch.ones(16,dtype=torch.bool),selected_index=0)
    published=[]
    controller=SimpleNamespace(metadata={'controller':'test'},publish=lambda route,view:published.append(view),next_command=lambda state,time_s:(np.array([.1,0.]),{}))
    monkeypatch.setattr(torch.cuda,'device',lambda *args:nullcontext())
    args=SimpleNamespace(policy_device='cpu',decision_latency_s=.2,max_steps=6 if finish=='smoke_limit' else None)
    episode=dict(episode_id='scene/goal_003',scene_id='scene',goal=[1.,0.],spl_distance_m=1.)
    row=isaac.run_episode(SimpleNamespace(is_running=lambda:True),Env(),Actor(),controller,NavigationGrid(grid_file(tmp_path)),np.eye(4),episode,args,tmp_path)
    assert row['goal_status']==finish and row['goal']['success']==(finish=='goal_reached')
    assert row['goal_decision_latency_ms']==pytest.approx(200.)
    execution=json.loads((tmp_path/'goal_execution.json').read_text())
    applied=[p for p in execution['plans'] if p['applied']]
    assert applied[0]['time_s']==applied[0]['observation_time_s']
    assert all(p['time_s']-p['observation_time_s']==pytest.approx(.2) for p in applied[1:])
    assert not execution['plans'][-1]['applied']
    assert row['goal_trajectory']['published_decisions']==len(applied)==len(published)


def test_isaac_exploration_runs_expert_then_goal_free_policy(tmp_path,monkeypatch):
    from kappanav.evaluation.simulation.navigation import NavigationGrid
    camera=SimpleNamespace(data=SimpleNamespace(output={'distance_to_image_plane':torch.ones(1,8,12,1)},
        intrinsic_matrices=torch.tensor([[[8.,0.,6.],[0.,8.,4.],[0.,0.,1.]]])))
    robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor([[1.,0.,0.,0.]]),
        root_pos_w=torch.zeros(1,3),root_lin_vel_w=torch.tensor([[.1,0.,0.]])))
    resets=[]
    class Env:
        step_dt=.1
        def __init__(self):self.robot,self.camera=robot,camera;self.contact=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(1,1,3)))
        def reset(self,start,goal,budget_s):resets.append((goal,budget_s));self.steps=0;robot.data.root_pos_w.zero_()
        def step(self,command):
            self.steps+=1;robot.data.root_pos_w[0,0]=self.steps*.01
            return 'budget_exhausted' if self.steps==6 else 'running'
    class History:
        def update(self,*args):self.selection={'indices':[0,0,0,0]}
    goals=[]
    class Actor:
        def new_history(self):return History()
        def step(self,history,goal,present):
            goals.append((goal.tolist(),present))
            path=torch.stack((torch.linspace(0.,1.,64),torch.zeros(64)),-1).repeat(16,1,1)
            return dict(paths=path,values=torch.zeros(16,8),scores=torch.zeros(16),clear=torch.ones(16,dtype=torch.bool),selected_index=0)
    routes=[]
    controller=SimpleNamespace(metadata={'controller':'test'},reset=lambda:None,publish=lambda route,view:routes.append(route),
        next_command=lambda state,time_s:(np.array([.1,0.]),{}))
    monkeypatch.setattr(torch.cuda,'device',lambda *args:nullcontext())
    args=SimpleNamespace(policy_device='cpu',decision_latency_s=.2,max_steps=None)
    episode=dict(episode_id='scene/goal_003',scene_id='scene',start=[0.,0.,0.],goal=[1.,0.],exploration_budget_s=.6,reachable_area_m2=100.)
    row=isaac.run_exploration(SimpleNamespace(is_running=lambda:True),Env(),Actor(),controller,NavigationGrid(grid_file(tmp_path)),np.eye(4),episode,args,tmp_path)
    assert resets==[(None,.6),(None,.6)]
    assert goals and all(goal==[0.,0.] and not present for goal,present in goals)
    assert row['expert_status']=='budget_exhausted' and row['exploration_status']=='budget_exhausted'
    assert set(row['exploration'])>={'coverage','exploration_spl','expert_valid'}
    for name in ('expert_execution.json','expert_trace.npz','exploration_execution.json','exploration_candidates.npz','exploration_curve.npz'):
        assert (tmp_path/name).is_file()


def test_isaac_entry_runs_each_scene_once_and_writes_one_summary(tmp_path,monkeypatch):
    scenes=[dict(scene_id=f'scene_{i}',asset_scene_id=f'scene_{i}') for i in range(2)]
    episodes=[dict(episode_id=f'{s["scene_id"]}/goal_{j:03d}',scene_id=s['scene_id']) for s in scenes for j in (0,99)]
    catalog=tmp_path/'tasks.json';catalog.write_text(json.dumps(dict(scenes=scenes,episodes=episodes)))
    checkpoint=tmp_path/'policy.pt';checkpoint.write_bytes(b'checkpoint')
    output=tmp_path/'run';calls=[]
    def evaluate(args,row,tasks,folder):
        folder.mkdir();calls.append([t['episode_id'] for t in tasks])
        (folder/'rows.json').write_text(json.dumps([result(t['episode_id'],t['episode_id'].endswith('000')) for t in tasks[::-1]]))
    class Process:
        exitcode=0
        def __init__(self,*,target,args):self.target,self.args=target,args
        def start(self):self.target(*self.args)
        def join(self):pass
    monkeypatch.setattr(isaac,'evaluate_scene',evaluate)
    monkeypatch.setattr(isaac.mp,'get_context',lambda _:SimpleNamespace(Process=Process))
    # This entry-point fake finishes synchronously; real process readiness has its own regression test.
    monkeypatch.setattr(isaac,'take_completed_scene',lambda running:running.pop(0))
    monkeypatch.setattr('sys.argv',['isaac','--benchmark',str(catalog),'--checkpoint',str(checkpoint),'--output',str(output),'--scenes',str(tmp_path),'--robot',str(tmp_path/'dingo.usd')])
    isaac.main()
    assert len(calls)==2 and sum(map(len,calls))==4
    summary=json.loads((output/'summary.json').read_text())
    assert summary['complete'] and summary['decision_latency_s']==.2 and summary['backend']=='isaac'
    assert [r['episode_id'] for r in summary['rows']]==[e['episode_id'] for e in episodes]
    assert summary['groups']['all']['episodes']==4 and summary['groups']['all']['success_rate']==.5


def test_latency_summary_excludes_unpublished_tail_and_immediate_first_plan():
    from kappanav.evaluation.benchmark.metrics import mean_decision_latency_ms
    plans=[dict(applied=True),dict(applied=True,decision_latency_ms=200.),dict(applied=False,decision_latency_ms=100.)]
    assert mean_decision_latency_ms(plans)==200.
    assert mean_decision_latency_ms(plans[:1]) is None


def test_habitat_entry_runs_one_pass(tmp_path,monkeypatch):
    from kappanav.evaluation.benchmark import run
    calls=[]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(run,'evaluate',lambda args:calls.append(args) or dict(groups=dict(all=dict(episodes=4000))))
    monkeypatch.setattr('sys.argv',['run','--output',str(tmp_path/'result')])
    run.main()
    assert len(calls)==1 and calls[0].output==tmp_path/'result'
