"""Canonical array replay is self-contained and shares physical states."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import numpy as np
import torch

from kappanav.local_planner.data import replay_pack
from kappanav.local_planner.data.capture import save_capture, load_capture
from kappanav.local_planner.data.array_store import write_columns
from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
from kappanav.local_planner.data.value_targets import RouteUtilityTeacher
from kappanav.common.experiment_io import write
from kappanav.local_planner.model.trajectory import IncrementalBSplineTrajectory


def array_actor():
    codec=IncrementalBSplineTrajectory(num_control_points=8,degree=3,num_path_points=64,
        control_increment_mean_xy_m=(0.,)*14,control_increment_std_xy_m=(1.,)*14)
    return SimpleNamespace(policy=SimpleNamespace(curve_codec=codec,planning_horizon_m=3.6),
                           head=SimpleNamespace(radius=.167584539),memory=None)


def make_array_replay(folder):
    actor=array_actor();folder=Path(folder);folder.mkdir(parents=True)
    original=folder/'capture';original.mkdir()
    arrays=dict(depths=np.arange(160,dtype=np.float16).reshape(2,8,10),
        noncontiguous=np.arange(12,dtype=np.float64).reshape(3,4).T,
        tensor=np.arange(10,dtype=np.float32)[2:7])
    save_capture(original/'capture.json',arrays,dict(case=dict(goal=[1.,2.])))
    raw=load_capture(original/'capture.json')
    plan=dict(frame_ids=np.array([0,1,1]),valid=np.array([True,False]))
    grid=SourceConfigurationGrid(np.ones((5,7),np.float32),np.zeros(2),.05,'canonical_world_xy')
    utility=RouteUtilityTeacher(SourceConfigurationSpaceQuery((grid,)),3.6,actor.head.radius)
    field=torch.ones(6,64,64)
    rows=[(dict(raw_path=str(original/'capture.json'),history_plan_path='plan',history_plan=plan,
        local_target_field=field,raw_sha256='raw',trial_id='scene',goal_present=mode,
        source_grid_sha256='grid',
        teacher_values=torch.zeros(14),cfm_target=torch.zeros(1,14),trajectory_target=torch.zeros(1,64,2),
        q_supervision_available=True,
        critic_replay_paths=torch.zeros(16,64,2),source_grid_index=0,
        q_expert_paths=torch.full((1,64,2),7.),
        _replay_sha256='row-binding'),raw,None,utility) for mode in (True,False)]
    replay_pack.write_shard(folder/'shard',rows)
    catalog=write_columns(folder/'catalog',dict(shard=np.zeros(2,np.int32),row=np.arange(2,dtype=np.int64),
        trial=np.zeros(2,np.int32),raw=np.zeros(2,np.int32),goal=np.array([True,False])))
    manifest=folder/'manifest.json'
    write(manifest,dict(format=replay_pack.FORMAT,split='train',shards=[dict(path='shard',rows=2)],
        catalog_path='catalog',catalog=catalog,trials=['scene'],raw_captures=['raw'],
        trajectory_contract=replay_pack.trajectory_contract(actor)))
    return actor,manifest,raw,field


def test_array_replay_roundtrip_is_self_contained(tmp_path,monkeypatch):
    actor,manifest,raw,field=make_array_replay(tmp_path/'data')
    def forbidden(*args,**kwargs):raise AssertionError('Training deserialized per-row tensor data')
    monkeypatch.setattr(torch,'load',forbidden)
    result=replay_pack.load_array_replay(actor,[manifest],split='train');a,b=result[0]
    assert a[1] is b[1] and a[0]['history_plan'] is b[0]['history_plan']
    assert a[0]['local_target_field'] is b[0]['local_target_field']
    assert a[3] is b[3] and a[0]['q_expert_paths'][0,0,0].item()==7.
    np.testing.assert_array_equal(a[1]['depths'],raw['depths']);assert a[1]['depths'].dtype==raw['depths'].dtype
    torch.testing.assert_close(a[0]['local_target_field'],field,rtol=0,atol=0)
    assert result[0].sources[0]==result[0].sources[1]==frozenset({('raw',0),('raw',1)})
    assert result[1].tolist()==[0] and result[2].tolist()==[1]
    # The snapshot is self-contained: it does not read the source captures.
    shutil.rmtree(manifest.parent/'capture')
    assert replay_pack.load_array_replay(actor,[manifest],split='train')[0][0][0]['q_expert_paths'][0,0,0].item()==7.


def test_parallel_shards_preserve_catalog_order_and_grid_offsets(tmp_path):
    import json
    actor,manifest,_,_=make_array_replay(tmp_path/'data')
    metadata=json.loads(manifest.read_text())
    metadata['shards']*=2
    order=np.array([1,0,1,0],np.int32)
    row=np.array([0,1,1,0],np.int64)
    metadata['catalog_path']='interleaved'
    metadata['catalog']=write_columns(manifest.parent/'interleaved',dict(
        shard=order,row=row,trial=np.zeros(4,np.int32),raw=np.zeros(4,np.int32),goal=row==0))
    manifest=manifest.with_name('interleaved.json')
    write(manifest,metadata)
    replay=replay_pack.load_array_replay(actor,[manifest],split='train')[0]
    assert [record[0]['source_grid_index'] for record in replay]==order.tolist()
    assert [record[0]['goal_present'] for record in replay]==[True,False,False,True]
    assert len(replay.utility.source_query.grids)==2
