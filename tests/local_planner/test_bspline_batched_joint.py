"""FP32 sample-mean oracle for actual batched native14 kernels and source contracts."""
from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy
import torch,numpy as np,pytest
from kappanav.local_planner import build_policy
from kappanav.local_planner.model.types import ConditionFeatures
from kappanav.local_planner.inference.bundle import LocalBSplineBundle
from kappanav.local_planner.training import bspline_objective as batched
from kappanav.local_planner.data.replay_targets import prepare_replay_targets, prepare_generator_targets
from kappanav.local_planner.features import conditions as conditioning
import reference_implementations as references
from kappanav.local_planner.model.network.policy import repeat_condition
from torch.nn import functional as F
from kappanav.local_planner.training.bspline_replay import (batch_indices, restore_full, full_state, epoch_budget,
                                                            save_checkpoint)
from kappanav.local_planner.data.bspline_dataset import case_groups, sample_index
from kappanav.local_planner.training.bspline_replay import capture_rng
from test_policy import tiny_config


def scalar_reference(actor,item,raw,tree,utility,draws):
    encoded,geometry=scalar.live_condition(actor,item,raw)
    cfm=encoded.tokens.new_zeros(())
    if item['teacher_values'] is not None:
        condition=encoded;teacher=item['teacher_values']
        clean=actor.policy.curve_codec.coordinates_from_values(teacher[None]);z=draws['noise'];time=draws['time']
        state=(1-time[:,None])*clean+time[:,None]*z
        velocity=actor.policy._predict_velocity(state,time,repeat_condition(condition,2),actor.policy.trajectory_decoder.project_condition_memory(condition))
        cfm=(velocity-(z-clean)).square().mean()
    bank={k:v[0] for k,v in actor.native_bank(encoded,source=draws['source'][None]).items()}
    gf,_=scalar.observed_risk(bank['values'],tree,item['point_error_m'],actor.head.radius)
    lh=scalar.geometry_reconstruction_loss(*geometry,item['local_target_field'][None])
    lq=encoded.tokens.new_zeros(())
    if item['q_supervision_available']:
        qbatch=dict(source_grid_index=torch.zeros(1,dtype=torch.long),source_origin_xy=torch.tensor(item['view'][:2],dtype=torch.float64)[None],source_yaw_rad=torch.tensor([item['view'][2]],dtype=torch.float64),point_goal=item['goal'][None],goal_present=encoded.goal_present[:,0].bool())
        n=len(item['q_expert_values']);paths=bank['paths']
        if n:
            expert,_=actor.policy.curve_codec.decode_values(item['q_expert_values']);paths=torch.cat((expert,paths))
        paths=paths.detach();target=utility(paths[None],qbatch).score
        recorded=item['critic_replay_paths'].detach();rt=utility(recorded[None],qbatch).score
        score=torch.cat((actor.policy.score_candidate_paths(encoded,item['goal'][None],paths[None]),
            actor.policy.score_candidate_paths(encoded,item['goal'][None],recorded[None])),-1)
        target=torch.cat((target,rt),-1)
        factor=.75 if n else 1.
        weight=torch.cat((torch.full((n,),.25/n) if n else torch.empty(0),
            torch.full((16,),factor/32),torch.full((len(recorded),),factor/(2*len(recorded)))))
        score=score-(score*weight).sum(-1,keepdim=True)/weight.sum()
        target=target-(target*weight).sum(-1,keepdim=True)/weight.sum()
        lq=(F.smooth_l1_loss(score,target,reduction='none')*weight).sum()
    trajectory=encoded.tokens.new_zeros(())
    if item['teacher_values'] is not None:
        targets,_=actor.policy.curve_codec.decode_values(item['trajectory_teacher_values'])
        def distance(a,b):
            position=((a-b)/actor.head.radius).square().mean((-1,-2))
            derivative=((torch.diff(a,dim=-2)-torch.diff(b,dim=-2))*(a.shape[-2]-1)/actor.policy.planning_horizon_m).square().mean((-1,-2))
            return (position+derivative+1e-6).sqrt()
        x=bank['paths']
        cross=torch.stack([distance(a,b) for a in x for b in targets]).mean()
        within=torch.stack([distance(a,b) for i,a in enumerate(x) for j,b in enumerate(x) if i!=j]).sum()/(2*16*15)
        # The batched form also sums the 16 diagonal terms sqrt(1e-6).
        trajectory=cross-within-16*1e-3/(2*16*15)
    return dict(CFM=cfm,TRAJ=trajectory,GF=gf,LH=lh,LQ=lq),bank


def batched_draws(records):
    """The random numbers batched_generation draws, in its order."""
    count=len(records);teachers=any(r[0]['teacher_values'] is not None for r in records)
    noise,time=(torch.randn(count,2,14),torch.rand(count,2)) if teachers else (None,None)
    source=torch.randn(count,16,14)
    return [dict(noise=None if noise is None else noise[i],time=None if time is None else time[i],source=source[i]) for i in range(count)]

scalar=SimpleNamespace(losses=scalar_reference,live_condition=None,observed_risk=None,geometry_reconstruction_loss=None)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.velocity=torch.nn.Linear(19,14);self.projections=0
    def project_condition_memory(self,condition):
        self.projections+=1
        value=condition.tokens.mean((1,2))[:,None]
        return ((value,value,condition.token_valid[:,:1]),)


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.trajectory_decoder=Decoder();self.trajectory_evaluator=torch.nn.Linear(3,1)
        self.curve_codec=build_policy(tiny_config()).curve_codec;self.integration_steps=2;self.planning_horizon_m=3.6
        self.register_buffer('inference_source',torch.randn(16,14));self.calls=[];self.q_calls=[]
    def _predict_velocity(self,state,time,condition,memory,step=0.):
        memory=memory[0][0];self.calls.append((len(state),len(memory)))
        m=memory.repeat_interleave(len(state)//len(memory),0)
        x=torch.cat((state,time[:,None],torch.full_like(time,step)[:,None],condition.goal_present,condition.terminal_goal[:,:1]*condition.goal_present,m),-1)
        return self.trajectory_decoder.velocity(x)
    def score_candidate_paths(self,condition,goal,paths):
        self.q_calls.append(tuple(paths.shape[:2]));batch,count=paths.shape[:2]
        actual=torch.where(condition.goal_present.bool(),goal,torch.zeros_like(goal))[:,0]
        path=paths.mean((-1,-2))
        x=torch.stack((path,condition.tokens.mean((1,2))[:,None]*path,actual[:,None]*path),-1)
        return self.trajectory_evaluator(x).squeeze(-1)


def fixture(monkeypatch):
    torch.manual_seed(321);policy=ToyPolicy();head=torch.nn.Linear(1,1,bias=False);head.radius=.3;reader=torch.nn.Linear(1,1,bias=False)
    actor=LocalBSplineBundle(None,policy,head,{},{});actor.reader=reader
    items=[];records=[];counts=dict(live=0)
    for index in range(6):
        goal=index%2==0;partial=index==3;teacher=torch.linspace(.02,.12,14)*(index+1)
        item=dict(history_plan=dict(points=[]),distance_label_available=True,raw_sha256=f'raw-{index}',history_plan_sha256=f'plan-{index}',task_state=dict(panel="J",case={}),source_grid_index=0,q_supervision_available=not partial,teacher_values=None if partial else teacher,goal_present=goal,goal=torch.tensor([.2*(index+1),.1]) if goal else torch.zeros(2),
            trajectory_teacher_values=torch.empty(0,14) if partial else teacher[None],
            q_expert_values=torch.stack([teacher]*([1,2,1,0,0,2][index])) if [1,2,1,0,0,2][index] else torch.zeros(0,14),
            critic_replay_paths=torch.full((16,64,2),.05*(index+1)),utility_goal=torch.tensor([.1*index,0.]),view=np.array([index,2.,0.]),point_error_m=0.,
            local_target_field=torch.full((6,2,2),.02*index),loss_availability=dict(CFM=False,GF=True,LH=True,LQ=False,own_expert=False) if partial else None)
        def utility(paths,batch,index=index):
            assert batch['source_origin_xy'][0,0]==index
            return SimpleNamespace(score=(paths[:,:,-1,0]*.1+batch['point_goal'][:,0,None]).detach())
        items.append(item);records.append((item,index,index+1.,utility))
    def live(actor,item,raw,**kwargs):
        counts['live']+=1;t=(actor.head.weight.mean()*actor.reader.weight.mean()+raw*.03).expand(1,4,3)
        encoded=ConditionFeatures(tokens=t,token_valid=torch.ones(1,4,dtype=torch.bool),metric_position=torch.zeros(1,4,3),surface_hit=torch.zeros(1,4,dtype=torch.bool),frame_age=torch.zeros(1,4),motion_token=torch.zeros(1,4,dtype=torch.bool),terminal_goal=item['goal'][None],goal_present=torch.tensor([[float(item['goal_present'])]]),configuration_field=actor.head.weight.mean().expand(1,6,2,2))
        return encoded,(actor.head.weight.mean().expand(1,6,2,2),actor.head.weight.mean().expand(1,2,2,2))
    risk=lambda values,tree,*args:(values.square().mean()*tree,{})
    lh=lambda field,logits,target,*args,**kwargs:(field-target).square().mean()
    for module in (references,scalar):monkeypatch.setattr(module,'live_condition',live)
    for module in (batched,scalar):monkeypatch.setattr(module,'geometry_reconstruction_loss',lh)
    def live_batch(actor,records,maps):
        values=[references.live_condition(actor,*r[:2]) for r in records]
        return conditioning.cat_conditions([c for c,_ in values]),tuple(torch.cat([v[1][i] for v in values]) for i in range(2))
    monkeypatch.setattr(batched,'live_conditions',live_batch)
    monkeypatch.setattr(scalar,'observed_risk',risk)
    from kappanav.local_planner.features import image_batch as bspline_image_batch
    monkeypatch.setattr(bspline_image_batch,'step_feature_maps',lambda *a,**kw:{})
    monkeypatch.setattr(batched,'_source_geometry_risk',lambda actor,values,paths,pad,records,**kw:torch.stack([risk(values[i],r[2])[0] for i,r in enumerate(records)]).mean())
    prepare_replay_targets(actor,records)
    return actor,items,records,counts


@pytest.mark.parametrize("rows", [slice(None), slice(0,3), slice(3,4)])
def test_true_batched_two_step_losses_and_all_gradients_match_fp32_sample_mean(monkeypatch,rows):
    actor,items,records,counts=fixture(monkeypatch);records=records[rows];params=list(actor.policy.parameters())+list(actor.head.parameters())+list(actor.reader.parameters())
    torch.manual_seed(1001);draws=batched_draws(records);reference_rng=torch.get_rng_state()
    reference=[scalar.losses(actor,*r,d)[0] for r,d in zip(records,draws)]
    expected={name:torch.stack([r[name] for r in reference]).mean() for name in reference[0]}
    oldgrads=torch.autograd.grad(sum(expected.values()),params,allow_unused=True)
    actor.policy.calls.clear();actor.policy.q_calls.clear();counts['live']=0
    actor.policy.trajectory_decoder.projections=0
    torch.manual_seed(1001);actual,bank=batched.batched_losses(actor,records)
    assert torch.equal(torch.get_rng_state(),reference_rng)
    forward_calls=list(actor.policy.calls)
    newgrads=torch.autograd.grad(sum(actual.values()),params,allow_unused=True)
    for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=2e-5,atol=2e-6)
    for old,new in zip(oldgrads,newgrads):
        assert (old is None)==(new is None)
        if old is not None:torch.testing.assert_close(new,old,rtol=3e-5,atol=3e-6)
    assert counts['live']==len(records)
    assert actor.policy.trajectory_decoder.projections==1
    count=len(records);teachers=any(r[0]['teacher_values'] is not None for r in records)
    assert forward_calls==([(count*2,count)] if teachers else [])+[(count*16,count)]*2
    assert actor.policy.q_calls==([(sum(r[0]['q_supervision_available'] for r in records),34)] if teachers else [])
    assert bank['paths'].shape==(count,16,64,2)


def test_goal_nogoal_native_sources_batched_without_unused_q_scoring(monkeypatch):
    actor,items,records,_=fixture(monkeypatch);conditions=[references.live_condition(actor,r[0],r[1])[0] for r in records]
    condition=conditioning.cat_conditions(conditions);reference=[actor.candidates(c,i['goal'])['values'] for c,i in zip(conditions,items)]
    actor.policy.q_calls.clear();bank=actor.native_bank(condition)
    assert actor.policy.q_calls==[]
    torch.testing.assert_close(bank['values'],torch.stack(reference),rtol=2e-5,atol=2e-6)
    # Decoder input non-goal mask never opens, and all original fields are kept when slicing.
    for index,c in enumerate(conditions):
        selected=batched.select_condition(condition,[index])
        for name in vars(c):torch.testing.assert_close(getattr(c,name),getattr(selected,name),rtol=0,atol=0)


def test_q_targets_are_scored_online_for_recorded_and_current_candidates(monkeypatch):
    actor,items,records,_=fixture(monkeypatch);item=items[0]
    condition,_=references.live_condition(actor,*records[0][:2]);calls=[]
    def utility(paths,batch):
        calls.append(paths.shape[1]);return records[0][3](paths,batch)
    a=references.prepare_critic_inputs(item,condition.goal_present,torch.zeros(16,64,2),utility,[item[n] for n in references.CRITIC_PATHS])
    references.label_critic_inputs([a])
    b=references.prepare_critic_inputs(item,condition.goal_present,torch.ones(16,64,2),utility,[item[n] for n in references.CRITIC_PATHS])
    references.label_critic_inputs([b])
    # One utility call scores expert, current and recorded candidates together, every time.
    bank=len(item['q_expert_paths'])+16+len(item['critic_replay_paths'])
    assert calls==[bank,bank]
    count=a['expert_count']
    assert torch.equal(a['recorded_targets'],b['recorded_targets'])
    assert torch.equal(a['targets'][:,:count],b['targets'][:,:count])
    assert not torch.equal(a['targets'][:,count:],b['targets'][:,count:])


def test_local_q_uses_only_current_goal_and_four_frame_novelty(monkeypatch):
    actor,items,records,_=fixture(monkeypatch);item=items[1]
    item['local_target_field']=torch.ones_like(item['local_target_field'])
    goal=torch.zeros(1,1);paths=torch.zeros(16,64,2)
    local=references.prepare_critic_inputs(item,goal,paths,records[1][3],[item[n] for n in references.CRITIC_PATHS])
    assert 'task_states' not in local['qbatch'] and 'utility_goal' not in local['qbatch']
    assert torch.equal(local['qbatch']['observed_field'],item['local_target_field'][None])
    references.label_critic_inputs([local])


def test_source_signed_cell_clearance_keeps_inside_wall_gradients():
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    signed=np.ones((30,30),np.float32);signed[10:20,10:20]=-1.
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(signed,np.array([-1.,-1.]),.1,'canonical_world_xy'),))
    points=torch.tensor([[1.5,.4],[.8,.4],[-.1,.7]],dtype=torch.float64,requires_grad=True)
    distance=query.signed_distance(points,0)
    torch.testing.assert_close(distance,torch.tensor([.5,-.2,.1],dtype=torch.float64))
    distance.sum().backward()
    assert torch.isfinite(points.grad).all() and (points.grad.norm(dim=-1)>0).all()


def test_source_GF_uses_metric_world_pose_and_backpropagates():
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    grid=np.ones((100,100),np.float32);grid[50:70,50:70]=-1.
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(grid,np.array([-2.5,-2.5]),.05,'canonical_world_xy'),))
    utility=SimpleNamespace(source_query=query)
    actor=SimpleNamespace(head=SimpleNamespace(radius=.2))
    t=torch.linspace(0,.9,257,dtype=torch.float64)
    paths=torch.stack((torch.zeros_like(t),-t),-1)[None].requires_grad_()
    values=torch.zeros(1,1,14);pad=torch.zeros(1,dtype=torch.float64)
    records=[(dict(view=[-.5,.5,np.pi/2],source_grid_index=0,teacher_values=torch.zeros(14)),{},None,utility)]
    loss=batched._source_geometry_risk(actor,values,paths,pad,records)
    assert loss>0
    loss.backward()
    assert torch.isfinite(paths.grad).all() and paths.grad.norm()>0
    records[0][0]['view']=[-1.5,-1.5,np.pi/2]
    assert batched._source_geometry_risk(actor,values,paths,pad,records)==0


def test_source_GF_ignores_rows_without_a_teacher():
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    grid=np.ones((100,100),np.float32);grid[50:70,50:70]=-1.
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(grid,np.array([-2.5,-2.5]),.05,'canonical_world_xy'),))
    utility=SimpleNamespace(source_query=query);actor=SimpleNamespace(head=SimpleNamespace(radius=.2))
    t=torch.linspace(0,.9,257,dtype=torch.float64)
    paths=torch.stack((torch.zeros_like(t),-t),-1)[None].expand(2,-1,-1).clone().requires_grad_()
    values=torch.zeros(2,1,14);pad=torch.zeros(2,dtype=torch.float64)
    row=lambda teacher:(dict(view=[-.5,.5,np.pi/2],source_grid_index=0,teacher_values=teacher),{},None,utility)
    both=batched._source_geometry_risk(actor,values,paths,pad,[row(torch.zeros(14)),row(None)])
    taught=batched._source_geometry_risk(actor,values[:1],paths[:1],pad[:1],[row(torch.zeros(14))])
    torch.testing.assert_close(both,taught/2)
    gradient,=torch.autograd.grad(both,paths)
    assert gradient[0].norm()>0 and gradient[1].norm()==0


@pytest.mark.parametrize('views', [([.1,.2,.6],[-.2,.1,-1.]), ([.1,.2,.6],[-.2,.1,-1.],[.1,.2,.6])])
def test_batched_GF_poses_preserve_per_scene_distances_and_gradients(monkeypatch,views):
    from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
    signed=np.ones((70,70),np.float32);signed[30:40,20:50]=-1
    query=SourceConfigurationSpaceQuery((SourceConfigurationGrid(signed,np.array([-1.75,-1.75]),.05),))
    actor=SimpleNamespace(head=SimpleNamespace(radius=.2));utility=SimpleNamespace(source_query=query)
    records=[(dict(view=view,source_grid_index=0,teacher_values=torch.zeros(14)),None,None,utility) for view in views]
    batch=len(records)
    paths=(torch.rand(batch,514,2,dtype=torch.float64,generator=torch.Generator().manual_seed(19))-.5).requires_grad_()
    values=torch.zeros(batch,2,14);pad=torch.linspace(.05,.15,batch*2,dtype=torch.float64,requires_grad=True)
    expected=[]
    for index,(item,*_) in enumerate(records):
        view=paths.new_tensor(item['view']);c,s=view[2].cos(),view[2].sin()
        rotation=torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
        world=paths[index]@rotation.T+view[:2]
        distance=query.signed_distance(world,0).reshape(2,257)
        probe=world.detach().requires_grad_()
        normal,=torch.autograd.grad(query.signed_distance(probe,0).sum(),probe)
        deficit=((pad.detach().reshape(batch,2,1)[index]-distance)/.2).clamp_min(0)
        deficit=deficit*batched._entry_side(normal.reshape(1,2,257,2),deficit.detach()[None]>0)[0]
        expected.append((4*torch.where(deficit<=1,deficit.square(),2*deficit-1).amax(-1)).mean())
    calls=[];original=query.signed_distance_batch
    def counted(points,index):
        calls.append(points.shape[0]*points.shape[1]);return original(points,index)
    monkeypatch.setattr(query,'signed_distance_batch',counted)
    expected=torch.stack(expected).mean();actual=batched._source_geometry_risk(actor,values,paths,pad,records)
    assert len(calls)==1 and sum(calls)==batch*514
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    result,=torch.autograd.grad(actual,paths,retain_graph=True);reference,=torch.autograd.grad(expected,paths,retain_graph=True)
    torch.testing.assert_close(result,reference,rtol=1e-12,atol=1e-12)
    # The sampling bound is a fixed margin: GF gives it no gradient.
    assert torch.autograd.grad(actual,pad,allow_unused=True)[0] is None



def test_duplicate_samples_share_only_current_step_encoding(monkeypatch):
    actor,items,records,counts=fixture(monkeypatch)
    records=[records[1],records[1]];params=list(actor.policy.parameters())+list(actor.head.parameters())+list(actor.reader.parameters())
    torch.manual_seed(51);expected=[scalar.losses(actor,*r,d)[0] for r,d in zip(records,batched_draws(records))]
    expected={k:torch.stack([r[k] for r in expected]).mean() for k in expected[0]}
    expected_grads=torch.autograd.grad(sum(expected.values()),params,allow_unused=True)
    counts['live']=0;torch.manual_seed(51);actual,_=batched.batched_losses(actor,records)
    actual_grads=torch.autograd.grad(sum(actual.values()),params,allow_unused=True)
    assert counts['live']==1
    for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=2e-5,atol=2e-6)
    for a,b in zip(actual_grads,expected_grads,strict=True):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6)
    batched.batched_losses(actor,records)
    assert counts['live']==2


@pytest.mark.parametrize('goal_present',[False,True])
def test_shared_history_preserves_each_goal_intent_and_all_loss_gradients(monkeypatch,goal_present):
    actor,items,records,counts=fixture(monkeypatch)
    twin=deepcopy(items[0]);twin.update(goal_present=goal_present,goal=torch.tensor([-.7,.3]) if goal_present else torch.zeros(2),
        teacher_values=(items[0]['teacher_values']*.5),
        trajectory_teacher_values=(items[0]['teacher_values']*.5)[None])
    rows=[records[0],(twin,*records[0][1:])]
    prepare_replay_targets(actor,rows)
    parameters=list(actor.policy.parameters())+list(actor.head.parameters())+list(actor.reader.parameters())
    torch.manual_seed(51);expected=[scalar.losses(actor,*row,d)[0] for row,d in zip(rows,batched_draws(rows))]
    expected={key:torch.stack([row[key] for row in expected]).mean() for key in expected[0]}
    expected_grads=torch.autograd.grad(sum(expected.values()),parameters,allow_unused=True)
    counts['live']=0;torch.manual_seed(51);actual,_=batched.batched_losses(actor,rows)
    actual_grads=torch.autograd.grad(sum(actual.values()),parameters,allow_unused=True)
    assert counts['live']==1
    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=2e-5,atol=2e-6)
    for actual_grad,expected_grad in zip(actual_grads,expected_grads,strict=True):
        assert (actual_grad is None)==(expected_grad is None)
        if actual_grad is not None:torch.testing.assert_close(actual_grad,expected_grad,rtol=3e-5,atol=3e-6)
    twin['history_plan_sha256']='different-four-frame-selection'
    batched.batched_losses(actor,rows)
    assert counts['live']==3


def test_real_batch_sampler_uses_one_uniform_task_cycle():
    rows=[(dict(trial_id=f't{i%8}',raw_sha256=f'r{i}'),None,None,None) for i in range(32)]
    columns=([r[0]['trial_id'] for r in rows],[r[0]['raw_sha256'] for r in rows],[False]*len(rows))
    g=case_groups(*columns,range(16));m=case_groups(*columns,range(16,32))
    for rank in range(8):
        assert batch_indices(g+m,3,7,rank,8,2)==[sample_index(g+m,160+rank*2+s) for s in range(2)]
        selected=batch_indices(g+m,3,7,rank,8,4)
        positions=[320+rank*4+s for s in range(4)]
        assert selected==[sample_index(g+m,p) for p in positions]


def test_production_adamw_schedule_ema_full_roundtrip_and_epoch_checkpoint(tmp_path,monkeypatch):
    import random,json
    from kappanav.local_planner.training.optimizer import build_optimizer, build_cosine_schedule
    from kappanav.local_planner.training.ema import ExponentialMovingAverage
    from kappanav.local_planner.training import bspline_replay as caller
    cfg=replace(tiny_config().training,epochs=50,warmup_epochs=5)
    goals=[[[0]*15] for _ in range(16)];memories=[[[0]*43] for _ in range(24)]
    for batch,steps in ((2,108),(4,54),(8,27)):
        budget=epoch_budget(goals+memories,8,batch,cfg,1e-5)
        assert budget['steps_per_epoch']==steps and budget['total_steps']==50*steps and budget['warmup_steps']==5*steps
    def make():
        model=torch.nn.ModuleDict(dict(policy=torch.nn.Linear(3,2),head=torch.nn.Linear(2,1)))
        opt=build_optimizer(model,1e-5,.01);schedule=build_cosine_schedule(opt,10,2,.01);ema=ExponentialMovingAverage(model,.9999)
        return model,opt,schedule,ema
    torch.manual_seed(18);np.random.seed(18);random.seed(18)
    model,opt,schedule,ema=make();assert not opt.state and ema.num_updates==0
    assert all(g['betas']==(.9,.95) and g['fused'] for g in opt.param_groups)
    assert [g['weight_decay'] for g in opt.param_groups]==[.01,0.]
    def update(m,o,s,e):
        x=torch.randn(4,3)+np.random.rand()+random.random();o.zero_grad(set_to_none=True)
        m['head'](m['policy'](x)).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();s.step();e.update()
    update(model,opt,schedule,ema);raw=deepcopy(model.state_dict());rng=capture_rng(torch.device('cpu'))
    contract=dict(world_size=8,effective_batch=16,row_counts=dict(all=345,goal=127,no_goal=218),epoch_budget=dict(total_steps=10,warmup_steps=2))
    state=full_state(model,opt,schedule,ema,[rng]*8,1,contract,'raw','ema',dict(sha256='warmstart_only',legacy_optimizer_restored=False));torch.save(state,tmp_path/'full.pt')
    update(model,opt,schedule,ema);expected=deepcopy(model.state_dict());expected_ema=deepcopy(ema.state_dict());expected_rng=torch.get_rng_state().clone()
    restored,ro,rs,re=make();restored.load_state_dict(raw)
    loaded=restore_full(tmp_path/'full.pt',restored,ro,rs,re,contract=contract,raw_sha='raw',rank=0,device=torch.device('cpu'));assert loaded['steps']==1
    update(restored,ro,rs,re)
    for name,value in restored.state_dict().items():torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
    assert rs.state_dict()==schedule.state_dict() and re.num_updates==2
    for name,value in re.shadow.items():torch.testing.assert_close(value,expected_ema['shadow'][name],rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    for badcontract,badsha in ((dict(contract,effective_batch=32),'raw'),(contract,'ema')):
        with pytest.raises(ValueError):restore_full(tmp_path/'full.pt',restored,ro,rs,re,contract=badcontract,raw_sha=badsha,rank=0,device=torch.device('cpu'))
    torch.save(dict(format='bspline_memory_training'),tmp_path/'legacy.pt')
    with pytest.raises(ValueError,match='another training contract'):restore_full(tmp_path/'legacy.pt',restored,ro,rs,re,contract=contract,raw_sha='raw',rank=0,device=torch.device('cpu'))
    def save_bundle(path,**kwargs):
        torch.save(dict(weights_kind=kwargs['provenance']['weights_kind'],components={k:deepcopy(kwargs[k].state_dict()) for k in ('policy','head')}),path)
    monkeypatch.setattr(caller,'save_bundle',save_bundle)
    actor=SimpleNamespace(policy=restored['policy'],head=restored['head'],provenance={},architecture={})
    snapshot=tmp_path/'checkpoints/epoch_001';payloads=[dict(rng=capture_rng(torch.device('cpu')),performance=dict(rank=i)) for i in range(8)]
    before=deepcopy(restored.state_dict())
    save_checkpoint(snapshot,actor,restored,ro,rs,re,dict(contract,gradient_communication='nccl_cuda_fp32_sum_mean',nccl_protocol='Simple',cnn_batch_size=16,cnn_feature_contract='same_sensor_step_full_image_batch_v1_not_B1_bitexact'),dict(sha256='warmstart'),{},2,1,2,dict(epochs=50),'parent',None,payloads,[],tmp_path/'source',create=True)
    proof=json.loads((snapshot/'result.json').read_text());assert proof['complete'] and proof['evaluation_weights_kind']=='ema'
    assert proof['provenance']['deterministic_algorithms'] is True
    assert proof['provenance']['nccl_protocol']=='Simple'
    assert torch.load(snapshot/'model.pt',weights_only=False)['weights_kind']=='raw'
    assert torch.load(snapshot/'ema_model.pt',weights_only=False)['weights_kind']=='ema'
    saved=torch.load(snapshot/'training_state.pt',weights_only=False);assert saved['steps']==saved['ema']['num_updates']==saved['scheduler']['last_epoch']==2
    for name,value in restored.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)


@pytest.mark.parametrize('pool_size',[0,1])
def test_partial_defined_q_keeps_gf_lh_and_no_generator_q_gradient(monkeypatch,pool_size):
    actor,items,records,_=fixture(monkeypatch)
    item=items[3];item['loss_availability']['LQ']=True;item['q_supervision_available']=True
    item['q_expert_values']=torch.full((pool_size,14),.1)
    prepare_replay_targets(actor,[records[3]])
    terms,_=batched.batched_losses(actor,[records[3]])
    assert terms['CFM']==0 and terms['GF']>0 and terms['LH']>0 and terms['LQ'].requires_grad
    decoder=list(actor.policy.trajectory_decoder.parameters())
    grads=torch.autograd.grad(terms['LQ'],decoder+list(actor.head.parameters())+list(actor.reader.parameters()),allow_unused=True)
    assert all(g is None for g in grads[:len(decoder)])
    assert all(g is not None and torch.isfinite(g).all() for g in grads[len(decoder):])


def test_partial_undefined_q_never_queries_source_and_retains_gf_lh(monkeypatch):
    actor,items,records,_=fixture(monkeypatch)
    item,raw,tree,_=records[3]
    def forbidden(*args):pytest.fail('Undefined source utility queried')
    terms,_=batched.batched_losses(actor,[(item,raw,tree,forbidden)])
    assert terms['CFM']==terms['LQ']==0 and terms['GF']>0 and terms['LH']>0
    parameters=list(actor.policy.trajectory_evaluator.parameters())
    assert all(g is None for g in torch.autograd.grad(sum(terms.values()),parameters,allow_unused=True))


def test_native_bank_preserves_generator_and_shared_perception_gradients(monkeypatch):
    actor,items,records,_=fixture(monkeypatch)
    parameters=list(actor.policy.parameters())+list(actor.head.parameters())+list(actor.reader.parameters())
    conditions=[references.live_condition(actor,*record[:2])[0] for record in records[:2]]
    bank=actor.native_bank(conditioning.cat_conditions(conditions))
    loss=bank['values'].square().mean()+bank['paths'].square().mean()
    gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    decoder_grad,head_grad,reader_grad=(gradients[0],gradients[-2],gradients[-1])
    assert decoder_grad.abs().sum()>0 and head_grad.abs().sum()>0 and reader_grad.abs().sum()>0


def test_native16_selects_sources_before_exactly_two_euler_calls(monkeypatch):
    actor,items,records,_=fixture(monkeypatch)
    condition=conditioning.cat_conditions([references.live_condition(actor,*r[:2])[0] for r in records[:2]])
    scale=torch.tensor(.3,requires_grad=True)
    calls=[]
    def velocity(state,time,condition,memory,step=0.):
        calls.append((state.shape, time.detach().clone(), condition.goal_present.detach().clone(), step))
        return scale*state+time[:,None]
    monkeypatch.setattr(actor.policy,'_predict_velocity',velocity)
    bank=actor.native_bank(condition)
    z=actor.policy.inference_source.expand(2,-1,-1)
    expected_state=(1-scale/2)**2*z-.5*(1-scale/2)-.25
    expected=actor.policy.curve_codec.values_from_coordinates(expected_state.flatten(0,1)).reshape(2,16,14)
    torch.testing.assert_close(bank['values'],expected,rtol=1e-6,atol=1e-6)
    actual_grad=torch.autograd.grad(bank['values'].square().sum(),scale,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(expected.square().sum(),scale)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-6,atol=1e-5)
    assert len(calls)==2 and all(shape==(32,14) and step==.5 for shape,_,_,step in calls)  # Two scenes, sixteen each.
    assert torch.equal(calls[0][1],torch.ones(32)) and torch.equal(calls[1][1],torch.full((32,),.5))
    assert calls[0][2][:16].all() and not calls[0][2][16:].any()


def test_energy_score_prefers_the_teacher_mixture_and_trains_every_candidate(monkeypatch):
    actor,items,_,_=fixture(monkeypatch)
    codec=actor.policy.curve_codec
    left,right=torch.zeros(14),torch.zeros(14);left[1::2]=.4;right[1::2]=-.4
    item=dict(items[0],trajectory_teacher_values=torch.stack((left,right)))
    prepare_generator_targets(actor,[(item,)])
    target,_=codec.decode_values(torch.stack((left,right)))
    covered=torch.cat((target[:1].expand(8,-1,-1),target[1:].expand(8,-1,-1)))[None]
    averaged=target.mean(0,keepdim=True).expand(16,-1,-1)[None]
    one_mode=target[:1].expand(16,-1,-1)[None]
    score=lambda paths:batched.generated_energy_score(actor,paths,[item])
    assert score(covered)<score(averaged) and score(covered)<score(one_mode)
    shifted=(covered+.05*torch.randn_like(covered)).requires_grad_()
    gradient=torch.autograd.grad(score(shifted),shifted)[0]
    assert (gradient.flatten(2).abs().sum(-1)>0).all()  # no dead candidate


def test_loss_gradient_ownership_includes_generated_trajectory(monkeypatch):
    actor,_,records,_=fixture(monkeypatch)
    terms,_=batched.batched_losses(actor,records)
    generator=list(actor.policy.trajectory_decoder.parameters())
    critic=list(actor.policy.trajectory_evaluator.parameters())
    shared=list(actor.head.parameters())+list(actor.reader.parameters())
    for name in ('CFM','TRAJ','GF'):
        grads=torch.autograd.grad(terms[name],generator+critic+shared,retain_graph=True,allow_unused=True)
        assert any(g is not None and g.abs().sum()>0 for g in grads[:len(generator)])
        assert all(g is None for g in grads[len(generator):len(generator)+len(critic)])
        assert all(g is not None and g.abs().sum()>0 for g in grads[-len(shared):])
    qgrads=torch.autograd.grad(terms['LQ'],generator+critic+shared,retain_graph=True,allow_unused=True)
    assert all(g is None for g in qgrads[:len(generator)])
    assert qgrads[len(generator)].abs().sum()>0
    assert qgrads[len(generator)+1].abs().sum()<1e-6  # A common score bias cannot change argmax.
    assert all(g is not None and g.abs().sum()>0 for g in qgrads[-len(shared):])


def test_energy_score_penalizes_oscillation_not_only_point_error(monkeypatch):
    actor,items,_,_=fixture(monkeypatch)
    teacher=torch.zeros(1,14)
    item=dict(items[0],trajectory_teacher_values=teacher)
    prepare_generator_targets(actor,[(item,)])
    target=actor.policy.curve_codec.decode_values(teacher)[0]
    shift=torch.ones_like(target);shift[:,0]=0
    oscillation=shift.clone();oscillation[:,::2]*=-1
    smooth=(target+shift)[:,None].requires_grad_()
    curled=(target+oscillation)[:,None].requires_grad_()
    smooth=smooth.expand(1,16,-1,-1);curled=curled.expand(1,16,-1,-1)
    smooth_loss=batched.generated_energy_score(actor,smooth,[item])
    curled_loss=batched.generated_energy_score(actor,curled,[item])
    assert curled_loss>smooth_loss*3
    assert torch.autograd.grad(curled_loss,curled)[0].abs().sum()>0
def test_cpu_generator_targets_preserve_all_teacher_modes_and_missing_own_teacher():
    from kappanav.local_planner.model.trajectory import IncrementalBSplineTrajectory
    codec=IncrementalBSplineTrajectory(8,3,64,(0.,)*14,(1.,)*14)
    actor=SimpleNamespace(policy=SimpleNamespace(curve_codec=codec))
    values=torch.linspace(-.1,.2,14)
    items=[dict(teacher_values=values,trajectory_teacher_values=values[None]),
        dict(teacher_values=None,trajectory_teacher_values=torch.stack((values,-values))),
        dict(teacher_values=None,trajectory_teacher_values=torch.empty(0,14))]
    prepare_generator_targets(actor,[(item,) for item in items])
    torch.testing.assert_close(items[0]['cfm_target'],codec.coordinates_from_values(values[None]))
    assert 'cfm_target' not in items[1] and 'cfm_target' not in items[2]
    assert 'trajectory_target' not in items[2]
    for item in items[:2]:
        expected,_=codec.decode_values(item['trajectory_teacher_values'])
        actual=item['trajectory_target']
        torch.testing.assert_close(actual,expected)
        assert actual.device.type=='cpu' and not actual.requires_grad


def test_critic_length_groups_preserve_scores_and_all_gradients(monkeypatch):
    from test_policy import make_encoded_condition
    torch.manual_seed(104)
    policy=build_policy(tiny_config()).eval()
    policy.trajectory_evaluator.neural_dtype=torch.float32
    base=make_encoded_condition(policy,3);lengths=[150,7,41];maximum=max(lengths)
    changes={}
    for name in ('tokens','token_valid','metric_position','surface_hit','frame_age','motion_token'):
        value=getattr(base,name);shape=(3,maximum,*value.shape[2:])
        extra=torch.randn(shape,dtype=value.dtype) if value.is_floating_point() else torch.ones(shape,dtype=value.dtype)
        if name=='token_valid':extra=torch.arange(maximum)[None]<torch.tensor(lengths)[:,None]
        changes[name]=torch.cat((value,extra),1)
    condition=replace(base,**changes,goal_present=torch.tensor([[1.],[0.],[1.]]))
    tokens=condition.tokens.detach().requires_grad_();field=condition.configuration_field.detach().requires_grad_()
    condition=replace(condition,tokens=tokens,configuration_field=field)
    paths=torch.randn(3,2,64,2,requires_grad=True);goals=condition.terminal_goal
    items=[dict(history_plan=dict(points=range(n))) for n in lengths]
    reference=policy.score_candidate_paths(condition,goals,paths)
    calls=[];original=policy.score_candidate_paths
    def scored(selected,goal,bank):
        calls.append(tuple(selected.tokens.shape[:2]));return original(selected,goal,bank)
    monkeypatch.setattr(policy,'score_candidate_paths',scored)
    actual=batched._score_critic_banks(policy,condition,goals,paths,items)
    assert calls==[(1,base.tokens.shape[1]+7),(2,base.tokens.shape[1]+150)]
    torch.testing.assert_close(actual,reference,rtol=2e-5,atol=2e-6)
    inputs=(tokens,field,paths,*policy.trajectory_evaluator.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),inputs),torch.autograd.grad(reference.sum(),inputs),strict=True):
        torch.testing.assert_close(a,b,rtol=5e-4,atol=2e-5)


def test_critic_skips_unavailable_longest_scene_without_reweighting_available_rows(monkeypatch):
    from test_policy import make_encoded_condition
    torch.manual_seed(108)
    policy=build_policy(tiny_config()).eval()
    policy.trajectory_evaluator.neural_dtype=torch.float32
    condition=make_encoded_condition(policy,3);base=condition.tokens.shape[1]
    lengths=[9,2,4];changes={}
    for name in ('tokens','token_valid','metric_position','surface_hit','frame_age','motion_token'):
        value=getattr(condition,name)
        extra=value.new_zeros(3,9,*value.shape[2:])
        if name=='token_valid':extra=torch.arange(9)[None]<torch.tensor(lengths)[:,None]
        changes[name]=torch.cat((value,extra),1)
    condition=replace(condition,**changes)
    tokens=condition.tokens.detach().requires_grad_();field=condition.configuration_field.detach().requires_grad_()
    condition=replace(condition,tokens=tokens,configuration_field=field)
    paths=torch.randn(3,2,64,2);recorded=torch.randn(2,64,2)
    targets=torch.tensor([[.3,-.5,.8,-.2]]);weights=torch.full((1,4),.25)
    items=[dict(history_plan=dict(points=range(n)),q_supervision_available=i>0,goal=condition.terminal_goal[i],
        q_expert_paths=torch.zeros(0,64,2),critic_replay_paths=recorded)
        for i,n in enumerate(lengths)]
    # Every available row's label targets: its two current and two recorded candidates.
    monkeypatch.setattr(batched,'critic_label_inputs',lambda rows,utilities:[])
    monkeypatch.setattr(batched,'label_critic_targets',lambda groups,native,present,fields,shape:targets.expand(len(native),-1))
    reference=[]
    for i in (1,2):
        selected=conditioning.select_condition(condition,[i]);bank=torch.cat((paths[i],recorded))[None]
        scores=policy.score_candidate_paths(selected,selected.terminal_goal,bank)
        reference.append(batched.relative_utility_loss(scores,targets,weights))
    expected=torch.stack(reference).sum()/3
    calls=[];original=policy.score_candidate_paths
    def scored(selected,goal,bank):
        calls.append(tuple(selected.tokens.shape[:2]));return original(selected,goal,bank)
    monkeypatch.setattr(policy,'score_candidate_paths',scored)
    actual=references.batched_critic(SimpleNamespace(policy=policy),condition,items,paths,[None]*3,fields=torch.zeros(3,1))
    assert calls==[(1,base+2),(1,base+4)]
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    inputs=(tokens,field,*policy.trajectory_evaluator.parameters())
    actual_grad=torch.autograd.grad(actual,inputs);expected_grad=torch.autograd.grad(expected,inputs)
    for a,b in zip(actual_grad,expected_grad,strict=True):
        torch.testing.assert_close(a,b,rtol=5e-4,atol=2e-5)
    assert not actual_grad[0][0].any() and not actual_grad[1][0].any()
