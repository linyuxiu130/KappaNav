"""The five training losses of one microbatch (docs/ARCHITECTURE.md section 3), each a mean over its rows.

batched_losses is the one entry: depth CNN (step_feature_maps) -> four-frame fusion and geometry head
(live_conditions) -> LH on the head's field; the generator's CFM and its 16-candidate bank (batched_generation) ->
TRAJ (generated_energy_score) and GF (_source_geometry_risk) on the bank; Q's scores of the bank, the row's same-pose
experts and its recorded candidates against the route utility labelled on a side thread (start_critic -> LQ).

Every `*_inputs` function is the host side of the function it names: it packs that step's CPU tensors so the loader
thread can build them and HostTree moves them in one pinned transfer per dtype; each device function builds its own
inputs when given none. Shapes: B rows, C=16 candidates, P=64 decoded points, S=257 dense samples, D=14
standardized control-increment coordinates.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from kappanav.local_planner.features.conditions import live_conditions, select_condition
from kappanav.local_planner.data.boundary_batch import signed_distance_rows
from kappanav.local_planner.data.bspline_replay import dense_curve, SAMPLES
from kappanav.local_planner.features.staging import HostTree, Keep
from kappanav.local_planner.features.temporal_geometry import geometry_reconstruction_loss
from kappanav.local_planner.model.network.policy import INFERENCE_CANDIDATES, repeat_condition
from kappanav.local_planner.data.replay_targets import critic_sources

OBJECTIVE='native_decoupled_euler2_energy_four_frame'  # recorded in checkpoints; names this objective
CFM_QUERIES=2  # uniform-time bridge queries per teacher
OWN_TEACHER_WEIGHT=.5  # TRAJ mass of the row's own teacher in its same-pose pool
# ConditionFeatures fields with one entry per scene token ([B,T,...]); the others are per row.
TOKEN_FIELDS=('tokens','token_valid','metric_position','surface_hit','frame_age','motion_token')
_LABEL_WORKER=ThreadPoolExecutor(max_workers=1)


def _truncate_tokens(condition,stop):
    """The condition with only its first `stop` scene tokens."""
    return replace(condition,**{name:getattr(condition,name)[:,:stop] for name in TOKEN_FIELDS})


def _aligned_condition(condition):
    """Pad the scene tokens to a multiple of 16 with invalid tokens under autograd on CUDA (the efficient attention
    kernel aligns bias gradients to 16; invalid keys add no attention mass, so values are unchanged)."""
    if not condition.tokens.is_cuda or not torch.is_grad_enabled():
        return condition
    pad=(-condition.tokens.shape[1])%16
    values={}
    for name in TOKEN_FIELDS:
        value=getattr(condition,name)
        if pad:value=F.pad(value,(0,pad) if value.ndim==2 else (0,0,0,pad))
        values[name]=value.contiguous()
    return replace(condition,**values)


def generation_inputs(policy,items):
    """Host side of batched_generation: which rows have a teacher [B] and their clean CFM targets [B,D] (zeros
    without one)."""
    present=[item['teacher_values'] is not None for item in items]
    if not any(present):return dict(any=False)
    return dict(any=True,teacher=torch.tensor(present),clean=torch.cat([item['cfm_target'] if item['teacher_values'] is not None
        else torch.zeros(1,policy.curve_codec.coordinate_dim) for item in items]))


def batched_generation(actor,conditions,items,memory,*,host=None):
    """CFM loss (scalar) and the candidate bank (generate_candidate_bank: values [B,C,D], paths [B,C,P,2]).

    CFM: per row CFM_QUERIES times t~U(0,1), x_t=(1-t)x0+t*z, mean squared error of the step-0 velocity against
    z-x0, zero weight on rows without a teacher. The bank is the two-step sampler from fresh noise; its calls carry
    their step size, so the two losses train different functions and neither compromises with the other.
    """
    policy=actor.policy;device=conditions.tokens.device
    if host is None:host=HostTree(generation_inputs(policy,items)).to(device)
    cfm=conditions.tokens.new_zeros(())
    if host['any']:
        # Rows without a teacher run with a zero target and zero weight.
        teacher,clean=host['teacher'],host['clean']
        noise=torch.randn(len(items),CFM_QUERIES,clean.shape[-1],device=device)
        time=torch.rand(len(items),CFM_QUERIES,device=device)
        state=(1-time[...,None])*clean[:,None]+time[...,None]*noise
        velocity=policy._predict_velocity(state.flatten(0,1),time.flatten(),repeat_condition(conditions,CFM_QUERIES),memory)
        error=(velocity.unflatten(0,state.shape[:2])-(noise-clean[:,None])).square().mean((1,2))
        cfm=(error*teacher).mean()
    source=torch.randn(len(items),INFERENCE_CANDIDATES,policy.curve_codec.coordinate_dim,device=device)
    return cfm,actor.native_bank(conditions,memory=memory,source=source)


def _curve_features(actor,paths):
    """[...,P,2] paths -> [...,4P-2] features. Squared distance of two feature rows = mean squared position error over the
    robot radius + mean squared derivative error over the planning horizon."""
    derivative=torch.diff(paths,dim=-2)*(paths.shape[-2]-1)
    return torch.cat(((paths/actor.head.radius).flatten(-2)/(2*paths.shape[-2])**.5,
        (derivative/actor.policy.planning_horizon_m).flatten(-2)/(2*derivative.shape[-2])**.5),-1)


def teacher_weights(count):
    """The row's own teacher (first) carries OWN_TEACHER_WEIGHT; the other same-pose teachers share the rest."""
    if count==1:return torch.ones(1)
    return torch.cat((torch.full((1,),OWN_TEACHER_WEIGHT),torch.full((count-1,),(1-OWN_TEACHER_WEIGHT)/(count-1))))


def energy_inputs(items):
    """Host side of generated_energy_score: rows grouped by teacher-pool size K, with their decoded teachers [n,K,P,2]
    and pool weights [K]."""
    groups=defaultdict(list)
    for index,item in enumerate(items):
        if len(item['trajectory_teacher_values']):groups[len(item['trajectory_teacher_values'])].append(index)
    return dict(count=len(items),rows=torch.tensor([i for indices in groups.values() for i in indices],dtype=torch.long),
        groups=[dict(ids=torch.as_tensor(indices),teachers=torch.stack([items[i]['trajectory_target'] for i in indices]),
            weights=teacher_weights(size)) for size,indices in groups.items()])


def _row_losses(template,count,rows,values):
    """Per-row losses, zero outside `rows`: exact copies of `values`, as stacking per-row scalars gives."""
    losses=template.new_zeros(count)
    return losses.index_copy(0,rows,torch.cat(values)) if values else losses


def generated_energy_score(actor,paths,items,*,host=None):
    """TRAJ (scalar): mean over rows of the energy score E|X-Y| - E|X-X'|/2 of the candidates X (paths [B,C,P,2])
    against the weighted same-pose teacher pool Y; zero for rows without a pool.

    Strictly proper: its minimiser is the weighted teacher distribution. Against a single teacher that minimiser is
    a point (by the triangle inequality the score is >= 0, with equality only when the candidates sit on the
    teacher), so the bank collapses; the pool of every certified teacher at the pose (own goal, relabelled goals,
    exploration) spreads the candidates over the modes while the own teacher keeps half the mass.
    """
    if host is None:host=HostTree(energy_inputs(items)).to(paths.device)
    # The epsilon keeps the gradient finite for coincident curves.
    distance=lambda a,b:((a[:,:,None]-b[:,None]).square().sum(-1)+1e-6).sqrt()
    scores=[]
    for group in host['groups']:
        candidates=_curve_features(actor,paths.index_select(0,group['ids']))
        teachers=_curve_features(actor,group['teachers'])
        count=candidates.shape[1]
        weights=group['weights'].to(candidates.dtype)
        scores.append((distance(candidates,teachers)*weights).sum(-1).mean(-1)-distance(candidates,candidates).sum((1,2))/(2*count*(count-1)))
    return _row_losses(paths,host['count'],host['rows'],scores).mean()


def relative_utility_loss(scores,targets,weights):
    """Weighted smooth-L1 between Q scores and utility targets [R,K], each minus its weighted row mean: Q learns the
    differences argmax uses; a scene-wide offset is irrelevant. Mean over rows of the weighted sum over slots."""
    mass=weights.sum(-1,keepdim=True).clamp_min(1e-12)
    predicted=scores-(scores*weights).sum(-1,keepdim=True)/mass
    target=targets-(targets*weights).sum(-1,keepdim=True)/mass
    return (F.smooth_l1_loss(predicted,target,reduction='none')*weights).sum(-1).mean()


def critic_bank_inputs(items):
    """Host side of _score_critic_banks: two length groups when the scene token counts differ."""
    lengths=[len(item['history_plan']['points']) for item in items]
    if max(lengths)==min(lengths):return dict(maximum=max(lengths),groups=None)
    rows=sorted(range(len(items)),key=lengths.__getitem__)
    middle=len(rows)//2
    return dict(maximum=max(lengths),groups=[dict(length=max(lengths[row] for row in part),ids=torch.as_tensor(part))
        for part in (rows[:middle],rows[middle:])],inverse=torch.tensor(sorted(range(len(rows)),key=rows.__getitem__)))


def _score_critic_banks(policy,condition,goals,banks,items,*,host=None):
    """Q scores [R,K] of banks [R,K,P,2]. Rows are scored in two groups of similar scene-token count, each cut to
    its own longest row, which removes most fine-cell padding while keeping every BEV/motion token."""
    if host is None:host=HostTree(critic_bank_inputs(items)).to(banks.device)
    base=condition.tokens.shape[1]-host['maximum']
    if host['groups'] is None:return policy.score_candidate_paths(_aligned_condition(condition),goals,banks)
    scores=[]
    for group in host['groups']:
        stop=base+group['length'];ids=group['ids']
        selected=_truncate_tokens(select_condition(condition,ids),stop)
        scores.append(policy.score_candidate_paths(_aligned_condition(selected),goals.index_select(0,ids),banks.index_select(0,ids)))
    return torch.cat(scores).index_select(0,host['inverse'])


def _bank_padding(experts,recorded):
    """Per-row fixed paths ([n_i,P,2] each) zero-padded to [R,max n,P,2], as pad_sequence pads, and their validity
    masks [R,max n]."""
    def padded(values):
        counts=torch.as_tensor([len(v) for v in values])
        return pad_sequence(values,batch_first=True),torch.arange(int(counts.max()))[None]<counts[:,None]
    (experts,expert_mask),(recorded,recorded_mask)=padded(experts),padded(recorded)
    return dict(experts=experts,expert_mask=expert_mask,recorded=recorded,recorded_mask=recorded_mask)


def critic_label_inputs(rows,utilities):
    """Each utility oracle's rows (in order), their fixed paths and source coordinates for the label thread."""
    groups={}
    for position,utility in enumerate(utilities):groups.setdefault(id(utility),(utility,[]))[1].append(position)
    result=[]
    for utility,positions in groups.values():
        batch=critic_sources([rows[i] for i in positions])
        metadata=batch.pop('source_metadata_cpu')
        result.append(dict(utility=utility,positions=positions,ids=None if len(groups)==1 else torch.as_tensor(positions),
            metadata=Keep(metadata),batch=batch,
            **_bank_padding([rows[i]['q_expert_paths'] for i in positions],[rows[i]['critic_replay_paths'] for i in positions])))
    return result


def candidate_weights(expert_counts,recorded_counts,experts,native,recorded):
    """[R, experts+native+recorded] Q weights of the [experts | native | recorded] slots: a quarter on the row's experts (if any), the rest split
    evenly between its current and recorded candidates; padding slots weigh nothing."""
    count=np.asarray(expert_counts);replay=np.asarray(recorded_counts)
    factor=np.where(count>0,.75,1.)
    weights=np.zeros((len(count),experts+native+recorded),dtype=np.float32)
    weights[:,:experts]=np.where(np.arange(experts)<count[:,None],(.25/np.maximum(count,1))[:,None],0.)
    weights[:,experts:experts+native]=(factor/(2*native))[:,None]
    weights[:,experts+native:]=np.where(np.arange(recorded)<replay[:,None],(factor/(2*np.maximum(replay,1)))[:,None],0.)
    return weights


def critic_inputs(items,utilities,*,fields=False,native=INFERENCE_CANDIDATES):
    """Host side of start_critic: Q-supervised rows, their fixed expert/recorded paths, candidate weights,
    length groups and the label oracle's inputs (with `fields`, also every row's local target field)."""
    indices=[i for i,item in enumerate(items) if item['q_supervision_available']]
    if not indices:return None
    lengths=[len(item['history_plan']['points']) for item in items]
    rows=[items[i] for i in indices]
    padding=_bank_padding([item['q_expert_paths'] for item in rows],[item['critic_replay_paths'] for item in rows])
    experts,recorded=padding['experts'].shape[1],padding['recorded'].shape[1]
    weights=candidate_weights([len(item['q_expert_paths']) for item in rows],[len(item['critic_replay_paths']) for item in rows],experts,native,recorded)
    subset=None if len(indices)==len(items) else dict(ids=torch.as_tensor(indices),maximum=max(lengths),length=max(lengths[i] for i in indices))
    return dict(subset=subset,count=len(items),rows=len(rows),native=native,weights=torch.from_numpy(weights),
        fields=torch.stack([item['local_target_field'] for item in items]) if fields else None,
        goals=torch.stack([item['goal'] for item in rows]),banks=critic_bank_inputs(rows),labels=critic_label_inputs(rows,[utilities[i] for i in indices]),
        target_mask=torch.cat((padding['expert_mask'],torch.ones(len(rows),native,dtype=torch.bool),padding['recorded_mask']),1),
        experts=padding['experts'],recorded=padding['recorded'])


def label_critic_targets(groups,native,present,fields,shape):
    """The current utility of every row's expert, current and recorded candidates, exactly as the reference label_critic_inputs
    scores them: one query per oracle over its rows' banks, each padded with the row's own first current candidate
    to the oracle's largest expert/recorded count. Returns [rows, experts+native+recorded] with the scores of
    padding slots left for the caller to zero."""
    experts,recorded=shape;results=[]
    for group in groups:
        ids=group['ids'];select=(lambda value:value) if ids is None else (lambda value:value.index_select(0,ids))
        rows=select(native);first=rows[:,:1]
        fill=lambda bank,mask:torch.where(mask[...,None,None],bank,first)
        paths=torch.cat((fill(group['experts'],group['expert_mask']),rows,fill(group['recorded'],group['recorded_mask'])),1)
        batch=dict(group['batch'],observed_field=select(fields),goal_present=select(present),source_metadata_cpu=group['metadata'])
        scores=group['utility'](paths,batch).score.detach()
        own,count=group['experts'].shape[1],rows.shape[1]
        results.append(torch.cat((F.pad(scores[:,:own],(0,experts-own)),scores[:,own:own+count],
            F.pad(scores[:,own+count:],(0,recorded-group['recorded'].shape[1]))),1))
    if len(groups)==1:return results[0]
    order=torch.tensor([i for group in groups for i in group['positions']])
    return torch.cat(results).index_select(0,torch.argsort(order).to(native.device))


_LABEL_STREAMS={}


def _label_in_background(groups,native,present,fields,shape,ready):
    """label_critic_targets on a side stream that waits only for `ready` (the current bank), not for the work the
    forward queues meanwhile."""
    if ready is None:return label_critic_targets(groups,native,present,fields,shape),None
    stream=_LABEL_STREAMS.setdefault(native.device,torch.cuda.Stream(native.device))
    used=[native,present,fields]
    for group in groups:
        used+=[group[key] for key in ('experts','expert_mask','recorded','recorded_mask')]+list(group['batch'].values())
        if group['ids'] is not None:used.append(group['ids'])
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        # Tensors of the forward's stream stay allocated until this stream has read them.
        for value in used:
            if value.is_cuda:value.record_stream(stream)
        targets=label_critic_targets(groups,native,present,fields,shape)
        done=torch.cuda.Event();done.record(stream)
    return targets,done


class PendingCritic:
    """Q's scores of one microbatch and the future of their utility targets; result() is LQ (scalar): the relative
    utility loss scaled by (Q-supervised rows)/(rows), so it is a mean over all rows like the other terms."""
    def __init__(self,scores=None,future=None,mask=None,weights=None,rows=0,batch_size=1,zero=None):
        self.scores,self.future,self.mask,self.weights,self.rows,self.batch_size,self.zero=scores,future,mask,weights,rows,batch_size,zero

    def result(self):
        if self.future is None:return self.zero
        targets,done=self.future.result()
        if done is not None:
            stream=torch.cuda.current_stream(targets.device);stream.wait_event(done);targets.record_stream(stream)
        # Padding slots carry zero targets, as pad_sequence pads them.
        targets=torch.where(self.mask,targets,torch.zeros((),dtype=targets.dtype,device=targets.device))
        return relative_utility_loss(self.scores.float(),targets,self.weights)*self.rows/self.batch_size


def start_critic(actor,condition,items,paths,utilities,*,host=None,fields=None):
    """Queue Q's scores of the bank; its targets are labelled meanwhile (PendingCritic.result waits for them).
    `fields` are the rows' local target fields (the LH targets) when already on the device."""
    if host is None:
        host=critic_inputs(items,utilities,fields=fields is None,native=paths.shape[1])
        if host is not None:host=HostTree(host).to(paths.device)
    if host is None:return PendingCritic(zero=condition.tokens.new_zeros(()))
    if paths.shape[1]!=host['native']:raise ValueError('Q candidate weights assume the native bank size')
    # Unavailable rows have zero Q weight; do not run their candidate banks.
    subset=host['subset']
    if fields is None:fields=host['fields']
    if subset is not None:
        stop=condition.tokens.shape[1]-subset['maximum']+subset['length'];ids=subset['ids']
        condition=_truncate_tokens(select_condition(condition,ids),stop)
        paths=paths.index_select(0,ids);fields=fields.index_select(0,ids)
    native=paths.detach();present=condition.goal_present[:,0].bool()
    ready=None
    if native.is_cuda:ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(native.device))
    future=_LABEL_WORKER.submit(_label_in_background,host['labels'],native,present,fields,
        (host['experts'].shape[1],host['recorded'].shape[1]),ready)
    banks=torch.cat((host['experts'],native,host['recorded']),1)
    scores=_score_critic_banks(actor.policy,condition,host['goals'],banks,None,host=host['banks'])
    return PendingCritic(scores,future,host['target_mask'],host['weights'],host['rows'],host['count'])


def _entry_side(normal,violating):
    """Samples still measured from the face where the curve first violates: [R,C,S] mask.

    A C-space obstacle is a thin wall inflated by the robot radius. Past its midline the nearest boundary is the
    far face, so the margin there shrinks as the curve goes deeper and its gradient pushes the curve through.
    Execution stops at first contact anyway; the samples up to the first normal reversal keep the penetration
    depth and a gradient that always points back out of the entry face.
    """
    samples=violating.shape[-1]
    first=violating.to(torch.int8).argmax(-1)
    entry=normal.gather(2,first[...,None,None].expand(-1,-1,1,2))
    # Only reversals from the first violation on: before it the curve may pass between two walls.
    after=torch.arange(samples,device=first.device)>=first[...,None]
    reversed_=(((normal*entry).sum(-1)<0)&after).to(torch.int8)
    return reversed_.cumsum(-1)==0


def geometry_inputs(records):
    """Host side of _source_geometry_risk: the rows' views [B,3] and the rows with a teacher, grouped by source
    C-space oracle."""
    groups={}
    for i,record in enumerate(records):
        if record[0]['teacher_values'] is not None:groups.setdefault(id(record[3].source_query),(record[3].source_query,[]))[1].append(i)
    rows=[i for _,members in groups.values() for i in members]
    return dict(count=len(records),views=torch.from_numpy(np.stack([r[0]['view'] for r in records])),rows=torch.tensor(rows,dtype=torch.long),
        groups=[dict(query=query,size=len(members),ids=torch.as_tensor(members),grids=[records[i][0]['source_grid_index'] for i in members],
            grid_ids=torch.as_tensor([records[i][0]['source_grid_index'] for i in members],dtype=torch.long)) for query,members in groups.values()])


def _source_geometry_risk(actor,values,paths,pad,records,*,host=None):
    """GF (scalar): mean over rows of the candidate-mean of 4*Huber of the entry-side C-space shortfall over the
    robot radius, maximised over each candidate's dense samples; zero on rows without a teacher (as CFM and TRAJ).
    `values` [B,C,D] gives the bank shape; paths [B,C*S,2] float64 are its dense samples, pad [B*C] their sampling
    error bound. With a collision-free teacher GF vanishes at
    the imitation optimum; without one its zero set includes curves that barely move, and shortening is
    the fastest way to reach it."""
    if host is None:host=HostTree(geometry_inputs(records)).to(paths.device)
    batch,count=values.shape[:2]
    views=host['views'].to(dtype=paths.dtype)
    c,s=views[:,2].cos(),views[:,2].sin()
    rotation=torch.stack((torch.stack((c,-s),-1),torch.stack((s,c),-1)),-2)
    world=paths@rotation.transpose(-1,-2)+views[:,None,:2]
    losses=[]
    for group in host['groups']:
        ids=group['ids'];size=group['size']
        retain_graph=torch.is_grad_enabled()
        with torch.enable_grad():
            probe=world.index_select(0,ids).requires_grad_(True)
            margin=signed_distance_rows(group['query'],probe,group['grid_ids'] if probe.is_cuda else group['grids']).reshape(size,count,SAMPLES)
            # Reuse the exact boundary search for both GF and its entry-face normal.
            normal,=torch.autograd.grad(margin.sum(),probe,retain_graph=retain_graph)
        # The sampling bound is a fixed safety margin for certifying the continuous curve from its samples; detached,
        # so GF's gradient moves the geometry only and never rewards a slower (shorter) curve for a smaller bound.
        deficit=((pad.detach().reshape(batch,count,1).index_select(0,ids)-margin)/actor.head.radius).clamp_min(0)
        deficit=deficit*_entry_side(normal.reshape(size,count,SAMPLES,2),deficit.detach()>0)
        risks=4*torch.where(deficit<=1,deficit.square(),2*deficit-1).amax(-1)
        losses.append(risks.mean(-1))
    return _row_losses(paths,host['count'],host['rows'],losses).mean()


def task_inputs(actor,records,*,fields=False,geometry=True):
    """Host side of native_task_losses (without `geometry`, GF prepares its own)."""
    items=[r[0] for r in records]
    return dict(generation=generation_inputs(actor.policy,items),energy=energy_inputs(items),
        critic=critic_inputs(items,[r[3] for r in records],fields=fields),geometry=geometry_inputs(records) if geometry else None)


def native_task_losses(actor,condition,records,lh_loss,*,host=None,fields=None):
    """The five losses {CFM, TRAJ, GF, LH, LQ} (in this order: the total is their sum) and the bank, for one batch of
    conditions [B]. `lh_loss()` computes LH; `fields` are the rows' local target fields [B,6,64,64] on the device.

    Q's targets are labelled on the label thread while GF and LH queue behind the critic's scores, so the device has
    work while the labels finish.
    """
    if host is None:host=HostTree(task_inputs(actor,records,fields=fields is None,geometry=False)).to(condition.tokens.device)
    items=[r[0] for r in records]
    generator_condition=_aligned_condition(condition)
    memory=actor.policy.trajectory_decoder.project_condition_memory(generator_condition)
    cfm,bank=batched_generation(actor,generator_condition,items,memory,host=host['generation'])
    trajectory=generated_energy_score(actor,bank['paths'],items,host=host['energy'])
    batch=bank['values'].shape[0]
    paths,pad=dense_curve(bank['values'].flatten(0,1))
    paths=paths.reshape(batch,-1,2).double()
    if host['critic'] is None:critic=PendingCritic(zero=condition.tokens.new_zeros(()))
    else:critic=start_critic(actor,condition,items,bank['paths'],[r[3] for r in records],host=host['critic'],fields=fields)
    gf=_source_geometry_risk(actor,bank['values'],paths,pad,records,host=host['geometry'])
    lh=lh_loss()
    return dict(CFM=cfm,TRAJ=trajectory,GF=gf,LH=lh,LQ=critic.result()),bank


def scene_order(records):
    """Rows' scenes, deduplicated in first-seen order (each scene represented by its last row)."""
    keys=[(r[0]['raw_sha256'],r[0]['history_plan_sha256']) for r in records]
    unique={key:r for key,r in zip(keys,records,strict=True)}
    positions={key:index for index,key in enumerate(unique)}
    return list(unique.values()),[positions[key] for key in keys]


def loss_inputs(actor,records,*,tasks=True):
    """Host side of batched_losses after the scene encoding (with `tasks`, also native_task_losses')."""
    items=[r[0] for r in records];_,positions=scene_order(records)
    return dict(ids=torch.tensor(positions),goals=torch.stack([item['goal'] for item in items]),
        present=torch.tensor([item['goal_present'] for item in items],dtype=torch.float32)[:,None],
        fields=torch.stack([item['local_target_field'] for item in items]),
        distance_available=torch.as_tensor([item['distance_label_available'] for item in items]),tasks=task_inputs(actor,records) if tasks else None)


def microbatch_inputs(actor,records,*,cnn_batch_size=64,bucket=None):
    """Every host input of one training batched_losses call (image chunks, fusion, rasters, targets, critic),
    built on the loader thread; replay records only (feature-map shapes follow from the image sizes)."""
    from kappanav.local_planner.features.image_batch import feature_map_inputs
    from kappanav.local_planner.features.conditions import condition_inputs
    scenes,_=scene_order(records)
    return dict(cnn=feature_map_inputs(records,cnn_batch_size),scenes=condition_inputs(actor,scenes,bucket=bucket),losses=loss_inputs(actor,records))


def batched_losses(actor,records,*,cnn_cache=None,cnn_batch_size=64):
    """({CFM, TRAJ, GF, LH, LQ} scalars, the generated bank) of one microbatch of replay records. Records prepared by
    prepare_training_batch carry their host inputs (`.host`); `cnn_cache` supplies image features (validation)."""
    from kappanav.local_planner.features.image_batch import step_feature_maps
    prepared=getattr(records,'host',None)
    device=next(actor.policy.parameters()).device
    host=None if prepared is None else prepared.to(device)
    scenes,positions=scene_order(records)
    feature_maps=step_feature_maps(actor,records,cache=cnn_cache,cnn_batch_size=cnn_batch_size,
        **({} if host is None or cnn_cache is not None else dict(host=host['cnn'])))
    encoded,(field,logits)=live_conditions(actor,scenes,feature_maps,**({} if host is None else dict(host=host['scenes'])))
    losses=HostTree(loss_inputs(actor,records,tasks=False)).to(encoded.tokens.device) if host is None else host['losses']
    tasks=losses['tasks']
    ids,goals,present,fields=losses['ids'],losses['goals'],losses['present'].to(encoded.goal_present.dtype),losses['fields']
    condition=select_condition(encoded,ids);field=field.index_select(0,ids);logits=logits.index_select(0,ids)
    condition=replace(condition,terminal_goal=torch.where(present.bool(),goals,torch.zeros_like(goals)),goal_present=present)
    lh=lambda:geometry_reconstruction_loss(field,logits,fields,distance_available=losses['distance_available'])
    return native_task_losses(actor,condition,records,lh,**({} if tasks is None else dict(host=tasks)),fields=fields)
