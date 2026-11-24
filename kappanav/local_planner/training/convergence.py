"""Per-epoch validation of the EMA weights: the five losses and three behaviour metrics, each a trial-uniform mean
over every validation state and an equally large trial-uniform sample of training states (fixed RNG)."""
from collections import defaultdict
import numpy as np
import torch
import torch.distributed as dist
from kappanav.local_planner.training.bspline_objective import batched_losses


def monitored_rows(trial_ids,share):
    """Evenly spaced rows of every trial, at least one; the trial-uniform mean keeps its expectation."""
    order=np.argsort(trial_ids,kind='stable');rows=[]
    for part in np.split(order,np.cumsum(np.bincount(trial_ids))[:-1]):
        if len(part):rows.append(part[np.unique(np.linspace(0,len(part)-1,max(1,round(len(part)*share))).round().astype(int))])
    return np.concatenate(rows)


@torch.no_grad()
def behavior_metrics(actor,batch,feature_maps):
    """What closed-loop behaviour depends on, scored by the privileged route utility on the deployed choice:
    utility of the Q-selected candidate (deployment rule: Q argmax among perceived-field-clear candidates),
    utility of the best of the generated candidates, and whether the selected one collides. The fitted-loss
    terms measure closeness to a single teacher per state; these measure the decision itself."""
    from kappanav.local_planner.features.conditions import live_conditions
    from kappanav.local_planner.data.replay_targets import critic_sources
    condition,_=live_conditions(actor,batch,feature_maps)
    bank=actor.candidate_batches(condition,condition.terminal_goal)
    device=bank['paths'].device;paths=bank['paths'].float().cpu();selected=bank['selected_index'].cpu()
    utility=batch[0][3]
    if any(row[3] is not utility for row in batch):raise ValueError('behavior_metrics scores one scene per batch')
    rows=[item for item,*_ in batch]
    inputs=dict(critic_sources(rows),observed_field=torch.stack([item['local_target_field'] for item in rows]),
        goal_present=torch.tensor([bool(item['goal_present']) for item in rows]))
    result=utility(paths,inputs)
    rows=torch.arange(len(batch));score=result.score.cpu();failed=result.failed.cpu().float()
    # On the losses' device: validation stacks them with the fitted terms.
    return {name:value.to(device) for name,value in dict(SEL_U=score[rows,selected],BEST_U=score.amax(-1),SEL_FAIL=failed[rows,selected]).items()}


@torch.no_grad()
def validation_losses(actor,records,rank,world,device,control,cnn_cache=None,*,batch_size=2,share=1.):
    """{CFM, TRAJ, GF, LH, LQ, SEL_U, BEST_U, SEL_FAIL}: trial-uniform means over `share` of each trial's rows
    (monitored_rows), the trials split across ranks and the sums all-reduced on the `control` group."""
    names=('CFM','TRAJ','GF','LH','LQ','SEL_U','BEST_U','SEL_FAIL')
    totals=torch.zeros(len(names)+1,dtype=torch.float64,device=device)
    trial_ids=records.case_columns[0];selected=monitored_rows(trial_ids,share) if share<1 else np.arange(len(records))
    counts=np.bincount(trial_ids[selected]);trial_count=int(np.count_nonzero(counts))
    tasks=defaultdict(list)
    for index in np.sort(selected)[rank::world]:tasks[int(trial_ids[index])].append(int(index))
    with torch.random.fork_rng(devices=[device] if device.type=='cuda' else []):
        torch.manual_seed(20261028+rank)
        for trial,indices in tasks.items():
            rows=records.select(indices)
            if cnn_cache is None:
                from kappanav.local_planner.features.image_batch import step_feature_maps
                trial_cache=step_feature_maps(actor,list(rows))
            else:trial_cache=cnn_cache
            for start in range(0,len(rows),batch_size):
                batch=list(rows[start:start+batch_size])
                terms,_=batched_losses(actor,batch,cnn_cache=trial_cache)
                terms.update({name:value.mean() for name,value in behavior_metrics(actor,batch,trial_cache).items()})
                weight=len(batch)/(trial_count*counts[trial])
                totals[0]+=weight
                totals[1:]+=torch.stack([terms[k].detach().double().to(device) for k in names])*weight
    totals=totals.cpu();dist.all_reduce(totals,group=control)
    return dict(zip(names,(totals[1:]/totals[0]).tolist(),strict=True))

@torch.no_grad()
def comparison_losses(actor,training,validation,rank,world,device,control,cnn_cache=None,*,batch_size=2):
    """Every validation state and an equally large trial-uniform sample of training states,
    retaining image maps for just one physical scene."""
    share=min(1.,len(validation)/len(training))
    return {name:validation_losses(actor,records,rank,world,device,control,cnn_cache,batch_size=batch_size,share=part)
        for name,records,part in (('train',training,share),('validation',validation,1.))}
