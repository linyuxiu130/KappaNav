"""Navigation and expert-relative exploration metrics,independent of policy execution."""
import numpy as np


def goal_metrics(reached,reference_m,travel_m):
    """SR and SPL; the reference is the shortest path, or the straight start-goal distance under the reference benchmark's protocol."""
    return dict(success=bool(reached),spl=float(reached)*reference_m/max(reference_m,travel_m))


def first_area_time(times,areas,target):
    # Plateaus are valid during backtracking;use the first crossing,not a linear-growth assumption.
    index=int(np.searchsorted(areas,target,side='left'))
    if index==0:return float(times[0])
    before=index-1
    fraction=(target-areas[before])/(areas[index]-areas[before])
    return float(times[before]+fraction*(times[index]-times[before]))


def exploration_metrics(times,areas,expert_times,expert_areas,reachable_area_m2,collision,*,initial_area_m2=0.,expert_collision=False,expert_failure=False,failure=False):
    """Areas are cumulative newly observed reachable free area,excluding the common initial view.

    Expert and agent run from the same start for the same budget with the same sensor/controller.
    E-SPL is a benchmark definition here,not a metric of the reference benchmark or an optimal-cover guarantee.
    """
    area=float(areas[-1]);reference=float(expert_areas[-1]);common=min(area,reference)
    if common==0:
        relative=0.;agent_time=expert_time=0.
    else:
        agent_time=first_area_time(times,areas,common)
        expert_time=first_area_time(expert_times,expert_areas,common)
        relative=min(area/reference,1.)*expert_time/max(expert_time,agent_time)
    valid=not (expert_collision or expert_failure)
    return dict(expert_valid=valid,explored_area_m2=area+initial_area_m2,new_area_m2=area,coverage=(area+initial_area_m2)/reachable_area_m2,
        expert_area_m2=reference,common_area_m2=common,agent_common_area_time_s=agent_time,
        expert_common_area_time_s=expert_time,relative_area_time_efficiency=relative if valid else None,
        exploration_spl=(0. if collision or failure else relative) if valid else None)

def summarize(rows):
    """Groups: each source, each shortest-path range of tasks that have one (range_m), and all."""
    from kappanav.evaluation.benchmark.trajectory_metrics import aggregate_trajectory_metrics
    result={}
    ranges=sorted({tuple(r['range_m']) for r in rows if r.get('range_m')})
    groups={**{source:[r for r in rows if r['source']==source] for source in sorted({r['source'] for r in rows})},
        **{f'{low:g}-{high:g} m':[r for r in rows if r.get('range_m') and tuple(r['range_m'])==(low,high)] for low,high in ranges},
        'all':rows}
    for name,group in groups.items():
        if not group:continue
        result[name]=dict(episodes=len(group),success_count=sum(r['goal']['success'] for r in group),
            success_rate=float(np.mean([r['goal']['success'] for r in group])),spl=float(np.mean([r['goal']['spl'] for r in group])),
            goal_collision_rate=float(np.mean([r['goal_collision'] for r in group])),
            goal_controller_failure_rate=float(np.mean([r['goal_status']=='controller_failure' for r in group])),
            goal_generation=aggregate_trajectory_metrics([r['goal_trajectory'] for r in group])
                if all(r['goal_trajectory'] is not None for r in group) else None)
        if all('exploration' in r for r in group):
            valid=[r['exploration']['exploration_spl'] for r in group if r['exploration']['expert_valid']]
            result[name].update(exploration_reference_valid_episodes=len(valid),
                exploration_reference_collision_rate=float(np.mean([r['expert_collision'] for r in group])),
                exploration_reference_controller_failure_rate=float(np.mean([r['expert_status']=='controller_failure' for r in group])),
                exploration_controller_failure_rate=float(np.mean([r['exploration_status']=='controller_failure' for r in group])),
                exploration_generation=aggregate_trajectory_metrics([r['exploration_trajectory'] for r in group]),
                mean_new_area_m2=float(np.mean([r['exploration']['new_area_m2'] for r in group])),
                mean_explored_area_m2=float(np.mean([r['exploration']['explored_area_m2'] for r in group])),
                mean_expert_new_area_m2=float(np.mean([r['exploration']['expert_area_m2'] for r in group])),
                coverage=float(np.mean([r['exploration']['coverage'] for r in group])),
                exploration_spl=float(np.mean(valid)) if len(valid)==len(group) else None,
                exploration_collision_rate=float(np.mean([r['exploration_collision'] for r in group])))
    return result


def mean_decision_latency_ms(plans):
    """Exclude the immediate first plan and decisions that were never published before termination."""
    latencies=[p['decision_latency_ms'] for p in plans[1:] if p['applied']]
    return float(np.mean(latencies)) if latencies else None
