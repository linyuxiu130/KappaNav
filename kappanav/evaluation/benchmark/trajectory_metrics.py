"""Offline candidate collision counts;never alter policy selection or execution."""
import numpy as np
import torch
from kappanav.local_planner.model.trajectory import path_arc_length


@torch.no_grad()
def trajectory_metrics(navigation,raw,plans,device):
    oracle=navigation.oracle;collisions=[]
    paths=raw['candidate_paths'];count=paths.shape[1]
    for start in range(0,len(paths),32):
        batch=torch.as_tensor(paths[start:start+32],device=device)
        views=torch.as_tensor(raw['acquired_views'][start:start+32],device=device,dtype=torch.float64)
        flat=batch.flatten(0,1)
        horizon=max(float(path_arc_length(flat.double()).max()),1e-8)
        query=oracle.query(flat,torch.zeros(len(flat),device=device,dtype=torch.long),
            views[:,:2].repeat_interleave(count,0),views[:,2].repeat_interleave(count),horizon)
        collisions.append((query.minimum_clearance_m<0).cpu().numpy().reshape(-1,count))
    collision=np.concatenate(collisions)
    rows=np.arange(len(collision));selected=collision[rows,raw['selected_indices']]
    # The selector trusts the perceived field: a selected curve it showed clear but that collides is a perception miss.
    clear=np.asarray(raw['candidate_clear'],dtype=bool);selected_clear=clear[rows,raw['selected_indices']]
    # plans[i] is decision i's publication; a controller failure ends the episode before the last decision is published.
    published=np.zeros(len(paths),dtype=bool);published[:len(plans)]=[p['applied'] for p in plans]
    return dict(decisions=len(paths),candidates=int(collision.size),candidate_collisions=int(collision.sum()),
        selected_collisions=int(selected.sum()),no_safe_candidate_decisions=int(collision.all(-1).sum()),
        field_missed_candidate_collisions=int((clear&collision).sum()),field_missed_selected_collisions=int((selected_clear&selected).sum()),
        no_field_clear_candidate_decisions=int((~clear).all(-1).sum()),
        published_decisions=int(published.sum()),published_selected_collisions=int(selected[published].sum()))


def aggregate_trajectory_metrics(rows):
    counts={key:sum(row[key] for row in rows) for key in rows[0]}
    available=counts['decisions']-counts['no_safe_candidate_decisions']
    selection_errors=counts['selected_collisions']-counts['no_safe_candidate_decisions']
    return dict(**counts,selection_error_with_safe_candidate_count=selection_errors,
        selection_error_with_safe_candidate_rate=selection_errors/available if available else None,candidate_collision_rate=counts['candidate_collisions']/counts['candidates'],
        selected_collision_rate=counts['selected_collisions']/counts['decisions'],
        no_safe_candidate_rate=counts['no_safe_candidate_decisions']/counts['decisions'],
        field_missed_candidate_rate=counts['field_missed_candidate_collisions']/max(counts['candidate_collisions'],1),
        no_field_clear_candidate_rate=counts['no_field_clear_candidate_decisions']/counts['decisions'],
        published_selected_collision_rate=counts['published_selected_collisions']/counts['published_decisions'])
