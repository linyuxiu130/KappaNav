"""Current array dataset inputs, expert pools and task-balanced state sampling."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import torch

SAMPLING = 'uniform_trial_all_state_cycle'
TRAIN_SAMPLING = 'uniform_physical_trial_goal_mode_random_anchor_cycle'


def dataset_manifests(paths):
    """The train/validation split manifests and row counts of one or several published datasets (roots or
    dataset.json files), concatenated in the given order. Datasets combine only under one trajectory contract."""
    paths=[paths] if isinstance(paths,(str,Path)) else list(paths)
    manifests=dict(train=[],validation=[]);rows=dict(train=0,validation=0);contract=None
    for path in map(Path,paths):
        if path.is_dir():path=path/'dataset.json'
        path=path.resolve();summary=json.loads(path.read_text())
        if summary.get('format')!='curvenav_array_dataset':raise ValueError(f'Expected a published dataset.json: {path}')
        if contract is None:contract=summary.get('trajectory_contract')
        elif summary.get('trajectory_contract')!=contract:raise ValueError(f'{path} has another trajectory contract than {paths[0]}')
        for part in ('train','validation'):
            manifests[part]+=[(path.parent/entry).resolve() for entry in summary['manifests'][part]];rows[part]+=int(summary['rows'][part])
    return manifests,rows


def case_groups(trials, raw_keys, present, indices, *, stratify_conditions=False):
    """Group trials/conditions; retain source order as stable anchor IDs for sampling."""
    cases = {}
    for index in indices:
        key=trials[index]
        cases.setdefault(key, {}).setdefault(raw_keys[index], []).append(index)
    result=[]
    for histories in cases.values():
        states=[i for rows in histories.values() for i in rows];group=[states]
        if stratify_conditions:
            phases={}
            for i in states:
                key=bool(present[i])
                phases.setdefault(key,[]).append(i)
            if len(phases)>1:group.extend(phases.values())
        result.append(group)
    return result

@lru_cache(maxsize=1024)
def _anchor_order(seed, trial, cycle, length):
    # Equal-sized goal/no-goal strata share the order to reuse their current CNN features.
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial, cycle]))
    return tuple(rng.permutation(length))


def sample_index(groups, position, *, seed=None):
    trial = position % len(groups)
    group = groups[trial]
    visit = position // len(groups)
    if len(group)>1:
        states=group[1+visit%(len(group)-1)];visit//=len(group)-1
    else:states=group[0]
    cycle, slot = divmod(visit, len(states))
    if seed is not None:
        slot = _anchor_order(seed, trial, cycle, len(states))[slot]
    return states[slot]

def attach_q_expert_pools(items):
    """Group admitted experts by source geometry and exact physical capture pose."""
    groups = {}
    for item, *_ in items:
        key = (item['source_grid_sha256'], np.asarray(item['view'], dtype=np.float64).tobytes())
        groups.setdefault(key, []).append(item)
    for members in groups.values():
        experts = {}
        for item in members:
            if item['teacher_values'] is None:
                continue
            values = item['teacher_values'].float().contiguous()
            experts.setdefault(values.numpy().tobytes(), values)
        for item in members:
            own = None if item['teacher_values'] is None else item['teacher_values'].float().contiguous().numpy().tobytes()
            ordered = ([own] if own is not None else []) + [key for key in experts if key != own]
            item['q_expert_values'] = torch.stack([experts[key] for key in ordered]) if ordered else torch.empty(0, 14)

def attach_trajectory_teacher_pools(items):
    """G's teacher pool at a row: its own teacher first, then every other certified teacher at the same causal
    history and pose (relabelled goals, exploration). generated_energy_score gives the own teacher half the mass.
    A row whose own teacher failed qualification has no G target; another row's teacher cannot stand in for it.
    """
    groups = {}
    for item, *_ in items:
        key = (item['raw_sha256'], item['acquisitions'], item['history_plan_sha256'],
               np.asarray(item['view'], dtype=np.float64).tobytes())
        groups.setdefault(key, []).append(item)
    for members in groups.values():
        teachers = {}
        for item in members:
            if item['teacher_values'] is not None:
                value = item['teacher_values'].float().contiguous()
                teachers.setdefault(value.numpy().tobytes(), value)
        for item in members:
            if item['teacher_values'] is None:
                item['trajectory_teacher_values'] = torch.empty(0, 14);continue
            own = item['teacher_values'].float().contiguous()
            others = [value for key, value in teachers.items() if key != own.numpy().tobytes()]
            item['trajectory_teacher_values'] = torch.stack([own] + others)
