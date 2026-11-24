"""Compile capture records into native training scenes and publish their dataset indexes."""
from copy import deepcopy
from pathlib import Path
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from kappanav.local_planner.data.array_store import write_columns
from kappanav.local_planner.data.bspline_dataset import attach_q_expert_pools, attach_trajectory_teacher_pools
from kappanav.local_planner.data.replay_pack import FORMAT, write_shard, trajectory_contract
from kappanav.common.experiment_io import sha, write
from kappanav.local_planner.data.replay_targets import prepare_replay_targets


DATASET_FORMAT = 'curvenav_array_dataset'

TRAINING_FIELDS = {
    'step', 'view', 'acquisitions', 'teacher_values', 'trial_id', 'raw_path', 'raw_sha256',
    'history_plan_path', 'history_plan_sha256', 'source_grid_sha256', 'history_plan',
    'local_target_field', 'distance_label_available', 'goal_present', 'goal', 'source_grid_index',
    'q_supervision_available', 'critic_replay_paths', 'trajectory_teacher_values', 'cfm_target',
    'trajectory_target', 'q_expert_paths',
}


def prepare_scene(actor, records, destination, *, provenance):
    """Compile a scene in memory and atomically publish its final training shard."""
    destination = Path(destination)
    marker = destination/'scene.json'
    contract = trajectory_contract(actor)
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata['provenance'] != provenance or metadata.get('trajectory_contract') != contract:
            raise ValueError(f'Scene inputs changed: {destination}; use a new output directory')
        return metadata
    split = provenance['split']
    captures = []
    for item, *_ in records:
        key = (item['trial_id'], item['raw_sha256'])
        if not captures or (captures[-1]['trial_id'], captures[-1]['raw_sha256']) != key:
            captures.append(dict(trial_id=key[0], raw_sha256=key[1], goals=[]))
        captures[-1]['goals'].append(bool(item['goal_present']))
    metadata = dict(format=FORMAT, split=split, rows=len(records), captures=captures,
                    provenance=provenance, trajectory_contract=contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.scene-') as temporary:
        prepared = Path(temporary)/'shard'
        if records:
            attach_q_expert_pools(records)
            attach_trajectory_teacher_pools(records)
            # Immutable targets are CPU arrays; only their small codec is needed from the GPU model.
            target_actor = SimpleNamespace(policy=SimpleNamespace(curve_codec=deepcopy(actor.policy.curve_codec).cpu()))
            prepare_replay_targets(target_actor, records)
            clean = [({key:value for key, value in item.items() if key in TRAINING_FIELDS},
                      raw, tree, utility) for item, raw, tree, utility in records]
            write_shard(prepared, clean)
            metadata['shard_manifest_sha256'] = sha(prepared/'manifest.json')
        else:
            prepared.mkdir()
        write(prepared/'scene.json', metadata)
        os.rename(prepared, destination)
    return metadata


def _scene_contract(scenes):
    contract = scenes[0][1]['trajectory_contract'] if scenes else None
    if any(metadata['trajectory_contract'] != contract for _, metadata in scenes):
        raise ValueError('Scene trajectory representations or normalization differ; rebuild with one bundle contract')
    return contract


def _scene_catalog(scenes, folder, split, contract):
    count = sum(metadata['rows'] for _, metadata in scenes)
    manifest_path = folder/'manifest.json'
    inputs = [dict(path=str(path), sha256=sha(path)) for path, _ in scenes]
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get('scenes') != inputs:
            raise ValueError(f'Dataset scenes changed: {folder}; use a new output directory')
        return manifest_path, count
    columns = {name:np.empty(count, dtype=dtype) for name, dtype in
               [('shard', np.int32), ('row', np.int64), ('trial', np.int32), ('raw', np.int32), ('goal', bool)]}
    trials, raws, shards, cursor = {}, {}, [], 0
    for path, metadata in scenes:
        if not metadata['rows']:
            continue
        shard = len(shards)
        start = cursor
        for capture in metadata['captures']:
            trial, raw = capture['trial_id'], capture['raw_sha256']
            trials.setdefault(trial, len(trials));raws.setdefault(raw, len(raws))
            stop = cursor + len(capture['goals'])
            columns['trial'][cursor:stop] = trials[trial]
            columns['raw'][cursor:stop] = raws[raw]
            columns['goal'][cursor:stop] = capture['goals']
            cursor = stop
        columns['shard'][start:cursor] = shard
        columns['row'][start:cursor] = np.arange(metadata['rows'])
        shards.append(dict(path=os.path.relpath(path.parent, folder), rows=metadata['rows']))
    with tempfile.TemporaryDirectory(dir=folder.parent, prefix='.catalog-') as temporary:
        prepared = Path(temporary)/split
        prepared.mkdir()
        catalog = write_columns(prepared/'catalog', columns)
        write(prepared/'manifest.json', dict(format=FORMAT, split=split, shards=shards,
              catalog_path='catalog', catalog=catalog, trials=list(trials), raw_captures=list(raws),
              scenes=inputs, trajectory_contract=contract))
        os.rename(prepared, folder)
    return manifest_path, count


def finalize_scenes(scene_paths, output):
    """Publish only catalogs: scene NPY payloads are already in their training format."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scenes = []
    for path in scene_paths:
        path = Path(path)
        path = (path/'scene.json' if path.is_dir() else path).resolve()
        metadata = json.loads(path.read_text())
        if metadata['rows'] and sha(path.parent/'manifest.json') != metadata['shard_manifest_sha256']:
            raise ValueError(f'Scene manifest changed: {path}')
        scenes.append((path, metadata))
    contract = _scene_contract(scenes)
    inputs = [dict(path=str(path), sha256=sha(path)) for path, _ in scenes]
    dataset = output/'dataset.json'
    if dataset.exists():
        if json.loads(dataset.read_text()).get('scenes') != inputs:
            raise ValueError(f'Dataset scenes changed: {output}; use a new output directory')
        return dataset
    manifests, rows = {}, {}
    for split in ('train', 'validation'):
        part = [(path, metadata) for path, metadata in scenes if metadata['split'] == split]
        if sum(metadata['rows'] for _, metadata in part):
            path, rows[split] = _scene_catalog(part, output/split, split, contract)
            manifests[split] = [str(path.resolve())]
        else:
            manifests[split], rows[split] = [], 0
    write(dataset, dict(format=DATASET_FORMAT, manifests=manifests, rows=rows,
                        scenes=inputs, trajectory_contract=contract))
    return dataset
