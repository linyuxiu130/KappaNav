"""The training replay format: NPY columns, shared frames/states, lazy row views."""
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os

import numpy as np

from kappanav.local_planner.data.array_store import ArrayTable, write_table
from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery
from kappanav.local_planner.data.value_targets import RouteUtilityTeacher
from kappanav.common.experiment_io import sha, write
from kappanav.local_planner.model.trajectory.control_points import INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE


FORMAT = 'curvenav_array_replay'


def trajectory_contract(actor):
    """The physical representation and normalization used to compile fixed targets."""
    codec = actor.policy.curve_codec
    return dict(parameterization=INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE,
                num_control_points=codec.num_control_points, degree=codec.degree,
                num_path_points=codec.num_path_points, coordinate_dim=codec.coordinate_dim,
                basis_sha256=hashlib.sha256(codec.basis.detach().cpu().numpy().tobytes()).hexdigest(),
                increment_mean_xy_m=codec.control_increment_mean_xy_m.detach().cpu().tolist(),
                increment_std_xy_m=codec.control_increment_std_xy_m.detach().cpu().tolist(),
                planning_horizon_m=actor.policy.planning_horizon_m, robot_radius_m=actor.head.radius)


def read_replay_manifest(path, *, split=None, contract=None):
    """A published split manifest; it must be this format, the requested split, and the model's trajectory contract
    (representation and normalization), the one property a model and its training rows have to share."""
    metadata = json.loads(Path(path).read_text())
    if metadata.get('format') != FORMAT or (split is not None and metadata['split'] != split) \
            or (contract is not None and metadata['trajectory_contract'] != contract):
        raise ValueError(f'Replay manifest is not a {split or "train/validation"} split of this model contract: {path}')
    return metadata


def write_shard(folder, records):
    """Columns for one source grid: rows, their shared physical states and captures, and the grids."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    raw_ids, state_ids, raws, states, rows = {}, {}, [], [], []
    for item, raw, _, _ in records:
        raw_key, state_key = item['raw_path'], item['history_plan_path']
        if raw_key not in raw_ids:
            raw_ids[raw_key] = len(raws)
            raws.append(raw)
        if state_key not in state_ids:
            state_ids[state_key] = len(states)
            states.append(dict(plan=item['history_plan'], field=item['local_target_field'],
                               frame_sources=np.unique(item['history_plan']['frame_ids'])))
        row = {key:value for key, value in item.items() if key not in ('history_plan', 'local_target_field')}
        row['_raw_index'] = raw_ids[raw_key]
        row['_state_index'] = state_ids[state_key]
        rows.append(row)
    depth_folder = folder/'raw'; depth_folder.mkdir()
    files = {}
    for index, raw in enumerate(raws):
        name = f'{index:04d}.npy'
        os.link(raw['_depth_path'], depth_folder/name)
        depth = raw['depths']
        files[name] = dict(shape=list(depth.shape), dtype=depth.dtype.str)
    raw_table = dict(rows=len(raws), schema=dict(kind='dict', fields=dict(
        depths=dict(kind='array_file', values=list(files)))), files=files)
    grids = records[0][3].source_query.grids
    metadata = dict(format=FORMAT, tables={
        'rows':write_table(folder/'rows', rows),
        'states':write_table(folder/'states', states),
        'raw':raw_table,
        'grids':write_table(folder/'grids', [dict((key, getattr(grid, key))
                                                for key in grid.__dataclass_fields__) for grid in grids])})
    write(folder/'manifest.json', metadata)
    return metadata


class _Row(Mapping):
    """Decode columns on demand; audit metadata is not rebuilt in the training loop."""
    __slots__ = ('table', 'index', 'values')

    def __init__(self, table, index, values):
        self.table, self.index, self.values = table, index, values

    def __getitem__(self, name):
        if name in ('_raw_index', '_state_index'):
            raise KeyError(name)
        if name not in self.values:
            self.values[name] = self.table.field(self.index, name)
        return self.values[name]

    def __iter__(self):
        for name in self.table.metadata['schema']['fields']:
            if name not in ('_raw_index', '_state_index') and self.table.has_field(self.index, name):
                yield name
        for key in self.values:
            if key not in self.table.metadata['schema']['fields']:yield key

    def __len__(self):
        return sum(1 for _ in self)


class _Shard:
    def __init__(self, folder):
        self.folder = Path(folder)
        metadata = json.loads((self.folder/'manifest.json').read_text())
        if metadata.get('format') != FORMAT:
            raise ValueError(f'Unsupported training shard format: {self.folder}')
        self.tables = {name:ArrayTable(self.folder/name, spec)
                       for name, spec in metadata['tables'].items()}
        self.raw = {}
        self.states = OrderedDict()

    def state(self, index):
        if index not in self.states:
            if len(self.states) == 256:
                self.states.popitem(last=False)
            self.states[index] = self.tables['states'][index]
        self.states.move_to_end(index)
        return self.states[index]

    def record(self, index, grid_offset):
        table = self.tables['rows']
        raw_index, state_index = table.field(index, '_raw_index'), table.field(index, '_state_index')
        if raw_index not in self.raw:
            self.raw[raw_index] = self.tables['raw'][raw_index]
            depth = self.tables['raw'].metadata['schema']['fields']['depths']
            if depth['kind'] == 'array_file':
                self.raw[raw_index]['_depth_path'] = str((self.folder/'raw'/depth['values'][raw_index]).resolve())
        state = self.state(state_index)
        values=dict(history_plan=state['plan'], local_target_field=state['field'],source_grid_index=table.field(index, 'source_grid_index') + grid_offset)
        item = _Row(table, index, values)
        return item, self.raw[raw_index]


class _Selection(Sequence):
    def __init__(self, parent, indices):
        self.parent, self.indices = parent, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return _Selection(self.parent, self.indices[index]) if isinstance(index, slice) else self.parent[self.indices[index]]


class _Sources(Sequence):
    def __init__(self, replay):
        self.replay = replay

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, index):
        shard, row = self.replay.location(index)
        table = shard.tables['rows']
        state = table.field(row, '_state_index')
        raw_sha = self.replay.raw_names[self.replay.columns['raw'][index]]
        frames = shard.tables['states'].field(state,'frame_sources')
        return frozenset((raw_sha, int(frame)) for frame in frames)


class ArrayReplay(Sequence):
    def __init__(self, manifests, actor, *, split):
        if isinstance(manifests, (str, Path)): manifests = [manifests]
        self.manifests = [Path(path).resolve() for path in manifests]
        contract = trajectory_contract(actor)
        parts = [read_replay_manifest(path, split=split, contract=contract) for path in self.manifests]
        entries, shard_starts = [], []
        for path, metadata in zip(self.manifests, parts, strict=True):
            shard_starts.append(len(entries))
            entries.extend((path.parent/entry['path'], entry) for entry in metadata['shards'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.shards = list(pool.map(lambda entry: _Shard(entry[0]), entries))
        columns = {name:[] for name in ('shard', 'row', 'trial', 'raw', 'goal')}
        trial_ids, raw_ids = {}, {}
        for path, metadata, offset in zip(self.manifests, parts, shard_starts, strict=True):
            catalog = ArrayTable(path.parent/metadata['catalog_path'], metadata['catalog'])
            part = {name:catalog.column(name) for name in columns}
            for name, names, identities in (('trial', metadata['trials'], trial_ids),
                                            ('raw', metadata['raw_captures'], raw_ids)):
                mapping = np.asarray([identities.setdefault(value, len(identities)) for value in names], dtype=np.int64)
                part[name] = mapping[part[name]]
            part['shard'] = part['shard'] + offset
            for name, value in part.items(): columns[name].append(value)
        self.columns = {name:np.concatenate(values) for name, values in columns.items()}
        self.raw_names = tuple(raw_ids)
        self.case_columns = tuple(self.columns[name] for name in ('trial', 'raw', 'goal'))
        self.sources = _Sources(self)
        grids, self.source_grid_ids = [], set()
        self.grid_offsets = []
        for shard in self.shards:
            self.grid_offsets.append(len(grids))
            grids.extend(SourceConfigurationGrid(**row) for row in shard.tables['grids'])
            table = shard.tables['rows']
            if len(table): self.source_grid_ids.add(table.field(0, 'source_grid_sha256'))
        self.utility = RouteUtilityTeacher(SourceConfigurationSpaceQuery(tuple(grids)),
                                           actor.policy.planning_horizon_m, actor.head.radius)

    def __len__(self):
        return len(self.columns['row'])

    def select(self, indices):
        return _Selection(self, indices)

    def location(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        shard = int(self.columns['shard'][index])
        return self.shards[shard], int(self.columns['row'][index])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _Selection(self, range(len(self))[index])
        shard_index = int(self.columns['shard'][index])
        shard, row = self.location(index)
        item, raw = shard.record(row, self.grid_offsets[shard_index])
        return item, raw, None, self.utility


def load_array_replay(actor, manifests, *, split):
    """The split's array replay, its goal and no-goal row indices, and the files it was read from."""
    replay = ArrayReplay(manifests, actor, split=split)
    present = replay.case_columns[2]
    files = {str(path):sha(path) for path in replay.manifests}
    files.update((str(shard.folder/'manifest.json'), sha(shard.folder/'manifest.json')) for shard in replay.shards)
    return replay, np.flatnonzero(present), np.flatnonzero(~present), dict(format=FORMAT, files=files, rows=len(replay))
