"""Columnar NPY storage for replay tables; reconstruct only requested rows."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch



_MISSING = object()


class _Writer:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.files = {}

    def array(self, value):
        name = f'{len(self.files):04d}.npy'
        path = self.folder / name
        np.save(path, value, allow_pickle=False)
        self.files[name] = dict(shape=list(value.shape), dtype=value.dtype.str)
        return name

    def dense(self, values, dtype, shapes):
        fixed = all(shape == shapes[0] for shape in shapes)
        name = f'{len(self.files):04d}.npy'
        path = self.folder / name
        shape = (len(values), *shapes[0]) if fixed else (sum(int(np.prod(s)) for s in shapes),)
        target = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        offset = 0
        offsets = [0]
        for index, value in enumerate(values):
            if fixed:
                target[index] = value
            else:
                target[offset:offset + value.size] = value.reshape(-1)
                offset += value.size
                offsets.append(offset)
        target.flush()
        del target
        self.files[name] = dict(shape=list(shape), dtype=dtype.str)
        result = dict(kind='array', values=name)
        if not fixed:
            result.update(offsets=self.array(np.asarray(offsets, dtype=np.int64)),
                          shapes=self.array(np.asarray(shapes, dtype=np.int64)))
        return result

    def node(self, values):
        values = [value.item() if isinstance(value, np.generic) else value for value in values]
        valid = [value is not None and value is not _MISSING for value in values]
        if not all(valid):
            indices = np.full(len(values), -1, dtype=np.int64)
            indices[[i for i, value in enumerate(values) if value is _MISSING]] = -2
            indices[valid] = np.arange(sum(valid))
            return dict(kind='optional', index=self.array(indices),
                        child=self.node([v for v, keep in zip(values, valid) if keep]) if any(valid) else None)
        first = values[0]
        if isinstance(first, dict):
            keys = dict.fromkeys(key for value in values for key in value)
            return dict(kind='dict', fields={key:self.node([value.get(key, _MISSING) for value in values])
                                            for key in keys})
        if isinstance(first, (tuple, list)):
            if all(len(value) == len(first) for value in values):
                return dict(kind='tuple' if isinstance(first, tuple) else 'list',
                            children=[self.node([value[i] for value in values]) for i in range(len(first))])
            offsets = np.r_[0, np.cumsum([len(value) for value in values])].astype(np.int64)
            return dict(kind='ragged_tuple' if isinstance(first, tuple) else 'ragged_list',
                        offsets=self.array(offsets), child=self.node([v for value in values for v in value]))
        if isinstance(first, (torch.Tensor, np.ndarray)):
            tensor = isinstance(first, torch.Tensor)
            arrays = [value.detach().cpu().numpy() if tensor else value for value in values]
            dtype = arrays[0].dtype
            result = self.dense(arrays, dtype, [tuple(value.shape) for value in arrays])
            result['tensor'] = tensor
            return result
        if isinstance(first, (str, bool, int, float)):
            if all(value == first for value in values):
                return dict(kind='constant', value=first)
            if isinstance(first, str):
                pool = dict.fromkeys(values)
                codes = {value:index for index, value in enumerate(pool)}
                encoded = [value.encode('utf-8') for value in pool]
                return dict(kind='string', codes=self.array(np.asarray([codes[v] for v in values], dtype=np.int32)),
                            data=self.array(np.frombuffer(b''.join(encoded), dtype=np.uint8)),
                            offsets=self.array(np.r_[0, np.cumsum([len(v) for v in encoded])].astype(np.int64)))
            return dict(kind='scalar', values=self.array(np.asarray(values)))
        raise TypeError(f'Unsupported replay column: {type(first).__name__}')


def write_table(folder, rows):
    if not rows:
        return dict(rows=0, schema=None, files={})
    writer = _Writer(folder)
    return dict(rows=len(rows), schema=writer.node(rows), files=writer.files)


def write_columns(folder, columns):
    writer = _Writer(folder)
    count = len(next(iter(columns.values())))
    fields = {key:dict(kind='scalar', values=writer.array(value)) for key, value in columns.items()}
    return dict(rows=count, schema=dict(kind='dict', fields=fields), files=writer.files)


class ArrayTable:
    def __init__(self, folder, metadata):
        self.folder = Path(folder)
        self.metadata = metadata
        def load(name):
            return name, np.load(self.folder/name, allow_pickle=False)
        # Read each array once, sequentially, into memory. The data live on GPFS, where faulting
        # memory-mapped pages in is ~100x slower than a sequential read; the forkserver loads the
        # tables before creating ranks, which then share these pages copy-on-write.
        if len(metadata['files']) <= 1:
            self._arrays = dict(map(load, metadata['files']))
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:
                self._arrays = dict(pool.map(load, metadata['files']))

    def __len__(self):
        return self.metadata['rows']

    def column(self, name):
        return self.array(self.metadata['schema']['fields'][name]['values'])

    def field(self, row, name):
        value = self._read(self.metadata['schema']['fields'][name], row)
        if value is _MISSING:
            raise KeyError(name)
        return value

    def has_field(self, row, name):
        node = self.metadata['schema']['fields'][name]
        return node['kind'] != 'optional' or self.array(node['index'])[row] != -2

    def array(self, name):
        return self._arrays[name]

    def _read(self, node, row):
        kind = node['kind']
        if kind == 'optional':
            index = int(self.array(node['index'])[row])
            return None if index == -1 else _MISSING if index == -2 else self._read(node['child'], index)
        if kind == 'dict':
            result = {key:self._read(child, row) for key, child in node['fields'].items()}
            return {key:value for key, value in result.items() if value is not _MISSING}
        if kind in ('tuple', 'list'):
            values = [self._read(child, row) for child in node['children']]
            return tuple(values) if kind == 'tuple' else values
        if kind in ('ragged_tuple', 'ragged_list'):
            start, stop = self.array(node['offsets'])[row:row + 2]
            values = [self._read(node['child'], index) for index in range(start, stop)]
            return tuple(values) if kind == 'ragged_tuple' else values
        if kind == 'constant':
            return node['value']
        if kind == 'array_file':
            return self.array(node['values'][row])
        if kind == 'string':
            code = int(self.array(node['codes'])[row])
            start, stop = self.array(node['offsets'])[code:code + 2]
            return self.array(node['data'])[start:stop].tobytes().decode('utf-8')
        value = self.array(node['values'])
        if kind == 'scalar':
            return value[row].item()
        if 'offsets' in node:
            start, stop = self.array(node['offsets'])[row:row + 2]
            value = value[start:stop].reshape(tuple(self.array(node['shapes'])[row]))
        else:
            value = value[row]
        return torch.from_numpy(np.asarray(value)) if node['tensor'] else value

    def __getitem__(self, row):
        if row < 0:
            row += len(self)
        return self._read(self.metadata['schema'], row)
