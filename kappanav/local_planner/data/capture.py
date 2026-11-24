"""Immutable capture arrays in NPY files, with hash-bound JSON acquisition metadata."""
import json
from pathlib import Path
import numpy as np
from kappanav.common.experiment_io import sha, write

FORMAT = 'curvenav_capture'


def save_capture(path, arrays, meta):
    path = Path(path)
    files = {}
    for name, value in arrays.items():
        value = np.asarray(value)
        target = path.parent/(name+'.npy')
        with target.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        files[name] = dict(shape=list(value.shape), dtype=value.dtype.str, sha256=sha(target))
    write(path, dict(format=FORMAT, arrays=files, metadata=meta))


def load_capture(path, *, arrays=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest['format'] != FORMAT:
        raise ValueError(f'Expected a native NPY capture: {path}')
    raw = {}
    for name in manifest['arrays'] if arrays is None else arrays:
        entry = manifest['arrays'][name]
        target = path.parent/(name+'.npy')
        if sha(target) != entry['sha256']:
            raise ValueError(f'Capture array changed: {target}')
        raw[name] = np.load(target, allow_pickle=False)
    raw.update(manifest['metadata'])
    if 'depths' in raw:
        raw['_depth_path'] = str((path.parent/'depths.npy').resolve())
    return raw
