"""The training and validation replay, opened once in the forkserver before the CUDA ranks fork (they share the
arrays copy-on-write): TRAINING and VALIDATION are load_array_replay results, CASES the trial/goal-mode sampling
groups and SOURCE_KEYS each row's frame identities (for balanced_batch_indices)."""
import json
import os
from pathlib import Path

import torch

from kappanav.local_planner.data.bspline_dataset import case_groups
from kappanav.local_planner.data.replay_pack import load_array_replay
from kappanav.local_planner.inference.bundle import load_bundle


def _load():
    torch.set_num_threads(1)
    bundle, training, validation = json.loads(os.environ['KAPPANAV_SHARED_REPLAY_INPUTS'])
    actor = load_bundle(Path(bundle), 'cpu')
    train = load_array_replay(actor, list(map(Path, training)), split='train')
    held_out = load_array_replay(actor, list(map(Path, validation)), split='validation') if validation else ([], [], [], None)
    rows = train[0]
    if held_out[0] and rows.source_grid_ids & held_out[0].source_grid_ids:
        raise ValueError('Training and validation contain the same physical source grid')
    trials, raw_keys, present = rows.case_columns
    cases = case_groups(trials, raw_keys, present, range(len(rows)), stratify_conditions=True)
    return train, held_out, rows.sources, cases


TRAINING, VALIDATION, SOURCE_KEYS, CASES = _load()
LOADER_PID = os.getpid()
