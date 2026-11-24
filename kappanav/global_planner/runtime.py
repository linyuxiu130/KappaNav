"""Atomic checkpoints, pinned model identity and exact epoch-boundary recovery."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModel

from kappanav.global_planner.dataset import WORK_GSD_M
from kappanav.global_planner.taxonomy import CLASSES

BACKBONE = Path(os.environ.get('KAPPANAV_GLOBAL_BACKBONE', '/shibo_huang/models/dinov3-sat'))
ARCHITECTURE = 'dinov3-sat-vitl16-dense128-gsd-film-masked-readout-local4'
TASK = 'ground-value9'
SAT_SHA256 = 'eadcf0ffc02418b6c22a885ea1a7aaeeef84fbf0f5bb4d0b7d1d36e68a964f48'


def stream_digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return stream_digest(stream)


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, records):
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(record, allow_nan=False) + '\n' for record in records))
    temporary.replace(path)


def atomic_save(state, path):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def source_identity():
    package = Path(__file__).parent
    return {str(path.relative_to(package)): digest(path) for path in sorted(package.rglob('*.py'))}


def distributed_device(requested):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    device = (torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
              if world > 1 and requested != 'cpu' else torch.device(requested))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return device, rank, world


def rng_state(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def load_backbone(directory, *, pretrained=True):
    """Load SAT weights, or build a skeleton for immediate full-checkpoint restoration."""
    identity = json.loads((directory / 'source.json').read_text())
    if identity.get('original_sha256') != SAT_SHA256:
        raise ValueError('Expected the pinned official DINOv3-SAT ViT-L/16 weights')
    identity['config_sha256'] = digest(directory / 'config.json')
    if pretrained:
        if digest(directory / 'model.safetensors') != identity['model_sha256']:
            raise ValueError('SAT weight file differs from its recorded SHA256')
        model = AutoModel.from_pretrained(directory, local_files_only=True, attn_implementation='sdpa')
    else:
        from transformers.modeling_utils import no_init_weights
        # A complete checkpoint already contains the adapted encoder; load its weights only once.
        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        # Checkpoint restoration overwrites every parameter; keep constructor-built RoPE buffers.
        with no_init_weights():
            model = AutoModel.from_config(config, attn_implementation='sdpa')
    return model, identity


def validate_checkpoint(state):
    if (state.get('architecture') != ARCHITECTURE or state.get('task') != TASK
            or state.get('classes') != list(CLASSES)):
        raise ValueError('Checkpoint must match the surface architecture and class order')


def load_model(checkpoint, device):
    from kappanav.global_planner.network import GlobePlanner
    with Path(checkpoint).open('rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
        validate_checkpoint(state)
        # Keep provenance tied to this inode if training atomically replaces best.pt meanwhile.
        stream.seek(0)
        checksum = stream_digest(stream)
    backbone, identity = load_backbone(BACKBONE, pretrained=False)
    if identity != state['backbone']:
        raise ValueError('Checkpoint backbone identity changed')
    model = GlobePlanner(backbone, WORK_GSD_M)
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False), {**state, 'loaded_checkpoint_sha256': checksum}
