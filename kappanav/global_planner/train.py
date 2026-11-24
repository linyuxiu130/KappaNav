"""Multi-source surface training: weighted sources, varied ground resolution and extent, set-valued supervision.

Model selection: the lowest mean class-balanced error of the 0-255 value over the held-out training scenes.
"""
import json
import math
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from kappanav.global_planner.dataset import (DRONE_SHARE, EXCLUDED, WORK_GSD_M, CropBatches, Source, TrainCrops,
                                             collate, normalize_uint8_batch)
from kappanav.global_planner.evaluate import evaluate, evaluation_sets
from kappanav.global_planner.loss import SurfaceCriterion
from kappanav.global_planner.network import GlobePlanner
from kappanav.global_planner.runtime import (ARCHITECTURE, BACKBONE, TASK, atomic_save, load_backbone, restore_rng,
                                             rng_state, source_identity, validate_checkpoint, write_json, write_jsonl)
from kappanav.global_planner.taxonomy import CLASSES

# What a resume must reproduce. `init` (a warm start names the weights it began from) and the source digest are
# provenance; `train_rows` follows the store, which is cleaned between runs, so neither decides identity.
RECIPE = ('sources', 'source_weights', 'steps', 'batch_size_per_rank', 'world_size', 'lr', 'backbone_lr', 'seed',
          'eval_every', 'eval_images', 'torch')


def same_recipe(saved, new):
    return all(saved.get(key) == new.get(key) for key in RECIPE)


def learning_rate(update, total, peak):
    warmup = min(1000, max(1, total // 20))
    if update < warmup:
        return peak * (update + 1) / warmup
    progress = (update - warmup) / max(1, total - warmup - 1)
    return peak * .5 * (1 + math.cos(math.pi * progress))


def source_weights(sources, overrides):
    """Crop share of each source: drone imagery gets DRONE_SHARE in total and the rest the remainder; within each
    group in proportion to sqrt(training rows). NAME=W overrides a share."""
    raw = [math.sqrt(len(s.split('train'))) for s in sources]
    drone = [s.meta['sensor'] == 'drone' for s in sources]
    weights = []
    for value, is_drone in zip(raw, drone):
        group = sum(r for r, d in zip(raw, drone) if d == is_drone)
        share = DRONE_SHARE if is_drone else 1 - DRONE_SHARE
        weights.append(share * value / group if group else 0.)
    for item in overrides:
        name, value = item.split('=')
        weights[[s.name for s in sources].index(name)] = float(value)
    return weights


def train(args, device, rank, world):
    if rank == 0:
        if args.resume:
            if not (args.output / 'last.pt').is_file():
                raise FileNotFoundError('Resume requires output/last.pt')
        else:
            args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    names = args.sources or sorted(p.name for p in args.store.iterdir()
                                   if (p / 'manifest.jsonl').is_file() and p.name not in EXCLUDED)
    sources = [s for s in (Source(name, args.store) for name in names) if s.labelled]  # supervised ones
    names = [s.name for s in sources]
    crops = TrainCrops(sources)
    weights = source_weights(sources, args.weight)
    torch.manual_seed(args.seed)
    backbone, backbone_identity = load_backbone(BACKBONE, pretrained=not args.resume)
    model = GlobePlanner(backbone, WORK_GSD_M).to(device)
    head = list(model.head.parameters())
    encoder = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': head, 'lr': args.lr, 'base_lr': args.lr},
                                   {'params': encoder, 'lr': args.backbone_lr, 'base_lr': args.backbone_lr}],
                                  weight_decay=.01)
    config = {'sources': names, 'source_weights': dict(zip(names, weights)),
              'train_rows': {s.name: len(s.split('train')) for s in sources}, 'steps': args.steps,
              'batch_size_per_rank': args.batch_size, 'world_size': world, 'lr': args.lr,
              'backbone_lr': args.backbone_lr, 'seed': args.seed, 'eval_every': args.eval_every,
              'eval_images': args.eval_images, 'source_sha256': source_identity(), 'torch': torch.__version__}
    step, best, history = 0, -1., []
    if args.resume:
        state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        validate_checkpoint(state)
        if not same_recipe(state['config'], config):
            raise ValueError('Resume requires the same sources, recipe and topology')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'][rank], device)
        step, best, history = state['step'], state['best'], state['history']
    else:
        if args.init is not None:   # warm start: weights of an earlier run, new schedule and sources
            init = torch.load(args.init, map_location='cpu', weights_only=False)
            validate_checkpoint(init)
            model.load_state_dict(init['model'], strict=True)
            config['init'] = str(args.init.resolve())
    if not args.resume and rank == 0:
        write_json(args.output / 'config.json', {**config, 'architecture': ARCHITECTURE, 'classes': CLASSES,
                   'backbone': backbone_identity, 'trainable_parameters': sum(p.numel() for p in head + encoder)})
    wrapped = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else model
    criterion = SurfaceCriterion()
    sets = evaluation_sets(args.store, names, args.eval_images, role='select')
    loader = DataLoader(crops, batch_sampler=CropBatches(crops, weights, args.batch_size, args.steps, args.seed, rank, step),
                        num_workers=args.workers, collate_fn=collate, pin_memory=device.type == 'cuda',
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    wrapped.train()
    totals, started = np.zeros(2), time.perf_counter()
    for images, labels, gsd in loader:
        images, gsd = images.to(device, non_blocking=True), gsd.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        factor = learning_rate(step, args.steps, 1.)
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr'] * factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = wrapped(normalize_uint8_batch(images), gsd)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head + encoder, 1., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        totals += [loss.item(), 1]
        if rank == 0 and step % 50 == 0:
            print(json.dumps({'step': step, 'surface': totals[0] / totals[1],
                              'lr': args.lr * factor, 'grad_norm': norm.item(), 'side': images.shape[-1],
                              'batch': len(images), 'seconds': time.perf_counter() - started}), flush=True)
        if step % args.eval_every and step != args.steps:
            continue
        record = {'step': step, 'surface': float(totals[0] / max(totals[1], 1)),
                  'validation': evaluate(model, sets, device, rank, world)}
        wrapped.train()
        totals = np.zeros(2)
        score = 255. - record['validation']['mean_value_error']
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state(device))
        else:
            states[0] = rng_state(device)
        if rank == 0:
            history.append(record)
            state = {'architecture': ARCHITECTURE, 'task': TASK, 'classes': list(CLASSES), 'backbone': backbone_identity,
                     'config': config, 'step': step, 'model': model.state_dict(), 'best': max(best, score),
                     'validation': record['validation'], 'history': history}
            if score > best:
                atomic_save(state, args.output / 'best.pt')
            atomic_save({**state, 'optimizer': optimizer.state_dict(), 'rng': states}, args.output / 'last.pt')
            write_jsonl(args.output / 'metrics.jsonl', history)
            write_json(args.output / 'status.json', {'state': 'complete' if step == args.steps else 'training',
                       'step': step, 'best_value_error': 255. - max(best, score)})
            print(json.dumps(record), flush=True)
        best = max(best, score)
        if world > 1:
            dist.barrier()
