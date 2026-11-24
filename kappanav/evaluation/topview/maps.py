"""Traversability maps of the simulated worlds for global_planner.route: the global model's 0-255 value on the top
view of every Wanderland scene or of NavVerse cities, scored against the robot's C-space.

    python -m kappanav.evaluation.topview.maps wanderland --checkpoint MODEL --output DIR
    python -m kappanav.evaluation.topview.maps navverse --checkpoint MODEL --output DIR --scene vcp_amsterdam

DIR/<scene_id>.npz: value ([x, y], NaN = not observed), origin_xy, cell_m, unobserved_value. Both worlds end where
their data ends, so an unobserved cell is as bad as a predicted obstacle.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from kappanav.global_planner.dataset import WORK_GSD_M
from kappanav.global_planner.inference import predict
from kappanav.global_planner.metrics import auc
from kappanav.global_planner.runtime import load_model
from kappanav.global_planner.taxonomy import VALUE_FLOOR, traversability

SCENES = Path(os.environ.get('KAPPANAV_WANDERLAND_SCENES', '/shibo_huang/data/wanderland/curvenav'))
AERIAL = Path(os.environ.get('KAPPANAV_AERIAL', '/shibo_huang/data/topview'))
SAMPLES = 30_000     # scored cells per scene and area
UNKNOWN = 255        # a cell outside the robot's navigation grid


def value_map(model, rgb, cell_m, device):
    """The 0-255 value of an RGB top view, at the top view's own cells."""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (round(w * cell_m / WORK_GSD_M), round(h * cell_m / WORK_GSD_M)), interpolation=cv2.INTER_AREA)
    value = traversability(predict(model, small, WORK_GSD_M, device))
    return np.maximum(cv2.resize(value.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR), VALUE_FLOOR)


def save(output, scene_id, value, seen, origin_xy, cell_m):
    value = np.where(seen, value, np.nan).astype(np.float32)
    np.savez_compressed(output / f'{scene_id}.npz', value=value, origin_xy=origin_xy, cell_m=cell_m,
                        unobserved_value=VALUE_FLOOR)


def robot_label(maps, navigation):
    """Robot C-space on the map grid: 1 free, 0 blocked, UNKNOWN outside the navigation grid."""
    shape, cell, origin = maps['dom_rgb'].shape[:2], float(maps['map_cell_m']), maps['map_origin_xy']
    free, nav_cell, nav_origin = navigation['free'], float(navigation['cell_size_m']), navigation['origin_xy']
    idx = [np.floor((origin[a] + (np.arange(shape[a]) + .5) * cell - nav_origin[a]) / nav_cell).astype(int)
           for a in range(2)]
    ok = [(i >= 0) & (i < free.shape[a]) for a, i in enumerate(idx)]
    label = np.full(shape, UNKNOWN, np.uint8)
    label[np.ix_(*ok)] = free[np.ix_(idx[0][ok[0]], idx[1][ok[1]])]
    return label


def sample(value, label, mask, rng):
    i = np.flatnonzero(mask)
    i = rng.choice(i, min(len(i), SAMPLES), replace=False)
    return value.ravel()[i], label.ravel()[i] == 1


def summary(v, y):
    return dict(cells=len(y), free=float(y.mean()), auc=auc(v, y), value_free=float(v[y].mean()) if y.any() else None,
                value_blocked=float(v[~y].mean()) if (~y).any() else None)


def wanderland(args, model, device):
    """By scene domain and canopy: a drone sees no floor indoors or under a roof, so those are reported apart."""
    robot = os.environ.get('KAPPANAV_ROBOT', 'scout_mini')
    rng, groups = np.random.default_rng(0), {}
    for scene in json.loads((SCENES / f'tasks_{robot}.json').read_text())['scenes']:
        with np.load(SCENES / 'scenes' / scene['scene_id'] / 'maps.npz') as m, np.load(scene['navigation']) as n:
            maps, label = dict(m), robot_label(m, n)
        cell, seen = float(maps['map_cell_m']), maps['dom_valid']
        value = value_map(model, maps['dom_rgb'], cell, device)
        save(args.output, scene['scene_id'], value, seen, maps['map_origin_xy'], cell)
        scored, domain = seen & (label != UNKNOWN), scene['domain'].removesuffix('_guess')
        for area, mask in (('open', ~maps['covered']), ('covered', maps['covered'])):
            for name in (f'{domain}_{area}', f'{domain}_all'):
                groups.setdefault(name, []).append(sample(value, label, scored & mask, rng))
    return {name: summary(*map(np.concatenate, zip(*parts))) for name, parts in sorted(groups.items())}


def city_mosaic(scene_id):
    """A city's navverse_aerial tiles joined: rgb, the robot's C-space, origin, cell, and where a tile was rendered."""
    tiles = sorted((AERIAL / 'navverse' / scene_id).glob('*.npz'))
    if not tiles:
        raise FileNotFoundError(f'no aerial tiles for {scene_id}: run evaluation.topview.navverse_aerial')
    entries = [dict(np.load(t)) for t in tiles]
    side, cell = entries[0]['rgb'].shape[0], float(entries[0]['gsd_m'])
    origins = np.array([e['origin_xy'] for e in entries])
    low = origins.min(0)
    index = np.rint((origins - low) / (side * cell)).astype(int)
    shape = tuple((index.max(0) + 1) * side)
    rgb, label, seen = np.zeros(shape + (3,), np.uint8), np.full(shape, UNKNOWN, np.uint8), np.zeros(shape, bool)
    for (i, j), e in zip(index, entries):
        window = np.s_[i * side:(i + 1) * side, j * side:(j + 1) * side]
        rgb[window], label[window], seen[window] = e['rgb'], e['traversable'], True
    return rgb, label, low, cell, seen


def navverse(args, model, device):
    rng, report = np.random.default_rng(0), {}
    for scene_id in args.scene:
        rgb, label, origin, cell, seen = city_mosaic(scene_id)
        step = round(WORK_GSD_M / cell)                       # the city map is kept at the model's own resolution
        X, Y = (n // step * step for n in seen.shape)
        small = cv2.resize(rgb[:X, :Y], (Y // step, X // step), interpolation=cv2.INTER_AREA)
        label, seen = label[step // 2:X:step, step // 2:Y:step], seen[step // 2:X:step, step // 2:Y:step]
        value = value_map(model, small, cell * step, device)
        save(args.output, scene_id, value, seen, origin, cell * step)
        report[scene_id] = summary(*sample(value, label, seen & (label != UNKNOWN), rng))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('world', choices=('wanderland', 'navverse'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', nargs='+', default=['vcp_amsterdam'], help='NavVerse cities')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(args.checkpoint, device)
    report = dict(wanderland=wanderland, navverse=navverse)[args.world](args, model, device)
    print(json.dumps(report, indent=1))
    (args.output / 'report.json').write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
