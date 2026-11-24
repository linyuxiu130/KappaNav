"""The multi-source ground-surface store (/shibo_huang/data/globe, written by kappanav.global_planner.prepare) and its crops.

A source is <root>/<source>/{meta.json, manifest.jsonl, images/<file>.jpg, labels/<file>.png (native ids)}; a
manifest row names its file and whether it is labelled.
Native ids become CLASSES bitmasks through taxonomy.label_table, so the store never changes with the taxonomy.

Training crops imitate the input the model meets: a nadir image at a ground resolution drawn log-uniformly from
GSD_RANGE_M (never finer than the source), and an extent from SIDES (a drone at 5 m sees ~9 m, at 70 m ~125 m). Labels
are resampled by majority over native classes, never interpolated. Sources are drawn by weight, then images uniformly.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from kappanav.global_planner.taxonomy import label_table

STORE = Path(os.environ.get('GLOBE_DATA', '/shibo_huang/data/globe'))   # the same variable the writer honours
GSD_RANGE_M = (.08, .25)
WORK_GSD_M = .15             # evaluation and default inference resolution
SIDES = (64, 128, 192, 256, 384, 512)
SIDE_WEIGHTS = (1, 2, 2, 3, 3, 4)
SAT_MEAN, SAT_STD = (.430, .411, .296), (.213, .156, .143)
EXCLUDED = ('rescuenet', 'floodnet', 'global_south')   # in the store, not used: disaster areas, unlabelled imagery
DRONE_SHARE = .5                       # of the training crops; the rest is DOM and satellite
MIN_GROUPS = 8
TEST_SHARE = .15                       # of a source whose author split does not separate its scenes
SELECT_SHARE = .1                      # of the training rows, held out so the model is never chosen on the test rows
VALIDATION_SPLITS = ('val', 'validation', 'test', 'test2')


def normalize_uint8_batch(rgb):
    if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError('Expected uint8 RGB [B,3,H,W]')
    x = rgb.float() / 255
    return (x - x.new_tensor(SAT_MEAN)[None, :, None, None]) / x.new_tensor(SAT_STD)[None, :, None, None]


class Source:
    """One source of the store: rows and label table."""
    def __init__(self, name, store=STORE):
        self.name, self.folder = name, Path(store) / name
        self.meta = json.loads((self.folder / 'meta.json').read_text())
        self.rows = [json.loads(line) for line in (self.folder / 'manifest.jsonl').read_text().splitlines()]
        self.labelled = any(r.get('labelled', True) for r in self.rows)
        self.lut = label_table(name, self.meta['classes']) if self.labelled else None
        self.kept = self.test_ids()
        self.select = self.selection_ids()

    def test_ids(self):
        """Ids of the labelled rows kept for testing, the same in every run. The author's val/test rows, when no scene
        group of theirs is also in training. Otherwise whole scene groups up to TEST_SHARE of the rows, in hash order;
        a source with fewer than MIN_GROUPS groups keeps the author's split, or rows in hash order if it has none
        (then training and test share a place: scene_disjoint is False)."""
        rows = [r for r in self.rows if r.get('labelled', True)]
        order = lambda key: hashlib.sha256(f'{self.name}/{key}'.encode()).hexdigest()
        author = [r for r in rows if r['split'] in VALIDATION_SPLITS]
        groups = {r['scene_group'] for r in rows}
        shared = {r['scene_group'] for r in author} & {r['scene_group'] for r in rows if r['split'] not in VALIDATION_SPLITS}
        self.scene_disjoint = self.meta.get('scene_groups_verified', True)
        if author and not shared:
            return {r['id'] for r in author}
        kept = set()
        if len(groups) >= MIN_GROUPS:
            for group in sorted(groups, key=order):
                members = {r['id'] for r in rows if r['scene_group'] == group}
                if len(kept) + len(members) <= TEST_SHARE * len(rows) * 1.5 and len(kept) < TEST_SHARE * len(rows):
                    kept |= members
            return kept
        self.scene_disjoint = False
        if author:
            return {r['id'] for r in author}
        ids = sorted((r['id'] for r in rows), key=order)
        return set(ids[:round(TEST_SHARE * len(ids))])

    def selection_ids(self):
        """A fixed part of the training rows, held out to choose the model. Whole scene groups in hash order, so a
        place stays on one side; `test_ids` (the reported split) is untouched by this.

        A source of few, large groups would otherwise contribute nothing: when no group fits the budget, the smallest
        one is held out anyway. A source with a single group cannot be split by place and contributes no rows (its
        test split is already reported with scene_disjoint=False)."""
        rows = [r for r in self.rows if r.get('labelled', True) and r['id'] not in self.kept]
        groups = {}
        for row in rows:
            groups.setdefault(row['scene_group'], set()).add(row['id'])
        order = lambda key: hashlib.sha256(f'{self.name}/select/{key}'.encode()).hexdigest()
        target, chosen = SELECT_SHARE * len(rows), set()
        for group in sorted(groups, key=order):
            if len(chosen) >= target:
                break
            if len(groups[group]) <= target:
                chosen |= groups[group]
        if not chosen and len(groups) > 1:
            chosen = set(groups[min(groups, key=lambda group: (len(groups[group]), order(group)))])
        return chosen

    def split(self, role):
        """The labelled rows of 'train', 'select' (chooses the model) or 'val' (the fixed test ids, for reporting)."""
        rows = [r for r in self.rows if r.get('labelled', True)]
        if role == 'val':
            return [r for r in rows if r['id'] in self.kept]
        if role == 'select':
            return [r for r in rows if r['id'] in self.select]
        return [r for r in rows if r['id'] not in self.kept and r['id'] not in self.select]

    def read(self, row):
        """RGB uint8 HxWx3, native ids uint8 HxW (None if unlabelled)."""
        file = row.get('file', row['id'])
        rgb = np.asarray(Image.open(self.folder / 'images' / f'{file}.jpg').convert('RGB'))
        ids = np.asarray(Image.open(self.folder / 'labels' / f'{file}.png')) if row.get('labelled', True) else None
        return rgb, ids


def resample_ids(ids, size):
    """Native ids resampled to size (w, h) by majority: the area fraction of every class present, then argmax."""
    present = np.unique(ids)
    if len(present) == 1:
        return np.full(size[::-1], present[0], np.uint8)
    fractions = np.stack([cv2.resize((ids == c).astype(np.float32), size, interpolation=cv2.INTER_AREA) for c in present])
    return present[fractions.argmax(0)].astype(np.uint8)


def resample(rgb, ids, size):
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    ids = None if ids is None else resample_ids(ids, size)
    return rgb, ids


def photometric(rgb, rng):
    """Appearance changes across cameras, seasons, haze and compression; geometry is untouched."""
    x = rgb.astype(np.float32) / 255
    x = x * rng.uniform(.75, 1.25) + rng.uniform(-.08, .08)                     # exposure
    mean = x.mean((0, 1), keepdims=True)
    x = mean + (x - mean) * rng.uniform(.7, 1.3)                                # contrast
    grey = x.mean(2, keepdims=True)
    x = grey + (x - grey) * rng.uniform(.6, 1.4)                                # saturation
    x = x * np.array([rng.uniform(.9, 1.1) for _ in range(3)], np.float32)      # white balance / season tint
    if rng.random() < .2:
        x = x * (1 - (a := rng.uniform(.05, .3))) + a * rng.uniform(.7, .95)    # haze
    x = np.clip(x, 0, 1) ** rng.uniform(.8, 1.25)                               # gamma
    if rng.random() < .3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(.4, 1.2))
    if rng.random() < .3:
        x = x + rng.normal(0, rng.uniform(.005, .02), x.shape).astype(np.float32)
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if rng.random() < .3:
        ok, buffer = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        out = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    return out


def fills(row, side):
    """Whether a row's image can supply `side` pixels without padding: it covers min(height, width) * gsd_m metres of
    ground, and a `side`-pixel crop at the finest working resolution needs side * GSD_RANGE_M[0] of them."""
    return min(row['height'], row['width']) * float(row['gsd_m']) >= side * GSD_RANGE_M[0]


class TrainCrops(torch.utils.data.Dataset):
    """Item (source index, row index, side, seed) -> (rgb 3xSxS uint8, label bitmask SxS int32, gsd).
    Bitmask 0 = no supervision. Only rows that fill the requested side are ever drawn (see `eligible`), so every
    pixel of a crop is real ground."""
    def __init__(self, sources):
        self.sources = sources
        self.rows = [s.split('train') for s in sources]
        # side -> source index -> the rows whose image covers it; a source is absent when no row does.
        self.eligible = {side: {i: [j for j, row in enumerate(rows) if fills(row, side)]
                                for i, rows in enumerate(self.rows)} for side in SIDES}
        self.eligible = {side: {i: rows for i, rows in per_source.items() if rows}
                         for side, per_source in self.eligible.items()}
        self.sides = [{side for side, per_source in self.eligible.items() if i in per_source}
                      for i in range(len(sources))]

    def __getitem__(self, item):
        s, r, side, seed = item
        source, row = self.sources[s], self.rows[s][r]
        rng = np.random.default_rng(seed)
        store = float(row['gsd_m'])
        short = min(int(row['height']), int(row['width']))
        finest, coarsest = max(GSD_RANGE_M[0], store), min(GSD_RANGE_M[1], short * store / side)
        if coarsest < finest:
            raise ValueError(f'{source.name} {row["id"]}: a {side}-pixel crop needs gsd <= {short * store / side:.4f}')
        gsd = math.exp(rng.uniform(math.log(finest), math.log(coarsest)))
        extent = min(short, max(1, math.ceil(side * gsd / store)))     # crop side in store pixels, never finer than gsd
        rgb, ids = source.read(row)
        h, w = rgb.shape[:2]
        y, x = int(rng.integers(0, h - extent + 1)), int(rng.integers(0, w - extent + 1))
        window = np.s_[y:y + extent, x:x + extent]
        rgb, ids = resample(rgb[window], ids[window], (side, side))
        label = source.lut[ids].astype(np.int32)
        k, flip = int(rng.integers(4)), bool(rng.integers(2))
        arrays = [np.rot90(a, k) for a in (rgb, label)]
        if flip:
            arrays = [a[:, ::-1] for a in arrays]
        rgb, label = arrays
        rgb = photometric(np.ascontiguousarray(rgb), rng)
        return (torch.from_numpy(rgb).permute(2, 0, 1).contiguous(), torch.from_numpy(np.ascontiguousarray(label)),
                # The resolution of the crop as it was actually cut, which rounding can move by a pixel.
                torch.tensor(extent * store / side, dtype=torch.float32))


class CropBatches(torch.utils.data.Sampler):
    """Batches of TrainCrops items for one rank: each batch has one side (SIDES by SIDE_WEIGHTS) and only draws
    sources whose images can fill that side; within a batch a source is drawn by `weights`, then a row uniformly.
    The side depends on the step only, so every rank runs the same shapes and none waits for a slower one; the crops
    are deterministic in (seed, rank, batch index)."""
    def __init__(self, crops, weights, batch_size, steps, seed, rank=0, start=0):
        self.counts = [len(r) for r in crops.rows]
        self.weights = np.array([w if c else 0. for w, c in zip(weights, self.counts)], np.float64)
        if not self.weights.sum():
            raise ValueError('No training rows in the weighted sources')
        self.weights /= self.weights.sum()
        self.eligible = crops.eligible
        self.choices = np.array([i for i, side in enumerate(SIDES) if crops.eligible[side]])
        if not len(self.choices):
            raise ValueError('No source can fill any crop side')
        weights_by_side = np.array([SIDE_WEIGHTS[i] for i in self.choices], np.float64)
        self.side_weights = weights_by_side / weights_by_side.sum()
        self.batch_size, self.steps, self.seed, self.rank, self.start = batch_size, steps, seed, rank, start

    def __len__(self):
        return self.steps - self.start

    def __iter__(self):
        for step in range(self.start, self.steps):
            side = SIDES[self.choices[np.random.default_rng((self.seed, step)).choice(len(self.choices), p=self.side_weights)]]
            rng = np.random.default_rng((self.seed, self.rank, step))
            weights = np.array([w if i in self.eligible[side] else 0. for i, w in enumerate(self.weights)])
            weights = weights / weights.sum()
            batch = min(8 * self.batch_size, max(1, round(self.batch_size * (512 / side) ** 2)))  # same pixels per batch
            sources = rng.choice(len(weights), batch, p=weights)
            yield [(int(s), int(rng.choice(self.eligible[side][int(s)])), int(side), int(rng.integers(2**31)))
                   for s in sources]


def collate(items):
    rgb, label, gsd = zip(*items)
    return torch.stack(rgb), torch.stack(label), torch.stack(gsd)


def evaluation_image(source, row):
    """A whole validation image at WORK_GSD_M (or its own, if coarser): rgb, bitmask labels, gsd."""
    rgb, ids = source.read(row)
    store = float(row['gsd_m'])
    gsd = max(WORK_GSD_M, store)
    size = (max(16, round(rgb.shape[1] * store / gsd)), max(16, round(rgb.shape[0] * store / gsd)))
    rgb, ids = resample(rgb, ids, size)
    return rgb, source.lut[ids].astype(np.int32), gsd
