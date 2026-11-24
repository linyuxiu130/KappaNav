"""Aerial training data in one store: every source as RGB and its authors' own class ids at a common ground scale.

    <store>/<source>/meta.json        sensor, provenance, licence, native ground scale, the authors' class list
    <store>/<source>/manifest.jsonl   one image per line: id, split, scene_group, gsd_m, native_gsd_m, altitude_m,
                                      width, height, rgb_sha256 (of the native RGB, for duplicate detection)
    <store>/<source>/images/<id>.jpg  RGB at gsd_m = max(native, MIN_GSD_M), area resampled
    <store>/<source>/labels/<id>.png  uint8 native class ids at the same scale, majority per output pixel (absent for
                                      unlabelled sources)
    <store>/<source>/preview.jpg      a few images | labels, for inspection

Labels are never remapped here: the class mapping belongs to the training code, so a taxonomy change needs no
conversion. A source is written to <source>.partial and renamed when complete.
"""
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
STORE = Path(os.environ.get('GLOBE_DATA', '/shibo_huang/data/globe'))
MIN_GSD_M = .05


class Skip(Exception):
    """Raised by Item.load for an image that cannot be used (the reason is recorded in meta.json)."""


@dataclass
class Item:
    """One image of a source. load() returns (rgb HxWx3 uint8, label HxW uint8 or None); it must be picklable (a module-level function or functools.partial)."""
    id: str
    split: str
    scene_group: str
    native_gsd_m: float
    load: Callable
    altitude_m: float = None
    extra: dict = None  # further manifest fields, e.g. gsd_estimated


def resample_rgb(rgb, size):
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BOX))


def mode_pool(label, size):
    """Majority class of the source pixels falling in each output pixel (size = (width, height), smaller)."""
    h, w = label.shape
    rows = (np.arange(h) * size[1] // h)[:, None]
    cols = (np.arange(w) * size[0] // w)[None, :]
    cell = (rows * size[0] + cols).ravel()
    values, codes = np.unique(label, return_inverse=True)
    counts = np.bincount(cell * len(values) + codes.ravel(), minlength=size[0] * size[1] * len(values))
    return values[counts.reshape(-1, len(values)).argmax(1)].reshape(size[1], size[0]).astype(np.uint8)


def convert(folder, item):
    try:
        rgb, label = item.load()
    except Skip as reason:
        return dict(id=item.id, skipped=str(reason))
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'{item.id}: RGB must be HxWx3 uint8, got {rgb.dtype} {rgb.shape}')
    if label is not None and label.shape != rgb.shape[:2]:
        raise ValueError(f'{item.id}: label {label.shape} vs image {rgb.shape[:2]}')
    sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    gsd = max(item.native_gsd_m, MIN_GSD_M)
    scale = item.native_gsd_m / gsd
    size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
    name = item.id.replace('/', '__')
    if size != (rgb.shape[1], rgb.shape[0]):
        rgb = resample_rgb(rgb, size)
        label = None if label is None else mode_pool(label, size)
    Image.fromarray(rgb).save(folder / 'images' / f'{name}.jpg', quality=95)
    if label is not None:
        Image.fromarray(label).save(folder / 'labels' / f'{name}.png')
    fractions = None
    if label is not None:
        values, counts = np.unique(label, return_counts=True)
        fractions = {int(v): round(float(c) / label.size, 5) for v, c in zip(values, counts)}
    row = dict(id=item.id, file=name, split=item.split, scene_group=item.scene_group, gsd_m=round(gsd, 5),
                native_gsd_m=round(item.native_gsd_m, 5), altitude_m=item.altitude_m, width=size[0], height=size[1],
                labelled=label is not None, class_fraction=fractions, rgb_sha256=sha)
    return dict(row, **(item.extra or {}))


def _convert(args):
    return convert(*args)


def _histogram(path):
    return np.bincount(np.asarray(Image.open(path)).ravel(), minlength=256)


def check_labels(folder, meta, rows, workers):
    """Fail a labelled source whose labels cannot be right: no pixel outside the ignore id, an id outside the class
    list, or a listed class that never occurs (unless meta['absent_classes'] lists its id). Returns the pixel
    fraction of every class that occurs (ignore id included)."""
    labelled = [r for r in rows if r['labelled']]
    if not labelled:
        return None
    with Pool(workers) as pool:
        hist = sum(pool.map(_histogram, [folder / 'labels' / f"{r['file']}.png" for r in labelled], chunksize=16))
    classes, ignore = meta['classes'], meta.get('ignore_id')
    used = [i for i in range(256) if hist[i]]
    outside = [i for i in used if i >= len(classes) and i != ignore]
    valid = hist.sum() - (hist[ignore] if ignore is not None else 0)
    absent = [i for i in range(len(classes)) if not hist[i] and i != ignore and i not in meta.get('absent_classes', [])]
    if outside or valid == 0 or absent:
        raise ValueError(f'label check failed: ids outside the class list {outside}, non-ignore pixels {valid}, '
                         f'classes never occurring {absent}; fractions {dict((i, round(hist[i] / hist.sum(), 4)) for i in used)}')
    return {(classes[i] if i < len(classes) else 'ignore'): round(float(hist[i] / hist.sum()), 5) for i in used}


def write_source(name, meta, items, workers=16):
    """Convert all items of a source into <store>/<name>, dropping exact RGB duplicates (the first id is kept)."""
    final, folder = STORE / name, STORE / f'{name}.partial'
    if folder.exists():
        shutil.rmtree(folder)
    for sub in ('images', 'labels'):
        (folder / sub).mkdir(parents=True)
    with Pool(workers) as pool:
        rows = pool.map(_convert, [(folder, item) for item in items], chunksize=1)
    skipped = [r for r in rows if 'skipped' in r]
    rows = sorted((r for r in rows if 'skipped' not in r), key=lambda r: r['id'])
    seen, kept, duplicates = set(), [], []
    for row in rows:
        if row['rgb_sha256'] in seen:
            duplicates.append(row['id'])
            for sub, ext in (('images', 'jpg'), ('labels', 'png')):
                (folder / sub / f"{row['file']}.{ext}").unlink(missing_ok=True)
            continue
        seen.add(row['rgb_sha256'])
        kept.append(row)
    if not any((folder / 'labels').iterdir()):
        (folder / 'labels').rmdir()
    fractions = check_labels(folder, meta, kept, workers)
    meta = dict(meta, source=name, class_pixel_fraction=fractions, images=len(kept), dropped_exact_duplicates=duplicates, skipped=skipped,
                splits={s: sum(r['split'] == s for r in kept) for s in sorted({r['split'] for r in kept})},
                scene_groups=len({r['scene_group'] for r in kept}), store_gsd_m=f'max(native, {MIN_GSD_M})')
    (folder / 'meta.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False) + '\n')
    with (folder / 'manifest.jsonl').open('w') as f:
        for row in kept:
            f.write(json.dumps(row) + '\n')
    preview(folder, kept, meta)
    if final.exists():
        shutil.rmtree(final)
    folder.rename(final)
    return meta


def palette(n=256):
    rng = np.random.default_rng(7)
    colours = rng.integers(40, 256, (n, 3), dtype=np.uint8)
    colours[0] = 0
    return colours


def preview(folder, rows, meta, count=4, height=300):
    """count images spread over the source: image | label coloured."""
    picks = [rows[i] for i in np.linspace(0, len(rows) - 1, min(count, len(rows))).astype(int)]
    colours, strips = palette(), []
    for row in picks:
        rgb = np.asarray(Image.open(folder / 'images' / f"{row['file']}.jpg"))
        panels = [rgb]
        if row['labelled']:
            label = np.asarray(Image.open(folder / 'labels' / f"{row['file']}.png"))
            panels.append((rgb * .35 + colours[label] * .65).astype(np.uint8))
        w = max(1, round(rgb.shape[1] * height / rgb.shape[0]))
        strips.append(np.hstack([np.asarray(Image.fromarray(p).resize((w, height), Image.NEAREST)) for p in panels]))
    width = max(s.shape[1] for s in strips)
    sheet = np.vstack([np.pad(s, ((0, 4), (0, width - s.shape[1]), (0, 0))) for s in strips])
    Image.fromarray(sheet).save(folder / 'preview.jpg', quality=85)
