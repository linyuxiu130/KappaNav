import json

import numpy as np
import pytest
from PIL import Image

from kappanav.global_planner.dataset import (GSD_RANGE_M, WORK_GSD_M, CropBatches, Source, TrainCrops, collate,
                                             evaluation_image, resample_ids)
from kappanav.global_planner.taxonomy import bits

CLASSES = ['Background', 'Person', 'Road', 'Vegetation', 'Car', 'Construction']


def make_source(root, name, classes, splits=('train', 'train', 'val'), size=(300, 200), gsd=.05,
                sensor='drone'):
    folder = root / name
    for sub in ('images', 'labels'):
        (folder / sub).mkdir(parents=True)
    (folder / 'meta.json').write_text(json.dumps({'classes': classes, 'sensor': sensor}))
    rows = []
    for i, split in enumerate(splits):
        rgb = np.random.default_rng(i).integers(0, 255, size[::-1] + (3,), dtype=np.uint8)
        ids = np.full(size[::-1], 2, np.uint8)
        ids[:, size[0] // 2:] = 5
        Image.fromarray(rgb).save(folder / 'images' / f'{i}.jpg')
        Image.fromarray(ids).save(folder / 'labels' / f'{i}.png')
        rows.append({'id': str(i), 'split': split, 'scene_group': str(i), 'gsd_m': gsd, 'width': size[0], 'height': size[1]})
    (folder / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return Source(name, root)


def test_majority_resampling_keeps_the_dominant_native_class():
    ids = np.zeros((10, 10), np.uint8)
    ids[:, 7:] = 3
    out = resample_ids(ids, (2, 2))
    assert (out[:, 0] == 0).all() and (out[:, 1] == 3).all()


def test_selection_rows_are_held_out_of_training_and_never_come_from_the_test_split(tmp_path):
    """The model is chosen on `select` rows, so they must not be trained on, and the reported test rows stay untouched."""
    source = make_source(tmp_path, 'aeroscapes', CLASSES, splits=('train',) * 60)
    ids = {role: {r['id'] for r in source.split(role)} for role in ('train', 'select', 'val')}
    group = {r['id']: r['scene_group'] for r in source.rows}
    assert ids['select'] and not ids['select'] & ids['train'] and not ids['select'] & ids['val']
    assert ids['train'] | ids['select'] | ids['val'] == set(group)
    assert not {group[i] for i in ids['select']} & {group[i] for i in ids['train']}
    assert Source('aeroscapes', tmp_path).select == source.select               # the same in every run


def test_test_ids_are_unchanged_by_the_selection_split(tmp_path):
    source = make_source(tmp_path, 'aeroscapes', CLASSES, splits=('train',) * 60)
    assert source.kept == source.test_ids()


def test_a_source_of_few_large_groups_still_holds_one_out(tmp_path):
    """Four places of 25 rows each: no group fits a 10% budget, so the smallest is held out rather than nothing."""
    source = make_source(tmp_path, 'aeroscapes', CLASSES, splits=('train',) * 100)
    for i, row in enumerate(source.rows):
        row['scene_group'] = str(i // 25)
    source.kept = source.test_ids()
    source.select = source.selection_ids()
    groups = {r['scene_group'] for r in source.rows if r['id'] not in source.kept}
    chosen = {r['scene_group'] for r in source.rows if r['id'] in source.select}
    assert len(chosen) == 1 and chosen <= groups                       # exactly one whole place, never a fragment
    assert not source.select & {r['id'] for r in source.split('train')}


def test_a_single_place_source_holds_out_nothing_for_selection(tmp_path):
    source = make_source(tmp_path, 'aeroscapes', CLASSES, splits=('train',) * 40)
    for row in source.rows:
        row['scene_group'] = 'one'
    source.kept, source.select = source.test_ids(), source.selection_ids()
    assert source.select == set() and source.split('train')


def test_training_never_sees_a_selection_row(tmp_path):
    source = make_source(tmp_path, 'aeroscapes', CLASSES, splits=('train',) * 60)
    selected = {r['id'] for r in source.split('select')}
    crops = TrainCrops([source])
    assert selected and not selected & {r['id'] for r in crops.rows[0]}
    ids = np.zeros((10, 10), np.uint8)
    ids[:, 7:] = 3
    out = resample_ids(ids, (2, 2))
    assert (out[:, 0] == 0).all() and (out[:, 1] == 3).all()


def test_test_rows_are_whole_scene_groups_and_stable(tmp_path):
    source = make_source(tmp_path, 'aeroscapes', CLASSES, splits=('train',) * 40)
    group = {r['id']: r['scene_group'] for r in source.rows}
    train, select, test = ({r['id'] for r in source.split(k)} for k in ('train', 'select', 'val'))
    assert test and not train & test and len(train | select | test) == 40 and source.scene_disjoint
    assert not {group[i] for i in train} & {group[i] for i in test}
    assert not {group[i] for i in select} & {group[i] for i in test}
    assert Source('aeroscapes', tmp_path).kept == source.kept


def test_a_single_place_source_still_keeps_test_rows_and_says_they_share_it(tmp_path):
    source = make_source(tmp_path, 'aeroscapes', CLASSES, splits=('train',) * 20)
    for r in source.rows:
        r['scene_group'] = 'one'
    source.kept = source.test_ids()
    assert len(source.split('val')) == 3 and not source.scene_disjoint


def test_crops_have_the_batch_side_bitmask_labels(tmp_path):
    crops = TrainCrops([make_source(tmp_path, 'aeroscapes', CLASSES)])
    batches = list(CropBatches(crops, [1.], 2, 6, seed=0))
    assert batches == list(CropBatches(crops, [1.], 2, 6, seed=0))          # deterministic
    for batch in batches:
        rgb, label, gsd = collate([crops[item] for item in batch])
        side = batch[0][2]
        assert rgb.shape[-2:] == (side, side) and label.shape[-2:] == (side, side)
        assert set(label.unique().tolist()) <= {0, bits(('road', 'dirt')), bits(('obstacle',))}
        assert ((gsd >= .08) & (gsd <= .25)).all()


def test_evaluation_image_is_at_the_working_resolution(tmp_path):
    source = make_source(tmp_path, 'aeroscapes', CLASSES)
    rgb, labels, gsd = evaluation_image(source, source.split('val')[0])
    assert gsd == .15 and rgb.shape[:2] == (67, 100) and labels.shape == (67, 100)


def test_a_source_only_offers_the_crop_sides_its_images_can_fill(tmp_path):
    """A 200-pixel short side at 0.05 m/px is 10 m of ground: no 128-pixel crop fits in it at 0.08 m/px."""
    source = make_source(tmp_path, 'aeroscapes', CLASSES, size=(300, 200), gsd=.05)
    assert TrainCrops([source]).sides == [{64}]


def test_a_crop_is_all_real_ground_and_reports_the_resolution_it_was_cut_at(tmp_path):
    source = make_source(tmp_path, 'aeroscapes', CLASSES, size=(300, 200), gsd=.05)
    crops = TrainCrops([source])
    for seed in range(8):
        _, _, gsd = crops[(0, 0, 64, seed)]
        assert .08 <= float(gsd) <= 200 * .05 / 64 + 1e-6
    with pytest.raises(ValueError, match='needs gsd'):
        crops[(0, 0, 512, 0)]


def test_batches_only_draw_sources_that_can_fill_the_batch_side(tmp_path):
    tiny = make_source(tmp_path, 'aeroscapes', CLASSES, size=(300, 200), gsd=.05)            # 64-pixel crops only
    large = make_source(tmp_path, 'potsdam', ['impervious surfaces'], size=(2048, 2048), gsd=.05)
    crops = TrainCrops([tiny, large])
    assert crops.sides == [{64}, {64, 128, 192, 256, 384, 512}]
    for batch in CropBatches(crops, [1., 1.], 2, 40, seed=0):
        side = batch[0][2]
        if side > 64:
            assert {item[0] for item in batch} == {1}
    assert {batch[0][2] for batch in CropBatches(TrainCrops([tiny]), [1.], 2, 20, seed=0)} == {64}


def test_drone_sources_get_half_of_the_crops_and_unlabelled_rows_none(tmp_path):
    pytest.importorskip('transformers')
    from kappanav.global_planner.train import source_weights
    drone = make_source(tmp_path, 'aeroscapes', CLASSES)
    dom = make_source(tmp_path, 'okutama_50m', ['background'] * 6, splits=('train',) * 30, sensor='dom')
    assert np.allclose(source_weights([drone, dom], []), [.5, .5])
    drone.rows[0]['labelled'] = False
    trained = {r['id'] for r in drone.split('train')} | {r['id'] for r in drone.split('select')}
    assert '0' not in trained                                          # an unlabelled row is never trained on


def test_working_resolution_keeps_the_ground_inside_the_trained_range():
    from kappanav.global_planner.inference import working_resolution
    image = np.zeros((200, 300, 3), np.uint8)
    assert working_resolution(image, .2)[1] == .2 and working_resolution(image, .2)[0] is image
    fine, gsd = working_resolution(image, .05)
    assert gsd == WORK_GSD_M and fine.shape[:2] == (67, 100)
    coarse, gsd = working_resolution(image, .5)
    assert gsd == GSD_RANGE_M[1] and coarse.shape[:2] == (400, 600)
