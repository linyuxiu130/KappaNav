"""Sources already in the curated raw archive (/shibo_huang/datasets/traversability_semantic): its catalog lists every
image/label pair with the authors' split; quarantined pairs are left out."""
import re
import sqlite3
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from kappanav.global_planner.prepare import Item

ARCHIVE = Path('/shibo_huang/datasets/traversability_semantic')


def catalog(source):
    with sqlite3.connect(ARCHIVE / 'catalog.sqlite') as db:
        return db.execute("select id, image, label, source_split, scene_group from pairs where source = ? and "
                          "quality_status not like 'quarantine%' order by id", (source,)).fetchall()


def rgb_of(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def ids_of(path, channel=None):
    with Image.open(path) as image:
        a = np.asarray(image)  # palette PNGs keep their indices
    return a if a.ndim == 2 else a[..., channel or 0]


def load_ids(image, label):
    return rgb_of(image), ids_of(label)


# ---------------------------------------------------------------- MESSI
MESSI_CLASSES = ['background', 'bicycle', 'building', 'fence', 'other objects', 'person', 'pole', 'rough terrain', 'shed',
                 'soft terrain', 'stairs', 'transportation terrain', 'vegetation', 'vehicle', 'walking terrain', 'water']
MAVIC2PRO_M_PER_PX_PER_M = 13.2e-3 / (10.26e-3 * 5472)  # L1D-20c: 1" sensor width / focal length / image width
NADIR_TOLERANCE_DEG = 10.


def dji_xmp(path):
    with open(path, 'rb') as f:
        head = f.read(1 << 17).decode('latin-1')
    def value(key):
        m = re.search(rf'{key}="([-+0-9.]+)"', head)
        return float(m.group(1)) if m else None
    return value('RelativeAltitude'), value('GimbalPitchDegree')


def messi():
    items, skipped = [], []
    for sid, image, label, split, _ in catalog('messi'):
        altitude, pitch = dji_xmp(image)
        if altitude is None or pitch is None or abs(pitch + 90) > NADIR_TOLERANCE_DEG:
            skipped.append(dict(id=sid, altitude_m=altitude, gimbal_pitch_deg=pitch))
            continue
        city = 'IrYamim' if 'IrYamim' in image else 'Agamim'
        items.append(Item(sid, 'trainval' if split == 'author_train_val_unsplit' else split, f'MESSI/{city}',
                          altitude * MAVIC2PRO_M_PER_PX_PER_M, partial(load_ids, image, label), round(altitude, 1)))
    meta = dict(sensor='drone', url='https://github.com/messi-dataset/messi-dataset', license='see authors (research)',
                classes=MESSI_CLASSES, ignore_id=0,
                native_gsd='per image from DJI XMP RelativeAltitude x 13.2 mm / (10.26 mm x 5472 px) (Mavic 2 Pro)',
                altitude='per image, 5-120 m (RelativeAltitude above take-off)',
                notes=['Agamim (train/val) and Ir Yamim (test) neighbourhoods; Path and Descend flights over Agamim '
                       'overlap, so scene_group is the neighbourhood',
                       '"transportation terrain" = roads, "walking terrain" = sidewalks/paths',
                       f'kept only nadir frames (|gimbal pitch + 90| <= {NADIR_TOLERANCE_DEG} deg); skipped: {len(skipped)}'],
                skipped_non_nadir=skipped)
    return meta, items


# ---------------------------------------------------------------- Semantic Drone (TU Graz)
SEMANTIC_DRONE_CLASSES = ['unlabeled', 'paved-area', 'dirt', 'grass', 'gravel', 'water', 'rocks', 'pool', 'vegetation',
                          'roof', 'wall', 'window', 'door', 'fence', 'fence-pole', 'person', 'dog', 'car', 'bicycle',
                          'tree', 'bald-tree', 'ar-marker', 'obstacle', 'conflicting']
SEMANTIC_DRONE_RGB = [(0, 0, 0), (128, 64, 128), (130, 76, 0), (0, 102, 0), (112, 103, 87), (28, 42, 168), (48, 41, 30),
                      (0, 50, 89), (107, 142, 35), (70, 70, 70), (102, 102, 156), (254, 228, 12), (254, 148, 12),
                      (190, 153, 153), (153, 153, 153), (255, 22, 96), (102, 51, 0), (9, 143, 150), (119, 11, 32),
                      (51, 51, 0), (190, 250, 190), (112, 150, 146), (2, 135, 115), (255, 0, 0)]
SEMANTIC_DRONE_GSD_M = (5 * 23.5e-3 / 16e-3 / 6000, 30 * 23.5e-3 / 16e-3 / 6000)  # Sony a6000 APS-C, 16 mm, 6000 px


def load_semantic_drone(image, label):
    rgb = rgb_of(label).astype(np.int32)
    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    lut = {(r << 16) | (g << 8) | b: i for i, (r, g, b) in enumerate(SEMANTIC_DRONE_RGB)}
    ids = np.zeros(key.shape, np.uint8)
    for k, i in lut.items():
        ids[key == k] = i
    return rgb_of(image), ids


def semantic_drone():
    gsd = float(np.sqrt(np.prod(SEMANTIC_DRONE_GSD_M)))
    items = [Item(sid, split, 'SemanticDrone/unknown', gsd, partial(load_semantic_drone, image, label))
             for sid, image, label, split, _ in catalog('semantic_drone')]
    meta = dict(sensor='drone', url='https://www.tugraz.at/index.php?id=22387', license='research only (TU Graz ICG)',
                classes=SEMANTIC_DRONE_CLASSES, ignore_id=0,
                native_gsd=f'unknown per image: 5-30 m altitude -> {SEMANTIC_DRONE_GSD_M[0]:.4f}-{SEMANTIC_DRONE_GSD_M[1]:.4f} m; '
                           f'native_gsd_m is their geometric mean ({gsd:.4f}): stored scale uncertain by up to x2.4',
                altitude='5-30 m (authors), nadir', notes=[
                    'labels decoded from the authors\' 24-colour table; id 23 "conflicting" should be ignored like 0',
                    'all public images are the authors\' training set; capture groups unknown (one scene_group)'])
    return meta, items


# ---------------------------------------------------------------- RescueNet


# ---------------------------------------------------------------- Okutama (50 m subset)
OKUTAMA_CLASSES = ['background', 'outdoor structures', 'buildings', 'paved ground', 'non-paved ground', 'train tracks',
                   'plants', 'wheeled vehicles', 'water', 'people']
OKUTAMA_GSD_M = 50 * 6.17e-3 / 3.61e-3 / 3840  # Phantom 4: 1/2.3" 6.17 mm width, 3.61 mm focal, 3840 px


def okutama_50m():
    items = [Item(sid, split, scene, OKUTAMA_GSD_M, partial(load_ids, image, label), 50.)
             for sid, image, label, split, scene in catalog('okutama_50m')]
    meta = dict(sensor='drone', url='https://github.com/agoncalves/okutama-swiss-dataset (Speth et al. 2022)',
                license='see authors', classes=OKUTAMA_CLASSES, ignore_id=0,
                native_gsd=f'{OKUTAMA_GSD_M:.4f} m (50 m, Phantom 4 4K)', altitude='50 m, nadir',
                notes=['labels are grey PNGs (channel 0 = id); polygon contours, not fine boundaries',
                       '"plants" mixes grass, bushes and trees'])
    return meta, items


# ---------------------------------------------------------------- MAVSD (90 degree frames only)
MAVSD_CLASSES = ['background', 'solidago canadensis', 'person', 'road', 'pine', 'bush', 'reed', 'stone', 'cypress',
                 'water', 'building', 'vegetable garden', 'electric bicycle']
MAVSD_GSD_M = (15 * 6.17e-3 / 4.5e-3 / 1920, 40 * 6.17e-3 / 4.5e-3 / 1920)


def mavsd():
    gsd = float(np.sqrt(np.prod(MAVSD_GSD_M)))
    items = [Item(sid, split, scene, gsd, partial(load_ids, image, label))
             for sid, image, label, split, scene in catalog('mavsd') if '/90degree/' in image]
    meta = dict(sensor='drone', url='https://doi.org/10.6084/m9.figshare.28428707.v1', license='CC BY 4.0 (figshare)',
                classes=MAVSD_CLASSES, ignore_id=0,
                native_gsd=f'not published: 15-40 m with a DJI M300 1920x1080 camera; native_gsd_m = {gsd:.4f} assumes '
                           f'the H20T wide camera (1/2.3", 4.5 mm), checked against object sizes in the preview',
                altitude='15-40 m (authors); only the 90 degree (nadir) folder is kept, 30/45/60 degree are oblique',
                notes=['id 0 = unannotated (about 75% of pixels): ignore',
                       'class order from the paper list (ids verified on masks: 3 road, 11 vegetable garden)',
                       'author train/test share 21 capture prefixes: not geographically independent'])
    return meta, items


# ---------------------------------------------------------------- MAVSD oblique (30, 45 and 60 degree folders)
MAVSD_RAD_PER_PX = 6.17e-3 / 4.5e-3 / 1920  # H20T wide camera: 1/2.3" sensor width / focal length / image width
MAVSD_ALTITUDE_M = (15 + 40) / 2  # centre of the authors' 15-40 m: the altitude of a single frame is unknown


def mavsd_oblique():
    items = []
    for sid, image, label, split, scene in catalog('mavsd'):
        pitch = {'30degree': 30, '45degree': 45, '60degree': 60}.get(Path(image).parent.parent.name)
        if pitch is None:
            continue
        centre_m = MAVSD_ALTITUDE_M / np.sin(np.deg2rad(pitch))  # distance along the optical axis to flat ground
        items.append(Item(sid, split, scene, float(centre_m * MAVSD_RAD_PER_PX), partial(load_ids, image, label), None,
                          dict(view='oblique', pitch_deg=pitch, gsd_estimated=True, altitude_range_m=[15, 40])))
    meta = dict(sensor='drone', url='https://doi.org/10.6084/m9.figshare.28428707.v1', license='CC BY 4.0 (figshare)',
                classes=MAVSD_CLASSES, ignore_id=0, absent_classes=[10],  # building: no frame of the oblique folders has one
                native_gsd=f'approximate, at the image centre, horizontal direction: (altitude / sin(pitch)) x {MAVSD_RAD_PER_PX:.3e} rad/px '
                           f'with the altitude taken as {MAVSD_ALTITUDE_M} m (centre of the authors\' 15-40 m: x0.6 to x1.6) and the H20T '
                           'wide camera (1/2.3", 4.5 mm, 1920 px); no object-size check is possible (forest and scrub scenes, hardly any cars)',
                altitude='15-40 m (authors); per image unknown',
                pitch_deg='folder name = camera angle of the authors (30, 45 or 60; 90 = straight down); assumed measured from the horizontal',
                notes=['the oblique folders of the MAVSD archive; the 90 degree folder is source mavsd',
                       'id 0 = unannotated: ignore; class order as source mavsd',
                       'gsd_m varies strongly over an oblique image (far pixels are coarser); only the centre value is stored',
                       'scene_group = capture prefix (date and time of the flight), as source mavsd'])
    return meta, items


SOURCES = dict(messi=messi, semantic_drone=semantic_drone, okutama_50m=okutama_50m, mavsd=mavsd, mavsd_oblique=mavsd_oblique)
