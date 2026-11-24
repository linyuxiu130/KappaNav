"""Sources downloaded from Hugging Face (through hf-mirror) into DOWNLOADS, converted from the authors' archives."""
import io
import json
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image

from kappanav.global_planner.prepare import STORE, Item, Skip

DOWNLOADS = Path('/shibo_huang/data/globe_downloads')
Image.MAX_IMAGE_PIXELS = None


def from_zip(archive, name):
    with zipfile.ZipFile(archive) as z:
        return z.read(name)


# ---------------------------------------------------------------- FLAIR #1 (IGN aerial orthophotos, 0.2 m)
FLAIR_CLASSES = ['nodata', 'building', 'pervious surface', 'impervious surface', 'bare soil', 'water', 'coniferous',
                 'deciduous', 'brushwood', 'vineyard', 'herbaceous vegetation', 'agricultural land', 'plowed land',
                 'swimming pool', 'snow', 'clear cut', 'mixed', 'ligneous', 'greenhouse', 'other']


def load_flair(archive, image, label):
    import rasterio
    try:
        with rasterio.open(io.BytesIO(from_zip(archive, image))) as src:
            rgb = np.ascontiguousarray(src.read((1, 2, 3)).transpose(1, 2, 0))
        with rasterio.open(io.BytesIO(from_zip(archive, label))) as src:
            ids = src.read(1)
    except Exception as error:
        raise Skip(f'unreadable: {type(error).__name__}')
    if rgb.dtype != np.uint8 or ids.shape != rgb.shape[:2]:
        raise Skip(f'unusable: image {rgb.dtype} {rgb.shape}, mask {ids.shape}')
    if not ids.any():
        raise Skip('mask is all nodata (0)')
    if not rgb.any():
        raise Skip('image is all zero')
    return rgb, ids.astype(np.uint8)


def flair():
    metadata = json.loads((DOWNLOADS / 'flair/aux-data/flair_aerial_metadata.json').read_text())
    items = []
    for split, folder in (('trainval', 'train-val'), ('test', 'flair#1-test'), ('test2', 'flair#2-test')):
        for archive in sorted((DOWNLOADS / 'flair/data' / folder).glob('*.zip')):
            with zipfile.ZipFile(archive) as z:
                names = [n for n in z.namelist() if n.startswith('aerial/') and n.endswith('.tif')]
            domain = archive.stem
            for name in names:
                zone, file = name.split('/')[1:]
                patch = file[len('IMG_'):-len('.tif')]
                meta = metadata.get(f'IMG_{patch}', {})
                items.append(Item(f'{domain}/{zone}/{patch}', split, f'{domain}/{zone}', .2,
                                  partial(load_flair, archive, name, f'labels/{zone}/MSK_{patch}.tif'),
                                  None))
    meta = dict(sensor='dom', url='https://huggingface.co/datasets/IGNF/FLAIR-1-2 (FLAIR #1 aerial)', license='Etalab 2.0',
                classes=FLAIR_CLASSES, ignore_id=0,
                native_gsd='0.2 m orthophotos (IGN BD ORTHO), 512x512 patches; RGB kept, NIR and elevation dropped',
                altitude='airborne survey camera (orthorectified)',
                notes=['split "trainval" = the 40 public train/val domains (the official 32/8 division is only given as '
                       'a figure); "test" = flair#1-test and "test2" = flair#2-test (10 domains each, other patches of the same domains); scene_group = domain/zone (e.g. D004_2021/Z10_UU)',
                       'mask values 1-19 as the authors; 0 never occurs (kept as ignore)',
                       'classes 14-19 (snow, clear cut, mixed, ligneous, greenhouse, other) are rare; the authors merge '
                       'them into "other" for their 13-class baseline'])
    return meta, items


# ---------------------------------------------------------------- FloodNet (Hurricane Harvey, DJI Mavic Pro at 200 ft)


# ---------------------------------------------------------------- ISPRS Potsdam (5 cm true orthophoto, 512 px tiles)
# ids of the aamante/potsdam label images (ISPRS reference with eroded object boundaries); ids found from the images:
# 0 = sand/construction clutter, 6 = the eroded boundary ring and the zero padding of the edge tiles
POTSDAM_CLASSES = ['clutter/background', 'impervious surfaces', 'building', 'low vegetation', 'tree', 'car',
                   'void (eroded object boundaries, tile padding)']
POTSDAM_TILE = 512  # the 6000 x 6000 px ISPRS tiles are cut into a 12 x 12 grid of 512 px tiles, zero padded (void)


def load_potsdam(image, label):
    import rasterio
    with rasterio.open(image) as src:
        rgb = src.read((1, 2, 3)).transpose(1, 2, 0)
    if rgb.dtype != np.uint8:
        raise Skip(f'RGB dtype {rgb.dtype}')
    return np.ascontiguousarray(rgb), np.asarray(Image.open(label))


def load_potsdam_window(image, label, x, y):
    import rasterio
    from rasterio.windows import Window
    window = Window(x * POTSDAM_TILE, y * POTSDAM_TILE, POTSDAM_TILE, POTSDAM_TILE)
    with rasterio.open(image) as src:
        rgb = src.read((1, 2, 3), window=window, boundless=True, fill_value=0).transpose(1, 2, 0)
    with rasterio.open(label) as src:
        ids = src.read(1, window=window, boundless=True, fill_value=6)
    if rgb.dtype != np.uint8 or ids.dtype != np.uint8:
        raise Skip(f'dtype {rgb.dtype} {ids.dtype}')
    return np.ascontiguousarray(rgb), ids


def potsdam():
    root = DOWNLOADS / 'potsdam/Potsdam'
    names = lambda split: set((root / 'splits' / f'{split}.txt').read_text().split())
    train_tiles, val_tiles = names('all_train'), names('val')
    items = []
    for image in sorted((root / 'all_train/RGBIR').glob('*.tif')):
        tile = '_'.join(image.stem.split('_')[1:3])
        assert image.stem in train_tiles
        items.append(Item(image.stem, 'val' if image.stem in val_tiles else 'train', f'Potsdam/{tile}', .05,
                          partial(load_potsdam, image, root / 'all_train/gt_no_boundary' / f'{image.stem}.png')))
    for name in sorted(names('test')):
        tile = name[len('potsdam_'):]
        for y in range(12):
            for x in range(12):
                items.append(Item(f'{name}_{x}_{y}', 'test', f'Potsdam/{tile}', .05,
                                  partial(load_potsdam_window, root / 'test/RGBIR' / f'{name}.tif',
                                          root / 'test/gt_no_boundary' / f'{name}.tif', x, y)))
    meta = dict(sensor='dom', url='https://huggingface.co/datasets/aamante/potsdam (ISPRS 2D Potsdam, RGB of RGBIR)',
                license='ISPRS benchmark terms (research)', classes=POTSDAM_CLASSES, ignore_id=6,
                native_gsd='0.05 m true orthophoto (GeoTIFF pixel size 0.05 m)',
                altitude='airborne survey (true orthophoto)',
                notes=['labels are the ids of the aamante/potsdam label images (ISPRS reference, eroded boundaries); '
                       'the class order was identified from the images (0 = sand/construction clutter, 6 = thin ring '
                       'around objects and the zero padding of edge tiles), no README states it',
                       'file names potsdam_<col>_<row>_<x>_<y>: ISPRS tile and the 512 px tile column x, row y '
                       '(12 x 12 per 6000 px tile, last row/column zero padded and labelled 6)',
                       'splits as the repository: val = splits/val.txt (4 tiles 2_11, 4_11, 6_11, 6_9, taken from the '
                       '23 train tiles), train = the rest of all_train, test = the 14 ISPRS test tiles (labels '
                       'released by ISPRS); train tile 7_10 is not in the repository; scene_group = ISPRS tile'])
    return meta, items


# ---------------------------------------------------------------- unlabelled drone orthomosaics (Global South cities)
TILE_PX = 1024          # tile side at the store scale
MAX_NODATA = .1         # tiles with more black (no data) border than this are left out


def ground_metres(src):
    """Pixel size in metres of a GeoTIFF in a projected (metre) or geographic (degree) CRS."""
    if src.crs.is_geographic:
        lat = np.deg2rad((src.bounds.top + src.bounds.bottom) / 2)
        return abs(src.res[0]) * 111320 * np.cos(lat), abs(src.res[1]) * 110574
    return abs(src.res[0]), abs(src.res[1])


def load_ortho_tile(path, col, row, native_px):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    with rasterio.open(path) as src:
        a = src.read((1, 2, 3), window=Window(col, row, native_px, native_px), out_shape=(3, TILE_PX, TILE_PX),
                     resampling=Resampling.average, boundless=True, fill_value=0)
        alpha = src.count >= 4 and src.read(4, window=Window(col, row, native_px, native_px), out_shape=(TILE_PX, TILE_PX),
                                            resampling=Resampling.nearest, boundless=True, fill_value=0)
    rgb = np.ascontiguousarray(a.transpose(1, 2, 0))
    nodata = (rgb.max(-1) == 0) if alpha is False else (alpha == 0)
    if nodata.mean() > MAX_NODATA:
        raise Skip('mostly outside the orthomosaic')
    return rgb, None


def global_south():
    import rasterio
    items, files = [], []
    for path in sorted((DOWNLOADS / 'global_south').glob('*/*.tif')):
        with rasterio.open(path) as src:
            gx, gy = ground_metres(src)
            if src.count < 3 or src.dtypes[0] != 'uint8':
                continue
            native = (gx + gy) / 2
            store = max(native, .05)
            native_px = round(TILE_PX * store / native)
            files.append(dict(file=f'{path.parent.name}/{path.name}', native_gsd_m=round(native, 4), crs=str(src.crs)))
            for row in range(0, src.height, native_px):
                for col in range(0, src.width, native_px):
                    items.append(Item(f'{path.parent.name}/{path.stem}/{row // native_px:03d}_{col // native_px:03d}',
                                      'unlabeled', f'GlobalSouth/{path.parent.name}/{path.stem}', store,
                                      partial(load_ortho_tile, path, col, row, native_px)))
    meta = dict(sensor='drone', url='https://huggingface.co/datasets/sparsedata/global-south-urban-drone-imagery',
                license='CC BY 4.0 (credit each provider: CREDITS.txt per city)', classes=[], ignore_id=None,
                native_gsd='per orthomosaic (orthomosaics listed below), 2-10 cm; tiles are read already averaged to '
                           'the store scale, so native_gsd_m in the manifest is the store scale',
                altitude='drone orthomosaics', orthomosaics=files,
                notes=['no labels: for self-training / domain coverage (Africa, South America, Southeast Asia)',
                       f'{TILE_PX} px tiles; tiles with more than {MAX_NODATA:.0%} no-data border are left out'])
    return meta, items


# ---------------------------------------------------------------- UAVScenes (DJI M300 + Hikvision camera, 80-90 m, nadir)
UAVSCENES_CLASSES = ['background', 'roof', 'dirt_motor_road', 'paved_motor_road', 'river', 'pool', 'bridge', '(unnamed 7)',
                     '(unnamed 8)', 'container', 'airstrip', 'traffic_barrier', '(unnamed 12)', 'green_field', 'wild_field',
                     'solar_board', 'umbrella', 'transparent_roof', 'car_park', 'paved_walk', 'sedan', '(unnamed 21)',
                     '(unnamed 22)', '(unnamed 23)', 'truck', '(unnamed 25)']  # names of cmap.py of the authors
UAVSCENES_CRUISE_M = {'AMtown': 80, 'HKairport': 80, 'HKisland': 90}  # MARS-LVIG table, above the take-off point


def load_uavscenes(image, label):
    return np.asarray(Image.open(image).convert('RGB')), np.asarray(Image.open(label))


def uavscenes():
    root = DOWNLOADS / 'uavscenes'
    items, runs = [], {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        run = folder.name
        rtk = np.genfromtxt(folder / 'rtk.csv', delimiter=',', skip_header=1, usecols=(1, 4))
        focal = json.loads((folder / 'sampleinfos.json').read_text())[0]['P3x3'][0][0]
        cruise = next(v for k, v in UAVSCENES_CRUISE_M.items() if run.startswith(k))
        gsd = cruise / focal
        frames = sorted(folder.glob('*.jpg'))
        runs[run] = dict(frames=len(frames), cruise_altitude_m=cruise, focal_px=round(focal, 1), gsd_m=round(gsd, 4))
        for image in frames:
            height = float(np.interp(float(image.stem), rtk[:, 0], rtk[:, 1]) - rtk[0, 1])
            items.append(Item(f'{run}/{image.stem}', 'all', run, gsd, partial(load_uavscenes, image, image.with_suffix('.png')),
                              round(height, 1), dict(gsd_estimated=True)))
    meta = dict(sensor='drone', url='https://huggingface.co/datasets/sijieaaa/UAVScenes (interval-5 release, every 10th frame of it)',
                license='CC BY-NC-SA 4.0 (academic use), built on MARS-LVIG', classes=UAVSCENES_CLASSES, ignore_id=0,
                absent_classes=[7, 8, 12, 21, 22, 23],  # unnamed ids that do not occur in this subsample
                native_gsd='estimated: cruise altitude of the MARS-LVIG table / focal length in pixels of the authors\' '
                           'intrinsics (P3x3); true height above ground varies with terrain',
                altitude='80 m (AMtown, HKairport) and 90 m (HKisland) above the take-off point, camera looking straight down '
                         '(rotation z axis of every pose checked ~ -1); altitude_m per image = RTK height minus the first RTK '
                         'height of the run; AMvalley (130 m) is not included',
                runs=runs,
                notes=['labels are the authors\' frame labels projected from the annotated 3D map plus instance fixing: coarse, with '
                       'unlabelled (0) halos around objects; classes 7, 8, 12, 21, 22, 23, 25 are unnamed in cmap.py',
                       'one frame of ten of the interval-5 release (2 Hz), runs are different passes over four areas; '
                       'no author split: split "all", scene_group = run (keep a run on one side of any split)',
                       'scenes: rural town (AMtown), airfield (HKairport), islands/forest/coast (HKisland)'])
    return meta, items


# ---------------------------------------------------------------- AeroScapes (Autel drone, 5-50 m, 720p), near-nadir sequences only
AEROSCAPES_CLASSES = ['Background', 'Person', 'Bike', 'Car', 'Drone', 'Boat', 'Animal', 'Obstacle', 'Construction', 'Vegetation',
                      'Road', 'Sky']


def load_aeroscapes(image, label):
    return np.asarray(Image.open(image).convert('RGB')), np.asarray(Image.open(label))


def aeroscapes():
    root = DOWNLOADS / 'aeroscapes'
    selection = json.loads((DOWNLOADS / 'aeroscapes_check/selection.json').read_text())
    items = []
    for seq, info in sorted(selection['kept'].items()):
        for name in info['frames']:
            items.append(Item(f'{seq}/{name}', info['split'], f'Aeroscapes/{seq}', info['gsd_m'],
                              partial(load_aeroscapes, root / seq / f'{name}.jpg', root / seq / f'{name}.png'),
                              None, dict(gsd_estimated=True)))
    meta = dict(sensor='drone', url='https://huggingface.co/datasets/dronefreak/Aeroscapes (redistribution of AeroScapes, Nigam et al. 2018)',
                license='CC BY-SA 4.0', classes=AEROSCAPES_CLASSES, ignore_id=None,
                absent_classes=selection['absent_classes'],
                native_gsd='not published: estimated per sequence from the masks of objects of known size (see sequences)',
                altitude='5-50 m (authors); per image unknown',
                sequences_kept=selection['kept_summary'], sequences_dropped=selection['dropped'],
                notes=['only near-nadir video sequences are kept: frames with any sky pixel are dropped; every sequence was judged on two '
                       'sample frames and the survivors again on four (start, 1/3, 2/3, end) and dropped for converging parallel '
                       'edges or people seen from the side (ground-level zoo sequences, horizon views, sea-only sequences, '
                       'oblique roads and paths)',
                       'every 4th remaining frame of a sequence is kept (near-identical neighbours thinned); '
                       'the authors\' train/val split is kept (no sequence is in both)',
                       'gsd_m is an estimate (gsd_estimated true) from car masks (4.5 x 1.8 m, at least 6 masks per sequence); '
                       'sequences whose scale rested only on person masks were dropped (person masks are not a reliable ruler)'])
    return meta, items


# ---------------------------------------------------------------- Kust4K (UAV RGB-thermal urban traffic scenes), RGB day frames
KUST4K_CLASSES = ['unlabelled', 'road', 'building', 'motorcycle', 'car', 'truck', 'tree', 'person', 'traffic facilities']
KUST4K_ROOT = Path('/shibo_huang/datasets/traversability_semantic/sources/kust4k/original')  # read only
KUST4K_CAR_M2 = 4.5 * 1.8


def kust4k_car_areas(label):
    """Pixel areas of the car masks (id 4) that do not touch the image border."""
    from scipy import ndimage
    ids = np.asarray(Image.open(label))
    labelled, count = ndimage.label(ids == 4)
    areas = []
    for index, box in enumerate(ndimage.find_objects(labelled), 1):
        if box[0].start > 0 and box[1].start > 0 and box[0].stop < ids.shape[0] and box[1].stop < ids.shape[1]:
            areas.append(int((labelled[box] == index).sum()))
    return [a for a in areas if a >= 250]


def load_kust4k(image, label):
    return np.asarray(Image.open(image).convert('RGB')), np.asarray(Image.open(label))


def kust4k():
    files = []
    for author_split in ('train', 'val', 'test'):
        for image in sorted((KUST4K_ROOT / author_split / 'images').glob('*D.png')):
            files.append((author_split, image, KUST4K_ROOT / author_split / 'labels' / image.name))
    block = lambda image: f'Kust4K/block{int(image.stem[:-1]) // 200:02d}'
    cars = {}
    for _, image, label in files:
        cars.setdefault(block(image), []).extend(kust4k_car_areas(label))
    every = [a for areas in cars.values() for a in areas]
    gsd = {b: float(np.sqrt(KUST4K_CAR_M2 / np.median(a if len(a) >= 10 else every))) for b, a in cars.items()}
    items = [Item(image.stem, 'all', block(image), gsd[block(image)], partial(load_kust4k, image, label), None,
                  dict(gsd_estimated=True, author_split=author_split)) for author_split, image, label in files]
    evidence = {b: dict(gsd_m=round(gsd[b], 4), car_masks=len(cars[b]), basis='block' if len(cars[b]) >= 10 else 'all blocks')
                for b in sorted(cars)}
    meta = dict(sensor='drone', url='https://figshare.com/articles/dataset/29476610 (Kust4K, v3)', license='CC BY 4.0',
                classes=KUST4K_CLASSES, ignore_id=0,
                native_gsd='estimated per block (gsd_estimated true): median car mask area of the block (a car taken as 4.5 x 1.8 m '
                           'seen from above, so +-30 %); the altitude changes between and within videos',
                altitude='not published; nadir', gsd_by_block=evidence,
                notes=['RGB daytime frames only (ids ending in D); night frames (N) and the thermal images are not used',
                       'class names are the paper\'s eight categories (Road, Building, Motorcycle, Car, Truck, Tree, Person, '
                       'Traffic Facilities) assigned to ids 1-8 in that order; checked by median mask sizes (motorcycle 431 px, '
                       'car 2691, truck 4517, person 173); id 0 (23 % of pixels) is treated as unlabelled',
                       'the authors\' train/val/test split is random over neighbouring frames of the same videos, so it is kept '
                       'only as the manifest field author_split; split is "all" and scene_group = block of 200 consecutive '
                       'frame numbers (flights are not published: the block is an unverified proxy for a video)',
                       'frames are video samples of road scenes: far fewer independent scenes than images'])
    return meta, items

# ---------------------------------------------------------------- UAVid (oblique, 4K, ~50 m) and UAVid++ (its relabelled copy)
UAVID_CLASSES = ['clutter', 'building', 'road', 'static car', 'tree', 'vegetation', 'human', 'moving car']
UAVID_MIRROR = DOWNLOADS / 'dl_UAVid-2020'
UAVIDPP_ROOT = Path('/shibo_huang/datasets/traversability_semantic/sources/uavidplusplus/original')  # read only
UAVIDPP_CLASSES = ['clutter', 'building (wall)', 'road', 'tree', 'low vegetation', 'water', 'sky', 'roof', 'static car', 'moving car', 'human']
UAVIDPP_RGB = [(0, 0, 0), (128, 0, 0), (128, 64, 128), (0, 128, 0), (128, 128, 0), (0, 0, 255), (128, 255, 255), (70, 70, 70),
               (192, 0, 192), (64, 0, 128), (64, 64, 0)]  # dataloader.py of github.com/vivichiciudean/uavidplusplus-code
UAVID_CAR_VISIBLE_M2 = 8.5  # image area of a 4.5 x 1.8 x 1.45 m car seen 45 degrees from the vertical: roof 8.1 x .71 plus about 2.8 of side faces
UAVID_PITCH_DEG = 45  # "camera angle around 45 degrees to the vertical" (Lyu et al. 2020), so also 45 above the horizon
UAVID_ALTITUDE_M = 50  # "safe flying height around 50 meters" (same paper)


def uavid_car_sizes(path):
    """sqrt of the pixel area of every car mask (static and moving) that does not touch the border and centres in the middle band."""
    from scipy import ndimage
    ids = np.asarray(Image.open(path))
    labelled, _ = ndimage.label(np.isin(ids, (3, 7)))
    sizes = []
    for index, box in enumerate(ndimage.find_objects(labelled), 1):
        if box[0].start > 0 and box[1].start > 0 and box[0].stop < ids.shape[0] and box[1].stop < ids.shape[1]:
            area = int((labelled[box] == index).sum())
            if area >= 800 and .35 < (box[0].start + box[0].stop) / 2 / ids.shape[0] < .7:
                sizes.append(area ** .5)
    return sizes


def uavid_gsd():
    """gsd (m, horizontal, at the image centre) per sequence from the car masks of the UAVid release: the upper quartile of the
    mask sizes (occlusion only makes masks smaller) of the centre band; sequences with fewer than 20 such cars use all sequences."""
    from multiprocessing import Pool
    paths = sorted((UAVID_MIRROR / 'masks').glob('*/*.png'))
    with Pool(16) as pool:
        sizes = pool.map(uavid_car_sizes, paths)
    by_seq = {}
    for path, s in zip(paths, sizes):
        by_seq.setdefault(path.stem.split('_')[0], []).extend(s)
    pooled = float(np.percentile([x for s in by_seq.values() for x in s], 75))
    table = {}
    for seq, s in by_seq.items():
        own = len(s) >= 20
        px = float(np.percentile(s, 75)) if own else pooled
        table[seq] = dict(gsd_m=round(UAVID_CAR_VISIBLE_M2 ** .5 / px, 4), car_masks=len(s), car_px=round(px, 1), basis='sequence' if own else 'all sequences')
    return table


def load_uavid(image, label):
    return np.asarray(Image.open(image).convert('RGB')), np.asarray(Image.open(label))


def load_uavidpp(image, label):
    rgb = np.asarray(Image.open(image).convert('RGB'))
    colour = np.asarray(Image.open(label).convert('RGB')).astype(np.int32)
    key = (colour[..., 0] << 16) | (colour[..., 1] << 8) | colour[..., 2]
    ids = np.full(key.shape, 255, np.uint8)
    for i, (r, g, b) in enumerate(UAVIDPP_RGB):
        ids[key == (r << 16) | (g << 8) | b] = i
    if (ids == 255).any():
        raise ValueError(f'{label}: colours outside the UAVid++ palette')
    return rgb, ids


def uavid_extra(table, seq):
    return dict(view='oblique', pitch_deg=UAVID_PITCH_DEG, gsd_estimated=True, gsd_basis=table[seq]['basis'])


def uavid():
    """The images of UAVid that UAVid++ does not relabel (verified pixel-identical: UAVid++ is the 270 train and val frames)."""
    table = uavid_gsd()
    done = {f'{p.parent.parent.name}_{p.stem}' for p in UAVIDPP_ROOT.glob('*/seq*/Images/*.png')}
    items = []
    for image in sorted((UAVID_MIRROR / 'images').glob('*/*.png')):
        if image.stem in done:
            continue
        seq = image.stem.split('_')[0]
        items.append(Item(f'{seq}/{image.stem.split("_")[1]}', image.parent.name, f'UAVid/{seq}', table[seq]['gsd_m'],
                          partial(load_uavid, image, UAVID_MIRROR / 'masks' / image.parent.name / image.name), UAVID_ALTITUDE_M,
                          uavid_extra(table, seq)))
    meta = dict(sensor='drone', url='https://huggingface.co/datasets/dronefreak/UAVid-2020 (redistribution of UAVid, Lyu et al. 2020)',
                license='CC BY-NC-SA 4.0', classes=UAVID_CLASSES, ignore_id=255,
                native_gsd='approximate, horizontal at the image centre, per sequence from car masks (a car seen from 45 degrees covers about '
                           f'{UAVID_CAR_VISIBLE_M2} m2): gsd = sqrt({UAVID_CAR_VISIBLE_M2}) / upper-quartile sqrt(mask area); the nominal geometry '
                           '(50 m, 45 degrees, Phantom 3/4 4K, about 2000 px focal length) gives about 0.034 m; far pixels are coarser',
                altitude='"around 50 m", camera 45 degrees from the vertical (paper); 4K frames 3840x2160 or 4096x2160',
                gsd_by_sequence=table,
                notes=['only the 150 test-split images: the 270 train and val images are in source uavidplusplus (pixel-identical RGB, finer labels)',
                       'clutter (id 0) includes sky, unlabelled and other objects; the authors use all 8 classes',
                       'masks of the mirror are single-channel ids 0-7 in the order of its data.yaml; 255 (a few pixels, unrecognised colours of the conversion) is ignore'])
    return meta, items


def uavidplusplus():
    table = uavid_gsd()
    items = []
    for image in sorted(UAVIDPP_ROOT.glob('*/seq*/Images/*.png')):
        seq, split = image.parent.parent.name, image.parent.parent.parent.name[len('uavid_'):]
        items.append(Item(f'{seq}/{image.stem}', split, f'UAVid/{seq}', table[seq]['gsd_m'],
                          partial(load_uavidpp, image, image.parent.parent / 'Labels' / image.name), UAVID_ALTITUDE_M, uavid_extra(table, seq)))
    meta = dict(sensor='drone', url='https://huggingface.co/datasets/vivianchiciudean/uavidplusplus (local archive copy; Chiciudean et al. 2026)',
                license='CC BY-NC-SA 4.0', classes=UAVIDPP_CLASSES, ignore_id=None, label_palette_rgb=UAVIDPP_RGB,
                native_gsd='as source uavid (same images): per sequence from car masks of the UAVid release', gsd_by_sequence=table,
                altitude='"around 50 m", camera 45 degrees from the vertical (UAVid paper); 4K frames',
                notes=['the same 270 frames as UAVid train (200) and val (70), pixel-identical, with the corrected labels and the classes '
                       'water, sky and roof added; the 150 test frames are in source uavid with the original labels',
                       'colour labels decoded with the palette of the authors\' dataloader.py; clutter (id 0) is the remaining unlabelled/other class',
                       'scene_group = UAVid sequence (same names as source uavid)'])
    return meta, items

# ---------------------------------------------------------------- AeroScapes, the sequences not in source aeroscapes (oblique and horizon views)
AEROSCAPES_MIRROR = DOWNLOADS / 'dl_Aeroscapes/data'
AEROSCAPES_NOT_AERIAL = {'038': 'ground-level zoo cameras', '115': 'open sea only', '116': 'open sea only'}  # sequence-name prefixes
AEROSCAPES_MAX_SKY = .10
AEROSCAPES_DIFF = 12.  # mean grey-level difference of 64x36 thumbnails that makes a frame "new" (consecutive frames: median 10.7)
AEROSCAPES_GAP = (2, 4)  # keep every 2nd to 4th frame of a sequence
AEROSCAPES_WINDOW = 8  # frames either side whose object masks are pooled for the ground scale
AEROSCAPES_CAR_M2, AEROSCAPES_PERSON_M2 = 8.5, .55  # visible area of a 4.5 x 1.8 x 1.45 m car and of a 1.7 m person seen between nadir and 45 degrees


def aeroscapes_frame(args):
    from scipy import ndimage
    split, name = args
    ids = np.asarray(Image.open(AEROSCAPES_MIRROR / 'masks' / split / f'{name}.png'))
    thumb = np.asarray(Image.open(AEROSCAPES_MIRROR / 'images' / split / f'{name}.jpg').convert('L').resize((64, 36), Image.BOX), np.float32)
    objects = {}
    for key, cls, minimum in (('car', 3, 100), ('person', 1, 40)):
        labelled, _ = ndimage.label(ids == cls)
        sizes = []
        for index, box in enumerate(ndimage.find_objects(labelled), 1):
            if box[0].start > 0 and box[1].start > 0 and box[0].stop < ids.shape[0] and box[1].stop < ids.shape[1]:
                area = int((labelled[box] == index).sum())
                if area >= minimum:
                    sizes.append(area ** .5)
        objects[key] = sizes
    return name, dict(split=split, sky=float((ids == 11).mean()), thumb=thumb, hist=np.bincount(ids.ravel(), minlength=12), **objects)


def aeroscapes_plan():
    """Select and scale the frames: no sky above 10 %, not a zoo or sea-only sequence, thinned by image difference; ground scale per frame
    from the car masks (else the person masks) of the frames within +-8 of it; frames without enough objects to measure are dropped."""
    from multiprocessing import Pool
    already = {json.loads(line)['scene_group'].split('/')[1] for line in (STORE / 'aeroscapes/manifest.jsonl').read_text().splitlines()}
    jobs = [(s, p.stem) for s in ('train', 'val') for p in sorted((AEROSCAPES_MIRROR / 'images' / s).glob('*.jpg'))
            if p.stem[:6] not in already and p.stem[:3] not in AEROSCAPES_NOT_AERIAL]
    with Pool(16) as pool:
        stats = dict(pool.map(aeroscapes_frame, jobs, chunksize=8))
    sequences = {}
    for name in sorted(stats):
        sequences.setdefault(name[:6], []).append(name)
    plan, dropped = {}, dict(sky=0, thinned=0, unmeasurable=0)
    for seq, names in sequences.items():
        last, kept = None, []
        for i, name in enumerate(names):
            if stats[name]['sky'] > AEROSCAPES_MAX_SKY:
                dropped['sky'] += 1
                continue
            if last is not None:
                gap = i - last[0]
                new = float(np.abs(stats[name]['thumb'] - last[1]).mean()) >= AEROSCAPES_DIFF
                if gap < AEROSCAPES_GAP[0] or (gap < AEROSCAPES_GAP[1] and not new):
                    dropped['thinned'] += 1
                    continue
            kept.append(i)
            last = (i, stats[name]['thumb'])
        for i in kept:
            near = names[max(0, i - AEROSCAPES_WINDOW):i + AEROSCAPES_WINDOW + 1]
            cars = [x for n in near for x in stats[n]['car']]
            people = [x for n in near for x in stats[n]['person']]
            if len(cars) >= 5:
                gsd, basis, count = AEROSCAPES_CAR_M2 ** .5 / np.median(cars), 'cars', len(cars)
            elif len(people) >= 10:
                gsd, basis, count = AEROSCAPES_PERSON_M2 ** .5 / np.median(people), 'people', len(people)
            else:
                dropped['unmeasurable'] += 1
                continue
            plan[names[i]] = dict(split=stats[names[i]]['split'], gsd_m=float(gsd), basis=basis, masks=count, hist=stats[names[i]]['hist'])
    return plan, dropped


def aeroscapes_oblique():
    plan, dropped = aeroscapes_plan()
    used = sum(p['hist'] for p in plan.values())
    items = [Item(f'{name[:6]}/{name}', p['split'], f'Aeroscapes/{name[:6]}', p['gsd_m'],
                  partial(load_aeroscapes, AEROSCAPES_MIRROR / 'images' / p['split'] / f'{name}.jpg', AEROSCAPES_MIRROR / 'masks' / p['split'] / f'{name}.png'),
                  None, dict(view='oblique', pitch_deg=None, gsd_estimated=True, gsd_basis=p['basis'], gsd_objects=p['masks']))
             for name, p in sorted(plan.items())]
    meta = dict(sensor='drone', url='https://huggingface.co/datasets/dronefreak/Aeroscapes (redistribution of AeroScapes, Nigam et al. 2018)',
                license='CC BY-SA 4.0', classes=AEROSCAPES_CLASSES, ignore_id=None,
                absent_classes=[i for i in range(len(AEROSCAPES_CLASSES)) if not used[i]],
                native_gsd='not published: estimated per frame from the masks of objects of known size in the frames within +-8 of it '
                           f'(median of the car masks, a car seen between nadir and 45 degrees covering {AEROSCAPES_CAR_M2} m2, if there are at least 5; '
                           f'else of the person masks, {AEROSCAPES_PERSON_M2} m2, at least 10); ground scale at the image centre is only approximate in '
                           'oblique frames; person-based values (gsd_basis people) are uncertain by about +-40 %',
                altitude='5-50 m (authors); per image unknown', pitch='unknown per frame: sequences range from near-nadir to oblique',
                notes=['every sequence of AeroScapes that is not in source aeroscapes (which holds the near-nadir sequences judged on car masks), '
                       'except ground-level zoo sequences (038xxx) and sea-only sequences (115xxx, 116xxx)',
                       f'frames with more than {AEROSCAPES_MAX_SKY:.0%} sky pixels are dropped; the remaining sky stays labelled Sky',
                       f'thinned to every {AEROSCAPES_GAP[0]}nd-{AEROSCAPES_GAP[1]}th frame: a frame is kept at least {AEROSCAPES_GAP[0]} and at most '
                       f'{AEROSCAPES_GAP[1]} frames after the last kept one, as soon as its mean grey-level difference (64x36 thumbnails) exceeds {AEROSCAPES_DIFF}',
                       'view is "oblique" for all rows (the near-nadir sequences are in source aeroscapes) but pitch_deg is not known, so some frames here are nearly nadir',
                       'the authors\' train/val split is kept (no sequence is in both)'],
                dropped_frames=dropped)
    return meta, items


SOURCES = dict(flair=flair, potsdam=potsdam, global_south=global_south, uavscenes=uavscenes, aeroscapes=aeroscapes, kust4k=kust4k, uavid=uavid, uavidplusplus=uavidplusplus, aeroscapes_oblique=aeroscapes_oblique)
