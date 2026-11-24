"""The ground-surface classes, how every source's native classes map onto them, and how good each is to drive on.

CLASSES are the surfaces a global route chooses between, in order of preference. A source class that does not resolve
to one of them is a SET of them (for example 'vegetation' = {shrub, tree}); the loss then only says what the pixel is
not.

Traversability is one graded value per pixel, 0-255: the expected value of the class posterior, so it changes smoothly
with the model's certainty. VALUE holds the robot's priorities (Scout Mini). A route pays 1 / value per metre.
"""
import numpy as np

CLASSES = ('road', 'dirt', 'grass', 'field', 'tree', 'shrub', 'obstacle', 'vehicle', 'water')
COLORS = np.array([[128, 64, 128], [210, 170, 110], [107, 182, 35], [230, 220, 90], [20, 90, 20], [60, 120, 90],
                   [180, 60, 60], [0, 0, 230], [50, 110, 210]], np.uint8)
IGNORE = frozenset()

# road: every paved surface (carriageway, pavement, square, car park); dirt: tracks, gravel, bare soil; field: crops,
# vineyards and plowed land; tree: the ground under a crown is unseen; obstacle: buildings, walls, rails, rocks.
VALUE = {'road': 255, 'dirt': 190, 'grass': 130, 'field': 70, 'tree': 50, 'shrub': 25, 'obstacle': 0, 'vehicle': 0,
         'water': 0}
VALUES = np.array([VALUE[c] for c in CLASSES], np.float32)
DRIVABLE, BLOCKED = ('road', 'dirt', 'grass'), ('shrub', 'obstacle', 'vehicle', 'water')   # field and tree: undecided
VALUE_FLOOR = 5.                                                                           # a blocked metre costs 51


def bits(names):
    """The bitmask of a set of class names (0 = ignore)."""
    return sum(1 << CLASSES.index(n) for n in names)


def traversability(posterior):
    """Graded value 0-255 (float) of a class posterior (classes on axis -3)."""
    return (posterior * VALUES.reshape(-1, 1, 1)).sum(-3)


def route_cost(value):
    """Cost per metre: 1 on a road, more on worse ground."""
    return 255. / np.maximum(value, VALUE_FLOOR)


def drivable_probability(posterior):
    return sum(posterior[CLASSES.index(c)] for c in DRIVABLE)


def _norm(name):
    return ' '.join(name.lower().replace('_', ' ').replace('-', ' ').split())


# Native class name (as the authors name it) -> CLASSES set. IGNORE: no supervision (unlabelled, people, markers, snow).
SOURCE_CLASSES = {
    'semantic_drone': {
        'unlabeled': IGNORE, 'paved area': {'road'}, 'dirt': {'dirt'}, 'grass': {'grass'},
        'gravel': {'dirt'}, 'water': {'water'}, 'rocks': {'obstacle'}, 'pool': {'water'},
        'vegetation': {'shrub', 'tree'}, 'roof': {'obstacle'}, 'wall': {'obstacle'}, 'window': {'obstacle'},
        'door': {'obstacle'}, 'fence': {'obstacle'}, 'fence pole': {'obstacle'}, 'person': IGNORE, 'dog': IGNORE,
        'car': {'vehicle'}, 'bicycle': {'obstacle'}, 'tree': {'tree'}, 'bald tree': {'tree'}, 'ar marker': IGNORE,
        'obstacle': {'obstacle'}, 'conflicting': IGNORE},
    'messi': {
        'background': IGNORE, 'bicycle': {'obstacle'}, 'building': {'obstacle'}, 'fence': {'obstacle'},
        'other objects': {'obstacle'}, 'person': IGNORE, 'pole': {'obstacle'}, 'rough terrain': {'dirt', 'shrub'},
        'shed': {'obstacle'}, 'soft terrain': {'grass', 'dirt'}, 'stairs': {'obstacle'},
        'transportation terrain': {'road'}, 'vegetation': {'shrub', 'tree'}, 'vehicle': {'vehicle'},
        'walking terrain': {'road'}, 'water': {'water'}},
    'okutama_50m': {
        'background': IGNORE, 'outdoor structures': {'obstacle'}, 'buildings': {'obstacle'},
        'paved ground': {'road'}, 'non paved ground': {'dirt', 'grass'}, 'train tracks': {'obstacle'},
        'plants': {'grass', 'field', 'shrub', 'tree'}, 'wheeled vehicles': {'vehicle'}, 'water': {'water'},
        'people': IGNORE},
    'mavsd': {   # rural weed survey, 15-40 m; about a quarter of the pixels are labelled, the rest is background
        'background': IGNORE, 'solidago canadensis': {'shrub'}, 'person': IGNORE, 'road': {'road', 'dirt'},
        'pine': {'tree'}, 'bush': {'shrub'}, 'reed': {'shrub'}, 'stone': {'obstacle'}, 'cypress': {'tree'},
        'water': {'water'}, 'building': {'obstacle'}, 'vegetable garden': {'field'}, 'electric bicycle': {'obstacle'}},
    'uavid': {   # the authors' test frames; train and val are in uavidplusplus with corrected labels
        'clutter': {'road', 'dirt', 'obstacle', 'water'},
        'building': {'obstacle'}, 'road': {'road'}, 'tree': {'tree'}, 'vegetation': {'grass', 'shrub', 'field'},
        'moving car': {'vehicle'}, 'static car': {'vehicle'}, 'human': IGNORE},
    'uavidplusplus': {   # the same frames as UAVid train and val, with water, sky and roof labelled
        'clutter': {'road', 'dirt', 'obstacle'},
        'building (wall)': {'obstacle'}, 'roof': {'obstacle'}, 'road': {'road'}, 'tree': {'tree'},
        'low vegetation': {'grass', 'shrub', 'field'}, 'water': {'water'}, 'sky': IGNORE,
        'static car': {'vehicle'}, 'moving car': {'vehicle'}, 'human': IGNORE},
    'flair': {   # no vehicle class: the few parked cars stay the surface they stand on
        'nodata': IGNORE, 'building': {'obstacle'}, 'pervious surface': {'dirt'},
        'impervious surface': {'road'}, 'bare soil': {'dirt', 'obstacle'}, 'water': {'water'},
        'coniferous': {'tree'}, 'deciduous': {'tree'}, 'brushwood': {'shrub'}, 'vineyard': {'field'},
        'herbaceous vegetation': {'grass'}, 'agricultural land': {'field'}, 'plowed land': {'field'},
        'swimming pool': {'water'}, 'snow': IGNORE, 'clear cut': {'dirt', 'grass', 'shrub'}, 'mixed': {'tree'},
        'ligneous': {'shrub', 'tree'}, 'greenhouse': {'obstacle'}, 'other': IGNORE},
    'aeroscapes': {   # 'road' is any paved or unpaved carriageway; background mixes paving, courts and soil beds: no information
        'background': IGNORE, 'person': IGNORE, 'bike': {'obstacle'},
        'car': {'vehicle'}, 'drone': IGNORE, 'boat': IGNORE, 'animal': IGNORE, 'obstacle': {'obstacle'},
        'construction': {'obstacle'}, 'vegetation': {'grass', 'field', 'shrub', 'tree'},
        'road': {'road', 'dirt'}, 'sky': IGNORE},
    'uavscenes': {   # labels projected from a 3D map; green_field covers trees as well as grass, wild_field also bare rock
        'background': IGNORE, 'roof': {'obstacle'}, 'dirt_motor_road': {'dirt'}, 'paved_motor_road': {'road'},
        'river': {'water'}, 'pool': {'water'}, 'bridge': {'road', 'obstacle'},
        'container': {'obstacle'}, 'airstrip': {'road'}, 'traffic_barrier': {'obstacle'},
        'green_field': {'grass', 'field', 'shrub', 'tree'}, 'wild_field': {'dirt', 'grass', 'shrub', 'obstacle'},
        'solar_board': {'obstacle'}, 'umbrella': {'obstacle'}, 'transparent_roof': {'obstacle'},
        'car_park': {'road'}, 'paved_walk': {'road'}, 'sedan': {'vehicle'}, 'truck': {'vehicle'},
        '(unnamed 7)': IGNORE,
        '(unnamed 8)': IGNORE,
        '(unnamed 12)': IGNORE,
        '(unnamed 21)': IGNORE,
        '(unnamed 22)': IGNORE,
        '(unnamed 23)': IGNORE,
        '(unnamed 25)': IGNORE,
    },
    'kust4k': {   # road scenes; sidewalks and grass are in 'unlabelled'
        'unlabelled': IGNORE, 'road': {'road'}, 'building': {'obstacle'}, 'motorcycle': {'vehicle'},
        'car': {'vehicle'}, 'truck': {'vehicle'}, 'tree': {'shrub', 'tree'}, 'person': IGNORE,
        'traffic facilities': {'obstacle'}},
    'potsdam': {
        'impervious surfaces': {'road'}, 'building': {'obstacle'}, 'low vegetation': {'dirt', 'grass', 'shrub'},
        'tree': {'tree'}, 'car': {'vehicle'}, 'void (eroded object boundaries, tile padding)': IGNORE,
        'clutter/background': IGNORE},
}

# The oblique releases of AeroScapes and MAVSD keep the authors' class list, so they share that source's mapping.
for _oblique, _nadir in (('aeroscapes_oblique', 'aeroscapes'), ('mavsd_oblique', 'mavsd')):
    SOURCE_CLASSES[_oblique] = dict(SOURCE_CLASSES[_nadir])


def label_table(source, native_classes):
    """uint16 LUT native id -> CLASSES bitmask (0 = ignore) for a source's native class list [id -> name]. Every
    native class must be mapped explicitly."""
    if source not in SOURCE_CLASSES:
        raise KeyError(f'source {source!r} has no taxonomy mapping (add it to SOURCE_CLASSES); its native classes are '
                       f'{list(native_classes)}')
    table = {_norm(k): v for k, v in SOURCE_CLASSES[source].items()}
    lut = np.zeros(256, np.uint16)
    for native_id, name in enumerate(native_classes):
        if name is None:
            continue
        if _norm(name) not in table:
            raise KeyError(f'{source}: native class {name!r} has no mapping')
        lut[native_id] = bits(table[_norm(name)])
    return lut
