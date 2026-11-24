"""NavVerse cities as CurveNav point-goal scenes for the selected robot, with one navigation C-space for the whole
city. A scene is vcp_<city> (the outdoor city, NavVerse's vcp episodes) or innout_<city> (the same city with
building interiors open to the street through doors and storefronts, NavVerse's innout episodes, which run from
indoors to outdoors).

Frames: the city USD is metric and Z-up; canonical XY = USD XY, Habitat = (X, Z, -Y). Official episodes keep their XY
and wxyz heading; their Z belongs to NavVerse's runtime ground alignment and is never used: heights come from the
city's own terrain.

Per city, robot-independent (--export):
  assets/stages/<city>.glb (+ stage/scene configs)   the whole city: collision, navigation and depth
  assets/stages/<city>.appearance.glb                the same triangles with their base colours: the rendered asset
  scenes/<city>/terrain.npz                          the terrain triangles: ground (all but water), water, and each
                                                     region (region_<road|sidewalk|crosswalk|curb|green|water|Terrain>)
  scenes/<city>/geometry.json                        export metadata
Per city and robot (KAPPANAV_ROBOT), in scenes/<city>/<robot>/:
  navigation.npz, navigation.json                    the robot's C-space of the whole city, with ground height
<city>_<robot>_tasks.json       the official episodes the robot can do (and the rejected ones with the reason)
<city>_<robot>_long_tasks.json  sampled long-distance tasks (--long)
Every task of a city is evaluated on the one city scene (stage and C-space) and carries its C-space shortest path.

The city navmesh is built in tiles (Recast's voxel grid bounds the size of one build) that overlap by more than
the robot radius, so every tile's interior is exact and the tiles join without seams. Water is rendered but is not
ground. A building without a door is a closed shell whose floor is navmesh too: a free component mostly under a roof
(INDOOR_SHARE of it COVERED_HEIGHT_M below some surface) is such a floor and is dropped. An interior that opens to
the street (innout) belongs to the street's component and stays. Curbs, steps and slopes are judged by the robot's own limits (NavVerse's navmesh climbs 0.8 m), so an
official episode the robot cannot do is rejected with its reason, never made feasible by loosening the robot.
This is CurveNav's point-goal protocol (docs/EVALUATION.md), not NavVerse's PlaceNav scoring.
Run with scripts/evaluation/prepare_navverse.sh.
"""
import argparse, json, math, os, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from kappanav.common.physical import ROBOT
from kappanav.evaluation.simulation.assets import SCENE_DATASET, export_geometry, triangulate_faces, write_stage
from kappanav.evaluation.simulation.navigation import (BODY_GROUND_RADIUS_M, known_ground, navmesh_ground,
                                                       navmesh_settings, navmesh_triangles, raster_surface,
                                                       save_navigation_grid)

ROOT = Path(os.environ.get('KAPPANAV_NAVVERSE', '/shibo_huang/data/navverse/release'))
OUT = Path(os.environ.get('KAPPANAV_NAVVERSE_SCENES', '/shibo_huang/data/navverse/curvenav'))
ASSETS = OUT / 'assets'

GRID_M = 0.05                # CurveNav navigation grid
TILE_CELLS = 1000            # navmesh tile: 50 m; Detour holds 65535 vertices per build, about 100 m of dense street
TILE_SPLIT_MIN_CELLS = 500   # a tile whose build fails is split in four down to this size
TILE_OVERLAP_CELLS = 40      # 2 m beyond each tile edge: Recast erodes the clipped edge by the robot radius
WATER_ABOVE_GROUND_M = 0.05  # terrain less than this above the water surface is under water
SNAP_MAX_XY_M = 0.5          # official endpoints must snap this close to the robot's C-space
GROUND_FILL_M = 25.0         # ground with no terrain region (a building's floor) is at the terrain this near
COVERED_HEIGHT_M = 2.0      # a surface this far above the ground covers it (a roof, a ceiling, a canopy)
INDOOR_SHARE = 0.8           # a free component this much covered is the inside of a building shell
ENDPOINT_CLEARANCE_M = 0.10  # sampled starts and goals keep this clearance (simulation.tasks)
LONG_EDGES_M = (100., 200., 400.)  # default shortest-path bin edges of the long tasks
LONG_ATTEMPTS = 200          # starts tried for one long task before the city is declared unable to hold it


SOURCES = {'vcp': ('outdoor', 'vcp'), 'innout': ('innout', 'innout')}  # scene prefix: USD variant, episode set


def scene_files(scene_id):
    """The city USD and the official episodes of a scene."""
    prefix, city = scene_id.split('_', 1)
    variant, episodes = SOURCES[prefix]
    return (ROOT / 'vc_plus' / city.upper() / f'{city.upper()}_{variant}.usd',
            ROOT / 'episodes' / episodes / f'{scene_id}.json')


def heading(q):
    """Yaw about +Z of a NavVerse wxyz quaternion."""
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError('invalid NavVerse wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# ---------------------------------------------------------------- city geometry
def export_scene(scene_id):
    """The whole city as a Habitat stage, and its terrain regions as triangles in the USD frame."""
    from pxr import Usd, UsdGeom
    usd = scene_files(scene_id)[0]
    meta = export_geometry(usd, ROOT, ASSETS / 'stages' / f'{scene_id}.glb', root_path='/World', fix_blankets=False,
                           appearance=ASSETS / 'stages' / f'{scene_id}.appearance.glb', ground='/TerrainRegions/')
    write_stage(ASSETS, scene_id)
    stage, transforms, ground, water, regions = Usd.Stage.Open(str(usd)), UsdGeom.XformCache(), [], [], {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or '/TerrainRegions/' not in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        world = np.asarray(transforms.GetLocalToWorldTransform(prim))  # acts on row vectors
        points = np.asarray(mesh.GetPointsAttr().Get(), float) @ world[:3, :3] + world[3, :3]
        faces = triangulate_faces(points, np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int),
                                  np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int))
        (water if prim.GetName() == 'water' else ground).append(points[faces])
        regions.setdefault(f'region_{prim.GetName()}', []).append(points[faces])
    if not ground:
        raise ValueError(f'no terrain regions: {scene_id}')
    folder = OUT / 'scenes' / scene_id
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / 'terrain.npz', triangles=np.concatenate(ground),
             water=np.concatenate(water) if water else np.empty((0, 3, 3)),
             **{name: np.concatenate(parts) for name, parts in regions.items()})
    meta.update(scene_id=scene_id, source_usd=str(usd), frames='canonical XY = USD XY, Z up; habitat = (X, Z, -Y)')
    (folder / 'geometry.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


# ---------------------------------------------------------------- city navigation
_CITY = None  # per tile worker: (collision mesh in the Habitat frame, its faces' XY bounds, terrain, water triangles)


def load_city(scene_id):
    global _CITY
    import trimesh
    scene = trimesh.load(ASSETS / 'stages' / f'{scene_id}.glb', force='scene', process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        if '/TerrainRegions/water' in node:  # water does not carry the robot
            continue
        matrix, key = scene.graph[node]
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(matrix)
        meshes.append(mesh)
    mesh = trimesh.util.concatenate(meshes)
    xy = (mesh.vertices[:, [0, 2]] * [1., -1.])[mesh.faces]
    with np.load(OUT / 'scenes' / scene_id / 'terrain.npz') as data:
        _CITY = mesh, xy.min(1), xy.max(1), data['triangles'], data['water']


def tile_ground(lo, shape):
    """The selected robot's navigable ground on the grid (lo, shape): its navmesh of the city clipped to the grid,
    kept where it lies on the ground and not on water. The ground is the terrain regions, continued under buildings
(where NavVerse authors no region) at the nearest terrain height within GROUND_FILL_M: a floor level with the
street (an open interior) is ground, a roof is not. Layers: ground (height, NaN off it), covered (a surface
    COVERED_HEIGHT_M above the ground) and, for a rectangular robot, body (the body ground: navmesh with
    BODY_GROUND_RADIUS_M). None when a navmesh build fails."""
    import habitat_sim, trimesh
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    city, face_lo, face_hi, terrain, water = _CITY
    hi = lo + np.array(shape) * GRID_M
    pad = int(round(GROUND_FILL_M / GRID_M))
    wide_lo, wide_shape = lo - pad * GRID_M, (shape[0] + 2 * pad, shape[1] + 2 * pad)
    ground_z, water_z = raster_surface(terrain, wide_lo, wide_shape, GRID_M), raster_surface(water, wide_lo, wide_shape, GRID_M)
    regionless = ~np.isfinite(ground_z) & ~np.isfinite(water_z)
    if regionless.any() and np.isfinite(ground_z).any():
        nearest = distance_transform_edt(~np.isfinite(ground_z), return_distances=False, return_indices=True)
        ground_z[regionless] = ground_z[tuple(nearest)][regionless]
    ground_z[np.isfinite(water_z) & ~(ground_z > water_z + WATER_ABOVE_GROUND_M)] = np.nan
    ground_z = ground_z[pad:pad + shape[0], pad:pad + shape[1]]
    axes = [lo[a] + (np.arange(shape[a]) + .5) * GRID_M for a in (0, 1)]
    ground = RegularGridInterpolator(axes, ground_z, method='nearest', bounds_error=False, fill_value=np.nan)
    layers = dict(ground=np.full(shape, np.nan), covered=np.zeros(shape, bool),
                  **({} if ROBOT.body_half_extent_m is None else dict(body=np.zeros(shape, bool))))
    faces = city.faces[(face_hi >= lo).all(1) & (face_lo <= hi).all(1)]
    if not len(faces):
        return layers
    used, faces = np.unique(faces, return_inverse=True)
    mesh = trimesh.Trimesh(city.vertices[used], faces.reshape(-1, 3), process=False)
    for origin, normal in (([lo[0], 0, 0], [1, 0, 0]), ([hi[0], 0, 0], [-1, 0, 0]),
                           ([0, 0, -hi[1]], [0, 0, 1]), ([0, 0, -lo[1]], [0, 0, -1])):
        mesh = mesh.slice_plane(origin, normal)  # clipped triangles bound Recast's voxel grid
    if not len(mesh.faces):
        return layers
    top = raster_surface(mesh.vertices[mesh.faces][..., [0, 2, 1]] * [1., -1., 1.], lo, shape, GRID_M)  # habitat (X, Z, -Y)
    layers['covered'] = top - ground_z > COVERED_HEIGHT_M
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(doubleSided=True))
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        mesh.export(folder / 'tile.glb')
        (folder / 'tile.stage_config.json').write_text(json.dumps(dict(
            render_asset='tile.glb', collision_asset='tile.glb', up=[0, 1, 0], front=[0, 0, -1], units_to_meters=1.)))
        (folder / 'tile.scene_instance.json').write_text(json.dumps(dict(
            stage_instance=dict(template_name=str(folder / 'tile.stage_config.json')), object_instances=[])))
        sc = habitat_sim.SimulatorConfiguration()
        sc.scene_id, sc.enable_physics = str(folder / 'tile.scene_instance.json'), False
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = []
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [agent]))
        try:
            for name, settings in (('ground', navmesh_settings()), ('body', navmesh_settings(BODY_GROUND_RADIUS_M))):
                if name not in layers:
                    continue
                if not sim.recompute_navmesh(sim.pathfinder, settings):
                    return None
                surface = navmesh_ground(navmesh_triangles(sim.pathfinder), ground, lo, shape, spacing=GRID_M)
                layers[name] = surface if name == 'ground' else np.isfinite(surface)
            return layers
        finally:
            sim.close()


def tile(lo, shape):
    """tile_ground of the grid without its TILE_OVERLAP_CELLS border. A failed build (more navmesh than one Detour
    build holds, or nothing the robot can stand on) is split in four; at the smallest size it is no ground."""
    m = TILE_OVERLAP_CELLS
    built = tile_ground(lo, shape)
    if built is not None:
        return {name: layer[m:-m, m:-m] for name, layer in built.items()}
    inner = np.array(shape) - 2 * m
    if inner.max() <= TILE_SPLIT_MIN_CELLS:
        return dict(ground=np.full(inner, np.nan), covered=np.zeros(inner, bool),
                    **({} if ROBOT.body_half_extent_m is None else dict(body=np.zeros(inner, bool))))
    half, result = inner // 2, {}
    for a, na in ((0, half[0]), (half[0], inner[0] - half[0])):
        for b, nb in ((0, half[1]), (half[1], inner[1] - half[1])):
            for name, layer in tile(lo + np.array([a, b]) * GRID_M, (int(na) + 2 * m, int(nb) + 2 * m)).items():
                result.setdefault(name, np.empty(inner, layer.dtype))[a:a + na, b:b + nb] = layer
    return result


def _city_tile(origin, a, b, cells):
    m = TILE_OVERLAP_CELLS
    shape = tuple(int(min(TILE_CELLS, cells[k] - (a, b)[k])) + 2 * m for k in (0, 1))
    return a, b, tile(origin + (np.array([a, b]) - m) * GRID_M, shape)


def indoor(free, covered):
    """Free 8-connected components with INDOOR_SHARE of their cells covered: the insides of building shells."""
    from scipy import ndimage
    labels, count = ndimage.label(free, structure=np.ones((3, 3), bool))
    area = np.bincount(labels.ravel(), minlength=count + 1)
    under = np.bincount(labels.ravel(), weights=covered.ravel(), minlength=count + 1)
    inside = under >= INDOOR_SHARE * np.maximum(area, 1)
    inside[0] = False
    return inside[labels]


def navigation_pipeline():
    """Hash of the code the city navigation is built from (not the task code, which never changes it)."""
    from kappanav.common.geometry import Grid
    from kappanav.common.experiment_io import code_digest
    from kappanav.evaluation.simulation import navigation as n
    return code_digest(load_city, tile_ground, tile, _city_tile, build_navigation, n.navmesh_settings, n.navmesh_triangles,
                       n.navmesh_ground, n.known_ground, n._raster_surface, n.raster_surface, n.save_navigation_grid, Grid,
                       indoor, repr((GRID_M, TILE_CELLS, TILE_SPLIT_MIN_CELLS, TILE_OVERLAP_CELLS, WATER_ABOVE_GROUND_M,
                                     BODY_GROUND_RADIUS_M, GROUND_FILL_M, COVERED_HEIGHT_M, INDOOR_SHARE)))


def build_navigation(scene_id, workers):
    """scenes/<city>/<robot>/navigation.npz: the robot's C-space of the whole city. Reused while the code, the robot
    and the exported geometry are unchanged."""
    folder = OUT / 'scenes' / scene_id / ROBOT.name
    geometry = json.loads((OUT / 'scenes' / scene_id / 'geometry.json').read_text())
    inputs = json.loads(json.dumps(dict(robot=asdict(ROBOT), pipeline=navigation_pipeline(),
                                        geometry={k: geometry[k] for k in ('triangles', 'size_bytes', 'bounds_m')})))
    if (folder / 'navigation.json').exists() and json.loads((folder / 'navigation.json').read_text())['inputs'] == inputs:
        return
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(OUT / 'scenes' / scene_id / 'terrain.npz') as data:
        bounds = data['triangles'][..., :2].reshape(-1, 2)
        origin = np.floor(bounds.min(0) / GRID_M) * GRID_M
        cells = np.ceil((bounds.max(0) - origin) / GRID_M).astype(int)
    city = dict(ground=np.full(cells, np.nan, np.float32), covered=np.zeros(cells, bool),
                **({} if ROBOT.body_half_extent_m is None else dict(body=np.zeros(cells, bool))))
    tiles = [(a, b) for a in range(0, cells[0], TILE_CELLS) for b in range(0, cells[1], TILE_CELLS)]
    with ProcessPoolExecutor(workers, initializer=load_city, initargs=(scene_id,)) as pool:
        for n, future in enumerate(as_completed([pool.submit(_city_tile, origin, a, b, cells) for a, b in tiles]), 1):
            a, b, layers = future.result()
            for name, layer in layers.items():
                city[name][a:a + layer.shape[0], b:b + layer.shape[1]] = layer
            print(f'tile {n}/{len(tiles)}', flush=True)
    ground_z, body_free = city['ground'], city.get('body')
    episodes = json.loads(scene_files(scene_id)[1].read_text())
    walked = np.concatenate([np.asarray(g['reference_path'])[:, :2] for e in episodes for g in e['goals']])
    free, info = known_ground(np.isfinite(ground_z), origin, walked, spacing=GRID_M)
    inside = indoor(free, city['covered'])
    free &= ~inside
    info.update(indoor_dropped_m2=round(float(inside.sum() * GRID_M ** 2), 1))
    if not free.any():
        raise ValueError(f'no navigable ground for {ROBOT.name} in {scene_id}')
    ground_z[~free] = np.nan
    save_navigation_grid(folder / 'navigation.npz', free, origin, GRID_M, ground_z=ground_z, body_free=body_free)
    info.update(inputs=inputs, navigation_shape=[int(c) for c in cells], tiles=len(tiles),
                navigation_free_area_m2=round(float(free.sum() * GRID_M ** 2), 1),
                **({} if body_free is None else dict(body_ground_area_m2=round(float(body_free.sum() * GRID_M ** 2), 1))))
    (folder / 'navigation.json').write_text(json.dumps(info, indent=2) + '\n')


# ---------------------------------------------------------------- tasks
class City:
    """The city C-space and the shortest paths on it."""
    def __init__(self, scene_id):
        from scipy import ndimage
        from kappanav.common.geometry import Grid
        self.scene_id, self.folder = scene_id, OUT / 'scenes' / scene_id / ROBOT.name
        self.grid = Grid.load(self.folder / 'navigation.npz')
        self.components = self.grid.components
        self.boxes = ndimage.find_objects(self.components)
        self.cost = np.where(self.grid.free, np.float32(1), np.float32(np.inf))  # route.line_of_sight's map
        self.scene = dict(scene_id=scene_id, source='navverse', domain='outdoor', assets=str(ASSETS),
                          asset_scene_id=scene_id, scene_dataset=SCENE_DATASET, navigation=str(self.folder / 'navigation.npz'))

    def distances(self, start, targets=None, limit=np.inf):
        """Shortest-path lengths from cell `start` within its connected component: (field, the component's cells, their
        offset in the city grid). With `targets` (cells) the search stops once they are settled; cells farther than
        `limit` are not expanded (their values are not final)."""
        from kappanav.common.grid_dijkstra import grid_distances
        label = self.components[tuple(start)]
        box = self.boxes[label - 1]
        offset = np.array([box[0].start, box[1].start])
        free = np.ascontiguousarray(self.components[box] == label)
        local = None if targets is None else np.asarray(targets) - offset
        return grid_distances(free, GRID_M, int(start[0] - offset[0]), int(start[1] - offset[1]), local, limit), free, offset

    def task(self, name, start, yaw, goal, extra, distances=None):
        """The episode of the shortest-path task between cells `start` and `goal` (one component), with its C-space
        shortest path. `distances`: self.distances(start) if already known."""
        from kappanav.common.grid_dijkstra import descend
        from kappanav.global_planner.route import cell_polyline
        from kappanav.evaluation.simulation.realtime import ARRIVAL_HOLD_S
        field, free, offset = distances or self.distances(start, [goal])
        local = goal - offset
        length = float(field[tuple(local)])
        cells = descend(field, free, GRID_M, int(local[0]), int(local[1]))[::-1] + offset  # start to goal
        start_xy, goal_xy = self.grid.grid_to_world(start), self.grid.grid_to_world(goal)
        path = cell_polyline(cells, self.cost, self.grid.origin_xy, GRID_M, start_xy, goal_xy)
        return dict(episode_id=f'{self.scene_id}/{name}', scene_id=self.scene_id, start=[*start_xy.tolist(), float(yaw)],
                    goal=goal_xy.tolist(), shortest_path_m=length, spl_distance_m=float(np.linalg.norm(goal_xy - start_xy)),
                    shortest_path=path.round(3).tolist(), goal_budget_s=max(122., 6 * length + 30), collision_ends=False,
                    arrival_hold_s=ARRIVAL_HOLD_S, **extra)


def official_tasks(city):
    """The official episodes for the selected robot: (episodes, rejected)."""
    from kappanav.common.geometry import PlanningError
    episodes, rejected = [], []
    for e in json.loads(scene_files(city.scene_id)[1].read_text()):
        target = e['goals'][e.get('closest_goal_idx', 0)]
        try:
            start, goal = (city.grid.snap(np.asarray(p[:2]), SNAP_MAX_XY_M) for p in (e['start_position'], target['location']))
            if city.components[tuple(start)] != city.components[tuple(goal)]:
                raise PlanningError('endpoints disconnected for selected robot')
        except PlanningError as error:
            rejected.append(dict(episode_id=e['episode_id'], reason=str(error)))
            continue
        episodes.append(city.task(f'goal_{e["episode_id"]:03d}', start, heading(e['start_rotation']), goal, dict(official=e)))
    return episodes, rejected


_TASK_CITY = None  # the City the long-task workers inherit


def long_task(low, high, index, seed):
    """Long task `index` of the range [low, high) m: a start uniform over the free cells with ENDPOINT_CLEARANCE_M,
    its heading uniform, and a goal uniform over the cells of its component at that shortest-path distance. Its
    random stream is (seed, low, index), so the task does not depend on the other tasks or the worker count."""
    city = _TASK_CITY
    rng = np.random.default_rng((seed, int(low), index))
    shape = np.array(city.grid.free.shape)
    for _ in range(LONG_ATTEMPTS):
        start = np.array(np.unravel_index(rng.integers(shape.prod()), shape))
        if not (city.grid.free[tuple(start)] and city.grid.clearance_m[tuple(start)] >= ENDPOINT_CLEARANCE_M):
            continue
        field, free, offset = known = city.distances(start, limit=high)
        window = tuple(slice(o, o + n) for o, n in zip(offset, field.shape))
        goals = np.argwhere(free & (city.grid.clearance_m[window] >= ENDPOINT_CLEARANCE_M) & (field >= low) & (field < high))
        if len(goals):
            return city.task(f'{int(low):03d}m_{index:04d}', start, rng.uniform(-np.pi, np.pi),
                             goals[rng.integers(len(goals))] + offset, dict(range_m=[low, high]), known)
    raise RuntimeError(f'{city.scene_id}: no {low:g}-{high:g} m task in {LONG_ATTEMPTS} starts')


def long_tasks(city, count, ranges, workers, seed=0):
    """`count` long tasks split evenly over the shortest-path `ranges` (lowest first), in task order."""
    global _TASK_CITY
    import multiprocessing
    _TASK_CITY = city
    jobs = [(low, high, i, seed) for k, (low, high) in enumerate(ranges)
            for i in range(count // len(ranges) + (k < count % len(ranges)))]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(long_task, *job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            if n % 50 == 0 or n == len(futures):
                print(f'long tasks {n}/{len(futures)}', flush=True)
        return [future.result() for future in futures]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', default='vcp_amsterdam', help='vcp_<city> or innout_<city>')
    ap.add_argument('--export', action='store_true', help='(re)export the city geometry first')
    ap.add_argument('--long', type=int, default=0, metavar='N', help='also sample N long tasks, split evenly over the --edges bins')
    ap.add_argument('--edges', type=float, nargs='+', default=list(LONG_EDGES_M), metavar='M',
                    help='shortest-path bin edges of the long tasks (default %(default)s)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8, help='navmesh tile and long-task processes')
    args = ap.parse_args()
    if args.export:
        print(json.dumps(export_scene(args.scene)), flush=True)
    build_navigation(args.scene, args.workers)
    city = City(args.scene)
    catalog = dict(robot=ROBOT.name, scenes=[city.scene])
    episodes, rejected = official_tasks(city)
    (OUT / f'{args.scene}_{ROBOT.name}_tasks.json').write_text(json.dumps(dict(
        catalog, protocol=f'navverse_{ROBOT.name}_pointgoal', episodes=episodes, rejected=rejected), indent=1) + '\n')
    print(json.dumps(dict(robot=ROBOT.name, scene=args.scene, valid=len(episodes), rejected=len(rejected))), flush=True)
    if args.long:
        ranges = list(zip(args.edges[:-1], args.edges[1:]))
        episodes = long_tasks(city, args.long, ranges, args.workers, args.seed)
        (OUT / f'{args.scene}_{ROBOT.name}_long_tasks.json').write_text(json.dumps(dict(
            catalog, protocol=f'navverse_{ROBOT.name}_pointgoal_long', ranges_m=ranges, seed=args.seed,
            episodes=episodes), indent=1) + '\n')
        print(json.dumps(dict(long_tasks=len(episodes))), flush=True)


if __name__ == '__main__':
    main()
