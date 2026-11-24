"""Turn raw Wanderland v1 scenes into CurveNav navigation scenes: one scene geometry, a navigation layer per robot.

Frames (all metric, right-handed):
  dataset  : Wanderland files (raw_pcd.ply, scene.usdz collider, episodes.json). +Y points DOWN
             (COLMAP); episode start/goal positions are rig-camera poses about 1.2 m above ground.
  isaac    : wanderland-lab's world, Z-up: isaac = (x, z, -y).
  habitat  : Y-up glTF for Habitat/CurveNav: habitat = (x, -y, -z), a 180 deg turn about X.
  canonical: CurveNav's planar frame (X_h, -Z_h) = (x, z) = isaac XY. Every 2D output here uses it.

Per scene, robot-independent (rebuilt when the raw inputs or the geometry code change):
  assets/stages/<id>.glb (+ stage/scene configs)  collision mesh REBUILT from raw LiDAR (collider.py), closed by
                                                  boundary walls where the scan ends
  scenes/<id>/maps.npz                            LiDAR ground height, top-down RGB "aerial" (DOM), DSM
  scenes/<id>/scene.json, preview.jpg             metadata and a visual check
Per scene and robot (KAPPANAV_ROBOT), in scenes/<id>/<robot>/:
  scene.navmesh                                   the robot's navmesh (radius, height, step, slope)
  navigation.npz                                  CurveNav C-space grid
  episodes.json                                   official episodes in canonical frame, validated for the robot
  navigation.json                                 navigation metadata
Run with the CurveNav python (habitat_sim) and one free GPU: scripts/evaluation/prepare_wanderland.sh.
"""
import argparse, csv, hashlib, json, math, os, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from kappanav.evaluation.simulation.assets import SCENE_DATASET, write_stage
from kappanav.evaluation.simulation.navigation import (BODY_GROUND_RADIUS_M, navmesh_settings, navmesh_triangles,
                                                       rasterize_navigation)
from kappanav.common.physical import ROBOT
from kappanav.evaluation.simulation.wanderland import collider
from kappanav.evaluation.simulation.wanderland.usd_collider import read_collider

ROOT = Path(os.environ.get('KAPPANAV_WANDERLAND', 'data/wanderland'))   # raw/ and the public manifest
RAW = ROOT / 'raw'
OUT = Path(os.environ.get('KAPPANAV_WANDERLAND_SCENES', ROOT / 'processed'))
ASSETS = OUT / 'assets'

GRID_M = 0.05          # CurveNav navigation grid
MAP_M = 0.10           # top-down aerial / DSM
SURFACE_RANK = 3       # DSM = 3rd highest point of a cell: single floaters above a surface are not the surface
SURFACE_LAYER_M = 0.25 # DOM colour = mean of the points this far below the surface
FILL_GAP_M = 0.3       # DOM / DSM fill gaps between observed cells up to this width
MIN_POINTS = 2         # a cell with a single LiDAR return is a gap, not a surface
GROUND_CELL_M = collider.GROUND_M
ON_GROUND_M = 0.10     # navmesh within this height of the LiDAR ground counts as ground
FRAME_CHECK_MAX_M = 0.15  # released collider (local points) must lie on the LiDAR cloud
ENDPOINT_HEIGHT_M = (-0.3, 2.5)  # episode endpoints (rig camera) above the LiDAR ground below them
CAMERA_HEIGHT_M = (0.8, 2.0)  # median capture-camera height above the LiDAR ground
WALKED_RADIUS_M = 0.5    # ground within this distance of a capture camera was walked by the operator
ISLAND_MIN_M2 = 20.0     # unwalked free components smaller than this are object tops (car roofs, boxes)
SNAP_MAX_XY_M = 0.5    # episode endpoints must snap this close to the navmesh
CANONICAL_TO_HABITAT = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # (X, Z, -Y)


def dataset_to_canonical(p):
    p = np.asarray(p, np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)  # isaac (x, z, -y)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def read_ply_xyzrgb(path, stride=1):
    with open(path, 'rb') as f:
        props, n = [], 0
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property') and 'list' not in line:
                props.append(line.split()[1:])
            elif line == 'end_header':
                break
        off = f.tell()
    types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4'}
    dt = np.dtype([(name, '<' + types[t]) for t, name in props])
    v = np.memmap(path, dtype=dt, mode='r', offset=off, shape=(n,))[::stride]
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    rgb = np.stack([v['red'], v['green'], v['blue']], 1).astype(np.uint8)
    return xyz, rgb


# ---------------------------------------------------------------- stage assets
def export_stage(scene_id, verts, faces):
    """Write the rebuilt canonical mesh as a Habitat stage."""
    import trimesh
    if not np.isfinite(verts).all():
        raise ValueError('nonfinite rebuilt vertices')
    mesh = trimesh.Trimesh(verts @ CANONICAL_TO_HABITAT.T, faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(doubleSided=True))
    (ASSETS / 'stages').mkdir(parents=True, exist_ok=True)
    glb = ASSETS / 'stages' / f'{scene_id}.glb'
    tmp = glb.with_suffix('.glb.partial')
    tmp.write_bytes(mesh.export(file_type='glb', include_normals=False))
    os.replace(tmp, glb)
    write_stage(ASSETS, scene_id)
    return dict(collider_vertices=int(len(verts)), collider_triangles=int(len(faces)),
                collider_bounds_canonical=[verts.min(0).round(3).tolist(), verts.max(0).round(3).tolist()])


# ---------------------------------------------------------------- navigation
def build_navigation(scene_id, out_dir, gpu, ground, cameras_xy):
    """The robot's navmesh of the rebuilt scene, rasterized to the CurveNav grid on the LiDAR ground; ground the
    capture operator walked is known ground (navigation.rasterize_navigation). A rectangular robot also gets its
    body ground (navmesh with BODY_GROUND_RADIUS_M) on the same grid."""
    import habitat_sim
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_dataset_config_file = str(ASSETS / SCENE_DATASET)
    sc.scene_id = str(ASSETS / 'scenes' / f'{scene_id}.scene_instance.json')
    sc.gpu_device_id, sc.enable_physics = gpu, False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [agent]))
    try:
        if not sim.recompute_navmesh(sim.pathfinder, navmesh_settings()):
            raise ValueError('navmesh build failed')
        sim.pathfinder.save_nav_mesh(str(out_dir / 'scene.navmesh'))
        triangles = navmesh_triangles(sim.pathfinder)
        stats = dict(navmesh_area_m2=round(float(sim.pathfinder.navigable_area), 1),
                     navmesh_islands=int(sim.pathfinder.num_islands))
        body = None
        if ROBOT.body_half_extent_m is not None:
            if not sim.recompute_navmesh(sim.pathfinder, navmesh_settings(BODY_GROUND_RADIUS_M)):
                raise ValueError('body ground navmesh build failed')
            body = navmesh_triangles(sim.pathfinder)
        nav, info = rasterize_navigation(triangles, out_dir, ground, cameras_xy, body_triangles=body, spacing=GRID_M,
                                         ground_tolerance=ON_GROUND_M, walked_radius=WALKED_RADIUS_M,
                                         island_min_m2=ISLAND_MIN_M2)
        return nav, dict(stats, **info)
    finally:
        sim.close()


# ---------------------------------------------------------------- episodes
def convert_episodes(raw_dir, ground, navigation_path):
    from kappanav.common.geometry import Grid, PlanningError
    from kappanav.common.grid_dijkstra import grid_distances
    from kappanav.common.physical import TURN_CENTRE_FORWARD_OFFSET_M
    grid = Grid.load(navigation_path)
    # Enough room for base_link to swing around the turn centre during an in-place turn.
    starts = Grid(grid.free & (grid.clearance_m >= 2 * TURN_CENTRE_FORWARD_OFFSET_M),
                  grid.clearance_m, grid.origin_xy, grid.cell_size_m)
    raw = json.loads((raw_dir / 'episodes.json').read_text())['episodes']
    out = []
    for e in raw:
        rec = dict(episode_id=e['episode_id'], dataset_geodesic_m=e.get('geodesic_distance'),
                   goal_radius_m=e['goals'][0].get('radius'), instruction=e.get('instruction', ''))
        ends, reasons = {}, []
        for key, p in (('start', e['start_position']), ('goal', e['goals'][0]['position'])):
            point = dataset_to_canonical(p)
            rec[f'{key}_original_xyz'] = point.tolist()
            gz = float(ground(point[None, :2])[0])
            height = float(point[2] - gz)
            rec[f'{key}_camera_height_m'] = height if math.isfinite(height) else None
            if not math.isfinite(height) or not ENDPOINT_HEIGHT_M[0] <= height <= ENDPOINT_HEIGHT_M[1]:
                reasons.append(f'{key}_not_ground_level')
            try:
                cell = (starts if key == 'start' else grid).snap(point[:2], SNAP_MAX_XY_M)
            except PlanningError:
                reasons.append(f'{key}_off_navigation_grid')
                rec[f'{key}_xy'] = point[:2].tolist()
                rec[f'{key}_ground_z'] = gz if math.isfinite(gz) else None
                rec[f'{key}_snap_xy_m'] = None
                continue
            xy = grid.grid_to_world(cell)
            ends[key] = cell
            rec[f'{key}_xy'] = xy.tolist()
            rec[f'{key}_ground_z'] = float(ground(xy[None])[0])
            rec[f'{key}_snap_xy_m'] = float(np.linalg.norm(xy - point[:2]))
            if not math.isfinite(rec[f'{key}_ground_z']):
                reasons.append(f'{key}_off_ground')
        q = e.get('start_rotation')
        if q is not None:
            x, y, z, w = q
            n = math.sqrt(x*x + y*y + z*z + w*w); x, y, z, w = x/n, y/n, z/n, w/n
            fwd = np.array([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)])
            fc = dataset_to_canonical(fwd)
            rec['start_heading_rad'] = float(math.atan2(fc[1], fc[0]))
        ref = e.get('reference_path') or []
        rec['reference_path_xy'] = dataset_to_canonical(ref)[:, :2].tolist() if ref else []
        distance = None
        if not reasons:
            start, goal = ends['start'], ends['goal']
            if grid.components[tuple(start)] == grid.components[tuple(goal)]:
                d = grid_distances(grid.free, grid.cell_size_m, int(goal[0]), int(goal[1]), np.array([start]))
                distance = float(d[tuple(start)])
            if distance is None or not math.isfinite(distance):
                reasons.append('no_path')
                distance = None
        rec['geodesic_m'] = distance
        rec['valid'] = not reasons
        rec['invalid_reasons'] = reasons
        out.append(rec)
    return out


# ---------------------------------------------------------------- top-down maps
def topdown_maps(c, rgb, ground, ground_origin):
    """Aerial-like products on a MAP_M canonical grid covering the whole LiDAR ground grid, indexed [x, y]:
    dsm_z, the visible top surface: the SURFACE_RANK-th highest point of a cell with at least MIN_POINTS, so isolated floaters above a
    surface do not become the surface; dom_rgb, the mean colour of the points within SURFACE_LAYER_M below it; both
    filled across gaps of at most FILL_GAP_M between observed cells (sparse far-range LiDAR), dom_valid marks where
    they hold. ground_rgb is the colour of the lowest point, points_per_cell the raw count, covered whether the
    surface is >= 2.2 m above the LiDAR ground (canopy, roofs, ceilings).
    The LiDAR was coloured from ground-level cameras, so canopy and roof tops carry what those cameras saw (often
    sky): dom_rgb is a stand-in for an aerial image where the ground is open, not under cover."""
    from scipy.ndimage import distance_transform_edt
    ox, oy = ground_origin
    W = int(math.ceil(ground.shape[0] * GROUND_CELL_M / MAP_M)); H = int(math.ceil(ground.shape[1] * GROUND_CELL_M / MAP_M))
    i = np.floor((c[:, 0] - ox) / MAP_M).astype(np.int64); j = np.floor((c[:, 1] - oy) / MAP_M).astype(np.int64)
    k = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    lin, z, col = i[k] * H + j[k], c[k, 2], rgb[k]
    order = np.lexsort((-z, lin))                      # by cell, highest first
    lin, z, col = lin[order], z[order], col[order]
    count = np.bincount(lin, minlength=W * H).astype(np.int32)
    has = count >= MIN_POINTS
    first = np.concatenate([[0], np.cumsum(count)[:-1]])
    surface = np.full(W * H, np.nan); surface[has] = z[first[has] + np.minimum(SURFACE_RANK, count[has]) - 1]
    lowest = first + count - 1
    under = np.full((W * H, 3), 255, np.uint8); under[has] = col[lowest[has]]
    layer = (z <= surface[lin]) & (z >= surface[lin] - SURFACE_LAYER_M)
    n = np.bincount(lin[layer], minlength=W * H)
    dom = np.stack([np.bincount(lin[layer], col[layer, a].astype(np.float64), W * H) for a in range(3)], 1)
    dom = np.rint(dom / np.maximum(n, 1)[:, None]).astype(np.uint8)
    dom, surface, has = dom.reshape(W, H, 3), surface.reshape(W, H), has.reshape(W, H)
    gap, (ni, nj) = distance_transform_edt(~has, return_indices=True)
    valid = gap * MAP_M <= FILL_GAP_M
    dom = np.where(valid[..., None], dom[ni, nj], 255).astype(np.uint8)
    dsm = np.where(valid, surface[ni, nj], np.nan).astype(np.float32)
    r = int(round(MAP_M / GROUND_CELL_M))
    gz = ground[np.clip(np.arange(W) * r, 0, ground.shape[0] - 1)][:, np.clip(np.arange(H) * r, 0, ground.shape[1] - 1)]
    covered = valid & np.isfinite(gz) & (np.nan_to_num(dsm, nan=-np.inf) > gz + 2.2)
    return dict(dom_rgb=dom, dom_valid=valid, ground_rgb=under.reshape(W, H, 3), dsm_z=dsm,
                points_per_cell=count.reshape(W, H), covered=covered,
                map_origin_xy=np.array([ox, oy]), map_cell_m=MAP_M), int(k.sum())


def free_on_map(maps, nav):
    """Navigation free cells resampled onto the map grid (map cell = free if its centre's nav cell is free)."""
    W, H = maps['dom_rgb'].shape[:2]
    cx = maps['map_origin_xy'][0] + (np.arange(W) + .5) * MAP_M
    cy = maps['map_origin_xy'][1] + (np.arange(H) + .5) * MAP_M
    ni = np.floor((cx - nav['origin_xy'][0]) / GRID_M).astype(int)
    nj = np.floor((cy - nav['origin_xy'][1]) / GRID_M).astype(int)
    vi, vj = (ni >= 0) & (ni < nav['free'].shape[0]), (nj >= 0) & (nj < nav['free'].shape[1])
    out = np.zeros((W, H), bool)
    out[np.ix_(vi, vj)] = nav['free'][np.ix_(ni[vi], nj[vj])]
    return out


def preview(maps, free, episodes, path):
    from PIL import Image, ImageDraw
    base = maps['dom_rgb'].transpose(1, 0, 2)[::-1].astype(np.float32)  # rows = +Y up the page
    W, H = maps['dom_rgb'].shape[:2]
    m = free.T[::-1]
    base[m] = base[m] * .6 + np.array([0, 255, 0]) * .4
    im = Image.fromarray(base.astype(np.uint8))
    d = ImageDraw.Draw(im)
    ox, oy = maps['map_origin_xy']
    px = lambda p: ((p[0] - ox) / MAP_M, H - (p[1] - oy) / MAP_M)
    for e in episodes:
        c = (255, 140, 0) if e['valid'] else (220, 0, 220)
        if len(e['reference_path_xy']) > 1:
            d.line([px(p) for p in e['reference_path_xy']], fill=c, width=2)
        s, g = px(e['start_xy']), px(e['goal_xy'])
        d.ellipse([s[0]-4, s[1]-4, s[0]+4, s[1]+4], fill=(0, 150, 255))
        d.ellipse([g[0]-4, g[1]-4, g[0]+4, g[1]+4], fill=(230, 0, 0))
    d.rectangle([8, im.height - 14, 8 + 10 / MAP_M, im.height - 9], fill=(0, 0, 0))
    s = min(1.0, 1200 / max(im.size))
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    im.save(path, quality=85)


def frame_check(raw_dir, P):
    """Median distance from the released collider's local points to the canonical LiDAR cloud."""
    from scipy.spatial import cKDTree
    verts, _, info = read_collider(raw_dir / 'scene.usdz')
    if info['meters_per_unit'] != 1.0:
        raise ValueError(f'unexpected metersPerUnit {info}')
    q = dataset_to_canonical(verts[:: max(1, len(verts) // 20000)])
    return round(float(np.median(cKDTree(P[::5]).query(q)[0])), 3)


def read_camera_centers(raw_dir):
    """COLMAP camera centres (sparse/0/images.bin) in the canonical frame: the capture trajectory."""
    import struct
    centres = []
    with open(raw_dir / 'sparse' / '0' / 'images.bin', 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack('<idddddddi', f.read(64))
            while f.read(1) != b'\0':
                pass
            f.seek(24 * struct.unpack('<Q', f.read(8))[0], 1)
            R = np.array([[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                          [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                          [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]])
            centres.append(-R.T @ np.array([tx, ty, tz]))
    return dataset_to_canonical(np.array(centres))


def endpoint_check(raw_dir, ground):
    """Fraction of episode start/goal poses whose height above the LiDAR ground below them is plausible."""
    eps = json.loads((raw_dir / 'episodes.json').read_text())['episodes']
    pts = dataset_to_canonical([e['start_position'] for e in eps] + [e['goals'][0]['position'] for e in eps])
    h = pts[:, 2] - ground(pts[:, :2])
    ok = np.isfinite(h) & (h >= ENDPOINT_HEIGHT_M[0]) & (h <= ENDPOINT_HEIGHT_M[1])
    return round(float(ok.mean()), 3), [round(float(x), 2) for x in np.nanpercentile(h, [10, 50, 90])]


# ---------------------------------------------------------------- driver
REQUIRED = ('episodes.json', 'raw_pcd.ply', 'sparse/0/images.bin')


def geometry_pipeline():
    """Hash of the code the robot-independent geometry layer is built from."""
    from kappanav.common.experiment_io import code_digest
    from kappanav.evaluation.simulation import assets
    from kappanav.evaluation.simulation.wanderland import usd_collider
    return code_digest(build_geometry, read_ply_xyzrgb, dataset_to_canonical, frame_check, export_stage, topdown_maps,
                       assets.write_stage, collider, usd_collider,
                       repr((MAP_M, SURFACE_RANK, SURFACE_LAYER_M, FILL_GAP_M, MIN_POINTS, FRAME_CHECK_MAX_M)))


def navigation_pipeline():
    """Hash of the code a robot's navigation layer and tasks are built from."""
    from kappanav.common.geometry import Grid
    from kappanav.common.experiment_io import code_digest
    from kappanav.evaluation.simulation import navigation as n
    return code_digest(process, build_navigation, convert_episodes, read_camera_centers, n.navmesh_settings,
                       n.navmesh_triangles, n.rasterize_navigation, n.navmesh_ground, n.known_ground, n._raster_surface,
                       n.save_navigation_grid, Grid, repr((GRID_M, ON_GROUND_M, WALKED_RADIUS_M, ISLAND_MIN_M2, SNAP_MAX_XY_M,
                                                           ENDPOINT_HEIGHT_M, BODY_GROUND_RADIUS_M)))


def build_geometry(sid, raw_dir, out_dir, inputs, meta):
    """Collision mesh, stage assets and top-down maps of one scene; returns the (updated) metadata."""
    t0 = time.time()
    xyz, rgb = read_ply_xyzrgb(raw_dir / 'raw_pcd.ply')
    P = dataset_to_canonical(xyz)
    del xyz
    if (raw_dir / 'scene.usdz').exists():
        frame_m = frame_check(raw_dir, P)
        meta['frame_check_collider_to_lidar_m'] = frame_m
        if frame_m > FRAME_CHECK_MAX_M:
            raise ValueError(f'released collider and raw_pcd disagree by {frame_m} m: frames differ')
    verts, faces, ground, ground_origin, rebuild_info = collider.rebuild(P, rgb)
    meta.update(rebuild_info)
    meta.update(export_stage(sid, verts, faces))
    del verts, faces
    maps, npts = topdown_maps(P, rgb, ground, ground_origin)
    np.savez_compressed(out_dir / 'maps.npz', lidar_ground_z=ground, lidar_ground_origin_xy=ground_origin,
                        lidar_ground_cell_m=collider.GROUND_M, **maps)
    meta.update(points=npts, extent_m=[round(maps['dom_rgb'].shape[0] * MAP_M, 1), round(maps['dom_rgb'].shape[1] * MAP_M, 1)],
                geometry_inputs=inputs, geometry_seconds=round(time.time() - t0, 1))
    return meta


def process(row, gpu, force=False):
    sid, raw_dir = row['scene_id'], RAW / row['data_path']
    out_dir = OUT / 'scenes' / sid
    nav_dir = out_dir / ROBOT.name
    meta_path, nav_meta_path = out_dir / 'scene.json', nav_dir / 'navigation.json'
    missing = [f for f in REQUIRED if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f'missing required files: {missing}')
    inputs = {f: sha256(raw_dir / f) for f in ('scene.usdz', 'episodes.json') if (raw_dir / f).exists()}
    inputs['raw_pcd.ply_size'] = (raw_dir / 'raw_pcd.ply').stat().st_size
    inputs['pipeline'] = geometry_pipeline()
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fresh = force or meta.get('geometry_inputs') != inputs
    # what the navigation layer depends on: the geometry, its code and the robot's body, height, step, slope, turn centre
    nav_inputs = json.loads(json.dumps(dict(geometry=inputs, robot=ROBOT.name, pipeline=navigation_pipeline(), **{
        k: getattr(ROBOT, k) for k in ('footprint_radius_m', 'body_half_extent_m', 'collision_bottom_z_m', 'collision_top_z_m',
                                       'maximum_step_m', 'maximum_slope_degrees', 'turn_centre_forward_m')})))
    if not fresh and nav_meta_path.exists() and json.loads(nav_meta_path.read_text()).get('inputs') == nav_inputs:
        return sid, 'up-to-date'
    out_dir.mkdir(parents=True, exist_ok=True)
    nav_dir.mkdir(exist_ok=True)
    if fresh:
        meta = build_geometry(sid, raw_dir, out_dir, inputs, dict(
            scene_id=sid, release=row['release_version'], quality_tier=row['quality_tier'],
            quality_tags=[t for t in row['quality_tags'].split('|') if t],
            frames='canonical XY = isaac XY = (x_dataset, z_dataset); z_up = -y_dataset'))
    t0 = time.time()
    with np.load(out_dir / 'maps.npz') as data:
        maps = dict(data)
    from scipy.interpolate import RegularGridInterpolator
    gaxis = [maps['lidar_ground_origin_xy'][a] + (np.arange(maps['lidar_ground_z'].shape[a]) + .5) * collider.GROUND_M
             for a in (0, 1)]
    ground = RegularGridInterpolator(gaxis, maps['lidar_ground_z'], bounds_error=False, fill_value=np.nan)
    if fresh:
        ok_frac, heights = endpoint_check(raw_dir, ground)
        meta.update(frame_check_endpoint_ok_fraction=ok_frac, endpoint_height_above_ground_p10_50_90=heights)
    cameras = read_camera_centers(raw_dir)
    if fresh:
        cam_h = cameras[:, 2] - ground(cameras[:, :2])
        cam_med = float(np.nanmedian(cam_h)) if np.isfinite(cam_h).any() else float('nan')
        meta.update(capture_cameras=int(len(cameras)), camera_height_above_ground_median_m=round(cam_med, 3))
        if not CAMERA_HEIGHT_M[0] <= cam_med <= CAMERA_HEIGHT_M[1]:
            raise ValueError(f'capture cameras sit {cam_med:.2f} m above the LiDAR ground: frames differ')
        # Camera centres and task endpoints share COLMAP's measured frame. Elevated endpoints
        # (stairs/upper floors) are rejected per task, not mistaken for an entire misaligned scene.
        from scipy.spatial import cKDTree
        raw_eps = json.loads((raw_dir / 'episodes.json').read_text())['episodes']
        endpoints = dataset_to_canonical([e['start_position'] for e in raw_eps] + [e['goals'][0]['position'] for e in raw_eps])
        frame_distance = float(np.median(cKDTree(cameras).query(endpoints)[0]))
        meta['frame_check_endpoint_to_camera_m'] = frame_distance
        if frame_distance > FRAME_CHECK_MAX_M:
            raise ValueError(f'episode endpoints and capture cameras disagree by {frame_distance:.3f} m')
    nav, navinfo = build_navigation(sid, nav_dir, gpu, ground, cameras[:, :2])
    episodes = convert_episodes(raw_dir, ground, nav_dir / 'navigation.npz')
    free_map = free_on_map(maps, nav)
    (nav_dir / 'episodes.json').write_text(json.dumps(episodes, indent=1, ensure_ascii=False))
    cov = float(maps['covered'][free_map].mean()) if free_map.any() else None
    valid = [e for e in episodes if e['valid']]
    navinfo.update(inputs=nav_inputs, covered_fraction_of_free=round(cov, 3) if cov is not None else None,
                   indoor_guess=bool(cov is not None and cov > 0.9),
                   episodes=len(episodes), episodes_valid=len(valid),
                   invalid_reasons=dict(sorted({r: sum(r in e['invalid_reasons'] for e in episodes)
                                                for e in episodes for r in e['invalid_reasons']}.items())),
                   geodesic_m=[min(e['geodesic_m'] for e in valid), max(e['geodesic_m'] for e in valid)] if valid else None,
                   seconds=round(time.time() - t0, 1))
    preview(maps, free_map, episodes, nav_dir / 'preview.jpg')
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    nav_meta_path.write_text(json.dumps(navinfo, indent=1, ensure_ascii=False))
    return sid, 'ok'


def write_index():
    rows = []
    for p in sorted((OUT / 'scenes').glob(f'*/{ROBOT.name}/navigation.json')):
        m, n = json.loads((p.parent.parent / 'scene.json').read_text()), json.loads(p.read_text())
        rows.append(dict({k: m.get(k) for k in ('scene_id', 'quality_tier', 'quality_tags', 'extent_m')},
                         **{k: n.get(k) for k in ('navmesh_area_m2', 'navmesh_islands', 'covered_fraction_of_free',
                                                  'indoor_guess', 'episodes', 'episodes_valid', 'invalid_reasons',
                                                  'geodesic_m')}))
    with open(OUT / f'index_{ROBOT.name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['scene_id'])
        w.writeheader()
        for r in rows:
            w.writerow({k: ('|'.join(v) if isinstance(v, list) and v and isinstance(v[0], str) else
                            json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    return len(rows)


def _work(row, gpu, force):
    try:
        return process(row, gpu, force)
    except Exception as exc:  # report and continue with the other scenes
        return row['scene_id'], 'FAILED ' + ''.join(traceback.format_exception_only(type(exc), exc)).strip()[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scenes', nargs='*', help='scene ids (default: every v1 scene in the manifest)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0, help='index within CUDA_VISIBLE_DEVICES')
    ap.add_argument('--force', action='store_true', help='rebuild the scene geometry even if its inputs are unchanged')
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(ROOT / 'wanderland_public_manifest.csv')) if r['release_version'] == 'v1']
    if a.scenes:
        rows = [r for r in rows if r['scene_id'] in set(a.scenes)]
    (OUT / 'scenes').mkdir(parents=True, exist_ok=True)
    print(f'{len(rows)} scenes, robot {ROBOT.name}, output {OUT}', flush=True)
    failures = []
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(_work, r, a.gpu, a.force) for r in rows]
        for i, f in enumerate(as_completed(futs), 1):
            sid, status = f.result()
            print(f'[{i}/{len(rows)}] {sid} {status}', flush=True)
            if status.startswith('FAILED'): failures.append(dict(scene_id=sid, error=status))
    print('index rows', write_index(), flush=True)
    (OUT / f'prepare_status_{ROBOT.name}.json').write_text(json.dumps(dict(scenes=len(rows), failed=failures), indent=2))
    if failures: raise SystemExit(1)


if __name__ == '__main__':
    main()
