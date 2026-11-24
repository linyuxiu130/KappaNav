"""NavVerse cities seen straight down: the top views evaluation.topview.maps reads.

A city's appearance asset (simulation.navverse --export) is rendered with an orthographic camera at BASE_GSD_M, one
TILE_CELLS square at a time, aligned with the robot's navigation grid. A tile is OUT/navverse/<city>/<x>_<y>.npz with
rgb, traversable (the selected robot's C-space: 1 free, 0 blocked, 255 outside it), gsd_m and origin_xy, indexed [x, y]
like every CurveNav grid.

    python -m kappanav.evaluation.topview.navverse_aerial --scene vcp_amsterdam
"""
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from kappanav.common.physical import ROBOT
from kappanav.evaluation.simulation import navverse
from kappanav.evaluation.simulation.assets import SCENE_DATASET

OUT = Path(os.environ.get('KAPPANAV_AERIAL', '/shibo_huang/data/topview'))
BASE_GSD_M = .05
TILE_CELLS = 1280                # 64 m at BASE_GSD_M
SUPERSAMPLE = 2                  # rendered at this many pixels per tile pixel and area-averaged: sharper than one
                                 # mip-filtered sample per pixel (texture minification at nadir)


def city_triangles(scene_id):
    """Every rendered triangle of the city in canonical (x, y, z), with its faces' XY bounds."""
    import trimesh
    scene = trimesh.load(navverse.ASSETS / 'stages' / f'{scene_id}.glb', force='scene', process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        matrix, key = scene.graph[node]
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(matrix)
        parts.append(mesh.vertices[mesh.faces][..., [0, 2, 1]] * [1., -1., 1.])  # habitat (X, Z, -Y)
    triangles = np.concatenate(parts)
    return triangles, triangles[..., :2].min(1), triangles[..., :2].max(1)


class OrthographicCamera:
    """A colour camera looking straight down at the city stage, one TILE_CELLS square at BASE_GSD_M per render.
    Habitat's orthographic view is 1/ortho_scale metres wide, x to the image right and y up; rendered at SUPERSAMPLE
    times the tile's pixels and area-averaged down."""
    def __init__(self, scene_id, gpu=0):
        import habitat_sim, magnum as mn
        settings = habitat_sim.SimulatorConfiguration()
        settings.scene_dataset_config_file = str(navverse.ASSETS / SCENE_DATASET)
        settings.scene_id = str(navverse.ASSETS / 'scenes' / f'{scene_id}.scene_instance.json')
        settings.gpu_device_id, settings.enable_physics = gpu, False
        color = habitat_sim.CameraSensorSpec()
        color.uuid, color.sensor_type, color.sensor_subtype = 'color', habitat_sim.SensorType.COLOR, habitat_sim.SensorSubType.ORTHOGRAPHIC
        color.resolution, color.near, color.far = [TILE_CELLS * SUPERSAMPLE] * 2, .01, 2000.
        color.position, color.orientation = mn.Vector3(0, 0, 0), mn.Vector3(-math.pi / 2, 0, 0)
        color.ortho_scale = 1. / (TILE_CELLS * BASE_GSD_M)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [color]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(settings, [agent]))

    def render(self, centre_xy, height):
        """RGB of the square centred at centre_xy, indexed [x, y]."""
        import habitat_sim
        state = habitat_sim.AgentState()
        state.position = np.array([centre_xy[0], height, -centre_xy[1]], np.float32)
        state.rotation = np.quaternion(1, 0, 0, 0)
        self.sim.get_agent(0).set_state(state, reset_sensors=True)
        image = cv2.resize(np.asarray(self.sim.get_sensor_observations()['color'])[..., :3], (TILE_CELLS, TILE_CELLS),
                           interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image[::-1].transpose(1, 0, 2))

    def close(self):
        self.sim.close()


def build(scene_id, gpu=0):
    """The city's tiles in the store (rewritten); returns their count."""
    from kappanav.common.geometry import Grid
    folder = OUT / 'navverse' / scene_id
    folder.mkdir(parents=True, exist_ok=True)
    grid = Grid.load(navverse.OUT / 'scenes' / scene_id / ROBOT.name / 'navigation.npz')
    triangles, low, high = city_triangles(scene_id)
    height = float(triangles[..., 2].max()) + 20.
    camera = OrthographicCamera(scene_id, gpu)
    tiles = 0
    try:
        for a in range(0, grid.free.shape[0], TILE_CELLS):
            for b in range(0, grid.free.shape[1], TILE_CELLS):
                lo = grid.origin_xy + np.array([a, b]) * BASE_GSD_M
                hi = lo + TILE_CELLS * BASE_GSD_M
                if not ((high >= lo).all(1) & (low <= hi).all(1)).any():
                    continue
                free = np.full((TILE_CELLS, TILE_CELLS), 255, np.uint8)
                part = grid.free[a:a + TILE_CELLS, b:b + TILE_CELLS]
                free[:part.shape[0], :part.shape[1]] = part
                np.savez_compressed(folder / f'{a // TILE_CELLS:02d}_{b // TILE_CELLS:02d}.npz', traversable=free,
                                    rgb=camera.render((lo + hi) / 2, height), gsd_m=np.float32(BASE_GSD_M), origin_xy=lo)
                tiles += 1
    finally:
        camera.close()
    return dict(scene=scene_id, robot=ROBOT.name, tiles=tiles, tile_m=TILE_CELLS * BASE_GSD_M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', nargs='+', default=['vcp_amsterdam'], help='vcp_<city> (exported and navigated)')
    ap.add_argument('--gpu', type=int, default=0, help='index within CUDA_VISIBLE_DEVICES')
    args = ap.parse_args()
    for scene_id in args.scene:
        print(json.dumps(build(scene_id, args.gpu)), flush=True)


if __name__ == '__main__':
    main()
