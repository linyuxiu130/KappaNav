"""Episode videos of an evaluation run, for presentations: what the robot's camera saw, where it was on the drone's
map, and how the episode went.

    python -m kappanav.evaluation.benchmark.video RUN [--episodes ID ...] [--heatmap DIR] [--output DIR]

RUN is an evaluation.run output directory; videos go to RUN/videos/<episode>.mp4 (1920 x 1080, one frame per decision,
at the pose of that decision's observation). Layout:
  main      the camera's colour image, re-rendered at the recorded poses (the evaluation renders depth only):
            NavVerse from the city's textured appearance asset (Habitat), Wanderland from the dataset's 3D Gaussian
            splatting (evaluation.splat, in the gsplat environment KAPPANAV_GSPLAT_PYTHON); the chosen plan and the
            current subgoal drawn on the ground; the policy's depth input inset
  local     heading-up map around the robot: the aerial image with the top-view model's P(traversable) over it, the
            global route, its fixed subgoals (the current one highlighted), the driven trajectory and the chosen plan
  overview  the whole route and trajectory on the same map, with the local window
  timeline  distance to the goal, speed and contact over the episode
The subgoal is replayed with route.Subgoals from the recorded poses, so it is the one the policy was given. The heat
map is DIR/<scene_id>.npz (evaluation.topview.maps: value [x, y], origin_xy, cell_m); by default the run's own
route maps when they were a directory.
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kappanav.global_planner.route import Subgoals

W, H = 1920, 1080
HEADER = 56
MAIN = (0, HEADER, 1280, 720)            # x, y, width, height
LOCAL = (1280, HEADER, 640, 560)
OVERVIEW = (1280, HEADER + 560, 640, H - HEADER - 560)
TIMELINE = (0, HEADER + 720, 1280, H - HEADER - 720)
DEPTH_INSET = (20, HEADER + 720 - 20 - 162, 288, 162)
LOCAL_M_PER_PX = .06                     # the local map is 38 m wide
LOCAL_ROBOT = (.5, .72)                  # the robot's place in the local map (fractions of width, height)
HEAT_ALPHA = .45
DEPTH_MAX_M = 5.                         # scene.MAX_DEPTH_M: the policy's depth is clipped there
SPLAT_CHUNK = 96                         # frames per gsplat process call
AERIAL = Path(os.environ.get('KAPPANAV_AERIAL', '/shibo_huang/data/topview'))  # evaluation.topview.navverse_aerial.OUT
GSPLAT_PYTHON = os.environ.get('KAPPANAV_GSPLAT_PYTHON', '/shibo_huang/envs/curvenav-gsplat/bin/python')
WANDERLAND_RAW = Path(os.environ.get('KAPPANAV_WANDERLAND', '/shibo_huang/data/wanderland')) / 'raw'

ROUTE, SUBGOAL, CURRENT, DRIVEN, PLAN, GOAL = (255, 255, 255), (235, 235, 235), (0, 200, 255), (255, 140, 0), \
    (255, 214, 0), (235, 64, 52)
INK, PANEL = (18, 18, 22), (28, 29, 34)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def _lut(name):
    from matplotlib import colormaps
    return (colormaps[name](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


HEAT, DEPTH = _lut('RdYlGn'), _lut('turbo')


# ---------------------------------------------------------------- maps
def raster(source, lo, hi):
    """The part of an [x, y] raster (array, origin, cell) covering world box lo..hi, with its origin."""
    array, origin, cell = source
    i0, j0 = np.maximum(np.floor((np.asarray(lo) - origin) / cell).astype(int), 0)
    i1, j1 = np.minimum(np.ceil((np.asarray(hi) - origin) / cell).astype(int), array.shape[:2])
    return array[i0:max(i1, i0 + 1), j0:max(j1, j0 + 1)], origin + np.array([i0, j0]) * cell, cell


def sample(source, xy, fill, linear=True):
    """An [x, y] raster sampled at world points xy [..., 2] (bilinear or nearest), fill off the raster."""
    array, origin, cell = raster(source, xy.reshape(-1, 2).min(0) - 1, xy.reshape(-1, 2).max(0) + 1)
    rows = ((xy[..., 0] - origin[0]) / cell - .5).astype(np.float32)
    cols = ((xy[..., 1] - origin[1]) / cell - .5).astype(np.float32)
    # Replicated borders so a cell centre at the edge is its own value; points off the raster get `fill`.
    out = cv2.remap(np.ascontiguousarray(array), cols, rows, cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE)
    off = (rows < -.5) | (cols < -.5) | (rows > array.shape[0] - .5) | (cols > array.shape[1] - .5)
    out[off] = fill
    return out


def aerial_image(row, lo, hi):
    """The drone's view of world box lo..hi as an [x, y] RGB raster: Wanderland's LiDAR orthophoto (maps.npz), or
    NavVerse's rendered aerial tiles (evaluation.topview.navverse_aerial)."""
    if row['source'] == 'wanderland':
        with np.load(row['maps']) as m:
            rgb = np.where(m['dom_valid'][..., None], m['dom_rgb'], 60).astype(np.uint8)
            return raster((rgb, m['map_origin_xy'], float(m['map_cell_m'])), lo, hi)
    tiles = sorted((AERIAL / 'navverse' / row['scene_id']).glob('*.npz'))
    if not tiles:
        raise FileNotFoundError(f'no aerial tiles for {row["scene_id"]}: run evaluation.topview.navverse_aerial')
    with np.load(tiles[0]) as t:
        side, cell = t['rgb'].shape[0], float(t['gsd_m'])
    picked = []
    for path in tiles:
        with np.load(path) as t:
            o = t['origin_xy']
            if (o + side * cell >= lo).all() and (o <= hi).all():
                picked.append((o, t['rgb']))
    origin = np.min([o for o, _ in picked], 0)
    index = [np.rint((o - origin) / (side * cell)).astype(int) for o, _ in picked]
    mosaic = np.full((*((np.max(index, 0) + 1) * side), 3), 60, np.uint8)
    for (i, j), (_, rgb) in zip(index, picked):
        mosaic[i * side:(i + 1) * side, j * side:(j + 1) * side] = rgb
    return mosaic, origin, cell


def heat_overlay(rgb, p):
    """The aerial image with P(traversable) laid over it (NaN: none)."""
    seen = np.isfinite(p)
    colour = HEAT[np.clip(np.nan_to_num(p) * 255, 0, 255).astype(np.uint8)]
    a = (HEAT_ALPHA * seen)[..., None]
    return (rgb * (1 - a) + colour * a).astype(np.uint8)


def local_frame(view):
    """The heading-up local map of the robot at view (x, y, yaw): world xy -> pixel function, and each pixel's world xy."""
    x, y, yaw = view
    forward, right = np.array([np.cos(yaw), np.sin(yaw)]), np.array([np.sin(yaw), -np.cos(yaw)])
    centre = np.array([LOCAL[2] * LOCAL_ROBOT[0], LOCAL[3] * LOCAL_ROBOT[1]])
    def to_px(xy):
        d = np.asarray(xy, float).reshape(-1, 2) - [x, y]
        return centre + np.stack([d @ right, -(d @ forward)], 1) / LOCAL_M_PER_PX
    u, v = np.meshgrid(np.arange(LOCAL[2]) + .5, np.arange(LOCAL[3]) + .5)
    world = (np.array([x, y]) + ((u - centre[0]) * LOCAL_M_PER_PX)[..., None] * right
             + ((centre[1] - v) * LOCAL_M_PER_PX)[..., None] * forward)
    return to_px, world


# ---------------------------------------------------------------- camera
class Camera:
    """The robot camera's colour image at body poses, at the main panel's size (the evaluated camera's field of view)."""
    def __init__(self, row, navigation):
        from kappanav.evaluation.simulation import scene
        self.row, self.navigation, self.scene = row, navigation, scene
        runtime, _, self.camera_to_body = scene.sensor_assets()
        k = scene.rendered_intrinsics()
        s = MAIN[2] / k.width
        self.camera = dict(runtime['camera'], image_width=MAIN[2], image_height=MAIN[3],
                           focal_x_px=k.fx * s, focal_y_px=k.fy * s)
        self.K = np.array([[k.fx * s, 0, MAIN[2] / 2], [0, k.fy * s, MAIN[3] / 2], [0, 0, 1]])

    def pose(self, view):
        """Optical camera to canonical world (x right, y down, z forward)."""
        z = self.navigation.ground_height(view[:2])
        return self.scene.body_pose(view, ground_z=z).astype(np.float64) @ self.camera_to_body

    def images(self, views):
        """(rgb, depth) per view, in order; sky (no surface) filled with a gradient."""
        if self.row['source'] == 'wanderland':
            yield from self._splats(views)
        else:
            yield from self._habitat(views)

    def _habitat(self, views):
        sim = self.scene.create_simulator(Path(self.row['assets']), self.row['asset_scene_id'], 0, self.camera,
                                          self.row['scene_dataset'], True)
        try:
            for view in views:
                rgb, depth = self.scene.raw_observation(sim, view, ground_z=self.navigation.ground_height(view[:2]), rgb=True)
                yield sky(rgb, depth), depth
        finally:
            sim.close()

    def _splats(self, views):
        ply = next(WANDERLAND_RAW.glob(f"data/*/*/{self.row['scene_id']}/3dgs.ply"))
        with tempfile.TemporaryDirectory() as folder:
            poses, out = Path(folder) / 'poses.npz', Path(folder) / 'out.npz'
            for k in range(0, len(views), SPLAT_CHUNK):
                np.savez(poses, camera_to_world=np.stack([self.pose(v) for v in views[k:k + SPLAT_CHUNK]]), K=self.K,
                         width=MAIN[2], height=MAIN[3])
                subprocess.run([GSPLAT_PYTHON, '-m', 'kappanav.evaluation.benchmark.splat', str(ply), str(poses), str(out)], check=True)
                with np.load(out) as o:
                    for rgb, depth in zip(o['rgb'], o['depth']):
                        yield sky(rgb, depth), depth


def sky(rgb, depth):
    """Pixels without a surface (depth inf) as a plain sky gradient."""
    blank = ~np.isfinite(depth)
    if blank.any():
        t = np.linspace(0, 1, rgb.shape[0])[:, None, None]
        gradient = ((1 - t) * np.array([120, 160, 205]) + t * np.array([200, 215, 230])).astype(np.uint8)
        rgb = np.where(blank[..., None], np.broadcast_to(gradient, rgb.shape), rgb)
    return np.ascontiguousarray(rgb)


# ---------------------------------------------------------------- drawing
def polyline(image, points, colour, width, outline=None, alpha=1.):
    """Antialiased polyline in pixel coordinates, optionally on a darker outline, optionally translucent."""
    points = np.asarray(points, np.float64)
    if len(points) < 2:
        return
    layer = image.copy() if alpha < 1 else image
    pts = [np.round(points * 16).astype(np.int32)]
    if outline is not None:
        cv2.polylines(layer, pts, False, outline, width + 3, cv2.LINE_AA, 4)
    cv2.polylines(layer, pts, False, colour, width, cv2.LINE_AA, 4)
    if alpha < 1:
        cv2.addWeighted(layer, alpha, image, 1 - alpha, 0, image)


def disc(image, point, radius, colour, ring=None):
    c = tuple(np.round(np.asarray(point) * 16).astype(int))
    if ring is not None:
        cv2.circle(image, c, int((radius + 2) * 16), ring, -1, cv2.LINE_AA, 4)
    cv2.circle(image, c, int(radius * 16), colour, -1, cv2.LINE_AA, 4)


def arrow(image, point, angle, size, colour=(255, 255, 255)):
    """The robot: a filled arrow at pixel `point` pointing at image angle `angle` (0 = right, counter-clockwise up)."""
    shape = np.array([[1., 0], [-.7, .62], [-.35, 0], [-.7, -.62]]) * size
    c, s = np.cos(angle), np.sin(angle)
    pts = np.round((shape @ np.array([[c, -s], [s, c]]).T * [1, -1] + point) * 16).astype(np.int32)
    cv2.fillPoly(image, [pts], INK, cv2.LINE_AA, 4)
    inner = np.round((shape * .78 @ np.array([[c, -s], [s, c]]).T * [1, -1] + point) * 16).astype(np.int32)
    cv2.fillPoly(image, [inner], colour, cv2.LINE_AA, 4)


class Text:
    """PIL text on the frame."""
    fonts = {}

    def __init__(self, image):
        self.pil = Image.fromarray(image)
        self.draw = ImageDraw.Draw(self.pil)

    def __call__(self, xy, text, size=18, colour=(240, 240, 240), bold=False, anchor='la', shadow=True):
        font = Text.fonts.setdefault((size, bold), ImageFont.truetype(BOLD if bold else FONT, size))
        if shadow:
            self.draw.text((xy[0] + 1, xy[1] + 1), text, font=font, fill=(0, 0, 0), anchor=anchor)
        self.draw.text(xy, text, font=font, fill=colour, anchor=anchor)

    def pill(self, xy, text, colour, size=18):
        font = Text.fonts.setdefault((size, True), ImageFont.truetype(BOLD, size))
        w = self.draw.textlength(text, font=font)
        self.draw.rounded_rectangle((xy[0] - w - 28, xy[1] - 15, xy[0], xy[1] + 15), 15, fill=colour)
        self.draw.text((xy[0] - w / 2 - 14, xy[1]), text, font=font, fill=(255, 255, 255), anchor='mm')

    def done(self):
        return np.asarray(self.pil)


def framed(image, box, width=2, colour=(235, 235, 235)):
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), colour, width, cv2.LINE_AA)


# ---------------------------------------------------------------- episode
class Episode:
    """One episode's recorded run, its task and the maps it is drawn on."""
    def __init__(self, run, task, row, heatmap):
        from kappanav.evaluation.simulation.navigation import NavigationGrid
        folder = next(run.glob(f"worker_*/{task['episode_id']}"))
        self.task, self.row = task, row
        self.result = json.loads((folder / 'result.json').read_text())
        with np.load(folder / 'goal_candidates.npz') as data:
            self.d = dict(data)
        with np.load(folder / 'goal_trace.npz') as data:
            stopped = data['sweep_margin_m'] != data['attempted_sweep_margin_m']  # a step cut short at an obstacle
            self.trace = dict(time=data['time_s'], view=data['view'], speed=np.abs(data['linear_speed_mps']),
                              stopped=stopped)
        self.navigation = NavigationGrid(row['navigation'])
        self.goal = np.asarray(task['goal'], float)
        self.route = np.asarray(task['route'], float) if task.get('route') else None
        views = self.d['acquired_views']
        if self.route is not None:
            subgoals = Subgoals(self.route, task.get('route_doubtful') or ())
            self.waypoints = subgoals.points
            self.current = np.array([subgoals.target(v[:2]) for v in views])
            self.index = np.array([int(np.flatnonzero((subgoals.points == c).all(1))[0]) for c in self.current])
        else:
            self.waypoints, self.current = self.goal[None], np.repeat(self.goal[None], len(views), 0)
            self.index = np.zeros(len(views), int)
        # Maps over everything the episode can show: the route, the trajectory, the local window around it.
        points = np.vstack([self.trace['view'][:, :2], self.goal[None], self.waypoints,
                            *([] if self.route is None else [self.route])])
        reach = LOCAL_M_PER_PX * np.hypot(LOCAL[2], LOCAL[3])
        lo, hi = points.min(0) - reach, points.max(0) + reach
        self.aerial = aerial_image(row, lo, hi)
        with np.load(heatmap) as m:
            self.heat = raster((m['value'].astype(np.float32) / 255, m['origin_xy'], float(m['cell_m'])), lo, hi)
        self._overview(points)
        self._timeline()

    def local(self, k):
        view = self.d['acquired_views'][k]
        to_px, world = local_frame(view)
        rgb = sample(self.aerial, world, (60, 60, 60))
        image = heat_overlay(rgb, sample(self.heat, world, np.nan))
        self._vectors(image, to_px, k, 2)
        arrow(image, to_px(view[:2])[0], np.pi / 2, 15)
        return image

    def _vectors(self, image, to_px, k, scale):
        """Route, subgoals, driven trajectory, plan and goal in a map panel."""
        t = self.d['timestamps'][k]
        if self.route is not None:
            polyline(image, to_px(self.route), ROUTE, scale, INK, .9)
        polyline(image, to_px(self.trace['view'][self.trace['time'] <= t + 1e-9, :2]), DRIVEN, scale + 1, INK)
        for i, w in enumerate(self.waypoints[:-1]):
            if i > self.index[k]:
                disc(image, to_px(w)[0], 1.5 * scale, SUBGOAL, INK)
        if 'candidate_paths' in self.d:
            polyline(image, to_px(self.plan(k)), PLAN, max(1, scale - 1))
        disc(image, to_px(self.goal)[0], 3 * scale, GOAL, (255, 255, 255))
        if self.index[k] < len(self.waypoints) - 1:
            disc(image, to_px(self.current[k])[0], 2.5 * scale, CURRENT, (255, 255, 255))

    def plan(self, k):
        view = self.d['acquired_views'][k]
        c, s = np.cos(view[2]), np.sin(view[2])
        return self.d['candidate_paths'][k, self.d['selected_indices'][k]] @ np.array([[c, s], [-s, c]]) + view[:2]

    # -- overview, north up
    def _overview(self, points):
        lo, hi = points.min(0) - 12, points.max(0) + 12
        res = max((hi[0] - lo[0]) / OVERVIEW[2], (hi[1] - lo[1]) / OVERVIEW[3])
        centre = (lo + hi) / 2
        self.over_px = lambda xy: (np.asarray(xy, float).reshape(-1, 2) - centre) / res * [1, -1] \
            + [OVERVIEW[2] / 2, OVERVIEW[3] / 2]
        u, v = np.meshgrid(np.arange(OVERVIEW[2]) + .5, np.arange(OVERVIEW[3]) + .5)
        world = np.stack([centre[0] + (u - OVERVIEW[2] / 2) * res, centre[1] - (v - OVERVIEW[3] / 2) * res], -1)
        image = heat_overlay(sample(self.aerial, world, (60, 60, 60)), sample(self.heat, world, np.nan))
        if self.route is not None:
            polyline(image, self.over_px(self.route), ROUTE, 2, INK, .9)
        disc(image, self.over_px(self.task['start'][:2])[0], 5, (70, 140, 255), (255, 255, 255))
        disc(image, self.over_px(self.goal)[0], 6, GOAL, (255, 255, 255))
        self.over_base = image

    def overview(self, k):
        image = self.over_base.copy()
        view, t = self.d['acquired_views'][k], self.d['timestamps'][k]
        polyline(image, self.over_px(self.trace['view'][self.trace['time'] <= t + 1e-9, :2]), DRIVEN, 3, INK)
        _, world = local_frame(view)
        corners = world[[0, 0, -1, -1], [0, -1, -1, 0]]
        cv2.polylines(image, [np.round(self.over_px(corners) * 16).astype(np.int32)], True, (255, 255, 255), 1,
                      cv2.LINE_AA, 4)
        arrow(image, self.over_px(view[:2])[0], view[2], 9)
        return image

    # -- timeline
    def _timeline(self):
        x0, y0, w, h = 70, 44, TIMELINE[2] - 100, TIMELINE[3] - 90
        times, views = self.d['timestamps'], self.d['acquired_views']
        end = max(float(self.trace['time'][-1]), float(times[-1]), 1.)
        distance = np.linalg.norm(views[:, :2] - self.goal, axis=1)
        top = max(float(distance.max()), 1.)
        self.time_px = lambda t: x0 + np.asarray(t) / end * w
        image = np.full((TIMELINE[3], TIMELINE[2], 3), PANEL, np.uint8)
        for f in (0, .25, .5, .75, 1):
            cv2.line(image, (x0, int(y0 + h * f)), (x0 + w, int(y0 + h * f)), (52, 54, 60), 1)
        # contact spans: a band under the plot
        stopped, t = self.trace['stopped'], self.trace['time']
        edges = np.flatnonzero(np.diff(np.r_[0, stopped.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            cv2.rectangle(image, (int(self.time_px(t[a])), y0 + h + 8), (int(np.ceil(self.time_px(t[min(b, len(t) - 1)]))),
                          y0 + h + 18), (220, 60, 60), -1)
        speed = np.interp(times, t, self.trace['speed'])
        top_speed = max(float(speed.max()), .5)
        polyline(image, np.stack([self.time_px(times), y0 + h - speed / top_speed * h], 1), (120, 170, 255), 1)
        polyline(image, np.stack([self.time_px(times), y0 + h - distance / top * h], 1), (255, 255, 255), 2)
        text = Text(image)
        text((x0, 14), 'distance to goal', 16, (255, 255, 255), bold=True)
        text((x0 + 170, 14), 'speed', 16, (120, 170, 255), bold=True)
        text((x0 + 240, 14), 'contact', 16, (220, 60, 60), bold=True)
        text((x0 - 8, y0), f'{top:.0f} m', 13, (170, 170, 170), anchor='ra')
        text((x0 - 8, y0 + h), '0', 13, (170, 170, 170), anchor='rd')
        text((x0 + w, y0 - 6), f'{top_speed:.1f} m/s', 13, (120, 170, 255), anchor='rd')
        for s in np.arange(0, end + 1e-9, nice_step(end)):
            text((self.time_px(s), y0 + h + 24), f'{s:.0f} s', 13, (170, 170, 170), anchor='ma')
        self.timeline_base, self.plot = text.done(), (y0, h)
        self.distance = distance

    def timeline(self, k):
        image = self.timeline_base.copy()
        x = self.time_px(self.d['timestamps'][k])
        y0, h = self.plot
        cv2.line(image, (int(round(x)), y0 - 4), (int(round(x)), y0 + h + 18), (255, 214, 0), 2, cv2.LINE_AA)
        return image

    # -- main view
    def main(self, k, rgb, camera):
        image = rgb.copy()
        view = self.d['acquired_views'][k]
        world_to_camera = np.linalg.inv(camera.pose(view))
        def project(xy):
            xy = np.asarray(xy, float).reshape(-1, 2)
            z = np.array([self.navigation.ground_height(p) if self._on_grid(p) else np.nan for p in xy])
            p = world_to_camera @ np.c_[xy, z, np.ones(len(xy))].T
            return (camera.K @ p[:3]).T, p[2]
        if 'candidate_paths' in self.d:
            uvw, depth = project(self.plan(k))
            keep = depth > .2
            if keep.sum() > 1:
                polyline(image, uvw[keep, :2] / uvw[keep, 2:], PLAN, 5, INK, .8)
        target = self.current[k]
        uvw, depth = project(target)
        if depth[0] > .3:
            u, v = uvw[0, :2] / uvw[0, 2]
            if 0 <= u < MAIN[2] and 0 <= v < MAIN[3]:
                disc(image, (u, v), 9, CURRENT if self.index[k] < len(self.waypoints) - 1 else GOAL, (255, 255, 255))
        return image

    def _on_grid(self, xy):
        cell = (np.asarray(xy) - self.navigation.origin) / self.navigation.spacing
        return (cell >= 0).all() and (cell < self.navigation.free.shape).all()

    def depth_inset(self, k):
        depth = self.d['depths'][k].astype(np.float32)
        colour = DEPTH[np.clip(np.nan_to_num(depth, nan=0, posinf=DEPTH_MAX_M) / DEPTH_MAX_M * 255, 0, 255).astype(np.uint8)]
        colour[~(depth > 0)] = 0
        return cv2.resize(colour, DEPTH_INSET[2:], interpolation=cv2.INTER_AREA)

    # -- frame
    def frame(self, k, rgb, camera):
        image = np.full((H, W, 3), PANEL, np.uint8)
        x, y, w, h = MAIN
        image[y:y + h, x:x + w] = self.main(k, rgb, camera)
        if 'depths' in self.d:
            dx, dy, dw, dh = DEPTH_INSET
            image[dy:dy + dh, dx:dx + dw] = self.depth_inset(k)
            framed(image, DEPTH_INSET)
        for box, panel in ((LOCAL, self.local(k)), (OVERVIEW, self.overview(k)), (TIMELINE, self.timeline(k))):
            bx, by, bw, bh = box
            image[by:by + bh, bx:bx + bw] = panel
        cv2.line(image, (LOCAL[0], HEADER), (LOCAL[0], H), INK, 3)
        cv2.line(image, (LOCAL[0], OVERVIEW[1]), (W, OVERVIEW[1]), INK, 3)
        cv2.rectangle(image, (0, 0), (W, HEADER - 1), INK, -1)
        text = Text(image)
        source = {'wanderland': 'Wanderland', 'navverse': 'NavVerse'}.get(self.row['source'], self.row['source'])
        text((20, HEADER // 2), source, 22, bold=True, anchor='lm', shadow=False)
        offset = 20 + text.draw.textlength(source, font=Text.fonts[22, True]) + 18
        text((offset, HEADER // 2), self.task['episode_id'], 20, (190, 190, 190), anchor='lm', shadow=False)
        t, last = float(self.d['timestamps'][k]), k == len(self.d['timestamps']) - 1
        text((W - 230, HEADER // 2), f"t  {t:5.1f} / {self.task['goal_budget_s']:.0f} s", 20, anchor='rm', shadow=False)
        status = self.result['goal_status'] if last else 'running'
        label, colour = {'goal_reached': ('success', (46, 160, 67)), 'running': ('running', (90, 92, 100))}.get(
            status, ('failed: ' + status.replace('_', ' '), (200, 50, 50)))
        text.pill((W - 20, HEADER // 2), label, colour)
        text((DEPTH_INSET[0] + 8, DEPTH_INSET[1] + 6), 'depth (policy input)', 14)
        text((x + 20, y + 16), 'camera', 18, bold=True)
        remaining = self.distance[k]
        text((x + w - 20, y + 16), f'goal {remaining:.0f} m', 18, bold=True, anchor='ra')
        text((LOCAL[0] + 16, LOCAL[1] + 12), 'drone map  ·  P(traversable)', 16, bold=True)
        text((OVERVIEW[0] + 16, OVERVIEW[1] + 12), 'route overview', 16, bold=True)
        legend = (LOCAL[0] + LOCAL[2] - 176, LOCAL[1] + 16)
        image = text.done().copy()
        image[legend[1]:legend[1] + 10, legend[0]:legend[0] + 160] = cv2.resize(HEAT[None], (160, 10))
        text = Text(image)
        text((legend[0], legend[1] + 14), '0', 13)
        text((legend[0] + 160, legend[1] + 14), '1', 13, anchor='ra')
        key = [(ROUTE, 'route'), (CURRENT, 'subgoal'), (DRIVEN, 'driven'), (PLAN, 'plan'), (GOAL, 'goal')]
        for i, (colour, name) in enumerate(key):
            px = LOCAL[0] + 16 + i * 120
            text.draw.ellipse((px, LOCAL[1] + LOCAL[3] - 26, px + 12, LOCAL[1] + LOCAL[3] - 14), fill=colour, outline=INK)
            text((px + 18, LOCAL[1] + LOCAL[3] - 20), name, 14, anchor='lm')
        return text.done()


def nice_step(span):
    for step in (5, 10, 20, 30, 60, 120, 300, 600):
        if span / step <= 10:
            return step
    return 1200


def render(run, episode_id, heatmap_dir, output, fps, max_frames):
    import imageio.v2 as imageio
    from kappanav.evaluation.simulation import scene
    scene.configure(scene.DEFAULT_RUNTIME)
    tasks = json.loads((run / 'tasks.json').read_text())
    task = next(e for e in tasks['episodes'] if e['episode_id'] == episode_id)
    row = next(s for s in tasks['scenes'] if s['scene_id'] == task['scene_id'])
    episode = Episode(run, task, row, Path(heatmap_dir) / f"{row['scene_id']}.npz")
    count = len(episode.d['timestamps'])
    frames = sorted({*range(0, count, -(-count // max_frames)), count - 1})
    camera = Camera(row, episode.navigation)
    path = output / (episode_id.replace('/', '__') + '.mp4')
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=8) as writer:
        for k, (rgb, _) in zip(frames, camera.images(episode.d['acquired_views'][frames])):
            writer.append_data(episode.frame(k, rgb, camera))
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    parser.add_argument('--episodes', nargs='+', help='episode ids (default: every episode of the run)')
    parser.add_argument('--heatmap', type=Path, help="directory of <scene_id>.npz traversability maps (default: the run's route maps)")
    parser.add_argument('--output', type=Path, help='default RUN/videos')
    parser.add_argument('--fps', type=float, default=10., help='frames per second; one frame per decision (0.2 s)')
    parser.add_argument('--max-frames', type=int, default=3000, help='longer episodes use every n-th decision')
    args = parser.parse_args()
    tasks = json.loads((args.run / 'tasks.json').read_text())
    heatmap = args.heatmap or (Path(tasks['route_maps']) if tasks.get('route_maps') not in (None, 'oracle') else None)
    if heatmap is None:
        parser.error('the run planned no routes on a predicted map: give --heatmap DIR')
    output = args.output or args.run / 'videos'
    output.mkdir(parents=True, exist_ok=True)
    for episode in args.episodes or [e['episode_id'] for e in tasks['episodes']]:
        print(render(args.run, episode, heatmap, output, args.fps, args.max_frames), flush=True)


if __name__ == '__main__':
    main()
