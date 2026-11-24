"""Global route on a traversability map: the long-range stage of global route -> subgoals -> CurveNav.

A map is the 0-255 traversability value (taxonomy.VALUE: road, dirt, grass, ... obstacle) on a world-aligned grid
indexed [x, y] (NaN = not observed). The route minimises 255 / value per metre, so it keeps to roads and takes grass
or worse only when the detour costs more. The policy is given fixed subgoals along the route, one at a time (Subgoals): every corner the route turns at
(line_of_sight keeps exactly those, so each leg is a straight chord the map clears) and points splitting longer legs
to at most SUBGOAL_SPACING_M, none inside a stretch the map doubts (value < DOUBT_VALUE), whose end gets one instead: there
the policy finds its own way across, as it does towards a bare goal.

The oracle map is the robot's navigation C-space: its route is the task's own C-space shortest path
(`shortest_path`, written with the task). The evaluator plans routes itself (evaluation.run --maps); this command
writes them for inspection:

    python -m kappanav.global_planner.route --tasks tasks.json --maps oracle|DIR --output routed.json  # DIR/<scene>.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
from skimage.graph import MCP_Geometric

from kappanav.global_planner.taxonomy import route_cost as cost_per_metre

DOUBT_VALUE = 100.        # below grass (130), above fields (70)
UNOBSERVED_VALUE = 100.   # cells the map does not see, unless the map file says otherwise (unobserved_value)
SUBGOAL_SPACING_M = 6.   # subgoals at most this far apart along the route (CurveNav's point goals are 3-12 m)
SUBGOAL_REACHED_M = 2.   # the policy is given the next subgoal within this distance of the current one


class NoRoute(ValueError):
    pass


def route_cost(value, unobserved_value=UNOBSERVED_VALUE):
    """Per-metre cost of every cell; inf where the value is 0 (predicted maps stop at taxonomy.VALUE_FLOOR: a predicted
    obstacle is avoided, not impossible)."""
    v = np.where(np.isfinite(value), value, unobserved_value)
    return np.where(v > 0, cost_per_metre(v), np.inf)


def plan_route(value, origin_xy, cell_m, start_xy, goal_xy, unobserved_value=UNOBSERVED_VALUE):
    """The least-cost route as polyline vertices in world XY, from start to goal (cell_polyline)."""
    cost = route_cost(value, unobserved_value)
    origin = np.asarray(origin_xy, np.float64)
    def cell(xy):
        return tuple(np.clip(np.floor((np.asarray(xy) - origin) / cell_m).astype(int), 0, np.array(cost.shape) - 1))
    mcp = MCP_Geometric(cost, fully_connected=True)
    costs, _ = mcp.find_costs([cell(start_xy)], [cell(goal_xy)])
    if not np.isfinite(costs[cell(goal_xy)]):
        raise NoRoute(f'goal {goal_xy} is not connected to start {start_xy} on the map')
    return cell_polyline(mcp.traceback(cell(goal_xy)), cost, origin, cell_m, start_xy, goal_xy)


def doubtful_stretches(polyline, value, origin_xy, cell_m, unobserved_value=UNOBSERVED_VALUE, step_m=.25):
    """Arc-length intervals [from, to] along the polyline where the map gives value < DOUBT_VALUE."""
    s = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))])
    along = np.append(np.arange(0., s[-1], step_m), s[-1])
    xy = np.stack([np.interp(along, s, polyline[:, a]) for a in range(2)], 1)
    ij = np.clip(np.floor((xy - np.asarray(origin_xy)) / cell_m).astype(int), 0, np.array(value.shape) - 1)
    v = value[ij[:, 0], ij[:, 1]]
    low = np.where(np.isfinite(v), v, unobserved_value) < DOUBT_VALUE
    edges = np.flatnonzero(np.diff(np.concatenate([[0], low.astype(int), [0]])))
    return [[float(along[a]), float(along[min(b, len(along) - 1)])] for a, b in zip(edges[::2], edges[1::2])]


def line_of_sight(cells, cost):
    """Indices of the cell path to keep as vertices: from each vertex, the furthest later cell whose straight chord
    crosses no cell costlier than the path itself between them, so the polyline never cuts a corner the optimal
    path went around."""
    keep, i, n = [0], 0, len(cells)
    while i < n - 1:
        j, worst = i + 1, cost[tuple(cells[i + 1])]
        while j + 1 < n:
            worst = max(worst, cost[tuple(cells[j + 1])])
            k = int(np.abs(cells[j + 1] - cells[i]).max()) * 4 + 1
            chord = np.rint(np.linspace(cells[i], cells[j + 1], k)).astype(int)
            if cost[chord[:, 0], chord[:, 1]].max() > worst:
                break
            j += 1
        keep.append(j)
        i = j
    return np.asarray(keep)


def subgoal_points(route, doubtful=()):
    """Arc lengths of the subgoals on a route polyline (start first, goal last): its vertices and points splitting
    each leg evenly to at most SUBGOAL_SPACING_M, without those inside a doubtful stretch [from, to], whose end
    is one instead. The goal is the last."""
    s = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
    along = [np.linspace(a, b, int(np.ceil((b - a) / SUBGOAL_SPACING_M)) + 1)[1:] for a, b in zip(s[:-1], s[1:]) if b > a]
    along = np.concatenate(along) if along else np.array([s[-1]])
    for low, high in doubtful:
        along = np.append(along[(along <= low) | (along >= high)], min(high, s[-1]))
    return np.unique(along), s


class Subgoals:
    """The policy's point goal on a route: its fixed subgoals (subgoal_points) in order. The current one is passed
    when the robot comes within SUBGOAL_REACHED_M of it or its progress along the route reaches it. Progress is the
    robot's projection on the route, searched only just ahead of the previous progress and never moving back, so a
    later leg that passes nearby is not taken for progress."""
    def __init__(self, route, doubtful=()):
        self.route = np.asarray(route, np.float64)
        self.along, self.s = subgoal_points(self.route, doubtful)
        self.points = self.point(self.along)
        self.progress, self.index = 0., 0

    def point(self, s):
        """Route points at arc lengths s."""
        return np.stack([np.interp(s, self.s, self.route[:, a]) for a in range(2)], -1)

    def target(self, position):
        a, b = self.route[:-1], self.route[1:]
        d = b - a
        length = np.maximum((d * d).sum(1), 1e-12)
        t = np.clip(((position - a) * d).sum(1) / length, 0, 1)
        distance = np.linalg.norm(a + t[:, None] * d - position, axis=1)
        along = self.s[:-1] + t * np.sqrt(length)
        window = (along >= self.progress - 1e-6) & (along <= self.progress + 2 * SUBGOAL_SPACING_M)
        if window.any():
            self.progress = max(self.progress, along[np.flatnonzero(window)[np.argmin(distance[window])]])
        while self.index < len(self.along) - 1 and (
                self.progress >= self.along[self.index]
                or np.linalg.norm(self.points[self.index] - position) <= SUBGOAL_REACHED_M):
            self.index += 1
        return self.points[self.index]


def oracle_map(navigation_path):
    """The simulator's navigation C-space (free = base_link may stand there) as a 0/255 map."""
    with np.load(navigation_path) as nav:
        return nav['free'].astype(np.float32) * 255, nav['origin_xy'], float(nav['cell_size_m']), UNOBSERVED_VALUE


def load_map(source, scene):
    """(value, origin, cell, unobserved_value) of a scene: 'oracle' or a directory of predicted <scene_id>.npz."""
    if source == 'oracle':
        return oracle_map(scene['navigation'])
    with np.load(Path(source) / f"{scene['scene_id']}.npz") as m:
        return (m['value'], m['origin_xy'], float(m['cell_m']),
                float(m['unobserved_value']) if 'unobserved_value' in m else UNOBSERVED_VALUE)


def cell_polyline(cells, cost, origin_xy, cell_m, start_xy, goal_xy):
    """World XY vertices of a cell path from start to goal: the exact endpoints and the cells line_of_sight keeps."""
    keep = line_of_sight(np.asarray(cells), cost)
    inner = np.asarray(origin_xy, np.float64) + (np.asarray(cells)[keep[1:-1]] + .5) * cell_m
    return np.vstack([start_xy, inner.reshape(-1, 2), goal_xy])


def plan_routes(data, maps):
    """The task list with each episode's global route on `maps` ('oracle' or a directory of predicted
    <scene_id>.npz): route (polyline from start to goal), route_doubtful (Subgoals skips them), route_length_m."""
    scenes = {s['scene_id']: s for s in data['scenes']}
    loaded, unreachable = {}, []
    for episode in data['episodes']:
        if maps == 'oracle':
            route, doubtful = np.asarray(episode['shortest_path'], np.float64), []
        else:
            sid = episode['scene_id']
            if sid not in loaded:
                loaded[sid] = load_map(maps, scenes[sid])
            value, origin, cell, unobserved_value = loaded[sid]
            try:
                route = plan_route(value, origin, cell, episode['start'][:2], episode['goal'], unobserved_value)
            except NoRoute:
                unreachable.append(episode['episode_id'])
                continue
            doubtful = doubtful_stretches(route, value, origin, cell, unobserved_value)
        episode['route'] = route.round(3).tolist()
        episode['route_doubtful'] = [[round(a, 2), round(b, 2)] for a, b in doubtful]
        episode['route_length_m'] = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    if unreachable:
        raise NoRoute(f'goal not connected to start on the map: {unreachable}')
    return dict(data, route_maps=maps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tasks', type=Path, required=True, help='benchmark task list')
    parser.add_argument('--maps', required=True, help="'oracle' (navigation C-space) or a directory of <scene_id>.npz")
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = plan_routes(json.loads(args.tasks.read_text()), args.maps)
    args.output.write_text(json.dumps(data, indent=1) + '\n')
    lengths = np.array([e['route_length_m'] / e['shortest_path_m'] for e in data['episodes']])
    print(f"{len(lengths)} routes, length / shortest path: median {np.median(lengths):.3f}, max {lengths.max():.3f}")


if __name__ == '__main__':
    main()
