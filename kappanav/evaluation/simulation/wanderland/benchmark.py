"""Export the selected robot's valid Wanderland tasks to CurveNav's closed-loop evaluator (evaluation.run).

CurveNav's evaluation rules (docs/EVALUATION.md): success is a 4 s hold within 0.5 m of the goal, contact does not
end an episode (the body stops at the obstacle and may turn or back out), and SPL uses the straight start-goal
distance. One change for long routes: the budget is max(122, 6 x grid geodesic + 30) s, not a fixed 122 s (61 m at
the 0.5 m/s top speed). The original Wanderland task stays in `official`; `shortest_path` is the task's C-space
shortest path (the oracle route). This is an integration benchmark, not the Wanderland protocol.
"""
import argparse, json
import math
from pathlib import Path

import numpy as np

from kappanav.common.grid_dijkstra import descend, grid_distances
from kappanav.common.geometry import Grid
from kappanav.global_planner.route import cell_polyline
from kappanav.common.physical import ROBOT
from kappanav.evaluation.simulation.assets import SCENE_DATASET
from kappanav.evaluation.simulation.realtime import ARRIVAL_HOLD_S
from kappanav.evaluation.simulation.wanderland.prepare import OUT


def shortest_path(grid, cost, start_xy, goal_xy):
    """The C-space shortest path between two cell centres as polyline vertices (route.cell_polyline), and its length."""
    start, goal = grid.world_to_grid(np.asarray([start_xy, goal_xy]))
    field = grid_distances(grid.free, grid.cell_size_m, int(start[0]), int(start[1]), goal[None])
    cells = descend(field, grid.free, grid.cell_size_m, int(goal[0]), int(goal[1]))[::-1]
    return cell_polyline(cells, cost, grid.origin_xy, grid.cell_size_m, start_xy, goal_xy), float(field[tuple(goal)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('scenes', nargs='*')
    ap.add_argument('--per-scene', type=int, help='shortest valid official tasks per scene; default all')
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    folders = [OUT / 'scenes' / sid for sid in args.scenes] if args.scenes else sorted((OUT / 'scenes').iterdir())
    if args.per_scene is not None and args.per_scene < 1:
        ap.error('--per-scene must be positive')
    rows, episodes, excluded = [], [], []
    for folder in folders:
        sid = folder.name
        nav = json.loads((folder / ROBOT.name / 'navigation.json').read_text())
        valid = [e for e in json.loads((folder / ROBOT.name / 'episodes.json').read_text()) if e['valid']]
        if args.per_scene is not None:
            valid = sorted(valid, key=lambda e: (e['geodesic_m'], e['episode_id']))[:args.per_scene]
        if not valid:
            excluded.append(sid)
            continue
        rows.append(dict(scene_id=sid, source='wanderland', domain='indoor_guess' if nav['indoor_guess'] else 'outdoor_guess',
                         assets=str(OUT / 'assets'), asset_scene_id=sid, scene_dataset=SCENE_DATASET,
                         navigation=str(folder / ROBOT.name / 'navigation.npz'), maps=str(folder / 'maps.npz')))
        grid = Grid.load(folder / ROBOT.name / 'navigation.npz')
        cost = np.where(grid.free, 1., np.inf)
        for e in valid:
            path, length = shortest_path(grid, cost, e['start_xy'], e['goal_xy'])
            if not math.isclose(length, e['geodesic_m'], rel_tol=1e-6):
                raise ValueError(f'{sid} episode {e["episode_id"]}: shortest path {length} m, task says {e["geodesic_m"]} m')
            episodes.append(dict(episode_id=f'{sid}/goal_{e["episode_id"]:03d}', scene_id=sid,
                                 start=[*e['start_xy'], e['start_heading_rad']], goal=e['goal_xy'],
                                 shortest_path_m=e['geodesic_m'], spl_distance_m=math.dist(e['start_xy'], e['goal_xy']),
                                 shortest_path=path.round(3).tolist(),
                                 goal_budget_s=max(122., 6 * e['geodesic_m'] + 30), collision_ends=False,
                                 arrival_hold_s=ARRIVAL_HOLD_S, official=e))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol=f'wanderland_{ROBOT.name}_pointgoal', robot=ROBOT.name, scenes=rows,
        episodes=episodes, goal_tolerance_m=.5, excluded_scenes_without_valid_tasks=excluded,
        selection='all valid tasks' if args.per_scene is None else f'{args.per_scene} shortest valid official tasks per requested scene'),
        indent=2) + '\n')
    print(json.dumps(dict(robot=ROBOT.name, scenes=len(rows), episodes=len(episodes), output=str(args.output))))


if __name__ == '__main__':
    main()
