"""Build the benchmark from the 40 evaluation scenes and all 100 official point-goal pairs per scene.

The rules follow the reference benchmark's evaluation code: its own start/goal pairs per
scene (ROOT/pointgoal/SCENE.npy, an imported input with its PROVENANCE.json; rows [start_x, start_y, goal_x, goal_y, start_yaw] in canonical world XY),
122 s episodes, success 4 s after the robot is first within 0.5 m below 0.25 m/s (the official latched rule,
simulation.realtime.ARRIVAL_SPEED_MPS), contact does not end an episode, and SPL uses the straight start-goal distance.
All 4000 episodes retain the official initial heading. Endpoints keep their official position unless the robot could not
turn in place at the start or the goal lies outside the C-space; those move (benchmark_episodes); no filtering or
subsampling.

usage: python -m kappanav.evaluation.benchmark.prepare [--root data/benchmark] --worker W --workers N   (geometry, per process)
       python -m kappanav.evaluation.benchmark.prepare [--root data/benchmark]                          (ROOT/tasks.json)
ROOT/assets holds the converted scenes (kappanav.evaluation.benchmark.assets) with scenes.json.
"""
from pathlib import Path
import argparse,json
import numpy as np
from kappanav.evaluation.simulation import scene
from kappanav.evaluation.simulation.navigation import NavigationGrid, prepare_scene
from kappanav.evaluation.simulation.realtime import ARRIVAL_HOLD_S
from kappanav.common.physical import ROBOT, BODY_OBSTACLE_MIN_Z_M, ROBOT_FOOTPRINT_RADIUS_M
from kappanav.evaluation.benchmark.protocol import GOAL_BUDGET_S, SCENE_COUNT, EPISODES_PER_SCENE
from kappanav.evaluation.simulation.tasks import turns_in_place

def visible_clearance_m(intrinsics):
    """Signed contact distance beyond which every body-band obstacle in front of the robot is inside the image: the
    lowest band point (BODY_OBSTACLE_MIN_Z_M) enters the lower image edge this far ahead of base_link."""
    lowest=np.deg2rad(ROBOT.camera_downward_pitch_degrees)+np.arctan(intrinsics[1][2]/intrinsics[1][1])
    ahead=ROBOT.camera_forward_m+(ROBOT.camera_height_m-BODY_OBSTACLE_MIN_Z_M)/np.tan(lowest)
    return float(ahead-ROBOT_FOOTPRINT_RADIUS_M)


def _nearest(navigation,point,component,margin):
    """The free cell centre of `component` with signed contact distance >= `margin` nearest to `point`."""
    grid=navigation.grid;cells=np.argwhere((grid.components==component)&(grid.clearance_m>=margin))
    world=grid.grid_to_world(cells);nearest=world[int(np.argmin(np.linalg.norm(world-point,axis=1)))]
    assert navigation.clearance(np.repeat(nearest[None],2,0))>=margin
    return nearest,float(np.linalg.norm(nearest-point))


def benchmark_episodes(row,pairs,*,start_clearance_m):
    """Official evaluation episodes of one scene, in file order. An official start the robot cannot turn in place at
    (its axle midpoint less than the axle offset inside the C-space) moves to the nearest position with signed contact
    distance >= `start_clearance_m`, so that no body obstacle near it is hidden below the image; an official goal
    outside the C-space moves to the nearest cell inside it. Both stay in the region of the other endpoint (else the
    largest) and keep the heading; the official tuple is kept in `official`."""
    if pairs.shape!=(EPISODES_PER_SCENE,5) or not np.isfinite(pairs).all():
        raise ValueError(f'{row["scene_id"]}: expected {EPISODES_PER_SCENE} finite [start_x,start_y,goal_x,goal_y,yaw] pairs')
    navigation=NavigationGrid(row['navigation']);grid=navigation.grid
    counts=np.bincount(grid.components.ravel());counts[0]=0
    area=float(counts.max()*grid.cell_size_m**2);largest=int(counts.argmax())
    clearance=lambda point:navigation.clearance(np.repeat(np.asarray(point,dtype=np.float64)[None],2,0))
    region=lambda point:int(grid.components[tuple(grid.world_to_grid(np.asarray(point)))]) if clearance(point)>=0 else None
    episodes=[]
    for index,(x,y,gx,gy,yaw) in enumerate(pairs):
        start,goal=np.array([x,y]),np.array([gx,gy]);moved=[0.,0.]
        if not turns_in_place(navigation,(x,y,yaw)):
            start,moved[0]=_nearest(navigation,start,region(goal) or largest,start_clearance_m)
        if clearance(goal)<0:goal,moved[1]=_nearest(navigation,goal,region(start),0.)
        episodes.append(dict(episode_id=row['scene_id']+f'/goal_{index:03d}',scene_id=row['scene_id'],episode_index=index,
            start=[float(start[0]),float(start[1]),float(yaw)],goal=[float(goal[0]),float(goal[1])],
            official=[float(x),float(y),float(gx),float(gy),float(yaw)],endpoint_moved_m=moved,
            spl_distance_m=float(np.linalg.norm(goal-start)),goal_budget_s=GOAL_BUDGET_S,collision_ends=False,
            arrival_hold_s=ARRIVAL_HOLD_S,reachable_area_m2=area,exploration_budget_s=600.))
    return episodes


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,default=Path('data/benchmark'))
    parser.add_argument('--worker',type=int)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--runtime',type=Path,default=scene.DEFAULT_RUNTIME);args=parser.parse_args()
    scene.configure(args.runtime)
    rows=json.loads((args.root/'assets'/'scenes.json').read_text())['scenes']
    if len(rows)!=SCENE_COUNT or len({row['scene_id'] for row in rows})!=SCENE_COUNT:
        raise ValueError(f'Expected {SCENE_COUNT} distinct evaluation scenes')
    destination=args.root/'navigation'
    if args.worker is not None:
        for row in rows[args.worker::args.workers]:
            if (destination/row['scene_id']/'scene.json').is_file():continue
            result=prepare_scene(row,destination);print(result['scene_id'],flush=True)
        return
    margin=visible_clearance_m(scene.sensor_assets()[1])
    prepared=[json.loads((destination/row['scene_id']/'scene.json').read_text()) for row in rows]
    episodes=[episode for row in prepared for episode in
        benchmark_episodes(row,np.load(args.root/'pointgoal'/(row['asset_scene_id']+'.npy')),start_clearance_m=margin)]
    catalog=dict(protocol='official_pointgoal_4000',complete=True,scenes=prepared,episodes=episodes,
        goal_tolerance_m=.5,goal_budget_s=GOAL_BUDGET_S,grid_spacing_m=.05,
        reference='https://github.com/InternRobotics/NavDP open-source point-goal evaluation',
        selection='All 100 official pairs in each of 40 evaluation scenes; official heading, budget and arrival rule; inadmissible endpoints moved to the nearest admissible position')
    (args.root/'tasks.json').write_text(json.dumps(catalog,indent=2)+'\n')
    print(json.dumps(dict(scenes=len(prepared),episodes=len(episodes))),flush=True)


if __name__=='__main__':main()
