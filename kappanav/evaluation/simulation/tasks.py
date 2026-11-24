"""Deterministic long-range start/goal sampling on shared navigation geometry."""
import numpy as np
from scipy.ndimage import label
from kappanav.common.grid_dijkstra import grid_distances
from kappanav.evaluation.simulation.navigation import NavigationGrid
from kappanav.evaluation.simulation.realtime import GOAL_TOLERANCE_M
from kappanav.common.physical import TURN_CENTRE_FORWARD_OFFSET_M

# Start-goal geodesic distance of the official point-goal benchmark tasks; farther goals reach the training data
# through multi-goal relabelling (prepare_replay.label_records), not through longer expert drives.
GOAL_GEODESIC_RANGE_M=(3.,12.)


def turns_in_place(navigation,view):
    """Whether the robot at `view` can turn a full circle in place: base_link swings about the turn centre at the
    turn-centre offset, so every position on that circle is free iff the turn centre has that much signed clearance."""
    x,y,yaw=view;axle=np.array([x,y])+TURN_CENTRE_FORWARD_OFFSET_M*np.array([np.cos(yaw),np.sin(yaw)])
    return float(navigation.clearance(np.repeat(axle[None],2,0)))>=TURN_CENTRE_FORWARD_OFFSET_M


def sample_episodes(row,*,seed,count,distance_range_m=GOAL_GEODESIC_RANGE_M):
    navigation=NavigationGrid(row['navigation']);grid=navigation.grid
    components,sizes=label(grid.free)
    counts=np.bincount(components.ravel());counts[0]=0
    reachable=components==counts.argmax()
    eligible=np.argwhere(reachable&(grid.clearance_m>=.10))
    rng=np.random.default_rng(seed);episodes=[]
    minimum,maximum=distance_range_m
    for attempt in range(200):
        start=eligible[rng.integers(len(eligible))]
        distances=grid_distances(grid.free,grid.cell_size_m,int(start[0]),int(start[1]))
        goals=np.argwhere(reachable&(grid.clearance_m>=.10)&(distances>=minimum)&(distances<=maximum))
        goals=goals[np.linalg.norm((goals-start)*grid.cell_size_m,axis=-1)>GOAL_TOLERANCE_M]
        if not len(goals):continue
        goal=goals[rng.integers(len(goals))];xy=grid.grid_to_world(np.stack((start,goal)))
        episodes.append(dict(episode_id=row['scene_id']+f'/start_{len(episodes):03d}',scene_id=row['scene_id'],
            start=[*xy[0].tolist(),float(rng.uniform(-np.pi,np.pi))],goal=xy[1].tolist(),
            shortest_path_m=float(distances[tuple(goal)]),reachable_area_m2=float(reachable.sum()*grid.cell_size_m**2),
            goal_budget_s=float(3*distances[tuple(goal)]/.5+30),exploration_budget_s=600.))
        if len(episodes)==count:return episodes
    raise ValueError(f'Cannot sample {count} tasks at geodesic distance {minimum}-{maximum}m: {row["scene_id"]}')



def wall_episodes(row,*,seed,count,clearance_max_m=.10,contact_m=.03,facing=.7,facing_jitter_rad=np.pi/3,geodesic_range_m=GOAL_GEODESIC_RANGE_M):
    """Escape tasks the long-range catalog never visits: the start stands within clearance_max_m of an obstacle
    (half of them in contact, clearance <= contact_m) and, with probability `facing`, faces it. Official
    evaluation starts are 11% below 0.10 m and 40% below 0.20 m, while sample_episodes keeps starts >= 0.10 m.
    Goals lie at the official 3-12 m geodesic range. Only poses the robot can turn in place at are kept
    (turns_in_place, the rule the benchmark applies to its starts): elsewhere a forward-only robot cannot leave."""
    navigation=NavigationGrid(row['navigation']);grid=navigation.grid
    components,_=label(grid.free)
    counts=np.bincount(components.ravel());counts[0]=0
    reachable=components==counts.argmax()
    # Cell axes are world x,y; the clearance gradient points away from the nearest obstacle.
    gx,gy=np.gradient(grid.clearance_m.astype(np.float64))
    pools=(np.argwhere(reachable&(grid.clearance_m<=contact_m)),
           np.argwhere(reachable&(grid.clearance_m>contact_m)&(grid.clearance_m<clearance_max_m)))
    rng=np.random.default_rng(seed);episodes=[]
    for attempt in range(20*count):
        pool=pools[len(episodes)%2] if len(pools[len(episodes)%2]) else pools[1-len(episodes)%2]
        if not len(pool):break
        start=pool[rng.integers(len(pool))]
        distances=grid_distances(grid.free,grid.cell_size_m,int(start[0]),int(start[1]))
        goals=np.argwhere(reachable&(grid.clearance_m>=.10)&(distances>=geodesic_range_m[0])&(distances<=geodesic_range_m[1]))
        if not len(goals):continue
        goal=goals[rng.integers(len(goals))];xy=grid.grid_to_world(np.stack((start,goal)))
        away=np.array([gx[tuple(start)],gy[tuple(start)]])
        if rng.random()<facing and np.linalg.norm(away)>0:
            yaw=float(np.arctan2(-away[1],-away[0])+rng.uniform(-facing_jitter_rad,facing_jitter_rad))
        else:yaw=float(rng.uniform(-np.pi,np.pi))
        if not turns_in_place(navigation,(*xy[0],yaw)):continue
        episodes.append(dict(episode_id=row['scene_id']+f'/wall_{len(episodes):03d}',scene_id=row['scene_id'],
            start=[*xy[0].tolist(),float(np.angle(np.exp(1j*yaw)))],goal=xy[1].tolist(),
            start_clearance_m=float(grid.clearance_m[tuple(start)]),shortest_path_m=float(distances[tuple(goal)]),
            goal_budget_s=float(3*distances[tuple(goal)]/.5+30),exploration_budget_s=600.))
        if len(episodes)==count:break
    # A scene without 3 m of reachable geodesic room has no escape task; the catalog reports it as short.
    return episodes
