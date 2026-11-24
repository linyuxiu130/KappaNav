"""A deterministic frontier reference,with global geometry and actual MPC execution."""
from pathlib import Path
import numpy as np
from kappanav.common.grid_dijkstra import grid_distances
from kappanav.common.geometry import _astar, _simplify_route
from kappanav.evaluation.benchmark.coverage import ExplorationCoverage
from kappanav.evaluation.simulation import scene
from kappanav.evaluation.simulation.realtime import RealtimeMotion


TARGET_REACHED_M=.15


def exploration_route(navigation,coverage,view,*,snap_radius_m=.25):
    """Route to the geodesically farthest unobserved reachable cell, planned on the cells A* may use.

    Target choice, reachability and the route share one traversable set (free and at least the minimum
    clearance); a start next to an obstacle is snapped to the nearest such cell. None when no cell is near.
    """
    grid=navigation.grid
    allowed=grid.free&(grid.clearance_m+1e-9>=grid.minimum_clearance_m)
    here=grid.world_to_grid(view[:2])
    if (here>=0).all() and (here<allowed.shape).all() and allowed[tuple(here)]:start=here
    else:
        radius=max(1,int(np.ceil(snap_radius_m/grid.cell_size_m)))
        lo=np.maximum(here-radius,0);hi=np.minimum(here+radius+1,allowed.shape)
        near=np.argwhere(allowed[lo[0]:hi[0],lo[1]:hi[1]])+lo
        if not len(near):return None
        gap=np.linalg.norm(grid.grid_to_world(near)-view[:2],axis=-1)
        if gap.min()>snap_radius_m+grid.cell_size_m/np.sqrt(2):return None
        start=near[gap.argmin()]
    distances=grid_distances(allowed,grid.cell_size_m,int(start[0]),int(start[1]))
    cells=np.argwhere(coverage.reachable&~coverage.known&np.isfinite(distances))
    targets=grid.grid_to_world(cells)
    cells=cells[np.linalg.norm(targets-view[:2],axis=-1)>TARGET_REACHED_M]
    if not len(cells):return None
    target=cells[distances[cells[:,0],cells[:,1]].argmax()]
    path=grid.grid_to_world(_astar(grid,start,target))
    path[0]=view[:2]
    return _simplify_route(grid,path)


def run_exploration_expert(row,episode,navigation,controller,*,simulator=None):
    runtime,k,camera=scene.sensor_assets()
    sim=scene.create_simulator(Path(row['assets']),row['asset_scene_id'],0,runtime['camera'],row['scene_dataset']) if simulator is None else simulator
    coverage=ExplorationCoverage(navigation,episode['start'],k,camera)
    initial=scene.observation(sim,episode['start'],k,ground_z=navigation.ground_height(episode['start'][:2]))
    coverage.update(initial,episode['start'],0.)
    budget=episode['exploration_budget_s']
    motion=RealtimeMotion(episode['start'],navigation.body_clearance,budget,controller=controller)
    current_target=None;plans=[]
    def decide(state):
        # Observe, extend coverage and replan once the current target is seen or reached; False ends the episode.
        nonlocal current_target
        depth=scene.observation(sim,state['view'],k,ground_z=navigation.ground_height(state['view'][:2]))
        coverage.update(depth,state['view'],state['time_s'])
        if (current_target is None or
            coverage.known[tuple(navigation.grid.world_to_grid(current_target))] or
            np.linalg.norm(state['view'][:2]-current_target)<TARGET_REACHED_M):
            route=exploration_route(navigation,coverage,state['view'])
            if route is None:return False
            publication=motion.publish(route,state['time_s']);current_target=route[-1]
            plans.append(dict(world_path=route.tolist(),observation_time_s=state['time_s'],**publication))
        return True
    try:
        # Simulation time: one observation every 0.1 s of motion, the controller at its own period in between.
        steps=max(1,round(.1/controller.period))
        while motion.status=='running':
            state=motion.snapshot()
            if state['status']!='running' or not decide(state):break
            motion.control_steps(steps)
        final=motion.snapshot()
        if not plans:
            # No route from the start (e.g. a start deep inside an obstacle): the reference is invalid, not zero coverage.
            final=dict(final,status='planning_failure',failure_reason='no traversable cell near the start')
        return dict(status=final['status'],failure_reason=final['failure_reason'],collision=final['status']=='collision',motion_trace=motion.trace,plans=plans,controller=controller.metadata,
            times=coverage.times,areas=coverage.areas,initial_area_m2=coverage.initial_area_m2,
            path_m=final['path_m'],seconds=final['time_s'],reference='farthest_unobserved_geodesic_MPC',
            optimality='Privileged deterministic reference;not globally optimal exploration')
    finally:
        if simulator is None:sim.close()
