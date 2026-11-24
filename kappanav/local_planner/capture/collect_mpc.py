"""Collect fresh physical MPC trajectories without running a navigation network."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from kappanav.local_planner.data.capture import save_capture
from kappanav.local_planner.data.bspline_teachers import expert_route, pack_routes
from kappanav.common.geometry import PlanningError, _spatial_route, path_length
from kappanav.local_planner.data import MPC_SOURCE
from kappanav.evaluation.simulation import scene
from kappanav.evaluation.simulation.renderer import camera_to_body, mounted_camera
from kappanav.evaluation.simulation.navigation import NavigationGrid
from kappanav.evaluation.simulation.realtime import RealtimeMotion, GOAL_TOLERANCE_M
from kappanav.local_planner.control.mpc import axle_state


# Expert frames are 0.1 s apart; the expert replans (and the label stage has a teacher) every ANCHOR_STRIDE frames.
# Every frame still enters the history.
ANCHOR_STRIDE=2
# The expert parks the wheel axle midpoint on the goal: the MPC tracks that point (the differential drive's unicycle
# point), whereas base_link trails it by TURN_CENTRE_FORWARD_OFFSET_M and cannot be placed exactly driving forward.
PARKED_M=.02


def parked(view,goal):
    return float(np.linalg.norm(axle_state(view)[:2]-goal))<=PARKED_M


def acquisition_contract():
    source=Path(__file__).resolve().parents[2]
    planner,common=source/'local_planner',source/'common'
    files=[planner/'capture/collect_mpc.py',common/'depth.py',common/'geometry.py',common/'physical.py']
    files+=[planner/'data'/name for name in ('bspline_teachers.py','capture.py','observed_geometry.py','privileged.py')]
    # Task sampling (simulation/tasks.py) is not acquisition: a capture's task is checked by its whole case.
    files+=[path for path in (source/'evaluation/simulation').glob('*.py') if path.name!='tasks.py']+sorted((planner/'control').glob('*.py'))
    digest=hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(source)).encode());digest.update(path.read_bytes())
    return dict(code_sha256=digest.hexdigest(),runtime_sha256=scene.RUNTIME_ASSETS_SHA256,anchor_stride=ANCHOR_STRIDE)


def episode_camera(spec,camera,pitch_up_range=None):
    """(renderer camera settings, mount pitch-up in degrees or None) of one capture. Without pitch_up_range it is the
    runtime camera; with (low, high) it is the Dingo mount at a pitch-up drawn uniformly from the case id alone, so a
    capture's viewpoint does not depend on workers, order or resumption."""
    if pitch_up_range is None:return dict(camera),None
    seed=int.from_bytes(hashlib.sha256(spec['case_id'].encode()).digest()[:8],'little')
    up=float(np.random.default_rng(seed).uniform(*pitch_up_range))
    return mounted_camera(camera,up),up


def collect(spec,folder,controller,*,simulator,observe=None,plan=None,camera=None):
    """Return the summary and raw arrays of one expert capture. The expert drives the teacher: at every anchor frame it plans expert_route from its current
    state and the MPC follows that curve until the next anchor, so the recorded routes are the anchors' teachers.
    observe(view) renders a depth and plan(view) is expert_route toward the goal (defaults: this simulator and
    this process); a capture worker passes its render service and helper processes. camera is the episode's
    renderer camera settings (episode_camera; default: the runtime camera); observe must render with it."""
    case=spec['case'];folder.mkdir()
    shutil.copyfile(case['navigation'],folder/'source_grid.npz')
    navigation=NavigationGrid(folder/'source_grid.npz');grid=navigation.grid
    start=np.asarray(case['start'],dtype=np.float64);goal=np.asarray(case['goal'],dtype=np.float64)
    planning_error=None;motion=None;budget=0.
    try:
        geodesic=path_length(_spatial_route(grid,start[:2],goal)[0]);budget=float(3*geodesic/controller.max_speed+30)
    except PlanningError as problem:planning_error=str(problem)
    runtime,k,_=scene.sensor_assets()
    if camera is None:camera=dict(runtime['camera'])
    if observe is None:observe=lambda view:scene.observation(simulator,view,k,camera)
    pose=camera_to_body(camera)
    if plan is None:plan=lambda view:expert_route(grid,view,goal)
    route=plan(start) if planning_error is None else None
    if planning_error is None and route is None:planning_error='no forward expert route from the start'
    depths=[];views=[];timestamps=[];turning=[];turn_directions=[];routes=[];first_reached=None
    # Contact as in the benchmark physics: the body is held at an obstacle and turns about the axle where the obstacle
    # only has to push it back slightly; turning in place swings base_link about the axle, which a start at a wall needs.
    if route is not None:motion=RealtimeMotion(start,navigation.body_clearance,budget,controller=controller,collision_ends=False)
    while True:
        state=motion.snapshot() if motion is not None else dict(view=start,time_s=0.,status='running')
        if state['status']!='running':break
        view=state['view'];frame=len(views)
        depths.append(observe(view));views.append(view.copy());timestamps.append(state['time_s'])
        if frame%ANCHOR_STRIDE==0:
            if frame:route=plan(view)
            routes.append(route)
            if route is not None:motion.publish(route,state['time_s'])
        distance=np.linalg.norm(view[:2]-goal)
        if first_reached is None and distance<=GOAL_TOLERANCE_M:first_reached=len(views)-1
        if motion is None or parked(view,goal):
            turning.append(False);turn_directions.append(0);break
        motion.control_steps(1)
        action=motion.control_trace[-1]
        turning.append(bool(action.get('turning_in_place',False)))
        turn_directions.append(int(action.get('turn_direction',0)))
    final=motion.snapshot() if motion is not None else dict(status='planning_failure',view=start,path_m=0.,time_s=0.)
    distance=float(np.linalg.norm(final['view'][:2]-goal))
    positive=motion is not None and final['status']!='collision' and parked(final['view'],goal)
    eligible=first_reached if positive else len(views)
    count=len(views)
    arrays=dict(depths=np.asarray(depths),acquired_views=np.asarray(views),timestamps=np.asarray(timestamps),
        camera_intrinsics=np.repeat(k[None],count,axis=0),camera_to_body=np.repeat(pose[None],count,axis=0),
        body_to_world=np.stack([scene.body_pose(view) for view in views]),**pack_routes(routes),
        turning_in_place=np.asarray(turning,dtype=np.bool_),turn_direction=np.asarray(turn_directions,dtype=np.int8))
    metadata=dict(case=case,camera=camera,eligible_anchor_count=eligible,anchor_stride=ANCHOR_STRIDE,positive_expert=positive,
        planning_error=planning_error,controller=controller.metadata,acquisition=acquisition_contract(),
        provenance=dict(source_kind=MPC_SOURCE,split=spec['split'],trial_id=spec['trial_id'],driver='expert'),
        capture_clock='fixed physical control intervals;render/solver compute time excluded from training timestamps')
    save_capture(folder/'capture.json',arrays,metadata)
    result=dict(case_id=spec['case_id'],trial_id=spec['trial_id'],split=spec['split'],frames=count,
        eligible_anchors=eligible,positive_expert=positive,status='expert_complete' if positive else final['status'],
        planning_error=planning_error,teacher_anchors=sum(r is not None for r in routes),anchors=len(routes),
        final_distance_m=distance,path_m=final['path_m'],seconds=final['time_s'],contacts=motion.contacts if motion else 0)
    result.update(turning_in_place_frames=int(np.count_nonzero(turning)),turning_in_place_seconds=round(controller.period*np.count_nonzero(turning),3))
    if not positive:
        (folder/'failure.json').write_text(json.dumps(dict(**result,trace=motion.trace if motion else [],
            controls=motion.control_trace if motion else []),indent=2)+'\n')
    (folder/'collection.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,dict(arrays,**metadata)
