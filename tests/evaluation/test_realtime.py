import numpy as np
import pytest
from pathlib import Path
from kappanav.evaluation.simulation.realtime import RealtimeMotion, GOAL_TOLERANCE_M
from kappanav.evaluation.simulation.scene import clearance as box_clearance

def collision(boxes):
    return lambda poses,padding:box_clearance(np.asarray(poses)[:,:2],boxes,radius=ROBOT_FOOTPRINT_RADIUS_M+padding)[0]

from kappanav.local_planner.control.mpc import PathMPC, axle_state
from kappanav.common.physical import ROBOT_FOOTPRINT_RADIUS_M, TURN_CENTRE_FORWARD_OFFSET_M


class Clock:
    def __init__(self):self.time=0.
    def __call__(self):return self.time


@pytest.fixture(scope='module')
def controller(tmp_path_factory):
    return PathMPC(tmp_path_factory.mktemp('realtime_mpc')/'generated')


def make(controller,duration=10.,boxes=((10.,10.,11.,11.),),goal=None,velocity=None):
    clock=Clock();motion=RealtimeMotion([0.,0.,0.],collision(boxes),duration,goal,controller=controller,initial_velocity=velocity,clock=clock)
    motion.publish(np.array([[0.,0.],[3.,0.]]),0.)
    return clock,motion


def test_solver_failure_is_a_visible_terminal_case(controller):
    from kappanav.local_planner.control.mpc import MPCSolveError
    _,motion=make(controller)
    def fail():raise MPCSolveError('out-of-bound command:[0.6,0.]')
    motion.control_step=fail
    motion.control_steps(3)
    state=motion.snapshot()
    assert state['status']=='controller_failure'
    assert 'out-of-bound' in state['failure_reason']


def test_unexpected_control_error_is_not_swallowed(controller):
    _,motion=make(controller)
    def fail():raise ValueError('broken reference shape')
    motion.control_step=fail
    with pytest.raises(ValueError,match='broken reference'):motion.control_steps(1)


@pytest.mark.parametrize('terminal',['collision','goal_reached','budget_exhausted'])
def test_late_solver_failure_preserves_physical_terminal_status(controller,terminal):
    from kappanav.local_planner.control.mpc import MPCSolveError
    _,motion=make(controller)
    def fail():
        motion.status=terminal
        raise MPCSolveError('late solve failure')
    motion.control_step=fail;motion.control_steps(3)
    assert motion.snapshot()['status']==terminal
    assert motion.snapshot()['failure_reason']=='late solve failure'


def advance(clock,motion,duration,period=.05):
    for tick in np.arange(clock.time+period,duration+period/2,period):
        motion.control_step();clock.time=float(tick);state=motion.snapshot()
        if state['status']!='running':return state
    return motion.snapshot()


def test_variable_intervals_preserve_continuous_physics(controller):
    snapshots=[]
    for ticks in ([.1,.2,.3,.7,1.],[.5,1.]):
        clock,motion=make(controller);motion.command[:]=[.5,0.]
        for tick in ticks:
            clock.time=tick;state=motion.snapshot();assert state['time_s']==tick
        expected=.5*(1-motion.response_s*(1-np.exp(-1/motion.response_s)))
        assert state['view'][0]==pytest.approx(expected,abs=1e-12)
        assert np.linalg.norm(state['velocity_mps'])<=.5+1e-12
        snapshots.append(state['view'])
    np.testing.assert_allclose(*snapshots,atol=1e-12)


def test_delayed_plan_and_control_handover_preserve_pose_speed_and_old_motion(controller):
    clock,motion=make(controller,velocity=[.5,0.])
    clock.time=.5;observed=motion.snapshot();clock.time=.7;before=motion.snapshot()
    np.testing.assert_array_equal(before['velocity_mps'],[.5,0.])
    publication=motion.publish(np.array([[observed['view'][0],0.],[observed['view'][0],2.]]),observed['time_s'])
    assert publication['observation_age_s']==pytest.approx(.2)
    np.testing.assert_array_equal(publication['view'],before['view'])
    np.testing.assert_array_equal(publication['velocity_mps'],before['velocity_mps'])
    original=controller.solve
    def delayed(*args):
        clock.time+=.15
        return original(*args)
    controller.solve=delayed
    try:motion.control_step()
    finally:controller.solve=original
    assert motion.view[0]>before['view'][0]
    assert motion.control_trace[-1]['actuation_latency_ms']==pytest.approx(150.)
    assert motion.view[2]==0  # A new command is adopted without teleporting the body angle.
    state=advance(clock,motion,8.)  # A 90-degree turn in place takes pi/(2*omega_max) at the in-place yaw response.
    assert abs(state['view'][2])>0 and state['view'][1]>0


@pytest.mark.dingo
def test_wall_contact_is_continuous_and_terminal(controller):
    clock,motion=make(controller,boxes=((.5,-1.,.7,1.),));motion.command[:]=[.5,0.]
    clock.time=4.;state=motion.snapshot();contact=.5-ROBOT_FOOTPRINT_RADIUS_M
    assert state['status']=='collision'
    assert contact-1e-4<=state['view'][0]<=contact+1e-9
    assert not motion.publish(np.array([[0.,0.],[0.,2.]]),0.)['applied']
    assert motion.trace[-1]['attempted_sweep_margin_m']<0
    assert max(r['numerical_sweep_padding_m'] for r in motion.trace)<1e-4


def test_mpc_reaches_goal_before_later_wall(controller):
    clock,motion=make(controller,boxes=((1.5,-1.,1.7,1.),),goal=[1.,0.])
    state=advance(clock,motion,4.)
    assert state['status']=='goal_reached'
    assert np.linalg.norm(state['view'][:2]-[1.,0.])<=GOAL_TOLERANCE_M+1e-8
    assert abs(state['view'][2])<1e-6


def test_budget_is_wall_seconds_not_controller_or_planner_calls(controller):
    clock,motion=make(controller,duration=.7);motion.command[:]=[.5,0.]
    clock.time=1.;state=motion.snapshot()
    assert state['status']=='budget_exhausted' and state['time_s']==.7
    assert state['path_m']<.35 and motion.trace[-1]['status']=='budget_exhausted'


def test_initial_overlap_and_backwards_clock_fail(controller):
    clock,motion=make(controller);clock.time=.2;motion.snapshot();clock.time=.1


def test_progress_does_not_jump_to_a_later_self_crossing_segment(controller):
    clock=Clock();point=np.array([.49,.51]);direction=point/np.linalg.norm(point);yaw=np.arctan2(direction[1],direction[0])
    motion=RealtimeMotion([0.,0.,yaw],collision([[10.,10.,11.,11.]]),5.,controller=controller,initial_velocity=.5*direction,clock=clock)
    motion.command[:]=[.5,0.];motion.publish(np.stack((np.zeros(2),3*direction)),0.)
    clock.time=np.linalg.norm(point)/.5;before=motion.snapshot();np.testing.assert_allclose(before['view'][:2],point,atol=1e-12)
    route=np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.],[2.,0.]])
    motion.publish(route,0.);controller.reference(motion.view)
    assert controller.segment==0 and controller.progress<np.sqrt(2)


@pytest.mark.dingo
def test_xy_u_turn_makes_mpc_turn_and_drive_out_with_no_heading_prediction(controller):
    clock,motion=make(controller,duration=25.,goal=[-1.5,.6],boxes=((.9,-1.,1.1,1.),(-3.,1.,1.1,1.2),(-3.,-1.2,1.1,-1.)))
    theta=np.linspace(0,np.pi,100);arc=np.column_stack((.3*np.sin(theta),.3*(1-np.cos(theta))))
    route=np.vstack((arc,np.column_stack((np.linspace(-.02,-2.,100),np.full(100,.6)))))
    motion.publish(route,0.);state=advance(clock,motion,25.)
    assert state['status']=='goal_reached'
    assert np.linalg.norm(state['view'][:2]-[-1.5,.6])<=GOAL_TOLERANCE_M+1e-8
    assert min(r['sweep_margin_m'] for r in motion.trace)>0
    assert max(abs(r['yaw_rate_rad_s']) for r in motion.trace)<=.5+1e-5
    assert max(abs(r['linear_speed_mps']) for r in motion.trace)<=.5+1e-5
    points=np.asarray([r['view'][:2] for r in motion.trace]);delta=np.diff(route,axis=0)
    offset=points[:,None]-route[:-1]
    fraction=((offset*delta).sum(-1)/(delta*delta).sum(-1)).clip(0,1)
    assert np.linalg.norm(offset-fraction[...,None]*delta,axis=-1).min(-1).max()<.02


def test_rear_reference_is_reached_by_turning_never_by_reversing(controller):
    controller.reset()
    reference=np.column_stack((-np.arange(controller.horizon+1)*.05,np.zeros(controller.horizon+1)))
    command,_=controller.solve(np.zeros(4),reference,0.)
    assert command[0]>=0 and abs(command[1])>.1


@pytest.mark.dingo
def test_safe_indoor_curve_does_not_collide_while_tracking(controller):
    from kappanav.evaluation.simulation.navigation import NavigationGrid
    fixture=Path(__file__).parents[1]/'fixtures/indoor_xy_tracking.npz'
    with np.load(fixture) as data:route=data['reference'];start=data['start_view']
    navigation=NavigationGrid(fixture);clock=Clock()
    motion=RealtimeMotion(start,navigation.body_clearance,60.,controller=controller,clock=clock)
    motion.publish(route,0.)
    for tick in range(600):
        motion.control_step();clock.time=(tick+1)*controller.period;state=motion.snapshot()
        if state['status']!='running' or np.linalg.norm(axle_state(state['view'])[:2]-route[-1])<.005:break
    assert state['status']!='collision'
    assert np.linalg.norm(axle_state(state['view'])[:2]-route[-1])<.005
    points=np.asarray([axle_state(row['view'])[:2] for row in motion.trace]);delta=np.diff(route,axis=0)
    # The route leaves backwards: the robot first turns in place about the axle midpoint (base_link swings on
    # its circle, the axle stays off the route by the offset), and from joining the route the axle stays on it.
    offset=points[:,None]-route[:-1];fraction=((offset*delta).sum(-1)/(delta*delta).sum(-1)).clip(0,1)
    deviation=np.linalg.norm(offset-fraction[...,None]*delta,axis=-1).min(-1);joined=int(np.argmax(deviation<.005))
    turn=np.unwrap([row['view'][2] for row in motion.trace[:joined+1]])
    assert abs(turn[-1]-turn[0])>np.deg2rad(90) and deviation[:joined].max()<TURN_CENTRE_FORWARD_OFFSET_M+.005
    assert deviation[joined:].max()<.01


@pytest.mark.parametrize('fixture',['mpc_reversing_turn.npz','mpc_reversing_turn_long.npz'])
def test_reversing_turn_handover_produces_a_valid_realtime_command(controller,fixture):
    # Captured states at which full SQP exhausted100 and500 iterations respectively.
    with np.load(Path(__file__).parents[1]/'fixtures'/fixture) as sample:
        controller.reset()
        command,stats=controller.solve(sample['state'],sample['reference'],float(sample['timestamp']))
    assert np.isfinite(command).all()
    assert np.abs(command).max()<=.5+1e-6
    assert stats['control_bound_residual']<=1e-6


def test_u_turn_reference_speed_stays_bounded_below_by_the_turn_rate_law(controller):
    angles=np.linspace(0,np.pi,40)
    route=np.column_stack((.05*np.sin(angles),.05-.05*np.cos(angles)))
    controller.reset();controller.publish(np.vstack(([[0.,0.]],route[1:])),np.zeros(3))
    assert controller.curvature.max()<=np.pi/(2*.167584539)+1e-6
    reference=controller.reference(np.zeros(3))
    assert np.linalg.norm(np.diff(reference,axis=0),axis=1).max()/.1>.03


@pytest.mark.dingo
def test_reference_speed_is_the_fastest_that_still_stops_at_the_plan_end(controller):
    def first_step(length):
        controller.reset();controller.publish(np.array([[0.,0.],[length,0.]]),np.zeros(3))
        return np.linalg.norm(np.diff(controller.reference(np.zeros(3)),axis=0),axis=1)[0]/.1
    assert first_step(1.)==pytest.approx(.5)      # a short replanned curve no longer crawls
    # stopping distance speed*(response+period) fits what remains ahead of the axle midpoint
    assert first_step(.1)==pytest.approx((.1-TURN_CENTRE_FORWARD_OFFSET_M)/.25)


@pytest.mark.parametrize('ends',[True,False])
@pytest.mark.dingo
def test_contact_ends_the_episode_only_when_collisions_end_it(controller,ends):
    clock=Clock();wall=((1.,-1.,2.,1.),)
    motion=RealtimeMotion([0.,0.,0.],collision(wall),20.,controller=controller,initial_velocity=[.5,0.],clock=clock,collision_ends=ends)
    motion.command[:]=[.5,0.]
    clock.time=4.;state=motion.snapshot()
    assert state['status']==('collision' if ends else 'running')
    stop=1.-ROBOT_FOOTPRINT_RADIUS_M
    assert state['view'][0]<=stop+1e-6 and abs(state['view'][0]-stop)<1e-3
    if ends:return
    assert state['time_s']==4. and abs(state['linear_speed_mps'])<1e-9 and motion.contacts>0
    # Held at the wall it does not slide or drive into it, but it still turns about the axle: facing the wall, the turn
    # would push base_link into it, and the wall pushes it back by less than CONTACT_PIVOT_PUSH_M.
    motion.command[:]=[.5,.5];clock.time=4.2;held=motion.snapshot()
    assert held['view'][0]<=stop+1e-6 and held['view'][2]>.05 and abs(held['view'][1])<.02
    motion.command[:]=[0.,.5];clock.time=4.4;turned=motion.snapshot()
    assert turned['view'][0]<=stop+1e-6 and turned['view'][2]>held['view'][2]+.05
    motion.command[:]=[0.,0.];motion.view[2]=np.pi;motion.command[:]=[.5,0.];clock.time=6.;away=motion.snapshot()
    assert away['view'][0]<stop-.5 and away['status']=='running'


def test_arrival_latches_when_slow_near_the_goal_and_succeeds_four_seconds_later(controller):
    from kappanav.evaluation.simulation.realtime import ARRIVAL_HOLD_S
    clock=Clock()
    motion=RealtimeMotion([0.,0.,0.],collision(((10.,10.,11.,11.),)),30.,[1.,0.],controller=controller,initial_velocity=[.5,0.],clock=clock,arrival_hold_s=ARRIVAL_HOLD_S)
    motion.command[:]=[.5,0.];clock.time=4.;passing=motion.snapshot()
    assert passing['status']=='running' and passing['view'][0]>1.5          # drove through the goal disc at speed: no latch
    assert motion.latched_s is None
    motion.view[:2]=[1.2,0.];motion.command[:]=[0.,0.];clock.time=5.;state=motion.snapshot()
    assert state['status']=='running' and motion.latched_s is not None        # slow near the goal: the timer latches
    latched=motion.latched_s
    motion.view[:2]=[3.,0.];clock.time=latched+ARRIVAL_HOLD_S-.15;state=motion.snapshot()
    assert state['status']=='running'                                         # left the goal; still latched
    clock.time=latched+ARRIVAL_HOLD_S+.05;state=motion.snapshot()
    assert state['status']=='goal_reached'                                    # 40 steps after the latch


def test_navigation_clearance_measures_penetration_so_an_overlapping_start_can_leave(tmp_path):
    import numpy as np
    from kappanav.evaluation.simulation.navigation import NavigationGrid
    free=np.ones((40,40),bool);free[10:30,10:30]=False
    np.savez_compressed(tmp_path/'nav.npz',free=free,clearance_m=np.zeros((40,40),np.float32),origin_xy=np.zeros(2),
        cell_size_m=.05,coordinate_frame='canonical_world_xy')
    grid=NavigationGrid(tmp_path/'nav.npz')
    start=np.array([.6,.6])                                   # 3 cells inside the block's corner
    depth=grid.clearance(np.stack((start,start)),0.)
    assert depth<-.1
    outward=grid.clearance(np.stack((start,start-[.1,0.])),0.)  # toward the nearest free side
    inward=grid.clearance(np.stack((start,start+[.1,.1])),0.)   # toward the block's centre
    assert outward>=depth and inward<depth


def test_compiled_navigation_clearance_equals_the_box_index_composition_bit_for_bit(tmp_path):
    from kappanav.evaluation.simulation.navigation import NavigationGrid
    rng=np.random.default_rng(3)
    free=np.ones((90,70),bool)
    for _ in range(12):
        x,y=rng.integers(0,80),rng.integers(0,60);free[x:x+rng.integers(1,12),y:y+rng.integers(1,12)]=False
    free[[0,-1]]=False;free[:,[0,-1]]=False
    np.savez_compressed(tmp_path/'nav.npz',free=free,clearance_m=np.zeros(free.shape,np.float32),origin_xy=np.array([-1.3,2.7]),
        cell_size_m=.05,coordinate_frame='canonical_world_xy')
    grid=NavigationGrid(tmp_path/'nav.npz')
    def original(path,padding):
        cells=grid.grid.path_cells(path)
        if (cells<0).any() or (cells>=grid.free.shape).any():return -np.inf
        inside=~grid.free[cells[:,0],cells[:,1]]
        if inside.any():return -float(grid.penetration_m[cells[inside,0],cells[inside,1]].max())-padding
        return float(grid.boundary.clearance(path[None],radius=padding)[0])
    low,high=grid.origin-.3,grid.origin+np.array(free.shape)*grid.spacing+.3
    for n in range(4000):
        a=low+rng.random(2)*(high-low)
        path=np.stack((a,a+rng.normal(size=2)*rng.choice([0.,1e-3,.05,.5]))) if n%3 else a+np.cumsum(rng.normal(size=(4,2))*.2,0)
        padding=float(rng.choice([0.,1e-6,.01]))
        assert np.float64(grid.clearance(path,padding)).tobytes()==np.float64(original(path,padding)).tobytes()


def test_contact_pivot_turns_against_a_wall_but_not_into_a_deep_pocket():
    from kappanav.evaluation.simulation.realtime import _contact_pivot, CONTACT_PIVOT_PUSH_M
    wall=collision(((1.,-1.,2.,1.),));stop=1.-ROBOT_FOOTPRINT_RADIUS_M
    view=np.array([stop,0.,0.])
    turned=_contact_pivot(view,.3,wall,0.)
    assert turned is not None and turned[2]==pytest.approx(.3) and wall(np.stack((turned[:2],turned[:2])),0.)>=0.
    assert np.linalg.norm(turned[:2]-view[:2])<TURN_CENTRE_FORWARD_OFFSET_M*.3+CONTACT_PIVOT_PUSH_M
    # Boxed in on every side: no free pose within the push.
    pocket=collision(((stop-1.,.01,stop+1.,1.),(stop-1.,-1.,stop+1.,-.01),(1.,-1.,2.,1.),(-1.,-1.,stop-ROBOT_FOOTPRINT_RADIUS_M-.005,1.)))
    assert _contact_pivot(view,.3,pocket,0.) is None


def test_publish_hands_the_controller_the_observation_pose():
    seen=[]
    class Recorder:
        period=.1
        def reset(self):pass
        def publish(self,route,view):seen.append(np.array(view))
        def next_command(self,state,time_s):return np.array([.5,0.]),{}
    clock=Clock();motion=RealtimeMotion([0.,0.,0.],collision(((10.,10.,11.,11.),)),5.,controller=Recorder(),clock=clock)
    motion.control_steps(2);observed=motion.snapshot()['view'].copy();motion.control_steps(2)
    motion.publish(np.array([[0.,0.],[1.,0.]]),.2,observed)
    np.testing.assert_array_equal(seen[-1],observed)
    motion.publish(np.array([[0.,0.],[1.,0.]]),.4)
    np.testing.assert_array_equal(seen[-1],motion.view)


def test_yaw_response_follows_the_measured_table_and_is_full_only_when_driving():
    from kappanav.evaluation.simulation.realtime import yaw_response, YAW_RESPONSE_GAIN
    assert yaw_response(0.)==YAW_RESPONSE_GAIN[0]<.6 and yaw_response(-.5)==yaw_response(.5)==YAW_RESPONSE_GAIN[-1]
    assert all(a<=b for a,b in zip(YAW_RESPONSE_GAIN,YAW_RESPONSE_GAIN[1:]))


def test_runtime_camera_is_the_rest_pose_of_the_mounted_camera():
    import json
    from kappanav.common.physical import rest_camera_pose
    from kappanav.evaluation.simulation.scene import DEFAULT_RUNTIME
    camera=json.load(open(Path(__file__).resolve().parents[2]/DEFAULT_RUNTIME))
    forward,height,pitch=rest_camera_pose()
    assert camera['camera']['forward_offset_m']==pytest.approx(forward,abs=1e-6)
    assert camera['camera']['height_m']==pytest.approx(height,abs=1e-6)
    assert camera['camera']['downward_pitch_degrees']==pytest.approx(pitch,abs=1e-6)
    m=np.array(camera['camera_to_body']);assert m[0,3]==pytest.approx(forward,abs=1e-6) and m[2,3]==pytest.approx(camera['camera']['height_m'],abs=1e-6)
    assert np.degrees(np.arcsin(-m[2,2]))==pytest.approx(pitch,abs=1e-4)


def test_rectangular_body_clears_a_wall_inside_its_footprint_circle_but_never_straddles_a_curb(tmp_path):
    import kappanav.evaluation.simulation.navigation as nav
    from kappanav.evaluation.simulation.navigation import BODY_GROUND_RADIUS_M, NavigationGrid, save_navigation_grid
    from scipy.ndimage import distance_transform_edt
    radius, r0, spacing = ROBOT_FOOTPRINT_RADIUS_M, BODY_GROUND_RADIUS_M, .01
    half = np.array((.6 * radius, .6 * radius))
    old = nav.BODY_OUTLINE
    try:
        nav.BODY_OUTLINE = nav._outline(half - r0)
        # A wall from y = 1.5 m: the C-space is it grown by the footprint radius, the body ground by r0.
        occupied = np.zeros((400, 200), bool); occupied[:, 150:] = True
        distance = (distance_transform_edt(~occupied) - .5) * spacing       # cell centre to the wall face
        save_navigation_grid(tmp_path / 'wall.npz', distance >= radius, np.zeros(2), spacing, body_free=distance >= r0)
        wall = NavigationGrid(tmp_path / 'wall.npz')
        y = 1.5 - .8 * radius                                                # inside the circle's reach, outside a narrower body
        assert wall.body_clearance([[2., y, 0.]] * 2) == pytest.approx(.2 * radius, abs=spacing)  # the body is 0.2 R clear
        assert wall.body_clearance([[2., 1.5 - .5 * radius, 0.]] * 2) < 0    # a body edge beyond the wall face
        # A curb along y = 1 m: a step edge with no width, which the body may touch but never straddle.
        edge = np.abs((np.arange(200) + .5) * spacing - 1.)[None].repeat(400, 0)
        save_navigation_grid(tmp_path / 'curb.npz', edge >= radius, np.zeros(2), spacing, body_free=edge >= r0)
        curb = NavigationGrid(tmp_path / 'curb.npz')
        assert curb.body_clearance([[2., 1. - half[1] - .05, 0.]] * 2) == pytest.approx(.05, abs=spacing)
        assert curb.body_clearance([[2., 1., 0.]] * 2) < 0                   # astride the edge
        assert curb.body_clearance([[2., 1. + half[1] / 2, 0.]] * 2) < 0      # half over it
        nav.BODY_OUTLINE = None
        assert wall.body_clearance([[2., y, 0.]] * 2) < 0                    # the circle touches the wall
    finally:
        nav.BODY_OUTLINE = old
