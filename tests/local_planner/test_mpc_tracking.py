from pathlib import Path
import numpy as np
import pytest
from kappanav.local_planner.control.mpc import PathMPC, unicycle_step, body_step, axle_state
from kappanav.common.physical import TURN_CENTRE_FORWARD_OFFSET_M as AXLE


@pytest.fixture(scope='module')
def tracker(tmp_path_factory):
    return PathMPC(tmp_path_factory.mktemp('forward_mpc')/'codegen')


def test_straight_motion_retains_exact_speed_servo_solution():
    state=np.array([0.,0.,0.,0.]);actual=unicycle_step(state,[.5,0.],1.,.15)
    assert actual[0]==pytest.approx(.5*(1-.15*(1-np.exp(-1/.15))),abs=1e-12)
    assert actual[3]==pytest.approx(.5*(1-np.exp(-1/.15)),abs=1e-12)
    assert actual[1]==actual[2]==0


def test_turning_kinematics_changes_body_direction_and_is_interval_independent():
    state=np.array([0.,0.,0.,.3]);whole=unicycle_step(state,[.3,.5],2.,.15)
    pieces=state.copy()
    for dt in [.13,.22,.08,.57,1.]:pieces=unicycle_step(pieces,[.3,.5],dt,.15)
    np.testing.assert_allclose(whole,pieces,rtol=0,atol=1e-12)
    assert whole[2]==pytest.approx(1.)
    assert whole[0]==pytest.approx(.6*np.sin(1.)) and whole[1]==pytest.approx(.6*(1-np.cos(1.)))


def test_stationary_xy_can_rotate_in_the_tracker_without_a_generator_yaw_channel():
    actual=unicycle_step(np.array([1.,2.,.2,0.]),[0.,.5],1.,.15)
    np.testing.assert_array_equal(actual[:2],[1.,2.]);assert actual[2]==pytest.approx(.7)


def test_base_link_trails_the_axle_and_swings_on_its_circle_in_an_in_place_turn():
    start=np.array([1.,2.,.2,0.]);turned=body_step(start,[0.,.5],1.,.15)
    np.testing.assert_allclose(axle_state(turned)[:2],axle_state(start)[:2],rtol=0,atol=1e-12)
    assert turned[2]==pytest.approx(.7) and np.linalg.norm(turned[:2]-start[:2])==pytest.approx(2*AXLE*np.sin(.25))
    moving=np.array([0.,0.,.3,.2]);np.testing.assert_allclose(axle_state(body_step(moving,[.3,.4],.7,.15)),
        unicycle_step(axle_state(moving),[.3,.4],.7,.15),rtol=0,atol=1e-12)


def test_rear_route_turns_in_place_before_forward_motion(tracker):
    tracker.reset();state=np.array([0.,0.,0.,0.])
    reference=np.column_stack((np.linspace(0.,-2.,16),np.zeros(16)))
    commands=[];turning=[]
    for step in range(90):
        command,stats=tracker.solve(state,reference,step*.1)
        commands.append(command);turning.append(stats['turning_in_place'])
        state=body_step(state,command,.1,.15)
        if not stats['turning_in_place'] and command[0]>1e-3:break
    assert turning[0] and commands[0][0]==pytest.approx(0.,abs=1e-8)
    assert any(not flag for flag in turning)
    assert commands[-1][0]>1e-3
    assert all(command[0]>=-1e-5 for command in commands)


def test_turn_mode_hysteresis(tracker):
    tracker.reset();reference=np.column_stack((np.linspace(0.,2.,16),np.zeros(16)))
    for yaw in (65.,55.):
        command,stats=tracker.solve(np.array([0.,0.,np.deg2rad(yaw),0.]),reference,0.)
        assert stats['turning_in_place'] and command[0]==pytest.approx(0.,abs=1e-8)
    _,stats=tracker.solve(np.array([0.,0.,np.deg2rad(19.),0.]),reference,0.)
    assert not stats['turning_in_place']
    _,stats=tracker.solve(np.array([0.,0.,np.deg2rad(55.),0.]),reference,0.)
    assert not stats['turning_in_place']


@pytest.mark.parametrize('name',['mpc_reversing_turn.npz','mpc_reversing_turn_long.npz'])
def test_historical_reverse_state_commands_forward_speed(tracker,name):
    fixture=np.load(Path(__file__).parents[1]/'fixtures'/name)
    tracker.reset();command,_=tracker.solve(fixture['state'],fixture['reference'],float(fixture['timestamp']))
    assert command[0]>=-1e-5


def test_reference_mpc_drives_ahead_reverses_behind_and_stops_after_its_plan(tmp_path):
    from kappanav.local_planner.control.reference_mpc import ReferenceMPC, N
    controller=ReferenceMPC(tmp_path/'reference_mpc')
    ahead=np.column_stack((np.linspace(0,2,30),np.zeros(30)));behind=-ahead[:15]/2
    controller.publish(ahead,np.zeros(3));assert controller.next_command(None,0.)[0][0]==pytest.approx(.5,abs=.01)
    controller.publish(behind,np.zeros(3));assert controller.next_command(None,0.)[0][0]<-.1
    for _ in range(N-1):controller.next_command(None,0.)
    np.testing.assert_array_equal(controller.next_command(None,0.)[0],[0.,0.])


def test_reference_mpc_solves_in_the_frame_the_plan_was_made_from(tmp_path):
    """A plan made at the observation pose and published 0.2 s later drives ahead; solved in the later frame its start
    would lie behind the robot and the solve would back up first."""
    from kappanav.local_planner.control.reference_mpc import ReferenceMPC
    controller=ReferenceMPC(tmp_path/'reference_mpc')
    plan=np.column_stack((np.linspace(0,2,30),np.zeros(30)))
    controller.publish(plan,np.zeros(3))
    assert controller.next_command(None,0.)[0][0]==pytest.approx(.5,abs=.01)
