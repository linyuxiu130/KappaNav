"""Shared source-planner geometry and simulator pose contracts."""
import numpy as np
import pytest
from kappanav.evaluation.simulation.renderer import base_position_from_navmesh
from kappanav.common.geometry import Grid, _plan, _spatial_route, path_length


def source_route(grid,start,goal,yaw):
    """The planner's certified curve along the whole clearance-aware route from a start heading."""
    return _plan(grid,_spatial_route(grid,start,goal)[0],yaw)
from kappanav.common.physical import ROBOT_BASE_HEIGHT_ABOVE_GROUND_M


def test_clearance_aware_planner_is_safe_and_deterministic() -> None:
    free = np.ones((80, 80), dtype=bool)
    free[25:55, 36:44] = False
    from scipy.ndimage import distance_transform_edt

    clearance = (
        distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]
        * 0.05
    )
    clearance[~free] = 0.0
    grid = Grid(free, clearance.astype(np.float32), np.zeros(2), 0.05)

    first = source_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]), 0.)
    second = source_route(grid, np.array([0.6, 2.0]), np.array([3.4, 2.0]), 0.)

    assert grid.safe(first.path_xy)
    assert np.allclose(first.path_xy, second.path_xy)
    assert path_length(first.path_xy) < 4.5


def test_render_pose_uses_the_dingo_base_link_height() -> None:
    floor_position = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    base_position = base_position_from_navmesh(floor_position)

    assert np.array_equal(floor_position, np.array([1.0, 2.0, 3.0]))
    assert base_position == pytest.approx(
        [1.0, 2.0 + ROBOT_BASE_HEIGHT_ABOVE_GROUND_M, 3.0]
    )


def test_clearance_curve_energy_gradient_matches_finite_difference():
    from scipy.interpolate import BSpline
    from scipy.ndimage import distance_transform_edt
    from scipy.optimize._numdiff import approx_derivative
    from kappanav.common.geometry import _travel_objective, _bending_objective
    free = np.ones((80,80),bool)
    free[30:40,30:40] = False
    distance = distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*.05
    grid = Grid(free,distance,np.zeros(2),.05)
    controls = np.array([[.617,.793],[1.037,.719],[1.843,1.017],[2.413,.613]])
    u = np.linspace(0,1,37)
    identity = BSpline([0,0,0,0,1,1,1,1],np.eye(4),3)
    def objective(controls):
        travel, travel_gradient = _travel_objective(
            controls, identity(u), identity.derivative()(u), np.ones(37)/37, grid)
        bending, bending_gradient = _bending_objective(
            controls, identity.derivative()(u), identity.derivative(2)(u), np.ones(37)/37)
        return travel + bending, travel_gradient + bending_gradient
    value, gradient = objective(controls)
    numerical = approx_derivative(lambda x:objective(x.reshape(4,2))[0],controls.ravel(),method='3-point',abs_step=1e-6)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient.ravel(),numerical.ravel(),rtol=2e-4,atol=2e-5)


def test_constraint_cache_tracks_values_when_solver_mutates_its_array():
    from kappanav.common.geometry import _cached_value_and_jacobian
    calls = []
    @_cached_value_and_jacobian
    def constraint(values):
        calls.append(values.copy())
        return values**2, np.diag(2 * values)
    values = np.array([1., 2.])
    first = constraint(values)
    assert constraint(values.copy()) is first
    values[0] = 3.
    second = constraint(values)
    np.testing.assert_array_equal(second[0], [9., 4.])
    np.testing.assert_array_equal(second[1], [[6., 0.], [0., 4.]])
    values[0] = 1.
    np.testing.assert_array_equal(constraint(values)[0], first[0])
    assert len(calls) == 3


def test_clearance_optimization_keeps_a_narrow_feasible_corridor():
    from scipy.ndimage import distance_transform_edt
    free = np.zeros((80,80),bool)
    free[:,28:34] = True
    distance = distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*.05
    grid = Grid(free,distance,np.zeros(2),.05)
    plan = source_route(grid,np.array([.5,1.55]),np.array([3.5,1.55]), 0.)
    assert grid.safe(plan.path_xy)
    assert grid.clearance(plan.path_xy).min() <= .15+1e-6


@pytest.mark.parametrize("goal", [[2.5, 3.5], [1.5, 2.5], [2.3, 2.5]])
def test_side_and_rear_goals_leave_along_body_x(goal):
    from scipy.ndimage import distance_transform_edt
    free = np.ones((100, 100), bool)
    clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * .05
    grid = Grid(free, clearance, np.zeros(2), .05)
    plan = source_route(grid, np.array([2.5, 2.5]), np.array(goal), 0.)
    np.testing.assert_allclose(plan.curve([0., 1.]), [[2.5, 2.5], goal], atol=1e-9)
    tangent = plan.curve.derivative()(0.)
    assert tangent[0]>0 and tangent[1]==pytest.approx(0.,abs=1e-10)
    assert grid.safe(plan.path_xy)
    assert np.max(np.linalg.norm(np.diff(plan.path_xy, axis=0), axis=1)) <= .03 + 1e-6


@pytest.mark.parametrize('narrow', [False, True])
def test_start_facing_the_goal_gives_a_straight_forward_label(narrow):
    from scipy.ndimage import distance_transform_edt
    free = np.ones((80, 40), dtype=bool)
    if narrow:
        free[:, :12] = False
        free[:, 28:] = False
    distance = distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1]*.05
    grid = Grid(free, np.where(free, distance, 0).astype(np.float32), np.zeros(2), .05)
    start, goal = np.array([3.2, 1.]), np.array([.8, 1.])
    plan = source_route(grid, start, goal, np.pi)
    np.testing.assert_array_equal(plan.curve(0.), start)
    tangent = plan.curve.derivative()(0.)
    assert tangent[0] < 0 and abs(tangent[1]) < 1e-8
    assert grid.safe(plan.path_xy)
    assert np.abs(plan.path_xy[:, 1] - 1.).max() < 1e-6


def test_physical_cspace_does_not_add_a_second_clearance_gate():
    from kappanav.common.geometry import MIN_CLEARANCE_M
    free=np.ones((80,40),bool);clearance=np.full(free.shape,.085,np.float32)
    grid=Grid(free,clearance,np.zeros(2),.05)
    start,goal=np.array([.5,1.]),np.array([2.5,1.])
    assert grid.minimum_clearance_m==MIN_CLEARANCE_M==0.
    plan=source_route(grid,start,goal,0.)
    assert grid.safe(plan.path_xy)
    assert not grid.safe(plan.path_xy,minimum_m=.1)


def test_planning_clearance_uses_signed_distance_to_the_actual_cell_boundary():
    free=np.ones((5,5),bool);free[2,2]=False
    grid=Grid(free,np.zeros((5,5)),np.zeros(2),1.)
    assert grid.signed_clearance_m[2,2]==-.5
    assert grid.signed_clearance_m[1,2]==.5
    assert grid.signed_clearance_m[1,1]==pytest.approx(np.sqrt(.5))
    assert grid.signed_clearance_m[0,0]==.5


def _open_grid(shape=(120,120)):
    from scipy.ndimage import distance_transform_edt
    free=np.ones(shape,bool)
    distance=distance_transform_edt(np.pad(free,1))[1:-1,1:-1]*.05-np.sqrt(2)*.025
    return Grid(free,np.maximum(distance,0).astype(np.float32),np.zeros(2),.05)


def test_u_turn_label_leaves_along_body_x_then_turns_back():
    grid=_open_grid()
    start,goal=np.array([3.,3.]),np.array([1.,3.])
    plan=source_route(grid,start,goal,0.)
    tangent=plan.curve.derivative()(0.)
    assert tangent[0]>0 and abs(tangent[1])<1e-8
    assert plan.path_xy[:,0].max()>start[0]+.05
    np.testing.assert_allclose(plan.path_xy[-1],goal,atol=1e-6)
    assert grid.safe(plan.path_xy)


@pytest.mark.dingo
def test_wall_ahead_of_the_start_turns_in_place_toward_the_route():
    from kappanav.local_planner.data.bspline_teachers import expert_route, forward_route, start_heading
    from kappanav.common.geometry import PlanningError
    grid=_open_grid((60,60))
    grid.free[32:,:]=False
    start,goal=np.array([1.585,1.5]),np.array([1.585,.5])
    with pytest.raises(PlanningError):
        source_route(grid,start,goal,0.)
    # The route leaves 90 degrees off heading 0: the controller turns in place, so the curve starts along it.
    route=forward_route(grid,(*start,0.),goal)
    assert route is not None and grid.safe(route)
    np.testing.assert_allclose(route[0],start)
    leaving=route[1]-route[0];heading=start_heading((*start,0.),route)
    assert leaving[1]<0 and abs(np.angle(np.exp(1j*(np.arctan2(leaving[1],leaving[0])-heading))))<.05
    np.testing.assert_allclose(expert_route(grid,(*start,0.),goal),route)


def test_start_heading_keeps_the_body_heading_below_the_controller_turn_threshold():
    from kappanav.local_planner.data.bspline_teachers import start_heading
    from kappanav.local_planner.control.mpc import TURN_ENTER_RAD
    reach=np.array([[0.,0.],[1.,0.]])
    assert start_heading((0.,0.,TURN_ENTER_RAD*.9),reach)==pytest.approx(TURN_ENTER_RAD*.9)
    assert start_heading((0.,0.,TURN_ENTER_RAD*1.1),reach)==pytest.approx(0.)
    assert start_heading((0.,0.,np.pi),reach)==pytest.approx(0.)


def test_boundary_distance_is_exact_and_matches_the_cell_centre_field():
    free=np.ones((7,7),bool);free[3,3]=False
    grid=Grid(free,np.zeros((7,7)),np.zeros(2),1.)
    centres=grid.grid_to_world(np.argwhere(np.ones_like(free)))
    distance,gradient=grid.boundary_distance(centres)
    np.testing.assert_allclose(distance,grid.signed_clearance_m.ravel(),atol=1e-12)
    # Between cell centres the exact distance sees the obstacle corner.
    value,direction=grid.boundary_distance(np.array([[2.9,2.9]]))
    assert value[0]==pytest.approx(np.hypot(.1,.1))
    np.testing.assert_allclose(direction[0],-np.array([1.,1.])/np.sqrt(2))


def test_shared_reach_gives_the_same_extra_goals_and_exploration_targets():
    from types import SimpleNamespace
    from kappanav.local_planner.data.bspline_teachers import exploration_targets, reachable_cells, relabelled_goals
    grid=_open_grid((200,200));grid.free[80:90,20:180]=False
    actor=SimpleNamespace(policy=SimpleNamespace(planning_horizon_m=3.6))
    view=np.array([2.,6.,.4]);reach=reachable_cells(grid,view)
    first=relabelled_goals(grid,view,np.random.default_rng(3))
    second=relabelled_goals(grid,view,np.random.default_rng(3),reach)
    assert len(first)==3 and all(np.array_equal(a,b) for a,b in zip(first,second))
    targets=exploration_targets(actor,grid,view)
    assert len(targets) and all(np.array_equal(a,b) for a,b in zip(targets,exploration_targets(actor,grid,view,reach)))
    assert reachable_cells(grid,np.array([-1.,0.,0.])) is None


def test_reachable_cells_are_the_finite_grid_distances():
    from kappanav.local_planner.data.bspline_teachers import reachable_cells
    from kappanav.common.grid_dijkstra import grid_distances
    rng=np.random.default_rng(0)
    for trial in range(10):
        grid=_open_grid((60,60));grid.free[:]=rng.random((60,60))>.3
        here=np.argwhere(grid.free)[rng.integers(grid.free.sum())]
        view=np.array([*grid.grid_to_world(here[None])[0],0.])
        finite=np.isfinite(grid_distances(grid.free,grid.cell_size_m,int(here[0]),int(here[1]))).reshape(grid.free.shape)
        np.testing.assert_array_equal(reachable_cells(grid,view),finite)
