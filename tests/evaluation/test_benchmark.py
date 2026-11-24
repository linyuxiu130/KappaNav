import json
from pathlib import Path
import numpy as np
from kappanav.evaluation.simulation.navigation import NavigationGrid
from kappanav.evaluation.simulation.tasks import sample_episodes
from kappanav.evaluation.benchmark.coverage import ExplorationCoverage


def save_grid(path,free,spacing=1.,origin=(0.,0.)):
    np.savez(path,free=free,clearance_m=free.astype(float),origin_xy=origin,cell_size_m=spacing)


def test_collision_traces_short_corner_crossing_and_closed_boundaries(tmp_path):
    free=np.ones((5,5),bool);free[2,2]=False
    path=tmp_path/'grid.npz';save_grid(path,free);navigation=NavigationGrid(path)
    assert navigation.clearance([[1.5,2.5],[3.5,2.5]])<0
    assert navigation.clearance([[1.5,2.],[3.5,2.]])<0
    assert navigation.clearance([[1.5,1.5],[3.5,1.5]])>0
    assert navigation.clearance([[-.01,1.],[1.,1.]])<0


def test_maze_sampling_uses_geodesic_fifty_to_hundred_meters(tmp_path):
    free=np.zeros((43,11),bool)
    for y in (1,5,9):free[1:42,y]=True
    free[41,1:6]=True;free[1,5:10]=True
    path=tmp_path/'maze.npz';save_grid(path,free,spacing=.5)
    row=dict(scene_id='maze/test',source='maze',navigation=str(path))
    episodes=sample_episodes(row,seed=42,count=10,distance_range_m=(50.,100.))
    assert episodes==sample_episodes(row,seed=42,count=10,distance_range_m=(50.,100.))
    assert all(50<=episode['shortest_path_m']<=100 for episode in episodes)
    assert all(np.linalg.norm(np.asarray(e['start'][:2])-e['goal'])<50 for e in episodes)


def test_coverage_keeps_cumulative_area_during_return_to_initial_view(tmp_path):
    free=np.ones((80,80),bool);path=tmp_path/'grid.npz';save_grid(path,free,spacing=.05,origin=(-2.,-2.))
    runtime=json.loads((Path(__file__).resolve().parents[2]/'configs/runtime.json').read_text())
    coverage=ExplorationCoverage(NavigationGrid(path),[0.,0.,0.],np.asarray(runtime['intrinsics']),np.asarray(runtime['camera_to_body']))
    depth=np.ones((128,224),np.float16)
    coverage.update(depth,[0.,0.,0.],0.)
    coverage.update(depth,[0.,0.,0.],1.)
    assert coverage.areas==[0.,0.]
    coverage.update(depth,[0.,0.,np.pi],2.)
    assert coverage.areas[-1]>0
    coverage.update(depth,[0.,0.,0.],3.)
    assert coverage.areas[-1]==coverage.areas[-2]


def test_sampling_rejects_a_map_whose_pairs_are_already_within_goal_tolerance(tmp_path):
    import pytest
    path=tmp_path/'tiny.npz';save_grid(path,np.ones((2,2),bool),spacing=.1)
    with pytest.raises(ValueError,match='Cannot sample'):
        sample_episodes(dict(scene_id='tiny',navigation=str(path)),seed=42,count=1)


def test_execution_uses_actual_cspace_boundary_not_zero_free_cell_clearance(tmp_path):
    path=tmp_path/'zero_margin.npz';free=np.ones((5,5),bool);free[3:]=False
    np.savez(path,free=free,clearance_m=np.zeros((5,5)),origin_xy=np.zeros(2),cell_size_m=1.)
    navigation=NavigationGrid(path)
    assert navigation.clearance([[1.5,1.5],[1.6,1.5]],.01)>1.
    assert navigation.clearance([[2.9,1.5],[2.999,1.5]],.002)<0
    assert navigation.clearance([[2.9,1.5],[3.,1.5]])<0


def test_planner_rejects_a_corner_cut_shorter_than_its_old_sample_spacing(tmp_path):
    free=np.ones((3,3),bool);free[1,1]=False
    path=tmp_path/'corner.npz';save_grid(path,free)
    navigation=NavigationGrid(path)
    crossing=np.array([[.994,1.008],[1.008,.994]])
    assert not navigation.grid.safe(crossing)
    assert navigation.clearance(crossing)<0


def test_exploration_expert_targets_unobserved_space_not_an_already_reached_centroid(tmp_path):
    from types import SimpleNamespace
    from kappanav.evaluation.benchmark.expert import exploration_route, TARGET_REACHED_M
    path=tmp_path/'expert.npz';free=np.ones((12,5),bool);save_grid(path,free)
    navigation=NavigationGrid(path);known=np.zeros_like(free);known[2:5]=True
    view=np.array([2.9,1.5,0.])
    route=exploration_route(navigation,SimpleNamespace(reachable=free,known=known),view)
    np.testing.assert_equal(route[0],view[:2])
    assert np.linalg.norm(route[-1]-view[:2])>TARGET_REACHED_M
    assert route[-1,0]>10. and navigation.grid.safe(route)


def test_exploration_expert_plans_from_a_start_inside_an_obstacle_edge(tmp_path):
    # Contact lets a body overlap an obstacle edge; the reference snaps to the nearest traversable cell.
    from types import SimpleNamespace
    from kappanav.evaluation.benchmark.expert import exploration_route
    free=np.ones((60,20),bool);free[:,:3]=False
    path=tmp_path/'wall.npz';np.savez(path,free=free,clearance_m=free.astype(float),origin_xy=(0.,0.),cell_size_m=.05)
    navigation=NavigationGrid(path)
    view=np.array([.5,.125,0.])                  # cell (10,2): inside the obstacle, one cell from free space
    assert not free[10,2]
    known=np.zeros_like(free);known[:15]=True
    route=exploration_route(navigation,SimpleNamespace(reachable=free,known=known),view)
    assert route is not None and np.allclose(route[0],view[:2]) and route[-1,0]>2.
    assert exploration_route(navigation,SimpleNamespace(reachable=free,known=known),np.array([.5,-1.,0.])) is None
