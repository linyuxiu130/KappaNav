import numpy as np
import pytest
from kappanav.evaluation.simulation.assets import triangulate_faces, habitat_from_usd
from kappanav.evaluation.simulation.navigation import raster_surface
from kappanav.evaluation.simulation.navverse import heading


def test_navverse_wxyz_heading_and_axis_mapping():
    assert heading([np.cos(.3),0,0,np.sin(.3)])==pytest.approx(.6)
    with pytest.raises(ValueError):heading([0,0,0,0])
    basis=habitat_from_usd('Z',1)
    np.testing.assert_allclose(basis@np.array([1.,2.,102.,1.]),[1.,102.,-2.,1.])


def test_usd_quads_and_holes_keep_authored_winding():
    points=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    f=triangulate_faces(points,np.array([0,1,2,3,0,2,3]),np.array([4,3]),[1])
    np.testing.assert_array_equal(f,[[0,1,2],[0,2,3]])
    tri=points[f];assert (np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])[:,2]>0).all()


def test_source_surface_raster_keeps_slope_and_missing_space():
    tri=np.array([[[0,0,100],[1,0,101],[1,1,101]],[[0,0,100],[1,1,101],[0,1,100]]],float)
    z=raster_surface(tri,np.zeros(2),(20,20),.1)
    np.testing.assert_allclose(z[:10,:10],np.repeat((100+(np.arange(10)+.5)*.1)[:,None],10,1),atol=1e-5)
    assert np.isnan(z[15,15])


def test_wrong_robot_catalog_is_rejected_before_starting_workers(tmp_path):
    import json
    from types import SimpleNamespace
    from kappanav.common.physical import ROBOT
    from kappanav.evaluation.benchmark.run import evaluate
    path=tmp_path/'tasks.json';path.write_text(json.dumps(dict(robot='dingo' if ROBOT.name=='scout_mini' else 'scout_mini')))
    output=tmp_path/'run'
    with pytest.raises(ValueError,match='benchmark robot'):
        evaluate(SimpleNamespace(benchmark=path,output=output,gpus=[0],workers_per_gpu=1))
    assert not output.exists()


def test_descend_follows_the_shortest_path_around_a_wall_and_its_polyline_never_cuts_it():
    from kappanav.common.grid_dijkstra import grid_distances, descend
    from kappanav.global_planner.route import cell_polyline
    free=np.ones((20,20),bool);free[10,:15]=False
    field=grid_distances(free,.05,2,2,np.array([[18,2]]))
    path=descend(field,free,.05,18,2)
    assert (path[0]==[18,2]).all() and (path[-1]==[2,2]).all() and free[path[:,0],path[:,1]].all()
    assert .05*np.linalg.norm(np.diff(path,axis=0),axis=1).sum()==pytest.approx(field[18,2])
    cost=np.where(free,1.,np.inf);start,goal=(path[-1]+.5)*.05,(path[0]+.5)*.05
    polyline=cell_polyline(path[::-1],cost,np.zeros(2),.05,start,goal)
    assert np.allclose(polyline[0],start) and np.allclose(polyline[-1],goal) and len(polyline)>2
    chords=np.concatenate([np.linspace(a,b,400) for a,b in zip(polyline[:-1],polyline[1:])])
    assert free[tuple(np.floor(chords/.05).astype(int).T)].all()


def test_indoor_components_are_the_mostly_covered_ones():
    from kappanav.evaluation.simulation.navverse import indoor
    free=np.zeros((20,20),bool);covered=np.zeros((20,20),bool)
    free[1:4,:]=True;covered[1:4,15:]=True       # a street, partly under an arcade: outdoor
    free[8:18,2:12]=True;covered[8:18,2:12]=True;covered[8,2:12]=False  # a closed shell's floor, 90% under its roof
    inside=indoor(free,covered)
    assert not inside[1:4].any() and inside[8:18,2:12].all() and not inside[~free].any()
