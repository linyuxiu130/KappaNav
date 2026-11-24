import numpy as np
from kappanav.global_planner.route import plan_route, NoRoute, Subgoals, doubtful_stretches, SUBGOAL_SPACING_M


def test_route_goes_around_a_wall_and_never_cuts_it():
    p=np.full((100,100),255,np.float32);p[50,0:90]=0             # a wall with a gap at the far end
    path=plan_route(p,np.zeros(2),.05,(1.,1.),(4.,1.))
    q=np.concatenate([np.linspace(a,b,200) for a,b in zip(path[:-1],path[1:])])
    ij=np.floor(q/.05).astype(int)
    assert (p[ij[:,0],ij[:,1]]>0).all() and np.allclose(path[0],(1.,1.)) and np.allclose(path[-1],(4.,1.))
    assert np.linalg.norm(np.diff(path,axis=0),axis=1).sum()>7  # around the wall end at y = 4.5, not the straight 3 m


def test_disconnected_goal_is_reported():
    p=np.full((60,60),255,np.float32);p[30,:]=0
    try:plan_route(p,np.zeros(2),.05,(.5,.5),(2.5,.5));raise AssertionError
    except NoRoute:pass


def test_subgoals_are_the_corners_and_even_splits_of_longer_legs():
    g=Subgoals([[0,0],[10,0],[10,3]])
    assert np.allclose(g.points,[[5,0],[10,0],[10,3]])                # 10 m leg in two, the corner, the goal
    assert np.diff(np.r_[0.,g.along]).max()<=SUBGOAL_SPACING_M


def test_subgoals_advance_on_reach_or_progress_and_never_jump_to_a_nearby_later_leg():
    g=Subgoals([[0,0],[10,0],[10,10]])
    assert np.allclose(g.target(np.array([0.,0.])),(5,0))
    assert np.allclose(g.target(np.array([3.5,.5])),(10,0))          # within reach of (5, 0)
    assert np.allclose(g.target(np.array([10.5,1.])),(10,5))         # past the corner along the route
    u=Subgoals([[0,0],[20,0],[20,1],[0,1]])                          # the way back passes 1 m from the way out
    assert np.allclose(u.target(np.array([.5,.9])),u.points[0])


def test_no_subgoal_inside_a_doubtful_stretch_but_one_at_its_end():
    p=np.full((400,20),255,np.float32);p[100:160]=25                   # a doubtful band across x in [5, 8) m
    route=np.array([[0.,.5],[18.,.5]])
    (a,b),=doubtful_stretches(route,p,np.zeros(2),.05)
    assert abs(a-5.)<.3 and abs(b-8.)<.3
    g=Subgoals(route,[(a,b)])
    assert not ((g.along>a)&(g.along<b)).any() and np.isclose(g.along,b).any() and np.allclose(g.points[-1],route[-1])


def test_the_route_prefers_road_to_grass_until_the_detour_is_too_long():
    def length(detour):
        v=np.full((100,100),130,np.float32);v[:,:]=130;v[10:90,detour]=255;v[10,10:detour]=v[89,10:detour]=255
        r=plan_route(v,np.zeros(2),.1,(1.05,1.05),(8.95,1.05));return np.linalg.norm(np.diff(r,axis=0),axis=1).sum()
    assert length(25)>9 and length(90)<8                              # road cost 1, grass 2 per metre
