import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from kappanav.evaluation.simulation.navigation import save_navigation_grid
from kappanav.evaluation.simulation.wanderland.collider import ground_surface, boundary_walls, rebuild
from kappanav.evaluation.simulation.wanderland.prepare import convert_episodes, rasterize_navigation, REQUIRED


def cloud(height, *, dense=True):
    x,y=np.meshgrid(np.arange(40)*.05+.025,np.arange(40)*.05+.025,indexing='ij')
    xy=np.c_[x.ravel(),y.ravel()]
    if dense:xy=np.concatenate([xy+offset for offset in [(-.002,-.002),(.002,-.002),(-.002,.002),(.002,.002)]])
    return np.c_[xy,height(xy)]


def grey(points):
    return np.full((len(points),3),128,np.uint8)


def coloured(points,rgb=(128,128,128)):
    return points,np.tile(np.asarray(rgb,np.uint8),(len(points),1))


def test_grass_ground_is_smoothed_and_grey_kerb_is_not():
    rng=np.random.default_rng(0)
    noise=lambda xy:rng.choice([0.,.04],len(xy),p=[.9,.1])         # blades lift a few cells by 4 cm
    pts,_=coloured(cloud(lambda xy:noise(xy)*(xy[:,0]<1.)+(xy[:,0]>=1.)*.15))
    rgb=np.where((pts[:,0]<1.)[:,None],np.uint8([60,140,50]),np.uint8([128,128,128])).astype(np.uint8)
    ground,_,_=ground_surface(pts,rgb)
    assert np.nanmax(np.abs(np.diff(ground[3:16,3:37],axis=0)))<.01   # grass: no cm-scale spikes left
    assert np.nanmax(np.diff(ground[15:25,8:32],axis=0))>.1            # the grey kerb at x = 1 m stays a step


def test_sparse_continuous_slope_has_no_coarse_terraces():
    points=cloud(lambda xy:.10*xy[:,0]+.05*xy[:,1],dense=False)
    ground,_,_=ground_surface(points,grey(points))
    inner=ground[6:-6,6:-6]
    assert np.isfinite(inner).all()
    assert np.abs(np.diff(inner,axis=0)).max()<.012
    assert np.abs(np.diff(inner,axis=1)).max()<.012


def test_real_kerb_and_narrow_raised_floor_survive():
    ground,_,_=ground_surface(*coloured(cloud(lambda xy:(xy[:,0]>=1.)*.15)))
    assert np.nanmedian(ground[10:15,8:32])<.02
    assert np.nanmedian(ground[25:30,8:32])>.13
    assert np.nanmax(np.diff(ground[15:25,8:32],axis=0))>.1
    ground,_,_=ground_surface(*coloured(cloud(lambda xy:((xy[:,0]>.5)&(xy[:,0]<1.5))*.6)))
    assert np.nanmedian(ground[15:25,8:32])>.55


def test_small_overhead_patch_does_not_create_wall_to_ground():
    ground,_,_=ground_surface(*coloured(cloud(lambda xy:((xy[:,0]>.8)&(xy[:,0]<1.05)&(xy[:,1]>.8)&(xy[:,1]<1.05))*2.6)))
    assert np.isfinite(ground[16:21,16:21]).all()
    assert np.max(ground[16:21,16:21])<.05


def test_terrace_is_smoothed_but_step_is_preserved():
    low,_,_=ground_surface(*coloured(cloud(lambda xy:(xy[:,0]>=1.)*.044)))
    high,_,_=ground_surface(*coloured(cloud(lambda xy:(xy[:,0]>=1.)*.051)))
    assert np.nanmax(np.diff(low[15:25,8:32],axis=0))<.03
    assert np.nanmax(np.diff(high[15:25,8:32],axis=0))>=.05


def test_filled_gap_does_not_reintroduce_terrace():
    points=cloud(lambda xy:(xy[:,0]>=1.)*.044)
    points=points[(points[:,0]<.9)|(points[:,0]>1.1)]
    ground,_,_=ground_surface(points,grey(points))
    assert np.isfinite(ground[16:24,8:32]).all()
    assert np.max(np.abs(np.diff(ground[16:24,8:32],axis=0)))<.03


def test_boundary_walls_close_every_open_ground_edge():
    ground=np.full((6,6),np.nan,np.float32);ground[1:5,1:5]=.2;ground[2,2]=np.nan  # a ring of ground around a hole
    v,f=boundary_walls(ground,np.zeros(2),.05)
    assert len(f)==2*(16+4)                    # 16 outer sides of the 4x4 block, 4 sides facing the hole
    assert np.isclose(v[:,2].min(),.2-.3) and np.isclose(v[:,2].max(),.2+1.)
    # the walls stand on cell faces: x or y is a multiple of the cell size for both corners of every quad
    on_face=lambda a:np.isclose(np.round(a/.05)*.05,a)
    assert (on_face(v[:,0])|on_face(v[:,1])).all()


def test_rebuilt_mesh_includes_walls():
    verts,faces,ground,_,info=rebuild(*coloured(cloud(lambda xy:0*xy[:,0])))
    assert info['boundary_wall_triangles']>0 and verts[:,2].max()>.9


def test_tasks_use_filtered_grid_and_preserve_upper_floor_rejection(tmp_path):
    free=np.ones((40,40),bool);free[:2,:]=False;free[18:22,:]=False
    nav=tmp_path/'nav.npz';save_navigation_grid(nav,free,np.zeros(2),.05)
    raw=[dict(episode_id=1,start_position=[.04,-1.2,.5],goals=[dict(position=[.7,-1.2,.5])],start_rotation=[0,0,0,1]),
         dict(episode_id=2,start_position=[.2,-6.,.5],goals=[dict(position=[.7,-1.2,.5])]),
         dict(episode_id=3,start_position=[.2,-1.2,.5],goals=[dict(position=[1.5,-1.2,.5])])]
    (tmp_path/'episodes.json').write_text(json.dumps(dict(episodes=raw)))
    ground=RegularGridInterpolator((np.arange(42)*.05,np.arange(42)*.05),np.zeros((42,42)))
    eps=convert_episodes(tmp_path,ground,nav)
    assert eps[0]['valid'] and eps[0]['start_xy'][0]>.1
    assert eps[0]['start_original_xyz'][:2]==[.04,.5]
    assert 'start_not_ground_level' in eps[1]['invalid_reasons']
    assert 'no_path' in eps[2]['invalid_reasons']
    assert 'scene.usdz' not in REQUIRED
    with np.load(nav) as n:
        for e in eps:
            if e['valid']:
                for key in ('start','goal'):
                    ij=np.floor(np.asarray(e[key+'_xy'])/.05).astype(int)
                    assert n['free'][tuple(ij)]


def test_upper_walkable_surface_does_not_hide_ground(tmp_path):
    from kappanav.common.geometry import Grid
    floor=np.array([[[0,0,0],[1,0,0],[1,1,0]],[[0,0,0],[1,1,0],[0,1,0]]],float)
    roof=floor+np.array([0,0,3.])
    canonical=np.concatenate([floor,roof])
    nav,_=rasterize_navigation(canonical,tmp_path,lambda xy:np.zeros(len(xy)),np.array([[.5,.5]]),body_triangles=canonical)
    grid=Grid.load(tmp_path/'navigation.npz');cell=grid.world_to_grid([.5,.5])
    assert grid.free[tuple(cell)]
    assert nav['ground_z'][tuple(cell)]==0.
    with np.load(tmp_path/'navigation.npz') as data:assert data['body_free'][tuple(cell)]
