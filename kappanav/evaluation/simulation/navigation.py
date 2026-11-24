"""Shared immutable navigation C-space; it is never supplied to the policy."""
from pathlib import Path
from functools import cached_property
import json
from kappanav.local_planner.data.privileged import SourceConfigurationGrid
from kappanav.local_planner.data.observed_geometry import BoxIndex
from kappanav.common.physical import (ROBOT, ROBOT_FOOTPRINT_RADIUS_M, ROBOT_COLLISION_HEIGHT_M,
                                      MAXIMUM_TRAVERSABLE_HEIGHT_M, MAXIMUM_TRAVERSABLE_SLOPE_DEGREES)
from kappanav.evaluation.simulation import scene
import numpy as np
from scipy.ndimage import distance_transform_edt
from numba import njit
from kappanav.common.geometry import Grid, _path_cells


class NavigationGrid:
    def __init__(self,path):
        self.grid=Grid.load(Path(path))
        with np.load(path) as data:
            self.ground_z=data['ground_z'] if 'ground_z' in data else None
            self.body_free=data['body_free'] if 'body_free' in data else None
        if self.ground_z is not None:
            valid=np.isfinite(self.ground_z)
            if self.ground_z.shape != self.grid.free.shape or not valid[self.grid.free].all():
                raise ValueError('terrain must have a finite height for every navigation cell')
            # Extend across blocked cells for continuous camera height at a contact boundary.
            nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
            self.ground_z=self.ground_z[tuple(nearest)]
        self.free=self.grid.free
        self.origin=self.grid.origin_xy
        self.spacing=self.grid.cell_size_m
        self.signed=np.where(self.free,self.grid.clearance_m,-self.spacing)

    def ground_height(self,xy):
        """Ground height in canonical Z-up coordinates; older flat-floor assets have height zero."""
        if self.ground_z is None:return 0.
        from scipy.ndimage import map_coordinates
        cell=(np.asarray(xy)-self.origin)/self.spacing
        if not ((cell>=0).all() and (cell<self.free.shape).all()):
            raise ValueError(f'position outside terrain: {xy}')
        return float(map_coordinates(self.ground_z,(cell-.5)[:,None],order=1,mode='nearest')[0])

    @cached_property
    def oracle(self):
        """The source C-space oracle on this grid (trajectory metrics), built once per grid."""
        from kappanav.local_planner.data.privileged import SourceConfigurationSpaceQuery
        return SourceConfigurationSpaceQuery((SourceConfigurationGrid(self.signed,self.origin,self.spacing,'canonical_world_xy'),))

    @cached_property
    def boundary(self):
        """The C-space boundary segments as boxes, indexed for the exact contact distance."""
        segments=SourceConfigurationGrid(self.signed,self.origin,self.spacing,'canonical_world_xy').boundary_segments
        return BoxIndex(np.concatenate((segments.min(1),segments.max(1)),1))

    @cached_property
    def contact_buckets(self):
        """The boundary boxes of `boundary` bucketed on a uniform grid (CSR), for the compiled contact query."""
        boxes=self.boundary.boxes;origin=self.origin.astype(np.float64)
        if not len(boxes):return boxes,np.zeros(2,np.int64),np.zeros(0,np.int64),origin,np.zeros(2,np.int64)
        low=np.floor((boxes[:,:2]-origin)/CONTACT_BUCKET_M).astype(np.int64)
        high=np.floor((boxes[:,2:]-origin)/CONTACT_BUCKET_M).astype(np.int64)
        shift=low.min(0);low-=shift;high-=shift;shape=high.max(0)+1
        members=[[] for _ in range(int(shape[0]*shape[1]))]
        for box,((lx,ly),(hx,hy)) in enumerate(zip(low,high)):
            for i in range(lx,hx+1):
                for j in range(ly,hy+1):members[i*shape[1]+j].append(box)
        starts=np.r_[0,np.cumsum([len(m) for m in members])].astype(np.int64)
        flat=np.asarray([box for m in members for box in m],dtype=np.int64)
        return boxes,starts,flat,origin+shift*CONTACT_BUCKET_M,shape.astype(np.int64)

    def clearance(self,path,padding=0.):
        """Exact signed contact distance of a polyline: -inf outside the grid, minus the deepest penetration if
        it touches a blocked cell, else its distance to the C-space boundary; minus `padding`. One compiled call;
        the boundary minimum is the exact one of BoxIndex.clearance (same boxes, same arithmetic)."""
        boxes,starts,members,bucket_origin,shape=self.contact_buckets
        return float(_contact_clearance(np.ascontiguousarray(path,dtype=np.float64),self.origin.astype(np.float64),float(self.spacing),
            self.free,self.penetration_m,float(padding),boxes,starts,members,bucket_origin,shape))

    def body_clearance(self,poses,padding=0.):
        """Signed contact distance of the robot body over a short sweep of poses (rows x, y, yaw), minus `padding`.
        A circular body: clearance of the base_link path in the C-space. A rectangular body
        (ROBOT.body_half_extent_m): the least signed distance into the body ground (body_signed_m) of its outline
        shrunk by BODY_GROUND_RADIUS_M, at each pose."""
        poses=np.asarray(poses,dtype=np.float64)
        if BODY_OUTLINE is None:return self.clearance(poses[:,:2],padding)
        return _outline_clearance(self.body_signed_m,self.origin,float(self.spacing),poses[:,:2],np.cos(poses[:,2]),np.sin(poses[:,2]),
            BODY_OUTLINE)-padding

    @cached_property
    def body_signed_m(self):
        """Signed distance to the edge of the body ground (body_free), on cell centres: positive on it, negative off
        it, zero half way between a body-ground cell and its blocked neighbour."""
        if self.body_free is None:
            raise ValueError(f'{ROBOT.name} needs the body ground layer (body_free) in its navigation grid; rebuild it')
        inside=distance_transform_edt(self.body_free);outside=distance_transform_edt(~self.body_free)
        return np.where(self.body_free,inside-.5,.5-outside)*self.spacing

    @cached_property
    def penetration_m(self):
        """How deep each blocked cell lies inside the C-space obstacle: distance to the nearest free cell."""
        return (distance_transform_edt(~self.free)*self.spacing).astype(np.float64)


CONTACT_BUCKET_M=.25


# A rectangular body is tested against the body ground: the robot's navmesh built with this small radius instead of
# the footprint radius. A curb or other step edge has no width, so the footprint-radius C-space (obstacles grown by
# 0.42 m) cannot stop a rectangle from straddling it: the edge lies within 0.42 m of free C-space on both sides. On
# the body ground the edge is a band 2 x BODY_GROUND_RADIUS_M wide, which an outline shrunk by the same radius cannot
# cross; against a wall the shrunk outline on the eroded ground is the body against the wall.
BODY_GROUND_RADIUS_M=.10


def _outline(half_extent,spacing=.02):
    """Points every `spacing` m around a rectangle of the given half length and half width, corners included."""
    a,b=half_extent
    edge=lambda half,n:np.linspace(-half,half,n)
    na,nb=int(np.ceil(2*a/spacing))+1,int(np.ceil(2*b/spacing))+1
    return np.concatenate((np.stack((edge(a,na),np.full(na,b)),1),np.stack((edge(a,na),np.full(na,-b)),1),
        np.stack((np.full(nb,a),edge(b,nb)),1),np.stack((np.full(nb,-a),edge(b,nb)),1)))


BODY_OUTLINE=None if ROBOT.body_half_extent_m is None else _outline(np.asarray(ROBOT.body_half_extent_m)-BODY_GROUND_RADIUS_M)


@njit(cache=True,nogil=True)
def _outline_clearance(distance,origin,spacing,positions,cosine,sine,outline):
    """The least bilinear sample of the cell-centred `distance` grid over the outline at each pose; a point off the
    grid counts as -1 (blocked). Operation for operation scipy.ndimage.map_coordinates(order=1, mode='constant',
    cval=-1) on the same points."""
    rows,columns=distance.shape;least=np.inf
    for pose in range(len(positions)):
        c=cosine[pose];s=sine[pose]
        for point in range(len(outline)):
            u=(positions[pose,0]+c*outline[point,0]-s*outline[point,1]-origin[0])/spacing-.5
            v=(positions[pose,1]+s*outline[point,0]+c*outline[point,1]-origin[1])/spacing-.5
            if u<0 or v<0 or u>rows-1 or v>columns-1:value=-1.
            else:
                i=int(np.floor(u));j=int(np.floor(v));a=u-i;b=v-j;i1=min(i+1,rows-1);j1=min(j+1,columns-1)
                value=distance[i,j]*(1-a)*(1-b)+distance[i,j1]*(1-a)*b+distance[i1,j]*a*(1-b)+distance[i1,j1]*a*b
            if value<least:least=value
    return least


@njit(cache=True,nogil=True)
def _box_squared(ax,ay,bx,by,dx,dy,denominator,lx,ly,hx,hy):
    """observed_geometry._minimum_squared for one segment and one box, operation for operation."""
    enter=0.;leave=1.;intersects=True
    if dx==0:
        if ax<lx or ax>hx:intersects=False
    else:
        a=(lx-ax)/dx;b=(hx-ax)/dx
        enter=max(enter,min(a,b));leave=min(leave,max(a,b))
    if dy==0:
        if ay<ly or ay>hy:intersects=False
    else:
        a=(ly-ay)/dy;b=(hy-ay)/dy
        enter=max(enter,min(a,b));leave=min(leave,max(a,b))
    if intersects and enter<=leave:return 0.
    ex=max(lx-ax,ax-hx,0.);ey=max(ly-ay,ay-hy,0.)
    best=ex*ex+ey*ey
    ex=max(lx-bx,bx-hx,0.);ey=max(ly-by,by-hy,0.)
    best=min(best,ex*ex+ey*ey)
    for corner in range(4):
        cx=lx if corner==0 or corner==3 else hx
        cy=ly if corner<2 else hy
        fraction=((cx-ax)*dx+(cy-ay)*dy)/denominator
        fraction=min(1.,max(0.,fraction))
        ex=cx-(ax+fraction*dx);ey=cy-(ay+fraction*dy)
        best=min(best,ex*ex+ey*ey)
    return best


@njit(cache=True,nogil=True)
def _bucket_squared(path,delta,denominator,boxes,starts,members,bucket,best):
    for k in range(starts[bucket],starts[bucket+1]):
        lx,ly,hx,hy=boxes[members[k]]
        for segment in range(len(path)-1):
            value=_box_squared(path[segment,0],path[segment,1],path[segment+1,0],path[segment+1,1],
                delta[segment,0],delta[segment,1],denominator[segment],lx,ly,hx,hy)
            if value<best:best=value
    return best


@njit(cache=True,nogil=True)
def _contact_clearance(path,origin,spacing,free,penetration,padding,boxes,starts,members,bucket_origin,shape):
    coordinates=np.empty_like(path)
    for i in range(len(path)):
        for axis in range(2):coordinates[i,axis]=(path[i,axis]-origin[axis])/spacing
    cells=_path_cells(coordinates)
    for i in range(len(cells)):
        if cells[i,0]<0 or cells[i,1]<0 or cells[i,0]>=free.shape[0] or cells[i,1]>=free.shape[1]:return -np.inf
    inside=False;deepest=-np.inf
    for i in range(len(cells)):
        if not free[cells[i,0],cells[i,1]]:
            inside=True;deepest=max(deepest,penetration[cells[i,0],cells[i,1]])
    if inside:return -deepest-padding
    if len(boxes)==0:return np.inf-padding
    delta=np.empty((len(path)-1,2));denominator=np.empty(len(path)-1)
    for segment in range(len(path)-1):
        delta[segment,0]=path[segment+1,0]-path[segment,0];delta[segment,1]=path[segment+1,1]-path[segment,1]
        denominator[segment]=max(delta[segment,0]**2+delta[segment,1]**2,1e-20)
    # Rings of buckets around the path's bounding box: a box first met in ring k lies at least (k-1) buckets
    # from the path, so the search stops once that bound exceeds the best distance; the minimum is exact.
    low=np.empty(2,np.int64);high=np.empty(2,np.int64)
    for axis in range(2):
        low[axis]=np.int64(np.floor((path[:,axis].min()-bucket_origin[axis])/CONTACT_BUCKET_M))
        high[axis]=np.int64(np.floor((path[:,axis].max()-bucket_origin[axis])/CONTACT_BUCKET_M))
    best=np.inf;ring=0
    while True:
        if ring>0:
            covered=ring-1  # rings 0..ring-1 are done
            if low[0]-covered<=0 and low[1]-covered<=0 and high[0]+covered>=shape[0]-1 and high[1]+covered>=shape[1]-1:break
            bound=(ring-1)*CONTACT_BUCKET_M-1e-6
            if bound>0 and bound*bound>best:break
        for i in range(max(low[0]-ring,0),min(high[0]+ring,shape[0]-1)+1):
            border=ring==0 or i==low[0]-ring or i==high[0]+ring
            for j in range(max(low[1]-ring,0),min(high[1]+ring,shape[1]-1)+1):
                if border or j==low[1]-ring or j==high[1]+ring:
                    best=_bucket_squared(path,delta,denominator,boxes,starts,members,i*shape[1]+j,best)
        ring+=1
    return np.sqrt(best)-padding


def save_navigation_grid(path,free,origin,spacing,*,ground_z=None,body_free=None):
    """The robot's C-space raster: a cell is free iff its centre is in the C-space (the navmesh already includes the
    robot radius), with each free cell's exact distance to the free/blocked cell interface. One geometry serves
    contact, planning and supervision. An extra sqrt(2)*cell erosion (every cell entirely inside the C-space) made the
    simulated robot ~7 cm wider than the Dingo: it removed 16% of the navmesh C-space and 8-26% of the official
    reference navigable area of the evaluation scenes. Planned curves keep LABEL_CLEARANCE_M from this boundary.
    body_free: the body ground of a rectangular robot (BODY_GROUND_RADIUS_M) on the same grid."""
    free=np.asarray(free,dtype=bool)
    clearance=Grid(free,np.zeros(free.shape,np.float32),np.asarray(origin,dtype=np.float64),float(spacing)).signed_clearance_m
    np.savez_compressed(path,free=free,clearance_m=np.maximum(clearance,0).astype(np.float32),
        origin_xy=origin,cell_size_m=spacing,coordinate_frame='canonical_world_xy',
        **({} if ground_z is None else dict(ground_z=ground_z)),**({} if body_free is None else dict(body_free=np.asarray(body_free,bool))))


def navigation_from_navmesh(navmesh,path,spacing=.05):
    """save_navigation_grid from a saved Habitat navmesh, rasterized at `spacing` (canonical XY is (X,-Z))."""
    import habitat_sim
    finder=habitat_sim.PathFinder();finder.load_nav_mesh(str(navmesh))
    lower,_=finder.get_bounds();top=finder.get_topdown_view(spacing,0.,.05)
    origin=np.array([lower[0]-.5*spacing,-lower[2]-(top.shape[0]-.5)*spacing])
    save_navigation_grid(path,top.T[:,::-1].copy(),origin,spacing)


def navmesh_settings(radius=ROBOT_FOOTPRINT_RADIUS_M):
    """Recast settings of the selected robot: footprint radius (or the body ground's), body height, step and slope.
    Every scene source (benchmark, Wanderland, NavVerse) builds its navmesh with these."""
    import habitat_sim
    settings=habitat_sim.NavMeshSettings();settings.set_defaults()
    settings.cell_size=.025;settings.cell_height=.01
    settings.agent_radius=radius;settings.agent_height=ROBOT_COLLISION_HEIGHT_M
    settings.agent_max_climb=MAXIMUM_TRAVERSABLE_HEIGHT_M;settings.agent_max_slope=MAXIMUM_TRAVERSABLE_SLOPE_DEGREES
    # Furniture placed by the scene instance (HSSD) is part of the obstacle field; the other sources have none.
    settings.include_static_objects=True
    return settings


def prepare_scene(row,destination):
    folder=destination/row['scene_id'];folder.mkdir(parents=True)
    path=folder/'navigation.npz'
    runtime,_,_=scene.sensor_assets()
    sim=scene.create_simulator(Path(row['assets']),row['asset_scene_id'],0,runtime['camera'],row['scene_dataset'])
    try:
        if not sim.recompute_navmesh(sim.pathfinder,navmesh_settings()):
            raise ValueError(f'No physical navigation mesh: {row["scene_id"]}')
        sim.pathfinder.save_nav_mesh(str(folder/'scene.navmesh'))
        navigation_from_navmesh(folder/'scene.navmesh',path)
    finally:sim.close()
    result=dict(row,navigation=str(path.resolve()))
    (folder/'scene.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


@njit(cache=True,nogil=True)
def _raster_surface(triangles,origin,shape,step,reference,tolerance,edge):
    grid=np.full(shape,np.nan)
    for t in range(len(triangles)):
        a,b,c=triangles[t,0],triangles[t,1],triangles[t,2]
        d=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(d)<1e-12:continue
        i0=max(int(np.floor((min(a[0],b[0],c[0])-origin[0])/step)),0);i1=min(int(np.ceil((max(a[0],b[0],c[0])-origin[0])/step)),shape[0]-1)
        j0=max(int(np.floor((min(a[1],b[1],c[1])-origin[1])/step)),0);j1=min(int(np.ceil((max(a[1],b[1],c[1])-origin[1])/step)),shape[1]-1)
        for i in range(i0,i1+1):
            x=origin[0]+(i+.5)*step
            for j in range(j0,j1+1):
                y=origin[1]+(j+.5)*step
                u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/d
                v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/d
                w=1-u-v
                if u<edge or v<edge or w<edge:continue
                z=u*a[2]+v*b[2]+w*c[2]
                if reference.size and not abs(z-reference[i,j])<=tolerance:continue
                if np.isnan(grid[i,j]) or z>grid[i,j]:grid[i,j]=z
    return grid


def raster_surface(triangles,origin,shape,step):
    """Highest surface of canonical (x, y, z) triangles at each cell centre of an [x, y] grid; NaN where none."""
    return _raster_surface(np.asarray(triangles,np.float64),np.asarray(origin,np.float64),(int(shape[0]),int(shape[1])),
        float(step),np.empty((0,0)),0.,-1e-8)


def navmesh_triangles(pathfinder):
    """The navmesh triangles as canonical (x, y, z)."""
    triangles=np.asarray(pathfinder.build_navmesh_vertices(-1),np.float64).reshape(-1,3,3)
    return np.stack([triangles[...,0],-triangles[...,2],triangles[...,1]],-1)


def navmesh_ground(triangles,ground,origin,shape,*,spacing=.05,tolerance=.10):
    """The navmesh on the terrain, as a ground height per cell of an [x, y] grid (NaN: not navigable ground). Only
    navmesh within `tolerance` of the terrain height `ground(xy)` (NaN = no ground there) counts, selected per
    triangle before overlapping floors and roofs are combined: navmesh on obstacle tops stays blocked."""
    origin=np.asarray(origin,np.float64);shape=(int(shape[0]),int(shape[1]))
    centres=np.stack(np.meshgrid(origin[0]+(np.arange(shape[0])+.5)*spacing,origin[1]+(np.arange(shape[1])+.5)*spacing,
        indexing='ij'),-1).reshape(-1,2)
    reference=np.ascontiguousarray(ground(centres).reshape(shape),dtype=np.float64)
    return _raster_surface(triangles,origin,shape,float(spacing),reference,float(tolerance),-1e-9)


def known_ground(on_ground,origin,walked_xy,*,spacing=.05,walked_radius=.5,island_min_m2=20.):
    """The navigable ground to keep: 8-connected components that pass within walked_radius (in grid steps) of a
    `walked_xy` point (known ground: the capture trajectory, official routes) or cover at least island_min_m2.
    Small unvisited components are object tops (car roofs, boxes) or slivers. Returns the mask and statistics."""
    from scipy import ndimage as nd
    labels,count=nd.label(on_ground,structure=np.ones((3,3),bool))
    area=np.bincount(labels.ravel(),minlength=count+1)*spacing**2
    keep=area>=island_min_m2
    cells=np.floor((np.asarray(walked_xy,np.float64).reshape(-1,2)-origin)/spacing).astype(np.int64)
    reach=int(round(walked_radius/spacing));offsets=np.argwhere(np.abs(np.indices((2*reach+1,)*2)-reach).sum(0)<=reach)-reach
    cells=cells[(cells>=0).all(1)&(cells<on_ground.shape).all(1)]
    near=(cells[:,None]+offsets[None]).reshape(-1,2)
    near=near[(near>=0).all(1)&(near<on_ground.shape).all(1)]
    keep[np.unique(labels[near[:,0],near[:,1]])]=True
    keep[0]=False
    return keep[labels],dict(navigation_components_kept=int(keep.sum()),navigation_components_dropped=int(count-keep.sum()),
        navigation_dropped_m2=round(float(area[1:][~keep[1:]].sum()),1))


def rasterize_navigation(triangles, out_dir, ground, walked_xy, *, body_triangles=None, spacing=.05, ground_tolerance=.10,
                         walked_radius=.5, island_min_m2=20.):
    """One scene's navmesh triangles (navmesh_triangles) as out_dir/navigation.npz on a grid covering them:
    navmesh_ground, then known_ground; body_triangles (the body ground navmesh of a rectangular robot) on the same
    grid. The grid reaches a footprint radius past the navmesh, so a body standing at its edge is still on the grid.
    Returns the grid and its statistics."""
    margin = int(np.ceil(ROBOT_FOOTPRINT_RADIUS_M / spacing)) + 1
    lo = np.floor(triangles[..., :2].reshape(-1, 2).min(0) / spacing) * spacing - margin * spacing
    shape = np.ceil((triangles[..., :2].reshape(-1, 2).max(0) - lo) / spacing).astype(int) + margin + 1
    top = navmesh_ground(triangles, ground, lo, shape, spacing=spacing, tolerance=ground_tolerance)
    on_ground, info = known_ground(np.isfinite(top), lo, walked_xy, spacing=spacing, walked_radius=walked_radius,
                                   island_min_m2=island_min_m2)
    ground_z = np.where(on_ground, top, np.nan).astype(np.float32)
    body = None if body_triangles is None else np.isfinite(
        navmesh_ground(body_triangles, ground, lo, shape, spacing=spacing, tolerance=ground_tolerance))
    save_navigation_grid(out_dir / 'navigation.npz', on_ground, lo, spacing, ground_z=ground_z, body_free=body)
    return dict(origin_xy=lo, free=on_ground, ground_z=ground_z), dict(
        navmesh_triangles=int(len(triangles)), navigation_free_cells=int(on_ground.sum()),
        navigation_free_area_m2=round(float(on_ground.sum() * spacing ** 2), 1),
        navigation_shape=[int(shape[0]), int(shape[1])], **info)
