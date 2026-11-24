"""Offline geometry targets from causal depth-only ObservedTeacher state.

The caller updates the teacher with each actual depth and executed=False before
reading a target. These labels are supervision only, never policy inputs.
"""

from functools import lru_cache
from itertools import product
import math
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F




def field_from_observed_hits(tree, point_error_m, observed_free, horizon, radius):
    """Use the same body-frame hits and conservative error subtraction as GF."""
    if tree.n==0:
        field=torch.zeros(6,64,64)
        field[3]=field[5]=torch.as_tensor(observed_free,dtype=torch.float32).reshape(64,64)
        return field
    resolution=2*horizon/63
    axis=np.arange(64)*resolution-horizon
    x,y=np.meshgrid(axis,axis,indexing='xy')
    body=np.stack((x,y),-1)
    distance=nearest_distances(tree,body.reshape(-1,2)).reshape(64,64)
    clearance=torch.from_numpy((distance-point_error_m-radius).astype(np.float32))[None,None]
    padded=F.pad(clearance,(1,1,1,1),mode='replicate')
    dx=(padded[:,:,1:-1,2:]-padded[:,:,1:-1,:-2])/(2*resolution)
    dy=(padded[:,:,2:,1:-1]-padded[:,:,:-2,1:-1])/(2*resolution)
    overlap=(clearance<=0).float()
    free=torch.as_tensor(observed_free,dtype=torch.float32).reshape(1,1,64,64)*(1-overlap)
    return torch.cat((clearance,dx,dy,torch.maximum(free,overlap),overlap,free),1)[0]


def nearest_distances(tree, queries):
    """tree.query(queries)[0], bit for bit: the minimum over the points of sqrt((q-p)_x^2+(q-p)_y^2), searched
    through the points' buckets with Lipschitz bounds from the previous query (field queries are a dense grid
    far from most hits, where the KD tree visits many nodes). Falls back to the tree where the check fails."""
    queries=np.ascontiguousarray(queries,dtype=np.float64)
    points=np.ascontiguousarray(tree.data,dtype=np.float64)
    if (tree.n==0 or tree.m!=2 or not np.isfinite(queries).all() or not np.isfinite(points).all()
            or np.ptp(points,0).max()>1e6*_BUCKET_M or not _nearest_exact()):
        return tree.query(queries,workers=1)[0]
    return _nearest(queries,points,_BUCKET_M)


# Point bucket side and the number of consecutive queries sharing one bucket preselection.
_BUCKET_M,CHUNK=.05,8


@njit(cache=True,nogil=True)
def _nearest(queries,points,cell):
    """Exact nearest distances. Every query's distance is bounded above by its predecessor's distance plus their
    separation (with a relative margin far above rounding): a run of CHUNK queries keeps the buckets whose point
    box is within the run's bound of the run's box. A query starts from its predecessor's nearest point and skips
    the buckets whose box is farther than its best squared distance so far (a box's rounded squared distance never
    exceeds that of a point inside it), so the minimum is the minimum over all points."""
    lx,ly=points[:,0].min(),points[:,1].min()
    nx=int((points[:,0].max()-lx)/cell)+1;ny=int((points[:,1].max()-ly)/cell)+1
    key=np.empty(len(points),np.int64)
    for j in range(len(points)):
        key[j]=min(int((points[j,1]-ly)/cell),ny-1)*nx+min(int((points[j,0]-lx)/cell),nx-1)
    order=np.argsort(key,kind='mergesort');sorted_points=points[order];sorted_key=key[order]
    starts=[0]
    for j in range(1,len(points)):
        if sorted_key[j]!=sorted_key[j-1]:starts.append(j)
    starts.append(len(points))
    buckets=len(starts)-1;box=np.empty((buckets,4))
    for b in range(buckets):
        part=sorted_points[starts[b]:starts[b+1]]
        box[b,0],box[b,1],box[b,2],box[b,3]=part[:,0].min(),part[:,1].min(),part[:,0].max(),part[:,1].max()
    output=np.empty(len(queries));candidates=np.empty(buckets,np.int64);nearest=0
    for first in range(0,len(queries),CHUNK):
        last=min(first+CHUNK,len(queries))
        x0,y0=queries[first:last,0].min(),queries[first:last,1].min()
        x1,y1=queries[first:last,0].max(),queries[first:last,1].max()
        if first==0:
            bound=np.inf
        else:
            px,py=queries[first-1,0],queries[first-1,1]
            sx,sy=max(abs(x0-px),abs(x1-px)),max(abs(y0-py),abs(y1-py))
            bound=(output[first-1]+math.sqrt(sx*sx+sy*sy))*(1+1e-6)+1e-9
        bound=bound*bound;count=0
        for b in range(buckets):
            ex=max(box[b,0]-x1,x0-box[b,2],0.);ey=max(box[b,1]-y1,y0-box[b,3],0.)
            if ex*ex+ey*ey<=bound:
                candidates[count]=b;count+=1
        for i in range(first,last):
            qx,qy=queries[i,0],queries[i,1]
            # The predecessor's nearest point bounds this query's minimum: an attained squared distance.
            best=np.inf
            if i:
                dx=qx-sorted_points[nearest,0];dy=qy-sorted_points[nearest,1];best=dx*dx+dy*dy
            for c in range(count):
                b=candidates[c]
                ex=max(box[b,0]-qx,qx-box[b,2],0.);ey=max(box[b,1]-qy,qy-box[b,3],0.)
                if ex*ex+ey*ey>best:continue
                for j in range(starts[b],starts[b+1]):
                    dx=qx-sorted_points[j,0];dy=qy-sorted_points[j,1]
                    distance=dx*dx+dy*dy
                    if distance<best:best=distance;nearest=j
            output[i]=math.sqrt(best)
    return output


@lru_cache(maxsize=1)
def _nearest_exact():
    """Whether _nearest reproduces cKDTree's distances here (same IEEE arithmetic, no fused multiply-add)."""
    rng=np.random.default_rng(2718)
    for count in (1,7,300,3000):
        points=np.concatenate((rng.normal(0,2,(count,2)),rng.uniform(-1,1,(count,1))*[1,0]+[0,1.5]))
        queries=np.concatenate((rng.normal(0,4,(2000,2)),points[:200]+rng.normal(0,1e-7,(min(200,len(points)),2))))
        tree=cKDTree(points)
        if not np.array_equal(_nearest(queries,tree.data,_BUCKET_M),tree.query(queries,workers=1)[0]):
            return False
    return True


def footprint_observed_per_point(teacher, points, radius):
    """Vectorize the teacher's exact disk/fully-observed-column predicate."""
    points=np.asarray(points,dtype=float).reshape(-1,2)
    reach=int(np.ceil(radius/teacher.grid))+1
    offsets=sorted(product(range(-reach,reach+1),repeat=2),key=lambda xy:xy[0]**2+xy[1]**2)
    base=np.floor(points/teacher.grid).astype(int)
    lower=base.min(0)-reach;shape=base.max(0)+reach-lower+1
    raster=np.zeros(tuple(shape),bool)
    known=np.asarray([key for key,value in teacher.cells.items() if value==255],dtype=int).reshape(-1,2)-lower
    known=known[((known>=0)&(known<shape)).all(-1)]
    raster[known[:,0],known[:,1]]=True
    covered=np.ones(len(points),bool)
    for offset in offsets:
        active=np.flatnonzero(covered)
        if not len(active):break
        keys=base[active]+offset
        distance=np.maximum(abs((keys+.5)*teacher.grid-points[active])-teacher.grid/2,0)
        overlap=np.linalg.norm(distance,axis=-1)<=radius
        indices=keys-lower
        covered[active]=raster[indices[:,0],indices[:,1]]|~overlap
    return covered


def target_from_observations(teacher, pose, k, camera, projector, tree, point_error_m):
    """Return the CPU supervision field[6,64,64], without mutation.

    ``pose`` is the current 4x4 body-to-world matrix. The teacher must contain
    this acquisition already. Unknown cells have support/free zero; their
    represented distance is not an observed-free-space certificate.
    """
    horizon=projector.planning_horizon_m
    radius=projector.radius
    free=footprint_observed_per_point(teacher,sample_points(pose,horizon),radius)
    field=field_from_observed_hits(tree,point_error_m,free,horizon,radius)
    return field


def sample_points(pose, horizon):
    """The field's 64x64 body-frame sample points in the world, [64,64,2]."""
    pose=np.asarray(pose,dtype=float)
    resolution=2*horizon/63
    axis=np.arange(64)*resolution-horizon
    x,y=np.meshgrid(axis,axis,indexing='xy')
    body=np.stack((x,y),-1)
    return body@pose[:2,:2].T+pose[:2,3]


@njit(cache=True,nogil=True)
def footprint_observed_raster(masks,origin_x,origin_y,points,radius,grid):
    """footprint_observed_per_point on a dense route-cell mask raster masks[y-origin_y,x-origin_x] (cells outside
    it are unobserved): a point is covered when every cell its radius-disk overlaps is fully observed (255)."""
    reach=int(math.ceil(radius/grid))+1
    height,width=masks.shape
    covered=np.ones(len(points),dtype=np.bool_)
    for index in range(len(points)):
        px,py=points[index,0],points[index,1]
        bx,by=int(math.floor(px/grid)),int(math.floor(py/grid))
        for ox in range(-reach,reach+1):
            for oy in range(-reach,reach+1):
                kx,ky=bx+ox,by+oy
                column,row=kx-origin_x,ky-origin_y
                if 0<=row<height and 0<=column<width and masks[row,column]==255:
                    continue
                dx=max(abs((kx+.5)*grid-px)-grid/2,0.)
                dy=max(abs((ky+.5)*grid-py)-grid/2,0.)
                if math.sqrt(dx*dx+dy*dy)<=radius:
                    covered[index]=False;break
            if not covered[index]:break
    return covered


@njit(cache=True,nogil=True)
def _minimum_squared(paths,delta,denominator,boxes):
    output=np.empty(len(paths),np.float64)
    for candidate in range(len(paths)):
        best=np.inf
        for segment in range(paths.shape[1]-1):
            ax,ay=paths[candidate,segment];bx,by=paths[candidate,segment+1]
            dx,dy=delta[candidate,segment]
            for box in range(len(boxes)):
                lx,ly,hx,hy=boxes[box];enter=0.;leave=1.;intersects=True
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
                if intersects and enter<=leave:
                    best=0.;break
                ex=max(lx-ax,ax-hx,0.);ey=max(ly-ay,ay-hy,0.)
                best=min(best,ex*ex+ey*ey)
                ex=max(lx-bx,bx-hx,0.);ey=max(ly-by,by-hy,0.)
                best=min(best,ex*ex+ey*ey)
                for corner in range(4):
                    cx=lx if corner==0 or corner==3 else hx
                    cy=ly if corner<2 else hy
                    fraction=((cx-ax)*dx+(cy-ay)*dy)/denominator[candidate,segment]
                    fraction=min(1.,max(0.,fraction))
                    ex=cx-(ax+fraction*dx);ey=cy-(ay+fraction*dy)
                    best=min(best,ex*ex+ey*ey)
            if best==0:break
        output[candidate]=best
    return output


class BoxIndex:
    """Axis-aligned boxes [lx,ly,hx,hy] indexed by their centres, for exact minimum path-to-box distances.

    The paths share a start. Every box closer than (path reach + the farthest path start's nearest-box distance)
    can hold the minimum; its centre lies within that bound plus the largest box half-diagonal, and a path start's
    nearest centre bounds its nearest-box distance from above. The candidates are therefore a superset of the
    boxes that can hold the minimum, and the minimum is the brute-force one, while a query touches only nearby
    boxes instead of every boundary box of the scene."""

    # Boxes are cut into pieces at most this long per axis before indexing: a scene's merged boundary segments run
    # tens of metres, and one long box would widen every query to the whole scene. The pieces cover exactly the
    # same set, so the distance to their union is the distance to the box.
    PIECE_M = .2

    def __init__(self, boxes):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4); pieces = []
        counts = np.maximum(np.ceil((boxes[:, 2:] - boxes[:, :2]) / self.PIECE_M).astype(np.int64), 1)
        for (lx, ly, hx, hy), (nx, ny) in zip(boxes, counts):
            xs, ys = np.linspace(lx, hx, nx + 1), np.linspace(ly, hy, ny + 1)
            pieces.append(np.stack(np.broadcast_arrays(xs[:-1, None], ys[None, :-1], xs[1:, None], ys[None, 1:]), -1).reshape(-1, 4))
        self.boxes = np.ascontiguousarray(np.concatenate(pieces) if pieces else boxes)
        self.half = float(np.linalg.norm(self.boxes[:, 2:] - self.boxes[:, :2], axis=1).max() / 2) if len(self.boxes) else 0.
        self.tree = cKDTree((self.boxes[:, :2] + self.boxes[:, 2:]) / 2) if len(self.boxes) else None

    def clearance(self, paths, *, radius):
        paths = np.ascontiguousarray(paths, dtype=np.float64)
        if self.tree is None: return np.full(len(paths), np.inf) - radius
        center = paths[0, 0]; reach = np.linalg.norm(paths - center, axis=-1).max()
        upper = self.tree.query(paths[:, 0], k=1)[0].max()
        nearby = self.boxes[np.sort(self.tree.query_ball_point(center, reach + upper + self.half + 1e-7))]
        delta = np.diff(paths, axis=1); denominator = np.maximum((delta**2).sum(-1), 1e-20)
        return np.sqrt(_minimum_squared(paths, delta, denominator, nearby)) - radius
