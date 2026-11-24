"""The observation classes that define the LH target field of a state: ObservedBSplineRoutes (observed-free route
cells and occupied voxels of the frames) and ObservedHitHistory (their depth hits and error bound).
local_replay.replayed_labels replays a state's selected frames through them; fused_labels reproduces it bit for bit.
"""
from itertools import product
import math
import sys
import numpy as np
from kappanav.local_planner.features.obstacle_memory import VOXEL_M, volumes_observed_clear
from kappanav.common.physical import ROBOT_BASE_HEIGHT_ABOVE_GROUND_M


def body_pose(view):
    c,s=math.cos(float(view[2])),math.sin(float(view[2]))
    return np.asarray([[c,-s,0.,view[0]],[s,c,0.,view[1]],
        [0.,0.,1.,ROBOT_BASE_HEIGHT_ABOVE_GROUND_M],[0.,0.,0.,1.]],dtype=np.float32)


class ObservedBSplineRoutes:
    grid = .05
    radius = .167584539
    margin = .10
    waypoint_tolerance = .005
    forecast_guard = .0075
    maximum = 5.
    band = (.05, .161981515)

    def __init__(self, k, camera):
        self.k = np.asarray(k, dtype=float).copy()
        self.camera = np.asarray(camera, dtype=float).copy()
        self.cells = {}
        self.occupied = set()
        self.corners = np.asarray(list(product((0., 1.), repeat=3)))
        self.view = None
        self.updates = 0
        self._rays = None

    def update(self, depth, view, executed=True, executed_sweep=None):
        """Consume one actually acquired normalized optical-Z image and its pose.

        A supplied sweep contains actual SE2 poses, not virtual observations.
        Only the physical radius is used; curved segments shrink their chord
        capsule by the arc sagitta. Without a sweep only this pose is certified.
        """
        depth = np.asarray(depth, dtype=np.float32)
        view = np.asarray(view, dtype=float)
        sweep = view[None] if executed_sweep is None else np.asarray(executed_sweep, dtype=float)
        if self._rays is None:
            y, x = np.indices(depth.shape, dtype=float)
            self._rays = np.stack(((x+.5-self.k[0, 2])/self.k[0, 0],
                                  (y+.5-self.k[1, 2])/self.k[1, 1], np.ones_like(x)), -1)
        pose = body_pose(view)
        camera_world = pose @ self.camera
        world = (self._rays*(depth*self.maximum)[..., None]) @ camera_world[:3, :3].T + camera_world[:3, 3]
        observed = depth > 0
        bounds = np.concatenate((world[observed, :2], sweep[:, :2]-self.radius, sweep[:, :2]+self.radius))
        lo = np.floor(bounds.min(0)/self.grid).astype(int)-1
        hi = np.floor(bounds.max(0)/self.grid).astype(int)+1
        x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        keys = np.column_stack((x.ravel(), y.ravel()))
        masks = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), dtype=np.uint8)
        if self.cells:
            previous = np.asarray(list(self.cells))-lo
            inside = ((previous >= 0) & (previous <= hi-lo)).all(-1)
            xy = previous[inside]
            masks[xy[:, 1], xy[:, 0]] = np.fromiter(self.cells.values(), dtype=np.uint8)[inside]
        masks = masks.ravel()
        pending = masks != 255
        missing = (masks[pending, None] & (1 << np.arange(8))) == 0
        heights = np.linspace(*self.band, 9)
        lower = np.column_stack((np.repeat(keys[pending]*self.grid, 8, axis=0),
                                 np.tile(heights[:-1], int(pending.sum()))))[missing.ravel()]
        size = np.array([self.grid, self.grid, heights[1]-heights[0]])
        # A slab cannot clear unless its lower corner is in the image frustum.
        optical = (lower[:, None]-camera_world[:3, 3]) @ camera_world[:3, :3]
        z = optical[:, 0, 2]
        positive_z = np.where(z > 0, z, 1.)
        u = self.k[0, 0]*optical[:, 0, 0]/positive_z+self.k[0, 2]-.5
        v = self.k[1, 1]*optical[:, 0, 1]/positive_z+self.k[1, 2]-.5
        h, w = depth.shape
        visible = (z > 0) & (u >= -1e-9) & (u <= w-1+1e-9) & (v >= -1e-9) & (v <= h-1+1e-9)
        clear = np.zeros(missing.shape, bool)
        boxes = lower[visible, None]+self.corners*size
        visible_clear = np.zeros(len(lower), bool)
        visible_clear[visible] = volumes_observed_clear(boxes, depth, self.k, camera_world, self.maximum)
        clear[missing] = visible_clear
        bits = (clear.astype(np.uint16) * (1 << np.arange(8))).sum(1).astype(np.uint8)
        masks[pending] |= bits
        if executed:
            near = (((keys+.5)*self.grid >= sweep[:, :2].min(0)-self.radius-self.grid) &
                    ((keys+.5)*self.grid <= sweep[:, :2].max(0)+self.radius+self.grid)).all(-1)
            corners = (keys[near, None]+np.asarray(list(product((0., 1.), repeat=2))))*self.grid
            certified = np.zeros(int(near.sum()), bool)
            segments = zip(sweep[:-1], sweep[1:]) if len(sweep) > 1 else [(view, view)]
            for a, b in segments:
                delta = b[:2]-a[:2]
                length = np.linalg.norm(delta)
                angle = abs(np.arctan2(np.sin(b[2]-a[2]), np.cos(b[2]-a[2])))
                sagitta = length*.5*np.tan(angle*.25)
                if length:
                    along = np.clip(((corners-a[:2])*delta).sum(-1)/(length*length), 0, 1)
                    nearest = a[:2]+along[..., None]*delta
                else:
                    nearest = a[:2]
                certified |= np.linalg.norm(corners-nearest, axis=-1).max(-1) <= self.radius-sagitta
            masks[np.flatnonzero(near)[certified]] = 255
        nonzero = masks != 0
        for key, mask in zip(keys[nonzero].tolist(), masks[nonzero]):
            self.cells[tuple(key)] = int(mask)
        if self.occupied:
            voxels = np.asarray(sorted(self.occupied), dtype=np.int64)
            boxes = (voxels[:, None]+self.corners)*VOXEL_M
            cleared = volumes_observed_clear(boxes, depth, self.k, camera_world, self.maximum)
            self.occupied.difference_update(map(tuple, voxels[cleared]))
        hit = observed & (depth < 1) & (world[..., 2] >= self.band[0]) & (world[..., 2] <= self.band[1])
        self.occupied.update(map(tuple, np.unique(np.floor(world[hit]/VOXEL_M).astype(np.int64), axis=0)))
        for key in self._occupied_cells():
            self.cells.pop(key, None)
        self.view = view.copy()
        self.updates += 1

    def _occupied_cells(self):
        if not self.occupied:
            return set()
        voxels = np.asarray(list(self.occupied))[:, :2]
        lo = np.floor(voxels*VOXEL_M/self.grid).astype(int)
        hi = np.floor(np.nextafter((voxels+1)*VOXEL_M, -np.inf)/self.grid).astype(int)
        return {tuple(cell) for dx, dy in product((0, 1), repeat=2)
                for cell in (lo+[dx, dy])[((lo+[dx, dy]) <= hi).all(-1)]}

    def _raster(self, view):
        occupied = self._occupied_cells()
        keys = list(self.cells) + list(occupied) + [tuple(np.floor(view[:2]/self.grid).astype(int))]
        keys = np.asarray(keys)
        lo, hi = keys.min(0)-5, keys.max(0)+5
        shape = tuple((hi-lo+1)[::-1])
        free, blocked = np.zeros(shape, bool), np.zeros(shape, bool)
        for key, value in self.cells.items():
            x, y = np.asarray(key)-lo
            free[y, x] = value == 255
        for key in occupied:
            x, y = np.asarray(key)-lo
            blocked[y, x] = True
        free &= ~blocked
        return free, blocked, lo

    def footprint_observed(self, points, radius=None):
        """Conservative disk coverage by fully observed body-height grid columns."""
        radius = self.radius if radius is None else radius
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        reach = int(np.ceil(radius/self.grid))+1
        offsets = np.asarray(list(product(range(-reach, reach+1), repeat=2)))
        keys = np.floor(points[:, None]/self.grid).astype(int)+offsets
        distance = np.maximum(abs((keys+.5)*self.grid-points[:, None])-self.grid/2, 0)
        overlap = np.linalg.norm(distance, axis=-1) <= radius
        needed = np.unique(keys[overlap], axis=0)
        return all(self.cells.get(tuple(key), 0) == 255 for key in needed)


    def occupied_clearance(self, points):
        """Clearance to retained measured occupied voxels, never scene boxes."""
        if not self.occupied:
            return float('inf')
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        lower = np.unique(np.asarray(list(self.occupied))[:, :2], axis=0)*VOXEL_M
        upper = lower+VOXEL_M
        reach = self.radius+self.margin+self.forecast_guard
        nearby = ((upper >= points.min(0)-reach) & (lower <= points.max(0)+reach)).all(-1)
        if not nearby.any():
            return float('inf')
        distances = np.maximum(np.maximum(lower[None, nearby]-points[:, None],
                                           points[:, None]-upper[None, nearby]), 0)
        return float(np.linalg.norm(distances, axis=-1).min()-self.radius)






RECORD=np.dtype([('world',np.float64,(3,)),('frame',np.uint32),('pixel',np.uint32)])

class ObservedHitHistory:
    maximum=ObservedBSplineRoutes.maximum
    band=ObservedBSplineRoutes.band

    def __init__(self,k,camera):
        self.k=np.asarray(k,dtype=float).copy();self.camera=np.asarray(camera,dtype=float).copy()
        self.live={};self.corners=np.asarray(list(product((0.,1.),repeat=3)))
        self.rays=None;self.updates=0;self.view=None;self.error_bound=None

    def update(self,depth,view,*,pose=None):
        """Accumulate hits using the supplied full world-from-body pose, or the planar capture pose."""
        depth=np.asarray(depth,dtype=np.float32);view=np.asarray(view,dtype=float)
        if self.rays is None:
            y,x=np.indices(depth.shape,dtype=float)
            self.rays=np.stack(((x+.5-self.k[0,2])/self.k[0,0],(y+.5-self.k[1,2])/self.k[1,1],np.ones_like(x)),-1)
            self.error_bound=float(5*np.finfo(np.float16).eps*np.linalg.norm(self.rays,axis=-1).max())
        camera_world=(body_pose(view) if pose is None else np.asarray(pose,dtype=float))@self.camera
        # The target definition: depth is FP32 and depth*5 happens before FP64 rays.
        world=(self.rays*(depth*self.maximum)[...,None])@camera_world[:3,:3].T+camera_world[:3,3]
        cleared_keys=set();removed_points=0
        if self.live:
            voxels=np.asarray(sorted(self.live),dtype=np.int64)
            cleared=volumes_observed_clear((voxels[:,None]+self.corners)*VOXEL_M,depth,self.k,camera_world,self.maximum)
            cleared_keys=set(map(tuple,voxels[cleared]))
            for key in cleared_keys:
                removed_points+=sum(len(chunk) for chunk in self.live.pop(key))
        hit=(depth>0)&(depth<1)&(world[...,2]>=self.band[0])&(world[...,2]<=self.band[1])
        pixels=np.flatnonzero(hit);points=world.reshape(-1,3)[pixels]
        keys,inverse=np.unique(np.floor(points/VOXEL_M).astype(np.int64),axis=0,return_inverse=True)
        records=np.empty(len(points),dtype=RECORD)
        records['world']=points;records['frame']=self.updates;records['pixel']=pixels
        order=np.argsort(inverse,kind='stable');ends=np.cumsum(np.bincount(inverse,minlength=len(keys)))
        start=0
        for key,end in zip(keys,ends,strict=True):
            self.live.setdefault(tuple(key),[]).append(records[order[start:end]].copy());start=end
        self.view=view.copy();self.updates+=1
        return dict(frame=self.updates-1,hits=len(points),cleared_voxels=len(cleared_keys),
            removed_points=removed_points,cleared_then_rehit=sum(tuple(key) in cleared_keys for key in keys))

    def records(self):
        chunks=[chunk for key in sorted(self.live) for chunk in self.live[key]]
        return np.concatenate(chunks) if chunks else np.empty(0,dtype=RECORD)

    def memory(self):
        chunks=[chunk for values in self.live.values() for chunk in values]
        return dict(points=sum(len(chunk) for chunk in chunks),voxels=len(self.live),chunks=len(chunks),
            array_bytes=sum(chunk.nbytes for chunk in chunks),
            approximate_python_and_arrays_bytes=sys.getsizeof(self.live)+sum(sys.getsizeof(k)+sys.getsizeof(v)
                for k,v in self.live.items())+sum(sys.getsizeof(chunk) for chunk in chunks))
