"""Four-frame observed geometry targets,independent of learned image features.

local_labels is defined by `replayed_labels`: the anchor's selected frames replayed, in order, through
ObservedBSplineRoutes (executed=False) and ObservedHitHistory, then target_from_observations. The fused path
computes the same field, KD tree and error bound bit for bit, several times faster:
- both classes back-project the same world points and keep the same occupied voxels (a voxel is occupied while
  it holds a live hit), so one back-projection and one clearing test per frame serve both;
- the route cells live in one dense mask raster, and the frustum and box-clearing tests run per slab in numba
  kernels with numpy's operation order and dtypes (no temporaries over the frustum's bounding box);
- a route cell's mask depends only on its own history, the frames and the occupied voxels, and the field reads
  only cells within the footprint reach of its 64x64 sample points, so other cells are never tested.
The kernels' projections reproduce numpy's matmuls only if this machine's BLAS rounds a 3-term dot product
without fusing; that is checked once per process against numpy, and the replay is used where it does not hold.
"""
from functools import lru_cache
import math
import numpy as np
import torch
from numba import njit
from scipy.spatial import cKDTree
from kappanav.local_planner.features.obstacle_memory import VOXEL_M
from kappanav.local_planner.data.observed_bspline_routes import ObservedBSplineRoutes, ObservedHitHistory, body_pose
from kappanav.local_planner.data.observed_geometry import (footprint_observed_raster, field_from_observed_hits,
                                                           sample_points, target_from_observations)

# The clearing tolerance of volumes_observed_clear (normalized FP16 depth storage).
_TOLERANCE = float(ObservedBSplineRoutes.maximum * np.finfo(np.float16).eps)


@torch.no_grad()
def local_labels(actor, raw, plan):
    try:
        return fused_labels(actor, raw, plan)
    except _Unsupported:
        return replayed_labels(actor, raw, plan)


def _anchor(raw, plan):
    selected = sorted(set(plan['selection']['indices']))
    current = selected[-1]
    return selected, raw['body_to_world'][current], raw['camera_intrinsics'][current], raw['camera_to_body'][current]


@torch.no_grad()
def replayed_labels(actor, raw, plan):
    """The definition of local_labels: replay the selected frames through the observation classes."""
    selected, pose, k, camera = _anchor(raw, plan)
    observed = ObservedBSplineRoutes(k, camera); hits = ObservedHitHistory(k, camera)
    for i in selected:
        observed.update(raw['depths'][i], raw['acquired_views'][i], executed=False)
        hits.update(raw['depths'][i], raw['acquired_views'][i])
    points = (hits.records()['world'][:, :2]-pose[:2, 3])@pose[:2, :2]
    tree = cKDTree(points)
    field = target_from_observations(observed, pose, k, camera, actor.policy.depth_encoder.metric_projector, tree, hits.error_bound)
    return field, tree, hits.error_bound


@torch.no_grad()
def fused_labels(actor, raw, plan):
    """replayed_labels' outputs, bit for bit, by the fused path (see the module docstring)."""
    maximum, band, grid = ObservedBSplineRoutes.maximum, ObservedBSplineRoutes.band, ObservedBSplineRoutes.grid
    selected, pose, k, camera = _anchor(raw, plan)
    k = np.asarray(k, dtype=float).copy(); camera = np.asarray(camera, dtype=float).copy()
    projector = actor.policy.depth_encoder.metric_projector
    heights = np.linspace(*band, 9); scale = np.float32(maximum)
    frames = []
    for i in selected:
        depth = np.asarray(raw['depths'][i], dtype=np.float32); view = np.asarray(raw['acquired_views'][i], dtype=float)
        if not frames:
            if not _kernels_exact(depth.shape):
                raise _Unsupported
            rays, error_bound = _rays(k.tobytes(), depth.shape)
        elif depth.shape != frames[0][0].shape:
            raise _Unsupported
        camera_world = body_pose(view)@camera
        points, bounds = _back_project(depth, rays, camera_world, scale, band[0], band[1])
        if not np.isfinite(bounds).all():
            raise _Unsupported
        # ObservedBSplineRoutes.update's cell box: observed points and the (unexecuted) pose's footprint.
        bounds = np.concatenate((bounds, view[None, :2]-ObservedBSplineRoutes.radius, view[None, :2]+ObservedBSplineRoutes.radius))
        lo = np.floor(bounds.min(0)/grid).astype(int)-1; hi = np.floor(bounds.max(0)/grid).astype(int)+1
        frames.append((depth, camera_world, points, lo, hi))
    origin = np.min([frame[3] for frame in frames], 0); top = np.max([frame[4] for frame in frames], 0)
    masks = np.zeros(tuple((top-origin+1)[::-1]), dtype=np.uint8)
    sample = sample_points(pose, projector.planning_horizon_m)
    needed = np.zeros(masks.shape, dtype=np.bool_)
    _mark_footprint_cells(needed, origin[0], origin[1], sample.reshape(-1, 2), projector.radius, grid)
    # The hit history as per-frame arrays: a hit lives until a later frame clears its voxel. Voxels are codes
    # ordered like their (x, y, z) tuples, so sorting the live hits by (code, frame, pixel) is records()' order.
    codes, located, alive = [], [], []
    for depth, camera_world, points, lo, hi in frames:
        _clear_slabs(masks, needed, origin[0], origin[1], lo[0], lo[1], hi[0], hi[1], heights, grid,
                     camera_world, k, depth, scale, _TOLERANCE)
        if codes:
            live = np.unique(np.concatenate([code[keep] for code, keep in zip(codes, alive)]))
            cleared = live[_clear_voxels(_decode(live), VOXEL_M, camera_world, k, depth, scale, _TOLERANCE)]
            if len(cleared):
                for code, keep in zip(codes, alive):
                    keep &= ~np.isin(code, cleared, assume_unique=False)
        codes.append(_encode(np.floor(points/VOXEL_M).astype(np.int64))); located.append(points[:, :2]); alive.append(np.ones(len(points), bool))
        live = np.unique(np.concatenate([code[keep] for code, keep in zip(codes, alive)]))
        if len(live):
            _pop_occupied_cells(masks, origin, _decode(live)[:, :2], grid)
    code = np.concatenate([code[keep] for code, keep in zip(codes, alive)])
    xy = np.concatenate([xy[keep] for xy, keep in zip(located, alive)])[np.argsort(code, kind='stable')]
    tree = cKDTree((xy-pose[:2, 3])@pose[:2, :2])
    free = footprint_observed_raster(masks, origin[0], origin[1], sample.reshape(-1, 2), projector.radius, grid)
    return field_from_observed_hits(tree, error_bound, free, projector.planning_horizon_m, projector.radius), tree, error_bound


_BIAS, _BITS = 1 << 20, 21


def _encode(voxels):
    """Voxel (x, y, z) rows as int64 codes ordered like the tuples."""
    if len(voxels) and (voxels.min() < -_BIAS or voxels.max() >= _BIAS):
        raise _Unsupported
    voxels = voxels+_BIAS
    return (voxels[:, 0] << 2*_BITS) | (voxels[:, 1] << _BITS) | voxels[:, 2]


def _decode(codes):
    mask = (1 << _BITS)-1
    return np.stack((codes >> 2*_BITS, (codes >> _BITS) & mask, codes & mask), -1)-_BIAS


class _Unsupported(Exception):
    """Inputs outside what the fused path reproduces (non-finite depth, voxel codes beyond 26 km from the origin,
    differing image shapes, or kernels not matching numpy here): the replay labels the anchor."""


def _pop_occupied_cells(masks, origin, voxels, grid):
    """ObservedBSplineRoutes._occupied_cells: every route cell under an occupied voxel's footprint is reset."""
    lo = np.floor(voxels*VOXEL_M/grid).astype(int)
    hi = np.floor(np.nextafter((voxels+1)*VOXEL_M, -np.inf)/grid).astype(int)
    shape = np.asarray(masks.shape[::-1])
    for dx in (0, 1):
        for dy in (0, 1):
            cell = lo+[dx, dy]
            cell = cell[(cell <= hi).all(-1)]-origin
            cell = cell[((cell >= 0) & (cell < shape)).all(-1)]
            masks[cell[:, 1], cell[:, 0]] = 0


@lru_cache(maxsize=8)
def _rays(intrinsic, shape):
    """ObservedBSplineRoutes' FP64 pixel rays and ObservedHitHistory's error bound for intrinsics k (bytes)."""
    k = np.frombuffer(intrinsic, dtype=float).reshape(3, 3)
    y, x = np.indices(shape, dtype=float)
    rays = np.stack(((x+.5-k[0, 2])/k[0, 0], (y+.5-k[1, 2])/k[1, 1], np.ones_like(x)), -1)
    rays.flags.writeable = False
    return rays, float(5*np.finfo(np.float16).eps*np.linalg.norm(rays, axis=-1).max())


@njit(cache=True, nogil=True)
def _back_project(depth, rays, camera_world, scale, low, high):
    """The classes' world = (rays*(depth*maximum)[...,None])@R.T+t, per pixel: (FP32 depth*maximum) widened, a
    matmul row left to right, then +t. Returns the hits (valid depth below the maximum and world z within the
    body band) in pixel order and the bounds [[min x, min y], [max x, max y]] of the observed (depth>0) points;
    the bounds are NaN if a depth is not finite."""
    h, w = depth.shape
    points = np.empty((h*w, 3)); count = 0
    bounds = np.full((2, 2), np.nan); first = True
    for row in range(h):
        for column in range(w):
            value = depth[row, column]
            if not np.isfinite(value):
                return points[:0].copy(), np.full((2, 2), np.nan)
            if not value > 0:
                continue
            metric = np.float64(value*scale)
            p0, p1, p2 = rays[row, column, 0]*metric, rays[row, column, 1]*metric, rays[row, column, 2]*metric
            x = p0*camera_world[0, 0]+p1*camera_world[0, 1]+p2*camera_world[0, 2]+camera_world[0, 3]
            y = p0*camera_world[1, 0]+p1*camera_world[1, 1]+p2*camera_world[1, 2]+camera_world[1, 3]
            z = p0*camera_world[2, 0]+p1*camera_world[2, 1]+p2*camera_world[2, 2]+camera_world[2, 3]
            if first:
                bounds[0, 0] = bounds[1, 0] = x; bounds[0, 1] = bounds[1, 1] = y; first = False
            else:
                bounds[0, 0] = min(bounds[0, 0], x); bounds[1, 0] = max(bounds[1, 0], x)
                bounds[0, 1] = min(bounds[0, 1], y); bounds[1, 1] = max(bounds[1, 1], y)
            if value < 1 and z >= low and z <= high:
                points[count, 0] = x; points[count, 1] = y; points[count, 2] = z; count += 1
    return points[:count].copy(), bounds[:0].copy() if first else bounds


@njit(cache=True, nogil=True)
def _planar(x0, x1, y0, y1, camera_world, partial):
    """The first two terms of the optical coordinates of the four planar corners (corner = 2*(x is x1)+(y is y1)):
    a 3x3 matmul row is a*b0+c*b1+d*b2, evaluated left to right, so (a*b0+c*b1) is shared by every height."""
    for corner in range(4):
        dx = (x1 if corner & 2 else x0)-camera_world[0, 3]
        dy = (y1 if corner & 1 else y0)-camera_world[1, 3]
        for j in range(3):
            partial[corner, j] = dx*camera_world[0, j]+dy*camera_world[1, j]


@njit(cache=True, nogil=True)
def _vertical(z, camera_world, term):
    dz = z-camera_world[2, 3]
    for j in range(3):
        term[j] = dz*camera_world[2, j]


@njit(cache=True, nogil=True)
def _box_clear(partial, lower, upper, k, depth, scale, tolerance):
    """volumes_observed_clear for one box (planar corner partials, the lower and upper heights' terms): every
    corner projects into the image in front of the camera and the whole rectangle of pixels their projections
    touch is measured deeper (FP32 depth*maximum) than the farthest corner plus the tolerance."""
    h, w = depth.shape
    left, top, right, bottom, farthest = w, h, -1, -1, -np.inf
    for corner in range(8):
        term = upper if corner & 1 else lower
        o0 = partial[corner >> 1, 0]+term[0]
        o1 = partial[corner >> 1, 1]+term[1]
        o2 = partial[corner >> 1, 2]+term[2]
        if not o2 > 0:
            return False
        u = k[0, 0]*o0/o2+k[0, 2]-.5
        v = k[1, 1]*o1/o2+k[1, 2]-.5
        if not (u >= 0 and u <= w-1 and v >= 0 and v <= h-1):
            return False
        a, b = int(math.floor(u)), int(math.floor(v))
        left, top = min(left, a), min(top, b)
        right, bottom = max(right, min(a+1, w-1)), max(bottom, min(b+1, h-1))
        farthest = max(farthest, o2)
        # The corner's own pixel is in the rectangle and the threshold only grows: an early refutation.
        if not depth[b, a]*scale > farthest+tolerance:
            return False
    threshold = farthest+tolerance
    # (minimum depth)*maximum > threshold for every pixel's FP32 product: rounding is monotone, NaN never clears.
    for row in range(top, bottom+1):
        for column in range(left, right+1):
            if not depth[row, column]*scale > threshold:
                return False
    return True


@njit(cache=True, nogil=True)
def _clear_slabs(masks, needed, origin_x, origin_y, lo_x, lo_y, hi_x, hi_y, heights, grid, camera_world, k, depth, scale, tolerance):
    """ObservedBSplineRoutes.update's free-slab bits for the cells of box [lo, hi] that the field can read: a
    missing slab is tested only when its lower corner is in the image frustum (with the update's 1e-9 slack).
    Slab boxes are lower+{0,1}^3*(grid, grid, heights[1]-heights[0]), and lower+0. is lower."""
    h, w = depth.shape
    slab = heights[1]-heights[0]
    terms = np.empty((8, 2, 3)); partial = np.empty((4, 3))
    for bit in range(8):
        _vertical(heights[bit], camera_world, terms[bit, 0]); _vertical(heights[bit]+slab, camera_world, terms[bit, 1])
    for ky in range(lo_y, hi_y+1):
        for kx in range(lo_x, hi_x+1):
            row, column = ky-origin_y, kx-origin_x
            mask = masks[row, column]
            if mask == 255 or not needed[row, column]:
                continue
            lx, ly = kx*grid, ky*grid
            _planar(lx, lx+grid, ly, ly+grid, camera_world, partial)
            for bit in range(8):
                if mask & (1 << bit):
                    continue
                o0 = partial[0, 0]+terms[bit, 0, 0]
                o1 = partial[0, 1]+terms[bit, 0, 1]
                z = partial[0, 2]+terms[bit, 0, 2]
                if not z > 0:
                    continue
                u = k[0, 0]*o0/z+k[0, 2]-.5
                v = k[1, 1]*o1/z+k[1, 2]-.5
                if not (u >= -1e-9 and u <= w-1+1e-9 and v >= -1e-9 and v <= h-1+1e-9):
                    continue
                if _box_clear(partial, terms[bit, 0], terms[bit, 1], k, depth, scale, tolerance):
                    mask |= 1 << bit
            masks[row, column] = mask


@njit(cache=True, nogil=True)
def _clear_voxels(voxels, size, camera_world, k, depth, scale, tolerance):
    """volumes_observed_clear of the occupied voxels [N,3] (boxes (voxel+{0,1}^3)*size)."""
    cleared = np.zeros(len(voxels), dtype=np.bool_)
    partial = np.empty((4, 3)); lower = np.empty(3); upper = np.empty(3)
    for index in range(len(voxels)):
        x, y, z = float(voxels[index, 0]), float(voxels[index, 1]), float(voxels[index, 2])
        _planar((x+0.)*size, (x+1.)*size, (y+0.)*size, (y+1.)*size, camera_world, partial)
        _vertical((z+0.)*size, camera_world, lower); _vertical((z+1.)*size, camera_world, upper)
        cleared[index] = _box_clear(partial, lower, upper, k, depth, scale, tolerance)
    return cleared


@njit(cache=True, nogil=True)
def _mark_footprint_cells(needed, origin_x, origin_y, points, radius, grid):
    """Every cell footprint_observed_raster can read for these points (its offset square around each point's cell)."""
    reach = int(math.ceil(radius/grid))+1
    height, width = needed.shape
    for index in range(len(points)):
        bx, by = int(math.floor(points[index, 0]/grid)), int(math.floor(points[index, 1]/grid))
        for column in range(max(bx-reach-origin_x, 0), min(bx+reach-origin_x, width-1)+1):
            for row in range(max(by-reach-origin_y, 0), min(by+reach-origin_y, height-1)+1):
                needed[row, column] = True


@lru_cache(maxsize=8)
def _kernels_exact(shape):
    """Whether the kernels' projections equal numpy's matmuls here (they assume the BLAS rounds 3-term rows
    left to right without fused multiply-adds), in the layouts the classes use, for images of this shape."""
    rng = np.random.default_rng(1729)
    k = np.asarray([[.51*shape[1], 0., shape[1]/2], [0., .51*shape[1], shape[0]/2], [0., 0., 1.]])
    for _ in range(16):
        angle = rng.uniform(-np.pi, np.pi, 3)
        camera = np.eye(4); camera[:3, :3] = _rotation(angle); camera[:3, 3] = rng.normal(0, .3, 3)
        camera_world = body_pose(np.r_[rng.normal(0, 20, 2), angle[0]])@camera
        lower = rng.normal(0, 4, (512, 3)); boxes = rng.normal(0, 4, (64, 8, 3))
        depth = rng.uniform(-.1, 1.1, shape).astype(np.float16).astype(np.float32)
        rays = _rays(k.tobytes(), shape)[0]
        world = (rays*(depth*ObservedBSplineRoutes.maximum)[..., None])@camera_world[:3, :3].T+camera_world[:3, 3]
        points, bounds = _back_project(depth, rays, camera_world, np.float32(ObservedBSplineRoutes.maximum), -np.inf, np.inf)
        if not (np.array_equal(_project(lower, camera_world), ((lower[:, None]-camera_world[:3, 3])@camera_world[:3, :3])[:, 0]) and
                np.array_equal(_project(boxes.reshape(-1, 3), camera_world), ((boxes-camera_world[:3, 3])@camera_world[:3, :3]).reshape(-1, 3)) and
                np.array_equal(points, world[(depth > 0) & (depth < 1)])):
            return False
    return True


def _rotation(angle):
    a, b, c = angle
    x = np.asarray([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    y = np.asarray([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    z = np.asarray([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return z@y@x


@njit(cache=True, nogil=True)
def _project(points, camera_world):
    """(points-t)@R through the kernels' _planar and _vertical."""
    output = np.empty((len(points), 3)); partial = np.empty((4, 3)); term = np.empty(3)
    for index in range(len(points)):
        _planar(points[index, 0], points[index, 0], points[index, 1], points[index, 1], camera_world, partial)
        _vertical(points[index, 2], camera_world, term)
        for j in range(3):
            output[index, j] = partial[0, j]+term[j]
    return output
