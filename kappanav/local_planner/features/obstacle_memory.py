"""Shared calibrated depth visibility and raster geometry for four-frame observations."""

import numpy as np

from kappanav.common.physical import PATH_CONFIGURATION_QUERY_SPACING_M

GRID_SIZE = 64
VOXEL_M = PATH_CONFIGURATION_QUERY_SPACING_M


def _rectangle_minimum(image, x0, y0, x1, y1):
    """Exact inclusive rectangle minima using four overlapping dyadic blocks."""
    height, width = y1 - y0 + 1, x1 - x0 + 1
    ky = np.floor(np.log2(height)).astype(np.int64)
    kx = np.floor(np.log2(width)).astype(np.int64)
    result = np.empty(len(x0), dtype=image.dtype)
    # FP16 values widen exactly; return minima in the original dtype below.
    # NumPy CPU minimum is much faster in FP32 than software FP16 arithmetic.
    rows = image.astype(np.float32) if image.dtype == np.float16 else image
    for j in range(int(ky.max(initial=0)) + 1):
        if j:
            offset = 1 << (j - 1)
            rows = np.minimum(rows[:-offset], rows[offset:])
        blocks = rows
        for i in range(int(kx.max(initial=0)) + 1):
            if i:
                offset = 1 << (i - 1)
                blocks = np.minimum(blocks[:, :-offset], blocks[:, offset:])
            selected = (ky == j) & (kx == i)
            left, top = x0[selected], y0[selected]
            right = x1[selected] - (1 << i) + 1
            bottom = y1[selected] - (1 << j) + 1
            result[selected] = np.minimum(
                np.minimum(blocks[top, left], blocks[top, right]),
                np.minimum(blocks[bottom, left], blocks[bottom, right]),
            )
    return result


def volumes_observed_clear(world, depth, intrinsic, camera_to_world, maximum):
    """Entire boxes are in front of valid depth; invalid or occluded pixels never clear.

    world is [N,8,3] calibrated box corners. Shared by occupied-memory clearing
    and experimental free-volume evidence, including FP16 depth tolerance.
    """
    optical = (world-camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = optical[..., 2]
    # Behind-camera corners cannot clear a voxel; only project positive Z.
    positive_z = np.where(z > 0, z, 1.0)
    u = intrinsic[0,0]*optical[...,0]/positive_z + intrinsic[0,2] - .5
    v = intrinsic[1,1]*optical[...,1]/positive_z + intrinsic[1,2] - .5
    h, w = depth.shape
    inside = ((z > 0) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)).all(1)
    clear = np.zeros(len(world), dtype=bool)
    if not inside.any():
        return clear
    # Invisible boxes cannot be refuted. Do not query their clipped image rectangles.
    u, v, z = u[inside], v[inside], z[inside]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    measured = _rectangle_minimum(
        depth, x0.min(1), y0.min(1), x1.max(1), y1.max(1)
    ) * maximum
    # The entire projected voxel rectangle must be observed free.
    # This tolerance covers normalized FP16 depth storage, not scene motion.
    clear[inside] = measured > z.max(1) + maximum * np.finfo(np.float16).eps
    return clear
