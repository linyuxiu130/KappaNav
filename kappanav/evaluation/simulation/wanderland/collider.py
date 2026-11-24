"""Rebuild a ground-robot collision mesh from a Wanderland raw LiDAR cloud (robot-independent scene geometry).

Why: the released scene.usdz collider is a marching-cubes iso-surface of point *density*
(8x4x8 cm voxels, iso = mean density, quadric decimation, 5 Taubin passes; see
wanderland/reconstruction/gsplat/examples/datasets/mesh_from_pointcloud.py). On walkable ground it is
~3.5x rougher than the LiDAR (p90 height change over 10 cm: 4.9 cm vs 1.4 cm) and sits ~5 cm above it,
so a 5 cm-climb robot such as CurveNav's Dingo finds no connected ground at all.

What: in the canonical Z-up frame (x, z, -y of the dataset),
  ground   : lowest dense point layer per GROUND_M cell, 3x3 median over observed cells (keeps kerbs as
             steps), a wider median on green (grass) cells, small holes filled; meshed as a height field. Cells never observed stay open.
  obstacles: non-ground points 3 cm..OBST_TOP_M above local ground and within NEAR_GROUND_M of observed
             ground, except low green vegetation (grass blades below GRASS_TOP_M, excess green >= GRASS_EXG),
             which a wheeled base drives through, voxelized in (x, y, height above ground), closed by one voxel, small components dropped,
             meshed with marching cubes per tile and lifted back onto the ground surface.
  boundary : the scan ends where the ground does. Every open edge of the observed ground (the scene border and
             unobserved gaps wider than HOLE_FILL_M) gets a vertical wall, so the depth camera sees where the
             world ends instead of looking into a void that the navigation C-space treats as blocked.
"""
import math
import warnings

import numpy as np
from scipy import ndimage as nd

GROUND_M = 0.05          # ground height-field cell
DENSE_POINTS = 4         # a ground layer needs this many points ...
DENSE_BAND_M = 0.03      # ... within this height band
COARSE_M = 0.20          # second scale for sparse floors
LAYER_JUMP_M = 0.5       # a fine cell this far above the coarse layer read a ceiling/table/canopy
HOLE_FILL_M = 0.30       # unobserved ground gaps up to this size are interpolated
OBST_M = 0.05            # obstacle voxel
OBST_MIN_M = 0.03        # points this close to the ground are ground texture, not obstacles
OBST_TOP_M = 4.0         # obstacles are kept up to this height above ground (camera view, not roofs)
GRASS_TOP_M = 0.20       # green points below this are grass the chassis presses down, not obstacles ...
GRASS_EXG = 0.10         # ... green = excess green (2g - r - b) / (r + g + b); kerbs, rocks, bins stay obstacles
GRASS_SMOOTH_M = 0.25    # green ground is median-smoothed over this window (what a wheel on grass feels)
MIN_COMPONENT_VOXELS = 8
NEAR_GROUND_M = 5.0      # obstacles farther than this from observed ground are dropped
TILE = 256               # marching-cubes tile (voxels per side)
TERRACE_M = 0.05         # height differences below this are reconstruction terraces, not steps (the lowest
                         # step any simulated robot is stopped by; robots apply their own limits on navigation)
WALL_TOP_M = 1.0         # boundary walls rise this far above the ground edge ...
WALL_FOOT_M = 0.3        # ... and reach this far below it


def _fill_nearest(a):
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = nd.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _lowest_dense(P, lo, cell):
    """Per cell, the median of the lowest DENSE_POINTS points spanning at most DENSE_BAND_M (NaN if none)."""
    ij = np.floor((P[:, :2] - lo) / cell).astype(np.int64)
    W, H = ij.max(0) + 1
    c = ij[:, 0] * H + ij[:, 1]
    order = np.lexsort((P[:, 2], c))
    c, z = c[order], P[order, 2]
    k = DENSE_POINTS
    dense = np.zeros(len(z), bool)
    dense[:-(k - 1)] = (c[k - 1:] == c[:-(k - 1)]) & (z[k - 1:] - z[:-(k - 1)] <= DENSE_BAND_M)
    first = np.nonzero(dense)[0]
    ids, at = np.unique(c[first], return_index=True)
    out = np.full(W * H, np.nan, np.float32)
    out[ids] = z[first[at] + (k - 1) // 2]
    return out.reshape(W, H)


def _ground_greenness(P, rgb, ground, lo):
    """Mean excess green of the points within DENSE_BAND_M of each cell's ground height (NaN where none)."""
    W, H = ground.shape
    ij = np.minimum(np.floor((P[:, :2] - lo) / GROUND_M).astype(np.int64), [W - 1, H - 1])
    c = ij[:, 0] * H + ij[:, 1]
    near = np.abs(P[:, 2] - ground.ravel()[c]) <= DENSE_BAND_M
    n = np.bincount(c[near], minlength=W * H)
    exg = np.bincount(c[near], excess_green(rgb[near]), minlength=W * H)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (exg / n).reshape(W, H)


def ground_surface(P, rgb):
    """Return (ground_z [W,H] with NaN where unobserved, origin_xy, is_ground) on the GROUND_M grid.

    Ground is the lowest *dense* layer of each cell: the lowest DENSE_POINTS points spanning at most DENSE_BAND_M,
    with a COARSE_M fallback where the 5 cm cell is too sparse (see below).
    Sparse returns below a floor (reflections, noise) do not qualify; walls, trunks and poles contribute their
    foot. No neighbourhood window is involved, so narrow raised ground (1.5 m kerbed sidewalks, corridors on
    a raised floor) is kept; objects whose underside is unseen (cars) become solid blocks from their lowest
    visible point, and the navigation step removes the small plateaus they form.
    """
    lo = P[:, :2].min(0)
    ground = _lowest_dense(P, lo, GROUND_M)
    W, H = ground.shape
    # Sparse floors (dark or glossy surfaces, the scanner's blind spot) may lack a dense layer in a 5 cm cell, which
    # then reads the next dense layer up (ceiling, table, canopy). The same rule on COARSE_M cells sees enough of
    # the floor; use it where the fine cell is empty or sits a whole layer (>= LAYER_JUMP_M) above it. Kerbs and
    # steps are far below LAYER_JUMP_M, so their fine-scale edges are untouched.
    r = int(round(COARSE_M / GROUND_M))
    coarse_grid = _lowest_dense(P, lo, COARSE_M)
    coarse = np.repeat(np.repeat(coarse_grid, r, 0), r, 1)[:W, :H]
    if coarse.shape != (W, H):
        coarse = np.pad(coarse, ((0, W - coarse.shape[0]), (0, H - coarse.shape[1])), constant_values=np.nan)
    # Replicating coarse cells turns a continuous sparse slope into 20 cm terraces.
    # Interpolate only within a 5 cm height range; real kerbs remain discontinuities.
    i, j = np.meshgrid((np.arange(W) + .5) / r - .5, (np.arange(H) + .5) / r - .5, indexing='ij')
    ii = np.clip(np.floor(i).astype(int), 0, coarse_grid.shape[0] - 2)
    jj = np.clip(np.floor(j).astype(int), 0, coarse_grid.shape[1] - 2)
    tx, ty = np.clip(i - ii, 0, 1), np.clip(j - jj, 0, 1)
    samples = np.stack([coarse_grid[ii, jj], coarse_grid[ii + 1, jj],
                        coarse_grid[ii, jj + 1], coarse_grid[ii + 1, jj + 1]])
    smooth = np.isfinite(samples).all(0) & (np.ptp(samples, axis=0) <= TERRACE_M)
    interpolated = (samples[0] * (1 - tx) * (1 - ty) + samples[1] * tx * (1 - ty)
                    + samples[2] * (1 - tx) * ty + samples[3] * tx * ty)
    coarse = np.where(smooth, interpolated, coarse)
    with np.errstate(invalid='ignore'):
        use_coarse = np.isfinite(coarse) & (~np.isfinite(ground) | (ground > coarse + LAYER_JUMP_M))
    ground = np.where(use_coarse, coarse, ground).astype(np.float32)
    # A sub-footprint patch of canopy/roof returns is an object, not a raised floor.
    # Remove that unsupported layer; the existing small-hole rule reconstructs the ground below.
    local_low = nd.minimum_filter(np.where(np.isfinite(ground), ground, np.inf), size=7)
    elevated = ground > local_low + LAYER_JUMP_M
    components, count = nd.label(elevated, structure=np.ones((3, 3), bool))
    for label, box in enumerate(nd.find_objects(components), 1):
        if box is not None and all(axis.stop - axis.start <= int(round(HOLE_FILL_M / GROUND_M)) for axis in box):
            patch = ground[box]
            patch[components[box] == label] = np.nan
    seen = np.isfinite(ground)
    # median over observed neighbours only (keeps kerb steps, removes single-cell spikes and pits)
    pad = np.pad(ground, 1, constant_values=np.nan)
    stack = np.stack([pad[a:a + W, b:b + H] for a in range(3) for b in range(3)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN neighbourhoods stay NaN
        med = np.nanmedian(stack, axis=0)
    ground = np.where(seen, med, np.nan).astype(np.float32)
    # Grass: blades scatter the lowest dense layer by a few cm, and on a 5 cm grid that reads as >30 deg micro-slopes
    # and >7 cm steps at a few percent of cells, each of which the navmesh then erodes by the robot radius. A wheel
    # presses grass down; on green ground take the median over GRASS_SMOOTH_M. Non-green cells (kerbs, steps) keep
    # their own height.
    green = _ground_greenness(P, rgb, ground, lo) >= GRASS_EXG
    if green.any():
        k = int(round(GRASS_SMOOTH_M / GROUND_M)) | 1
        pad = np.pad(ground, k // 2, constant_values=np.nan)
        stack = np.stack([pad[a:a + W, b:b + H] for a in range(k) for b in range(k)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ground = np.where(green, np.nanmedian(stack, axis=0), ground).astype(np.float32)
        del stack
    # fill small unobserved gaps from their observed neighbourhood
    r = int(round(HOLE_FILL_M / GROUND_M / 2))
    closed = nd.binary_closing(seen, structure=np.ones((2 * r + 1, 2 * r + 1)), border_value=0)
    gap = closed & ~seen
    if gap.any():
        ground[gap] = _fill_nearest(ground)[gap]
    # Suppress terraces below TERRACE_M without rounding a real kerb.
    pad = np.pad(ground, 1, constant_values=np.nan)
    stack = np.stack([pad[a:a + W, b:b + H] for a in range(3) for b in range(3)])
    close = np.isfinite(stack) & (np.abs(stack - ground) <= TERRACE_M)
    ground = np.divide(np.where(close, stack, 0).sum(0), close.sum(0),
                       out=ground.copy(), where=close.any(0)).astype(np.float32)
    ij = np.minimum(np.floor((P[:, :2] - lo) / GROUND_M).astype(np.int64), [W - 1, H - 1])
    gz = ground[ij[:, 0], ij[:, 1]]
    is_ground = np.abs(P[:, 2] - gz) <= DENSE_BAND_M
    return ground, lo.astype(np.float64), is_ground


def heightfield_mesh(ground, origin, cell):
    """Two triangles per pair of adjacent observed cells' centres (a cell-centre lattice)."""
    W, H = ground.shape
    ok = np.isfinite(ground)
    vid = np.full((W, H), -1, np.int64)
    vid[ok] = np.arange(ok.sum())
    i, j = np.nonzero(ok)
    verts = np.c_[origin[0] + (i + .5) * cell, origin[1] + (j + .5) * cell, ground[ok]]
    a, b, c, d = vid[:-1, :-1], vid[1:, :-1], vid[1:, 1:], vid[:-1, 1:]
    faces = []
    for t in ((a, b, c), (a, c, d)):
        m = (t[0] >= 0) & (t[1] >= 0) & (t[2] >= 0)
        faces.append(np.stack([t[0][m], t[1][m], t[2][m]], 1))
    return verts, np.concatenate(faces)  # counter-clockwise seen from +Z: normals point up


def boundary_walls(ground, origin, cell):
    """Vertical quads on every side of an observed ground cell that faces an unobserved one (or the grid border),
    from WALL_FOOT_M below to WALL_TOP_M above that cell's ground."""
    seen = np.isfinite(ground)
    pad = np.pad(seen, 1, constant_values=False)
    verts, faces = [], []
    # (neighbour offset, the face's two corners in cell units relative to the cell's lower corner)
    for (di, dj), (a, b) in (((1, 0), ((1, 0), (1, 1))), ((-1, 0), ((0, 1), (0, 0))),
                             ((0, 1), ((1, 1), (0, 1))), ((0, -1), ((0, 0), (1, 0)))):
        open_side = seen & ~pad[1 + di:1 + di + seen.shape[0], 1 + dj:1 + dj + seen.shape[1]]
        i, j = np.nonzero(open_side)
        if not len(i):
            continue
        g = ground[i, j]
        corner = lambda c: origin + (np.c_[i, j] + c) * cell
        pa, pb = corner(a), corner(b)
        lo, hi = g - WALL_FOOT_M, g + WALL_TOP_M
        quad = np.stack([np.c_[pa, lo], np.c_[pb, lo], np.c_[pb, hi], np.c_[pa, hi]], 1)  # [n, 4, 3]
        base = sum(len(v) for v in verts)
        verts.append(quad.reshape(-1, 3))
        k = base + 4 * np.arange(len(i))[:, None]
        faces.append(np.concatenate([k + [0, 1, 2], k + [0, 2, 3]]))
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(verts), np.concatenate(faces).astype(np.int64)


def _bilinear(grid, origin, cell, xy):
    f = (xy - origin) / cell - .5
    i0 = np.clip(np.floor(f).astype(np.int64), 0, np.array(grid.shape) - 2)
    t = np.clip(f - i0, 0, 1)
    g00, g10 = grid[i0[:, 0], i0[:, 1]], grid[i0[:, 0] + 1, i0[:, 1]]
    g01, g11 = grid[i0[:, 0], i0[:, 1] + 1], grid[i0[:, 0] + 1, i0[:, 1] + 1]
    return (g00 * (1 - t[:, 0]) * (1 - t[:, 1]) + g10 * t[:, 0] * (1 - t[:, 1])
            + g01 * (1 - t[:, 0]) * t[:, 1] + g11 * t[:, 0] * t[:, 1])


def excess_green(rgb):
    rgb = rgb.astype(np.float32)
    return (2 * rgb[:, 1] - rgb[:, 0] - rgb[:, 2]) / (rgb.sum(1) + 1)


def obstacle_mesh(P, rgb, ground, origin):
    """Obstacles voxelized in (x, y, height-above-ground); vertices are lifted back by the ground height.
    Only points within NEAR_GROUND_M of observed ground matter for a ground robot and its camera."""
    from skimage import measure
    assert OBST_M == GROUND_M, 'obstacle columns share the ground grid'
    seen = np.isfinite(ground)
    near = nd.distance_transform_edt(~seen) * GROUND_M <= NEAR_GROUND_M
    gref = _fill_nearest(ground)
    gi = np.floor((P[:, :2] - origin) / GROUND_M).astype(np.int64)
    inside = (gi >= 0).all(1) & (gi < ground.shape).all(1)
    P, rgb, gi = P[inside], rgb[inside], gi[inside]
    h = P[:, 2] - gref[gi[:, 0], gi[:, 1]]
    grass = (h < GRASS_TOP_M) & (excess_green(rgb) >= GRASS_EXG)
    keep = (h > OBST_MIN_M) & (h < OBST_TOP_M) & ~grass & near[gi[:, 0], gi[:, 1]]
    gi, k = gi[keep], np.floor(h[keep] / OBST_M).astype(np.int64)
    if len(k) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    levels = int(math.ceil(OBST_TOP_M / OBST_M))
    W, H = ground.shape
    cols = np.zeros((W, H, levels), bool)
    cols[gi[:, 0], gi[:, 1], k] = True
    halo, ball = 2, np.ones((3, 3, 3), bool)
    verts, faces, off = [], [], 0
    for x0 in range(0, W, TILE):
        for y0 in range(0, H, TILE):
            xa, ya = max(x0 - halo, 0), max(y0 - halo, 0)
            block = cols[xa:x0 + TILE + halo, ya:y0 + TILE + halo]
            if not block.any():
                continue
            block = nd.binary_closing(block, structure=ball)
            lab, n = nd.label(block, structure=ball)
            if n:
                sizes = np.bincount(lab.ravel())
                good = sizes >= MIN_COMPONENT_VOXELS
                good[0] = False
                block = good[lab]
            # Mesh only this tile's own cells (+1 to stitch with the next tile).
            core = block[x0 - xa:x0 - xa + TILE + 1, y0 - ya:y0 - ya + TILE + 1]
            if not core.any():
                continue
            vol = np.pad(core, 1).astype(np.float32)
            vv, ff, _, _ = measure.marching_cubes(vol, 0.5)
            ij = vv[:, :2] - 1 + [x0, y0]            # cell index space; cell c spans [c, c+1)
            xy = origin + (ij + .5) * GROUND_M
            z = _bilinear(gref, origin, GROUND_M, xy) + (vv[:, 2] - 1 + .5) * OBST_M
            verts.append(np.c_[xy, z])
            faces.append(ff[:, ::-1] + off)  # skimage winds solids inward; make normals point outward
            off += len(vv)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.concatenate(verts), np.concatenate(faces).astype(np.int64)


def rebuild(P, rgb):
    """P: canonical Z-up points [N,3], rgb their colours [N,3]. Returns verts, faces (canonical) and a summary."""
    ground, origin, is_ground = ground_surface(P, rgb)
    gv, gf = heightfield_mesh(ground, origin, GROUND_M)
    ov, of = obstacle_mesh(P, rgb, ground, origin)
    wv, wf = boundary_walls(ground, origin, GROUND_M)
    verts = np.concatenate([gv, ov, wv])
    faces = np.concatenate([gf, of + len(gv), wf + len(gv) + len(ov)])
    info = dict(collider_source='rebuilt_from_raw_pcd', ground_cells=int(np.isfinite(ground).sum()),
                ground_area_m2=round(float(np.isfinite(ground).sum() * GROUND_M ** 2), 1),
                ground_point_fraction=round(float(is_ground.mean()), 3),
                ground_triangles=int(len(gf)), obstacle_triangles=int(len(of)), boundary_wall_triangles=int(len(wf)),
                params=dict(ground_m=GROUND_M, dense_points=DENSE_POINTS, dense_band_m=DENSE_BAND_M, coarse_m=COARSE_M,
                            layer_jump_m=LAYER_JUMP_M,
                            hole_fill_m=HOLE_FILL_M, obstacle_voxel_m=OBST_M, obstacle_top_m=OBST_TOP_M,
                            grass_top_m=GRASS_TOP_M, grass_exg=GRASS_EXG, grass_smooth_m=GRASS_SMOOTH_M,
                            terrace_m=TERRACE_M, wall_top_m=WALL_TOP_M, wall_foot_m=WALL_FOOT_M))
    return verts, faces, ground, origin, info
