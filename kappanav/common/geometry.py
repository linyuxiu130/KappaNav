"""Shared geometry and clearance-aware route planning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import heapq
import math
import threading
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, PPoly, RectBivariateSpline
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from numba import njit

from kappanav.common.physical import ROBOT_FOOTPRINT_RADIUS_M


SAFETY_STEP_M = 0.025
MIN_CLEARANCE_M = 0.0
# Shortest P0-P1 handle: a tighter body-x departure would be a cusp, not a label.
MIN_START_HANDLE_M = ROBOT_FOOTPRINT_RADIUS_M / 2
# Exact-boundary margin at the constraint samples; covers the chord between samples.
FIT_MARGIN_M = 0.01
# 12.5 cm turning radius: at 0.5 rad/s the robot still moves at 6 cm/s. An open
# U-turn label measures about 7 rad/m; a near-in-place turn is far above.
MAX_TURN_RATE_PER_M = 8.0
# Turning-radius rule of the label: the widest radius whose curve certifies.
# Solved by continuation from the unconstrained curve (0), each radius warm
# started from the last certified one. Steps of 2.5 cm keep each warm start
# near feasible (10 cm steps stall SLSQP at the first step); 0.5 m is a maze
# corridor's widest U-turn (1.4 m free minus the body).
# Widening stops before the curve passes closer than this to the source
# boundary: executed labels are certified with a ~1 cm sampling pad, and a
# radius bought with millimetres of clearance loses them to tracking error.
LABEL_CLEARANCE_M = 0.05
# _smooth's clearance rows are in centimetres (the turning rows and the objective are O(1)).
CONSTRAINT_UNIT_M = 0.01
# Re-solves of _smooth that add constraint samples where the dense certificate fails.
CERTIFICATE_REFINEMENTS = 3
TURN_RADII_M = (0.0, *np.round(np.arange(1 / MAX_TURN_RATE_PER_M, 0.5 + 1e-9, 0.025), 3))


@njit(cache=True, nogil=True)
def _cubic_basis(t, x, span):
    """Cox-de Boor values and first derivatives of the 4 cubic B-splines nonzero on [t[span], t[span+1])."""
    left = np.empty(4); right = np.empty(4); n = np.zeros((4, 4))
    n[0, 0] = 1.
    for j in range(1, 4):
        left[j] = x - t[span + 1 - j]; right[j] = t[span + j] - x; saved = 0.
        for r in range(j):
            n[j, r] = right[r + 1] + left[j - r]
            temp = n[r, j - 1] / n[j, r]
            n[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        n[j, j] = saved
    value = np.empty(4); derivative = np.empty(4)
    for r in range(4): value[r] = n[r, 3]
    # Degree-3 derivative from the degree-2 values: 3*(N_{i,2}/(t_{i+3}-t_i) - N_{i+1,2}/(t_{i+4}-t_{i+1})).
    lower = np.zeros(5)
    for r in range(3): lower[r + 1] = n[r, 2]
    for r in range(4):
        i = span - 3 + r
        a = lower[r] / (t[i + 3] - t[i]) if t[i + 3] > t[i] else 0.
        b = lower[r + 1] / (t[i + 4] - t[i + 1]) if t[i + 4] > t[i + 1] else 0.
        derivative[r] = 3. * (a - b)
    return value, derivative

@njit(cache=True, nogil=True, inline='always')
def _cubic_basis_into(t, x, span, n, left, right, lower, value, derivative):
    """_cubic_basis into caller-owned scratch arrays: the same operations in the same order, no allocation."""
    n[0, 0] = 1.
    for j in range(1, 4):
        left[j] = x - t[span + 1 - j]; right[j] = t[span + j] - x; saved = 0.
        for r in range(j):
            n[j, r] = right[r + 1] + left[j - r]
            temp = n[r, j - 1] / n[j, r]
            n[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        n[j, j] = saved
    for r in range(4): value[r] = n[r, 3]
    lower[0] = 0.; lower[4] = 0.
    for r in range(3): lower[r + 1] = n[r, 2]
    for r in range(4):
        i = span - 3 + r
        a = lower[r] / (t[i + 3] - t[i]) if t[i + 3] > t[i] else 0.
        b = lower[r + 1] / (t[i + 4] - t[i + 1]) if t[i + 4] > t[i + 1] else 0.
        derivative[r] = 3. * (a - b)


@njit(cache=True, nogil=True)
def _bicubic_with_gradient_reference(tx, ty, c, x, y):
    """FITPACK bispev/parder for a cubic tensor spline: value, d/dx, d/dy in one pass (arguments clamped to the knot range)."""
    nx = len(tx) - 4; ny = len(ty) - 4
    out = np.empty((len(x), 3))
    n = np.zeros((4, 4)); left = np.empty(4); right = np.empty(4); lower = np.zeros(5)
    bx = np.empty(4); dbx = np.empty(4); by = np.empty(4); dby = np.empty(4)
    for p in range(len(x)):
        u = min(max(x[p], tx[3]), tx[nx]); v = min(max(y[p], ty[3]), ty[ny])
        i = min(max(np.searchsorted(tx, u, side='right') - 1, 3), nx - 1)
        j = min(max(np.searchsorted(ty, v, side='right') - 1, 3), ny - 1)
        _cubic_basis_into(tx, u, i, n, left, right, lower, bx, dbx)
        _cubic_basis_into(ty, v, j, n, left, right, lower, by, dby)
        f = 0.; fx = 0.; fy = 0.
        for a in range(4):
            row = (i - 3 + a) * ny + (j - 3)
            for b in range(4):
                coef = c[row + b]
                f += bx[a] * by[b] * coef; fx += dbx[a] * by[b] * coef; fy += bx[a] * dby[b] * coef
        out[p, 0] = f; out[p, 1] = fx; out[p, 2] = fy
    return out


@njit(cache=True, nogil=True, inline='always')
def _knot_span(t, u, last, step):
    """searchsorted(t, u, side='right') - 1 clamped to [3, last - 1] for u in [t[3], t[last]]: a guess from the mean
    knot step, corrected to the last knot <= u (the same index for any sorted t)."""
    g = int((u - t[3]) / step) + 3 if step > 0 else 3
    g = min(max(g, 0), len(t) - 1)
    while g + 1 < len(t) and t[g + 1] <= u: g += 1
    while g > 0 and t[g] > u: g -= 1
    return min(max(g, 3), last - 1)


@njit(cache=True, nogil=True, error_model='numpy')
def _cubic_basis_rows(t, xs, spans, value, derivative):
    """_cubic_basis_into for every point, unrolled into scalars (the same operations in the same order) and free
    of branches, so the loop vectorizes. No denominator is zero on a valid span (each is a sum of two
    non-negative knot distances spanning it), so numpy's error model changes nothing."""
    for p in range(len(xs)):
        x = xs[p]; s = spans[p]
        tm3 = t[s - 3]; tm2 = t[s - 2]; tm1 = t[s - 1]; t0 = t[s]; t1 = t[s + 1]; t2 = t[s + 2]; t3 = t[s + 3]; t4 = t[s + 4]
        left1 = x - t0; right1 = t1 - x
        n10 = right1 + left1; temp = 1. / n10; n01 = 0. + right1 * temp; n11 = left1 * temp
        left2 = x - tm1; right2 = t2 - x
        n20 = right1 + left2; temp = n01 / n20; n02 = 0. + right1 * temp; saved = left2 * temp
        n21 = right2 + left1; temp = n11 / n21; n12 = saved + right2 * temp; n22 = left1 * temp
        left3 = x - tm2; right3 = t3 - x
        n30 = right1 + left3; temp = n02 / n30; n03 = 0. + right1 * temp; saved = left3 * temp
        n31 = right2 + left2; temp = n12 / n31; n13 = saved + right2 * temp; saved = left2 * temp
        n32 = right3 + left1; temp = n22 / n32; n23 = saved + right3 * temp; n33 = left1 * temp
        # Degree-3 derivative from the degree-2 values, i = s - 3 + r: lower[r] / (t[i+3] - t[i]) - lower[r+1] / (t[i+4] - t[i+1]).
        a0 = 0. / (t0 - tm3) if t0 > tm3 else 0.
        b0 = n02 / (t1 - tm2) if t1 > tm2 else 0.
        b1 = n12 / (t2 - tm1) if t2 > tm1 else 0.
        b2 = n22 / (t3 - t0) if t3 > t0 else 0.
        b3 = 0. / (t4 - t1) if t4 > t1 else 0.
        value[p, 0] = n03; value[p, 1] = n13; value[p, 2] = n23; value[p, 3] = n33
        derivative[p, 0] = 3. * (a0 - b0); derivative[p, 1] = 3. * (b0 - b1)
        derivative[p, 2] = 3. * (b1 - b2); derivative[p, 3] = 3. * (b2 - b3)


@njit(cache=True, nogil=True, error_model='numpy')
def _bicubic_with_gradient(tx, ty, c, x, y):
    """_bicubic_with_gradient_reference, bit for bit: knot spans by a corrected guess instead of a binary search,
    and the basis recursion of all points in one vectorizable pass per axis. A point outside the knot range
    after clamping (nan) sends the whole call to the reference."""
    nx = len(tx) - 4; ny = len(ty) - 4; n = len(x)
    step_x = (tx[nx] - tx[3]) / max(nx - 3, 1); step_y = (ty[ny] - ty[3]) / max(ny - 3, 1)
    us = np.empty(n); vs = np.empty(n); si = np.empty(n, np.int64); sj = np.empty(n, np.int64)
    for p in range(n):
        u = min(max(x[p], tx[3]), tx[nx]); v = min(max(y[p], ty[3]), ty[ny])
        if not (tx[3] <= u <= tx[nx] and ty[3] <= v <= ty[ny]):
            return _bicubic_with_gradient_reference(tx, ty, c, x, y)
        us[p] = u; vs[p] = v
        si[p] = _knot_span(tx, u, nx, step_x); sj[p] = _knot_span(ty, v, ny, step_y)
    bx = np.empty((n, 4)); dbx = np.empty((n, 4)); by = np.empty((n, 4)); dby = np.empty((n, 4))
    _cubic_basis_rows(tx, us, si, bx, dbx)
    _cubic_basis_rows(ty, vs, sj, by, dby)
    out = np.empty((n, 3))
    for p in range(n):
        i = si[p]; j = sj[p]
        f = 0.; fx = 0.; fy = 0.
        for a in range(4):
            row = (i - 3 + a) * ny + (j - 3)
            for b in range(4):
                coef = c[row + b]
                f += bx[p, a] * by[p, b] * coef; fx += dbx[p, a] * by[p, b] * coef; fy += bx[p, a] * dby[p, b] * coef
        out[p, 0] = f; out[p, 1] = fx; out[p, 2] = fy
    return out


@njit(cache=True, nogil=True)
def _path_samples(coordinates):
    """Vertices, grid-line crossings and their midpoints in grid coordinates."""
    total = len(coordinates)
    pieces = []
    for s in range(len(coordinates) - 1):
        a = coordinates[s]; b = coordinates[s + 1]
        fractions = [0., 1.]
        for axis in range(2):
            if a[axis] != b[axis]:
                edge = np.floor(min(a[axis], b[axis])) + 1.
                stop = np.ceil(max(a[axis], b[axis]))
                while edge < stop:
                    fractions.append((edge - a[axis]) / (b[axis] - a[axis])); edge += 1.
        f = np.unique(np.array(fractions))
        f = np.concatenate((f, (f[1:] + f[:-1]) / 2))
        piece = np.empty((len(f), 2))
        for i in range(len(f)):
            piece[i, 0] = a[0] + f[i] * (b[0] - a[0]); piece[i, 1] = a[1] + f[i] * (b[1] - a[1])
        pieces.append(piece); total += len(f)
    samples = np.empty((total, 2)); samples[:len(coordinates)] = coordinates; k = len(coordinates)
    for piece in pieces:
        samples[k:k + len(piece)] = piece; k += len(piece)
    return samples


@njit(cache=True, nogil=True)
def _path_cells(coordinates):
    """Four candidate cells per sample; a grid-line contact touches both sides."""
    samples = _path_samples(coordinates); total = len(samples)
    eps = 4 * np.finfo(np.float64).eps
    cells = np.empty((4 * total, 2), dtype=np.int64)
    for i in range(total):
        h = np.empty(2, dtype=np.int64); l = np.empty(2, dtype=np.int64)
        for axis in range(2):
            c = samples[i, axis]; r = np.rint(c)
            if abs(c - r) <= eps * max(abs(c), 1.): c = r
            h[axis] = np.int64(np.floor(c)); l[axis] = np.int64(np.ceil(c)) - 1
        cells[i, 0] = h[0]; cells[i, 1] = h[1]
        cells[total + i, 0] = l[0]; cells[total + i, 1] = l[1]
        cells[2 * total + i, 0] = l[0]; cells[2 * total + i, 1] = h[1]
        cells[3 * total + i, 0] = h[0]; cells[3 * total + i, 1] = l[1]
    return cells


@njit(cache=True, nogil=True)
def _safe_path(coordinates, free, clearance, minimum, margin):
    eps = 4 * np.finfo(np.float64).eps
    for sample in _path_samples(coordinates):
        x, y = sample[0], sample[1]
        rx, ry = np.rint(x), np.rint(y)
        if abs(x - rx) <= eps * max(abs(x), 1.): x = rx
        if abs(y - ry) <= eps * max(abs(y), 1.): y = ry
        hx, hy = np.int64(np.floor(x)), np.int64(np.floor(y))
        lx, ly = np.int64(np.ceil(x)) - 1, np.int64(np.ceil(y)) - 1
        if not (0 <= lx <= hx < free.shape[0] and 0 <= ly <= hy < free.shape[1]): return False
        for i, j in ((hx, hy), (lx, ly), (lx, hy), (hx, ly)):
            if not free[i, j] or not clearance[i, j] + margin >= minimum: return False
    return True


@njit(cache=True, nogil=True)
def _astar_cells(allowed, signed, cell_size, sx, sy, tx, ty):
    """Clearance-weighted 8-neighbour A*; same costs, heuristic and heap order as the reference. Empty if unreachable."""
    nx_, ny_ = allowed.shape
    cost = np.full((nx_, ny_), np.inf); parent = np.full((nx_, ny_, 2), -1, dtype=np.int64)
    closed = np.zeros((nx_, ny_), dtype=np.bool_)
    cost[sx, sy] = 0.
    moves = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
    queue = [(0.0, 0.0, float(sx), float(sy))]
    while len(queue):
        _, current, fx, fy = heapq.heappop(queue)
        x = int(fx); y = int(fy)
        if closed[x, y] or current > cost[x, y] + 1e-12:
            continue
        if x == tx and y == ty:
            path = [(tx, ty)]
            while path[-1][0] != sx or path[-1][1] != sy:
                px = parent[path[-1][0], path[-1][1], 0]; py = parent[path[-1][0], path[-1][1], 1]
                if px < 0:
                    return np.empty((0, 2), dtype=np.int32)
                path.append((px, py))
            out = np.empty((len(path), 2), dtype=np.int32)
            for i in range(len(path)):
                out[i, 0] = path[len(path) - 1 - i][0]; out[i, 1] = path[len(path) - 1 - i][1]
            return out
        closed[x, y] = True
        for dx, dy, step in moves:
            nx = x + dx; ny = y + dy
            if not (0 <= nx < nx_ and 0 <= ny < ny_) or not allowed[nx, ny]:
                continue
            if dx != 0 and dy != 0 and (not allowed[x + dx, y] or not allowed[x, y + dy]):
                continue
            clear = 0.5 * (signed[x, y] + signed[nx, ny])
            trial = current + step * cell_size * (1.0 + math.exp(1.0 - clear / ROBOT_FOOTPRINT_RADIUS_M))
            if trial + 1e-12 >= cost[nx, ny]:
                continue
            cost[nx, ny] = trial; parent[nx, ny, 0] = x; parent[nx, ny, 1] = y
            heuristic = math.hypot(nx - tx, ny - ty) * cell_size
            heapq.heappush(queue, (trial + heuristic, trial, float(nx), float(ny)))
    return np.empty((0, 2), dtype=np.int32)


@njit(cache=True, nogil=True)
def _nearest_boundary(xy, origin, h, start, end, bucket_start, nbx, nby):
    """Exact nearest free/obstacle cell edge: edges bucketed by the grid cell of their midpoint, searched in
    rings of buckets around each point until the next ring cannot be closer (its edges lie at least
    (k - 1/2) cells away). Returns the distance and the unit offset from the edge to the point."""
    distance = np.empty(len(xy)); offset = np.empty((len(xy), 2))
    for p in range(len(xy)):
        px = xy[p, 0]; py = xy[p, 1]
        qx = int(np.floor((px - origin[0]) / h)); qy = int(np.floor((py - origin[1]) / h))
        first = max(max(0 - qx, qx - (nbx - 1)), max(0 - qy, qy - (nby - 1)), 0)
        last = max(abs(qx), abs(qx - (nbx - 1)), abs(qy), abs(qy - (nby - 1)))
        best = np.inf; bx = 0.; by = 0.
        for k in range(first, last + 1):
            for ix in range(qx - k, qx + k + 1):
                if ix < 0 or ix >= nbx: continue
                step = 1 if (ix == qx - k or ix == qx + k) else 2 * k
                iy = qy - k
                while iy <= qy + k:
                    if 0 <= iy < nby:
                        b = ix * nby + iy
                        for s in range(bucket_start[b], bucket_start[b + 1]):
                            ax = start[s, 0]; ay = start[s, 1]; dx = end[s, 0] - ax; dy = end[s, 1] - ay
                            f = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
                            f = min(max(f, 0.), 1.)
                            ox = px - (ax + f * dx); oy = py - (ay + f * dy); d = math.sqrt(ox * ox + oy * oy)
                            if d < best: best = d; bx = ox; by = oy
                    iy += step if k > 0 else 1
            if best <= (k - .5) * h: break
        distance[p] = best; scale = max(best, 1e-12); offset[p, 0] = bx / scale; offset[p, 1] = by / scale
    return distance, offset


# Grid.boundary_distance: points farther than BOUNDARY_RING_M from the boundary (by the cell-centre distance) try
# their BOUNDARY_CANDIDATES nearest edge midpoints first; nearer points, and any the candidates may miss, use the
# ring search, whose cost grows with the squared distance in cells.
BOUNDARY_CANDIDATES = 32
BOUNDARY_RING_M = 0.8
# Fewer far points than this per call are cheaper by ring search than by one KD query (its fixed call cost).
BOUNDARY_KD_MIN_POINTS = 8


@njit(cache=True, nogil=True)
def _nearest_candidate_boundary(xy, origin, h, start, end, candidates, farthest, nby):
    """_nearest_boundary restricted to each point's candidate edges, where these provably hold every edge it could
    return: edges are h long, so an edge within the true distance d* <= best of a point has its midpoint within
    best + h/2, and `farthest` is the distance of the farthest candidate midpoint. Equal distances keep the edge
    _nearest_boundary meets first (ring k, then sorted edge index), so values and offsets are the same.
    `exact[p]` is False where the candidates may miss an edge; those points need _nearest_boundary."""
    distance = np.empty(len(xy)); offset = np.empty((len(xy), 2)); exact = np.empty(len(xy), np.bool_)
    for p in range(len(xy)):
        px = xy[p, 0]; py = xy[p, 1]
        qx = int(np.floor((px - origin[0]) / h)); qy = int(np.floor((py - origin[1]) / h))
        best = np.inf; bx = 0.; by = 0.; ring = 0; first = 0
        for c in range(candidates.shape[1]):
            s = candidates[p, c]
            ax = start[s, 0]; ay = start[s, 1]; dx = end[s, 0] - ax; dy = end[s, 1] - ay
            f = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
            f = min(max(f, 0.), 1.)
            ox = px - (ax + f * dx); oy = py - (ay + f * dy); d = math.sqrt(ox * ox + oy * oy)
            mx = int(np.floor(((ax + end[s, 0]) / 2 - origin[0]) / h)); my = int(np.floor(((ay + end[s, 1]) / 2 - origin[1]) / h))
            k = max(abs(mx - qx), abs(my - qy))
            if d < best or (d == best and (k < ring or (k == ring and s < first))):
                best = d; bx = ox; by = oy; ring = k; first = s
        exact[p] = farthest[p] > best + h / 2 + 1e-9
        distance[p] = best; scale = max(best, 1e-12); offset[p, 0] = bx / scale; offset[p, 1] = by / scale
    return distance, offset, exact


@njit(cache=True, nogil=True, inline='always')
def _ring_nearest(px, py, qx, qy, h, start, end, bucket_start, nbx, nby):
    """_nearest_boundary for one point (the same rings, bucket order, arithmetic and break)."""
    first = max(max(0 - qx, qx - (nbx - 1)), max(0 - qy, qy - (nby - 1)), 0)
    last = max(abs(qx), abs(qx - (nbx - 1)), abs(qy), abs(qy - (nby - 1)))
    best = np.inf; bx = 0.; by = 0.
    for k in range(first, last + 1):
        for ix in range(qx - k, qx + k + 1):
            if ix < 0 or ix >= nbx: continue
            step = 1 if (ix == qx - k or ix == qx + k) else 2 * k
            iy = qy - k
            while iy <= qy + k:
                if 0 <= iy < nby:
                    b = ix * nby + iy
                    for s in range(bucket_start[b], bucket_start[b + 1]):
                        ax = start[s, 0]; ay = start[s, 1]; dx = end[s, 0] - ax; dy = end[s, 1] - ay
                        f = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
                        f = min(max(f, 0.), 1.)
                        ox = px - (ax + f * dx); oy = py - (ay + f * dy); d = math.sqrt(ox * ox + oy * oy)
                        if d < best: best = d; bx = ox; by = oy
                iy += step if k > 0 else 1
        if best <= (k - .5) * h: break
    return best, bx, by


# A cell's candidate edges reach this far beyond its centre's nearest-edge distance (in cells): every point of the
# cell is within h/sqrt(2) of the centre, so its nearest edge lies within D + sqrt(2) h of the centre.
_CELL_CANDIDATE_REACH = 1.5


@njit(cache=True, nogil=True)
def _cell_boundary(xy, origin, h, start, end, bucket_start, nbx, nby, free, slot, pointer, edges, state, value, gradient):
    """Grid.boundary_distance from per-cell candidate edge lists, built on first use and kept in `slot`/`pointer`/
    `edges` (`state` = lists, edges used). A cell's list holds every edge within D + 1.5 h of its centre (D: the
    centre's nearest-edge distance), in _nearest_boundary's visiting order (ring around the cell, then edge index),
    so the first strict minimum over it is the edge the ring search returns, with the same arithmetic. Points
    outside the bucket range use the ring search itself. Returns 1 where the cache arrays are full (nothing is
    returned then; grow and call again), 2 for a non-finite or out-of-range point (use the reference), else 0."""
    nx_ = free.shape[0]; ny_ = free.shape[1]
    for p in range(len(xy)):
        px = xy[p, 0]; py = xy[p, 1]
        fx = np.floor((px - origin[0]) / h); fy = np.floor((py - origin[1]) / h)
        if not (abs(fx) < 1e9 and abs(fy) < 1e9): return 2
        qx = int(fx); qy = int(fy)
        if 0 <= qx < nbx and 0 <= qy < nby:
            b = qx * nby + qy
            if slot[b] < 0:
                # Build this cell's list: the centre's distance, then every edge within reach, in visiting order.
                cx = origin[0] + (qx + .5) * h; cy = origin[1] + (qy + .5) * h
                centre, _, _ = _ring_nearest(cx, cy, qx, qy, h, start, end, bucket_start, nbx, nby)
                if not centre < np.inf: return 2
                reach = centre + _CELL_CANDIDATE_REACH * h
                rings = int(math.ceil(reach / h)) + 2
                lists = state[0]; used = state[1]
                if lists + 1 >= len(pointer): return 1
                n = used
                for k in range(rings + 1):
                    for ix in range(qx - k, qx + k + 1):
                        if ix < 0 or ix >= nbx: continue
                        step = 1 if (ix == qx - k or ix == qx + k) else 2 * k
                        iy = qy - k
                        while iy <= qy + k:
                            if 0 <= iy < nby:
                                c = ix * nby + iy
                                for s in range(bucket_start[c], bucket_start[c + 1]):
                                    ax = start[s, 0]; ay = start[s, 1]; dx = end[s, 0] - ax; dy = end[s, 1] - ay
                                    f = ((cx - ax) * dx + (cy - ay) * dy) / (dx * dx + dy * dy)
                                    f = min(max(f, 0.), 1.)
                                    ox = cx - (ax + f * dx); oy = cy - (ay + f * dy)
                                    if math.sqrt(ox * ox + oy * oy) <= reach:
                                        if n >= len(edges): return 1
                                        edges[n] = s; n += 1
                            iy += step if k > 0 else 1
                pointer[lists] = used; pointer[lists + 1] = n
                slot[b] = lists; state[0] = lists + 1; state[1] = n
            index = slot[b]
            best = np.inf; bx = 0.; by = 0.
            for e in range(pointer[index], pointer[index + 1]):
                s = edges[e]
                ax = start[s, 0]; ay = start[s, 1]; dx = end[s, 0] - ax; dy = end[s, 1] - ay
                f = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
                f = min(max(f, 0.), 1.)
                ox = px - (ax + f * dx); oy = py - (ay + f * dy); d = math.sqrt(ox * ox + oy * oy)
                if d < best: best = d; bx = ox; by = oy
        else:
            best, bx, by = _ring_nearest(px, py, qx, qy, h, start, end, bucket_start, nbx, nby)
        sign = 1. if (0 <= qx < nx_ and 0 <= qy < ny_ and free[qx, qy]) else -1.
        scale = max(best, 1e-12)
        value[p] = sign * best; gradient[p, 0] = sign * (bx / scale); gradient[p, 1] = sign * (by / scale)
    return 0


_CELL_LISTS_LOCK = threading.Lock()


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class Grid:
    free: np.ndarray
    clearance_m: np.ndarray
    origin_xy: np.ndarray
    cell_size_m: float
    minimum_clearance_m: float = MIN_CLEARANCE_M

    @classmethod
    def load(cls, path: Path) -> "Grid":
        with np.load(path) as values:
            return cls(
                values["free"],
                values["clearance_m"],
                values["origin_xy"],
                float(values["cell_size_m"]),
            )

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        return np.floor(
            (np.asarray(xy, dtype=np.float64) - self.origin_xy) / self.cell_size_m
        ).astype(np.int32)

    def grid_to_world(self, cells: np.ndarray) -> np.ndarray:
        return (
            self.origin_xy
            + (np.asarray(cells, dtype=np.float64) + 0.5) * self.cell_size_m
        )

    def snap(self, xy: np.ndarray, radius_m: float = 0.25) -> np.ndarray:
        center = self.world_to_grid(np.asarray(xy))
        radius = max(1, math.ceil(radius_m / self.cell_size_m))
        lo = np.maximum(center - radius, 0)
        hi = np.minimum(center + radius + 1, self.free.shape)
        cells = np.argwhere(self.free[lo[0] : hi[0], lo[1] : hi[1]]) + lo
        if not len(cells):
            raise PlanningError("no navigable grid cell near endpoint")
        distance = np.linalg.norm(self.grid_to_world(cells) - xy, axis=1)
        index = int(np.argmin(distance))
        if distance[index] > radius_m:
            raise PlanningError("endpoint is too far from the navigation grid")
        return cells[index]

    def clearance(self, xy: np.ndarray) -> np.ndarray:
        cells = self.world_to_grid(xy)
        values = np.full(len(cells), -np.inf, dtype=np.float64)
        valid = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < self.free.shape[0])
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < self.free.shape[1])
        )
        positions = np.flatnonzero(valid)
        inside = cells[valid]
        navigable = self.free[inside[:, 0], inside[:, 1]]
        positions, inside = positions[navigable], inside[navigable]
        values[positions] = self.clearance_m[inside[:, 0], inside[:, 1]]
        return values

    @cached_property
    def components(self):
        """Four-connected components of the free cells (the connectivity of grid_distances, whose diagonal steps
        need both side cells free); 0 outside free space."""
        from scipy.ndimage import label
        return label(self.free)[0]

    @cached_property
    def signed_clearance_m(self):
        # Half-cell boundary vertices/midpoints give exact distances at cell centres.
        padded=np.pad(self.free,1)
        vertical=np.argwhere(padded[1:,1:-1]!=padded[:-1,1:-1])
        horizontal=np.argwhere(padded[1:-1,1:]!=padded[1:-1,:-1])
        boundary=np.zeros(tuple(2*np.asarray(self.free.shape)+1),dtype=bool)
        for offset in (0,1,2):
            boundary[2*vertical[:,0],2*vertical[:,1]+offset]=True
            boundary[2*horizontal[:,0]+offset,2*horizontal[:,1]]=True
        distance=distance_transform_edt(~boundary,sampling=self.cell_size_m/2)[1::2,1::2]
        return np.where(self.free,distance,-distance)

    @cached_property
    def clearance_spline(self):
        # Smooth clearance for the travel cost only; the fit constraint uses boundary_distance.
        signed=self.signed_clearance_m
        axes = [
            self.origin_xy[k] + (np.arange(n) + 0.5) * self.cell_size_m
            for k, n in enumerate(self.free.shape)
        ]
        return RectBivariateSpline(*axes, signed)

    def clearance_gradient(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous surrogate value and its exact spline derivative."""
        tx, ty, c = self.clearance_spline.tck
        xy = np.asarray(xy, dtype=np.float64)
        result = _bicubic_with_gradient(tx, ty, c, np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1]))
        return result[:, 0], result[:, 1:]

    @cached_property
    def _interface(self):
        """Free/obstacle cell edges as world segments, bucketed by the grid cell holding their midpoint."""
        padded = np.pad(self.free, 1)
        h = self.cell_size_m
        rows = np.argwhere(padded[1:, 1:-1] != padded[:-1, 1:-1])
        cols = np.argwhere(padded[1:-1, 1:] != padded[1:-1, :-1])
        start = np.concatenate((rows, cols)) * h + self.origin_xy
        end = start + np.concatenate((np.tile([0., h], (len(rows), 1)), np.tile([h, 0.], (len(cols), 1))))
        nbx, nby = self.free.shape[0] + 1, self.free.shape[1] + 1
        bucket = np.floor(((start + end) / 2 - self.origin_xy) / h).astype(np.int64)
        key = bucket[:, 0] * nby + bucket[:, 1]
        order = np.argsort(key, kind='stable')
        bucket_start = np.searchsorted(key[order], np.arange(nbx * nby + 1))
        return (np.ascontiguousarray(start[order]), np.ascontiguousarray(end[order]), bucket_start.astype(np.int64), nbx, nby)

    @cached_property
    def _edge_index(self):
        start, end, _, _, _ = self._interface
        return cKDTree((start + end) / 2)

    @cached_property
    def _cell_lists(self):
        """_cell_boundary's per-cell candidate lists: [slot per bucket, list offsets, edge indices, (lists, used)]."""
        _, _, _, nbx, nby = self._interface
        return [np.full(nbx * nby, -1, np.int64), np.zeros(1025, np.int64), np.zeros(1 << 16, np.int64), np.zeros(2, np.int64)]

    def boundary_distance(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact signed distance to the closed free-cell boundary and its gradient.

        This is the quantity the closed-cell certificate tests; negative inside an
        obstacle cell or outside the grid. The values and gradients are those of
        _boundary_distance_reference (the ring search), from cached per-cell
        candidate edges (_cell_boundary): SLSQP's constraint rows revisit the same
        cells hundreds of times.
        """
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            return self._boundary_distance_reference(xy)
        start, end, bucket_start, nbx, nby = self._interface
        lists = self._cell_lists
        value = np.empty(len(xy)); gradient = np.empty((len(xy), 2))
        # The kernel extends the shared lists (it releases the GIL): one call at a time per process.
        with _CELL_LISTS_LOCK:
            while True:
                status = _cell_boundary(xy, np.asarray(self.origin_xy, dtype=np.float64), float(self.cell_size_m), start, end,
                                        bucket_start, nbx, nby, self.free, lists[0], lists[1], lists[2], lists[3], value, gradient)
                if status == 0:
                    return value, gradient
                if status == 2:
                    break
                lists[1] = np.concatenate((lists[1], np.zeros(len(lists[1]), np.int64)))
                lists[2] = np.concatenate((lists[2], np.zeros(len(lists[2]), np.int64)))
        return self._boundary_distance_reference(xy)

    def _boundary_distance_reference(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """boundary_distance by KD candidates for far points and the ring search for the others."""
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        start, end, bucket_start, nbx, nby = self._interface
        origin = np.asarray(self.origin_xy, dtype=np.float64)
        h = float(self.cell_size_m); count = min(BOUNDARY_CANDIDATES, len(start))
        cells = self.world_to_grid(xy)
        inside = (cells >= 0).all(1) & (cells < self.free.shape).all(1)
        far = np.full(len(xy), count > 1)
        if count > 1:
            far[inside] = np.abs(self.signed_clearance_m[cells[inside, 0], cells[inside, 1]]) > BOUNDARY_RING_M
        # Far from the boundary the candidate search below replaces the ring search with the same result.
        value = np.empty(len(xy)); gradient = np.empty((len(xy), 2)); ring = ~far
        if far.sum() >= BOUNDARY_KD_MIN_POINTS:
            points = np.ascontiguousarray(xy[far])
            farthest, candidates = self._edge_index.query(points, k=count)
            v, g, exact = _nearest_candidate_boundary(points, origin, h, start, end, np.ascontiguousarray(candidates), farthest[:, -1], nby)
            value[far], gradient[far] = v, g
            ring[np.flatnonzero(far)[~exact]] = True
        else:
            ring[:] = True
        if ring.any():
            value[ring], gradient[ring] = _nearest_boundary(np.ascontiguousarray(xy[ring]), origin, h, start, end, bucket_start, nbx, nby)
        free = np.zeros(len(xy), bool)
        free[inside] = self.free[cells[inside, 0], cells[inside, 1]]
        sign = np.where(free, 1., -1.)
        return sign * value, sign[:, None] * gradient

    def path_cells(self, path):
        path=np.ascontiguousarray(path,dtype=np.float64)
        return _path_cells((path-self.origin_xy)/self.cell_size_m)

    def safe(self, path: np.ndarray, minimum_m: float | None = None) -> bool:
        minimum_m = self.minimum_clearance_m if minimum_m is None else minimum_m
        path = np.ascontiguousarray(path, dtype=np.float64)
        # Match NumPy's scalar promotion, including float32 threshold rounding.
        dtype = np.result_type(self.clearance_m, 1e-9)
        minimum = np.result_type(dtype, minimum_m).type(minimum_m)
        return _safe_path((path-self.origin_xy)/self.cell_size_m, self.free, self.clearance_m, minimum, dtype.type(1e-9))


def polyline(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape [N,2], N >= 2")
    keep = np.concatenate(
        [[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    )
    path = path[keep]
    if len(path) < 2:
        raise ValueError("path has zero length")
    return path, np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )


def points_at_arc(path: np.ndarray, arcs_m: np.ndarray) -> np.ndarray:
    path, cumulative = polyline(path)
    query = np.clip(np.asarray(arcs_m), 0.0, cumulative[-1])
    return np.column_stack(
        [np.interp(query, cumulative, path[:, axis]) for axis in range(2)]
    )


def resample(path: np.ndarray, spacing_m: float) -> np.ndarray:
    path, cumulative = polyline(path)
    query = np.linspace(
        0.0, cumulative[-1], max(2, math.ceil(cumulative[-1] / spacing_m) + 1)
    )
    return points_at_arc(path, query)


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def curvature(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 3:
        return np.zeros(1)
    first, second, chord = (
        path[1:-1] - path[:-2],
        path[2:] - path[1:-1],
        path[2:] - path[:-2],
    )
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
        * np.linalg.norm(chord, axis=1)
    )
    cross = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return np.divide(
        2 * cross, denominator, out=np.zeros_like(cross), where=denominator > 1e-9
    )


def turn_rate_per_m(path: np.ndarray) -> np.ndarray:
    """Heading change per metre over one body diameter at every vertex.

    A vertex or near-cusp cannot demand infinite yaw rate; this is the quantity
    the MPC speed law divides the angular-speed limit by.
    """
    path = np.asarray(path, dtype=np.float64)
    delta = np.diff(path, axis=0)
    rate = np.zeros(len(path))
    if len(delta) < 2:
        return rate
    arc = np.r_[0., np.linalg.norm(delta, axis=1).cumsum()]
    turn = np.arctan2(delta[:-1, 0] * delta[1:, 1] - delta[:-1, 1] * delta[1:, 0], (delta[:-1] * delta[1:]).sum(-1))
    total = np.r_[0., np.cumsum(turn)]
    window = 2 * ROBOT_FOOTPRINT_RADIUS_M
    low = np.searchsorted(arc[1:-1], arc - window / 2, side="left")
    high = np.searchsorted(arc[1:-1], arc + window / 2, side="right")
    return np.abs(total[high] - total[low]) / window


@dataclass(frozen=True)
class Plan:
    curve: BSpline
    path_xy: np.ndarray
    metrics: dict[str, float]


def _astar(grid: Grid, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    allowed = grid.free & (grid.clearance_m + 1e-9 >= grid.minimum_clearance_m)
    cells = _astar_cells(allowed, grid.signed_clearance_m, float(grid.cell_size_m), int(start[0]), int(start[1]), int(goal[0]), int(goal[1]))
    if not len(cells):
        raise PlanningError("no safe grid route")
    return cells


@njit(cache=True, nogil=True)
def _polyline_arc(path):
    """polyline(): drop vertices within 1e-9 of their predecessor; cumulative arc. Same elementwise operations
    as np.diff/np.linalg.norm/np.cumsum (one add of two squares, correctly rounded sqrt, sequential cumsum)."""
    keep = np.empty(len(path), dtype=np.bool_); keep[0] = True
    for i in range(1, len(path)):
        dx = path[i, 0] - path[i - 1, 0]; dy = path[i, 1] - path[i - 1, 1]
        keep[i] = math.sqrt(dx * dx + dy * dy) > 1e-9
    kept = path[keep]
    arc = np.empty(len(kept)); arc[0] = 0.
    total = 0.
    for i in range(1, len(kept)):
        dx = kept[i, 0] - kept[i - 1, 0]; dy = kept[i, 1] - kept[i - 1, 1]
        total = total + math.sqrt(dx * dx + dy * dy); arc[i] = total
    return kept, arc


@njit(cache=True, nogil=True)
def _resample(path, spacing):
    """resample(path, spacing) with the NumPy 1.26 polyline/linspace/clip/interp arithmetic, bitwise. Returns an
    empty array where polyline() would raise (fewer than two distinct vertices); the caller then uses resample."""
    kept, arc = _polyline_arc(path)
    if len(kept) < 2:
        return np.empty((0, 2))
    length = arc[-1]
    count = max(2, math.ceil(length / spacing) + 1)
    step = length / (count - 1)
    kept, arc = _polyline_arc(kept)
    if len(kept) < 2:
        return np.empty((0, 2))
    last = len(arc) - 1
    out = np.empty((count, 2))
    for i in range(count):
        # linspace: arange * step + start, endpoint set to stop exactly.
        query = length if i == count - 1 else i * step + 0.
        query = min(max(query, 0.), arc[last])
        # np.interp: largest j with arc[j] <= query; exact knots and the end take the vertex.
        j = np.searchsorted(arc, query, side='right') - 1
        for axis in range(2):
            if j == last or arc[j] == query:
                out[i, axis] = kept[j, axis]
            else:
                slope = (kept[j + 1, axis] - kept[j, axis]) / (arc[j + 1] - arc[j])
                value = slope * (query - arc[j]) + kept[j, axis]
                if math.isnan(value):
                    value = slope * (query - arc[j + 1]) + kept[j + 1, axis]
                    if math.isnan(value) and kept[j, axis] == kept[j + 1, axis]:
                        value = kept[j, axis]
                out[i, axis] = value
    return out


def _path_cost(grid: Grid, path: np.ndarray) -> float:
    path = np.ascontiguousarray(path, dtype=np.float64)
    regular = path.ndim == 2 and path.shape[1] == 2 and len(path) >= 2 and np.isfinite(path).all()
    sampled = _resample(path, SAFETY_STEP_M) if regular else None
    if sampled is None or not len(sampled):
        sampled = resample(path, SAFETY_STEP_M)
    segment_length = np.linalg.norm(np.diff(sampled, axis=0), axis=1)
    tx, ty, c = grid.clearance_spline.tck
    clearance = _bicubic_with_gradient(tx, ty, c, np.ascontiguousarray(sampled[:, 0]), np.ascontiguousarray(sampled[:, 1]))[:, 0]
    if not np.isfinite(clearance).all():
        return math.inf
    segment_clearance = 0.5 * (clearance[:-1] + clearance[1:])
    return float(np.sum(segment_length * clearance_density(segment_clearance)))


@njit(cache=True, nogil=True, error_model='numpy')
def _first_safe_shortcut(path, index, following, free, clearance, minimum, margin, origin, h):
    """The largest f in (index + 1, following] whose direct segment path[[index, f]] passes Grid.safe (the same
    _safe_path call on the same coordinates), else index + 1."""
    direct = np.empty((2, 2))
    direct[0, 0] = (path[index, 0] - origin[0]) / h; direct[0, 1] = (path[index, 1] - origin[1]) / h
    while following > index + 1:
        direct[1, 0] = (path[following, 0] - origin[0]) / h; direct[1, 1] = (path[following, 1] - origin[1]) / h
        if _safe_path(direct, free, clearance, minimum, margin):
            return following
        following -= 1
    return following


@njit(cache=True, nogil=True, error_model='numpy')
def _path_cost_terms(path, spacing, tx, ty, c, radius):
    """_path_cost up to its np.exp: the resampled path's segment lengths and the exponents 1 - clearance / radius
    of its segment midpoints, and a status: 0, 1 for an infinite cost, 2 where _path_cost would take resample()
    (a path that is irregular or has fewer than two distinct vertices)."""
    empty = np.empty(0)
    if len(path) < 2: return 2, empty, empty
    for i in range(len(path)):
        if not (np.isfinite(path[i, 0]) and np.isfinite(path[i, 1])): return 2, empty, empty
    sampled = _resample(path, spacing)
    if not len(sampled): return 2, empty, empty
    value = _bicubic_with_gradient(tx, ty, c, np.ascontiguousarray(sampled[:, 0]), np.ascontiguousarray(sampled[:, 1]))
    for i in range(len(sampled)):
        if not np.isfinite(value[i, 0]): return 1, empty, empty
    lengths = np.empty(len(sampled) - 1); arguments = np.empty(len(sampled) - 1)
    for i in range(len(sampled) - 1):
        dx = sampled[i + 1, 0] - sampled[i, 0]; dy = sampled[i + 1, 1] - sampled[i, 1]
        lengths[i] = math.sqrt(dx * dx + dy * dy)
        arguments[i] = 1.0 - 0.5 * (value[i, 0] + value[i + 1, 0]) / radius
    return 0, lengths, arguments


@njit(cache=True, nogil=True, error_model='numpy')
def _shortcut_terms(path, index, following, spacing, tx, ty, c, radius):
    """_path_cost_terms of path[[index, following]] and path[index:following + 1], the exponents concatenated."""
    direct = np.empty((2, 2))
    direct[0] = path[index]; direct[1] = path[following]
    first, first_lengths, first_arguments = _path_cost_terms(direct, spacing, tx, ty, c, radius)
    second, second_lengths, second_arguments = _path_cost_terms(np.ascontiguousarray(path[index:following + 1]), spacing,
                                                                tx, ty, c, radius)
    return first, first_lengths, second, second_lengths, np.concatenate((first_arguments, second_arguments))


@njit(cache=True, nogil=True, error_model='numpy')
def _path_cost_sum(lengths, exponential, at):
    """np.sum(segment_length * (1.0 + np.exp(..))) of one _path_cost_terms block (its exponentials from `at`)."""
    rows = np.empty(len(lengths))
    for i in range(len(lengths)): rows[i] = lengths[i] * (1.0 + exponential[at + i])
    return 0. + _pairwise_sum(rows, 0, len(rows))


def _shortcut_costs(grid, path, index, following):
    """(_path_cost(grid, path[[index, following]]), _path_cost(grid, path[index:following + 1])) from the fused
    kernels: one np.exp over both paths' exponents (numpy's exp is the same per element in any batch)."""
    tx, ty, c = grid.clearance_spline.tck
    first, first_lengths, second, second_lengths, arguments = _shortcut_terms(
        path, index, following, SAFETY_STEP_M, tx, ty, c, ROBOT_FOOTPRINT_RADIUS_M)
    if first == 2 or second == 2:
        # The reference raises or resamples there; it is called in its own order.
        return _path_cost(grid, path[[index, following]]), _path_cost(grid, path[index:following + 1])
    exponential = np.exp(arguments)
    return (math.inf if first == 1 else float(_path_cost_sum(first_lengths, exponential, 0)),
            math.inf if second == 1 else float(_path_cost_sum(second_lengths, exponential, len(first_lengths))))


def _simplify_route(grid: Grid, path: np.ndarray, reach_m: float | None = None) -> np.ndarray:
    """Greedy shortcutting of a grid route. With `reach_m`, stop once the output's arc length exceeds it: the
    remaining vertices never change `_prefix(output, length)` for any length <= reach_m (same leading
    vertices, same arc cumsum), so callers that only keep such a prefix get it unchanged."""
    path = np.asarray(path)
    if not (_fused_objective_exact() and path.dtype == np.float64 and path.ndim == 2 and path.shape[1] == 2
            and np.isfinite(path).all()):
        return _simplify_route_reference(grid, path, reach_m)
    # _simplify_route_reference with the safe tests and costs from the fused kernels: the same candidates are
    # tested in the same order, and the same costs compared.
    path = np.ascontiguousarray(path)
    dtype = np.result_type(grid.clearance_m, 1e-9)
    minimum, margin = np.result_type(dtype, grid.minimum_clearance_m).type(grid.minimum_clearance_m), dtype.type(1e-9)
    origin, h = np.asarray(grid.origin_xy, dtype=np.float64), grid.cell_size_m
    output, index = [path[0]], 0
    while index < len(path) - 1:
        if reach_m is not None and len(output) > 1 and np.cumsum(
                np.linalg.norm(np.diff(np.asarray(output), axis=0), axis=1))[-1] > reach_m:
            break
        following = len(path) - 1
        while following > index + 1:
            following = _first_safe_shortcut(path, index, following, grid.free, grid.clearance_m, minimum, margin, origin, h)
            if following <= index + 1:
                break
            direct_cost, sub_cost = _shortcut_costs(grid, path, index, following)
            if direct_cost <= sub_cost + 1e-9:
                break
            following -= 1
        output.append(path[following])
        index = following
    return np.asarray(output)


def _simplify_route_reference(grid: Grid, path: np.ndarray, reach_m: float | None = None) -> np.ndarray:
    output, index = [path[0]], 0
    while index < len(path) - 1:
        if reach_m is not None and len(output) > 1 and np.cumsum(
                np.linalg.norm(np.diff(np.asarray(output), axis=0), axis=1))[-1] > reach_m:
            break
        following = len(path) - 1
        while following > index + 1:
            direct = path[[index, following]]
            if grid.safe(direct) and _path_cost(grid, direct) <= _path_cost(
                grid, path[index : following + 1]
            ) + 1e-9:
                break
            following -= 1
        output.append(path[following])
        index = following
    return np.asarray(output)


def clearance_density(clearance):
    """Dimensionless travel cost; extra clearance is scaled by body radius."""
    return 1.0 + np.exp(1.0 - clearance / ROBOT_FOOTPRINT_RADIUS_M)


def _travel_objective(controls, basis, first, weights, grid):
    """Integral rho(clearance) ds and its control-point gradient."""
    points, tangent = basis @ controls, first @ controls
    speed = np.linalg.norm(tangent, axis=1)
    if np.any(speed <= np.finfo(float).eps):
        raise PlanningError("curve optimization encountered a stationary tangent")
    distance, distance_gradient = grid.clearance_gradient(points)
    density = clearance_density(distance)
    dp = -(density - 1)[:, None] / ROBOT_FOOTPRINT_RADIUS_M * distance_gradient * speed[:, None]
    dv = density[:, None] * tangent / speed[:, None]
    return float(weights @ (density * speed)), (
        basis.T @ (weights[:, None] * dp) + first.T @ (weights[:, None] * dv)
    )


def _bending_objective(controls, first, second, weights):
    """Integral r_body^2 * curvature^2 ds, independent of parameterization."""
    tangent, acceleration = first @ controls, second @ controls
    return _bending_terms(tangent, acceleration, first, second, weights)


def _bending_terms(tangent, acceleration, first, second, weights):
    speed = np.linalg.norm(tangent, axis=1)
    cross = tangent[:, 0] * acceleration[:, 1] - tangent[:, 1] * acceleration[:, 0]
    radius2 = ROBOT_FOOTPRINT_RADIUS_M**2
    energy = radius2 * cross**2 / speed**5
    dv = ((2 * radius2 * cross / speed**5)[:, None]
          * np.column_stack((acceleration[:, 1], -acceleration[:, 0]))
          - (5 * energy / speed**2)[:, None] * tangent)
    da = (2 * radius2 * cross / speed**5)[:, None] * np.column_stack(
        (-tangent[:, 1], tangent[:, 0]))
    return float(weights @ energy), (
        first.T @ (weights[:, None] * dv) + second.T @ (weights[:, None] * da)
    )


_GAUSS_LOW = np.polynomial.legendre.leggauss(8)
_GAUSS_HIGH = np.polynomial.legendre.leggauss(16)
_GAUSS_TRAVEL = np.polynomial.legendre.leggauss(4)


def _bending_integral(velocity, acceleration, first, second, breaks, epsabs=1e-7, epsrel=1e-6, limit=10000):
    """Adaptive composite Gauss-Legendre of _bending_terms over [0, 1].

    Every open interval is integrated with 8 and 16 nodes at once; an interval
    whose two estimates differ by more than its length's share of the tolerance
    is bisected. Accepted intervals contribute their 16-node value and gradient.
    More than `limit` intervals (a trial curve close to a cusp) does not converge.
    """
    def nodes(lo, hi, rule):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        return (mid[:, None] + half[:, None] * rule[0]).ravel(), (half[:, None] * rule[1]).ravel()

    def energy(u):
        tangent, curve = velocity(u), acceleration(u)
        cross = tangent[:, 0] * curve[:, 1] - tangent[:, 1] * curve[:, 0]
        return ROBOT_FOOTPRINT_RADIUS_M**2 * cross**2 / np.einsum("ij,ij->i", tangent, tangent) ** 2.5

    lo, hi, accepted, total, count = breaks[:-1], breaks[1:], [], 0., 0
    while len(lo):
        count += len(lo)
        if count > limit:
            raise PlanningError("curve energy quadrature did not converge")
        (u_low, w_low), (u_high, w_high) = nodes(lo, hi, _GAUSS_LOW), nodes(lo, hi, _GAUSS_HIGH)
        low = (w_low * energy(u_low)).reshape(len(lo), -1).sum(1)
        high = (w_high * energy(u_high)).reshape(len(lo), -1).sum(1)
        if not np.isfinite(high).all():
            raise PlanningError("curve energy quadrature did not converge")
        tolerance = (epsabs + epsrel * (total + high.sum())) * (hi - lo)
        done = np.abs(high - low) <= tolerance
        accepted.append((lo[done], hi[done]))
        total += high[done].sum()
        mid = (lo[~done] + hi[~done]) / 2
        lo, hi = np.concatenate((lo[~done], mid)), np.concatenate((mid, hi[~done]))
    u, w = nodes(np.concatenate([a for a, _ in accepted]), np.concatenate([b for _, b in accepted]), _GAUSS_HIGH)
    return _bending_terms(velocity(u), acceleration(u), first(u), second(u), w)


def _spline_derivative(t, c, k, nu):
    """BSpline(t, c, k).derivative(nu) without its object and argument-checking overhead: the same padding and
    splder arithmetic as scipy, so the coefficients are bitwise identical."""
    if len(t) > len(c):
        c = np.concatenate((c, np.zeros((len(t) - len(c),) + c.shape[1:])))
    for _ in range(nu):
        dt = t[k + 1:-1] - t[1:-k - 1]
        c = (c[1:-1 - k] - c[:-2 - k]) * k / dt[:, None]
        c = np.concatenate((c, np.zeros((k,) + c.shape[1:])))
        t = t[1:-1]
        k -= 1
    return BSpline.construct_fast(t, c, k)


@njit(cache=True, nogil=True)
def _equal_vectors(a, b):
    """np.array_equal for two 1-D float64 arrays."""
    if len(a) != len(b): return False
    for i in range(len(a)):
        if a[i] != b[i]: return False
    return True


def _same_values(a, b):
    if isinstance(a, np.ndarray) and a.ndim == 1 and b.ndim == 1 and a.dtype == np.float64 and b.dtype == np.float64:
        return _equal_vectors(a, b)
    return np.array_equal(a, b)


def _cached_value_and_jacobian(function):
    previous = None
    def evaluate(values):
        nonlocal previous
        if previous is None or not _same_values(values, previous[0]):
            previous = values.copy(), function(values)
        return previous[1]
    return evaluate


# _smooth's objective and constraints fused into numba kernels (used where _fused_objective_exact holds). Every
# kernel repeats the reference expression's IEEE operations in the reference order (numba does not contract
# multiply-adds without fastmath). It also repeats the compiled routines the reference calls: FITPACK's
# splev/splder/fpbspl (PPoly.from_spline), scipy's evaluate_spline/_deBoor_D (BSpline.__call__), and four numpy
# reductions: two-element row sums (norm, einsum), the 3-term self-convolutions of np.convolve, and pairwise
# summation (np.sum). What it cannot repeat stays a numpy call on the same arrays: matrix products (BLAS, which
# fuses multiply-adds here), np.exp and np.power (numpy's SIMD loops, not libm), eigvals and np.roots (LAPACK).
# The kernels use numpy's error model: a division by zero gives inf or nan as in numpy, never ZeroDivisionError.


@njit(cache=True, nogil=True, error_model='numpy')
def _pairwise_leaf(a, start, n):
    """DOUBLE_pairwise_sum for n <= 128: sequential below 8, else eight interleaved partial sums."""
    if n < 8:
        res = 0.
        for i in range(n): res = res + a[start + i]
        return res
    r0 = a[start]; r1 = a[start + 1]; r2 = a[start + 2]; r3 = a[start + 3]
    r4 = a[start + 4]; r5 = a[start + 5]; r6 = a[start + 6]; r7 = a[start + 7]
    i = 8
    stop = n - (n % 8)
    while i < stop:
        r0 = r0 + a[start + i]; r1 = r1 + a[start + i + 1]; r2 = r2 + a[start + i + 2]; r3 = r3 + a[start + i + 3]
        r4 = r4 + a[start + i + 4]; r5 = r5 + a[start + i + 5]; r6 = r6 + a[start + i + 6]; r7 = r7 + a[start + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res = res + a[start + i]; i += 1
    return res


@njit(cache=True, nogil=True, error_model='numpy')
def _pairwise_sum(a, start, n):
    """numpy's pairwise summation (DOUBLE_pairwise_sum) of a[start:start+n]; a.sum() is 0. + this. Blocks of more
    than 128 split at n/2 rounded down to a multiple of 8; the recursion runs on an explicit stack (a cached
    recursive numba function crashes on load)."""
    if n <= 128:
        return _pairwise_leaf(a, start, n)
    frame_start = np.empty(64, np.int64); frame_n = np.empty(64, np.int64); frame_state = np.empty(64, np.int64)
    values = np.empty(64)
    top = 1; frame_start[0] = start; frame_n[0] = n; frame_state[0] = 0; depth = 0
    while top > 0:
        s = frame_start[top - 1]; m = frame_n[top - 1]
        if m <= 128:
            values[depth] = _pairwise_leaf(a, s, m); depth += 1; top -= 1
            continue
        half = m // 2; half -= half % 8
        state = frame_state[top - 1]
        if state == 0:
            frame_state[top - 1] = 1
            frame_start[top] = s; frame_n[top] = half; frame_state[top] = 0; top += 1
        elif state == 1:
            frame_state[top - 1] = 2
            frame_start[top] = s + half; frame_n[top] = m - half; frame_state[top] = 0; top += 1
        else:
            values[depth - 2] = values[depth - 2] + values[depth - 1]; depth -= 1; top -= 1
    return values[0]


@njit(cache=True, nogil=True, error_model='numpy')
def _np_sum(a):
    """a.sum() for a 1-D float64 array."""
    return 0. + _pairwise_sum(a, 0, len(a))


@njit(cache=True, nogil=True, error_model='numpy', inline='always')
def _fpbspl(t, k, x, l, h, hh):
    """FITPACK fpbspl (0-based l): the k+1 B-splines of degree k nonzero at x, t[l] <= x < t[l+1]."""
    h[0] = 1.
    for j in range(1, k + 1):
        for i in range(j):
            hh[i] = h[i]
        h[0] = 0.
        for i in range(1, j + 1):
            li = l + i; lj = li - j
            if t[li] == t[lj]:
                h[i] = 0.
                continue
            f = hh[i - 1] / (t[li] - t[lj])
            h[i - 1] = h[i - 1] + f * (t[li] - x)
            h[i] = f * (x - t[lj])


@njit(cache=True, nogil=True, error_model='numpy')
def _splder(t, c, k, nu, x, y):
    """FITPACK splder (splev for nu = 0) with e = 0, into y; Fortran's 1-based indices kept as offsets."""
    n = len(t); k1 = k + 1; k3 = k1 + 1; nk1 = n - k1
    wrk = np.empty(n); h = np.empty(20); hh = np.empty(19)
    for i in range(nk1): wrk[i] = c[i]
    l = 1; kk = k; nk2 = nk1
    for _ in range(nu):
        ak = float(kk)
        nk2 = nk2 - 1
        l1 = l
        for i in range(1, nk2 + 1):
            l1 = l1 + 1
            l2 = l1 + kk
            fac = t[l2 - 1] - t[l1 - 1]
            if fac <= 0.: continue
            wrk[i - 1] = ak * (wrk[i] - wrk[i - 1]) / fac
        l = l + 1
        kk = kk - 1
    if kk == 0:
        j = 1
        for i in range(len(x)):
            arg = x[i]
            while not (arg >= t[l - 1] or l + 1 == k3):
                l = l - 1; j = j - 1
            while not (arg < t[l] or l == nk1):
                l = l + 1; j = j + 1
            y[i] = wrk[j - 1]
        return
    l = k1; l1 = l + 1; k2 = k1 - nu
    for i in range(len(x)):
        arg = x[i]
        while not (arg >= t[l - 1] or l1 == k3):
            l1 = l; l = l - 1
        while not (arg < t[l1 - 1] or l == nk1):
            l = l1; l1 = l + 1
        _fpbspl(t, kk, arg, l - 1, h, hh)
        sp = 0.
        ll = l - k1
        for j in range(k2):
            ll = ll + 1
            sp = sp + wrk[ll - 1] * h[j]
        y[i] = sp


@njit(cache=True, nogil=True, error_model='numpy')
def _spline_values(t, c, k, x):
    """scipy's evaluate_spline (nu = 0, extrapolate): BSpline.construct_fast(t, c, k)(x) for a 2-D c."""
    m = c.shape[1]; out = np.empty((len(x), m)); h = np.empty(k + 1); hh = np.empty(k + 1)
    n = len(t) - k - 1
    interval = k
    for ip in range(len(x)):
        xval = x[ip]
        if xval != xval:
            for jp in range(m): out[ip, jp] = np.nan
            continue
        l = interval if (k < interval and interval < n) else k
        while xval < t[l] and l != k: l -= 1
        l += 1
        while xval >= t[l] and l != n: l += 1
        interval = l - 1
        h[0] = 1.
        for j in range(1, k + 1):
            for q in range(j): hh[q] = h[q]
            h[0] = 0.
            for q in range(1, j + 1):
                ind = interval + q; xb = t[ind]; xa = t[ind - j]
                if xb == xa:
                    h[q] = 0.
                    continue
                w = hh[q - 1] / (xb - xa)
                h[q - 1] = h[q - 1] + w * (xb - xval)
                h[q] = w * (xval - xa)
        for jp in range(m):
            s = 0.
            for a in range(k + 1):
                s = s + c[interval + a - k, jp] * h[a]
            out[ip, jp] = s
    return out


@njit(cache=True, nogil=True, error_model='numpy')
def _assemble(values, e, direction, first_control, last_control):
    """_smooth's assemble(values), e = math.exp(values[0])."""
    n = (len(values) - 1) // 2 + 3
    assembled = np.empty((n, 2))
    assembled[0, 0] = first_control[0]; assembled[0, 1] = first_control[1]
    assembled[1, 0] = first_control[0] + e * direction[0]; assembled[1, 1] = first_control[1] + e * direction[1]
    for i in range(n - 3):
        assembled[2 + i, 0] = values[1 + 2 * i]; assembled[2 + i, 1] = values[2 + 2 * i]
    assembled[n - 1, 0] = last_control[0]; assembled[n - 1, 1] = last_control[1]
    return assembled


@njit(cache=True, nogil=True, error_model='numpy')
def _objective_front(values, e, direction, first_control, last_control, knots, t1, x1, spans):
    """The objective up to the tangent-speed extrema: assemble(values); the coefficients of
    _spline_derivative(knots, assembled, 3, 1) (tangent) and of (knots, relative, 3, 1|2) (velocity,
    acceleration); per span the speed derivative (sum over both coordinates of np.convolve(c, c)[:-1] * [4,3,2,1],
    c a column of PPoly.from_spline of the tangent), the cubic-span mask and the cubic spans' companion matrices."""
    assembled = _assemble(values, e, direction, first_control, last_control)
    n = len(assembled)
    tangent = np.zeros((n + 2, 2)); velocity = np.zeros((n + 2, 2)); acceleration = np.zeros((n, 2))
    relative = np.empty((n, 2))
    for i in range(n):
        for k in range(2):
            relative[i, k] = assembled[i, k] - assembled[0, k]
    for i in range(n - 1):
        dt = knots[4 + i] - knots[1 + i]
        for k in range(2):
            tangent[i, k] = (assembled[i + 1, k] - assembled[i, k]) * 3. / dt
            velocity[i, k] = (relative[i + 1, k] - relative[i, k]) * 3. / dt
    for i in range(n - 2):
        dt = knots[4 + i] - knots[2 + i]
        for k in range(2):
            acceleration[i, k] = (velocity[i + 1, k] - velocity[i, k]) * 2. / dt
    # PPoly.from_spline((t1, tangent[:, k], 2)).c: rows splder(.., 2) / gamma(3), splder(.., 1) / gamma(2), splev / gamma(1).
    pieces = np.empty((2, 3, len(x1))); column = np.empty(n + 2); y = np.empty(len(x1))
    for k in range(2):
        for i in range(n + 2): column[i] = tangent[i, k]
        for nu in range(3):
            _splder(t1, column, 2, 2 - nu, x1, y)
            for i in range(len(x1)): pieces[k, nu, i] = y[i] / 2. if nu == 0 else y[i] / 1.
    derivatives = np.empty((len(spans), 4)); cubic = np.empty(len(spans), np.bool_)
    count = 0
    for row in range(len(spans)):
        i = spans[row]
        conv = np.zeros(4)
        for piece in range(2):
            a0 = pieces[piece, 0, i]; a1 = pieces[piece, 1, i]; a2 = pieces[piece, 2, i]
            # np.convolve(a, a)[:4]: dot products of 1, 2, 3 and 2 terms (the 2-term ones are 2 * a_i a_j however
            # the dot rounds), the 3-term one by numpy's small_correlate.
            p01 = a0 * a1; p12 = a1 * a2
            c0 = a0 * a0; c1 = p01 + p01; c2 = ((0. + a0 * a2) + a1 * a1) + a2 * a0; c3 = p12 + p12
            if piece == 0:
                conv[0] = 0. + c0; conv[1] = 0. + c1; conv[2] = 0. + c2; conv[3] = 0. + c3
            else:
                conv[0] = conv[0] + c0; conv[1] = conv[1] + c1; conv[2] = conv[2] + c2; conv[3] = conv[3] + c3
        derivatives[row, 0] = conv[0] * 4.; derivatives[row, 1] = conv[1] * 3.
        derivatives[row, 2] = conv[2] * 2.; derivatives[row, 3] = conv[3] * 1.
        cubic[row] = derivatives[row, 0] != 0 and derivatives[row, 3] != 0
        if cubic[row]: count += 1
    companions = np.zeros((count, 3, 3)); c = 0
    for row in range(len(spans)):
        if cubic[row]:
            companions[c, 1, 0] = 1.; companions[c, 2, 1] = 1.
            for j in range(3):
                companions[c, 0, j] = -derivatives[row, 1 + j] / derivatives[row, 0]
            c += 1
    return assembled, tangent, velocity, acceleration, derivatives, cubic, companions


@njit(cache=True, nogil=True, error_model='numpy')
def _objective_breaks(real, imag, cubic, spans, t1, knot_values, extra, tangent, eps):
    """np.asarray(sorted(set(breaks))) - the unique knots, lo + root for every real root strictly inside its
    cubic span (eigenvalues of its companion) and `extra` (the other spans' np.roots) - and whether the tangent is
    regular there: not np.any(norm(tangent(breaks), axis=1) <= eps * norm(tangent.c, axis=1).max())."""
    out = np.empty(len(knot_values) + 3 * len(spans) + len(extra)); n = 0
    for i in range(len(knot_values)): out[n] = knot_values[i]; n += 1
    c = 0
    for row in range(len(spans)):
        if not cubic[row]: continue
        lo = t1[spans[row]]; width = t1[spans[row] + 1] - lo
        for r in range(3):
            if imag[c, r] == 0 and 0 < real[c, r] and real[c, r] < width:
                out[n] = lo + real[c, r]; n += 1
        c += 1
    for i in range(len(extra)): out[n] = extra[i]; n += 1
    ordered = np.sort(out[:n])
    unique = np.empty(n); m = 0
    for i in range(n):
        if i == 0 or ordered[i] != ordered[i - 1]:
            unique[m] = ordered[i]; m += 1
    breaks = unique[:m]
    scale = -np.inf; nan = False
    for i in range(len(tangent)):
        s = math.sqrt(tangent[i, 0] * tangent[i, 0] + tangent[i, 1] * tangent[i, 1])
        if s != s: nan = True
        elif s > scale: scale = s
    cusp = False
    if not nan:
        bound = eps * scale
        at = _spline_values(t1, tangent, 2, breaks)
        for i in range(len(at)):
            if math.sqrt(at[i, 0] * at[i, 0] + at[i, 1] * at[i, 1]) <= bound:
                cusp = True
                break
    return breaks, cusp


@njit(cache=True, nogil=True, error_model='numpy')
def _row_norms(v):
    out = np.empty(len(v))
    for i in range(len(v)): out[i] = math.sqrt(v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1])
    return out


@njit(cache=True, nogil=True, error_model='numpy')
def _quadrature_nodes(lo, hi, low_x, low_w, high_x, high_w):
    """_bending_integral's nodes(lo, hi, LOW) and nodes(lo, hi, HIGH), concatenated."""
    m = len(lo); nl = len(low_x); nh = len(high_x)
    u = np.empty(m * (nl + nh)); w = np.empty(m * (nl + nh))
    for i in range(m):
        mid = (lo[i] + hi[i]) / 2; half = (hi[i] - lo[i]) / 2
        for j in range(nl):
            u[i * nl + j] = mid + half * low_x[j]; w[i * nl + j] = half * low_w[j]
        for j in range(nh):
            u[m * nl + i * nh + j] = mid + half * high_x[j]; w[m * nl + i * nh + j] = half * high_w[j]
    return u, w


@njit(cache=True, nogil=True, error_model='numpy')
def _cross_speed2(tangent, curve):
    """tangent x curve and np.einsum("ij,ij->i", tangent, tangent)."""
    cross = np.empty(len(tangent)); speed2 = np.empty(len(tangent))
    for i in range(len(tangent)):
        cross[i] = tangent[i, 0] * curve[i, 1] - tangent[i, 1] * curve[i, 0]
        speed2[i] = tangent[i, 0] * tangent[i, 0] + tangent[i, 1] * tangent[i, 1]
    return cross, speed2


@njit(cache=True, nogil=True, error_model='numpy')
def _quadrature_points(lo, hi, low_x, low_w, high_x, high_w, t1, velocity, t2, acceleration):
    """The 8- and 16-node points of the open intervals: weights, cross and |velocity|^2 there."""
    u, w = _quadrature_nodes(lo, hi, low_x, low_w, high_x, high_w)
    cross, speed2 = _cross_speed2(_spline_values(t1, velocity, 2, u), _spline_values(t2, acceleration, 1, u))
    return w, cross, speed2


@njit(cache=True, nogil=True, error_model='numpy')
def _quadrature_step(lo, hi, w, cross, power, radius2, nl, nh, total, epsabs, epsrel):
    """One pass of _bending_integral's adaptive loop: the 8/16-node estimates, the accepted intervals and the
    bisected rest. Returns (finite, accepted lo, accepted hi, next lo, next hi, total)."""
    m = len(lo)
    low = np.empty(m); high = np.empty(m); row = np.empty(max(nl, nh))
    for i in range(m):
        for j in range(nl):
            p = i * nl + j
            row[j] = w[p] * (radius2 * (cross[p] * cross[p]) / power[p])
        low[i] = 0. + _pairwise_sum(row, 0, nl)
        for j in range(nh):
            p = m * nl + i * nh + j
            row[j] = w[p] * (radius2 * (cross[p] * cross[p]) / power[p])
        high[i] = 0. + _pairwise_sum(row, 0, nh)
    for i in range(m):
        if not np.isfinite(high[i]):
            return False, lo[:0], hi[:0], lo[:0], hi[:0], total
    level = epsrel * (total + (0. + _pairwise_sum(high, 0, m)))
    done = np.empty(m, np.bool_); accepted = 0
    for i in range(m):
        done[i] = abs(high[i] - low[i]) <= (epsabs + level) * (hi[i] - lo[i])
        if done[i]: accepted += 1
    accepted_lo = np.empty(accepted); accepted_hi = np.empty(accepted); kept = np.empty(accepted)
    rest = m - accepted
    next_lo = np.empty(2 * rest); next_hi = np.empty(2 * rest)
    a = 0; r = 0
    for i in range(m):
        if done[i]:
            accepted_lo[a] = lo[i]; accepted_hi[a] = hi[i]; kept[a] = high[i]; a += 1
        else:
            mid = (lo[i] + hi[i]) / 2
            next_lo[r] = lo[i]; next_lo[rest + r] = mid
            next_hi[r] = mid; next_hi[rest + r] = hi[i]; r += 1
    return True, accepted_lo, accepted_hi, next_lo, next_hi, total + (0. + _pairwise_sum(kept, 0, accepted))


@njit(cache=True, nogil=True, error_model='numpy')
def _sparse_spline_values(t, k, x, columns, pointer, index, value):
    """_spline_values for a coefficient matrix given by rows (pointer, index, value) of its nonzeros. A skipped
    term c * h with c = 0 is +-0 (h is finite), and the sum it would join starts at +0. and can never be -0, so
    leaving it out changes no bit; columns without a nonzero term stay +0. as in the dense sum."""
    out = np.zeros((len(x), columns)); h = np.empty(k + 1); hh = np.empty(k + 1)
    n = len(t) - k - 1
    interval = k
    for ip in range(len(x)):
        xval = x[ip]
        if xval != xval:
            for jp in range(columns): out[ip, jp] = np.nan
            continue
        l = interval if (k < interval and interval < n) else k
        while xval < t[l] and l != k: l -= 1
        l += 1
        while xval >= t[l] and l != n: l += 1
        interval = l - 1
        h[0] = 1.
        for j in range(1, k + 1):
            for q in range(j): hh[q] = h[q]
            h[0] = 0.
            for q in range(1, j + 1):
                ind = interval + q; xb = t[ind]; xa = t[ind - j]
                if xb == xa:
                    h[q] = 0.
                    continue
                w = hh[q - 1] / (xb - xa)
                h[q - 1] = h[q - 1] + w * (xb - xval)
                h[q] = w * (xval - xa)
        for a in range(k + 1):
            r = interval + a - k
            for e in range(pointer[r], pointer[r + 1]):
                out[ip, index[e]] = out[ip, index[e]] + value[e] * h[a]
    return out


def _sparse_rows(c):
    """(columns, pointer, index, value) of the nonzeros of a 2-D coefficient matrix, row by row."""
    rows, cols = np.nonzero(c)
    return c.shape[1], np.r_[0, np.cumsum(np.bincount(rows, minlength=c.shape[0]))].astype(np.int64), cols.astype(np.int64), np.ascontiguousarray(c[rows, cols])


@njit(cache=True, nogil=True, error_model='numpy')
def _bending_points(lo, hi, high_x, high_w, t1, velocity, t2, acceleration, first_t, first_k, first_columns, first_pointer,
                    first_index, first_value, second_t, second_k, second_columns, second_pointer, second_index, second_value):
    """The accepted intervals' 16-node points: weights, velocity, acceleration, |velocity| and the dense
    first/second derivative bases there (from their nonzero coefficients)."""
    empty = np.empty(0)
    u, w = _quadrature_nodes(lo, hi, empty, empty, high_x, high_w)
    tangent = _spline_values(t1, velocity, 2, u)
    return (w, tangent, _spline_values(t2, acceleration, 1, u), _row_norms(tangent),
            _sparse_spline_values(first_t, first_k, u, first_columns, first_pointer, first_index, first_value),
            _sparse_spline_values(second_t, second_k, u, second_columns, second_pointer, second_index, second_value))


@njit(cache=True, nogil=True, error_model='numpy')
def _bending_rows(tangent, acceleration, power5, weights, radius2):
    """_bending_terms' energy and the weighted dv, da rows (power5 = speed ** 5 from numpy)."""
    n = len(tangent)
    energy = np.empty(n); wdv = np.empty((n, 2)); wda = np.empty((n, 2))
    for i in range(n):
        t0 = tangent[i, 0]; t1 = tangent[i, 1]; a0 = acceleration[i, 0]; a1 = acceleration[i, 1]
        speed = math.sqrt(t0 * t0 + t1 * t1)
        cross = t0 * a1 - t1 * a0
        e = radius2 * (cross * cross) / power5[i]
        energy[i] = e
        c2 = 2 * radius2 * cross / power5[i]
        e5 = 5 * e / (speed * speed)
        wdv[i, 0] = weights[i] * (c2 * a1 - e5 * t0); wdv[i, 1] = weights[i] * (c2 * -a0 - e5 * t1)
        wda[i, 0] = weights[i] * (c2 * -t1); wda[i, 1] = weights[i] * (c2 * t0)
    return energy, wdv, wda


@njit(cache=True, nogil=True, error_model='numpy')
def _travel_rows_before(tangent, tx, ty, c, points, radius):
    """_travel_objective up to the density: |tangent|, the stationary test, clearance_gradient(points) and the
    exponent 1 - clearance / radius."""
    n = len(tangent)
    speed = np.empty(n); stationary = False
    eps = np.finfo(np.float64).eps
    for i in range(n):
        speed[i] = math.sqrt(tangent[i, 0] * tangent[i, 0] + tangent[i, 1] * tangent[i, 1])
        if speed[i] <= eps: stationary = True
    clearance = _bicubic_with_gradient(tx, ty, c, np.ascontiguousarray(points[:, 0]), np.ascontiguousarray(points[:, 1]))
    argument = np.empty(n)
    for i in range(n):
        argument[i] = 1.0 - clearance[i, 0] / radius
    return speed, stationary, clearance, argument


@njit(cache=True, nogil=True, error_model='numpy')
def _travel_rows_after(exponential, speed, tangent, clearance, weights, radius):
    """density = 1 + exp(..); density * speed and the weighted dp, dv rows of _travel_objective."""
    n = len(speed)
    cost = np.empty(n); wdp = np.empty((n, 2)); wdv = np.empty((n, 2))
    for i in range(n):
        density = 1.0 + exponential[i]
        cost[i] = density * speed[i]
        factor = -(density - 1.) / radius
        for k in range(2):
            wdp[i, k] = weights[i] * (factor * clearance[i, 1 + k] * speed[i])
            wdv[i, k] = weights[i] * (density * tangent[i, k] / speed[i])
    return cost, wdp, wdv


@njit(cache=True, nogil=True, error_model='numpy')
def _clearance_rows(distance, gradient, basis, projected, minimum, margin, unit):
    """The clearance constraint's values and jacobian (projected = gradient @ (P1 - P0) from numpy)."""
    n = len(distance); controls = basis.shape[1]
    value = np.empty(n); jacobian = np.empty((n, 1 + 2 * (controls - 3)))
    for i in range(n):
        value[i] = (distance[i] - minimum - margin) / unit
        jacobian[i, 0] = basis[i, 1] * projected[i] / unit
        for c in range(2, controls - 1):
            jacobian[i, 1 + 2 * (c - 2)] = basis[i, c] * gradient[i, 0] / unit
            jacobian[i, 2 + 2 * (c - 2)] = basis[i, c] * gradient[i, 1] / unit
    return value, jacobian


@njit(cache=True, nogil=True, error_model='numpy')
def _turning_rows(v, a, first_basis, second_basis, scale, r2):
    """The turning constraint's |v|^2, cross and (samples, controls, 2) control gradient."""
    n = len(v); controls = first_basis.shape[1]
    speed2 = np.empty(n); cross = np.empty(n); gradient = np.empty((n, controls, 2))
    for i in range(n):
        v0 = v[i, 0]; v1 = v[i, 1]; a0 = a[i, 0]; a1 = a[i, 1]
        s2 = v0 * v0 + v1 * v1; x = v0 * a1 - v1 * a0
        speed2[i] = s2; cross[i] = x
        s6 = 6 * (s2 * s2); rc = 2 * r2 * x
        dv0 = scale * (s6 * v0 - rc * a1); dv1 = scale * (s6 * v1 - rc * -a0)
        nrc = -scale * rc
        da0 = nrc * -v1; da1 = nrc * v0
        for c in range(controls):
            gradient[i, c, 0] = first_basis[i, c] * dv0 + second_basis[i, c] * da0
            gradient[i, c, 1] = first_basis[i, c] * dv1 + second_basis[i, c] * da1
    return speed2, cross, gradient


@njit(cache=True, nogil=True, error_model='numpy')
def _turning_jacobian(gradient, projected):
    n = gradient.shape[0]; controls = gradient.shape[1]
    jacobian = np.empty((n, 1 + 2 * (controls - 3)))
    for i in range(n):
        jacobian[i, 0] = projected[i]
        for c in range(2, controls - 1):
            jacobian[i, 1 + 2 * (c - 2)] = gradient[i, c, 0]; jacobian[i, 2 + 2 * (c - 2)] = gradient[i, c, 1]
    return jacobian


try:  # np.linalg.eigvals' own LAPACK call (numpy 1.x), without its wrapper; checked by _fused_objective_exact.
    from numpy.linalg import _umath_linalg
    from numpy.linalg.linalg import get_linalg_error_extobj, _raise_linalgerror_eigenvalues_nonconvergence
    _EIGVALS_EXTOBJ = get_linalg_error_extobj(_raise_linalgerror_eigenvalues_nonconvergence)
except Exception:
    _EIGVALS_EXTOBJ = None


def _eigenvalue_parts(companions, direct=True):
    """Real and imaginary parts of np.linalg.eigvals(companions) (a stack of finite real square matrices calls
    its LAPACK routine directly; anything else goes through np.linalg.eigvals, which raises as the reference)."""
    if direct and _EIGVALS_EXTOBJ is not None and np.isfinite(companions).all():
        solved = _umath_linalg.eigvals(companions, signature='d->D', extobj=_EIGVALS_EXTOBJ)
    else:
        solved = np.linalg.eigvals(companions)
    return np.ascontiguousarray(solved.real), np.ascontiguousarray(solved.imag)


def _fused_bending(t1, velocity, t2, acceleration, first, second, breaks, radius2, epsabs=1e-7, epsrel=1e-6, limit=10000):
    """_bending_integral with velocity = (t1, velocity, 2), acceleration = (t2, acceleration, 1) and the dense
    derivative bases first, second as (t, k, *_sparse_rows(c)), from the fused kernels; the same nodes, sums and
    results."""
    low_x, low_w = _GAUSS_LOW; high_x, high_w = _GAUSS_HIGH
    lo, hi, accepted_lo, accepted_hi, total, count = breaks[:-1], breaks[1:], [], [], 0., 0
    while len(lo):
        count += len(lo)
        if count > limit:
            raise PlanningError("curve energy quadrature did not converge")
        w, cross, speed2 = _quadrature_points(lo, hi, low_x, low_w, high_x, high_w, t1, velocity, t2, acceleration)
        finite, done_lo, done_hi, lo, hi, total = _quadrature_step(lo, hi, w, cross, speed2 ** 2.5, radius2, len(low_x),
                                                                   len(high_x), total, epsabs, epsrel)
        if not finite:
            raise PlanningError("curve energy quadrature did not converge")
        accepted_lo.append(done_lo); accepted_hi.append(done_hi)
    lo, hi = (np.concatenate(accepted_lo), np.concatenate(accepted_hi)) if len(accepted_lo) > 1 else (accepted_lo[0], accepted_hi[0])
    w, tangent, curve, speed, first_u, second_u = _bending_points(lo, hi, high_x, high_w, t1, velocity, t2, acceleration,
                                                                  *first, *second)
    energy, wdv, wda = _bending_rows(tangent, curve, speed ** 5, w, radius2)
    return float(w @ energy), first_u.T @ wdv + second_u.T @ wda


@lru_cache(maxsize=None)
def _fused_objective_exact():
    """Whether the fused kernels reproduce numpy and scipy on this machine (BLAS, SIMD dispatch and compiler flags
    differ): the two-element row sums of norm and einsum, the 3-term self-convolutions of np.convolve, pairwise
    summation, PPoly.from_spline (FITPACK), BSpline.__call__ (evaluate_spline), the direct eigvals call, and np.exp /
    np.power giving each element the same value in any batch. Checked once per process on random data with a wide
    dynamic range and at the knots; _smooth and _simplify_route keep the reference code where any differs."""
    try:
        rng = np.random.default_rng(20240607)
        def wide(*shape):
            return rng.normal(size=shape) * np.exp(rng.normal(0, 6, shape))
        def same(a, b):
            return a.shape == b.shape and np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a), np.signbit(b))
        v = wide(4000, 2); v[:50] = 0.; v[50:100, 0] = -0.
        if not same(_row_norms(v), np.linalg.norm(v, axis=1)): return False
        if not same(_cross_speed2(v, v[::-1].copy())[1], np.einsum("ij,ij->i", v, v)): return False
        for n in (0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 64, 127, 128, 129, 255, 256, 257, 1000, 4099):
            a = wide(n)
            if not same(np.array(_np_sum(a)), np.array(a.sum())): return False
        for width in (8, 16):
            rows = wide(300, width)
            if not same(np.array([0. + _pairwise_sum(row, 0, width) for row in rows]), rows.sum(1)): return False
        # The fused paths evaluate np.exp and np.power on concatenations of the reference's arrays: numpy's loops
        # must give every element the same value at any offset and batch length.
        base = np.abs(wide(1031)); exponent = rng.normal(0, 3, 1031)
        whole = (base ** 2.5, base ** 5, np.exp(exponent))
        for start in range(0, 24):
            for length in (1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33, 100):
                part = slice(start, start + length)
                if not (same(base[part] ** 2.5, whole[0][part]) and same(base[part].copy() ** 5, whole[1][part])
                        and same(np.exp(exponent[part]), whole[2][part])):
                    return False
        companions = np.zeros((64, 3, 3)); companions[:, 1, 0] = 1.; companions[:, 2, 1] = 1.
        companions[:, 0] = wide(64, 3)
        if not all(same(a, b) for a, b in zip(_eigenvalue_parts(companions), _eigenvalue_parts(companions, direct=False))):
            return False
        for count in (4, 5, 6, 9, 17, 23, 40):
            for trial in range(6):
                knots = np.r_[np.zeros(3), np.linspace(0, 1, count - 2), np.ones(3)]
                controls = wide(count, 2) if trial < 4 else np.repeat(wide(1, 2), count, 0)
                if trial == 3: controls[count // 2] = controls[0]
                values = np.r_[rng.normal(), controls[2:-1].ravel()]
                direction = np.array([math.cos(values[0] * 7), math.sin(values[0] * 7)])
                t1 = np.ascontiguousarray(knots[1:-1]); x1 = np.ascontiguousarray(t1[:-1])
                spans = np.flatnonzero(np.diff(t1) > 0)
                fused = _objective_front(values, math.exp(values[0]), direction, controls[0], controls[-1], knots, t1, x1, spans)
                assembled = np.vstack((controls[0], controls[0] + math.exp(values[0]) * direction,
                                       values[1:].reshape(-1, 2), controls[-1]))
                relative = assembled - assembled[:1]
                tangent = _spline_derivative(knots, assembled, 3, 1)
                pieces = [PPoly.from_spline((tangent.t, tangent.c[:, k], tangent.k)) for k in range(2)]
                derivatives = np.asarray([sum(np.convolve(piece.c[:, i], piece.c[:, i]) for piece in pieces)[:-1]
                                          * np.arange(4, 0, -1) for i in spans])
                if not (same(fused[0], assembled) and same(fused[1], tangent.c)
                        and same(fused[2], _spline_derivative(knots, relative, 3, 1).c)
                        and same(fused[3], _spline_derivative(knots, relative, 3, 2).c) and same(fused[4], derivatives)):
                    return False
                x = np.r_[np.unique(knots), rng.uniform(-.1, 1.1, 200), np.nan]
                for spline in (tangent, _spline_derivative(knots, relative, 3, 2)):
                    c = np.ascontiguousarray(spline.c).reshape(spline.c.shape[0], -1)
                    if not same(_spline_values(np.ascontiguousarray(spline.t), c, spline.k, x), spline(x).reshape(len(x), -1)):
                        return False
                identity = BSpline(knots, np.eye(count), 3)
                for spline in (identity, identity.derivative(), identity.derivative(2)):
                    rows = _sparse_rows(spline.c.reshape(spline.c.shape[0], -1))
                    if not same(_sparse_spline_values(np.ascontiguousarray(spline.t), spline.k, x, *rows), spline(x)):
                        return False
        return True
    except Exception:
        return False


def _smooth(grid: Grid, path: np.ndarray, start_yaw: float, turn_radius_m: float,
            initial: BSpline | None = None) -> tuple[BSpline, np.ndarray]:
    """Optimize one continuous curve for length, clearance and turning effort,
    turning nowhere tighter than `turn_radius_m`; `initial` (a certified curve
    on the same route) replaces the route seed."""
    length = path_length(path)
    # One cubic representation at every length, including sub-body-length
    # routes. Four controls are the cubic basis minimum, not a turn constraint.
    count = max(4, math.ceil(length / (2 * ROBOT_FOOTPRINT_RADIUS_M)) + 1)
    controls = points_at_arc(path, np.linspace(0., length, count))
    degree = 3
    knots = np.r_[
        np.zeros(degree), np.linspace(0, 1, len(controls) - degree + 1), np.ones(degree)
    ]
    parameter = np.linspace(0, 1, max(2, math.ceil(length / SAFETY_STEP_M) + 1))
    identity = BSpline(knots, np.eye(len(controls)), degree)
    first, second = identity.derivative(), identity.derivative(2)
    # Gauss quadrature integrates straight spline segments exactly; trapezoids
    # can otherwise lower the apparent length by collapsing control spacing.
    edges = np.unique(np.r_[parameter, knots])
    nodes, gauss_weights = _GAUSS_TRAVEL
    half = np.diff(edges) / 2
    travel_u = ((edges[:-1] + half)[:, None] + half[:, None] * nodes).ravel()
    weights = (half[:, None] * gauss_weights).ravel()
    fused = _fused_objective_exact()
    if fused:
        # The dense bases from their nonzero coefficients (_sparse_spline_values, checked against BSpline.__call__).
        sparse = {id(spline): (np.ascontiguousarray(spline.t), spline.k, *_sparse_rows(spline.c.reshape(spline.c.shape[0], -1)))
                  for spline in (identity, first, second)}

        def dense_basis(spline, x):
            t, k, *rows = sparse[id(spline)]
            return _sparse_spline_values(t, k, np.ascontiguousarray(x, dtype=np.float64), *rows)
    else:
        def dense_basis(spline, x):
            return spline(x)
    travel_basis, travel_first = dense_basis(identity, travel_u), dense_basis(first, travel_u)

    # A clamped cubic has p'(0) parallel to P1-P0. Parameterize this
    # handle by a bounded positive length, so the prescribed heading cannot flip.
    direction = np.array([math.cos(start_yaw), math.sin(start_yaw)])
    offset = controls[1:] - controls[0]
    seed_yaw = math.atan2(offset[0, 1], offset[0, 0])
    turn = math.atan2(math.sin(start_yaw - seed_yaw), math.cos(start_yaw - seed_yaw))
    # Preserve the obstacle-avoiding A* seed away from the initial turn.
    # This initializes controls only; every interior control remains optimized.
    arcs = np.arange(count - 1) * length / (count - 1)
    angles = turn * np.linspace(1., 0., count - 1) * np.exp(-arcs / (2 * ROBOT_FOOTPRINT_RADIUS_M))
    c, s = np.cos(angles), np.sin(angles)
    controls[1:] = controls[0] + np.column_stack(
        (c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1])
    )

    def assemble(values):
        handle = math.exp(values[0]) * direction
        return np.vstack((controls[0], controls[0] + handle,
                          values[1:].reshape(-1, 2), controls[-1]))

    # Value-independent parts of the objective, hoisted out of the SLSQP loop.
    unique_knots = list(np.unique(knots))
    # The tangent spline's knots are knots[1:-1] (splder), so its nonempty spans are fixed.
    spans = np.flatnonzero(np.diff(knots[1:-1]) > 0)
    powers = np.arange(4, 0, -1)

    def objective(values):
        assembled = assemble(values)
        # Split at every tangent-speed extremum: a narrow curvature peak
        # must not hide between fixed quadrature nodes.
        tangent = _spline_derivative(knots, assembled, degree, 1)
        pieces = [PPoly.from_spline((tangent.t, tangent.c[:, k], tangent.k))
                  for k in range(2)]
        breaks = list(unique_knots)
        derivatives = np.asarray([sum(np.convolve(piece.c[:, i], piece.c[:, i]) for piece in pieces)[:-1]
                                  * powers for i in spans])
        cubic = (derivatives[:, 0] != 0) & (derivatives[:, -1] != 0)
        # The same companion matrices as np.roots, solved together across cubic spans.
        companions = np.zeros((int(cubic.sum()), 3, 3))
        companions[:, 1, 0] = 1.; companions[:, 2, 1] = 1.
        companions[:, 0, :] = -derivatives[cubic, 1:] / derivatives[cubic, 0, None]
        solved = iter(np.linalg.eigvals(companions))
        for i, coefficients, regular in zip(spans, derivatives, cubic):
            lo, hi = pieces[0].x[i:i + 2]
            for root in next(solved) if regular else np.roots(coefficients):
                if root.imag == 0 and 0 < root.real < hi - lo:
                    breaks.append(lo + float(root.real))

        scale = np.linalg.norm(tangent.c, axis=1).max()
        if np.any(np.linalg.norm(tangent(breaks), axis=1) <= np.finfo(float).eps * scale):
            # Infinite energy outside the regular-curve domain lets the
            # optimizer shorten its trial step without accepting a cusp.
            return math.inf, np.zeros_like(values)

        # Near a small tangent, basis @ world-controls loses significant digits.
        # Evaluate derivative splines directly in translated coordinates instead.
        relative = assembled - assembled[:1]
        velocity = _spline_derivative(knots, relative, degree, 1)
        acceleration = _spline_derivative(knots, relative, degree, 2)

        try:
            energy, energy_gradient = _bending_integral(
                velocity, acceleration, first, second, np.asarray(sorted(set(breaks))))
        except PlanningError:
            # A trial point next to a cusp, where the adaptive rule cannot converge: infinite energy as for a
            # cusp, so SLSQP shortens its step. (Raising here aborted 39% of the R=0 solves of real captures.)
            return math.inf, np.zeros_like(values)
        # Clearance uses the navigation grid's existing spatial quadrature;
        # only curvature needs adaptive integration around tangent extrema.
        cost, gradient = _travel_objective(assembled, travel_basis, travel_first, weights, grid)
        cost += energy
        gradient += energy_gradient
        return cost, np.concatenate(([gradient[1] @ (assembled[1] - assembled[0])],
                                     gradient[2:-1].ravel()))

    if fused:
        # The same objective from the fused kernels (see _fused_objective_exact); bit-identical values and gradients.
        t1, t2 = np.ascontiguousarray(knots[1:-1]), np.ascontiguousarray(knots[2:-2])
        x1 = np.ascontiguousarray(t1[:-1])
        knot_values = np.asarray(unique_knots)
        first_control, last_control = controls[0].copy(), controls[-1].copy()
        first_tck, second_tck = sparse[id(first)], sparse[id(second)]
        spline_x, spline_y, spline_c = grid.clearance_spline.tck
        eps, radius2 = np.finfo(float).eps, ROBOT_FOOTPRINT_RADIUS_M**2

        def assemble_fused(values):
            return _assemble(values, math.exp(values[0]), direction, first_control, last_control)

        def objective(values):
            assembled, tangent_c, velocity_c, acceleration_c, derivatives, cubic, companions = _objective_front(
                values, math.exp(values[0]), direction, first_control, last_control, knots, t1, x1, spans)
            real, imag = _eigenvalue_parts(companions)
            roots = []
            if not cubic.all():
                for row in np.flatnonzero(~cubic):
                    lo, hi = t1[spans[row]:spans[row] + 2]
                    for root in np.roots(derivatives[row]):
                        if root.imag == 0 and 0 < root.real < hi - lo:
                            roots.append(lo + float(root.real))
            breaks, cusp = _objective_breaks(real, imag, cubic, spans, t1, knot_values, np.asarray(roots, dtype=np.float64),
                                             tangent_c, eps)
            if cusp:
                return math.inf, np.zeros_like(values)
            try:
                energy, energy_gradient = _fused_bending(t1, velocity_c, t2, acceleration_c, first_tck, second_tck, breaks, radius2)
            except PlanningError:
                return math.inf, np.zeros_like(values)
            points, tangent = travel_basis @ assembled, travel_first @ assembled
            speed, stationary, clearance, argument = _travel_rows_before(tangent, spline_x, spline_y, spline_c, points,
                                                                         ROBOT_FOOTPRINT_RADIUS_M)
            if stationary:
                raise PlanningError("curve optimization encountered a stationary tangent")
            rows, wdp, wdv = _travel_rows_after(np.exp(argument), speed, tangent, clearance, weights, ROBOT_FOOTPRINT_RADIUS_M)
            cost = float(weights @ rows)
            gradient = travel_basis.T @ wdp + travel_first.T @ wdv
            cost += energy
            gradient += energy_gradient
            return cost, np.concatenate(([gradient[1] @ (assembled[1] - assembled[0])],
                                         gradient[2:-1].ravel()))

    def constraints(samples):
        """SLSQP inequality constraints at the parameter `samples`."""
        basis, first_basis, second_basis = dense_basis(identity, samples), dense_basis(first, samples), dense_basis(second, samples)

        # SLSQP requests the value and Jacobian separately at the same controls.
        @_cached_value_and_jacobian
        def clearance(values):
            # In centimetres: the same constraint, on the scale of the turning rows and the objective.
            assembled = assemble(values)
            distance, gradient = grid.boundary_distance(basis @ assembled)
            jacobian = np.column_stack((
                basis[:, 1] * (gradient @ (assembled[1] - assembled[0])),
                (basis[:, 2:-1, None] * gradient[:, None, :]).reshape(len(samples), -1),
            ))
            return (distance - grid.minimum_clearance_m - FIT_MARGIN_M) / CONSTRAINT_UNIT_M, jacobian / CONSTRAINT_UNIT_M

        @_cached_value_and_jacobian
        def turning(values):
            # curvature^2 <= 1/R^2 as the polynomial |v|^6 - R^2 cross^2 >= 0: no
            # division near a small tangent, and a cusp (v = 0) sits on the boundary.
            # Divided by length^6, so every sample is O(1) in parameter units.
            assembled = assemble(values)
            v, a = first_basis @ assembled, second_basis @ assembled
            speed2, cross = np.einsum("ij,ij->i", v, v), v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]
            scale, r2 = length ** -6, turn_radius_m**2
            dv = scale * ((6 * speed2**2)[:, None] * v
                          - (2 * r2 * cross)[:, None] * np.column_stack((a[:, 1], -a[:, 0])))
            da = -scale * (2 * r2 * cross)[:, None] * np.column_stack((-v[:, 1], v[:, 0]))
            gradient = first_basis[:, :, None] * dv[:, None] + second_basis[:, :, None] * da[:, None]
            jacobian = np.column_stack((gradient[:, 1] @ (assembled[1] - assembled[0]),
                                        gradient[:, 2:-1].reshape(len(samples), -1)))
            return scale * (speed2**3 - r2 * cross**2), jacobian

        if fused:
            @_cached_value_and_jacobian
            def clearance(values):
                assembled = assemble_fused(values)
                distance, gradient = grid.boundary_distance(basis @ assembled)
                return _clearance_rows(distance, gradient, basis, gradient @ (assembled[1] - assembled[0]),
                                       grid.minimum_clearance_m, FIT_MARGIN_M, CONSTRAINT_UNIT_M)

            @_cached_value_and_jacobian
            def turning(values):
                assembled = assemble_fused(values)
                v, a = first_basis @ assembled, second_basis @ assembled
                scale, r2 = length ** -6, turn_radius_m**2
                speed2, cross, gradient = _turning_rows(v, a, first_basis, second_basis, scale, r2)
                return scale * (speed2**3 - r2 * cross**2), _turning_jacobian(gradient, gradient[:, 1] @ (assembled[1] - assembled[0]))

        rows = [{"type": "ineq", "fun": lambda x: clearance(x)[0], "jac": lambda x: clearance(x)[1]}]
        # At R = 0 the turning rows read |v|^6 >= 0, which every curve satisfies; as ~L/SAFETY_STEP_M extra rows
        # they only made SLSQP's least-squares subproblem fail ("More than 3*n iterations", 64% of R=0 solves).
        if turn_radius_m > 0:
            rows.append({"type": "ineq", "fun": lambda x: turning(x)[0], "jac": lambda x: turning(x)[1]})
        return rows

    lower = grid.origin_xy + grid.cell_size_m * 0.5
    upper = grid.origin_xy + (np.array(grid.free.shape) - 0.5) * grid.cell_size_m
    reach = min((upper[k] - controls[0, k]) / direction[k] if direction[k] > 0
                else (lower[k] - controls[0, k]) / direction[k]
                for k in range(2) if abs(direction[k]) > np.finfo(float).eps)
    seed = controls if initial is None else initial.c
    values = np.r_[math.log(np.linalg.norm(seed[1] - seed[0])), seed[2:-1].ravel()]
    bounds = [(math.log(MIN_START_HANDLE_M), math.log(reach))] + list(zip(lower, upper)) * (len(controls) - 3)
    samples = parameter
    # The constraint samples are uniform in the spline parameter, so in arc length they can be wider than the
    # FIT_MARGIN_M chord allowance. Where the certificate fails, its failing intervals become constraint samples
    # too and the solve continues from its last point (cutting planes on the certificate itself).
    for refinement in range(CERTIFICATE_REFINEMENTS + 1):
        result = minimize(
            objective, values, jac=True, method="SLSQP", bounds=bounds, constraints=constraints(samples),
            options={"maxiter": 300, "ftol": 1e-6},
        )
        if not result.success:
            raise PlanningError(f"clearance curve optimization failed: {result.message}")
        curve = BSpline(knots, assemble(result.x), degree)
        # Derivative control hull bounds arc distance between certification samples.
        bound = np.linalg.norm(curve.derivative().c, axis=1).max()
        dense = np.linspace(0, 1, max(2, math.ceil(bound / SAFETY_STEP_M) + 1))
        sampled = curve(dense)
        if grid.safe(sampled):
            return curve, sampled
        failing = [k for k in range(len(dense) - 1) if not grid.safe(sampled[k:k + 2])]
        samples = np.unique(np.r_[samples, np.linspace(dense[failing], dense[np.add(failing, 1)], 5).ravel()])
        values = result.x
    raise PlanningError("optimized curve leaves the source configuration space")


def _spatial_route(grid: Grid, start_xy: np.ndarray, goal_xy: np.ndarray,
                   reach_m: float | None = None) -> tuple[np.ndarray, float]:
    """Clearance-aware A* polyline and the direction of its first segment (with `reach_m`, only a leading part
    whose arc exceeds reach_m or the whole route; see _simplify_route)."""
    cells = _astar(grid, grid.snap(start_xy), grid.snap(goal_xy))
    simplified = _simplify_route(grid, np.vstack([start_xy, grid.grid_to_world(cells), goal_xy]), reach_m)
    segments = np.diff(simplified, axis=0)
    nonzero = np.flatnonzero(np.any(segments != 0, axis=1))
    if not len(nonzero):
        raise PlanningError("spatial route has no nonzero initial segment")
    direction = segments[nonzero[0]]
    return simplified, math.atan2(direction[1], direction[0])


def _plan(grid: Grid, simplified: np.ndarray, start_yaw: float,
          ) -> Plan:
    """The certified curve with the widest turning radius in TURN_RADII_M that
    keeps LABEL_CLEARANCE_M (a tighter start turn is kept whatever its clearance)."""
    curve_spline, path = _smooth(grid, simplified, start_yaw, TURN_RADII_M[0])
    radius = TURN_RADII_M[0]
    for wider in TURN_RADII_M[1:]:
        if wider <= radius:
            continue
        if 1 / curvature(path).max() >= wider:
            radius = wider
            continue
        try:
            wider_spline, wider_path = _smooth(grid, simplified, start_yaw, wider, curve_spline)
        except PlanningError:
            break
        if radius >= 1 / MAX_TURN_RATE_PER_M and grid.boundary_distance(wider_path)[0].min() < LABEL_CLEARANCE_M:
            break
        curve_spline, path, radius = wider_spline, wider_path, wider
    if radius < 1 / MAX_TURN_RATE_PER_M:
        raise PlanningError("no forward curve turns wider than 1/MAX_TURN_RATE_PER_M")
    length = path_length(path)
    clear, curve = grid.boundary_distance(path)[0], curvature(path)
    heading = np.arctan2(np.diff(path, axis=0)[:, 1], np.diff(path, axis=0)[:, 0])
    heading_change = np.arctan2(
        np.sin(np.diff(heading)),
        np.cos(np.diff(heading)),
    )
    metrics = {
        "length_m": length,
        "minimum_clearance_m": float(clear.min()),
        "clearance_p05_m": float(np.percentile(clear, 5)),
        "curvature_p95": float(np.percentile(curve, 95)),
        "maximum_curvature": float(curve.max()),
        "maximum_turn_rate_per_m": float(turn_rate_per_m(path).max()),
        "total_turn_radians": float(np.abs(heading_change).sum()),
        "turn_radius_bound_m": radius,
    }
    return Plan(curve_spline, path.astype(np.float32), metrics)
