"""The fused planner kernels against the reference code paths they replace: bit-identical results."""
import pytest
import numpy as np
from scipy.ndimage import distance_transform_edt
from kappanav.common import geometry
from kappanav.common.geometry import Grid, PlanningError


def _grid():
    free = np.ones((90, 70), dtype=bool)
    free[25:55, 30:38] = False; free[60:64, 10:60] = False; free[:, :3] = False
    clearance = distance_transform_edt(np.pad(free, 1, constant_values=False))[1:-1, 1:-1] * .05
    clearance[~free] = 0.
    return Grid(free, clearance.astype(np.float32), np.array([-1.3, .7]), .05)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a), np.signbit(b))


def test_fused_kernels_reproduce_numpy_here():
    assert geometry._fused_objective_exact()


def test_boundary_distance_matches_the_reference_search():
    grid = _grid(); rng = np.random.default_rng(3)
    lo = grid.origin_xy; hi = lo + np.asarray(grid.free.shape) * grid.cell_size_m
    cells = rng.integers(0, grid.free.shape, (4000, 2))
    xy = np.r_[rng.uniform(lo - .7, hi + .7, (6000, 2)),
               grid.origin_xy + (cells + rng.choice([0., .25, .5, 1.], (4000, 2))) * grid.cell_size_m]
    for batch in (xy[:1], xy[:64], xy):
        value, gradient = grid.boundary_distance(batch)
        reference = grid._boundary_distance_reference(batch)
        assert _same(value, reference[0]) and _same(gradient, reference[1])
    # A non-finite point takes the reference path.
    assert _same(grid.boundary_distance(np.array([[np.nan, 1.]]))[0], grid._boundary_distance_reference(np.array([[np.nan, 1.]]))[0])


def test_bicubic_matches_the_reference_kernel():
    grid = _grid(); rng = np.random.default_rng(4)
    tx, ty, c = grid.clearance_spline.tck
    lo = grid.origin_xy; hi = lo + np.asarray(grid.free.shape) * grid.cell_size_m
    for xy in (rng.uniform(lo - 1, hi + 1, (5000, 2)), np.stack([rng.choice(tx, 500), rng.choice(ty, 500)], 1),
               np.array([[np.nan, 0.], [1., 1.]])):
        x, y = np.ascontiguousarray(xy[:, 0]), np.ascontiguousarray(xy[:, 1])
        assert _same(geometry._bicubic_with_gradient(tx, ty, c, x, y), geometry._bicubic_with_gradient_reference(tx, ty, c, x, y))


def _plans(grid, monkeypatch, fused):
    monkeypatch.setattr(geometry, '_fused_objective_exact', lambda: fused)
    rng = np.random.default_rng(5); out = []
    for start, goal in (((0., 1.5), (3.0, 1.6)), ((-1., 3.), (2.8, 3.2)), ((2.5, 1.0), (-0.9, 1.1)), ((0.2, 0.9), (0.4, 1.0))):
        start, goal = np.asarray(start), np.asarray(goal)
        try:
            route = geometry._spatial_route(grid, start, goal)[0]
        except PlanningError as error:
            out.append(('route', str(error))); continue
        out.append(('route', route))
        for yaw in (0., float(rng.uniform(-np.pi, np.pi)), np.pi):
            try:
                plan = geometry._plan(grid, route, yaw)
                out.append((plan.path_xy, plan.curve.c, plan.metrics))
            except PlanningError as error:
                out.append(('plan', str(error)))
    return out


@pytest.mark.dingo
def test_fused_planner_is_bit_identical_to_the_reference(monkeypatch):
    fused = _plans(_grid(), monkeypatch, True)
    reference = _plans(_grid(), monkeypatch, False)
    assert len(fused) == len(reference)
    for a, b in zip(fused, reference):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            if isinstance(x, dict):
                assert x == y
            elif isinstance(x, str):
                assert x == y
            else:
                assert _same(x, y)
