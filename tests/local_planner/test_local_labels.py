"""local_labels' fused path equals the replay through ObservedBSplineRoutes/ObservedHitHistory bit for bit."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial import cKDTree
from kappanav.local_planner.data import local_replay
from kappanav.local_planner.data.observed_bspline_routes import body_pose
from kappanav.local_planner.data.observed_geometry import _nearest, _nearest_exact, nearest_distances

K = np.asarray([[114.239494, 0., 112.], [0., 114.239494, 63.], [0., 0., 1.]], dtype=np.float32)
CAMERA = np.asarray([[0., -.17364818, .9848077, .14309], [-1., 0., 0., 0.], [0., -.9848077, -.17364818, .31266],
                     [0., 0., 0., 1.]], dtype=np.float32)
ACTOR = SimpleNamespace(policy=SimpleNamespace(depth_encoder=SimpleNamespace(
    metric_projector=SimpleNamespace(planning_horizon_m=3.5999999999999996, radius=0.167584539))))


def _render(view, boxes, rng, shape=(126, 224)):
    """Normalized optical-Z depth of a floor and vertical boxes [x0,y0,x1,y1,height], FP16 with dropouts."""
    world = body_pose(view).astype(float)@CAMERA.astype(float)
    y, x = np.indices(shape, dtype=float)
    rays = np.stack(((x+.5-K[0, 2])/K[0, 0], (y+.5-K[1, 2])/K[1, 1], np.ones_like(x)), -1)
    direction = rays@world[:3, :3].T; origin = world[:3, 3]
    depth = np.full(shape, np.inf)
    floor = -origin[2]/np.where(direction[..., 2] < 0, direction[..., 2], -np.inf)
    depth = np.where(floor > 0, np.minimum(depth, floor), depth)
    for x0, y0, x1, y1, height in boxes:
        lo, hi = np.asarray([x0, y0, 0.]), np.asarray([x1, y1, height])
        with np.errstate(divide='ignore', invalid='ignore'):
            a, b = (lo-origin)/direction, (hi-origin)/direction
        enter = np.nanmax(np.minimum(a, b), -1); leave = np.nanmin(np.maximum(a, b), -1)
        depth = np.where((enter <= leave) & (enter > 0), np.minimum(depth, enter), depth)
    depth = np.minimum(depth/5, 1.)
    depth[rng.random(shape) < .02] = 0
    return depth.astype(np.float16)


def _capture(seed):
    rng = np.random.default_rng(seed)
    boxes = [(-3., 1.2, 6., 1.5, 1.), (-3., -1.5, 6., -1.2, 1.), (2.2, -.3, 2.6, .4, .12), (3.5, .5, 3.9, .9, .6)]
    boxes += [(x, y, x+.2, y+.2, .1) for x, y in rng.uniform(-1, 4, (3, 2))]
    views = np.stack([np.linspace(0, 1.2, 12)+rng.normal(0, .02, 12), np.linspace(0, .3, 12)*rng.normal(),
                      np.linspace(0, 1, 12)*rng.normal(0, .6)], -1)
    return dict(depths=np.stack([_render(view, boxes, rng) for view in views]), acquired_views=views,
                body_to_world=np.stack([body_pose(view) for view in views]),
                camera_intrinsics=np.repeat(K[None], len(views), 0), camera_to_body=np.repeat(CAMERA[None], len(views), 0))


def _bits(array):
    array = np.ascontiguousarray(array.numpy() if hasattr(array, 'numpy') else array)
    return array.dtype, array.shape, array.tobytes()


@pytest.mark.parametrize('seed', range(4))
def test_fused_local_labels_equal_the_replay_bit_for_bit(seed):
    if not local_replay._kernels_exact((126, 224)):
        pytest.skip('this BLAS does not round like the kernels: local_labels replays')
    raw = _capture(seed); rng = np.random.default_rng(seed); free = hits = 0
    for _ in range(6):
        selected = sorted(rng.choice(12, size=int(rng.integers(1, 5)), replace=False).tolist())
        plan = dict(selection=dict(indices=selected))
        field, tree, error = local_replay.fused_labels(ACTOR, raw, plan)
        expected_field, expected_tree, expected_error = local_replay.replayed_labels(ACTOR, raw, plan)
        free += int(expected_field[5].sum()); hits += expected_tree.n
        assert _bits(field) == _bits(expected_field) and _bits(tree.data) == _bits(expected_tree.data)
        assert error == expected_error
    assert free and hits                      # the scene exercises observed free space and hit distances


def test_unsupported_depth_falls_back_to_the_replay():
    raw = _capture(0); raw['depths'] = raw['depths'].copy(); raw['depths'][3, 60, 100] = np.nan
    plan = dict(selection=dict(indices=[1, 3]))
    with pytest.raises(local_replay._Unsupported):
        local_replay.fused_labels(ACTOR, raw, plan)
    field, tree, error = local_replay.local_labels(ACTOR, raw, plan)
    expected = local_replay.replayed_labels(ACTOR, raw, plan)
    assert _bits(field) == _bits(expected[0]) and _bits(tree.data) == _bits(expected[1].data)


def test_nearest_distances_equal_the_kd_tree():
    if not _nearest_exact():
        pytest.skip('this scipy does not round like the kernel')
    rng = np.random.default_rng(5)
    for count in (1, 2, 50, 4000):
        points = np.concatenate((rng.normal(0, 2, (count, 2)), np.round(rng.normal(0, 1, (count, 2)), 1)))
        tree = cKDTree(points)
        axis = np.arange(64)*(7.2/63)-3.6
        grid = np.stack(np.meshgrid(axis, axis, indexing='xy'), -1).reshape(-1, 2)
        for queries in (grid, rng.normal(0, 5, (999, 2)), points[:300]):
            np.testing.assert_array_equal(nearest_distances(tree, queries), tree.query(queries)[0])
            for cell in (.01, .3, 10.):
                np.testing.assert_array_equal(_nearest(np.ascontiguousarray(queries), tree.data, cell), tree.query(queries)[0])
