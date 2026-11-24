import numpy as np
import pytest

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from kappanav.local_planner.data.goal_distance import GoalDistanceQuery
from kappanav.common.grid_dijkstra import grid_distances
from kappanav.local_planner.data.privileged import SourceConfigurationGrid, SourceConfigurationSpaceQuery


def scipy_reference(free, cell_size, cell):
    """Independent 8-neighbour graph with closed-cell diagonal contact, solved by scipy Dijkstra."""
    ids = np.arange(free.size).reshape(free.shape)
    rows, cols, lengths = [], [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        x0, x1 = slice(0, free.shape[0] - dx), slice(dx, free.shape[0])
        y0 = slice(max(0, -dy), min(free.shape[1], free.shape[1] - dy))
        y1 = slice(max(0, dy), min(free.shape[1], free.shape[1] + dy))
        valid = free[x0, y0] & free[x1, y1]
        if dx and dy:
            valid &= free[x1, y0] & free[x0, y1]
        a, b = ids[x0, y0][valid], ids[x1, y1][valid]
        rows.extend((a, b)); cols.extend((b, a))
        edge = np.full(len(a), np.hypot(dx, dy) * cell_size)
        lengths.extend((edge, edge))
    graph = coo_matrix((np.concatenate(lengths), (np.concatenate(rows), np.concatenate(cols))), shape=(free.size, free.size)).tocsr()
    return dijkstra(graph, directed=True, indices=int(np.ravel_multi_index(cell, free.shape))).astype(np.float32).reshape(free.shape)


@pytest.mark.parametrize("seed", range(6))
def test_implicit_grid_matches_scipy_including_blocked_corners(seed):
    pytest.importorskip("numba")
    from kappanav.common.grid_dijkstra import grid_distances
    rng = np.random.default_rng(seed)
    free = rng.random((55, 71)) > 0.25
    free[20, :] = False
    free[0, 0] = True
    free[0, 1] = free[1, 0] = False
    grid = SourceConfigurationGrid(np.where(free, 1., -1.).astype(np.float32), np.zeros(2), 0.025)
    cell = np.argwhere(free)[100]
    full = grid_distances(free, grid.cell_size_m, int(cell[0]), int(cell[1])).astype(np.float32)
    np.testing.assert_array_equal(full, scipy_reference(free, grid.cell_size_m, cell).astype(np.float32))
    isolated = grid_distances(free, grid.cell_size_m, 0, 0)
    assert np.isfinite(isolated).sum() == 1


@pytest.mark.parametrize('seed', range(6))
def test_local_queries_match_full_scipy_without_retaining_origin_fields(seed):
    rng = np.random.default_rng(seed)
    free = rng.random((55, 71)) > .25
    free[20, :] = False
    cell = np.argwhere(free)[100]
    grid = SourceConfigurationGrid(np.where(free, 1., -1.).astype(np.float32), np.zeros(2), .05)
    source = SourceConfigurationSpaceQuery((grid,))
    cells = np.concatenate((cell[None], rng.integers([0, 0], free.shape, (100, 2)), cell[None], [[-1, 0], [55, 71]]))
    query = GoalDistanceQuery(source)
    reference = grid_distances(free, grid.cell_size_m, int(cell[0]), int(cell[1])).astype(np.float32)
    inside = (cells >= 0).all(-1) & (cells < free.shape).all(-1)
    expected = np.full(len(cells), np.inf, np.float32)
    expected[inside] = reference[cells[inside, 0], cells[inside, 1]]
    np.testing.assert_array_equal(query.query(0, cell, cells), expected)


def test_parallel_rows_match_serial():
    import pickle
    free=np.ones((40,45),np.float32);free[10:25,20]=-1
    source=SourceConfigurationSpaceQuery((SourceConfigurationGrid(free,np.zeros(2),.05),))
    query=GoalDistanceQuery(source)
    cells=np.array([[[3,3],[4,6],[3,3]],[[4,4],[8,8],[7,7]],[[30,30],[32,31],[30,30]],[[5,6],[8,9],[7,7]]])
    expected=query.query_rows([0]*4,cells)
    actual=query.query_rows([0]*4,cells,parallel=True)
    for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    restored=pickle.loads(pickle.dumps(query));assert not restored.free_grids
    for a,b in zip(restored.query_rows([0]*4,cells),expected):np.testing.assert_array_equal(a,b)
