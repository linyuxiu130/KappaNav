"""Training-only obstacle-aware goal distances on immutable robot C-space."""

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from kappanav.common.grid_dijkstra import grid_distances_to_cells
from kappanav.local_planner.data.privileged import SourceConfigurationSpaceQuery

_POOL = None
# Threads per process for parallel row queries.
QUERY_THREADS = 8

class GoalDistanceQuery:
    """Geodesic distances from a goal cell to a few cells (candidate endpoints and the start) on immutable C-space.

    Each query runs the eight-neighbour Dijkstra from the goal and stops once every requested cell is settled; a
    settled distance does not depend on where the search stops, so the values equal the full field's. No per-goal
    field is kept: with relabelled goals almost every training row has its own goal, and caching full-scene fields
    grew each training rank by ~48 GB per epoch toward exhausting host memory, while a capped cache missed.
    """

    def __init__(self, source: SourceConfigurationSpaceQuery):
        self.source = source
        self.free_grids = {}
        self.components = {}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.free_grids = {}
        self.components = {}

    def connected(self, index: int, a: np.ndarray, b: np.ndarray) -> bool:
        """Whether grid_distances between two traversable cells is finite. Its diagonal steps need both side cells
        free, so its connectivity is exactly four-neighbour connectivity: one component labelling per grid
        replaces a full-scene Dijkstra per goal."""
        if index not in self.components:
            from scipy.ndimage import label
            self.components[index] = label(self._free(index))[0]
        labels = self.components[index]
        return bool(labels[tuple(a)] != 0 and labels[tuple(a)] == labels[tuple(b)])

    def _free(self, index):
        if index not in self.free_grids:
            self.free_grids[index] = self.source.grids[index].signed_clearance_m >= 0
        return self.free_grids[index]

    def _goal_grid(self, index, cell):
        grid = self.source.grids[index]
        if np.any(cell < 0) or np.any(cell >= grid.signed_clearance_m.shape):
            raise ValueError("training mission goal lies outside source C-space")
        if grid.signed_clearance_m[tuple(cell)] < 0:
            raise ValueError("training mission goal is not traversable in source C-space")
        return grid

    def query(self, index: int, goal_cell: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Distances from goal_cell to `cells` [..., 2] (inf outside the grid or unreachable)."""
        grid = self._goal_grid(index, goal_cell)
        values = np.full(cells.shape[:-1], np.inf, dtype=np.float32)
        inside = (cells >= 0).all(-1) & (cells < grid.signed_clearance_m.shape).all(-1)
        if inside.any():
            values[inside] = grid_distances_to_cells(self._free(index), grid.cell_size_m, int(goal_cell[0]),
                                                     int(goal_cell[1]), np.ascontiguousarray(cells[inside], dtype=np.int64))
        return values

    def query_rows(self, indices, cells, *, parallel=False):
        """query per row with the goal in the last column; with `parallel` the rows run on a thread pool (the
        Dijkstra kernel releases the GIL)."""
        global _POOL
        for index in set(indices): self._free(index)
        jobs = list(zip(indices, cells))
        run = lambda job: self.query(job[0], job[1][-1], job[1][:-1])
        if not parallel:
            return [run(job) for job in jobs]
        # GPU workers create this lazily; the CPU forkserver initializer stays thread-free.
        if _POOL is None: _POOL = ThreadPoolExecutor(max_workers=QUERY_THREADS)
        return list(_POOL.map(run, jobs))
