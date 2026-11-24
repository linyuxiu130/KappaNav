"""Exact eight-neighbour grid distances without materializing a sparse graph."""

import heapq
import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def grid_distances(free: np.ndarray, cell_size: float, gx: int, gy: int, targets=None, limit=np.inf) -> np.ndarray:
    """Distances from cell (gx, gy); inf where unreached. The search stops once every `targets` cell is settled,
    and does not expand cells farther than `limit`."""
    height, width = free.shape
    distances = np.full((height, width), np.inf, dtype=np.float64)
    pending = np.zeros((height, width), dtype=np.bool_)
    remaining = -1
    if targets is not None:
        remaining = 0
        for x, y in targets:
            if not pending[x, y]:
                pending[x, y] = True
                remaining += 1
    distances[gx, gy] = 0.0
    queue = [(0.0, gx * width + gy)]
    while queue and remaining != 0:
        distance, node = heapq.heappop(queue)
        x, y = node // width, node % width
        if distance != distances[x, y]:
            continue
        if distance > limit:
            break
        if remaining > 0 and pending[x, y]:
            pending[x, y] = False
            remaining -= 1
            if not remaining:
                break
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= height or ny >= width or not free[nx, ny]:
                    continue
                if dx != 0 and dy != 0 and (not free[nx, y] or not free[x, ny]):
                    continue
                edge = (math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0) * cell_size
                candidate = distance + edge
                if candidate < distances[nx, ny]:
                    distances[nx, ny] = candidate
                    heapq.heappush(queue, (candidate, nx * width + ny))
    return distances


@njit(cache=True, nogil=True)
def grid_distances_to_cells(free, cell_size, gx, gy, cells):
    """Stop the same Dijkstra search once every requested cell is settled."""
    distances = grid_distances(free, cell_size, gx, gy, cells)
    return np.array([distances[x, y] for x, y in cells], dtype=np.float32)


@njit(cache=True, nogil=True)
def descend(distances: np.ndarray, free: np.ndarray, cell_size: float, x: int, y: int) -> np.ndarray:
    """The cells of a shortest path from (x, y) down a grid_distances field over `free` to its source, both ends
    included: each step goes to the neighbour the field was relaxed through (same moves as grid_distances)."""
    height, width = free.shape
    path = [(x, y)]
    while distances[x, y] > 0:
        best, bx, by = np.inf, -1, -1
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if (dx == 0 and dy == 0) or nx < 0 or ny < 0 or nx >= height or ny >= width or not free[nx, ny]:
                    continue
                if dx != 0 and dy != 0 and (not free[nx, y] or not free[x, ny]):
                    continue
                through = distances[nx, ny] + (math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0) * cell_size
                if through < best:
                    best, bx, by = through, nx, ny
        if bx < 0:
            raise ValueError('no finite descent: the cell is not reached by the field')
        x, y = bx, by
        path.append((x, y))
    cells = np.empty((len(path), 2), np.int64)
    for i in range(len(path)):
        cells[i, 0], cells[i, 1] = path[i]
    return cells
