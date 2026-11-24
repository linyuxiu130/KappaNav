"""One source configuration-space oracle for supervision and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import Tensor

from kappanav.common.physical import EXTRA_CLEARANCE_M, PATH_CONFIGURATION_QUERY_SPACING_M
from kappanav.local_planner.model.trajectory import path_arc_length


SOURCE_CONFIGURATION_QUERY_SPACING_M = PATH_CONFIGURATION_QUERY_SPACING_M


def _closed_grid_cells(coordinates: Tensor) -> Tensor:
    """All cells touching a point; occupied cells include their edges and corners."""
    nearest = coordinates.round()
    # FP64 transform/interpolation roundoff at an analytically exact grid crossing.
    tolerance = 4 * torch.finfo(coordinates.dtype).eps * coordinates.abs().clamp_min(1)
    coordinates = torch.where((coordinates - nearest).abs() <= tolerance, nearest, coordinates)
    high = coordinates.floor().long()
    low = coordinates.ceil().long() - 1
    return torch.stack((
        high,
        torch.stack((low[..., 0], high[..., 1]), -1),
        torch.stack((high[..., 0], low[..., 1]), -1),
        low,
    ), -2)


@dataclass(frozen=True)
class SourceConfigurationGrid:
    """Dingo centre-clearance values with an explicit world coordinate frame."""

    signed_clearance_m: np.ndarray
    origin_xy: np.ndarray
    cell_size_m: float
    coordinate_frame: str = "habitat_xz"

    @classmethod
    def load(cls, path: Path) -> "SourceConfigurationGrid":
        with np.load(path) as values:
            free = np.asarray(values["free"], dtype=np.bool_)
            clearance = np.asarray(values["clearance_m"], dtype=np.float32)
            origin = np.asarray(values["origin_xy"], dtype=np.float64)
            cell_size = float(values["cell_size_m"])
            frame = str(values["coordinate_frame"].item()) if "coordinate_frame" in values else "habitat_xz"
        distance_inside_obstacle = distance_transform_edt(~free).astype(np.float32)
        signed = np.where(
            free,
            clearance,
            -distance_inside_obstacle * cell_size,
        ).astype(np.float32)
        return cls(signed, origin, cell_size, frame)

    def query_world(self, world_xy: np.ndarray) -> np.ndarray:
        """Use the same closed-cell clearance as continuous trajectory queries."""
        world_xy = np.asarray(world_xy, dtype=np.float64)
        cells = _closed_grid_cells(torch.from_numpy(
            (world_xy - self.origin_xy) / self.cell_size_m
        )).numpy()
        values = np.full(cells.shape[:-1], -np.inf, dtype=np.float32)
        inside = (
            (cells[..., 0] >= 0)
            & (cells[..., 0] < self.signed_clearance_m.shape[0])
            & (cells[..., 1] >= 0)
            & (cells[..., 1] < self.signed_clearance_m.shape[1])
        )
        values[inside] = self.signed_clearance_m[
            cells[inside, 0], cells[inside, 1]
        ]
        return values.min(-1)

    @cached_property
    def boundary_segments(self):
        free=self.signed_clearance_m>=0
        padded=np.pad(free,1)
        segments=[]
        # A run of touching collinear cell edges is the same closed segment.
        for edges,swap in ((padded[1:,1:-1]!=padded[:-1,1:-1],False),
                           ((padded[1:-1,1:]!=padded[1:-1,:-1]).T,True)):
            changes=np.diff(np.pad(edges.astype(np.int8),((0,0),(1,1))),axis=1)
            rows,starts=np.nonzero(changes==1)
            _,ends=np.nonzero(changes==-1)
            start=np.stack((rows,starts),-1);end=np.stack((rows,ends),-1)
            if swap:start,end=start[:,::-1],end[:,::-1]
            segments.append(np.stack((start,end),1))
        return self.origin_xy+np.concatenate(segments)*self.cell_size_m

@dataclass(frozen=True)
class SourcePathQuery:
    """One dense source-grid measurement of a robot-frame trajectory."""

    local_path: Tensor
    clearance_m: Tensor
    in_world_bounds: Tensor
    active: Tensor
    grid_cells: Tensor

    @property
    def minimum_clearance_m(self) -> Tensor:
        return self.clearance_m.masked_fill(~self.active, torch.inf).amin(dim=-1)

    @property
    def path_field_coverage_fraction(self) -> Tensor:
        return (self.in_world_bounds & self.active).sum(dim=-1) / self.active.sum(
            dim=-1
        ).clamp_min(1)


class SourceConfigurationSpaceQuery:
    """Query immutable source C-space at one spacing with one OOB definition.

    The oracle validates expert B-splines, supervises candidate scores, and
    independently evaluates trajectories.  It is never part of the deployed
    depth/PointGoal policy input.
    """

    def __init__(self, grids: tuple[SourceConfigurationGrid, ...]) -> None:
        self.grids = grids
        self._atlases: dict[torch.device, tuple[Tensor, ...]] = {}
        self._boundaries={}

    def signed_distance(self,world:Tensor,index:int)->Tensor:
        return self.signed_distance_batch(world[None],index)[0]

    def signed_distance_batch(self,world:Tensor,index:int)->Tensor:
        """Exact closed-cell distances; batch independent boundary searches on one map."""
        grid=self.grids[index];key=(world.device,index,world.dtype)
        if key not in self._boundaries:
            segments=torch.as_tensor(grid.boundary_segments,device=world.device,dtype=torch.float64)
            delta=segments[:,1]-segments[:,0]
            self._boundaries[key]=(segments[:,0],delta,delta.square().sum(-1),
                torch.as_tensor(grid.signed_clearance_m>=0,device=world.device),
                world.new_tensor(grid.origin_xy),world.new_tensor(grid.signed_clearance_m.shape))
        a,delta,length2,free,origin,shape=self._boundaries[key]
        batch=world.shape[0]
        with torch.no_grad():
            center=(world.detach().amin(1)+world.detach().amax(1))/2
            reach=(world.detach()-center[:,None]).norm(dim=-1).amax(1)
            fraction=(((center[:,None]-a)*delta).sum(-1)/length2).clamp(0,1)
            squared=(center[:,None]-a-fraction[...,None]*delta).square().sum(-1)
            # Each row keeps its own triangle-inequality bound, even for distant poses.
            upper=squared.amin(-1).sqrt()+2*reach+1e-12
            valid=squared<=upper[:,None].square()
            if world.is_cuda:
                from kappanav.local_planner.data._boundary_cuda import nearest_valid_boundary
                chosen=nearest_valid_boundary(world.detach(),a,delta,length2,valid)
                start=a[chosen];direction=delta[chosen];denominator=length2[chosen]
            else:
                width=int(valid.sum(-1).max().item())
                selected=(~valid).to(torch.int32).argsort(dim=-1,stable=True)[:,:width]
                active=valid.gather(1,selected)
                a=a[selected];delta=delta[selected];length2=length2[selected]
                nearest=[]
                for chunk in world.detach().split(max(1,(1<<20)//(batch*width)),dim=1):
                    fraction=(((chunk[:,:,None]-a[:,None])*delta[:,None]).sum(-1)/length2[:,None]).clamp(0,1)
                    distance=(chunk[:,:,None]-a[:,None]-fraction[...,None]*delta[:,None]).square().sum(-1)
                    nearest.append(distance.masked_fill(~active[:,None],torch.inf).argmin(-1))
                chosen=torch.cat(nearest,1)
                rows=torch.arange(batch,device=world.device)[:,None]
                start=a[rows,chosen];direction=delta[rows,chosen]
                denominator=length2.gather(1,chosen)
            coordinates=(world.detach()-origin)/grid.cell_size_m
            cells=_closed_grid_cells(coordinates)
            inside=(cells>=0).all(-1)&(cells<shape).all(-1)
            clipped=torch.stack((cells[...,0].clamp(0,free.shape[0]-1),cells[...,1].clamp(0,free.shape[1]-1)),-1)
            navigable=(inside&free[clipped[...,0],clipped[...,1]]).all(-1)
        fraction=(((world-start)*direction).sum(-1)/denominator).clamp(0,1)
        distance=(world-start-fraction[...,None]*direction).norm(dim=-1)
        return torch.where(navigable,distance,-distance)

    @classmethod
    def from_paths(cls, paths: tuple[Path, ...]) -> "SourceConfigurationSpaceQuery":
        return cls(tuple(SourceConfigurationGrid.load(path) for path in paths))

    def _atlas(self, device: torch.device) -> tuple[Tensor, ...]:
        """Pack immutable maps once; all scenes share one batched lookup."""
        if device not in self._atlases:
            sizes = np.array([grid.signed_clearance_m.shape for grid in self.grids], dtype=np.int64)
            offsets = np.r_[0, np.prod(sizes, axis=1).cumsum()[:-1]]
            self._atlases[device] = tuple(torch.as_tensor(value, device=device) for value in (
                np.concatenate([grid.signed_clearance_m.ravel() for grid in self.grids]),
                np.stack([grid.origin_xy for grid in self.grids]),
                np.array([grid.cell_size_m for grid in self.grids], dtype=np.float64),
                sizes, offsets,
                np.array([1. if grid.coordinate_frame == "canonical_world_xy" else -1. for grid in self.grids]),
            ))
        return self._atlases[device]

    def grid_coordinates(
        self, local: Tensor, indices: Tensor, origin: Tensor, yaw: Tensor,
    ) -> Tensor:
        """One FP64 robot-to-native-grid transform, outside neural autocast."""
        _, grid_origins, cell_sizes, _, _, lateral_sign = self._atlas(local.device)
        local, origin, yaw = local.double(), origin.double(), yaw.double()
        cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
        lateral = local[..., 1] * lateral_sign[indices, None]
        world = torch.stack((
            origin[:, None, 0] + cosine * local[..., 0] - sine * lateral,
            origin[:, None, 1] + sine * local[..., 0] + cosine * lateral,
        ), -1)
        return (world - grid_origins[indices, None]) / cell_sizes[indices, None, None]

    def point_cells(self, local: Tensor, indices: Tensor, origin: Tensor, yaw: Tensor) -> Tensor:
        return _closed_grid_cells(self.grid_coordinates(local, indices, origin, yaw))[..., 0, :]

    @staticmethod
    def _trace_segments(path: Tensor, grid_path: Tensor, horizon: float) -> tuple[Tensor, Tensor, Tensor]:
        """Sample every cell interior and boundary crossed, preserving vertices.

        Grid-boundary crossings partition each original segment into intervals
        lying in a single cell. Their midpoints cannot miss a short corner cut.
        Metric subdivisions also bound the spacing of visibility diagnostics.
        """
        delta = path[:, 1:].double() - path[:, :-1].double()
        length = delta.norm(dim=-1)
        start_arc = length.cumsum(-1) - length
        fraction = ((horizon - start_arc).clamp_min(0) / length.clamp_min(1e-30)).clamp_max(1)
        end = path[:, :-1].double() + fraction[..., None] * delta
        grid_start = grid_path[:, :-1]
        grid_end = grid_start + fraction[..., None] * (grid_path[:, 1:] - grid_start)
        grid_delta = grid_end - grid_start
        used_length = length * fraction
        events = [torch.zeros_like(length[..., None]), torch.ones_like(length[..., None])]
        for axis in range(2):
            low = torch.minimum(grid_start[..., axis], grid_end[..., axis]).floor()
            high = torch.maximum(grid_start[..., axis], grid_end[..., axis]).floor()
            count = int((high - low).max().item())
            boundaries = low[..., None] + torch.arange(1, count + 1, device=path.device)
            direction = grid_delta[..., axis, None]
            t = (boundaries - grid_start[..., axis, None]) / torch.where(direction != 0, direction, 1)
            events.append(torch.where((direction != 0) & (t > 0) & (t < 1), t, 1))
        spacing = SOURCE_CONFIGURATION_QUERY_SPACING_M
        count = int(torch.ceil(used_length.max() / spacing).item())
        distances = torch.arange(max(0, count - 1), device=path.device).add(1) * spacing
        events.append((distances / used_length[..., None].clamp_min(1e-30)).clamp_max(1))
        cuts = torch.cat(events, -1).sort(-1).values
        left, right = cuts[..., :-1], cuts[..., 1:]
        # Midpoint covers each crossed cell; the right endpoint also checks
        # isolated contacts at grid corners and retains every original vertex.
        t = torch.stack(((left + right) * 0.5, right), -1).flatten(-2)
        local = (path[:, :-1, None].double()
                 + t[..., None] * (end - path[:, :-1])[..., None, :]).flatten(1, 2)
        coordinates = (grid_start[..., None, :] + t[..., None] * grid_delta[..., None, :]).flatten(1, 2)
        valid = ((right > left) & (used_length[..., None] > 0)).repeat_interleave(2, -1).flatten(1, 2)
        local = torch.cat((path[:, :1].double(), local), 1)
        coordinates = torch.cat((grid_path[:, :1], coordinates), 1)
        valid = torch.cat((torch.ones_like(valid[:, :1]), valid), 1)
        # Stable compaction keeps active samples contiguous and in execution order.
        size = int(valid.sum(-1).max().item())
        order = (~valid).to(torch.int32).argsort(dim=-1, stable=True)[:, :size]
        active = valid.gather(1, order)
        local = local.gather(1, order[..., None].expand(-1, -1, 2))
        coordinates = coordinates.gather(1, order[..., None].expand(-1, -1, 2))
        last = (active.sum(-1) - 1)[:, None, None].expand(-1, 1, 2)
        local = torch.where(active[..., None], local, local.gather(1, last))
        coordinates = torch.where(active[..., None], coordinates, coordinates.gather(1, last))
        return local.float(), coordinates, active

    def query(
        self, path: Tensor, grid_index: Tensor, world_origin_xy: Tensor,
        world_yaw_rad: Tensor, planning_horizon_m: float,
    ) -> SourcePathQuery:
        """Trace the polyline against closed occupied cells, including corner contact."""
        indices = grid_index.long()
        coordinates = self.grid_coordinates(path, indices, world_origin_xy, world_yaw_rad)
        local, coordinates, active = self._trace_segments(path, coordinates, planning_horizon_m)
        cells = _closed_grid_cells(coordinates)
        values, _, _, sizes, offsets, _ = self._atlas(path.device)
        shape = sizes[indices, None, None]
        inside = (cells >= 0).all(-1) & (cells < shape).all(-1)
        address = offsets[indices, None, None] + cells[..., 0] * shape[..., 1] + cells[..., 1]
        # Select a valid address before the gather; out-of-map values remain unsafe.
        clearance = values[torch.where(inside, address, 0)].masked_fill(~inside, -planning_horizon_m)
        # Clearance covers every touching cell; progress uses the canonical cell
        # only after the common collision-free prefix has been established.
        return SourcePathQuery(local, clearance.amin(-1), inside.all(-1), active, cells[..., 0, :])

    def safety_metrics(
        self,
        path: Tensor,
        grid_index: Tensor,
        world_origin_xy: Tensor,
        world_yaw_rad: Tensor,
        planning_horizon_m: float,
    ) -> dict[str, Tensor]:
        """Use the source oracle as the sole physical safety metric."""
        query = self.query(
            path,
            grid_index,
            world_origin_xy,
            world_yaw_rad,
            planning_horizon_m,
        )
        return self.safety_metrics_from_query(path, query, planning_horizon_m)

    @staticmethod
    def safety_metrics_from_query(
        path: Tensor,
        query: SourcePathQuery,
        planning_horizon_m: float,
    ) -> dict[str, Tensor]:
        """Summarize one previously sampled source-C-space trajectory."""
        minimum = query.minimum_clearance_m
        return {
            "path_field_coverage_fraction": query.path_field_coverage_fraction,
            "min_clearance_m": minimum,
            "footprint_collision": minimum < 0.0,
            "safety_margin_violation": minimum < EXTRA_CLEARANCE_M,
            "max_margin_violation_m": torch.relu(EXTRA_CLEARANCE_M - minimum),
            "arc_length_beyond_local_horizon_m": torch.relu(
                path_arc_length(path.float()) - planning_horizon_m
            ),
        }
