"""Supervised route utility: geometry is the teacher, never an inference input."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from kappanav.local_planner.data.goal_distance import GoalDistanceQuery
from kappanav.local_planner.features.staging import stage
from kappanav.local_planner.data.privileged import SourceConfigurationSpaceQuery, SourcePathQuery
from kappanav.common.physical import EXTRA_CLEARANCE_M, ROBOT_FOOTPRINT_RADIUS_M
from kappanav.local_planner.control.mpc import (MAX_ANGULAR_SPEED_RAD_S, MAX_LINEAR_SPEED_MPS, TURN_ENTER_RAD,
                                                TURN_EXIT_RAD, TURN_LOOKAHEAD_M)
from kappanav.local_planner.model.trajectory import path_arc_length


# Contact-distance scale of the collision cost; not a supervision sampling interval.
EXECUTION_PREFIX_M = .25


@torch.no_grad()
def candidate_query(
    query: SourceConfigurationSpaceQuery,
    paths: Tensor,
    batch: dict[str, Tensor],
    planning_horizon_m: float,
) -> tuple[SourcePathQuery, Tensor]:
    count = paths.shape[1]
    flat = paths.flatten(0, 1).float()
    horizon = max(planning_horizon_m, path_arc_length(flat).max().item())
    result = query.query(
        flat,
        batch["source_grid_index"].repeat_interleave(count),
        batch["source_origin_xy"].repeat_interleave(count, dim=0),
        batch["source_yaw_rad"].repeat_interleave(count),
        horizon,
    )
    # OOB semantics must not depend on another candidate's length.
    clearance = result.clearance_m.masked_fill(~result.in_world_bounds, -planning_horizon_m)
    return result, clearance


@dataclass
class CandidateUtility:
    score: Tensor
    clearance_m: Tensor
    progress_m: Tensor
    failed: Tensor


def clearance_shortfall(clearance: Tensor, active: Tensor) -> Tensor:
    """Margin missing along a path, with the escape semantics of bspline_teachers.escape_clear.

    The required margin at a sample is the best clearance reached so far, capped at EXTRA_CLEARANCE_M, so a
    curve leaving contact is not charged for the start it cannot change. A path that never reaches the margin
    is charged what it still lacks at its best. From a start with clearance >= the margin this equals
    relu(EXTRA_CLEARANCE_M - min clearance), the plain margin penalty.
    """
    pad = EXTRA_CLEARANCE_M
    values = clearance.masked_fill(~active, -torch.inf)
    reached = torch.cummax(values, dim=-1).values.clamp(max=pad)
    dropped = (reached - clearance).masked_fill(~active, 0).amax(-1)
    unescaped = pad - values.amax(-1).clamp(max=pad)
    return torch.maximum(dropped, unescaped).clamp_min(0)


def in_place_turn_m(paths: Tensor) -> Tensor:
    """Driving distance the controller gives up turning on the spot before it follows each body-frame path.

    The rule of simulation.mpc: a path whose direction at TURN_LOOKAHEAD_M is TURN_ENTER_RAD or more off the
    heading is turned to within TURN_EXIT_RAD at MAX_ANGULAR_SPEED_RAD_S; that time is priced at
    MAX_LINEAR_SPEED_MPS. Without it a short path leaving backwards scores like a forward one, and a memoryless
    policy keeps turning toward a new backward stub after every turn.
    """
    paths = paths.float()
    arc = F.pad((paths[..., 1:, :] - paths[..., :-1, :]).norm(dim=-1).cumsum(-1), (1, 0))
    target = arc[..., -1:].clamp(max=TURN_LOOKAHEAD_M)
    upper = (arc < target).sum(-1, keepdim=True).clamp(1, arc.shape[-1] - 1)
    a0, a1 = arc.gather(-1, upper - 1), arc.gather(-1, upper)
    pick = lambda index: paths.gather(-2, index[..., None].expand(*index.shape, 2))
    p0, p1 = pick(upper - 1), pick(upper)
    ahead = p0 + ((target - a0) / (a1 - a0).clamp_min(1e-12))[..., None] * (p1 - p0) - paths[..., :1, :]
    error = torch.atan2(ahead[..., 1], ahead[..., 0]).abs()
    turn = torch.where((error >= TURN_ENTER_RAD) & (target > 1e-9), error - TURN_EXIT_RAD, torch.zeros_like(error))
    return (turn * (MAX_LINEAR_SPEED_MPS / MAX_ANGULAR_SPEED_RAD_S)).squeeze(-1)


class RouteUtilityTeacher:
    def __init__(self, source_query: SourceConfigurationSpaceQuery, planning_horizon_m: float, robot_radius_m: float = ROBOT_FOOTPRINT_RADIUS_M):
        self.robot_radius_m = robot_radius_m
        self.source_query = source_query
        self.horizon = planning_horizon_m
        self.goal_distance = GoalDistanceQuery(source_query)
    @torch.no_grad()
    def __call__(self, paths: Tensor, batch: dict[str, Tensor]) -> CandidateUtility:
        result, clearance = candidate_query(self.source_query, paths, batch, self.horizon)
        return self.score_paths(paths, batch, result, clearance)

    @torch.no_grad()
    def score_paths(self, paths: Tensor, batch: dict[str, Tensor],
                    result: SourcePathQuery, clearance: Tensor) -> CandidateUtility:
        """Score an existing trace, retaining the caller's sample width and reduction order."""
        b, count = paths.shape[:2]
        if 'source_metadata_cpu' in batch:
            indices=batch['source_metadata_cpu'][:,0].numpy().astype(np.int64)
        else:
            indices=batch['source_grid_index'].cpu().tolist()
        failure = (clearance < 0) & result.active
        # Geodesic progress still ends at a traversable grid cell, independently of physical contact.
        prefix = result.active & (failure.cumsum(-1) == 0)
        last = (prefix.sum(-1) - 1).clamp_min(0)
        rows = torch.arange(b * count, device=paths.device)
        endpoint_cells = result.grid_cells[rows, last].unflatten(0, (b, count))
        present = batch["goal_present"].bool().reshape(b)
        goal = torch.where(present[:, None], batch["point_goal"], torch.zeros_like(batch["point_goal"]))
        start_and_goal = torch.stack((torch.zeros_like(goal), goal), 1)
        fixed_cells = self.source_query.point_cells(
            start_and_goal, batch["source_grid_index"], batch["source_origin_xy"], batch["source_yaw_rad"],
        )
        cells = torch.cat((fixed_cells[:, :1], endpoint_cells, fixed_cells[:, 1:]), 1).cpu().numpy()
        distances_by_row=[]
        for distances in self.goal_distance.query_rows(indices,cells,parallel=paths.is_cuda):
            distances_by_row.append(distances)
        distances,=stage([torch.from_numpy(np.stack(distances_by_row))],paths.device)
        # Without a mission, reward traversable displacement from the start.
        progress=torch.where(present[:,None],distances[:,:1]-distances[:,1:],distances[:,1:])
        minimum = clearance.masked_fill(~result.active, torch.inf).amin(-1).unflatten(0, (b, count))
        contact = failure.any(-1).unflatten(0, (b, count)).float()
        length = path_arc_length(paths.flatten(0, 1).float()).unflatten(0, (b, count))
        sampled=result.local_path.unflatten(0,(b,count))
        segments=sampled[:,:,1:]-sampled[:,:,:-1]
        valid_segments=(prefix[:,1:]&prefix[:,:-1]).unflatten(0,(b,count))
        safe_length=segments.norm(dim=-1)*valid_segments
        # Distinguish an immediate collision from a later bad tail of the curve.
        contact_cost=contact*(1+torch.exp(-safe_length.sum(-1)/EXECUTION_PREFIX_M))
        if (~present).any():
            # Progress and novelty end at the same last physically/task-valid point.
            midpoint=(sampled[:,:,1:]+sampled[:,:,:-1])/(2*self.horizon)
            coverage=F.grid_sample(batch['observed_field'][:,3:4].float(),midpoint.flatten(1,2).unsqueeze(2),align_corners=True).reshape(b,count,-1)
            coverage_available=(midpoint.abs()<=1).all(-1)
            novelty=((1-coverage)*coverage_available*safe_length).sum(-1)/safe_length.sum(-1).clamp_min(1e-12)
            progress=progress*(1+(~present)[:,None]*novelty)
        # Time is charged as distance: the arc driven and the on-the-spot turn before it (in_place_turn_m).
        score = (
            (progress - 0.5 * (length + in_place_turn_m(paths))) / self.horizon
            - contact_cost
            - torch.asinh(clearance_shortfall(clearance, result.active).unflatten(0, (b, count)) / self.robot_radius_m)
        )
        return CandidateUtility(score, minimum, progress,failure.any(-1).unflatten(0,(b,count)))
