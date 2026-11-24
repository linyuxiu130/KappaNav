"""Planar cubic B-splines in standardized physical increment space."""

import torch
from torch import Tensor, nn

from kappanav.local_planner.model.precision import GEOMETRY_DTYPE

from kappanav.local_planner.model.trajectory.basis import bspline_basis_and_first_derivative


INCREMENTAL_CONTROL_PARAMETERIZATION_TYPE = (
    "standardized_physical_bspline_control_increments"
)
BSPLINE_CONTROL_POINTS = 8
BSPLINE_DEGREE = 3

def local_terminal_goal(point_goal: Tensor, planning_horizon_m: float) -> Tensor:
    """Clip PointGoal to the local terminal position without changing its direction."""
    point_goal = point_goal.float()
    distance = torch.linalg.vector_norm(point_goal, dim=-1, keepdim=True)
    direction = point_goal / distance.clamp_min(torch.finfo(point_goal.dtype).eps)
    endpoint = direction * distance.clamp_max(planning_horizon_m)
    return endpoint


class IncrementalBSplineTrajectory(nn.Module):
    """Decode standardized physical increments into one clamped cubic B-spline."""

    def __init__(
        self,
        num_control_points: int,
        degree: int,
        num_path_points: int,
        control_increment_mean_xy_m: tuple[float, ...],
        control_increment_std_xy_m: tuple[float, ...],
    ) -> None:
        super().__init__()
        self.num_control_tokens = num_control_points - 1
        self.coordinate_dim = 2 * self.num_control_tokens

        self.num_control_points = num_control_points
        self.degree = degree
        self.num_path_points = num_path_points
        basis, first_basis = bspline_basis_and_first_derivative(
            num_control_points, degree, num_path_points
        )
        learned_basis = basis[:, 2:-1]
        fit_inverse = torch.linalg.solve(
            learned_basis.T @ learned_basis,
            learned_basis.T,
        )
        self.register_buffer("basis", basis, persistent=True)
        # dp(s)/d(delta P_i) = sum_{j>=i} B_j(s). The fixed origin has no token.
        self.register_buffer(
            "increment_basis", basis[:, 1:].flip(1).cumsum(1).flip(1), persistent=True
        )
        self.register_buffer("first_basis", first_basis, persistent=True)
        self.register_buffer("fit_inverse", fit_inverse, persistent=True)
        self.register_buffer(
            "control_increment_mean_xy_m",
            torch.tensor(control_increment_mean_xy_m),
            persistent=True,
        )
        self.register_buffer(
            "control_increment_std_xy_m",
            torch.tensor(control_increment_std_xy_m),
            persistent=True,
        )

    @staticmethod
    def _increments(values: Tensor) -> Tensor:
        controls = values.float().reshape(values.shape[0], -1, 2)
        origin = torch.zeros(
            values.shape[0], 1, 2, device=values.device, dtype=GEOMETRY_DTYPE
        )
        return torch.diff(torch.cat((origin, controls), dim=1), dim=1)

    def values_from_coordinates(
        self,
        coordinates: Tensor,
    ) -> Tensor:
        increments = self.control_increment_mean_xy_m + (
            self.control_increment_std_xy_m * coordinates.float()
        )
        controls = increments.reshape(coordinates.shape[0], -1, 2).cumsum(dim=1)
        return controls.flatten(1)

    def coordinates_from_values(
        self,
        values: Tensor,
    ) -> Tensor:
        increments = self._increments(values).flatten(1)
        return (
            increments - self.control_increment_mean_xy_m
        ) / self.control_increment_std_xy_m

    def _controls_from_values(self, values: Tensor) -> Tensor:
        learned = values.float().reshape(values.shape[0], -1, 2)
        origin = torch.zeros(
            values.shape[0], 1, 2, device=values.device, dtype=GEOMETRY_DTYPE
        )
        return torch.cat((origin, learned), dim=1)

    def decode_values(self, values: Tensor) -> tuple[Tensor, Tensor]:
        controls = self._controls_from_values(values)
        path = torch.einsum("pc,bcd->bpd", self.basis, controls)
        tangent = torch.einsum("pc,bcd->bpd", self.first_basis, controls)
        heading = torch.atan2(tangent[..., 1], tangent[..., 0])
        return path, heading

    def decode_path(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.decode_values(self.values_from_coordinates(coordinates))

    def decode(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.decode_path(coordinates)

    @torch.no_grad()
    def project_expert(self, reference_path: Tensor) -> tuple[Tensor, Tensor]:
        reference_path = reference_path.float()
        # Arc-length parameterization gives p'(0)=(length,0) in the body frame.
        # For the clamped cubic, p'(0)=15*P1. Fix this boundary and the endpoint
        # before least squares, rather than clamping a fitted backwards tangent.
        length = torch.linalg.vector_norm(torch.diff(reference_path, dim=1), dim=-1).sum(1)
        first = torch.stack((length / 15.0, torch.zeros_like(length)), dim=1)
        last = reference_path[:, -1]
        residual = reference_path - self.basis[:, 1][None, :, None] * first[:, None]
        residual = residual - self.basis[:, -1][None, :, None] * last[:, None]
        interior = torch.einsum("cp,bpd->bcd", self.fit_inverse, residual)
        values = torch.cat((first[:, None], interior, last[:, None]), dim=1).flatten(1)
        path, _ = self.decode_values(values)
        return values, path

    def forward(
        self,
        coordinates: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.decode(coordinates)


def project_reference_tangent(codec, reference):
    """Reuse the native fit in the reference tangent frame; preserve its endpoint.

    Labels may require turning in place before translation. This does not add
    that action to the controller or establish closed-loop executability.
    """
    tangent = torch.nn.functional.normalize(reference[:, 1] - reference[:, 0], dim=-1)
    rotation = torch.stack((tangent, torch.stack((-tangent[:, 1], tangent[:, 0]), -1)), -1)
    aligned = torch.einsum('bpi,bij->bpj', reference, rotation)
    values, _ = codec.project_expert(aligned)
    values = torch.einsum('bpj,bij->bpi', values.reshape(-1, 7, 2), rotation).flatten(1)
    return values, codec.decode_values(values)[0]
