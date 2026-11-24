"""PointGoal-conditioned Flow Matching in physical curve coordinates."""

from dataclasses import fields, replace

import torch
from torch import Tensor, nn

from kappanav.local_planner.model.trajectory import IncrementalBSplineTrajectory
from kappanav.local_planner.model.types import ConditionFeatures


INFERENCE_SOURCE_SEED = 20_260_828
INFERENCE_CANDIDATES = 16
INTEGRATION_STEPS = 2


def repeat_condition(condition: ConditionFeatures, count: int) -> ConditionFeatures:
    # Projected K/V and the raster are shared; only query geometry/intent repeats.
    return ConditionFeatures(**{
        f.name: (getattr(condition,f.name) if f.name in ('configuration_field','tokens','token_valid') else
            getattr(condition, f.name).repeat_interleave(count, dim=0))
        for f in fields(condition)
    })


def generate_candidate_bank(policy, condition, *, memory=None, source=None):
    """Sixteen curves in INTEGRATION_STEPS jumps of size 1/INTEGRATION_STEPS.

    Every call is told its step size, so the sampler is its own function rather
    than the step-0 CFM velocity it shares weights with. Training passes fresh
    `source` noise; inference uses the fixed bank.
    """
    count = INFERENCE_CANDIDATES
    present = condition.goal_present.bool()
    condition = replace(condition, terminal_goal=torch.where(
        present, condition.terminal_goal, torch.zeros_like(condition.terminal_goal)))
    if source is None:
        source = policy.inference_source.expand(len(present), -1, -1)
    if memory is None:memory = policy.trajectory_decoder.project_condition_memory(condition)
    queries = repeat_condition(condition, count)
    step = 1 / INTEGRATION_STEPS
    state = source.flatten(0, 1)
    for index in range(INTEGRATION_STEPS):
        time = torch.full((len(state),), 1 - index*step, device=state.device)
        state = state - step*policy._predict_velocity(state, time, queries, memory, step)
    values = policy.curve_codec.values_from_coordinates(state)
    paths, tangent = policy.curve_codec.decode_values(values)
    shape = source.shape[:2]
    return dict(values=values.unflatten(0, shape), paths=paths.unflatten(0, shape), tangent=tangent.unflatten(0, shape))


class CurveNavPolicy(nn.Module):
    """Generate sixteen native curves and score their actual geometry."""

    def __init__(
        self,
        depth_encoder: nn.Module,
        condition_encoder: nn.Module,
        trajectory_decoder: nn.Module,
        trajectory_evaluator: nn.Module,
        curve_codec: IncrementalBSplineTrajectory,
        planning_horizon_m: float,
        integration_steps: int,
    ) -> None:
        super().__init__()
        self.depth_encoder = depth_encoder
        self.condition_encoder = condition_encoder
        self.trajectory_decoder = trajectory_decoder
        self.trajectory_evaluator = trajectory_evaluator
        self.curve_codec = curve_codec
        self._planning_horizon_m = float(planning_horizon_m)
        self.integration_steps = integration_steps
        generator = torch.Generator(device="cpu")
        generator.manual_seed(INFERENCE_SOURCE_SEED)
        # One fixed bank for both intents: training draws fresh noise, so the
        # learned map is valid for any draw and needs no per-intent slice.
        inference_source = torch.randn(
            INFERENCE_CANDIDATES,
            curve_codec.coordinate_dim,
            generator=generator,
        )
        self.register_buffer(
            "inference_source",
            inference_source,
            persistent=True,
        )

    @property
    def planning_horizon_m(self) -> float:
        return self._planning_horizon_m


    def _predict_velocity(
        self, state: Tensor, time: Tensor, condition: ConditionFeatures, memory, step: float = 0.
    ) -> Tensor:
        """Velocity of a jump of size `step` (0: the instantaneous CFM velocity)."""
        path, _ = self.curve_codec.decode(state)
        step = torch.full_like(time, step)
        return self.trajectory_decoder(state, time, step, path, condition, memory).float()


    def score_candidate_paths(
        self,
        encoded: ConditionFeatures,
        point_goal: Tensor,
        candidates: Tensor,
    ) -> Tensor:
        """Score a fixed candidate bank while keeping evaluator gradients."""
        batch, count = candidates.shape[:2]
        critic_memory = self.trajectory_evaluator.project_condition_memory(encoded)
        paths = candidates.flatten(0, 1)
        return self.trajectory_evaluator(
            paths,
            torch.where(encoded.goal_present.bool(), point_goal,
                torch.zeros_like(point_goal)).repeat_interleave(count, dim=0),
            repeat_condition(encoded, count),
            critic_memory,
        ).unflatten(0, (batch, count))
