"""A single goal-conditioned critic over the complete decoded metric curve."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kappanav.local_planner.model.configuration_space import sample_configuration_features
from kappanav.local_planner.model.layers import RMSNorm
from kappanav.local_planner.model.network.blocks import ConditionalTrajectoryBlock, ProjectedCondition, _GeometryCast
from kappanav.local_planner.model.precision import NEURAL_DTYPE
from kappanav.local_planner.model.types import ConditionFeatures


EVALUATOR_LAYERS = 2
SEGMENTS = 8


class EvidenceTrajectoryEvaluator(nn.Module):
    def __init__(self, model_dim: int, heads: int, planning_horizon_m: float):
        super().__init__()
        self.neural_dtype = NEURAL_DTYPE
        self.horizon = planning_horizon_m
        self.represented_geometry = True
        # Position, observed geometry, tangent, arc progress, and mission goal.
        self.input_embedding = nn.Sequential(
            nn.Linear(15, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.blocks = nn.ModuleList(
            ConditionalTrajectoryBlock(model_dim, heads, 0.0) for _ in range(EVALUATOR_LAYERS)
        )
        self.norm = RMSNorm(model_dim)
        self.pool_query = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.pool_query, std=model_dim ** -0.5)
        self.readout = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, 1)
        )

    def pool_tokens(self, tokens: Tensor) -> Tensor:
        weights = (tokens * self.pool_query).sum(-1).softmax(-1)
        pooled = (weights[..., None] * tokens).sum(1)
        return pooled

    def project_condition_memory(self, condition: ConditionFeatures) -> tuple[ProjectedCondition, ...]:
        with torch.autocast(device_type=condition.tokens.device.type, dtype=self.neural_dtype):
            return tuple(block.project_condition(condition) for block in self.blocks)

    def forward(
        self,
        path: Tensor,
        goal: Tensor,
        condition: ConditionFeatures,
        memory: tuple[ProjectedCondition, ...],
    ) -> Tensor:
        path = path.float()
        segments = path[:, 1:] - path[:, :-1]
        length = segments.norm(dim=-1)
        tangent = F.normalize(segments, dim=-1)
        tangent = torch.cat((tangent[:, :1], tangent), dim=1)
        arc_m = F.pad(length.cumsum(-1), (1, 0))
        arc = arc_m / self.horizon
        goal_length = goal.float().norm(dim=-1, keepdim=True)
        mission = torch.cat(
            (F.normalize(goal.float(), dim=-1), torch.log1p(goal_length / self.horizon), condition.goal_present.float()),
            dim=-1,
        )
        geometry = sample_configuration_features(
            condition.configuration_field, path, self.horizon,represented_geometry=self.represented_geometry
        )
        observed = geometry[..., :5]
        features = torch.cat((path / self.horizon, observed[..., :1] / self.horizon,
                              observed[..., 1:], geometry[..., 5:6], tangent, arc[..., None],
                              mission[:, None].expand(-1, path.shape[1], -1)), -1)
        # One token per segment of consecutive samples: ranking is as good as with one token per
        # sample (Q study: 4-64 tokens within noise) at 1/8 of the pairwise-bias and attention cost.
        # Like G's increments, pool the per-sample offset and distance so a token keeps its extent.
        segment = lambda x: x.unflatten(1, (SEGMENTS, -1)).mean(2)
        relative = (condition.metric_position[:, None, :, :2].float() - path[:, :, None]) / self.horizon
        distance = segment(relative.norm(dim=-1, keepdim=True))
        relative = segment(relative)
        shape = (*relative.shape[:-1], 1)
        pair = torch.cat((
            relative,
            condition.metric_position[:, None, :, 2:3].float().expand(shape) / self.horizon,
            distance,
            condition.surface_hit[:, None, :, None].expand(shape).float(),
            condition.frame_age[:, None, :, None].expand(shape).float(),
            condition.motion_token[:, None, :, None].expand(shape).float(),
        ), -1)
        with torch.autocast(device_type=path.device.type, dtype=self.neural_dtype):
            tokens = segment(self.input_embedding(features))
            neural_geometry = pair.detach().to(tokens.dtype)
            for block, projected in zip(self.blocks, memory, strict=True):
                tokens = block(tokens, projected, _GeometryCast.apply(pair, neural_geometry))
            tokens = self.norm(tokens).float()
        # Pool in FP32; the learned query can emphasize a local bottleneck.
        pooled = self.pool_tokens(tokens)
        with torch.autocast(device_type=path.device.type, dtype=self.neural_dtype):
            return self.readout(pooled).squeeze(-1).float()
