"""Transformer blocks for trajectory--condition interaction."""

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from kappanav.local_planner.model.layers import RMSNorm, SwiGLU
from kappanav.local_planner.model.types import ConditionFeatures


ProjectedCondition = tuple[Tensor, Tensor, Tensor]
_SAVED_BIAS_BYTES = 512 << 20
_BIAS_CHUNK_BYTES = 512 << 20


def _silu_value_and_grad(value, gradient):
    work = value if value.dtype == torch.float64 else value.float()
    sigmoid = work.sigmoid()
    return ((work * sigmoid).to(value.dtype),
            (gradient.to(work.dtype) * sigmoid * (1 + work * (1 - sigmoid))).to(value.dtype))


class _GeometryCast(torch.autograd.Function):
    """Share neural data while each block returns gradients in the input precision."""

    @staticmethod
    def forward(ctx, geometry, neural):
        ctx.geometry_dtype = geometry.dtype
        return neural.view_as(neural)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to(ctx.geometry_dtype), None


class _SiluLinear(torch.autograd.Function):
    """Save pre-SiLU values only; reconstruct value and derivative together."""

    @staticmethod
    def forward(ctx, value, weight, bias):
        ctx.save_for_backward(value, weight)
        ctx.bias_dtype = bias.dtype
        return functional.linear(functional.silu(value), weight.to(value.dtype), bias.to(value.dtype))

    @staticmethod
    def backward(ctx, gradient):
        value, weight = ctx.saved_tensors
        flat = gradient.reshape(-1, gradient.shape[-1])
        grad_hidden = flat @ weight.to(flat.dtype)
        hidden, grad_value = _silu_value_and_grad(value.reshape(-1, value.shape[-1]), grad_hidden)
        grad_weight = (flat.T @ hidden).to(weight.dtype)
        return grad_value.reshape_as(value), grad_weight, flat.sum(0).to(ctx.bias_dtype)


class ReusableConditionCrossAttention(nn.Module):
    """Cross-attention whose condition K/V projection is reusable."""

    def __init__(
        self,
        model_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed_dim = model_dim
        self.num_heads = heads
        self.head_dim = model_dim // heads
        self.dropout = float(dropout)
        self.saved_bias_bytes = _SAVED_BIAS_BYTES
        self.in_proj_weight = nn.Parameter(torch.empty(3 * model_dim, model_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * model_dim))
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.relative_bias = nn.Sequential(
            nn.Linear(7, model_dim // 4),
            nn.SiLU(),
            nn.Linear(model_dim // 4, heads),
        )
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def project_condition(self, condition: ConditionFeatures) -> ProjectedCondition:
        tokens = condition.tokens
        token_valid = condition.token_valid
        projected = functional.linear(
            tokens,
            self.in_proj_weight[self.embed_dim :],
            self.in_proj_bias[self.embed_dim :],
        )
        key, value = projected.unflatten(-1, (2, self.embed_dim)).unbind(-2)
        key = key.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        return key, value, token_valid[:, None, None]

    def forward(
        self,
        query: Tensor,
        projected_condition: ProjectedCondition,
        pair_geometry: Tensor,
    ) -> Tensor:
        projected_query = functional.linear(
            query,
            self.in_proj_weight[: self.embed_dim],
            self.in_proj_bias[: self.embed_dim],
        )
        projected_query = projected_query.unflatten(
            -1, (self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key, value, token_valid = projected_condition
        batch = key.shape[0]
        candidates = query.shape[0] // batch
        projected_query = projected_query.unflatten(0, (batch, candidates)).transpose(1, 2).flatten(2, 3)
        geometry = pair_geometry.unflatten(0, (batch, candidates)).flatten(1, 2)
        pieces = []
        # Save small bias activations; retain bounded recomputation for large banks.
        # Fused BF16 saves the 7D input and head mask, not the retired hidden activation.
        saved_width = (7 + self.num_heads if geometry.is_cuda and projected_query.dtype == torch.bfloat16
                       else self.relative_bias[0].out_features)
        bias_bytes = geometry.numel() // 7 * saved_width * projected_query.element_size()
        recompute = bias_bytes > self.saved_bias_bytes
        chunk = 128
        # Grow training chunks within the target; retain inference/dropout partitions.
        if torch.is_grad_enabled() and (not self.training or self.dropout == 0):
            per_query = batch * key.shape[-2] * self.relative_bias[0].out_features * projected_query.element_size()
            chunk = max(128, min(512, (_BIAS_CHUNK_BYTES // per_query) // 32 * 32))
        for queries, pairs in zip(projected_query.split(chunk, dim=2), geometry.split(chunk, dim=1), strict=True):
            if torch.is_grad_enabled() and recompute:
                piece = checkpoint(self._attend, queries, key, value, pairs, token_valid,
                    use_reentrant=False, preserve_rng_state=self.training and self.dropout > 0)
            else:
                piece = self._attend(queries, key, value, pairs, token_valid,
                    recompute_silu=torch.is_grad_enabled() and bias_bytes > _SAVED_BIAS_BYTES)
            pieces.append(piece)
        attended = torch.cat(pieces, dim=2)
        attended = attended.unflatten(2, (candidates, query.shape[1])).transpose(1, 2).flatten(0, 1)
        attended = attended.transpose(1, 2).flatten(-2)
        return self.out_proj(attended)

    def _attend(self, query, key, value, geometry, token_valid, *, recompute_silu=False):
        geometry = geometry.to(dtype=query.dtype)
        if geometry.is_cuda and geometry.dtype == torch.bfloat16:
            from kappanav.local_planner.model.network._geometry_bias import geometric_bias
            bias = geometric_bias(geometry.contiguous(), self.relative_bias[0].weight, self.relative_bias[0].bias,
                self.relative_bias[2].weight, self.relative_bias[2].bias)
        elif recompute_silu:
            preact = self.relative_bias[0](geometry)
            bias = _SiluLinear.apply(preact, self.relative_bias[2].weight, self.relative_bias[2].bias)
        else:
            bias = self.relative_bias(geometry)
        attention_mask = bias.permute(0, 3, 1, 2)
        attention_mask = attention_mask.masked_fill(~token_valid, float("-inf"))
        dropout_p=self.dropout if self.training else 0.0
        return functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=dropout_p,
        )


class ConditionalTrajectoryBlock(nn.Module):
    """Update trajectory tokens only through self- and metric scene attention."""

    def __init__(
        self,
        model_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_norm = RMSNorm(model_dim)
        self.self_attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.query_norm = RMSNorm(model_dim)
        self.memory_norm = RMSNorm(model_dim)
        self.cross_attention = ReusableConditionCrossAttention(
            model_dim,
            heads,
            dropout,
        )
        self.feed_forward_norm = RMSNorm(model_dim)
        self.feed_forward = SwiGLU(model_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trajectory: Tensor,
        projected_condition: ProjectedCondition,
        pair_geometry: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(trajectory)
        attended = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        trajectory = trajectory + self.dropout(attended)

        attended = self.cross_attention(
            self.query_norm(trajectory),
            projected_condition,
            pair_geometry,
        )
        trajectory = trajectory + self.dropout(attended)

        normalized = self.feed_forward_norm(trajectory)
        return trajectory + self.dropout(self.feed_forward(normalized))

    def project_condition(self, condition: ConditionFeatures) -> ProjectedCondition:
        return self.cross_attention.project_condition(
            ConditionFeatures(
                tokens=self.memory_norm(condition.tokens),
                token_valid=condition.token_valid,
                metric_position=condition.metric_position,
                surface_hit=condition.surface_hit,
                frame_age=condition.frame_age,
                motion_token=condition.motion_token,
                terminal_goal=condition.terminal_goal,
                goal_present=condition.goal_present,
                configuration_field=condition.configuration_field,
            )
        )
