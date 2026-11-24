"""Past-frame tokens of the condition: frame age only (the policy is not conditioned on its own motion)."""

import torch
from torch import Tensor, nn
from kappanav.local_planner.model.precision import NEURAL_DTYPE


HISTORICAL_STATE_FEATURES = "metric_xyz_rotation_columns_time"


class HistoricalMotionEncoder(nn.Module):
    """One token per past observation slot [B,F-1,D]: its frame age, with the motion fixed to a stationary history.

    The pose features are kept (identity transforms) so trained weights load unchanged; `observation_to_current`
    only provides the shape. Missing history is excluded by the condition memory attention mask.
    """

    def __init__(
        self,
        *,
        observation_frames: int,
        translation_scale_m: float,
        model_dim: int,
    ) -> None:
        super().__init__()
        self.neural_dtype = NEURAL_DTYPE
        self.state_tokens = observation_frames - 1
        self.translation_scale_m = float(translation_scale_m)
        self.projection = nn.Sequential(
            nn.Linear(10, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.slot_embedding = nn.Parameter(torch.zeros(1, self.state_tokens, model_dim))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)

    def forward(
        self,
        observation_to_current: Tensor,
        observation_age_s: Tensor,
    ) -> Tensor:
        # The policy is not conditioned on its own past motion: an expert's smooth motion predicts its next label
        # (causal confusion), and at deployment the copied motion feeds back. Past poses keep their geometric role
        # in fusion and token positions; only this token's motion content is fixed to a stationary history, the age
        # stays real. The feature layout is unchanged so trained weights load as they are.
        past = torch.eye(observation_to_current.shape[-1], dtype=torch.float32, device=observation_to_current.device
            ).expand_as(observation_to_current[:, :-1])
        features = torch.cat(
            (
                past[..., :3, 3] / self.translation_scale_m,
                past[..., :3, :2].flatten(-2),
                observation_age_s[:, :-1, None],
            ),
            dim=-1,
        )
        with torch.autocast(device_type=features.device.type, dtype=self.neural_dtype):
            encoded = self.projection(features)
        return encoded + self.slot_embedding
