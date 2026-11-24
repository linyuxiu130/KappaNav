"""Exponential moving average with one fused update over model parameters."""

from contextlib import contextmanager
from collections.abc import Iterator

import torch
from torch import nn


class ExponentialMovingAverage:
    """EMA of every trainable parameter with the warmup decay min(decay, (1+n)/(10+n)) after n updates."""
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        named_parameters = tuple(
            (name, parameter.detach())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self.decay = decay
        self.num_updates = 0
        self._parameter_names = tuple(name for name, _ in named_parameters)
        self._parameters = tuple(parameter for _, parameter in named_parameters)
        self.shadow = {
            name: parameter.clone() for name, parameter in named_parameters
        }

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates += 1
        effective_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        torch._foreach_lerp_(
            tuple(self.shadow.values()),
            self._parameters,
            1.0 - effective_decay,
        )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        named_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        for name, parameter in named_parameters:
            parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))

    @contextmanager
    @torch.no_grad()
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily expose the EMA policy without changing optimizer state."""
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        saved = tuple(parameter.detach().clone() for parameter in parameters)
        self.copy_to(model)
        try:
            yield
        finally:
            torch._foreach_copy_(parameters, saved)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore by parameter name, in this model's order (update() pairs shadow values with parameters by position)."""
        shadow = state["shadow"]
        if set(shadow) != set(self._parameter_names):
            raise ValueError("EMA parameters differ from the saved state")
        self.decay = float(state["decay"])
        self.num_updates = int(state["num_updates"])
        self.shadow = {
            name: shadow[name].detach().to(device=parameter.device, dtype=parameter.dtype).clone()
            for name, parameter in zip(self._parameter_names, self._parameters, strict=True)
        }
