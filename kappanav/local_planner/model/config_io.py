"""YAML-to-dataclass configuration loading and serialization."""

from dataclasses import asdict
from typing import Any, Mapping

from kappanav.local_planner.model.config import (CurveNavConfig, DataConfig, DepthEncoderConfig,
                                                 TrajectoryDecoderConfig, ConditionEncoderConfig, TrainingConfig,
                                                 TrajectoryConfig)


def config_from_mapping(raw: Mapping[str, Any]) -> CurveNavConfig:
    model = raw.get("model", {})
    trajectory = dict(model.get("trajectory", {}))
    for name in (
        "control_increment_mean_xy_m",
        "control_increment_std_xy_m",
    ):
        if name in trajectory:
            trajectory[name] = tuple(trajectory[name])
    config = CurveNavConfig(
        data=DataConfig(**raw.get("data", {})),
        trajectory=TrajectoryConfig(**trajectory),
        depth_encoder=DepthEncoderConfig(**model.get("depth_encoder", {})),
        condition_encoder=ConditionEncoderConfig(**model.get("condition_encoder", {})),
        trajectory_decoder=TrajectoryDecoderConfig(
            **model.get("trajectory_decoder", {})
        ),
        training=TrainingConfig(**raw.get("training", {})),
    )
    return config


def config_to_mapping(config: CurveNavConfig) -> dict:
    """Serialize the validated model and effective training settings in the YAML schema."""
    raw=asdict(config)
    return dict(data=raw.pop('data'),training=raw.pop('training'),model=raw)
