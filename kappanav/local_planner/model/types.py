"""Tensor contracts shared across CurveNav modules."""

from dataclasses import dataclass
from torch import Tensor


@dataclass
class PolicyCondition:
    """Robot-centric observations used to condition a local trajectory."""

    depth: Tensor
    point_goal: Tensor
    camera_intrinsics: Tensor
    camera_to_body: Tensor
    observation_to_current: Tensor
    observation_age_s: Tensor
    observation_valid: Tensor
    obstacle_memory: Tensor


@dataclass
class ConfigurationFeatures:
    """One target-independent metric BEV memory."""

    tokens: Tensor
    metric_position: Tensor
    observed_fraction: Tensor


@dataclass
class ConditionFeatures:
    """Current scene and selected historical features in the body frame."""

    tokens: Tensor
    token_valid: Tensor
    metric_position: Tensor
    surface_hit: Tensor  # Historical name:observed support for BEV/fine tokens,not hard occupancy.
    frame_age: Tensor
    motion_token: Tensor
    terminal_goal: Tensor
    goal_present: Tensor
    configuration_field: Tensor
