"""Typed configuration for the single CurveNav policy contract."""

from dataclasses import dataclass, field as dataclass_field

from kappanav.common.physical import ROBOT, ROBOT_FOOTPRINT_RADIUS_M, BODY_OBSTACLE_MIN_Z_M, ROBOT_COLLISION_TOP_Z_M


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/dataset"
    observation_frames: int = 4
    expert_waypoint_spacing_m: float = 0.15
    future_steps: int = 24
    image_height: int = 126
    image_width: int = 224
    max_depth_m: float = 5.0
    camera_forward_offset_m: float = ROBOT.camera_forward_m
    camera_height_m: float = ROBOT.camera_height_m
    camera_downward_pitch_degrees: float = ROBOT.camera_downward_pitch_degrees

    embodiment: str = ROBOT.name
    robot_radius_m: float = ROBOT_FOOTPRINT_RADIUS_M
    obstacle_min_z_m: float = BODY_OBSTACLE_MIN_Z_M
    obstacle_max_z_m: float = ROBOT_COLLISION_TOP_Z_M

    @property
    def robot_geometry(self):
        return (self.robot_radius_m, self.obstacle_min_z_m, self.obstacle_max_z_m)


@dataclass(frozen=True)
class TrajectoryConfig:
    num_control_points: int = 8
    spline_degree: int = 3
    num_path_points: int = 64
    # Untrained placeholders;dataset preparation supplies the qualified training statistics.
    control_increment_mean_xy_m: tuple[float, ...] = (0.0,) * 14
    control_increment_std_xy_m: tuple[float, ...] = (1.0,) * 14


@dataclass(frozen=True)
class DepthEncoderConfig:
    model_dim: int = 384
    frame_tokens_height: int = 8
    frame_tokens_width: int = 12
    dropout: float = 0.0
    geometry_context_m: float = 12.0


@dataclass(frozen=True)
class ConditionEncoderConfig:
    model_dim: int = 384
    bev_grid_size: int = 16


@dataclass(frozen=True)
class TrajectoryDecoderConfig:
    model_dim: int = 384
    transformer_layers: int = 12
    integration_steps: int = 2
    transformer_heads: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 4
    warmup_epochs: int = 1
    min_learning_rate_factor: float = 0.01
    checkpoint_every_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    ema_decay: float = 0.9999
    geometry_batch_size: int = 32


@dataclass(frozen=True)
class CurveNavConfig:
    data: DataConfig = dataclass_field(default_factory=DataConfig)
    trajectory: TrajectoryConfig = dataclass_field(default_factory=TrajectoryConfig)
    depth_encoder: DepthEncoderConfig = dataclass_field(
        default_factory=DepthEncoderConfig
    )
    condition_encoder: ConditionEncoderConfig = dataclass_field(
        default_factory=ConditionEncoderConfig
    )
    trajectory_decoder: TrajectoryDecoderConfig = dataclass_field(
        default_factory=TrajectoryDecoderConfig
    )
    training: TrainingConfig = dataclass_field(default_factory=TrainingConfig)
