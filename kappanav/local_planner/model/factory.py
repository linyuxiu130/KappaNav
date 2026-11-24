"""Composition root for the one CurveNav policy graph."""

from kappanav.local_planner.model.config import CurveNavConfig
from kappanav.local_planner.model.conditioning import PolicyConditionEncoder
from kappanav.local_planner.model.encoders import DepthObservationEncoder
from kappanav.local_planner.model.network import ConditionalCurveFlowDecoder, CurveNavPolicy
from kappanav.local_planner.model.network.bspline_evidence_evaluator import EvidenceTrajectoryEvaluator
from kappanav.local_planner.model.trajectory import IncrementalBSplineTrajectory


def build_policy(config: CurveNavConfig) -> CurveNavPolicy:
    """Assemble the only production policy graph."""

    trajectory = config.trajectory
    depth = config.depth_encoder
    condition = config.condition_encoder
    decoder = config.trajectory_decoder
    planning_horizon_m = (
        config.data.future_steps * config.data.expert_waypoint_spacing_m
    )

    depth_encoder = DepthObservationEncoder(
        model_dim=depth.model_dim,
        frame_tokens_height=depth.frame_tokens_height,
        frame_tokens_width=depth.frame_tokens_width,
        dropout=depth.dropout,
        max_depth_m=config.data.max_depth_m,
        planning_horizon_m=planning_horizon_m,
        geometry=config.data.robot_geometry,
    )
    condition_encoder = PolicyConditionEncoder(
        observation_frames=config.data.observation_frames,
        planning_horizon_m=planning_horizon_m,
        model_dim=condition.model_dim,
    )
    curve_codec = IncrementalBSplineTrajectory(
        num_control_points=trajectory.num_control_points,
        degree=trajectory.spline_degree,
        num_path_points=trajectory.num_path_points,
        control_increment_mean_xy_m=trajectory.control_increment_mean_xy_m,
        control_increment_std_xy_m=trajectory.control_increment_std_xy_m,
    )
    trajectory_decoder = ConditionalCurveFlowDecoder(
        control_tokens=curve_codec.num_control_tokens,
        coordinate_dim=curve_codec.coordinate_dim,
        model_dim=decoder.model_dim,
        layers=decoder.transformer_layers,
        heads=decoder.transformer_heads,
        dropout=decoder.dropout,
        planning_horizon_m=planning_horizon_m,
        path_to_increment_weight=curve_codec.increment_basis.transpose(0, 1),
    )
    return CurveNavPolicy(
        depth_encoder=depth_encoder,
        condition_encoder=condition_encoder,
        trajectory_decoder=trajectory_decoder,
        trajectory_evaluator=EvidenceTrajectoryEvaluator(decoder.model_dim, decoder.transformer_heads, planning_horizon_m),
        integration_steps=decoder.integration_steps,
        curve_codec=curve_codec,
        planning_horizon_m=planning_horizon_m,
    )
