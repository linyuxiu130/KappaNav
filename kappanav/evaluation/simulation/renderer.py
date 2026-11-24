"""Habitat-Sim camera and pose helpers for the shared collection and evaluation pipeline."""
import math
from pathlib import Path
from typing import Any
import numpy as np
from kappanav.common.physical import (ROBOT_COLLISION_HEIGHT_M, ROBOT_FOOTPRINT_RADIUS_M,
                                      ROBOT_BASE_HEIGHT_ABOVE_GROUND_M, rest_camera_pose)


def camera_contract(camera: dict[str, float | int]) -> dict[str, Any]:
    width = int(camera["image_width"])
    height = int(camera["image_height"])
    focal_x = float(camera["focal_x_px"])
    focal_y = float(camera["focal_y_px"])
    forward_offset = float(camera["forward_offset_m"])
    lateral_offset = float(camera.get("lateral_offset_m", 0.0))  # +y = left of base_link
    camera_height = float(camera["height_m"])
    pitch = math.radians(float(camera["downward_pitch_degrees"]))
    sine, cosine = math.sin(pitch), math.cos(pitch)
    horizontal_fov = math.degrees(2 * math.atan(width / (2 * focal_x)))
    return {
        "sensor": "Habitat-Sim pinhole depth camera",
        "measurement_type": "distance_to_image_plane",
        "depth": {
            "dtype": "float32",
            "unit": "m",
            "invalid": "not isfinite(value) or value <= 0",
            "near_m": 0.01,
            "far_m": 100.0,
        },
        "image": {
            "width": width,
            "height": height,
            "horizontal_fov_degrees": horizontal_fov,
            "K": [
                [focal_x, 0.0, width / 2],
                [0.0, focal_y, height / 2],
                [0.0, 0.0, 1.0],
            ],
        },
        "body_from_camera_optical": [
            [0.0, -sine, cosine, forward_offset],
            [-1.0, 0.0, 0.0, lateral_offset],
            [0.0, -cosine, -sine, camera_height],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def mounted_camera(camera: dict[str, float | int], pitch_up_degrees: float) -> dict[str, float | int]:
    """The camera settings with the pose of the robot's mount when base_link rests nose-up by pitch_up_degrees."""
    forward, height, pitch = rest_camera_pose(pitch_up_degrees)
    return dict(camera, forward_offset_m=forward, height_m=height, downward_pitch_degrees=pitch)


def camera_to_body(camera: dict[str, float | int]) -> np.ndarray:
    """The 4x4 optical-camera-to-level-body transform of these camera settings."""
    return np.asarray(camera_contract(camera)["body_from_camera_optical"], dtype=np.float32)


def place_camera(simulator: Any, camera: dict[str, float | int]) -> None:
    """Move the depth sensor to the pose of these camera settings (same convention as create_simulator)."""
    import magnum as mn

    node = simulator._sensors["depth"]._sensor_object.node
    node.translation = mn.Vector3(-float(camera.get("lateral_offset_m", 0.0)), float(camera["height_m"]),
                                  -float(camera["forward_offset_m"]))
    node.rotation = mn.Quaternion.rotation(
        mn.Rad(-math.radians(float(camera["downward_pitch_degrees"]))), mn.Vector3.x_axis())


def create_simulator(
    asset_root: Path,
    scene_id: str,
    gpu: int,
    camera: dict[str, float | int],
    scene_dataset: str,
    rgb: bool = False,
):
    """A Habitat simulator of the scene with the depth sensor ("depth") of these camera settings and, with `rgb`, a
    colour sensor ("color") at the same pose with the same image."""
    import habitat_sim

    settings = habitat_sim.SimulatorConfiguration()
    settings.scene_dataset_config_file = str(
        asset_root / scene_dataset
    )
    settings.scene_id = str(asset_root / "scenes" / f"{scene_id}.scene_instance.json")
    settings.gpu_device_id, settings.enable_physics = gpu, False
    settings.requires_textures = rgb  # depth needs the geometry only; a city's textures take gigabytes of GPU memory
    depth = habitat_sim.CameraSensorSpec()
    depth.uuid, depth.sensor_type = "depth", habitat_sim.SensorType.DEPTH
    depth.resolution, depth.hfov, depth.near, depth.far = (
        [int(camera["image_height"]), int(camera["image_width"])],
        camera_contract(camera)["image"]["horizontal_fov_degrees"],
        0.01,
        100.0,
    )
    depth.position = [  # the agent frame: +X right, +Y up, -Z forward
        -float(camera.get("lateral_offset_m", 0.0)),
        float(camera["height_m"]),
        -float(camera["forward_offset_m"]),
    ]
    depth.orientation = [
        -math.radians(float(camera["downward_pitch_degrees"])),
        0.0,
        0.0,
    ]
    sensors = [depth]
    if rgb:
        color = habitat_sim.CameraSensorSpec()
        color.uuid, color.sensor_type = "color", habitat_sim.SensorType.COLOR
        color.resolution, color.hfov, color.position, color.orientation = (
            depth.resolution, depth.hfov, depth.position, depth.orientation)
        sensors.append(color)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.height, agent.radius, agent.sensor_specifications = (
        ROBOT_COLLISION_HEIGHT_M,
        ROBOT_FOOTPRINT_RADIUS_M,
        sensors,
    )
    return habitat_sim.Simulator(habitat_sim.Configuration(settings, [agent]))


def base_position_from_navmesh(position: np.ndarray) -> np.ndarray:
    """Lift a floor contact point to the robot's base-link origin."""
    base_position = np.asarray(position, dtype=np.float32).copy()
    base_position[1] += ROBOT_BASE_HEIGHT_ABOVE_GROUND_M
    return base_position


def set_pose(simulator: Any, position: np.ndarray, heading: float) -> None:
    import habitat_sim

    habitat_yaw = math.atan2(-math.cos(heading), -math.sin(heading))
    state = habitat_sim.AgentState()
    state.position = base_position_from_navmesh(position)
    state.rotation = np.quaternion(
        math.cos(habitat_yaw / 2), 0.0, math.sin(habitat_yaw / 2), 0.0
    )
    simulator.get_agent(0).set_state(state, reset_sensors=True)
