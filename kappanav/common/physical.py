"""Physical contract of the simulated robot in base-link coordinates (x forward, y left, z up).

Two robots share every pipeline: the benchmark Dingo (the default, all existing benchmarks) and the AgileX Scout
Mini with its front Gemini 335 (the air-ground Wanderland evaluation). KAPPANAV_ROBOT selects one for the whole
process before anything imports this module; every module reads the selected robot's values from here.
"""
import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Robot:
    name: str
    # Exact circular configuration-space envelope of all moving collision shapes about the turn centre's path
    footprint_radius_m: float
    # A rectangular body's half length and half width about base_link: contact is tested on its outline at the actual
    # heading. None: the body is its footprint circle.
    body_half_extent_m: tuple
    collision_bottom_z_m: float       # wheel contact plane (base_link stands -this above the ground)
    collision_top_z_m: float
    maximum_step_m: float             # height difference the robot drives over
    maximum_slope_degrees: float
    # Kinematics: where the drive turns, the wheels behind the Isaac wheel-speed commands
    turn_centre_forward_m: float      # an in-place turn swings base_link on a circle of this radius
    wheel_radius_m: float
    wheel_track_m: float
    # Plant response (Isaac physics): first-order forward speed, yaw rate reached for a commanded one by speed
    speed_response_s: float
    yaw_response_speed_mps: tuple
    yaw_response_gain: tuple
    # Depth camera mount in base_link and base_link's rest attitude
    camera_forward_m: float
    camera_lateral_m: float
    camera_height_m: float
    camera_downward_pitch_degrees: float
    rest_pitch_up_degrees: float
    rest_base_height_m: float


DINGO = Robot(
    name='dingo',
    # From the enabled collision prims of the benchmark dingo.usd (sha256 43db9c54...cce20; offsets under its 0.5 scale)
    footprint_radius_m=0.167584539,
    body_half_extent_m=None,
    collision_bottom_z_m=-0.044000001,
    collision_top_z_m=0.117981499,
    maximum_step_m=0.05,               # below the 6.125 cm wheel radius
    maximum_slope_degrees=45.0,
    turn_centre_forward_m=0.0494919,   # the wheel axle midpoint (wheel joints at x = 0.0989838 under the 0.5 scale)
    wheel_radius_m=0.06125,
    wheel_track_m=0.22616,
    # Measured in the benchmark physics (Isaac): medians by forward speed
    speed_response_s=0.15,
    yaw_response_speed_mps=(0., .075, .15, .25, .45),
    yaw_response_gain=(.55, .69, .84, .91, .96),
    camera_forward_m=0.14309,
    camera_lateral_m=0.,
    camera_height_m=0.31266,
    camera_downward_pitch_degrees=10.0,
    # At rest on its casters base_link is pitched nose-up and stands this high (measured in Isaac)
    rest_pitch_up_degrees=6.0,
    rest_base_height_m=0.0512,
)

SCOUT_MINI = Robot(
    name='scout_mini',
    # Collision meshes of the scout_front_camera URDF (AgileX ugv_gazebo_sim + Gemini 335 front mount): the tyre
    # corners (|x| <= 0.3119, |y| <= 0.2933) bound the in-place turn about base_link; the camera housing is the top.
    footprint_radius_m=0.4182,
    body_half_extent_m=(0.3119, 0.2933),  # tyre corners; the circle above reaches 9-12 cm past the sides and ends
    collision_bottom_z_m=-0.1805,      # wheel centres z = -0.1005, radius 0.080
    collision_top_z_m=0.0805,
    maximum_step_m=0.07,               # published maximum climbable obstacle (no load)
    maximum_slope_degrees=30.0,        # published maximum climbable slope (no load)
    turn_centre_forward_m=0.,          # skid steer: wheels symmetric about base_link (x = +-0.232)
    wheel_radius_m=0.080,
    wheel_track_m=0.4546,              # tyre contact centres y = +-0.2273
    # Forward speed: measured in Isaac (calibrate_scout_mini.py; 0.5 m/s commanded, 0.5007 m/s steady). Yaw: PhysX
    # gives 0.01-0.17 for the wide tyres' line contact, which is not physical; Coulomb friction at the four contacts
    # of a skid steer (wheels at +-L/2, +-W/2, rims at the commanded wheel speeds) balances at a yaw rate
    # W^2 / (W^2 + L^2) = 0.49 of the commanded one at every forward speed, with the forward speed itself unchanged.
    speed_response_s=0.04,
    yaw_response_speed_mps=(0., .075, .15, .25, .45),
    yaw_response_gain=(.49,) * 5,
    # Depth origin = left IR = camera_link (gemini_mount_joint xyz 0.24277 0.025 0.068), looking level along +x
    camera_forward_m=0.24277,
    camera_lateral_m=0.025,
    camera_height_m=0.06800166,
    camera_downward_pitch_degrees=0.,
    rest_pitch_up_degrees=0.,          # four wheels: base_link rests level
    rest_base_height_m=0.1805,
)

ROBOTS = {robot.name: robot for robot in (DINGO, SCOUT_MINI)}
ROBOT = ROBOTS[os.environ.get('KAPPANAV_ROBOT', 'dingo')]

if ROBOT is not DINGO:
    # Numba compiles these module constants into cached machine code but keys its on-disk cache by source alone, so
    # another robot keeps a cache of its own (the Dingo's stays in __pycache__). Every compiled user of the robot
    # imports this module first, before its functions are decorated.
    from numba.core import config as _numba_config
    _numba_config.CACHE_DIR = os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')),
                                           'curvenav-numba', ROBOT.name)

ROBOT_FOOTPRINT_RADIUS_M = ROBOT.footprint_radius_m
ROBOT_COLLISION_BOTTOM_Z_M = ROBOT.collision_bottom_z_m
ROBOT_COLLISION_TOP_Z_M = ROBOT.collision_top_z_m
ROBOT_BASE_HEIGHT_ABOVE_GROUND_M = -ROBOT_COLLISION_BOTTOM_Z_M
ROBOT_COLLISION_HEIGHT_M = ROBOT_COLLISION_TOP_Z_M - ROBOT_COLLISION_BOTTOM_Z_M
TURN_CENTRE_FORWARD_OFFSET_M = ROBOT.turn_centre_forward_m

# Habitat uses the step height above the ground plane; depth points use base-link z and therefore need the translated
# lower bound below.
MAXIMUM_TRAVERSABLE_HEIGHT_M = ROBOT.maximum_step_m
MAXIMUM_TRAVERSABLE_SLOPE_DEGREES = ROBOT.maximum_slope_degrees
BODY_OBSTACLE_MIN_Z_M = ROBOT_COLLISION_BOTTOM_Z_M + MAXIMUM_TRAVERSABLE_HEIGHT_M
EXTRA_CLEARANCE_M = 0.10

# One common physical query spacing for source-truth certification and offline safety measurement.
PATH_CONFIGURATION_QUERY_SPACING_M = 0.025


def rest_camera_pose(pitch_up_degrees=ROBOT.rest_pitch_up_degrees):
    """(forward offset, height, downward pitch in degrees) of the camera in the level body frame (origin
    ROBOT_BASE_HEIGHT_ABOVE_GROUND_M above the floor) when base_link stands ROBOT.rest_base_height_m high, turned
    nose-up by pitch_up_degrees about its origin. The default is the measured rest pose of the robot."""
    up = math.radians(pitch_up_degrees)
    forward = ROBOT.camera_forward_m * math.cos(up) - ROBOT.camera_height_m * math.sin(up)
    height = ROBOT.camera_forward_m * math.sin(up) + ROBOT.camera_height_m * math.cos(up)
    height += ROBOT.rest_base_height_m - ROBOT_BASE_HEIGHT_ABOVE_GROUND_M
    return forward, height, ROBOT.camera_downward_pitch_degrees - pitch_up_degrees
