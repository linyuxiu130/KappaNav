import pytest
import numpy as np
from kappanav.local_planner.data.capture import save_capture, load_capture


def test_capture_round_trip_keeps_arrays_and_metadata_without_pickle(tmp_path):
    arrays=dict(depths=np.random.rand(3,4,5).astype(np.float32),turn_direction=np.array([0,1,-1],np.int8))
    save_capture(tmp_path/'capture.json',arrays,dict(case=dict(goal=[1.,2.]),positive_expert=True,eligible_anchor_count=2))
    raw=load_capture(tmp_path/'capture.json')
    np.testing.assert_array_equal(raw['depths'],arrays['depths'])
    assert raw['turn_direction'].dtype==np.int8
    assert raw['case']['goal']==[1.,2.] and raw['eligible_anchor_count']==2


def test_episode_camera_is_the_mount_at_a_pitch_drawn_from_the_case_alone():
    import json
    from pathlib import Path
    from kappanav.local_planner.capture.collect_mpc import episode_camera
    from kappanav.common.physical import rest_camera_pose
    runtime = json.loads((Path(__file__).resolve().parents[2]/'configs/runtime.json').read_text())['camera']
    assert episode_camera(dict(case_id='a'), runtime) == (runtime, None)
    draws = [episode_camera(dict(case_id=f'case_{i}'), runtime, (2., 9.)) for i in range(200)]
    ups = [up for _, up in draws]
    assert all(2. <= up <= 9. for up in ups) and max(ups)-min(ups) > 6.
    assert episode_camera(dict(case_id='case_7'), runtime, (2., 9.)) == draws[7]
    for camera, up in draws[:5]:
        forward, height, pitch = rest_camera_pose(up)
        assert (camera['forward_offset_m'], camera['height_m'], camera['downward_pitch_degrees']) == (forward, height, pitch)
        assert {key: value for key, value in camera.items() if key not in ('forward_offset_m', 'height_m', 'downward_pitch_degrees')} == \
            {key: value for key, value in runtime.items() if key not in ('forward_offset_m', 'height_m', 'downward_pitch_degrees')}


def test_mounted_camera_pose_is_the_level_mount_turned_about_base_link():
    import math
    import numpy as np
    from kappanav.common.physical import rest_camera_pose, ROBOT, ROBOT_BASE_HEIGHT_ABOVE_GROUND_M
    from kappanav.evaluation.simulation.renderer import camera_to_body, mounted_camera
    settings = dict(image_width=640, image_height=360, focal_x_px=326.4, focal_y_px=326.4)
    lift = np.eye(4); lift[2, 3] = ROBOT.rest_base_height_m-ROBOT_BASE_HEIGHT_ABOVE_GROUND_M
    level = np.linalg.inv(lift) @ camera_to_body(mounted_camera(settings, 0.))   # the mount in base_link
    for up in (0., 2.5, 6., 9.):
        turn = math.radians(up)
        nose_up = np.eye(4); nose_up[[0, 0, 2, 2], [0, 2, 0, 2]] = math.cos(turn), -math.sin(turn), math.sin(turn), math.cos(turn)
        np.testing.assert_allclose(camera_to_body(mounted_camera(settings, up)), lift @ nose_up @ level, atol=1e-6)
    np.testing.assert_allclose(rest_camera_pose(0.)[:2], (ROBOT.camera_forward_m, ROBOT.camera_height_m+ROBOT.rest_base_height_m-ROBOT_BASE_HEIGHT_ABOVE_GROUND_M), rtol=0, atol=1e-12)


@pytest.mark.dingo
def test_the_expert_parks_its_axle_midpoint_not_the_trailing_base_link():
    import numpy as np
    from kappanav.local_planner.capture.collect_mpc import parked, PARKED_M
    from kappanav.common.physical import TURN_CENTRE_FORWARD_OFFSET_M
    goal = np.array([1., 2.])
    behind = np.r_[goal-[TURN_CENTRE_FORWARD_OFFSET_M, 0.], 0.]   # base_link trailing an axle that is on the goal
    assert parked(behind, goal) and not parked(np.r_[goal, 0.], goal)
    assert not parked(behind+[0., PARKED_M+1e-3, 0.], goal)
