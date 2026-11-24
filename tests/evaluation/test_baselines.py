import io
import json
import sys

import numpy as np
import pytest

from kappanav.evaluation.benchmark import baselines


def test_client_request_round_trips_through_the_server_protocol(monkeypatch):
    """What BaselineClient sends is exactly what a baseline server reads (serving.protocol), and the NPZ reply is
    read back as the body-frame trajectory."""
    flask = pytest.importorskip('flask')
    sys.path.insert(0, str(baselines.ROOT))
    from serving.protocol import policy_response, read_depth, read_rgb
    app, seen = flask.Flask('probe'), {}

    @app.route('/pointgoal_step', methods=['POST'])
    def step():
        seen.update(rgb=read_rgb('image', 1), depth=read_depth(1), goal=json.loads(flask.request.form['goal_data']),
                    state=json.loads(flask.request.form['state_data']))
        return policy_response(np.array([[[0., 0.], [1., .5], [2., 1.]]], np.float32))

    tester = app.test_client()

    class Session:
        def post(self, url, files=None, data=None, json=None, timeout=None):
            form = dict(data, **{key: (io.BytesIO(bytes(value[1])), value[0]) for key, value in files.items()})
            reply = tester.post(url.split('127.0.0.1:0')[1], data=form, content_type='multipart/form-data')
            return type('Reply', (), dict(raise_for_status=lambda self: None, content=reply.data))()

    client = baselines.BaselineClient('sandplanner', 0)
    client.session = Session()
    rgb = np.random.default_rng(0).integers(0, 255, (4, 6, 3), dtype=np.uint8)
    depth = np.array([[1., np.inf, 2., 3., 4., 5.]] * 4, np.float32)
    state = dict(robot_pos=[1., 2., .2], robot_quat=[0., 0., 0., 1.], body_to_world=np.eye(4), camera_to_body=np.eye(4),
                 camera_intrinsics=np.eye(3), timestamps=.4)
    trajectory = client.step(rgb, depth, np.array([3., -1.], np.float32), state)
    assert np.array_equal(seen['rgb'][0], rgb) and np.array_equal(seen['depth'][0, ..., 0], depth)
    assert seen['goal'] == dict(goal_x=[3.], goal_y=[-1.])
    assert seen['state']['robot_pos'] == [[1., 2., .2]] and seen['state']['planning_goal'] == [[3., -1.]]
    np.testing.assert_allclose(trajectory, [[0., 0.], [1., .5], [2., 1.]])


def test_baseline_trajectory_is_tracked_in_its_planning_frame_at_its_observation(monkeypatch):
    from kappanav.evaluation.benchmark.run import BaselinePlanner
    from kappanav.evaluation.simulation import scene

    class Client:
        name, rgb, origin = 'stub', False, 'base'
        def reset(self, intrinsic): self.intrinsic = intrinsic
        def step(self, rgb, depth, goal, state): return np.array([[1., 0.], [2., 1.]])

    monkeypatch.setattr(scene, 'raw_observation', lambda sim, view, ground_z=0., rgb=False: (None, np.ones((2, 3), np.float32)))
    monkeypatch.setattr(scene, 'rendered_intrinsics', lambda: type('K', (), dict(fx=1., fy=1., cx=0., cy=0.))())
    monkeypatch.setattr(scene, 'sensor_assets', lambda: (None, None, np.eye(4)))
    planner = BaselinePlanner(Client())
    planner.start(None)
    view = np.array([5., 3., np.pi / 2])                        # facing +y: body x is world +y, body y is world -x
    world, record = planner.request(view, 0., np.eye(4), 0., np.zeros(2, np.float32), True)()
    np.testing.assert_allclose(world, [[5., 3.], [5., 4.], [4., 5.]], atol=1e-12)
    assert record == {} and planner.finish()['trajectories'].shape == (1, 2, 2)
    camera = np.eye(4); camera[:2, 3] = [.25, .05]                  # a camera 0.25 m ahead and 0.05 m left of the base
    monkeypatch.setattr(scene, 'sensor_assets', lambda: (None, None, camera))
    Client.origin = 'camera'; client = Client(); seen = {}
    client.step = lambda rgb, depth, goal, state: seen.setdefault('goal', goal) is not None and np.array([[1., 0.]])
    planner = BaselinePlanner(client); planner.start(None)
    world, _ = planner.request(view, 0., np.eye(4), 0., np.array([3., 1.], np.float32), True)()
    np.testing.assert_allclose(seen['goal'], [2.75, .95], atol=1e-6)             # the goal seen from the camera
    np.testing.assert_allclose(world[-1], [5. - .05, 3. + 1.25], atol=1e-12)     # the trajectory leaves from the camera
    with pytest.raises(ValueError, match='point-goal'):
        planner.request(view, 0., np.eye(4), 0., np.zeros(2, np.float32), False)


def test_every_registered_baseline_has_its_server_and_config_in_the_tree():
    for name, baseline in baselines.BASELINES.items():
        for argument in baseline.server:
            if argument.endswith(('.py', '.yaml')):
                assert (baselines.ROOT / argument).is_file(), (name, argument)
        assert (baselines.ROOT / baseline.cwd).is_dir()
