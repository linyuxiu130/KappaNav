"""Capture workers: helper admission, scene claims and helper planning equal to in-process planning."""
import threading
import numpy as np
from kappanav.local_planner.data.capture_helpers import Helpers, drive_plan
from kappanav.local_planner.capture.capture_pipeline import claim


def _wait(seconds):
    import time
    time.sleep(seconds)
    return seconds


def test_label_jobs_leave_a_helper_for_every_driving_capture():
    with Helpers(3) as helpers:
        helpers.drivers(2)
        first = helpers.label(_wait, .5)
        admitted = threading.Event()
        threading.Thread(target=lambda: (helpers.label(_wait, 0.), admitted.set()), daemon=True).start()
        assert not admitted.wait(.2)          # 3 helpers - 2 drives: one label job at a time
        assert first.result() == .5 and admitted.wait(5)
        helpers.drivers(-2)
        assert [helpers.label(abs, -i).result() for i in range(4)] == [0, 1, 2, 3]


def test_a_claimed_scene_is_skipped_until_released(tmp_path):
    with claim(tmp_path/'scene') as held:
        with claim(tmp_path/'scene') as other:
            assert held and not other
    with claim(tmp_path/'scene') as again:
        assert again


def test_helper_planning_equals_in_process_planning(tmp_path):
    from scipy.ndimage import distance_transform_edt
    from kappanav.local_planner.data.bspline_teachers import expert_route
    from kappanav.common.geometry import Grid
    free = np.ones((100, 100), bool); free[40:60, 30:70] = False
    clearance = np.maximum(distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]*.05-np.sqrt(2)*.025, 0)
    path = tmp_path/'navigation.npz'
    np.savez(path, free=free, clearance_m=clearance.astype(np.float32), origin_xy=np.zeros(2), cell_size_m=np.float64(.05))
    view, goal = np.array([1., 1., .2]), np.array([4., 4.])
    local = expert_route(Grid.load(path), view, goal)
    with Helpers(1) as helpers:
        remote = helpers.drive(drive_plan, str(path), view, goal)
    np.testing.assert_array_equal(local, remote)


def test_helper_results_cross_processes_as_exact_arrays_not_shared_tensors():
    import pickle
    import torch
    from kappanav.local_planner.data.capture_helpers import _pack, _unpack, _Array
    value = (torch.randn(5, 3), dict(tree=torch.arange(7, dtype=torch.int64), half=torch.randn(4).half()), [1.5, None, 'x'])
    packed = _pack(value)
    assert isinstance(packed[0], _Array) and isinstance(packed[1]['half'], _Array) and packed[2] == [1.5, None, 'x']
    assert b'torch' not in pickle.dumps(packed)   # no tensor storage, so nothing goes through /dev/shm
    restored = _unpack(pickle.loads(pickle.dumps(packed)))
    for a, b in ((value[0], restored[0]), (value[1]['tree'], restored[1]['tree']), (value[1]['half'], restored[1]['half'])):
        assert a.dtype == b.dtype and torch.equal(a, b)
