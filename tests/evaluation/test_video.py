import numpy as np
import pytest


def test_local_map_is_heading_up_and_its_pixels_invert_the_transform():
    video = pytest.importorskip('kappanav.evaluation.benchmark.video')
    view = np.array([3., -2., .7])
    to_px, world = video.local_frame(view)
    centre, ahead = to_px(view[:2])[0], to_px(view[:2] + 5 * np.array([np.cos(.7), np.sin(.7)]))[0]
    right = to_px(view[:2] + 2 * np.array([np.sin(.7), -np.cos(.7)]))[0]
    assert np.isclose(ahead[0], centre[0]) and ahead[1] < centre[1]      # forward is up
    assert right[0] > centre[0] and np.isclose(right[1], centre[1])      # the robot's right is right
    v, u = 100, 400                                                       # a pixel's world point maps back to it
    assert np.allclose(to_px(world[v, u])[0], (u + .5, v + .5))


def test_raster_sampling_uses_cell_centres():
    video = pytest.importorskip('kappanav.evaluation.benchmark.video')
    grid = np.arange(20, dtype=np.float32).reshape(4, 5)                  # [x, y], 1 m cells, origin (0, 0)
    xy = np.array([[[1.5, 2.5], [3.5, .5]]])
    assert np.allclose(video.sample((grid, np.zeros(2), 1.), xy, np.nan), [[grid[1, 2], grid[3, 0]]])


def test_ply_vertices_are_read_back():
    from kappanav.evaluation.benchmark.splat import read_ply
    import tempfile, os
    table = np.zeros(3, dtype=[('x', '<f4'), ('y', '<f4'), ('opacity', '<f4')]); table['x'] = [1, 2, 3]
    header = b'ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float opacity\nend_header\n'
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(header + table.tobytes())
    try:
        assert np.array_equal(read_ply(f.name), table)
    finally:
        os.unlink(f.name)
