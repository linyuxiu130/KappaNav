import pytest
import numpy as np


@pytest.mark.dingo
def test_wall_starts_are_near_obstacles_and_the_robot_can_turn_in_place_there(tmp_path):
    from kappanav.evaluation.simulation.navigation import save_navigation_grid, NavigationGrid
    from kappanav.evaluation.simulation.tasks import wall_episodes, turns_in_place
    free = np.ones((120, 120), bool); free[50:70, 50:70] = False; free[:, :3] = False; free[:3, :] = False
    path = tmp_path/'navigation.npz'; save_navigation_grid(path, free, np.array([-6., -6.]), .1)
    episodes = wall_episodes(dict(scene_id='s', navigation=str(path)), seed=3, count=20)
    navigation = NavigationGrid(path)
    assert len(episodes) == 20
    assert all(turns_in_place(navigation, e['start']) for e in episodes)
    assert all(e['start_clearance_m'] < .10 for e in episodes)
    # facing a wall in contact is the pose the rule removes
    contact = min(episodes, key=lambda e: e['start_clearance_m'])['start']
    assert not all(turns_in_place(navigation, (contact[0], contact[1], yaw)) for yaw in np.linspace(-np.pi, np.pi, 16))
