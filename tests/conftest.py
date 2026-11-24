import pytest
from kappanav.common.physical import DINGO, ROBOT


def pytest_configure(config):
    config.addinivalue_line('markers', "dingo: the test's scene and expected numbers are the Dingo's")


def pytest_collection_modifyitems(items):
    if ROBOT is DINGO:
        return
    for item in items:
        if item.get_closest_marker('dingo'):
            item.add_marker(pytest.mark.skip(reason=f"Dingo scene and numbers; selected robot is {ROBOT.name}"))
