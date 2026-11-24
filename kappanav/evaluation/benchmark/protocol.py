"""Shared benchmark defaults and simulation-time decision cadence, independent of either simulator."""
import math

DECISION_LATENCY_S=.2
GOAL_BUDGET_S=122.
SCENE_COUNT=40
EPISODES_PER_SCENE=100


def decision_steps(latency_s,period_s):
    """A delayed decision must land on a controller tick; never silently round the requested latency."""
    if not math.isfinite(latency_s) or latency_s<=0:
        raise ValueError('Decision latency must be finite and positive')
    steps=round(latency_s/period_s)
    if steps<1 or not math.isclose(steps*period_s,latency_s,rel_tol=0,abs_tol=1e-9):
        raise ValueError(f'Decision latency must be a positive multiple of {period_s:g} s')
    return steps
