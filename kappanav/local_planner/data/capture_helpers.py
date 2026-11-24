"""CPU helper processes of a capture worker: the expert's per-anchor planning and the anchors' label geometry.

A capture worker (one GPU process) drives a scene's captures concurrently and labels them; the CPU-heavy pure
geometry goes to these helpers: expert_route at every driving anchor, and per labelled anchor local_labels, the
exploration ranking with its first routes, and the relabelled goals' expert routes. Helpers run the same
functions on the same inputs (capture arrays read from the capture folder, the scene's navigation grid), without
CUDA, so every result equals the worker's own; only where it is computed changes. Inline computes them in the
calling process (the reference path, and the path without helpers).
"""
from concurrent.futures import Future, ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import multiprocessing
import os
import threading
import numpy as np
from kappanav.local_planner.data.bspline_teachers import (EXPLORATION_TEACHERS, _expert_from_reach, expert_route,
                                                          exploration_ranking)
from kappanav.local_planner.data.local_replay import local_labels
from kappanav.common.geometry import Grid


def label_context(actor):
    """The scalars of the actor that the label geometry reads."""
    projector = actor.policy.depth_encoder.metric_projector
    return dict(horizon=float(actor.policy.planning_horizon_m), projector_horizon=float(projector.planning_horizon_m),
                projector_radius=float(projector.radius), head_radius=float(actor.head.radius))


def _anchor_geometry(actor, frames, grid, utility, selected, view, targets):
    """local_labels at an anchor and, at an extra anchor (`targets` not None), its exploration ranking and the
    routes of its first EXPLORATION_TEACHERS sectors. ranking is False at an ordinary anchor."""
    field, tree, error = local_labels(actor, frames, dict(selection=dict(indices=selected)))
    if targets is None:
        return field, tree, error, False, None
    ranking = exploration_ranking(actor, grid, view, utility, field, targets=targets)
    if ranking is None:
        return field, tree, error, None, None
    reaches, order, _ = ranking
    return field, tree, error, ranking, {index: _expert_from_reach(grid, view, reaches[index]) for index in order[:EXPLORATION_TEACHERS]}


# Helper-process entry points: arguments are paths and small arrays; scene resources are loaded once per helper.
@lru_cache(maxsize=1)
def _actor(context):
    context = dict(context)
    projector = SimpleNamespace(planning_horizon_m=context['projector_horizon'], radius=context['projector_radius'])
    return SimpleNamespace(policy=SimpleNamespace(planning_horizon_m=context['horizon'],
                           depth_encoder=SimpleNamespace(metric_projector=projector)), head=SimpleNamespace(radius=context['head_radius']))


@lru_cache(maxsize=2)
def _grid(navigation):
    return Grid.load(Path(navigation))


@lru_cache(maxsize=2)
def _utility(navigation, horizon, radius):
    from kappanav.local_planner.data.privileged import SourceConfigurationSpaceQuery
    from kappanav.local_planner.data.value_targets import RouteUtilityTeacher
    return RouteUtilityTeacher(SourceConfigurationSpaceQuery.from_paths((Path(navigation),)), horizon, radius)


@lru_cache(maxsize=4)
def _frames(folder):
    """The capture arrays local_labels reads (depths memory-mapped)."""
    folder = Path(folder)
    frames = {name: np.load(folder/f'{name}.npy') for name in ('acquired_views', 'body_to_world', 'camera_intrinsics', 'camera_to_body')}
    frames['depths'] = np.load(folder/'depths.npy', mmap_mode='r')
    return frames


def anchor_geometry(context, folder, navigation, selected, view, targets):
    key = tuple(sorted(context.items()))
    return _anchor_geometry(_actor(key), _frames(folder), _grid(navigation), _utility(navigation, context['horizon'], context['head_radius']),
                            selected, view, targets)


def goal_route(navigation, view, goal):
    return expert_route(_grid(navigation), view, goal)


def drive_plan(navigation, view, goal):
    return expert_route(_grid(navigation), view, goal)


class Inline:
    """The label geometry computed in this process with the caller's own objects, as completed futures."""
    def __init__(self, actor, grid, utility, raw):
        self.actor, self.grid, self.utility, self.raw = actor, grid, utility, raw

    def label(self, function, *args):
        future = Future()
        if function is anchor_geometry:
            _, _, _, selected, view, targets = args
            future.set_result(_anchor_geometry(self.actor, self.raw, self.grid, self.utility, selected, view, targets))
        elif function is goal_route:
            future.set_result(expert_route(self.grid, *args[1:]))
        else:
            raise ValueError(f'Unknown label job {function}')
        return future


class _Array:
    """A CPU tensor sent between processes as a numpy array. A pool pickles torch tensors into /dev/shm files that
    stay mapped while the receiver holds them, and a worker holds a whole scene's labels until it compiles the scene;
    arrays are copied through the pipe instead. The round trip is exact (same dtype, shape and values)."""
    def __init__(self, tensor): self.array = tensor.numpy()


def _pack(value):
    import torch
    if isinstance(value, torch.Tensor): return _Array(value.detach().cpu())
    if isinstance(value, (tuple, list)): return type(value)(_pack(item) for item in value)
    if isinstance(value, dict): return {key: _pack(item) for key, item in value.items()}
    return value


def _unpack(value):
    import torch
    if isinstance(value, _Array): return torch.from_numpy(value.array)
    if isinstance(value, (tuple, list)): return type(value)(_unpack(item) for item in value)
    if isinstance(value, dict): return {key: _unpack(item) for key, item in value.items()}
    return value


def _call(function, *args):
    return _pack(function(*args))


def _initialize():
    import torch
    torch.set_num_threads(1)
    # Helpers are oversubscribed; the workers' render/control/label threads keep precedence over them.
    os.nice(5)


class Helpers:
    """A pool of CPU helper processes shared by a worker's driving threads and its label thread.

    A driving thread has at most one plan in flight and waits for it; label jobs are admitted only while
    in-flight label jobs leave a free helper for every driving capture, so drive plans never queue behind
    label jobs (the label backlog fills the helpers the drives do not use)."""
    def __init__(self, processes):
        self.size = int(processes)
        self.executor = ProcessPoolExecutor(self.size, mp_context=multiprocessing.get_context('spawn'), initializer=_initialize)
        self.condition = threading.Condition(); self.driving = 0; self.labelling = 0

    def __enter__(self): return self
    def __exit__(self, *_): self.executor.shutdown(cancel_futures=True)

    def drive(self, function, *args):
        return _unpack(self.executor.submit(_call, function, *args).result())

    def drivers(self, change):
        with self.condition:
            self.driving += change; self.condition.notify_all()

    def label(self, function, *args):
        with self.condition:
            self.condition.wait_for(lambda: self.labelling < max(1, self.size - self.driving))
            self.labelling += 1
        result = Future()
        def done(future):
            self._released(future)
            try: result.set_result(_unpack(future.result()))
            except BaseException as error: result.set_exception(error)
        self.executor.submit(_call, function, *args).add_done_callback(done)
        return result

    def _released(self, _):
        with self.condition:
            self.labelling -= 1; self.condition.notify_all()
