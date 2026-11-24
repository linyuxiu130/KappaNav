"""Load CPU weights and tasks once before evaluation CUDA workers are forked."""
import json,os
import numpy as np
from pathlib import Path
import torch
from kappanav.local_planner.inference.bundle import load_bundle

torch.set_num_threads(1)
checkpoint,benchmark=json.loads(os.environ['KAPPANAV_EVAL_INPUTS'])
ACTOR=load_bundle(Path(checkpoint),'cpu') if checkpoint else None  # None: a baseline planner is evaluated
DATA=json.loads(Path(benchmark).read_text())

from kappanav.evaluation.simulation.navigation import BODY_OUTLINE, NavigationGrid
from kappanav.common.geometry import _path_cells
from numba import float64

NAVIGATIONS={row['scene_id']:NavigationGrid(row['navigation']) for row in DATA['scenes']}
# Build the body's contact geometry and compile its query once, before workers inherit them: the boundary index for a
# disc body, the body ground's signed distance for a rectangular one (NavigationGrid.body_clearance).
for navigation in NAVIGATIONS.values():
    if BODY_OUTLINE is None:navigation.contact_buckets;navigation.penetration_m
    else:navigation.body_signed_m
    navigation.oracle._atlas(torch.device('cpu'))  # trajectory metrics
if NAVIGATIONS:
    _first=next(iter(NAVIGATIONS.values()));_first.body_clearance(np.r_[_first.origin+_first.spacing/2,0.][None].repeat(2,0))
# Compile the existing CPU kernels once, before workers inherit their code.
_path_cells.compile((float64[:,::1],))
from kappanav.local_planner.data.observed_geometry import _minimum_squared
_minimum_squared.compile((float64[:,:,::1],float64[:,:,::1],float64[:,::1],float64[:,::1]))

import habitat_sim  # noqa: F401  (loaded once here, before the evaluation workers fork)

from kappanav.local_planner.control.mpc import body_step
body_step.compile((float64[::1],float64[::1],float64,float64))

from kappanav.local_planner.control.mpc import _nominal_rollout
_nominal_rollout.compile((float64[::1],float64[:,::1],float64,float64))
