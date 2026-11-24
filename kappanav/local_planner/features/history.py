"""Rigid geometry shared by current four-frame preparation and deployment."""

import numpy as np


def inverse_rigid(transform: np.ndarray) -> np.ndarray:
    result = np.zeros_like(transform)
    result[..., :3, :3] = np.swapaxes(transform[..., :3, :3], -1, -2)
    result[..., :3, 3] = -(result[..., :3, :3] @ transform[..., :3, 3, None])[..., 0]
    result[..., 3, 3] = 1
    return result
