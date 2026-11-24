"""Calibrated metric-depth reprojection shared by data preparation and deployment."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        width: int,
        height: int,
    ) -> "PinholeIntrinsics":
        value = np.asarray(matrix, dtype=np.float64)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(value[0, 0]),
            fy=float(value[1, 1]),
            cx=float(value[0, 2]),
            cy=float(value[1, 2]),
        )

    def matrix(self) -> np.ndarray:
        return np.array(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )

    def at_resolution(self, width: int, height: int) -> "PinholeIntrinsics":
        """Resample the same horizontal and vertical field of view."""
        scale_x = width / self.width
        scale_y = height / self.height
        return PinholeIntrinsics(
            width=width,
            height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )


def preprocess_depth(
    depth_m: np.ndarray,
    *,
    source_intrinsics: PinholeIntrinsics,
    maximum_m: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize optical-Z; preserve FoV and invalid depth.

    Pixels have half-integer centres, as in the simulator rasterizer;
    K follows that same mapping. No depth interpolation crosses object edges.
    """
    y = np.minimum(
        ((np.arange(height) + 0.5) * depth_m.shape[0] / height).astype(int),
        depth_m.shape[0] - 1,
    )
    x = np.minimum(
        ((np.arange(width) + 0.5) * depth_m.shape[1] / width).astype(int),
        depth_m.shape[1] - 1,
    )
    depth = depth_m[y[:, None], x].astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    hit = valid & (depth < maximum_m)
    # Reserve 0 for unknown and 1 for finite measurements at/beyond the
    # planning range. FP16 rounding must not change a surface into either.
    smallest = float(np.nextafter(np.float16(0), np.float16(1)))
    largest = float(np.nextafter(np.float16(1), np.float16(0)))
    depth = np.where(
        hit, np.clip(depth / maximum_m, smallest, largest), np.where(valid, 1, 0)
    ).astype(np.float32)
    intrinsic = (
        source_intrinsics.at_resolution(width, height).matrix().astype(np.float32)
    )
    return depth, intrinsic
