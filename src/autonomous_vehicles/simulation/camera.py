"""Pinhole projection with an explicit camera coordinate convention.

Camera coordinates: x right, y down, z forward. Projection and unprojection
operate on metric depths along z, not Euclidean range. No image renderer or
real sensor calibration is implied by this analytical model.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 640
    height: int = 360
    fx: float = 400.0
    fy: float = 400.0
    cx: float = 319.5
    cy: float = 179.5
    near: float = 0.1

    def __post_init__(self):
        if min(self.width, self.height) < 1:
            raise ValueError("Image dimensions must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy, self.near]).all():
            raise ValueError("Intrinsics must be finite")
        if min(self.fx, self.fy, self.near) <= 0:
            raise ValueError("Focal lengths and near plane must be positive")
        if not 0 <= self.cx < self.width or not 0 <= self.cy < self.height:
            raise ValueError("Principal point must be inside image")

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        if min(width, height) < 1:
            raise ValueError("Target dimensions must be positive")
        sx, sy = width / self.width, height / self.height
        # Pixel centers transform consistently under resize.
        return CameraIntrinsics(
            width,
            height,
            self.fx * sx,
            self.fy * sy,
            (self.cx + 0.5) * sx - 0.5,
            (self.cy + 0.5) * sy - 0.5,
            self.near,
        )


@dataclass(frozen=True)
class Projection:
    pixels: np.ndarray
    depth: np.ndarray
    visible: np.ndarray


def project_points(points: np.ndarray, intrinsics: CameraIntrinsics) -> Projection:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx3 camera coordinates")
    depth = points[:, 2].copy()
    in_front = depth >= intrinsics.near
    pixels = np.full((len(points), 2), np.nan)
    pixels[in_front, 0] = intrinsics.fx * points[in_front, 0] / depth[in_front] + intrinsics.cx
    pixels[in_front, 1] = intrinsics.fy * points[in_front, 1] / depth[in_front] + intrinsics.cy
    visible = (
        in_front
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < intrinsics.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < intrinsics.height)
    )
    return Projection(pixels, depth, visible)


def unproject_pixels(
    pixels: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics
) -> np.ndarray:
    pixels, depth = np.asarray(pixels, dtype=float), np.asarray(depth, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),):
        raise ValueError("Expected Nx2 pixels and N depths")
    if not np.isfinite(pixels).all() or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Pixels must be finite and depths finite and positive")
    return np.column_stack(
        (
            (pixels[:, 0] - intrinsics.cx) / intrinsics.fx * depth,
            (pixels[:, 1] - intrinsics.cy) / intrinsics.fy * depth,
            depth,
        )
    )


def depth_to_points(
    depth_map: np.ndarray, intrinsics: CameraIntrinsics, stride: int = 1
) -> np.ndarray:
    depth_map = np.asarray(depth_map, dtype=float)
    if depth_map.shape != (intrinsics.height, intrinsics.width) or stride < 1:
        raise ValueError("Depth map must match camera dimensions and stride must be positive")
    rows, columns = np.mgrid[0 : intrinsics.height : stride, 0 : intrinsics.width : stride]
    depth = depth_map[::stride, ::stride].ravel()
    pixels = np.column_stack((columns.ravel(), rows.ravel()))
    valid = np.isfinite(depth) & (depth >= intrinsics.near)
    return unproject_pixels(pixels[valid], depth[valid], intrinsics)
