"""Projection roundtrips, depth validity, and filter covariance invariants."""

import numpy as np
import pytest

from autonomous_vehicles.evaluation.depth import depth_metrics, median_scale
from autonomous_vehicles.simulation.camera import (
    CameraIntrinsics,
    depth_to_points,
    project_points,
    unproject_pixels,
)
from autonomous_vehicles.simulation.localization import ConstantVelocityFilter, noisy_positions


def test_camera_projection_unprojection_roundtrip():
    camera = CameraIntrinsics()
    points = np.array([[0, 0, 10], [1, -0.5, 5], [-2, 1, 20]])
    projection = project_points(points, camera)
    assert projection.visible.all()
    np.testing.assert_allclose(projection.pixels[0], [camera.cx, camera.cy])
    restored = unproject_pixels(projection.pixels, projection.depth, camera)
    np.testing.assert_allclose(restored, points, atol=1e-12)


def test_behind_camera_and_offscreen_points_are_not_visible():
    camera = CameraIntrinsics()
    projection = project_points(np.array([[0, 0, -1], [100, 0, 1], [0, 0, 0.01]]), camera)
    assert not projection.visible.any()
    assert np.isnan(projection.pixels[0]).all()


def test_resize_intrinsics_preserves_rays_at_pixel_centers():
    camera = CameraIntrinsics()
    resized = camera.scaled(320, 180)
    assert resized.fx == camera.fx / 2
    assert resized.cx == (camera.cx + 0.5) / 2 - 0.5
    points = np.array([[1, 2, 10]])
    original = project_points(points, camera)
    small = project_points(points, resized)
    np.testing.assert_allclose(small.pixels, (original.pixels + 0.5) / 2 - 0.5)


def test_depth_to_points_drops_invalid_measurements():
    camera = CameraIntrinsics(width=4, height=2, fx=2, fy=2, cx=1.5, cy=0.5)
    depth = np.ones((2, 4)) * 5
    depth[0, 0] = np.nan
    depth[0, 1] = 0
    points = depth_to_points(depth, camera)
    assert points.shape == (6, 3)
    assert np.all(points[:, 2] == 5)


def test_perfect_depth_metrics_and_median_scale():
    target = np.array([1, 2, 4, 8], dtype=float)
    result = depth_metrics(target, target)
    assert result.mean_absolute_error == 0
    assert result.root_mean_squared_error == 0
    assert result.delta_1 == result.delta_2 == result.delta_3 == 1
    assert median_scale(target / 2, target) == 2


def test_depth_metrics_use_target_validity_mask():
    target = np.array([1, 2, 0, np.nan])
    prediction = np.array([2, 3, np.nan, np.nan])
    result = depth_metrics(prediction, target)
    assert result.valid_pixels == 2
    assert result.mean_absolute_error == 1
    assert result.absolute_relative_error == pytest.approx(0.75)
    with pytest.raises(ValueError):
        depth_metrics(np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        depth_metrics(np.ones(2), np.zeros(2))


def test_filter_rejects_far_outlier_without_mutation():
    estimator = ConstantVelocityFilter()
    estimator.predict(0.1)
    before = estimator.snapshot()
    after = estimator.update([1000, 1000])
    assert not after.accepted
    np.testing.assert_array_equal(before.state, after.state)
    np.testing.assert_array_equal(before.covariance, after.covariance)


def test_filter_converges_on_constant_velocity_and_preserves_psd():
    times = np.arange(80) * 0.1
    truth = np.column_stack((times * 2, times * -0.5))
    measurements = noisy_positions(truth, 0.3, 42)
    estimator = ConstantVelocityFilter()
    estimates = []
    for measurement in measurements:
        estimator.predict(0.1)
        estimate = estimator.update(measurement)
        estimates.append(estimate.position)
        np.testing.assert_allclose(estimate.covariance, estimate.covariance.T, atol=1e-12)
        assert np.linalg.eigvalsh(estimate.covariance).min() >= -1e-12
    filtered_error = np.mean((np.array(estimates)[20:] - truth[20:]) ** 2)
    raw_error = np.mean((measurements[20:] - truth[20:]) ** 2)
    assert filtered_error < raw_error
    np.testing.assert_allclose(estimator.snapshot().velocity, [2, -0.5], atol=0.3)


def test_filter_snapshots_do_not_alias_internal_state():
    estimator = ConstantVelocityFilter()
    snapshot = estimator.snapshot()
    snapshot.state[0] = 100
    assert estimator.state[0] == 0
    with pytest.raises(ValueError):
        estimator.predict(0)
    with pytest.raises(ValueError):
        estimator.update([float("nan"), 0])
