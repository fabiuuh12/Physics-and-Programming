"""Numerical and synthetic image regressions for the physics camera."""
import numpy as np
import pytest

pytest.importorskip('cv2')
from vision.phone_physics_camera import (
    MotionEstimator, PhoneTracker, demo_frame, physical_values, positive,
)


def test_quadratic_motion_with_irregular_timestamps_and_y_up():
    estimator = MotionEstimator()
    times = np.linspace(0, 0.44, 20) ** 1.1
    for t in times:
        result = estimator.add(t, (100 + 12*t + 3*t*t, 200 - 8*t + 2*t*t))
    middle = (times[0] + times[-1]) / 2
    velocity, acceleration = result
    np.testing.assert_allclose(velocity, [12 + 6*middle, 8 - 4*middle], atol=1e-8)
    np.testing.assert_allclose(acceleration, [6, -4], atol=1e-8)


def test_gap_and_duplicate_timestamp_clear_derivatives():
    estimator = MotionEstimator()
    for i in range(12):
        estimator.add(i/30, (i, 0))
    assert estimator.add(2, (80, 0)) is None
    assert len(estimator.samples) == 1
    assert estimator.add(2, (80, 0)) is None
    assert len(estimator.samples) == 1


def test_physical_units_require_scale_and_mass():
    motion = np.array([300., 400.]), np.array([0., -200.])
    v, a, force, energy = physical_values(motion, None, 0.2)
    assert force is energy is None
    v, a, force, energy = physical_values(motion, 0.01, None)
    assert force is energy is None
    v, a, force, energy = physical_values(motion, 0.01, 0.2)
    np.testing.assert_allclose(v, [3, 4])
    np.testing.assert_allclose(force, [0, -0.4])
    assert energy == pytest.approx(2.5)


def test_tracker_follows_phone_and_drops_lock_on_occlusion():
    tracker = PhoneTracker()
    frame, box = demo_frame(0)
    assert tracker.select(frame, box)
    for i in range(1, 61):
        frame, box = demo_frame(i/30)
        center = tracker.update(frame)
        assert center is not None
        np.testing.assert_allclose(center, [box[0]+52.5, box[1]+95], atol=2)
    assert tracker.update(np.zeros_like(frame)) is None
    assert not tracker.active


def test_textureless_selection_is_rejected():
    assert not PhoneTracker().select(np.zeros((100, 100, 3), np.uint8), (10, 10, 50, 50))


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_measurements_rejected(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        positive(value)
