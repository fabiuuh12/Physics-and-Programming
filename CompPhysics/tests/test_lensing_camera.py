"""Analytic lens-equation and hand-control regressions, without a webcam."""
import numpy as np
import pytest

pytest.importorskip('cv2')
from vision.lensing_camera import LensControl, PointLens, compose, synthetic_scene


def test_zero_strength_is_identity_without_artistic_mask():
    frame = synthetic_scene(320, 240, 0)
    output = PointLens(320, 240).apply(frame, (160, 120), 0, shadow=False)
    np.testing.assert_array_equal(frame, output)


def test_einstein_ring_maps_to_aligned_source():
    lens = PointLens(201, 201)
    mx, my, _ = lens.maps((100, 100), 40)
    for x, y in [(140, 100), (60, 100), (100, 140), (100, 60)]:
        assert mx[y, x] == pytest.approx(100)
        assert my[y, x] == pytest.approx(100)
    assert np.isfinite(mx).all() and np.isfinite(my).all()


def test_two_images_of_off_axis_source_satisfy_lens_equation():
    # beta=30, theta_E=40 => theta=+/- roots 57.72 and -27.72.
    beta, radius = 30., 40.
    roots = [(beta + sign*np.sqrt(beta**2+4*radius**2))/2 for sign in (1, -1)]
    for theta in roots:
        assert theta - radius**2/theta == pytest.approx(beta)
    # Exact integer example: beta=30, theta_E=20 => theta=40 and -10.
    mx, my, _ = PointLens(201, 201).maps((100, 100), 20)
    assert mx[100, 140] == pytest.approx(130)
    assert mx[100, 90] == pytest.approx(130)
    assert my[100, 140] == pytest.approx(100)


def test_lens_is_radially_symmetric():
    mx, my, _ = PointLens(201, 201).maps((100, 100), 30)
    assert mx[100, 150]-100 == pytest.approx(-(mx[100, 50]-100))
    assert mx[100, 150]-100 == pytest.approx(my[150, 100]-100)


def test_black_center_is_optional_and_output_is_finite():
    frame = np.full((120, 160, 3), 180, np.uint8)
    lens = PointLens(160, 120)
    assert lens.apply(frame, (80, 60), 20, shadow=True)[60, 80].sum() == 0
    np.testing.assert_array_equal(lens.apply(frame, (80, 60), 20, shadow=False)[60, 80], [180]*3)
    for center in [(0, 0), (159, 119), (80.5, 60.5)]:
        result = lens.apply(frame, center, 200)
        assert result.shape == frame.shape and result.dtype == np.uint8


def test_hand_spread_increases_strength_and_missing_hand_does_not_jump():
    points = np.zeros((21, 3))
    points[5, :2], points[17, :2] = [0.4, 0.6], [0.6, 0.6]
    points[4, :2], points[8, :2] = [0.48, 0.4], [0.52, 0.4]
    control = LensControl(np.array([480., 270.]))
    assert control.follow(points, 960, 540, 1, 0.2)
    small = control.radius
    points[4, 0], points[8, 0] = 0.3, 0.7
    assert control.follow(points, 960, 540, 2, 0.2)
    assert control.radius > small
    before = control.center.copy(), control.radius
    assert not control.follow(np.full((21, 3), np.nan), 960, 540, 3, 0.2)
    np.testing.assert_array_equal(control.center, before[0])
    assert control.radius == before[1]


def test_full_render_includes_unwarped_inset_and_ui():
    frame = synthetic_scene(960, 540, 0)
    lens = PointLens(960, 540)
    control = LensControl(np.array([480., 270.]))
    result = compose(frame, lens.apply(frame, control.center, control.radius),
                     control, True, True, True, False, 30, 'TEST', 0)
    assert result.shape == (712, 960, 3)
    # Sample the inset away from its border and label.
    import cv2
    expected = cv2.resize(frame, (240, 135))
    np.testing.assert_array_equal(result[54+387+5:54+387+30, 23:48], expected[5:30, 5:30])


@pytest.mark.parametrize('center,radius', [((np.nan, 1), 10), ((1, 1), -5), ((1, 1), np.inf)])
def test_bad_lens_parameters_are_rejected(center, radius):
    with pytest.raises(ValueError):
        PointLens(20, 20).maps(center, radius)
