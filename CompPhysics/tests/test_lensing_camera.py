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
                     control, guide=True, inset=True, frozen=False, now=0)
    assert result.shape == (648, 960, 3)
    # Sample the inset away from its border and label.
    import cv2
    expected = cv2.resize(frame, (190, 107), interpolation=cv2.INTER_CUBIC)
    np.testing.assert_array_equal(result[56+413+12:56+413+35, 32:55], expected[12:35, 12:35])


@pytest.mark.parametrize('center,radius', [((np.nan, 1), 10), ((1, 1), -5), ((1, 1), np.inf)])
def test_bad_lens_parameters_are_rejected(center, radius):
    with pytest.raises(ValueError):
        PointLens(20, 20).maps(center, radius)


def gesture(pinched=False, center=0.5, ratio=None):
    points = np.zeros((21, 3))
    points[5, :2], points[17, :2] = [0.4, 0.6], [0.6, 0.6]
    gap = 0.2 * (ratio if ratio is not None else 0.1 if pinched else 1.0)
    points[4, :2], points[8, :2] = [center-gap/2, 0.4], [center+gap/2, 0.4]
    return points


def test_left_pinch_stops_right_motion_on_the_same_frame():
    control = LensControl(np.array([480., 270.]))
    before = control.center.copy(), control.radius
    assert not control.update_hands({'left': gesture(True), 'right': gesture(center=0.8)},
                                    960, 540, 1, 0.1)
    assert control.locked
    np.testing.assert_array_equal(control.center, before[0])
    assert control.radius == before[1]
    for t in [1.1, 1.2, 2.0]:
        control.update_hands({'left': gesture(True), 'right': gesture(center=0.2)},
                             960, 540, t, 0.1)
        assert control.locked
        np.testing.assert_array_equal(control.center, before[0])
        assert control.radius == before[1]


def test_unlock_needs_two_new_pinches_with_observed_releases():
    control = LensControl(np.array([480., 270.]))
    control.update_left(gesture(True), 960, 540, 1)  # locking pinch
    control.update_left(gesture(False), 960, 540, 1.1)
    control.update_left(gesture(True), 960, 540, 1.2)  # first NEW pinch
    assert control.locked
    control.update_left(gesture(True), 960, 540, 1.3)  # holding isn't another tap
    assert control.locked
    control.update_left(gesture(False), 960, 540, 1.4)
    assert control.update_hands({'left': gesture(True), 'right': gesture(center=0.8)},
                                960, 540, 1.5, 0.1)
    assert not control.locked
    assert control.center[0] > 480
    assert control.radius != 115
    # The held second pinch must not immediately lock again.
    control.update_hands({'left': gesture(True)}, 960, 540, 1.6, 0.1)
    assert not control.locked


def test_slow_pinches_do_not_unlock_and_hand_loss_is_not_a_release():
    control = LensControl(np.array([480., 270.]))
    control.update_left(gesture(True), 960, 540, 1)
    control.update_left(gesture(False), 960, 540, 1.1)
    control.update_left(gesture(True), 960, 540, 1.2)
    control.update_left(None, 960, 540, 1.3)
    control.update_left(gesture(True), 960, 540, 1.4)
    assert control.locked
    control.update_left(gesture(False), 960, 540, 1.5)
    control.update_left(gesture(True), 960, 540, 2.2)  # too late; new first tap
    assert control.locked
    control.update_left(gesture(False), 960, 540, 2.3)
    control.update_left(gesture(True), 960, 540, 2.4)
    assert not control.locked


def test_left_never_steers_and_right_never_locks():
    control = LensControl(np.array([480., 270.]))
    assert not control.update_hands({'left': gesture(center=0.8)}, 960, 540, 1, 0.1)
    np.testing.assert_array_equal(control.center, [480, 270])
    assert control.update_hands({'right': gesture(True, center=0.8)}, 960, 540, 2, 0.1)
    assert not control.locked
    assert control.center[0] > 480


def test_pinch_hysteresis_ignores_boundary_jitter():
    control = LensControl(np.array([480., 270.]))
    control.update_left(gesture(ratio=0.34), 960, 540, 1)
    for i, ratio in enumerate([0.36, 0.50, 0.34, 0.45]):
        control.update_left(gesture(ratio=ratio), 960, 540, 1.1+i*0.1)
    assert control.locked and control.unlock_first == -np.inf


@pytest.mark.parametrize('tasks', [False, True])
def test_detector_labels_are_swapped_before_assigning_physical_hand_roles(tasks):
    from types import SimpleNamespace
    from vision.lensing_camera import HandInput
    physical_left, physical_right = gesture(True), gesture(center=0.8)
    observations = [SimpleNamespace(label='Right', landmarks=physical_left),
                    SimpleNamespace(label='Left', landmarks=physical_right)]
    adapter = HandInput.__new__(HandInput)
    adapter.tasks, adapter.timestamp = tasks, 0
    adapter.mp = SimpleNamespace(Image=lambda **kwargs: None,
                                 ImageFormat=SimpleNamespace(SRGB=0))
    adapter.detector = SimpleNamespace(process=lambda frame: None,
                                       detect_for_video=lambda image, timestamp: None)
    adapter.bridge = SimpleNamespace(
        _extract_task_hands=lambda result, mirror: observations,
        _extract_solution_hands=lambda result, mirror: observations,
        _pick_best_by_label=lambda hands, side: next(h for h in hands if h.label.lower() == side))
    roles = adapter.detect(np.zeros((540, 960, 3), np.uint8))
    assert roles['left'] is physical_left
    assert roles['right'] is physical_right



def test_source_star_has_white_core_broad_halo_and_can_move_or_hide():
    from vision.lensing_camera import SpaceScene
    sky = SpaceScene(320, 240)
    original = sky.base.copy()
    bright = sky.render((80, 120))
    assert bright[120, 80].min() >= 245
    assert bright[120, 120].sum() > original[120, 120].sum() + 20
    moved = sky.render((240, 120))
    assert moved[120, 240].min() >= 245
    np.testing.assert_array_equal(sky.render((80, 120), star_visible=False), original)
    np.testing.assert_array_equal(sky.base, original)


def test_lensed_star_bloom_brightens_ring_surroundings_without_filling_black_center():
    from vision.lensing_camera import SpaceScene, star_bloom
    sky = SpaceScene(320, 240)
    lensed = PointLens(320, 240).apply(sky.render((160, 120)), (160, 120), 45)
    before = lensed.copy()
    glowing = star_bloom(lensed)
    assert glowing.dtype == np.uint8 and glowing.shape == lensed.shape
    assert glowing.astype(float).sum() > lensed.astype(float).sum()
    assert glowing[120, 160].max() < 5
    np.testing.assert_array_equal(lensed, before)


@pytest.mark.parametrize('width,height', [(1710, 1074), (1366, 720), (800, 600), (1920, 1080)])
def test_window_fits_usable_display_including_title_bar(width, height):
    from vision.lensing_camera import DisplayArea, fitted_size
    w, h = fitted_size(960, 648, DisplayArea(width, height, scale=2))
    assert w+64 <= width and h+80 <= height
    assert w/h == pytest.approx(960/648, abs=0.003)


def test_retina_ui_is_rasterized_at_double_density():
    from vision.lensing_camera import ui_layout
    import cv2
    frame = synthetic_scene(960, 540, 0)
    control = LensControl(np.array([480., 270.]))
    lens = PointLens(960, 540).apply(frame, control.center, control.radius)
    low = compose(frame, lens, control, False, True, False, 0)
    high = compose(frame, lens, control, False, True, False, 0, render_scale=2)
    assert high.shape == (1296, 1920, 3)
    # UI is drawn with larger glyphs/shapes, not an enlargement of low-res UI.
    assert not np.array_equal(high[:112], cv2.resize(low[:56], (1920, 112), interpolation=cv2.INTER_CUBIC))
    assert ui_layout(960, 540)['camera'] == (392, 12, 478, 44)


def test_retina_clicks_reach_tabs_slider_and_viewport(monkeypatch):
    from vision import lensing_camera as app
    import cv2
    callback = None
    captures, fitted, events = [], [], []
    step = 0
    original_compose = app.compose

    def compose_capture(*args, **kwargs):
        captures.append((args[2].center.copy(), args[2].radius, args[7], args[8]))
        return original_compose(*args, **kwargs)

    def install(window, cb):
        nonlocal callback
        callback = cb

    def click(x, y):
        # HighGUI callbacks use framebuffer pixels, including on resized windows.
        callback(cv2.EVENT_LBUTTONDOWN, x*2, y*2, 0, None)
        callback(cv2.EVENT_LBUTTONUP, x*2, y*2, 0, None)

    def wait(delay):
        nonlocal step
        actions = [lambda: click(430, 28), lambda: click(520, 28),
                   lambda: click(930, 623), lambda: click(885, 28),
                   lambda: click(918, 90), lambda: click(640, 296)]
        if step == len(actions):
            return ord('q')
        actions[step]()
        step += 1
        return -1

    monkeypatch.setattr(app, 'display_area', lambda: app.DisplayArea(1710, 1074, 2))
    monkeypatch.setattr(app, 'fit_window', lambda w, h, screen: fitted.append((w, h, len(events))))
    monkeypatch.setattr(app, 'compose', compose_capture)
    monkeypatch.setattr(cv2, 'namedWindow', lambda *args: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', install)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: events.append('image'))
    monkeypatch.setattr(cv2, 'waitKey', wait)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    assert app.main(['--demo', '--background', 'space']) == 0
    assert fitted == [(960, 648, 1)]  # Fit AFTER first imshow, in logical units.
    assert captures[1][2] == 'camera' and captures[2][2] == 'space'
    assert captures[3][1] > 220
    assert captures[4][3] is True and captures[5][3] is False
    np.testing.assert_array_equal(captures[6][0], [640, 240])
