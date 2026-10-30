"""Physical invariants, event conservation and gesture checks for Particle Studio."""
import math
import numpy as np
import pytest

from vision.particle_studio_model import (
    DETECTOR_RADIUS, DISPLAY_C, MAGNETIC_K, Studio, Track, circle_crossings, toy_collision,
)


def test_exact_arc_matches_quarter_circle_and_preserves_energy():
    p, b, mass = .6, 1.3, .105658
    track = Track((0, 0), (p, 0), mass, 1)
    radius, energy = track.radius(b), track.energy
    omega = MAGNETIC_K*b*DISPLAY_C/energy
    track.advance(math.pi/(2*omega), b)
    np.testing.assert_allclose(track.pos, [radius, -radius], atol=1e-12)
    np.testing.assert_allclose(track.momentum, [0, -p], atol=1e-12)
    assert track.energy == pytest.approx(energy, rel=1e-14)


def test_opposite_charges_and_field_reversal_mirror_tracks():
    positive = Track((0, 0), (.6, 0), .1, 1)
    negative = Track((0, 0), (.6, 0), .1, -1)
    reversed_field = Track((0, 0), (.6, 0), .1, 1)
    for _ in range(150):
        positive.advance(.02, 1.2)
        negative.advance(.02, 1.2)
        reversed_field.advance(.02, -1.2)
    np.testing.assert_allclose(positive.pos*[1, -1], negative.pos, atol=1e-12)
    np.testing.assert_allclose(reversed_field.pos, negative.pos, atol=1e-12)
    assert np.linalg.norm(positive.momentum) == pytest.approx(.6, rel=1e-12)


def test_straight_tracks_when_neutral_or_field_is_zero():
    for charge, field in [(0, 3), (1, 0)]:
        track = Track((1, 2), (.4, .3), 0, charge)
        track.advance(1, field)
        np.testing.assert_allclose(track.pos, [1+DISPLAY_C*.8, 2+DISPLAY_C*.6])
        assert math.isinf(track.radius(field))


def test_magnetic_radius_scales_with_momentum_and_inverse_field():
    track = Track((0, 0), (.3, 0), .1, 1)
    assert track.radius(1) == pytest.approx(.3/MAGNETIC_K)
    assert track.radius(2) == pytest.approx(track.radius(1)/2)


@pytest.mark.parametrize('energy', [2., 4., 12.])
def test_toy_collision_conserves_energy_and_momentum(energy):
    for seed in range(8):
        event = toy_collision(energy, np.random.default_rng(seed), vertex=(.2, -.3))
        assert event.outgoing_energy == pytest.approx(energy, abs=1e-12)
        np.testing.assert_allclose(event.momentum_residual, [0, 0], atol=1e-14)
        assert len(event.momenta) == 6


def test_collision_launch_has_balanced_charge_and_keeps_original_vertex_accounting():
    studio = Studio(5)
    studio.collision_energy = 4.
    studio.collide()
    for _ in range(25):
        studio.step(.03)
    assert studio.events == 1 and len(studio.tracks) == 6
    assert sum(t.charge for t in studio.tracks) == 0
    assert studio.last_event.outgoing_energy == pytest.approx(4)
    np.testing.assert_allclose(studio.last_event.momentum_residual, [0, 0], atol=1e-14)
    # The event snapshot must not be mutated by later magnetic deflection.
    for _ in range(100):
        studio.step(.03)
    assert studio.last_event.outgoing_energy == pytest.approx(4)
    np.testing.assert_allclose(studio.last_event.momentum_residual, [0, 0], atol=1e-14)


def test_below_threshold_event_is_rejected():
    with pytest.raises(ValueError):
        toy_collision(.1, np.random.default_rng())


def test_detector_hits_lie_on_the_correct_layer():
    points = circle_crossings(np.array([-2., 0.]), np.array([2., 0.]), 1)
    assert len(points) == 2
    for point in points:
        assert np.linalg.norm(point) == pytest.approx(1)
    assert not circle_crossings(np.array([2., 2.]), np.array([3., 2.]), 1)


def hand(x, closed=True):
    p = np.zeros((21, 3))
    p[5, :2], p[17, :2] = [.4, .6], [.6, .6]
    gap = .02 if closed else .25
    p[4, :2], p[8, :2] = [x-gap/2, .5], [x+gap/2, .5]
    p[0, :2] = [x, .7]
    return p


def test_right_pinch_releases_at_limited_rate_and_loss_stops_release():
    pytest.importorskip('cv2')
    from vision.particle_studio import Gestures
    studio, controls = Studio(2), Gestures()
    controls.update({'right': hand(.6)}, 'tracks', studio, 960, 540, 1, .03)
    assert len(studio.tracks) == 2
    controls.update({'right': hand(.6)}, 'tracks', studio, 960, 540, 1.1, .03)
    assert len(studio.tracks) == 2
    controls.update({}, 'tracks', studio, 960, 540, 1.5, .03)
    assert len(studio.tracks) == 2 and not controls.points


def test_collision_gesture_requires_apart_then_together_and_release_to_rearm():
    pytest.importorskip('cv2')
    from vision.particle_studio import Gestures
    studio, controls = Studio(2), Gestures()
    controls.update({'left': hand(.48), 'right': hand(.52)}, 'collisions', studio, 960, 540, 1, .03)
    assert studio.pending is None
    controls.update({'left': hand(.25), 'right': hand(.75)}, 'collisions', studio, 960, 540, 1.1, .03)
    assert controls.armed
    initial_field = studio.magnetic_field
    controls.update({'left': hand(.47), 'right': hand(.53)}, 'collisions', studio, 960, 540, 1.2, .03)
    assert studio.pending is not None and controls.latched
    assert studio.magnetic_field == initial_field
    studio.pending = None
    controls.update({'left': hand(.25), 'right': hand(.75)}, 'collisions', studio, 960, 540, 1.3, .03)
    assert not controls.armed and studio.pending is None
    controls.update({'left': hand(.25, False), 'right': hand(.75, False)}, 'collisions', studio, 960, 540, 1.4, .03)
    assert not controls.latched


def test_hand_loss_cancels_armed_collision():
    pytest.importorskip('cv2')
    from vision.particle_studio import Gestures
    studio, controls = Studio(2), Gestures()
    controls.update({'left': hand(.25), 'right': hand(.75)}, 'collisions', studio, 960, 540, 1, .03)
    controls.update({}, 'collisions', studio, 960, 540, 1.1, .03)
    controls.update({'left': hand(.48), 'right': hand(.52)}, 'collisions', studio, 960, 540, 1.2, .03)
    assert studio.pending is None


def test_coordinate_mapping_is_centered_y_up_and_limited_to_detector():
    pytest.importorskip('cv2')
    from vision.particle_studio import to_world
    np.testing.assert_allclose(to_world((480, 270), 960, 540), [0, 0])
    assert to_world((480, 100), 960, 540)[1] > 0
    assert np.linalg.norm(to_world((0, 0), 960, 540)) < DETECTOR_RADIUS


def test_retina_tabs_sliders_reverse_field_and_scene_release(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    from vision import particle_studio as app
    callback = None
    captures, fitted = [], []
    step = 0
    original_render = app.render

    def record(*args, **kwargs):
        studio = args[1]
        captures.append((args[2], studio.magnetic_field, studio.collision_energy,
                         len(studio.tracks), args[5]))
        image = original_render(*args, **kwargs)
        assert image.shape == (1384, 1920, 3)
        return image

    def install(window, cb):
        nonlocal callback
        callback = cb

    def click(x, y):
        callback(cv2.EVENT_LBUTTONDOWN, x*2, y*2, 0, None)
        callback(cv2.EVENT_LBUTTONUP, x*2, y*2, 0, None)

    def wait(delay):
        nonlocal step
        actions = [lambda: click(540, 28), lambda: click(860, 623),
                   lambda: click(932, 668), lambda: click(880, 28),
                   lambda: click(918, 90), lambda: click(420, 28),
                   lambda: click(400, 668), lambda: click(480, 326)]
        if step == len(actions):
            return ord('q')
        actions[step]()
        step += 1
        return -1

    monkeypatch.setattr(app, 'display_area', lambda: app.DisplayArea(1710, 1074, 2))
    monkeypatch.setattr(app, 'fit_window', lambda w, h, screen, name: fitted.append((w, h, name)))
    monkeypatch.setattr(app, 'render', record)
    monkeypatch.setattr(cv2, 'namedWindow', lambda *args: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', install)
    monkeypatch.setattr(cv2, 'imshow', lambda *args: None)
    monkeypatch.setattr(cv2, 'waitKey', wait)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    monkeypatch.setattr(cv2, 'destroyAllWindows', lambda: None)
    assert app.main(['--demo']) == 0
    assert fitted == [(960, 692, 'Particle Studio')]
    assert captures[1][0] == 'collisions' and captures[2][1] < 0
    assert captures[3][2] == pytest.approx(12)
    assert captures[4][4] is True and captures[5][4] is False
    assert captures[6][0] == 'tracks'
    assert captures[7][1] == pytest.approx(-3)
    assert captures[8][3] == captures[7][3]+2
