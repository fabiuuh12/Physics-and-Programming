"""Physics-kernel, tracking-loss, and camera-free particle display checks."""
import numpy as np
import pytest

from vision.body_particle_model import (
    CM_PER_PIXEL, Transport, charged_range, compton_event, compton_mu, photon_mu,
)


def test_charged_ranges_are_monotone_and_resolve_different_scales():
    alpha = charged_range('alpha', np.geomspace(2, 10, 30))
    beta = charged_range('beta', np.geomspace(.1, 3, 30))
    assert np.all(np.diff(alpha) > 0) and np.all(np.diff(beta) > 0)
    assert charged_range('alpha', 5) == pytest.approx(.0035)
    assert charged_range('beta', 1) == pytest.approx(.412)
    assert charged_range('alpha', 5) < CM_PER_PIXEL
    assert charged_range('beta', 1) > 5*CM_PER_PIXEL


def test_alpha_stops_at_its_subpixel_material_range_and_counts_once():
    mask = np.zeros((20, 40), dtype=bool)
    mask[:, 10:30] = True
    engine = Transport('alpha', 5, seed=1)
    particle = engine.emit((1, 10))
    engine.advance(particle, 40, mask)
    assert not particle.alive and particle.energy == 0
    assert particle.pos[0] == pytest.approx(10+.0035/CM_PER_PIXEL)
    assert engine.counts['stopped'] == 1
    engine.finish(particle, 'stopped')
    assert engine.counts['stopped'] == 1


def test_beta_loses_energy_scatters_and_stops():
    mask = np.ones((300, 300), dtype=bool)
    engine = Transport('beta', 1, seed=2)
    particle = engine.emit((100, 150))
    engine.advance(particle, 1, mask)
    assert 0 < particle.energy < 1 and particle.scattered
    engine.advance(particle, 100, mask)
    assert not particle.alive and particle.energy == 0
    assert engine.counts['stopped'] == 1
    assert np.isfinite(particle.pos).all()


def test_gamma_coefficients_match_nist_water_samples():
    assert photon_mu(.1) == pytest.approx(.1707)
    assert photon_mu(.5) == pytest.approx(.09687)
    assert photon_mu(1) == pytest.approx(.07072)
    assert 0 < compton_mu(.662) < .2


def test_gamma_unscattered_survival_matches_exponential_attenuation():
    # A static 10 cm model slab, independent of a webcam or silhouette geometry.
    mask = np.zeros((10, 30), dtype=bool)
    mask[:, 10:20] = True
    engine = Transport('gamma', 1, seed=37, cm_per_pixel=1)
    n = 600
    for _ in range(n):
        particle = engine.emit((1, 5))
        engine.advance(particle, 50, mask)
        engine.particles.clear()
    expected = np.exp(-photon_mu(1)*10)
    assert engine.counts['through']/n == pytest.approx(expected, abs=.06)


def test_compton_energy_obeys_angle_relation_and_never_increases():
    rng = np.random.default_rng(3)
    for energy in (.05, .1, .662, 1):
        for _ in range(20):
            outgoing, angle = compton_event(energy, rng)
            expected = energy/(1+energy/.51099895*(1-np.cos(angle)))
            assert outgoing == pytest.approx(expected)
            assert 0 < outgoing <= energy


def test_empty_space_does_not_deposit_energy_or_count_as_transmission():
    engine = Transport('beta', 1, seed=9)
    particle = engine.emit((1, 10))
    engine.advance(particle, 50, np.zeros((20, 40), dtype=bool))
    assert not particle.touched and particle.energy == 1
    assert engine.counts['missed'] == 1 and engine.counts['through'] == 0


def test_reset_clears_old_comparison_and_selects_new_default_energy():
    engine = Transport('alpha', 5)
    engine.emit((1, 10))
    engine.counts['stopped'] = 6
    engine.reset('gamma')
    assert engine.kind == 'gamma' and engine.energy == pytest.approx(.662)
    assert not engine.particles and not any(engine.counts.values())


@pytest.mark.parametrize('value', [0, -1, np.inf, np.nan])
def test_invalid_energy_rejected(value):
    with pytest.raises(ValueError):
        Transport('alpha', value)


def test_silhouette_loss_and_timeout_clear_target():
    pytest.importorskip('cv2')
    from vision.body_particle_camera import Silhouette
    target = Silhouette()
    mask = np.ones((30, 30), np.float32)
    target.update(mask, 1)
    assert target.mask(mask.shape, 1.1).all()
    assert not target.mask(mask.shape, 1.4).any()
    target.update(mask, 2)
    target.update(None, 2.1)
    assert not target.mask(mask.shape, 2.1).any()
    target.update(np.full_like(mask, np.nan), 3)
    assert not target.mask(mask.shape, 3).any()


def test_retina_particle_render_preserves_camera_input():
    pytest.importorskip('cv2')
    from vision.body_particle_camera import demo_scene, render
    frame, probability = demo_scene(0)
    original = frame.copy()
    engine = Transport('gamma', .662, seed=4)
    image = render(frame, probability > .5, engine, (45, 270), True,
                   False, False, True, 2, demo=True)
    assert image.shape == (1364, 1920, 3)
    np.testing.assert_array_equal(frame, original)


def test_retina_camera_controls_select_energy_type_slow_motion_and_aim(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    from vision import body_particle_camera as app
    callback = None
    captures, fitted = [], []
    step = 0
    original_render = app.render

    def record(*args):
        captures.append((args[2].kind, args[2].energy, args[3].copy(), args[6], args[7]))
        return original_render(*args)

    def install(window, cb):
        nonlocal callback
        callback = cb

    def click(x, y):
        callback(cv2.EVENT_LBUTTONDOWN, x*2, y*2, 0, None)
        callback(cv2.EVENT_LBUTTONUP, x*2, y*2, 0, None)

    def wait(delay):
        nonlocal step
        actions = [lambda: click(480, 28), lambda: click(932, 623),
                   lambda: click(565, 28), lambda: click(880, 661),
                   lambda: click(880, 28), lambda: click(918, 90),
                   lambda: click(200, 330)]
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
    assert fitted == [(960, 682, 'Particle Lab')]
    assert captures[1][0] == 'beta' and captures[2][1] > 2.9
    assert captures[3][0] == 'gamma' and captures[3][1] == pytest.approx(.662)
    assert captures[4][3] is True
    assert captures[5][4] is True and captures[6][4] is False
    np.testing.assert_allclose(captures[7][2], [200, 274])
