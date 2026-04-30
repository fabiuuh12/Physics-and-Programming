from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from cislunar_nav.config import load_config
from cislunar_nav.dynamics import cr3bp_rhs, rk4_step, propagate_truth
from cislunar_nav.ekf import generate_measurements, run_filter
from cislunar_nav.measurements import geometry_strength, measurement, measurement_jacobian
from cislunar_nav.risk import navigation_risk


CONFIG = Path("Aerospace/research/experiments/baseline/config.json")


def test_cr3bp_rhs_is_finite() -> None:
    config = load_config(CONFIG)
    state = np.array(config["initial"]["true_state"], dtype=float)
    rhs = cr3bp_rhs(0.0, state, config["dynamics"]["mu"])
    assert rhs.shape == (4,)
    assert np.all(np.isfinite(rhs))


def test_rk4_short_horizon_stays_finite() -> None:
    config = load_config(CONFIG)
    state = np.array(config["initial"]["true_state"], dtype=float)
    for _ in range(20):
        state = rk4_step(state, 0.0, config["dynamics"]["dt"], config["dynamics"]["mu"])
    assert np.all(np.isfinite(state))
    assert np.linalg.norm(state[:2]) < 5.0


def test_measurement_and_jacobian_shape() -> None:
    config = load_config(CONFIG)
    state = np.array(config["initial"]["true_state"], dtype=float)
    observer = np.array(config["sensors"]["observers"][0], dtype=float)
    z = measurement(state, observer)
    h = measurement_jacobian(state, observer)
    assert z.shape == (2,)
    assert h.shape == (2, 4)
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(h))


def test_navigation_risk_monotonic_drivers() -> None:
    config = load_config(CONFIG)
    risk = config["risk"]
    p_small = np.diag([1.0e-6, 1.0e-6, 1.0e-8, 1.0e-8])
    p_large = 10.0 * p_small
    base = navigation_risk(p_small, geometry=0.7, missed=0.0, lighting=0.0, risk_config=risk)
    assert navigation_risk(p_large, geometry=0.7, missed=0.0, lighting=0.0, risk_config=risk) > base
    assert navigation_risk(p_small, geometry=0.1, missed=0.0, lighting=0.0, risk_config=risk) > base
    assert navigation_risk(p_small, geometry=0.7, missed=1.0, lighting=0.0, risk_config=risk) > base
    assert navigation_risk(p_small, geometry=0.7, missed=0.0, lighting=1.0, risk_config=risk) > base


def test_ekf_covariance_symmetric_and_psd() -> None:
    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config["dynamics"]["steps"] = 30
    truth = propagate_truth(np.array(config["initial"]["true_state"], dtype=float), 30, config["dynamics"]["dt"], config["dynamics"]["mu"])
    measurements = generate_measurements(truth, config)
    result = run_filter("baseline", truth, config, "baseline", measurements)
    for covariance in result.covariances:
        assert np.allclose(covariance, covariance.T, atol=1.0e-10)
        assert np.min(np.linalg.eigvalsh(covariance)) > -1.0e-10


def test_adaptive_reduces_to_baseline_when_risk_weights_zero() -> None:
    config = load_config(CONFIG)
    config = copy.deepcopy(config)
    config["dynamics"]["steps"] = 40
    for key in ["w_r", "w_v", "w_g", "w_m", "w_l"]:
        config["risk"][key] = 0.0
    truth = propagate_truth(np.array(config["initial"]["true_state"], dtype=float), 40, config["dynamics"]["dt"], config["dynamics"]["mu"])
    measurements = generate_measurements(truth, config)
    baseline = run_filter("baseline", truth, config, "baseline", measurements)
    adaptive = run_filter("adaptive_R", truth, config, "adaptive_R", measurements)
    assert np.allclose(baseline.estimates, adaptive.estimates)
    assert np.allclose(baseline.covariances, adaptive.covariances)


def test_geometry_strength_is_finite_and_bounded() -> None:
    config = load_config(CONFIG)
    state = np.array(config["initial"]["true_state"], dtype=float)
    observers = [np.array(item, dtype=float) for item in config["sensors"]["observers"]]
    strength = geometry_strength(state, observers)
    assert np.isfinite(strength)
    assert 0.0 <= strength <= 1.0
