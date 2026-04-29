"""Baseline and risk-adaptive EKF implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics import numerical_state_transition, rk4_step
from .measurements import geometry_strength, is_missed, lighting_penalty, measurement, measurement_jacobian, wrap_angle
from .risk import clipped_risk, covariance_sigmas, navigation_risk


@dataclass
class FilterResult:
    name: str
    estimates: np.ndarray
    covariances: np.ndarray
    risks: np.ndarray
    errors: np.ndarray
    geometry: np.ndarray
    missed: np.ndarray
    lighting: np.ndarray


def run_filter(
    name: str,
    truth: np.ndarray,
    config: dict[str, Any],
    mode: str,
    measurements: np.ndarray | None = None,
) -> FilterResult:
    """Run one EKF variant. mode is baseline, adaptive_R, or adaptive_Q."""
    dynamics = config["dynamics"]
    filt = config["filter"]
    sensor = config["sensors"]
    risk_conf = config["risk"]
    mu = float(dynamics["mu"])
    dt = float(dynamics["dt"])
    steps = int(dynamics["steps"])
    observers = [np.array(item, dtype=float) for item in sensor["observers"]]
    primary_observer = observers[0]
    rng = np.random.default_rng(int(config["seed"]))

    x = np.array(config["initial"]["estimate_state"], dtype=float)
    p = np.diag(np.array(filt["P0_diag"], dtype=float))
    q0 = np.diag(np.array(filt["Q0_diag"], dtype=float))
    r0 = np.diag(np.array(filt["R0_diag"], dtype=float))

    estimates = np.zeros((steps + 1, 4), dtype=float)
    covariances = np.zeros((steps + 1, 4, 4), dtype=float)
    risks = np.zeros(steps + 1, dtype=float)
    errors = np.zeros((steps + 1, 4), dtype=float)
    geoms = np.zeros(steps + 1, dtype=float)
    missed_flags = np.zeros(steps + 1, dtype=float)
    lighting_values = np.zeros(steps + 1, dtype=float)
    estimates[0] = x
    covariances[0] = p
    errors[0] = x - truth[0]

    for step in range(1, steps + 1):
        t = (step - 1) * dt
        phi = numerical_state_transition(x, t, dt, mu)
        geom_pre = geometry_strength(x, observers)
        miss = 1.0 if is_missed(step, sensor.get("missed_windows", [])) else 0.0
        light = lighting_penalty(step, sensor.get("lighting_windows", []), float(sensor.get("lighting_severity", 0.0)))
        pre_risk = navigation_risk(p, geom_pre, miss, light, risk_conf)
        risk_factor = clipped_risk(pre_risk, risk_conf)

        q = q0 * (1.0 + float(filt["alpha"]) * risk_factor) if mode == "adaptive_Q" else q0
        x_pred = rk4_step(x, t, dt, mu)
        p_pred = phi @ p @ phi.T + q

        geom = geometry_strength(x_pred, observers)
        miss = 1.0 if is_missed(step, sensor.get("missed_windows", [])) else 0.0
        light = lighting_penalty(step, sensor.get("lighting_windows", []), float(sensor.get("lighting_severity", 0.0)))
        risk = navigation_risk(p_pred, geom, miss, light, risk_conf)
        risk_factor = clipped_risk(risk, risk_conf)

        if miss:
            x = x_pred
            p = p_pred
        else:
            r = r0 * (1.0 + float(filt["beta"]) * risk_factor) if mode == "adaptive_R" else r0
            z = measurements[step] if measurements is not None else noisy_measurement(truth[step], primary_observer, r0, rng)
            h = measurement(x_pred, primary_observer)
            innovation = z - h
            innovation[1] = wrap_angle(innovation[1])
            jac = measurement_jacobian(x_pred, primary_observer)
            s = jac @ p_pred @ jac.T + r
            k = p_pred @ jac.T @ np.linalg.inv(s)
            x = x_pred + k @ innovation
            i = np.eye(4)
            p = (i - k @ jac) @ p_pred @ (i - k @ jac).T + k @ r @ k.T
            p = 0.5 * (p + p.T)

        estimates[step] = x
        covariances[step] = p
        risks[step] = risk
        errors[step] = x - truth[step]
        geoms[step] = geom
        missed_flags[step] = miss
        lighting_values[step] = light

    return FilterResult(name, estimates, covariances, risks, errors, geoms, missed_flags, lighting_values)


def noisy_measurement(state: np.ndarray, observer: np.ndarray, r_cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.multivariate_normal(np.zeros(2), r_cov)
    z = measurement(state, observer) + noise
    z[1] = wrap_angle(z[1])
    return z


def generate_measurements(truth: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    sensor = config["sensors"]
    filt = config["filter"]
    observer = np.array(sensor["observers"][0], dtype=float)
    r0 = np.diag(np.array(filt["R0_diag"], dtype=float))
    rng = np.random.default_rng(int(config["seed"]) + 1000)
    values = np.zeros((len(truth), 2), dtype=float)
    for idx, state in enumerate(truth):
        values[idx] = noisy_measurement(state, observer, r0, rng)
    return values


def summarize_result(result: FilterResult) -> dict[str, float | str]:
    pos_error = np.linalg.norm(result.errors[:, :2], axis=1)
    vel_error = np.linalg.norm(result.errors[:, 2:], axis=1)
    sigma_r = np.array([covariance_sigmas(p)[0] for p in result.covariances])
    sigma_v = np.array([covariance_sigmas(p)[1] for p in result.covariances])
    return {
        "filter": result.name,
        "final_position_error": float(pos_error[-1]),
        "rms_position_error": float(np.sqrt(np.mean(pos_error * pos_error))),
        "max_position_error": float(np.max(pos_error)),
        "final_velocity_error": float(vel_error[-1]),
        "rms_velocity_error": float(np.sqrt(np.mean(vel_error * vel_error))),
        "mean_risk": float(np.mean(result.risks)),
        "max_risk": float(np.max(result.risks)),
        "final_sigma_r": float(sigma_r[-1]),
        "final_sigma_v": float(sigma_v[-1]),
        "missed_fraction": float(np.mean(result.missed)),
    }
