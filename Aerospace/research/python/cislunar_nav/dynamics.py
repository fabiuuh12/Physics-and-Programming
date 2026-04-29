"""Planar circular restricted three-body problem dynamics."""

from __future__ import annotations

import numpy as np


def cr3bp_rhs(_t: float, state: np.ndarray, mu: float) -> np.ndarray:
    """Return planar CR3BP derivative for [x, y, vx, vy]."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu) ** 2 + y * y)
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y * y)
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / (r1**3) - mu * (x - 1.0 + mu) / (r2**3)
    ay = -2.0 * vx + y - (1.0 - mu) * y / (r1**3) - mu * y / (r2**3)
    return np.array([vx, vy, ax, ay], dtype=float)


def rk4_step(state: np.ndarray, t: float, dt: float, mu: float) -> np.ndarray:
    k1 = cr3bp_rhs(t, state, mu)
    k2 = cr3bp_rhs(t + 0.5 * dt, state + 0.5 * dt * k1, mu)
    k3 = cr3bp_rhs(t + 0.5 * dt, state + 0.5 * dt * k2, mu)
    k4 = cr3bp_rhs(t + dt, state + dt * k3, mu)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_truth(initial_state: np.ndarray, steps: int, dt: float, mu: float) -> np.ndarray:
    states = np.zeros((steps + 1, 4), dtype=float)
    states[0] = initial_state
    t = 0.0
    for idx in range(steps):
        states[idx + 1] = rk4_step(states[idx], t, dt, mu)
        t += dt
    return states


def numerical_state_transition(state: np.ndarray, t: float, dt: float, mu: float) -> np.ndarray:
    """Finite-difference state transition Jacobian for one RK4 step."""
    n = len(state)
    phi = np.zeros((n, n), dtype=float)
    eps = 1.0e-6
    for col in range(n):
        delta = np.zeros(n, dtype=float)
        scale = max(1.0, abs(float(state[col])))
        delta[col] = eps * scale
        plus = rk4_step(state + delta, t, dt, mu)
        minus = rk4_step(state - delta, t, dt, mu)
        phi[:, col] = (plus - minus) / (2.0 * delta[col])
    return phi
