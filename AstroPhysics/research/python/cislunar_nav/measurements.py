"""Range and bearing measurements for cislunar navigation."""

from __future__ import annotations

import numpy as np


def measurement(state: np.ndarray, observer: np.ndarray) -> np.ndarray:
    """Return [range, bearing] from observer to spacecraft."""
    dx = state[0] - observer[0]
    dy = state[1] - observer[1]
    rho = np.sqrt(dx * dx + dy * dy)
    bearing = np.arctan2(dy, dx)
    return np.array([rho, bearing], dtype=float)


def measurement_jacobian(state: np.ndarray, observer: np.ndarray) -> np.ndarray:
    dx = state[0] - observer[0]
    dy = state[1] - observer[1]
    q = max(dx * dx + dy * dy, 1.0e-12)
    rho = np.sqrt(q)
    h = np.zeros((2, 4), dtype=float)
    h[0, 0] = dx / rho
    h[0, 1] = dy / rho
    h[1, 0] = -dy / q
    h[1, 1] = dx / q
    return h


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def geometry_strength(state: np.ndarray, observers: list[np.ndarray]) -> float:
    """Return a normalized observability/geometry score from measurement Jacobians."""
    if not observers:
        return 0.0
    rows = [measurement_jacobian(state, observer)[:, :2] for observer in observers]
    h = np.vstack(rows)
    info = h.T @ h
    eig = np.linalg.eigvalsh(info)
    score = float(np.min(eig) / (np.max(eig) + 1.0e-12))
    return max(0.0, min(1.0, score))


def is_missed(step: int, windows: list[list[int]]) -> bool:
    return any(start <= step <= end for start, end in windows)


def lighting_penalty(step: int, windows: list[list[int]], severity: float) -> float:
    return float(severity) if is_missed(step, windows) else 0.0
