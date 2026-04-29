"""Finite-difference numerical utilities for the ASF prototype."""

from __future__ import annotations

import numpy as np


def gradient(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return central-difference gradient components d/dx and d/dy."""
    gy, gx = np.gradient(field, dy, dx, edge_order=2)
    return gx, gy


def laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return the second-order finite-difference Laplacian."""
    out = np.zeros_like(field, dtype=float)
    out[1:-1, 1:-1] = (
        (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) / (dx * dx)
        + (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / (dy * dy)
    )
    return out


def smooth_box(field: np.ndarray, passes: int = 1) -> np.ndarray:
    """Apply a conservative 3x3 box smoother to reduce grid-scale noise."""
    result = np.array(field, dtype=float, copy=True)
    for _ in range(max(0, passes)):
        padded = np.pad(result, 1, mode="edge")
        result = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return result


def poisson_jacobi(
    source: np.ndarray,
    dx: float,
    dy: float,
    iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, list[float]]:
    """Solve laplacian(phi)=source with zero Dirichlet boundaries."""
    phi = np.zeros_like(source, dtype=float)
    residuals: list[float] = []
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2.0 * (dx2 + dy2)

    for _ in range(iterations):
        old = phi.copy()
        phi[1:-1, 1:-1] = (
            dy2 * (old[1:-1, 2:] + old[1:-1, :-2])
            + dx2 * (old[2:, 1:-1] + old[:-2, 1:-1])
            - source[1:-1, 1:-1] * dx2 * dy2
        ) / denom
        residual = poisson_residual(phi, source, dx, dy)
        residuals.append(residual)
        if residual <= tolerance:
            break
    return phi, residuals


def poisson_residual(phi: np.ndarray, source: np.ndarray, dx: float, dy: float) -> float:
    """Return RMS residual for laplacian(phi)=source on interior cells."""
    diff = laplacian(phi, dx, dy) - source
    interior = diff[1:-1, 1:-1]
    return float(np.sqrt(np.mean(interior * interior)))
