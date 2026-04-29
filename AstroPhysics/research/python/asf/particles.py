"""Test-particle integration for ASF potential fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fields import acceleration


@dataclass
class ParticleState:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray


def initialize_ring_particles(config: dict[str, Any]) -> ParticleState:
    """Initialize particles on near-circular rings around the grid center."""
    pconf = config["particles"]
    grid = config["grid"]
    count = int(pconf["count"])
    rng = np.random.default_rng(int(config.get("seed", 0)))

    nx = int(grid["nx"])
    ny = int(grid["ny"])
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    radii = np.linspace(float(pconf["radius_min"]), float(pconf["radius_max"]), count)
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    theta += 0.03 * rng.normal(size=count)

    x = cx + radii * np.cos(theta)
    y = cy + radii * np.sin(theta)
    speed = np.sqrt(1.0 / np.maximum(radii, 1.0))
    vx = -speed * np.sin(theta)
    vy = speed * np.cos(theta)
    return ParticleState(x=x, y=y, vx=vx, vy=vy)


def integrate_particles(
    initial: ParticleState,
    potential: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    steps: int,
) -> np.ndarray:
    """Integrate particles in a fixed potential using semi-implicit Euler."""
    ax_grid, ay_grid = acceleration(potential, dx, dy)
    state = ParticleState(initial.x.copy(), initial.y.copy(), initial.vx.copy(), initial.vy.copy())
    history = np.zeros((steps + 1, len(state.x), 4), dtype=float)
    history[0, :, 0] = state.x
    history[0, :, 1] = state.y
    history[0, :, 2] = state.vx
    history[0, :, 3] = state.vy

    for step in range(1, steps + 1):
        ax = sample_grid(ax_grid, state.x, state.y)
        ay = sample_grid(ay_grid, state.x, state.y)
        state.vx += dt * ax
        state.vy += dt * ay
        state.x += dt * state.vx
        state.y += dt * state.vy
        state.x = np.clip(state.x, 1.0, potential.shape[1] - 2.0)
        state.y = np.clip(state.y, 1.0, potential.shape[0] - 2.0)
        history[step, :, 0] = state.x
        history[step, :, 1] = state.y
        history[step, :, 2] = state.vx
        history[step, :, 3] = state.vy
    return history


def sample_grid(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinearly sample a grid using array coordinates."""
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, values.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, values.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    return (
        (1.0 - wx) * (1.0 - wy) * values[y0, x0]
        + wx * (1.0 - wy) * values[y0, x1]
        + (1.0 - wx) * wy * values[y1, x0]
        + wx * wy * values[y1, x1]
    )
