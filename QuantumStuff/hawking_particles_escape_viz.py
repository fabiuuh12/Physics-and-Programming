"""
Stylized animation of particles escaping a black hole via Hawking radiation.

Scene elements:
- Curved background field hinting at spacetime curvature.
- Dark event horizon plus photon ring glow.
- Repeated bursts of discrete particle tracers that arc outward
  with slight bends before fading into space.

Run:
    python3 hawking_particles_escape_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")

# Scene / grid settings
EXTENT = 3.2
RESOLUTION = 720
FIGSIZE = (6, 6)

# Horizon parameters
HORIZON_RADIUS = 0.55
PHOTON_RING_WIDTH = 0.05
GLOW_WIDTH = 0.18

# Particle emission parameters
NUM_PARTICLES = 420
EMISSION_PERIOD = 2.6
PARTICLE_SPEED = 0.65
BEND_RATE = 1.2
DRIFT_RATE = 0.5
FADE_DISTANCE = 2.3
PARTICLE_BASE_SIZE = 25.0

RNG = np.random.default_rng(7)
EMISSION_PHASE = RNG.uniform(0.0, EMISSION_PERIOD, NUM_PARTICLES)
BASE_ANGLES = np.linspace(-np.pi, np.pi, NUM_PARTICLES, endpoint=False) + RNG.uniform(
    -0.35, 0.35, NUM_PARTICLES
)
CURVATURE = RNG.uniform(0.15, 0.45, NUM_PARTICLES)
SPIN_DIR = RNG.choice([-1.0, 1.0], size=NUM_PARTICLES)
COLOR_SEED = RNG.uniform(0.15, 1.0, NUM_PARTICLES)


def make_grid(resolution: int = RESOLUTION):
    axis = np.linspace(-EXTENT, EXTENT, resolution)
    xx, yy = np.meshgrid(axis, axis)
    r = np.sqrt(xx * xx + yy * yy) + 1e-9
    theta = np.arctan2(yy, xx)
    return axis, xx, yy, r, theta


AXIS, XX, YY, R, THETA = make_grid()


def generate_field(t: float) -> np.ndarray:
    """Compute the luminous background representing warped spacetime."""
    gradient = np.exp(-(R / EXTENT) ** 1.2)
    swirl = np.exp(-((R - (HORIZON_RADIUS + 0.35)) / 0.9) ** 2) * (
        0.5 + 0.5 * np.cos(3.0 * THETA - 0.4 * t)
    )
    ripples = np.exp(-((R - (HORIZON_RADIUS + 1.3)) / 1.5) ** 2) * (
        0.5 + 0.5 * np.sin(2.0 * THETA + 0.3 * t)
    )
    photon_ring = np.exp(-((R - (HORIZON_RADIUS + 0.08)) / PHOTON_RING_WIDTH) ** 2)
    glow = np.exp(-((R - HORIZON_RADIUS) / GLOW_WIDTH) ** 2)
    texture = np.sin(2.5 * XX + 1.7 * YY + 0.25 * t)
    texture = np.clip(texture, 0.0, None) ** 2

    field = (
        0.2 * texture * gradient
        + 0.45 * gradient
        + 1.3 * glow
        + 2.3 * photon_ring
        + 0.7 * swirl
        + 0.4 * ripples
    )

    field[R < HORIZON_RADIUS] = 0.0
    field = np.clip(field, 0.0, 1.0)
    return field


def particle_state(t: float):
    """Return particle offsets, colors, and sizes for time t."""
    # Time since each particle's most recent emission.
    time_since_emit = (t - EMISSION_PHASE) % EMISSION_PERIOD
    radius = HORIZON_RADIUS + PARTICLE_SPEED * time_since_emit
    mask = (radius >= HORIZON_RADIUS) & (radius <= EXTENT)
    if not np.any(mask):
        empty = np.empty((0,))
        return (
            np.empty((0, 2)),
            np.empty((0, 4)),
            empty,
        )

    radius = radius[mask]
    phase = time_since_emit[mask]
    base_angle = BASE_ANGLES[mask]
    curvature = CURVATURE[mask]
    spin = SPIN_DIR[mask]
    color_seed = COLOR_SEED[mask]

    bend = curvature * (1.0 - np.exp(-BEND_RATE * (radius - HORIZON_RADIUS)))
    theta = base_angle + spin * bend + 0.08 * np.sin(DRIFT_RATE * t + 3.0 * color_seed)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    offsets = np.column_stack([x, y])

    birth = np.exp(-((phase) / 0.28) ** 2)
    fade = np.exp(-(radius - HORIZON_RADIUS) / FADE_DISTANCE)
    intensity = np.clip(0.35 + birth + 1.1 * fade, 0.0, 1.4)
    sizes = PARTICLE_BASE_SIZE * (0.8 + 2.2 * intensity)
    color_vals = np.clip(color_seed * 0.6 + 0.4 * birth, 0.0, 1.0)

    cmap = plt.get_cmap("inferno")
    colors = cmap(color_vals)
    colors[:, 3] = np.clip(0.25 + 0.6 * intensity, 0.1, 1.0)

    return offsets, colors, sizes


def run_animation():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("black")
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect("equal")
    ax.axis("off")

    background = ax.imshow(
        generate_field(0.0),
        cmap="magma",
        origin="lower",
        extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
        interpolation="bilinear",
    )

    particle_scatter = ax.scatter(
        [],
        [],
        s=[],
        c=[],
        linewidths=0.0,
    )

    ax.set_title("Hawking Particle Escape", pad=16, color="white")

    def update(frame: int):
        t = frame * 0.15
        background.set_data(generate_field(t))
        offsets, colors, sizes = particle_state(t)
        particle_scatter.set_offsets(offsets)
        particle_scatter.set_facecolors(colors)
        particle_scatter.set_edgecolors(colors)
        particle_scatter.set_sizes(sizes)
        return (background, particle_scatter)

    anim = FuncAnimation(
        fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )
    return anim


def main():
    anim = run_animation()
    plt.show()


if __name__ == "__main__":
    main()
