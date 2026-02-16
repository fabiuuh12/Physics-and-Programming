

"""
dark_matter_viz.py — Stylized dark matter halo visualization.

Conceptual only:
- Visible galaxy (bright disk + stars).
- Surrounding dark matter halo as a smooth, semi-transparent glow.
- Faint "test particles" orbiting, showing the extended gravity of the halo.
- Very light gravitational-lensing-style shimmer near the center.

This is NOT a physical N-body sim — it's an intuition/visual piece.

Run:
    python dark_matter_viz.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")

# Canvas
FIGSIZE = (8, 8)
X_MIN, X_MAX = -8.0, 8.0
Y_MIN, Y_MAX = -8.0, 8.0

# Visible galaxy parameters (disc + stars)
NUM_VISIBLE_STARS = 260
DISC_RADIUS = 3.2
DISC_COLOR = "#ffd8a6"
CORE_COLOR = "#ffeccd"

# Dark matter halo parameters (stylized)
HALO_RADIUS = 7.0
HALO_CORE_RADIUS = 2.0
HALO_COLOR = "#6f7bff"
HALO_ALPHA = 0.42

# Test particles (to show "flat" rotation feel)
NUM_TRACERS = 80
TRACER_COLOR = "#9be7ff"
TRACER_SIZE = 26

# Rotation speeds (not physical, just to hint: inner/outer similar)
OMEGA_INNER = 0.02
OMEGA_OUTER = 0.016

# Weak shimmering lensing
LENSING_STRENGTH = 0.03


@dataclass
class Tracer:
    radius: float
    angle: float
    omega: float

    def step(self):
        self.angle += self.omega

    @property
    def pos(self) -> tuple[float, float]:
        return (
            self.radius * math.cos(self.angle),
            self.radius * math.sin(self.angle),
        )


def make_visible_stars():
    xs = []
    ys = []
    cs = []

    # compact bright core
    for _ in range(80):
        r = random.uniform(0.0, 0.6)
        theta = random.uniform(0, 2 * math.pi)
        xs.append(r * math.cos(theta))
        ys.append(r * math.sin(theta))
        cs.append(CORE_COLOR)

    # disc stars with slight spiral hint
    for _ in range(NUM_VISIBLE_STARS):
        u = random.random()
        r = (u ** 0.5) * DISC_RADIUS  # more in the center
        theta = random.uniform(0, 2 * math.pi)
        # add gentle two-arm spiral modulation
        theta += 0.4 * math.sin(2 * r)
        x = r * math.cos(theta)
        y = r * math.sin(theta) * 0.7  # flatten vertically
        xs.append(x)
        ys.append(y)
        cs.append(DISC_COLOR)

    return np.array(xs), np.array(ys), cs


def make_tracers() -> List[Tracer]:
    tracers: List[Tracer] = []
    for _ in range(NUM_TRACERS):
        r = random.uniform(DISC_RADIUS * 0.8, HALO_RADIUS * 0.95)
        angle = random.uniform(0, 2 * math.pi)
        # inner and outer radii get similar-ish angular speeds
        if r < 4.0:
            omega = random.uniform(OMEGA_INNER * 0.9, OMEGA_INNER * 1.2)
        else:
            omega = random.uniform(OMEGA_OUTER * 0.9, OMEGA_OUTER * 1.2)
        tracers.append(Tracer(radius=r, angle=angle, omega=omega))
    return tracers


def build_scene():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#02040a")

    # Label
    ax.text(
        0.0,
        Y_MIN + 0.5,
        "Galaxy + Dark Matter Halo (stylized)",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#9fb4ff",
    )

    return fig, ax


def halo_intensity_grid():
    """
    Precompute a smooth halo intensity map on a grid for imshow.
    Peaks at center, falls off slowly, extended beyond visible disc.
    """
    xs = np.linspace(X_MIN, X_MAX, 600)
    ys = np.linspace(Y_MIN, Y_MAX, 600)
    X, Y = np.meshgrid(xs, ys)
    r = np.sqrt(X**2 + Y**2) + 1e-6

    # Base profile: high in center, slow falloff (NFW-ish vibe but simple)
    inner = 1.0 / (1.0 + (r / HALO_CORE_RADIUS) ** 2)
    outer = 1.0 / (1.0 + (r / HALO_RADIUS) ** 3)
    halo = (inner + 1.8 * outer)

    # Normalize and clip
    halo /= halo.max()
    return xs, ys, halo


def lensing_offset(x: np.ndarray, y: np.ndarray, t: float):
    """
    Tiny wobble around the center to fake gravitational lensing shimmer.
    """
    r2 = x * x + y * y + 1e-6
    factor = LENSING_STRENGTH * (1.0 + 0.3 * math.sin(0.7 * t))
    dx = factor * x / r2
    dy = factor * y / r2
    return dx, dy


def run_animation():
    fig, ax = build_scene()

    # Precompute halo field
    xs, ys, halo = halo_intensity_grid()
    img = ax.imshow(
        halo,
        extent=(X_MIN, X_MAX, Y_MIN, Y_MAX),
        origin="lower",
        cmap="magma",
        alpha=HALO_ALPHA,
        zorder=1,
        interpolation="bilinear",
    )

    # Visible galaxy stars
    star_x, star_y, star_colors = make_visible_stars()
    stars_scatter = ax.scatter(
        star_x,
        star_y,
        s=8,
        c=star_colors,
        alpha=0.95,
        edgecolors="none",
        zorder=4,
    )

    # Tracer particles (orbits in the halo)
    tracers = make_tracers()
    tracer_scatter = ax.scatter(
        [],
        [],
        s=TRACER_SIZE,
        c=TRACER_COLOR,
        alpha=0.9,
        edgecolors="none",
        zorder=5,
    )

    frame_counter = {"value": 0}

    def update(_frame: int):
        frame_counter["value"] += 1
        t = frame_counter["value"]

        # Step tracers
        xs_t = []
        ys_t = []
        for tr in tracers:
            tr.step()
            x, y = tr.pos
            xs_t.append(x)
            ys_t.append(y)

        if xs_t:
            coords = np.column_stack((xs_t, ys_t))
            tracer_scatter.set_offsets(coords)

        # Update subtle lensing shimmer on halo
        # Apply a tiny, time-varying positional offset
        dx, dy = lensing_offset(0.0, 0.0, float(t) * 0.04)  # center-based scalar
        # vary alpha slightly like breathing
        alpha = HALO_ALPHA * (0.9 + 0.15 * math.sin(0.03 * t))
        img.set_alpha(alpha)

        return img, stars_scatter, tracer_scatter

    anim = FuncAnimation(
        fig,
        update,
        interval=40,
        blit=False,
        cache_frame_data=False,
    )

    return anim


if __name__ == "__main__":
    anim = run_animation()
    plt.show()