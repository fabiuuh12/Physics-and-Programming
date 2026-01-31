

"""
Quantum-style double-slit wave visualization.

Instead of individual particles, this shows a continuous wave field:
- A wave comes in from the left.
- The barrier blocks it except for two narrow slits.
- Past the slits, you see the interference pattern building up as ripples.

Purely visual & qualitative, not a precise physical simulator.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Scene geometry (arbitrary units)
X_MIN, X_MAX = -2.5, 10.5
Y_MIN, Y_MAX = -3.0, 3.0
EMITTER_X = -2.0
BARRIER_X = 2.0
SCREEN_X = 9.5

# Slit configuration (match your particle version)
SLIT_OFFSET = 1.0      # distance of the slit centers from y=0
SLIT_HEIGHT = 0.8
SLIT_WIDTH = 0.18

# Wave parameters
GRID_POINTS_X = 500
GRID_POINTS_Y = 260
WAVE_NUMBER = 7.5          # k: spatial frequency
ANGULAR_FREQ = 0.18        # omega: time frequency per frame step
DT = 1.0                   # time step per frame
DECAY_POWER = 0.55         # 1/sqrt(r)^p falloff for slit waves

# Visualization
CMAP = "magma"             # good on dark background
INTENSITY_CLIP = (0.0, 1.6)  # vmin, vmax for imshow


def make_grid():
    xs = np.linspace(X_MIN, X_MAX, GRID_POINTS_X)
    ys = np.linspace(Y_MIN, Y_MAX, GRID_POINTS_Y)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, X, Y


def build_scene():
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#05060d")

    # Barrier stripe
    barrier = Rectangle(
        (BARRIER_X - SLIT_WIDTH / 2, Y_MIN),
        SLIT_WIDTH,
        Y_MAX - Y_MIN,
        color="#444954",
        zorder=5,
    )
    ax.add_patch(barrier)

    # Carve out two slits as "holes" (same positions as particle sim)
    for offset in (-SLIT_OFFSET, SLIT_OFFSET):
        ax.add_patch(
            Rectangle(
                (BARRIER_X - SLIT_WIDTH / 2 - 0.02, offset - SLIT_HEIGHT / 2),
                SLIT_WIDTH + 0.04,
                SLIT_HEIGHT,
                color=ax.get_facecolor(),
                zorder=6,
            )
        )

    # Detection screen line on the right
    ax.plot(
        [SCREEN_X, SCREEN_X],
        [Y_MIN, Y_MAX],
        color="#7f8cfa",
        linewidth=1.8,
        alpha=0.7,
        zorder=7,
    )
    ax.text(
        SCREEN_X + 0.12,
        Y_MAX - 0.25,
        "Detection screen\n(intensity pattern)",
        color="#b9c4ff",
        ha="left",
        va="top",
        fontsize=9,
    )

    # Source label
    ax.text(
        EMITTER_X,
        Y_MIN + 0.3,
        "Incoming wavefront",
        color="#8df6ff",
        ha="left",
        va="bottom",
        fontsize=9,
    )

    return fig, ax


def compute_wave_field(X, Y, t):
    """
    Build a 2D scalar field representing the wave amplitude at time t.

    Left side: incoming plane wave moving right.
    Right side: two spherical-ish waves from each slit interfering.
    """
    # Base incoming plane wave from the left (propagating +x)
    phase_in = WAVE_NUMBER * (X - EMITTER_X) - ANGULAR_FREQ * t
    incoming = np.cos(phase_in)

    # Mask incoming only to x < barrier (so barrier actually blocks)
    incoming_mask = X < BARRIER_X
    field = np.zeros_like(X)
    field[incoming_mask] = incoming[incoming_mask]

    # Positions of slit centers
    slit1 = (BARRIER_X, -SLIT_OFFSET)
    slit2 = (BARRIER_X, SLIT_OFFSET)

    # Only compute slit-generated waves to the right of barrier
    right_region = X >= BARRIER_X

    # Distances from each slit
    r1 = np.sqrt((X - slit1[0]) ** 2 + (Y - slit1[1]) ** 2)
    r2 = np.sqrt((X - slit2[0]) ** 2 + (Y - slit2[1]) ** 2)

    # Avoid div-by-zero
    r1 = np.maximum(r1, 1e-3)
    r2 = np.maximum(r2, 1e-3)

    # Time-dependent phases for each slit wave
    phase1 = WAVE_NUMBER * r1 - ANGULAR_FREQ * t
    phase2 = WAVE_NUMBER * r2 - ANGULAR_FREQ * t

    # Slit waves with radial falloff so it's not uniform
    amp1 = np.cos(phase1) / (r1 ** DECAY_POWER)
    amp2 = np.cos(phase2) / (r2 ** DECAY_POWER)

    # Apply only on right side
    slit_field = amp1 + amp2
    field[right_region] = slit_field[right_region]

    return field


def run_animation():
    fig, ax = build_scene()
    xs, ys, X, Y = make_grid()

    # Initial frame
    t0 = 0.0
    field0 = compute_wave_field(X, Y, t0)
    intensity0 = field0 ** 2

    img = ax.imshow(
        intensity0,
        extent=(X_MIN, X_MAX, Y_MIN, Y_MAX),
        origin="lower",
        cmap=CMAP,
        vmin=INTENSITY_CLIP[0],
        vmax=INTENSITY_CLIP[1],
        alpha=0.95,
        zorder=1,
        interpolation="bilinear",
    )

    def update(frame):
        t = frame * DT
        field = compute_wave_field(X, Y, t)
        intensity = field * field

        # Optionally clip extremes for nicer contrast
        img.set_data(np.clip(intensity, INTENSITY_CLIP[0], INTENSITY_CLIP[1]))
        return (img,)

    anim = FuncAnimation(
        fig,
        update,
        interval=40,
        blit=True,
        cache_frame_data=False,
    )

    return anim


if __name__ == "__main__":
    anim = run_animation()
    plt.show()