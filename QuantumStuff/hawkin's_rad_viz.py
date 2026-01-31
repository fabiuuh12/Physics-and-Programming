"""
Animated sketch of Hawking radiation around a black hole.

Visual ingredients:
- Starfield background warped by gravity.
- Dark event horizon with a hot photon halo.
- Stochastic "quantum foam" flickering at the horizon.
- Expanding pulses that represent Hawking radiation quanta escaping.
- Broad lobes indicating preferential particle escape directions.

Run:
    python3 "hawkin's_rad_viz.py"
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

# Horizon + near-field parameters
HORIZON_RADIUS = 0.55
ERGOSPHERE_RADIUS = 0.85
HORIZON_WIDTH = 0.06
ERGOSPHERE_WIDTH = 0.18

# Quantum foam flicker
FOAM_SCALE = 22.0
FOAM_FLICKER_RATE = 1.2
FOAM_SHARPNESS = 3.6
FOAM_SHELL_WIDTH = 0.2

# Hawking radiation pulses
WAVE_COUNT = 5
WAVE_SPEED = 0.38
WAVE_THICKNESS = 0.12
WAVE_DECAY = 0.55
EMISSION_PERIOD = 1.3

# Broad escape lobes / spray
LOBE_WIDTH = 0.28
LOBE_OFFSET = 0.35
LOBE_OSC = 0.25

# Subtle polar jets to hint at symmetry axes
JET_WIDTH = 0.22
JET_FADE = 0.9
JET_FREQ = 0.6

# Light background
STAR_INTENSITY = 0.07


def make_grid(resolution: int = RESOLUTION):
    axis = np.linspace(-EXTENT, EXTENT, resolution)
    xx, yy = np.meshgrid(axis, axis)
    r = np.sqrt(xx * xx + yy * yy) + 1e-9
    theta = np.arctan2(yy, xx)
    return axis, xx, yy, r, theta


AXIS, XX, YY, R, THETA = make_grid()


def generate_frame(t: float) -> np.ndarray:
    # Background: warped starfield that dims toward the center.
    star_seed = (
        np.sin(2.1 * XX + 0.6 * t)
        + np.cos(2.7 * YY - 0.4 * t)
        + np.sin(3.5 * (XX + YY) + 0.32 * t)
    )
    stars = np.clip(star_seed, 0.0, None) ** 2
    stars = STAR_INTENSITY * stars / (stars.max() + 1e-9)
    vignette = np.exp(-((R / EXTENT) ** 2.2))
    background = stars * vignette

    # Hot photon halo hugging the horizon + ergosphere glow.
    horizon_glow = np.exp(-((R - HORIZON_RADIUS) / HORIZON_WIDTH) ** 2)
    ergosphere_glow = np.exp(-((R - ERGOSPHERE_RADIUS) / ERGOSPHERE_WIDTH) ** 2)

    # Quantum foam: noisy flicker localized just outside the horizon.
    foam_phase = FOAM_SCALE * (R - HORIZON_RADIUS) + 3.0 * THETA
    foam_pattern = np.sin(foam_phase + FOAM_FLICKER_RATE * t) + 0.6 * np.sin(
        0.7 * foam_phase - 1.7 * t
    )
    foam_pattern = np.clip(foam_pattern, 0.0, None) ** FOAM_SHARPNESS
    foam_shell = np.exp(-((R - (HORIZON_RADIUS + 0.12)) / FOAM_SHELL_WIDTH) ** 2)
    foam = foam_pattern * foam_shell

    # Hawking radiation pulses: concentric rings that drift outward over time.
    waves = np.zeros_like(R)
    for idx in range(WAVE_COUNT):
        emission_time = idx * EMISSION_PERIOD
        travel_time = max(t - emission_time, 0.0)
        if travel_time <= 0.0:
            continue
        radius = HORIZON_RADIUS + WAVE_SPEED * travel_time
        if radius - 3.0 * WAVE_THICKNESS > EXTENT:
            continue
        ring = np.exp(-((R - radius) / WAVE_THICKNESS) ** 2)
        damping = np.exp(-WAVE_DECAY * (radius - HORIZON_RADIUS))
        anisotropy = 0.65 + 0.35 * np.cos(THETA - 0.35 * t + idx * 0.8)
        waves += damping * ring * anisotropy

    # Escape lobes: emphasize that some quanta manage to climb out.
    lobes = np.exp(-((R - (HORIZON_RADIUS + LOBE_OFFSET)) / LOBE_WIDTH) ** 2)
    lobes *= (0.4 + 0.6 * np.cos(2.0 * (THETA - LOBE_OSC * t)) ** 2)
    lobes *= np.clip((R - HORIZON_RADIUS) / (EXTENT - HORIZON_RADIUS), 0.0, 1.0)

    # Subtle polar jets to add cinematic structure.
    jet_profile = np.exp(-(XX ** 2) / (2.0 * JET_WIDTH ** 2))
    jet_length = np.exp(-((np.abs(YY) - HORIZON_RADIUS) / (JET_FADE * EXTENT)) ** 2)
    jet_time = 0.45 + 0.55 * np.cos(JET_FREQ * t)
    jets = jet_time * jet_profile * jet_length
    jets[R < HORIZON_RADIUS] = 0.0

    # Event horizon mask.
    horizon_mask = R < HORIZON_RADIUS

    # Combine contributions.
    image = (
        background
        + 0.25 * np.exp(-(R / EXTENT) ** 1.4)
        + 2.2 * horizon_glow
        + 0.8 * ergosphere_glow
        + 1.4 * foam
        + 2.2 * waves
        + 0.9 * lobes
        + 0.6 * jets
    )

    image[horizon_mask] = 0.0
    image = np.clip(image, 0.0, 1.0)
    return image


def run_animation():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("black")
    ax.set_xlim(-EXTENT, EXTENT)
    ax.set_ylim(-EXTENT, EXTENT)
    ax.set_aspect("equal")
    ax.axis("off")

    img = ax.imshow(
        generate_frame(0.0),
        cmap="magma",
        origin="lower",
        extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
        interpolation="bilinear",
    )

    ax.set_title("Hawking Radiation Glow", pad=16, color="white")

    def update(frame: int):
        t = frame * 0.2
        img.set_data(generate_frame(t))
        return (img,)

    anim = FuncAnimation(
        fig,
        update,
        interval=50,
        blit=True,
        cache_frame_data=False,
    )

    return anim


def main():
    anim = run_animation()
    plt.show()


if __name__ == "__main__":
    main()
