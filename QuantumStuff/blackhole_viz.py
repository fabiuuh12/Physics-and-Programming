"""
Animated cinematic black hole + accretion disk visualization.

Stylized toy model:
- Dark event horizon in the center.
- Glowing photon ring hugging the horizon.
- Hot, swirling accretion disk with relativistic beaming.
- Lensed starfield bending around the hole.
- Subtle jets and a "sucking in light" falloff toward the center.
- Curved rays of light being pulled inward toward the horizon.

Run:
    python3 blackhole_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")

# Global visual / domain settings
EXTENT = 3.0
RESOLUTION = 800
FIGSIZE = (6, 6)

# Black hole + disk parameters
HORIZON_RADIUS = 0.42       # event horizon radius (visual)
PHOTON_SPHERE_R = 0.75      # bright photon ring radius
DISK_RADIUS = 1.5           # main accretion disk radius
DISK_WIDTH = 0.32
LENSING_STRENGTH = 4.0      # stronger = more warped background
DISK_ANGULAR_SPEED = 0.9    # rotation speed of bright structures
SPIRAL_ARMS = 3             # azimuthal structure in the disk
JET_STRENGTH = 0.45       # brightness of polar jets
JET_WIDTH = 0.22          # horizontal thickness of jets

LIGHT_RAY_COUNT = 6          # number of infalling light streams
LIGHT_RAY_THICKNESS = 0.12   # angular thickness of each stream
LIGHT_RAY_BRIGHTNESS = 0.65  # overall brightness scale for streams
LIGHT_RAY_TWIST = 1.7        # how strongly the paths curve as they fall in
LIGHT_RAY_INFLOW_SPEED = 0.16  # how fast the bright part of each stream slides inward

# Starfield / light "sucking" effect
STAR_BASE_INTENSITY = 0.06
SUCK_GROWTH_RATE = 0.05     # how fast the "dark influence" expands
SUCK_MAX_RADIUS = 2.4       # max radius of strong dimming region


def make_grid(resolution: int = RESOLUTION):
    axis = np.linspace(-EXTENT, EXTENT, resolution)
    xx, yy = np.meshgrid(axis, axis)
    r = np.sqrt(xx * xx + yy * yy) + 1e-9
    theta = np.arctan2(yy, xx)
    return axis, xx, yy, r, theta


AXIS, XX, YY, R, THETA = make_grid()


def lensed_coords(xx: np.ndarray, yy: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Crude lensing: nudge apparent positions of background light radially outward.
    Not physical, but gives the sense of bending.
    """
    # Deflection ~ LENSING_STRENGTH / r^2, directed radially outward.
    # Clamp to avoid insane values near center.
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = LENSING_STRENGTH / (r * r)
    alpha = np.clip(alpha, 0.0, 0.9)
    # Components of radial unit vector
    ux = xx / r
    uy = yy / r
    x_lensed = xx + alpha * ux
    y_lensed = yy + alpha * uy
    return x_lensed, y_lensed


def generate_frame(t: float) -> np.ndarray:
    """
    Generate a single frame of the stylized black hole scene at time t.
    """
    # Base: lensed background starfield with subtle twinkling
    x_l, y_l = lensed_coords(XX, YY, R)
    base = (
        np.sin(4.0 * x_l + 0.37 * t)
        + np.cos(5.0 * y_l - 0.23 * t)
        + np.sin(7.3 * x_l + 3.1 * y_l + 0.11 * t)
    )
    # Keep only positive lobes and square them for sharper star points
    stars = np.clip(base, 0.0, None) ** 2
    # Normalize star brightness frame-by-frame to keep contrast punchy
    stars = STAR_BASE_INTENSITY * stars / (stars.max() + 1e-9)
    # Vignette so the center region pops more than the outer edge
    vignette = np.exp(-((R / EXTENT) ** 2.2))
    starfield = stars * vignette

    # Photon sphere: thin bright ring around PHOTON_SPHERE_R
    photon_ring = 1.4 * np.exp(-((R - PHOTON_SPHERE_R) / 0.04) ** 2)

    # Swirling accretion disk: radial ring + azimuthal / spiral structure
    disk_radial = np.exp(-((R - DISK_RADIUS) / DISK_WIDTH) ** 2)

    # Relativistic Doppler beaming: approaching side boosted, receding dimmed
    doppler = 0.32 + 0.68 * (1.0 + np.cos(THETA - DISK_ANGULAR_SPEED * t)) / 2.0

    # Spiral / clumpy structures rotating with time
    spiral_pattern = 0.5 + 0.5 * np.cos(
        SPIRAL_ARMS * (THETA - DISK_ANGULAR_SPEED * 1.4 * t)
    )

    disk = disk_radial * doppler * spiral_pattern

    # Extra inner glow just outside the horizon to make it feel hotter
    inner_glow = 0.25 * np.exp(-((R - (HORIZON_RADIUS + 0.12)) / 0.15) ** 2)
    disk += inner_glow

    # "Sucking in light": evolving falloff that darkens closer to the hole
    suck_radius = HORIZON_RADIUS + (SUCK_MAX_RADIUS - HORIZON_RADIUS) * (
        1.0 - np.exp(-SUCK_GROWTH_RATE * t)
    )
    # Values near the center go toward 0, outside suck_radius approach 1
    hole_falloff = 1.0 - np.exp(-(R / suck_radius) ** 3.0)

    starfield *= hole_falloff
    disk *= hole_falloff

    # Curved infalling light rays being pulled toward the hole
    rays = np.zeros_like(R)
    ray_angles = np.linspace(-np.pi, np.pi, LIGHT_RAY_COUNT, endpoint=False)
    for a in ray_angles:
        # Spiral-like trajectory: far away it's at angle a,
        # closer in it twists due to the gravitational pull.
        target_theta = a + LIGHT_RAY_TWIST * (1.0 - np.exp(-0.9 * R))
        # Smallest signed angular difference between current angle and the curve
        dtheta = np.angle(np.exp(1j * (THETA - target_theta)))
        # Bright where we're close to the curve
        along_curve = np.exp(-(dtheta / LIGHT_RAY_THICKNESS) ** 2)
        # Only really visible outside the horizon; fade inwards for a pulled look
        radial_gate = np.clip((R - HORIZON_RADIUS) / (EXTENT * 0.95), 0.0, 1.0)
        # Make the bright segment slide inward over time
        infall_center = EXTENT - LIGHT_RAY_INFLOW_SPEED * t
        infall = np.exp(-((R - infall_center) / 1.8) ** 2)
        rays += along_curve * radial_gate * infall
    # Scale and apply same falloff so rays disappear into the hole
    rays *= LIGHT_RAY_BRIGHTNESS
    rays *= hole_falloff

    # Polar jets: thin beams shooting out along the vertical axis
    jet_profile = np.exp(-(XX ** 2) / (2.0 * JET_WIDTH ** 2))
    jet_length = np.exp(-((np.abs(YY) - HORIZON_RADIUS) / (EXTENT * 0.9)) ** 2)
    jet_time = 0.5 + 0.5 * np.cos(0.9 * t)
    jets = JET_STRENGTH * jet_time * jet_profile * jet_length
    jets[R < HORIZON_RADIUS] = 0.0

    # Event horizon: enforce deep black inside HORIZON_RADIUS
    horizon_mask = R < HORIZON_RADIUS

    # Combine contributions
    image = (
        starfield
        + 2.5 * disk
        + photon_ring
        + jets
        + rays
    )

    # Hard black center
    image[horizon_mask] = 0.0

    # Clip range for display
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
        generate_frame(t=0.0),
        cmap="inferno",
        origin="lower",
        extent=[-EXTENT, EXTENT, -EXTENT, EXTENT],
        interpolation="bilinear",
    )

    ax.set_title("Animated Black Hole & Accretion Disk", pad=18, color="white")

    def update(frame: int):
        # Scale time so motion is smooth and continuous
        t = frame * 0.25
        frame_data = generate_frame(t)
        img.set_data(frame_data)
        return (img,)

    anim = FuncAnimation(
        fig,
        update,
        interval=40,          # ms between frames
        blit=True,
        cache_frame_data=False,
    )

    return anim


def main():
    anim = run_animation()
    plt.show()


if __name__ == "__main__":
    main()
