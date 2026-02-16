"""Gravitational lensing visualization with an animated Einstein radius.

We model a point-mass lens in the center. A static textured background (galaxy + stars)
is sampled through the lens equation β = θ − (θ_E^2) θ / |θ|^2. As θ_E grows, the
background stretches into arcs and an Einstein ring.

Run:
    python3 gravitational_lensing_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


GRID = 220
DT = 0.04
N_FRAMES = 360
THETA_RANGE = 2.2

theta = np.linspace(-THETA_RANGE, THETA_RANGE, GRID)
THETA_X, THETA_Y = np.meshgrid(theta, theta)

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Gravitational Lensing (toy point lens)", color="white")


def background_texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    base = 0.35 + 0.12 * np.sin(2.3 * x) + 0.12 * np.cos(1.8 * y)
    spiral = np.exp(-((x - 0.7) ** 2 + (y + 0.2) ** 2) / 0.8) * (1 + 0.4 * np.sin(4 * np.arctan2(y + 0.2, x - 0.7)))

    rng = np.random.default_rng(4)
    stars = np.zeros_like(base)
    for _ in range(65):
        sx = rng.uniform(-THETA_RANGE, THETA_RANGE)
        sy = rng.uniform(-THETA_RANGE, THETA_RANGE)
        amp = rng.uniform(0.3, 1.0)
        width = rng.uniform(0.003, 0.02)
        stars += amp * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / width)

    tex = np.clip(base + 0.7 * spiral + stars, 0, 1)
    return tex


BACKGROUND = background_texture(THETA_X, THETA_Y)

img = ax.imshow(BACKGROUND, extent=(-THETA_RANGE, THETA_RANGE, -THETA_RANGE, THETA_RANGE), origin="lower", cmap="magma")

mass_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="#f6f1ff", fontsize=11, ha="left", va="top")
note_text = ax.text(0.02, 0.02, "Background sampled via β = θ − θ_E^2 θ / |θ|^2", transform=ax.transAxes, color="#ffdfa8", fontsize=9, ha="left")

lens_circle = plt.Circle((0, 0), 0.12, color="#9ff6ff", fill=False, linewidth=1.2, alpha=0.7)
ax.add_patch(lens_circle)


def lens_map(theta_e: float):
    r2 = THETA_X**2 + THETA_Y**2 + 1e-6
    beta_x = THETA_X - (theta_e**2) * THETA_X / r2
    beta_y = THETA_Y - (theta_e**2) * THETA_Y / r2
    return beta_x, beta_y


def sample_background(beta_x: np.ndarray, beta_y: np.ndarray) -> np.ndarray:
    # Convert beta coords back to indices
    scale = (GRID - 1) / (2 * THETA_RANGE)
    ix = np.clip(((beta_x + THETA_RANGE) * scale).astype(int), 0, GRID - 1)
    iy = np.clip(((beta_y + THETA_RANGE) * scale).astype(int), 0, GRID - 1)
    return BACKGROUND[iy, ix]


def update(frame: int):
    t = frame * DT
    theta_e = 0.25 + 0.35 * (0.5 + 0.5 * np.sin(0.6 * t))
    beta_x, beta_y = lens_map(theta_e)
    warped = sample_background(beta_x, beta_y)
    img.set_data(warped)

    lens_circle.set_radius(theta_e)
    mass_text.set_text(f"Einstein radius θ_E ≈ {theta_e:.2f} → effective lens mass ∝ θ_E²")

    return img, lens_circle, mass_text


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
