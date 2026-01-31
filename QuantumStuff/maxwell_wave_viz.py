"""Animated electromagnetic plane wave obeying Maxwell's equations in vacuum.

Features:
- Top plot: electric field E_y (blue) and magnetic field B_z (gold) traveling in +x.
- Bottom plot: field vectors sampled at discrete points, with the Poynting vector
  (energy flow) pointing along +x.
- Text readouts remind that E ⊥ B ⊥ direction of propagation and that the two fields
  remain in phase for a plane wave in vacuum.

Run:
    python3 maxwell_wave_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


C = 1.0  # set c = 1 for convenience
E0 = 1.0
WAVELENGTH = 2.5 * np.pi
K = 2.0 * np.pi / WAVELENGTH
OMEGA = C * K

DT = 0.04
N_FRAMES = 400

x_grid = np.linspace(0.0, 4.0 * np.pi, 600)
x_samples = np.linspace(0.2, 4.0 * np.pi - 0.2, 18)

plt.style.use("dark_background")
fig, (ax_wave, ax_vectors) = plt.subplots(2, 1, figsize=(9, 6.4), gridspec_kw={"height_ratios": [3, 2]}, sharex=True)


def wave_fields(t: float):
    phase = K * x_grid - OMEGA * t
    E = E0 * np.sin(phase)
    B = (E0 / C) * np.sin(phase)
    return E, B


def sample_fields(t: float):
    phase = K * x_samples - OMEGA * t
    E = E0 * np.sin(phase)
    B = (E0 / C) * np.sin(phase)
    return E, B


# Top plot styling
ax_wave.set_xlim(x_grid.min(), x_grid.max())
ax_wave.set_ylim(-1.8, 1.4)
ax_wave.set_ylabel("Field amplitude")
ax_wave.set_title("Electromagnetic plane wave: E ⊥ B ⊥ propagation")
ax_wave.grid(color="#101223", linewidth=0.6, alpha=0.6)

E_line, = ax_wave.plot([], [], color="#72d1ff", linewidth=2.5, label="E field (y-direction)")
B_line, = ax_wave.plot([], [], color="#ffd27c", linewidth=2.5, label="B field (z-direction)")
ax_wave.legend(loc="upper right")

# Baselines to separate the curves visually
ax_wave.axhline(0.0, color="#1a1a2f", linewidth=1.0)
ax_wave.axhline(-1.2, color="#1a1a2f", linewidth=1.0, linestyle="--", alpha=0.7)
ax_wave.text(x_grid.max() * 0.99, 0.15, "E_y", color="#72d1ff", ha="right")
ax_wave.text(x_grid.max() * 0.99, -1.05, "B_z (offset)", color="#ffd27c", ha="right")

ax_vectors.set_xlim(x_grid.min(), x_grid.max())
ax_vectors.set_ylim(-1.6, 1.6)
ax_vectors.set_xlabel("Propagation direction x")
ax_vectors.set_ylabel("Field axes")
ax_vectors.grid(color="#101223", linewidth=0.6, alpha=0.6)
ax_vectors.set_title("Local field vectors and Poynting direction S ∝ E × B")

zeros_samples = np.zeros_like(x_samples)
E_quiver = ax_vectors.quiver(
    x_samples,
    zeros_samples,
    zeros_samples,
    zeros_samples,
    color="#72d1ff",
    scale=8,
    width=0.008,
)
B_quiver = ax_vectors.quiver(
    x_samples,
    zeros_samples - 0.8,
    zeros_samples,
    zeros_samples,
    color="#ffd27c",
    scale=8,
    width=0.008,
)
S_arrow = ax_vectors.arrow(x_grid.min() + 0.3, 1.2, x_grid.max() * 0.2, 0.0, color="#8affbd", width=0.02)
S_text = ax_vectors.text(x_grid.min() + 0.3, 1.32, "Energy flow (S)", color="#8affbd", fontsize=10)

info_text = fig.text(
    0.5,
    0.04,
    "",
    ha="center",
    color="#f3f8ff",
    fontsize=11,
)


def update(frame: int):
    t = frame * DT
    E_vals, B_vals = wave_fields(t)
    E_line.set_data(x_grid, E_vals)
    # Offset B downwards for clarity
    B_line.set_data(x_grid, B_vals - 1.2)

    E_samples, B_samples = sample_fields(t)
    zeros = np.zeros_like(x_samples)
    E_quiver.set_UVC(zeros, E_samples)
    B_quiver.set_UVC(B_samples, zeros)

    info_text.set_text(
        "Fields stay in phase: B = E / c\n"
        "Blue arrows: E along +y, Gold arrows: B along +z (drawn horizontally for clarity)."
    )

    return E_line, B_line, info_text


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()
