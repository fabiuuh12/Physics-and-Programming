"""Heisenberg uncertainty visualization: position-space packet vs. momentum-space wave.

Highlights:
- Left panel shows a Gaussian wave packet in position space whose width breathes in time.
- Right panel shows the matching momentum distribution (Fourier partner) widening when the
  position packet narrows, illustrating Δx · Δp ≥ ħ / 2 (ħ set to 1 here).
- Text readouts show the instantaneous uncertainties and their product.

Run:
    python3 uncertainty_wavepacket_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Numeric setup
X_MIN, X_MAX = -7.0, 7.0
P_MIN, P_MAX = -10.0, 10.0
N_SAMPLES = 800

BASE_SIGMA_X = 0.9
BREATHING_FREQ = 0.35
P_AVG = 4.0

DT = 0.04
N_FRAMES = 520

plt.style.use("dark_background")

fig, (ax_x, ax_p) = plt.subplots(1, 2, figsize=(10, 4.5))

# Position-space plot
x_grid = np.linspace(X_MIN, X_MAX, N_SAMPLES)
pos_line, = ax_x.plot([], [], color="#7cd6ff", linewidth=2.3)
ax_x.fill_between(x_grid, 0, 0, color="#7cd6ff", alpha=0.2)
ax_x.set_xlim(X_MIN, X_MAX)
ax_x.set_ylim(0, 1.15)
ax_x.set_xlabel("Position x")
ax_x.set_ylabel("|ψ(x)|")
ax_x.set_title("Position-space wave packet")
ax_x.grid(color="#111", linewidth=0.6, alpha=0.6)

# Momentum-space plot
p_grid = np.linspace(P_MIN, P_MAX, N_SAMPLES)
mom_line, = ax_p.plot([], [], color="#ffb5a8", linewidth=2.3)
ax_p.fill_between(p_grid, 0, 0, color="#ffb5a8", alpha=0.2)
ax_p.set_xlim(P_MIN, P_MAX)
ax_p.set_ylim(0, 1.15)
ax_p.set_xlabel("Momentum p")
ax_p.set_ylabel("|φ(p)|")
ax_p.set_title("Momentum-space distribution")
ax_p.grid(color="#111", linewidth=0.6, alpha=0.6)

info_text = fig.text(0.5, 0.04, "", ha="center", color="#e8f7ff", fontsize=11)


def normalized_gaussian(grid: np.ndarray, sigma: float, center: float) -> np.ndarray:
    """Return |ψ| for a Gaussian normalized to peak at 1."""
    return np.exp(-0.5 * ((grid - center) / sigma) ** 2)


def update(frame: int):
    t = frame * DT

    sigma_x = BASE_SIGMA_X * (0.65 + 0.35 * (1 + np.sin(BREATHING_FREQ * t)))
    sigma_p = 0.5 / sigma_x  # Using ħ = 1 units.

    psi_mag = normalized_gaussian(x_grid, sigma_x, center=0.0)
    phi_mag = normalized_gaussian(p_grid, sigma_p, center=P_AVG)

    pos_line.set_data(x_grid, psi_mag)
    mom_line.set_data(p_grid, phi_mag)

    info_text.set_text(
        f"Δx ≈ {sigma_x:.2f}     Δp ≈ {sigma_p:.2f}     Δx·Δp ≈ {sigma_x * sigma_p:.2f}"
    )

    return pos_line, mom_line, info_text


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.show()
