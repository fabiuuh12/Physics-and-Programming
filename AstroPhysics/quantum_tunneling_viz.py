"""1D quantum tunneling of a Gaussian wave packet hitting a finite barrier.

What the animation shows:
- |ψ(x, t)|² for a packet approaching, interacting with, and partially tunneling through
a rectangular potential barrier.
- The potential barrier profile is drawn on the same axis for reference.
- Live readouts of reflection vs. transmission probability (integrals of |ψ|² on each side).

Numerics:
- Split-step Fourier method (Strang splitting) with ħ = m = 1 units.
- Not intended for precision, just an illustrative qualitative visualization.

Run:
    python3 quantum_tunneling_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Simulation grid
X_MIN, X_MAX = -20.0, 20.0
N_POINTS = 1024
DX = (X_MAX - X_MIN) / N_POINTS
x = np.linspace(X_MIN, X_MAX, N_POINTS, endpoint=False)

# Momentum space grid
k = np.fft.fftfreq(N_POINTS, d=DX) * 2.0 * np.pi

# Physical parameters (ħ = m = 1 units)
SIGMA = 1.5
K0 = 3.0
X0 = -10.0
V0 = 6.0
BARRIER_WIDTH = 3.0

DT = 0.004
STEPS_PER_FRAME = 6
N_FRAMES = 600

plt.style.use("dark_background")

# Initial wave packet
psi = (
    (1.0 / (np.pi * SIGMA**2)) ** 0.25
    * np.exp(-0.5 * ((x - X0) / SIGMA) ** 2)
    * np.exp(1j * K0 * x)
)

# Rectangular barrier
barrier = np.where(np.abs(x) < BARRIER_WIDTH / 2.0, V0, 0.0)

expV = np.exp(-1j * barrier * DT / 2.0)
expK = np.exp(-1j * (k**2) * DT / 2.0)


def split_step(psi_state: np.ndarray) -> np.ndarray:
    psi_state = expV * psi_state
    psi_k = np.fft.fft(psi_state)
    psi_k = expK * psi_k
    psi_state = np.fft.ifft(psi_k)
    psi_state = expV * psi_state
    return psi_state


fig, ax = plt.subplots(figsize=(9, 4.8))
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(0, 1.4)
ax.set_xlabel("Position x")
ax.set_ylabel("|ψ(x)|²")
ax.set_title("Quantum tunneling through a finite barrier")
ax.grid(color="#0d0f1f", linewidth=0.6, alpha=0.5)

wave_line, = ax.plot([], [], color="#7de0ff", linewidth=2.0)
barrier_line, = ax.plot([], [], color="#ffb997", linewidth=1.6, linestyle="--", alpha=0.8)

prob_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, color="#f8f7ff", fontsize=11)

# Precompute barrier drawing (scaled into axis range)
barrier_plot = barrier / V0 * 1.2
barrier_line.set_data(x, barrier_plot)


def update(frame: int):
    global psi
    for _ in range(STEPS_PER_FRAME):
        psi = split_step(psi)
        # Renormalize occasionally to avoid numerical drift
        norm = np.trapz(np.abs(psi) ** 2, x)
        psi /= np.sqrt(norm)

    density = np.abs(psi) ** 2
    wave_line.set_data(x, density)

    left_prob = np.trapz(density[x < -5], x[x < -5])
    right_prob = np.trapz(density[x > 5], x[x > 5])
    barrier_prob = 1.0 - left_prob - right_prob

    prob_text.set_text(
        f"Reflection ≈ {left_prob:.2f}    Transmission ≈ {right_prob:.2f}    "
        f"In-barrier ≈ {barrier_prob:.2f}"
    )

    return wave_line, barrier_line, prob_text


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * STEPS_PER_FRAME * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
