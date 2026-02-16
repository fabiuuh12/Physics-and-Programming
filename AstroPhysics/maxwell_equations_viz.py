"""
Maxwell's equations visualization for electromagnetic plane waves.

We animate a 1D plane wave that satisfies all four Maxwell equations in vacuum:
  1) ?·E = 0
  2) ?·B = 0
  3) ?×E = -?B/?t
  4) ?×B = µ0 e0 ?E/?t  (with c^2 = 1/(µ0 e0))

Run:
    python maxwell_equations_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters (vacuum plane wave)
C = 1.0
E0 = 1.0
WAVELENGTH = 2.5 * np.pi
K = 2.0 * np.pi / WAVELENGTH
OMEGA = C * K

DT = 0.04
N_FRAMES = 600

x = np.linspace(0.0, 4.0 * np.pi, 700)

plt.style.use("dark_background")
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)

ax_fields = axes[0, 0]
ax_farad = axes[0, 1]
ax_ampere = axes[1, 0]
ax_text = axes[1, 1]

# Top-left: E and B fields
ax_fields.set_title("Plane wave fields: E_y and B_z")
ax_fields.set_ylabel("Field amplitude")
ax_fields.grid(color="#111", alpha=0.6)
E_line, = ax_fields.plot([], [], color="#7de0ff", lw=2.2, label="E_y")
B_line, = ax_fields.plot([], [], color="#ffd27c", lw=2.0, label="B_z")
ax_fields.legend(loc="upper right", frameon=False)

# Top-right: Faraday's law
ax_farad.set_title("Faraday's law: ?×E = -?B/?t")
ax_farad.grid(color="#111", alpha=0.6)
farad_lhs, = ax_farad.plot([], [], color="#9fe0ff", lw=2.0, label="(?×E)_z")
farad_rhs, = ax_farad.plot([], [], color="#ffb578", lw=2.0, linestyle="--", label="-?B_z/?t")
ax_farad.legend(loc="upper right", frameon=False)

# Bottom-left: Ampere-Maxwell law
ax_ampere.set_title("Ampere–Maxwell: ?×B = (1/c^2) ?E/?t")
ax_ampere.set_xlabel("x")
ax_ampere.grid(color="#111", alpha=0.6)
amp_lhs, = ax_ampere.plot([], [], color="#9fe0ff", lw=2.0, label="(?×B)_y")
amp_rhs, = ax_ampere.plot([], [], color="#ffb578", lw=2.0, linestyle="--", label="(1/c^2) ?E_y/?t")
ax_ampere.legend(loc="upper right", frameon=False)

# Bottom-right: text panel with Gauss laws
ax_text.axis("off")
text = ax_text.text(
    0.0,
    0.95,
    "Gauss' laws in vacuum:\n\n"
    "?·E = 0\n"
    "?·B = 0\n\n"
    "Plane wave (propagation +x):\n"
    "E = y E0 sin(kx - ?t)\n"
    "B = ? (E0/c) sin(kx - ?t)",
    color="#e8f2ff",
    fontsize=11,
    va="top",
)


def fields(t: float):
    phase = K * x - OMEGA * t
    E = E0 * np.sin(phase)
    B = (E0 / C) * np.sin(phase)
    return E, B


def update(frame: int):
    t = frame * DT
    E, B = fields(t)

    # Spatial derivatives for curls (1D plane wave)
    dE_dx = np.gradient(E, x)
    dB_dx = np.gradient(B, x)

    # Time derivatives
    dB_dt = -OMEGA * (E0 / C) * np.cos(K * x - OMEGA * t)
    dE_dt = -OMEGA * E0 * np.cos(K * x - OMEGA * t)

    # For a 1D plane wave:
    # (?×E)_z = ?E_y/?x
    # (?×B)_y = -?B_z/?x
    curlE_z = dE_dx
    curlB_y = -dB_dx

    # Update plots
    E_line.set_data(x, E)
    B_line.set_data(x, B)

    farad_lhs.set_data(x, curlE_z)
    farad_rhs.set_data(x, -dB_dt)

    amp_lhs.set_data(x, curlB_y)
    amp_rhs.set_data(x, (1.0 / C**2) * dE_dt)

    # Autoscale y-limits for visibility
    ax_fields.set_ylim(-1.5, 1.5)
    ax_farad.set_ylim(-K * E0 * 1.2, K * E0 * 1.2)
    ax_ampere.set_ylim(-K * (E0 / C) * 1.2, K * (E0 / C) * 1.2)

    return E_line, B_line, farad_lhs, farad_rhs, amp_lhs, amp_rhs


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=DT * 1000.0, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
