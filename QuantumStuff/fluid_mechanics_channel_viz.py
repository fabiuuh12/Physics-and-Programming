"""
Fluid mechanics: laminar channel (Poiseuille) flow with tracer particles.

Velocity profile:
  u(y) = U_max (1 - (y/H)^2),   v = 0

Run:
    python fluid_mechanics_channel_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain
X_MIN, X_MAX = 0.0, 10.0
Y_MIN, Y_MAX = -1.2, 1.2
H = 1.0

# Flow parameters
U_MAX = 1.2
DT = 0.02
N_FRAMES = 700

# Tracers
N_PARTICLES = 800

plt.style.use("dark_background")

# Initialize tracers
rng = np.random.default_rng(2)
px = rng.uniform(X_MIN, X_MAX, N_PARTICLES)
py = rng.uniform(Y_MIN, Y_MAX, N_PARTICLES)

# Figure
fig = plt.figure(figsize=(9, 5.2))
ax_flow = fig.add_subplot(1, 1, 1)

ax_flow.set_xlim(X_MIN, X_MAX)
ax_flow.set_ylim(Y_MIN, Y_MAX)
ax_flow.set_facecolor("#05070f")
ax_flow.set_title("Laminar channel flow (Poiseuille profile)")
ax_flow.set_xticks([])
ax_flow.set_yticks([])

# Walls
ax_flow.plot([X_MIN, X_MAX], [H, H], color="#8fb3ff", lw=2.0, alpha=0.7)
ax_flow.plot([X_MIN, X_MAX], [-H, -H], color="#8fb3ff", lw=2.0, alpha=0.7)

# Tracers
scatter = ax_flow.scatter(px, py, s=6, c="#9fe0ff", alpha=0.8, edgecolors="none")

# Velocity profile overlay (right side)
y_prof = np.linspace(-H, H, 200)
u_prof = U_MAX * (1.0 - (y_prof / H) ** 2)
ax_flow.plot(X_MAX - 1.0 + 0.8 * (u_prof / U_MAX), y_prof, color="#ffd27c", lw=2.0)
ax_flow.text(X_MAX - 1.05, H + 0.1, "u(y)", color="#ffd27c", fontsize=9, ha="left")


def velocity(y: np.ndarray) -> np.ndarray:
    return U_MAX * (1.0 - (y / H) ** 2)


def update(_frame: int):
    global px, py

    # Advect particles
    u = velocity(py)
    px = px + u * DT

    # Wrap around to left when exiting
    out = px > X_MAX
    if np.any(out):
        px[out] = X_MIN
        py[out] = rng.uniform(Y_MIN, Y_MAX, out.sum())

    scatter.set_offsets(np.column_stack((px, py)))
    return (scatter,)


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=DT * 1000.0, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
