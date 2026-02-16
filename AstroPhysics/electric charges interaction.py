"""Electric field interaction between two point charges.

Features:
- Field lines (streamplot) for the combined field.
- Quiver arrows for field direction.
- Force arrows on each charge showing attraction/repulsion.

Run:
    python3 "electric charges interaction.py"
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# Charge setup (edit these to explore interactions)
Q1 = 1.0
Q2 = -1.0
POS1 = (-1.2, 0.0)
POS2 = (1.2, 0.0)
K = 1.0  # normalized Coulomb constant

# Grid for field visualization
GRID_LIM = 3.0
N = 240
x = np.linspace(-GRID_LIM, GRID_LIM, N)
y = np.linspace(-GRID_LIM, GRID_LIM, N)
X, Y = np.meshgrid(x, y)

softening = 0.08


def field_from_charge(q: float, pos: tuple[float, float]):
    dx = X - pos[0]
    dy = Y - pos[1]
    r2 = dx**2 + dy**2 + softening**2
    r = np.sqrt(r2)
    ex = K * q * dx / r**3
    ey = K * q * dy / r**3
    return ex, ey


Ex1, Ey1 = field_from_charge(Q1, POS1)
Ex2, Ey2 = field_from_charge(Q2, POS2)
Ex = Ex1 + Ex2
Ey = Ey1 + Ey2
E_mag = np.sqrt(Ex**2 + Ey**2)

# Compute force vectors on charges due to each other
r12 = np.array(POS2) - np.array(POS1)
r = np.linalg.norm(r12)
if r == 0:
    r = 1e-6
r_hat = r12 / r
F_mag = K * Q1 * Q2 / (r**2)
F1 = F_mag * r_hat
F2 = -F1

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("#0b0f1a")
ax.set_aspect("equal", "box")
ax.set_xlim(-GRID_LIM, GRID_LIM)
ax.set_ylim(-GRID_LIM, GRID_LIM)
ax.set_title("Electric Field Interaction of Two Charges", fontsize=14, pad=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(color="#101223", linewidth=0.6, alpha=0.6)

# Field lines
stream = ax.streamplot(
    X,
    Y,
    Ex,
    Ey,
    color=E_mag,
    linewidth=1.0,
    density=1.6,
    cmap="plasma",
    minlength=0.2,
)

# Quiver arrows (sampled to reduce clutter)
skip = (slice(None, None, 14), slice(None, None, 14))
ax.quiver(
    X[skip],
    Y[skip],
    Ex[skip],
    Ey[skip],
    color="#7dd3ff",
    scale=50,
    width=0.004,
    alpha=0.8,
)

# Charge markers
charge_color_1 = "#ff6b6b" if Q1 > 0 else "#5ea8ff"
charge_color_2 = "#ff6b6b" if Q2 > 0 else "#5ea8ff"
ax.scatter([POS1[0]], [POS1[1]], s=350, c=charge_color_1, edgecolors="white", linewidths=1.5, zorder=5)
ax.scatter([POS2[0]], [POS2[1]], s=350, c=charge_color_2, edgecolors="white", linewidths=1.5, zorder=5)
ax.text(POS1[0], POS1[1], "+" if Q1 > 0 else "-", ha="center", va="center", fontsize=18, color="white", zorder=6)
ax.text(POS2[0], POS2[1], "+" if Q2 > 0 else "-", ha="center", va="center", fontsize=18, color="white", zorder=6)

# Force arrows (interaction)
force_scale = 1.2  # visual scale only
ax.quiver(
    POS1[0],
    POS1[1],
    F1[0],
    F1[1],
    color="#8affbd",
    scale=1.0 / force_scale,
    width=0.012,
    zorder=6,
)
ax.quiver(
    POS2[0],
    POS2[1],
    F2[0],
    F2[1],
    color="#8affbd",
    scale=1.0 / force_scale,
    width=0.012,
    zorder=6,
)
ax.text(
    0.02,
    0.02,
    f"q1={Q1:+.1f}, q2={Q2:+.1f}",
    color="white",
    fontsize=10,
    transform=ax.transAxes,
    ha="left",
    va="bottom",
)

# Colorbar
cbar = fig.colorbar(stream.lines, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("|E| (normalized)")

plt.tight_layout()

if __name__ == "__main__":
    plt.show()
