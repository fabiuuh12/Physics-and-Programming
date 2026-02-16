"""Electric field visualization for a point charge.

Features:
- Streamlines for field lines.
- Quiver arrows for field direction/magnitude.
- Charge marker at the origin with sign label.

Run:
    python3 "electric field.py"
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# Charge configuration
Q = 1.0  # positive charge; set to -1.0 for negative
K = 1.0  # Coulomb constant in normalized units

# Grid for field visualization
GRID_LIM = 2.5
N = 200
x = np.linspace(-GRID_LIM, GRID_LIM, N)
y = np.linspace(-GRID_LIM, GRID_LIM, N)
X, Y = np.meshgrid(x, y)

# Electric field of a point charge: E = k q r / r^3
R2 = X**2 + Y**2
# Avoid singularity at the origin by adding a small softening
softening = 0.05
R2_safe = R2 + softening**2
R = np.sqrt(R2_safe)

Ex = K * Q * X / R**3
Ey = K * Q * Y / R**3
E_mag = np.sqrt(Ex**2 + Ey**2)

# Styling
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.set_facecolor("#0b0f1a")
ax.set_aspect("equal", "box")
ax.set_xlim(-GRID_LIM, GRID_LIM)
ax.set_ylim(-GRID_LIM, GRID_LIM)
ax.set_title("Electric Field of a Point Charge", fontsize=14, pad=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(color="#101223", linewidth=0.6, alpha=0.6)

# Field lines (streamplot)
stream = ax.streamplot(
    X,
    Y,
    Ex,
    Ey,
    color=E_mag,
    linewidth=1.1,
    density=1.6,
    cmap="plasma",
    minlength=0.2,
)

# Quiver arrows (sampled to reduce clutter)
skip = (slice(None, None, 12), slice(None, None, 12))
ax.quiver(
    X[skip],
    Y[skip],
    Ex[skip],
    Ey[skip],
    color="#7dd3ff",
    scale=40,
    width=0.004,
    alpha=0.8,
)

# Charge marker
charge_color = "#ff6b6b" if Q > 0 else "#5ea8ff"
ax.scatter([0], [0], s=400, c=charge_color, edgecolors="white", linewidths=1.5, zorder=5)
ax.text(0, 0, "+" if Q > 0 else "−", ha="center", va="center", fontsize=20, color="white", zorder=6)
ax.text(
    0.02,
    0.02,
    f"q = {Q:+.1f}",
    color="white",
    fontsize=10,
    transform=ax.transAxes,
    ha="left",
    va="bottom",
)

# Colorbar for field magnitude
cbar = fig.colorbar(stream.lines, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("|E| (normalized)")

plt.tight_layout()

if __name__ == "__main__":
    plt.show()
