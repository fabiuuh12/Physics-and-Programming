"""
Two-body Newtonian orbit visualization.

Equations:
  m1 r1'' = G m1 m2 (r2 - r1) / |r2 - r1|^3
  m2 r2'' = G m1 m2 (r1 - r2) / |r1 - r2|^3

Run:
    python two_body_orbit_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
G = 1.0
M1 = 1.0
M2 = 0.8

DT = 0.01
N_FRAMES = 1200

# Initial conditions (roughly elliptical)
r1 = np.array([-0.6, 0.0], dtype=float)
r2 = np.array([0.8, 0.0], dtype=float)

v1 = np.array([0.0, -0.7], dtype=float)
v2 = np.array([0.0, 0.9], dtype=float)

plt.style.use("dark_background")

# Histories
r1_hist = []
r2_hist = []


def accel(r1_val: np.ndarray, r2_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = r2_val - r1_val
    dist = np.linalg.norm(diff) + 1e-9
    a1 = G * M2 * diff / (dist ** 3)
    a2 = -G * M1 * diff / (dist ** 3)
    return a1, a2


def step_rk4(r1_val, r2_val, v1_val, v2_val, dt):
    a1, a2 = accel(r1_val, r2_val)

    k1r1, k1r2 = v1_val, v2_val
    k1v1, k1v2 = a1, a2

    a1, a2 = accel(r1_val + 0.5 * dt * k1r1, r2_val + 0.5 * dt * k1r2)
    k2r1, k2r2 = v1_val + 0.5 * dt * k1v1, v2_val + 0.5 * dt * k1v2
    k2v1, k2v2 = a1, a2

    a1, a2 = accel(r1_val + 0.5 * dt * k2r1, r2_val + 0.5 * dt * k2r2)
    k3r1, k3r2 = v1_val + 0.5 * dt * k2v1, v2_val + 0.5 * dt * k2v2
    k3v1, k3v2 = a1, a2

    a1, a2 = accel(r1_val + dt * k3r1, r2_val + dt * k3r2)
    k4r1, k4r2 = v1_val + dt * k3v1, v2_val + dt * k3v2
    k4v1, k4v2 = a1, a2

    r1_next = r1_val + (dt / 6.0) * (k1r1 + 2 * k2r1 + 2 * k3r1 + k4r1)
    r2_next = r2_val + (dt / 6.0) * (k1r2 + 2 * k2r2 + 2 * k3r2 + k4r2)
    v1_next = v1_val + (dt / 6.0) * (k1v1 + 2 * k2v1 + 2 * k3v1 + k4v1)
    v2_next = v2_val + (dt / 6.0) * (k1v2 + 2 * k2v2 + 2 * k3v2 + k4v2)

    return r1_next, r2_next, v1_next, v2_next


fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.set_aspect("equal")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_facecolor("#05070f")
ax.set_title("Two-body Newtonian orbit")
ax.set_xticks([])
ax.set_yticks([])

(r1_line,) = ax.plot([], [], color="#7de0ff", lw=1.6)
(r2_line,) = ax.plot([], [], color="#ffb578", lw=1.6)
(r1_dot,) = ax.plot([], [], marker="o", markersize=9, color="#7de0ff")
(r2_dot,) = ax.plot([], [], marker="o", markersize=9, color="#ffb578")

# Barycenter marker
ax.plot([0], [0], marker="+", color="#ffffff", markersize=8)


def update(frame: int):
    global r1, r2, v1, v2

    r1, r2, v1, v2 = step_rk4(r1, r2, v1, v2, DT)

    r1_hist.append(r1.copy())
    r2_hist.append(r2.copy())
    if len(r1_hist) > 2500:
        del r1_hist[:400]
        del r2_hist[:400]

    r1_arr = np.array(r1_hist)
    r2_arr = np.array(r2_hist)

    r1_line.set_data(r1_arr[:, 0], r1_arr[:, 1])
    r2_line.set_data(r2_arr[:, 0], r2_arr[:, 1])
    r1_dot.set_data([r1[0]], [r1[1]])
    r2_dot.set_data([r2[0]], [r2[1]])

    return r1_line, r2_line, r1_dot, r2_dot


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=DT * 1000.0, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
