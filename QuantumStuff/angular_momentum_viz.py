"""
Angular momentum visualization for a particle in a central force.

We show that L_z = m (x v_y - y v_x) stays constant in a central potential.

Run:
    python angular_momentum_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
MASS = 1.0
K = 1.2  # strength of central force F = -k r
DT = 0.02
N_FRAMES = 900

# Initial state
x, y = 1.6, 0.2
vx, vy = 0.0, 1.1

plt.style.use("dark_background")

# History
t_hist = []
L_hist = []


def accel(x_val: float, y_val: float) -> tuple[float, float]:
    ax = -(K / MASS) * x_val
    ay = -(K / MASS) * y_val
    return ax, ay


def step_rk4(x_val, y_val, vx_val, vy_val, dt):
    ax1, ay1 = accel(x_val, y_val)
    k1x, k1y = vx_val, vy_val
    k1vx, k1vy = ax1, ay1

    ax2, ay2 = accel(x_val + 0.5 * dt * k1x, y_val + 0.5 * dt * k1y)
    k2x, k2y = vx_val + 0.5 * dt * k1vx, vy_val + 0.5 * dt * k1vy
    k2vx, k2vy = ax2, ay2

    ax3, ay3 = accel(x_val + 0.5 * dt * k2x, y_val + 0.5 * dt * k2y)
    k3x, k3y = vx_val + 0.5 * dt * k2vx, vy_val + 0.5 * dt * k2vy
    k3vx, k3vy = ax3, ay3

    ax4, ay4 = accel(x_val + dt * k3x, y_val + dt * k3y)
    k4x, k4y = vx_val + dt * k3vx, vy_val + dt * k3vy
    k4vx, k4vy = ax4, ay4

    x_next = x_val + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_next = y_val + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    vx_next = vx_val + (dt / 6.0) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
    vy_next = vy_val + (dt / 6.0) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)

    return x_next, y_next, vx_next, vy_next


fig = plt.figure(figsize=(9, 5.5))
ax_orbit = fig.add_subplot(1, 2, 1)
ax_L = fig.add_subplot(1, 2, 2)

ax_orbit.set_aspect("equal")
ax_orbit.set_xlim(-2.2, 2.2)
ax_orbit.set_ylim(-2.2, 2.2)
ax_orbit.set_title("Central force orbit")
ax_orbit.set_facecolor("#05070f")
ax_orbit.set_xticks([])
ax_orbit.set_yticks([])

# Center marker
ax_orbit.plot([0], [0], marker="o", markersize=6, color="#ffffff")

(path_line,) = ax_orbit.plot([], [], color="#7de0ff", lw=1.8)
(particle_dot,) = ax_orbit.plot([], [], marker="o", markersize=10, color="#ffb578")
(velocity_arrow,) = ax_orbit.plot([], [], color="#9ef6ff", lw=1.4)

ax_L.set_title("Angular momentum Lz")
ax_L.set_xlabel("Time")
ax_L.set_ylabel("Lz")
ax_L.grid(color="#111", alpha=0.6)
ax_L.set_facecolor("#05070f")
L_line, = ax_L.plot([], [], color="#ffd27c", lw=2.0)

x_hist = []
y_hist = []


def update(frame: int):
    global x, y, vx, vy
    t = frame * DT

    x, y, vx, vy = step_rk4(x, y, vx, vy, DT)

    # Path history
    x_hist.append(x)
    y_hist.append(y)
    if len(x_hist) > 1200:
        del x_hist[:200]
        del y_hist[:200]

    path_line.set_data(x_hist, y_hist)
    particle_dot.set_data([x], [y])

    # Velocity arrow
    vx_n, vy_n = vx, vy
    velocity_arrow.set_data([x, x + 0.35 * vx_n], [y, y + 0.35 * vy_n])

    # Angular momentum Lz
    Lz = MASS * (x * vy - y * vx)
    t_hist.append(t)
    L_hist.append(Lz)
    L_line.set_data(t_hist, L_hist)
    if t > ax_L.get_xlim()[1] - 0.5:
        ax_L.set_xlim(0, t + 1.0)

    return path_line, particle_dot, velocity_arrow, L_line


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=DT * 1000.0, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
