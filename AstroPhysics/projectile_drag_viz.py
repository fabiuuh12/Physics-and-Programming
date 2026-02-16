"""
Projectile motion with quadratic drag.

Equations:
  m dv/dt = -m g y_hat - c |v| v

Run:
    python projectile_drag_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
MASS = 1.0
G = 9.81
C_D = 0.08  # drag coefficient

DT = 0.02
N_FRAMES = 600

# Initial conditions
x, y = 0.0, 0.0
v0 = 12.0
angle = np.deg2rad(50.0)
vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)

plt.style.use("dark_background")

# History
xh = []
yh = []


def accel(vx_val: float, vy_val: float) -> tuple[float, float]:
    speed = np.hypot(vx_val, vy_val)
    ax = -(C_D / MASS) * speed * vx_val
    ay = -G - (C_D / MASS) * speed * vy_val
    return ax, ay


def step_rk4(x_val, y_val, vx_val, vy_val, dt):
    ax1, ay1 = accel(vx_val, vy_val)
    k1x, k1y = vx_val, vy_val
    k1vx, k1vy = ax1, ay1

    ax2, ay2 = accel(vx_val + 0.5 * dt * k1vx, vy_val + 0.5 * dt * k1vy)
    k2x, k2y = vx_val + 0.5 * dt * k1vx, vy_val + 0.5 * dt * k1vy
    k2vx, k2vy = ax2, ay2

    ax3, ay3 = accel(vx_val + 0.5 * dt * k2vx, vy_val + 0.5 * dt * k2vy)
    k3x, k3y = vx_val + 0.5 * dt * k2vx, vy_val + 0.5 * dt * k2vy
    k3vx, k3vy = ax3, ay3

    ax4, ay4 = accel(vx_val + dt * k3vx, vy_val + dt * k3vy)
    k4x, k4y = vx_val + dt * k3vx, vy_val + dt * k3vy
    k4vx, k4vy = ax4, ay4

    x_next = x_val + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_next = y_val + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    vx_next = vx_val + (dt / 6.0) * (k1vx + 2 * k2vx + 2 * k3vx + k4vx)
    vy_next = vy_val + (dt / 6.0) * (k1vy + 2 * k2vy + 2 * k3vy + k4vy)

    return x_next, y_next, vx_next, vy_next


fig, ax = plt.subplots(figsize=(8.5, 5.0))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.set_facecolor("#05070f")
ax.set_title("Projectile motion with quadratic drag")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(color="#111", alpha=0.6)

(traj_line,) = ax.plot([], [], color="#7de0ff", lw=2.0)
(projectile_dot,) = ax.plot([], [], marker="o", markersize=8, color="#ffb578")

info_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="#e8f2ff", fontsize=10, va="top")


def update(frame: int):
    global x, y, vx, vy

    if y < 0:
        return traj_line, projectile_dot, info_text

    x, y, vx, vy = step_rk4(x, y, vx, vy, DT)

    xh.append(x)
    yh.append(y)

    traj_line.set_data(xh, yh)
    projectile_dot.set_data([x], [y])

    speed = np.hypot(vx, vy)
    info_text.set_text(f"v = {speed: .2f}   vx = {vx: .2f}   vy = {vy: .2f}")

    return traj_line, projectile_dot, info_text


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=DT * 1000.0, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
