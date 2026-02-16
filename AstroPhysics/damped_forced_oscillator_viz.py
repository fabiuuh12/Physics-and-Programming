"""
Damped and forced harmonic oscillator visualization.

Equation: x'' + 2 beta x' + omega0^2 x = (F0/m) cos(omega_d t)

Run:
    python damped_forced_oscillator_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
MASS = 1.0
OMEGA0 = 2.2
BETA = 0.18
F0 = 1.0
OMEGA_D = 1.6

DT = 0.02
N_FRAMES = 900

# Initial conditions
x = 0.8
v = 0.0

plt.style.use("dark_background")

# History
th = []
xh = []

def accel(x_val: float, v_val: float, t: float) -> float:
    return -(2.0 * BETA) * v_val - (OMEGA0 ** 2) * x_val + (F0 / MASS) * np.cos(OMEGA_D * t)


def step_rk4(x_val: float, v_val: float, t: float, dt: float) -> tuple[float, float]:
    k1x = v_val
    k1v = accel(x_val, v_val, t)

    k2x = v_val + 0.5 * dt * k1v
    k2v = accel(x_val + 0.5 * dt * k1x, v_val + 0.5 * dt * k1v, t + 0.5 * dt)

    k3x = v_val + 0.5 * dt * k2v
    k3v = accel(x_val + 0.5 * dt * k2x, v_val + 0.5 * dt * k2v, t + 0.5 * dt)

    k4x = v_val + dt * k3v
    k4v = accel(x_val + dt * k3x, v_val + dt * k3v, t + dt)

    x_next = x_val + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v_val + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x_next, v_next


fig = plt.figure(figsize=(9, 6))
ax_motion = fig.add_subplot(2, 1, 1)
ax_trace = fig.add_subplot(2, 1, 2)

ax_motion.set_xlim(-2.5, 2.5)
ax_motion.set_ylim(-0.9, 0.9)
ax_motion.set_xticks([])
ax_motion.set_yticks([])
ax_motion.set_facecolor("#05070f")
ax_motion.set_title("Damped + Driven Harmonic Oscillator")

(spring_line,) = ax_motion.plot([], [], color="#9fb8ff", lw=2.0)
(mass_dot,) = ax_motion.plot([], [], marker="o", markersize=16, color="#ffb578")
(driver_line,) = ax_motion.plot([], [], color="#7ef6c9", lw=2.0, alpha=0.8)

ax_trace.set_xlim(0, 14)
ax_trace.set_ylim(-1.7, 1.7)
ax_trace.set_xlabel("Time")
ax_trace.set_ylabel("x(t)")
ax_trace.grid(color="#111", alpha=0.6)
(x_line,) = ax_trace.plot([], [], color="#7de0ff", lw=2.0)

info_text = ax_motion.text(0.02, 0.85, "", transform=ax_motion.transAxes, fontsize=10, color="#e8f2ff")


def update(frame: int):
    global x, v
    t = frame * DT

    x, v = step_rk4(x, v, t, DT)

    # Spring
    n_coils = 16
    xs = np.linspace(-1.5, x, n_coils)
    ys = 0.12 * np.sin(np.linspace(0, 4 * np.pi, n_coils))
    spring_line.set_data(xs, ys)
    mass_dot.set_data([x], [0])

    # Driver signal (visual reference)
    drive = 0.6 * np.cos(OMEGA_D * t)
    driver_line.set_data([-2.2, -2.2], [0, drive])

    # History
    th.append(t)
    xh.append(x)
    x_line.set_data(th, xh)
    if t > ax_trace.get_xlim()[1] - 0.5:
        ax_trace.set_xlim(0, t + 1.0)

    info_text.set_text(f"x = {x: .3f}   v = {v: .3f}   drive = {drive: .2f}")

    return spring_line, mass_dot, driver_line, x_line, info_text


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=DT * 1000.0, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
