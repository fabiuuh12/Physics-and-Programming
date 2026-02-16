"""
Simple Harmonic Oscillator (mass-spring) visualization.

Equation: x'' + (k/m) x = 0

Run:
    python simple_harmonic_oscillator_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
MASS = 1.0
K = 4.0
OMEGA0 = np.sqrt(K / MASS)

DT = 0.02
N_FRAMES = 800

# Initial conditions
x0 = 1.0
v0 = 0.0

plt.style.use("dark_background")

# State
x = x0
v = v0

# For plotting history
t_hist = []
x_hist = []


def accel(x_val: float) -> float:
    return -(K / MASS) * x_val


def step_rk4(x_val: float, v_val: float, dt: float) -> tuple[float, float]:
    # x' = v, v' = a(x)
    k1x = v_val
    k1v = accel(x_val)

    k2x = v_val + 0.5 * dt * k1v
    k2v = accel(x_val + 0.5 * dt * k1x)

    k3x = v_val + 0.5 * dt * k2v
    k3v = accel(x_val + 0.5 * dt * k2x)

    k4x = v_val + dt * k3v
    k4v = accel(x_val + dt * k3x)

    x_next = x_val + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    v_next = v_val + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x_next, v_next


# Figure
fig = plt.figure(figsize=(8.5, 5.5))
ax_motion = fig.add_subplot(2, 1, 1)
ax_trace = fig.add_subplot(2, 1, 2)

ax_motion.set_xlim(-2.2, 2.2)
ax_motion.set_ylim(-0.8, 0.8)
ax_motion.set_xticks([])
ax_motion.set_yticks([])
ax_motion.set_title("Simple Harmonic Oscillator: mass-spring")
ax_motion.set_facecolor("#05070f")

# Spring anchor and mass
(anchor_line,) = ax_motion.plot([-2.0, -1.4], [0, 0], color="#6fa8ff", lw=2.0)
(spring_line,) = ax_motion.plot([], [], color="#9fc7ff", lw=2.0)
(mass_dot,) = ax_motion.plot([], [], marker="o", markersize=16, color="#ffd27c")

# Trace plot
ax_trace.set_xlim(0, 12)
ax_trace.set_ylim(-1.5, 1.5)
ax_trace.set_xlabel("Time")
ax_trace.set_ylabel("x(t)")
ax_trace.grid(color="#111" , alpha=0.6)
(x_line,) = ax_trace.plot([], [], color="#7de0ff", lw=2.0)

info_text = ax_motion.text(0.02, 0.85, "", transform=ax_motion.transAxes, fontsize=10, color="#e8f2ff")


def update(frame: int):
    global x, v
    t = frame * DT

    # Integrate
    x, v = step_rk4(x, v, DT)

    # Spring shape
    n_coils = 16
    xs = np.linspace(-1.4, x, n_coils)
    ys = 0.12 * np.sin(np.linspace(0, 4 * np.pi, n_coils))
    spring_line.set_data(xs, ys)
    mass_dot.set_data([x], [0])

    # History
    t_hist.append(t)
    x_hist.append(x)
    x_line.set_data(t_hist, x_hist)
    if t > ax_trace.get_xlim()[1] - 0.5:
        ax_trace.set_xlim(0, t + 1.0)

    energy = 0.5 * MASS * v * v + 0.5 * K * x * x
    info_text.set_text(f"x = {x: .3f}   v = {v: .3f}   E = {energy: .3f}")

    return spring_line, mass_dot, x_line, info_text


anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=DT * 1000.0, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
