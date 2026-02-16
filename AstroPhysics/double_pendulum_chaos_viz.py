"""Two nearly identical double pendulums diverging to showcase chaos.

What happens:
- Two double pendulums start with a minuscule difference in angle.
- Their motion is integrated numerically (RK4) and drawn on the same pivot.
- A log plot shows how the angular separation grows over time, a hallmark of chaos.

Run:
    python3 double_pendulum_chaos_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


G = 9.81
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0

DT = 0.01
SUBSTEPS = 5
N_FRAMES = 700

plt.style.use("dark_background")


class DoublePendulum:
    def __init__(self, theta1: float, theta2: float, omega1: float = 0.0, omega2: float = 0.0):
        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = omega1
        self.omega2 = omega2

    def _derivatives(self, theta1, theta2, omega1, omega2):
        delta = theta2 - theta1

        denom1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) ** 2
        denom2 = (L2 / L1) * denom1

        domega1 = (
            M2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta)
            + M2 * G * np.sin(theta2) * np.cos(delta)
            + M2 * L2 * omega2**2 * np.sin(delta)
            - (M1 + M2) * G * np.sin(theta1)
        ) / denom1

        domega2 = (
            -M2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta)
            + (M1 + M2)
            * (
                G * np.sin(theta1) * np.cos(delta)
                - L1 * omega1**2 * np.sin(delta)
                - G * np.sin(theta2)
            )
        ) / denom2

        return np.array([omega1, omega2, domega1, domega2])

    def step(self, dt: float):
        state = np.array([self.theta1, self.theta2, self.omega1, self.omega2])
        k1 = self._derivatives(*state)
        k2 = self._derivatives(*(state + 0.5 * dt * k1))
        k3 = self._derivatives(*(state + 0.5 * dt * k2))
        k4 = self._derivatives(*(state + dt * k3))

        state += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.theta1, self.theta2, self.omega1, self.omega2 = state

    def positions(self):
        x1 = L1 * np.sin(self.theta1)
        y1 = -L1 * np.cos(self.theta1)
        x2 = x1 + L2 * np.sin(self.theta2)
        y2 = y1 - L2 * np.cos(self.theta2)
        return (0.0, x1, x2), (0.0, y1, y2)


pend_a = DoublePendulum(theta1=np.pi / 2, theta2=-np.pi / 6)
pend_b = DoublePendulum(theta1=np.pi / 2 + 0.0008, theta2=-np.pi / 6)

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.3)
ax_pend = fig.add_subplot(gs[0])
ax_diff = fig.add_subplot(gs[1])

ax_pend.set_aspect("equal")
ax_pend.set_xlim(-2.2, 2.2)
ax_pend.set_ylim(-2.4, 1.0)
ax_pend.set_xticks([])
ax_pend.set_yticks([])
ax_pend.set_facecolor("#05070f")
ax_pend.set_title("Double Pendulum Chaos (blue vs orange)")

line_a, = ax_pend.plot([], [], "-o", color="#7ad3ff", lw=2.4, markersize=6)
line_b, = ax_pend.plot([], [], "-o", color="#ffb878", lw=2.4, markersize=6, alpha=0.85)

ax_diff.set_xlim(0, DT * SUBSTEPS * N_FRAMES)
ax_diff.set_yscale("log")
ax_diff.set_ylim(1e-6, 1.0)
ax_diff.set_xlabel("Time")
ax_diff.set_ylabel("|θ₂ₐ - θ₂ᵦ| (log scale)")
ax_diff.grid(color="#111422", linewidth=0.6, alpha=0.7)

diff_line, = ax_diff.plot([], [], color="#ffef9f", linewidth=2.0)

time_history: list[float] = []
diff_history: list[float] = []
info_text = fig.text(0.5, 0.05, "", ha="center", color="#f5f5ff", fontsize=11)


def update(frame: int):
    t = frame * DT * SUBSTEPS
    for _ in range(SUBSTEPS):
        pend_a.step(DT)
        pend_b.step(DT)

    xs_a, ys_a = pend_a.positions()
    xs_b, ys_b = pend_b.positions()

    line_a.set_data(xs_a, ys_a)
    line_b.set_data(xs_b, ys_b)

    diff = abs(pend_a.theta2 - pend_b.theta2)
    time_history.append(t)
    diff_history.append(diff)

    diff_line.set_data(time_history, diff_history)
    ax_diff.set_xlim(0, max(5.0, time_history[-1]))

    info_text.set_text(f"Current angular separation ≈ {diff: .2e} rad")

    return line_a, line_b, diff_line, info_text


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * SUBSTEPS * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()
