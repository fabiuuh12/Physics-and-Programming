"""Relativistic time-dilation race between a stay-at-home clock and a fast traveler.

Panels:
- Top: cartoon of Earth clock vs. spaceship clock, with spaceship sweeping back and forth.
- Bottom-left: Minkowski-style worldlines showing coordinate time (ct) vs. position.
- Bottom-right: Proper time accumulated by each clock as a function of coordinate time.

Run:
    python3 relativistic_time_dilation_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


C = 1.0  # natural units
DT = 0.04
N_FRAMES = 600

TRACK_CENTER = 2.6
TRACK_AMPLITUDE = 2.2
TRACK_OMEGA = 0.32  # angular frequency of spacecraft oscillation

plt.style.use("dark_background")

fig = plt.figure(figsize=(10, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2], hspace=0.33, wspace=0.28)

ax_scene = fig.add_subplot(gs[0, :])
ax_world = fig.add_subplot(gs[1, 0])
ax_tau = fig.add_subplot(gs[1, 1])


# ------------------
# Scene setup (top)
# ------------------
ax_scene.set_xlim(-0.5, 6.0)
ax_scene.set_ylim(-0.8, 1.4)
ax_scene.set_xticks([])
ax_scene.set_yticks([])
ax_scene.set_facecolor("#04050a")
ax_scene.set_title("Relativistic Time Dilation: Earth clock vs. fast traveler", fontsize=12)

ax_scene.plot([0, 5.5], [0.0, 0.0], color="#1e2a45", linewidth=3.2, alpha=0.9)
ax_scene.text(5.6, -0.05, "Space path", color="#6b78a0", fontsize=9, va="center")

earth_clock = Circle((-0.05, 0.25), 0.35, facecolor="#172133", edgecolor="#7fc5ff", linewidth=1.5)
ship_clock = Circle((TRACK_CENTER, 0.25), 0.32, facecolor="#311a22", edgecolor="#ffb3a8", linewidth=1.5)
ax_scene.add_patch(earth_clock)
ax_scene.add_patch(ship_clock)

earth_label = ax_scene.text(-0.05, -0.35, "Earth clock (rest)", color="#a2f6ff", fontsize=10, ha="center")
ship_label = ax_scene.text(
    TRACK_CENTER,
    -0.35,
    "Traveler clock (moving)",
    color="#ffccb7",
    fontsize=10,
    ha="center",
)

earth_time_text = ax_scene.text(-0.05, 0.3, "", color="#e0fbff", fontsize=13, ha="center")
ship_time_text = ax_scene.text(TRACK_CENTER, 0.3, "", color="#ffd7c8", fontsize=13, ha="center")
gamma_text = ax_scene.text(3.0, 0.95, "", color="#f8f8ff", fontsize=11)


# -----------------------------
# Worldline / proper time plots
# -----------------------------
ax_world.set_xlim(-0.5, 6.0)
ax_world.set_ylim(0.0, DT * N_FRAMES * 1.05)
ax_world.set_xlabel("Space (light-seconds)")
ax_world.set_ylabel("ct (seconds)")
ax_world.grid(color="#101223", linewidth=0.6, alpha=0.7)
ax_world.set_title("Worldlines")

world_earth_line, = ax_world.plot([], [], color="#88f6ff", linewidth=2.0, label="Earth")
world_ship_line, = ax_world.plot([], [], color="#ff9c85", linewidth=2.0, label="Traveler")
ax_world.legend(loc="upper right", fontsize=8)

ax_tau.set_xlim(0.0, DT * N_FRAMES)
ax_tau.set_ylim(0.0, DT * N_FRAMES)
ax_tau.set_xlabel("Coordinate time t")
ax_tau.set_ylabel("Proper time τ")
ax_tau.grid(color="#101223", linewidth=0.6, alpha=0.7)
ax_tau.set_title("Proper time accumulation")

tau_earth_line, = ax_tau.plot([], [], color="#91f0ff", linewidth=2.0, label="Earth τ")
tau_ship_line, = ax_tau.plot([], [], color="#ffc2a7", linewidth=2.0, label="Traveler τ")
ax_tau.legend(loc="upper left", fontsize=8)


# Histories
world_t: list[float] = []
world_x_ship: list[float] = []
world_x_earth: list[float] = []

tau_history_t: list[float] = []
tau_history_earth: list[float] = []
tau_history_ship: list[float] = []

earth_tau = 0.0
ship_tau = 0.0


def update(frame: int):
    global earth_tau, ship_tau

    t = frame * DT

    ship_x = TRACK_CENTER + TRACK_AMPLITUDE * np.sin(TRACK_OMEGA * t)
    ship_v = TRACK_AMPLITUDE * TRACK_OMEGA * np.cos(TRACK_OMEGA * t)
    beta = np.clip(ship_v / C, -0.999, 0.999)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    earth_tau += DT  # gamma = 1
    ship_tau += DT / gamma

    # Update scene
    ship_clock.center = (ship_x, 0.25)
    ship_label.set_position((ship_x, -0.35))
    ship_time_text.set_position((ship_x, 0.3))

    earth_time_text.set_text(f"{earth_tau:5.2f} s")
    ship_time_text.set_text(f"{ship_tau:5.2f} s")
    gamma_text.set_text(f"Instant speed ≈ {beta * C: .2f} c\nLorentz factor γ ≈ {gamma:4.2f}")

    # Record worldline data
    world_t.append(t)
    world_x_ship.append(ship_x)
    world_x_earth.append(0.0)

    world_ship_line.set_data(world_x_ship, world_t)
    world_earth_line.set_data(world_x_earth, world_t)

    ax_world.set_ylim(0.0, max(DT * N_FRAMES * 0.25, world_t[-1] + 1.0))

    # Proper time history
    tau_history_t.append(t)
    tau_history_earth.append(earth_tau)
    tau_history_ship.append(ship_tau)

    tau_earth_line.set_data(tau_history_t, tau_history_earth)
    tau_ship_line.set_data(tau_history_t, tau_history_ship)

    ax_tau.set_xlim(0.0, max(DT * N_FRAMES * 0.2, tau_history_t[-1] + 0.5))
    ax_tau.set_ylim(0.0, max(earth_tau, ship_tau) + 0.5)

    return (
        ship_clock,
        ship_label,
        earth_time_text,
        ship_time_text,
        gamma_text,
        world_ship_line,
        world_earth_line,
        tau_earth_line,
        tau_ship_line,
    )


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
