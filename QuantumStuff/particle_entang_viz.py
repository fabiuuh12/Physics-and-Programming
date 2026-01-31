"""
particle_entang_viz.py — Stylized particle entanglement visualization.

Purely conceptual & visual:
- An entangled pair is created at the center.
- The two particles fly apart in opposite directions.
- While unmeasured, they are shown as a neutral, shared color.
- When one side "measures" its particle (crossing a detector line),
  both partners instantly take on correlated opposite colors.
- Over time you see many pairs and their perfectly correlated outcomes.

This is NOT a physical simulator; it's an intuition aid.
Run:
    python particle_entang_viz.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.style.use("dark_background")

# Scene geometry
FIGSIZE = (12, 5)
X_MIN, X_MAX = -8.0, 8.0
Y_MIN, Y_MAX = -3.0, 3.0

SOURCE_X = 0.0
MEAS_L_X = -5.0
MEAS_R_X = 5.0

# Visual parameters
PAIR_SPAWN_INTERVAL = 10          # frames between new entangled pairs
MAX_PAIRS = 90

BASE_SPEED = 0.14
SPEED_JITTER = 0.04

UNMEASURED_COLOR = "#d9b3ff"      # shared color before collapse
SPIN_UP_COLOR = "#7fe9ff"         # e.g. "up"
SPIN_DOWN_COLOR = "#ffb16f"       # e.g. "down"

PARTICLE_SIZE = 65

TRAIL_FADE = 0.96                 # how quickly trails fade
TRAIL_LENGTH = 24                 # max stored positions per particle


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: str = UNMEASURED_COLOR
    measured: bool = False
    history: List[tuple[float, float]] = field(default_factory=list)

    def step(self):
        self.x += self.vx
        self.y += self.vy
        # Store positions for trail visualization
        self.history.append((self.x, self.y))
        if len(self.history) > TRAIL_LENGTH:
            self.history.pop(0)


@dataclass
class EntangledPair:
    left: Particle
    right: Particle
    collapsed: bool = False
    # outcome: +1/-1 for left; right is opposite if anti-correlated
    outcome_left: Optional[int] = None

    def step(self):
        self.left.step()
        self.right.step()

    def check_measurement_and_collapse(self):
        """
        If either particle crosses its measurement line for the first time,
        choose a random outcome and set perfectly anti-correlated colors.
        """
        if self.collapsed:
            return

        left_cross = self.left.x <= MEAS_L_X
        right_cross = self.right.x >= MEAS_R_X

        if not (left_cross or right_cross):
            return

        # Choose random spin outcome for "left": +1 or -1
        self.outcome_left = random.choice([-1, 1])
        self.collapsed = True

        # Assign colors based on outcome
        if self.outcome_left == +1:
            self.left.color = SPIN_UP_COLOR
            self.right.color = SPIN_DOWN_COLOR
        else:
            self.left.color = SPIN_DOWN_COLOR
            self.right.color = SPIN_UP_COLOR

        self.left.measured = True
        self.right.measured = True


def spawn_entangled_pair() -> EntangledPair:
    # Small random offset in y so pairs don't all align
    base_y = random.uniform(-1.4, 1.4)

    # symmetric velocities: left goes left, right goes right
    speed_l = BASE_SPEED + random.uniform(-SPEED_JITTER, SPEED_JITTER)
    speed_r = BASE_SPEED + random.uniform(-SPEED_JITTER, SPEED_JITTER)

    # tiny angular tilt so they fan out
    tilt = random.uniform(-0.16, 0.16)

    left = Particle(
        x=SOURCE_X,
        y=base_y,
        vx=-speed_l * math.cos(tilt),
        vy=speed_l * math.sin(tilt),
    )
    right = Particle(
        x=SOURCE_X,
        y=base_y,
        vx=speed_r * math.cos(tilt),
        vy=speed_r * math.sin(tilt),
    )
    return EntangledPair(left=left, right=right)


def build_scene():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#05060d")

    # Source label
    ax.text(
        SOURCE_X,
        Y_MIN + 0.15,
        "Entangled pair source",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#9fb4ff",
    )

    # Measurement regions
    ax.axvline(MEAS_L_X, color="#445", linestyle="--", linewidth=1.1, alpha=0.7)
    ax.axvline(MEAS_R_X, color="#445", linestyle="--", linewidth=1.1, alpha=0.7)

    ax.text(
        MEAS_L_X,
        Y_MAX - 0.2,
        "Detector A",
        ha="center",
        va="top",
        fontsize=8,
        color="#8df6ff",
    )
    ax.text(
        MEAS_R_X,
        Y_MAX - 0.2,
        "Detector B",
        ha="center",
        va="top",
        fontsize=8,
        color="#ffddaa",
    )

    # Legend-ish hint
    ax.text(
        X_MIN + 0.2,
        Y_MAX - 0.2,
        "Unmeasured pair: shared purple   |   Measured outcomes: teal / orange (perfectly opposite)",
        ha="left",
        va="top",
        fontsize=7,
        color="#b8c7ff",
    )

    return fig, ax


def run_animation():
    fig, ax = build_scene()

    pairs: List[EntangledPair] = []
    frame_counter = 0

    # We'll draw particles with a scatter for current positions,
    # and faint line segments for short trails.
    scatter = ax.scatter([], [], s=PARTICLE_SIZE, c=[], edgecolors="none", zorder=4)
    trail_lines = []

    def update(frame: int):
        nonlocal frame_counter, trail_lines
        frame_counter += 1

        # Spawn new pairs over time
        if frame_counter % PAIR_SPAWN_INTERVAL == 0 and len(pairs) < MAX_PAIRS:
            pairs.append(spawn_entangled_pair())

        # Step each pair
        for pair in pairs:
            pair.step()
            pair.check_measurement_and_collapse()

        # Remove particles that flew far off screen
        alive_pairs = []
        for pair in pairs:
            if (
                X_MIN - 1.0 <= pair.left.x <= X_MAX + 1.0
                and X_MIN - 1.0 <= pair.right.x <= X_MAX + 1.0
            ):
                alive_pairs.append(pair)
        pairs[:] = alive_pairs

        # Clear old trail artists
        for ln in trail_lines:
            ln.remove()
        trail_lines = []

        # Collect positions & colors
        xs = []
        ys = []
        cs = []

        for pair in pairs:
            for p in (pair.left, pair.right):
                xs.append(p.x)
                ys.append(p.y)
                cs.append(p.color)

                # draw trail as faint line following history
                if len(p.history) > 1:
                    hx, hy = zip(*p.history)
                    ln_color = p.color if p.measured else UNMEASURED_COLOR
                    alpha = 0.13 if p.measured else 0.09
                    (ln,) = ax.plot(
                        hx,
                        hy,
                        linewidth=1.0,
                        color=ln_color,
                        alpha=alpha,
                        zorder=2,
                    )
                    trail_lines.append(ln)

        if xs:
            coords = np.column_stack((xs, ys))
            scatter.set_offsets(coords)
            scatter.set_color(cs)
            scatter.set_sizes([PARTICLE_SIZE] * len(xs))
        else:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])

        return [scatter, *trail_lines]

    anim = FuncAnimation(
        fig,
        update,
        interval=40,
        blit=False,
        cache_frame_data=False,
    )

    return anim


if __name__ == "__main__":
    anim = run_animation()
    plt.show()
