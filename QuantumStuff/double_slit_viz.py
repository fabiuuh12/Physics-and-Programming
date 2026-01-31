"""
Double-slit beam made of tiny white particles.

Run this file to watch dots launch from the left, test the two narrow slits,
and, only if they line up with an opening, zip across to the detection wall
on the right.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Scene geometry (all values in arbitrary screen units)
X_MIN, X_MAX = -2.5, 10.5
Y_MIN, Y_MAX = -3.0, 3.0
EMITTER_X = -2.0
BARRIER_X = 2.0
SCREEN_X = 9.5

# Slit configuration
SLIT_OFFSET = 1.0      # distance of the slit centers from y=0
SLIT_HEIGHT = 0.8
SLIT_WIDTH = 0.18

# Particle behavior
PARTICLE_SPEED_RANGE = (0.08, 0.14)
SPAWN_INTERVAL = 1       # frames between new particles
INITIAL_PARTICLES = 18
MAX_LIVE_PARTICLES = 120
MAX_IMPACTS = 800
DIFFRACTION_SPREAD = 0.6
EMISSION_BAND = 1.8
BARRIER_JITTER = 0.35


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def inside_slit(y: float) -> bool:
    """
    Return True if y hits one of the two vertical slits in the barrier.
    """
    for center in (-SLIT_OFFSET, SLIT_OFFSET):
        if abs(y - center) <= SLIT_HEIGHT / 2:
            return True
    return False


@dataclass
class Particle:
    color: str
    start_y: float
    barrier_y: float = field(init=False)
    passes_slit: bool = field(init=False)
    final_y: float | None = field(init=False, default=None)
    speed: float = field(
        default_factory=lambda: random.uniform(*PARTICLE_SPEED_RANGE)
    )
    positions: List[np.ndarray] = field(init=False)
    segment_index: int = field(default=0, init=False)
    finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Where it actually hits the barrier line (with some jitter)
        self.barrier_y = clamp(
            self.start_y + random.gauss(0.0, BARRIER_JITTER),
            Y_MIN + 0.4,
            Y_MAX - 0.4,
        )

        # Does this trajectory line up with a slit?
        self.passes_slit = inside_slit(self.barrier_y)

        # If it passes, pick a detection y on the right (simulating diffraction)
        if self.passes_slit:
            spread = random.gauss(0.0, DIFFRACTION_SPREAD)
            self.final_y = clamp(
                self.barrier_y + spread,
                Y_MIN + 0.3,
                Y_MAX - 0.3,
            )

        # Build its path segments:
        # 1) source -> barrier
        self.positions = [
            np.array([EMITTER_X, self.start_y], dtype=float),
            np.array([BARRIER_X, self.barrier_y], dtype=float),
        ]

        # 2) if it passes a slit, barrier -> screen
        if self.passes_slit and self.final_y is not None:
            self.positions.append(np.array([SCREEN_X, self.final_y], dtype=float))

        self._pos = self.positions[0].copy()

    def update(self) -> None:
        """
        Move the particle along its polyline path by `speed` each frame.
        """
        if self.finished:
            return

        travel = self.speed

        while travel > 0 and not self.finished:
            target = self.positions[self.segment_index + 1]
            vector = target - self._pos
            distance = float(np.linalg.norm(vector))

            if distance < 1e-6:
                # Snap to target, advance to next segment
                self._pos = target
                self.segment_index += 1
                if self.segment_index >= len(self.positions) - 1:
                    self.finished = True
                continue

            if travel >= distance:
                # Reach this segment end and continue with remaining travel
                self._pos = target
                travel -= distance
                self.segment_index += 1
                if self.segment_index >= len(self.positions) - 1:
                    self.finished = True
            else:
                # Move partway along the current segment
                self._pos += (vector / distance) * travel
                travel = 0.0

    @property
    def coords(self) -> np.ndarray:
        return self._pos


def build_scene():
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.axis("off")
    ax.set_facecolor("#05060d")

    # Barrier with slits (solid bar first)
    barrier = Rectangle(
        (BARRIER_X - SLIT_WIDTH / 2, Y_MIN),
        SLIT_WIDTH,
        Y_MAX - Y_MIN,
        color="#444954",
        zorder=1,
    )
    ax.add_patch(barrier)

    # Carve out two "slits" using rectangles matching the background
    for offset in (-SLIT_OFFSET, SLIT_OFFSET):
        ax.add_patch(
            Rectangle(
                (BARRIER_X - SLIT_WIDTH / 2 - 0.02, offset - SLIT_HEIGHT / 2),
                SLIT_WIDTH + 0.04,
                SLIT_HEIGHT,
                color=ax.get_facecolor(),
                zorder=2,
            )
        )

    # Detection wall (right)
    ax.plot(
        [SCREEN_X, SCREEN_X],
        [Y_MIN, Y_MAX],
        color="#7f8cfa",
        linewidth=2,
        alpha=0.6,
    )
    ax.text(
        SCREEN_X + 0.1,
        Y_MAX - 0.3,
        "Detection screen",
        color="#b9c4ff",
        ha="left",
        va="top",
        fontsize=11,
    )

    # Source label (left)
    ax.text(
        EMITTER_X,
        Y_MIN + 0.3,
        "Electron / Photon Source",
        color="#8df6ff",
        ha="left",
        va="bottom",
        fontsize=10,
    )

    return fig, ax


def run_animation() -> None:
    fig, ax = build_scene()

    particles: List[Particle] = []
    impact_points: List[List[float]] = []

    # Moving particles (small bright dots)
    moving_scatter = ax.scatter(
        [],
        [],
        s=26,                     # size of flying particles
        facecolors="#ffffff",
        edgecolors="#66a3ff",
        linewidths=0.6,
        alpha=1.0,
        zorder=5,
    )

    # Hits on the detection screen
    impact_scatter = ax.scatter(
        [],
        [],
        s=18,
        color="#c7d5ff",
        alpha=0.75,
        zorder=4,
    )

    frame_counter = {"value": 0}

    def spawn_particle() -> None:
        particles.append(
            Particle(
                color="#ffffff",
                start_y=random.uniform(-EMISSION_BAND, EMISSION_BAND),
            )
        )

    # Initial burst
    for _ in range(INITIAL_PARTICLES):
        spawn_particle()

    def update(_frame):
        frame_counter["value"] += 1

        # Spawn new particles over time
        if (
            frame_counter["value"] % SPAWN_INTERVAL == 0
            and len(particles) < MAX_LIVE_PARTICLES
        ):
            spawn_particle()

        # Update motion
        for particle in particles:
            particle.update()

        # Collect finished particles
        alive_particles = []
        for particle in particles:
            if particle.finished:
                # Only record impact if it actually passed through a slit
                if particle.passes_slit and particle.final_y is not None:
                    impact_points.append([particle.coords[0], particle.coords[1]])
                # If it didn't pass a slit, it's absorbed at the barrier: no screen hit
            else:
                alive_particles.append(particle)

        particles[:] = alive_particles

        # Keep the impact list light
        if len(impact_points) > MAX_IMPACTS:
            del impact_points[: len(impact_points) - MAX_IMPACTS]

        # Update moving particle positions
        if particles:
            coords = np.array([p.coords for p in particles])
        else:
            coords = np.empty((0, 2))
        moving_scatter.set_offsets(coords)

        # Update impact scatter (only passed-through particles)
        if impact_points:
            impact_array = np.array(impact_points)
            impact_scatter.set_offsets(impact_array)
        else:
            impact_scatter.set_offsets(np.empty((0, 2)))

        return moving_scatter, impact_scatter

    anim = FuncAnimation(
        fig,
        update,
        interval=30,
        blit=False,
        cache_frame_data=False,
    )

    # Keep a reference to the animation so it is not garbage-collected
    return anim


if __name__ == "__main__":
    anim = run_animation()
    plt.show()