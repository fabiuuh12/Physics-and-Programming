"""
Particle Accelerator Visualization

Run this file to see a simple visualization inspired by a circular particle accelerator.
Small "atoms" (particles) accelerate around a ring, with occasional bright collision
events at the interaction point on the right side.

This is purely a visual toy, not a physically accurate simulation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Scene settings
plt.style.use("dark_background")

FIG_SIZE = (10, 10)
BACKGROUND_COLOR = "#020308"

# Accelerator geometry
CENTER = (0.0, 0.0)
RADIUS = 4.0

# Particle settings
NUM_PARTICLES = 80
MIN_SPEED = 0.02       # radians per frame
MAX_SPEED = 0.06
PARTICLE_SIZE = 28
PARTICLE_COLOR = "#9be7ff"

# "Atoms" (we'll treat them as neutral-ish blobs moving in the ring)
# A few larger ones mixed into the beam.
NUM_ATOMS = 14
ATOM_SIZE = 60
ATOM_COLOR = "#ffc6a5"

# Collision / interaction point
INTERACTION_ANGLE = 0.0        # 0 rad = (RADIUS, 0) on the right side
INTERACTION_WINDOW = 0.06      # how close in angle counts as "at the IP"
COLLISION_CHANCE = 0.35        # probability that a close pass triggers a flash
MAX_COLLISION_POINTS = 120

# Flash / collision visuals
FLASH_MAX_RADIUS = 0.6
FLASH_MIN_RADIUS = 0.2
FLASH_DECAY = 0.92
FLASH_COLOR = "#ffe66d"


@dataclass
class BeamParticle:
    angle: float
    speed: float
    radius: float
    size: float
    color: str
    alpha: float = field(default=1.0)

    def step(self) -> None:
        # Advance angle; wrap into [-pi, pi] range for stability/consistency
        self.angle += self.speed
        if self.angle > math.pi:
            self.angle -= 2.0 * math.pi

    @property
    def xy(self) -> tuple[float, float]:
        x = CENTER[0] + self.radius * math.cos(self.angle)
        y = CENTER[1] + self.radius * math.sin(self.angle)
        return x, y


@dataclass
class CollisionFlash:
    x: float
    y: float
    radius: float
    alpha: float

    def decay(self) -> None:
        self.radius *= FLASH_DECAY
        self.alpha *= FLASH_DECAY

    @property
    def is_alive(self) -> bool:
        return self.alpha > 0.05 and self.radius > 0.05


def make_beam() -> List[BeamParticle]:
    particles: List[BeamParticle] = []

    # Fast small particles
    for _ in range(NUM_PARTICLES):
        angle = random.uniform(-math.pi, math.pi)
        speed = random.uniform(MIN_SPEED, MAX_SPEED)
        # Slight random radial offset so the beam looks thicker
        r = RADIUS + random.uniform(-0.12, 0.12)
        particles.append(
            BeamParticle(
                angle=angle,
                speed=speed,
                radius=r,
                size=PARTICLE_SIZE,
                color=PARTICLE_COLOR,
                alpha=1.0,
            )
        )

    # Bigger "atoms" mixed into the ring (slightly slower, brighter)
    for _ in range(NUM_ATOMS):
        angle = random.uniform(-math.pi, math.pi)
        speed = random.uniform(MIN_SPEED * 0.35, MIN_SPEED * 0.9)
        r = RADIUS + random.uniform(-0.22, 0.22)
        particles.append(
            BeamParticle(
                angle=angle,
                speed=speed,
                radius=r,
                size=ATOM_SIZE,
                color=ATOM_COLOR,
                alpha=0.9,
            )
        )

    return particles


def build_scene():
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.axis("off")

    # Outer ring (accelerator)
    ring = Circle(
        CENTER,
        RADIUS,
        fill=False,
        linewidth=2.2,
        edgecolor="#3a4a6b",
        alpha=0.9,
    )
    ax.add_patch(ring)

    # Slight inner glow to make it feel like a beam pipe
    inner_ring = Circle(
        CENTER,
        RADIUS - 0.15,
        fill=False,
        linewidth=1.1,
        edgecolor="#1b263b",
        alpha=0.7,
    )
    ax.add_patch(inner_ring)

    # Mark the interaction point on the right
    ip_x = CENTER[0] + RADIUS * math.cos(INTERACTION_ANGLE)
    ip_y = CENTER[1] + RADIUS * math.sin(INTERACTION_ANGLE)
    ax.scatter(
        [ip_x],
        [ip_y],
        s=60,
        color="#7f8cff",
        alpha=0.95,
        zorder=5,
    )
    ax.text(
        ip_x + 0.15,
        ip_y + 0.28,
        "Interaction Point",
        color="#9fb4ff",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    # Label
    ax.text(
        0.0,
        -5.1,
        "Circular Particle Accelerator (atoms / particles circulating & colliding)",
        ha="center",
        va="top",
        fontsize=10,
        color="#b3c7ff",
    )

    return fig, ax


def run_animation():
    fig, ax = build_scene()

    beam = make_beam()
    flashes: List[CollisionFlash] = []

    # Scatter collection for particles
    scatter = ax.scatter(
        [],
        [],
        s=[],
        c=[],
        alpha=1.0,
        edgecolors="none",
        zorder=4,
    )

    # We'll draw collision flashes as Circle patches
    flash_patches: List[Circle] = []

    def spawn_collision(x: float, y: float):
        if len(flashes) >= MAX_COLLISION_POINTS:
            return
        radius = random.uniform(FLASH_MIN_RADIUS, FLASH_MAX_RADIUS)
        flashes.append(
            CollisionFlash(
                x=x,
                y=y,
                radius=radius,
                alpha=1.0,
            )
        )
        patch = Circle(
            (x, y),
            radius=radius,
            edgecolor=FLASH_COLOR,
            facecolor=FLASH_COLOR,
            alpha=1.0,
            linewidth=0.0,
            zorder=6,
        )
        flash_patches.append(patch)
        ax.add_patch(patch)

    def update(_frame):
        xs = []
        ys = []
        sizes = []
        colors = []

        # Update beam particles
        for p in beam:
            p.step()
            x, y = p.xy
            xs.append(x)
            ys.append(y)
            sizes.append(p.size)
            colors.append(p.color)

            # Check if this particle passes through the interaction region
            # normalize angle difference into [-pi, pi]
            diff = ((p.angle - INTERACTION_ANGLE + math.pi) % (2.0 * math.pi)) - math.pi
            if abs(diff) < INTERACTION_WINDOW and random.random() < COLLISION_CHANCE:
                spawn_collision(x, y)

        # Update scatter for particles
        scatter.set_offsets(np.column_stack((xs, ys)))
        scatter.set_sizes(np.array(sizes))
        scatter.set_color(colors)

        # Update and fade flashes
        alive_flashes: List[CollisionFlash] = []
        alive_patches: List[Circle] = []
        for flash, patch in list(zip(flashes, flash_patches)):
            flash.decay()
            if flash.is_alive:
                patch.center = (flash.x, flash.y)
                patch.radius = flash.radius
                patch.set_alpha(flash.alpha)
                alive_flashes.append(flash)
                alive_patches.append(patch)
            else:
                patch.remove()

        flashes[:] = alive_flashes
        flash_patches[:] = alive_patches

        return (scatter, *flash_patches)

    anim = FuncAnimation(
        fig,
        update,
        interval=30,
        blit=False,
        cache_frame_data=False,
    )

    return anim


if __name__ == "__main__":
    anim = run_animation()
    plt.show()
