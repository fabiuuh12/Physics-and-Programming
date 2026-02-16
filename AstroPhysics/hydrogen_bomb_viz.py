"""
hydrogen_fusion_viz.py — Stylized hydrogen fusion visualization (safe & non-technical).

This is an artistic animation of hydrogen-like particles fusing inside a stellar core.
It is NOT a physics-accurate model and includes no technical or operational details
about weapons or real reactors.

Visual idea:
- Small blue/teal "H" particles (protons) drifting in a hot plasma core.
- When a few get close, they "fuse" into a larger golden "He" cluster.
- Each fusion event emits bright photon streaks/light bursts.
- The background glows like a star's core, pulsing with energy.

Run:
    python hydrogen_bomb_viz.py
(or rename the file to hydrogen_fusion_viz.py if you prefer.)
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

plt.style.use("dark_background")

# Canvas
FIGSIZE = (8, 8)
X_MIN, X_MAX = -4.5, 4.5
Y_MIN, Y_MAX = -4.5, 4.5

# Core visual parameters (purely aesthetic)
NUM_H_PARTICLES = 70          # hydrogen-like particles
H_RADIUS = 0.11
H_COLOR = "#7fe9ff"

HE_RADIUS = 0.22
HE_COLOR = "#ffd86a"

PHOTON_COLOR = "#fff6d5"

MAX_HE_CLUSTERS = 40
MAX_PHOTONS = 260

FUSION_DISTANCE = 0.28        # distance threshold to consider particles "close"
FUSION_GROUP_SIZE = 3         # how many H join to make one He (visual)
FUSION_CHANCE = 0.16          # probability that a close group actually fuses

H_SPEED_RANGE = (0.01, 0.05)
PHOTON_SPEED_RANGE = (0.12, 0.28)

CORE_GLOW_STRENGTH = 0.85
PULSE_SPEED = 0.045


@dataclass
class HParticle:
    x: float
    y: float
    vx: float
    vy: float
    alive: bool = True

    def step(self):
        if not self.alive:
            return
        self.x += self.vx
        self.y += self.vy

        # Soft confinement: bounce back toward center
        r = math.sqrt(self.x * self.x + self.y * self.y) + 1e-9
        if r > 4.0:
            # nudge velocity inward
            self.vx += (-self.x / r) * 0.02
            self.vy += (-self.y / r) * 0.02

        # Gentle random jitter to feel like hot plasma
        self.vx += random.uniform(-0.002, 0.002)
        self.vy += random.uniform(-0.002, 0.002)

        # Slight clamp on speed
        s2 = self.vx * self.vx + self.vy * self.vy
        if s2 > 0.05 * 0.05:
            scale = 0.05 / math.sqrt(s2)
            self.vx *= scale
            self.vy *= scale


@dataclass
class HeCluster:
    x: float
    y: float
    radius: float = HE_RADIUS
    alpha: float = 1.0
    ttl: int = 600  # fade slowly

    def step(self):
        self.ttl -= 1
        # slow gentle pulsation and fade
        phase = (600 - self.ttl) * 0.08
        self.radius = HE_RADIUS * (1.0 + 0.16 * math.sin(phase))
        self.alpha = max(0.12, self.ttl / 600.0)

    @property
    def alive(self):
        return self.ttl > 0


@dataclass
class Photon:
    x: float
    y: float
    vx: float
    vy: float
    alpha: float = 1.0
    ttl: int = 40

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.ttl -= 1
        # fade out
        self.alpha = max(0.0, self.ttl / 40.0)

    @property
    def alive(self):
        return self.ttl > 0


def random_velocity():
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(*H_SPEED_RANGE)
    return math.cos(angle) * speed, math.sin(angle) * speed


def create_initial_h_particles() -> List[HParticle]:
    particles: List[HParticle] = []
    for _ in range(NUM_H_PARTICLES):
        r = random.uniform(0.0, 3.5)
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        vx, vy = random_velocity()
        particles.append(HParticle(x=x, y=y, vx=vx, vy=vy))
    return particles


def spawn_photons(x: float, y: float, count: int) -> List[Photon]:
    photons: List[Photon] = []
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*PHOTON_SPEED_RANGE)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        photons.append(Photon(x=x, y=y, vx=vx, vy=vy))
    return photons


def try_fusion(h_particles: List[HParticle], he_clusters: List[HeCluster], photons: List[Photon]):
    # Build a list of indices of still-alive H particles
    alive_indices = [i for i, p in enumerate(h_particles) if p.alive]
    random.shuffle(alive_indices)

    used = set()

    for i in alive_indices:
        if i in used:
            continue
        pi = h_particles[i]
        if not pi.alive:
            continue

        # Search neighbors close to this particle
        neighbors = [i]
        for j in alive_indices:
            if j == i or j in used:
                continue
            pj = h_particles[j]
            if not pj.alive:
                continue
            dx = pj.x - pi.x
            dy = pj.y - pi.y
            if dx * dx + dy * dy <= FUSION_DISTANCE * FUSION_DISTANCE:
                neighbors.append(j)
            if len(neighbors) >= FUSION_GROUP_SIZE:
                break

        if len(neighbors) >= FUSION_GROUP_SIZE and random.random() < FUSION_CHANCE:
            # Perform a stylized "fusion": mark H as gone, create one He, spawn photons
            xs = [h_particles[k].x for k in neighbors]
            ys = [h_particles[k].y for k in neighbors]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            # Mark used H as not alive
            for k in neighbors:
                h_particles[k].alive = False
                used.add(k)

            # Spawn helium-like cluster
            if len(he_clusters) < MAX_HE_CLUSTERS:
                he_clusters.append(HeCluster(x=cx, y=cy))

            # Spawn photons flying outwards
            new_photons = spawn_photons(cx, cy, count=random.randint(5, 14))
            photons.extend(new_photons)

    # Optionally respawn some new H over time to keep the scene alive
    while sum(1 for p in h_particles if p.alive) < NUM_H_PARTICLES:
        # Spawn near center with random direction
        r = random.uniform(0.0, 3.0)
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        vx, vy = random_velocity()
        h_particles.append(HParticle(x=x, y=y, vx=vx, vy=vy))


def build_scene():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#020308")

    # subtle label
    ax.text(
        0.0,
        Y_MIN + 0.4,
        "Stylized Hydrogen Fusion in a Stellar-like Core (non-physical visualization)",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#9fb4ff",
    )

    return fig, ax


def run_animation():
    fig, ax = build_scene()

    h_particles = create_initial_h_particles()
    he_clusters: List[HeCluster] = []
    photons: List[Photon] = []

    # scatter for hydrogen-like particles
    h_scatter = ax.scatter([], [], s=[], c=[], alpha=0.96, edgecolors="none", zorder=4)

    # helium clusters as patches
    he_patches: List[Circle] = []

    # photons as scatter
    photon_scatter = ax.scatter([], [], s=[], c=[], alpha=1.0, edgecolors="none", zorder=5)

    frame_counter = {"value": 0}

    def update(frame: int):
        frame_counter["value"] += 1
        t = frame_counter["value"]

        # Core background glow: radial gradient + breathing pulse
        # Compute once per frame as scalar background color mod.
        pulse = 0.9 + 0.12 * math.sin(PULSE_SPEED * t)
        # We don't redraw background per-pixel here (keeps things simple & fast).

        # Step H particles
        for p in h_particles:
            if p.alive:
                p.step()

        # Attempt stylized fusion events
        try_fusion(h_particles, he_clusters, photons)

        # Step He clusters
        for he in he_clusters:
            if he.alive:
                he.step()

        # Step photons
        for ph in photons:
            if ph.alive:
                ph.step()

        # Clean up dead photons and He clusters, limit counts
        photons[:] = [ph for ph in photons if ph.alive][:MAX_PHOTONS]
        he_clusters[:] = [he for he in he_clusters if he.alive][:MAX_HE_CLUSTERS]

        # Update hydrogen scatter
        h_coords = np.array([[p.x, p.y] for p in h_particles if p.alive])
        if len(h_coords) > 0:
            sizes = np.full(h_coords.shape[0], H_RADIUS * 600.0)
            colors = [H_COLOR] * h_coords.shape[0]
            h_scatter.set_offsets(h_coords)
            h_scatter.set_sizes(sizes)
            h_scatter.set_color(colors)
        else:
            h_scatter.set_offsets(np.empty((0, 2)))
            h_scatter.set_sizes([])

        # Update helium cluster patches
        for patch in he_patches:
            patch.remove()
        he_patches.clear()

        for he in he_clusters:
            c = Circle(
                (he.x, he.y),
                he.radius,
                facecolor=(*tuple(int(HE_COLOR[i:i+2], 16) / 255 for i in (1, 3, 5)), he.alpha),
                edgecolor="none",
                zorder=3,
            )
            he_patches.append(c)
            ax.add_patch(c)

        # Update photons scatter
        if photons:
            p_coords = np.array([[ph.x, ph.y] for ph in photons if ph.alive])
            if len(p_coords) > 0:
                p_sizes = np.full(p_coords.shape[0], 18.0)
                p_colors = [PHOTON_COLOR] * p_coords.shape[0]
                photon_scatter.set_offsets(p_coords)
                photon_scatter.set_sizes(p_sizes)
                photon_scatter.set_color(p_colors)
            else:
                photon_scatter.set_offsets(np.empty((0, 2)))
                photon_scatter.set_sizes([])
        else:
            photon_scatter.set_offsets(np.empty((0, 2)))
            photon_scatter.set_sizes([])

        # Slightly tint the background using the pulse (via axes facecolor)
        base = 0.02 * pulse
        ax.set_facecolor((base, base * 0.9, base * 1.4))

        artists = [h_scatter, photon_scatter] + he_patches
        return artists

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