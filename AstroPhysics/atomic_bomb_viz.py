"""
Stylized chain-reaction visualization (fictional / non-technical).

This is an artistic animation that visually represents a "cascade" where:
- Bright particles move around as incoming neutrons.
- Circular clusters represent atomic nuclei made of smaller nucleons.
- When struck, a nucleus flashes, its nucleons light up, and fragments plus new neutrons fly out.

This is NOT a real simulation of nuclear fission or weapons — it's purely for visualization.

Run:
    python atomic_bomb_viz.py
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

# Scene geometry
FIGSIZE = (10, 6)
X_MIN, X_MAX = -8.0, 8.0
Y_MIN, Y_MAX = -4.5, 4.5

# Visual/behavioral tuning (all purely artistic)
NUM_NUCLEI = 28                   # number of big stationary "nuclei"
NUCLEUS_RADIUS = 0.28
NUCLEUS_COLOR = "#ff8c6a"
FRAGMENT_COLOR = "#ffd86a"
NEUTRON_COLOR = "#9be7ff"

INITIAL_NEUTRONS = 6              # seed particles that float around
NEUTRON_SPEED = (0.06, 0.18)      # visual speed range (units/frame)
NEUTRON_SIZE = 28
FRAGMENT_SIZE = 48
SPLIT_FRAGMENTS = (3, 6)          # how many fragments spawn visually on a split
NEW_NEUTRONS_PER_SPLIT = (2, 5)   # how many new moving "neutrons" appear on split

SPLIT_GLOW_DURATION = 26          # frames of bright flash when a nucleus splits
MAX_ACTIVE_NEUTRONS = 220

# Safety note in program (visual only)
PROGRAM_NOTE = (
    "This is a stylized, non-physical animation. "
    "It does not simulate real nuclear behavior."
)


@dataclass
class MovingParticle:
    x: float
    y: float
    vx: float
    vy: float
    size: float = NEUTRON_SIZE
    color: str = NEUTRON_COLOR
    lifespan: int = 9999  # optional life in frames
    ttl: int = field(default=9999, init=False)

    def __post_init__(self):
        self.ttl = self.lifespan

    def step(self):
        self.x += self.vx
        self.y += self.vy
        # Bounce off walls softly
        if self.x < X_MIN or self.x > X_MAX:
            self.vx *= -1.0
            self.x = max(min(self.x, X_MAX), X_MIN)
        if self.y < Y_MIN or self.y > Y_MAX:
            self.vy *= -1.0
            self.y = max(min(self.y, Y_MAX), Y_MIN)
        self.ttl -= 1

    @property
    def alive(self):
        return self.ttl > 0


@dataclass
class Nucleus:
    x: float
    y: float
    radius: float = NUCLEUS_RADIUS
    split: bool = False
    split_timer: int = 0  # counts flash frames after splitting
    nucleon_offsets: List[tuple[float, float]] = field(default_factory=list)


@dataclass
class Fragment:
    x: float
    y: float
    vx: float
    vy: float
    size: float = FRAGMENT_SIZE
    color: str = FRAGMENT_COLOR
    ttl: int = 80

    def step(self):
        # Fragments slow down and fade
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.97
        self.vy *= 0.97
        self.ttl -= 1

    @property
    def alive(self):
        return self.ttl > 0


def make_nucleon_offsets(count: int, core_radius: float) -> List[tuple[float, float]]:
    """
    Create small offset positions for nucleons inside a nucleus.
    Purely visual: random cluster within the nuclear radius.
    """
    offsets: List[tuple[float, float]] = []
    for _ in range(count):
        ang = random.uniform(0, 2 * math.pi)
        # bias inward so cluster looks dense
        r = (random.random() ** 0.6) * (core_radius * 0.7)
        dx = r * math.cos(ang)
        dy = r * math.sin(ang)
        offsets.append((dx, dy))
    return offsets


def create_initial_nuclei() -> List[Nucleus]:
    nuclei: List[Nucleus] = []
    # Arrange nuclei in a loose grid/cluster across the field
    cols = 7
    rows = max(3, NUM_NUCLEI // cols)
    xs = np.linspace(X_MIN + 1.2, X_MAX - 1.2, cols)
    ys = np.linspace(Y_MIN + 0.8, Y_MAX - 0.8, rows)
    positions = []
    for i in range(rows):
        for j in range(cols):
            positions.append((xs[j] + random.uniform(-0.6, 0.6), ys[i] + random.uniform(-0.3, 0.3)))
    random.shuffle(positions)
    for i in range(min(NUM_NUCLEI, len(positions))):
        x, y = positions[i]
        # each nucleus gets a random number of visible nucleons
        n_nucleons = random.randint(8, 18)
        offsets = make_nucleon_offsets(n_nucleons, NUCLEUS_RADIUS)
        nuclei.append(Nucleus(x=x, y=y, nucleon_offsets=offsets))
    return nuclei


def spawn_neutron_from(x: float, y: float) -> MovingParticle:
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(*NEUTRON_SPEED)
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    return MovingParticle(x=x, y=y, vx=vx, vy=vy)


def run_animation():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#05060d")

    # Title + note
    title = ax.text(0.0, Y_MAX + 0.1, "Stylized Chain Reaction (fictional visualization)",
                    ha="center", va="bottom", color="#cfe7ff", fontsize=12)
    note = ax.text(0.0, Y_MIN - 0.6, PROGRAM_NOTE, ha="center", va="top", color="#9aa9c7", fontsize=8)

    nuclei = create_initial_nuclei()
    neutrons: List[MovingParticle] = [spawn_neutron_from(random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX)) for _ in range(INITIAL_NEUTRONS)]
    fragments: List[Fragment] = []
    split_count = 0

    # Visual artists: scatter for neutrons, patches for nuclei & fragments
    neutron_scatter = ax.scatter([], [], s=[], c=[], edgecolors="none", alpha=0.95, zorder=4)
    fragment_patches: List[Circle] = []
    nucleus_patches: List[Circle] = []
    for n in nuclei:
        c = Circle((n.x, n.y), n.radius, color=NUCLEUS_COLOR, zorder=3)
        nucleus_patches.append(c)
        ax.add_patch(c)

    # For each nucleus, create small nucleon dots inside (protons/neutrons)
    nucleus_nucleon_patches: List[List[Circle]] = []
    for nuc in nuclei:
        nucleon_p_list: List[Circle] = []
        for (dx, dy) in nuc.nucleon_offsets:
            # Randomly color some as "protons" and some as "neutrons"
            if random.random() < 0.5:
                col = "#ffb38a"  # warmer (proton-ish)
            else:
                col = "#9bbdff"  # cooler (neutron-ish)
            p = Circle((nuc.x + dx, nuc.y + dy), 0.07, color=col, alpha=0.55, zorder=4)
            nucleon_p_list.append(p)
            ax.add_patch(p)
        nucleus_nucleon_patches.append(nucleon_p_list)

    def try_hit(n: MovingParticle, nuc: Nucleus) -> bool:
        # simple distance check; purely visual "hit" probability added
        dx = n.x - nuc.x
        dy = n.y - nuc.y
        d2 = dx * dx + dy * dy
        hit = d2 <= (nuc.radius + 0.12) ** 2
        # add some randomness to hits so not every near pass splits
        if hit and random.random() < 0.36:
            return True
        return False

    frame_counter = {"value": 0}

    def update(_frame):
        frame_counter["value"] += 1

        # Move neutrons
        for p in neutrons:
            p.step()

        # Move fragments
        for f in fragments:
            f.step()

        # Check collisions: neutrons hitting non-split nuclei
        new_neutrons: List[MovingParticle] = []
        new_fragments: List[Fragment] = []
        for n_i, nuc in enumerate(nuclei):
            if nuc.split:
                # decrease split flash timer
                nuc.split_timer = max(0, nuc.split_timer - 1)
                # Darken patch while in split flash (handled below)
                continue
            # check all neutrons for hits
            for p in list(neutrons):
                if try_hit(p, nuc):
                    # Nucleus visually "splits" (artistic)
                    nuc.split = True
                    nuc.split_timer = SPLIT_GLOW_DURATION
                    nonlocal_frag_count = random.randint(*SPLIT_FRAGMENTS)
                    nonlocal_new_neutrons = random.randint(*NEW_NEUTRONS_PER_SPLIT)
                    # spawn fragments flying outward
                    for _ in range(nonlocal_frag_count):
                        ang = random.uniform(0, 2 * math.pi)
                        speed = random.uniform(0.8, 2.6)
                        vx = math.cos(ang) * speed * 0.06
                        vy = math.sin(ang) * speed * 0.06
                        new_fragments.append(Fragment(x=nuc.x, y=nuc.y, vx=vx, vy=vy))
                    # spawn new neutrons (visual)
                    for _ in range(nonlocal_new_neutrons):
                        new_neutrons.append(spawn_neutron_from(nuc.x, nuc.y))
                    # visual removal of the hitting neutron (it is consumed)
                    try:
                        neutrons.remove(p)
                    except ValueError:
                        pass
                    break  # only one neutron triggers the split at a time

        # Add newly spawned items
        fragments.extend(new_fragments)
        neutrons.extend(new_neutrons)

        # occasionally spawn background neutrons to keep it lively (but cap total)
        if frame_counter["value"] % 18 == 0 and len(neutrons) < MAX_ACTIVE_NEUTRONS:
            neutrons.append(spawn_neutron_from(random.uniform(X_MIN, X_MAX), random.uniform(Y_MIN, Y_MAX)))

        # Cull dead fragments
        fragments[:] = [f for f in fragments if f.alive]

        # Update nucleus patch + internal nucleon visuals
        for idx, (nuc, patch) in enumerate(zip(nuclei, nucleus_patches)):
            nucleon_p_list = nucleus_nucleon_patches[idx]
            if nuc.split:
                # flash glow when split
                glow = 0.6 + 0.4 * math.sin(0.45 * nuc.split_timer + 0.3)
                patch.set_facecolor((1.0, 0.5, 0.2, glow))
                patch.set_radius(nuc.radius * (1.0 + 0.16 * (SPLIT_GLOW_DURATION - nuc.split_timer)))

                # brighten and slightly "push out" nucleons to show internal disruption
                for j, p in enumerate(nucleon_p_list):
                    (dx, dy) = nuc.nucleon_offsets[j]
                    # scale outward based on remaining flash time
                    scale = 1.0 + 0.9 * (SPLIT_GLOW_DURATION - nuc.split_timer) / max(SPLIT_GLOW_DURATION, 1)
                    p.center = (nuc.x + dx * scale, nuc.y + dy * scale)
                    p.set_alpha(min(1.0, 0.4 + 0.7 * (SPLIT_GLOW_DURATION - nuc.split_timer) / max(SPLIT_GLOW_DURATION, 1)))
            else:
                # stable nucleus: default size/color and nucleons clustered
                patch.set_facecolor(NUCLEUS_COLOR)
                patch.set_radius(nuc.radius)
                for j, p in enumerate(nucleon_p_list):
                    (dx, dy) = nuc.nucleon_offsets[j]
                    p.center = (nuc.x + dx, nuc.y + dy)
                    p.set_alpha(0.55)

        # Update fragments drawing
        # remove old patched fragments
        for p in list(fragment_patches):
            p.remove()
        fragment_patches.clear()
        for f in fragments:
            c = Circle((f.x, f.y), 0.06 + 0.0008 * f.size, color=f.color, alpha=max(0.08, f.ttl / 100.0), zorder=5)
            fragment_patches.append(c)
            ax.add_patch(c)

        # Update neutron scatter
        if neutrons:
            coords = np.array([[p.x, p.y] for p in neutrons])
            sizes = np.array([p.size for p in neutrons])
            colors = [p.color for p in neutrons]
            neutron_scatter.set_offsets(coords)
            neutron_scatter.set_sizes(sizes)
            neutron_scatter.set_color(colors)
        else:
            neutron_scatter.set_offsets(np.empty((0, 2)))
            neutron_scatter.set_sizes([])

        # Return artists that changed
        artists = [neutron_scatter] + fragment_patches + nucleus_patches
        for plist in nucleus_nucleon_patches:
            artists.extend(plist)
        return artists

    anim = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    run_animation()