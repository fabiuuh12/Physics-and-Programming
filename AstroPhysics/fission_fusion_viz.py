"""
fission_fusion_viz.py — Stylized visualization of atomic fission & fusion.

This file renders a safe, artistic animation that contrasts two fictional scenes:
- Left half: glowing heavy nuclei occasionally split when playful "neutrons" bump into them,
  releasing bright fragments and more drifting particles.
- Right half: smaller atoms wander a plasma field and sometimes merge into luminous clusters
  with gentle energy waves.

It is not a scientific simulation, carries no technical detail about reactors or weapons,
and is intended purely as an educational art piece.

Run:
    python fission_fusion_viz.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

plt.style.use("dark_background")

# Canvas layout
FIGSIZE = (12, 6)
X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = -5.0, 5.0
FISSION_X_MIN, FISSION_X_MAX = -9.0, -1.0
FUSION_X_MIN, FUSION_X_MAX = 1.0, 9.0

# Visual tuning (pure aesthetics)
INITIAL_NEUTRONS = 10
MAX_NEUTRONS = 160
NEUTRON_SPEED_RANGE = (0.04, 0.1)
SPLIT_FLASH = 42
SPLIT_COOLDOWN = 90
FRAGMENT_SPEED_RANGE = (0.05, 0.18)

FUSION_ATOMS = 14
FUSION_DISTANCE = 0.65
FUSION_CHANCE = 0.25
SPARKS_PER_FUSION = (6, 12)

# Palette
BACKGROUND_COLOR = "#05060f"
FISSION_LABEL_COLOR = "#f6bf92"
FUSION_LABEL_COLOR = "#a8e4ff"
NUCLEUS_COLOR = "#ff9d6c"
NUCLEUS_FLASH_COLOR = "#ffe7c2"
NEUTRON_COLOR = "#9be8ff"
FRAGMENT_COLOR = "#ffd678"
SPARK_COLOR = "#7dfada"
FUSION_ATOM_COLOR = "#90d9ff"
FUSION_PRODUCT_COLOR = "#ffeec5"
FUSION_FLASH_COLOR = "#fff7e5"


def blend_colors(c1: str, c2: str, t: float) -> Tuple[float, float, float, float]:
    """Linear blend between two hex colors."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1, a1 = mcolors.to_rgba(c1)
    r2, g2, b2, a2 = mcolors.to_rgba(c2)
    return (
        r1 + (r2 - r1) * t,
        g1 + (g2 - g1) * t,
        b1 + (b2 - b1) * t,
        a1 + (a2 - a1) * t,
    )


@dataclass
class HeavyNucleus:
    x: float
    y: float
    base_radius: float = 0.65
    phase: float = field(default_factory=lambda: random.uniform(0, 2 * math.pi))
    flash: int = 0
    cooldown: int = 0

    def step(self):
        self.phase += 0.015
        if self.flash > 0:
            self.flash -= 1
        if self.cooldown > 0:
            self.cooldown -= 1

    @property
    def draw_radius(self) -> float:
        return self.base_radius * (1.0 + 0.07 * math.sin(self.phase))

    def glow_color(self):
        if self.flash > 0:
            t = self.flash / SPLIT_FLASH
            return blend_colors(NUCLEUS_COLOR, NUCLEUS_FLASH_COLOR, t)
        flicker = 0.3 + 0.2 * math.sin(self.phase * 3.0)
        return blend_colors(NUCLEUS_COLOR, NUCLEUS_FLASH_COLOR, flicker)

    def trigger(self):
        self.flash = SPLIT_FLASH
        self.cooldown = SPLIT_COOLDOWN


@dataclass
class Neutron:
    x: float
    y: float
    vx: float
    vy: float
    ttl: int = 520
    max_ttl: int = 520
    size: float = 38.0
    color: str = NEUTRON_COLOR

    def step(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < FISSION_X_MIN:
            self.vx = abs(self.vx)
        if self.x > FISSION_X_MAX:
            self.vx = -abs(self.vx)
        if self.y < Y_MIN + 0.5 or self.y > Y_MAX - 0.5:
            self.vy *= -1
        self.ttl -= 1

    @property
    def alive(self):
        return self.ttl > 0 and X_MIN - 2.0 <= self.x <= 2.0

    @property
    def alpha(self):
        return max(0.15, self.ttl / self.max_ttl)


@dataclass
class Fragment:
    x: float
    y: float
    vx: float
    vy: float
    ttl: int = 80
    max_ttl: int = 80
    size: float = 120.0
    color: str = FRAGMENT_COLOR

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.97
        self.vy *= 0.97
        self.ttl -= 1

    @property
    def alive(self):
        return self.ttl > 0 and X_MIN - 1.0 <= self.x <= 2.5

    @property
    def alpha(self):
        return max(0.0, self.ttl / self.max_ttl)


@dataclass
class FusionAtom:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    color: str = FUSION_ATOM_COLOR

    def step(self):
        # drift with small jitter and bounce in fusion arena
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.995
        self.vy *= 0.995
        self.vx += random.uniform(-0.002, 0.002)
        self.vy += random.uniform(-0.002, 0.002)
        if self.x < FUSION_X_MIN + 0.3 or self.x > FUSION_X_MAX - 0.3:
            self.vx *= -1
            self.x = min(max(self.x, FUSION_X_MIN + 0.3), FUSION_X_MAX - 0.3)
        if self.y < Y_MIN + 0.4 or self.y > Y_MAX - 0.4:
            self.vy *= -1
            self.y = min(max(self.y, Y_MIN + 0.4), Y_MAX - 0.4)

    @property
    def size(self):
        return (self.radius * 55) ** 2

    @property
    def alpha(self):
        return 0.9


@dataclass
class FusionProduct:
    x: float
    y: float
    base_radius: float = 0.45
    ttl: int = 160
    max_ttl: int = 160
    phase: float = field(default_factory=lambda: random.uniform(0, 2 * math.pi))
    patch: Circle | None = field(default=None, repr=False)

    def attach(self, ax):
        if self.patch is None:
            self.patch = Circle(
                (self.x, self.y),
                self.base_radius,
                color=FUSION_PRODUCT_COLOR,
                alpha=0.9,
                zorder=3,
                ec="#3b2f0f",
                lw=0.4,
            )
            ax.add_patch(self.patch)

    def step(self):
        self.phase += 0.04
        self.ttl -= 1
        radius = self.base_radius * (1.1 + 0.18 * math.sin(self.phase))
        alpha = max(0.0, self.ttl / self.max_ttl)
        if self.patch is not None:
            self.patch.center = (self.x, self.y)
            self.patch.set_radius(radius)
            self.patch.set_alpha(alpha)

    @property
    def alive(self):
        return self.ttl > 0


@dataclass
class FusionFlash:
    x: float
    y: float
    ttl: int = 34
    max_ttl: int = 34
    base_size: float = 80.0
    color: str = FUSION_FLASH_COLOR

    def step(self):
        self.ttl -= 1

    @property
    def alive(self):
        return self.ttl > 0

    @property
    def size(self):
        phase = 1.0 - (self.ttl / self.max_ttl)
        return self.base_size * (1.0 + phase * 2.8)

    @property
    def alpha(self):
        return max(0.0, self.ttl / self.max_ttl)


@dataclass
class Spark:
    x: float
    y: float
    vx: float
    vy: float
    ttl: int = 30
    max_ttl: int = 30
    size: float = 40.0
    color: str = SPARK_COLOR

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.94
        self.vy *= 0.94
        self.ttl -= 1

    @property
    def alive(self):
        return self.ttl > 0 and FUSION_X_MIN - 0.5 <= self.x <= X_MAX + 1.0

    @property
    def alpha(self):
        return max(0.0, self.ttl / self.max_ttl)


def make_neutron(x: float | None = None, y: float | None = None, directed: bool = True) -> Neutron:
    if x is None:
        x = random.uniform(FISSION_X_MIN - 2.0, FISSION_X_MIN - 0.6)
    if y is None:
        y = random.uniform(Y_MIN + 0.8, Y_MAX - 0.8)
    speed = random.uniform(*NEUTRON_SPEED_RANGE)
    if directed:
        angle = random.uniform(-0.35, 0.35)
    else:
        angle = random.uniform(0, 2 * math.pi)
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    ttl = random.randint(420, 620)
    return Neutron(x=x, y=y, vx=vx, vy=vy, ttl=ttl, max_ttl=ttl)


def spawn_fragment(x: float, y: float) -> Fragment:
    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(*FRAGMENT_SPEED_RANGE)
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    ttl = random.randint(60, 110)
    size = random.uniform(80, 150)
    return Fragment(x=x, y=y, vx=vx, vy=vy, ttl=ttl, max_ttl=ttl, size=size)


def reset_atom(atom: FusionAtom):
    atom.x = random.uniform(FUSION_X_MIN + 0.6, FUSION_X_MAX - 0.6)
    atom.y = random.uniform(Y_MIN + 0.7, Y_MAX - 0.7)
    speed = random.uniform(0.02, 0.06)
    angle = random.uniform(0, 2 * math.pi)
    atom.vx = math.cos(angle) * speed
    atom.vy = math.sin(angle) * speed
    atom.radius = random.uniform(0.15, 0.22)


def create_fusion_atoms() -> List[FusionAtom]:
    atoms: List[FusionAtom] = []
    for _ in range(FUSION_ATOMS):
        speed = random.uniform(0.02, 0.06)
        angle = random.uniform(0, 2 * math.pi)
        atom = FusionAtom(
            x=random.uniform(FUSION_X_MIN + 0.6, FUSION_X_MAX - 0.6),
            y=random.uniform(Y_MIN + 0.8, Y_MAX - 0.8),
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed,
            radius=random.uniform(0.15, 0.25),
        )
        atoms.append(atom)
    return atoms


def create_nuclei() -> List[HeavyNucleus]:
    nuclei: List[HeavyNucleus] = []
    rows = 2
    cols = 3
    xs = np.linspace(FISSION_X_MIN + 1.2, FISSION_X_MAX - 0.6, cols)
    ys = np.linspace(Y_MIN + 1.2, Y_MAX - 1.2, rows)
    jitter = 0.5
    for y in ys:
        for x in xs:
            nuclei.append(
                HeavyNucleus(
                    x=x + random.uniform(-jitter, jitter),
                    y=y + random.uniform(-jitter, jitter),
                    base_radius=random.uniform(0.55, 0.75),
                )
            )
    return nuclei


def update_scatter(scatter, items: Sequence):
    if not items:
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_sizes([])
        scatter.set_facecolors([])
        return
    offsets = np.array([[p.x, p.y] for p in items])
    sizes = [getattr(p, "size", 40.0) for p in items]
    colors = []
    for p in items:
        alpha = getattr(p, "alpha", 1.0)
        colors.append(mcolors.to_rgba(getattr(p, "color", "#ffffff"), alpha=alpha))
    scatter.set_offsets(offsets)
    scatter.set_sizes(sizes)
    scatter.set_facecolors(colors)


def run_animation():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.text(0.0, Y_MAX + 0.25, "Stylized Fission & Fusion Dance", ha="center", va="bottom", color="#f8fbff", fontsize=13)
    ax.text(
        0.0,
        Y_MIN - 0.7,
        "Playful art-only view: glowing nuclei split on the left, light atoms fuse on the right.",
        ha="center",
        va="top",
        color="#a2b6cc",
        fontsize=9,
    )
    ax.text((FISSION_X_MIN + FISSION_X_MAX) * 0.5, Y_MAX - 0.35, "Fission field", color=FISSION_LABEL_COLOR, ha="center", fontsize=11)
    ax.text((FUSION_X_MIN + FUSION_X_MAX) * 0.5, Y_MAX - 0.35, "Fusion field", color=FUSION_LABEL_COLOR, ha="center", fontsize=11)
    ax.plot([0, 0], [Y_MIN, Y_MAX], color="#1a1f2e", lw=1.2, alpha=0.6)

    nuclei = create_nuclei()
    neutrons: List[Neutron] = [make_neutron() for _ in range(INITIAL_NEUTRONS)]
    fragments: List[Fragment] = []
    fusion_atoms = create_fusion_atoms()
    fusion_products: List[FusionProduct] = []
    fusion_flashes: List[FusionFlash] = []
    sparks: List[Spark] = []

    nucleus_patches: List[Circle] = []
    for n in nuclei:
        patch = Circle((n.x, n.y), n.draw_radius, color=NUCLEUS_COLOR, alpha=0.85, ec="#4a1f12", lw=0.6, zorder=2)
        nucleus_patches.append(patch)
        ax.add_patch(patch)

    neutron_scatter = ax.scatter([], [], s=[], c=[], edgecolors="none", zorder=4)
    fragment_scatter = ax.scatter([], [], s=[], c=[], edgecolors="none", zorder=4)
    spark_scatter = ax.scatter([], [], s=[], c=[], edgecolors="none", zorder=4)
    fusion_atom_scatter = ax.scatter([], [], s=[], c=[], edgecolors="#142647", linewidths=0.4, zorder=4)
    fusion_flash_scatter = ax.scatter([], [], s=[], c=[], edgecolors="none", zorder=3)

    def collide_nuclei():
        for n in neutrons:
            if not n.alive:
                continue
            for nucleus in nuclei:
                if nucleus.cooldown > 0:
                    continue
                dx = n.x - nucleus.x
                dy = n.y - nucleus.y
                if dx * dx + dy * dy <= (nucleus.draw_radius + 0.15) ** 2:
                    nucleus.trigger()
                    n.ttl = 0
                    fragments.extend(spawn_fragment(nucleus.x, nucleus.y) for _ in range(random.randint(3, 6)))
                    extra = random.randint(1, 3)
                    for _ in range(extra):
                        neutrons.append(make_neutron(nucleus.x, nucleus.y, directed=False))
                    break

    def attempt_fusion():
        if len(fusion_atoms) < 2:
            return
        indices = list(range(len(fusion_atoms)))
        random.shuffle(indices)
        for i_idx in range(len(indices)):
            a = fusion_atoms[indices[i_idx]]
            for j_idx in range(i_idx + 1, len(indices)):
                b = fusion_atoms[indices[j_idx]]
                dx = b.x - a.x
                dy = b.y - a.y
                if dx * dx + dy * dy <= FUSION_DISTANCE * FUSION_DISTANCE:
                    if random.random() < FUSION_CHANCE:
                        cx = (a.x + b.x) * 0.5
                        cy = (a.y + b.y) * 0.5
                        flash = FusionFlash(x=cx, y=cy)
                        fusion_flashes.append(flash)
                        product = FusionProduct(x=cx, y=cy, base_radius=random.uniform(0.4, 0.55))
                        product.attach(ax)
                        fusion_products.append(product)
                        spark_count = random.randint(*SPARKS_PER_FUSION)
                        for _ in range(spark_count):
                            angle = random.uniform(0, 2 * math.pi)
                            speed = random.uniform(0.05, 0.16)
                            sparks.append(
                                Spark(
                                    x=cx,
                                    y=cy,
                                    vx=math.cos(angle) * speed,
                                    vy=math.sin(angle) * speed,
                                    ttl=random.randint(25, 40),
                                    max_ttl=random.randint(25, 40),
                                )
                            )
                        reset_atom(a)
                        reset_atom(b)
                    break

    def update(_frame):
        for nucleus, patch in zip(nuclei, nucleus_patches):
            nucleus.step()
            patch.center = (nucleus.x, nucleus.y)
            patch.set_radius(nucleus.draw_radius)
            patch.set_facecolor(nucleus.glow_color())

        for n in neutrons:
            n.step()
        collide_nuclei()
        neutrons[:] = [n for n in neutrons if n.alive]
        if len(neutrons) < MAX_NEUTRONS and random.random() < 0.2:
            neutrons.append(make_neutron())

        for frag in fragments:
            frag.step()
        fragments[:] = [f for f in fragments if f.alive]

        for atom in fusion_atoms:
            atom.step()
        attempt_fusion()

        for prod in fusion_products:
            prod.step()
        for prod in [p for p in fusion_products if not p.alive]:
            if prod.patch is not None:
                prod.patch.remove()
        fusion_products[:] = [p for p in fusion_products if p.alive]

        for flash in fusion_flashes:
            flash.step()
        fusion_flashes[:] = [f for f in fusion_flashes if f.alive]

        for sp in sparks:
            sp.step()
        sparks[:] = [s for s in sparks if s.alive]

        update_scatter(neutron_scatter, neutrons)
        update_scatter(fragment_scatter, fragments)
        update_scatter(spark_scatter, sparks)
        update_scatter(fusion_atom_scatter, fusion_atoms)
        update_scatter(fusion_flash_scatter, fusion_flashes)
        return []

    anim = FuncAnimation(fig, update, interval=30)
    plt.show()
    return anim


if __name__ == "__main__":
    run_animation()
