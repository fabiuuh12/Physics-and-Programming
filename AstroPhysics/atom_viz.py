"""
Atom Visualization (Intuition, not exact quantum mechanics)

Run this file to see:
- A compact, glowing nucleus made of protons/neutrons.
- Several electrons moving in fuzzy, changing orbits around it.
- A sense of:
    - Nucleus = tiny, dense core.
    - Electrons = delocalized cloud / orbitals rather than perfect circles.

This is a visual aid, not a physically accurate orbital solver.
Requirements:
    pip install matplotlib numpy
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Config
# -----------------------------
N_NUCLEONS = 40           # dots in the nucleus (protons+neutrons)
N_ELECTRONS = 12          # total electrons
SHELL_RADII = [0.6, 1.0, 1.4]  # approximate shells
INTERVAL_MS = 30

NUCLEUS_RADIUS = 0.18

np.random.seed(2)

# -----------------------------
# Build nucleus
# -----------------------------
def make_nucleus(n: int):
    """
    Random cluster of points in a small sphere for the nucleus.
    """
    r = NUCLEUS_RADIUS * np.random.rand(n) ** (1 / 3)
    theta = np.random.rand(n) * 2 * np.pi
    # 2D version: spread in x,y within radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

# -----------------------------
# Electrons model
# -----------------------------
class ElectronCloud:
    """
    Represent electrons as points that:
    - live near one of a few shell radii
    - move with their own angular speed
    - jitter radially a bit to look like a probability cloud
    """

    def __init__(self, n_electrons: int, shell_radii: list[float]):
        self.n = n_electrons
        self.shell_radii = np.asarray(shell_radii)

        # Assign electrons to shells
        shell_indices = np.random.randint(0, len(shell_radii), size=n_electrons)
        self.base_r = self.shell_radii[shell_indices]

        # Initial angles and speeds
        self.theta = np.random.rand(n_electrons) * 2 * np.pi
        self.omega = (np.random.rand(n_electrons) * 0.8 + 0.2) * np.random.choice(
            [-1, 1], size=n_electrons
        )

        # Small radial jitter amplitudes
        self.jitter_amp = 0.06 + 0.05 * np.random.rand(n_electrons)
        self.jitter_phase = np.random.rand(n_electrons) * 2 * np.pi
        self.jitter_speed = 0.04 + 0.04 * np.random.rand(n_electrons)

    def step(self, t: float):
        # Update angles
        self.theta += self.omega * 0.03

        # Update jitter phase
        self.jitter_phase += self.jitter_speed

        # Radius with jitter
        r = self.base_r + self.jitter_amp * np.sin(self.jitter_phase)

        # Convert to (x, y)
        x = r * np.cos(self.theta)
        y = r * np.sin(self.theta)

        return x, y

# -----------------------------
# Main visualization
# -----------------------------
def main():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")

    fig.suptitle("Atom Visualization (Nucleus + Electron Cloud)", color="white", fontsize=12)

    # Nucleus points
    nuc_x, nuc_y = make_nucleus(N_NUCLEONS)
    nucleus_scatter = ax.scatter(
        nuc_x, nuc_y,
        s=18,
        alpha=0.9,
    )

    # Soft halo around the nucleus
    nucleus_halo = plt.Circle(
        (0, 0),
        NUCLEUS_RADIUS * 2.2,
        fill=False,
        alpha=0.4,
    )
    ax.add_patch(nucleus_halo)

    # Shell guideline circles (faint)
    for r in SHELL_RADII:
        c = plt.Circle((0, 0), r, fill=False, alpha=0.12)
        ax.add_patch(c)

    # Radial "field lines" to make the electric field feel visible
    n_field_lines = 28
    for angle in np.linspace(0, 2 * np.pi, n_field_lines, endpoint=False):
        r_start = NUCLEUS_RADIUS * 2.2
        r_end = 1.9
        x0 = r_start * np.cos(angle)
        y0 = r_start * np.sin(angle)
        x1 = r_end * np.cos(angle)
        y1 = r_end * np.sin(angle)
        ax.plot(
            [x0, x1],
            [y0, y1],
            alpha=0.10,
            linewidth=0.8,
        )

    # Electric field boundary circles (inner and outer fields)
    field_radii = [0.9, 1.6]
    for fr in field_radii:
        circle = plt.Circle((0, 0), fr, fill=False, color="cyan", alpha=0.15, linewidth=1.5)
        ax.add_patch(circle)

    electrons = ElectronCloud(N_ELECTRONS, SHELL_RADII)
    e_x0, e_y0 = electrons.step(0.0)
    electron_scatter = ax.scatter(
        e_x0,
        e_y0,
        s=14,
        alpha=0.9,
    )

    # Text explanation
    text = ax.text(
        0.02,
        0.04,
        "Center: tiny, dense nucleus\n"
        "Around it: electrons as a moving cloud,\n"
        "not fixed little planets.",
        color="white",
        fontsize=8,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )

    def update(frame: int):
        t = frame * INTERVAL_MS / 1000.0
        ex, ey = electrons.step(t)
        electron_scatter.set_offsets(np.column_stack((ex, ey)))
        return electron_scatter, nucleus_scatter, text

    anim = FuncAnimation(
        fig,
        update,
        frames=800,
        interval=INTERVAL_MS,
        blit=False,
        repeat=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

if __name__ == "__main__":
    main()
