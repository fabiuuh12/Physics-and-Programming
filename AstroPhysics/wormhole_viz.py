"""
Wormhole Visualization (Not physically accurate, but vibes)

Run this file to see an intuitive / sci-fi style animation:
- Two "universes" on the left and right.
- A bright throat in the center.
- Particles spiral in from each side, get squeezed through the throat,
  and reappear on the opposite side — like traveling through a shortcut.

This is just to build intuition:
  Normal space = particles travel the long way around.
  Wormhole idea = a tunnel connecting distant regions so paths are shorter.

Requirements:
  pip install matplotlib numpy
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# -----------------------------
# Config
# -----------------------------
N_PARTICLES = 500
INTERVAL_MS = 40
R_MIN = 0.03          # radius at which particle "enters" throat
R_MAX = 1.25          # starting radius
SPIRAL_SPEED = 0.18   # angular speed
INFALL_SPEED = 0.012  # radial speed
THROAT_PULSE_SPEED = 0.09
STAR_COUNT = 480

# -----------------------------
# Particle model
# -----------------------------
class Particles:
    def __init__(self, n: int):
        # side: -1 = left universe, +1 = right universe
        self.side = np.random.choice([-1, 1], size=n)

        # radius and angle in polar coords around each side's center
        self.r = np.random.uniform(0.4, R_MAX, size=n)
        self.theta = np.random.uniform(0, 2 * np.pi, size=n)

        # each side has its own center
        self.left_center = np.array([-1.0, 0.0])
        self.right_center = np.array([1.0, 0.0])

        self.color_phase = np.random.rand(n)

    def _center_for_side(self, side_vals: np.ndarray) -> np.ndarray:
        """
        Return centers for each particle depending on its side.
        """
        centers = np.zeros((side_vals.size, 2))
        left_mask = side_vals == -1
        right_mask = ~left_mask
        centers[left_mask] = self.left_center
        centers[right_mask] = self.right_center
        return centers

    def step(self):
        """
        Update particle positions:
        - Spiral inward (r decreases, theta rotates).
        - When close to throat (r < R_MIN), send particle through to opposite side,
          reset r ~ R_MAX, keep some memory of angle for smooth look.
        """
        # Spiral in
        self.theta += SPIRAL_SPEED
        self.r -= INFALL_SPEED
        self.color_phase += 0.02

        # Check which particles hit the throat
        mask = self.r < R_MIN
        if np.any(mask):
            # Flip side (they exit on the opposite universe)
            self.side[mask] *= -1

            # Reset radius outward
            self.r[mask] = np.random.uniform(0.8, R_MAX, size=mask.sum())

            # Small random tweak in phase so it doesn't look too rigid
            self.theta[mask] += np.random.uniform(-0.6, 0.6, size=mask.sum())
            self.color_phase[mask] += np.random.uniform(0.0, 1.0, size=mask.sum())

        # Compute Cartesian positions relative to their side centers
        centers = self._center_for_side(self.side)
        x = centers[:, 0] + self.r * np.cos(self.theta)
        y = centers[:, 1] + self.r * np.sin(self.theta)

        # Color hint: hotter near the throat, cooler farther out, tinted by universe side
        radial_norm = np.clip((self.r - R_MIN) / (R_MAX - R_MIN), 0.0, 1.0)
        swirl_color = cm.plasma(0.2 + 0.75 * (1.0 - radial_norm))
        left_mask = self.side == -1
        swirl_color[left_mask, 2] = np.clip(swirl_color[left_mask, 2] + 0.18, 0, 1)
        swirl_color[~left_mask, 0] = np.clip(swirl_color[~left_mask, 0] + 0.18, 0, 1)
        swirl_color[:, 3] = 0.85

        # Pulse their brightness subtly
        flicker = 0.15 * np.sin(self.color_phase) + 0.85
        swirl_color[:, :3] *= flicker[:, None]

        return x, y, swirl_color


# -----------------------------
# Background geometry
# -----------------------------
def draw_background(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_facecolor("black")

    # scatter some stars behind everything
    star_x = np.random.uniform(-2.3, 2.3, STAR_COUNT)
    star_y = np.random.uniform(-1.6, 1.6, STAR_COUNT)
    star_sizes = np.random.uniform(3, 18, STAR_COUNT)
    star_alpha = np.random.uniform(0.15, 0.7, STAR_COUNT)
    ax.scatter(
        star_x,
        star_y,
        s=star_sizes,
        c="#9fd1ff",
        alpha=star_alpha,
        linewidths=0,
        zorder=0,
    )

    # "Space-time grid" warped toward the throat
    xs = np.linspace(-2.2, 2.2, 40)
    ys = np.linspace(-1.5, 1.5, 24)

    for y in ys:
        warped = []
        for x in xs:
            # radial distance from center throat
            r = np.sqrt(x**2 + y**2) + 1e-6
            # pull lines slightly toward center to mimic curvature
            factor = 1.0 / (1.0 + 0.4 / (r**2))
            warped.append((x * factor, y * factor))
        warped = np.array(warped)
        ax.plot(warped[:, 0], warped[:, 1], alpha=0.08)

    for x in xs:
        warped = []
        for y in ys:
            r = np.sqrt(x**2 + y**2) + 1e-6
            factor = 1.0 / (1.0 + 0.4 / (r**2))
            warped.append((x * factor, y * factor))
        warped = np.array(warped)
        ax.plot(warped[:, 0], warped[:, 1], alpha=0.08)

    # Left and right "universes" labels
    ax.text(-1.6, 1.25, "Universe A", color="white", fontsize=10, ha="center")
    ax.text(1.6, 1.25, "Universe B", color="white", fontsize=10, ha="center")

    # Wormhole throat (glowing rings at center)
    throat_radii = [0.12, 0.23, 0.34]
    for tr in throat_radii:
        c = plt.Circle((0, 0), tr, fill=False, alpha=0.6)
        ax.add_patch(c)

    # Central glow hint
    ax.scatter([0], [0], s=120, alpha=0.9)


# -----------------------------
# Main visualization
# -----------------------------
def main():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Wormhole Visualization (Shortcut Through Curved Spacetime)", color="white")

    draw_background(ax)

    # Initialize particles
    particles = Particles(N_PARTICLES)
    x0, y0, c0 = particles.step()
    scatter = ax.scatter(x0, y0, s=10, alpha=0.9, c=c0)

    # Glowing throat pulse
    throat_glow = plt.Circle((0, 0), 0.18, color="#d9f6ff", alpha=0.55)
    pulse_ring = plt.Circle((0, 0), 0.35, fill=False, linewidth=1.5, color="#8effd2", alpha=0.35)
    ax.add_patch(throat_glow)
    ax.add_patch(pulse_ring)

    # swirling filaments around the throat
    swirl_lines = []
    swirl_phases = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    for phase in swirl_phases:
        (line,) = ax.plot([], [], color="#9ef5ff", linewidth=1.1, alpha=0.35)
        swirl_lines.append((line, phase))
    swirl_artists = [line for line, _ in swirl_lines]
    theta_swirl = np.linspace(-np.pi, np.pi, 480)

    # Text explanation
    explain_text = ax.text(
        0.02,
        0.06,
        "Particles spiral into the central throat\n"
        "and emerge on the opposite side.\n"
        "Think of it as a shortcut linking two distant regions.",
        color="white",
        fontsize=8,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
    )

    def update(frame: int):
        x, y, colors = particles.step()
        scatter.set_offsets(np.column_stack((x, y)))
        scatter.set_color(colors)

        # make the wormhole pulse and swirl
        glow_radius = 0.13 + 0.06 * (0.5 + 0.5 * np.sin(frame * THROAT_PULSE_SPEED))
        throat_glow.set_radius(glow_radius)
        throat_glow.set_alpha(0.4 + 0.2 * (0.5 + 0.5 * np.cos(frame * THROAT_PULSE_SPEED)))
        pulse_radius = 0.28 + 0.16 * (0.5 + 0.5 * np.sin(frame * 0.04))
        pulse_ring.set_radius(pulse_radius)
        pulse_ring.set_alpha(0.2 + 0.25 * (0.5 + 0.5 * np.cos(frame * 0.05)))

        swirl_offset = 0.3 * np.sin(frame * 0.02)
        for line, phase in swirl_lines:
            radius = 0.22 + 0.08 * np.sin(3 * theta_swirl + phase + swirl_offset)
            twist = 0.08 * np.sin(theta_swirl * 2 + phase * 0.3 + 0.04 * frame)
            xs = (radius + twist) * np.cos(theta_swirl)
            ys = (radius - twist) * np.sin(theta_swirl)
            line.set_data(xs, ys)
            line.set_alpha(0.15 + 0.26 * (0.5 + 0.5 * np.sin(frame * 0.04 + phase)))

        return scatter, explain_text, throat_glow, pulse_ring, *swirl_artists

    anim = FuncAnimation(
        fig,
        update,
        frames=800,
        interval=INTERVAL_MS,
        blit=False,
        repeat=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


if __name__ == "__main__":
    main()
