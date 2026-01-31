"""
Entropy visualization: particles in a box going from ordered to disordered.

Run this file to see:
- A box with particles starting in a low-entropy state (clustered on the left).
- Over time, particles move randomly and spread out (higher entropy).
- A live plot of the Shannon entropy value as the system evolves.

This is NOT a perfect physics simulation.
It's an intuition pump to "feel" what increasing entropy / disorder looks like.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -----------------------------
# Config
# -----------------------------
N_PARTICLES = 400          # number of particles
BOX_SIZE = 1.0             # box from (0,0) to (1,1)
STEP_SIZE = 0.03           # random step size per frame
GRID_SIZE = 20             # grid resolution for entropy calc
INTERVAL_MS = 40           # animation speed (ms between frames)

# Seed for reproducibility (set to None for full randomness)
RNG_SEED = 42


# -----------------------------
# Helpers
# -----------------------------
def init_particles(n: int) -> np.ndarray:
    """
    Start particles in a low-entropy configuration:
    - All particles packed in the left half of the box.
    """
    x = np.random.uniform(0.0, 0.25, size=n)  # tightly packed left side
    y = np.random.uniform(0.3, 0.7, size=n)   # narrow vertical band
    return np.column_stack((x, y))


def random_step(positions: np.ndarray, step_size: float) -> np.ndarray:
    """
    Apply a small random step (2D random walk) to each particle.
    Reflect at the walls so particles stay inside the box.
    """
    steps = np.random.normal(loc=0.0, scale=step_size, size=positions.shape)
    new_pos = positions + steps

    # Hard wall reflection
    new_pos = np.clip(new_pos, 0.0, BOX_SIZE)
    return new_pos


def shannon_entropy(positions: np.ndarray, grid_size: int) -> float:
    """
    Approximate Shannon entropy of particle distribution:
    - Divide box into grid_size x grid_size cells.
    - Count how many particles fall in each cell.
    - Compute H = -sum(p_i * log2(p_i)).
    Max entropy is log2(grid_size^2) when all cells are equally likely.
    """
    # 2D histogram normalized to probabilities
    hist, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=grid_size,
        range=[[0.0, BOX_SIZE], [0.0, BOX_SIZE]],
    )
    probs = hist.ravel().astype(float)
    probs /= probs.sum()  # normalize

    # Filter out empty cells (p = 0)
    mask = probs > 0
    probs = probs[mask]

    if probs.size == 0:
        return 0.0

    return float(-np.sum(probs * np.log2(probs)))


# -----------------------------
# Main visualization
# -----------------------------
def main() -> None:
    # Optional reproducibility
    if RNG_SEED is not None:
        np.random.seed(RNG_SEED)

    # Initial particle positions (low entropy)
    positions = init_particles(N_PARTICLES)

    # Entropy tracking
    entropies = []
    max_entropy = np.log2(GRID_SIZE * GRID_SIZE)

    # Figure layout: top = box, bottom = entropy vs time
    fig = plt.figure(figsize=(8, 8))
    ax_box = fig.add_subplot(2, 1, 1)
    ax_entropy = fig.add_subplot(2, 1, 2)

    # --- Top: particle box ---
    ax_box.set_title("Entropy Visualization: From Order to Disorder")
    ax_box.set_xlim(0, BOX_SIZE)
    ax_box.set_ylim(0, BOX_SIZE)
    ax_box.set_aspect("equal")
    ax_box.set_xticks([])
    ax_box.set_yticks([])
    ax_box.set_facecolor("black")

    # Draw initial scatter
    scatter = ax_box.scatter(
        positions[:, 0],
        positions[:, 1],
        s=10,
    )

    # Text showing entropy value
    text_entropy = ax_box.text(
        0.02,
        0.96,
        "",
        color="white",
        fontsize=11,
        transform=ax_box.transAxes,
        verticalalignment="top",
    )
    text_desc = ax_box.text(
        0.02,
        0.88,
        "Watch particles spread out.\nMore spread = more microstates = higher entropy.",
        color="white",
        fontsize=9,
        transform=ax_box.transAxes,
        verticalalignment="top",
    )

    # --- Bottom: entropy over time ---
    ax_entropy.set_title("Shannon Entropy Over Time")
    ax_entropy.set_xlim(0, 300)  # will update dynamically
    ax_entropy.set_ylim(0, max_entropy * 1.05)
    ax_entropy.set_xlabel("Frame")
    ax_entropy.set_ylabel("Entropy (bits)")
    (entropy_line,) = ax_entropy.plot([], [], linewidth=2)
    entropy_fill = ax_entropy.fill_between([], [], alpha=0.2)

    # -----------------------------
    # Animation update function
    # -----------------------------
    def update(frame: int):
        nonlocal positions, entropies, entropy_fill

        # Randomly move particles around
        positions = random_step(positions, STEP_SIZE)

        # Update scatter plot
        scatter.set_offsets(positions)

        # Compute new entropy
        H = shannon_entropy(positions, GRID_SIZE)
        entropies.append(H)

        # Update text (show also as fraction of max entropy)
        frac = H / max_entropy if max_entropy > 0 else 0.0
        text_entropy.set_text(
            f"Entropy: {H:.2f} bits  ({frac * 100:5.1f}% of max)"
        )

        # Extend x-limit for entropy plot if needed
        if frame >= ax_entropy.get_xlim()[1] - 10:
            ax_entropy.set_xlim(0, frame + 50)

        # Update entropy line
        x_data = np.arange(len(entropies))
        entropy_line.set_data(x_data, entropies)

        # Update the filled region under the curve (remove previous, then redraw)
        if entropy_fill is not None:
            try:
                entropy_fill.remove()
            except Exception:
                pass
        new_fill = ax_entropy.fill_between(x_data, entropies, alpha=0.15)
        entropy_fill = new_fill

        return scatter, entropy_line, entropy_fill, text_entropy

    # Run animation
    anim = FuncAnimation(
        fig,
        update,
        frames=600,
        interval=INTERVAL_MS,
        blit=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
