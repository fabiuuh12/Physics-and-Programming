"""
Simple visualization of a quantum-like electron "cloud".

The animation samples the particle's position from a time-varying
probability distribution (superposition of stationary states in a 2D well).
Run with: python3 quantum_particle_viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wavefunction(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """
    Combine a few 2D standing waves with time-dependent phases.
    Returns the probability density |psi|^2.
    """
    psi = (
        np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-1j * 2 * np.pi * t)
        + 0.6 * np.sin(2 * np.pi * x) * np.sin(np.pi * y) * np.exp(1j * 1.5 * np.pi * t)
        + 0.4 * np.sin(np.pi * x) * np.sin(2 * np.pi * y) * np.exp(-1j * np.pi * t)
    )
    density = np.abs(psi) ** 2
    return density / density.sum()


def sample_position(density: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Sample (x, y) based on the probability density."""
    flat = density.ravel()
    idx = np.random.choice(len(flat), p=flat)
    i, j = np.unravel_index(idx, density.shape)
    return xs[j], ys[i]


def main():
    resolution = 200
    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Quantum Particle Cloud (|ψ|² heatmap + sampled position)")

    initial = np.zeros_like(xx)
    heatmap = ax.imshow(
        initial,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap="magma",
        vmin=0,
        vmax=1,
        alpha=0.8,
    )
    particle, = ax.plot([], [], "wo", markersize=6)

    def init():
        heatmap.set_data(initial)
        particle.set_data([], [])
        return heatmap, particle

    def update(frame):
        t = frame / 40
        density = wavefunction(xx, yy, t)
        display = density / density.max()
        heatmap.set_data(display)
        x, y = sample_position(density, xs, ys)
        particle.set_data([x], [y])
        return heatmap, particle

    anim = FuncAnimation(
        fig, update, frames=400, init_func=init, interval=40, blit=True
    )
    plt.tight_layout()
    plt.show()
    return anim


if __name__ == "__main__":
    main()
