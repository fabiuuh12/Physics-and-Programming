"""Schrödinger's cat thought-experiment visual: superposition decoheres into a definite outcome.

What the animation shows:
- Left bars track amplitudes for |alive> and |dead> components plus a fading coherence overlay.
- Right heatmap mimics the environment gaining information about the system (decoherence).
- Once the coherence drops below a threshold, the wavefunction "collapses" to a random outcome,
and the progress text signals that measurement has finished.

Run:
    python3 schrodinger_cat_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DT = 0.04
N_FRAMES = 420
DECOHERENCE_RATE = 0.22
ENV_SPEED = 0.35
COHERENCE_THRESHOLD = 0.12

GRID_SIZE = 90

plt.style.use("dark_background")

fig = plt.figure(figsize=(8.5, 5))
gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 0.6], hspace=0.32, wspace=0.25)

ax_bars = fig.add_subplot(gs[0, 0])
ax_env = fig.add_subplot(gs[0, 1])
ax_text = fig.add_subplot(gs[1, :])
ax_text.axis("off")

# -----------------------
# Left: probability bars
# -----------------------
ax_bars.set_xlim(0, 1.05)
ax_bars.set_ylim(-0.5, 1.5)
ax_bars.set_xticks([0, 0.5, 1.0])
ax_bars.set_yticks([0, 1])
ax_bars.set_yticklabels(["|dead⟩", "|alive⟩"], fontsize=11)
ax_bars.set_title("Cat state amplitudes", fontsize=12)
ax_bars.grid(axis="x", color="#151528", alpha=0.5)

alive_bar = ax_bars.barh(1, 0.5, color="#7ef5c9", alpha=0.85)[0]
dead_bar = ax_bars.barh(0, 0.5, color="#ff9ea9", alpha=0.85)[0]

coherence_patch = ax_bars.barh(
    [0, 1],
    [0.5, 0.5],
    color="#8eb4ff",
    alpha=0.35,
)

ax_bars.text(0.02, 1.25, "Quantum superposition", color="#dfe6ff", fontsize=10)

# -----------------------
# Right: environment map
# -----------------------
env_field = np.zeros((GRID_SIZE, GRID_SIZE))
env_img = ax_env.imshow(
    env_field,
    cmap="plasma",
    interpolation="bilinear",
    origin="lower",
    vmin=-1,
    vmax=1,
)
ax_env.set_title("Environment picking up information", fontsize=12)
ax_env.set_xticks([])
ax_env.set_yticks([])

coherence_text = fig.text(0.5, 0.07, "", color="#f5f8ff", ha="center", fontsize=11)
status_text = fig.text(0.5, 0.02, "", color="#ffffc6", ha="center", fontsize=10)

collapsed = False
collapsed_state = None


def update(frame: int):
    global collapsed, collapsed_state

    t = frame * DT
    coherence = np.exp(-DECOHERENCE_RATE * t)

    # Introduce a mild oscillation in amplitudes while coherent
    if collapsed:
        if collapsed_state == "alive":
            alive_prob = 1.0
            dead_prob = 0.0
        else:
            alive_prob = 0.0
            dead_prob = 1.0
    else:
        alive_prob = 0.5 + 0.15 * np.sin(0.8 * t)
        dead_prob = 1.0 - alive_prob
        if coherence <= COHERENCE_THRESHOLD:
            collapsed = True
            collapsed_state = np.random.choice(["alive", "dead"], p=[alive_prob, dead_prob])

    alive_bar.set_width(alive_prob)
    dead_bar.set_width(dead_prob)

    if collapsed:
        coherence_overlay = 0.0
    else:
        coherence_overlay = coherence

    for c_patch, width in zip(coherence_patch, [alive_prob, dead_prob]):
        c_patch.set_width(width * coherence_overlay)
        c_patch.set_alpha(0.4 * coherence_overlay)

    # Environment heatmap evolves with noise + coherence level
    x = np.linspace(-np.pi, np.pi, GRID_SIZE)
    y = np.linspace(-np.pi, np.pi, GRID_SIZE)
    X, Y = np.meshgrid(x, y)
    phase = ENV_SPEED * t
    env_pattern = (
        0.6 * np.sin(1.6 * X + phase)
        + 0.5 * np.cos(1.2 * Y - 0.5 * phase)
        + 0.3 * np.sin(0.8 * (X + Y) + 0.9 * phase)
    )
    env_pattern *= (1.2 - coherence)
    env_img.set_data(env_pattern)

    coherence_text.set_text(f"Coherence ≈ {coherence:0.3f}  (falls as environment learns the state)")

    if collapsed:
        state_label = collapsed_state.upper() if isinstance(collapsed_state, str) else "UNKNOWN"
        status_text.set_text(f"Measurement complete → cat is definitely {state_label} 🐈")
    else:
        status_text.set_text("System + environment still entangled … no definite outcome yet.")
    return alive_bar, dead_bar, env_img, coherence_text, status_text, *coherence_patch


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    plt.show()
