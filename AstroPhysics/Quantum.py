"""
Binary vs Quantum Computing Visualization

Run this file to get an *intuition* for the difference between:
- A classical bit: only ever 0 or 1.
- A qubit: can be in a superposition (a|0> + b|1>) with probabilities that vary smoothly.

What this shows (4 panels):
- Top-left: A single classical bit randomly flips between 0 and 1.
- Bottom-left: Empirical probabilities of the classical bit so far (how often we've seen 0 vs 1).
- Top-right: A single qubit state on the Bloch circle (2D cross-section).
- Bottom-right: The qubit's current probabilities P(0) and P(1) from its state.

This is NOT a full simulator of quantum algorithms.
It's just a clear picture of “discrete vs continuous / definite vs superposed”.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Config
# -----------------------------
N_FRAMES = 400
INTERVAL_MS = 60

# Classical bit tracking
classical_history = []

# Qubit state parameters:
# |ψ> = cos(theta/2)|0> + e^{iφ} sin(theta/2)|1>
# We'll smoothly vary theta over time to show changing superposition.
def qubit_state(frame: int) -> tuple[complex, complex]:
    # Make theta sweep back and forth between 0 and π
    # (0 -> pure |0>, π -> pure |1>, middle -> balanced superposition).
    t = frame / 80.0
    theta = (np.sin(t) * 0.5 + 0.5) * np.pi  # in [0, π]
    phi = 0.0  # keep phase fixed here for simplicity

    a = np.cos(theta / 2.0)
    b = np.exp(1j * phi) * np.sin(theta / 2.0)
    return a, b


# -----------------------------
# Figure + Axes
# -----------------------------
plt.style.use("default")
fig = plt.figure(figsize=(10, 6))

ax_classical = fig.add_subplot(2, 2, 1)
ax_classical_prob = fig.add_subplot(2, 2, 3)
ax_qubit = fig.add_subplot(2, 2, 2)
ax_qubit_prob = fig.add_subplot(2, 2, 4)

fig.suptitle("Classical Bit vs Qubit (Intuition Visualizer)", fontsize=14)

# -----------------------------
# Top-left: Classical bit (0 or 1)
# -----------------------------
ax_classical.set_title("Classical Bit: Only 0 or 1")
ax_classical.set_xticks([])
ax_classical.set_yticks([])
ax_classical.set_xlim(0, 1)
ax_classical.set_ylim(0, 1)

classical_text = ax_classical.text(
    0.5,
    0.5,
    "0",
    ha="center",
    va="center",
    fontsize=40,
)

# -----------------------------
# Bottom-left: Classical probabilities
# -----------------------------
ax_classical_prob.set_title("Classical Bit: Observed Frequencies")
ax_classical_prob.set_ylim(0, 1)
ax_classical_prob.set_xticks([0, 1])
ax_classical_prob.set_xticklabels(["P(0)", "P(1)"])
classical_bars = ax_classical_prob.bar([0, 1], [1.0, 0.0])

# -----------------------------
# Top-right: Qubit on Bloch circle (2D slice)
# -----------------------------
ax_qubit.set_title("Qubit State on Bloch Circle")
ax_qubit.set_aspect("equal")
ax_qubit.set_xlim(-1.1, 1.1)
ax_qubit.set_ylim(-1.1, 1.1)
ax_qubit.set_xticks([])
ax_qubit.set_yticks([])

# Draw unit circle
circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--", alpha=0.5)
ax_qubit.add_patch(circle)

# Bloch vector arrow (we only show projection in X-Z plane)
bloch_arrow = ax_qubit.arrow(
    0,
    0,
    0,
    1,
    head_width=0.06,
    length_includes_head=True,
)

qubit_label = ax_qubit.text(
    0.0,
    -1.25,
    "",
    ha="center",
    va="center",
    fontsize=9,
)

# -----------------------------
# Bottom-right: Qubit probabilities
# -----------------------------
ax_qubit_prob.set_title("Qubit: Instantaneous Probabilities")
ax_qubit_prob.set_ylim(0, 1)
ax_qubit_prob.set_xticks([0, 1])
ax_qubit_prob.set_xticklabels(["P(0)", "P(1)"])
qubit_bars = ax_qubit_prob.bar([0, 1], [1.0, 0.0])

# -----------------------------
# Update function
# -----------------------------
def update(frame: int):
    global classical_history
    global bloch_arrow

    # ---- Classical bit update ----
    bit = np.random.randint(0, 2)  # 0 or 1
    classical_history.append(bit)

    p0 = classical_history.count(0) / len(classical_history)
    p1 = 1.0 - p0

    classical_text.set_text(str(bit))
    classical_bars[0].set_height(p0)
    classical_bars[1].set_height(p1)

    # ---- Qubit update ----
    a, b = qubit_state(frame)
    p0_q = float(np.abs(a) ** 2)
    p1_q = float(np.abs(b) ** 2)

    # Bloch vector components (phi fixed = 0)
    # n_x = sin(theta), n_z = cos(theta)
    # from a = cos(theta/2), b = sin(theta/2)
    theta = 2 * np.arccos(a.real)  # since phi = 0 and a≥0 in this param
    n_x = np.sin(theta)
    n_z = np.cos(theta)

    # Update Bloch arrow by removing and redrawing
    for art in list(ax_qubit.lines) + list(ax_qubit.patches):
        # keep the circle, but remove old arrows if any
        if art is bloch_arrow or isinstance(art, plt.Arrow):
            try:
                art.remove()
            except Exception:
                pass

    # Draw updated arrow
    new_arrow = ax_qubit.arrow(
        0,
        0,
        n_x,
        n_z,
        head_width=0.06,
        length_includes_head=True,
    )

    # Store reference so our "remove" logic can see it next frame
    bloch_arrow = new_arrow

    # Update qubit probability bars
    qubit_bars[0].set_height(p0_q)
    qubit_bars[1].set_height(p1_q)

    # Text label for qubit state
    qubit_label.set_text(
        f"|ψ⟩ = {a.real:+.2f}|0⟩ + {b.real:+.2f}|1⟩   "
        f"(P(0)={p0_q:.2f}, P(1)={p1_q:.2f})"
    )

    return (
        classical_text,
        *classical_bars,
        bloch_arrow,
        *qubit_bars,
        qubit_label,
    )


# -----------------------------
# Run animation
# -----------------------------
def main() -> None:
    anim = FuncAnimation(
        fig,
        update,
        frames=N_FRAMES,
        interval=INTERVAL_MS,
        blit=False,
        repeat=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
