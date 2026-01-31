"""Animated cheat-sheet for the four laws of thermodynamics.

Panels:
1. Zeroth law – three bodies equilibrate to the same temperature when in mutual
   thermal contact.
2. First law – internal energy changes track heat input versus work done by/on
   a gas in a piston.
3. Second law – a gas expands to fill available volume while entropy (disorder)
   increases.
4. Third law – entropy drops toward zero as temperature approaches absolute zero.

Run:
    python3 thermodynamics_laws_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch


DT = 0.05
N_FRAMES = 520

N_GAS_PARTICLES = 80
N_MIX_PARTICLES = 140

plt.style.use("dark_background")


def temp_to_color(temp_k: float) -> tuple[float, float, float, float]:
    """Map temperature (Kelvin-ish) to a color for discs."""
    # Normalize to [0, 1] for the inferno colormap.
    norm = np.clip((temp_k - 270.0) / 80.0, 0.0, 1.0)
    return plt.cm.inferno(norm)


fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

ax_zero = fig.add_subplot(gs[0, 0])
ax_first = fig.add_subplot(gs[0, 1])
ax_second = fig.add_subplot(gs[1, 0])
ax_third = fig.add_subplot(gs[1, 1])


# -------------------------------
# Zeroth law panel configuration
# -------------------------------
ax_zero.set_title("Zeroth Law: Thermal Equilibrium", fontsize=11, color="#f4f4ff")
ax_zero.set_xlim(-1.1, 1.1)
ax_zero.set_ylim(-0.1, 1.25)
ax_zero.set_xticks([])
ax_zero.set_yticks([])
ax_zero.set_facecolor("#06060c")

zeroth_positions = {
    "A": (-0.7, 0.6),
    "B": (0.0, 0.6),
    "C": (0.7, 0.6),
}

zeroth_circles: dict[str, Circle] = {}
zeroth_text: dict[str, plt.Text] = {}

# Connection lines A-B and B-C (thermal contact)
ax_zero.plot(
    [zeroth_positions["A"][0], zeroth_positions["B"][0]],
    [zeroth_positions["A"][1], zeroth_positions["B"][1]],
    color="#1f324a",
    linewidth=2.5,
    alpha=0.7,
)
ax_zero.plot(
    [zeroth_positions["B"][0], zeroth_positions["C"][0]],
    [zeroth_positions["B"][1], zeroth_positions["C"][1]],
    color="#1f324a",
    linewidth=2.5,
    alpha=0.7,
)

for label, (x_pos, y_pos) in zeroth_positions.items():
    circle = Circle((x_pos, y_pos), 0.28, edgecolor="#ffffff", linewidth=1.0)
    ax_zero.add_patch(circle)
    zeroth_circles[label] = circle
    text = ax_zero.text(
        x_pos,
        y_pos - 0.45,
        "",
        color="#d2e9ff",
        fontsize=9,
        ha="center",
    )
    zeroth_text[label] = text

ax_zero.text(0.0, 0.05, "If A = B and B = C → A = C", color="#9fe2ff", fontsize=10, ha="center")


# ------------------------------
# First law panel configuration
# ------------------------------
ax_first.set_title("First Law: ΔU = Q − W", fontsize=11, color="#f4f4ff")
ax_first.set_xlim(0.0, 1.45)
ax_first.set_ylim(0.0, 1.55)
ax_first.set_xticks([])
ax_first.set_yticks([])
ax_first.set_facecolor("#04040a")

container = Rectangle(
    (0.3, 0.25),
    0.6,
    1.0,
    linewidth=1.4,
    edgecolor="#dce3ff",
    facecolor="none",
)
ax_first.add_patch(container)

piston = Rectangle(
    (0.3, 0.95),
    0.6,
    0.08,
    color="#818ff7",
    alpha=0.85,
)
ax_first.add_patch(piston)

gas_points = np.column_stack(
    (
        np.random.rand(N_GAS_PARTICLES),
        np.random.rand(N_GAS_PARTICLES),
    )
)
gas_scatter = ax_first.scatter(
    0.3 + gas_points[:, 0] * 0.6,
    0.3 + gas_points[:, 1] * 0.6,
    s=18,
    color="#7cf9ff",
    alpha=0.85,
)

heat_arrow = FancyArrowPatch(
    (0.17, 0.75),
    (0.3, 0.75),
    arrowstyle="->",
    mutation_scale=18,
    linewidth=1.4,
    color="#ffb38f",
)
ax_first.add_patch(heat_arrow)

work_arrow = FancyArrowPatch(
    (0.6, 1.34),
    (0.6, 1.12),
    arrowstyle="->",
    mutation_scale=18,
    linewidth=1.4,
    color="#8fffdf",
)
ax_first.add_patch(work_arrow)

heat_text = ax_first.text(0.05, 0.83, "", color="#ffbf9d", fontsize=9)
work_text = ax_first.text(0.95, 1.42, "", color="#8effd7", fontsize=9, ha="center")
delta_u_text = ax_first.text(0.6, 0.18, "", color="#f7f7ff", fontsize=10, ha="center")


# -------------------------------
# Second law panel configuration
# -------------------------------
ax_second.set_title("Second Law: Entropy increases (isolated system)", fontsize=11, color="#f4f4ff")
ax_second.set_xlim(0.0, 1.05)
ax_second.set_ylim(0.0, 1.0)
ax_second.set_xticks([])
ax_second.set_yticks([])
ax_second.set_facecolor("#07070e")

chamber = Rectangle(
    (0.05, 0.1),
    0.9,
    0.8,
    linewidth=1.3,
    edgecolor="#cfd8ff",
    facecolor="none",
)
ax_second.add_patch(chamber)
partition_line = ax_second.axvline(0.5, color="#38446d", linestyle="--", linewidth=1.0, alpha=0.8)

mix_initial = np.column_stack(
    (
        0.05 + 0.35 * np.random.rand(N_MIX_PARTICLES),
        0.15 + 0.7 * np.random.rand(N_MIX_PARTICLES),
    )
)
mix_final = np.column_stack(
    (
        0.05 + 0.9 * np.random.rand(N_MIX_PARTICLES),
        0.15 + 0.7 * np.random.rand(N_MIX_PARTICLES),
    )
)
mix_scatter = ax_second.scatter(
    mix_initial[:, 0],
    mix_initial[:, 1],
    s=12,
    color="#8be0ff",
    alpha=0.9,
)
entropy_text = ax_second.text(0.52, 0.03, "", color="#f9f4bf", fontsize=10, ha="center")


# ------------------------------
# Third law panel configuration
# ------------------------------
ax_third.set_title("Third Law: S → 0 as T → 0 K", fontsize=11, color="#f4f4ff")
ax_third.set_xlim(0, 420)
ax_third.set_ylim(0, 5.0)
ax_third.set_xlabel("Temperature (K)", color="#dbe8ff")
ax_third.set_ylabel("Entropy (arb. units)", color="#dbe8ff")
ax_third.grid(color="#1a1a26", linewidth=0.6, alpha=0.6)

temp_curve = np.linspace(1.0, 420.0, 400)
entropy_curve = np.log(temp_curve + 1.0)
ax_third.plot(temp_curve, entropy_curve, color="#8be7ff", linewidth=2.0)
third_marker, = ax_third.plot([], [], marker="o", color="#ffe57d", markersize=6)
cooling_text = ax_third.text(230, 4.3, "", color="#ffe57d", fontsize=10)


# ===========================
# Animation update per frame
# ===========================

def update(frame: int):
    t = frame * DT

    # Zeroth law equilibration
    target_temp = 300.0 + 6.0 * np.sin(0.35 * t)
    offsets = np.array([14.0, -10.0, 20.0]) * np.exp(-0.35 * t)
    temps = {
        label: target_temp + offset
        for label, offset in zip(zeroth_positions.keys(), offsets)
    }

    for label, circle in zeroth_circles.items():
        T = temps[label]
        circle.set_facecolor(temp_to_color(T))
        zeroth_text[label].set_text(f"{label}: {T:.0f} K")

    # First law energy bookkeeping
    piston_frac = 0.55 + 0.35 * np.sin(0.4 * t)
    gas_height = 0.35 + piston_frac * 0.75
    piston.set_y(gas_height)

    jitter = 0.015 * np.sin(1.5 * t + gas_points[:, 0] * 5.0)
    gas_offsets = np.column_stack(
        (
            0.3 + gas_points[:, 0] * 0.6,
            0.3 + gas_points[:, 1] * (gas_height - 0.32) + jitter,
        )
    )
    gas_scatter.set_offsets(gas_offsets)

    heat_flow = 35.0 + 15.0 * np.sin(0.45 * t + 0.6)
    work_done = 28.0 + 12.0 * np.sin(0.45 * t)
    delta_u = heat_flow - work_done

    heat_text.set_text(f"Q in ≈ {heat_flow:4.1f}")
    work_text.set_text(f"W out ≈ {work_done:4.1f}")
    delta_u_text.set_text(f"ΔU ≈ {delta_u:4.1f}")

    # Update arrow extents
    heat_arrow.set_positions(
        (0.17, gas_height - 0.1),
        (0.3, gas_height - 0.1),
    )
    work_arrow.set_positions(
        (0.6, gas_height + 0.18),
        (0.6, gas_height + 0.4 + 0.05 * np.sin(0.3 * t)),
    )

    # Second law mixing / entropy increase
    mix_fraction = 1.0 - np.exp(-0.25 * t)
    blend_positions = (1.0 - mix_fraction) * mix_initial + mix_fraction * mix_final
    mix_scatter.set_offsets(blend_positions)
    partition_line.set_alpha(max(0.0, 0.6 * (1.0 - mix_fraction)))
    entropy_text.set_text(f"Entropy ↑ {mix_fraction * 100:4.0f}%")

    # Third law cooling curve
    temperature = 320.0 * np.exp(-0.012 * t) + 2.0
    entropy_value = np.log(temperature + 1.0)
    third_marker.set_data([temperature], [entropy_value])
    cooling_text.set_text(f"T ≈ {temperature:5.1f} K\nS ≈ {entropy_value:4.2f}")

    return (
        *zeroth_circles.values(),
        piston,
        gas_scatter,
        heat_arrow,
        work_arrow,
        mix_scatter,
        partition_line,
        third_marker,
    )


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=DT * 1000.0,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
