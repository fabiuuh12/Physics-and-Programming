"""Michelson interferometer visualization with a passing gravitational wave.

What you'll see:
- An L-shaped interferometer (think LIGO) with a beam splitter and two mirrors.
- A gravitational wave stretches one arm while squeezing the other, shown by mirrors
  oscillating and wave crests shifting along each arm.
- The returning beams interfere at the detector, and the resulting intensity trace
  is plotted in real time underneath the geometry.

Run:
    python3 interferometer_gw_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle


# ===============================================================
# Parameters (tweak these to explore different qualitative regimes)
# ===============================================================
LIGHT_WAVELENGTH = 0.55        # effective laser wavelength (arbitrary units)
ARM_LENGTH = 2.6               # nominal length of each interferometer arm
GW_AMPLITUDE = 0.18            # dimensionless strain amplitude (toy scale)
GW_FREQUENCY = 0.32            # Hz-equivalent for the toy gravitational wave
PHASE_VELOCITY = 6.0           # how fast the visual wave ripples slide along the arms
NUM_WAVE_SAMPLES = 180         # resolution for the drawn wave traces

DT = 0.04                      # time per animation frame
N_FRAMES = 650
HISTORY_WINDOW = 18.0          # seconds shown in the intensity plot


# =====================
# Figure configuration
# =====================

plt.style.use("dark_background")

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.32)

ax_setup = fig.add_subplot(gs[0])
ax_intensity = fig.add_subplot(gs[1])

for ax in (ax_setup, ax_intensity):
    ax.tick_params(colors="#aaaaaa")

ax_setup.set_xlim(-0.6, ARM_LENGTH + 0.7)
ax_setup.set_ylim(-0.8, ARM_LENGTH + 0.7)
ax_setup.set_aspect("equal")
ax_setup.set_xticks([])
ax_setup.set_yticks([])
ax_setup.set_facecolor("#05050a")
ax_setup.set_title("Michelson Interferometer + Passing Gravitational Wave", fontsize=12)

# Draw the rigid vacuum tubes for context.
ax_setup.plot([0, ARM_LENGTH * 1.05], [0, 0], color="#2f3b52", linewidth=2.6, alpha=0.6)
ax_setup.plot([0, 0], [0, ARM_LENGTH * 1.05], color="#2f3b52", linewidth=2.6, alpha=0.6)

# Beam splitter & mirrors
beam_splitter = Rectangle(
    (-0.11, -0.11), 0.22, 0.22,
    angle=45,
    linewidth=1.3,
    edgecolor="#f0f0ff",
    facecolor="#535cff",
    alpha=0.9,
)
ax_setup.add_patch(beam_splitter)
ax_setup.text(-0.35, -0.35, "Beam\nSplitter", color="#d8ddff", fontsize=9, ha="center")

mirror_x = Rectangle(
    (ARM_LENGTH - 0.08, -0.22),
    0.14,
    0.44,
    linewidth=1.4,
    edgecolor="#ffe0b5",
    facecolor="#f8b684",
)
ax_setup.add_patch(mirror_x)
ax_setup.text(ARM_LENGTH + 0.18, -0.05, "Mirror X", color="#ffdcb2", fontsize=9)

mirror_y = Rectangle(
    (-0.22, ARM_LENGTH - 0.08),
    0.44,
    0.14,
    linewidth=1.4,
    edgecolor="#ffe0b5",
    facecolor="#f8b684",
)
ax_setup.add_patch(mirror_y)
ax_setup.text(-0.32, ARM_LENGTH + 0.15, "Mirror Y", color="#ffdcb2", fontsize=9, rotation=90)

# Detector indicator (brightness encodes interference output)
detector = Circle(
    (0.35, -0.35),
    0.16,
    facecolor="#111419",
    edgecolor="#ffffff",
    linewidth=1.0,
)
ax_setup.add_patch(detector)
ax_setup.text(0.35, -0.62, "Detector", color="#cfe8ff", fontsize=9, ha="center")

# Laser arrow feeding the splitter
laser_arrow = FancyArrowPatch(
    (-0.55, -0.35),
    (-0.12, -0.08),
    arrowstyle="->",
    mutation_scale=15,
    linewidth=1.2,
    color="#ff9b71",
)
ax_setup.add_patch(laser_arrow)
ax_setup.text(-0.72, -0.35, "Laser", color="#ffc4a1", fontsize=9, ha="center")

# Wave traces along each arm
x_wave_line, = ax_setup.plot([], [], color="#94f5ff", linewidth=2.2)
y_wave_line, = ax_setup.plot([], [], color="#fff68b", linewidth=2.2)

# Text overlays
strain_text = ax_setup.text(
    ARM_LENGTH * 0.45,
    ARM_LENGTH + 0.3,
    "",
    color="#8ef0ff",
    fontsize=10,
)
phase_text = ax_setup.text(
    ARM_LENGTH * 0.45,
    ARM_LENGTH + 0.05,
    "",
    color="#f6f2ff",
    fontsize=10,
)

# Set up the intensity subplot
ax_intensity.set_facecolor("#05050a")
ax_intensity.set_xlim(0, HISTORY_WINDOW)
ax_intensity.set_ylim(0, 1.05)
ax_intensity.set_xlabel("Time (s)", color="#cccccc")
ax_intensity.set_ylabel("Detector intensity (arb. units)", color="#cccccc")
ax_intensity.grid(color="#222222", linewidth=0.6, alpha=0.5)

intensity_line, = ax_intensity.plot([], [], color="#9fe8ff", linewidth=1.8)
intensity_point, = ax_intensity.plot([], [], marker="o", color="#ffffff", markersize=5)

time_history: list[float] = []
intensity_history: list[float] = []


# ====================
# Animation update
# ====================

def update(frame: int):
    t = frame * DT

    # Toy gravitational-wave strain (plus polarization)
    strain = GW_AMPLITUDE * np.sin(2.0 * np.pi * GW_FREQUENCY * t)
    arm_length_x = ARM_LENGTH * (1.0 + 0.5 * strain)
    arm_length_y = ARM_LENGTH * (1.0 - 0.5 * strain)

    # Update mirror positions to reflect stretched/squeezed arms.
    mirror_x.set_x(arm_length_x - 0.08)
    mirror_y.set_y(arm_length_y - 0.08)

    # Wave visual along X arm
    x_positions = np.linspace(0.0, arm_length_x, NUM_WAVE_SAMPLES)
    phase_x = 2.0 * np.pi * (x_positions / LIGHT_WAVELENGTH - PHASE_VELOCITY * t)
    x_displacement = 0.07 * np.sin(phase_x)
    x_wave_line.set_data(x_positions, x_displacement)

    # Wave visual along Y arm
    y_positions = np.linspace(0.0, arm_length_y, NUM_WAVE_SAMPLES)
    phase_y = 2.0 * np.pi * (y_positions / LIGHT_WAVELENGTH - PHASE_VELOCITY * t + np.pi / 2.0)
    y_displacement = 0.07 * np.sin(phase_y)
    y_wave_line.set_data(y_displacement, y_positions)

    # Interference pattern at the detector
    phase_difference = (4.0 * np.pi / LIGHT_WAVELENGTH) * (arm_length_x - arm_length_y)
    intensity = 0.5 * (1.0 + np.cos(phase_difference))

    # Update detector brightness (clamp for safety)
    detector_color = np.clip(intensity, 0.0, 1.0)
    detector.set_facecolor((0.2 + 0.6 * detector_color, 0.05, 0.15 + 0.7 * detector_color))

    # Update text overlays
    strain_text.set_text(f"Strain h(t) ≈ {strain:+.3f}")
    phase_text.set_text(f"Phase difference ≈ {phase_difference/np.pi:+.2f} π rad")

    # Rolling history for the intensity plot
    time_history.append(t)
    intensity_history.append(intensity)
    while time_history and (t - time_history[0]) > HISTORY_WINDOW:
        time_history.pop(0)
        intensity_history.pop(0)

    intensity_line.set_data(time_history, intensity_history)
    intensity_point.set_data([time_history[-1]], [intensity_history[-1]])

    # Keep the intensity axis view sliding with time
    if time_history:
        t0 = max(0.0, time_history[-1] - HISTORY_WINDOW)
        ax_intensity.set_xlim(t0, t0 + HISTORY_WINDOW)

    return (
        x_wave_line,
        y_wave_line,
        mirror_x,
        mirror_y,
        detector,
        intensity_line,
        intensity_point,
        strain_text,
        phase_text,
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
