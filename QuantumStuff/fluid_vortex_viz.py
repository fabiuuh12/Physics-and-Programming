"""2D toy fluid visualization with a cylinder generating a Kármán vortex street.

This is not a CFD solver, just a qualitative illustration:
- Smoke tracers advect in a prescribed velocity field resembling flow past a cylinder.
- The wake alternates swirls to mimic vortex shedding (Reynolds-number vibes).
- A quiver overlay shows local velocity directions, and text highlights shedding frequency.

Run:
    python3 fluid_vortex_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DT = 0.03
N_FRAMES = 500

SMOKE_COUNT = 1600
DOMAIN = (-2.0, 4.0, -1.5, 1.5)  # xmin, xmax, ymin, ymax
CYL_RADIUS = 0.35
BASE_FLOW = 1.0
SHEDDING_FREQ = 0.7

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.set_xlim(DOMAIN[0], DOMAIN[1])
ax.set_ylim(DOMAIN[2], DOMAIN[3])
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor("#030711")
ax.set_title("Kármán Vortex Street (toy model, not full Navier–Stokes)", color="white")

theta = np.linspace(0, 2 * np.pi, 200)
ax.plot(CYL_RADIUS * np.cos(theta), CYL_RADIUS * np.sin(theta), color="#8dd7ff", linewidth=2.0)
ax.text(-0.02, -0.65, "Cylinder obstacle", color="#bfe5ff", ha="center")


def initial_smoke(n: int):
    xs = np.random.uniform(DOMAIN[0], DOMAIN[0] + 0.5, n)
    ys = np.random.uniform(DOMAIN[2], DOMAIN[3], n)
    phase = np.random.rand(n)
    return xs, ys, phase


smoke_x, smoke_y, smoke_phase = initial_smoke(SMOKE_COUNT)


def velocity_field(x: np.ndarray, y: np.ndarray, t: float):
    """Semi-procedural velocity pattern."""
    base_u = BASE_FLOW * np.ones_like(x)
    base_v = np.zeros_like(y)

    r2 = x**2 + y**2
    mask = r2 < CYL_RADIUS**2
    base_u[mask] = 0.0
    base_v[mask] = 0.0

    # Potential flow deflection around cylinder
    with np.errstate(divide="ignore", invalid="ignore"):
        deflect = CYL_RADIUS**2 / np.maximum(r2, CYL_RADIUS**2)
    u_deflect = -deflect * (x * y) * 1.2
    v_deflect = deflect * (1 - 2 * (y**2) / np.maximum(r2, CYL_RADIUS**2)) * 0.6

    # Vortex street wake (alternating sign behind cylinder)
    wake_mask = x > CYL_RADIUS
    wake_strength = np.exp(-0.6 * (x - CYL_RADIUS))
    shedding = np.sin(2 * np.pi * (SHEDDING_FREQ * t - 0.35 * x))
    u_wake = np.zeros_like(x)
    v_wake = np.zeros_like(y)
    u_wake[wake_mask] = 0.25 * wake_strength[wake_mask] * np.sin(np.pi * y[wake_mask])
    v_wake[wake_mask] = 0.6 * wake_strength[wake_mask] * shedding[wake_mask]

    u = base_u + u_deflect + u_wake
    v = base_v + v_deflect + v_wake

    return u, v


smoke_scatter = ax.scatter(smoke_x, smoke_y, s=6, c="#9fe0ff", alpha=0.8)

grid_x = np.linspace(DOMAIN[0], DOMAIN[1], 18)
grid_y = np.linspace(DOMAIN[2], DOMAIN[3], 12)
GX, GY = np.meshgrid(grid_x, grid_y)
quiver = ax.quiver(GX, GY, np.zeros_like(GX), np.zeros_like(GY), color="#ffaa7e", alpha=0.6)

info_text = ax.text(
    0.02,
    0.94,
    "",
    transform=ax.transAxes,
    color="#f9f7ff",
    fontsize=10,
    ha="left",
    va="top",
)


def step_particles(t: float):
    global smoke_x, smoke_y, smoke_phase
    u, v = velocity_field(smoke_x, smoke_y, t)
    v += 0.1 * np.sin(2 * np.pi * (smoke_phase + t * 0.2))  # slight turbulence

    smoke_x += u * DT
    smoke_y += v * DT

    # Wrap particles that leave the domain
    reset_mask = (smoke_x > DOMAIN[1]) | (smoke_y > DOMAIN[3]) | (smoke_y < DOMAIN[2]) | (
        (smoke_x**2 + smoke_y**2) < (CYL_RADIUS * 0.95) ** 2
    )
    num_reset = reset_mask.sum()
    if num_reset > 0:
        smoke_x[reset_mask], smoke_y[reset_mask], smoke_phase[reset_mask] = initial_smoke(num_reset)


def update(frame: int):
    t = frame * DT
    step_particles(t)
    smoke_scatter.set_offsets(np.column_stack((smoke_x, smoke_y)))

    u_grid, v_grid = velocity_field(GX.ravel(), GY.ravel(), t)
    quiver.set_UVC(u_grid.reshape(GX.shape), v_grid.reshape(GY.shape))

    info_text.set_text(
        f"Synthetic Reynolds vibe: cylinder shedding @ {SHEDDING_FREQ:.2f} Hz –"
        f" vortices alternate downstream."
    )

    return smoke_scatter, quiver, info_text


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
