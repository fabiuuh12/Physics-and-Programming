"""
Animated visualization of gravitational lensing showing how the Sun bends spacetime and the path of light coming from a background star.

Stylized GR-inspired toy model (not numerically exact):
- Warped spacetime grid
- Massive Sun near the center
- Curved light path grazing the Sun before reaching Earth
- Apparent straight-line extension pointing toward the apparent star position
- Subtle camera orbit to highlight the curvature well

Run:
    python3 4th_dimension_viz.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 needed for 3D projection


# =========================
# Parameters & helpers
# =========================

# Spacetime grid
GRID_N = 80
X_MIN, X_MAX = -6.0, 6.0
Y_MIN, Y_MAX = -3.5, 3.5

WELL_DEPTH = 2.4
SOFTENING = 0.35  # avoid singularity at r = 0

# Animation
N_FRAMES = 420
INTERVAL_MS = 25

# Visual tuning
BG_COLOR = "#02030a"
SURFACE_CMAP = "viridis"
GRID_ALPHA = 0.23

SUN_COLOR = "#ffdd55"
EARTH_COLOR = "#4aa3ff"
STAR_ACTUAL_COLOR = "#ffbb66"
STAR_APP_COLOR = "#ffe0aa"
LIGHT_PATH_COLOR = "#c75bff"

SUN_R = 0.9
EARTH_R = 0.25
STAR_R = 0.28


def spacetime_height(x, y):
    """Return z(x,y) for the warped spacetime sheet."""
    r = np.sqrt(x * x + y * y)
    z0 = -WELL_DEPTH / np.sqrt(r * r + SOFTENING * SOFTENING)
    # Normalize & scale so edges are near 0, center deepest
    z0_min = -WELL_DEPTH / np.sqrt(SOFTENING * SOFTENING)
    z = (z0 - z0_min) / (0.0 - z0_min + 1e-9)  # map [z0_min,0] -> [1,0]
    return -z * WELL_DEPTH


def make_sphere(cx, cy, cz, r, n=40):
    """Generate sphere surface coordinates."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    uu, vv = np.meshgrid(u, v)
    xs = cx + r * np.cos(uu) * np.sin(vv)
    ys = cy + r * np.sin(uu) * np.sin(vv)
    zs = cz + r * np.cos(vv)
    return xs, ys, zs


# =========================
# Build scene geometry
# =========================

# Grid
x = np.linspace(X_MIN, X_MAX, GRID_N)
y = np.linspace(Y_MIN, Y_MAX, GRID_N)
X, Y = np.meshgrid(x, y)
Z = spacetime_height(X, Y)

# Sun at origin, slightly above bottom of the well
sun_cx, sun_cy = 0.0, 0.0
sun_cz = spacetime_height(0.0, 0.0) + SUN_R * 0.25

# Earth on right
earth_cx, earth_cy = 4.7, 0.6
earth_cz = spacetime_height(earth_cx, earth_cy) + EARTH_R * 0.4

# Distant star (actual position) on left, above grid
star_cx, star_cy = -4.8, 1.6
star_cz = spacetime_height(star_cx, star_cy) + STAR_R * 0.9

# =========================
# Light path (curved geodesic-ish)
# =========================

# Parameter from 0 (star) to 1 (Earth)
t = np.linspace(0.0, 1.0, 400)

# Rough base straight line
x_base = star_cx + (earth_cx - star_cx) * t
y_base = star_cy + (earth_cy - star_cy) * t

# Add a "bending" term that pulls the path inward near the Sun
# Peak bending near x ~ 0; sign chosen so ray dips toward the Sun.
bend = -1.3 * np.exp(-((x_base - 0.0) ** 2) / (2.0 * 1.1**2))
y_path = y_base + bend

# Sample z from spacetime surface so the ray hugs the warped grid
z_path = spacetime_height(x_base, y_path) + 0.05

# =========================
# Apparent straight-line extension
# =========================

# Take final direction of curved path near Earth
dx = x_base[-1] - x_base[-6]
dy = y_path[-1] - y_path[-6]
norm = np.hypot(dx, dy) + 1e-9
ux, uy = dx / norm, dy / norm

# Extend backward from Earth along this direction
L_app = 5.5
s_ext = np.linspace(0.0, 1.0, 150)
x_app = earth_cx - ux * L_app * s_ext
y_app = earth_cy - uy * L_app * s_ext
z_app = spacetime_height(x_app, y_app) + 0.06

# Apparent star position = end of extension
star_app_cx, star_app_cy, star_app_cz = x_app[-1], y_app[-1], z_app[-1] + STAR_R * 0.6


# =========================
# Matplotlib setup
# =========================

plt.style.use("dark_background")

fig = plt.figure(figsize=(9, 5.2))
ax = fig.add_subplot(111, projection="3d")

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Camera
elev0 = 28
azim0 = -55
ax.view_init(elev=elev0, azim=azim0)

# Limits
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_zlim(-WELL_DEPTH * 1.25, 2.2)

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_title(
    "Gravitational Lensing: Star, Sun, Curved Light Path, and Apparent Position",
    color="white",
    pad=14,
    fontsize=11,
)

# Spacetime sheet
surface = ax.plot_surface(
    X,
    Y,
    Z,
    rstride=1,
    cstride=1,
    cmap=SURFACE_CMAP,
    linewidth=0,
    antialiased=True,
    alpha=0.96,
)

# Grid lines overlay
grid_lines = []
step = 4
for i in range(0, GRID_N, step):
    # Constant y
    (gy,) = ax.plot(
        X[i, :],
        Y[i, :],
        Z[i, :],
        color="white",
        alpha=GRID_ALPHA,
        linewidth=0.35,
    )
    grid_lines.append(gy)
    # Constant x
    (gx,) = ax.plot(
        X[:, i],
        Y[:, i],
        Z[:, i],
        color="white",
        alpha=GRID_ALPHA,
        linewidth=0.35,
    )
    grid_lines.append(gx)

# Sun
sun_x, sun_y, sun_z = make_sphere(sun_cx, sun_cy, sun_cz, SUN_R)
sun_surf = ax.plot_surface(
    sun_x,
    sun_y,
    sun_z,
    color=SUN_COLOR,
    linewidth=0,
    antialiased=True,
    shade=True,
)

# Earth
earth_x, earth_y, earth_z = make_sphere(earth_cx, earth_cy, earth_cz, EARTH_R)
earth_surf = ax.plot_surface(
    earth_x,
    earth_y,
    earth_z,
    color=EARTH_COLOR,
    linewidth=0,
    antialiased=True,
    shade=True,
)

# Actual star (true position)
star_x, star_y, star_z = make_sphere(star_cx, star_cy, star_cz, STAR_R)
star_actual_surf = ax.plot_surface(
    star_x,
    star_y,
    star_z,
    color=STAR_ACTUAL_COLOR,
    linewidth=0,
    antialiased=True,
    shade=True,
)

# Apparent star (where light seems to come from)
star_app_x, star_app_y, star_app_z = make_sphere(
    star_app_cx, star_app_cy, star_app_cz, STAR_R * 0.7
)
star_app_surf = ax.plot_surface(
    star_app_x,
    star_app_y,
    star_app_z,
    color=STAR_APP_COLOR,
    linewidth=0,
    antialiased=True,
    shade=True,
    alpha=0.95,
)

# Curved light path (solid)
(light_path_line,) = ax.plot(
    x_base,
    y_path,
    z_path,
    color=LIGHT_PATH_COLOR,
    linewidth=2.2,
    alpha=1.0,
    label="Curved light path",
)

# Apparent straight-line path (dashed)
(app_path_line,) = ax.plot(
    x_app,
    y_app,
    z_app,
    color=LIGHT_PATH_COLOR,
    linewidth=1.4,
    alpha=0.95,
    linestyle="--",
    label="Apparent path",
)

# Labels
ax.text(sun_cx, sun_cy, sun_cz + SUN_R * 1.4, "Sun", color="white", fontsize=9, ha="center")
ax.text(
    earth_cx,
    earth_cy,
    earth_cz + EARTH_R * 1.7,
    "Earth",
    color="white",
    fontsize=9,
    ha="center",
)
ax.text(
    star_cx,
    star_cy,
    star_cz + STAR_R * 1.8,
    "Actual star",
    color="white",
    fontsize=8,
    ha="center",
)
ax.text(
    star_app_cx,
    star_app_cy,
    star_app_cz + STAR_R * 1.5,
    "Apparent star",
    color="white",
    fontsize=8,
    ha="center",
)

ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=7)


# =========================
# Animation
# =========================


def update(frame: int):
    # Small camera orbit for depth effect
    azim = azim0 + 40.0 * np.sin(2 * np.pi * frame / N_FRAMES)
    elev = elev0 + 3.0 * np.sin(2 * np.pi * frame / (N_FRAMES * 0.8))
    ax.view_init(elev=elev, azim=azim)
    return (
        surface,
        sun_surf,
        earth_surf,
        star_actual_surf,
        star_app_surf,
        light_path_line,
        app_path_line,
        *grid_lines,
    )


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=INTERVAL_MS,
    blit=False,
)


if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
