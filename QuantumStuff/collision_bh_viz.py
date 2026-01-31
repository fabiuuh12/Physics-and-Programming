

"""
Animated visualization of two black holes spiraling in and colliding.

Stylized GR-inspired toy model (not numerically exact):
- Two dark cores with bright photon rings and accretion glow.
- They orbit inward, merge into a single larger black hole.
- The spacetime "brightness" ripples outward like gravitational waves.
- Designed to feel cinematic but run in plain matplotlib.

Run:
    python3 collision_bh_viz.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Global parameters
# =========================

RES = 700                 # resolution of grid
EXTENT = 4.5              # physical size in both x,y
X_MIN, X_MAX = -EXTENT, EXTENT
Y_MIN, Y_MAX = -EXTENT, EXTENT

N_FRAMES = 520
INTERVAL_MS = 18

# Black hole + ring tuning
BH_CORE_R = 0.25
BH_PHOTON_RING_R = 0.38
BH_RING_WIDTH = 0.055
BH_GLOW = 0.55

MERGED_BH_CORE_R = 0.38
MERGED_PHOTON_RING_R = 0.6

# Orbit / inspiral tuning
INITIAL_SEP = 2.8
FINAL_SEP = 0.15
INSPIRAL_FRAC = 0.7  # fraction of frames spent spiraling in
ORBIT_FREQ = 1.6

# Gravitational wave ripple tuning
GW_START_FRAC = 0.4  # when ripples start getting visible
GW_SPEED = 0.22
GW_DECAY = 0.35
GW_FREQ = 16.0

# Starfield / background
STARFIELD_INTENSITY = 0.32

plt.style.use("dark_background")

# Precompute grid
x = np.linspace(X_MIN, X_MAX, RES)
y = np.linspace(Y_MIN, Y_MAX, RES)
XX, YY = np.meshgrid(x, y)
R = np.sqrt(XX**2 + YY**2)
TH = np.arctan2(YY, XX)


def soft_potential(xc: float, yc: float, eps: float = 0.18) -> np.ndarray:
    """Simple softened 1/r potential for "spacetime well" look."""
    dx = XX - xc
    dy = YY - yc
    rr = np.sqrt(dx * dx + dy * dy + eps * eps)
    return 1.0 / rr


def bh_mask(xc: float, yc: float, r: float) -> np.ndarray:
    """Boolean mask for inside a black hole core."""
    dx = XX - xc
    dy = YY - yc
    return (dx * dx + dy * dy) <= r * r


def bh_photon_ring(xc: float, yc: float, ring_r: float, width: float) -> np.ndarray:
    dx = XX - xc
    dy = YY - yc
    rr = np.sqrt(dx * dx + dy * dy)
    return np.exp(-((rr - ring_r) / width) ** 2)


def orbit_positions(t_norm: float) -> tuple[float, float, float, float]:
    """
    Compute positions (x1,y1,x2,y2) of two black holes for normalized time t_norm in [0,1].
    Early: nearly circular orbit at INITIAL_SEP.
    Towards t=INSPIRAL_FRAC: separation shrinks to FINAL_SEP.
    After inspiral: they basically merge at origin.
    """
    if t_norm >= 1.0:
        t_norm = 1.0

    # Inspiral factor from 0 -> 1 over first INSPIRAL_FRAC of evolution
    if t_norm < INSPIRAL_FRAC:
        f = t_norm / INSPIRAL_FRAC
        sep = INITIAL_SEP + (FINAL_SEP - INITIAL_SEP) * f
    else:
        sep = FINAL_SEP

    # Orbital phase speeds up slightly as they inspiral
    phase = ORBIT_FREQ * (t_norm + 0.4 * t_norm**2) * 2.0 * np.pi

    x1 = -0.5 * sep * np.cos(phase)
    y1 = -0.5 * sep * np.sin(phase)
    x2 = +0.5 * sep * np.cos(phase)
    y2 = +0.5 * sep * np.sin(phase)

    # After inspiral, smoothly snap them toward origin for merger
    if t_norm >= INSPIRAL_FRAC:
        blend = (t_norm - INSPIRAL_FRAC) / (1.0 - INSPIRAL_FRAC + 1e-9)
        blend = np.clip(blend, 0.0, 1.0)
        x1 *= (1.0 - blend)
        y1 *= (1.0 - blend)
        x2 *= (1.0 - blend)
        y2 *= (1.0 - blend)

    return x1, y1, x2, y2


def gravitational_wave_ripples(t_norm: float) -> np.ndarray:
    """
    Simple circular sinusoidal ripples emanating from the origin,
    turned on after GW_START_FRAC.
    """
    if t_norm < GW_START_FRAC:
        return np.zeros_like(R)

    local_t = (t_norm - GW_START_FRAC) / (1.0 - GW_START_FRAC + 1e-9)
    # Radius of main ripple peak
    r0 = GW_SPEED * local_t * EXTENT * 5.0

    # Oscillatory ripple: sin(k (r - r0)) envelope
    phase = GW_FREQ * (R - r0)
    env = np.exp(-GW_DECAY * np.abs(R - r0))
    ripples = env * np.sin(phase)

    # Only outward-going (positive R > small core)
    ripples *= (R > MERGED_BH_CORE_R * 1.4).astype(float)
    # Scale down amplitude
    return 0.18 * ripples


def generate_frame(frame: int) -> np.ndarray:
    t_norm = frame / (N_FRAMES - 1)

    # Background starfield (simple phasey interference pattern)
    base = (
        np.sin(4.0 * XX + 0.7 * t_norm * 10)
        + np.cos(3.6 * YY - 0.9 * t_norm * 10)
        + np.sin(2.3 * XX + 1.9 * YY + 1.3 * t_norm * 10)
    )
    stars = np.clip(base, 0.0, None) ** 1.8
    stars = STARFIELD_INTENSITY * stars / (stars.max() + 1e-9)

    # Two BH positions
    x1, y1, x2, y2 = orbit_positions(t_norm)

    # Potential wells for visual curvature
    pot = 0.9 * soft_potential(x1, y1) + 0.9 * soft_potential(x2, y2)

    # Normalize and invert potential so deeper wells are brighter accretion zones
    pot_norm = pot / (pot.max() + 1e-9)

    # Base image: stars warped by potential
    image = stars + 0.4 * pot_norm

    # Individual BH photon rings + glow
    ring1 = bh_photon_ring(x1, y1, BH_PHOTON_RING_R, BH_RING_WIDTH)
    ring2 = bh_photon_ring(x2, y2, BH_PHOTON_RING_R, BH_RING_WIDTH)

    # Accretion-like glow around each BH
    glow1 = BH_GLOW * np.exp(-((soft_potential(x1, y1) * 0.8) ** -1.15))
    glow2 = BH_GLOW * np.exp(-((soft_potential(x2, y2) * 0.8) ** -1.15))

    image += 1.5 * (ring1 + ring2) + 0.45 * (glow1 + glow2)

    # Merge phase: fade out binary structure, fade in single bigger BH + rings
    if t_norm >= INSPIRAL_FRAC:
        merge_alpha = (t_norm - INSPIRAL_FRAC) / (1.0 - INSPIRAL_FRAC + 1e-9)
        merge_alpha = np.clip(merge_alpha, 0.0, 1.0)

        # Suppress double structure gradually
        image *= (1.0 - 0.7 * merge_alpha)

        # Single central BH rings
        center_ring = bh_photon_ring(0.0, 0.0, MERGED_PHOTON_RING_R, BH_RING_WIDTH * 1.25)
        center_glow = 0.9 * np.exp(-((soft_potential(0.0, 0.0) * 0.72) ** -1.12))

        image += merge_alpha * (2.4 * center_ring + 0.9 * center_glow)

    # Gravitational wave ripples added on top
    gw = gravitational_wave_ripples(t_norm)
    image += gw

    # Apply black-hole cores (hard cut to black)
    core1 = bh_mask(x1, y1, BH_CORE_R)
    core2 = bh_mask(x2, y2, BH_CORE_R)

    if t_norm >= INSPIRAL_FRAC:
        # blended core for merged BH
        merge_alpha = (t_norm - INSPIRAL_FRAC) / (1.0 - INSPIRAL_FRAC + 1e-9)
        merged_core = bh_mask(0.0, 0.0, MERGED_BH_CORE_R)
        core = ((1.0 - merge_alpha) * (core1 | core2)) | (merge_alpha * merged_core > 0)
    else:
        core = core1 | core2

    image[core] = 0.0

    # Clip to [0,1]
    return np.clip(image, 0.0, 1.0)


# =========================
# Matplotlib animation setup
# =========================

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

ax.set_xticks([])
ax.set_yticks([])

im = ax.imshow(
    generate_frame(0),
    extent=(X_MIN, X_MAX, Y_MIN, Y_MAX),
    origin="lower",
    cmap="magma",
    interpolation="bilinear",
)

title = ax.set_title(
    "Colliding Black Holes (stylized)",
    color="white",
    fontsize=11,
    pad=10,
)


def update(frame: int):
    img = generate_frame(frame)
    im.set_data(img)
    # Slight zoom / drift for cinematic feel
    return (im, title)


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