"""Animated visualization of the Doppler effect for waves from a moving source.

Stylized 2D toy model:
- A source moves to the right at a constant speed (less than wave speed).
- Inbound circular wavefronts converge on the source at regular intervals.
- In front of the motion, wavefronts are bunched up (higher frequency / blue-shift).
- Behind the motion, wavefronts are spread out (lower frequency / red-shift).
- Back half emits expanding waves while the leading edge receives incoming waves.
- One (or two) observer points show what they would "hear"/"see".

Run:
    python3 doppler_eff_viz.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc

# =========================
# Physical-style parameters
# =========================

WAVE_SPEED = 1.0          # speed of waves (e.g. sound/light in this toy)
SOURCE_SPEED = 0.55       # source speed (must be < WAVE_SPEED here)
EMIT_PERIOD = 0.25        # time between emitted wavefronts
MAX_RADIUS = 8.0          # when waves are far enough, drop them
INBOUND_START_RADIUS = MAX_RADIUS

# Simulation timing
DT = 0.04                 # time step per frame
N_FRAMES = 550

# Spatial window
X_MIN, X_MAX = -4.5, 6.0
Y_MIN, Y_MAX = -3.0, 3.0

# Source motion
SOURCE_Y = 0.0
SOURCE_X_START = -3.0

# Observer positions
OBSERVER_FRONT = (3.5, 0.0)     # in front of motion
OBSERVER_BACK = (-3.5, 0.0)     # behind the source (optional)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color: tuple[int, int, int]) -> str:
    return '#' + ''.join(f'{min(255,max(0,int(c))):02x}' for c in rgb_color)


def lerp_color(start_hex: str, end_hex: str, frac: float) -> str:
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)
    frac = min(max(frac, 0.0), 1.0)
    interp = tuple(
        int(start_rgb[i] + frac * (end_rgb[i] - start_rgb[i]))
        for i in range(3)
    )
    return rgb_to_hex(interp)


# =========================
# Bookkeeping for wavefronts
# =========================

class Wavefront:
    def __init__(self, t0: float, mode: str, color: str):
        self.t0 = t0
        self.mode = mode  # "inbound" or "outbound"
        self.artist: Arc | None = None
        self.color = color

    def radius(self, t: float) -> float:
        age = t - self.t0
        if self.mode == "inbound":
            return max(0.0, INBOUND_START_RADIUS - WAVE_SPEED * age)
        return max(0.0, WAVE_SPEED * age)

    def alive(self, t: float) -> bool:
        if t < self.t0:
            return False
        r = self.radius(t)
        if self.mode == "inbound":
            return r > 0.0
        return r <= MAX_RADIUS

    def theta_limits(self) -> tuple[float, float]:
        # Split the circle into front (inbound) and rear (outbound) semicircles.
        if self.mode == "inbound":
            return (-90.0, 90.0)
        return (90.0, 270.0)


emitted_waves: list[Wavefront] = []
last_emit_time = 0.0


# =========================
# Matplotlib setup
# =========================

plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(8, 4.8))
fig.patch.set_facecolor("#05050a")
ax.set_facecolor("#05050a")

ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_aspect("equal", adjustable="box")

ax.set_xticks([])
ax.set_yticks([])

title = ax.set_title(
    "Doppler Effect: Moving Source & Wavefronts",
    color="white",
    fontsize=12,
    pad=10,
)

# Source artist
(source_dot,) = ax.plot(
    [], [],
    marker="o",
    markersize=8,
    color="#ffffff",
    label="Source",
)

# Observers
ax.plot(
    OBSERVER_FRONT[0], OBSERVER_FRONT[1],
    marker="o",
    markersize=6,
    color="#6fb6ff",
)
ax.text(
    OBSERVER_FRONT[0],
    OBSERVER_FRONT[1] + 0.35,
    "Observer (higher pitch / blue-shift)",
    color="#6fb6ff",
    fontsize=7,
    ha="center",
)

ax.plot(
    OBSERVER_BACK[0], OBSERVER_BACK[1],
    marker="o",
    markersize=6,
    color="#ff8888",
)
ax.text(
    OBSERVER_BACK[0],
    OBSERVER_BACK[1] + 0.35,
    "Observer (lower pitch / red-shift)",
    color="#ff8888",
    fontsize=7,
    ha="center",
)

# Motion arrow hint
ax.arrow(
    X_MIN + 0.4,
    Y_MIN + 0.5,
    1.0,
    0.0,
    head_width=0.2,
    head_length=0.25,
    linewidth=1.2,
    color="white",
    length_includes_head=True,
)
ax.text(
    X_MIN + 1.9,
    Y_MIN + 0.5,
    "Source motion",
    color="white",
    fontsize=7,
    ha="left",
    va="center",
)

# For legend
ax.legend(loc="upper left", frameon=False, fontsize=7)


# =========================
# Animation update
# =========================

def update(frame: int):
    global last_emit_time, emitted_waves

    t = frame * DT

    # Update source position
    x_s = SOURCE_X_START + SOURCE_SPEED * t
    y_s = SOURCE_Y
    source_dot.set_data([x_s], [y_s])

    # Emit new wavefronts at fixed period
    if t - last_emit_time >= EMIT_PERIOD:
        emitted_waves.append(Wavefront(t, mode="outbound", color="#ff9494"))
        emitted_waves.append(Wavefront(t, mode="inbound", color="#9fcbff"))
        last_emit_time = t

    # Clean up old artists
    alive_waves: list[Wavefront] = []
    for wf in emitted_waves:
        if wf.alive(t):
            alive_waves.append(wf)
        else:
            if wf.artist is not None:
                wf.artist.remove()
    emitted_waves = alive_waves

    # Draw / update circles
    for wf in emitted_waves:
        r = wf.radius(t)
        if r <= 0.0:
            continue
        if wf.artist is None:
            theta1, theta2 = wf.theta_limits()
            arc = Arc(
                (x_s, y_s),
                width=2 * r,
                height=2 * r,
                theta1=theta1,
                theta2=theta2,
                linewidth=1.0 if wf.mode == "inbound" else 0.8,
                color=wf.color,
                alpha=0.9,
            )
            ax.add_patch(arc)
            wf.artist = arc
        wf.artist.center = (x_s, y_s)
        wf.artist.set_width(2 * r)
        wf.artist.set_height(2 * r)

        # Fade inbound as they approach, outbound as they expand.
        if wf.mode == "inbound":
            alpha = max(0.15, r / INBOUND_START_RADIUS)
            color = lerp_color("#9fcbff", "#ffffff", 1.0 - r / INBOUND_START_RADIUS)
        else:
            alpha = max(0.15, 1.0 - r / MAX_RADIUS)
            color = lerp_color("#ffb29e", "#ff5f5f", min(r / MAX_RADIUS, 1.0))
        wf.artist.set_alpha(alpha)
        wf.artist.set_edgecolor(color)

    return (source_dot, title, *[wf.artist for wf in emitted_waves if wf.artist is not None])


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
