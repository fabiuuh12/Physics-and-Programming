"""
Animated 1D quantum wave packet visualization.

Stylized toy model:
- A complex Gaussian wave packet moving left-to-right.
- Shows both the real part of the wavefunction ψ(x, t) and the probability density |ψ|².
- Slight dispersion over time so it "smears out" like a real free particle packet.
- Dark background + glow for a cinematic / Feynman-lecture vibe.

Run:
    python3 quantum_wave_viz.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------------
# Spatial + temporal grid parameters
# ----------------------------------
X_MIN, X_MAX = -20.0, 20.0
N_X = 1500

T_MAX = 40.0          # total "time"
N_FRAMES = 600        # animation frames

# Wave packet parameters
X0 = -8.0             # initial center
SIGMA0 = 2.0          # initial width
K0 = 2.4              # central wavenumber (sets oscillation frequency in space)
V = 0.55              # group velocity (how fast the packet center moves)
DISPERSION = 0.018    # how quickly the packet spreads
PHASE_SPEED = 1.1     # controls how fast the internal ripples move

# Visual tuning
BG_COLOR = "#050510"
PROB_COLOR = "#33bbff"    # probability density glow
REAL_COLOR = "#ffaa33"    # real part of psi(x) curve
ALPHA_FILL = 0.22

# Precompute spatial axis
x = np.linspace(X_MIN, X_MAX, N_X)


def psi_t(x: np.ndarray, t: float) -> np.ndarray:
    """
    Stylized time-evolved Gaussian wave packet.

    Not a strict exact solution; tuned for clarity and aesthetics:
    - Packet center drifts with velocity V.
    - Width slowly increases with time to mimic dispersion.
    - Phase oscillations move with PHASE_SPEED and K0.
    """
    # Time-dependent width to fake dispersion
    sigma_t = np.sqrt(SIGMA0**2 + (DISPERSION * t) ** 2)

    # Center of the packet
    x_c = X0 + V * t

    # Gaussian envelope
    envelope = np.exp(-((x - x_c) ** 2) / (2.0 * sigma_t**2))

    # Oscillatory complex phase (carrier wave)
    phase = K0 * (x - x_c) - PHASE_SPEED * t

    return envelope * np.exp(1j * phase)


# -----------------------
# Matplotlib setup
# -----------------------
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Axis limits
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(-1.2, 1.4)

ax.set_xlabel("Position  x", color="white")
ax.set_ylabel("Wave amplitude / Probability", color="white")
ax.set_title("Quantum Wave Packet (ψ and |ψ|²)", color="white", pad=12)

# Thin horizontal line at y=0 for reference
ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.9)

# Initial data
psi0 = psi_t(x, 0.0)
real0 = psi0.real
prob0 = np.abs(psi0) ** 2
prob0 /= prob0.max() + 1e-12  # normalize peak to 1

# Plot objects
(real_line,) = ax.plot(x, real0, REAL_COLOR, linewidth=1.4, label="Re[ψ(x, t)]")
prob_fill = ax.fill_between(
    x, 0, prob0, color=PROB_COLOR, alpha=ALPHA_FILL, linewidth=0.0
)

# Legend
ax.legend(loc="upper right", frameon=False, labelcolor="white")


def update(frame: int):
    """Update function for each animation frame."""
    # Map frame -> time
    t = (frame / (N_FRAMES - 1)) * T_MAX

    psi = psi_t(x, t)
    real_psi = psi.real
    prob = np.abs(psi) ** 2
    prob /= prob.max() + 1e-12  # keep probability lobe visually consistent

    # Update real part line
    real_line.set_ydata(real_psi)

    # To update fill_between, we need to remove and redraw it
    global prob_fill
    prob_fill.remove()
    prob_fill = ax.fill_between(
        x, 0, prob, color=PROB_COLOR, alpha=ALPHA_FILL, linewidth=0.0
    )

    return real_line, prob_fill


anim = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=20,
    blit=False,
)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()