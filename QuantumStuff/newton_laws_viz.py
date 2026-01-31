"""Newton's three fundamental laws illustrated with simple animations.

Panels:
1. Law I – An ice puck glides at constant velocity when no net force acts.
2. Law II – A thruster applies force F(t), and the cart's acceleration matches a = F/m.
3. Law III – Two carts push off each other; impulses are equal and opposite.

Run:
    python3 newton_laws_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle


DT = 0.04
N_FRAMES = 400

plt.style.use("dark_background")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#05070f")

axes[0].set_title("Law I: inertia (ΣF = 0 ⇒ v constant)")
axes[1].set_title("Law II: ΣF = m a")
axes[2].set_title("Law III: equal & opposite forces")

# -----------------
# Law I components
# -----------------
ax1 = axes[0]
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 2)
puck = Rectangle((0.5, 0.8), 0.5, 0.4, color="#7de8ff")
ax1.add_patch(puck)
trail, = ax1.plot([], [], color="#6bd8ff", alpha=0.7)
inertia_text = ax1.text(0.3, 1.7, "No net force → motion stays uniform", color="#dcecff", fontsize=9)
force_arrow = FancyArrowPatch(
    (2.2, 0.6),
    (2.2, 0.6),
    arrowstyle="->",
    linewidth=0.0,
    color="#ff9d9d",
)
ax1.add_patch(force_arrow)
positions_x = []

# -----------------
# Law II components
# -----------------
ax2 = axes[1]
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 2)
cart = Rectangle((1.5, 0.6), 1.1, 0.6, color="#ffa77a")
ax2.add_patch(cart)
thruster = FancyArrowPatch((1.4, 0.9), (1.4, 0.9), arrowstyle="-|>", color="#ffd29c", linewidth=2.0)
ax2.add_patch(thruster)
acc_text = ax2.text(0.2, 1.7, "", color="#fff4d6", fontsize=9)
cart_vel = 0.0
cart_pos = 1.5
mass = 2.0

# -----------------
# Law III components
# -----------------
ax3 = axes[2]
ax3.set_xlim(0, 6)
ax3.set_ylim(0, 2)
cart_left = Rectangle((2.2, 1.0), 0.9, 0.4, color="#9ee5ff")
cart_right = Rectangle((2.9, 1.0), 0.9, 0.4, color="#ff9aba")
ax3.add_patch(cart_left)
ax3.add_patch(cart_right)
spring = ax3.plot([], [], color="#ffe897", linewidth=2.0)[0]
action_arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle="<->", color="#f7ff9e", linewidth=1.8)
ax3.add_patch(action_arrow)
law3_text = ax3.text(0.3, 1.7, "Action = −Reaction impulses", color="#f6ffd2", fontsize=9)
left_vel = -0.15
right_vel = 0.15
spring_phase = np.linspace(0, 1, 20)


def update(frame: int):
    global cart_pos, cart_vel, left_vel, right_vel
    t = frame * DT

    # Law I: puck moves at constant velocity except for a brief force pulse.
    v = 0.8
    x = 0.5 + v * t
    if 2.5 < t < 3.0:
        v += 0.3 * np.sin(40 * (t - 2.5))
        x = 0.5 + 0.8 * 2.5 + 0.3 * (1 - np.cos(40 * (t - 2.5))) / 40
        force_arrow.set_positions((x + 0.2, 0.55), (x + 0.2, 1.25))
        force_arrow.set_linewidth(2.0)
    else:
        force_arrow.set_positions((x + 0.2, 0.55), (x + 0.2, 0.55))
        force_arrow.set_linewidth(0.0)
    puck.set_x((x % 7) - 0.5)
    positions_x.append(puck.get_x() + 0.25)
    positions_x[:] = positions_x[-80:]
    trail.set_data(np.array(positions_x), np.ones_like(positions_x) * 1.0)

    # Law II: apply time-varying force to cart
    force = 2.5 * np.sin(0.9 * t) + 1.2
    accel = force / mass
    cart_vel += accel * DT
    cart_pos += cart_vel * DT
    if cart_pos > 4.2:
        cart_pos = 1.5
        cart_vel = 0.0
    cart.set_x(cart_pos)
    thruster.set_positions((cart_pos - 0.05, 0.9), (cart_pos - 0.05 - 0.4 - 0.12 * np.sin(3 * t), 0.9))
    acc_text.set_text(f"Force {force:4.1f} N → a = F/m ≈ {accel:4.2f} m/s²")

    # Law III: carts push off via spring
    compression = 0.3 + 0.15 * np.cos(1.5 * t)
    center = 2.6
    left_x = center - compression
    right_x = center + compression
    cart_left.set_x(left_x - cart_left.get_width())
    cart_right.set_x(right_x)
    spring_x = np.linspace(left_x, right_x, len(spring_phase))
    spring_y = 1.2 + 0.05 * np.sin(10 * spring_phase)
    spring.set_data(spring_x, spring_y)
    left_impulse = -0.8 * np.sin(1.5 * t)
    right_impulse = -left_impulse
    action_arrow.set_positions((left_x + 0.1, 0.6), (right_x + 0.8, 0.6))
    law3_text.set_text(f"Impulse left = {left_impulse:+.2f}, right = {right_impulse:+.2f}")

    return puck, trail, force_arrow, cart, thruster, spring, action_arrow


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
