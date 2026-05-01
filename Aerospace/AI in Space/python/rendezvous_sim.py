from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MU_EARTH = 398_600.4418  # km^3 / s^2
EARTH_RADIUS = 6_378.137  # km


@dataclass(frozen=True)
class SimConfig:
    dt: float = 10.0
    duration: float = 7_200.0
    decision_interval: float = 60.0
    lookahead: float = 900.0
    max_thrust_accel: float = 2.0e-5  # km / s^2, equal to 0.02 m / s^2
    fuel_cost_per_burn: float = 0.0012
    success_distance: float = 5.0
    success_relative_speed: float = 0.020


ACTIONS = (
    "coast",
    "prograde",
    "retrograde",
    "radial_out",
    "radial_in",
)


def norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def gravity(position: np.ndarray) -> np.ndarray:
    radius = norm(position)
    return -MU_EARTH * position / radius**3


def thrust_direction(position: np.ndarray, velocity: np.ndarray, action: str) -> np.ndarray:
    if action == "coast":
        return np.zeros(2)

    radial = position / norm(position)
    prograde = velocity / norm(velocity)

    if action == "prograde":
        return prograde
    if action == "retrograde":
        return -prograde
    if action == "radial_out":
        return radial
    if action == "radial_in":
        return -radial

    raise ValueError(f"Unknown action: {action}")


def acceleration(position: np.ndarray, velocity: np.ndarray, action: str, cfg: SimConfig) -> np.ndarray:
    return gravity(position) + cfg.max_thrust_accel * thrust_direction(position, velocity, action)


def rk4_step(position: np.ndarray, velocity: np.ndarray, action: str, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    def deriv(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return vel, acceleration(pos, vel, action, cfg)

    dt = cfg.dt
    k1_r, k1_v = deriv(position, velocity)
    k2_r, k2_v = deriv(position + 0.5 * dt * k1_r, velocity + 0.5 * dt * k1_v)
    k3_r, k3_v = deriv(position + 0.5 * dt * k2_r, velocity + 0.5 * dt * k2_v)
    k4_r, k4_v = deriv(position + dt * k3_r, velocity + dt * k3_v)

    next_position = position + dt * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r) / 6.0
    next_velocity = velocity + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    return next_position, next_velocity


def circular_state(radius_km: float, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    speed = math.sqrt(MU_EARTH / radius_km)
    position = radius_km * np.array([math.cos(angle_rad), math.sin(angle_rad)])
    velocity = speed * np.array([-math.sin(angle_rad), math.cos(angle_rad)])
    return position, velocity


def rollout_cost(
    chaser_r: np.ndarray,
    chaser_v: np.ndarray,
    target_r: np.ndarray,
    target_v: np.ndarray,
    action: str,
    cfg: SimConfig,
) -> float:
    steps = int(cfg.lookahead / cfg.dt)
    min_distance = float("inf")

    for _ in range(steps):
        chaser_r, chaser_v = rk4_step(chaser_r, chaser_v, action, cfg)
        target_r, target_v = rk4_step(target_r, target_v, "coast", cfg)
        min_distance = min(min_distance, norm(chaser_r - target_r))

    distance = norm(chaser_r - target_r)
    relative_speed = norm(chaser_v - target_v)
    fuel_penalty = 0.0 if action == "coast" else cfg.fuel_cost_per_burn * cfg.lookahead

    return distance + 900.0 * relative_speed + 0.2 * min_distance + fuel_penalty


def choose_action(
    chaser_r: np.ndarray,
    chaser_v: np.ndarray,
    target_r: np.ndarray,
    target_v: np.ndarray,
    cfg: SimConfig,
) -> str:
    costs = {
        action: rollout_cost(chaser_r.copy(), chaser_v.copy(), target_r.copy(), target_v.copy(), action, cfg)
        for action in ACTIONS
    }
    return min(costs, key=costs.get)


def run_simulation(cfg: SimConfig) -> dict[str, np.ndarray | list[str]]:
    target_r, target_v = circular_state(EARTH_RADIUS + 500.0, 0.0)
    chaser_r, chaser_v = circular_state(EARTH_RADIUS + 485.0, -0.045)

    steps = int(cfg.duration / cfg.dt)
    decision_steps = max(1, int(cfg.decision_interval / cfg.dt))

    times: list[float] = []
    distances: list[float] = []
    relative_speeds: list[float] = []
    fuel_used: list[float] = []
    chaser_positions: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    actions: list[str] = []

    current_action = "coast"
    fuel = 0.0

    for step in range(steps + 1):
        if step % decision_steps == 0:
            current_action = choose_action(chaser_r, chaser_v, target_r, target_v, cfg)

        times.append(step * cfg.dt / 60.0)
        distances.append(norm(chaser_r - target_r))
        relative_speeds.append(norm(chaser_v - target_v))
        fuel_used.append(fuel)
        chaser_positions.append(chaser_r.copy())
        target_positions.append(target_r.copy())
        actions.append(current_action)

        if (
            distances[-1] <= cfg.success_distance
            and relative_speeds[-1] <= cfg.success_relative_speed
        ):
            break

        if current_action != "coast":
            fuel += cfg.max_thrust_accel * cfg.dt

        chaser_r, chaser_v = rk4_step(chaser_r, chaser_v, current_action, cfg)
        target_r, target_v = rk4_step(target_r, target_v, "coast", cfg)

    return {
        "times": np.array(times),
        "distances": np.array(distances),
        "relative_speeds": np.array(relative_speeds),
        "fuel_used": np.array(fuel_used),
        "chaser_positions": np.vstack(chaser_positions),
        "target_positions": np.vstack(target_positions),
        "actions": actions,
    }


def save_plots(result: dict[str, np.ndarray | list[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    times = result["times"]
    distances = result["distances"]
    relative_speeds = result["relative_speeds"]
    chaser_positions = result["chaser_positions"]
    target_positions = result["target_positions"]

    assert isinstance(times, np.ndarray)
    assert isinstance(distances, np.ndarray)
    assert isinstance(relative_speeds, np.ndarray)
    assert isinstance(chaser_positions, np.ndarray)
    assert isinstance(target_positions, np.ndarray)

    plt.figure(figsize=(8, 8))
    earth = plt.Circle((0, 0), EARTH_RADIUS, color="#1f77b4", alpha=0.25)
    axis = plt.gca()
    axis.add_patch(earth)
    plt.plot(target_positions[:, 0], target_positions[:, 1], label="target", linewidth=1.8)
    plt.plot(chaser_positions[:, 0], chaser_positions[:, 1], label="AI chaser", linewidth=1.8)
    plt.scatter(target_positions[-1, 0], target_positions[-1, 1], s=30)
    plt.scatter(chaser_positions[-1, 0], chaser_positions[-1, 1], s=30)
    plt.axis("equal")
    plt.xlabel("x position (km)")
    plt.ylabel("y position (km)")
    plt.title("Autonomous orbital rendezvous attempt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(times, distances)
    plt.xlabel("time (min)")
    plt.ylabel("relative distance (km)")
    plt.title("Chaser distance to target")
    plt.tight_layout()
    plt.savefig(output_dir / "distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(times, relative_speeds)
    plt.xlabel("time (min)")
    plt.ylabel("relative speed (km/s)")
    plt.title("Relative speed to target")
    plt.tight_layout()
    plt.savefig(output_dir / "relative_speed.png", dpi=180)
    plt.close()


def main() -> None:
    cfg = SimConfig()
    result = run_simulation(cfg)
    output_dir = Path(__file__).resolve().parents[1] / "simulations" / "first_rendezvous"
    save_plots(result, output_dir)

    actions = result["actions"]
    distances = result["distances"]
    relative_speeds = result["relative_speeds"]
    fuel_used = result["fuel_used"]
    times = result["times"]

    assert isinstance(actions, list)
    assert isinstance(distances, np.ndarray)
    assert isinstance(relative_speeds, np.ndarray)
    assert isinstance(fuel_used, np.ndarray)
    assert isinstance(times, np.ndarray)

    action_counts = {action: actions.count(action) for action in ACTIONS}
    success = distances[-1] <= cfg.success_distance and relative_speeds[-1] <= cfg.success_relative_speed

    print("Autonomous orbital rendezvous simulation")
    print(f"success: {success}")
    print(f"simulated time: {times[-1]:.1f} min")
    print(f"initial distance: {distances[0]:.2f} km")
    print(f"final distance: {distances[-1]:.2f} km")
    print(f"minimum distance: {distances.min():.2f} km")
    print(f"final relative speed: {relative_speeds[-1]:.4f} km/s")
    print(f"fuel proxy, delta-v used: {fuel_used[-1] * 1000.0:.2f} m/s")
    print(f"action counts: {action_counts}")
    print(f"plots written to: {output_dir}")


if __name__ == "__main__":
    main()
