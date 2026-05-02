from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from q_learning import DEFAULT_FIXED_POLICY_PATH, DEFAULT_RANDOMIZED_POLICY_PATH, load_policy, q_policy_action_index
from rendezvous_env import EnvConfig, RendezvousEnv
from rendezvous_sim import EARTH_RADIUS, SimConfig, choose_action


ACTION_COLORS = {
    "coast": "#8a8f98",
    "prograde": "#31c48d",
    "retrograde": "#f05252",
    "radial_out": "#f59e0b",
    "radial_in": "#3f83f8",
}


EpisodeResult = dict[str, np.ndarray | list[str] | list[bool]]


def greedy_action_index(env: RendezvousEnv) -> int:
    action = choose_action(
        env.chaser_r.copy(),
        env.chaser_v.copy(),
        env.target_r.copy(),
        env.target_v.copy(),
        env.cfg.sim,
    )
    return env.action_names.index(action)


def select_action(
    env: RendezvousEnv,
    policy: str,
    rng: random.Random,
    q_table: dict[str, list[float]] | None,
) -> int:
    if policy == "greedy":
        return greedy_action_index(env)
    if policy == "random":
        return rng.randrange(env.n_actions)
    if policy == "qlearn":
        if q_table is None:
            raise ValueError("qlearn policy requires a loaded Q table")
        return q_policy_action_index(env, q_table)
    raise ValueError(f"Unknown policy: {policy}")


def record_episode(
    policy: str,
    seed: int,
    q_table: dict[str, list[float]] | None = None,
    *,
    randomized: bool = False,
) -> EpisodeResult:
    env = RendezvousEnv(EnvConfig())
    env.reset(randomize=randomized, seed=seed)
    rng = random.Random(seed)
    decision_steps = max(1, int(env.cfg.sim.decision_interval / env.cfg.sim.dt))

    chaser_positions: list[np.ndarray] = [env.chaser_r.copy()]
    target_positions: list[np.ndarray] = [env.target_r.copy()]
    distances: list[float] = [env.previous_distance]
    relative_speeds: list[float] = [float(np.linalg.norm(env.chaser_v - env.target_v))]
    fuel_used: list[float] = [env.fuel_delta_v]
    rewards: list[float] = [0.0]
    times: list[float] = [0.0]
    actions: list[str] = ["coast"]
    success: list[bool] = [False]

    done = False
    step = 0
    action_index = env.action_names.index("coast")
    while not done:
        if step % decision_steps == 0:
            action_index = select_action(env, policy, rng, q_table)

        _, reward, done, info = env.step(action_index)
        step += 1

        chaser_positions.append(env.chaser_r.copy())
        target_positions.append(env.target_r.copy())
        distances.append(info["distance_km"])
        relative_speeds.append(info["relative_speed_km_s"])
        fuel_used.append(info["fuel_delta_v_km_s"])
        rewards.append(reward)
        times.append(info["elapsed_s"] / 60.0)
        actions.append(info["action"])
        success.append(info["success"])

    return {
        "chaser_positions": np.vstack(chaser_positions),
        "target_positions": np.vstack(target_positions),
        "distances": np.array(distances),
        "relative_speeds": np.array(relative_speeds),
        "fuel_used": np.array(fuel_used),
        "rewards": np.array(rewards),
        "times": np.array(times),
        "actions": actions,
        "success": success,
    }


def get_arrays(result: EpisodeResult) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[bool],
]:
    chaser_positions = result["chaser_positions"]
    target_positions = result["target_positions"]
    distances = result["distances"]
    relative_speeds = result["relative_speeds"]
    fuel_used = result["fuel_used"]
    rewards = result["rewards"]
    times = result["times"]
    actions = result["actions"]
    success = result["success"]

    assert isinstance(chaser_positions, np.ndarray)
    assert isinstance(target_positions, np.ndarray)
    assert isinstance(distances, np.ndarray)
    assert isinstance(relative_speeds, np.ndarray)
    assert isinstance(fuel_used, np.ndarray)
    assert isinstance(rewards, np.ndarray)
    assert isinstance(times, np.ndarray)
    assert isinstance(actions, list)
    assert isinstance(success, list)

    return chaser_positions, target_positions, distances, relative_speeds, fuel_used, rewards, times, actions, success


def save_episode_gif(result: EpisodeResult, output_path: Path, policy: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (
        chaser_positions,
        target_positions,
        distances,
        relative_speeds,
        fuel_used,
        rewards,
        times,
        actions,
        success,
    ) = get_arrays(result)

    frame_stride = 4
    frame_indices = list(range(0, len(times), frame_stride))
    if frame_indices[-1] != len(times) - 1:
        frame_indices.append(len(times) - 1)

    fig, (orbit_ax, telemetry_ax) = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    fig.patch.set_facecolor("#111827")

    for axis in (orbit_ax, telemetry_ax):
        axis.set_facecolor("#0b1020")
        axis.tick_params(colors="#d1d5db")
        for spine in axis.spines.values():
            spine.set_color("#374151")

    orbit_ax.set_aspect("equal", adjustable="box")
    orbit_ax.set_title(f"{policy.title()} Rendezvous Episode", color="#f9fafb", fontsize=13)
    orbit_ax.set_xlabel("x position (km)", color="#d1d5db")
    orbit_ax.set_ylabel("y position (km)", color="#d1d5db")

    all_positions = np.vstack([chaser_positions, target_positions])
    center = all_positions.mean(axis=0)
    span = np.ptp(all_positions, axis=0).max()
    half_width = max(360.0, span * 0.55)
    orbit_ax.set_xlim(center[0] - half_width, center[0] + half_width)
    orbit_ax.set_ylim(center[1] - half_width, center[1] + half_width)

    earth = plt.Circle((0, 0), EARTH_RADIUS, color="#2563eb", alpha=0.18)
    orbit_ax.add_patch(earth)

    target_path, = orbit_ax.plot([], [], color="#e5e7eb", linewidth=1.8, label="target path")
    chaser_path, = orbit_ax.plot([], [], color="#38bdf8", linewidth=2.0, label="chaser path")
    target_marker = orbit_ax.scatter([], [], s=55, color="#f9fafb", edgecolor="#111827", zorder=4)
    chaser_marker = orbit_ax.scatter([], [], s=65, color="#38bdf8", edgecolor="#111827", zorder=5)
    thrust_line, = orbit_ax.plot([], [], color="#f59e0b", linewidth=2.5, alpha=0.9)
    orbit_ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#f9fafb", loc="upper left")

    zoom_ax = inset_axes(orbit_ax, width="37%", height="37%", loc="lower right", borderpad=1.2)
    zoom_ax.set_facecolor("#111827")
    zoom_ax.set_aspect("equal", adjustable="box")
    zoom_ax.tick_params(colors="#d1d5db", labelsize=7)
    for spine in zoom_ax.spines.values():
        spine.set_color("#64748b")
    zoom_ax.set_title("final approach", color="#f9fafb", fontsize=8)
    zoom_target_path, = zoom_ax.plot([], [], color="#e5e7eb", linewidth=1.3)
    zoom_chaser_path, = zoom_ax.plot([], [], color="#38bdf8", linewidth=1.5)
    zoom_target_marker = zoom_ax.scatter([], [], s=28, color="#f9fafb", edgecolor="#111827", zorder=4)
    zoom_chaser_marker = zoom_ax.scatter([], [], s=34, color="#38bdf8", edgecolor="#111827", zorder=5)

    telemetry_ax.set_title("Episode Telemetry", color="#f9fafb", fontsize=13)
    telemetry_ax.set_xlabel("time (min)", color="#d1d5db")
    telemetry_ax.set_ylabel("distance (km)", color="#d1d5db")
    telemetry_ax.set_xlim(float(times[0]), float(times[-1]))
    telemetry_ax.set_ylim(0.0, max(float(distances.max()) * 1.08, 10.0))
    telemetry_ax.grid(color="#1f2937", linewidth=0.8)
    distance_line, = telemetry_ax.plot([], [], color="#38bdf8", linewidth=2.0, label="distance")
    cursor_line = telemetry_ax.axvline(0.0, color="#f59e0b", linewidth=1.5)
    success_line = telemetry_ax.axhline(
        SimConfig().success_distance,
        color="#31c48d",
        linestyle="--",
        linewidth=1.2,
        label="success distance",
    )
    success_line.set_alpha(0.85)
    telemetry_ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#f9fafb", loc="upper right")

    status_text = fig.text(
        0.52,
        0.04,
        "",
        color="#f9fafb",
        fontsize=10,
        family="monospace",
        ha="left",
        va="bottom",
    )

    def update(frame_index: int) -> tuple[Any, ...]:
        action = actions[frame_index]
        color = ACTION_COLORS[action]

        target_path.set_data(target_positions[: frame_index + 1, 0], target_positions[: frame_index + 1, 1])
        chaser_path.set_data(chaser_positions[: frame_index + 1, 0], chaser_positions[: frame_index + 1, 1])
        target_marker.set_offsets(target_positions[frame_index])
        chaser_marker.set_offsets(chaser_positions[frame_index])
        chaser_marker.set_color(color if action != "coast" else "#38bdf8")

        if action == "coast":
            thrust_line.set_data([], [])
        else:
            direction = chaser_positions[frame_index] - chaser_positions[max(0, frame_index - 1)]
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 0.0:
                unit = direction / direction_norm
                start = chaser_positions[frame_index]
                end = start - unit * 80.0
                thrust_line.set_data([start[0], end[0]], [start[1], end[1]])
                thrust_line.set_color(color)

        zoom_start = max(0, frame_index - 80)
        zoom_points = np.vstack(
            [
                target_positions[zoom_start : frame_index + 1],
                chaser_positions[zoom_start : frame_index + 1],
            ]
        )
        zoom_center = zoom_points.mean(axis=0)
        zoom_span = max(float(np.ptp(zoom_points, axis=0).max()), 20.0)
        zoom_half_width = max(20.0, zoom_span * 0.65)
        zoom_ax.set_xlim(zoom_center[0] - zoom_half_width, zoom_center[0] + zoom_half_width)
        zoom_ax.set_ylim(zoom_center[1] - zoom_half_width, zoom_center[1] + zoom_half_width)
        zoom_target_path.set_data(target_positions[zoom_start : frame_index + 1, 0], target_positions[zoom_start : frame_index + 1, 1])
        zoom_chaser_path.set_data(chaser_positions[zoom_start : frame_index + 1, 0], chaser_positions[zoom_start : frame_index + 1, 1])
        zoom_target_marker.set_offsets(target_positions[frame_index])
        zoom_chaser_marker.set_offsets(chaser_positions[frame_index])
        zoom_chaser_marker.set_color(color if action != "coast" else "#38bdf8")

        distance_line.set_data(times[: frame_index + 1], distances[: frame_index + 1])
        cursor_line.set_xdata([times[frame_index], times[frame_index]])

        status = "SUCCESS" if success[frame_index] else "RUNNING"
        status_text.set_text(
            f"time={times[frame_index]:6.1f} min   action={action:10s}   status={status}\n"
            f"distance={distances[frame_index]:7.2f} km   "
            f"rel_speed={relative_speeds[frame_index]:.4f} km/s   "
            f"delta_v={fuel_used[frame_index] * 1000.0:6.2f} m/s   "
            f"reward={rewards[frame_index]:7.2f}"
        )

        return (
            target_path,
            chaser_path,
            target_marker,
            chaser_marker,
            thrust_line,
            zoom_target_path,
            zoom_chaser_path,
            zoom_target_marker,
            zoom_chaser_marker,
            distance_line,
            cursor_line,
            status_text,
        )

    animation = FuncAnimation(fig, update, frames=frame_indices, interval=70, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=14))
    plt.close(fig)


def default_output_path(policy: str, randomized: bool, seed: int) -> Path:
    suffix = f"randomized_seed_{seed}" if randomized else "rendezvous"
    return Path(__file__).resolve().parents[1] / "simulations" / "episode_viz" / f"{policy}_{suffix}.gif"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a rendezvous policy replay as an animated GIF.")
    parser.add_argument("--policy", choices=["greedy", "random", "qlearn"], default="greedy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--q-policy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    q_table = None
    if args.policy == "qlearn":
        q_policy_path = args.q_policy or (
            DEFAULT_RANDOMIZED_POLICY_PATH if args.randomized else DEFAULT_FIXED_POLICY_PATH
        )
        q_table, metadata = load_policy(q_policy_path)
        print(
            "loaded q-learning policy:"
            f" path={q_policy_path}"
            f" episodes={metadata['episodes']}"
            f" training_successes={metadata['successes']}"
            f" states={metadata['states']}"
        )

    result = record_episode(args.policy, args.seed, q_table, randomized=args.randomized)
    output_path = args.output or default_output_path(args.policy, args.randomized, args.seed)
    title_policy = f"{args.policy} randomized" if args.randomized else args.policy
    save_episode_gif(result, output_path, title_policy)

    (
        _,
        _,
        distances,
        relative_speeds,
        fuel_used,
        _,
        times,
        actions,
        success,
    ) = get_arrays(result)

    print(f"{args.policy.title()} rendezvous episode visualization")
    print(f"success: {success[-1]}")
    print(f"simulated time: {times[-1]:.1f} min")
    print(f"final distance: {distances[-1]:.2f} km")
    print(f"final relative speed: {relative_speeds[-1]:.4f} km/s")
    print(f"fuel proxy, delta-v used: {fuel_used[-1] * 1000.0:.2f} m/s")
    print(f"action counts: {dict(Counter(actions))}")
    print(f"animation written to: {output_path}")


if __name__ == "__main__":
    main()
