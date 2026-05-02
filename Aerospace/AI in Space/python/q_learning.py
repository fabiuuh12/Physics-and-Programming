from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from rendezvous_env import Difficulty, EnvConfig, RendezvousEnv
from rendezvous_sim import choose_action, norm


DEFAULT_FIXED_POLICY_PATH = Path(__file__).resolve().parents[1] / "simulations" / "q_learning" / "q_policy_fixed.json"
DEFAULT_RANDOMIZED_POLICY_PATH = Path(__file__).resolve().parents[1] / "simulations" / "q_learning" / "q_policy_randomized.json"
DEFAULT_POLICY_PATH = DEFAULT_FIXED_POLICY_PATH


def default_policy_path(randomized: bool, difficulty: Difficulty) -> Path:
    if not randomized:
        return DEFAULT_FIXED_POLICY_PATH
    if difficulty == "full":
        return DEFAULT_RANDOMIZED_POLICY_PATH
    return Path(__file__).resolve().parents[1] / "simulations" / "q_learning" / f"q_policy_{difficulty}.json"


def bucket(value: float, edges: list[float]) -> int:
    return int(np.digitize([value], edges)[0])


def discretize_state(env: RendezvousEnv) -> str:
    relative_r = env.chaser_r - env.target_r
    relative_v = env.chaser_v - env.target_v
    radial = env.target_r / norm(env.target_r)
    along_track = env.target_v / norm(env.target_v)
    distance = norm(relative_r)
    line_of_sight = relative_r / max(distance, 1.0e-9)

    rel_radial = float(np.dot(relative_r, radial))
    rel_along = float(np.dot(relative_r, along_track))
    vel_radial = float(np.dot(relative_v, radial))
    vel_along = float(np.dot(relative_v, along_track))
    closing_speed = -float(np.dot(relative_v, line_of_sight))
    relative_speed = norm(relative_v)
    fuel_remaining = max(0.0, env.cfg.max_delta_v - env.fuel_delta_v)
    time_fraction = env.elapsed / env.cfg.sim.duration

    state = (
        bucket(distance, [5, 10, 20, 40, 80, 140, 220, 340, 520]),
        bucket(rel_radial, [-60, -30, -15, -5, 0, 5, 15, 30, 60]),
        bucket(rel_along, [-420, -260, -150, -80, -35, -10, 0, 10, 35, 80, 150, 260, 420]),
        bucket(vel_radial, [-0.06, -0.03, -0.015, -0.005, 0.0, 0.005, 0.015, 0.03, 0.06]),
        bucket(vel_along, [-0.06, -0.03, -0.015, -0.005, 0.0, 0.005, 0.015, 0.03, 0.06]),
        bucket(closing_speed, [-0.04, -0.02, -0.005, 0.0, 0.005, 0.015, 0.03, 0.06]),
        bucket(relative_speed, [0.015, 0.02, 0.035, 0.06, 0.10, 0.16, 0.25]),
        bucket(fuel_remaining, [0.02, 0.05, 0.08, 0.11]),
        bucket(time_fraction, [0.35, 0.7, 0.9]),
    )
    return "|".join(str(part) for part in state)


def q_values(table: dict[str, list[float]], state: str, n_actions: int) -> list[float]:
    if state not in table:
        table[state] = [0.0] * n_actions
    return table[state]


def choose_q_action(
    table: dict[str, list[float]],
    state: str,
    n_actions: int,
    rng: random.Random,
    epsilon: float,
) -> int:
    if rng.random() < epsilon:
        return rng.randrange(n_actions)

    values = q_values(table, state, n_actions)
    best_value = max(values)
    best_actions = [index for index, value in enumerate(values) if value == best_value]
    return rng.choice(best_actions)


def run_decision_action(env: RendezvousEnv, action_index: int) -> tuple[float, bool, dict[str, Any]]:
    decision_steps = max(1, int(env.cfg.sim.decision_interval / env.cfg.sim.dt))
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    for _ in range(decision_steps):
        _, reward, done, info = env.step(action_index)
        total_reward += reward
        if done:
            break

    return total_reward, done, info


def decision_reward(
    start_distance: float,
    start_relative_speed: float,
    action_index: int,
    env: RendezvousEnv,
    done: bool,
    info: dict[str, Any],
) -> float:
    end_distance = float(info["distance_km"])
    end_relative_speed = float(info["relative_speed_km_s"])
    progress = start_distance - end_distance
    near_factor = max(0.0, min(1.0, (100.0 - end_distance) / 100.0))

    reward = 1.8 * progress
    reward -= 0.015 * end_distance
    reward -= 10.0 * end_relative_speed

    if env.action_names[action_index] != "coast":
        reward -= 0.15

    if progress < -1.0:
        reward += 0.6 * progress
    if end_distance < 40.0 and end_relative_speed <= 0.035:
        reward += 12.0
    if end_distance < 20.0 and end_relative_speed <= 0.025:
        reward += 25.0
    if end_distance < 10.0 and end_relative_speed <= env.cfg.sim.success_relative_speed:
        reward += 45.0

    excess_near_speed = max(0.0, end_relative_speed - env.cfg.sim.success_relative_speed)
    reward -= near_factor * 450.0 * excess_near_speed

    if end_distance < 20.0 and end_relative_speed > start_relative_speed:
        reward -= 8.0
    if info.get("success", False):
        reward += 300.0
    if info.get("unsafe_approach", False):
        reward -= 120.0
    if info.get("fuel_empty", False) and not info.get("success", False):
        reward -= 70.0
    if info.get("timed_out", False) and not info.get("success", False):
        reward -= 55.0 + 0.05 * end_distance
    if done and not info.get("success", False) and end_distance <= env.cfg.sim.success_distance:
        reward -= 80.0

    return reward


def greedy_action_index(env: RendezvousEnv) -> int:
    action = choose_action(
        env.chaser_r.copy(),
        env.chaser_v.copy(),
        env.target_r.copy(),
        env.target_v.copy(),
        env.cfg.sim,
    )
    return env.action_names.index(action)


def warm_start_from_greedy(
    table: dict[str, list[float]],
    env: RendezvousEnv,
    *,
    randomized: bool,
    difficulty: Difficulty,
    seed: int,
    scenarios: int,
) -> int:
    seeded_states = 0
    effective_scenarios = scenarios if randomized else 1

    for scenario_index in range(max(1, effective_scenarios)):
        env.reset(randomize=randomized, seed=seed + scenario_index, difficulty=difficulty)
        done = False

        while not done:
            state = discretize_state(env)
            action_index = greedy_action_index(env)
            values = q_values(table, state, env.n_actions)
            for index in range(env.n_actions):
                values[index] = min(values[index], -25.0)
            values[action_index] = 200.0
            seeded_states += 1
            _, done, _ = run_decision_action(env, action_index)

    return seeded_states


def train_q_learning(
    episodes: int,
    seed: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    warm_start: bool,
    randomized: bool,
    difficulty: Difficulty,
    warm_start_scenarios: int,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    rng = random.Random(seed)
    env = RendezvousEnv(EnvConfig())
    table: dict[str, list[float]] = {}
    seeded_states = (
        warm_start_from_greedy(
            table,
            env,
            randomized=randomized,
            difficulty=difficulty,
            seed=seed,
            scenarios=warm_start_scenarios,
        )
        if warm_start
        else 0
    )

    successes = 0
    best_distance = float("inf")
    best_episode = 0
    last_info: dict[str, Any] = {}

    for episode in range(episodes):
        env.reset(randomize=randomized, seed=seed + episode, difficulty=difficulty)
        fraction = episode / max(1, episodes - 1)
        epsilon = epsilon_start * (1.0 - fraction) + epsilon_end * fraction
        done = False

        while not done:
            state = discretize_state(env)
            action_index = choose_q_action(table, state, env.n_actions, rng, epsilon)
            start_distance = norm(env.chaser_r - env.target_r)
            start_relative_speed = norm(env.chaser_v - env.target_v)
            _, done, info = run_decision_action(env, action_index)
            reward = decision_reward(start_distance, start_relative_speed, action_index, env, done, info)
            next_state = discretize_state(env)

            values = q_values(table, state, env.n_actions)
            next_values = q_values(table, next_state, env.n_actions)
            target = reward if done else reward + gamma * max(next_values)
            values[action_index] += alpha * (target - values[action_index])
            last_info = info

        final_distance = float(last_info.get("distance_km", float("inf")))
        if final_distance < best_distance:
            best_distance = final_distance
            best_episode = episode + 1
        if last_info.get("success", False):
            successes += 1

    refreshed_seeded_states = (
        warm_start_from_greedy(
            table,
            env,
            randomized=randomized,
            difficulty=difficulty,
            seed=seed,
            scenarios=warm_start_scenarios,
        )
        if warm_start
        else 0
    )
    metadata = {
        "episodes": episodes,
        "seed": seed,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "warm_start": warm_start,
        "randomized": randomized,
        "difficulty": difficulty,
        "warm_start_scenarios": warm_start_scenarios,
        "seeded_states": seeded_states,
        "refreshed_seeded_states": refreshed_seeded_states,
        "successes": successes,
        "best_distance_km": best_distance,
        "best_episode": best_episode,
        "states": len(table),
        "actions": list(env.action_names),
    }
    return table, metadata


def save_policy(table: dict[str, list[float]], metadata: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "q_table": table}, indent=2), encoding="utf-8")


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> tuple[dict[str, list[float]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["q_table"], data["metadata"]


def q_policy_action_index(env: RendezvousEnv, table: dict[str, list[float]]) -> int:
    state = discretize_state(env)
    values = q_values(table, state, env.n_actions)
    return int(np.argmax(values))


def evaluate_one_policy(
    table: dict[str, list[float]],
    *,
    randomize: bool,
    seed: int,
    difficulty: Difficulty = "full",
) -> dict[str, Any]:
    env = RendezvousEnv(EnvConfig())
    env.reset(randomize=randomize, seed=seed, difficulty=difficulty)
    done = False
    total_reward = 0.0
    info: dict[str, Any] = {}
    decisions = 0

    while not done:
        action_index = q_policy_action_index(env, table)
        reward, done, info = run_decision_action(env, action_index)
        total_reward += reward
        decisions += 1

    info["total_reward"] = total_reward
    info["decisions"] = decisions
    return info


def evaluate_policy(
    table: dict[str, list[float]],
    *,
    randomize: bool,
    seed: int,
    episodes: int,
    difficulty: Difficulty = "full",
) -> dict[str, Any]:
    runs = [
        evaluate_one_policy(table, randomize=randomize, seed=seed + index, difficulty=difficulty)
        for index in range(episodes)
    ]
    successes = sum(1 for run in runs if run["success"])
    distances = np.array([run["distance_km"] for run in runs], dtype=np.float64)
    speeds = np.array([run["relative_speed_km_s"] for run in runs], dtype=np.float64)
    delta_v = np.array([run["fuel_delta_v_km_s"] * 1000.0 for run in runs], dtype=np.float64)

    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_final_distance_km": float(distances.mean()),
        "median_final_distance_km": float(np.median(distances)),
        "best_final_distance_km": float(distances.min()),
        "mean_relative_speed_km_s": float(speeds.mean()),
        "mean_delta_v_m_s": float(delta_v.mean()),
        "first_run": runs[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small tabular Q-learning rendezvous agent.")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--epsilon-start", type=float, default=0.9)
    parser.add_argument("--epsilon-end", type=float, default=0.04)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--difficulty", choices=["easy", "medium", "full"], default="full")
    parser.add_argument("--warm-start-scenarios", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_path = args.output or default_policy_path(args.randomized, args.difficulty)

    table, metadata = train_q_learning(
        episodes=args.episodes,
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        warm_start=not args.no_warm_start,
        randomized=args.randomized,
        difficulty=args.difficulty,
        warm_start_scenarios=args.warm_start_scenarios,
    )
    save_policy(table, metadata, output_path)
    evaluation = evaluate_policy(
        table,
        randomize=args.randomized,
        seed=args.seed + 10_000,
        episodes=args.eval_episodes,
        difficulty=args.difficulty,
    )

    print("Tabular Q-learning rendezvous agent")
    print(f"episodes: {metadata['episodes']}")
    print(
        "warm start:"
        f" {metadata['warm_start']}"
        f" randomized={metadata['randomized']}"
        f" difficulty={metadata['difficulty']}"
        f" warm_start_scenarios={metadata['warm_start_scenarios']}"
        f" seeded_states={metadata['seeded_states']}"
        f" refreshed_seeded_states={metadata['refreshed_seeded_states']}"
    )
    print(f"training successes: {metadata['successes']}")
    print(f"states learned: {metadata['states']}")
    print(f"best training distance: {metadata['best_distance_km']:.2f} km on episode {metadata['best_episode']}")
    print(
        "evaluation:"
        f" {evaluation['successes']}/{evaluation['episodes']} success"
        f" ({evaluation['success_rate'] * 100.0:.1f}%)"
    )
    print(f"mean final distance: {evaluation['mean_final_distance_km']:.2f} km")
    print(f"median final distance: {evaluation['median_final_distance_km']:.2f} km")
    print(f"best final distance: {evaluation['best_final_distance_km']:.2f} km")
    print(f"mean relative speed: {evaluation['mean_relative_speed_km_s']:.4f} km/s")
    print(f"mean delta-v: {evaluation['mean_delta_v_m_s']:.2f} m/s")
    print(f"policy written to: {output_path}")


if __name__ == "__main__":
    main()
