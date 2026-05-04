from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from policy_eval import greedy_action_index, summarize
from q_learning import decision_reward, run_decision_action
from rendezvous_env import Difficulty, EnvConfig, RendezvousEnv
from rendezvous_sim import norm


DEFAULT_LINEAR_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "simulations" / "q_learning" / "linear_q_policy.json"
)


def state_features(env: RendezvousEnv) -> np.ndarray:
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

    values = np.array(
        [
            1.0,
            np.clip(distance / 300.0, 0.0, 2.0),
            np.clip(rel_radial / 80.0, -2.0, 2.0),
            np.clip(rel_along / 420.0, -2.0, 2.0),
            np.clip(vel_radial / 0.08, -2.0, 2.0),
            np.clip(vel_along / 0.08, -2.0, 2.0),
            np.clip(closing_speed / 0.08, -2.0, 2.0),
            np.clip(relative_speed / 0.20, 0.0, 2.0),
            np.clip(fuel_remaining / env.cfg.max_delta_v, 0.0, 1.0),
            np.clip(time_fraction, 0.0, 1.0),
            np.clip((100.0 - distance) / 100.0, -2.0, 1.0),
            np.clip((20.0 - distance) / 20.0, -2.0, 1.0),
        ],
        dtype=np.float64,
    )
    return values


def q_values(weights: np.ndarray, env: RendezvousEnv) -> np.ndarray:
    return weights @ state_features(env)


def choose_action(weights: np.ndarray, env: RendezvousEnv, rng: random.Random, epsilon: float) -> int:
    if rng.random() < epsilon:
        return rng.randrange(env.n_actions)
    values = q_values(weights, env)
    best_value = float(values.max())
    best_actions = [index for index, value in enumerate(values) if value == best_value]
    return rng.choice(best_actions)


def warm_start_linear(
    weights: np.ndarray,
    *,
    difficulty: Difficulty,
    seed: int,
    scenarios: int,
    passes: int,
    alpha: float,
) -> int:
    env = RendezvousEnv(EnvConfig())
    updates = 0
    for _ in range(passes):
        for scenario_index in range(scenarios):
            env.reset(randomize=True, seed=seed + scenario_index, difficulty=difficulty)
            done = False
            while not done:
                features = state_features(env)
                greedy_index = greedy_action_index(env)
                values = weights @ features
                for action_index in range(env.n_actions):
                    target = 2.0 if action_index == greedy_index else -1.0
                    error = target - values[action_index]
                    weights[action_index] += alpha * error * features
                    updates += 1
                _, done, _ = run_decision_action(env, greedy_index)
    return updates


def train_linear_q(
    *,
    episodes: int,
    seed: int,
    difficulty: Difficulty,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    warm_start_scenarios: int,
    warm_start_passes: int,
    warm_start_alpha: float,
    reward_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    env = RendezvousEnv(EnvConfig())
    weights = np.zeros((env.n_actions, len(state_features(env))), dtype=np.float64)
    warm_start_updates = warm_start_linear(
        weights,
        difficulty=difficulty,
        seed=seed,
        scenarios=warm_start_scenarios,
        passes=warm_start_passes,
        alpha=warm_start_alpha,
    )

    successes = 0
    best_distance = float("inf")
    best_episode = 0

    for episode in range(episodes):
        env.reset(randomize=True, seed=seed + episode, difficulty=difficulty)
        fraction = episode / max(1, episodes - 1)
        epsilon = epsilon_start * (1.0 - fraction) + epsilon_end * fraction
        done = False
        last_info: dict[str, Any] = {}

        while not done:
            features = state_features(env)
            action_index = choose_action(weights, env, rng, epsilon)
            start_distance = norm(env.chaser_r - env.target_r)
            start_relative_speed = norm(env.chaser_v - env.target_v)
            _, done, info = run_decision_action(env, action_index)
            reward = decision_reward(start_distance, start_relative_speed, action_index, env, done, info)
            scaled_reward = reward_scale * reward
            target = scaled_reward if done else scaled_reward + gamma * float(q_values(weights, env).max())
            error = target - float(weights[action_index] @ features)
            weights[action_index] += alpha * error * features
            last_info = info

        final_distance = float(last_info.get("distance_km", float("inf")))
        if final_distance < best_distance:
            best_distance = final_distance
            best_episode = episode + 1
        if last_info.get("success", False):
            successes += 1

    metadata = {
        "episodes": episodes,
        "seed": seed,
        "difficulty": difficulty,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "warm_start_scenarios": warm_start_scenarios,
        "warm_start_passes": warm_start_passes,
        "warm_start_alpha": warm_start_alpha,
        "reward_scale": reward_scale,
        "warm_start_updates": warm_start_updates,
        "training_successes": successes,
        "best_distance_km": best_distance,
        "best_episode": best_episode,
        "actions": list(env.action_names),
    }
    return weights, metadata


def run_linear_episode(weights: np.ndarray, seed: int, *, difficulty: Difficulty) -> dict[str, Any]:
    env = RendezvousEnv(EnvConfig())
    env.reset(randomize=True, seed=seed, difficulty=difficulty)
    done = False
    total_reward = 0.0
    decisions = 0
    info: dict[str, Any] = {}

    while not done:
        action_index = int(np.argmax(q_values(weights, env)))
        reward, done, info = run_decision_action(env, action_index)
        total_reward += reward
        decisions += 1

    info["total_reward"] = total_reward
    info["decisions"] = decisions
    return info


def evaluate_linear_policy(
    weights: np.ndarray,
    *,
    difficulty: Difficulty,
    seed: int,
    episodes: int,
) -> list[dict[str, Any]]:
    return [run_linear_episode(weights, seed + index, difficulty=difficulty) for index in range(episodes)]


def save_policy(weights: np.ndarray, metadata: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "weights": weights.tolist()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_policy(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["weights"], dtype=np.float64), payload["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear Q approximator for rendezvous.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--difficulty", choices=["easy", "medium", "full"], default="medium")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--epsilon-start", type=float, default=0.6)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--warm-start-scenarios", type=int, default=24)
    parser.add_argument("--warm-start-passes", type=int, default=3)
    parser.add_argument("--warm-start-alpha", type=float, default=0.02)
    parser.add_argument("--reward-scale", type=float, default=0.02)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_LINEAR_POLICY_PATH)
    args = parser.parse_args()

    weights, metadata = train_linear_q(
        episodes=args.episodes,
        seed=args.seed,
        difficulty=args.difficulty,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        warm_start_scenarios=args.warm_start_scenarios,
        warm_start_passes=args.warm_start_passes,
        warm_start_alpha=args.warm_start_alpha,
        reward_scale=args.reward_scale,
    )
    runs = evaluate_linear_policy(
        weights,
        difficulty=args.difficulty,
        seed=args.seed + 10_000,
        episodes=args.eval_episodes,
    )
    save_policy(weights, metadata, args.output)

    print("Linear Q rendezvous agent")
    print(f"episodes: {metadata['episodes']}")
    print(f"training successes: {metadata['training_successes']}")
    print(f"best training distance: {metadata['best_distance_km']:.2f} km on episode {metadata['best_episode']}")
    print(summarize("linear", runs))
    print(f"policy written to: {args.output}")


if __name__ == "__main__":
    main()
