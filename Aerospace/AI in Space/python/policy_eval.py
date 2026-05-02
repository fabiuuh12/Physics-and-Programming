from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np

from q_learning import default_policy_path, load_policy, q_policy_action_index, run_decision_action
from rendezvous_env import Difficulty, EnvConfig, RendezvousEnv
from rendezvous_sim import choose_action


def greedy_action_index(env: RendezvousEnv) -> int:
    action = choose_action(
        env.chaser_r.copy(),
        env.chaser_v.copy(),
        env.target_r.copy(),
        env.target_v.copy(),
        env.cfg.sim,
    )
    return env.action_names.index(action)


def run_episode(
    policy: str,
    seed: int,
    *,
    difficulty: Difficulty,
    q_table: dict[str, list[float]] | None,
) -> dict[str, Any]:
    env = RendezvousEnv(EnvConfig())
    env.reset(randomize=True, seed=seed, difficulty=difficulty)
    rng = random.Random(seed)
    done = False
    total_reward = 0.0
    decisions = 0
    info: dict[str, Any] = {}

    while not done:
        if policy == "random":
            action_index = rng.randrange(env.n_actions)
        elif policy == "greedy":
            action_index = greedy_action_index(env)
        elif policy == "qlearn":
            if q_table is None:
                raise ValueError("qlearn policy requires q_table")
            action_index = q_policy_action_index(env, q_table)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        reward, done, info = run_decision_action(env, action_index)
        total_reward += reward
        decisions += 1

    info["total_reward"] = total_reward
    info["decisions"] = decisions
    return info


def summarize(policy: str, runs: list[dict[str, Any]]) -> str:
    successes = sum(1 for run in runs if run["success"])
    distances = np.array([run["distance_km"] for run in runs], dtype=np.float64)
    speeds = np.array([run["relative_speed_km_s"] for run in runs], dtype=np.float64)
    delta_v = np.array([run["fuel_delta_v_km_s"] * 1000.0 for run in runs], dtype=np.float64)
    return (
        f"{policy:>7s}: "
        f"success={successes:2d}/{len(runs):2d} "
        f"mean_distance={distances.mean():7.2f} km "
        f"median_distance={np.median(distances):7.2f} km "
        f"best={distances.min():6.2f} km "
        f"mean_speed={speeds.mean():.4f} km/s "
        f"mean_dv={delta_v.mean():6.2f} m/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rendezvous policies on randomized scenarios.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "full"], default="easy")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=10011)
    parser.add_argument("--q-policy", type=Path)
    args = parser.parse_args()

    q_policy_path = args.q_policy or default_policy_path(True, args.difficulty)
    q_table = None
    if q_policy_path.exists():
        q_table, metadata = load_policy(q_policy_path)
        print(
            "loaded q-learning policy:"
            f" path={q_policy_path}"
            f" difficulty={metadata.get('difficulty', 'unknown')}"
            f" episodes={metadata['episodes']}"
        )
    else:
        print(f"q-learning policy missing: {q_policy_path}")

    print(f"Policy evaluation difficulty={args.difficulty} episodes={args.episodes}")
    for policy in ("random", "greedy", "qlearn"):
        if policy == "qlearn" and q_table is None:
            continue
        runs = [
            run_episode(policy, args.seed + index, difficulty=args.difficulty, q_table=q_table)
            for index in range(args.episodes)
        ]
        print(summarize(policy, runs))


if __name__ == "__main__":
    main()
