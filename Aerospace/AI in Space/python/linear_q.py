from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from policy_eval import greedy_action_index, summarize
from q_learning import decision_reward, run_decision_action
from rendezvous_env import Difficulty, EnvConfig, RendezvousEnv
from rendezvous_sim import norm


DEFAULT_LINEAR_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "simulations" / "q_learning" / "linear_q_policy_medium_next.json"
)
LINEAR_GUARD_DISTANCE_MARGIN_KM = 0.25
LINEAR_GUARD_SPEED_MARGIN_KM_S = 0.0015
LINEAR_GREEDY_PRIOR = 0.6
WARM_START_BEST_VALUE = 2.5
WARM_START_GREEDY_VALUE = 1.5
WARM_START_OTHER_VALUE = -1.0


@dataclass(frozen=True)
class LinearDecision:
    action_index: int
    linear_action_index: int
    greedy_action_index: int
    reason: str


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

    distance_scale = np.clip(distance / 300.0, 0.0, 2.0)
    rel_radial_scale = np.clip(rel_radial / 80.0, -2.0, 2.0)
    rel_along_scale = np.clip(rel_along / 420.0, -2.0, 2.0)
    vel_radial_scale = np.clip(vel_radial / 0.08, -2.0, 2.0)
    vel_along_scale = np.clip(vel_along / 0.08, -2.0, 2.0)
    closing_scale = np.clip(closing_speed / 0.08, -2.0, 2.0)
    speed_scale = np.clip(relative_speed / 0.20, 0.0, 2.0)
    fuel_scale = np.clip(fuel_remaining / env.cfg.max_delta_v, 0.0, 1.0)
    time_scale = np.clip(time_fraction, 0.0, 1.0)
    near_100 = np.clip((100.0 - distance) / 100.0, -2.0, 1.0)
    near_20 = np.clip((20.0 - distance) / 20.0, -2.0, 1.0)

    values = np.array(
        [
            1.0,
            distance_scale,
            rel_radial_scale,
            rel_along_scale,
            vel_radial_scale,
            vel_along_scale,
            closing_scale,
            speed_scale,
            fuel_scale,
            time_scale,
            near_100,
            near_20,
            distance_scale * distance_scale,
            rel_radial_scale * vel_radial_scale,
            rel_along_scale * vel_along_scale,
            closing_scale * near_100,
            speed_scale * near_20,
            rel_radial_scale * near_20,
            rel_along_scale * near_20,
            vel_radial_scale * near_20,
            vel_along_scale * near_20,
            fuel_scale * time_scale,
        ],
        dtype=np.float64,
    )
    return values


def q_values(
    weights: np.ndarray,
    env: RendezvousEnv,
    greedy_index: int | None = None,
    *,
    use_greedy_prior: bool = False,
) -> np.ndarray:
    values = weights @ state_features(env)
    if not use_greedy_prior:
        return values

    values = values.copy()
    if greedy_index is None:
        greedy_index = greedy_action_index(env)
    values[greedy_index] += LINEAR_GREEDY_PRIOR
    return values


def choose_action(weights: np.ndarray, env: RendezvousEnv, rng: random.Random, epsilon: float) -> int:
    if rng.random() < epsilon:
        return rng.randrange(env.n_actions)
    values = q_values(weights, env)
    best_value = float(values.max())
    best_actions = [index for index, value in enumerate(values) if value == best_value]
    return rng.choice(best_actions)


def projected_decision_info(env: RendezvousEnv, action_index: int) -> dict[str, Any]:
    trial_env = copy.deepcopy(env)
    _, _, info = run_decision_action(trial_env, action_index)
    return info


def projected_linear_action_is_better(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    distance_margin_km: float,
    speed_margin_km_s: float,
) -> bool:
    if candidate.get("success", False):
        return True
    if baseline.get("success", False):
        return False
    if candidate.get("unsafe_approach", False) or candidate.get("earth_collision", False):
        return False

    candidate_distance = float(candidate["distance_km"])
    baseline_distance = float(baseline["distance_km"])
    candidate_speed = float(candidate["relative_speed_km_s"])
    baseline_speed = float(baseline["relative_speed_km_s"])

    closer = candidate_distance <= baseline_distance - distance_margin_km
    speed_ok = candidate_speed <= baseline_speed + speed_margin_km_s
    return closer and speed_ok


def best_projected_action_index(
    env: RendezvousEnv,
    *,
    distance_margin_km: float,
    speed_margin_km_s: float,
) -> int:
    greedy_index = greedy_action_index(env)
    greedy_info = projected_decision_info(env, greedy_index)
    best_index = greedy_index
    best_score = -float(greedy_info["distance_km"]) - 25.0 * float(greedy_info["relative_speed_km_s"])

    for action_index in range(env.n_actions):
        if action_index == greedy_index:
            continue
        action_info = projected_decision_info(env, action_index)
        if not projected_linear_action_is_better(
            action_info,
            greedy_info,
            distance_margin_km=distance_margin_km,
            speed_margin_km_s=speed_margin_km_s,
        ):
            continue
        score = -float(action_info["distance_km"]) - 25.0 * float(action_info["relative_speed_km_s"])
        if score > best_score:
            best_index = action_index
            best_score = score

    return best_index


def linear_residual_decision(
    weights: np.ndarray,
    env: RendezvousEnv,
    *,
    guard: bool,
    distance_margin_km: float,
    speed_margin_km_s: float,
) -> LinearDecision:
    greedy_index = greedy_action_index(env)
    linear_index = int(np.argmax(q_values(weights, env, greedy_index, use_greedy_prior=True)))
    if not guard:
        if linear_index == greedy_index:
            return LinearDecision(linear_index, linear_index, greedy_index, "linear_raw_matches_greedy")
        return LinearDecision(linear_index, linear_index, greedy_index, "linear_raw")
    if linear_index == greedy_index:
        return LinearDecision(greedy_index, linear_index, greedy_index, "linear_matches_greedy")

    linear_info = projected_decision_info(env, linear_index)
    greedy_info = projected_decision_info(env, greedy_index)
    if projected_linear_action_is_better(
        linear_info,
        greedy_info,
        distance_margin_km=distance_margin_km,
        speed_margin_km_s=speed_margin_km_s,
    ):
        return LinearDecision(linear_index, linear_index, greedy_index, "linear_projected_better")
    return LinearDecision(greedy_index, linear_index, greedy_index, "linear_guard")


def warm_start_linear(
    weights: np.ndarray,
    *,
    difficulty: Difficulty,
    seed: int,
    scenarios: int,
    passes: int,
    alpha: float,
    projected_scenarios: int,
) -> int:
    env = RendezvousEnv(EnvConfig())
    updates = 0
    for pass_index in range(passes):
        for scenario_index in range(scenarios):
            env.reset(randomize=True, seed=seed + scenario_index, difficulty=difficulty)
            done = False
            while not done:
                features = state_features(env)
                greedy_index = greedy_action_index(env)
                use_projection_label = pass_index == 0 and scenario_index < projected_scenarios
                best_index = (
                    best_projected_action_index(
                        env,
                        distance_margin_km=LINEAR_GUARD_DISTANCE_MARGIN_KM,
                        speed_margin_km_s=LINEAR_GUARD_SPEED_MARGIN_KM_S,
                    )
                    if use_projection_label
                    else greedy_index
                )
                values = q_values(weights, env, greedy_index, use_greedy_prior=True)
                for action_index in range(env.n_actions):
                    if action_index == best_index:
                        target = WARM_START_BEST_VALUE
                    elif action_index == greedy_index:
                        target = WARM_START_GREEDY_VALUE
                    else:
                        target = WARM_START_OTHER_VALUE
                    error = target - values[action_index]
                    weights[action_index] += alpha * error * features
                    updates += 1
                _, done, _ = run_decision_action(env, best_index)
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
    projected_warm_start_scenarios: int,
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
        projected_scenarios=projected_warm_start_scenarios,
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
        "projected_warm_start_scenarios": projected_warm_start_scenarios,
        "reward_scale": reward_scale,
        "warm_start_updates": warm_start_updates,
        "training_successes": successes,
        "best_distance_km": best_distance,
        "best_episode": best_episode,
        "actions": list(env.action_names),
    }
    return weights, metadata


def run_linear_episode(
    weights: np.ndarray,
    seed: int,
    *,
    difficulty: Difficulty,
    guard: bool,
    distance_margin_km: float,
    speed_margin_km_s: float,
) -> dict[str, Any]:
    env = RendezvousEnv(EnvConfig())
    env.reset(randomize=True, seed=seed, difficulty=difficulty)
    done = False
    total_reward = 0.0
    decisions = 0
    decision_reasons: dict[str, int] = {}
    info: dict[str, Any] = {}

    while not done:
        decision = linear_residual_decision(
            weights,
            env,
            guard=guard,
            distance_margin_km=distance_margin_km,
            speed_margin_km_s=speed_margin_km_s,
        )
        action_index = decision.action_index
        decision_reasons[decision.reason] = decision_reasons.get(decision.reason, 0) + 1
        reward, done, info = run_decision_action(env, action_index)
        total_reward += reward
        decisions += 1

    info["total_reward"] = total_reward
    info["decisions"] = decisions
    info["linear_decision_reasons"] = decision_reasons
    return info


def run_greedy_episode(seed: int, *, difficulty: Difficulty) -> dict[str, Any]:
    env = RendezvousEnv(EnvConfig())
    env.reset(randomize=True, seed=seed, difficulty=difficulty)
    done = False
    total_reward = 0.0
    decisions = 0
    info: dict[str, Any] = {}

    while not done:
        action_index = greedy_action_index(env)
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
    guard: bool = True,
    distance_margin_km: float = LINEAR_GUARD_DISTANCE_MARGIN_KM,
    speed_margin_km_s: float = LINEAR_GUARD_SPEED_MARGIN_KM_S,
) -> list[dict[str, Any]]:
    return [
        run_linear_episode(
            weights,
            seed + index,
            difficulty=difficulty,
            guard=guard,
            distance_margin_km=distance_margin_km,
            speed_margin_km_s=speed_margin_km_s,
        )
        for index in range(episodes)
    ]


def evaluate_greedy_policy(
    *,
    difficulty: Difficulty,
    seed: int,
    episodes: int,
) -> list[dict[str, Any]]:
    return [run_greedy_episode(seed + index, difficulty=difficulty) for index in range(episodes)]


def summarize_linear_diagnostics(runs: list[dict[str, Any]]) -> str:
    reasons: dict[str, int] = {}
    for run in runs:
        for reason, count in run.get("linear_decision_reasons", {}).items():
            reasons[reason] = reasons.get(reason, 0) + count

    total = sum(reasons.values())
    if total == 0:
        return "linear diagnostics: no decisions recorded"

    useful = reasons.get("linear_projected_better", 0)
    raw = reasons.get("linear_raw", 0)
    raw_matches = reasons.get("linear_raw_matches_greedy", 0)
    matches = reasons.get("linear_matches_greedy", 0)
    guarded = reasons.get("linear_guard", 0)
    own = raw + useful
    fallback = total - own
    return (
        "linear diagnostics:"
        f" useful_interventions={useful}/{total} ({useful / total * 100.0:.1f}%)"
        f" own={own}/{total} ({own / total * 100.0:.1f}%)"
        f" greedy={fallback}/{total} ({fallback / total * 100.0:.1f}%)"
        f" linear_raw={raw}"
        f" linear_raw_matches_greedy={raw_matches}"
        f" linear_matches_greedy={matches}"
        f" linear_projected_better={useful}"
        f" linear_guard={guarded}"
    )


def print_linear_report(label: str, runs: list[dict[str, Any]], *, diagnostics: bool) -> None:
    print(summarize(label, runs))
    if diagnostics:
        print(summarize_linear_diagnostics(runs))


def print_comparison_report(
    weights: np.ndarray,
    *,
    difficulty: Difficulty,
    seed: int,
    episodes: int,
    distance_margin_km: float,
    speed_margin_km_s: float,
) -> None:
    greedy_runs = evaluate_greedy_policy(difficulty=difficulty, seed=seed, episodes=episodes)
    raw_runs = evaluate_linear_policy(
        weights,
        difficulty=difficulty,
        seed=seed,
        episodes=episodes,
        guard=False,
        distance_margin_km=distance_margin_km,
        speed_margin_km_s=speed_margin_km_s,
    )
    guarded_runs = evaluate_linear_policy(
        weights,
        difficulty=difficulty,
        seed=seed,
        episodes=episodes,
        guard=True,
        distance_margin_km=distance_margin_km,
        speed_margin_km_s=speed_margin_km_s,
    )

    print(f"Policy comparison difficulty={difficulty} episodes={episodes}")
    print(summarize("greedy", greedy_runs))
    print_linear_report("linear_raw", raw_runs, diagnostics=True)
    print_linear_report("linear_guarded", guarded_runs, diagnostics=True)


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
    parser.add_argument("--warm-start-scenarios", type=int, default=10)
    parser.add_argument("--warm-start-passes", type=int, default=1)
    parser.add_argument("--warm-start-alpha", type=float, default=0.02)
    parser.add_argument("--projected-warm-start-scenarios", type=int, default=1)
    parser.add_argument("--reward-scale", type=float, default=0.02)
    parser.add_argument("--eval-episodes", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_LINEAR_POLICY_PATH)
    parser.add_argument("--load-policy", type=Path, help="Evaluate an existing linear policy instead of training.")
    parser.add_argument("--raw-linear", action="store_true", help="Evaluate the linear argmax without greedy residual guard.")
    parser.add_argument("--compare", action="store_true", help="Evaluate greedy, raw linear, and guarded linear on the same seeds.")
    parser.add_argument("--distance-margin", type=float, default=LINEAR_GUARD_DISTANCE_MARGIN_KM)
    parser.add_argument("--speed-margin", type=float, default=LINEAR_GUARD_SPEED_MARGIN_KM_S)
    args = parser.parse_args()

    if args.load_policy is not None:
        weights, metadata = load_policy(args.load_policy)
        print(
            "loaded linear policy:"
            f" path={args.load_policy}"
            f" difficulty={metadata.get('difficulty', 'unknown')}"
            f" episodes={metadata.get('episodes', 'unknown')}"
        )
    else:
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
            projected_warm_start_scenarios=args.projected_warm_start_scenarios,
            reward_scale=args.reward_scale,
        )
        save_policy(weights, metadata, args.output)

    print("Linear Q rendezvous agent")
    print(f"episodes: {metadata['episodes']}")
    print(f"training successes: {metadata['training_successes']}")
    print(f"best training distance: {metadata['best_distance_km']:.2f} km on episode {metadata['best_episode']}")
    eval_seed = args.seed + 10_000
    if args.compare:
        print_comparison_report(
            weights,
            difficulty=args.difficulty,
            seed=eval_seed,
            episodes=args.eval_episodes,
            distance_margin_km=args.distance_margin,
            speed_margin_km_s=args.speed_margin,
        )
    else:
        runs = evaluate_linear_policy(
            weights,
            difficulty=args.difficulty,
            seed=eval_seed,
            episodes=args.eval_episodes,
            guard=not args.raw_linear,
            distance_margin_km=args.distance_margin,
            speed_margin_km_s=args.speed_margin,
        )
        label = "linear_raw" if args.raw_linear else "linear_guarded"
        print_linear_report(label, runs, diagnostics=True)
    if args.load_policy is None:
        print(f"policy written to: {args.output}")


if __name__ == "__main__":
    main()
