from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from policy_eval import run_episode, summarize
from q_learning import default_policy_path, load_policy
from rendezvous_env import Difficulty


DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "full")
POLICIES = ("random", "greedy", "qlearn")
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "simulations" / "q_learning" / "curriculum_eval.json"
)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def evaluate_difficulty(
    difficulty: Difficulty,
    *,
    episodes: int,
    seed: int,
    q_policy_path: Path | None,
) -> dict[str, Any]:
    resolved_q_policy_path = q_policy_path or default_policy_path(True, difficulty)
    q_table: dict[str, list[float]] | None = None
    q_metadata: dict[str, Any] | None = None

    if resolved_q_policy_path.exists():
        q_table, q_metadata = load_policy(resolved_q_policy_path)

    policy_results: dict[str, Any] = {}
    for policy in POLICIES:
        if policy == "qlearn" and q_table is None:
            policy_results[policy] = {
                "available": False,
                "policy_path": str(resolved_q_policy_path),
                "runs": [],
            }
            continue

        runs = [
            run_episode(policy, seed + index, difficulty=difficulty, q_table=q_table)
            for index in range(episodes)
        ]
        successes = sum(1 for run in runs if run["success"])
        distances = [run["distance_km"] for run in runs]
        speeds = [run["relative_speed_km_s"] for run in runs]
        delta_v = [run["fuel_delta_v_km_s"] * 1000.0 for run in runs]

        policy_results[policy] = {
            "available": True,
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "mean_final_distance_km": sum(distances) / episodes,
            "median_final_distance_km": median(distances),
            "best_final_distance_km": min(distances),
            "mean_relative_speed_km_s": sum(speeds) / episodes,
            "mean_delta_v_m_s": sum(delta_v) / episodes,
            "runs": runs,
        }

    return {
        "difficulty": difficulty,
        "episodes": episodes,
        "seed": seed,
        "q_policy_path": str(resolved_q_policy_path),
        "q_policy_metadata": q_metadata,
        "policies": policy_results,
    }


def markdown_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| difficulty | policy | success | mean distance (km) | best (km) | mean speed (km/s) | mean dv (m/s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        difficulty = result["difficulty"]
        for policy in POLICIES:
            policy_result = result["policies"][policy]
            if not policy_result["available"]:
                lines.append(f"| {difficulty} | {policy} | missing | - | - | - | - |")
                continue
            lines.append(
                f"| {difficulty} | {policy} | "
                f"{policy_result['successes']}/{policy_result['episodes']} | "
                f"{policy_result['mean_final_distance_km']:.2f} | "
                f"{policy_result['best_final_distance_km']:.2f} | "
                f"{policy_result['mean_relative_speed_km_s']:.4f} | "
                f"{policy_result['mean_delta_v_m_s']:.2f} |"
            )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rendezvous policies across curriculum difficulties.")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=10011)
    parser.add_argument("--difficulty", choices=DIFFICULTIES, action="append")
    parser.add_argument("--q-policy", type=Path, help="Use one Q policy for all requested difficulties.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    difficulties = tuple(args.difficulty) if args.difficulty else DIFFICULTIES
    results = [
        evaluate_difficulty(
            difficulty,
            episodes=args.episodes,
            seed=args.seed,
            q_policy_path=args.q_policy,
        )
        for difficulty in difficulties
    ]

    for result in results:
        print(f"Policy evaluation difficulty={result['difficulty']} episodes={args.episodes}")
        for policy in POLICIES:
            policy_result = result["policies"][policy]
            if not policy_result["available"]:
                print(f"{policy:>7s}: missing policy at {result['q_policy_path']}")
                continue
            print(summarize(policy, policy_result["runs"]))
        print()

    print(markdown_table(results))

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nresults written to: {args.output}")


if __name__ == "__main__":
    main()
