from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from curriculum_eval import DIFFICULTIES, evaluate_difficulty, markdown_table  # noqa: E402
from policy_eval import summarize  # noqa: E402
from rendezvous_env import Difficulty  # noqa: E402


DEFAULT_OUTPUT_PATH = ROOT / "simulations" / "q_learning" / "curriculum_eval.json"


def print_results(results: list[dict[str, Any]], episodes: int) -> None:
    for result in results:
        print(f"Policy evaluation difficulty={result['difficulty']} episodes={episodes}")
        for policy, policy_result in result["policies"].items():
            if not policy_result["available"]:
                print(f"{policy:>7s}: missing policy at {result['q_policy_path']}")
                continue
            print(summarize(policy, policy_result["runs"]))
        print()

    print(markdown_table(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the official AI in Space rendezvous benchmark.")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=10011)
    parser.add_argument("--difficulty", choices=DIFFICULTIES, action="append")
    parser.add_argument("--q-policy", type=Path, help="Use one Q policy for all requested difficulties.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    difficulties: tuple[Difficulty, ...] = tuple(args.difficulty) if args.difficulty else DIFFICULTIES
    results = [
        evaluate_difficulty(
            difficulty,
            episodes=args.episodes,
            seed=args.seed,
            q_policy_path=args.q_policy,
        )
        for difficulty in difficulties
    ]

    print_results(results, args.episodes)

    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nresults written to: {args.output}")


if __name__ == "__main__":
    main()
