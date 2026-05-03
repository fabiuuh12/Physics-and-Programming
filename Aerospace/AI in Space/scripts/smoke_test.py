from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / "simulations" / ".matplotlib"))
    return env


def executable_path(name: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidates = [
        ROOT / "build" / f"{name}{suffix}",
        ROOT / "build" / "Debug" / f"{name}{suffix}",
        ROOT / "build" / "Release" / f"{name}{suffix}",
        ROOT / "build" / "RelWithDebInfo" / f"{name}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick cross-platform project checks.")
    parser.add_argument("--skip-cpp", action="store_true", help="Skip CMake build and C++ executable check.")
    parser.add_argument(
        "--quick-ai",
        action="store_true",
        help="Run a tiny Q-learning training pass to verify the AI training path.",
    )
    args = parser.parse_args()

    env = python_env()
    py = sys.executable

    run([py, "python/rendezvous_sim.py"], env=env)
    run([py, "python/random_policy.py", "--randomized", "--difficulty", "easy", "--seed", "21"], env=env)
    run([py, "python/policy_eval.py", "--difficulty", "easy", "--episodes", "3"], env=env)

    if args.quick_ai:
        run(
            [
                py,
                "python/q_learning.py",
                "--randomized",
                "--difficulty",
                "easy",
                "--episodes",
                "3",
                "--eval-episodes",
                "1",
                "--warm-start-scenarios",
                "2",
                "--output",
                "simulations/q_learning/q_policy_smoke.json",
            ],
            env=env,
        )

    if not args.skip_cpp:
        run(["cmake", "-S", ".", "-B", "build"])
        run(["cmake", "--build", "build"])
        run([str(executable_path("rendezvous_cpp"))])

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
