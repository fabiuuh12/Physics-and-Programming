# AI in Space

Research and simulation work for autonomous spacecraft decision-making.

## Focus

The core idea is to build spacecraft simulations where an AI agent chooses actions under realistic space constraints:

- orbital dynamics
- fuel limits
- collision risk
- sensor uncertainty
- communication delay
- power and thermal limits
- later: nuclear-electric propulsion and relativistic timing corrections

## First Project

Autonomous orbital rendezvous in 2D.

The agent controls a spacecraft trying to approach a target satellite while minimizing fuel use and avoiding unsafe trajectories.

Initial action set:

- coast
- thrust prograde
- thrust retrograde
- thrust radial outward
- thrust radial inward

Initial reward terms:

- reward getting closer to the target
- penalize fuel use
- penalize high relative speed near the target
- heavily penalize collisions or unsafe approaches
- reward successful rendezvous

## Roadmap

1. Build a simple 2D orbital mechanics environment.
2. Add a rule-based planner for comparison.
3. Train a reinforcement learning agent.
4. Add debris avoidance and sensor noise.
5. Add low-thrust electric propulsion.
6. Add nuclear-electric power and thermal constraints.
7. Add relativistic timing/navigation corrections where they matter.

## Folders

- `python/`: source code and experiments
- `cpp/`: fast dynamics and simulation kernels
- `simulations/`: scenario definitions and simulation outputs
- `notes/`: research notes and derivations
- `papers/`: drafts, references, and writeups

## Language Split

Use both Python and C++ from the beginning.

- Python is for experiments, AI agents, plotting, notebooks, reward design, and scenario iteration.
- C++ is for dynamics, long simulation batches, custom integrators, and performance-sensitive kernels.

The first milestone is to keep a small rendezvous simulator in both languages so the project grows with a hybrid workflow from day one.

## Working On Mac And Windows

Use Git as the source of truth between machines:

```bash
git pull
git status
git add .
git commit -m "Describe the change"
git push
```

Then pull on the other computer before continuing work there.

Recommended Python setup on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

Recommended Python setup on Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test runs the Python simulator, a randomized baseline, policy evaluation, the CMake C++ build, and the C++ executable. Use `python scripts/smoke_test.py --skip-cpp` if CMake or a C++ compiler is not installed yet.

Use `python` in project commands after activating the virtual environment. On macOS, `python3` also works if you are outside the virtual environment.

Run the unit tests:

```bash
python -m pytest
```

Run the official policy benchmark:

```bash
python scripts/benchmark.py --episodes 24
```

The full guarded Q-learning benchmark can take several minutes because each learned action is checked against a one-decision greedy projection. Use `--no-write` when you want a read-only scorecard.

## Current Next Step

The first planner is a greedy lookahead baseline. The next AI milestone is to make the learned policy beat that baseline, then compare tabular Q-learning against a linear function approximator.

The current Q-learning policy is warm-started from the greedy planner so there is already a successful replay path. Future training should reduce that dependence by improving state bins, reward shaping, and randomized initial conditions.

Randomized scenarios vary target altitude, chaser altitude, and starting phase angle. Use them to check whether a policy is learning a reusable strategy instead of memorizing the first setup. Difficulty levels stage the spread as `easy`, `medium`, and `full`.

## Running The First Simulations

Python:

```bash
MPLBACKEND=Agg MPLCONFIGDIR="simulations/.matplotlib" python3 python/rendezvous_sim.py
```

Random policy baseline using the RL-style environment:

```bash
python3 python/random_policy.py
python3 python/random_policy.py --randomized --seed 21
python3 python/random_policy.py --randomized --difficulty easy --seed 21
```

Train a first tabular Q-learning agent:

```bash
python3 python/q_learning.py
python3 python/q_learning.py --randomized --episodes 1200
python3 python/q_learning.py --randomized --difficulty easy --episodes 1200
python3 python/q_learning.py --randomized --difficulty medium --episodes 900 --init-policy simulations/q_learning/q_policy_easy.json --output simulations/q_learning/q_policy_medium_from_easy.json
python3 python/policy_eval.py --difficulty easy
python scripts/benchmark.py --episodes 24
```

The fixed run writes `simulations/q_learning/q_policy_fixed.json`. Randomized curriculum runs write policies such as `simulations/q_learning/q_policy_easy.json`, `q_policy_medium.json`, and `q_policy_randomized.json` for the full range.

Current easy-difficulty comparison over 24 seeds after improving the Q-learning state, reward, and adding projection guards for learned actions:

- random: 0/24 successes
- greedy: 20/24 successes
- guarded qlearn: 20/24 successes

The guarded tabular policy now matches greedy on easy randomized cases. It still needs to beat the planner rather than mostly relying on projection guards, but the benchmark gives a stable scorecard for the next AI upgrade.

Current easy guarded Q-learning mean final distance is `4.91 km`, down from `23.25 km` before the fallback and `82.58 km` before the state/reward update.

Current curriculum sweep over 24 seeds:

| difficulty | random | greedy | qlearn |
| --- | ---: | ---: | ---: |
| easy | 0/24 | 20/24 | 20/24 |
| medium | 0/24 | 7/24 | 7/24 |
| full | 0/24 | 5/24 | 5/24 |

The guarded medium policy now matches the greedy baseline on the 24-seed benchmark after adding a one-decision projection guard. Medium Q-learning improved from `2/24` successes and `52.35 km` mean final distance to `7/24` successes and `19.28 km` mean final distance. The next practical learning target is making the learned table beat greedy on medium instead of mostly relying on the guard.

On the full difficulty sweep, guarded Q-learning currently matches greedy at `5/24` successes with `44.89 km` mean final distance. That makes the full-range policy useful as a baseline check, but not yet an improvement over the planner.

An easy-to-medium transfer run improved mean distance to `107.59 km` but scored `0/24`, so it is useful diagnostic data rather than the new default medium policy.

Episode animations:

```bash
MPLBACKEND=Agg MPLCONFIGDIR="simulations/.matplotlib" python3 python/episode_viz.py --policy greedy
MPLBACKEND=Agg MPLCONFIGDIR="simulations/.matplotlib" python3 python/episode_viz.py --policy random
MPLBACKEND=Agg MPLCONFIGDIR="simulations/.matplotlib" python3 python/episode_viz.py --policy qlearn
MPLBACKEND=Agg MPLCONFIGDIR="simulations/.matplotlib" python3 python/episode_viz.py --policy qlearn --randomized --seed 10024
MPLBACKEND=Agg MPLCONFIGDIR="simulations/.matplotlib" python3 python/episode_viz.py --policy qlearn --randomized --difficulty easy --seed 10024
```

C++ with CMake:

```bash
cmake -S . -B build
cmake --build build
./build/rendezvous_cpp
```

C++ without CMake:

```bash
mkdir -p build
clang++ -std=c++17 -Icpp/include cpp/src/orbit.cpp cpp/apps/rendezvous_cpp.cpp -o build/rendezvous_cpp
./build/rendezvous_cpp
```

First paper:

```bash
PATH="$PWD/tools/tinytex/TinyTeX/bin/universal-darwin:$PATH" pdflatex -interaction=nonstopmode -halt-on-error -output-directory=papers papers/ai_in_space_foundation.tex
```
