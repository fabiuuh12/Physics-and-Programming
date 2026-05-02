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

## Current Next Step

The first planner is a greedy lookahead baseline. The next AI milestone is to train a small tabular or linear agent against `python/rendezvous_env.py`, then compare it against both the random policy and the greedy planner.

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
python3 python/policy_eval.py --difficulty easy
```

The fixed run writes `simulations/q_learning/q_policy_fixed.json`. Randomized curriculum runs write policies such as `simulations/q_learning/q_policy_easy.json`, `q_policy_medium.json`, and `q_policy_randomized.json` for the full range.

Current easy-difficulty comparison over 24 seeds after improving the Q-learning state and reward:

- random: 0/24 successes
- greedy: 20/24 successes
- qlearn: 11/24 successes

The tabular learner now solves nearly half of the easy randomized cases. It still trails the greedy lookahead planner, but the updated state/reward design moved it from `1/24` to `11/24`.

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
