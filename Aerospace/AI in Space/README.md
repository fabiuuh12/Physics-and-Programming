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

## Running The First Simulations

Python:

```bash
MPLBACKEND=Agg MPLCONFIGDIR="simulations/.matplotlib" python3 python/rendezvous_sim.py
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
