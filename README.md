# Physics and Programming

Interactive physics demonstrations in C++ and Python, plus a collection of Python security and file utilities. The physics demos are educational explorations; their assumptions and visual scales vary by program.

## Start here

From this folder, open the numbered program menu:

```bash
python3 explore.py
```

On Windows, use `python` instead of `python3`. The launcher itself needs only Python 3.10+. Running a program may require additional setup: see the [getting-started guide](docs/GETTING_STARTED.md).

```bash
python3 explore.py list gravity
python3 explore.py info three_body_problem_viz
python3 explore.py run three_body_problem_viz
python3 explore.py doctor
```

C++ programs are configured and built automatically through CMake before they run. Python scripts use the same Python environment as the launcher. Program arguments go after the program name.

## Find your way

| I want to… | Go here |
| --- | --- |
| Install dependencies and run my first demo | [Getting started](docs/GETTING_STARTED.md) |
| Browse every demo, script, and supporting Python module | [Program catalog](docs/CATALOG.md) |
| Understand the folder structure | [Repository map](docs/STRUCTURE.md) |
| Learn physics-demo controls and model assumptions | [Physics guide](CompPhysics/README.md) |
| Understand the portfolio's scope | [Portfolio notes](CompPhysics/PORTFOLIO.md) |

## Try these first

| Demo | What to explore | Run |
| --- | --- | --- |
| Projectile drag | Compare trajectories with and without air resistance | `python3 explore.py run projectile_drag_viz` |
| Forced oscillator | Damping, driving and resonance | `python3 explore.py run damped_forced_oscillator_viz` |
| Three-body problem | Orbital motion and gravitational forces | `python3 explore.py run three_body_problem_viz` |
| Quantum tunneling | Wavepacket evolution and scattering | `python3 explore.py run quantum_tunneling_viz` |
| Hohmann transfer | Transfer geometry and orbital maneuvers | `python3 explore.py run hohmann_transfer_viz` |

Each graphical demo displays its own controls. Webcam and hand-controlled programs may require a companion bridge; check their topic documentation before starting.

## Check the numerical models

```bash
python3 -m pip install -r CompPhysics/requirements-dev.txt
python3 -m pytest CompPhysics/tests -q
```

The suite covers Python quantum-search calculations and rendering-independent C++ numerical checks. The C++ checks require a C++17 compiler and skip when none is available. These checks do not validate every visualization.
