# Aerospace and Astrophysics Simulation Portfolio

This folder is a portfolio of interactive physics simulations built around aerospace-relevant topics: orbital mechanics, gravitational fields, electromagnetic fields, fluids, plasma behavior, optics, thermodynamics, and numerical/visual intuition for quantum systems.

The strongest presentation value is in the C++/raylib visual simulations. They show comfort with compiled systems, real-time rendering loops, vector math, physical models, camera controls, and clear interactive UI overlays.

## Highlights

- **Orbital and gravitational dynamics:** two-body orbits, three-body motion, Lagrange points, gravitational lensing, microlensing, spacetime visualizations, black hole accretion, and gravitational wave interferometer intuition.
- **Aerospace-adjacent field physics:** magnetospheres, solar wind, Maxwell waves, Poynting vector energy flow, plasma confinement, and fluid vortex/channel flow.
- **Mechanics and controls intuition:** projectile drag, aerodynamics, angular momentum, harmonic oscillators, chaotic pendulums, and hand-tracked interactive labs.
- **Scientific communication:** each visualization is designed to make an abstract model inspectable through motion, geometry, labels, and live controls.
- **Build discipline:** CMake-based C++ targets, isolated Python demos, vendored third-party dependencies, and a focused pytest suite for the Grover-search math helper.

## Technical Stack

- C++17
- CMake 3.20+
- raylib for real-time 2D/3D visualization
- Python 3 for hand-tracking bridges, optics demos, math utilities, and tests
- pytest and NumPy for Python-side numerical checks

## Folder Map

| Folder | Purpose |
| --- | --- |
| `astronomy/` | Stellar, galactic, exoplanet, pulsar, solar-wind, and mission visualizations |
| `gravity/` | Orbital mechanics, black holes, lensing, gravitational waves, and spacetime demos |
| `mechanics/` | Classical mechanics, oscillators, drag, aerodynamics, and interactive hand labs |
| `electromagnetism/` | Electric fields, Maxwell waves, magnetospheres, and EM energy flow |
| `fluids/` | Channel flow and vortex visualization |
| `plasma/` | Tokamak confinement visualization |
| `relativity/` | Time dilation, Doppler effects, and Penrose diagram visualization |
| `quantum/` | Quantum visualization demos and Grover-search visualization |
| `particle_physics/` | Higgs-field, accelerator, entanglement, and Feynman diagram demos |
| `thermodynamics/` | Thermal laws, entropy, and pseudo-thermal camera demos |
| `optics/` | Python optics scenes and hand-driven optical demonstrations |
| `vision/` | Hand-tracking bridges and shared controls for interactive demos |
| `mathematics/` | Equation animation tooling and supporting math documents |
| `tests/` | Python tests for numerical helper functions |
| `third_party/` | Vendored raylib headers/source used by local builds |

## Recommended Demos

These are good first demos for an aerospace or space-systems reviewer:

| Target | Source | Why it is relevant |
| --- | --- | --- |
| `aerodynamics_viz_cpp` | `mechanics/aerodynamics_viz.cpp` | Visualizes aerodynamic intuition and force behavior |
| `three_body_problem_viz_cpp` | `gravity/three_body_problem_viz.cpp` | Demonstrates nonlinear orbital dynamics |
| `gravity_lagrange_viz_cpp` | `gravity/gravity_lagrange_viz.cpp` | Shows Lagrange-point intuition for mission design |
| `magnetosphere_solar_wind_viz_cpp` | `electromagnetism/magnetosphere_solar_wind_viz.cpp` | Connects space weather, charged particles, and planetary fields |
| `tokamak_confinement_viz_cpp` | `plasma/tokamak_confinement_viz.cpp` | Shows field-guided plasma confinement concepts |
| `gravitational_lensing_playground_viz_cpp` | `gravity/gravitational_lensing_playground_viz.cpp` | Communicates spacetime curvature through interactive optics-like behavior |
| `orbital_construction_hand_lab_viz_cpp` | `astronomy/orbital_construction_hand_lab_viz.cpp` | Combines orbital visualization with live hand-control input |

## Build C++ Visualizations

From this folder:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target aerodynamics_viz_cpp
./build-native/aerodynamics_viz_cpp
```

Build all configured C++ targets:

```bash
cmake --build build-native
```

If CMake cannot find raylib, install raylib with your platform package manager or point CMake at the raylib package configuration using `CMAKE_PREFIX_PATH`.

## Run Python Tests

From this folder:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

## Notes For Reviewers

- Files under `build/`, `build-native/`, `build-cmake/`, and `bin/` are local generated outputs or convenience binaries, not the primary source of truth.
- The primary code to review is in the domain folders listed above and in `CMakeLists.txt`.
- Some interactive demos use webcam/hand-tracking bridges; they can be reviewed independently from the C++ visualizations.
