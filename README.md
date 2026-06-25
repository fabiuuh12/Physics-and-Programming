# Physics-and-Programming

Mixed repository of computational physics simulations and practical cybersecurity tooling.

## Current Repo Snapshot

- `CompPhysics/`: C++ Raylib visualizers plus Python vision/interaction bridges.
- `CompPhysics/CyberTools/`: standalone Python security and analysis scripts.
- C++ visualizers: many standalone `.cpp` simulation programs in `CompPhysics/` (plus vendored third-party sources).
- CyberTools scripts: 24 Python utilities across auth/network/forensics/web/file/integrity/osint.

## Repository Layout

- `CompPhysics/astronomy`, `gravity`, `mechanics`, `quantum`, `nuclear`, `particle_physics`, `relativity`, `thermodynamics`, `electromagnetism`, `fluids`, `waves`, `dimensions`
- `CompPhysics/vision`
- `CompPhysics/DefensiveSys`
- `CompPhysics/mathematics`
- `CompPhysics/tests`
- `CompPhysics/mathematics/docs`
- `CompPhysics/third_party` (vendored Raylib source/headers)
- `CompPhysics/CyberTools/auth`, `file_ops`, `forensics`, `integrity_hashing`, `network`, `osint`, `web`

## CompPhysics (C++ and Vision)

- Build system: `CompPhysics/CMakeLists.txt`
- Model: one executable target per simulation `.cpp`
- Example C++ targets:
- `solar_system_spacetime_viz_cpp`
- `quantum_search_cpp`
- `gravitational_lensing_viz_cpp`
- `defensive_sys_3d_cpp`
- `vision_two_hands_scene_cpp`

Vision and bridge scripts in `CompPhysics/vision`:

- `hand_planet_overlay.py` (hand-tracked atom/neutron/star/blackhole interaction scene)
- `floating_hand_avatar.py`
- `hologram_control.py` (webcam multi-hologram controller with fist cycling plus one-hand/two-hand gestures)
- `webcam_finger_tracker.py`
- `two_hand_bridge.py` (UDP hand bridge)
- `two_hands_scene.cpp` (C++ visual scene receiver)

Defensive simulation bridge:

- `CompPhysics/DefensiveSys/hand_turret_sim.py`
- `CompPhysics/DefensiveSys/defensive_sys_3d.cpp`

## Featured Demos

| Demo | Topic | Target |
| --- | --- | --- |
| Aerodynamics | Mechanics and fluid intuition | `aerodynamics_viz_cpp` |
| Hohmann Transfer | Orbital mechanics | `hohmann_transfer_viz_cpp` |
| Launch Window Porkchop | Mission design trade space | `launch_window_porkchop_viz_cpp` |
| Three Body Problem | Nonlinear orbital dynamics | `three_body_problem_viz_cpp` |
| Gravitational Lensing Playground | Relativity and optics intuition | `gravitational_lensing_playground_viz_cpp` |
| Helical Wave Laboratory | Waves, interference, and polarization | `helical_wave_lab_viz_cpp` |
| Tokamak Confinement | Plasma physics | `tokamak_confinement_viz_cpp` |

Add screenshots or GIFs under `media/` when capturing demos locally, then link them from this table.

## CyberTools

Python security utilities grouped by domain:

- `auth`: password/JWT/logon helpers
- `network`: DNS/subdomain/port/certificate checks
- `web`: URL heuristics, headers audit, local HTTP server
- `forensics`: IOC extraction, log/IP summaries, suspicious name checks
- `file_ops`: bulk rename, duplicate finder, JSON->CSV, extension audit
- `integrity_hashing`: inventory/check/decode helpers
- `osint`: consolidated OSINT helper script

## Prerequisites

- CMake 3.20+
- C++17 compiler (MSVC, clang, or gcc)
- Raylib available to CMake (`find_package(raylib CONFIG REQUIRED)`)
- Python 3.10+

macOS setup:

```bash
brew install cmake raylib python
python3 -m pip install -r CompPhysics/requirements.txt -r CompPhysics/requirements-dev.txt
```

Python packages used across the repo:

- `numpy`
- `opencv-python`
- `mediapipe`
- `pytest`
- `requests`

Install Python deps:

```bash
python3 -m pip install -r CompPhysics/requirements.txt -r CompPhysics/requirements-dev.txt
```

## Build C++ Programs

From repo root:

```bash
cd CompPhysics
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --config Release
```

macOS quick demo:

```bash
cd CompPhysics
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target aerodynamics_viz_cpp
./build-native/aerodynamics_viz_cpp
```

Build one target:

```bash
cmake --build build-native --config Release --target solar_system_spacetime_viz_cpp
```

Run (macOS/Linux example):

```bash
./CompPhysics/build-native/solar_system_spacetime_viz_cpp
```

Run (Windows example):

```powershell
.\CompPhysics\build-native\Release\solar_system_spacetime_viz_cpp.exe
```

## Run Python Programs

Examples from repo root:

```bash
python3 CompPhysics/vision/hand_planet_overlay.py
python3 CompPhysics/vision/floating_hand_avatar.py
python3 CompPhysics/vision/hologram_control.py
python3 CompPhysics/vision/two_hand_bridge.py
python3 CompPhysics/DefensiveSys/hand_turret_sim.py
python3 "CompPhysics/CyberTools/network/Certificate Expiry Checker.py" --host example.com
```

## Tests

Current automated tests:

- `CompPhysics/tests/test_quantum_search.py`

Run:

```bash
PYTHONPATH=CompPhysics pytest CompPhysics/tests -q
```

## Notes

- Many C++ programs open real-time interactive windows.
- Some script filenames include spaces/apostrophes; quote paths in shell commands.
- `CompPhysics/build*` and `__pycache__` directories are generated artifacts.
