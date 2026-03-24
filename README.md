# Physics-and-Programming

Mixed repository of interactive physics/astrophysics simulations and practical cybersecurity tooling.

## Current Repo Snapshot

- `AstroPhysics/`: C++ Raylib visualizers plus Python vision/interaction bridges.
- `CyberTools/`: standalone Python security and analysis scripts.
- C++ visualizers: 63 `.cpp` simulation programs in `AstroPhysics/` (plus vendored third-party sources).
- CyberTools scripts: 24 Python utilities across auth/network/forensics/web/file/integrity/osint.

## Repository Layout

- `AstroPhysics/astronomy`, `gravity`, `mechanics`, `quantum`, `nuclear`, `particle_physics`, `relativity`, `thermodynamics`, `electromagnetism`, `fluids`, `dimensions`
- `AstroPhysics/vision`
- `AstroPhysics/DefensiveSys`
- `AstroPhysics/tests`
- `AstroPhysics/docs`
- `AstroPhysics/third_party` (vendored Raylib source/headers)
- `CyberTools/auth`, `file_ops`, `forensics`, `integrity_hashing`, `network`, `osint`, `web`

## AstroPhysics (C++ and Vision)

- Build system: `AstroPhysics/CMakeLists.txt`
- Model: one executable target per simulation `.cpp`
- Example C++ targets:
- `solar_system_spacetime_viz_cpp`
- `quantum_search_cpp`
- `gravitational_lensing_viz_cpp`
- `defensive_sys_3d_cpp`
- `vision_two_hands_scene_cpp`

Vision and bridge scripts in `AstroPhysics/vision`:

- `hand_planet_overlay.py` (hand-tracked atom/neutron/star/blackhole interaction scene)
- `floating_hand_avatar.py`
- `hologram_control.py` (webcam multi-hologram controller with fist cycling plus one-hand/two-hand gestures)
- `webcam_finger_tracker.py`
- `two_hand_bridge.py` (UDP hand bridge)
- `two_hands_scene.cpp` (C++ visual scene receiver)

Defensive simulation bridge:

- `AstroPhysics/DefensiveSys/hand_turret_sim.py`
- `AstroPhysics/DefensiveSys/defensive_sys_3d.cpp`

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

Python packages used across repo:

- `numpy`
- `opencv-python`
- `mediapipe`
- `pytest`
- `requests`

Install Python deps:

```bash
python3 -m pip install numpy opencv-python mediapipe pytest requests
```

## Build C++ Programs

From repo root:

```bash
cd AstroPhysics
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --config Release
```

Build one target:

```bash
cmake --build build-native --config Release --target solar_system_spacetime_viz_cpp
```

Run (macOS/Linux example):

```bash
./AstroPhysics/build-native/solar_system_spacetime_viz_cpp
```

Run (Windows example):

```powershell
.\AstroPhysics\build-native\Release\solar_system_spacetime_viz_cpp.exe
```

## Run Python Programs

Examples from repo root:

```bash
python3 AstroPhysics/vision/hand_planet_overlay.py
python3 AstroPhysics/vision/floating_hand_avatar.py
python3 AstroPhysics/vision/hologram_control.py
python3 AstroPhysics/vision/two_hand_bridge.py
python3 AstroPhysics/DefensiveSys/hand_turret_sim.py
python3 "CyberTools/network/Certificate Expiry Checker.py" --host example.com
```

## Tests

Current automated tests:

- `AstroPhysics/tests/test_quantum_search.py`

Run:

```bash
PYTHONPATH=AstroPhysics pytest AstroPhysics/tests -q
```

## Notes

- Many C++ programs open real-time interactive windows.
- Some script filenames include spaces/apostrophes; quote paths in shell commands.
- `AstroPhysics/build*` and `__pycache__` directories are generated artifacts.
