# Physics-and-Programming

Mixed repository of interactive astrophysics/physics visualizations in C++ and practical cybersecurity utilities in Python.

## Repository Layout

- `AstroPhysics/`
  - Raylib-based C++ visualizers (orbital mechanics, relativity, quantum concepts, fluid dynamics, EM, etc.).
  - CMake build with one executable per `.cpp` file.
  - Topic-based folders (`quantum/`, `gravity/`, `mechanics/`, `thermodynamics/`, ...).
  - Includes `docs/QuantumStuff_Equations.tex` and `docs/QuantumStuff_Equations.pdf`.
- `CyberTools/`
  - Defensive/local Python scripts for analysis and hygiene tasks.
  - Topic-based folders (`auth/`, `network/`, `web/`, `forensics/`, ...).
- `external/`
  - External or third-party assets.

## Prerequisites

### For `AstroPhysics` (C++)

- CMake 3.20+
- C++17 compiler (MSVC, clang, or gcc)
- Raylib available to CMake (`find_package(raylib CONFIG REQUIRED)`)

Windows note:
This repo has previously been built with Visual Studio Build Tools + vcpkg-style package paths.

### For `CyberTools` (Python)

- Python 3.10+
- Most scripts use only the standard library.
- `CyberTools/osint/OSINT Tool.py` uses `requests`.
- `AstroPhysics/tests/test_quantum_search.py` uses `pytest` + `numpy`.

Install optional Python deps:

```powershell
python -m pip install requests pytest numpy
```

## Build and Run AstroPhysics Visualizers

From repo root:

```powershell
cd AstroPhysics
cmake -S . -B build -DCMAKE_PREFIX_PATH="C:\vcpkg\installed\x64-windows"
cmake --build build --config Release
```

Run a target from the build output folder:

```powershell
.\build\Release\solar_system_spacetime_viz_cpp.exe
```

Build a single target:

```powershell
cmake --build build --config Release --target solar_system_spacetime_viz_cpp
```

## Run CyberTools Scripts

From repo root:

```powershell
cd CyberTools
python "auth/Password Strength Checker.py" --password "example"
python "web/URL Heuristic Scanner.py" --url "https://example.com/login"
python "network/Certificate Expiry Checker.py" --host "example.com"
```

## Run Tests

Current tests are in `AstroPhysics/tests`.

From repo root:

```powershell
$env:PYTHONPATH = "AstroPhysics"
pytest AstroPhysics/tests -q
```

## Notes

- Many visualizers are interactive and open a windowed render loop.
- Some filenames include spaces/apostrophes; quote paths in terminal commands.
- Build outputs are generated in local build directories (for example, `AstroPhysics/build-native/` and `AstroPhysics/build-vcpkg/`).
