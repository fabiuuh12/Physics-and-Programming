# Getting started

Run commands from the repository root: the folder containing `explore.py`.

## 1. Choose what to run

Install Python 3.10 or newer, then browse without installing any other packages:

```bash
python3 explore.py
python3 explore.py list
python3 explore.py list optics
python3 explore.py info projectile_drag_viz
```

Use `python` on Windows. The [catalog](CATALOG.md) links directly to source files and separates programs from supporting modules. Supporting modules and headers are used by other programs; they are not standalone apps.

## 2. Set up C++ demos

You need CMake 3.20+, a C++17 compiler, raylib with its CMake package configuration, and a graphical desktop.

On macOS with Homebrew and the Apple command-line tools installed:

```bash
brew install cmake raylib
python3 explore.py run projectile_drag_viz
```

On Linux, install CMake, a C++17 compiler, raylib and its graphics dependencies using your distribution's package manager or raylib's installation instructions. On Windows, install CMake, Visual Studio C++ build tools, and raylib through your preferred C++ package manager. Run the launcher from a terminal configured for that compiler.

If raylib is installed in a custom location, configure the launcher's build directory once:

```bash
cmake -S CompPhysics -B CompPhysics/build-launcher -DCMAKE_PREFIX_PATH="/path/to/raylib/install"
```

For a package manager that requires a CMake toolchain file, pass `-DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake` in that initial configuration instead.

The launcher uses `CompPhysics/build-launcher/` so it does not interfere with existing local builds. CMake checks source and header changes on each run and builds only the selected target and its dependencies.

```bash
python3 explore.py run three_body_problem_viz
python3 explore.py run --dry-run three_body_problem_viz
```

The dry run prints commands without compiling or opening a window. Existing `CompPhysics/run` and `CompPhysics/apps/` shortcuts remain available for local setups that have already built vendored raylib.

## 3. Set up Python programs

Create an isolated environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux with `source .venv/bin/activate`, or in Windows PowerShell with `.venv\Scripts\Activate.ps1`. Then install the shared dependencies:

```bash
python -m pip install -r CompPhysics/requirements.txt
python explore.py list "Python script"
python explore.py run vision/two_hand_bridge.py
```

Some standard-library utilities do not need the shared packages. Individual scripts may have additional platform requirements. Camera programs need webcam permission and an available camera. Bridge programs send input to a separate visualization; starting a bridge alone does not start its C++ scene. See the source docstring and topic README for pairings and controls.

Forward arguments by placing them after the script name. Quote paths containing spaces:

```bash
python explore.py run "CyberTools/network/Certificate Expiry Checker.py" --file /absolute/path/to/hosts.txt
```

For this certificate checker, the input file contains hostnames, one per line.

All launched programs start with the repository root as their working directory. Use absolute paths for input/output arguments when you want a different location. File utilities perform their normal operations when launched; inspect their source and inputs before running them.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Unsure what is installed | Run `python3 explore.py doctor` |
| `cmake` not found | Install CMake and reopen your terminal |
| raylib package cannot be found | Install raylib's development package or configure its prefix as above |
| Compiler not found | Install a C++17 toolchain; on Windows use its developer terminal |
| Python import fails | Activate your environment and install the requirements using that environment's Python |
| More than one program matches | Use the full relative source path shown by `list` |
| Supporting module cannot be launched | Select a script or C++ demo from the catalog instead |
| Window or webcam fails to open | Use a local graphical session and check camera permissions |

## Validation and development

```bash
python -m pip install -r CompPhysics/requirements-dev.txt
python -m pytest CompPhysics/tests -q
```

To build everything directly:

```bash
cmake -S CompPhysics -B CompPhysics/build-launcher -DCMAKE_BUILD_TYPE=Release
cmake --build CompPhysics/build-launcher --config Release
```

Numerical checks do not require a graphics window. `CompPhysics/tests/graphics_smoke.py` provides separate opt-in desktop checks and currently looks for binaries in `CompPhysics/build-native/`; see its source before using it.
