# Getting started

Run these commands from the repository root, the folder containing `explore.py`. Use `python` instead of `python3` on Windows.

## Browse first

Install Python 3.10 or newer. Browsing requires no extra packages:

```bash
python3 explore.py
python3 explore.py list gravity
python3 explore.py list cybertools
python3 explore.py info projectile_drag_viz
python3 explore.py doctor
```

The [catalog](CATALOG.md) links to each source file and distinguishes standalone programs from supporting modules. The [folder map](STRUCTURE.md) explains the complete layout.

## C++ demonstrations

Install CMake 3.20+, a C++17 compiler, raylib with its CMake package configuration, and use a graphical desktop.

On macOS with Homebrew and Apple command-line tools:

```bash
brew install cmake raylib
python3 explore.py run projectile_drag_viz
```

On Linux, install CMake, a C++17 compiler, raylib and its graphics dependencies through your distribution's package manager or raylib's installation instructions. On Windows, use CMake, Visual Studio C++ build tools, and a raylib package installation. Use a terminal configured for that compiler.

The launcher configures `CompPhysics/output/build/`, builds the selected target, then opens it. CMake rebuilds changed sources and headers. To inspect commands without executing:

```bash
python3 explore.py run --dry-run three_body_problem_viz
```

For a custom raylib installation, configure the build directory once:

```bash
cmake -S CompPhysics -B CompPhysics/output/build -DCMAKE_PREFIX_PATH="/path/to/raylib/install"
```

If your package manager uses a CMake toolchain file, supply `-DCMAKE_TOOLCHAIN_FILE=/path/to/toolchain.cmake` instead. To build everything:

```bash
cmake -S CompPhysics -B CompPhysics/output/build -DCMAKE_BUILD_TYPE=Release
cmake --build CompPhysics/output/build --config Release
```

### Existing local raylib builds

The direct compiler launcher is available for macOS/Linux setups with an already compiled vendored raylib library:

```bash
./CompPhysics/scripts/run --list
./CompPhysics/scripts/run --build-only projectile_drag_viz
./CompPhysics/scripts/run projectile_drag_viz
./CompPhysics/scripts/run --install-launchers
./CompPhysics/output/launchers/projectile_drag_viz
```

It uses `output/deps/raylib/libraylib.a` (or a library already built in the vendored source), writes executables to `output/native/`, and command shortcuts to `output/launchers/`. The old `run` and `apps/` paths have moved into these folders. The Python launcher uses CMake and does not require this local archive.

Vendored raylib is a Git submodule. To obtain it on a new checkout, use `git submodule update --init --recursive`. For normal CMake builds the separately installed raylib package is sufficient.

## Python programs

Create an isolated environment, activate it, and install dependencies:

```bash
python3 -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then:

```bash
python -m pip install -r CompPhysics/config/requirements.txt
python explore.py list "Python script"
python explore.py run vision/two_hand_bridge.py
```

Some utilities use only the standard library. Individual scripts may need additional platform-specific packages. Camera programs need a webcam and permission. Hand-input bridges may need a separate C++ scene: read the source or topic README for pairings and controls.

Forward program arguments after its name, quoting paths with spaces:

```bash
python explore.py run "cybertools/network/Certificate Expiry Checker.py" --help
python explore.py run "cybertools/network/Certificate Expiry Checker.py" --file /absolute/path/to/hosts.txt
```

The certificate checker's input is one hostname per line. Some utilities modify files; inspect their inputs and source before running them.

Programs launched through `explore.py` start at the repository root. The launcher routes Python bytecode and matplotlib caches under `CompPhysics/output/cache/`. Bundled camera models live in `resources/models/`, live-control messages in `output/runtime/`, and recorded observation databases in `output/data/`. User-supplied output arguments still select their requested locations.

## Tests and graphics checks

```bash
python -m pip install -r CompPhysics/config/requirements-dev.txt
python CompPhysics/scripts/check.py
```

Tests cover launcher behavior, folder/resource paths, Python numerical calculations and rendering-independent C++ checks. C++ checks need a C++17 compiler and skip if none is available. Pytest caches live in `output/cache/pytest/`.

Desktop graphics checks are opt-in. With the local raylib archive available:

```bash
./CompPhysics/scripts/run --build-all
python CompPhysics/tests/graphics_smoke.py
```

They use `output/native/` binaries and write previews, logs and a report into `output/reports/`.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Unsure what is installed | Run `python3 explore.py doctor` |
| CMake or compiler missing | Install them and reopen a terminal configured for the compiler |
| raylib CMake package missing | Install raylib development files or set its prefix/toolchain as above |
| Moved CMake cache fails | Start a fresh build in `output/build/`; old caches are preserved in `output/legacy/` |
| Python import fails | Activate the environment and install requirements with that environment's Python |
| Multiple program matches | Use the full source-relative path from `list` |
| Supporting module cannot run | Choose a standalone script or C++ demo |
| Webcam or window fails | Use a local graphical desktop and check camera permission |
