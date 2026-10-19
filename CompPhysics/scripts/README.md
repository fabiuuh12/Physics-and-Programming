# Project scripts

Run from the repository root:

| Script | Purpose |
| --- | --- |
| `run` | Directly compile and run a C++ demo using an existing local raylib archive |
| `launch-viz` | Internal dispatcher for generated `output/launchers/` shortcuts |
| `check.py` | Run numerical and layout tests without creating source-local bytecode caches |
| `update_catalog.py` | Regenerate the linked program catalog |

The shared menu is `python3 explore.py`; it handles Python scripts and CMake-based C++ builds. The direct launcher is an alternative for existing macOS/Linux local builds.

```bash
./CompPhysics/scripts/run --list
./CompPhysics/scripts/run --build-only projectile_drag_viz
./CompPhysics/scripts/run projectile_drag_viz
./CompPhysics/scripts/run --install-launchers
python3 CompPhysics/scripts/update_catalog.py
```

Build outputs and shortcuts stay under `output/`. See [Getting started](../docs/GETTING_STARTED.md) for prerequisites.
