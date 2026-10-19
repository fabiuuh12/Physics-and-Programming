# Repository map

| Location | What belongs here |
| --- | --- |
| `explore.py` | Shared entry point: menu, search, file information, setup checks and launch commands |
| `docs/` | Getting-started instructions, navigation and the program catalog |
| `CompPhysics/simulations/<topic>/` | Physics demos grouped by subject; see the catalog for the complete topic list |
| `CompPhysics/vision/` | Camera programs, hand-input bridges and their model assets |
| `CompPhysics/optics/` | Shared Python optics scenes used by vision runners |
| `CyberTools/<category>/` | Security, network, forensics and file utilities |
| `CompPhysics/common/` | Shared C++ rendering helpers and numerical models |
| `CompPhysics/assets/` | Bundled fonts and other runtime assets |
| `CompPhysics/tests/` | Numerical regression tests and opt-in graphics checks |
| `CompPhysics/third_party/` | External raylib code; separate from project-authored simulations |
| `CompPhysics/CMakeLists.txt` | Authoritative mapping between C++ demo sources and build targets |
| `CompPhysics/requirements*.txt` | Python runtime and development dependencies |
| `media/` | Place screenshots and demo recordings here |
| `.github/workflows/` | Automated Python tests and a representative macOS build |

## Source versus generated files

Edit topic sources and shared helpers. Build directories (`build*`), `bin/`, Python caches and virtual environments are generated local files. `apps/` contains convenience launcher links; source files live in the topic folders. Build directories and convenience links remain local. Physics source folders now live under `simulations/`; CyberTools lives at the repository root. Short demo names and previous topic-relative paths still resolve through `explore.py`. Direct source-file paths must use the new locations.

## Adding a program

For C++, add the source in the appropriate topic folder and register an `add_executable(target "simulations/topic/source.cpp")` with raylib linking in `CMakeLists.txt`. The launcher discovers those declarations directly.

For Python, put a standalone script in the relevant topic folder and use the conventional `if __name__ == "__main__":` entry point. Python files without that guard are listed as supporting modules. Keep reusable functions in modules and avoid work at import time.

Document required inputs, dependencies, controls and model assumptions in the source or a topic README. Regenerate the linked catalog with `python3 tools/update_catalog.py` after adding or renaming files. The live launcher discovers programs on every invocation.

## CompPhysics at a glance

```text
CompPhysics/
├── simulations/       Physics demos grouped by subject
├── vision/            Camera input and hand-tracking bridges
├── optics/            Shared Python optics scenes
├── common/            Shared C++ graphics and numerical models
├── assets/            Fonts and runtime resources
├── tests/             Numerical and graphics checks
├── third_party/       External dependencies
├── CMakeLists.txt     C++ build targets
└── README.md          Physics overview and selected demo controls
```

The [simulation index](../CompPhysics/simulations/README.md) explains the subject folders. [CyberTools](../CyberTools/README.md) has its own entry page. Local build and cache folders may also appear on your computer; they are excluded from Git.
