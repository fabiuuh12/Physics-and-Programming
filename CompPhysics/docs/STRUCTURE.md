# Folder map

Everything in the collection stays under `CompPhysics/`. The repository root contains the shared `explore.py` entry point and repository/editor configuration.

```text
CompPhysics/
├── src/                  Code you edit and run
│   ├── simulations/      Physics and mathematics, grouped by subject
│   ├── cybertools/       Security, network, forensics and file utilities
│   ├── vision/           Camera experiments and hand-input bridges
│   ├── optics/           Shared Python optics scenes
│   └── common/           Shared C++ graphics and numerical helpers
├── resources/            Inputs bundled with the project
│   ├── fonts/
│   ├── models/           MediaPipe camera models
│   └── media/            Curated screenshots and recordings
├── config/               Python dependency lists
├── scripts/              Build/run shortcuts and catalog maintenance
├── docs/                 Guides, portfolio notes and reference documents
├── tests/                Numerical regression and optional graphics checks
├── vendor/               External raylib sources and headers
├── output/               All local generated files
│   ├── build/            Active CMake build tree
│   ├── native/           Direct-compiler executables
│   ├── launchers/        Generated command shortcuts
│   ├── deps/             Compiled local dependencies
│   ├── cache/            Python, pytest and matplotlib caches
│   ├── runtime/          Live camera-control messages
│   ├── data/             Recorded observation databases
│   ├── reports/          Graphics previews and check reports
│   └── legacy/           Preserved older builds and binaries
├── CMakeLists.txt        C++ build entry point
└── README.md             Collection overview and selected demo controls
```

`src/` contains project code. `resources/` contains reusable input files. `vendor/` contains external code. `output/` contains generated files and is excluded from Git, apart from its [directory guide](../output/README.md). New output directories are created when needed.

Existing build directories and binaries were moved intact into `output/legacy/`. Their CMake caches may contain old absolute paths: create a fresh build in `output/build/`. Existing command shortcuts were moved to `output/launchers/` and their links were repaired. Existing caches and Finder metadata were preserved under `output/cache/`.

Short launcher names still work. Older topic-relative launcher arguments, such as `gravity/three_body_problem_viz.cpp`, and `CyberTools/...` spellings also resolve. Direct file paths must use the new locations.

## Adding files

- Put physics demos in `src/simulations/<topic>/`, utilities in `src/cybertools/<category>/`, and camera bridges in `src/vision/`.
- Register standalone C++ programs with `add_executable(target "src/.../program.cpp")` and raylib linking in `CMakeLists.txt`.
- Give standalone Python scripts a conventional `if __name__ == "__main__":` entry point. Other Python files appear as supporting modules in the catalog.
- Put shared fonts, models, and curated images in `resources/`. Put generated data, recordings, and reports in `output/`.
- Put technical notes in `docs/` or a topic README. The mathematics reference book is under [`docs/reference/mathematics/`](reference/mathematics/).
- Regenerate the file index with `python3 CompPhysics/scripts/update_catalog.py` from the repository root. The live launcher discovers files automatically.
