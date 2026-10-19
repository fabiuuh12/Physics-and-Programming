# Generated output

Everything in this folder except this guide is local and excluded from Git.

| Folder | Contents |
| --- | --- |
| `build/` | Active CMake build tree and executables |
| `native/` | Executables produced by the direct compiler launcher |
| `launchers/` | Generated command shortcuts |
| `deps/` | Local compiled dependency libraries and object files |
| `cache/` | Python, pytest, matplotlib and preserved older caches |
| `runtime/` | Live hand-control messages |
| `data/` | Locally recorded observation databases |
| `reports/` | Graphics previews, logs and validation reports |
| `legacy/` | Preserved pre-reorganization builds and binaries |

The old build folders were moved intact into `legacy/`. CMake build caches can contain absolute paths, so use a fresh build in `build/` rather than trying to continue one of those moved builds. Nothing in `legacy/` is needed by the launcher. Keep it until you no longer need the old outputs.
