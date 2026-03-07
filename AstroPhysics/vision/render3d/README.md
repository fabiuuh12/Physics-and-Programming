# Vision 3D Renderer

This folder is the right place for high-quality C++ rendering used by the hand-tracking Python app.

## Why here
- It is under `vision/`, so it stays close to `hand_planet_overlay.py`.
- It keeps render-only C++ code separate from Python tracking logic.
- It matches the existing `vision` executable pattern in `CMakeLists.txt`.

## Current file
- `astro_interactions_renderer_3d.cpp`
  - UDP-bridge 3D renderer for:
    - Floating hand avatars (left/right)
    - Planet
    - Star
    - Black hole
    - Pair interactions (including star+star supernova)
  - Receives state packets from `hand_planet_overlay.py` on UDP `127.0.0.1:50506`
  - Objects are rendered above hand anchors to mimic floating-over-hand AR behavior.

## Controls
- `Z`: cycle left object
- `X`: cycle right object
- Left body move: `A D W S R F`
- Right body move: arrow keys + `PAGE_UP/PAGE_DOWN`
- Mouse drag: orbit camera
- Mouse wheel: zoom
- `Q`: quit
- Note: manual controls are active only when UDP bridge packets are not arriving.

## Build
From repo root:

```bash
bash AstroPhysics/tools/build_vision_astro_renderer.sh
```

This uses `AstroPhysics/cmakew.sh` to find CMake even if it is not in your shell `PATH`.
If CMake is unavailable, it falls back to direct `clang++` build using your existing `build-native/raylib-local` object cache and creates a symlink in `build-cmake/`.

## Run (bridge mode)
1. Start renderer:
```bash
./AstroPhysics/build-cmake/vision_astro_interactions_3d_cpp
```

Equivalent fallback output path:
```bash
./AstroPhysics/build-native/vision_astro_interactions_3d_cpp
```

2. In another terminal, start Python tracker (bridge is enabled by default):
```bash
python3 AstroPhysics/vision/hand_planet_overlay.py
```

## Bridge env vars (Python sender)
- `ASTRO_RENDER_BRIDGE=0` disables UDP sending.
- `ASTRO_RENDER_BRIDGE_HOST` defaults to `127.0.0.1`.
- `ASTRO_RENDER_BRIDGE_PORT` defaults to `50506`.
