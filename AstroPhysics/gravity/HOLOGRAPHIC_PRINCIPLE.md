# Holographic Principle Bridge

Folders:

- `AstroPhysics/vision/holographic_principle_bridge.py`: launches the existing webcam hand tracker bridge
- `AstroPhysics/vision/live_controls.txt`: bridge file written by Python
- `AstroPhysics/gravity/holographic_principle_viz.cpp`: C++ raylib visualization that consumes the bridge

Run in two terminals:

```bash
python3 AstroPhysics/vision/holographic_principle_bridge.py
```

```bash
./AstroPhysics/build-native/holographic_principle_viz_cpp
```

If you need to build the C++ target first:

```bash
cmake -S AstroPhysics -B AstroPhysics/build-cmake
cmake --build AstroPhysics/build-cmake --target holographic_principle_viz_cpp
```
