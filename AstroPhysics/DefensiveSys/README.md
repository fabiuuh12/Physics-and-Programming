# DefensiveSys

`DefensiveSys` is the dedicated module for defensive simulation prototypes.

## Current Layout

- Python hand bridge: `hand_turret_sim.py`
- C++ 3D sim: `defensive_sys_3d.cpp`

Python provides only hand input data; C++ owns rendering and game logic.

## Bridge Contract (UDP localhost)

Target: `127.0.0.1:50505`

Packet format (ASCII CSV):

`timestamp,left_valid,left_x,left_y,right_valid,right_pinch`

- `left_x`, `left_y` are normalized in `[0,1]`.
- `right_pinch` is binary (`1` when pinch is active).

## Run

1. Start Python hand bridge:
`python3 AstroPhysics/DefensiveSys/hand_turret_sim.py`

2. Build and run C++ sim target:
- CMake target: `defensive_sys_3d_cpp`

Example:
`cmake --build AstroPhysics/build-native --target defensive_sys_3d_cpp`
