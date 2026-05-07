# Orbital Mechanics Visualizations

Interactive C++/raylib demos for orbital design and aerospace mission intuition.

## Demos

| Target | Source | Focus |
| --- | --- | --- |
| `hohmann_transfer_viz_cpp` | `hohmann_transfer_viz.cpp` | Circular parking orbit, target orbit, transfer ellipse, burn points, phase angle, and delta-v budget |
| `launch_window_porkchop_viz_cpp` | `launch_window_porkchop_viz.cpp` | Departure/arrival date trade space, synthetic C3 contours, time of flight, and best-window marker |
| `solar_system_orbit_planner_viz_cpp` | `solar_system_orbit_planner_viz.cpp` | KSP-style top-down solar-system planner with editable burns and predicted spacecraft trajectory |

## Controls

### Hohmann Transfer

- `LEFT/RIGHT`: adjust target orbit radius
- `UP/DOWN`: adjust parking orbit radius
- `A/D`: scrub spacecraft along the transfer ellipse
- `SPACE`: animate or pause the transfer
- `R`: reset

### Launch Window Porkchop

- `LEFT/RIGHT`: move the selected departure day
- `UP/DOWN`: move the selected arrival day
- `W/S`: change the synodic-period assumption
- `R`: reset

### Solar System Orbit Planner

- Drag from the spacecraft: draw a maneuver burn vector
- `ENTER`: apply the currently previewed burn
- `SPACE`: pause or resume simulation
- `+/-`: change time warp
- Mouse wheel: zoom
- `F`: toggle camera between Sun and spacecraft
- `R`: reset to an Earth-like starting orbit
