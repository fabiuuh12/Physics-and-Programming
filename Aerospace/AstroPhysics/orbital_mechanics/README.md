# Orbital Mechanics Visualizations

Interactive C++/raylib demos for orbital design and aerospace mission intuition.

## Demos

| Target | Source | Focus |
| --- | --- | --- |
| `hohmann_transfer_viz_cpp` | `hohmann_transfer_viz.cpp` | Circular parking orbit, target orbit, transfer ellipse, burn points, phase angle, and delta-v budget |
| `launch_window_porkchop_viz_cpp` | `launch_window_porkchop_viz.cpp` | Departure/arrival date trade space, synthetic C3 contours, time of flight, and best-window marker |

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

