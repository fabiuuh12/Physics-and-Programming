# Spacetime Sandbox

An empty grid you can build on, in the direct scene style of
`gravity_well_grid_viz` and `solar_system_spacetime_viz`. Place several objects,
drag them around, and watch their wells combine. The default interface is just
the scene, an object picker, and short control hints.

```bash
./CompPhysics/scripts/run spacetime_geometry_viz
# Or use the repository menu:
python3 explore.py run spacetime_geometry_viz
```

## Play

Choose **Planet**, **Star**, **Black hole**, or **Wormhole** from the bottom
picker, or press `1`–`4`. Click an empty spot on the grid to place an object.
Click an existing object to select it; drag it to move it. New objects stay where
you place them. More massive objects make deeper wells, and nearby wells blend.
Up to 24 objects can be placed. Reaching the limit does not replace existing objects.

| Control | Action |
| --- | --- |
| Left click on empty grid | Place the chosen object |
| Left click / drag an object | Select / move that object |
| `1`–`4` / `Tab` | Choose / cycle the placement type |
| `+` / `-` | Increase / decrease the selected object's mass |
| `Delete` / `Backspace` | Remove the selected object |
| Right drag | Orbit the camera |
| Wheel | Zoom |
| Middle drag | Pan |
| `O` | Reset the camera |
| `R` | Clear all placed objects |
| `P` / `Space` | Pause surface and ring animation |
| `G` | Toggle three grid layers |
| `T` | Toggle the faint translucent surface |
| `[` / `]` | Decrease / increase grid deformation |
| `H` | Toggle the compact help overlay |

Placement intersects the deformed grid. A colored cursor ring previews the
placement point. Dragging uses a horizontal plane at the object's initial
height, keeping its horizontal motion stable while the surrounding grid changes.
Objects are kept within the inner grid. Orbiting and panning use separate mouse
buttons so they do not accidentally place objects.

The former clock, cone, ray, metric and settings panels have been removed.
The scene starts empty every launch; this version does not save layouts or
integrate object motion. Wormholes are visual mouths, not simulated passages.

## Visual model

This is an illustrative sandbox in arbitrary scene units. Each object adds a
softened radial well. The combined field passes through a smooth saturation
function to keep overlapping wells finite; the outer grid edge stays anchored.
Default masses and radii are tuned for interaction and visibility. They are not
calibrated Earth/Sun masses, event horizons, or physical spacetime metrics.

Planets use shaded procedural land/ocean colors, stars have animated warm
surfaces and a glow, black holes have dark centers and animated disk rings, and
wormhole mouths have violet rings. These are visual cues. The grid is not a
solution of Einstein's equations, and combining the wells does not superpose
exact relativistic spacetimes. Layered mode repeats the same deformed grid with
vertical reference lines.

## Checks and captures

```bash
python3 CompPhysics/scripts/check.py
python3 CompPhysics/tests/graphics_smoke.py --filter spacetime_geometry --frames 60
python3 CompPhysics/tests/spacetime_geometry_graphics.py
```

The numerical checks cover flat initialization, localized wells, symmetry,
mass response, combined deformation, finite overlapping wells, anchored edges,
movement bounds, mass limits, object limits and clearing. The desktop graphics
script checks the empty grid, four isolated object types and a mixed scene in
both grid modes. It requires Pillow from `config/requirements.txt`.

```bash
CompPhysics/output/native/spacetime_geometry_viz_cpp \
  --scene mixed --capture CompPhysics/output/reports/spacetime_sandbox.png
```

Capture options are `--scene empty|mixed|planet|star|blackhole|wormhole`,
`--lattice 0|1`, `--frames 1..1200`, and `--capture path.png`. The normal launch
always uses the empty scene unless a scene is explicitly selected on the command
line. Captures use a fixed 1/60 animation step; generated outputs stay under
`CompPhysics/output/`.
