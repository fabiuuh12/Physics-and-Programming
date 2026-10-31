# Spacetime Geometry Lab

An interactive 3D grid for exploring spatial curvature, light paths, causal
cones and gravitational clock comparisons. The grid fills the scene; floating
controls let you change the geometry and inspect its consequences.

```bash
./CompPhysics/scripts/run spacetime_geometry_viz
# Also available in the numbered menu:
python3 explore.py run spacetime_geometry_viz
```

## Geometries

| Preset | Geometry and interpretation |
| --- | --- |
| Flat space | Straight null rays, flat grid, equal clock rates |
| Planet | Exterior Schwarzschild geometry, Earth-like `rs/R = 1.39e-9` |
| Star | Exterior Schwarzschild geometry, Sun-like `rs/R = 4.24e-6` |
| Black hole | Schwarzschild exterior, capture boundary and photon-sphere ring |
| Wormhole | Hypothetical ultrastatic Ellis metric, two regions joined at a throat |

These are **one central geometry at a time**. The lab does not superpose multiple
Schwarzschild metrics or evolve an Einstein-equation solution. Existing gravity
and spacetime programs remain available alongside this new experiment.

The reference source radius is one length unit for the planet and star presets.
`G = c = 1`; each preset has its own reference length, so a screen distance is
not a shared kilometer scale across the presets. The mass slider multiplies
`rs/R` while keeping that reference radius fixed. The two weak-field presets
retain almost straight light paths and almost equal clocks: visual height
magnification does not artificially strengthen their gravity.

## Controls

Drag the scene to orbit and scroll to zoom. The window can be resized; pointer
coordinates are mapped into the same layout used for drawing. Dragging controls
never rotates the camera. Press `H` for an in-app guide.

| Control | Action |
| --- | --- |
| `1`–`5` / top buttons | Choose geometry; reset clocks and geometry parameters |
| Mass / throat slider | Change central mass or wormhole throat radius |
| Display height slider | Exaggerate embedding height only |
| Probe radius slider | Move the static clock probe and sample causal directions |
| Beam offset slider | Change ray impact offsets; negative values mirror the beam |
| `G` | Single curved plane / three-layer lattice |
| `S` | Toggle translucent spatial surface |
| `L` | Toggle light paths and moving markers |
| `C` | Toggle causal inset |
| `K` | Toggle clock probe and elapsed-clock readouts |
| `F` | Single ray / nine-ray beam |
| `Space` | Pause animation and clock accumulation |
| `R` | Reset both elapsed clocks |
| `O` | Restore the camera, including the low wormhole viewing angle |
| `A` | Toggle automatic orbit |

Changing the geometry, mass or probe resets elapsed clocks. Changing display
height changes only the view. Toggling clock readouts does not stop the clocks.
The ray outcomes refer to the whole integrated guide path, not the current dot.
Dots loop along the guides for visibility; they are not calibrated flight times.

## What the grid means

For Schwarzschild geometry, the exterior constant-time equatorial spatial slice
has metric

```
dl² = dr² / (1 - rs/r) + r² dphi²
```

Its Euclidean embedding is a Flamm paraboloid. The unscaled height used here is

```
z(r) = 2 sqrt(rs (r-rs)) - 2 sqrt(rs (16-rs))
```

The outer reference rim is at `r = 16`. The grid excludes the material interior
or the region inside the horizon. The source spheres are schematic markers.
A black hole's horizon and the `r = 1.5 rs` photon sphere have different radii.

The height is an **extrinsic spatial embedding coordinate, not time**, and the
sphere is not pulling a physical fabric downward. In the planet and star
presets the height gain is explicitly magnified by thousands of times. Changing
the display-height slider distorts the embedding drawing but leaves the metric,
ray solver and clock calculations unchanged. Only gain 1 is isometric.

The 3D lattice repeats the same spatial slice at three vertical offsets, with
connecting reference lines. Those layers make deformation visible in volume;
they are not extra time dimensions or a reconstruction of the full 3D spatial
metric. The single-plane setting is the direct equatorial embedding view.

## Null rays and clocks

The Schwarzschild ray solver uses the exact equatorial null-orbit equation in
coordinate projection `x = r cos(phi), z = r sin(phi)`:

```
d²(x,z)/dlambda² = -3 rs L² (x,z) / (2 r⁵)
L = x dz/dlambda - z dx/dlambda
E² = |d(x,z)/dlambda|² - rs L²/r³
```

Fourth-order Runge–Kutta integration follows rays from a finite emitter line
`x = -14`. The displayed beam offset is that line's coordinate offset; it is
not precisely the asymptotic impact parameter `|L/E|`. Step size decreases near
the center. Paths stop at the material surface, just outside the horizon, at the
outer boundary or after a finite integration budget. A near-critical ray may
spend a long time orbiting before its ultimate fate is resolved. Guide paths are
lifted slightly off the surface to avoid flicker, and continue through the
background beyond the grid rim for orientation.

The clock at infinity accumulates the common simulation time. A static exterior
probe accumulates `d tau/dt = sqrt(1-rs/r)`. A static clock cannot be placed on or
inside a black-hole horizon, so the display says **No static observer** there.
The star and planet clock probe is kept outside the source surface because this
program does not implement their interior metrics. Elapsed values use normalized
time units, not a specified SI second shared between different presets.

## Causal inset

The inset is separate from the spatial embedding. Its vertical axis is the
horizon-penetrating time coordinate `T = v-r`, where `v` is ingoing
Eddington–Finkelstein time, and its horizontal axis is areal radius `r`.
The cone shows **radial** future null directions at the probe location:

```
inward:   dr/dT = -1
outward:  dr/dT = (r-rs)/(r+rs)
```

At the horizon the outward edge is vertical; inside it, both future edges point
toward smaller radius. These are coordinate slopes, not locally measured light
speeds. Every local freely falling observer still measures light speed `c`.
The diagram suppresses angular directions and draws a tangent cone at the
probe; the triangle is not an extended global light ray.

## Wormhole model

The Ellis metric is an ultrastatic, hypothetical traversable geometry:

```
ds² = -dt² + dl² + (l²+b²) dOmega²
r(l) = sqrt(l²+b²)
embedding height = b asinh(l/b)
```

The two signs of the proper radial coordinate `l` correspond to the two spatial
regions. Null trajectories integrate `dl/dt`, angular momentum and angle through
`l = 0`; no teleportation shortcut is used. There is no horizon and no static
gravitational clock-rate difference in this particular metric. The causal inset
uses a local proper-radial section with axes `t` and `l` and equal inward/outward
slopes. The probe slider continues to specify the areal radius in the upper region.

This wormhole requires exotic stress-energy and is not an observed object or
an Einstein–Rosen bridge. Rotating holes, time-dependent metrics, gravitational
waves, multiple sources and interior matter solutions are outside this version.

## Checks and captures

```bash
python3 CompPhysics/scripts/check.py
python3 CompPhysics/tests/graphics_smoke.py --filter spacetime_geometry --frames 60
python3 CompPhysics/tests/spacetime_geometry_graphics.py
```

The numerical suite checks the embedding slope, clock lapse, horizon cone,
straight-ray limit, reflection symmetry, capture, energy conservation, the
weak-deflection limit, the unstable photon orbit and Ellis throat traversal.
The optional desktop script renders fourteen profiles, covering both grid modes,
all presets, an interior-horizon probe, strong curvature and a thin throat. It
requires Pillow from `config/requirements.txt`; the simulation itself needs only
raylib and a graphical desktop. Outputs remain under `output/reports/`.

```bash
CompPhysics/output/native/spacetime_geometry_viz_cpp \
  --preset 4 --lattice 0 \
  --capture CompPhysics/output/reports/wormhole_grid.png
```

Additional capture options: `--mass .3..2.5`, `--warp .25..2`, `--probe .55..10`,
`--impact -6..6`, `--frames 1..1200`. Captures use a fixed 1/60 playback step.

## References

- [David Tong: Schwarzschild geometry and geodesics](https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S1.html)
- [David Tong: black holes and horizon-penetrating coordinates](https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S6.html)
- [Einstein Online: gravity, geodesics and curvature](https://www.einstein-online.info/en/spotlight/geometry_force/)
- [Ellis's original drainhole solution](https://doi.org/10.1063/1.1666161)
