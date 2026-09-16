# Fluid simulations

## Shallow-water sandbox

From the repository root:

```bash
./CompPhysics/run shallow_water_sandbox_viz
```

A 16 m by 12 m basin with an evolving 3D water surface. Start with a ripple,
release a wet-bed dam break, or watch flow pass through a narrow gate.

| Control | Action |
| --- | --- |
| Left click water | Create a localized mound by redistributing existing water |
| Right mouse drag / wheel | Orbit around the focus point / zoom |
| WASD | Translate the camera across the basin, relative to the view |
| Q / E | Move down / up |
| Middle drag or Shift + left drag | Pan in the screen plane |
| Shift + right drag | Pan across the horizontal plane |
| Shift | Move faster with the keyboard |
| F / T | Return to home view / look from above |
| Space / R | Pause / reset current scene |
| 0 / 1 / 2 / 3 | Still water / ripple / dam break / narrow gate |
| C / V | Toggle depth vs speed colors / velocity arrows |
| - / = | Halve / double simulation speed (0.25–2x) |
| H | Show / hide the compact controls guide |

The light experiment panel has clickable presets and surface-view controls.
The floating transport bar controls pause, reset, and help. Clicking these
panels does not disturb the water or rotate the camera.

The model solves the two-dimensional, depth-averaged shallow-water equations
on a 64 by 48 finite-volume grid. State variables are depth and the two
depth-integrated momenta. Rusanov fluxes transfer equal and opposite quantities
across shared faces; reflecting ghost states impose closed basin and obstacle
walls. An adaptive timestep satisfies a conservative two-dimensional CFL bound.
Gravity is 9.81 m/s². The HUD reports volume drift relative to the current preset.
Clicking raises a Gaussian mound and withdraws the same volume across wet cells.

This first version uses a flat bed and initially wet cells. It has no terrain
editing, drying/flooding, physical viscosity, breaking-wave overturning, or
vertical fluid dynamics. The first-order scheme numerically smooths waves;
volume conservation alone is not a measure of solution accuracy. Displayed
height variations are exaggerated 1.5x, and colors saturate at the legend limits.
The narrow gate is a fixed obstacle preset, already open when the scene starts.

Reference: [Clawpack's shallow-water dam-break example](https://www.clawpack.org/gallery/pyclaw/gallery/dam_break.html).

After building, run the numerical checks without opening a window:

```bash
./CompPhysics/build-native/shallow_water_sandbox_viz_cpp --self-test
```

Checks cover 1,200 steps per preset, positive depths, finite state variables, volume
conservation (including repeated disturbances), still-water equilibrium, and
transverse symmetry of the dam break. These are invariant checks, not a
grid-convergence study or validation against experimental data.

For a brief graphics smoke test (five frames, then exit):

```bash
./CompPhysics/build-native/shallow_water_sandbox_viz_cpp --smoke-test
```

This saves a screenshot to `/tmp/shallow_water_sandbox.png`.

## Other demos

- `fluid_mechanics_channel_viz`: tracers in a prescribed laminar velocity profile.
- `fluid_vortex_viz`: vortex visualization.

## Engine airflow lab

```bash
./CompPhysics/run engine_airflow_viz
```

Three independent 3D component tabs, with the shallow-water sandbox's light
control panel, orbit/pan camera, and floating playback bar:

1. **Turbo:** isolated compressor and exhaust turbine on a shared shaft. Fresh
   air enters axially, turns outward through the compressor, follows the volute,
   and leaves through the outlet. A separate hot exhaust route drives the turbine.
2. **Supercharger:** isolated **twin-screw supercharger** inspired by the supplied cutaway reference:
   a rectangular casing, long drive snout, pulley, timing gears, and opposite-hand
   helical rotors. The male rotor has three lobes and the female five grooves;
   the female rotates oppositely at 3/5 of the male speed. Air travels axially
   from the rear intake to a bottom discharge near the drive end. Shrinking
   colored markers illustrate internal compression. Profiles and flow paths are
   schematic, not conjugate manufacturing geometry or a sealed-pocket solution.
3. **Engine:** an enlarged single-cylinder four-stroke cutaway. Moving valves
   gate the intake and exhaust streams. Gas particles and ribbons inside the
   cylinder show intake, compression, power, and exhaust as the piston moves.

Moving directional arrows, flowing ribbons and traveling particles make the air visible through open
housings. Toggle the cutaway to see the exterior. Each tab shows only its own
component; no full engine plumbing or intercooler is included.

| Control | Action |
| --- | --- |
| Tabs / 1, 2, 3 | Turbo / supercharger / engine |
| Sliders | Drive or engine speed, drive load or throttle |
| Left or right drag / arrow keys / wheel | Orbit / orbit / zoom |
| Middle drag / Shift + left or right drag | Pan |
| WASD / Q, E | Move camera focus horizontally / vertically |
| Shift | Faster keyboard movement |
| F / T | Home / top view |
| Space / R | Pause / reset the current tab |
| C | Open or close the cutaway |
| V / P / G | Toggle ribbons / particles / direction arrows |
| - / = | Animation speed, 0.125–2x |
| H | Controls guide |

### Model and limits

The visualization uses **prescribed flow paths, not CFD**. The continuous ribbons
are an animation, not a computed fluid surface like the shallow-water solver.
It does not resolve turbulence, boundary layers, compressor maps, surge,
combustion, or valve pressure waves. Geometry and drive response are schematic.
Particles are visual markers, not conserved parcels of gas. Colors identify
flow stages, not a calibrated temperature field. Cylinder motion is slowed 60x;
rotor speeds are illustrative. No engine power predictions are made.

The turbo tab shows an illustrative outlet pressure and temperature.
Ambient conditions are 101.325 kPa and 298.15 K. Temperature uses
`T2 = T1 * (1 + (PR^(2/7) - 1) / 0.72)`; outlet pressure is ambient pressure
multiplied by PR. There is no downstream intercooler in these standalone views.
Drive speed is an engine-equivalent input, not actual turbo shaft RPM.

Turbo PR is `1 + 1.1 * spool`, where spool exponentially approaches
`clamp((RPM-1000)/4500 * load, 0, 1)` with a 1.1 s rise and 0.45 s fall time.
The twin-screw tab does not display compressor pressure/temperature estimates:
its shrinking markers illustrate decreasing pocket volume without solving
compression, leakage, or moving-boundary flow. The earlier centrifugal estimate remains only in the tested helper.
The turbo assumptions are illustrative, not calibrated component predictions. The engine's throttle changes
visual tracer travel speed; the gas illustration does not solve cylinder filling.

Physical background: [NASA compressor thermodynamics](https://www.grc.nasa.gov/www/k-12/airplane/compth.html)
, [Whipple twin-screw background](https://whipplesuperchargers.com/ft-2434-faq.html),
and [Garrett turbo components](https://www.garrettmotion.com/ja/turbocharger-technology/how-a-turbo-works/basic/).

After building:

```bash
./CompPhysics/build-native/engine_airflow_viz_cpp --self-test
./CompPhysics/build-native/engine_airflow_viz_cpp --smoke-test
./CompPhysics/build-native/engine_airflow_viz_cpp --smoke-test --supercharger
./CompPhysics/build-native/engine_airflow_viz_cpp --smoke-test --engine
```

Smoke checks save `/tmp/engine_airflow_tab_0.png`, `_1.png`, or `_2.png` and exit.
`--na` is retained as an alias for `--engine`. Numerical checks cover the existing
thermodynamic helpers, spool timestep independence, finite states, flow-route
continuity and endpoints, and piston stroke direction and periodicity. These
checks do not validate a CFD solution or measured engine performance.
