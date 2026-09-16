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
