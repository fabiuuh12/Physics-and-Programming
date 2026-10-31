# Physics and Programming Collection

Physics simulations, camera experiments, mathematics tools, and CyberTools all live together here. The folder layout separates project code, reusable resources, dependencies, configuration, documentation and generated outputs.

| Folder | Purpose |
| --- | --- |
| [src](src/README.md) | All project-authored code: simulations, CyberTools, vision, optics and shared helpers |
| [resources](resources/README.md) | Bundled fonts, camera models and curated demo media |
| [config](config/README.md) | Python dependency lists |
| [scripts](scripts/README.md) | Build/run launchers and catalog maintenance |
| [docs](docs/STRUCTURE.md) | Setup guides, folder map, portfolio notes and reference documents |
| [tests](tests/) | Numerical regression and optional graphics checks |
| [vendor](vendor/) | External raylib sources and headers |
| [output](output/README.md) | Builds, binaries, shortcuts, caches, runtime data and reports |

From the repository root:

```bash
python3 explore.py
python3 explore.py list cybertools
python3 explore.py run three_body_problem_viz
```

Read [Getting started](docs/GETTING_STARTED.md) for installation, the [program catalog](docs/CATALOG.md) for every file, and the [folder map](docs/STRUCTURE.md) for the complete structure.

## Good first physics demos

| Demo | Explore |
| --- | --- |
| `projectile_drag_viz` | Compare trajectories with and without drag |
| `damped_forced_oscillator_viz` | Damping, forcing and resonance |
| `three_body_problem_viz` | Orbits and gravitational forces |
| `quantum_tunneling_viz` | Wavepacket evolution and scattering |
| `hohmann_transfer_viz` | Transfer geometry and orbital maneuvers |

These are educational models. Their assumptions and visual scales vary by program. Selected demos are documented below; other demos show controls in their windows or source files.

## Aerodynamics Wind Tunnel Lab

Run `./CompPhysics/scripts/run aerodynamics_viz` from the repository root.

- Drag to orbit; wheel to zoom. `1` selects side view; `2` selects perspective.
- Left/right changes wind speed; up/down changes body scale; `[` / `]` adjusts the roof.
- `F` toggles drag and vertical aerodynamic force; `N` toggles their resultant.
- `C` switches between surface flow-speed colors and shaded blue bodywork; `T` toggles trails and wake ribbons.
- `-` / `=` adjusts the shared force-arrow scale. `P` pauses flow; `R` resets experiment parameters and particles.

The fixed body experiences drag in the wind direction. Vertical aerodynamic force
can change sign as the roof parameter changes. The dashboard uses an educational
coefficient model: `q = 0.5 rho U²`, `D = q A Cd`, and `L = q A Cl`, with
`rho = 1.225 kg/m³` and `A = 2.2 scale² m²`. Coefficients are illustrative,
not measured or integrated from the flow field. The resultant combines these two
aerodynamic loads; the mount balances them. Surface colors represent relative
flow speed, not pressure. Particle motion is an illustrative flow field, not CFD.

## Three-body Orbital Lab

Run `./CompPhysics/scripts/run three_body_problem_viz` from the repository root.

- Drag to orbit; wheel to zoom; `O` selects a near-top view.
- `1` / `2` / `3` selects Lagrange Triangle, Braided Chaos, or Binary Capture.
- `Tab` cycles the inspected body; the dashboard's A/B/C buttons also select it.
- `V` toggles cyan velocity arrows for all bodies. `F` toggles dashed gold gravitational-force arrows for the inspected body. `A` toggles its purple net-acceleration arrow.
- `T` toggles fading trails and direction arrowheads; `B` toggles the barycenter.
- `[` / `]` changes arrow size; `-` / `+` changes simulation speed. `P` pauses; `R` resets the current preset.

Arrows use the actual velocity and softened gravity model. The two pairwise forces
sum to the selected body's mass times its net acceleration. Velocity, force, and
acceleration have separate display scales, shown on the dashboard, in arbitrary
simulation units. Their arrow lengths should only be compared within a vector type.
Arrows longer than 4.5 scene units are capped and their labels receive an asterisk.
Trail arrowheads follow the chronological direction of the recorded trajectory.
The equal-mass triangle starts equilateral with the circular speed appropriate to
the softened model; it is a symmetric orbit, not a claim of perturbation stability.

## Quasar Core and Wormhole Gateway

Launch either refreshed scene from the repository root:

```bash
./CompPhysics/scripts/run quasar_core_viz
./CompPhysics/scripts/run wormhole_gateway_viz
```

Both scenes have a dedicated viewport, clickable display controls, particle trails,
and motion arrows calculated from their animated trajectories. Drag inside the
viewport to orbit; scroll to zoom. `V` toggles arrows; `G` toggles guides (quasar)
or the translucent surface (wormhole). `,` / `.` adjusts arrow size; `P` pauses;
`R` resets the experiment. The original parameter controls remain available in
the bottom help strip.

Quasar: `1`–`4` selects views, `T` toggles trails, and `F` triggers a flare. Gold
arrows follow the disk's rotation and radial/vertical oscillations; cyan arrows
follow the outward jet packets. Disk and jet display scales are separate. Jet
radial contraction now uses elapsed time rather than a per-frame multiplier.

Wormhole: `1`–`3` selects views, `4` / `T` starts a transit, `L` toggles trails,
and `M` changes the destination palette. Cyan arrows combine axial flow, swirl,
and the changing tube radius. Gold/purple arrows distinguish energy streaks in
opposite directions. Transit progress and the entrance/exit pulses have separate
readouts. The default view shows both mouths and the throat.

Arrows are capped at 2.6 scene units for readability. These are illustrative
animations in arbitrary scene units: the quasar does not solve GR or MHD, and
the wormhole is conceptual geometry, not a physical spacetime solution.
