# Body particle lab

A live webcam experiment: your silhouette is the shape of a simplified material
target. Simulated alpha, beta and gamma beams pass through, scatter or stop in
that shape. Pose segmentation runs locally using the bundled MediaPipe model.
This is a visualization of transport in matter, not a scan of your tissues or a
measurement of radiation or dose.

## Run

From the repository root:

```bash
python3 explore.py run body_particle_camera
```

Requirements are already listed in `config/requirements.txt`: OpenCV, NumPy,
MediaPipe and Pillow. The app reuses the bundled pose model and Inter font, so
there are no model downloads. On macOS, allow camera access for the application
used to launch Python. Try `--camera 1` if the camera index differs.

Stand back enough for your shoulders, torso and arms to be visible. A thin
outline marks the tracked target. The beam initially aims at the silhouette's
center; click/drag the scene to aim it manually. A lost or stale pose clears the
target and tracks, then pauses emission until tracking returns. A live failure
never silently substitutes the demo silhouette.

| Control | Action |
| --- | --- |
| Alpha / Beta / Gamma tabs, or `1` / `2` / `3` | Select particle type |
| Energy slider | Change monoenergetic beam energy |
| Click/drag the camera view | Place the beam source and aim horizontally |
| `Space` | Pause/resume particles; camera tracking keeps running |
| Slow-motion button / `M` | Quarter-speed animation |
| `R` | Clear tracks/counts and restore automatic aim |
| Controls / `?` | Show/hide shortcuts |
| `F` | Fit the window to the usable display |
| `Q` / `Esc` | Quit; Esc first closes an open Controls panel |

Changing particle type or energy clears particles and counts so the comparison
never mixes configurations. **Through** counts tracks that entered the
target but had no modeled deflection or stopping, **Scattered** counts tracks
that leave the view after scattering, and **Stopped** counts tracks terminated
in the target. A particle must exit the view for through/scattered classification.
Misses and expired tracks are separate internal counters, excluded from these
three categories. These are event counts, not dose or exposure rates.

The normal app makes no recordings. `--snapshot` explicitly saves the final
frame and `--report` explicitly saves counts/model settings as JSON.

## Model

- **Geometry:** the segmented body/clothing outline becomes a homogeneous 2D
  water-like target at 1 g/cm³. Each image pixel is assigned 0.05 cm of material
  along the simulated path. This is an arbitrary model scale, not a measurement
  of your body size, depth, skin, bones, organs or composition. The exterior is
  treated as **vacuum**, so alpha particles can reach the target regardless of
  their drawn distance from the source. Air stopping is not simulated.
- **Alpha (2–10 MeV):** an illustrative range law `R = 0.0035 (E/5)^1.5 cm`
  sets a subpixel stopping distance, anchored at a 35 μm range for a 5 MeV alpha.
  The particle stops close to the boundary. The outline and flashes are enlarged
  for visibility; they do not depict microscopic tracks at literal screen scale.
- **Beta electrons (0.1–3 MeV):** the empirical Katz–Penfold mass-range expression
  provides a path-length budget at unit density. Below 2.5 MeV,
  `R = 0.412 E^(1.265 - 0.0954 ln(E)) cm`; above it, `R = 0.530 E - 0.106 cm`.
  Remaining energy is estimated by inverting this range law. A qualitative
  stochastic angular walk depicts multiple scattering. Applying this empirical
  range as a path budget with the angular walk is an educational approximation,
  not ESTAR transport or a validated penetration-depth model. The beam has a
  selected electron energy rather than a radioactive beta-decay spectrum.
- **Gamma photons (0.05–1 MeV initial energy):** log-log interpolation of NIST
  liquid-water total attenuation coefficients sets exponential optical depths.
  Free-electron Klein–Nishina cross sections and angular sampling model Compton
  events, with outgoing photon energy `E' = E/[1+(E/0.51099895)(1-cos(theta))]`.
  The free-electron Compton fraction is capped at the total attenuation rate.
  Residual channels are grouped as removal from the tracked beam; the **Stopped**
  category does not distinguish photoelectric absorption from other removal
  channels, including coherent removal. Secondary electron tracks, atomic binding,
  3D angular projection and detailed tissue response are not reconstructed.
  Initial energies stay below pair-production threshold.
- **Display:** path integration uses steps up to 0.5 pixels. Subpixel alpha ranges
  stop within a step; silhouette boundary resolution remains pixel-limited.
  All types travel at 300 image pixels per animation second, unrelated to their
  real speeds. Color, glow and event flashes are display choices. Slow motion
  scales animation time. Track lifetime and particle-count caps keep the live
  scene responsive; timed-out tracks are not reported as transmitted particles.

A moving silhouette is a changing material geometry. Counts from this interactive
view are not a calibrated attenuation experiment. A separate static-slab test
checks unscattered gamma transmission against `exp(-mu * thickness)`.

## Sources

- [NIST liquid-water photon attenuation coefficients](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html): the numerical attenuation samples embedded in the model.
- [CERN, Introduction to detectors](https://cds.cern.ch/record/292287/files/B00004683.pdf): empirical electron range relations and scattering context.
- [NIST stopping-power databases](https://www.nist.gov/pml/stopping-power-range-tables-electrons-protons-and-helium-ions): reference methods for more complete charged-particle models; the app does not claim to implement these databases.
- [NRC radiation basics](https://www.nrc.gov/facilities-safety/radiation-protection/radiation-and-its-health-effects/radiation-basics): qualitative differences between radiation types.

## Camera-free preview

```bash
python3 CompPhysics/src/vision/body_particle_camera.py --demo
python3 CompPhysics/src/vision/body_particle_camera.py --demo --type beta --energy 3
python3 CompPhysics/src/vision/body_particle_camera.py --demo --headless --type gamma --frames 180 --snapshot CompPhysics/output/reports/body_particles_gamma.png --report CompPhysics/output/reports/body_particles_gamma.json
```

The generated moving silhouette is explicitly labeled Demo. It exercises the
same particle model and display code. Add `--ui-scale 2` for a Retina-sized image,
`--show-controls` to preview the controls panel, or `--seed 42` for a reproducible
particle history.
