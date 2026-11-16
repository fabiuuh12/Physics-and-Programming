# Particle physics simulations

## Collider event display

`particle_acc_viz.cpp` is a real-time, orbitable cutaway of a general-purpose
particle detector. Two proton bunches meet at the interaction point, generated
particles curve through a solenoidal magnetic field, and energy deposits appear
in the detector subsystem they reach. The sequence repeats with a reproducible
new event suitable for screen recording.

During propagation, luminous particle heads make the event direction readable.
High-momentum tracks use a soft halo and bright core; photons remain straight and
gold, muons form clean penetrating violet tracks, and electrons or hadrons create
distinct compact showers when they reach their calorimeter layers.

Run it from the repository root:

```bash
./CompPhysics/scripts/run particle_acc_viz
```

Controls:

- **Space/P** pauses the presentation; **R** restarts the current event.
- **N** generates a new collision.
- **E** or the energy button cycles the collision energy through 3.5, 7.0, and
  13.6 TeV. Higher energy increases the chance of a displayed Higgs candidate.
- **C** switches between the cinematic camera and a free camera. In free-camera
  mode, drag to orbit and use the wheel to zoom.
- **W** or the beam button switches the two incoming proton bunches between
  particle clouds and localized Gaussian wave packets with internal 3D phase.
- **1/2/3** selects slow, normal, or double speed.
- **Left/Right** changes the displayed solenoid field strength.
- **D** hides detector geometry and **T** hides reconstructed tracks.
- **H** reveals or hides the interface. It starts hidden for a clean simulation
  and recording view; all keyboard controls remain active without it.

The visualization is an educational event display rather than a full Monte
Carlo detector simulation. Event species and momenta are generated from a fixed,
authored distribution. Charged trajectories preserve the important relationship
between curvature, charge sign, transverse momentum, and magnetic-field strength;
neutral particles travel straight. Electromagnetic particles stop near the
electromagnetic calorimeter, hadrons reach the hadronic calorimeter, and muons
continue into the outer chambers. Display units, material interactions, showers,
decays, detector inefficiencies, and reconstruction uncertainty are simplified.
The wave-beam view is an interpretive visualization of the incoming quantum
states, not a literal macroscopic helix traveled by each proton. After the
collision, the display returns to conventional reconstructed detector tracks.

Higgs candidates are intentionally enriched so they appear often enough for an
interactive demonstration; this is not the physical per-collision production
rate. Because the Higgs decays essentially at the interaction point, the display
does not draw a Higgs track. It instead highlights a two-photon or four-lepton
final state and reports a reconstructed candidate mass near 125 GeV.

The implementation separates deterministic event physics in `collider/model.h`
from GPU-facing drawing in `collider/renderer.h`. Detector tiles use cached meshes
and instanced batches. Tracks share one immediate-mode line batch, glow particles
share one texture, and the 3D scene renders at 88% resolution before a lightweight
bloom and tone-mapping pass. The interface remains native-resolution. This keeps
the cinematic view responsive while preserving anti-aliased detector edges.

For automated visual checks, set `COMPPHYSICS_COLLIDER_TIME` to a point in the
11.5-second sequence and `COMPPHYSICS_SMOKE_FRAMES` to the number of frames before
capture. The preview is written under `CompPhysics/output/reports/`.

Other programs in this folder explore Feynman diagrams, entanglement, and Higgs
field concepts.
