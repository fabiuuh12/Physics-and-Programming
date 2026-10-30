# Particle Studio

One hand-controlled app with two modes: a magnetic track chamber and a toy
particle-collision display. It uses the existing webcam/MediaPipe hand tracker
and a fitted Retina interface. Nothing is recorded automatically.

```bash
python3 explore.py run particle_studio
```

Dependencies are already listed in `config/requirements.txt`. The bundled hand
model and Inter font are reused; no download is needed. Allow webcam access for
your Python launcher if prompted. `--camera 1` selects another camera and
`--no-hands` keeps the webcam but uses mouse/keyboard input only. If hand tracking
cannot initialize, the camera and mouse controls remain available.

## Tracks

Hold a **right-hand thumb/index pinch** to release particles. Their release
point follows the pinch midpoint, and the wrist-to-index direction sets their
initial direction. Pinch emission is limited to one packet every 0.32 seconds.
Spread the **left thumb/index** to increase magnetic-field strength. The left
hand changes the field continuously; a manual field adjustment has a brief
1.5-second override so the slider remains usable.

Positive tracks are amber and negative tracks blue. Reverse the field and their
bending reverses. `P` cycles electron/positron, muon/antimuon, proton/antiproton,
and neutral photon packets. Photons travel straight in the magnetic field.
The species and momentum controls affect new releases. Existing particles keep
their momentum magnitude and species while responding to the current field.

A few sample packets appear when this mode opens so the detector view is
immediately visible. Click anywhere in the chamber or press Enter for mouse-only
release. Hand release positions are limited to the 3.6 m inner model region.

## Collisions

Pinch **both hands while they are apart**, then bring the pinches together.
The gesture arms above a separation of 23% of the image width and fires below
12%. Release both pinches to rearm. Losing a hand cancels an armed gesture and
requires an observed release before another event can be armed. The gesture
holds the magnetic-field setting while the hands converge.

Two neutral, back-to-back beams approach a vertex, then six charged pion tracks
emerge. The event's center-of-mass energy is adjustable between 2 and 12 GeV.
Random final-state 2D momentum vectors are centered to sum to zero, then given a
common scale so their relativistic on-shell energies sum to the input energy.
Three positive and three negative charges balance the neutral incoming state.

This is a **kinematic toy model**, not an LHC/Standard Model event generator.
It does not calculate cross sections, matrix elements, real particle yields,
showers, decays or a uniform Lorentz-invariant phase-space distribution. The
random angular/multiplicity choices are illustrative. Conservation checks refer
to the collision vertex. The original event snapshot is retained independently
of subsequent magnetic track bending; the external field changes the particles'
instantaneous total momentum.

An example event starts when Collision mode is opened. Press **Collide**, click
the scene, or press Enter for another event. Each new event clears the previous
tracks. A gesture vertex stays within the central 1.5 m model region.

## Controls

| Input | Action |
| --- | --- |
| Tracks / Collisions tabs, or `1` / `2` | Select mode |
| Right pinch held | Release track packets |
| Left thumb/index spread | Field strength |
| Both pinches apart, then together | Trigger a collision |
| Release / Collide button, click scene, or Enter | Manual release/event |
| Field slider | Set field magnitude, 0–3 T |
| Reverse B / `V` | Reverse field direction |
| Momentum slider (Tracks) | 0.08–2 GeV/c for new tracks |
| Next event slider (Collisions) | 2–12 GeV CM energy for the next event |
| `P` | Cycle track species |
| `C` | Toggle faint full-camera background; small camera preview remains |
| `Space` | Pause motion and gesture actions |
| `R` | Clear tracks, hits and pending event |
| `F` | Fit window to usable display |
| Controls / `?` | Show or hide shortcuts |
| `Q` / Esc | Quit; Esc first dismisses an open Controls panel |

## Track model

The chamber is a planar transverse slice of a uniform magnetic field in vacuum.
Position is in meters, momentum in GeV/c, rest mass in GeV/c² and field in tesla.
Positive field points out of the image (dots); negative field points into it
(crosses). World y points upward. The curvature radius is

`R = p_T / (0.299792458 |q| |B|)`.

Each timestep uses the exact constant-field arc. Momentum magnitude and
`E = sqrt(p² + m²)` remain constant. Every species uses the same slowed clock,
with c drawn as 3 model meters per animation second; these are not real transit
times. Changing B interactively applies an instantaneous magnetic-only field
setting and omits the electric field induced by a real changing field.

There is no gas transport, ionization energy loss, synchrotron radiation,
material scattering or particle decay. Tracks consequently form circular arcs
rather than physical energy-loss spirals. The detector rings flash at the first
geometric crossing by each track; they are illustrative layers, not simulated
silicon signals. Tracks outside 4 m or older than 6.5 animation seconds fade
from the display; this is a display lifetime, not a physical decay law.

References: [CERN, Particle Detection: Trackers](https://indico.cern.ch/event/44587/contributions/1108845/attachments/943247/1337924/Herten-Tracking1-Print.pdf),
[PDG, Particle Detectors at Accelerators](https://pdgweb.lbl.gov/2023/reviews/rpp2023-rev-particle-detectors-accel.pdf).

## Camera-free preview

```bash
python3 CompPhysics/src/vision/particle_studio.py --demo
python3 CompPhysics/src/vision/particle_studio.py --demo --mode collisions
python3 CompPhysics/src/vision/particle_studio.py --demo --headless --mode collisions --frames 90 --snapshot CompPhysics/output/reports/particle_studio_collisions.png --report CompPhysics/output/reports/particle_studio_collisions.json
```

The demo labels its camera placeholder **Demo** and emits example packets/events
at fixed intervals. It uses the same model and rendering as the live version.
`--ui-scale 2` creates a Retina-sized preview, `--show-controls` displays shortcuts,
and `--seed 7` makes toy events reproducible. `--snapshot` and `--report` are
explicit file outputs; normal live use saves no camera images or recordings.
