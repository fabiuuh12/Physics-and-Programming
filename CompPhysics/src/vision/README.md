# Camera experiments

## Gravitational lensing camera

```bash
python3 explore.py run lensing_camera
```

Your mirrored webcam feed becomes a background source plane behind a simulated
point-mass lens. Move your **right** thumb/index midpoint to move the lens; spread
those fingers to increase lens strength. **Pinch your left thumb and index once**
to lock position and size immediately. Release, then make **two new quick left
pinches** (within 0.8 seconds, releasing between them) to resume following the
right hand. The locking pinch does not count toward unlocking. A smaller inset
shows the original image.

**Pinch with both hands and hold**, then move your hands apart to stretch the
source wavelength toward red, or together to compress it toward violet.
The lens position and radius hold during this gesture. Release both pinches
to return to lens controls; the chosen wavelength stays selected.
No target selection, mass entry or calibration is needed.

Uses existing dependencies: OpenCV, NumPy, MediaPipe and the bundled
`resources/models/hand_landmarker.task`. There are no model downloads. If hand
tracking cannot initialize, mouse controls remain available. On macOS, launch
from Terminal and allow Terminal under **Privacy & Security → Camera**.

```bash
python3 CompPhysics/src/vision/lensing_camera.py --camera 1
python3 CompPhysics/src/vision/lensing_camera.py --no-hands
```

### Controls

| Input | Action |
| --- | --- |
| Both thumb/index pinches held; vary distance between hands | Stretch/compress source wavelength |
| Right thumb/index midpoint | Lens position |
| Right thumb/index separation, normalized by palm width | Lens strength |
| Left thumb/index pinch | Lock lens position and size |
| Two new left pinches within 0.8 s | Unlock and resume right-hand control |
| `L` | Toggle lock manually |
| `S` | Switch camera / procedural space background (camera still tracks hands) |
| Right-click / right-drag in space | Place/move the source star in the source plane |
| `A` in space | Align source star with lens for an Einstein ring |
| `T` | Toggle source star visibility |
| `C` | Toggle illustrative source-color treatment |
| `[` / `]` | Decrease/increase wavelength by 10 nm |
| Mouse drag | Move lens; temporarily overrides gestures |
| `+` / `-`, or mouse wheel | Increase/decrease Einstein radius |
| `H` | Switch hand control on/off; disable to keep manual strength |
| `G` | Toggle Einstein-radius guide |
| `B` | Toggle stylized black central mask |
| `I` | Toggle original-camera inset |
| `Space` | Freeze/unfreeze the source image; mouse controls still work |
| `R` | Reset lens position and strength |
| `Q` / `Esc` | Quit |

With hand control enabled and the lens unlocked, gestures set strength again
on the next detection.
The lens holds its position and strength when the right hand disappears. Both
hands are detected; detector labels are swapped at the input boundary to match
your physical right/left hands in this mirrored camera setup.
The left hand never steers the lens; the right hand never triggers the lock.
Left-only lock/unlock gestures require the right pinch to be open: both pinches
closed together control wavelength instead.
Holding a pinch counts once, and an observed release is required before another
pinch counts. Losing the left hand does not count as releasing it. A pinch is
recognized on the next camera detection and takes priority over right-hand
motion in that frame.

While locked, mouse movement and strength controls also hold; `L` unlocks and `R`
resets/unlocks. The header says **LOCKED**, and the status line shows when the
first unlocking pinch has been received. The webcam itself keeps running.
`Space` independently freezes the source frame and pauses gesture detection;
unfreeze it before using gestures again. Gesture indicators disappear after
0.35 seconds without a valid observation. Keep your thumbs and index fingers
visible. No camera recordings are written automatically.

### Two-hand wavelength gesture

Two simultaneous pinches take priority over the one-hand lens controls. Their
starting separation anchors the selected wavelength; subsequent separation
scales it proportionally, smoothed over 0.07 seconds and clamped to 380–750 nm.
The readout gives frequency using `f = c / lambda` (THz), alongside a spatial
wave sketch and spectrum ruler. The wave animation is enormously slowed down;
it does not depict optical oscillations at real speed. Source colors are
approximate screen RGB representations of visible wavelengths.

The color treatment turns on when you enter wavelength control. `C` toggles it
off again; in webcam mode it is an illustrative tint, **not a measurement or
reconstruction of the room's spectrum**. A space source star always uses the
selected wavelength color. Its position is independent of the lens and can be
changed even with the lens locked. The first switch to space aligns it behind
the lens to show a ring immediately. `A` realigns it later; move
it off-axis to compare the separated images. `S` changes only the background;
hand tracking continues to use the live webcam, shown in the inset.

An existing lens lock survives wavelength adjustment. If the left pinch arrives
up to 0.35 seconds before the right pinch, its new lock is consumed as part of
the two-hand gesture. An observed release of both visible pinches ends the
transition back to lens controls without generating lock/unlock taps. Losing
tracking holds wavelength; reacquiring both pinches anchors a new adjustment
to the retained value. `R` resets the lens, wavelength (550 nm), and tint.

### Model and visual meaning

The image remapping uses the thin-lens point-mass equation in pixel coordinates:

`beta = theta - theta_E² * theta / |theta|²`

Here `theta` is the output pixel's displacement from the lens, `beta` gives the
source-image sample, and `theta_E` is the adjustable Einstein radius. An aligned
point source maps to the Einstein ring. An off-axis source can have two images.
Brightness is sampled from the original image; no glowing disk or stars are
painted onto the live camera feed. Samples outside the camera's field of view
are dark. The central singular sample is regularized numerically.

The black central disk is an optional artistic mask of radius `0.24 theta_E`,
not a computed event horizon or black-hole shadow. The Einstein-radius guide is
also an overlay and does not imply all source scenes produce a luminous ring.
This is an educational weak-field thin-lens mapping, not a near-horizon GR ray
tracer. The room is treated as a flat source image, with no depth reconstruction.
Relative mass is `(theta_E / 115 px)²` at fixed assumed lens geometry; no actual
mass, distance or camera calibration is inferred.

Within this geometric-optics model, lensing is achromatic: all selected
wavelengths use the same deflection map. The two-hand gesture changes the
source-light wavelength, not lens mass or gravitational bending. It is not a
calculation of gravitational redshift or Doppler shift. The color treatment is
a visualization, not a physical spectral transformation of an RGB camera.

References: [Narayan & Bartelmann, Lectures on Gravitational Lensing](https://www.astro.umd.edu/~richard/ASTR680/Naryan_Bartlemann_Lensing.pdf),
[OpenCV inverse image remapping](https://docs.opencv.org/4.1.0/da/d54/group__imgproc__transform.html),
[NASA on achromatic lensing](https://imagine.gsfc.nasa.gov/ask_astro/relativity.html),
[NASA visible light](https://science.nasa.gov/ems/09_visiblelight/).

### Camera-free preview

```bash
python3 CompPhysics/src/vision/lensing_camera.py --demo
python3 CompPhysics/src/vision/lensing_camera.py --background space
python3 CompPhysics/src/vision/lensing_camera.py --demo --background space --wavelength 680 --tint
python3 CompPhysics/src/vision/lensing_camera.py --demo --headless --frames 120 --snapshot CompPhysics/output/reports/lensing_camera_demo.png
```

The demo generates an animated sky with planets and a grid so distortion is
visible without a webcam. It uses the same lens and UI rendering as the live
version. It does not use the AI-generated concept mockup.
