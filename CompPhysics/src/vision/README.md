# Camera experiments

## Gravitational lensing camera

```bash
python3 explore.py run lensing_camera
```

Your mirrored webcam feed becomes a background source plane behind a simulated
point-mass lens. Move your thumb/index midpoint to move the lens; spread the two
fingers to increase lens strength. A smaller inset shows the original image.
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
| Thumb/index midpoint | Lens position |
| Thumb/index separation, normalized by palm width | Lens strength |
| Mouse drag | Move lens; temporarily overrides gestures |
| `+` / `-`, or mouse wheel | Increase/decrease Einstein radius |
| `H` | Switch hand control on/off; disable to keep manual strength |
| `G` | Toggle Einstein-radius guide |
| `B` | Toggle stylized black central mask |
| `I` | Toggle original-camera inset |
| `Space` | Freeze/unfreeze the source image; mouse controls still work |
| `R` | Reset lens position and strength |
| `Q` / `Esc` | Quit |

With hand control enabled, gestures set strength again on the next detection.
The lens holds its position and strength when the hand disappears. Gesture
indicators disappear after 0.35 seconds without a valid observation. Only one
hand is used; switching visible hands may move the target. Keep your thumb and
index visible. No camera recordings are written automatically.

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

References: [Narayan & Bartelmann, Lectures on Gravitational Lensing](https://www.astro.umd.edu/~richard/ASTR680/Naryan_Bartlemann_Lensing.pdf),
[OpenCV inverse image remapping](https://docs.opencv.org/4.1.0/da/d54/group__imgproc__transform.html).

### Camera-free preview

```bash
python3 CompPhysics/src/vision/lensing_camera.py --demo
python3 CompPhysics/src/vision/lensing_camera.py --demo --headless --frames 120 --snapshot CompPhysics/output/reports/lensing_camera_demo.png
```

The demo generates an animated sky with planets and a grid so distortion is
visible without a webcam. It uses the same lens and UI rendering as the live
version. It does not use the AI-generated concept mockup.
