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

No target selection, mass entry or calibration is needed.

Uses OpenCV, NumPy, MediaPipe, Pillow for text rendering, and the bundled
`resources/models/hand_landmarker.task`. There are no model downloads. If hand
tracking cannot initialize, mouse controls remain available. On macOS, launch
from Terminal and allow Terminal under **Privacy & Security → Camera**.

```bash
python3 CompPhysics/src/vision/lensing_camera.py --camera 1
python3 CompPhysics/src/vision/lensing_camera.py --no-hands
```

### Interface

A small Camera/Space switch changes the background. Drag the **Lens size** slider
at the bottom to adjust strength; it holds while the lens is locked. The bottom
status shows when the lens is following, locked, paused, or using mouse control.
The smaller camera preview can be hidden with `I`.

The window opens at a size that fits the usable display, reserving room for the
macOS title bar and screen edges. On macOS its aspect ratio is maintained during
resizing to avoid an empty strip beside the scene. Press `F` to fit it again
after moving displays or resizing. Buttons, text, and sliders are drawn at the
display's pixel density (2× on Retina) while the camera/lens simulation stays at
its original resolution. Mouse coordinates are mapped back to the same logical
layout, including when the window is resized.

Use `--ui-scale 1`, `--ui-scale 2`, or `--ui-scale 3` to override the UI density.
The display size is read locally; this needs no screen-recording permission.

Click **Controls** at the top right, or press `?`, for the shortcut panel.
Click its close button or press **Esc** to dismiss it. The scene has no permanent
FPS counter or numerical dashboard. Text uses the bundled Inter typeface.

### Controls

| Input | Action |
| --- | --- |
| Controls button / `?` | Show or hide shortcut panel |
| Camera / Space buttons | Switch background |
| Lens size slider | Adjust lens strength while unlocked |
| `F` | Fit window to usable screen area |
| Right thumb/index midpoint | Lens position |
| Right thumb/index separation, normalized by palm width | Lens strength |
| Left thumb/index pinch | Lock lens position and size |
| Two new left pinches within 0.8 s | Unlock and resume right-hand control |
| `L` | Toggle lock manually |
| `S` | Switch camera / procedural space background (camera still tracks hands) |
| Right-click / right-drag in space | Place/move the source star in the source plane |
| `A` in space | Align source star with lens for an Einstein ring |
| `T` | Toggle source star visibility |
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
Holding a pinch counts once, and an observed release is required before another
pinch counts. Losing the left hand does not count as releasing it. A pinch is
recognized on the next camera detection and takes priority over right-hand
motion in that frame.

While locked, mouse movement and strength controls also hold; `L` unlocks and `R`
resets/unlocks. The bottom status says **Locked**, and shows when the
first unlocking pinch has been received. The webcam itself keeps running.
`Space` independently freezes the source frame and pauses gesture detection;
unfreeze it before using gestures again. Gesture indicators disappear after
0.35 seconds without a valid observation. Keep your thumbs and index fingers
visible. No camera recordings are written automatically.

### Bright source star

Press `S` to switch between your room and a procedural starfield while the
camera keeps tracking your hands. The first switch to space puts a bright star
behind the lens to form a ring immediately. Right-click/right-drag to move the
source star off-center and see its split images. `A` realigns it; `T` hides it.
The camera inset stays visible so you can see your hands.

The source has a white-hot core and a broad warm halo. Soft screen bloom follows
the bright lensed pixels, making the resulting rings and arcs shine. This glow
is a display effect, not a calculation of stellar luminosity or illumination of
surrounding matter. Camera mode keeps the original source colors and has no
added bloom. Left-hand locking still holds only the lens, so you can move the
star independently while the lens is locked.

### Model and visual meaning

The image remapping uses the thin-lens point-mass equation in pixel coordinates:

`beta = theta - theta_E² * theta / |theta|²`

Here `theta` is the output pixel's displacement from the lens, `beta` gives the
source-image sample, and `theta_E` is the adjustable Einstein radius. An aligned
point source maps to the Einstein ring. An off-axis source can have two images.
Camera-mode brightness is sampled from the original image; no glowing disk or
stars are painted onto the live camera feed. Space mode uses a generated
source star and adds display bloom after lens remapping. Samples outside the camera's field of view
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
python3 CompPhysics/src/vision/lensing_camera.py --background space
python3 CompPhysics/src/vision/lensing_camera.py --demo --background space
python3 CompPhysics/src/vision/lensing_camera.py --demo --headless --frames 120 --snapshot CompPhysics/output/reports/lensing_camera_demo.png
```

The demo generates an animated sky with planets and a grid so distortion is
visible without a webcam. It uses the same lens and UI rendering as the live
version. It does not use the AI-generated concept mockup.
