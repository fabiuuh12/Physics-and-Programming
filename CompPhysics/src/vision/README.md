# Camera experiments

## Phone physics camera

From the repository root:

```bash
python3 explore.py run phone_physics_camera
```

Requires `opencv-python` and `numpy` (already in the project requirements).
Allow camera access for your terminal/Python application if requested.
For a different camera or an optional known phone mass:

```bash
python3 CompPhysics/src/vision/phone_physics_camera.py --camera 1 --mass-g 200
```

Use your phone's actual mass; 200 g above is only an example.

1. Keep the webcam stationary. Hold the phone with its detailed face visible,
   preferably a lit screen with a static image or its camera lenses.
2. Press **S**, draw a tight rectangle around the phone, and press **Enter**.
   This selects the target manually; the program does not recognize phone models.
3. Move the phone slowly sideways or up/down while keeping it at the same depth
   and orientation. Cyan shows velocity, gold acceleration, and the trail shows
   the selected region's center. Keep the phone in view; after loss, press **S**
   to select it again. Plain reflective surfaces and occlusion can break tracking.
4. Press **C** to freeze the image. Click the endpoints of a known length in the
   phone's motion plane (for example, its measured height), type the length in
   centimeters, and press **Enter**. Pixel measurements now become metric estimates.
5. Press **M**, type the actual phone mass in grams, and press **Enter** to enable
   estimated net force (pink) and in-plane kinetic energy. **Esc** cancels entry.
6. **R** clears the target but preserves calibration/mass. **Q** quits outside entry mode.

### What the numbers mean

Sparse Lucas–Kanade optical flow with forward/backward checks tracks features
inside the selection. A RANSAC similarity transform follows the region. A local
quadratic fit over up to 0.45 seconds estimates velocity and acceleration using
actual timestamps. Derivatives refer to the middle of this window (roughly
0.22 seconds of lag), while the box and arrow origins follow the current target.
The fit needs at least 0.18 seconds and six samples. Lost tracking clears motion
readings; lock requires reselection. These checks reduce drift but cannot rule
out tracking the wrong visible features.

Coordinates are image-plane x right and y up. Net force is `m * a`; kinetic
energy is `0.5 * m * |v|²`. These are estimates of **projected translational
motion of the selected region**, not a measurement of separate gravity, hand,
friction or contact forces, center of mass, rotational energy, or full 3D motion.
There is no gravity arrow because hand support and gravity cannot be separated
from this measurement. Energy is a scalar numeric readout.

Physical units require a known length at the object's depth. Camera movement,
perspective, out-of-plane movement, rotation, blur and tracking noise bias the
results, especially acceleration. Recalibrate after moving the camera or changing
the object's depth. A stationary hand-held phone should have near-zero **net**
force even though gravity and hand support act on it.

Velocity arrows use 0.22 seconds of travel; acceleration arrows use 0.045 s²;
force arrows use 3000 pixels per newton. All are capped at 170 pixels and marked
when capped. Compare lengths within the same quantity only.

### Camera-free preview

```bash
python3 CompPhysics/src/vision/phone_physics_camera.py --demo --mass-g 200
python3 CompPhysics/src/vision/phone_physics_camera.py --demo --headless --frames 180 --mass-g 200 --snapshot CompPhysics/output/reports/phone_physics_demo.png
```

The synthetic demo uses a known 15 cm phone height and a generated textured
rectangle. It checks tracking and rendering, not real webcam performance.
Tracking implementation follows the [OpenCV optical flow APIs](https://docs.opencv.org/4.6.0/dc/d6b/group__video__track.html).
