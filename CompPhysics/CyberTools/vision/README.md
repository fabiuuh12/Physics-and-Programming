# Face Presence Scanner

`Face Presence Scanner.py` opens the webcam and displays a restrained monitoring interface around visible, front-facing faces. It includes stable presence reporting, multiple-face counts, session duration, frame rate, and lighting/focus feedback. Processing stays local, and the program does not identify people, save frames, or record video.

From the `CompPhysics` folder, install dependencies and run it with:

```powershell
python -m pip install -r requirements.txt
python "CyberTools/vision/Face Presence Scanner.py"
```

If the default webcam is not the right one, select another camera index:

```powershell
python "CyberTools/vision/Face Presence Scanner.py" --camera 1
```

Controls: `Q` or `Esc` quits, `F` toggles fullscreen, `M` toggles mirroring, and `H` toggles the help text.

Optional settings include `--no-mirror`, `--width`, `--height`, and `--detect-every N`. Run with `--help` to see all options.

This is face *detection*, not identity recognition. A “searching” result only means that the detector cannot currently see a clear front-facing face; it does not prove that no person is present.

## Scene Observation Recorder

`Scene Observation Recorder.py` is a people-only body-pose tracker. The live view uses a segmentation-based person silhouette and pose-defined outlines for the head, torso, arms, and legs. Pose data is stored as named, normalized landmark coordinates and visibility scores. General object recognition is intentionally disabled so CPU time is available for pose tracking and future avatar control.

For each sufficiently visible pose, the recorder also samples pose-defined upper- and lower-clothing regions. It reports a coarse primary color, an optional secondary color, and the primary color's share of sampled pixels. Clothing colors are approximate visible appearance: lighting, shadows, patterns, and overlapping objects can change the result.

```powershell
python "CyberTools/vision/Scene Observation Recorder.py"
```

Press `R` to start or stop recording structured observations. The default database is `CyberTools/vision/data/scene_observations.sqlite3`. Camera images are never stored. Inspect a recorded dataset with:

```powershell
python "CyberTools/vision/inspect_observations.py"
```

The database is suitable as input to a later movement-analysis or avatar-control pipeline, but collecting observations alone does not retrain the detector.

### Performance controls

Pose inference runs independently from the display rate, and segmentation contours are cached between pose updates. The default favors a smooth preview. Lower intervals increase responsiveness but use more CPU:

```powershell
python "CyberTools/vision/Scene Observation Recorder.py" --pose-every 3
```

The recorder does not assume a fixed four-person scene. It supports a configurable simultaneous-pose capacity (default `8`, accepted range `1-20`). Higher values require more CPU and enough image resolution for each person to remain visible:

```powershell
python "CyberTools/vision/Scene Observation Recorder.py" --max-people 12
```

Person segmentation masks are combined and temporally stabilized before the green outline is drawn. Adjust the balance between responsiveness and stability with `--outline-smoothing` from `0` through `0.9`:

```powershell
python "CyberTools/vision/Scene Observation Recorder.py" --outline-smoothing 0.68
```

### Self-described characteristics

Sensitive identity attributes such as race are never inferred from camera images. If a participant explicitly chooses to supply characteristics, copy `subject_profile.example.json`, edit the values, and attach it to the recording session:

```powershell
python "CyberTools/vision/Scene Observation Recorder.py" --profile "CyberTools/vision/subject_profile.example.json"
```

The profile is stored separately as self-described session metadata. Appearance observations and self-described attributes are not presented as equivalent facts.

## 3D Pose Avatar

The people tracker streams each detected person's 33 MediaPipe world landmarks to localhost UDP port `50525`. It sends coordinates and visibility scores only; camera frames are not transmitted. Disable streaming with `--no-pose-stream` or change the destination with `--pose-udp-host` and `--pose-udp-port`.

Build and run the C++/raylib procedural avatar from the `CompPhysics` folder:

```powershell
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native --target pose_avatar_3d_cpp --config Release
./build-native/Release/pose_avatar_3d_cpp.exe
```

Run `Scene Observation Recorder.py` at the same time. The avatar listens for the primary person, retargets the pose to a stable humanoid scale, and smoothly returns to an idle pose if the stream disconnects.

The procedural avatar uses fixed anatomical bone lengths while following the detected joint directions. This prevents noisy camera-depth estimates from stretching limbs. Adaptive smoothing suppresses small resting jitter but reacts faster to deliberate movement, and joints with low visibility temporarily hold their last reliable position.

The current character includes a layered torso, tapered limbs, articulated pose-level hands, shoes, neck, ears, hair, eyes, nose, and mouth. It is intentionally procedural: this establishes and debugs the motion-retargeting pipeline before a rigged external model is introduced.

Avatar controls: hold the right mouse button and drag to orbit, use the mouse wheel to zoom, press `M` to mirror the pose, `R` to reset the camera, and `F` for fullscreen.
