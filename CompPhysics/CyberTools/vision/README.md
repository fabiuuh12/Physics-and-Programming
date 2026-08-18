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

## Integrated 2D Avatar

The Python tracker includes a 2D character in the same process. No UDP bridge, C++ executable, or second application is required. The primary detected person's landmarks are normalized around the hips and torso before adaptive smoothing is applied, allowing the character to keep a stable size while following body movement.

The character includes a face, hair, layered torso, jointed arms and legs, pose-level fingers, hands, and shoes. Detected upper- and lower-clothing colors are applied to the character when those measurements are reliable. If tracking is lost, the character smoothly relaxes to an idle pose.

Press `V` to cycle through:

1. Split camera and avatar view
2. Avatar-only view
3. Camera-only view

The 2D rig is the movement-development environment. Movement recognition and joint-history debugging will be added here before considering another 3D renderer.
