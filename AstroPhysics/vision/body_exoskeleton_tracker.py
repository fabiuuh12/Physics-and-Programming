#!/usr/bin/env python3
"""
Full-body webcam exoskeleton tracker.

Features:
- MediaPipe Pose full-body tracking
- Mechanical exoskeleton overlay following the body
- Smoothed landmark motion to reduce jitter
- Movement and posture state detection
- FPS and state HUD

Controls:
- q: quit
- m: toggle mirror
"""

from __future__ import annotations

import math
import os
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:
    print(f"Missing dependency: {exc}. Install with: pip install opencv-python mediapipe numpy")
    sys.exit(1)


WINDOW_NAME = "Body Exoskeleton Tracker"
MODULE_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = MODULE_DIR / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
FPS_EMA = 0.90
SMOOTH_ALPHA = 0.34
VISIBILITY_THRESHOLD = 0.45

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

EXO_BONES: tuple[tuple[int, int], ...] = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
)


@dataclass
class PoseMotionState:
    movement_label: str = "searching"
    crouching: bool = False
    arms_up: bool = False
    lean_label: str = "center"
    stride_label: str = "narrow"
    center_speed: float = 0.0


class FPSCounter:
    def __init__(self, ema: float = FPS_EMA) -> None:
        self.ema = ema
        self._last_ts: float | None = None
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_ts is not None:
            dt = now - self._last_ts
            if dt > 0.0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0.0 else self.ema * self._fps + (1.0 - self.ema) * inst
        self._last_ts = now
        return self._fps


class LandmarkSmoother:
    def __init__(self, alpha: float = SMOOTH_ALPHA) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 0.99))
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        current = landmarks.astype(np.float32, copy=True)
        if self._state is None or self._state.shape != current.shape:
            self._state = current
        else:
            self._state = (1.0 - self.alpha) * self._state + self.alpha * current
        return self._state.copy()


class PoseTracker:
    def __init__(self) -> None:
        self.backend = ""
        self._pose = None
        self._mp = mp

    def setup(self) -> None:
        if hasattr(self._mp, "solutions") and hasattr(self._mp.solutions, "pose"):
            self._pose = self._mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.backend = "solutions"
            return

        if hasattr(self._mp, "tasks") and hasattr(self._mp.tasks, "vision"):
            self._ensure_pose_model()
            BaseOptions = self._mp.tasks.BaseOptions
            PoseLandmarker = self._mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = self._mp.tasks.vision.PoseLandmarkerOptions
            RunningMode = self._mp.tasks.vision.RunningMode
            self._pose = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                    running_mode=RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.55,
                    min_pose_presence_confidence=0.55,
                    min_tracking_confidence=0.55,
                    output_segmentation_masks=False,
                )
            )
            self.backend = "tasks"
            return

        raise RuntimeError("No supported MediaPipe pose API is available in this environment.")

    def _ensure_pose_model(self) -> None:
        if POSE_MODEL_PATH.exists():
            return
        POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        except urllib.error.URLError as exc:
            try:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(POSE_MODEL_URL, context=ctx) as response:
                    POSE_MODEL_PATH.write_bytes(response.read())
            except Exception as inner_exc:
                raise RuntimeError(f"Unable to download pose model from {POSE_MODEL_URL}: {inner_exc}") from exc

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        if self.backend == "solutions":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
            if not result.pose_landmarks:
                return None
            return np.asarray(
                [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in result.pose_landmarks.landmark],
                dtype=np.float32,
            )

        if self.backend == "tasks":
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            result = self._pose.detect_for_video(mp_image, timestamp_ms)
            if not getattr(result, "pose_landmarks", None):
                return None
            landmarks = result.pose_landmarks[0]
            return np.asarray(
                [[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in landmarks],
                dtype=np.float32,
            )

        return None

    def close(self) -> None:
        if self._pose is not None and hasattr(self._pose, "close"):
            self._pose.close()


def _xy(landmarks: np.ndarray, idx: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    x = int(np.clip(landmarks[idx, 0], 0.0, 1.0) * frame_w)
    y = int(np.clip(landmarks[idx, 1], 0.0, 1.0) * frame_h)
    return (x, y)


def _visible(landmarks: np.ndarray, idx: int) -> bool:
    return bool(landmarks[idx, 3] >= VISIBILITY_THRESHOLD)


def _all_visible(landmarks: np.ndarray, indices: Iterable[int]) -> bool:
    return all(_visible(landmarks, idx) for idx in indices)


def _midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _norm_distance(landmarks: np.ndarray, a: int, b: int) -> float:
    pa = landmarks[a, :2]
    pb = landmarks[b, :2]
    return float(np.linalg.norm(pa - pb))


def _draw_segment(frame: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: tuple[int, int, int]) -> None:
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    length = math.hypot(dx, dy)
    if length < 1.0:
        return

    nx = -dy / length
    ny = dx / length
    off = int(max(2.0, min(5.0, length * 0.05)))
    oa = (int(ax + nx * off), int(ay + ny * off))
    ob = (int(bx + nx * off), int(by + ny * off))
    ia = (int(ax - nx * off), int(ay - ny * off))
    ib = (int(bx - nx * off), int(by - ny * off))

    cv2.line(frame, oa, ob, (35, 42, 56), 6, cv2.LINE_AA)
    cv2.line(frame, ia, ib, (35, 42, 56), 6, cv2.LINE_AA)
    cv2.line(frame, a, b, color, 4, cv2.LINE_AA)


def _draw_joint(frame: np.ndarray, p: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    cv2.circle(frame, p, radius + 3, (20, 24, 34), -1, cv2.LINE_AA)
    cv2.circle(frame, p, radius, color, 2, cv2.LINE_AA)
    cv2.circle(frame, p, max(2, radius // 2), (240, 244, 255), -1, cv2.LINE_AA)


def _draw_torso_plate(frame: np.ndarray, landmarks: np.ndarray, frame_w: int, frame_h: int) -> None:
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not _all_visible(landmarks, needed):
        return

    ls = _xy(landmarks, LEFT_SHOULDER, frame_w, frame_h)
    rs = _xy(landmarks, RIGHT_SHOULDER, frame_w, frame_h)
    lh = _xy(landmarks, LEFT_HIP, frame_w, frame_h)
    rh = _xy(landmarks, RIGHT_HIP, frame_w, frame_h)
    chest = _midpoint(ls, rs)
    pelvis = _midpoint(lh, rh)

    pts = np.array(
        [
            ls,
            rs,
            (int(rs[0] * 0.82 + rh[0] * 0.18), int(rs[1] * 0.76 + rh[1] * 0.24)),
            rh,
            lh,
            (int(ls[0] * 0.82 + lh[0] * 0.18), int(ls[1] * 0.76 + lh[1] * 0.24)),
        ],
        dtype=np.int32,
    )
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, pts, (44, 68, 102))
    cv2.addWeighted(overlay, 0.24, frame, 0.76, 0.0, frame)
    cv2.polylines(frame, [pts], True, (124, 210, 255), 2, cv2.LINE_AA)
    cv2.line(frame, chest, pelvis, (155, 232, 255), 2, cv2.LINE_AA)


def _draw_head_helmet(frame: np.ndarray, landmarks: np.ndarray, frame_w: int, frame_h: int) -> None:
    needed = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)
    if not _all_visible(landmarks, needed):
        return

    nose = _xy(landmarks, NOSE, frame_w, frame_h)
    ls = _xy(landmarks, LEFT_SHOULDER, frame_w, frame_h)
    rs = _xy(landmarks, RIGHT_SHOULDER, frame_w, frame_h)
    shoulder_width = max(18, int(math.hypot(rs[0] - ls[0], rs[1] - ls[1]) * 0.24))
    center = (nose[0], int(nose[1] - shoulder_width * 0.28))
    axes = (shoulder_width, int(shoulder_width * 1.18))

    overlay = frame.copy()
    cv2.ellipse(overlay, center, axes, 0, 0, 360, (52, 72, 108), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0.0, frame)
    cv2.ellipse(frame, center, axes, 0, 0, 360, (150, 226, 255), 2, cv2.LINE_AA)
    visor_y = center[1] + int(axes[1] * 0.08)
    cv2.line(
        frame,
        (center[0] - int(axes[0] * 0.68), visor_y),
        (center[0] + int(axes[0] * 0.68), visor_y),
        (118, 252, 220),
        2,
        cv2.LINE_AA,
    )


def _movement_state(
    landmarks: np.ndarray,
    prev_center: np.ndarray | None,
    dt: float,
) -> tuple[PoseMotionState, np.ndarray | None]:
    state = PoseMotionState()
    needed_center = (LEFT_HIP, RIGHT_HIP)
    if not _all_visible(landmarks, needed_center):
        return state, None

    center = 0.5 * (landmarks[LEFT_HIP, :2] + landmarks[RIGHT_HIP, :2])
    if prev_center is not None and dt > 1.0e-4:
        speed = float(np.linalg.norm(center - prev_center) / dt)
        state.center_speed = speed
        if speed < 0.02:
            state.movement_label = "steady"
        elif speed < 0.08:
            state.movement_label = "moving"
        else:
            state.movement_label = "fast"
    else:
        state.movement_label = "steady"

    if _all_visible(landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE)):
        shoulder_y = 0.5 * (landmarks[LEFT_SHOULDER, 1] + landmarks[RIGHT_SHOULDER, 1])
        hip_y = 0.5 * (landmarks[LEFT_HIP, 1] + landmarks[RIGHT_HIP, 1])
        knee_y = 0.5 * (landmarks[LEFT_KNEE, 1] + landmarks[RIGHT_KNEE, 1])
        torso = max(1.0e-4, hip_y - shoulder_y)
        leg = max(1.0e-4, knee_y - hip_y)
        state.crouching = (torso / leg) > 1.12

    if _all_visible(landmarks, (LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER)):
        state.arms_up = landmarks[LEFT_WRIST, 1] < landmarks[LEFT_SHOULDER, 1] and landmarks[RIGHT_WRIST, 1] < landmarks[RIGHT_SHOULDER, 1]

    if _all_visible(landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)):
        shoulders_x = 0.5 * (landmarks[LEFT_SHOULDER, 0] + landmarks[RIGHT_SHOULDER, 0])
        hips_x = 0.5 * (landmarks[LEFT_HIP, 0] + landmarks[RIGHT_HIP, 0])
        delta = shoulders_x - hips_x
        if delta < -0.025:
            state.lean_label = "left"
        elif delta > 0.025:
            state.lean_label = "right"
        else:
            state.lean_label = "center"

    if _all_visible(landmarks, (LEFT_ANKLE, RIGHT_ANKLE, LEFT_HIP, RIGHT_HIP)):
        ankle_span = _norm_distance(landmarks, LEFT_ANKLE, RIGHT_ANKLE)
        hip_span = _norm_distance(landmarks, LEFT_HIP, RIGHT_HIP)
        ratio = ankle_span / max(1.0e-4, hip_span)
        if ratio > 1.55:
            state.stride_label = "wide"
        elif ratio > 1.12:
            state.stride_label = "open"
        else:
            state.stride_label = "narrow"

    return state, center.astype(np.float32)


def _draw_hud(frame: np.ndarray, fps: float, mirror: bool, pose_state: PoseMotionState, tracked: bool) -> None:
    h, w = frame.shape[:2]
    panel_h = 96
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.64, frame, 0.36, 0.0, frame)
    cv2.line(frame, (0, h - panel_h), (w, h - panel_h), (84, 106, 144), 1, cv2.LINE_AA)

    line1 = (
        f"pose:{'locked' if tracked else 'searching'}  mirror:{'on' if mirror else 'off'}  fps:{fps:4.1f}"
        f"  motion:{pose_state.movement_label}  speed:{pose_state.center_speed:0.3f}"
    )
    line2 = (
        f"crouch:{int(pose_state.crouching)}  arms_up:{int(pose_state.arms_up)}"
        f"  lean:{pose_state.lean_label}  stance:{pose_state.stride_label}"
    )
    cv2.putText(frame, line1, (14, h - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (222, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(frame, line2, (14, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (128, 220, 255), 1, cv2.LINE_AA)


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (camera index 0).")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        pose_tracker = PoseTracker()
        pose_tracker.setup()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        cap.release()
        return 1

    fps_counter = FPSCounter()
    smoother = LandmarkSmoother()
    mirror = True
    prev_center: np.ndarray | None = None
    prev_state_ts = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: webcam frame read failed.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        frame_h, frame_w = frame.shape[:2]
        fps = fps_counter.update()
        timestamp_ms = int(time.time() * 1000)

        now = time.perf_counter()
        dt = max(1.0e-4, now - prev_state_ts)
        prev_state_ts = now

        tracked = False
        pose_state = PoseMotionState()
        raw = pose_tracker.detect(frame, timestamp_ms)

        if raw is not None:
            landmarks = smoother.smooth(raw)
            tracked = True
            pose_state, prev_center = _movement_state(landmarks, prev_center, dt)

            _draw_torso_plate(frame, landmarks, frame_w, frame_h)
            _draw_head_helmet(frame, landmarks, frame_w, frame_h)

            for a_idx, b_idx in EXO_BONES:
                if not _all_visible(landmarks, (a_idx, b_idx)):
                    continue
                a = _xy(landmarks, a_idx, frame_w, frame_h)
                b = _xy(landmarks, b_idx, frame_w, frame_h)
                _draw_segment(frame, a, b, (116, 214, 255))

            for idx in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST,
                        LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE):
                if not _visible(landmarks, idx):
                    continue
                _draw_joint(frame, _xy(landmarks, idx, frame_w, frame_h), 7, (118, 255, 216))
        else:
            smoother.reset()
            prev_center = None

        _draw_hud(frame, fps, mirror, pose_state, tracked)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mirror = not mirror

    pose_tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
