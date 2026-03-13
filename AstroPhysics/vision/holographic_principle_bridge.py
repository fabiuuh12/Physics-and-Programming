#!/usr/bin/env python3
"""
Dedicated hand-tracking bridge for the holographic principle visualization.

Controls:
- Right-hand thumb/index pinch starts hologram encoding control
- While active, spreading thumb/index apart increases encoding progress
- Left hand steers view rotation/pitch when available

This writes the same `live_controls.txt` bridge format consumed by the
C++ raylib app in `AstroPhysics/gravity/holographic_principle_viz.cpp`.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:
    print(
        "Missing dependency: "
        f"{exc}. Install with: pip install opencv-python mediapipe numpy"
    )
    sys.exit(1)


WINDOW_NAME = "Holographic Principle Bridge"
CONTROL_OUTPUT_PATH = Path(__file__).resolve().parent / "live_controls.txt"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17

PINCH_ACTIVATE_RATIO = 0.42
PINCH_RELEASE_RATIO = 1.10
ZOOM_MIN = 0.45
ZOOM_MAX = 2.60
ZOOM_GAIN = 1.65
CONTROL_WRITE_HZ = 30.0


@dataclass
class HandObservation:
    label: str
    score: float
    landmarks: List[Tuple[float, float, float]]


@dataclass
class BridgeState:
    zoom: float = 0.45
    rotation_deg: float = 0.0
    pitch_deg: float = 0.0
    wave_amp: float = 1.0
    paused: bool = False
    zoom_line_active: bool = False
    zoom_line_ax: float = 0.5
    zoom_line_ay: float = 0.5
    zoom_line_bx: float = 0.5
    zoom_line_by: float = 0.5
    label: str = "Unknown"
    gesture: str = "idle"
    right_zoom_latched: bool = False
    right_zoom_anchor_metric: float = 0.0
    right_zoom_anchor_zoom: float = ZOOM_MIN
    last_write_ts: float = 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _extract_solution_hands(result: object, mirror: bool) -> List[HandObservation]:
    out: List[HandObservation] = []
    if not getattr(result, "multi_hand_landmarks", None):
        return out
    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        label = "Unknown"
        score = 1.0
        if result.multi_handedness and idx < len(result.multi_handedness):
            c = result.multi_handedness[idx].classification[0]
            label = c.label or "Unknown"
            score = float(c.score)
        if mirror:
            if label.lower() == "left":
                label = "Right"
            elif label.lower() == "right":
                label = "Left"
        out.append(
            HandObservation(
                label=label,
                score=score,
                landmarks=[(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
            )
        )
    return out


def _extract_task_hands(result: object, mirror: bool) -> List[HandObservation]:
    out: List[HandObservation] = []
    if not getattr(result, "hand_landmarks", None):
        return out
    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        label = "Unknown"
        score = 1.0
        if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
            c = result.handedness[idx][0]
            label = (
                getattr(c, "category_name", None)
                or getattr(c, "display_name", None)
                or "Unknown"
            )
            score = float(getattr(c, "score", 1.0))
        if mirror:
            if label.lower() == "left":
                label = "Right"
            elif label.lower() == "right":
                label = "Left"
        out.append(
            HandObservation(
                label=label,
                score=score,
                landmarks=[(lm.x, lm.y, lm.z) for lm in hand_landmarks],
            )
        )
    return out


def _best_hand_by_label(hands: List[HandObservation], wanted: str) -> HandObservation | None:
    filtered = [hand for hand in hands if hand.label.lower() == wanted]
    if not filtered:
        return None
    return max(filtered, key=lambda hand: hand.score)


def _palm_scale(landmarks: List[Tuple[float, float, float]]) -> float:
    pts = np.asarray(landmarks, dtype=np.float32)
    return max(1e-4, float(np.linalg.norm(pts[INDEX_MCP, :2] - pts[PINKY_MCP, :2])))


def _pinch_ratio(landmarks: List[Tuple[float, float, float]]) -> float:
    pts = np.asarray(landmarks, dtype=np.float32)
    pinch_dist = float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_TIP, :2]))
    return pinch_dist / _palm_scale(landmarks)


def _export_live_controls(state: BridgeState, now: float) -> None:
    interval = 1.0 / max(1e-3, CONTROL_WRITE_HZ)
    if now - state.last_write_ts < interval:
        return
    state.last_write_ts = now

    timestamp_ms = int(time.time() * 1000)
    payload = (
        f"timestamp_ms={timestamp_ms}\n"
        f"zoom={state.zoom:.6f}\n"
        f"rotation_deg={state.rotation_deg:.6f}\n"
        f"pitch_deg={state.pitch_deg:.6f}\n"
        f"wave_amp={state.wave_amp:.6f}\n"
        f"paused={1 if state.paused else 0}\n"
        f"camera_locked=0\n"
        f"zoom_line_active={1 if state.zoom_line_active else 0}\n"
        f"zoom_line_ax={state.zoom_line_ax:.6f}\n"
        f"zoom_line_ay={state.zoom_line_ay:.6f}\n"
        f"zoom_line_bx={state.zoom_line_bx:.6f}\n"
        f"zoom_line_by={state.zoom_line_by:.6f}\n"
        f"label={state.label}\n"
        f"gesture={state.gesture}\n"
        f"pinch_ratio=0.000000\n"
        f"n_inc_count=0\n"
        f"n_dec_count=0\n"
    )
    tmp_path = CONTROL_OUTPUT_PATH.with_suffix(".tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, CONTROL_OUTPUT_PATH)
    except OSError:
        pass


def _tracker_setup():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        return "solutions", hands, None

    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MediaPipe task model missing: {MODEL_PATH}")
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            running_mode=vision.RunningMode.VIDEO,
        )
        tracker = vision.HandLandmarker.create_from_options(options)
        return "tasks", None, tracker

    raise RuntimeError("Unsupported mediapipe build: expected solutions or tasks.vision.")


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open the default webcam.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    try:
        mode, hands, task_tracker = _tracker_setup()
    except Exception as exc:
        print(f"Error: {exc}")
        cap.release()
        return 1

    drawer = getattr(getattr(mp, "solutions", None), "drawing_utils", None)
    task_timestamp_ms = 0

    state = BridgeState()
    mirror = True
    last_t = time.perf_counter()
    fps = 60.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1120, 680)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read webcam frame.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(1e-4, min(0.05, now - last_t))
            last_t = now
            fps = 0.92 * fps + 0.08 * (1.0 / dt)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if mode == "tasks":
                task_timestamp_ms = max(task_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = task_tracker.detect_for_video(mp_image, task_timestamp_ms)
                observations = _extract_task_hands(result, mirror)
            else:
                result = hands.process(rgb)
                observations = _extract_solution_hands(result, mirror)

            if mode == "solutions" and result.multi_hand_landmarks and drawer is not None:
                for lm in result.multi_hand_landmarks:
                    drawer.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)

            right = _best_hand_by_label(observations, "right")
            left = _best_hand_by_label(observations, "left")
            control_hand = left or right

            state.zoom_line_active = False
            state.label = control_hand.label if control_hand is not None else "Unknown"
            state.gesture = "idle"

            if control_hand is not None:
                wrist_x = control_hand.landmarks[WRIST][0]
                wrist_y = control_hand.landmarks[WRIST][1]
                state.rotation_deg = _clamp((0.50 - wrist_x) * 170.0, -90.0, 90.0)
                state.pitch_deg = _clamp((0.52 - wrist_y) * 130.0, -70.0, 70.0)

            if right is not None:
                ratio = _pinch_ratio(right.landmarks)
                thumb = right.landmarks[THUMB_TIP]
                index = right.landmarks[INDEX_TIP]
                state.zoom_line_ax = thumb[0]
                state.zoom_line_ay = thumb[1]
                state.zoom_line_bx = index[0]
                state.zoom_line_by = index[1]

                if not state.right_zoom_latched and ratio <= PINCH_ACTIVATE_RATIO:
                    state.right_zoom_latched = True
                    state.right_zoom_anchor_metric = ratio
                    state.right_zoom_anchor_zoom = state.zoom

                if state.right_zoom_latched:
                    state.zoom_line_active = True
                    target_zoom = state.right_zoom_anchor_zoom + (ratio - state.right_zoom_anchor_metric) * ZOOM_GAIN
                    state.zoom = _clamp(target_zoom, ZOOM_MIN, ZOOM_MAX)
                    state.wave_amp = 0.25 + (state.zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN) * (1.80 - 0.25)
                    state.gesture = "right_zoom"
                    if ratio >= PINCH_RELEASE_RATIO:
                        state.right_zoom_latched = False
                else:
                    state.gesture = "tracking"

                h, w = frame.shape[:2]
                a = (int(thumb[0] * w), int(thumb[1] * h))
                b = (int(index[0] * w), int(index[1] * h))
                color = (110, 240, 255) if state.zoom_line_active else (100, 160, 220)
                cv2.line(frame, a, b, color, 3 if state.zoom_line_active else 1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"zoom {state.zoom:.2f}",
                    (22, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (130, 240, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                state.right_zoom_latched = False

            _export_live_controls(state, now)

            cv2.putText(
                frame,
                f"Hologram bridge  fps {fps:4.1f}",
                (22, frame.shape[0] - 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (220, 230, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "right pinch then spread = encode hologram | q quit",
                (22, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (190, 205, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mirror = not mirror
    finally:
        if hands is not None:
            hands.close()
        if task_tracker is not None:
            task_tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
