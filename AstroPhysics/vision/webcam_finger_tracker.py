#!/usr/bin/env python3
"""
Webcam hand tracker test.

Features:
- Red fingertip markers
- Temporal smoothing to reduce jitter
- Hand confidence thresholding
- FPS overlay
- Hand label overlays (Left/Right)
- Gesture detection: pinch, open hand, fist, point, two-finger pinch
- Gesture-mapped camera controls:
  - pinch/two_finger_pinch -> zoom
  - dual-pinch line center movement -> camera rotation/pitch
  - one-hand fist -> camera lock/unlock
  - two-hand fists -> pause/play (debounced toggle)
- Astrophysics wave panel:
  - pinch/two_finger_pinch -> wave amplification

Press 'q' to quit.
"""

import math
import os
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

# Some MediaPipe builds import matplotlib internally; keep its cache writable.
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
        f"{exc}. Install with: pip install opencv-python mediapipe"
    )
    sys.exit(1)

# Landmark indices (MediaPipe 21-hand-landmark format)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

TIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Tracking/gesture tuning
SMOOTHING_ALPHA = 0.35
HAND_CONFIDENCE_THRESHOLD = 0.55
FPS_EMA = 0.90
# Keep false for natural handedness mapping (Left=left hand, Right=right hand).
SWAP_HANDEDNESS_LABELS = False

# Camera-control tuning
ZOOM_MIN = 0.05
ZOOM_MAX = 2.60
WAVE_AMP_MIN = 0.25
WAVE_AMP_MAX = 1.80
CONTROL_LERP = 0.20
ZOOM_GESTURE_LERP = 0.30
ZOOM_RATIO_POWER = 1.55
AMP_RATIO_POWER = 1.10
ZOOM_RATIO_MIN = 0.45
ZOOM_RATIO_MAX = 2.20
ZOOM_RATIO_DEADBAND = 0.045
ZOOM_SNAP_MARKS = (1.00, 1.50, 2.00)
ZOOM_SNAP_WINDOW = 0.12
ZOOM_SNAP_STRENGTH = 0.55
ZOOM_VEL_NEUTRAL = 0.04
ZOOM_VEL_GLOW_REF = 0.95
ZOOM_DEPTH_DELTA_REF = 0.14
FIST_TOGGLE_COOLDOWN_SEC = 0.70
FIST_HOLD_TO_TOGGLE_SEC = 0.25
FIST_RELEASE_RESET_SEC = 0.25
CONTROL_WRITE_HZ = 30.0
WRIST_SMOOTH_ALPHA = 0.35
WRIST_HOLD_DEADZONE_DEG = 12.0
WRIST_MAX_EFFECT_DEG = 75.0
WRIST_ROTATE_MAX_DEG_PER_SEC = 210.0
HAND_X_DEADZONE_NORM = 0.038
HAND_X_MAX_EFFECT_NORM = 0.26
HAND_X_ROTATE_MAX_DEG_PER_SEC = 225.0
HAND_X_RESPONSE_EXP = 0.78
HAND_X_MIN_SPEED_FRACTION = 0.20
ROTATION_ACTIVATE_DELAY_SEC = 0.20
NEUTRAL_ANGLE_ADAPT_ALPHA = 0.12
NEUTRAL_X_ADAPT_ALPHA = 0.02
PITCH_CENTER_Y = 0.55
PITCH_DEADZONE_NORM = 0.07
PITCH_MAX_EFFECT_NORM = 0.34
PITCH_MAX_DEG_PER_SEC = 115.0
PITCH_MIN_DEG = -68.0
PITCH_MAX_DEG = 68.0
ZOOM_MOVE_ALPHA = 0.35
ZOOM_MOVE_X_DEADZONE_NORM = 0.035
ZOOM_MOVE_Y_DEADZONE_NORM = 0.040
ZOOM_MOVE_MAX_EFFECT_NORM = 0.30
ZOOM_MOVE_X_MAX_DEG_PER_SEC = 140.0
ZOOM_MOVE_Y_MAX_DEG_PER_SEC = 140.0
ZOOM_MOVE_RESPONSE_EXP = 1.15
ZOOM_MOVE_ZOOM_DAMPING_EXP = 1.25
ZOOM_MOVE_ZOOM_DAMPING_MIN = 0.28
TWO_HAND_DISTANCE_NEAR = 1.2
TWO_HAND_DISTANCE_FAR = 4.8
PINCH_POSE_THRESHOLD = 0.36
PINCH_STEP_COOLDOWN_SEC = 0.30
PINCH_STATE_DOUBLE_WINDOW_SEC = 0.55
PINCH_STATE_HOLD_SEC = 0.65
PINCH_STATE_RESOLVE_HOLD_SEC = 0.45

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
CONTROL_OUTPUT_PATH = Path(__file__).resolve().parent / "live_controls.txt"


@dataclass
class HandObservation:
    label: str
    score: float
    landmarks: List[Tuple[float, float, float]]


@dataclass
class CameraControlState:
    zoom: float = 1.0
    rotation_deg: float = 0.0
    pitch_deg: float = 0.0
    wave_amp: float = 1.0
    paused: bool = False
    camera_locked: bool = False
    demo_phase: float = 0.0
    single_fist_latched: bool = False
    dual_fist_latched: bool = False
    single_fist_hold_start_ts: float = 0.0
    dual_fist_hold_start_ts: float = 0.0
    fist_release_start_ts: float = 0.0
    last_single_toggle_ts: float = 0.0
    last_dual_toggle_ts: float = 0.0
    last_frame_ts: float = 0.0
    last_write_ts: float = 0.0
    source_label: str = "Unknown"
    source_gesture: str = "none"
    source_pinch_ratio: float = 0.0
    wrist_has_prev: bool = False
    wrist_prev_deg: float = 0.0
    wrist_smooth_deg: float = 0.0
    right_pinch_latched: bool = False
    left_pinch_latched: bool = False
    last_n_step_ts: float = 0.0
    n_inc_count: int = 0
    n_dec_count: int = 0
    zoom_gesture_active: bool = False
    zoom_anchor_metric: float = 0.0
    zoom_anchor_zoom: float = 1.0
    zoom_anchor_amp: float = 1.0
    zoom_line_active: bool = False
    zoom_line_ax: float = 0.5
    zoom_line_ay: float = 0.5
    zoom_line_bx: float = 0.5
    zoom_line_by: float = 0.5
    zoom_velocity: float = 0.0
    zoom_confidence: float = 0.0
    zoom_move_anchor_active: bool = False
    zoom_move_anchor_x: float = 0.50
    zoom_move_anchor_y: float = 0.50
    zoom_move_smooth_x: float = 0.50
    zoom_move_smooth_y: float = 0.50
    neutral_initialized: bool = False
    neutral_wrist_deg: float = -90.0
    neutral_wrist_x: float = 0.50
    rotation_enable_ts: float = 0.0
    right_pinch_pending_count: int = 0
    left_pinch_pending_count: int = 0
    right_pinch_pending_deadline_ts: float = 0.0
    left_pinch_pending_deadline_ts: float = 0.0
    right_pinch_state: str = "idle"
    left_pinch_state: str = "idle"
    right_pinch_state_until_ts: float = 0.0
    left_pinch_state_until_ts: float = 0.0
    pinch_suppressed_reason: str = "none"


@dataclass
class CameraWindowState:
    fullscreen: bool = False


class FPSCounter:
    def __init__(self, ema: float = FPS_EMA):
        self.ema = ema
        self._last_ts: float | None = None
        self._fps = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_ts is not None:
            dt = now - self._last_ts
            if dt > 0:
                inst_fps = 1.0 / dt
                if self._fps == 0.0:
                    self._fps = inst_fps
                else:
                    self._fps = self.ema * self._fps + (1.0 - self.ema) * inst_fps
        self._last_ts = now
        return self._fps


class LandmarkSmoother:
    def __init__(self, alpha: float = SMOOTHING_ALPHA):
        self.alpha = alpha
        self._state: Dict[str, List[Tuple[float, float, float]]] = {}

    def smooth(
        self, key: str, current: Sequence[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        prev = self._state.get(key)
        if prev is None or len(prev) != len(current):
            smoothed = list(current)
        else:
            smoothed = []
            for (px, py, pz), (cx, cy, cz) in zip(prev, current):
                sx = px * (1.0 - self.alpha) + cx * self.alpha
                sy = py * (1.0 - self.alpha) + cy * self.alpha
                sz = pz * (1.0 - self.alpha) + cz * self.alpha
                smoothed.append((sx, sy, sz))
        self._state[key] = smoothed
        return smoothed

    def prune(self, active_keys: Set[str]) -> None:
        stale = [k for k in self._state if k not in active_keys]
        for key in stale:
            del self._state[key]


def ensure_hand_model(model_path: Path) -> Path:
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    if model_path.exists() and model_path.stat().st_size == 0:
        model_path.unlink()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {model_path}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            print("SSL cert verification failed; retrying download without verification.")
            try:
                insecure_ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(MODEL_URL, context=insecure_ctx) as src:
                    with model_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Failed to download the hand model after SSL fallback."
                ) from fallback_exc
        else:
            raise RuntimeError(
                "Failed to download the hand model. "
                "Check internet access and retry."
            ) from exc
    except Exception as exc:
        raise RuntimeError("Failed to download the hand model.") from exc
    return model_path


def _dist_xy(
    landmarks: Sequence[Tuple[float, float, float]], idx_a: int, idx_b: int
) -> float:
    ax, ay, _ = landmarks[idx_a]
    bx, by, _ = landmarks[idx_b]
    return math.hypot(ax - bx, ay - by)


def _palm_scale(landmarks: Sequence[Tuple[float, float, float]]) -> float:
    scale = (
        _dist_xy(landmarks, WRIST, INDEX_MCP)
        + _dist_xy(landmarks, WRIST, PINKY_MCP)
        + _dist_xy(landmarks, INDEX_MCP, PINKY_MCP)
    ) / 3.0
    return max(scale, 1e-4)


def _finger_extended(
    landmarks: Sequence[Tuple[float, float, float]], tip: int, pip: int, palm_scale: float
) -> bool:
    tip_to_wrist = _dist_xy(landmarks, tip, WRIST)
    pip_to_wrist = _dist_xy(landmarks, pip, WRIST)
    return tip_to_wrist > (pip_to_wrist + 0.08 * palm_scale)


def classify_gesture(landmarks: Sequence[Tuple[float, float, float]]) -> str:
    palm_scale = _palm_scale(landmarks)

    thumb_index = _dist_xy(landmarks, THUMB_TIP, INDEX_TIP)
    thumb_middle = _dist_xy(landmarks, THUMB_TIP, MIDDLE_TIP)

    index_ext = _finger_extended(landmarks, INDEX_TIP, INDEX_PIP, palm_scale)
    middle_ext = _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, palm_scale)
    ring_ext = _finger_extended(landmarks, RING_TIP, RING_PIP, palm_scale)
    pinky_ext = _finger_extended(landmarks, PINKY_TIP, PINKY_PIP, palm_scale)

    if thumb_index < 0.36 * palm_scale and thumb_middle < 0.36 * palm_scale:
        return "two_finger_pinch"
    if thumb_index < 0.32 * palm_scale:
        return "pinch"
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "open_hand"
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "point"

    finger_to_wrist = [
        _dist_xy(landmarks, INDEX_TIP, WRIST),
        _dist_xy(landmarks, MIDDLE_TIP, WRIST),
        _dist_xy(landmarks, RING_TIP, WRIST),
        _dist_xy(landmarks, PINKY_TIP, WRIST),
    ]
    if (
        not index_ext
        and not middle_ext
        and not ring_ext
        and not pinky_ext
        and max(finger_to_wrist) < 1.85 * palm_scale
    ):
        return "fist"

    return "none"


def _pinch_ratio(landmarks: Sequence[Tuple[float, float, float]]) -> float:
    return _dist_xy(landmarks, THUMB_TIP, INDEX_TIP) / _palm_scale(landmarks)


def _wrist_angle_deg(landmarks: Sequence[Tuple[float, float, float]]) -> float:
    wx, wy, _ = landmarks[WRIST]
    mx, my, _ = landmarks[MIDDLE_MCP]
    return math.degrees(math.atan2(my - wy, mx - wx))


def _normalize_label(label: str) -> str:
    clean = (label or "Unknown").strip() or "Unknown"
    if not SWAP_HANDEDNESS_LABELS:
        return clean

    lower = clean.lower()
    if lower == "left":
        return "Right"
    if lower == "right":
        return "Left"
    return clean


def _prepare_for_mirrored_display(hands: Sequence[HandObservation]) -> List[HandObservation]:
    prepared: List[HandObservation] = []
    for hand in hands:
        mirrored_landmarks = [
            (min(1.0, max(0.0, 1.0 - x)), y, z) for x, y, z in hand.landmarks
        ]
        prepared.append(
            HandObservation(
                label=_normalize_label(hand.label),
                score=hand.score,
                landmarks=mirrored_landmarks,
            )
        )
    return prepared


def _draw_overlay_text(frame, fps: float, hand_count: int, control: CameraControlState) -> None:
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Hands: {hand_count}  Conf>={HAND_CONFIDENCE_THRESHOLD:.2f}",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Camera zoom={control.zoom:.2f}  Wave amp={control.wave_amp:.2f}  "
        f"yaw={control.rotation_deg:.1f}  pitch={control.pitch_deg:.1f}  "
        f"{'PAUSED' if control.paused else 'PLAY'}  "
        f"{'LOCKED' if control.camera_locked else 'UNLOCKED'}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "1 fist=cam lock  2 fists=pause  dual pinch=zoom+move camera  Right/Left pinch=]/[",
        (10, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def _draw_state_machine_hud(frame, control: CameraControlState) -> None:
    panel_x = 10
    panel_y = 126
    panel_w = 430
    panel_h = 106
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (42, 42, 42), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (78, 78, 78), 1)

    zoom_txt = "zooming" if control.zoom_gesture_active else "idle"
    zoom_col = (120, 235, 255) if control.zoom_gesture_active else (188, 188, 188)
    cv2.putText(
        frame,
        "STATE MACHINE HUD",
        (panel_x + 10, panel_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (250, 240, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"zoom: {zoom_txt}",
        (panel_x + 10, panel_y + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        zoom_col,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"right pinch: {control.right_pinch_state}",
        (panel_x + 10, panel_y + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (215, 240, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"left pinch:  {control.left_pinch_state}",
        (panel_x + 10, panel_y + 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (215, 240, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"suppressed: {control.pinch_suppressed_reason}",
        (panel_x + 10, panel_y + 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 210, 240),
        1,
        cv2.LINE_AA,
    )


def _draw_controls_guide(frame) -> None:
    h, w, _ = frame.shape
    bar_h = 112
    top = max(0, h - bar_h)

    # Soft dark strip at bottom to keep the main view clean and text readable.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.line(frame, (0, top), (w, top), (70, 70, 70), 1)

    title = "CONTROLS"
    line_1 = "One-hand fist: camera lock/unlock | Two-hand fists: pause/play"
    line_2 = "Dual thumb-index pinch: spread apart=zoom in, together=zoom out"
    line_3 = "Single pinch: Right hand=] (n+1) | Left hand=[ (n-1)"
    line_4 = "While dual pinch is active: move pinch line center to rotate/pitch camera"

    # Shadow + foreground text for better readability on any background.
    def draw_text(txt: str, x: int, y: int, size: float, color: Tuple[int, int, int], thick: int):
        cv2.putText(
            frame, txt, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA
        )

    draw_text(title, 16, top + 22, 0.62, (255, 245, 180), 2)
    draw_text(line_1, 16, top + 42, 0.50, (235, 235, 235), 1)
    draw_text(line_2, 16, top + 60, 0.49, (230, 230, 210), 1)
    draw_text(line_3, 16, top + 78, 0.49, (220, 220, 220), 1)
    draw_text(line_4, 16, top + 96, 0.49, (215, 215, 215), 1)


def _draw_camera_preview(frame, control: CameraControlState) -> None:
    h, w, _ = frame.shape
    panel_w = 250
    panel_h = 210
    panel_x = max(10, w - panel_w - 12)
    panel_y = 12

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (55, 55, 55), 2)
    cv2.putText(
        frame,
        "Camera Preview",
        (panel_x + 14, panel_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    cx = panel_x + panel_w // 2
    cy = panel_y + panel_h // 2 + 12
    cube_scale = 34.0 * control.zoom

    yaw = math.radians(control.rotation_deg)
    pitch = math.radians(control.pitch_deg * 0.9)
    cyaw = math.cos(yaw)
    syaw = math.sin(yaw)
    cpitch = math.cos(pitch)
    spitch = math.sin(pitch)

    # Wireframe cube so yaw/pitch changes are obvious.
    cube = [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    projected: List[Tuple[int, int, float]] = []
    rotated: List[Tuple[float, float, float]] = []
    for x, y, z in cube:
        # yaw around Y axis
        x1 = x * cyaw + z * syaw
        z1 = -x * syaw + z * cyaw
        # pitch around X axis
        y2 = y * cpitch - z1 * spitch
        z2 = y * spitch + z1 * cpitch

        perspective = 2.7 / (3.5 + z2)
        sx = int(cx + x1 * cube_scale * perspective)
        sy = int(cy + y2 * cube_scale * perspective)
        projected.append((sx, sy, z2))
        rotated.append((x1, y2, z2))

    # Fill faces first so the cube is solid (not see-through), then draw edges on top.
    faces = [
        (0, 1, 2, 3),  # back
        (4, 5, 6, 7),  # front
        (0, 3, 7, 4),  # left
        (1, 2, 6, 5),  # right
        (3, 2, 6, 7),  # top
        (0, 1, 5, 4),  # bottom
    ]
    light = (-0.40, -0.30, -0.86)
    light_mag = max(1e-6, math.sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]))
    lx, ly, lz = (light[0] / light_mag, light[1] / light_mag, light[2] / light_mag)
    base_color = (108, 185, 235)  # BGR
    face_draw_list: List[Tuple[float, np.ndarray, Tuple[int, int, int]]] = []

    for face in faces:
        p0 = rotated[face[0]]
        p1 = rotated[face[1]]
        p2 = rotated[face[2]]
        ux, uy, uz = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        vx, vy, vz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        nmag = max(1e-6, math.sqrt(nx * nx + ny * ny + nz * nz))
        nx, ny, nz = (nx / nmag, ny / nmag, nz / nmag)

        shade = 0.35 + 0.65 * abs(nx * lx + ny * ly + nz * lz)
        face_color = (
            int(_clamp(base_color[0] * shade, 0.0, 255.0)),
            int(_clamp(base_color[1] * shade, 0.0, 255.0)),
            int(_clamp(base_color[2] * shade, 0.0, 255.0)),
        )

        poly = np.array(
            [(projected[idx][0], projected[idx][1]) for idx in face],
            dtype=np.int32,
        )
        depth = sum(rotated[idx][2] for idx in face) / 4.0
        face_draw_list.append((depth, poly, face_color))

    # Painter's algorithm: far faces first, near faces last.
    face_draw_list.sort(key=lambda item: item[0], reverse=True)
    for _, poly, face_color in face_draw_list:
        cv2.fillConvexPoly(frame, poly, face_color, cv2.LINE_AA)

    for a, b in edges:
        p0 = projected[a]
        p1 = projected[b]
        back_edge = (p0[2] + p1[2]) < 0.0
        color = (85, 140, 170) if back_edge else (90, 210, 255)
        thick = 1 if back_edge else 2
        cv2.line(frame, (p0[0], p0[1]), (p1[0], p1[1]), color, thick, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"yaw {control.rotation_deg:.0f}  pitch {control.pitch_deg:.0f}",
        (panel_x + 14, panel_y + panel_h - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (188, 210, 230),
        1,
        cv2.LINE_AA,
    )

    state_text = "PAUSED" if control.paused else "PLAY"
    color = (90, 200, 255) if control.paused else (120, 255, 120)
    cv2.putText(
        frame,
        state_text,
        (panel_x + 14, panel_y + panel_h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_wave_panel(frame, control: CameraControlState) -> None:
    h, w, _ = frame.shape
    panel_w = 340
    panel_h = 180
    panel_x = max(10, w - panel_w - 12)
    panel_y = min(h - panel_h - 12, 230)
    panel_y = max(12, panel_y)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (55, 55, 55), 2)
    cv2.putText(
        frame,
        "Astro Wave (Amplification Demo)",
        (panel_x + 10, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )

    graph_left = panel_x + 16
    graph_right = panel_x + panel_w - 16
    graph_top = panel_y + 38
    graph_bottom = panel_y + panel_h - 20
    graph_mid = (graph_top + graph_bottom) // 2

    cv2.line(frame, (graph_left, graph_mid), (graph_right, graph_mid), (85, 85, 85), 1)

    amp_norm = (control.wave_amp - WAVE_AMP_MIN) / (WAVE_AMP_MAX - WAVE_AMP_MIN)
    amp_norm = max(0.0, min(1.0, amp_norm))
    amplitude_px = 12 + int(50 * amp_norm)

    freq_cycles = 2.2
    width = graph_right - graph_left
    prev_x = graph_left
    prev_y = graph_mid
    for i in range(1, width + 1):
        x = graph_left + i
        t = i / max(1, width)
        angle = 2.0 * math.pi * freq_cycles * t + control.demo_phase * 1.3
        y = int(graph_mid + amplitude_px * math.sin(angle))
        cv2.line(frame, (prev_x, prev_y), (x, y), (100, 220, 255), 2)
        prev_x, prev_y = x, y

    cv2.putText(
        frame,
        f"Amplitude: {control.wave_amp:.2f}",
        (panel_x + 12, panel_y + panel_h - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 225, 130),
        1,
        cv2.LINE_AA,
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _two_hand_targets(distance_metric: float) -> Tuple[float, float]:
    norm = (distance_metric - TWO_HAND_DISTANCE_NEAR) / (
        TWO_HAND_DISTANCE_FAR - TWO_HAND_DISTANCE_NEAR
    )
    norm = _clamp(norm, 0.0, 1.0)

    target_zoom = ZOOM_MIN + norm * (ZOOM_MAX - ZOOM_MIN)
    target_amp = WAVE_AMP_MIN + norm * (WAVE_AMP_MAX - WAVE_AMP_MIN)
    return target_zoom, target_amp


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _angle_delta_deg(current: float, previous: float) -> float:
    return _normalize_angle_deg(current - previous)


def _wrist_hold_rotation_speed_deg_per_sec(wrist_deg: float, neutral_deg: float) -> float:
    offset = _normalize_angle_deg(wrist_deg - neutral_deg)
    mag = abs(offset)
    if mag <= WRIST_HOLD_DEADZONE_DEG:
        return 0.0

    norm = (mag - WRIST_HOLD_DEADZONE_DEG) / max(
        1e-6, WRIST_MAX_EFFECT_DEG - WRIST_HOLD_DEADZONE_DEG
    )
    norm = _clamp(norm, 0.0, 1.0)
    speed = WRIST_ROTATE_MAX_DEG_PER_SEC * norm
    return speed if offset > 0 else -speed


def _hand_x_rotation_speed_deg_per_sec(wrist_x: float, neutral_x: float) -> float:
    offset = wrist_x - neutral_x
    mag = abs(offset)
    if mag <= HAND_X_DEADZONE_NORM:
        return 0.0

    norm = (mag - HAND_X_DEADZONE_NORM) / max(
        1e-6, HAND_X_MAX_EFFECT_NORM - HAND_X_DEADZONE_NORM
    )
    norm = _clamp(norm, 0.0, 1.0)
    norm = norm ** HAND_X_RESPONSE_EXP
    speed_norm = HAND_X_MIN_SPEED_FRACTION + (1.0 - HAND_X_MIN_SPEED_FRACTION) * norm
    speed = HAND_X_ROTATE_MAX_DEG_PER_SEC * speed_norm
    return speed if offset > 0 else -speed


def _pitch_speed_deg_per_sec(wrist_y: float) -> float:
    # Symmetric pitch response for up/down hand movement.
    offset = PITCH_CENTER_Y - wrist_y
    mag = abs(offset)
    if mag <= PITCH_DEADZONE_NORM:
        return 0.0

    norm = (mag - PITCH_DEADZONE_NORM) / max(
        1e-6, PITCH_MAX_EFFECT_NORM - PITCH_DEADZONE_NORM
    )
    norm = _clamp(norm, 0.0, 1.0)
    speed = PITCH_MAX_DEG_PER_SEC * norm
    return speed if offset > 0 else -speed


def _axis_speed_from_offset(
    offset: float,
    deadzone: float,
    max_effect: float,
    max_speed: float,
    response_exp: float,
) -> float:
    mag = abs(offset)
    if mag <= deadzone:
        return 0.0

    norm = (mag - deadzone) / max(1e-6, max_effect - deadzone)
    norm = _clamp(norm, 0.0, 1.0)
    norm = norm ** response_exp
    speed = max_speed * norm
    return speed if offset > 0 else -speed


def _zoom_move_sensitivity_scale(zoom: float) -> float:
    # When zoomed in, damp pan/tilt response so tiny hand motion doesn't over-rotate.
    z = max(1.0, zoom)
    scale = 1.0 / (z ** ZOOM_MOVE_ZOOM_DAMPING_EXP)
    return _clamp(scale, ZOOM_MOVE_ZOOM_DAMPING_MIN, 1.0)


def _is_thumb_index_pinch_pose(landmarks: Sequence[Tuple[float, float, float]]) -> Tuple[bool, float]:
    palm = _palm_scale(landmarks)
    ti_dist = _dist_xy(landmarks, THUMB_TIP, INDEX_TIP)
    pinching = ti_dist < PINCH_POSE_THRESHOLD * palm
    return pinching, palm


def _two_hand_zoom_pair(candidates: Sequence[dict]) -> Tuple[dict, dict, float] | None:
    eligible = [c for c in candidates if c.get("pinch_pose", False)]
    if len(eligible) < 2:
        return None

    eligible.sort(key=lambda c: c["score"], reverse=True)
    a, b = eligible[0], eligible[1]
    ax, ay = a["pair_center"]
    bx, by = b["pair_center"]
    center_dist = math.hypot(ax - bx, ay - by)
    scale = max(1e-4, 0.5 * (a["palm_scale"] + b["palm_scale"]))
    metric = center_dist / scale
    return a, b, metric


def _two_hand_zoom_metric(candidates: Sequence[dict]) -> float | None:
    zoom_pair = _two_hand_zoom_pair(candidates)
    if zoom_pair is None:
        return None
    _, _, metric = zoom_pair
    return metric


def _zoom_pinch_confidence(a: dict, b: dict) -> float:
    # Lower pinch ratios => stronger pinch confidence.
    denom = max(1e-6, PINCH_POSE_THRESHOLD * 0.75)
    conf_a = _clamp((PINCH_POSE_THRESHOLD - a["pinch_ratio"]) / denom, 0.0, 1.0)
    conf_b = _clamp((PINCH_POSE_THRESHOLD - b["pinch_ratio"]) / denom, 0.0, 1.0)
    return 0.5 * (conf_a + conf_b)


def _zoom_color_for_velocity(zoom_velocity: float) -> Tuple[int, int, int]:
    # BGR colors for OpenCV.
    # User preference: blue while moving, green while mostly still.
    if abs(zoom_velocity) > ZOOM_VEL_NEUTRAL:
        return (255, 120, 35)  # vivid blue (moving)
    return (35, 255, 95)  # vivid green (still)


def _scale_bgr(color: Tuple[int, int, int], scale: float) -> Tuple[int, int, int]:
    return (
        int(_clamp(color[0] * scale, 0.0, 255.0)),
        int(_clamp(color[1] * scale, 0.0, 255.0)),
        int(_clamp(color[2] * scale, 0.0, 255.0)),
    )


def _lerp_bgr(
    a: Tuple[int, int, int], b: Tuple[int, int, int], t: float
) -> Tuple[int, int, int]:
    return (
        int((1.0 - t) * a[0] + t * b[0]),
        int((1.0 - t) * a[1] + t * b[1]),
        int((1.0 - t) * a[2] + t * b[2]),
    )


def _draw_zoom_depth_line(
    frame,
    p_far: Tuple[int, int],
    p_near: Tuple[int, int],
    line_color: Tuple[int, int, int],
    depth_strength: float,
    glow_norm: float,
) -> None:
    # Perspective cue: far end dimmer/thinner, near end brighter/thicker.
    segments = 20
    base_thickness = 2 + int(2 * glow_norm)
    far_color = _scale_bgr(line_color, 0.72)
    near_color = _scale_bgr(line_color, 1.08)
    glow_color = _scale_bgr(line_color, 0.52)

    # Soft colored glow so the line stays vivid instead of looking black.
    cv2.line(frame, p_far, p_near, glow_color, base_thickness + 6, cv2.LINE_AA)

    for i in range(segments):
        t0 = i / segments
        t1 = (i + 1) / segments
        t_mid = 0.5 * (t0 + t1)

        x0 = int((1.0 - t0) * p_far[0] + t0 * p_near[0])
        y0 = int((1.0 - t0) * p_far[1] + t0 * p_near[1])
        x1 = int((1.0 - t1) * p_far[0] + t1 * p_near[0])
        y1 = int((1.0 - t1) * p_far[1] + t1 * p_near[1])

        color = _lerp_bgr(far_color, near_color, t_mid)
        thick_scale = 0.92 + depth_strength * (0.12 + 0.78 * t_mid)
        thickness = max(2, int(base_thickness * thick_scale))
        cv2.line(frame, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

    far_r = 5 + int(2 * depth_strength)
    near_r = 5 + int(5 * depth_strength)
    cv2.circle(frame, p_far, far_r, far_color, -1, cv2.LINE_AA)
    cv2.circle(frame, p_near, near_r + 2, _scale_bgr(near_color, 0.72), 2, cv2.LINE_AA)
    cv2.circle(frame, p_near, near_r, near_color, -1, cv2.LINE_AA)


def _export_live_controls(control: CameraControlState, now: float) -> None:
    interval = 1.0 / max(1e-3, CONTROL_WRITE_HZ)
    if now - control.last_write_ts < interval:
        return
    control.last_write_ts = now

    timestamp_ms = int(time.time() * 1000)
    payload = (
        f"timestamp_ms={timestamp_ms}\n"
        f"zoom={control.zoom:.6f}\n"
        f"rotation_deg={control.rotation_deg:.6f}\n"
        f"pitch_deg={control.pitch_deg:.6f}\n"
        f"wave_amp={control.wave_amp:.6f}\n"
        f"paused={1 if control.paused else 0}\n"
        f"camera_locked={1 if control.camera_locked else 0}\n"
        f"zoom_line_active={1 if control.zoom_line_active else 0}\n"
        f"zoom_line_ax={control.zoom_line_ax:.6f}\n"
        f"zoom_line_ay={control.zoom_line_ay:.6f}\n"
        f"zoom_line_bx={control.zoom_line_bx:.6f}\n"
        f"zoom_line_by={control.zoom_line_by:.6f}\n"
        f"label={control.source_label}\n"
        f"gesture={control.source_gesture}\n"
        f"pinch_ratio={control.source_pinch_ratio:.6f}\n"
        f"n_inc_count={control.n_inc_count}\n"
        f"n_dec_count={control.n_dec_count}\n"
    )

    tmp_path = CONTROL_OUTPUT_PATH.with_suffix(".tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, CONTROL_OUTPUT_PATH)
    except OSError:
        # Keep tracker running even if file write fails intermittently.
        pass


def _select_control_hand(candidates: Sequence[dict]) -> dict | None:
    if not candidates:
        return None

    right_hands = [c for c in candidates if c["label"].lower() == "right"]
    if right_hands:
        return max(right_hands, key=lambda c: c["score"])
    return max(candidates, key=lambda c: c["score"])


def _best_hand_by_label(candidates: Sequence[dict], wanted_label: str) -> dict | None:
    wanted = wanted_label.lower()
    matching = [c for c in candidates if c["label"].lower() == wanted]
    if not matching:
        return None
    return max(matching, key=lambda c: c["score"])


def _update_camera_controls(control: CameraControlState, candidates: Sequence[dict]) -> None:
    now = time.perf_counter()
    if control.last_frame_ts == 0.0:
        control.last_frame_ts = now
    dt = max(0.0, now - control.last_frame_ts)
    control.last_frame_ts = now

    active = _select_control_hand(candidates)
    control.pinch_suppressed_reason = "none"

    if active is not None:
        control.source_label = active["label"]
        control.source_gesture = active["gesture"]
        control.source_pinch_ratio = active["pinch_ratio"]

        zoom_before = control.zoom
        zoom_pair = _two_hand_zoom_pair(candidates)
        zoom_metric = zoom_pair[2] if zoom_pair is not None else None
        if zoom_pair is not None:
            a, b, _ = zoom_pair
            control.zoom_line_active = True
            control.zoom_line_ax, control.zoom_line_ay = a["pair_center"]
            control.zoom_line_bx, control.zoom_line_by = b["pair_center"]
            control.zoom_confidence = _zoom_pinch_confidence(a, b)
        else:
            control.zoom_line_active = False
            control.zoom_confidence = 0.0

        if zoom_metric is not None:
            if not control.zoom_gesture_active:
                control.zoom_gesture_active = True
                control.zoom_anchor_metric = max(1e-4, zoom_metric)
                control.zoom_anchor_zoom = control.zoom
                control.zoom_anchor_amp = control.wave_amp
                if zoom_pair is not None:
                    a, b, _ = zoom_pair
                    cx = 0.5 * (a["pair_center"][0] + b["pair_center"][0])
                    cy = 0.5 * (a["pair_center"][1] + b["pair_center"][1])
                    control.zoom_move_anchor_active = True
                    control.zoom_move_anchor_x = cx
                    control.zoom_move_anchor_y = cy
                    control.zoom_move_smooth_x = cx
                    control.zoom_move_smooth_y = cy

            ratio = zoom_metric / max(1e-4, control.zoom_anchor_metric)
            if abs(ratio - 1.0) < ZOOM_RATIO_DEADBAND:
                ratio = 1.0
            ratio = _clamp(ratio, ZOOM_RATIO_MIN, ZOOM_RATIO_MAX)
            target_zoom = _clamp(
                control.zoom_anchor_zoom * (ratio ** ZOOM_RATIO_POWER),
                ZOOM_MIN,
                ZOOM_MAX,
            )
            nearest_snap = min(ZOOM_SNAP_MARKS, key=lambda s: abs(target_zoom - s))
            snap_dist = abs(target_zoom - nearest_snap)
            if snap_dist < ZOOM_SNAP_WINDOW:
                snap_t = 1.0 - snap_dist / max(1e-6, ZOOM_SNAP_WINDOW)
                snap_alpha = ZOOM_SNAP_STRENGTH * snap_t
                target_zoom = (1.0 - snap_alpha) * target_zoom + snap_alpha * nearest_snap
            target_amp = _clamp(
                control.zoom_anchor_amp * (ratio ** AMP_RATIO_POWER),
                WAVE_AMP_MIN,
                WAVE_AMP_MAX,
            )
            control.zoom = (1.0 - ZOOM_GESTURE_LERP) * control.zoom + ZOOM_GESTURE_LERP * target_zoom
            control.wave_amp = (1.0 - ZOOM_GESTURE_LERP) * control.wave_amp + ZOOM_GESTURE_LERP * target_amp
        else:
            control.zoom_gesture_active = False
            control.zoom_move_anchor_active = False

        if dt > 1e-6:
            control.zoom_velocity = (control.zoom - zoom_before) / dt
        else:
            control.zoom_velocity = 0.0

        if not control.camera_locked:
            if zoom_pair is not None and dt > 1e-6:
                a, b, _ = zoom_pair
                cx = 0.5 * (a["pair_center"][0] + b["pair_center"][0])
                cy = 0.5 * (a["pair_center"][1] + b["pair_center"][1])

                if not control.zoom_move_anchor_active:
                    control.zoom_move_anchor_active = True
                    control.zoom_move_anchor_x = cx
                    control.zoom_move_anchor_y = cy
                    control.zoom_move_smooth_x = cx
                    control.zoom_move_smooth_y = cy
                else:
                    control.zoom_move_smooth_x = (
                        (1.0 - ZOOM_MOVE_ALPHA) * control.zoom_move_smooth_x + ZOOM_MOVE_ALPHA * cx
                    )
                    control.zoom_move_smooth_y = (
                        (1.0 - ZOOM_MOVE_ALPHA) * control.zoom_move_smooth_y + ZOOM_MOVE_ALPHA * cy
                    )

                # Continuous drive: while the dual-pinch line center is held off-center,
                # keep rotating/pitching instead of depending on a transient anchor delta.
                x_offset = 0.50 - control.zoom_move_smooth_x
                y_offset = PITCH_CENTER_Y - control.zoom_move_smooth_y
                rot_speed = _axis_speed_from_offset(
                    x_offset,
                    ZOOM_MOVE_X_DEADZONE_NORM,
                    ZOOM_MOVE_MAX_EFFECT_NORM,
                    ZOOM_MOVE_X_MAX_DEG_PER_SEC,
                    ZOOM_MOVE_RESPONSE_EXP,
                )
                pitch_speed = _axis_speed_from_offset(
                    y_offset,
                    ZOOM_MOVE_Y_DEADZONE_NORM,
                    ZOOM_MOVE_MAX_EFFECT_NORM,
                    ZOOM_MOVE_Y_MAX_DEG_PER_SEC,
                    ZOOM_MOVE_RESPONSE_EXP,
                )
                zoom_scale = _zoom_move_sensitivity_scale(control.zoom)
                rot_speed *= zoom_scale
                pitch_speed *= zoom_scale

                if rot_speed != 0.0:
                    control.rotation_deg = _normalize_angle_deg(control.rotation_deg + rot_speed * dt)
                if pitch_speed != 0.0:
                    control.pitch_deg = _normalize_angle_deg(control.pitch_deg + pitch_speed * dt)
            else:
                control.zoom_move_anchor_active = False
        else:
            control.zoom_move_anchor_active = False

        # One-shot pinch mappings for quantum level stepping:
        # Right-hand pinch => ] (n+1), Left-hand pinch => [ (n-1).
        right_hand = _best_hand_by_label(candidates, "right")
        left_hand = _best_hand_by_label(candidates, "left")
        right_pinching = bool(right_hand and right_hand["pinch_pose"])
        left_pinching = bool(left_hand and left_hand["pinch_pose"])
        zoom_active = zoom_metric is not None

        if right_pinching:
            cooldown_ok = (now - control.last_n_step_ts) >= PINCH_STEP_COOLDOWN_SEC
            is_edge = not control.right_pinch_latched
            if is_edge:
                if zoom_active:
                    control.pinch_suppressed_reason = "zooming"
                    control.right_pinch_pending_count = 0
                    control.right_pinch_pending_deadline_ts = 0.0
                    control.right_pinch_state = "suppressed"
                    control.right_pinch_state_until_ts = now + PINCH_STATE_HOLD_SEC
                elif not cooldown_ok:
                    control.pinch_suppressed_reason = "cooldown"
                    control.right_pinch_state = "suppressed"
                    control.right_pinch_state_until_ts = now + PINCH_STATE_HOLD_SEC
                else:
                    control.right_pinch_pending_count += 1
                    if control.right_pinch_pending_count >= 2:
                        control.right_pinch_state = "double-pinch detected"
                        control.right_pinch_state_until_ts = now + PINCH_STATE_HOLD_SEC
                        control.right_pinch_pending_count = 0
                        control.right_pinch_pending_deadline_ts = 0.0
                    else:
                        control.right_pinch_state = "single-pinch pending"
                        control.right_pinch_pending_deadline_ts = now + PINCH_STATE_DOUBLE_WINDOW_SEC
                        control.right_pinch_state_until_ts = control.right_pinch_pending_deadline_ts

            if is_edge and (not zoom_active) and cooldown_ok:
                control.n_inc_count += 1
                control.last_n_step_ts = now
            control.right_pinch_latched = True
        else:
            control.right_pinch_latched = False

        if left_pinching:
            cooldown_ok = (now - control.last_n_step_ts) >= PINCH_STEP_COOLDOWN_SEC
            is_edge = not control.left_pinch_latched
            if is_edge:
                if zoom_active:
                    control.pinch_suppressed_reason = "zooming"
                    control.left_pinch_pending_count = 0
                    control.left_pinch_pending_deadline_ts = 0.0
                    control.left_pinch_state = "suppressed"
                    control.left_pinch_state_until_ts = now + PINCH_STATE_HOLD_SEC
                elif not cooldown_ok:
                    control.pinch_suppressed_reason = "cooldown"
                    control.left_pinch_state = "suppressed"
                    control.left_pinch_state_until_ts = now + PINCH_STATE_HOLD_SEC
                else:
                    control.left_pinch_pending_count += 1
                    if control.left_pinch_pending_count >= 2:
                        control.left_pinch_state = "double-pinch detected"
                        control.left_pinch_state_until_ts = now + PINCH_STATE_HOLD_SEC
                        control.left_pinch_pending_count = 0
                        control.left_pinch_pending_deadline_ts = 0.0
                    else:
                        control.left_pinch_state = "single-pinch pending"
                        control.left_pinch_pending_deadline_ts = now + PINCH_STATE_DOUBLE_WINDOW_SEC
                        control.left_pinch_state_until_ts = control.left_pinch_pending_deadline_ts

            if is_edge and (not zoom_active) and cooldown_ok:
                control.n_dec_count += 1
                control.last_n_step_ts = now
            control.left_pinch_latched = True
        else:
            control.left_pinch_latched = False

        fist_count = sum(1 for c in candidates if c["gesture"] == "fist")
        if fist_count >= 2:
            if control.dual_fist_hold_start_ts == 0.0:
                control.dual_fist_hold_start_ts = now
            control.single_fist_hold_start_ts = 0.0
            control.fist_release_start_ts = 0.0

            hold_ok = (now - control.dual_fist_hold_start_ts) >= FIST_HOLD_TO_TOGGLE_SEC
            cooldown_ok = (now - control.last_dual_toggle_ts) >= FIST_TOGGLE_COOLDOWN_SEC
            if hold_ok and cooldown_ok and not control.dual_fist_latched:
                control.paused = not control.paused
                control.last_dual_toggle_ts = now
                control.dual_fist_latched = True
        elif fist_count == 1 and len(candidates) == 1:
            if control.single_fist_hold_start_ts == 0.0:
                control.single_fist_hold_start_ts = now
            control.dual_fist_hold_start_ts = 0.0
            control.fist_release_start_ts = 0.0

            hold_ok = (now - control.single_fist_hold_start_ts) >= FIST_HOLD_TO_TOGGLE_SEC
            cooldown_ok = (now - control.last_single_toggle_ts) >= FIST_TOGGLE_COOLDOWN_SEC
            if hold_ok and cooldown_ok and not control.single_fist_latched:
                control.camera_locked = not control.camera_locked
                control.last_single_toggle_ts = now
                control.single_fist_latched = True
        else:
            control.single_fist_hold_start_ts = 0.0
            control.dual_fist_hold_start_ts = 0.0
            if control.fist_release_start_ts == 0.0:
                control.fist_release_start_ts = now
            if (now - control.fist_release_start_ts) >= FIST_RELEASE_RESET_SEC:
                control.single_fist_latched = False
                control.dual_fist_latched = False
    else:
        control.single_fist_hold_start_ts = 0.0
        control.dual_fist_hold_start_ts = 0.0
        if control.fist_release_start_ts == 0.0:
            control.fist_release_start_ts = now
        if (now - control.fist_release_start_ts) >= FIST_RELEASE_RESET_SEC:
            control.single_fist_latched = False
            control.dual_fist_latched = False
        control.source_label = "Unknown"
        control.source_gesture = "none"
        control.source_pinch_ratio = 0.0
        control.wrist_has_prev = False
        control.neutral_initialized = False
        control.zoom_gesture_active = False
        control.zoom_line_active = False
        control.zoom_velocity = 0.0
        control.zoom_confidence = 0.0
        control.zoom_move_anchor_active = False
        control.right_pinch_latched = False
        control.left_pinch_latched = False

    if control.right_pinch_pending_count == 1 and now >= control.right_pinch_pending_deadline_ts > 0.0:
        control.right_pinch_state = "single-pinch detected"
        control.right_pinch_state_until_ts = now + PINCH_STATE_RESOLVE_HOLD_SEC
        control.right_pinch_pending_count = 0
        control.right_pinch_pending_deadline_ts = 0.0
    if control.left_pinch_pending_count == 1 and now >= control.left_pinch_pending_deadline_ts > 0.0:
        control.left_pinch_state = "single-pinch detected"
        control.left_pinch_state_until_ts = now + PINCH_STATE_RESOLVE_HOLD_SEC
        control.left_pinch_pending_count = 0
        control.left_pinch_pending_deadline_ts = 0.0

    if control.right_pinch_pending_count == 0 and now >= control.right_pinch_state_until_ts > 0.0:
        control.right_pinch_state = "idle"
        control.right_pinch_state_until_ts = 0.0
    if control.left_pinch_pending_count == 0 and now >= control.left_pinch_state_until_ts > 0.0:
        control.left_pinch_state = "idle"
        control.left_pinch_state_until_ts = 0.0

    if not control.paused:
        control.demo_phase += dt * 2.2

    _export_live_controls(control, now)


def _render_tracked_hands(
    frame,
    hands: Sequence[HandObservation],
    smoother: LandmarkSmoother,
    fps_counter: FPSCounter,
    control: CameraControlState,
    window_state: CameraWindowState,
    window_name: str,
) -> bool:
    h, w, _ = frame.shape

    counts_by_label: Dict[str, int] = {}
    active_keys: Set[str] = set()
    kept_hands = 0
    control_candidates: List[dict] = []

    for hand in hands:
        if hand.score < HAND_CONFIDENCE_THRESHOLD:
            continue

        kept_hands += 1
        label = hand.label or "Unknown"
        counts_by_label[label] = counts_by_label.get(label, 0) + 1
        hand_key = f"{label}:{counts_by_label[label]}"
        active_keys.add(hand_key)

        smoothed = smoother.smooth(hand_key, hand.landmarks)
        gesture = classify_gesture(smoothed)
        pinch_pose, palm_scale = _is_thumb_index_pinch_pose(smoothed)
        tx, ty, tz = smoothed[THUMB_TIP]
        ix, iy, iz = smoothed[INDEX_TIP]

        control_candidates.append(
            {
                "label": label,
                "score": hand.score,
                "gesture": gesture,
                "pinch_ratio": _pinch_ratio(smoothed),
                "wrist_angle": _wrist_angle_deg(smoothed),
                "wrist_x": smoothed[WRIST][0],
                "wrist_y": smoothed[WRIST][1],
                "pinch_pose": pinch_pose,
                "pair_center": ((tx + ix) * 0.5, (ty + iy) * 0.5),
                "pair_depth": 0.5 * (tz + iz),
                "palm_scale": palm_scale,
            }
        )

        for tip_id in TIP_IDS:
            x_norm, y_norm, _ = smoothed[tip_id]
            x_px = int(x_norm * w)
            y_px = int(y_norm * h)
            cv2.circle(frame, (x_px, y_px), 8, (0, 0, 255), -1)
            cv2.circle(frame, (x_px, y_px), 10, (255, 255, 255), 2)

        wrist_x, wrist_y, _ = smoothed[WRIST]
        text_x = max(10, int(wrist_x * w) - 30)
        text_y = max(30, int(wrist_y * h) - 20)

        cv2.putText(
            frame,
            f"{label} ({hand.score:.2f})",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (90, 255, 90),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            gesture.replace("_", " "),
            (text_x, text_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 230, 120),
            2,
            cv2.LINE_AA,
        )

    smoother.prune(active_keys)
    _update_camera_controls(control, control_candidates)
    zoom_pair = _two_hand_zoom_pair(control_candidates)
    if zoom_pair is not None and control.zoom_gesture_active:
        a, b, _ = zoom_pair
        ax, ay = a["pair_center"]
        bx, by = b["pair_center"]
        az = a.get("pair_depth", 0.0)
        bz = b.get("pair_depth", 0.0)

        line_color = _zoom_color_for_velocity(control.zoom_velocity)
        glow_norm = _clamp(abs(control.zoom_velocity) / ZOOM_VEL_GLOW_REF, 0.0, 1.0)
        depth_strength = _clamp(abs(az - bz) / ZOOM_DEPTH_DELTA_REF, 0.0, 1.0)

        p0 = (int(ax * w), int(ay * h))
        p1 = (int(bx * w), int(by * h))
        if az <= bz:
            p_near, p_far = p0, p1
        else:
            p_near, p_far = p1, p0
        _draw_zoom_depth_line(frame, p_far, p_near, line_color, depth_strength, glow_norm)

        mx = (p0[0] + p1[0]) // 2
        my = (p0[1] + p1[1]) // 2

        delta_pct = 0.0
        if control.zoom_anchor_zoom > 1e-6:
            delta_pct = (control.zoom / control.zoom_anchor_zoom - 1.0) * 100.0
        zoom_label = f"Zoom x{control.zoom:.2f} ({delta_pct:+.0f}%)"
        cv2.putText(
            frame,
            zoom_label,
            (mx + 1, my - 8 + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            zoom_label,
            (mx, my - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (245, 252, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"depth {int(depth_strength * 100):d}%",
            (mx, my + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (205, 230, 245),
            1,
            cv2.LINE_AA,
        )

        # Pinch confidence bar.
        bar_w = 110
        bar_h = 8
        bar_x = mx - bar_w // 2
        bar_y = my + 44
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 35, 35), -1)
        fill_w = int(bar_w * _clamp(control.zoom_confidence, 0.0, 1.0))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), line_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (215, 215, 215), 1)
        conf_txt = f"pinch {int(control.zoom_confidence * 100):d}%"
        cv2.putText(
            frame,
            conf_txt,
            (bar_x, bar_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

        # Snap marks near the bar.
        tick_y0 = bar_y + 26
        tick_y1 = tick_y0 + 7
        mark_spacing = bar_w // (len(ZOOM_SNAP_MARKS) - 1)
        for idx, mark in enumerate(ZOOM_SNAP_MARKS):
            tx = bar_x + idx * mark_spacing
            near_mark = abs(control.zoom - mark) <= ZOOM_SNAP_WINDOW
            tick_color = (120, 255, 170) if near_mark else (190, 190, 190)
            thickness = 3 if near_mark else 1
            cv2.line(frame, (tx, tick_y0), (tx, tick_y1), tick_color, thickness, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{mark:.1f}",
                (tx - 10, tick_y1 + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

    fps = fps_counter.update()
    _draw_overlay_text(frame, fps=fps, hand_count=kept_hands, control=control)
    _draw_state_machine_hud(frame, control)
    _draw_camera_preview(frame, control)
    _draw_wave_panel(frame, control)
    _draw_controls_guide(frame)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("f"), ord("F")):
        window_state.fullscreen = not window_state.fullscreen
        mode = cv2.WINDOW_FULLSCREEN if window_state.fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
    return key != ord("q")


def _extract_solution_hands(result) -> List[HandObservation]:
    extracted: List[HandObservation] = []

    if not result.multi_hand_landmarks:
        return extracted

    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        label = "Unknown"
        score = 1.0

        if result.multi_handedness and idx < len(result.multi_handedness):
            classification = result.multi_handedness[idx].classification[0]
            label = classification.label or "Unknown"
            score = float(classification.score)

        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        extracted.append(HandObservation(label=label, score=score, landmarks=landmarks))

    return extracted


def _extract_task_hands(result) -> List[HandObservation]:
    extracted: List[HandObservation] = []

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        label = "Unknown"
        score = 1.0

        if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
            best = result.handedness[idx][0]
            label = (
                getattr(best, "category_name", None)
                or getattr(best, "display_name", None)
                or "Unknown"
            )
            score = float(getattr(best, "score", 1.0))

        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        extracted.append(HandObservation(label=label, score=score, landmarks=landmarks))

    return extracted


def run_with_solutions(cap) -> None:
    mp_hands = mp.solutions.hands
    smoother = LandmarkSmoother()
    fps_counter = FPSCounter()
    control = CameraControlState()
    window_name = "Webcam Finger Tracker"
    window_state = CameraWindowState()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 820)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                continue

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            display_frame = cv2.flip(raw_frame, 1)
            extracted_hands = _prepare_for_mirrored_display(_extract_solution_hands(result))

            if not _render_tracked_hands(
                display_frame,
                extracted_hands,
                smoother,
                fps_counter,
                control,
                window_state,
                window_name,
            ):
                break


def run_with_tasks(cap) -> None:
    model_path = ensure_hand_model(MODEL_PATH)
    vision = mp.tasks.vision
    smoother = LandmarkSmoother()
    fps_counter = FPSCounter()
    control = CameraControlState()
    window_name = "Webcam Finger Tracker"
    window_state = CameraWindowState()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 820)

    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        last_timestamp_ms = 0
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                continue

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = max(last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
            last_timestamp_ms = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            display_frame = cv2.flip(raw_frame, 1)
            extracted_hands = _prepare_for_mirrored_display(_extract_task_hands(result))

            if not _render_tracked_hands(
                display_frame,
                extracted_hands,
                smoother,
                fps_counter,
                control,
                window_state,
                window_name,
            ):
                break


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (camera index 0).")
        return 1

    try:
        if hasattr(mp, "solutions"):
            run_with_solutions(cap)
        elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            run_with_tasks(cap)
        else:
            print(
                "Unsupported mediapipe build: expected either `solutions` or "
                "`tasks.vision` APIs."
            )
            return 1
    except RuntimeError as exc:
        error_text = str(exc)
        if "kGpuService" in error_text or "NSOpenGLPixelFormat" in error_text:
            print("Runtime error: MediaPipe failed to initialize OpenGL services.")
            print("Try running from a local GUI terminal session.")
            print(
                "If it still fails, use Python 3.11 in a virtualenv and reinstall "
                "opencv-python + mediapipe."
            )
        else:
            print(f"Runtime error: {exc}")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
