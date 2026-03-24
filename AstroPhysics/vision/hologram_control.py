#!/usr/bin/env python3
"""
Webcam hologram control.

Detailed hologram showcase with webcam hand control:
- one-hand pinch drags the active hologram
- two-hand pinch resizes and rotates it
- one-hand fist cycles to the next hologram
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


WINDOW_NAME = "Hologram Control"
MODULE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES: Tuple[Path, ...] = (
    MODULE_DIR / "models" / "hand_landmarker.task",
    MODULE_DIR.parent / "DefensiveSys" / "models" / "hand_landmarker.task",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])
CAPTURE_W = 1280
CAPTURE_H = 720
HUD_MARGIN = 22
MAX_TRACKED_HANDS = 4

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

SMOOTH_ALPHA = 0.28
PINCH_POSE_THRESHOLD = 0.36
MIN_SHAPE_SIZE = 90.0
MAX_SHAPE_SIZE = 360.0
DRAG_CATCH_MARGIN = 150.0
DUAL_ROT_GAIN = 3.6
DUAL_CENTER_GAIN = 3.4
DUAL_ZW_GAIN = 0.55
AUTO_ROT_ZW_SPEED = 0.42
FIST_SWITCH_COOLDOWN_SEC = 0.75
DUAL_CENTER_DEADZONE_NORM = 0.028
DUAL_CENTER_MAX_EFFECT_NORM = 0.24
DUAL_ANGLE_DEADZONE_RAD = math.radians(5.0)
DUAL_ANGLE_MAX_EFFECT_RAD = math.radians(50.0)
DUAL_AXIS_RESPONSE_EXP = 1.08
DUAL_YAW_MAX_SPEED = 2.30
DUAL_PITCH_MAX_SPEED = 2.05
DUAL_ROLL_MAX_SPEED = 2.10
IDLE_ROT_XZ_SPEED = 0.10
IDLE_ROT_YZ_SPEED = -0.08
UI_IDLE_FADE_DELAY_SEC = 1.2
UI_IDLE_FADE_DURATION_SEC = 1.2
TRANSITION_DURATION_SEC = 0.62
DEPTH_CONTROL_GAIN = 1800.0
MAX_DEPTH_OFFSET_FACTOR = 1.55
DETAIL_LOW_FPS = 28.0
DETAIL_MED_FPS = 42.0
PARAM_EDIT_GAIN_X = 2.2
PARAM_EDIT_GAIN_Y = 2.6
POSITION_VEL_BLEND = 0.24
DEPTH_VEL_BLEND = 0.22
ROTATION_VEL_BLEND = 0.20
TRANSLATION_DAMPING = 0.91
DEPTH_DAMPING = 0.89
ROTATION_DAMPING = 0.93
BOUNDARY_SPRING_GAIN = 7.6
ROTATION_SPRING_GAIN = 1.95
WOBBLE_DAMPING = 0.82
WOBBLE_TRANSLATION_GAIN = 0.0026
WOBBLE_ROTATION_GAIN = 0.22
WOBBLE_MAX_ANGLE = 0.26

HOLOGRAM_NAMES: Tuple[str, ...] = (
    "Tesseract",
    "Orbital Atom",
    "Black Hole Lens",
    "Solar Orrery",
    "Magnetic Cage",
)

HYPERCUBE_VERTICES = np.array(list(product((-1.0, 1.0), repeat=4)), dtype=np.float32)
HYPERCUBE_EDGES: Tuple[Tuple[int, int], ...] = tuple(
    (i, j)
    for i in range(len(HYPERCUBE_VERTICES))
    for j in range(i + 1, len(HYPERCUBE_VERTICES))
    if bin(i ^ j).count("1") == 1
)


@dataclass
class HandObservation:
    label: str
    score: float
    landmarks: List[Tuple[float, float, float]]


@dataclass
class HologramState:
    center_x: float
    center_y: float
    size: float
    rot_xy: float = 0.25
    rot_xz: float = 0.85
    rot_yz: float = -0.35
    rot_xw: float = 0.55
    rot_yw: float = -0.28
    rot_zw: float = 0.40
    depth_offset: float = 0.0
    bob_phase: float = 0.0
    dragging: bool = False
    drag_offset_x: float = 0.0
    drag_offset_y: float = 0.0
    drag_anchor_depth: float = 0.0
    drag_anchor_depth_offset: float = 0.0
    dual_active: bool = False
    dual_anchor_metric: float = 0.0
    dual_anchor_size: float = 0.0
    dual_anchor_angle: float = 0.0
    dual_anchor_center_x: float = 0.5
    dual_anchor_center_y: float = 0.5
    dual_anchor_depth: float = 0.0
    dual_anchor_depth_offset: float = 0.0
    dual_anchor_rot_xy: float = 0.0
    dual_anchor_rot_xw: float = 0.0
    dual_anchor_rot_yw: float = 0.0
    dual_anchor_rot_zw: float = 0.0
    mode_param_a: float = 0.50
    mode_param_b: float = 0.50
    dual_anchor_param_a: float = 0.50
    dual_anchor_param_b: float = 0.50
    mode_index: int = 0
    fist_latched: bool = False
    last_mode_switch_ts: float = 0.0
    previous_mode_index: int = 0
    transition_start_ts: float = -10.0
    last_interaction_ts: float = 0.0
    detail_level: int = 2
    animation_time: float = 0.0
    frozen: bool = False
    water_slosh_x: float = 0.0
    water_slosh_y: float = 0.0
    water_fill: float = 0.58
    water_wave_phase: float = 0.0
    prev_center_x: float = 0.0
    prev_center_y: float = 0.0
    prev_depth_offset: float = 0.0
    prev_rot_xy: float = 0.0
    prev_rot_xz: float = 0.0
    prev_rot_yz: float = 0.0
    prev_rot_xw: float = 0.0
    prev_rot_yw: float = 0.0
    prev_rot_zw: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_depth: float = 0.0
    vel_rot_xy: float = 0.0
    vel_rot_xz: float = 0.0
    vel_rot_yz: float = 0.0
    vel_rot_xw: float = 0.0
    vel_rot_yw: float = 0.0
    vel_rot_zw: float = 0.0
    wobble_xy: float = 0.0
    wobble_xz: float = 0.0
    wobble_yz: float = 0.0


class MultiLandmarkSmoother:
    def __init__(self, alpha: float = SMOOTH_ALPHA) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 0.99))
        self._state: Dict[str, np.ndarray] = {}

    def smooth(self, key: str, points: Sequence[Tuple[float, float, float]]) -> np.ndarray:
        current = np.asarray(points, dtype=np.float32)
        prev = self._state.get(key)
        if prev is None or prev.shape != current.shape:
            self._state[key] = current.copy()
        else:
            self._state[key] = (1.0 - self.alpha) * prev + self.alpha * current
        return self._state[key].copy()

    def prune(self, active_keys: Sequence[str]) -> None:
        keep = set(active_keys)
        for key in list(self._state.keys()):
            if key not in keep:
                del self._state[key]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def axis_speed_from_offset(
    offset: float,
    deadzone: float,
    max_effect: float,
    max_speed: float,
    response_exp: float = DUAL_AXIS_RESPONSE_EXP,
) -> float:
    mag = abs(offset)
    if mag <= deadzone:
        return 0.0
    norm = (mag - deadzone) / max(1e-6, max_effect - deadzone)
    norm = clamp(norm, 0.0, 1.0)
    norm = norm ** response_exp
    speed = max_speed * norm
    return speed if offset > 0.0 else -speed


def smoothstep01(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ui_visibility(now: float, last_interaction_ts: float) -> float:
    idle = max(0.0, now - last_interaction_ts)
    if idle <= UI_IDLE_FADE_DELAY_SEC:
        return 1.0
    fade = 1.0 - (idle - UI_IDLE_FADE_DELAY_SEC) / UI_IDLE_FADE_DURATION_SEC
    return smoothstep01(fade)


def transition_progress(now: float, transition_start_ts: float) -> float:
    if transition_start_ts <= 0.0:
        return 1.0
    return clamp((now - transition_start_ts) / TRANSITION_DURATION_SEC, 0.0, 1.0)


def detail_level_from_fps(fps: float) -> int:
    if fps < DETAIL_LOW_FPS:
        return 0
    if fps < DETAIL_MED_FPS:
        return 1
    return 2


def detail_value(state: HologramState, low: int, medium: int, high: int) -> int:
    return (low, medium, high)[int(clamp(state.detail_level, 0, 2))]


def neutral_rotations_for_mode(mode_name: str) -> Tuple[float, float, float, float, float, float]:
    if mode_name == "Tesseract":
        return 0.25, 0.85, -0.35, 0.55, -0.28, 0.40
    if mode_name == "Orbital Atom":
        return 0.18, 0.78, -0.16, 0.0, 0.0, 0.0
    if mode_name == "Black Hole Lens":
        return 0.14, 1.04, -0.04, 0.0, 0.0, 0.0
    if mode_name == "Solar Orrery":
        return 0.16, 0.90, 0.04, 0.0, 0.0, 0.0
    if mode_name == "Magnetic Cage":
        return 0.22, 0.76, 0.18, 0.0, 0.0, 0.0
    return 0.08, 0.70, -0.06, 0.0, 0.0, 0.0


def update_secondary_dynamics(state: HologramState, dt: float, width: int, height: int) -> None:
    if state.prev_center_x == 0.0 and state.prev_center_y == 0.0:
        state.prev_center_x = state.center_x
        state.prev_center_y = state.center_y
        state.prev_depth_offset = state.depth_offset
        state.prev_rot_xy = state.rot_xy
        state.prev_rot_xz = state.rot_xz
        state.prev_rot_yz = state.rot_yz
        state.prev_rot_xw = state.rot_xw
        state.prev_rot_yw = state.rot_yw
        state.prev_rot_zw = state.rot_zw
        return

    dt = max(1e-4, dt)
    dt60 = max(1.0, dt * 60.0)
    dx = state.center_x - state.prev_center_x
    dy = state.center_y - state.prev_center_y
    dz = state.depth_offset - state.prev_depth_offset
    droll = wrap_angle(state.rot_xy - state.prev_rot_xy)
    dpitch = wrap_angle(state.rot_xz - state.prev_rot_xz)
    dyaw = wrap_angle(state.rot_yz - state.prev_rot_yz)
    dxw = wrap_angle(state.rot_xw - state.prev_rot_xw)
    dyw = wrap_angle(state.rot_yw - state.prev_rot_yw)
    dzw = wrap_angle(state.rot_zw - state.prev_rot_zw)

    inst_vel_x = dx / dt
    inst_vel_y = dy / dt
    inst_vel_depth = dz / dt
    inst_rot_xy = droll / dt
    inst_rot_xz = dpitch / dt
    inst_rot_yz = dyaw / dt
    inst_rot_xw = dxw / dt
    inst_rot_yw = dyw / dt
    inst_rot_zw = dzw / dt

    impulse_x = (-dx * 0.010) + (dyaw * 0.65) + (droll * 0.30)
    impulse_y = (-dy * 0.012) + (-dz * 0.00045) + (dpitch * 0.78)

    state.water_slosh_x = state.water_slosh_x * (0.90 ** max(1.0, dt * 60.0)) + impulse_x
    state.water_slosh_y = state.water_slosh_y * (0.88 ** max(1.0, dt * 60.0)) + impulse_y
    state.water_slosh_x = clamp(state.water_slosh_x, -1.3, 1.3)
    state.water_slosh_y = clamp(state.water_slosh_y, -1.3, 1.3)
    state.water_wave_phase += dt * (1.2 + 1.6 * (abs(state.water_slosh_x) + abs(state.water_slosh_y)))

    if state.dragging:
        state.vel_x = (1.0 - POSITION_VEL_BLEND) * state.vel_x + POSITION_VEL_BLEND * inst_vel_x
        state.vel_y = (1.0 - POSITION_VEL_BLEND) * state.vel_y + POSITION_VEL_BLEND * inst_vel_y
        state.vel_depth = (1.0 - DEPTH_VEL_BLEND) * state.vel_depth + DEPTH_VEL_BLEND * inst_vel_depth
    if state.dragging or state.dual_active:
        state.vel_rot_xy = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_xy + ROTATION_VEL_BLEND * inst_rot_xy
        state.vel_rot_xz = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_xz + ROTATION_VEL_BLEND * inst_rot_xz
        state.vel_rot_yz = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_yz + ROTATION_VEL_BLEND * inst_rot_yz
        state.vel_rot_xw = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_xw + ROTATION_VEL_BLEND * inst_rot_xw
        state.vel_rot_yw = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_yw + ROTATION_VEL_BLEND * inst_rot_yw
        state.vel_rot_zw = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_zw + ROTATION_VEL_BLEND * inst_rot_zw

    wobble_drag = WOBBLE_DAMPING ** dt60
    state.wobble_xy = clamp(
        state.wobble_xy * wobble_drag + (-dx * WOBBLE_TRANSLATION_GAIN) + droll * WOBBLE_ROTATION_GAIN,
        -WOBBLE_MAX_ANGLE,
        WOBBLE_MAX_ANGLE,
    )
    state.wobble_xz = clamp(
        state.wobble_xz * wobble_drag + (-dy * WOBBLE_TRANSLATION_GAIN) + dpitch * WOBBLE_ROTATION_GAIN,
        -WOBBLE_MAX_ANGLE,
        WOBBLE_MAX_ANGLE,
    )
    state.wobble_yz = clamp(
        state.wobble_yz * wobble_drag + (-dz * 0.00022) + dyaw * (WOBBLE_ROTATION_GAIN * 0.9),
        -WOBBLE_MAX_ANGLE,
        WOBBLE_MAX_ANGLE,
    )

    if not state.dragging:
        trans_drag = TRANSLATION_DAMPING ** dt60
        depth_drag = DEPTH_DAMPING ** dt60
        state.vel_x *= trans_drag
        state.vel_y *= trans_drag
        state.vel_depth *= depth_drag
        state.center_x += state.vel_x * dt
        state.center_y += state.vel_y * dt
        state.depth_offset += state.vel_depth * dt

    rot_drag = ROTATION_DAMPING ** dt60
    if not state.dual_active:
        state.vel_rot_xy *= rot_drag
        state.vel_rot_xz *= rot_drag
        state.vel_rot_yz *= rot_drag
        state.vel_rot_xw *= rot_drag
        state.vel_rot_yw *= rot_drag
        state.vel_rot_zw *= rot_drag

        mode_name = HOLOGRAM_NAMES[state.mode_index]
        neutral_xy, neutral_xz, neutral_yz, neutral_xw, neutral_yw, neutral_zw = neutral_rotations_for_mode(mode_name)
        state.vel_rot_xy += (neutral_xy - state.rot_xy) * ROTATION_SPRING_GAIN * dt
        state.vel_rot_xz += (neutral_xz - state.rot_xz) * ROTATION_SPRING_GAIN * dt
        state.vel_rot_yz += (neutral_yz - state.rot_yz) * ROTATION_SPRING_GAIN * dt
        state.vel_rot_xw += (neutral_xw - state.rot_xw) * (ROTATION_SPRING_GAIN * 0.7) * dt
        state.vel_rot_yw += (neutral_yw - state.rot_yw) * (ROTATION_SPRING_GAIN * 0.7) * dt
        state.vel_rot_zw += (neutral_zw - state.rot_zw) * (ROTATION_SPRING_GAIN * 0.45) * dt

        state.rot_xy += state.vel_rot_xy * dt
        state.rot_xz += state.vel_rot_xz * dt
        state.rot_yz += state.vel_rot_yz * dt
        state.rot_xw += state.vel_rot_xw * dt
        state.rot_yw += state.vel_rot_yw * dt
        state.rot_zw += state.vel_rot_zw * dt

    min_x = 130.0
    max_x = width - 130.0
    min_y = 185.0
    max_y = height - 130.0
    max_depth = max(width, height) * MAX_DEPTH_OFFSET_FACTOR
    if state.center_x < min_x:
        state.vel_x += (min_x - state.center_x) * BOUNDARY_SPRING_GAIN * dt
        state.center_x = min_x
    elif state.center_x > max_x:
        state.vel_x -= (state.center_x - max_x) * BOUNDARY_SPRING_GAIN * dt
        state.center_x = max_x
    if state.center_y < min_y:
        state.vel_y += (min_y - state.center_y) * BOUNDARY_SPRING_GAIN * dt
        state.center_y = min_y
    elif state.center_y > max_y:
        state.vel_y -= (state.center_y - max_y) * BOUNDARY_SPRING_GAIN * dt
        state.center_y = max_y
    state.depth_offset = clamp(state.depth_offset, -max_depth, max_depth)

    state.prev_center_x = state.center_x
    state.prev_center_y = state.center_y
    state.prev_depth_offset = state.depth_offset
    state.prev_rot_xy = state.rot_xy
    state.prev_rot_xz = state.rot_xz
    state.prev_rot_yz = state.rot_yz
    state.prev_rot_xw = state.rot_xw
    state.prev_rot_yw = state.rot_yw
    state.prev_rot_zw = state.rot_zw


def normalize_label(label: str, mirror: bool) -> str:
    if not mirror:
        return label
    lowered = label.lower()
    if lowered == "left":
        return "Right"
    if lowered == "right":
        return "Left"
    return label


def extract_solution_hands(result: object, mirror: bool) -> List[HandObservation]:
    hands: List[HandObservation] = []
    if not getattr(result, "multi_hand_landmarks", None):
        return hands
    for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
        label = "Unknown"
        score = 1.0
        if result.multi_handedness and idx < len(result.multi_handedness):
            classification = result.multi_handedness[idx].classification[0]
            label = classification.label or "Unknown"
            score = float(classification.score)
        hands.append(
            HandObservation(
                label=normalize_label(label, mirror),
                score=score,
                landmarks=[(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
            )
        )
    return hands


def extract_task_hands(result: object, mirror: bool) -> List[HandObservation]:
    hands: List[HandObservation] = []
    if not getattr(result, "hand_landmarks", None):
        return hands
    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        label = "Unknown"
        score = 1.0
        if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
            classification = result.handedness[idx][0]
            label = (
                getattr(classification, "category_name", None)
                or getattr(classification, "display_name", None)
                or "Unknown"
            )
            score = float(getattr(classification, "score", 1.0))
        hands.append(
            HandObservation(
                label=normalize_label(label, mirror),
                score=score,
                landmarks=[(lm.x, lm.y, lm.z) for lm in hand_landmarks],
            )
        )
    return hands


def dist_xy(points: Sequence[Tuple[float, float, float]] | np.ndarray, idx_a: int, idx_b: int) -> float:
    ax, ay, _ = points[idx_a]
    bx, by, _ = points[idx_b]
    return math.hypot(ax - bx, ay - by)


def palm_scale(points: Sequence[Tuple[float, float, float]] | np.ndarray) -> float:
    scale = (
        dist_xy(points, WRIST, INDEX_MCP)
        + dist_xy(points, WRIST, PINKY_MCP)
        + dist_xy(points, INDEX_MCP, PINKY_MCP)
    ) / 3.0
    return max(scale, 1e-4)


def finger_extended(
    points: Sequence[Tuple[float, float, float]] | np.ndarray,
    tip_idx: int,
    pip_idx: int,
    palm: float,
) -> bool:
    tip_to_wrist = dist_xy(points, tip_idx, WRIST)
    pip_to_wrist = dist_xy(points, pip_idx, WRIST)
    return tip_to_wrist > (pip_to_wrist + 0.08 * palm)


def classify_gesture(points: Sequence[Tuple[float, float, float]] | np.ndarray) -> str:
    palm = palm_scale(points)
    thumb_index = dist_xy(points, THUMB_TIP, INDEX_TIP)
    thumb_middle = dist_xy(points, THUMB_TIP, MIDDLE_TIP)

    index_ext = finger_extended(points, INDEX_TIP, INDEX_PIP, palm)
    middle_ext = finger_extended(points, MIDDLE_TIP, MIDDLE_PIP, palm)
    ring_ext = finger_extended(points, RING_TIP, RING_PIP, palm)
    pinky_ext = finger_extended(points, PINKY_TIP, PINKY_PIP, palm)

    if thumb_index < 0.36 * palm and thumb_middle < 0.36 * palm:
        return "two_finger_pinch"
    if thumb_index < 0.32 * palm:
        return "pinch"
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "open_hand"

    finger_to_wrist = [
        dist_xy(points, INDEX_TIP, WRIST),
        dist_xy(points, MIDDLE_TIP, WRIST),
        dist_xy(points, RING_TIP, WRIST),
        dist_xy(points, PINKY_TIP, WRIST),
    ]
    if (
        not index_ext
        and not middle_ext
        and not ring_ext
        and not pinky_ext
        and max(finger_to_wrist) < 1.85 * palm
    ):
        return "fist"
    return "none"


def pinch_ratio(points: Sequence[Tuple[float, float, float]] | np.ndarray) -> float:
    return dist_xy(points, THUMB_TIP, INDEX_TIP) / palm_scale(points)


def pinch_pose(points: Sequence[Tuple[float, float, float]] | np.ndarray) -> Tuple[bool, float]:
    palm = palm_scale(points)
    return dist_xy(points, THUMB_TIP, INDEX_TIP) < PINCH_POSE_THRESHOLD * palm, palm


def wrist_x(hand: HandObservation) -> float:
    return float(hand.landmarks[WRIST][0])


def pinch_center_norm(points: np.ndarray) -> Tuple[float, float]:
    thumb = points[THUMB_TIP, :2]
    index_tip = points[INDEX_TIP, :2]
    center = 0.5 * (thumb + index_tip)
    return float(center[0]), float(center[1])


def pinch_center_px(points: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    cx, cy = pinch_center_norm(points)
    return cx * width, cy * height


def prepare_candidates(
    hands: List[HandObservation],
    smoother: MultiLandmarkSmoother,
) -> List[dict]:
    ordered = sorted(hands, key=wrist_x)[:MAX_TRACKED_HANDS]
    active_keys: List[str] = []
    candidates: List[dict] = []
    for idx, hand in enumerate(ordered):
        key = f"slot{idx}"
        active_keys.append(key)
        smoothed = smoother.smooth(key, hand.landmarks)
        pinching, palm = pinch_pose(smoothed)
        cx, cy = pinch_center_norm(smoothed)
        pair_depth = 0.5 * (float(smoothed[THUMB_TIP, 2]) + float(smoothed[INDEX_TIP, 2]))
        candidates.append(
            {
                "slot": key,
                "label": hand.label,
                "score": hand.score,
                "points": smoothed,
                "pinch_ratio": pinch_ratio(smoothed),
                "pinch_pose": pinching,
                "palm_scale": palm,
                "pair_center": (cx, cy),
                "pair_depth": pair_depth,
                "gesture": classify_gesture(smoothed),
            }
        )
    smoother.prune(active_keys)
    return candidates


def two_hand_pinch_pair(candidates: Sequence[dict]) -> Tuple[dict, dict, float] | None:
    eligible = [c for c in candidates if c["pinch_pose"]]
    if len(eligible) < 2:
        return None
    eligible.sort(key=lambda item: item["score"], reverse=True)
    a, b = eligible[0], eligible[1]
    ax, ay = a["pair_center"]
    bx, by = b["pair_center"]
    center_dist = math.hypot(ax - bx, ay - by)
    scale = max(1e-4, 0.5 * (a["palm_scale"] + b["palm_scale"]))
    metric = center_dist / scale
    return a, b, metric


def reset_state(width: int, height: int) -> HologramState:
    return HologramState(
        center_x=width * 0.50,
        center_y=height * 0.55,
        size=min(width, height) * 0.24,
    )


def update_controls(
    state: HologramState,
    candidates: Sequence[dict],
    width: int,
    height: int,
    dt: float,
) -> str:
    dual_pair = two_hand_pinch_pair(candidates)
    pinching = [c for c in candidates if c["pinch_pose"]]

    if dual_pair is not None:
        a, b, metric = dual_pair
        ax, ay = a["pair_center"]
        bx, by = b["pair_center"]
        center_x = 0.5 * (ax + bx)
        center_y = 0.5 * (ay + by)
        avg_depth = 0.5 * (a["pair_depth"] + b["pair_depth"])
        line_angle = math.atan2(by - ay, bx - ax)
        edit_mode = (
            a["gesture"] == "two_finger_pinch"
            and b["gesture"] == "two_finger_pinch"
        )
        state.dragging = False

        if not state.dual_active:
            state.dual_active = True
            state.dual_anchor_metric = max(1e-4, metric)
            state.dual_anchor_size = state.size
            state.dual_anchor_angle = line_angle
            state.dual_anchor_center_x = center_x
            state.dual_anchor_center_y = center_y
            state.dual_anchor_depth = avg_depth
            state.dual_anchor_depth_offset = state.depth_offset
            state.dual_anchor_rot_xy = state.rot_xy
            state.dual_anchor_rot_xw = state.rot_xw
            state.dual_anchor_rot_yw = state.rot_yw
            state.dual_anchor_rot_zw = state.rot_zw
            state.dual_anchor_param_a = state.mode_param_a
            state.dual_anchor_param_b = state.mode_param_b
        else:
            prev_depth_offset = state.depth_offset
            scale = metric / max(1e-4, state.dual_anchor_metric)
            state.size = clamp(state.dual_anchor_size * scale, MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
            depth_delta = avg_depth - state.dual_anchor_depth
            max_depth = max(width, height) * MAX_DEPTH_OFFSET_FACTOR
            state.depth_offset = clamp(
                state.dual_anchor_depth_offset - depth_delta * DEPTH_CONTROL_GAIN,
                -max_depth,
                max_depth,
            )
            x_offset = center_x - state.dual_anchor_center_x
            y_offset = center_y - state.dual_anchor_center_y
            angle_offset = wrap_angle(line_angle - state.dual_anchor_angle)

            if edit_mode:
                state.mode_param_a = clamp(
                    state.dual_anchor_param_a + x_offset * PARAM_EDIT_GAIN_X,
                    0.0,
                    1.0,
                )
                state.mode_param_b = clamp(
                    state.dual_anchor_param_b - y_offset * PARAM_EDIT_GAIN_Y,
                    0.0,
                    1.0,
                )
                state.vel_depth = (1.0 - DEPTH_VEL_BLEND) * state.vel_depth + DEPTH_VEL_BLEND * (
                    (state.depth_offset - prev_depth_offset) / max(dt, 1e-4)
                )
                return "edit"

            yaw_speed = axis_speed_from_offset(
                x_offset,
                DUAL_CENTER_DEADZONE_NORM,
                DUAL_CENTER_MAX_EFFECT_NORM,
                DUAL_YAW_MAX_SPEED,
            )
            pitch_speed = axis_speed_from_offset(
                y_offset,
                DUAL_CENTER_DEADZONE_NORM,
                DUAL_CENTER_MAX_EFFECT_NORM,
                DUAL_PITCH_MAX_SPEED,
            )
            roll_speed = axis_speed_from_offset(
                angle_offset,
                DUAL_ANGLE_DEADZONE_RAD,
                DUAL_ANGLE_MAX_EFFECT_RAD,
                DUAL_ROLL_MAX_SPEED,
            )

            state.rot_yw += yaw_speed * dt
            state.rot_yz += 0.58 * yaw_speed * dt
            state.rot_xw += pitch_speed * dt
            state.rot_xz += 0.54 * pitch_speed * dt
            state.rot_xy += roll_speed * dt
            state.rot_zw = state.dual_anchor_rot_zw + DUAL_ZW_GAIN * (
                metric - state.dual_anchor_metric
            )
            state.vel_depth = (1.0 - DEPTH_VEL_BLEND) * state.vel_depth + DEPTH_VEL_BLEND * (
                (state.depth_offset - prev_depth_offset) / max(dt, 1e-4)
            )
            state.vel_rot_yw = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_yw + ROTATION_VEL_BLEND * yaw_speed
            state.vel_rot_yz = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_yz + ROTATION_VEL_BLEND * (0.58 * yaw_speed)
            state.vel_rot_xw = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_xw + ROTATION_VEL_BLEND * pitch_speed
            state.vel_rot_xz = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_xz + ROTATION_VEL_BLEND * (0.54 * pitch_speed)
            state.vel_rot_xy = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_xy + ROTATION_VEL_BLEND * roll_speed
            state.vel_rot_zw = (1.0 - ROTATION_VEL_BLEND) * state.vel_rot_zw + ROTATION_VEL_BLEND * (
                DUAL_ZW_GAIN * (metric - state.dual_anchor_metric) / max(dt, 1e-4)
            )
        return "dual" if not edit_mode else "edit"

    state.dual_active = False

    if len(pinching) == 1:
        hand = pinching[0]
        pinch_x, pinch_y = pinch_center_px(hand["points"], width, height)
        catch_radius = max(DRAG_CATCH_MARGIN, state.size * 0.95)

        if not state.dragging:
            if math.hypot(state.center_x - pinch_x, state.center_y - pinch_y) <= catch_radius:
                state.dragging = True
                state.drag_offset_x = state.center_x - pinch_x
                state.drag_offset_y = state.center_y - pinch_y
                state.drag_anchor_depth = hand["pair_depth"]
                state.drag_anchor_depth_offset = state.depth_offset
        if state.dragging:
            prev_center_x = state.center_x
            prev_center_y = state.center_y
            prev_depth_offset = state.depth_offset
            state.center_x = clamp(pinch_x + state.drag_offset_x, 130.0, width - 130.0)
            state.center_y = clamp(pinch_y + state.drag_offset_y, 185.0, height - 130.0)
            depth_delta = hand["pair_depth"] - state.drag_anchor_depth
            max_depth = max(width, height) * MAX_DEPTH_OFFSET_FACTOR
            state.depth_offset = clamp(
                state.drag_anchor_depth_offset - depth_delta * DEPTH_CONTROL_GAIN,
                -max_depth,
                max_depth,
            )
            inv_dt = 1.0 / max(dt, 1e-4)
            state.vel_x = (1.0 - POSITION_VEL_BLEND) * state.vel_x + POSITION_VEL_BLEND * (
                (state.center_x - prev_center_x) * inv_dt
            )
            state.vel_y = (1.0 - POSITION_VEL_BLEND) * state.vel_y + POSITION_VEL_BLEND * (
                (state.center_y - prev_center_y) * inv_dt
            )
            state.vel_depth = (1.0 - DEPTH_VEL_BLEND) * state.vel_depth + DEPTH_VEL_BLEND * (
                (state.depth_offset - prev_depth_offset) * inv_dt
            )
            return "drag"
        return "single"

    state.dragging = False
    return "idle"


def update_mode_cycle(state: HologramState, candidates: Sequence[dict], now: float) -> bool:
    fists = [c for c in candidates if c["gesture"] == "fist"]
    pinching = any(c["pinch_pose"] for c in candidates)
    if len(fists) == 1 and not pinching:
        cooldown_ok = (now - state.last_mode_switch_ts) >= FIST_SWITCH_COOLDOWN_SEC
        if cooldown_ok and not state.fist_latched:
            state.previous_mode_index = state.mode_index
            direction = -1 if fists[0]["label"].lower() == "left" else 1
            state.mode_index = (state.mode_index + direction) % len(HOLOGRAM_NAMES)
            state.last_mode_switch_ts = now
            state.transition_start_ts = now
            state.fist_latched = True
            return True
    else:
        state.fist_latched = False
    return False


def rotate_plane(points: np.ndarray, axis_a: int, axis_b: int, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    out = points.copy()
    a = points[:, axis_a]
    b = points[:, axis_b]
    out[:, axis_a] = c * a - s * b
    out[:, axis_b] = s * a + c * b
    return out


def rotate_2d(x: float, y: float, angle: float) -> Tuple[float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def ellipse_point(
    cx: float,
    cy: float,
    axis_x: float,
    axis_y: float,
    angle: float,
    param: float,
) -> Tuple[float, float]:
    x = axis_x * math.cos(param)
    y = axis_y * math.sin(param)
    xr, yr = rotate_2d(x, y, angle)
    return cx + xr, cy + yr


def scale_bgr(color: Tuple[int, int, int], scale: float) -> Tuple[int, int, int]:
    return (
        int(clamp(color[0] * scale, 0.0, 255.0)),
        int(clamp(color[1] * scale, 0.0, 255.0)),
        int(clamp(color[2] * scale, 0.0, 255.0)),
    )


def project_points_3d(
    points: np.ndarray,
    state: HologramState,
    center_x: float | None = None,
    center_y: float | None = None,
    rot_xy: float = 0.0,
    rot_xz: float = 0.0,
    rot_yz: float = 0.0,
    depth: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = points.astype(np.float32).copy()
    pts = rotate_plane(pts, 0, 1, (state.rot_xy + state.wobble_xy) * 0.62 + rot_xy)
    pts = rotate_plane(pts, 0, 2, (state.rot_xz + state.wobble_xz) * 0.60 + rot_xz)
    pts = rotate_plane(pts, 1, 2, (state.rot_yz + state.wobble_yz) * 0.60 + rot_yz)
    pts[:, 2] += state.depth_offset
    px = state.center_x if center_x is None else center_x
    py = state.center_y if center_y is None else center_y
    proj_depth = max(state.size * 4.8, 760.0) if depth is None else depth
    z = np.maximum(1e-4, proj_depth - pts[:, 2])
    scale = proj_depth / z
    projected = np.column_stack([px + pts[:, 0] * scale, py + pts[:, 1] * scale])
    return projected.astype(np.int32), pts


def draw_depth_polyline(
    frame: np.ndarray,
    projected: np.ndarray,
    points3d: np.ndarray,
    color: Tuple[int, int, int],
    closed: bool = True,
) -> None:
    if len(projected) < 2:
        return
    z_min = float(np.min(points3d[:, 2]))
    z_max = float(np.max(points3d[:, 2]))
    z_span = max(1e-4, z_max - z_min)
    limit = len(projected) if closed else len(projected) - 1
    for i in range(limit):
        j = (i + 1) % len(projected)
        avg_z = 0.5 * (points3d[i, 2] + points3d[j, 2])
        near = (avg_z - z_min) / z_span
        line_color = scale_bgr(color, 0.55 + 0.55 * near)
        glow_color = scale_bgr(line_color, 0.42)
        thickness = max(1, int(1 + 2.2 * near))
        p0 = tuple(int(v) for v in projected[i])
        p1 = tuple(int(v) for v in projected[j])
        cv2.line(frame, p0, p1, glow_color, thickness + 4, cv2.LINE_AA)
        cv2.line(frame, p0, p1, line_color, thickness, cv2.LINE_AA)


def draw_depth_points(
    frame: np.ndarray,
    projected: np.ndarray,
    points3d: np.ndarray,
    color: Tuple[int, int, int],
    radius_scale: float = 1.0,
) -> None:
    z_min = float(np.min(points3d[:, 2]))
    z_max = float(np.max(points3d[:, 2]))
    z_span = max(1e-4, z_max - z_min)
    order = np.argsort(points3d[:, 2])
    for idx in order:
        near = (points3d[idx, 2] - z_min) / z_span
        radius = max(2, int(radius_scale * (2 + 3 * near)))
        glow = scale_bgr(color, 0.46 + 0.38 * near)
        px, py = int(projected[idx, 0]), int(projected[idx, 1])
        cv2.circle(frame, (px, py), radius + 3, glow, 1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 248, 230), -1, cv2.LINE_AA)


def make_ring_points(radius: float, samples: int, wobble: float = 0.0, phase: float = 0.0) -> np.ndarray:
    pts = []
    for i in range(samples):
        theta = (i / samples) * 2.0 * math.pi
        y = wobble * math.sin(theta * 2.0 + phase)
        pts.append((radius * math.cos(theta), y, radius * math.sin(theta)))
    return np.asarray(pts, dtype=np.float32)


def apply_cinematic_background(frame: np.ndarray, state: HologramState, t: float) -> None:
    return


def draw_hologram_aura(frame: np.ndarray, state: HologramState, t: float) -> None:
    overlay = frame.copy()
    cx = int(state.center_x)
    cy = int(state.center_y)
    pulse = 0.92 + 0.08 * math.sin(t * 2.6)
    for idx, color in enumerate(((118, 214, 255), (255, 192, 96), (96, 246, 214))):
        radius = int(state.size * pulse * (0.96 + 0.16 * idx))
        cv2.ellipse(
            overlay,
            (cx, cy),
            (radius, max(18, radius // (4 + idx))),
            math.degrees(state.rot_xy * 0.25 + idx * 0.4),
            0.0,
            360.0,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0.0, frame)


def draw_scanlines(image: np.ndarray) -> None:
    return


def project_tesseract(state: HologramState, t: float) -> Tuple[np.ndarray, np.ndarray]:
    points = HYPERCUBE_VERTICES.copy() * (state.size * 0.46)
    plane_mix = 2.0 * state.mode_param_a - 1.0
    fold = 0.45 + 0.85 * state.mode_param_b
    x_plane_gain = 1.0 + 0.95 * (1.0 - max(0.0, plane_mix))
    y_plane_gain = 1.0 + 0.95 * max(0.0, plane_mix)
    fold_drive = 0.18 + fold * 0.50
    points = rotate_plane(points, 0, 1, state.rot_xy + state.wobble_xy)
    points = rotate_plane(points, 0, 2, state.rot_xz + state.wobble_xz)
    points = rotate_plane(points, 1, 2, state.rot_yz + state.wobble_yz)
    points = rotate_plane(points, 0, 3, state.rot_xw * x_plane_gain + fold_drive * math.sin(t * 1.15 + plane_mix * 1.1))
    points = rotate_plane(points, 1, 3, state.rot_yw * y_plane_gain + fold_drive * math.cos(t * 1.05 - plane_mix * 1.1))
    points = rotate_plane(points, 2, 3, state.rot_zw + AUTO_ROT_ZW_SPEED * t + fold * (0.28 + 0.22 * abs(plane_mix)))
    points[:, 3] *= fold
    points[:, 2] += state.depth_offset

    w_depth = max(state.size * 3.3, 420.0)
    w_factor = w_depth / np.maximum(1e-4, w_depth - points[:, 3])
    xyz = points[:, :3] * w_factor[:, None]

    z_depth = max(state.size * 4.0, 640.0)
    z_factor = z_depth / np.maximum(1e-4, z_depth - xyz[:, 2])
    projected = np.column_stack(
        [
            state.center_x + xyz[:, 0] * z_factor,
            state.center_y + xyz[:, 1] * z_factor,
        ]
    )
    return projected.astype(np.int32), points


def draw_tesseract(frame: np.ndarray, state: HologramState, t: float) -> None:
    pts, raw4d = project_tesseract(state, t)
    overlay = frame.copy()
    glow = frame.copy()
    bob_y = int(18.0 * math.sin(1.9 * t + state.bob_phase))

    inner = [idx for idx, vertex in enumerate(HYPERCUBE_VERTICES) if vertex[3] < 0.0]
    outer = [idx for idx, vertex in enumerate(HYPERCUBE_VERTICES) if vertex[3] > 0.0]
    inner_poly = np.array([pts[i] for i in inner[:4]], dtype=np.int32)
    outer_poly = np.array([pts[i] for i in outer[:4]], dtype=np.int32)
    if len(inner_poly) == 4:
        cv2.fillConvexPoly(overlay, inner_poly, (52, 88, 186), cv2.LINE_AA)
    if len(outer_poly) == 4:
        cv2.fillConvexPoly(overlay, outer_poly, (92, 156, 255), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.14, frame, 0.86, 0.0, frame)

    w_values = raw4d[:, 3]
    w_min = float(np.min(w_values))
    w_max = float(np.max(w_values))
    w_span = max(1e-4, w_max - w_min)

    for a, b in HYPERCUBE_EDGES:
        mix = ((w_values[a] + w_values[b]) * 0.5 - w_min) / w_span
        edge_color = (
            int(255 - 120 * mix),
            int(160 + 80 * mix),
            int(64 + 170 * mix),
        )
        thickness = 2 if mix > 0.5 else 1
        p0 = tuple(int(v) for v in pts[a])
        p1 = tuple(int(v) for v in pts[b])
        cv2.line(glow, p0, p1, edge_color, 6, cv2.LINE_AA)
        cv2.line(frame, p0, p1, edge_color, thickness, cv2.LINE_AA)

    cv2.addWeighted(glow, 0.13, frame, 0.87, 0.0, frame)

    for idx in np.argsort(raw4d[:, 2]):
        glow_mix = (w_values[idx] - w_min) / w_span
        px, py = int(pts[idx, 0]), int(pts[idx, 1])
        color = (
            int(255 - 85 * glow_mix),
            int(208 + 40 * glow_mix),
            int(188 + 50 * glow_mix),
        )
        radius = int(max(3.0, state.size * 0.024))
        cv2.circle(frame, (px, py), radius + 3, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius, (255, 248, 226), -1, cv2.LINE_AA)

    ring_radius = int(state.size * 0.98)
    cv2.ellipse(
        frame,
        (int(state.center_x), int(state.center_y + bob_y)),
        (ring_radius, max(16, ring_radius // 4)),
        0.0,
        0.0,
        360.0,
        (255, 238, 148),
        1,
        cv2.LINE_AA,
    )


def draw_water_cylinder(frame: np.ndarray, state: HologramState, t: float) -> None:
    size = state.size
    radius = size * (0.20 + 0.14 * state.mode_param_b)
    height = size * (0.95 + 0.45 * (1.0 - state.mode_param_b))
    fill = 0.24 + 0.58 * state.mode_param_a
    samples = detail_value(state, 24, 36, 52)

    thetas = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False, dtype=np.float32)
    top_ring = np.column_stack(
        [radius * np.cos(thetas), np.full_like(thetas, -height * 0.5), radius * np.sin(thetas)]
    ).astype(np.float32)
    bottom_ring = np.column_stack(
        [radius * np.cos(thetas), np.full_like(thetas, height * 0.5), radius * np.sin(thetas)]
    ).astype(np.float32)

    water_level = height * (0.5 - fill)
    slosh_amp = size * 0.10
    wave = size * 0.018 * np.sin(3.0 * thetas + state.water_wave_phase)
    water_ring = np.column_stack(
        [
            radius * np.cos(thetas) * 0.96,
            np.full_like(thetas, water_level)
            + slosh_amp * state.water_slosh_x * np.cos(thetas)
            + slosh_amp * state.water_slosh_y * np.sin(thetas)
            + wave,
            radius * np.sin(thetas) * 0.96,
        ]
    ).astype(np.float32)

    top_proj, top_world = project_points_3d(top_ring, state, rot_xy=0.06, rot_xz=0.10, rot_yz=0.0)
    bottom_proj, bottom_world = project_points_3d(bottom_ring, state, rot_xy=0.06, rot_xz=0.10, rot_yz=0.0)
    water_proj, water_world = project_points_3d(water_ring, state, rot_xy=0.06, rot_xz=0.10, rot_yz=0.0)

    shell = frame.copy()
    if len(water_proj) >= 3:
        cv2.fillConvexPoly(shell, water_proj.astype(np.int32), (110, 190, 255), cv2.LINE_AA)
        cv2.addWeighted(shell, 0.16, frame, 0.84, 0.0, frame)

    draw_depth_polyline(frame, top_proj, top_world, (188, 236, 255))
    draw_depth_polyline(frame, bottom_proj, bottom_world, (124, 184, 238))
    draw_depth_polyline(frame, water_proj, water_world, (255, 238, 176))

    for idx in range(0, samples, max(1, samples // 12)):
        seg_proj = np.asarray([top_proj[idx], bottom_proj[idx]], dtype=np.int32)
        seg_world = np.asarray([top_world[idx], bottom_world[idx]], dtype=np.float32)
        draw_depth_polyline(frame, seg_proj, seg_world, (142, 220, 255), closed=False)

    for idx in range(0, samples, max(1, samples // 10)):
        seg_proj = np.asarray([water_proj[idx], bottom_proj[idx]], dtype=np.int32)
        seg_world = np.asarray([water_world[idx], bottom_world[idx]], dtype=np.float32)
        draw_depth_polyline(frame, seg_proj, seg_world, (86, 164, 240), closed=False)

    bubble_count = detail_value(state, 8, 14, 22)
    for bi in range(bubble_count):
        theta = ((bi / max(1, bubble_count)) * 2.0 * math.pi) + t * 0.35 + bi * 0.11
        bubble_h = water_level + (0.10 + 0.88 * ((bi * 0.37 + t * 0.28) % 1.0)) * (height * 0.5 - water_level)
        bubble = np.asarray(
            [[
                radius * 0.58 * math.cos(theta),
                bubble_h + size * 0.01 * math.sin(t * 2.0 + bi),
                radius * 0.58 * math.sin(theta),
            ]],
            dtype=np.float32,
        )
        bubble_proj, bubble_world = project_points_3d(bubble, state, rot_xy=0.06, rot_xz=0.10, rot_yz=0.0)
        draw_depth_points(frame, bubble_proj, bubble_world, (214, 244, 255), radius_scale=0.55)

    cap_center = np.asarray([[0.0, -height * 0.56, 0.0], [0.0, height * 0.56, 0.0]], dtype=np.float32)
    cap_proj, cap_world = project_points_3d(cap_center, state, rot_xy=0.06, rot_xz=0.10, rot_yz=0.0)
    for idx, color in enumerate(((255, 220, 138), (118, 220, 255))):
        cv2.circle(frame, (int(cap_proj[idx, 0]), int(cap_proj[idx, 1])), int(size * 0.045), color, 1, cv2.LINE_AA)


def draw_orbital_atom(frame: np.ndarray, state: HologramState, t: float) -> None:
    cx = state.center_x
    cy = state.center_y + 12.0 * math.sin(1.7 * t)
    size = state.size
    orbit_count = 3 + int(round(state.mode_param_a * 4.0))
    cloud_density = detail_value(state, 10, 18, 26) + int(state.mode_param_b * 8.0)
    core_pts = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (size * 0.10, 0.0, 0.0),
            (-size * 0.08, size * 0.03, size * 0.04),
            (size * 0.02, -size * 0.07, -size * 0.05),
            (-size * 0.04, size * 0.08, -size * 0.03),
        ],
        dtype=np.float32,
    )
    core_proj, core_world = project_points_3d(core_pts, state, center_x=cx, center_y=cy)
    for idx, color in enumerate(((255, 170, 74), (255, 216, 118), (255, 246, 220), (116, 250, 232), (146, 220, 255))):
        px, py = int(core_proj[idx, 0]), int(core_proj[idx, 1])
        cv2.circle(frame, (px, py), int(size * (0.10 if idx == 0 else 0.032)), scale_bgr(color, 0.35), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), int(size * (0.055 if idx == 0 else 0.017)), color, -1, cv2.LINE_AA)

    palette = (
        (255, 220, 120),
        (118, 255, 220),
        (255, 156, 86),
        (146, 220, 255),
        (255, 184, 128),
        (112, 214, 255),
        (196, 255, 146),
    )
    orbit_specs = []
    for idx in range(orbit_count):
        orbit_specs.append(
            (
                size * (0.58 - 0.05 * idx),
                size * (0.018 + 0.014 * ((idx % 3) / 2.0) + 0.010 * state.mode_param_b),
                0.12 + 0.48 * math.sin(t * (0.18 + 0.08 * idx) + idx * 0.9),
                0.90 + 0.28 * idx + 0.22 * state.mode_param_a,
                ((-1.0) ** idx) * (0.42 + 0.22 * idx),
                1.4 * idx,
                palette[idx % len(palette)],
            )
        )
    for radius, wobble, rxy, rxz, ryz, phase, color in orbit_specs:
        ring = make_ring_points(radius, detail_value(state, 56, 78, 96), wobble=wobble, phase=phase + t)
        proj, world = project_points_3d(
            ring,
            state,
            center_x=cx,
            center_y=cy,
            rot_xy=rxy + t * 0.26,
            rot_xz=rxz,
            rot_yz=ryz,
        )
        draw_depth_polyline(frame, proj, world, color)
        electron_idx = int(((t * (0.9 + abs(rxy))) + phase) * 18.0) % len(proj)
        ep = (int(proj[electron_idx, 0]), int(proj[electron_idx, 1]))
        cv2.circle(frame, ep, 10, scale_bgr(color, 0.44), 1, cv2.LINE_AA)
        cv2.circle(frame, ep, 5, (255, 248, 232), -1, cv2.LINE_AA)
        trail_count = 5
        for trail in range(1, trail_count + 1):
            ti = (electron_idx - trail * 2) % len(proj)
            alpha = 1.0 - trail / (trail_count + 1)
            tp = (int(proj[ti, 0]), int(proj[ti, 1]))
            cv2.circle(frame, tp, max(1, int(4 * alpha)), scale_bgr(color, 0.25 + 0.25 * alpha), -1, cv2.LINE_AA)

    shell = make_ring_points(size * 0.18, 48, wobble=size * 0.02, phase=t * 1.7)
    shell_proj, shell_world = project_points_3d(shell, state, center_x=cx, center_y=cy, rot_xy=t * 0.8, rot_xz=1.0, rot_yz=0.6)
    draw_depth_polyline(frame, shell_proj, shell_world, (255, 182, 86))
    draw_depth_points(frame, shell_proj[::6], shell_world[::6], (255, 210, 126), radius_scale=1.2)
    for ci, cloud_r in enumerate((0.22, 0.28, 0.34)):
        cloud = make_ring_points(size * cloud_r, cloud_density, wobble=size * 0.035, phase=t * (1.2 + cloud_r + 0.2 * ci))
        cloud_proj, cloud_world = project_points_3d(
            cloud,
            state,
            center_x=cx,
            center_y=cy,
            rot_xy=t * (0.5 + cloud_r),
            rot_xz=0.8 + cloud_r,
            rot_yz=0.5 + 0.18 * ci,
        )
        draw_depth_points(frame, cloud_proj[::5], cloud_world[::5], (255, 184, 98), radius_scale=0.9)


def draw_black_hole_lens(frame: np.ndarray, state: HologramState, t: float) -> None:
    cx = state.center_x
    cy = state.center_y + 10.0 * math.sin(1.5 * t)
    size = state.size
    radius = int(size * 0.24)
    disk_thickness = 0.12 + 0.22 * state.mode_param_a
    jet_intensity = 0.18 + 0.82 * state.mode_param_b

    for layer, color in enumerate(((255, 164, 66), (255, 206, 108), (118, 214, 255))):
        ring = make_ring_points(size * (0.42 + 0.08 * layer), detail_value(state, 72, 96, 120), wobble=size * 0.015, phase=t * (1.0 + 0.2 * layer))
        ring[:, 1] *= disk_thickness + 0.04 * layer
        proj, world = project_points_3d(
            ring,
            state,
            center_x=cx,
            center_y=cy,
            rot_xy=t * (0.22 + 0.05 * layer),
            rot_xz=1.18 + 0.1 * math.sin(t * 0.4),
            rot_yz=-0.26 + 0.18 * layer,
        )
        draw_depth_polyline(frame, proj, world, color)

    for i in range(detail_value(state, 10, 14, 18)):
        band = np.linspace(-1.0, 1.0, detail_value(state, 18, 24, 30), dtype=np.float32)
        pts = np.column_stack(
            [
                band * size * (0.88 + 0.05 * math.sin(i + t)),
                np.full_like(band, (i - 7.5) * size * 0.045),
                np.sin(band * math.pi + i * 0.28 + t * 1.4) * size * 0.10,
            ]
        )
        proj, world = project_points_3d(pts, state, center_x=cx, center_y=cy, rot_xy=0.25, rot_xz=1.06, rot_yz=0.0)
        draw_depth_polyline(frame, proj, world, (76, 118, 190), closed=False)

    photon = make_ring_points(size * 0.66, 84, wobble=size * 0.04, phase=t * 0.8)
    photon[:, 1] += size * 0.05 * np.sin(np.linspace(0.0, 2.0 * math.pi, len(photon), endpoint=False) * 3.0 + t)
    proj, world = project_points_3d(photon, state, center_x=cx, center_y=cy, rot_xy=t * 0.34, rot_xz=1.35, rot_yz=0.52)
    draw_depth_polyline(frame, proj, world, (255, 236, 148))

    jets = []
    jet_steps = detail_value(state, 10, 14, 20)
    for direction in (-1.0, 1.0):
        for j in range(jet_steps):
            depth_t = j / max(1, jet_steps - 1)
            jets.append(
                (
                    size * 0.05 * jet_intensity * math.sin(t * 3.0 + j * 0.4),
                    direction * depth_t * size * (0.55 + 0.70 * jet_intensity),
                    size * 0.18 * jet_intensity * math.cos(t * 1.6 + j * 0.5),
                )
            )
    jet_pts = np.asarray(jets, dtype=np.float32)
    jet_proj, jet_world = project_points_3d(jet_pts, state, center_x=cx, center_y=cy, rot_xy=0.0, rot_xz=1.52, rot_yz=0.0)
    draw_depth_points(frame, jet_proj, jet_world, (118, 214, 255), radius_scale=0.8)

    debris = []
    for i in range(detail_value(state, 16, 24, 32)):
        theta = t * (0.8 + 0.04 * i) + i * 0.52
        debris.append(
            (
                size * (0.70 + 0.09 * math.sin(i)) * math.cos(theta),
                size * (0.04 + 0.05 * disk_thickness) * math.sin(theta * 1.8),
                size * (0.70 + 0.09 * math.sin(i)) * math.sin(theta),
            )
        )
    debris_pts = np.asarray(debris, dtype=np.float32)
    debris_proj, debris_world = project_points_3d(debris_pts, state, center_x=cx, center_y=cy, rot_xy=0.38, rot_xz=1.28, rot_yz=0.48)
    draw_depth_points(frame, debris_proj, debris_world, (255, 224, 146), radius_scale=0.9)

    cv2.circle(frame, (int(cx), int(cy)), radius + 18, (34, 28, 18), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(cx), int(cy)), radius + 6, (255, 190, 82), 1, cv2.LINE_AA)
    cv2.circle(frame, (int(cx), int(cy)), radius, (0, 0, 0), -1, cv2.LINE_AA)


def draw_dna_helix(frame: np.ndarray, state: HologramState, t: float) -> None:
    cx = state.center_x
    cy = state.center_y
    size = state.size
    amp = size * (0.18 + 0.22 * state.mode_param_b)
    height = size * 1.70
    twist_turns = 3.8 + 4.6 * state.mode_param_a
    samples = detail_value(state, 36, 48, 64)
    strand_a = []
    strand_b = []
    for i in range(samples):
        v = i / (samples - 1)
        local_y = (v - 0.5) * height
        phase = v * twist_turns * math.pi + t * (1.5 + 1.2 * state.mode_param_a) + state.rot_xw
        strand_a.append((amp * math.cos(phase), local_y, amp * math.sin(phase)))
        strand_b.append((amp * math.cos(phase + math.pi), local_y, amp * math.sin(phase + math.pi)))
    a_pts = np.asarray(strand_a, dtype=np.float32)
    b_pts = np.asarray(strand_b, dtype=np.float32)
    a_proj, a_world = project_points_3d(a_pts, state, center_x=cx, center_y=cy, rot_xy=0.24, rot_xz=0.20, rot_yz=0.10)
    b_proj, b_world = project_points_3d(b_pts, state, center_x=cx, center_y=cy, rot_xy=0.24, rot_xz=0.20, rot_yz=0.10)
    draw_depth_polyline(frame, a_proj, a_world, (255, 206, 118), closed=False)
    draw_depth_polyline(frame, b_proj, b_world, (118, 248, 255), closed=False)
    draw_depth_points(frame, a_proj[::3], a_world[::3], (255, 216, 132), radius_scale=1.0)
    draw_depth_points(frame, b_proj[::3], b_world[::3], (126, 248, 255), radius_scale=1.0)

    for i in range(0, samples, 3):
        rung = np.asarray([a_pts[i], b_pts[i]], dtype=np.float32)
        rung_proj, rung_world = project_points_3d(rung, state, center_x=cx, center_y=cy, rot_xy=0.24, rot_xz=0.20, rot_yz=0.10)
        rung_color = (255, 164 + (i % 4) * 18, 98 + (i % 3) * 48)
        draw_depth_polyline(frame, rung_proj, rung_world, rung_color, closed=False)

    packet_idx = int(t * 8.0) % samples
    for pts_proj, pts_world, color in ((a_proj, a_world, (255, 208, 132)), (b_proj, b_world, (132, 246, 255))):
        px, py = int(pts_proj[packet_idx, 0]), int(pts_proj[packet_idx, 1])
        cv2.circle(frame, (px, py), 9, scale_bgr(color, 0.42), 1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, (255, 248, 232), -1, cv2.LINE_AA)

    cage_angles = (0.0, math.pi * 0.5, math.pi, math.pi * 1.5)
    for cage_angle in cage_angles:
        cage = []
        cage_steps = detail_value(state, 10, 16, 22)
        for i in range(cage_steps):
            depth_t = i / max(1, cage_steps - 1)
            y = (depth_t - 0.5) * height
            cage.append(
                (
                    size * (0.28 + 0.18 * state.mode_param_b) * math.cos(cage_angle),
                    y,
                    size * (0.28 + 0.18 * state.mode_param_b) * math.sin(cage_angle),
                )
            )
        cage_pts = np.asarray(cage, dtype=np.float32)
        cage_proj, cage_world = project_points_3d(cage_pts, state, center_x=cx, center_y=cy, rot_xy=0.22, rot_xz=0.18, rot_yz=0.08)
        draw_depth_polyline(frame, cage_proj, cage_world, (86, 136, 220), closed=False)


def draw_crystal_lattice(frame: np.ndarray, state: HologramState, t: float) -> None:
    size = state.size
    density = 3 + int(round(state.mode_param_a * 2.0))
    spacing = size * 0.24
    jitter = size * 0.020 * (0.4 + state.mode_param_b)
    coords = np.linspace(-(density - 1) * 0.5, (density - 1) * 0.5, density, dtype=np.float32)
    points = []
    index_map: Dict[Tuple[int, int, int], int] = {}
    idx = 0
    for xi, x in enumerate(coords):
        for yi, y in enumerate(coords):
            for zi, z in enumerate(coords):
                px = x * spacing + jitter * math.sin(t * 1.1 + xi + zi * 0.4)
                py = y * spacing + jitter * math.cos(t * 0.9 + yi + xi * 0.3)
                pz = z * spacing + jitter * math.sin(t * 1.3 + zi + yi * 0.2)
                points.append((px, py, pz))
                index_map[(xi, yi, zi)] = idx
                idx += 1
    pts = np.asarray(points, dtype=np.float32)
    proj, world = project_points_3d(pts, state)
    draw_depth_points(frame, proj, world, (158, 244, 255), radius_scale=0.9)
    for xi in range(density):
        for yi in range(density):
            for zi in range(density):
                i0 = index_map[(xi, yi, zi)]
                for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
                    xn = xi + dx
                    yn = yi + dy
                    zn = zi + dz
                    if xn < density and yn < density and zn < density:
                        i1 = index_map[(xn, yn, zn)]
                        seg_proj = np.asarray([proj[i0], proj[i1]], dtype=np.int32)
                        seg_world = np.asarray([world[i0], world[i1]], dtype=np.float32)
                        draw_depth_polyline(frame, seg_proj, seg_world, (255, 216, 146), closed=False)


def draw_solar_orrery(frame: np.ndarray, state: HologramState, t: float) -> None:
    cx = state.center_x
    cy = state.center_y
    size = state.size
    speed_mult = 0.45 + 2.15 * state.mode_param_a
    planet_count = 4 + int(round(state.mode_param_b * 2.0))
    orbit_spacing = size * 0.11
    trail_steps = 5 + int(round(state.mode_param_b * 12.0))
    sun = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    sun_proj, _ = project_points_3d(sun, state, center_x=cx, center_y=cy)
    cv2.circle(frame, (int(sun_proj[0, 0]), int(sun_proj[0, 1])), int(size * 0.10), (255, 214, 112), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(sun_proj[0, 0]), int(sun_proj[0, 1])), int(size * 0.17), (255, 170, 74), 1, cv2.LINE_AA)
    colors = ((118, 220, 255), (255, 186, 96), (126, 255, 198), (255, 236, 150), (196, 220, 255), (255, 146, 126), (128, 255, 146))
    for i in range(planet_count):
        orbit_r = size * 0.20 + orbit_spacing * i
        ring = make_ring_points(orbit_r, detail_value(state, 42, 64, 84), wobble=size * 0.012, phase=t * 0.5 + i)
        proj, world = project_points_3d(
            ring,
            state,
            center_x=cx,
            center_y=cy,
            rot_xy=0.16 + 0.10 * i,
            rot_xz=0.80 + 0.16 * math.sin(i + t * 0.2),
            rot_yz=0.10 * i,
        )
        color = colors[i % len(colors)]
        draw_depth_polyline(frame, proj, world, color)
        base_speed = speed_mult * (0.55 + 0.17 * i)
        angle = t * base_speed + i * 0.78
        flatten = 0.74 + 0.16 * math.sin(i * 0.8)
        planet_pt = np.asarray(
            [[
                orbit_r * math.cos(angle),
                size * 0.02 * math.sin(angle * 1.8 + i),
                orbit_r * flatten * math.sin(angle),
            ]],
            dtype=np.float32,
        )
        planet_proj, planet_world = project_points_3d(
            planet_pt,
            state,
            center_x=cx,
            center_y=cy,
            rot_xy=0.16 + 0.10 * i,
            rot_xz=0.80 + 0.16 * math.sin(i + t * 0.2),
            rot_yz=0.10 * i,
        )
        trail = []
        for step in range(trail_steps):
            trail_angle = angle - step * 0.12 * base_speed
            trail.append(
                (
                    orbit_r * math.cos(trail_angle),
                    size * 0.02 * math.sin(trail_angle * 1.8 + i),
                    orbit_r * flatten * math.sin(trail_angle),
                )
            )
        trail_pts = np.asarray(trail, dtype=np.float32)
        trail_proj, trail_world = project_points_3d(
            trail_pts,
            state,
            center_x=cx,
            center_y=cy,
            rot_xy=0.16 + 0.10 * i,
            rot_xz=0.80 + 0.16 * math.sin(i + t * 0.2),
            rot_yz=0.10 * i,
        )
        draw_depth_polyline(frame, trail_proj, trail_world, scale_bgr(color, 0.72), closed=False)
        draw_depth_points(frame, trail_proj[::2], trail_world[::2], scale_bgr(color, 0.82), radius_scale=0.55)
        p = (int(planet_proj[0, 0]), int(planet_proj[0, 1]))
        cv2.circle(frame, p, 4 + (i % 3), (255, 248, 232), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 8 + (i % 2), scale_bgr(color, 0.42), 1, cv2.LINE_AA)


def draw_magnetic_cage(frame: np.ndarray, state: HologramState, t: float) -> None:
    cx = state.center_x
    cy = state.center_y
    size = state.size
    loop_count = 4 + int(round(state.mode_param_a * 4.0))
    core_glow = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    core_proj, _ = project_points_3d(core_glow, state, center_x=cx, center_y=cy)
    core_strength = 0.5 + 0.5 * state.mode_param_b
    cv2.circle(frame, (int(core_proj[0, 0]), int(core_proj[0, 1])), int(size * (0.08 + 0.04 * core_strength)), (255, 182, 92), -1, cv2.LINE_AA)
    for i in range(loop_count):
        ring = make_ring_points(size * (0.24 + 0.09 * i), detail_value(state, 46, 64, 84), wobble=size * 0.03, phase=t * 1.1 + i)
        proj, world = project_points_3d(
            ring,
            state,
            center_x=cx,
            center_y=cy,
            rot_xy=t * 0.3 + i * 0.4,
            rot_xz=0.7 + i * 0.3,
            rot_yz=0.26 * i,
        )
        draw_depth_polyline(frame, proj, world, (118, 230, 255))
    for i in range(14):
        spark = np.asarray(
            [[
                size * (0.16 + 0.50 * state.mode_param_b) * math.cos(t * 2.4 + i),
                size * 0.26 * math.sin(t * 1.9 + i * 0.6),
                size * (0.16 + 0.50 * state.mode_param_b) * math.sin(t * 2.4 + i),
            ]],
            dtype=np.float32,
        )
        proj, _ = project_points_3d(spark, state, center_x=cx, center_y=cy)
        cv2.circle(frame, (int(proj[0, 0]), int(proj[0, 1])), 3, (255, 244, 220), -1, cv2.LINE_AA)


def draw_city_core(frame: np.ndarray, state: HologramState, t: float) -> None:
    size = state.size
    road_y = size * 0.34
    block_half = size * (0.52 + 0.12 * state.mode_param_a)
    street_half = size * 0.12
    tower_scale = 0.75 + 0.75 * state.mode_param_b

    roads = [
        np.asarray(
            [
                (-block_half, road_y, -street_half),
                (block_half, road_y, -street_half),
                (block_half, road_y, street_half),
                (-block_half, road_y, street_half),
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                (-street_half, road_y, -block_half),
                (street_half, road_y, -block_half),
                (street_half, road_y, block_half),
                (-street_half, road_y, block_half),
            ],
            dtype=np.float32,
        ),
    ]
    for road in roads:
        proj, world = project_points_3d(road, state, rot_xy=0.14, rot_xz=0.92, rot_yz=0.0)
        draw_depth_polyline(frame, proj, world, (90, 162, 216), closed=True)

    lane_marks = []
    dash_count = 6
    for idx in range(dash_count):
        dash_t = -0.78 + idx * 0.31 + 0.03 * math.sin(t * 1.5 + idx)
        lane_marks.append((dash_t * block_half, road_y - size * 0.001, 0.0))
        lane_marks.append((0.0, road_y - size * 0.001, dash_t * block_half))
    lane_pts = np.asarray(lane_marks, dtype=np.float32)
    lane_proj, lane_world = project_points_3d(lane_pts, state, rot_xy=0.14, rot_xz=0.92, rot_yz=0.0)
    draw_depth_points(frame, lane_proj, lane_world, (255, 220, 150), radius_scale=0.55)

    def draw_box(center_x: float, center_z: float, footprint_x: float, footprint_z: float, height: float, color: Tuple[int, int, int]) -> None:
        box = np.asarray(
            [
                (center_x - footprint_x, road_y, center_z - footprint_z),
                (center_x + footprint_x, road_y, center_z - footprint_z),
                (center_x + footprint_x, road_y, center_z + footprint_z),
                (center_x - footprint_x, road_y, center_z + footprint_z),
                (center_x - footprint_x, road_y - height, center_z - footprint_z),
                (center_x + footprint_x, road_y - height, center_z - footprint_z),
                (center_x + footprint_x, road_y - height, center_z + footprint_z),
                (center_x - footprint_x, road_y - height, center_z + footprint_z),
            ],
            dtype=np.float32,
        )
        proj, world = project_points_3d(box, state, rot_xy=0.14, rot_xz=0.92, rot_yz=0.0)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            seg_proj = np.asarray([proj[a], proj[b]], dtype=np.int32)
            seg_world = np.asarray([world[a], world[b]], dtype=np.float32)
            draw_depth_polyline(frame, seg_proj, seg_world, color, closed=False)

        windows = []
        rows = max(2, int(height / max(1.0, size * 0.10)))
        cols = max(2, int((footprint_x * 2.0) / max(1.0, size * 0.08)))
        for row in range(rows):
            for col in range(cols):
                windows.append(
                    (
                        center_x - footprint_x * 0.68 + col * (footprint_x * 1.36 / max(1, cols - 1)),
                        road_y - height * (0.18 + 0.62 * row / max(1, rows - 1)),
                        center_z + footprint_z + 0.5,
                    )
                )
        if windows:
            win_pts = np.asarray(windows, dtype=np.float32)
            win_proj, win_world = project_points_3d(win_pts, state, rot_xy=0.14, rot_xz=0.92, rot_yz=0.0)
            draw_depth_points(frame, win_proj, win_world, (255, 228, 164), radius_scale=0.40)

    def draw_house(center_x: float, center_z: float, footprint_x: float, footprint_z: float, height: float) -> None:
        wall_h = height * 0.62
        draw_box(center_x, center_z, footprint_x, footprint_z, wall_h, (122, 236, 255))
        roof = np.asarray(
            [
                (center_x, road_y - height, center_z - footprint_z),
                (center_x, road_y - height, center_z + footprint_z),
                (center_x - footprint_x, road_y - wall_h, center_z - footprint_z),
                (center_x + footprint_x, road_y - wall_h, center_z - footprint_z),
                (center_x + footprint_x, road_y - wall_h, center_z + footprint_z),
                (center_x - footprint_x, road_y - wall_h, center_z + footprint_z),
            ],
            dtype=np.float32,
        )
        roof_proj, roof_world = project_points_3d(roof, state, rot_xy=0.14, rot_xz=0.92, rot_yz=0.0)
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 3), (4, 5)):
            seg_proj = np.asarray([roof_proj[a], roof_proj[b]], dtype=np.int32)
            seg_world = np.asarray([roof_world[a], roof_world[b]], dtype=np.float32)
            draw_depth_polyline(frame, seg_proj, seg_world, (255, 212, 132), closed=False)

    buildings = [
        (-size * 0.28, -size * 0.28, size * 0.10, size * 0.09, size * 0.44 * tower_scale, (118, 232, 255)),
        (size * 0.28, -size * 0.26, size * 0.09, size * 0.08, size * 0.34 * tower_scale, (118, 232, 255)),
        (-size * 0.26, size * 0.28, size * 0.08, size * 0.07, size * 0.26 * tower_scale, (122, 236, 255)),
    ]
    for center_x, center_z, foot_x, foot_z, height, color in buildings:
        draw_box(center_x, center_z, foot_x, foot_z, height, color)

    houses = [
        (size * 0.26, size * 0.24, size * 0.11, size * 0.08, size * 0.24),
        (size * 0.08, size * 0.36, size * 0.09, size * 0.07, size * 0.20),
    ]
    for center_x, center_z, foot_x, foot_z, height in houses:
        draw_house(center_x, center_z, foot_x, foot_z, height)

    lamp_posts = []
    for sign_x in (-1.0, 1.0):
        lamp_posts.extend(
            [
                (sign_x * size * 0.16, road_y, -size * 0.16),
                (sign_x * size * 0.16, road_y - size * 0.16, -size * 0.16),
                (sign_x * size * 0.16, road_y, size * 0.16),
                (sign_x * size * 0.16, road_y - size * 0.16, size * 0.16),
            ]
        )
    lamp_pts = np.asarray(lamp_posts, dtype=np.float32)
    lamp_proj, lamp_world = project_points_3d(lamp_pts, state, rot_xy=0.14, rot_xz=0.92, rot_yz=0.0)
    for i in range(0, len(lamp_proj), 2):
        seg_proj = np.asarray([lamp_proj[i], lamp_proj[i + 1]], dtype=np.int32)
        seg_world = np.asarray([lamp_world[i], lamp_world[i + 1]], dtype=np.float32)
        draw_depth_polyline(frame, seg_proj, seg_world, (112, 206, 255), closed=False)
        px, py = int(lamp_proj[i + 1, 0]), int(lamp_proj[i + 1, 1])
        cv2.circle(frame, (px, py), 3, (255, 232, 176), -1, cv2.LINE_AA)


def draw_hologram_named(frame: np.ndarray, state: HologramState, t: float, mode: str) -> None:
    if mode == "Tesseract":
        draw_tesseract(frame, state, t)
    elif mode == "Orbital Atom":
        draw_orbital_atom(frame, state, t)
    elif mode == "Black Hole Lens":
        draw_black_hole_lens(frame, state, t)
    elif mode == "Solar Orrery":
        draw_solar_orrery(frame, state, t)
    elif mode == "Magnetic Cage":
        draw_magnetic_cage(frame, state, t)


def draw_hologram_mode(frame: np.ndarray, state: HologramState, t: float, mode_index: int) -> None:
    draw_hologram_named(frame, state, t, HOLOGRAM_NAMES[mode_index])


def draw_mode_transition_particles(frame: np.ndarray, state: HologramState, t: float, progress: float, mode: str) -> None:
    fade = 1.0 - smoothstep01(progress)
    if fade <= 0.0:
        return
    cx = state.center_x
    cy = state.center_y
    if mode == "Orbital Atom":
        for i in range(18):
            angle = i * 0.42 + t * 0.9
            radius = state.size * (0.14 + progress * (0.9 + 0.08 * i))
            px = int(cx + math.cos(angle) * radius)
            py = int(cy + math.sin(angle) * radius * 0.55)
            cv2.circle(frame, (px, py), 3, scale_bgr((255, 220, 128), fade), -1, cv2.LINE_AA)
    elif mode == "Black Hole Lens":
        for i in range(4):
            radius = int(state.size * (0.60 - 0.12 * i) * fade)
            if radius > 6:
                cv2.ellipse(
                    frame,
                    (int(cx), int(cy)),
                    (radius, max(6, radius // 3)),
                    math.degrees(state.rot_xy * 0.5 + i * 0.2),
                    0.0,
                    360.0,
                    scale_bgr((255, 188, 92), 0.6 * fade),
                    1,
                    cv2.LINE_AA,
                )
        cv2.circle(frame, (int(cx), int(cy)), int(state.size * 0.10 * fade), (0, 0, 0), -1, cv2.LINE_AA)
    elif mode == "Tesseract":
        temp = HologramState(**{**state.__dict__})
        temp.size = state.size * (1.0 - 0.36 * progress)
        temp.rot_xw += progress * 2.2
        temp.rot_yw -= progress * 1.6
        overlay = np.zeros_like(frame)
        draw_tesseract(overlay, temp, t)
        cv2.addWeighted(overlay, 0.24 * fade, frame, 1.0, 0.0, frame)
    elif mode == "Solar Orrery":
        for i in range(5):
            ring_r = state.size * (0.14 + 0.15 * i) * (1.0 - 0.55 * progress)
            if ring_r <= 4.0:
                continue
            cv2.ellipse(
                frame,
                (int(cx), int(cy)),
                (int(ring_r), max(4, int(ring_r * 0.42))),
                math.degrees(state.rot_xy * 0.4 + i * 0.1),
                0.0,
                360.0,
                scale_bgr((180, 228, 255), 0.20 + 0.45 * fade),
                1,
                cv2.LINE_AA,
            )
    elif mode == "Magnetic Cage":
        for i in range(8):
            theta = i * (math.pi * 0.25) + t
            pts = []
            for step in range(10):
                mix = step / 9.0
                pts.append(
                    (
                        cx + math.cos(theta) * state.size * (0.18 + 0.44 * mix * fade),
                        cy + (mix - 0.5) * state.size * (0.86 - 0.40 * progress),
                    )
                )
            for step in range(len(pts) - 1):
                cv2.line(
                    frame,
                    (int(pts[step][0]), int(pts[step][1])),
                    (int(pts[step + 1][0]), int(pts[step + 1][1])),
                    scale_bgr((126, 238, 255), 0.40 * fade),
                    1,
                    cv2.LINE_AA,
                )
    else:
        for i in range(20):
            angle = i * 0.31 + t
            radius = state.size * progress * (0.24 + 0.03 * i)
            px = int(cx + math.cos(angle) * radius)
            py = int(cy + math.sin(angle) * radius * 0.6)
            cv2.circle(frame, (px, py), 2, scale_bgr((180, 220, 255), 0.4 * fade), -1, cv2.LINE_AA)


def draw_mode_arrival_effect(frame: np.ndarray, state: HologramState, t: float, progress: float, mode: str) -> None:
    appear = smoothstep01(progress)
    if appear <= 0.0:
        return
    cx = state.center_x
    cy = state.center_y

    overlay = np.zeros_like(frame)
    temp = HologramState(**{**state.__dict__})
    temp.size = max(18.0, state.size * (0.18 + 0.82 * appear))
    temp.depth_offset = state.depth_offset - state.size * (1.0 - appear) * 0.25
    if mode == "Tesseract":
        temp.rot_xw += (1.0 - appear) * 2.8
        temp.rot_yw -= (1.0 - appear) * 2.2
    elif mode == "Orbital Atom":
        temp.mode_param_b *= appear
    elif mode == "Solar Orrery":
        temp.mode_param_b = max(temp.mode_param_b, 1.0 - appear)
    elif mode == "Magnetic Cage":
        temp.mode_param_a *= appear
    draw_hologram_named(overlay, temp, t, mode)
    cv2.addWeighted(overlay, 0.18 + 0.34 * appear, frame, 1.0, 0.0, frame)

    scan_count = 6
    for idx in range(scan_count):
        band_y = int(cy + (idx / max(1, scan_count - 1) - 0.5) * state.size * 1.3)
        band_half_w = int(state.size * (0.18 + 0.58 * appear))
        cv2.line(
            frame,
            (int(cx - band_half_w), band_y),
            (int(cx + band_half_w), band_y),
            scale_bgr((120, 222, 255), 0.10 + 0.14 * (1.0 - appear)),
            1,
            cv2.LINE_AA,
        )

    for idx in range(18):
        angle = idx * 0.35 + t * 0.8
        radius = state.size * (0.15 + (1.0 - appear) * 0.85)
        px = int(cx + math.cos(angle) * radius * (1.0 - appear))
        py = int(cy + math.sin(angle) * radius * 0.62 * (1.0 - appear))
        cv2.circle(frame, (px, py), 2, scale_bgr((210, 236, 255), 0.22 + 0.34 * appear), -1, cv2.LINE_AA)


def draw_transition_overlay(frame: np.ndarray, state: HologramState, anim_t: float, now: float) -> None:
    progress = transition_progress(now, state.transition_start_ts)
    if progress >= 1.0:
        return
    fade_prev = 1.0 - smoothstep01(progress)
    prev_layer = np.zeros_like(frame)
    draw_hologram_mode(prev_layer, state, anim_t - 0.18, state.previous_mode_index)
    cv2.addWeighted(prev_layer, 0.38 * fade_prev, frame, 1.0, 0.0, frame)
    draw_mode_transition_particles(frame, state, anim_t, progress, HOLOGRAM_NAMES[state.previous_mode_index])
    draw_mode_arrival_effect(frame, state, anim_t, progress, HOLOGRAM_NAMES[state.mode_index])

    sweep = frame.copy()
    max_r = int(state.size * (1.4 + 0.5 * progress))
    for idx in range(3):
        r = int(max_r * (0.52 + 0.22 * idx))
        alpha = 0.28 * fade_prev * (1.0 - 0.22 * idx)
        cv2.ellipse(
            sweep,
            (int(state.center_x), int(state.center_y)),
            (r, max(10, r // 4)),
            math.degrees(state.rot_xy * 0.4),
            0.0,
            360.0,
            (255, 224, 138),
            1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(sweep, alpha, frame, 1.0 - alpha, 0.0, frame)
    beam_alpha = 0.18 * fade_prev
    if beam_alpha > 0.0:
        beam = frame.copy()
        beam_y = int(state.center_y - state.size * (0.8 - 0.6 * progress))
        cv2.rectangle(
            beam,
            (int(state.center_x - state.size * 0.85), beam_y),
            (int(state.center_x + state.size * 0.85), beam_y + max(6, int(state.size * 0.06))),
            (120, 220, 255),
            -1,
        )
        cv2.addWeighted(beam, beam_alpha, frame, 1.0 - beam_alpha, 0.0, frame)


def draw_hologram(frame: np.ndarray, state: HologramState, anim_t: float, now: float) -> None:
    draw_hologram_mode(frame, state, anim_t, state.mode_index)
    draw_hologram_aura(frame, state, anim_t)
    draw_transition_overlay(frame, state, anim_t, now)


def draw_hand(frame: np.ndarray, candidate: dict, visibility: float) -> None:
    h, w = frame.shape[:2]
    points = candidate["points"]
    pinch_x, pinch_y = pinch_center_px(points, w, h)
    if candidate["gesture"] != "fist" and not candidate["pinch_pose"]:
        return
    if visibility <= 0.02:
        return
    accent = (255, 236, 148) if candidate["pinch_pose"] else (182, 204, 228)
    if candidate["gesture"] == "fist":
        accent = (255, 176, 98)
    accent = scale_bgr(accent, 0.28 + 0.72 * visibility)
    cv2.circle(frame, (int(pinch_x), int(pinch_y)), 8, accent, 1, cv2.LINE_AA)
    cv2.circle(frame, (int(pinch_x), int(pinch_y)), 2, accent, -1, cv2.LINE_AA)


def draw_dual_link(frame: np.ndarray, pair: Tuple[dict, dict, float], visibility: float) -> None:
    if visibility <= 0.02:
        return
    h, w = frame.shape[:2]
    a, b, metric = pair
    ax, ay = a["pair_center"]
    bx, by = b["pair_center"]
    p0 = (int(ax * w), int(ay * h))
    p1 = (int(bx * w), int(by * h))
    line_color = (255, 236, 148) if metric > 0.0 else (182, 204, 228)
    cv2.line(frame, p0, p1, scale_bgr(line_color, (0.10 + 0.22 * visibility)), 6, cv2.LINE_AA)
    cv2.line(frame, p0, p1, scale_bgr(line_color, (0.26 + 0.62 * visibility)), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, state: HologramState, now: float) -> None:
    visibility = ui_visibility(now, state.last_interaction_ts)
    if visibility <= 0.03:
        return
    current_name = HOLOGRAM_NAMES[state.mode_index]
    panel = frame.copy()
    cv2.rectangle(panel, (HUD_MARGIN, HUD_MARGIN), (HUD_MARGIN + 226, HUD_MARGIN + 40), (8, 12, 20), -1)
    cv2.addWeighted(panel, 0.08 + 0.22 * visibility, frame, 0.92 - 0.22 * visibility, 0.0, frame)
    cv2.putText(
        frame,
        current_name,
        (HUD_MARGIN + 14, HUD_MARGIN + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        scale_bgr((238, 242, 250), 0.42 + 0.58 * visibility),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "L/R fist cycle",
        (HUD_MARGIN + 148, HUD_MARGIN + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        scale_bgr((162, 178, 206), 0.34 + 0.66 * visibility),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (camera index 0).")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    use_tasks = False
    tracker_solution = None
    tracker_tasks = None
    task_timestamp_ms = 0

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        tracker_solution = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_TRACKED_HANDS,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
    elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        if not MODEL_PATH.exists():
            print("Error: missing hand landmark model. Checked:")
            for candidate in MODEL_CANDIDATES:
                print(f"  - {candidate}")
            cap.release()
            return 1
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(MODEL_PATH),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_TRACKED_HANDS,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.55,
        )
        tracker_tasks = vision.HandLandmarker.create_from_options(options)
        use_tasks = True
    else:
        print("Unsupported mediapipe build: expected `solutions` or `tasks.vision` APIs.")
        cap.release()
        return 1

    smoother = MultiLandmarkSmoother(alpha=SMOOTH_ALPHA)
    mirror = True
    state: HologramState | None = None
    last_t = time.perf_counter()
    fps = 60.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read webcam frame.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAPTURE_W, CAPTURE_H), interpolation=cv2.INTER_LINEAR)

            now = time.perf_counter()
            dt = max(1e-4, min(0.05, now - last_t))
            last_t = now
            fps = 0.88 * fps + 0.12 * (1.0 / dt)

            if state is None:
                state = reset_state(frame.shape[1], frame.shape[0])
                state.last_interaction_ts = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if use_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = tracker_tasks.detect_for_video(mp_image, task_timestamp_ms)
                task_timestamp_ms += max(1, int(dt * 1000.0))
                hands = extract_task_hands(result, mirror)
            else:
                result = tracker_solution.process(rgb)
                hands = extract_solution_hands(result, mirror)

            candidates = prepare_candidates(hands, smoother)
            state.detail_level = detail_level_from_fps(fps)
            mode = update_controls(state, candidates, frame.shape[1], frame.shape[0], dt)
            changed = update_mode_cycle(state, candidates, now)
            interacting = changed or any(
                candidate["pinch_pose"] or candidate["gesture"] in {"fist", "open_hand", "two_finger_pinch"}
                for candidate in candidates
            )
            if interacting:
                state.last_interaction_ts = now

            if changed:
                state.rot_xy += 0.35
                state.rot_xw -= 0.18
                state.mode_param_a = 0.5
                state.mode_param_b = 0.5
                state.vel_rot_xy *= 0.45
                state.vel_rot_xz *= 0.45
                state.vel_rot_yz *= 0.45
                state.vel_rot_xw *= 0.45
                state.vel_rot_yw *= 0.45
                state.vel_rot_zw *= 0.45

            if mode != "dual":
                state.vel_rot_xz += dt * IDLE_ROT_XZ_SPEED * 1.4
                state.vel_rot_yz += dt * IDLE_ROT_YZ_SPEED * 1.4

            state.frozen = any(c["gesture"] == "open_hand" for c in candidates) and not any(
                c["pinch_pose"] for c in candidates
            )
            if not state.frozen:
                state.animation_time += dt
            anim_t = state.animation_time
            update_secondary_dynamics(state, dt, frame.shape[1], frame.shape[0])

            apply_cinematic_background(frame, state, anim_t)
            draw_hologram(frame, state, anim_t, now)
            dual_pair = two_hand_pinch_pair(candidates)
            if dual_pair is not None:
                draw_dual_link(frame, dual_pair, ui_visibility(now, state.last_interaction_ts))
            for candidate in candidates:
                draw_hand(frame, candidate, ui_visibility(now, state.last_interaction_ts))

            draw_scanlines(frame)
            draw_hud(frame, state, now)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("r"):
                saved_mode = state.mode_index
                state = reset_state(frame.shape[1], frame.shape[0])
                state.mode_index = saved_mode
                state.last_interaction_ts = now

    finally:
        if tracker_solution is not None:
            tracker_solution.close()
        if tracker_tasks is not None:
            tracker_tasks.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
