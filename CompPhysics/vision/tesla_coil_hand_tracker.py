#!/usr/bin/env python3
"""
Webcam hand tracker with a safe Tesla-coil plasma overlay.

This is visual-only. It does not drive real coils or any external hardware.

Controls:
- Move your hand toward the coil to trigger arcs.
- Get closer to intensify the discharge and plasma bloom.
- Press '=' or '-' for coarse voltage changes.
- Press '[' or ']' for fine voltage changes.
- Press '0' to reset voltage to the default level.
- Press 'r' to reset the effect state.
- Press 'q' to quit.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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


WINDOW_NAME = "Tesla Coil Hand Tracker"
MODULE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES: Tuple[Path, ...] = (
    MODULE_DIR / "models" / "hand_landmarker.task",
    MODULE_DIR.parent / "DefensiveSys" / "models" / "hand_landmarker.task",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])

CAPTURE_W = 1280
CAPTURE_H = 720
MAX_HANDS = 2
SMOOTH_ALPHA = 0.30

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

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

LANDMARK_NAMES: Tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


@dataclass
class HandObservation:
    label: str
    score: float
    landmarks: List[Tuple[float, float, float]]


@dataclass
class SimulationState:
    active: bool = False
    voltage: float = 0.35
    voltage_target: float = 0.35
    proximity: float = 0.0
    reaction_strength: float = 0.0
    contact_label: str = "none"
    transition_burst: float = 0.0
    fps_ema: float = 60.0


class LandmarkSmoother:
    def __init__(self, alpha: float = SMOOTH_ALPHA) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 0.99))
        self._pose: np.ndarray | None = None

    def smooth(self, landmarks: Sequence[Tuple[float, float, float]]) -> np.ndarray:
        current = np.asarray(landmarks, dtype=np.float32)
        if self._pose is None or self._pose.shape != current.shape:
            self._pose = current.copy()
        else:
            self._pose = (1.0 - self.alpha) * self._pose + self.alpha * current
        return self._pose.copy()

    def reset(self) -> None:
        self._pose = None


def _normalize_label(label: str, mirror: bool) -> str:
    if not mirror:
        return label
    ll = label.lower()
    if ll == "left":
        return "Right"
    if ll == "right":
        return "Left"
    return label


def _extract_solution_hands(result: object, mirror: bool) -> List[HandObservation]:
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
                label=_normalize_label(label, mirror),
                score=score,
                landmarks=[(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
            )
        )
    return hands


def _extract_task_hands(result: object, mirror: bool) -> List[HandObservation]:
    hands: List[HandObservation] = []
    if not getattr(result, "hand_landmarks", None):
        return hands
    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        label = "Unknown"
        score = 1.0
        if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
            category = result.handedness[idx][0]
            label = (
                getattr(category, "category_name", None)
                or getattr(category, "display_name", None)
                or "Unknown"
            )
            score = float(getattr(category, "score", 1.0))
        hands.append(
            HandObservation(
                label=_normalize_label(label, mirror),
                score=score,
                landmarks=[(lm.x, lm.y, lm.z) for lm in hand_landmarks],
            )
        )
    return hands


def _tracker_setup():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
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
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
            running_mode=vision.RunningMode.VIDEO,
        )
        tracker = vision.HandLandmarker.create_from_options(options)
        return "tasks", None, tracker

    raise RuntimeError("Unsupported mediapipe build: expected solutions or tasks.vision.")


def dist_xy(points: Sequence[Tuple[float, float, float]] | np.ndarray, a: int, b: int) -> float:
    ax, ay, _ = points[a]
    bx, by, _ = points[b]
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
    return dist_xy(points, tip_idx, WRIST) > (dist_xy(points, pip_idx, WRIST) + 0.08 * palm)


def openness_score(points: Sequence[Tuple[float, float, float]] | np.ndarray) -> float:
    palm = palm_scale(points)
    specs = (
        (INDEX_TIP, INDEX_PIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        (RING_TIP, RING_PIP, 13),
        (PINKY_TIP, PINKY_PIP, PINKY_MCP),
    )
    scores: List[float] = []
    for tip_i, pip_i, mcp_i in specs:
        tip_to_wrist = dist_xy(points, tip_i, WRIST)
        pip_to_wrist = dist_xy(points, pip_i, WRIST)
        mcp_to_wrist = dist_xy(points, mcp_i, WRIST)
        denom = max(0.22 * palm, (mcp_to_wrist - pip_to_wrist) + 0.95 * palm)
        score = np.clip((tip_to_wrist - pip_to_wrist) / denom, 0.0, 1.0)
        scores.append(float(score))
    return float(np.mean(scores))


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    n1 = float(np.linalg.norm(ab))
    n2 = float(np.linalg.norm(cb))
    if n1 <= 1e-5 or n2 <= 1e-5:
        return 180.0
    cosang = float(np.dot(ab, cb) / (n1 * n2))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def fist_score(points: np.ndarray) -> float:
    palm = palm_scale(points)
    fingers = (
        (5, 6, 8),
        (9, 10, 12),
        (13, 14, 16),
        (17, 18, 20),
    )
    scores: List[float] = []
    for mcp_i, pip_i, tip_i in fingers:
        mcp = points[mcp_i, :2]
        pip = points[pip_i, :2]
        tip = points[tip_i, :2]
        tip_norm = float(np.linalg.norm(tip - mcp) / palm)
        dist_closed = float(np.clip((1.02 - tip_norm) / 0.50, 0.0, 1.0))
        ang = joint_angle_deg(mcp, pip, tip)
        ang_closed = float(np.clip((155.0 - ang) / 70.0, 0.0, 1.0))
        scores.append(0.55 * dist_closed + 0.45 * ang_closed)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def classify_gesture(points: np.ndarray) -> str:
    open_score = openness_score(points)
    closed_score = fist_score(points)
    if open_score >= 0.68:
        return "open_hand"
    if closed_score >= 0.60:
        return "fist"
    return "none"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def pick_control_hand(hands: Sequence[HandObservation]) -> HandObservation | None:
    if not hands:
        return None
    return max(hands, key=lambda hand: hand.score)


def add_glow(
    canvas: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    strength: float,
) -> None:
    radius = max(4, int(radius))
    strength = clamp(strength, 0.0, 1.0)
    for scale, alpha in ((2.8, 0.05), (1.8, 0.08), (1.15, 0.14), (0.62, 0.28)):
        overlay = canvas.copy()
        cv2.circle(
            overlay,
            center,
            max(2, int(radius * scale)),
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha * strength, canvas, 1.0 - alpha * strength, 0.0, canvas)


def draw_coil(
    canvas: np.ndarray,
    emitter: Tuple[int, int],
    voltage: float,
    firing: bool,
) -> None:
    x, y = emitter
    tower_h = 235
    tower_w = 38
    tower_top = y + 28
    base_y = y + tower_h
    active_strength = 0.45 + 0.55 * voltage if firing else 0.10 + 0.16 * voltage

    cv2.ellipse(canvas, (x + 8, base_y + 10), (95, 22), 0.0, 0.0, 360.0, (8, 10, 16), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x - 84, base_y - 18), (x + 84, base_y + 30), (30, 34, 50), -1)
    cv2.rectangle(canvas, (x - 74, base_y - 8), (x + 74, base_y + 18), (40, 46, 68), -1)
    cv2.rectangle(canvas, (x - 22, tower_top), (x + 22, base_y), (56, 66, 98), -1)
    cv2.rectangle(canvas, (x - 28, tower_top - 36), (x + 28, tower_top + 14), (92, 104, 142), -1)
    cv2.rectangle(canvas, (x - 18, tower_top + 10), (x + 18, base_y - 10), (72, 84, 118), 1)
    cv2.line(canvas, (x - 54, base_y - 2), (x - 28, tower_top + 16), (88, 96, 122), 2, cv2.LINE_AA)
    cv2.line(canvas, (x + 54, base_y - 2), (x + 28, tower_top + 16), (88, 96, 122), 2, cv2.LINE_AA)

    for idx in range(12):
        cy = tower_top + 8 + idx * 16
        copper = (
            int(82 + idx * 6),
            int(122 + idx * 5),
            int(198 + idx * 2),
        )
        cv2.ellipse(canvas, (x, cy), (34, 7), 0.0, 0.0, 360.0, copper, 2, cv2.LINE_AA)

    cv2.ellipse(canvas, (x, y), (30, 14), 0.0, 0.0, 360.0, (184, 190, 206), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x, y), (22, 8), 0.0, 0.0, 360.0, (225, 228, 238), 2, cv2.LINE_AA)
    cv2.line(canvas, (x, y + 8), (x, tower_top - 6), (126, 136, 168), 3, cv2.LINE_AA)

    add_glow(canvas, emitter, int(36 + 22 * active_strength), (255, 205, 90), 0.48 + 0.30 * active_strength)
    add_glow(canvas, emitter, int(56 + 42 * active_strength), (255, 158, 52), 0.22 + 0.18 * active_strength)
    cv2.circle(canvas, emitter, 14, (248, 238, 224), -1, cv2.LINE_AA)
    cv2.circle(canvas, emitter, 6, (255, 255, 255), -1, cv2.LINE_AA)


def draw_discharge(
    canvas: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    rng: np.random.Generator,
    voltage: float,
) -> None:
    start_v = np.asarray(start, dtype=np.float32)
    end_v = np.asarray(end, dtype=np.float32)
    direction = end_v - start_v
    length = float(np.linalg.norm(direction))
    if length <= 1.0:
        return

    direction /= length
    perp = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    segments = 9 + int(length / 72.0)
    jagged = 3.0 + 11.0 * voltage
    points = [start_v]
    for idx in range(1, segments):
        t = idx / segments
        base = start_v + (end_v - start_v) * t
        envelope = math.sin(math.pi * t) ** 1.2
        offset = perp * rng.normal(0.0, jagged * envelope)
        base += offset.astype(np.float32)
        points.append(base)
    points.append(end_v)
    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))

    glow = canvas.copy()
    cv2.polylines(glow, [pts], False, (255, 140, 52), max(3, int(4 + 4 * voltage)), cv2.LINE_AA)
    cv2.addWeighted(glow, 0.18, canvas, 0.82, 0.0, canvas)
    cv2.polylines(canvas, [pts], False, (255, 215, 130), max(2, int(2 + 2 * voltage)), cv2.LINE_AA)
    cv2.polylines(canvas, [pts], False, (255, 250, 242), 1, cv2.LINE_AA)


def draw_plasma(
    canvas: np.ndarray,
    target: Tuple[int, int],
    voltage: float,
    burst: float,
    rng: np.random.Generator,
) -> None:
    plasma = clamp((voltage - 0.55) / 0.45, 0.0, 1.0)
    if plasma <= 0.0:
        return

    center = np.asarray(target, dtype=np.int32)
    core_radius = int(10 + 24 * plasma + 8 * burst)
    add_glow(canvas, tuple(center), int(core_radius * 2.0), (255, 125, 44), 0.18 + 0.22 * plasma)
    add_glow(canvas, tuple(center), int(core_radius * 1.1), (255, 190, 100), 0.24 + 0.20 * plasma)
    cv2.circle(canvas, tuple(center), max(3, int(core_radius * 0.18)), (255, 250, 240), -1, cv2.LINE_AA)

    for _ in range(5 + int(plasma * 8)):
        angle = rng.uniform(0.0, math.tau)
        radius = rng.uniform(core_radius * 0.10, core_radius * (0.55 + 0.35 * plasma))
        px = int(center[0] + math.cos(angle) * radius)
        py = int(center[1] + math.sin(angle) * radius * rng.uniform(0.55, 1.25))
        size = int(rng.uniform(1.0, 2.0 + 2.0 * plasma))
        color = (255, int(rng.uniform(180, 245)), int(rng.uniform(110, 180)))
        cv2.circle(canvas, (px, py), size, color, -1, cv2.LINE_AA)


def draw_corona_streamers(
    canvas: np.ndarray,
    emitter: Tuple[int, int],
    voltage: float,
    reaction_strength: float,
    rng: np.random.Generator,
) -> None:
    streamer_count = 3 + int(voltage * 9.0)
    reach = 16.0 + 26.0 * voltage + 22.0 * reaction_strength
    for _ in range(streamer_count):
        angle = rng.uniform(-1.15, 1.15)
        tip = (
            int(emitter[0] + math.cos(angle) * reach * rng.uniform(0.6, 1.0)),
            int(emitter[1] - abs(math.sin(angle)) * reach * rng.uniform(0.3, 1.1)),
        )
        draw_discharge(canvas, emitter, tip, rng, clamp(0.16 + 0.55 * voltage, 0.0, 1.0))


def draw_contact_detail(
    canvas: np.ndarray,
    target: Tuple[int, int],
    voltage: float,
    reaction_strength: float,
    rng: np.random.Generator,
) -> None:
    add_glow(canvas, target, int(10 + 26 * reaction_strength), (255, 214, 130), 0.15 + 0.28 * reaction_strength)
    add_glow(canvas, target, int(18 + 34 * reaction_strength), (255, 120, 44), 0.06 + 0.15 * reaction_strength)
    for _ in range(1 + int(4 * reaction_strength)):
        angle = rng.uniform(0.0, math.tau)
        radius = rng.uniform(10.0, 18.0 + 24.0 * reaction_strength)
        px = int(target[0] + math.cos(angle) * radius)
        py = int(target[1] + math.sin(angle) * radius)
        cv2.circle(
            canvas,
            (px, py),
            1 + int(rng.uniform(0.0, 2.0 + 2.0 * voltage)),
            (255, int(200 + 40 * reaction_strength), int(120 + 40 * reaction_strength)),
            -1,
            cv2.LINE_AA,
        )


def draw_voltage_meter(
    canvas: np.ndarray,
    voltage: float,
    reaction_strength: float,
) -> None:
    panel_x = canvas.shape[1] - 84
    panel_y = 72
    panel_h = 250
    panel_w = 34
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x - 22, panel_y - 18), (panel_x + panel_w + 22, panel_y + panel_h + 18), (16, 20, 28), -1)
    cv2.addWeighted(overlay, 0.48, canvas, 0.52, 0.0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (92, 110, 144), 2)

    fill_h = int((panel_h - 8) * clamp(voltage, 0.0, 1.0))
    if fill_h > 0:
        grad = np.linspace(0.0, 1.0, fill_h, dtype=np.float32)[:, None]
        region = np.zeros((fill_h, panel_w - 8, 3), dtype=np.uint8)
        region[..., 0] = np.clip(40 + 120 * grad, 0, 255).astype(np.uint8)
        region[..., 1] = np.clip(100 + 90 * grad, 0, 255).astype(np.uint8)
        region[..., 2] = np.clip(230 + 25 * grad, 0, 255).astype(np.uint8)
        y0 = panel_y + panel_h - 4 - fill_h
        x0 = panel_x + 4
        canvas[y0:y0 + fill_h, x0:x0 + panel_w - 8] = region

    reaction_y = panel_y + panel_h - int(panel_h * clamp(reaction_strength, 0.0, 1.0))
    cv2.line(canvas, (panel_x - 10, reaction_y), (panel_x + panel_w + 10, reaction_y), (255, 210, 120), 2, cv2.LINE_AA)
    for idx in range(6):
        ty = panel_y + idx * (panel_h // 5)
        cv2.line(canvas, (panel_x - 8, ty), (panel_x + panel_w + 8, ty), (70, 82, 106), 1, cv2.LINE_AA)
    cv2.putText(canvas, "kV", (panel_x - 2, panel_y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (225, 236, 255), 1, cv2.LINE_AA)


def draw_hand_overlay(frame: np.ndarray, points: np.ndarray | None) -> None:
    if points is None:
        return
    h, w = frame.shape[:2]
    for a, b in HAND_CONNECTIONS:
        ax, ay = points[a, :2]
        bx, by = points[b, :2]
        p0 = (int(ax * w), int(ay * h))
        p1 = (int(bx * w), int(by * h))
        cv2.line(frame, p0, p1, (70, 235, 255), 2, cv2.LINE_AA)
    for x, y, _ in points:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (255, 255, 255), -1, cv2.LINE_AA)


def nearest_hand_point(
    points: np.ndarray | None,
    width: int,
    height: int,
    emitter: Tuple[int, int],
) -> Tuple[Tuple[int, int], str]:
    if points is None:
        return (int(width * 0.26), int(height * 0.34)), "none"
    pts_px = np.empty((points.shape[0], 2), dtype=np.float32)
    pts_px[:, 0] = np.clip(points[:, 0] * width, 0.0, width - 1.0)
    pts_px[:, 1] = np.clip(points[:, 1] * height, 0.0, height - 1.0)
    deltas = pts_px - np.asarray(emitter, dtype=np.float32)
    idx = int(np.argmin(np.linalg.norm(deltas, axis=1)))
    point = pts_px[idx]
    return (int(point[0]), int(point[1])), LANDMARK_NAMES[idx]


def proximity_strength(
    target: Tuple[int, int],
    emitter: Tuple[int, int],
    width: int,
    height: int,
) -> float:
    dist = math.hypot(target[0] - emitter[0], target[1] - emitter[1])
    near = 0.06 * min(width, height)
    far = 0.34 * min(width, height)
    return clamp(1.0 - (dist - near) / max(1.0, far - near), 0.0, 1.0)


def style_camera_frame(frame: np.ndarray, voltage: float, active: bool) -> np.ndarray:
    canvas = frame.copy()
    shadow = canvas.copy()
    cv2.rectangle(shadow, (0, 0), (canvas.shape[1], canvas.shape[0]), (8, 12, 22), -1)
    shadow_alpha = 0.08 + 0.16 * voltage
    cv2.addWeighted(shadow, shadow_alpha, canvas, 1.0 - shadow_alpha, 0.0, canvas)

    vignette = canvas.copy()
    cv2.rectangle(vignette, (0, 0), (canvas.shape[1], canvas.shape[0]), (10, 10, 16), 36)
    cv2.addWeighted(vignette, 0.22, canvas, 0.78, 0.0, canvas)

    if active:
        tint = canvas.copy()
        cv2.rectangle(tint, (0, 0), (canvas.shape[1], canvas.shape[0]), (48, 28, 10), -1)
        cv2.addWeighted(tint, 0.05 + 0.08 * voltage, canvas, 0.95 - 0.08 * voltage, 0.0, canvas)

    return canvas


def draw_hud(
    canvas: np.ndarray,
    state: SimulationState,
    hand_label: str,
) -> None:
    lines = (
        f"Hand: {hand_label}",
        f"Contact: {state.contact_label}",
        f"Arc: {'ACTIVE' if state.active else 'IDLE'}",
        f"Proximity: {state.proximity:0.2f}",
        f"Voltage: {state.voltage:0.2f}",
        f"Reaction: {state.reaction_strength:0.2f}",
        "Raise voltage, then move the nearest part of your hand toward the coil",
        "'='/'-' coarse voltage  |  '['/']' fine voltage",
        "'0' reset voltage  |  'r' reset state  |  'q' quit",
        "Visual simulation only. No real hardware control.",
    )
    x = 28
    y = 34
    for idx, line in enumerate(lines):
        color = (220, 238, 255)
        if idx == 2 and state.active:
            color = (155, 245, 255)
        if idx == 3 and state.active:
            color = (140, 255, 190)
        cv2.putText(canvas, line, (x, y + idx * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (12, 16, 22), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (x, y + idx * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 1, cv2.LINE_AA)

    fps_line = f"FPS: {state.fps_ema:0.1f}"
    cv2.putText(
        canvas,
        fps_line,
        (canvas.shape[1] - 150, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (220, 238, 255),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open the default webcam.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    try:
        mode, tracker_solution, tracker_tasks = _tracker_setup()
    except Exception as exc:
        print(f"Error: {exc}")
        cap.release()
        return 1

    state = SimulationState()
    smoother = LandmarkSmoother()
    mirror = True
    task_timestamp_ms = 0
    frame_idx = 0
    last_ts = time.perf_counter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read webcam frame.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(1e-4, min(0.05, now - last_ts))
            last_ts = now
            state.fps_ema = 0.92 * state.fps_ema + 0.08 * (1.0 / dt)
            state.transition_burst = max(0.0, state.transition_burst - dt * 0.85)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if mode == "tasks":
                task_timestamp_ms = max(task_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = tracker_tasks.detect_for_video(mp_image, task_timestamp_ms)
                hands = _extract_task_hands(result, mirror)
            else:
                result = tracker_solution.process(rgb)
                hands = _extract_solution_hands(result, mirror)

            control_hand = pick_control_hand(hands)
            smoothed: np.ndarray | None = None
            hand_label = "Unknown"
            contact_label = "none"
            proximity = 0.0

            if control_hand is not None:
                smoothed = smoother.smooth(control_hand.landmarks)
                hand_label = control_hand.label
            else:
                smoother.reset()
                hand_label = "No hand"

            frame_idx += 1
            h, w = frame.shape[:2]
            emitter = (int(w * 0.10), int(h * 0.44))
            ground_y = int(h * 0.78)
            target, contact_label = nearest_hand_point(smoothed, w, h, emitter)
            if smoothed is not None:
                proximity = proximity_strength(target, emitter, w, h)

            state.proximity = lerp(state.proximity, proximity, 0.20)
            state.contact_label = contact_label
            state.voltage_target = clamp(state.voltage_target, 0.0, 1.0)
            state.voltage = lerp(state.voltage, state.voltage_target, 0.12)
            reaction_strength = state.voltage * state.proximity
            state.reaction_strength = reaction_strength
            state.active = reaction_strength >= 0.08

            if state.active and reaction_strength >= 0.12 and state.transition_burst <= 0.05:
                state.transition_burst = 0.10 + 0.35 * reaction_strength

            rng = np.random.default_rng(frame_idx * 97 + int(state.voltage * 1000.0))
            canvas = style_camera_frame(frame, state.voltage, state.active)

            floor = canvas.copy()
            cv2.rectangle(floor, (0, ground_y), (w, h), (18, 18, 28), -1)
            cv2.addWeighted(floor, 0.38, canvas, 0.62, 0.0, canvas)

            draw_coil(canvas, emitter, state.voltage, state.active)
            draw_corona_streamers(canvas, emitter, state.voltage, reaction_strength, rng)

            if state.active:
                draw_discharge(
                    canvas,
                    emitter,
                    target,
                    rng,
                    clamp(0.20 + 0.90 * reaction_strength + 0.25 * state.transition_burst, 0.0, 1.0),
                )
                draw_contact_detail(canvas, target, state.voltage, reaction_strength, rng)
                add_glow(canvas, target, int(14 + 36 * reaction_strength), (255, 205, 120), 0.16 + 0.26 * reaction_strength)
                add_glow(canvas, target, int(24 + 48 * reaction_strength), (255, 125, 48), 0.08 + 0.14 * reaction_strength)
                draw_plasma(
                    canvas,
                    target,
                    clamp(reaction_strength + 0.18 * state.transition_burst, 0.0, 1.0),
                    state.transition_burst,
                    rng,
                )
            else:
                corona = 0.10 + 0.18 * state.voltage
                add_glow(canvas, emitter, int(24 + 28 * state.voltage), (255, 180, 90), corona)

            draw_hand_overlay(canvas, smoothed)
            if smoothed is not None:
                cv2.circle(canvas, target, 8, (255, 245, 220), -1, cv2.LINE_AA)
                cv2.circle(canvas, target, 22, (255, 170, 80), 1, cv2.LINE_AA)

            draw_voltage_meter(canvas, state.voltage, reaction_strength)
            draw_hud(canvas, state, hand_label)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("-"):
                state.voltage_target = clamp(state.voltage_target - 0.08, 0.0, 1.0)
            elif key == ord("="):
                state.voltage_target = clamp(state.voltage_target + 0.08, 0.0, 1.0)
            elif key == ord("["):
                state.voltage_target = clamp(state.voltage_target - 0.02, 0.0, 1.0)
            elif key == ord("]"):
                state.voltage_target = clamp(state.voltage_target + 0.02, 0.0, 1.0)
            elif key == ord("0"):
                state.voltage_target = 0.35
            elif key == ord("r"):
                state.active = False
                state.voltage = 0.35
                state.voltage_target = 0.35
                state.proximity = 0.0
                state.reaction_strength = 0.0
                state.contact_label = "none"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if tracker_solution is not None:
            tracker_solution.close()
        if tracker_tasks is not None:
            tracker_tasks.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
