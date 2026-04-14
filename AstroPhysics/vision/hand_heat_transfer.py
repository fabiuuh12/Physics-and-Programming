#!/usr/bin/env python3
"""
Gesture-controlled heat transfer lab.

Floating hand avatars act as thermal reservoirs. Each hand cycles through
thermal states with a fist, and heat packets travel from the hotter hand
toward the colder hand.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import cv2
import mediapipe as mp
import numpy as np

try:
    from AppKit import NSScreen
except Exception:
    NSScreen = None


WINDOW_NAME = "Hand Heat Transfer"
MODULE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES: Tuple[Path, ...] = (
    MODULE_DIR / "models" / "hand_landmarker.task",
    MODULE_DIR.parent / "DefensiveSys" / "models" / "hand_landmarker.task",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])

CANVAS_W = 1320
CANVAS_H = 860
CAPTURE_W = 960
CAPTURE_H = 540

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

SLOT_KEYS: Tuple[str, str] = ("slot0", "slot1")
HAND_STALE_S = 0.45
SMOOTH_ALPHA = 0.30
HAND_REACH_X_GAIN = 1.02
HAND_REACH_Y_GAIN = 1.06
HAND_MIN_PALM_NORM = 0.035
HAND_MAX_PALM_NORM = 0.17
HAND_NORMALIZED_SCALE = 82.0
FIST_ON_SCORE = 0.60
FIST_OFF_SCORE = 0.38
FIST_COOLDOWN_S = 0.55
THERMAL_BALANCE_DEADBAND_C = 6.0

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
PALM_POLY = (0, 1, 5, 9, 13, 17)


@dataclass(frozen=True)
class ThermalStateSpec:
    name: str
    temp_c: float
    palm_color: Tuple[int, int, int]
    line_color: Tuple[int, int, int]
    joint_color: Tuple[int, int, int]
    glow_color: Tuple[int, int, int]


THERMAL_STATES: Tuple[ThermalStateSpec, ...] = (
    ThermalStateSpec(
        name="ice",
        temp_c=-30.0,
        palm_color=(208, 126, 54),
        line_color=(255, 220, 172),
        joint_color=(255, 246, 226),
        glow_color=(255, 188, 130),
    ),
    ThermalStateSpec(
        name="liquid",
        temp_c=24.0,
        palm_color=(196, 168, 68),
        line_color=(244, 228, 180),
        joint_color=(255, 244, 214),
        glow_color=(220, 206, 128),
    ),
    ThermalStateSpec(
        name="steam",
        temp_c=135.0,
        palm_color=(84, 152, 232),
        line_color=(122, 192, 255),
        joint_color=(190, 226, 255),
        glow_color=(106, 182, 255),
    ),
    ThermalStateSpec(
        name="plasma",
        temp_c=420.0,
        palm_color=(76, 214, 255),
        line_color=(164, 236, 255),
        joint_color=(228, 246, 255),
        glow_color=(132, 232, 255),
    ),
)


@dataclass
class HandObservation:
    label: str
    score: float
    landmarks: List[Tuple[float, float, float]]


@dataclass
class TrackedHand:
    pose: np.ndarray
    label: str
    score: float
    last_seen: float


@dataclass
class SceneState:
    left_state_index: int = 2
    right_state_index: int = 0
    left_fist_latched: bool = False
    right_fist_latched: bool = False
    left_last_toggle_ts: float = 0.0
    right_last_toggle_ts: float = 0.0


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


def _get_screen_size() -> Tuple[int, int]:
    if NSScreen is not None:
        try:
            frame = NSScreen.mainScreen().frame()
            width = int(frame.size.width)
            height = int(frame.size.height)
            if width > 0 and height > 0:
                return width, height
        except Exception:
            pass
    return CANVAS_W, CANVAS_H


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
        out.append(
            HandObservation(
                label=_normalize_label(label, mirror),
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
        out.append(
            HandObservation(
                label=_normalize_label(label, mirror),
                score=score,
                landmarks=[(lm.x, lm.y, lm.z) for lm in hand_landmarks],
            )
        )
    return out


def _wrist_xy(hand: HandObservation) -> np.ndarray:
    return np.asarray(hand.landmarks[WRIST][:2], dtype=np.float32)


def _slot_wrist_xy(tracked: Dict[str, TrackedHand], slot: str) -> np.ndarray | None:
    item = tracked.get(slot)
    if item is None:
        return None
    return item.pose[WRIST, :2].astype(np.float32)


def _assign_hands_to_slots(
    hands: List[HandObservation],
    tracked: Dict[str, TrackedHand],
) -> List[Tuple[str, HandObservation]]:
    if not hands:
        return []

    hands = sorted(hands, key=lambda h: h.score, reverse=True)[:2]
    existing = [slot for slot in SLOT_KEYS if slot in tracked]

    if len(hands) == 1:
        hand = hands[0]
        if existing:
            wrist = _wrist_xy(hand)
            dists: List[Tuple[float, str]] = []
            for slot in existing:
                sw = _slot_wrist_xy(tracked, slot)
                if sw is not None:
                    dists.append((float(np.linalg.norm(wrist - sw)), slot))
            if dists:
                return [(min(dists, key=lambda item: item[0])[1], hand)]
        for slot in SLOT_KEYS:
            if slot not in tracked:
                return [(slot, hand)]
        return [(SLOT_KEYS[0], hand)]

    h0, h1 = hands
    w0 = _wrist_xy(h0)
    w1 = _wrist_xy(h1)

    if all(slot in tracked for slot in SLOT_KEYS):
        s0 = _slot_wrist_xy(tracked, SLOT_KEYS[0])
        s1 = _slot_wrist_xy(tracked, SLOT_KEYS[1])
        if s0 is not None and s1 is not None:
            cost_direct = float(np.linalg.norm(w0 - s0) + np.linalg.norm(w1 - s1))
            cost_cross = float(np.linalg.norm(w1 - s0) + np.linalg.norm(w0 - s1))
            if cost_direct <= cost_cross:
                return [(SLOT_KEYS[0], h0), (SLOT_KEYS[1], h1)]
            return [(SLOT_KEYS[0], h1), (SLOT_KEYS[1], h0)]

    if w0[0] <= w1[0]:
        return [(SLOT_KEYS[0], h0), (SLOT_KEYS[1], h1)]
    return [(SLOT_KEYS[0], h1), (SLOT_KEYS[1], h0)]


def _best_hand_by_label(
    tracked: Dict[str, TrackedHand],
    label: str,
    now: float,
) -> TrackedHand | None:
    want = label.lower()
    best: TrackedHand | None = None
    best_score = -1.0
    for hand in tracked.values():
        if now - hand.last_seen > HAND_STALE_S:
            continue
        if hand.label.lower() != want:
            continue
        if hand.score > best_score:
            best = hand
            best_score = hand.score
    return best


def _best_hand_by_side(
    tracked: Dict[str, TrackedHand],
    left_side: bool,
    now: float,
) -> TrackedHand | None:
    candidates = [hand for hand in tracked.values() if (now - hand.last_seen) <= HAND_STALE_S]
    if not candidates:
        return None
    picker = min if left_side else max
    return picker(candidates, key=lambda hand: float(hand.pose[WRIST, 0]))


def _palm_size(points: np.ndarray) -> float:
    return float(
        (
            np.linalg.norm(points[WRIST, :2] - points[INDEX_MCP, :2])
            + np.linalg.norm(points[WRIST, :2] - points[PINKY_MCP, :2])
            + np.linalg.norm(points[INDEX_MCP, :2] - points[PINKY_MCP, :2])
        ) / 3.0
    )


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    n1 = float(np.linalg.norm(ab))
    n2 = float(np.linalg.norm(cb))
    if n1 <= 1e-5 or n2 <= 1e-5:
        return 180.0
    cosang = float(np.dot(ab, cb) / (n1 * n2))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def _fist_score(points: np.ndarray) -> float:
    palm = max(1e-4, _palm_size(points))
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
        ang = _joint_angle_deg(mcp, pip, tip)
        ang_closed = float(np.clip((155.0 - ang) / 70.0, 0.0, 1.0))
        scores.append(0.55 * dist_closed + 0.45 * ang_closed)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _update_state_toggle(
    hand: TrackedHand | None,
    latched: bool,
    last_toggle_ts: float,
    state_index: int,
    now: float,
) -> Tuple[bool, float, int]:
    if hand is None:
        return False, last_toggle_ts, state_index
    score = _fist_score(hand.pose)
    if not latched:
        if score >= FIST_ON_SCORE and (now - last_toggle_ts) >= FIST_COOLDOWN_S:
            return True, now, (state_index + 1) % len(THERMAL_STATES)
        return False, last_toggle_ts, state_index
    if score <= FIST_OFF_SCORE:
        return False, last_toggle_ts, state_index
    return True, last_toggle_ts, state_index


def _default_hand_center(side: str) -> np.ndarray:
    if side == "left":
        return np.asarray([0.30, 0.57], dtype=np.float32)
    return np.asarray([0.70, 0.57], dtype=np.float32)


def _hand_canvas_points(points: np.ndarray, canvas_w: int, canvas_h: int, t: float) -> np.ndarray:
    pts = points.copy()
    wrist = pts[WRIST, :2]
    rel = pts[:, :2] - wrist
    tx = (wrist[0] - 0.5) * canvas_w * HAND_REACH_X_GAIN
    ty = (wrist[1] - 0.5) * canvas_h * HAND_REACH_Y_GAIN
    palm = (
        np.linalg.norm(pts[WRIST, :2] - pts[INDEX_MCP, :2])
        + np.linalg.norm(pts[WRIST, :2] - pts[PINKY_MCP, :2])
        + np.linalg.norm(pts[INDEX_MCP, :2] - pts[PINKY_MCP, :2])
    ) / 3.0
    palm = float(np.clip(palm, HAND_MIN_PALM_NORM, HAND_MAX_PALM_NORM))
    rel = rel / palm
    bob = 10.0 * math.sin(1.28 * t)
    cx = 0.5 * canvas_w + tx
    cy = 0.50 * canvas_h + ty + bob
    out = np.zeros((21, 2), dtype=np.float32)
    out[:, 0] = cx + rel[:, 0] * HAND_NORMALIZED_SCALE
    out[:, 1] = cy + rel[:, 1] * HAND_NORMALIZED_SCALE
    return out


def _thermal_avatar_from_hand(
    hand: TrackedHand | None,
    side: str,
    t: float,
) -> Dict[str, object]:
    if hand is None:
        center = _default_hand_center(side)
        return {
            "active": False,
            "center_norm": center,
            "center_px": np.asarray([center[0] * CANVAS_W, center[1] * CANVAS_H], dtype=np.float32),
            "draw_pts": None,
            "label": side.title(),
            "palm_size": 0.06,
        }

    draw_pts = _hand_canvas_points(hand.pose, CANVAS_W, CANVAS_H, t)
    center_norm = np.clip(hand.pose[[WRIST, INDEX_MCP, PINKY_MCP], :2].mean(axis=0), 0.05, 0.95)
    center_px = draw_pts[[WRIST, INDEX_MCP, PINKY_MCP], :2].mean(axis=0)
    return {
        "active": True,
        "center_norm": center_norm.astype(np.float32),
        "center_px": center_px.astype(np.float32),
        "draw_pts": draw_pts,
        "label": hand.label,
        "palm_size": float(np.clip(_palm_size(hand.pose), 0.05, 0.20)),
    }


def _state_temp_norm(state: ThermalStateSpec) -> float:
    low = THERMAL_STATES[0].temp_c
    high = THERMAL_STATES[-1].temp_c
    return float(np.clip((state.temp_c - low) / (high - low), 0.0, 1.0))


def _gradient_background(
    canvas: np.ndarray,
    t: float,
    left_state: ThermalStateSpec,
    right_state: ThermalStateSpec,
) -> None:
    h, w = canvas.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    canvas[..., 0] = np.clip(22.0 + 28.0 * (1.0 - y) + 12.0 * x, 0, 255).astype(np.uint8)
    canvas[..., 1] = np.clip(14.0 + 22.0 * (1.0 - y) + 10.0 * x, 0, 255).astype(np.uint8)
    canvas[..., 2] = np.clip(18.0 + 28.0 * y, 0, 255).astype(np.uint8)

    overlay = canvas.copy()
    left_norm = _state_temp_norm(left_state)
    right_norm = _state_temp_norm(right_state)
    left_glow = tuple(int(c) for c in left_state.glow_color)
    right_glow = tuple(int(c) for c in right_state.glow_color)
    cv2.circle(
        overlay,
        (int(w * 0.24), int(h * (0.28 + 0.03 * math.sin(0.8 * t)))),
        int(220 + 80 * left_norm),
        left_glow,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        overlay,
        (int(w * 0.80), int(h * (0.70 + 0.03 * math.cos(0.7 * t)))),
        int(220 + 100 * right_norm),
        right_glow,
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0.0, canvas)


def _draw_energy_grid(canvas: np.ndarray, left_state: ThermalStateSpec, right_state: ThermalStateSpec) -> None:
    mix = 0.5 * (_state_temp_norm(left_state) + _state_temp_norm(right_state))
    color = (
        int(34 + 36 * (1.0 - mix)),
        int(42 + 34 * mix),
        int(58 + 54 * mix),
    )
    for x in range(0, CANVAS_W, 72):
        cv2.line(canvas, (x, 0), (x, CANVAS_H), color, 1, cv2.LINE_AA)
    for y in range(0, CANVAS_H, 72):
        cv2.line(canvas, (0, y), (CANVAS_W, y), color, 1, cv2.LINE_AA)


def _draw_hand_avatar(
    canvas: np.ndarray,
    avatar: Dict[str, object],
    state: ThermalStateSpec,
    active: bool,
    now: float,
) -> None:
    center_px = np.asarray(avatar["center_px"], dtype=np.float32)
    cx, cy = int(center_px[0]), int(center_px[1])
    temp_norm = _state_temp_norm(state)
    aura_r = int(36 + 34 * temp_norm)

    glow = canvas.copy()
    cv2.circle(glow, (cx, cy), aura_r, state.glow_color, -1, cv2.LINE_AA)
    cv2.circle(glow, (cx, cy), int(aura_r * 1.55), state.glow_color, 1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.13 if active else 0.06, canvas, 0.87 if active else 0.94, 0.0, canvas)

    if not active or avatar["draw_pts"] is None:
        cv2.circle(canvas, (cx, cy), 20, state.line_color, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{avatar['label']} {state.name}",
            (cx - 48, cy - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (156, 170, 190),
            1,
            cv2.LINE_AA,
        )
        return

    draw_pts = np.asarray(avatar["draw_pts"], dtype=np.float32)
    palm_poly = np.array([[int(draw_pts[i, 0]), int(draw_pts[i, 1])] for i in PALM_POLY], dtype=np.int32)
    cv2.fillConvexPoly(canvas, palm_poly, state.palm_color, cv2.LINE_AA)

    for a, b in HAND_CONNECTIONS:
        p0 = (int(draw_pts[a, 0]), int(draw_pts[a, 1]))
        p1 = (int(draw_pts[b, 0]), int(draw_pts[b, 1]))
        cv2.line(canvas, p0, p1, state.line_color, 2, cv2.LINE_AA)

    for i in range(21):
        px, py = int(draw_pts[i, 0]), int(draw_pts[i, 1])
        radius = 5 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(canvas, (px, py), radius + 2, (36, 36, 40), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, state.joint_color, -1, cv2.LINE_AA)

    for ring in (1.05, 1.50):
        cv2.circle(canvas, (cx, cy), int(aura_r * ring), state.line_color, 1, cv2.LINE_AA)

    if state.name == "ice":
        for idx in range(6):
            ang = now * 0.5 + idx * (math.pi / 3.0)
            p0 = (int(cx + math.cos(ang) * 20), int(cy + math.sin(ang) * 20))
            p1 = (int(cx + math.cos(ang) * 34), int(cy + math.sin(ang) * 34))
            cv2.line(canvas, p0, p1, state.joint_color, 1, cv2.LINE_AA)
    elif state.name in {"steam", "plasma"}:
        for idx in range(4):
            phase = now * (1.1 + 0.12 * idx) + idx * 0.9
            px = int(cx - 18 + idx * 12 + 4 * math.sin(phase))
            py = int(cy - aura_r - 18 - 16 * abs(math.sin(phase)))
            cv2.circle(canvas, (px, py), 5 if state.name == "plasma" else 4, state.glow_color, 1, cv2.LINE_AA)

    text_color = (232, 238, 248) if active else (146, 160, 180)
    cv2.putText(
        canvas,
        f"{avatar['label']} {state.name}",
        (int(draw_pts[WRIST, 0]) - 44, int(draw_pts[WRIST, 1]) - 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{state.temp_c:0.0f} C",
        (int(draw_pts[WRIST, 0]) - 34, int(draw_pts[WRIST, 1]) - 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        text_color,
        1,
        cv2.LINE_AA,
    )


def _eval_quad_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    omt = 1.0 - t
    return (omt * omt) * p0 + 2.0 * omt * t * p1 + (t * t) * p2


def _bridge_control_point(p0: np.ndarray, p2: np.ndarray) -> np.ndarray:
    mid = 0.5 * (p0 + p2)
    delta = p2 - p0
    sep = float(np.linalg.norm(delta))
    if sep < 1e-5:
        return mid
    perp = np.asarray([-delta[1], delta[0]], dtype=np.float32) / sep
    if perp[1] > 0.0:
        perp = -perp
    return mid + perp * (0.18 * sep + 28.0)


def _draw_heat_bridge(
    canvas: np.ndarray,
    left_avatar: Dict[str, object],
    right_avatar: Dict[str, object],
    left_state: ThermalStateSpec,
    right_state: ThermalStateSpec,
    now: float,
) -> Tuple[float, float, str]:
    if not left_avatar["active"] or not right_avatar["active"]:
        return 0.0, 0.0, "tracking"

    p0 = np.asarray(left_avatar["center_px"], dtype=np.float32)
    p2 = np.asarray(right_avatar["center_px"], dtype=np.float32)
    sep = float(np.linalg.norm(p2 - p0))
    if sep < 16.0:
        return sep / CANVAS_W, 0.0, "balanced"

    temp_diff = left_state.temp_c - right_state.temp_c
    sep_norm = float(np.linalg.norm(
        np.asarray(left_avatar["center_norm"], dtype=np.float32)
        - np.asarray(right_avatar["center_norm"], dtype=np.float32)
    ))
    proximity = float(np.clip(1.12 - sep_norm, 0.18, 1.0))
    flux = float(np.clip((abs(temp_diff) / 450.0) * (0.34 + 0.66 * proximity), 0.0, 1.0))
    direction = "balanced"
    if temp_diff > THERMAL_BALANCE_DEADBAND_C:
        direction = "L -> R"
        hot_state = left_state
        cold_state = right_state
        forward = True
    elif temp_diff < -THERMAL_BALANCE_DEADBAND_C:
        direction = "R -> L"
        hot_state = right_state
        cold_state = left_state
        forward = False
    else:
        hot_state = left_state
        cold_state = right_state
        forward = True
        flux *= 0.24

    p1 = _bridge_control_point(p0, p2)
    path = np.asarray([
        _eval_quad_bezier(p0, p1, p2, t) for t in np.linspace(0.0, 1.0, 48)
    ], dtype=np.int32)

    overlay = canvas.copy()
    bridge_color = hot_state.glow_color if abs(temp_diff) > THERMAL_BALANCE_DEADBAND_C else (198, 206, 216)
    cv2.polylines(overlay, [path], False, bridge_color, 12, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.11 + 0.15 * flux, canvas, 0.89 - 0.15 * flux, 0.0, canvas)
    cv2.polylines(canvas, [path], False, bridge_color, 2, cv2.LINE_AA)

    packet_count = int(5 + 12 * flux)
    speed = 0.22 + 0.70 * flux
    for idx in range(packet_count):
        phase = (now * speed + idx / max(1, packet_count)) % 1.0
        if not forward:
            phase = 1.0 - phase
        pt = _eval_quad_bezier(p0, p1, p2, phase)
        tail = _eval_quad_bezier(p0, p1, p2, max(0.0, min(1.0, phase - 0.035 if forward else phase + 0.035)))
        pt_i = (int(pt[0]), int(pt[1]))
        tail_i = (int(tail[0]), int(tail[1]))
        radius = 4 if idx % 2 == 0 else 3
        cv2.line(canvas, tail_i, pt_i, hot_state.line_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, pt_i, radius + 4, hot_state.glow_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt_i, radius, hot_state.joint_color, -1, cv2.LINE_AA)

    cold_glow = canvas.copy()
    cold_center = np.asarray(right_avatar["center_px"] if forward else left_avatar["center_px"], dtype=np.float32)
    cv2.circle(cold_glow, (int(cold_center[0]), int(cold_center[1])), int(18 + 26 * flux), cold_state.glow_color, -1, cv2.LINE_AA)
    cv2.addWeighted(cold_glow, 0.07 + 0.10 * flux, canvas, 0.93 - 0.10 * flux, 0.0, canvas)
    return sep / CANVAS_W, flux, direction


def _draw_temperature_gauges(
    canvas: np.ndarray,
    scene: SceneState,
    left_state: ThermalStateSpec,
    right_state: ThermalStateSpec,
) -> None:
    panel_x = 28
    panel_y = 240
    panel_w = 250
    panel_h = 138
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.74, canvas, 0.26, 0.0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (86, 96, 118), 1, cv2.LINE_AA)

    states = (
        ("left", left_state, scene.left_state_index),
        ("right", right_state, scene.right_state_index),
    )
    for row, (label, state, idx) in enumerate(states):
        y = panel_y + 30 + row * 48
        cv2.putText(canvas, f"{label}  {state.name}", (panel_x + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (228, 234, 244), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{state.temp_c:6.0f} C", (panel_x + 142, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, state.line_color, 1, cv2.LINE_AA)
        for dot in range(len(THERMAL_STATES)):
            cx = panel_x + 22 + dot * 28
            color = THERMAL_STATES[dot].glow_color if dot == idx else (84, 92, 108)
            cv2.circle(canvas, (cx, y + 18), 7 if dot == idx else 5, color, -1, cv2.LINE_AA)


def _draw_webcam_inset(canvas: np.ndarray, frame: np.ndarray, hands: List[HandObservation]) -> None:
    margin = 18
    inset_w = 300
    inset_h = 180
    preview = frame.copy()
    ph, pw = preview.shape[:2]
    for hand in hands:
        pts = []
        color = (72, 232, 255) if hand.label.lower() == "right" else (136, 255, 156)
        for x, y, _ in hand.landmarks:
            px = int(np.clip(x * pw, 0, pw - 1))
            py = int(np.clip(y * ph, 0, ph - 1))
            pts.append((px, py))
        if len(pts) >= 21:
            for a, b in HAND_CONNECTIONS:
                cv2.line(preview, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for px, py in pts:
                cv2.circle(preview, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
    preview = cv2.resize(preview, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    panel = canvas.copy()
    cv2.rectangle(panel, (margin - 8, margin - 8), (margin + inset_w + 8, margin + inset_h + 34), (10, 12, 18), -1)
    cv2.addWeighted(panel, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.rectangle(canvas, (margin - 8, margin - 8), (margin + inset_w + 8, margin + inset_h + 34), (88, 96, 118), 1, cv2.LINE_AA)
    canvas[margin: margin + inset_h, margin: margin + inset_w] = preview
    cv2.putText(canvas, "webcam tracking", (margin + 8, margin + inset_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (216, 226, 244), 1, cv2.LINE_AA)


def _draw_hud(
    canvas: np.ndarray,
    left_state: ThermalStateSpec,
    right_state: ThermalStateSpec,
    flux: float,
    direction: str,
    separation: float,
    fps: float,
    left_active: bool,
    right_active: bool,
) -> None:
    temp_diff = left_state.temp_c - right_state.temp_c
    cv2.putText(canvas, "Hand Heat Transfer", (364, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.06, (236, 240, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Make a fist with either hand to cycle its thermal state. Heat flows from the hotter hand to the colder one.", (364, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (176, 194, 220), 1, cv2.LINE_AA)

    panel_x = 344
    panel_y = CANVAS_H - 106
    panel_w = 720
    panel_h = 66
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (84, 96, 118), 1, cv2.LINE_AA)

    stat_font = 0.50
    top_y = panel_y + 25
    dy = 23
    cv2.putText(canvas, f"L  {left_state.name:>6}  {left_state.temp_c:6.0f} C", (panel_x + 18, top_y), cv2.FONT_HERSHEY_SIMPLEX, stat_font, left_state.line_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"R  {right_state.name:>6}  {right_state.temp_c:6.0f} C", (panel_x + 18, top_y + dy), cv2.FONT_HERSHEY_SIMPLEX, stat_font, right_state.line_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"deltaT  {temp_diff:6.0f} C", (panel_x + 274, top_y), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (228, 234, 244), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"heat flux  {100.0 * flux:5.0f} %", (panel_x + 274, top_y + dy), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (228, 234, 244), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"direction  {direction}", (panel_x + 492, top_y), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (228, 234, 244), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"sep  {separation:4.2f}", (panel_x + 492, top_y + dy), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (228, 234, 244), 1, cv2.LINE_AA)

    footer_font = 0.40
    cv2.putText(canvas, f"L:{'ok' if left_active else 'def'} R:{'ok' if right_active else 'def'} fps:{fps:4.1f}", (20, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, footer_font, (150, 170, 198), 1, cv2.LINE_AA)
    cv2.putText(canvas, "[m] mirror  [f] full  [r] reset states  [q] quit", (CANVAS_W - 316, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, footer_font, (150, 170, 198), 1, cv2.LINE_AA)


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
            max_num_hands=2,
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
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        tracker_tasks = vision.HandLandmarker.create_from_options(options)
        use_tasks = True
    else:
        print("Unsupported mediapipe build: expected `solutions` or `tasks.vision` APIs.")
        cap.release()
        return 1

    tracked: Dict[str, TrackedHand] = {}
    smoother = MultiLandmarkSmoother(alpha=SMOOTH_ALPHA)
    scene = SceneState()
    mirror = True
    fullscreen = True
    screen_w, screen_h = _get_screen_size()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_t = time.perf_counter()
    fps = 60.0

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
            fps = 0.90 * fps + 0.10 * (1.0 / dt)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if use_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = tracker_tasks.detect_for_video(mp_image, task_timestamp_ms)
                task_timestamp_ms += max(1, int(dt * 1000.0))
                extracted = _extract_task_hands(result, mirror=mirror)
            else:
                result = tracker_solution.process(rgb)
                extracted = _extract_solution_hands(result, mirror=mirror)

            assignments = _assign_hands_to_slots(extracted, tracked)
            active_keys: List[str] = []
            for hand_key, hand in assignments:
                active_keys.append(hand_key)
                tracked[hand_key] = TrackedHand(
                    pose=smoother.smooth(hand_key, hand.landmarks),
                    label=hand.label,
                    score=hand.score,
                    last_seen=now,
                )
            for key in list(tracked.keys()):
                if key not in active_keys and (now - tracked[key].last_seen) > HAND_STALE_S:
                    del tracked[key]
            smoother.prune(tracked.keys())

            left_hand = _best_hand_by_label(tracked, "left", now)
            right_hand = _best_hand_by_label(tracked, "right", now)
            if left_hand is None:
                left_hand = _best_hand_by_side(tracked, left_side=True, now=now)
            if right_hand is None:
                right_hand = _best_hand_by_side(tracked, left_side=False, now=now)

            scene.left_fist_latched, scene.left_last_toggle_ts, scene.left_state_index = _update_state_toggle(
                left_hand,
                scene.left_fist_latched,
                scene.left_last_toggle_ts,
                scene.left_state_index,
                now,
            )
            scene.right_fist_latched, scene.right_last_toggle_ts, scene.right_state_index = _update_state_toggle(
                right_hand,
                scene.right_fist_latched,
                scene.right_last_toggle_ts,
                scene.right_state_index,
                now,
            )

            left_state = THERMAL_STATES[scene.left_state_index]
            right_state = THERMAL_STATES[scene.right_state_index]
            left_avatar = _thermal_avatar_from_hand(left_hand, "left", now)
            right_avatar = _thermal_avatar_from_hand(right_hand, "right", now)

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            _gradient_background(canvas, now, left_state, right_state)
            _draw_energy_grid(canvas, left_state, right_state)
            separation, flux, direction = _draw_heat_bridge(
                canvas,
                left_avatar,
                right_avatar,
                left_state,
                right_state,
                now,
            )
            _draw_hand_avatar(canvas, left_avatar, left_state, left_hand is not None, now)
            _draw_hand_avatar(canvas, right_avatar, right_state, right_hand is not None, now)
            _draw_temperature_gauges(canvas, scene, left_state, right_state)
            _draw_webcam_inset(canvas, frame, extracted)
            _draw_hud(
                canvas,
                left_state,
                right_state,
                flux,
                direction,
                separation,
                fps,
                left_hand is not None,
                right_hand is not None,
            )

            display = canvas
            if fullscreen and (screen_w != CANVAS_W or screen_h != CANVAS_H):
                display = cv2.resize(canvas, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
                if not fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
            if key == ord("m"):
                mirror = not mirror
                scene.left_fist_latched = False
                scene.right_fist_latched = False
            if key == ord("r"):
                scene = SceneState()
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
