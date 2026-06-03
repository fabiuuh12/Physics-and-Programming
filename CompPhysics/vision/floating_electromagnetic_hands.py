#!/usr/bin/env python3
"""
Gesture-controlled electromagnetism field lab.

Floating hand avatars remain visible and act as charged sources.
Each hand can flip polarity with a fist, and either pinch can inject
test-charge packets into the field.
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


WINDOW_NAME = "Floating Electromagnetic Hands"
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
PINKY_MCP = 17

SLOT_KEYS: Tuple[str, str] = ("slot0", "slot1")
HAND_STALE_S = 0.45
SMOOTH_ALPHA = 0.30
PINCH_CLOSE_RATIO = 0.36
PINCH_RELEASE_RATIO = 0.48

HAND_REACH_X_GAIN = 1.00
HAND_REACH_Y_GAIN = 1.02
HAND_MIN_PALM_NORM = 0.035
HAND_MAX_PALM_NORM = 0.17
HAND_NORMALIZED_SCALE = 82.0

FIELD_COLS = 18
FIELD_ROWS = 12
STREAMLINE_SEEDS = 14
MAX_PARTICLES = 110
FIELD_GLOW_W = 220
FIELD_GLOW_H = 144
FIST_ON_SCORE = 0.60
FIST_OFF_SCORE = 0.38
FIST_COOLDOWN_S = 0.55
CONTINUOUS_SPAWN_HZ = 7.0

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
PALM_POLY = (0, 1, 5, 9, 13, 17)


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
class Particle:
    pos: np.ndarray
    prev_pos: np.ndarray
    vel: np.ndarray
    charge: float
    ttl: float


@dataclass
class SceneState:
    left_pinch_latched: bool = False
    right_pinch_latched: bool = False
    injected_count: int = 0
    left_source_negative: bool = False
    right_source_negative: bool = True
    left_fist_latched: bool = False
    right_fist_latched: bool = False
    left_last_toggle_ts: float = 0.0
    right_last_toggle_ts: float = 0.0
    spawn_accumulator: float = 0.0


class MultiLandmarkSmoother:
    def __init__(self, alpha: float = SMOOTH_ALPHA) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 0.99))
        self._state: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._state.clear()

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
    max_age: float,
) -> TrackedHand | None:
    want = label.lower()
    best: TrackedHand | None = None
    best_score = -1.0
    for hand in tracked.values():
        if now - hand.last_seen > max_age:
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
    max_age: float,
) -> TrackedHand | None:
    candidates = [
        hand for hand in tracked.values() if (now - hand.last_seen) <= max_age
    ]
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
        )
        / 3.0
    )


def _pinch_ratio(points: np.ndarray) -> float:
    palm = max(1e-4, _palm_size(points))
    pinch = np.linalg.norm(points[THUMB_TIP, :2] - points[INDEX_TIP, :2])
    return float(pinch / palm)


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


def _update_charge_toggle(
    hand: TrackedHand | None,
    latched: bool,
    last_toggle_ts: float,
    negative: bool,
    now: float,
) -> Tuple[bool, float, bool]:
    if hand is None:
        return False, last_toggle_ts, negative
    score = _fist_score(hand.pose)
    if not latched:
        if score >= FIST_ON_SCORE and (now - last_toggle_ts) >= FIST_COOLDOWN_S:
            return True, now, not negative
        return False, last_toggle_ts, negative
    if score <= FIST_OFF_SCORE:
        return False, last_toggle_ts, negative
    return True, last_toggle_ts, negative


def _update_pinch_latch(
    hand: TrackedHand | None,
    latched: bool,
) -> Tuple[bool, bool]:
    if hand is None:
        return False, False
    ratio = _pinch_ratio(hand.pose)
    if latched:
        latched = ratio < PINCH_RELEASE_RATIO
        return latched, False
    if ratio < PINCH_CLOSE_RATIO:
        return True, True
    return False, False


def _default_hand_center(side: str) -> np.ndarray:
    if side == "left":
        return np.asarray([0.30, 0.56], dtype=np.float32)
    return np.asarray([0.70, 0.56], dtype=np.float32)


def _hand_canvas_points(points: np.ndarray, canvas_w: int, canvas_h: int, t: float) -> np.ndarray:
    pts = points.copy()
    wrist = pts[0, :2]
    rel = pts[:, :2] - wrist
    tx = (wrist[0] - 0.5) * canvas_w * HAND_REACH_X_GAIN
    ty = (wrist[1] - 0.5) * canvas_h * HAND_REACH_Y_GAIN
    palm = (
        np.linalg.norm(pts[0, :2] - pts[5, :2])
        + np.linalg.norm(pts[0, :2] - pts[17, :2])
        + np.linalg.norm(pts[5, :2] - pts[17, :2])
    ) / 3.0
    palm = float(np.clip(palm, HAND_MIN_PALM_NORM, HAND_MAX_PALM_NORM))
    rel = rel / palm
    bob = 10.0 * math.sin(1.35 * t)
    cx = 0.5 * canvas_w + tx
    cy = 0.50 * canvas_h + ty + bob
    out = np.zeros((21, 2), dtype=np.float32)
    out[:, 0] = cx + rel[:, 0] * HAND_NORMALIZED_SCALE
    out[:, 1] = cy + rel[:, 1] * HAND_NORMALIZED_SCALE
    return out


def _charge_source_from_hand(
    hand: TrackedHand | None,
    side: str,
    t: float,
    source_negative: bool,
) -> Dict[str, object]:
    if hand is None:
        center = _default_hand_center(side)
        center_px = np.asarray([center[0] * CANVAS_W, center[1] * CANVAS_H], dtype=np.float32)
        return {
            "active": False,
            "center_norm": center,
            "center_px": center_px,
            "charge": -1.0 if source_negative else 1.0,
            "strength": 0.9,
            "draw_pts": None,
            "label": side.title(),
            "index_tip_norm": center.copy(),
        }

    draw_pts = _hand_canvas_points(hand.pose, CANVAS_W, CANVAS_H, t)
    center_norm = np.clip(hand.pose[[WRIST, INDEX_MCP, PINKY_MCP], :2].mean(axis=0), 0.05, 0.95)
    center_px = draw_pts[[WRIST, INDEX_MCP, PINKY_MCP], :2].mean(axis=0)
    palm_size = _palm_size(hand.pose)
    strength = float(np.clip(0.8 + 5.5 * palm_size, 0.8, 1.6))
    return {
        "active": True,
        "center_norm": center_norm.astype(np.float32),
        "center_px": center_px.astype(np.float32),
        "charge": -1.0 if source_negative else 1.0,
        "strength": strength,
        "draw_pts": draw_pts,
        "label": hand.label,
        "index_tip_norm": np.asarray(hand.pose[INDEX_TIP, :2], dtype=np.float32),
    }


def _field_vector(pos: np.ndarray, sources: Sequence[Dict[str, object]]) -> np.ndarray:
    e = np.zeros(2, dtype=np.float32)
    for source in sources:
        center = np.asarray(source["center_norm"], dtype=np.float32)
        charge = float(source["charge"]) * float(source["strength"])
        d = pos - center
        r2 = float(np.dot(d, d)) + 0.0025
        e += (charge / (r2 * math.sqrt(r2))) * d
    return e


def _seed_particles() -> List[Particle]:
    return []


def _emit_particles(
    particles: List[Particle],
    source: Dict[str, object],
    sources: Sequence[Dict[str, object]],
    count: int,
    ttl: float,
    speed_scale: float,
    spread_scale: float,
) -> None:
    pos = np.asarray(source["index_tip_norm"], dtype=np.float32)
    tip_dir = pos - np.asarray(source["center_norm"], dtype=np.float32)
    norm = float(np.linalg.norm(tip_dir))
    if norm < 1e-4:
        tip_dir = np.asarray([0.10, -0.02], dtype=np.float32)
    else:
        tip_dir /= norm

    field_push = _field_vector(pos, sources)
    field_norm = float(np.linalg.norm(field_push))
    if field_norm > 1e-4:
        field_push /= field_norm

    charge = float(source["charge"])
    expected_dir = tip_dir + float(np.sign(charge) if charge != 0.0 else 1.0) * 0.70 * field_push
    expected_norm = float(np.linalg.norm(expected_dir))
    if expected_norm > 1e-4:
        expected_dir /= expected_norm
    base = speed_scale * (0.26 * expected_dir)
    for i in range(count):
        jitter = np.asarray([
            spread_scale * 0.018 * math.cos(i * 0.83),
            spread_scale * 0.018 * math.sin(i * 1.11),
        ], dtype=np.float32)
        particles.append(
            Particle(
                pos=pos.copy() + 0.01 * tip_dir,
                prev_pos=pos.copy() + 0.01 * tip_dir,
                vel=base + jitter,
                charge=charge,
                ttl=ttl,
            )
        )
    if len(particles) > MAX_PARTICLES:
        del particles[: len(particles) - MAX_PARTICLES]


def _update_particles(
    particles: List[Particle],
    sources: Sequence[Dict[str, object]],
    dt: float,
) -> Tuple[int, float, float]:
    alive: List[Particle] = []
    avg_field = 0.0
    alignment_sum = 0.0
    alignment_count = 0
    for particle in particles:
        field = _field_vector(particle.pos, sources)
        field_mag = float(np.linalg.norm(field))
        avg_field += field_mag
        particle.prev_pos = particle.pos.copy()
        particle.vel += particle.charge * field * (0.038 * dt)
        speed = float(np.linalg.norm(particle.vel))
        if speed > 0.56:
            particle.vel *= 0.56 / speed
            speed = 0.56
        if speed > 0.02 and field_mag > 0.03:
            expected = (field / field_mag) * float(np.sign(particle.charge) if particle.charge != 0.0 else 1.0)
            alignment = float(np.dot(particle.vel / speed, expected))
            alignment_sum += 0.5 * (alignment + 1.0)
            alignment_count += 1
        particle.pos += particle.vel * dt
        particle.ttl -= dt
        if (
            particle.ttl > 0.0
            and -0.10 <= particle.pos[0] <= 1.10
            and -0.10 <= particle.pos[1] <= 1.10
        ):
            alive.append(particle)
    particles[:] = alive
    if alive:
        avg_field /= len(alive)
    alignment_quality = alignment_sum / max(1, alignment_count)
    return len(alive), avg_field, alignment_quality


def _gradient_background(canvas: np.ndarray, t: float) -> None:
    h, w = canvas.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    canvas[..., 0] = np.clip(18.0 + 22.0 * (1.0 - y) + 12.0 * x, 0, 255).astype(np.uint8)
    canvas[..., 1] = np.clip(10.0 + 14.0 * (1.0 - y) + 8.0 * x, 0, 255).astype(np.uint8)
    canvas[..., 2] = np.clip(16.0 + 18.0 * (1.0 - y), 0, 255).astype(np.uint8)
    overlay = canvas.copy()
    cv2.circle(overlay, (int(w * 0.34), int(h * (0.28 + 0.03 * math.sin(0.8 * t)))), 240, (68, 36, 18), -1, cv2.LINE_AA)
    cv2.circle(overlay, (int(w * 0.76), int(h * (0.72 + 0.03 * math.cos(0.7 * t)))), 320, (18, 24, 58), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0, canvas)


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


def _draw_field_glow(canvas: np.ndarray, sources: Sequence[Dict[str, object]]) -> None:
    xs = np.linspace(0.0, 1.0, FIELD_GLOW_W, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, FIELD_GLOW_H, dtype=np.float32)[:, None]
    ex = np.zeros((FIELD_GLOW_H, FIELD_GLOW_W), dtype=np.float32)
    ey = np.zeros((FIELD_GLOW_H, FIELD_GLOW_W), dtype=np.float32)

    for source in sources:
        center = np.asarray(source["center_norm"], dtype=np.float32)
        charge = float(source["charge"]) * float(source["strength"])
        dx = xs - center[0]
        dy = ys - center[1]
        r2 = dx * dx + dy * dy + 0.0035
        inv = charge / (r2 * np.sqrt(r2))
        ex += inv * dx
        ey += inv * dy

    mag = np.sqrt(ex * ex + ey * ey)
    mag = np.log1p(0.18 * mag)
    mag /= max(1e-5, float(np.max(mag)))

    glow = np.zeros((FIELD_GLOW_H, FIELD_GLOW_W, 3), dtype=np.uint8)
    glow[..., 0] = np.clip(34 + 80 * mag, 0, 255).astype(np.uint8)
    glow[..., 1] = np.clip(20 + 68 * mag, 0, 255).astype(np.uint8)
    glow[..., 2] = np.clip(28 + 118 * mag, 0, 255).astype(np.uint8)
    glow = cv2.resize(glow, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_LINEAR)
    cv2.addWeighted(glow, 0.18, canvas, 0.82, 0.0, canvas)


def _draw_field_arrows(canvas: np.ndarray, sources: Sequence[Dict[str, object]]) -> float:
    total = 0.0
    count = 0
    for row in range(FIELD_ROWS):
        y = 0.12 + row * (0.72 / max(1, FIELD_ROWS - 1))
        for col in range(FIELD_COLS):
            x = 0.10 + col * (0.80 / max(1, FIELD_COLS - 1))
            pos = np.asarray([x, y], dtype=np.float32)
            field = _field_vector(pos, sources)
            mag = float(np.linalg.norm(field))
            total += mag
            count += 1
            if mag < 0.06:
                continue
            direction = field / max(mag, 1e-6)
            start = (int(x * CANVAS_W), int(y * CANVAS_H))
            step = int(9 + 10 * min(1.0, mag * 0.18))
            end = (
                int(start[0] + direction[0] * step),
                int(start[1] + direction[1] * step),
            )
            color = (
                76,
                int(110 + 90 * min(1.0, mag * 0.14)),
                int(170 + 70 * min(1.0, mag * 0.10)),
            )
            cv2.arrowedLine(canvas, start, end, color, 1, cv2.LINE_AA, tipLength=0.35)
    return total / max(1, count)


def _trace_streamline(
    seed: np.ndarray,
    sources: Sequence[Dict[str, object]],
    step_size: float,
    steps: int,
) -> np.ndarray:
    pts: List[Tuple[int, int]] = []
    pos = seed.astype(np.float32).copy()
    for _ in range(steps):
        if pos[0] < 0.02 or pos[0] > 0.98 or pos[1] < 0.02 or pos[1] > 0.98:
            break
        field = _field_vector(pos, sources)
        mag = float(np.linalg.norm(field))
        if mag < 0.035:
            break
        pos = pos + (field / mag) * step_size
        pts.append((int(pos[0] * CANVAS_W), int(pos[1] * CANVAS_H)))
    if len(pts) < 2:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(pts, dtype=np.int32)


def _draw_streamlines(canvas: np.ndarray, sources: Sequence[Dict[str, object]]) -> None:
    for source in sources:
        center = np.asarray(source["center_norm"], dtype=np.float32)
        charge = float(source["charge"])
        base_angle = 0.0 if charge > 0.0 else math.pi / STREAMLINE_SEEDS
        color = (96, 182, 255) if charge > 0.0 else (255, 170, 120)
        for i in range(STREAMLINE_SEEDS):
            angle = base_angle + i * (2.0 * math.pi / STREAMLINE_SEEDS)
            seed = center + 0.065 * np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)
            forward = _trace_streamline(seed, sources, 0.010 if charge > 0.0 else -0.010, 68)
            if forward.size:
                cv2.polylines(canvas, [forward], False, color, 1, cv2.LINE_AA)


def _draw_particles(canvas: np.ndarray, particles: Sequence[Particle], sources: Sequence[Dict[str, object]]) -> None:
    overlay = canvas.copy()
    for particle in particles:
        px0 = int(particle.prev_pos[0] * CANVAS_W)
        py0 = int(particle.prev_pos[1] * CANVAS_H)
        px1 = int(particle.pos[0] * CANVAS_W)
        py1 = int(particle.pos[1] * CANVAS_H)
        if px1 < 0 or py1 < 0 or px1 >= CANVAS_W or py1 >= CANVAS_H:
            continue
        color = (255, 216, 132) if particle.charge > 0 else (120, 220, 255)
        vel = particle.pos - particle.prev_pos
        speed = float(np.linalg.norm(vel))
        field = _field_vector(particle.pos, sources)
        field_mag = float(np.linalg.norm(field))
        if speed > 1e-5 and field_mag > 0.03:
            expected = (field / field_mag) * float(np.sign(particle.charge) if particle.charge != 0.0 else 1.0)
            align = float(np.dot(vel / speed, expected))
            if align < 0.25:
                color = (110, 116, 255)
        cv2.line(overlay, (px0, py0), (px1, py1), color, 1, cv2.LINE_AA)
        cv2.circle(overlay, (px1, py1), 7, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0.0, canvas)
    for particle in particles:
        px = int(particle.pos[0] * CANVAS_W)
        py = int(particle.pos[1] * CANVAS_H)
        if 0 <= px < CANVAS_W and 0 <= py < CANVAS_H:
            color = (255, 216, 132) if particle.charge > 0 else (120, 220, 255)
            vel = particle.pos - particle.prev_pos
            speed = float(np.linalg.norm(vel))
            field = _field_vector(particle.pos, sources)
            field_mag = float(np.linalg.norm(field))
            if speed > 1e-5 and field_mag > 0.03:
                expected = (field / field_mag) * float(np.sign(particle.charge) if particle.charge != 0.0 else 1.0)
                align = float(np.dot(vel / speed, expected))
                if align < 0.25:
                    color = (110, 116, 255)
            cv2.circle(canvas, (px, py), 2, color, -1, cv2.LINE_AA)


def _draw_source_core(canvas: np.ndarray, source: Dict[str, object]) -> None:
    cx, cy = map(int, np.asarray(source["center_px"], dtype=np.float32))
    charge = float(source["charge"])
    color = (92, 186, 255) if charge > 0.0 else (255, 164, 110)
    sign = "+" if charge > 0.0 else "-"
    strength = float(source["strength"])
    aura_r = int(34 + 20 * strength)
    cv2.circle(canvas, (cx, cy), aura_r, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), int(aura_r * 1.55), color, 1, cv2.LINE_AA)
    glow = canvas.copy()
    cv2.circle(glow, (cx, cy), 40, color, -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.16, canvas, 0.84, 0.0, canvas)
    cv2.circle(canvas, (cx, cy), 18, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 28, (38, 44, 62), 1, cv2.LINE_AA)
    cv2.putText(canvas, sign, (cx - 9, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (248, 248, 250), 2, cv2.LINE_AA)


def _draw_source_bridge(canvas: np.ndarray, left_source: Dict[str, object], right_source: Dict[str, object]) -> float:
    p0 = np.asarray(left_source["center_px"], dtype=np.float32)
    p1 = np.asarray(right_source["center_px"], dtype=np.float32)
    sep = float(np.linalg.norm(p1 - p0))
    same_sign = float(left_source["charge"]) * float(right_source["charge"]) > 0.0
    color = (120, 170, 255) if same_sign else (255, 188, 120)
    mid = 0.5 * (p0 + p1)
    lift = np.asarray([0.0, -0.16 * sep], dtype=np.float32)
    curve = np.array([p0, mid + lift, p1], dtype=np.float32)
    path = []
    for t in np.linspace(0.0, 1.0, 24):
        a = (1.0 - t) * (1.0 - t)
        b = 2.0 * (1.0 - t) * t
        c = t * t
        pt = a * curve[0] + b * curve[1] + c * curve[2]
        path.append((int(pt[0]), int(pt[1])))
    cv2.polylines(canvas, [np.asarray(path, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    return sep / float(CANVAS_W)


def _draw_pinch_feedback(canvas: np.ndarray, source: Dict[str, object], pinch_active: bool) -> None:
    if not pinch_active:
        return
    tip = np.asarray(source["index_tip_norm"], dtype=np.float32)
    tip_px = (int(tip[0] * CANVAS_W), int(tip[1] * CANVAS_H))
    center_px = tuple(map(int, np.asarray(source["center_px"], dtype=np.float32)))
    color = (120, 220, 255)
    cv2.line(canvas, center_px, tip_px, color, 2, cv2.LINE_AA)
    for radius in (10, 18, 26):
        cv2.circle(canvas, tip_px, radius, color, 1, cv2.LINE_AA)


def _draw_hand_avatar(canvas: np.ndarray, source: Dict[str, object], active: bool) -> None:
    draw_pts = source["draw_pts"]
    if draw_pts is None:
        return
    charge = float(source["charge"])
    palm_color = (42, 98, 190) if charge > 0.0 else (196, 112, 48)
    line_color = (255, 224, 148) if charge > 0.0 else (255, 196, 120)
    joint_color = (118, 228, 255) if charge > 0.0 else (255, 178, 142)

    palm_poly = np.array([[int(draw_pts[i, 0]), int(draw_pts[i, 1])] for i in PALM_POLY], dtype=np.int32)
    cv2.fillConvexPoly(canvas, palm_poly, palm_color, cv2.LINE_AA)
    for a, b in HAND_CONNECTIONS:
        p0 = (int(draw_pts[a, 0]), int(draw_pts[a, 1]))
        p1 = (int(draw_pts[b, 0]), int(draw_pts[b, 1]))
        cv2.line(canvas, p0, p1, line_color, 2, cv2.LINE_AA)
    for i in range(21):
        px, py = int(draw_pts[i, 0]), int(draw_pts[i, 1])
        radius = 5 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(canvas, (px, py), radius + 2, (36, 36, 40), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius, joint_color, -1, cv2.LINE_AA)

    label = f"{source['label']} {'+' if charge > 0 else '-'}"
    text_color = (220, 232, 248) if active else (150, 162, 182)
    cv2.putText(
        canvas,
        label,
        (int(draw_pts[0, 0]) - 42, int(draw_pts[0, 1]) - 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        text_color,
        1,
        cv2.LINE_AA,
    )
    tip = (int(draw_pts[INDEX_TIP, 0]), int(draw_pts[INDEX_TIP, 1]))
    cv2.circle(canvas, tip, 10, joint_color, 1, cv2.LINE_AA)


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
    field_mean: float,
    particle_count: int,
    avg_particle_field: float,
    alignment_quality: float,
    source_sep: float,
    scene: SceneState,
    fps: float,
    left_active: bool,
    right_active: bool,
) -> None:
    cv2.putText(canvas, "Electromagnetism Field Lab", (352, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.02, (236, 240, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Fists flip hand charge. Arrows show force on a positive test charge. Either pinch injects packets.", (352, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (176, 194, 220), 1, cv2.LINE_AA)

    panel_x = 350
    panel_y = CANVAS_H - 104
    panel_w = 670
    panel_h = 62
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (84, 96, 118), 1, cv2.LINE_AA)

    left_x = panel_x + 22
    mid_x = panel_x + 232
    right_x = panel_x + 430
    top_y = panel_y + 24
    dy = 22
    stat_font = 0.48
    cv2.putText(canvas, f"<|E|> grid  {field_mean:5.2f}", (left_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"particles  {particle_count:d}", (left_x, top_y + dy), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"shots  {scene.injected_count:d}", (mid_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"<|E|> tracers  {avg_particle_field:5.2f}", (mid_x, top_y + dy), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (226, 232, 248), 1, cv2.LINE_AA)
    right_sign = "-" if scene.right_source_negative else "+"
    left_sign = "-" if scene.left_source_negative else "+"
    cv2.putText(canvas, f"L {left_sign}   R {right_sign}", (right_x, top_y), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"match  {100.0 * alignment_quality:4.0f}%   sep {source_sep:4.2f}", (right_x, top_y + dy), cv2.FONT_HERSHEY_SIMPLEX, stat_font, (226, 232, 248), 1, cv2.LINE_AA)

    footer_font = 0.40
    cv2.putText(canvas, f"L:{'ok' if left_active else 'def'} R:{'ok' if right_active else 'def'} fps:{fps:4.1f}", (20, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, footer_font, (150, 170, 198), 1, cv2.LINE_AA)
    cv2.putText(canvas, "[m] mirror  [f] full  [r] clear  [q] quit", (CANVAS_W - 250, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, footer_font, (150, 170, 198), 1, cv2.LINE_AA)


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
    particles = _seed_particles()
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
            extracted: List[HandObservation]
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

            left_hand = _best_hand_by_label(tracked, "left", now, HAND_STALE_S)
            right_hand = _best_hand_by_label(tracked, "right", now, HAND_STALE_S)
            if left_hand is None:
                left_hand = _best_hand_by_side(tracked, left_side=True, now=now, max_age=HAND_STALE_S)
            if right_hand is None:
                right_hand = _best_hand_by_side(tracked, left_side=False, now=now, max_age=HAND_STALE_S)

            left_pinch_active, left_pinch_fired = _update_pinch_latch(left_hand, scene.left_pinch_latched)
            scene.left_pinch_latched = left_pinch_active
            right_pinch_active, right_pinch_fired = _update_pinch_latch(right_hand, scene.right_pinch_latched)
            scene.right_pinch_latched = right_pinch_active

            scene.left_fist_latched, scene.left_last_toggle_ts, scene.left_source_negative = _update_charge_toggle(
                left_hand,
                scene.left_fist_latched,
                scene.left_last_toggle_ts,
                scene.left_source_negative,
                now,
            )
            scene.right_fist_latched, scene.right_last_toggle_ts, scene.right_source_negative = _update_charge_toggle(
                right_hand,
                scene.right_fist_latched,
                scene.right_last_toggle_ts,
                scene.right_source_negative,
                now,
            )

            left_source = _charge_source_from_hand(left_hand, "left", now, source_negative=scene.left_source_negative)
            right_source = _charge_source_from_hand(right_hand, "right", now, source_negative=scene.right_source_negative)
            sources = (left_source, right_source)

            if left_pinch_fired:
                _emit_particles(particles, left_source, sources, count=8, ttl=3.2, speed_scale=1.0, spread_scale=1.0)
                scene.injected_count += 1
            if right_pinch_fired:
                _emit_particles(particles, right_source, sources, count=8, ttl=3.2, speed_scale=1.0, spread_scale=1.0)
                scene.injected_count += 1

            scene.spawn_accumulator += dt * CONTINUOUS_SPAWN_HZ
            while scene.spawn_accumulator >= 1.0:
                _emit_particles(particles, left_source, sources, count=1, ttl=2.6, speed_scale=0.75, spread_scale=0.45)
                _emit_particles(particles, right_source, sources, count=1, ttl=2.6, speed_scale=0.75, spread_scale=0.45)
                scene.spawn_accumulator -= 1.0

            particle_count, avg_particle_field, alignment_quality = _update_particles(particles, sources, dt)

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            _gradient_background(canvas, now)
            _draw_field_glow(canvas, sources)
            field_mean = _draw_field_arrows(canvas, sources)
            _draw_streamlines(canvas, sources)
            _draw_particles(canvas, particles, sources)
            source_sep = _draw_source_bridge(canvas, left_source, right_source)
            _draw_source_core(canvas, left_source)
            _draw_source_core(canvas, right_source)
            _draw_hand_avatar(canvas, left_source, left_hand is not None)
            _draw_hand_avatar(canvas, right_source, right_hand is not None)
            _draw_pinch_feedback(canvas, left_source, left_pinch_active)
            _draw_pinch_feedback(canvas, right_source, right_pinch_active)
            _draw_webcam_inset(canvas, frame, extracted)
            _draw_hud(
                canvas,
                field_mean,
                particle_count,
                avg_particle_field,
                alignment_quality,
                source_sep,
                scene,
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
                scene.left_pinch_latched = False
                scene.right_pinch_latched = False
            if key == ord("r"):
                particles.clear()
                scene.injected_count = 0
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
