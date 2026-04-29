#!/usr/bin/env python3
"""
Standalone floating hand avatar base scene.

This is the clean baseline version for future projects: webcam tracking,
floating hand avatars, a light HUD, and no physics-specific behavior.
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


WINDOW_NAME = "Floating Hand Avatar"
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
PINKY_MCP = 17

SLOT_KEYS: Tuple[str, str] = ("slot0", "slot1")
HAND_STALE_S = 0.45
SMOOTH_ALPHA = 0.32
PINCH_CLOSE_RATIO = 0.36
PINCH_RELEASE_RATIO = 0.48

HAND_REACH_X_GAIN = 1.02
HAND_REACH_Y_GAIN = 1.06
HAND_MIN_PALM_NORM = 0.035
HAND_MAX_PALM_NORM = 0.17
HAND_NORMALIZED_SCALE = 82.0

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
    candidates = [hand for hand in tracked.values() if (now - hand.last_seen) <= max_age]
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


def _gradient_background(canvas: np.ndarray, t: float) -> None:
    h, w = canvas.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    canvas[..., 0] = np.clip(20.0 + 28.0 * (1.0 - y) + 8.0 * x, 0, 255).astype(np.uint8)
    canvas[..., 1] = np.clip(12.0 + 18.0 * (1.0 - y) + 10.0 * x, 0, 255).astype(np.uint8)
    canvas[..., 2] = np.clip(26.0 + 18.0 * (1.0 - y), 0, 255).astype(np.uint8)

    overlay = canvas.copy()
    cv2.circle(overlay, (int(w * 0.36), int(h * (0.28 + 0.03 * math.sin(0.8 * t)))), 240, (80, 42, 26), -1, cv2.LINE_AA)
    cv2.circle(overlay, (int(w * 0.70), int(h * (0.68 + 0.04 * math.cos(0.9 * t)))), 320, (24, 28, 62), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0, canvas)


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
    bob = 12.0 * math.sin(1.35 * t)
    cx = 0.5 * canvas_w + tx
    cy = 0.5 * canvas_h + ty + bob
    out = np.zeros((21, 2), dtype=np.float32)
    out[:, 0] = cx + rel[:, 0] * HAND_NORMALIZED_SCALE
    out[:, 1] = cy + rel[:, 1] * HAND_NORMALIZED_SCALE
    return out


def _draw_avatar(canvas: np.ndarray, hand: TrackedHand, now: float, fade: float) -> None:
    draw_pts = _hand_canvas_points(hand.pose, canvas.shape[1], canvas.shape[0], now)
    if fade <= 0.001:
        return

    palm_poly = np.array([[int(draw_pts[i, 0]), int(draw_pts[i, 1])] for i in PALM_POLY], dtype=np.int32)
    glow = canvas.copy()
    cv2.fillConvexPoly(glow, palm_poly, (58, 120, 222), cv2.LINE_AA)
    cv2.addWeighted(glow, 0.12 + 0.08 * fade, canvas, 0.88 - 0.08 * fade, 0.0, canvas)
    cv2.fillConvexPoly(canvas, palm_poly, (42, 92, 185), cv2.LINE_AA)

    for a, b in HAND_CONNECTIONS:
        p0 = (int(draw_pts[a, 0]), int(draw_pts[a, 1]))
        p1 = (int(draw_pts[b, 0]), int(draw_pts[b, 1]))
        cv2.line(canvas, p0, p1, (255, 214, 120), 2, cv2.LINE_AA)

    for i in range(21):
        px, py = int(draw_pts[i, 0]), int(draw_pts[i, 1])
        r = 5 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(canvas, (px, py), r + 2, (34, 34, 38), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), r, (112, 226, 255), -1, cv2.LINE_AA)

    tip = (int(draw_pts[INDEX_TIP, 0]), int(draw_pts[INDEX_TIP, 1]))
    ratio = _pinch_ratio(hand.pose)
    tip_color = (118, 246, 182) if ratio < PINCH_CLOSE_RATIO else (126, 216, 255)
    cv2.circle(canvas, tip, 10, tip_color, 1, cv2.LINE_AA)

    cx = int(draw_pts[0, 0])
    cy = int(draw_pts[0, 1])
    cv2.putText(
        canvas,
        f"{hand.label} {hand.score:.2f}",
        (cx - 56, cy - 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 232, 250),
        1,
        cv2.LINE_AA,
    )


def _draw_webcam_inset(canvas: np.ndarray, frame: np.ndarray, hands: List[HandObservation]) -> None:
    margin = 18
    inset_w = 280
    inset_h = 168
    preview = frame.copy()
    ph, pw = preview.shape[:2]

    for hand in hands:
        color = (72, 232, 255) if hand.label.lower() == "right" else (136, 255, 156)
        pts = []
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
    cv2.addWeighted(panel, 0.70, canvas, 0.30, 0.0, canvas)
    cv2.rectangle(canvas, (margin - 8, margin - 8), (margin + inset_w + 8, margin + inset_h + 34), (88, 96, 118), 1, cv2.LINE_AA)
    canvas[margin: margin + inset_h, margin: margin + inset_w] = preview
    cv2.putText(canvas, "webcam tracking", (margin + 8, margin + inset_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (216, 226, 244), 1, cv2.LINE_AA)


def _draw_hud(
    canvas: np.ndarray,
    fps: float,
    mirror: bool,
    left: TrackedHand | None,
    right: TrackedHand | None,
) -> None:
    cv2.putText(canvas, "Floating Hand Avatar", (348, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.02, (236, 240, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Baseline scene for future experiments: tracking, avatars, and clean motion only.", (348, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (176, 194, 220), 1, cv2.LINE_AA)

    panel_x = 348
    panel_y = CANVAS_H - 98
    panel_w = 560
    panel_h = 58
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (84, 96, 118), 1, cv2.LINE_AA)

    mirror_s = "on" if mirror else "off"
    left_s = "pinch" if (left is not None and _pinch_ratio(left.pose) < PINCH_CLOSE_RATIO) else "open"
    right_s = "pinch" if (right is not None and _pinch_ratio(right.pose) < PINCH_CLOSE_RATIO) else "open"
    cv2.putText(canvas, f"mirror {mirror_s}", (panel_x + 18, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"left {left_s}", (panel_x + 18, panel_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"right {right_s}", (panel_x + 180, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"fps {fps:4.1f}", (panel_x + 180, panel_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (226, 232, 248), 1, cv2.LINE_AA)
    cv2.putText(canvas, "keys: [m] mirror  [f] full  [r] reset  [q] quit", (panel_x + 320, panel_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (166, 184, 210), 1, cv2.LINE_AA)


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

            left = _best_hand_by_label(tracked, "left", now, HAND_STALE_S)
            right = _best_hand_by_label(tracked, "right", now, HAND_STALE_S)
            if left is None:
                left = _best_hand_by_side(tracked, left_side=True, now=now, max_age=HAND_STALE_S)
            if right is None:
                right = _best_hand_by_side(tracked, left_side=False, now=now, max_age=HAND_STALE_S)

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            _gradient_background(canvas, now)
            _draw_webcam_inset(canvas, frame, extracted)

            stale_keys: List[str] = []
            for key, hand in tracked.items():
                age = now - hand.last_seen
                if age > 1.3:
                    stale_keys.append(key)
                    continue
                fade = float(np.clip(1.0 - age / 1.0, 0.0, 1.0))
                _draw_avatar(canvas, hand, now, fade)
            for key in stale_keys:
                del tracked[key]

            _draw_hud(canvas, fps, mirror, left, right)

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
            if key == ord("r"):
                smoother.reset()
                tracked.clear()
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
