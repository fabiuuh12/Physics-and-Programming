"""
DefensiveSys hand-tracking bridge (Python).

This app tracks both hands and sends control data over UDP for a C++ 3D sim:
- Left hand index fingertip -> aim (normalized x/y)
- Right hand pinch          -> fire trigger

UDP packet format (ASCII CSV):
timestamp,left_valid,left_x,left_y,right_valid,right_pinch
"""

from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES: Tuple[Path, ...] = (
    MODULE_DIR / "models" / "hand_landmarker.task",
    MODULE_DIR.parent / "vision" / "models" / "hand_landmarker.task",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])
WINDOW_NAME = "DefensiveSys Hand Tracking Bridge"

UDP_HOST = "127.0.0.1"
UDP_PORT = 50505

SLOT_KEYS: Tuple[str, str] = ("slot0", "slot1")
CANVAS_W = 1280
CANVAS_H = 820
CAPTURE_W = 960
CAPTURE_H = 540
DETECT_W = 640
DETECT_H = 360
DETECTION_STRIDE = 2

SWAP_LABELS_ON_MIRROR = True
HAND_REACH_X_GAIN = 1.04
HAND_REACH_Y_GAIN = 1.08
HAND_MIN_PALM_NORM = 0.035
HAND_MAX_PALM_NORM = 0.17
HAND_NORMALIZED_SCALE = 78.0

HAND_STALE_S = 0.45
PINCH_CLOSE_RATIO = 0.39
PINCH_RELEASE_RATIO = 0.54

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
PALM_POLY = (0, 1, 5, 9, 13, 17)
_BG_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


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
    def __init__(self, alpha: float = 0.33) -> None:
        self.alpha = alpha
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
        stale = [k for k in self._state if k not in keep]
        for k in stale:
            del self._state[k]


def _extract_solution_hands(result: object) -> List[HandObservation]:
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
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        out.append(HandObservation(label=label, score=score, landmarks=landmarks))
    return out


def _extract_task_hands(result: object) -> List[HandObservation]:
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
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        out.append(HandObservation(label=label, score=score, landmarks=landmarks))
    return out


def _normalize_label_for_display(label: str, mirror: bool) -> str:
    if not SWAP_LABELS_ON_MIRROR or not mirror:
        return label
    ll = label.lower()
    if ll == "left":
        return "Right"
    if ll == "right":
        return "Left"
    return label


def _normalize_hands_for_display(hands: List[HandObservation], mirror: bool) -> List[HandObservation]:
    out: List[HandObservation] = []
    for h in hands:
        out.append(
            HandObservation(
                label=_normalize_label_for_display(h.label, mirror),
                score=h.score,
                landmarks=h.landmarks,
            )
        )
    return out


def _wrist_xy(hand: HandObservation) -> np.ndarray:
    return np.asarray(hand.landmarks[0][:2], dtype=np.float32)


def _slot_wrist_xy(tracked: Dict[str, TrackedHand], slot: str) -> np.ndarray | None:
    item = tracked.get(slot)
    if item is None:
        return None
    return item.pose[0, :2].astype(np.float32)


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
            wx = _wrist_xy(hand)
            dists: List[Tuple[float, str]] = []
            for slot in existing:
                sw = _slot_wrist_xy(tracked, slot)
                if sw is not None:
                    dists.append((float(np.linalg.norm(wx - sw)), slot))
            if dists:
                return [(min(dists, key=lambda t: t[0])[1], hand)]
        for slot in SLOT_KEYS:
            if slot not in tracked:
                return [(slot, hand)]
        return [(SLOT_KEYS[0], hand)]

    h0, h1 = hands[0], hands[1]
    w0 = _wrist_xy(h0)
    w1 = _wrist_xy(h1)

    if all(slot in tracked for slot in SLOT_KEYS):
        s0 = _slot_wrist_xy(tracked, SLOT_KEYS[0])
        s1 = _slot_wrist_xy(tracked, SLOT_KEYS[1])
        if s0 is not None and s1 is not None:
            c_direct = float(np.linalg.norm(w0 - s0) + np.linalg.norm(w1 - s1))
            c_cross = float(np.linalg.norm(w1 - s0) + np.linalg.norm(w0 - s1))
            if c_direct <= c_cross:
                return [(SLOT_KEYS[0], h0), (SLOT_KEYS[1], h1)]
            return [(SLOT_KEYS[0], h1), (SLOT_KEYS[1], h0)]

    if len(existing) == 1:
        used = existing[0]
        free = SLOT_KEYS[1] if used == SLOT_KEYS[0] else SLOT_KEYS[0]
        sw = _slot_wrist_xy(tracked, used)
        if sw is not None:
            d0 = float(np.linalg.norm(w0 - sw))
            d1 = float(np.linalg.norm(w1 - sw))
            if d0 <= d1:
                return [(used, h0), (free, h1)]
            return [(used, h1), (free, h0)]
        return [(used, h0), (free, h1)]

    if w0[0] <= w1[0]:
        return [(SLOT_KEYS[0], h0), (SLOT_KEYS[1], h1)]
    return [(SLOT_KEYS[0], h1), (SLOT_KEYS[1], h0)]


def _gradient_background(height: int, width: int, t: float) -> np.ndarray:
    key = (height, width)
    base = _BG_CACHE.get(key)
    if base is None:
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        b = np.clip(24.0 + 22.0 * (1.0 - y) + 8.0 * x, 0, 255)
        g = np.clip(16.0 + 18.0 * (1.0 - y) + 7.0 * x, 0, 255)
        r = np.clip(30.0 + 20.0 * (1.0 - y) + 8.0 * x, 0, 255)
        base = np.dstack((b, g, r)).astype(np.uint8)
        _BG_CACHE[key] = base
    bg = base.copy()
    cx = int(width * (0.5 + 0.08 * math.sin(0.8 * t)))
    cy = int(height * (0.45 + 0.07 * math.cos(0.9 * t)))
    overlay = bg.copy()
    cv2.circle(overlay, (cx, cy), int(0.19 * min(width, height)), (76, 58, 42), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, bg, 0.75, 0.0, bg)
    return bg


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
    bob = 12.0 * math.sin(1.6 * t)
    cx = 0.5 * canvas_w + tx
    cy = 0.50 * canvas_h + ty + bob
    out = np.zeros((21, 2), dtype=np.float32)
    out[:, 0] = cx + rel[:, 0] * HAND_NORMALIZED_SCALE
    out[:, 1] = cy + rel[:, 1] * HAND_NORMALIZED_SCALE
    return out


def _pinch_ratio(points: np.ndarray) -> float:
    palm = (
        np.linalg.norm(points[0, :2] - points[5, :2])
        + np.linalg.norm(points[0, :2] - points[17, :2])
        + np.linalg.norm(points[5, :2] - points[17, :2])
    ) / 3.0
    pinch_dist = np.linalg.norm(points[4, :2] - points[8, :2])
    return float(pinch_dist / max(1e-4, palm))


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


def _send_bridge_packet(
    sock: socket.socket,
    left: TrackedHand | None,
    right: TrackedHand | None,
    right_pinched: bool,
    now: float,
) -> None:
    left_valid = 1 if left is not None else 0
    right_valid = 1 if right is not None else 0
    lx = 0.5
    ly = 0.5
    if left is not None:
        lx = float(np.clip(left.pose[8, 0], 0.0, 1.0))
        ly = float(np.clip(left.pose[8, 1], 0.0, 1.0))
    msg = f"{now:.3f},{left_valid},{lx:.5f},{ly:.5f},{right_valid},{1 if right_pinched else 0}\n"
    sock.sendto(msg.encode("ascii"), (UDP_HOST, UDP_PORT))


def _draw_avatar(canvas: np.ndarray, hand: TrackedHand, now: float, fade: float) -> None:
    draw_pts = _hand_canvas_points(hand.pose, canvas.shape[1], canvas.shape[0], now)
    if fade <= 0.001:
        return
    palm_poly = np.array([[int(draw_pts[i, 0]), int(draw_pts[i, 1])] for i in PALM_POLY], dtype=np.int32)
    cv2.fillConvexPoly(canvas, palm_poly, (42, 92, 185), cv2.LINE_AA)
    for a, b in HAND_CONNECTIONS:
        p0 = (int(draw_pts[a, 0]), int(draw_pts[a, 1]))
        p1 = (int(draw_pts[b, 0]), int(draw_pts[b, 1]))
        cv2.line(canvas, p0, p1, (255, 214, 120), 2, cv2.LINE_AA)
    for i in range(21):
        px, py = int(draw_pts[i, 0]), int(draw_pts[i, 1])
        r = 5 if i in (4, 8, 12, 16, 20) else 4
        cv2.circle(canvas, (px, py), r + 2, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), r, (112, 226, 255), -1, cv2.LINE_AA)
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
    margin = 16
    inset_w = 320
    inset_h = 192
    preview = frame.copy()
    h, w = preview.shape[:2]
    for hand in hands:
        color = (60, 225, 255) if hand.label.lower() == "right" else (130, 255, 130)
        pts = []
        for x, y, _ in hand.landmarks:
            px = int(np.clip(x * w, 0, w - 1))
            py = int(np.clip(y * h, 0, h - 1))
            pts.append((px, py))
        for a, b in HAND_CONNECTIONS:
            cv2.line(preview, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(preview, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
    preview = cv2.resize(preview, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    panel = canvas.copy()
    cv2.rectangle(panel, (margin - 6, margin - 6), (margin + inset_w + 6, margin + inset_h + 30), (10, 14, 22), -1)
    cv2.addWeighted(panel, 0.65, canvas, 0.35, 0.0, canvas)
    cv2.rectangle(canvas, (margin - 6, margin - 6), (margin + inset_w + 6, margin + inset_h + 30), (86, 98, 122), 1, cv2.LINE_AA)
    canvas[margin : margin + inset_h, margin : margin + inset_w] = preview
    cv2.putText(canvas, "webcam", (margin + 7, margin + inset_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (214, 227, 246), 1, cv2.LINE_AA)


def _draw_hud(
    canvas: np.ndarray,
    fps: float,
    mirror: bool,
    live_hands: int,
    tracked_hands: int,
    left: TrackedHand | None,
    right: TrackedHand | None,
    right_pinched: bool,
) -> None:
    h, w = canvas.shape[:2]
    panel_top = h - 74
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, panel_top), (w, h), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0.0, canvas)
    cv2.line(canvas, (0, panel_top), (w, panel_top), (75, 85, 104), 1, cv2.LINE_AA)

    mirror_s = "mirror:on" if mirror else "mirror:off"
    left_s = "ok" if left is not None else "none"
    right_s = "ok" if right is not None else "none"
    pinch_s = "down" if right_pinched else "up"
    cv2.putText(
        canvas,
        (
            f"HAND BRIDGE  |  UDP:{UDP_HOST}:{UDP_PORT}  |  left:{left_s} right:{right_s} pinch:{pinch_s}"
            f"  |  live:{live_hands} tracked:{tracked_hands}  |  {mirror_s}  |  fps:{fps:0.1f}"
        ),
        (18, h - 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 226, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "left hand index controls aim; right pinch controls fire | keys: [m] mirror [s] reset [q] quit",
        (18, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (172, 194, 224),
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

    if hasattr(mp, "solutions"):
        hands = mp.solutions.hands
        tracker_solution = hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
    elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        if not MODEL_PATH.exists():
            print("Error: missing hand landmark model. Checked:")
            for cand in MODEL_CANDIDATES:
                print(f"  - {cand}")
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
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        tracker_tasks = vision.HandLandmarker.create_from_options(options)
        use_tasks = True
    else:
        print("Unsupported mediapipe build: expected `solutions` or `tasks.vision` APIs.")
        cap.release()
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    mirror = True
    smoother = MultiLandmarkSmoother(alpha=0.32)
    tracked: Dict[str, TrackedHand] = {}
    right_pinched = False
    last_t = time.perf_counter()
    fps = 60.0
    frame_idx = 0
    last_extracted: List[HandObservation] = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

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

            frame_idx += 1
            if (frame_idx % DETECTION_STRIDE == 0) or (not last_extracted):
                detect_frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
                if use_tasks:
                    task_timestamp_ms = max(task_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = tracker_tasks.detect_for_video(mp_image, task_timestamp_ms)
                    extracted = _extract_task_hands(result)
                else:
                    result = tracker_solution.process(rgb)
                    extracted = _extract_solution_hands(result)
                last_extracted = extracted
            else:
                extracted = last_extracted

            extracted = _normalize_hands_for_display(extracted, mirror=mirror)
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
            smoother.prune(active_keys)

            left = _best_hand_by_label(tracked, "left", now, HAND_STALE_S)
            right = _best_hand_by_label(tracked, "right", now, HAND_STALE_S)
            if right is None:
                right_pinched = False
            else:
                ratio = _pinch_ratio(right.pose)
                if right_pinched:
                    right_pinched = ratio < PINCH_RELEASE_RATIO
                else:
                    right_pinched = ratio < PINCH_CLOSE_RATIO

            _send_bridge_packet(sock, left, right, right_pinched, now)

            canvas = _gradient_background(CANVAS_H, CANVAS_W, now)
            _draw_webcam_inset(canvas, frame, extracted)

            stale_keys = []
            drawn = 0
            for key, hand in tracked.items():
                age = now - hand.last_seen
                if age > 1.4:
                    stale_keys.append(key)
                    continue
                fade = float(np.clip(1.0 - age / 1.1, 0.0, 1.0))
                _draw_avatar(canvas, hand, now, fade)
                if fade > 0.01:
                    drawn += 1
            for key in stale_keys:
                del tracked[key]

            if left is not None:
                draw_pts = _hand_canvas_points(left.pose, CANVAS_W, CANVAS_H, now)
                tx = int(draw_pts[8, 0])
                ty = int(draw_pts[8, 1])
                cv2.circle(canvas, (tx, ty), 16, (118, 246, 182), 1, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 6, (118, 246, 182), 1, cv2.LINE_AA)

            _draw_hud(canvas, fps, mirror, len(extracted), drawn, left, right, right_pinched)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("m"), ord("s")):
                if key == ord("m"):
                    mirror = not mirror
                smoother.reset()
                tracked.clear()
                last_extracted = []
                right_pinched = False
    finally:
        if tracker_solution is not None:
            tracker_solution.close()
        if tracker_tasks is not None:
            tracker_tasks.close()
        sock.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
