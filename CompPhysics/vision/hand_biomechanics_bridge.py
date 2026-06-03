#!/usr/bin/env python3
"""
Two-hand webcam bridge for the C++ hand biomechanics visualizer.

Transport:
- UDP localhost:50515
- CSV packet
- timestamp,left_valid,left_pinched,left_score,(x,y,z)*21,right_valid,right_pinched,right_score,(x,y,z)*21
"""

from __future__ import annotations

import os
import socket
import sys
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


UDP_HOST = "127.0.0.1"
UDP_PORT = 50515
FRAME_UDP_PORT = 50516
WINDOW_NAME = "Hand Biomechanics Bridge"
MODULE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES: Tuple[Path, ...] = (
    MODULE_DIR / "models" / "hand_landmarker.task",
    MODULE_DIR.parent / "DefensiveSys" / "models" / "hand_landmarker.task",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])

CAPTURE_W = 1280
CAPTURE_H = 720
SMOOTH_ALPHA_MIN = 0.18
SMOOTH_ALPHA_MAX = 0.52
ANCHOR_SMOOTH_ALPHA = 0.18
PINCH_CLOSE_RATIO = 0.40
PINCH_RELEASE_RATIO = 0.52
HAND_STALE_S = 0.45
PREVIEW_SEND_HZ = 30.0
PREVIEW_W = 320
PREVIEW_H = 180
PREVIEW_JPEG_QUALITY = 55

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
SLOT_KEYS: Tuple[str, str] = ("slot0", "slot1")


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
    pinched: bool
    last_seen: float


class MultiLandmarkSmoother:
    def __init__(self, alpha_min: float = SMOOTH_ALPHA_MIN, alpha_max: float = SMOOTH_ALPHA_MAX) -> None:
        self.alpha_min = float(np.clip(alpha_min, 0.01, 0.99))
        self.alpha_max = float(np.clip(alpha_max, self.alpha_min, 0.99))
        self._state: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._state.clear()

    def smooth(self, key: str, points: Sequence[Tuple[float, float, float]]) -> np.ndarray:
        current = np.asarray(points, dtype=np.float32)
        prev = self._state.get(key)
        if prev is None or prev.shape != current.shape:
            self._state[key] = current.copy()
        else:
            motion = float(np.mean(np.linalg.norm(current[:, :2] - prev[:, :2], axis=1)))
            motion_t = float(np.clip((motion - 0.0015) / 0.030, 0.0, 1.0))
            alpha = (1.0 - motion_t) * self.alpha_min + motion_t * self.alpha_max
            self._state[key] = (1.0 - alpha) * prev + alpha * current
        return self._state[key].copy()

    def prune(self, active_keys: Sequence[str]) -> None:
        keep = set(active_keys)
        for key in list(self._state.keys()):
            if key not in keep:
                del self._state[key]


class PalmAnchorStabilizer:
    def __init__(self, alpha: float = ANCHOR_SMOOTH_ALPHA) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 0.99))
        self._anchors: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._anchors.clear()

    def stabilize(self, key: str, pose: np.ndarray) -> np.ndarray:
        pose = pose.copy()
        anchor = (
            0.62 * pose[WRIST]
            + 0.19 * pose[INDEX_MCP]
            + 0.19 * pose[PINKY_MCP]
        ).astype(np.float32)
        prev = self._anchors.get(key)
        if prev is None:
            self._anchors[key] = anchor.copy()
            return pose

        stable = (1.0 - self.alpha) * prev + self.alpha * anchor
        self._anchors[key] = stable.astype(np.float32)
        delta = stable - anchor
        pose += delta
        return pose

    def prune(self, active_keys: Sequence[str]) -> None:
        keep = set(active_keys)
        for key in list(self._anchors.keys()):
            if key not in keep:
                del self._anchors[key]


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
    palm = max(1.0e-4, _palm_size(points))
    pinch = np.linalg.norm(points[THUMB_TIP, :2] - points[INDEX_TIP, :2])
    return float(pinch / palm)


def _best_hand_by_label(tracked: Dict[str, TrackedHand], label: str, now: float) -> TrackedHand | None:
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


def _best_hand_by_side(tracked: Dict[str, TrackedHand], now: float, right_side: bool) -> TrackedHand | None:
    candidates = [hand for hand in tracked.values() if (now - hand.last_seen) <= HAND_STALE_S]
    if not candidates:
        return None
    picker = max if right_side else min
    return picker(candidates, key=lambda hand: float(hand.pose[WRIST, 0]))


def _select_left_right_unique(tracked: Dict[str, TrackedHand], now: float) -> Tuple[TrackedHand | None, TrackedHand | None]:
    alive = [(slot, hand) for slot, hand in tracked.items() if (now - hand.last_seen) <= HAND_STALE_S]
    if not alive:
        return None, None

    used_slots = set()
    left_item: Tuple[str, TrackedHand] | None = None
    right_item: Tuple[str, TrackedHand] | None = None

    left_labeled = [item for item in alive if item[1].label.lower() == "left"]
    right_labeled = [item for item in alive if item[1].label.lower() == "right"]
    if left_labeled:
        left_item = max(left_labeled, key=lambda item: item[1].score)
        used_slots.add(left_item[0])
    if right_labeled:
        candidates = [item for item in right_labeled if item[0] not in used_slots]
        if candidates:
            right_item = max(candidates, key=lambda item: item[1].score)
            used_slots.add(right_item[0])

    remaining = [item for item in alive if item[0] not in used_slots]
    if left_item is None and remaining:
        left_item = min(remaining, key=lambda item: float(item[1].pose[WRIST, 0]))
        used_slots.add(left_item[0])
        remaining = [item for item in alive if item[0] not in used_slots]
    if right_item is None and remaining:
        right_item = max(remaining, key=lambda item: float(item[1].pose[WRIST, 0]))

    return (left_item[1] if left_item is not None else None, right_item[1] if right_item is not None else None)


def _append_hand_packet(parts: List[str], hand: TrackedHand | None) -> None:
    if hand is None:
        parts.extend(("0", "0", "0.0000"))
        for _ in range(21):
            parts.extend(("0.000000", "0.000000", "0.000000"))
        return

    parts.extend((
        "1",
        "1" if hand.pinched else "0",
        f"{hand.score:.4f}",
    ))
    for x, y, z in hand.pose:
        parts.extend((f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"))


def _send_packet(sock: socket.socket, left: TrackedHand | None, right: TrackedHand | None, now: float) -> None:
    parts = [f"{now:.3f}"]
    _append_hand_packet(parts, left)
    _append_hand_packet(parts, right)
    sock.sendto((",".join(parts) + "\n").encode("ascii"), (UDP_HOST, UDP_PORT))


def _send_preview_frame(sock: socket.socket, frame: np.ndarray) -> None:
    preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY],
    )
    if not ok:
        return
    data = encoded.tobytes()
    if len(data) >= 65000:
        return
    try:
        sock.sendto(data, (UDP_HOST, FRAME_UDP_PORT))
    except OSError:
        return


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    mirror: bool,
    left: TrackedHand | None,
    right: TrackedHand | None,
    tracked_count: int,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 78), (w, h), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
    cv2.line(frame, (0, h - 78), (w, h - 78), (96, 103, 122), 1, cv2.LINE_AA)

    if left is None:
        left_s = "L:none"
    else:
        left_s = f"L:{left.score:.2f} p{int(left.pinched)} r{_pinch_ratio(left.pose):.2f}"
    if right is None:
        right_s = "R:none"
    else:
        right_s = f"R:{right.score:.2f} p{int(right.pinched)} r{_pinch_ratio(right.pose):.2f}"

    status = f"udp:{UDP_HOST}:{UDP_PORT}  tracked:{tracked_count}  mirror:{'on' if mirror else 'off'}  fps:{fps:.1f}"
    cv2.putText(frame, status, (14, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 228, 250), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{left_s}  {right_s}", (14, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 190, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "keys: [m] mirror  [r] reset  [q] quit", (w - 270, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 190, 220), 1, cv2.LINE_AA)


def _draw_tracked_overlay(frame: np.ndarray, left: TrackedHand | None, right: TrackedHand | None) -> None:
    h, w = frame.shape[:2]
    for hand, color in ((left, (136, 255, 196)), (right, (72, 220, 255))):
        if hand is None:
            continue
        pts: List[Tuple[int, int]] = []
        for x, y, _ in hand.pose:
            px = int(np.clip(x * w, 0, w - 1))
            py = int(np.clip(y * h, 0, h - 1))
            pts.append((px, py))
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for idx, (px, py) in enumerate(pts):
            radius = 5 if idx in (4, 8, 12, 16, 20) else 4
            cv2.circle(frame, (px, py), radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), radius + 2, color, 1, cv2.LINE_AA)


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
    drawer = None
    conn_style = None
    lmk_style = None
    task_timestamp_ms = 0

    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        if not MODEL_PATH.exists():
            print(f"Error: missing hand landmark model at {MODEL_PATH}")
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
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        tracker_tasks = vision.HandLandmarker.create_from_options(options)
        use_tasks = True
    elif hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        tracker_solution = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        drawer = mp.solutions.drawing_utils
        conn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()
        lmk_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
    else:
        print("Unsupported mediapipe build: expected `solutions` or `tasks.vision` APIs.")
        cap.release()
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    mirror = True
    tracked: Dict[str, TrackedHand] = {}
    smoother = MultiLandmarkSmoother()
    anchor_stabilizer = PalmAnchorStabilizer()
    pinch_latched: Dict[str, bool] = {slot: False for slot in SLOT_KEYS}
    last_t = time.perf_counter()
    fps = 60.0
    last_preview_send_ts = 0.0

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
            dt = max(1.0e-4, min(0.05, now - last_t))
            last_t = now
            fps = 0.92 * fps + 0.08 * (1.0 / dt)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if use_tasks:
                task_timestamp_ms = max(task_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = tracker_tasks.detect_for_video(mp_image, task_timestamp_ms)
                extracted = _extract_task_hands(result, mirror=mirror)
            else:
                result = tracker_solution.process(rgb)
                extracted = _extract_solution_hands(result, mirror=mirror)
                for lm in (result.multi_hand_landmarks or []):
                    drawer.draw_landmarks(
                        frame,
                        lm,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        lmk_style,
                        conn_style,
                    )

            assignments = _assign_hands_to_slots(extracted, tracked)
            active_keys: List[str] = []
            for key, hand in assignments:
                pose = smoother.smooth(key, hand.landmarks)
                pose = anchor_stabilizer.stabilize(key, pose)
                ratio = _pinch_ratio(pose)
                if key in pinch_latched:
                    if pinch_latched[key]:
                        pinch = ratio < PINCH_RELEASE_RATIO
                    else:
                        pinch = ratio < PINCH_CLOSE_RATIO
                    pinch_latched[key] = pinch
                else:
                    pinch = ratio < PINCH_CLOSE_RATIO

                tracked[key] = TrackedHand(
                    pose=pose,
                    label=hand.label,
                    score=hand.score,
                    pinched=pinch,
                    last_seen=now,
                )
                active_keys.append(key)

            for key in list(tracked.keys()):
                if key not in active_keys and (now - tracked[key].last_seen) > HAND_STALE_S:
                    del tracked[key]
            smoother.prune(tracked.keys())
            anchor_stabilizer.prune(tracked.keys())

            left_hand, right_hand = _select_left_right_unique(tracked, now)

            _draw_tracked_overlay(frame, left_hand, right_hand)
            _send_packet(sock, left_hand, right_hand, now)
            _draw_hud(frame, fps, mirror, left_hand, right_hand, len(extracted))
            if (now - last_preview_send_ts) >= (1.0 / PREVIEW_SEND_HZ):
                _send_preview_frame(sock, frame)
                last_preview_send_ts = now
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("r"):
                tracked.clear()
                smoother.reset()
                anchor_stabilizer.reset()
                pinch_latched = {slot: False for slot in SLOT_KEYS}
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
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
