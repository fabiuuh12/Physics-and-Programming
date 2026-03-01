#!/usr/bin/env python3
"""
Two-hand webcam bridge for the Vision scene.

Bridge transport:
- UDP localhost:50505
- ASCII CSV
- Backward-compatible first 6 fields:
  timestamp,left_valid,left_x,left_y,right_valid,right_pinch
- Extended fields:
  right_x,right_y,left_pinch
"""

from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


UDP_HOST = "127.0.0.1"
UDP_PORT = 50505
WINDOW_NAME = "Vision Two-Hand Bridge"
MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODULE_DIR / "models" / "hand_landmarker.task"

SMOOTH_ALPHA = 0.34
PINCH_CLOSE_RATIO = 0.40
PINCH_RELEASE_RATIO = 0.54
SWAP_LABELS_ON_MIRROR = True

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17


@dataclass
class HandObservation:
    label: str
    score: float
    landmarks: List[Tuple[float, float, float]]


@dataclass
class HandState:
    valid: bool = False
    x: float = 0.5
    y: float = 0.5
    pinched: bool = False


def _normalize_label(label: str, mirror: bool) -> str:
    if not mirror or not SWAP_LABELS_ON_MIRROR:
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

    for idx, lms in enumerate(result.multi_hand_landmarks):
        label = "Unknown"
        score = 0.0
        if result.multi_handedness and idx < len(result.multi_handedness):
            c = result.multi_handedness[idx].classification[0]
            label = _normalize_label(c.label or "Unknown", mirror)
            score = float(c.score)
        points = [(lm.x, lm.y, lm.z) for lm in lms.landmark]
        out.append(HandObservation(label=label, score=score, landmarks=points))
    return out


def _extract_task_hands(result: object, mirror: bool) -> List[HandObservation]:
    out: List[HandObservation] = []
    if not getattr(result, "hand_landmarks", None):
        return out

    for idx, hand_landmarks in enumerate(result.hand_landmarks):
        label = "Unknown"
        score = 0.0
        if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
            c = result.handedness[idx][0]
            raw_label = (
                getattr(c, "category_name", None)
                or getattr(c, "display_name", None)
                or "Unknown"
            )
            label = _normalize_label(raw_label, mirror)
            score = float(getattr(c, "score", 0.0))
        points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        out.append(HandObservation(label=label, score=score, landmarks=points))
    return out


def _palm_size(hand: HandObservation) -> float:
    p = np.asarray(hand.landmarks, dtype=np.float32)
    a = np.linalg.norm(p[WRIST, :2] - p[INDEX_MCP, :2])
    b = np.linalg.norm(p[WRIST, :2] - p[PINKY_MCP, :2])
    c = np.linalg.norm(p[INDEX_MCP, :2] - p[PINKY_MCP, :2])
    return float(max(1e-4, (a + b + c) / 3.0))


def _pinch_ratio(hand: HandObservation) -> float:
    p = np.asarray(hand.landmarks, dtype=np.float32)
    d = np.linalg.norm(p[THUMB_TIP, :2] - p[INDEX_TIP, :2])
    return float(d / _palm_size(hand))


def _pick_best_by_label(hands: List[HandObservation], label: str) -> HandObservation | None:
    want = label.lower()
    best: HandObservation | None = None
    best_score = -1.0
    for hand in hands:
        if hand.label.lower() != want:
            continue
        if hand.score > best_score:
            best = hand
            best_score = hand.score
    return best


def _smooth_xy(prev: Tuple[float, float], curr: Tuple[float, float], alpha: float) -> Tuple[float, float]:
    px, py = prev
    cx, cy = curr
    return ((1.0 - alpha) * px + alpha * cx, (1.0 - alpha) * py + alpha * cy)


def _update_hand_state(
    prev: HandState,
    hand: HandObservation | None,
    smooth_prev: Tuple[float, float],
) -> Tuple[HandState, Tuple[float, float]]:
    if hand is None:
        return HandState(valid=False, x=smooth_prev[0], y=smooth_prev[1], pinched=False), smooth_prev

    x = float(np.clip(hand.landmarks[INDEX_TIP][0], 0.0, 1.0))
    y = float(np.clip(hand.landmarks[INDEX_TIP][1], 0.0, 1.0))
    sx, sy = _smooth_xy(smooth_prev, (x, y), SMOOTH_ALPHA)
    ratio = _pinch_ratio(hand)
    pinched = prev.pinched
    if pinched:
        pinched = ratio < PINCH_RELEASE_RATIO
    else:
        pinched = ratio < PINCH_CLOSE_RATIO

    return HandState(valid=True, x=sx, y=sy, pinched=pinched), (sx, sy)


def _send_bridge_packet(sock: socket.socket, left: HandState, right: HandState, now: float) -> None:
    msg = (
        f"{now:.3f},"
        f"{1 if left.valid else 0},{left.x:.5f},{left.y:.5f},"
        f"{1 if right.valid else 0},{1 if right.pinched else 0},"
        f"{right.x:.5f},{right.y:.5f},"
        f"{1 if left.pinched else 0}\n"
    )
    sock.sendto(msg.encode("ascii"), (UDP_HOST, UDP_PORT))


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    mirror: bool,
    left: HandState,
    right: HandState,
    hand_count: int,
) -> None:
    h, w = frame.shape[:2]
    panel_h = 72
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (18, 20, 26), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)
    cv2.line(frame, (0, h - panel_h), (w, h - panel_h), (96, 103, 122), 1, cv2.LINE_AA)

    s = (
        f"udp:{UDP_HOST}:{UDP_PORT}  hands:{hand_count}  mirror:{'on' if mirror else 'off'}  fps:{fps:.1f}"
        f"  left:{'ok' if left.valid else 'none'}({left.x:.2f},{left.y:.2f})"
        f"  right:{'ok' if right.valid else 'none'}({right.x:.2f},{right.y:.2f})"
        f"  pinch L:{int(left.pinched)} R:{int(right.pinched)}"
    )
    cv2.putText(frame, s, (14, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 228, 250), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "keys: [m] mirror  [q] quit",
        (14, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 190, 220),
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (camera index 0).")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    use_tasks = False
    hands = None
    hand_landmarker = None
    drawer = None
    conn_style = None
    lmk_style = None
    task_timestamp_ms = 0

    if hasattr(mp, "solutions"):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        drawer = mp.solutions.drawing_utils
        conn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()
        lmk_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
    elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
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
            min_hand_detection_confidence=0.50,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
        use_tasks = True
    else:
        print("Error: unsupported mediapipe build (expected solutions or tasks.vision).")
        cap.release()
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)

    mirror = True
    left = HandState()
    right = HandState()
    left_smooth = (0.5, 0.5)
    right_smooth = (0.5, 0.5)

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
            if use_tasks:
                task_timestamp_ms = max(task_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hand_landmarker.detect_for_video(mp_image, task_timestamp_ms)
                extracted = _extract_task_hands(result, mirror)
            else:
                result = hands.process(rgb)
                extracted = _extract_solution_hands(result, mirror)
                for lm in (result.multi_hand_landmarks or []):
                    drawer.draw_landmarks(
                        frame,
                        lm,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        lmk_style,
                        conn_style,
                    )

            left_obs = _pick_best_by_label(extracted, "left")
            right_obs = _pick_best_by_label(extracted, "right")
            left, left_smooth = _update_hand_state(left, left_obs, left_smooth)
            right, right_smooth = _update_hand_state(right, right_obs, right_smooth)

            _send_bridge_packet(sock, left, right, now)
            _draw_hud(frame, fps, mirror, left, right, len(extracted))
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mirror = not mirror
    finally:
        if hands is not None:
            hands.close()
        if hand_landmarker is not None:
            hand_landmarker.close()
        sock.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
