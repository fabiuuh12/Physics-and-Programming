#!/usr/bin/env python3
from __future__ import annotations

import math
import os
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

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:
    print(
        "Missing dependency: "
        f"{exc}. Install with: pip install opencv-python mediapipe numpy"
    )
    raise SystemExit(1)

try:
    from AppKit import NSScreen
except Exception:
    NSScreen = None

MODULE_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODULE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from optics.hand_optics_scenes import CANVAS_H, CANVAS_W, OpticsHandScene, SceneHand, SUPPORTED_MODES


CAPTURE_W = 960
CAPTURE_H = 540
HAND_STALE_S = 0.45
SMOOTH_ALPHA = 0.30
PINCH_CLOSE_RATIO = 0.36
FIST_ON_SCORE = 0.60
SCENE_CENTER_ALPHA = 0.24
SCENE_CENTER_ALPHA_BENCH = 0.16
SCENE_ANGLE_ALPHA = 0.20
SCENE_ANGLE_ALPHA_BENCH = 0.22
SCENE_DECAY_ALPHA = 0.08
SCENE_PINCH_ALPHA = 0.24
SCENE_SPAN_ALPHA = 0.22
SCENE_PRESENCE_ALPHA = 0.26
HAND_ANGLE_GAIN = 1.28
HAND_ANGLE_LIMIT_RAD = math.radians(82.0)

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
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

MODEL_CANDIDATES: Tuple[Path, ...] = (
    MODULE_DIR / "models" / "hand_landmarker.task",
    MODULE_DIR.parent / "DefensiveSys" / "models" / "hand_landmarker.task",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])


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
class FilteredSceneHandState:
    center_norm: np.ndarray
    angle_rad: float
    pinch_strength: float
    span_norm: float
    presence: float = 0.0


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
    low = label.lower()
    if low == "left":
        return "Right"
    if low == "right":
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
            label = getattr(c, "category_name", None) or getattr(c, "display_name", None) or "Unknown"
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


def _assign_hands_to_slots(hands: List[HandObservation], tracked: Dict[str, TrackedHand]) -> List[Tuple[str, HandObservation]]:
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


def _best_hand_by_label(tracked: Dict[str, TrackedHand], label: str, now: float) -> TrackedHand | None:
    low = label.lower()
    best: TrackedHand | None = None
    for item in tracked.values():
        if (now - item.last_seen) > HAND_STALE_S:
            continue
        if item.label.lower() != low:
            continue
        if best is None or item.score > best.score:
            best = item
    return best


def _best_hand_by_side(tracked: Dict[str, TrackedHand], left_side: bool, now: float) -> TrackedHand | None:
    best: TrackedHand | None = None
    best_x = None
    for item in tracked.values():
        if (now - item.last_seen) > HAND_STALE_S:
            continue
        wrist_x = float(item.pose[WRIST, 0])
        if best is None:
            best = item
            best_x = wrist_x
            continue
        if left_side and wrist_x < float(best_x):
            best = item
            best_x = wrist_x
        elif not left_side and wrist_x > float(best_x):
            best = item
            best_x = wrist_x
    return best


def _palm_size(points: np.ndarray) -> float:
    wrist = points[WRIST, :2]
    index = points[INDEX_MCP, :2]
    pinky = points[PINKY_MCP, :2]
    size = 0.5 * float(np.linalg.norm(index - wrist) + np.linalg.norm(pinky - wrist))
    return max(1e-4, size)


def _pinch_ratio(points: np.ndarray) -> float:
    palm = _palm_size(points)
    pinch = float(np.linalg.norm(points[THUMB_TIP, :2] - points[INDEX_TIP, :2]))
    return pinch / palm


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < 1e-5 or nbc < 1e-5:
        return 180.0
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _fist_score(points: np.ndarray) -> float:
    curl_angles = (
        _joint_angle_deg(points[INDEX_TIP, :2], points[INDEX_MCP, :2], points[WRIST, :2]),
        _joint_angle_deg(points[MIDDLE_TIP, :2], points[MIDDLE_MCP, :2], points[WRIST, :2]),
        _joint_angle_deg(points[RING_TIP, :2], points[PINKY_MCP, :2], points[WRIST, :2]),
        _joint_angle_deg(points[PINKY_TIP, :2], points[PINKY_MCP, :2], points[WRIST, :2]),
    )
    mean_curl = sum(max(0.0, 180.0 - angle) for angle in curl_angles) / (4.0 * 120.0)
    fingertip_span = (
        float(np.linalg.norm(points[INDEX_TIP, :2] - points[PINKY_TIP, :2])) / _palm_size(points)
    )
    compactness = 1.0 - np.clip(fingertip_span / 1.6, 0.0, 1.0)
    return float(np.clip(0.62 * mean_curl + 0.38 * compactness, 0.0, 1.0))


def _default_scene_hand(side: str) -> SceneHand:
    center_x = 0.28 if side == "left" else 0.72
    angle = 0.0 if side == "left" else np.pi * 0.5
    return SceneHand(
        side=side,
        center_norm=np.asarray([center_x, 0.52], dtype=np.float32),
        angle_rad=float(angle),
        pinch=False,
        pinch_strength=0.0,
        fist=False,
        span_norm=0.08,
        valid=False,
    )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _blend_angle(prev: float, target: float, alpha: float) -> float:
    return prev + _wrap_angle(target - prev) * alpha


def _normalize_xy(vec: np.ndarray, fallback: tuple[float, float] = (0.0, -1.0)) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _steer_from_upright(angle: float, gain: float, limit_rad: float) -> float:
    relative = _wrap_angle(angle + 0.5 * math.pi)
    steered = float(np.clip(relative * gain, -limit_rad, limit_rad))
    return _wrap_angle(steered - 0.5 * math.pi)


def _estimate_hand_angle(points: np.ndarray) -> float:
    wrist = points[WRIST, :2]
    knuckle_center = (
        points[INDEX_MCP, :2] * 0.30
        + points[MIDDLE_MCP, :2] * 0.45
        + points[PINKY_MCP, :2] * 0.25
    )
    fingertip_center = (
        points[INDEX_TIP, :2]
        + points[MIDDLE_TIP, :2]
        + points[RING_TIP, :2]
        + points[PINKY_TIP, :2]
    ) * 0.25

    # Mix knuckle and fingertip direction so smaller palm turns produce a clearer response.
    forward = (knuckle_center - wrist) * 0.58 + (fingertip_center - wrist) * 0.42
    forward = _normalize_xy(forward)
    raw_angle = float(np.arctan2(forward[1], forward[0]))
    return _steer_from_upright(raw_angle, HAND_ANGLE_GAIN, HAND_ANGLE_LIMIT_RAD)


def _default_filter_state(side: str) -> FilteredSceneHandState:
    default = _default_scene_hand(side)
    return FilteredSceneHandState(
        center_norm=default.center_norm.copy(),
        angle_rad=default.angle_rad,
        pinch_strength=0.0,
        span_norm=default.span_norm,
        presence=0.0,
    )


def _tracked_to_scene_hand(tracked: TrackedHand | None, side: str) -> SceneHand:
    if tracked is None:
        return _default_scene_hand(side)

    points = tracked.pose
    center = (
        points[WRIST, :2] * 0.34
        + points[INDEX_MCP, :2] * 0.24
        + points[MIDDLE_MCP, :2] * 0.18
        + points[PINKY_MCP, :2] * 0.24
    ).astype(np.float32)
    angle = _estimate_hand_angle(points)
    span_norm = float(np.clip(_palm_size(points) * 1.20, 0.05, 0.24))
    pinch_ratio = _pinch_ratio(points)
    pinch_strength = float(np.clip((PINCH_CLOSE_RATIO - pinch_ratio) / max(0.12, PINCH_CLOSE_RATIO), 0.0, 1.0))
    return SceneHand(
        side=side,
        center_norm=np.clip(center, 0.02, 0.98),
        angle_rad=angle,
        pinch=pinch_ratio < PINCH_CLOSE_RATIO,
        pinch_strength=pinch_strength,
        fist=_fist_score(points) >= FIST_ON_SCORE,
        span_norm=span_norm,
        valid=True,
    )


def _smooth_scene_hand(
    state: FilteredSceneHandState,
    raw: SceneHand,
    side: str,
    mode: str,
) -> SceneHand:
    fallback = _default_scene_hand(side)
    target = raw if raw.valid else fallback

    center_alpha = SCENE_CENTER_ALPHA_BENCH if mode == "bench" else SCENE_CENTER_ALPHA
    angle_alpha = SCENE_ANGLE_ALPHA_BENCH if mode == "bench" else SCENE_ANGLE_ALPHA
    if not raw.valid:
        center_alpha = SCENE_DECAY_ALPHA
        angle_alpha = min(angle_alpha, SCENE_DECAY_ALPHA)

    state.center_norm = (
        (1.0 - center_alpha) * state.center_norm + center_alpha * target.center_norm
    ).astype(np.float32)
    state.angle_rad = _blend_angle(state.angle_rad, target.angle_rad, angle_alpha)
    state.pinch_strength = (1.0 - SCENE_PINCH_ALPHA) * state.pinch_strength + SCENE_PINCH_ALPHA * target.pinch_strength
    state.span_norm = (1.0 - SCENE_SPAN_ALPHA) * state.span_norm + SCENE_SPAN_ALPHA * target.span_norm
    state.presence = (1.0 - SCENE_PRESENCE_ALPHA) * state.presence + SCENE_PRESENCE_ALPHA * (1.0 if raw.valid else 0.0)

    return SceneHand(
        side=side,
        center_norm=np.clip(state.center_norm, 0.02, 0.98),
        angle_rad=state.angle_rad,
        pinch=bool(raw.valid and state.pinch_strength > 0.28),
        pinch_strength=float(np.clip(state.pinch_strength, 0.0, 1.0)),
        fist=bool(raw.valid and raw.fist),
        span_norm=float(np.clip(state.span_norm, 0.05, 0.24)),
        valid=bool(raw.valid or state.presence > 0.30),
    )


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


def run_scene(mode: str, window_name: str) -> int:
    if mode not in SUPPORTED_MODES:
        print(f"Unsupported mode {mode!r}. Choose from: {', '.join(sorted(SUPPORTED_MODES))}")
        return 1

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
    filtered_hands = {
        "left": _default_filter_state("left"),
        "right": _default_filter_state("right"),
    }
    smoother = MultiLandmarkSmoother(alpha=SMOOTH_ALPHA)
    scene = OpticsHandScene(mode)
    mirror = True
    fullscreen = True
    screen_w, screen_h = _get_screen_size()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

            left_raw = _tracked_to_scene_hand(left_hand, "left")
            right_raw = _tracked_to_scene_hand(right_hand, "right")
            hands = {
                "left": _smooth_scene_hand(filtered_hands["left"], left_raw, "left", mode),
                "right": _smooth_scene_hand(filtered_hands["right"], right_raw, "right", mode),
            }

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            scene.draw(canvas, hands, now, fps)
            _draw_webcam_inset(canvas, frame, extracted)

            display = canvas
            if fullscreen and (screen_w != CANVAS_W or screen_h != CANVAS_H):
                display = cv2.resize(canvas, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                fullscreen = not fullscreen
                mode_flag = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode_flag)
                if not fullscreen:
                    cv2.resizeWindow(window_name, CANVAS_W, CANVAS_H)
            if key == ord("m"):
                mirror = not mirror
            if key == ord("r"):
                scene.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if tracker_solution is not None:
            tracker_solution.close()
        if tracker_tasks is not None:
            tracker_tasks.close()

    return 0
