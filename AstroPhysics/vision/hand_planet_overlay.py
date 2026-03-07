#!/usr/bin/env python3
"""
Hand object overlay lab.

Both hands cycle: earth <-> star <-> blackhole

Make a fist to cycle that hand's object.
When two objects get too close, pair-specific interaction appears at midpoint.
When hands separate, objects return to normal.

Controls:
- q: quit
- m: toggle mirror
- [: smaller objects
- ]: larger objects
- -: lower objects
- =: raise objects
"""

from __future__ import annotations

import math
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Some MediaPipe builds import matplotlib internally; keep cache writable.
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.framework.formats import landmark_pb2
except Exception:
    landmark_pb2 = None


WINDOW_NAME = "Hand Astro Objects"
MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODULE_DIR / "models" / "hand_landmarker.task"

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

ANCHOR_ALPHA = 0.30
RADIUS_ALPHA = 0.24

FIST_ON_SCORE = 0.58
FIST_ON_EMA_SCORE = 0.52
FIST_OFF_SCORE = 0.35
FIST_HOLD_SEC = 0.12
FIST_SWITCH_COOLDOWN_SEC = 0.55
FIST_EMA_ALPHA = 0.24
FIST_MIN_CURLED_FINGERS = 2
INTERACT_THRESHOLD_BOOST = 2.10
INTERACT_ENTER_RATIO = 1.25
INTERACT_EXIT_RATIO = 1.90
INTERACT_BLEND_ALPHA = 0.30
INTERACT_DWELL_SEC = 0.06
NEAR_FREEZE_NORM_DIST = 2.8
CYCLE_FREEZE_NORM_DIST = 1.08
HAND_TRACK_MAX_JUMP_NORM = 0.25
HAND_MISSING_GRACE_SEC = 0.22
CLOSE_RANGE_ALPHA_SCALE = 0.55
MIN_HANDEDNESS_SCORE = 0.60
DUPLICATE_CENTER_NORM = 0.14
DUPLICATE_PALM_RATIO_MIN = 0.58
DUPLICATE_LANDMARK_DIST_NORM = 0.030
MP_MIN_DETECTION_CONF = 0.62
MP_MIN_TRACKING_CONF = 0.60
MP_MIN_PRESENCE_CONF = 0.60
RENDER_BRIDGE_ENABLED = os.environ.get("ASTRO_RENDER_BRIDGE", "1") != "0"
RENDER_BRIDGE_HOST = os.environ.get("ASTRO_RENDER_BRIDGE_HOST", "127.0.0.1")
RENDER_BRIDGE_PORT = int(os.environ.get("ASTRO_RENDER_BRIDGE_PORT", "50506"))

CACHE_MAX = 320
PHASE_BUCKETS = 96
RADIUS_QUANT = 2


@dataclass
class HandObservation:
    label: str
    score: float
    landmarks: List[Tuple[float, float, float]]


@dataclass
class BodyState:
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0
    valid: bool = False


@dataclass
class HandCycleState:
    object_index: int = 0
    fist_latched: bool = False
    fist_since_ts: float = 0.0
    last_switch_ts: float = 0.0
    fist_score_ema: float = 0.0
    fist_score_live: float = 0.0
    curled_fingers: int = 0


@dataclass
class HandTrackState:
    x: float = 0.5
    y: float = 0.5
    active: bool = False
    last_seen_ts: float = 0.0


@dataclass
class BodyParams:
    kind: str
    name: str
    mass: float
    radius_scale: float
    spin: float
    color_temp: float
    emissive: float
    danger_radius: float


@dataclass
class InteractionState:
    active: bool = False
    pair: Tuple[str, str] | None = None
    label: str = "none"
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0
    strength: float = 0.0
    close_since_ts: float = 0.0
    locked_pair: Tuple[str, str] | None = None


@dataclass
class Patch:
    image: np.ndarray
    alpha: np.ndarray
    center: int


def _extract_solution_hands(result: object, mirror: bool) -> List[HandObservation]:
    out: List[HandObservation] = []
    if not getattr(result, "multi_hand_landmarks", None):
        return out

    for idx, lms in enumerate(result.multi_hand_landmarks):
        label = "Unknown"
        score = 0.0
        if result.multi_handedness and idx < len(result.multi_handedness):
            c = result.multi_handedness[idx].classification[0]
            label = c.label or "Unknown"
            score = float(c.score)
        if score > 0.0 and score < MIN_HANDEDNESS_SCORE:
            continue
        if mirror:
            if label.lower() == "left":
                label = "Right"
            elif label.lower() == "right":
                label = "Left"
        points = [(lm.x, lm.y, lm.z) for lm in lms.landmark]
        out.append(HandObservation(label=label, score=score, landmarks=points))
    return _suppress_duplicate_hands(out)


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
            label = raw_label
            score = float(getattr(c, "score", 0.0))
        if score > 0.0 and score < MIN_HANDEDNESS_SCORE:
            continue
        if mirror:
            if label.lower() == "left":
                label = "Right"
            elif label.lower() == "right":
                label = "Left"
        points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        out.append(HandObservation(label=label, score=score, landmarks=points))
    return _suppress_duplicate_hands(out)


def _palm_center_and_size(hand: HandObservation) -> Tuple[Tuple[float, float], float]:
    p = np.asarray(hand.landmarks, dtype=np.float32)
    center = (p[[WRIST, INDEX_MCP, PINKY_MCP], 0:2].mean(axis=0)).tolist()
    size = float(np.linalg.norm(p[INDEX_MCP, 0:2] - p[PINKY_MCP, 0:2]))
    return (float(center[0]), float(center[1])), max(0.04, size)


def _hand_weight(hand: HandObservation) -> float:
    _, palm_size = _palm_center_and_size(hand)
    return hand.score + palm_size * 2.0


def _landmark_mean_dist(a: HandObservation, b: HandObservation) -> float:
    pa = np.asarray(a.landmarks, dtype=np.float32)[:, 0:2]
    pb = np.asarray(b.landmarks, dtype=np.float32)[:, 0:2]
    n = min(len(pa), len(pb))
    if n <= 0:
        return 1e9
    d = np.linalg.norm(pa[:n] - pb[:n], axis=1)
    return float(np.mean(d))


def _suppress_duplicate_hands(hands: List[HandObservation]) -> List[HandObservation]:
    if len(hands) <= 1:
        return hands

    ranked = sorted(hands, key=_hand_weight, reverse=True)
    kept: List[HandObservation] = []

    for hand in ranked:
        (cx, cy), palm = _palm_center_and_size(hand)
        duplicate = False
        for kept_hand in kept:
            (kx, ky), kp = _palm_center_and_size(kept_hand)
            center_d = math.hypot(cx - kx, cy - ky)
            ratio = min(palm, kp) / max(palm, kp)
            lm_d = _landmark_mean_dist(hand, kept_hand)
            if center_d <= DUPLICATE_CENTER_NORM and ratio >= DUPLICATE_PALM_RATIO_MIN:
                duplicate = True
                break
            if center_d <= (DUPLICATE_CENTER_NORM * 1.9) and lm_d <= DUPLICATE_LANDMARK_DIST_NORM:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(hand)
        if len(kept) >= 2:
            break

    kept.sort(key=lambda h: _center_of_hand(h)[0])
    return kept


def _split_hands_by_side(hands: List[HandObservation]) -> Tuple[HandObservation | None, HandObservation | None]:
    """
    Stable hand slot assignment by screen side.
    Returns: (left_slot_hand, right_slot_hand)
    """
    if not hands:
        return None, None

    scored = []
    for hand in hands:
        (cx, _), _ = _palm_center_and_size(hand)
        scored.append((cx, hand))
    scored.sort(key=lambda item: item[0])

    if len(scored) == 1:
        cx, hand = scored[0]
        return (hand, None) if cx < 0.5 else (None, hand)

    return scored[0][1], scored[-1][1]


def _center_of_hand(hand: HandObservation) -> Tuple[float, float]:
    (cx, cy), _ = _palm_center_and_size(hand)
    return cx, cy


def _update_track(track: HandTrackState, hand: HandObservation | None, now_ts: float) -> None:
    if hand is not None:
        cx, cy = _center_of_hand(hand)
        track.x = cx
        track.y = cy
        track.active = True
        track.last_seen_ts = now_ts
    elif track.active and (now_ts - track.last_seen_ts) > HAND_MISSING_GRACE_SEC:
        track.active = False


def _stable_assign_hands(
    observations: List[HandObservation],
    left_track: HandTrackState,
    right_track: HandTrackState,
    now_ts: float,
) -> Tuple[HandObservation | None, HandObservation | None]:
    """
    Stable per-side assignment using previous slot positions first, then side fallback.
    """
    if not observations:
        return None, None

    centers = [_center_of_hand(h) for h in observations]
    used: set[int] = set()
    assigned_left: HandObservation | None = None
    assigned_right: HandObservation | None = None

    def nearest_unused(tx: float, ty: float) -> Tuple[int | None, float]:
        best_idx = None
        best_d = 1e9
        for i, (cx, cy) in enumerate(centers):
            if i in used:
                continue
            d = math.hypot(cx - tx, cy - ty)
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx, best_d

    if left_track.active and (now_ts - left_track.last_seen_ts) <= HAND_MISSING_GRACE_SEC:
        i, d = nearest_unused(left_track.x, left_track.y)
        if i is not None and d <= HAND_TRACK_MAX_JUMP_NORM:
            assigned_left = observations[i]
            used.add(i)

    if right_track.active and (now_ts - right_track.last_seen_ts) <= HAND_MISSING_GRACE_SEC:
        i, d = nearest_unused(right_track.x, right_track.y)
        if i is not None and d <= HAND_TRACK_MAX_JUMP_NORM:
            assigned_right = observations[i]
            used.add(i)

    remaining = [i for i in range(len(observations)) if i not in used]
    if remaining:
        remaining_sorted = sorted(remaining, key=lambda i: centers[i][0])
        if assigned_left is None and assigned_right is None:
            if len(remaining_sorted) == 1:
                i = remaining_sorted[0]
                if centers[i][0] < 0.5:
                    assigned_left = observations[i]
                else:
                    assigned_right = observations[i]
            else:
                assigned_left = observations[remaining_sorted[0]]
                assigned_right = observations[remaining_sorted[-1]]
        elif assigned_left is None:
            assigned_left = observations[remaining_sorted[0]]
        elif assigned_right is None:
            assigned_right = observations[remaining_sorted[-1]]

    return assigned_left, assigned_right


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


def _fist_score(hand: HandObservation) -> Tuple[float, int]:
    """
    Returns (closure_score [0..1], curled_finger_count) for index/middle/ring/pinky.
    """
    p = np.asarray(hand.landmarks, dtype=np.float32)
    palm = np.linalg.norm(p[INDEX_MCP, 0:2] - p[PINKY_MCP, 0:2])
    palm = float(max(1e-4, palm))

    fingers = [
        (5, 6, 8),    # index
        (9, 10, 12),  # middle
        (13, 14, 16), # ring
        (17, 18, 20), # pinky
    ]

    scores: List[float] = []
    curled = 0
    for mcp_i, pip_i, tip_i in fingers:
        mcp = p[mcp_i, 0:2]
        pip = p[pip_i, 0:2]
        tip = p[tip_i, 0:2]

        tip_norm = float(np.linalg.norm(tip - mcp) / palm)
        # Lower tip distance to MCP means more closed.
        dist_closed = float(np.clip((1.05 - tip_norm) / 0.49, 0.0, 1.0))

        ang = _joint_angle_deg(mcp, pip, tip)
        # Lower PIP angle means more curled.
        ang_closed = float(np.clip((155.0 - ang) / 70.0, 0.0, 1.0))

        finger_score = 0.55 * dist_closed + 0.45 * ang_closed
        scores.append(finger_score)
        if finger_score >= 0.70:
            curled += 1

    score = float(np.clip(np.mean(scores), 0.0, 1.0))
    return score, curled


def _update_hand_cycle(hand: HandObservation | None, cycle: HandCycleState, now_ts: float, num_objects: int) -> bool:
    if hand is None:
        cycle.fist_latched = False
        cycle.fist_since_ts = 0.0
        cycle.fist_score_ema = 0.0
        cycle.fist_score_live = 0.0
        cycle.curled_fingers = 0
        return False

    score, curled = _fist_score(hand)
    cycle.fist_score_live = score
    cycle.curled_fingers = curled
    if cycle.fist_score_ema <= 0.0:
        cycle.fist_score_ema = score
    else:
        a = float(np.clip(FIST_EMA_ALPHA, 0.01, 0.99))
        cycle.fist_score_ema = (1.0 - a) * cycle.fist_score_ema + a * score

    fist_on = (
        score >= FIST_ON_SCORE
        and cycle.fist_score_ema >= FIST_ON_EMA_SCORE
        and curled >= FIST_MIN_CURLED_FINGERS
    )
    fist_off = cycle.fist_score_ema <= FIST_OFF_SCORE or curled <= 1

    switched = False
    if not cycle.fist_latched:
        if fist_on:
            if cycle.fist_since_ts <= 0.0:
                cycle.fist_since_ts = now_ts
            if (
                (now_ts - cycle.fist_since_ts) >= FIST_HOLD_SEC
                and (now_ts - cycle.last_switch_ts) >= FIST_SWITCH_COOLDOWN_SEC
            ):
                cycle.object_index = (cycle.object_index + 1) % max(1, num_objects)
                cycle.last_switch_ts = now_ts
                cycle.fist_latched = True
                cycle.fist_since_ts = 0.0
                switched = True
        else:
            cycle.fist_since_ts = 0.0
    elif fist_off:
        cycle.fist_latched = False

    return switched


def _undo_cycle_switch(cycle: HandCycleState, num_objects: int) -> None:
    cycle.object_index = (cycle.object_index - 1) % max(1, num_objects)
    cycle.fist_latched = False
    cycle.fist_since_ts = 0.0


def _update_body_state(
    state: BodyState,
    hand: HandObservation | None,
    frame_shape: Tuple[int, int, int],
    params: BodyParams,
    global_size_scale: float,
    lift_scale: float,
    keep_previous_on_missing: bool = False,
    position_alpha: float = ANCHOR_ALPHA,
) -> None:
    if hand is None:
        if keep_previous_on_missing and state.valid:
            return
        state.valid = False
        return

    h, w = frame_shape[:2]
    (cx_n, cy_n), palm_size_n = _palm_center_and_size(hand)
    cx = float(np.clip(cx_n * w, 0, w - 1))
    cy = float(np.clip(cy_n * h, 0, h - 1))
    palm_px = max(18.0, palm_size_n * max(w, h))

    target_x = cx
    target_y = cy - lift_scale * palm_px
    target_radius = max(10.0, palm_px * 0.50 * global_size_scale * params.radius_scale)

    pos_alpha = float(np.clip(position_alpha, 0.05, 0.95))
    if state.valid:
        state.x = (1.0 - pos_alpha) * state.x + pos_alpha * target_x
        state.y = (1.0 - pos_alpha) * state.y + pos_alpha * target_y
        state.radius = (1.0 - RADIUS_ALPHA) * state.radius + RADIUS_ALPHA * target_radius
    else:
        state.x = target_x
        state.y = target_y
        state.radius = target_radius
        state.valid = True


def _temp_to_bgr(color_temp: float) -> Tuple[float, float, float]:
    # Approximate blackbody-like color ramp for star tones.
    t = float(np.clip((color_temp - 2600.0) / (32000.0 - 2600.0), 0.0, 1.0))
    # Anchor colors in BGR.
    cool = np.array([80.0, 170.0, 255.0], dtype=np.float32)   # warm K/M (orange)
    sun = np.array([170.0, 230.0, 255.0], dtype=np.float32)   # G/F
    hot = np.array([255.0, 225.0, 190.0], dtype=np.float32)   # O/B (blue-white)
    if t < 0.42:
        u = t / 0.42
        c = cool * (1.0 - u) + sun * u
    else:
        u = (t - 0.42) / 0.58
        c = sun * (1.0 - u) + hot * u
    return float(c[0]), float(c[1]), float(c[2])


class EarthRenderer:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, int, int], Patch] = {}

    def _key(self, radius: int, main_phase: float, cloud_phase: float, params: BodyParams) -> Tuple[int, int, int, int]:
        rq = max(8, int(round(radius / float(RADIUS_QUANT)) * RADIUS_QUANT))
        pb_main = int((main_phase % (2.0 * math.pi)) / (2.0 * math.pi) * PHASE_BUCKETS) % PHASE_BUCKETS
        pb_cloud = int((cloud_phase % (2.0 * math.pi)) / (2.0 * math.pi) * PHASE_BUCKETS) % PHASE_BUCKETS
        em = int(np.clip(params.emissive * 10.0, 0.0, 50.0))
        return rq, pb_main, pb_cloud, em

    def _get_patch(self, radius: int, main_phase: float, cloud_phase: float, params: BodyParams) -> Patch:
        key = self._key(radius, main_phase, cloud_phase, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        patch = self._build_patch(*key)
        self._cache[key] = patch
        if len(self._cache) > CACHE_MAX:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        return patch

    def _build_patch(self, radius: int, phase_bucket: int, cloud_bucket: int, emissive_bucket: int) -> Patch:
        phase = (2.0 * math.pi * phase_bucket) / float(PHASE_BUCKETS)
        cloud_phase = (2.0 * math.pi * cloud_bucket) / float(PHASE_BUCKETS)
        emissive = 0.8 + 0.02 * emissive_bucket

        pad = int(radius * 1.14) + 6
        size = 2 * pad + 1
        cy = cx = pad

        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx.astype(np.float32) - cx) / float(radius)
        ny = (yy.astype(np.float32) - cy) / float(radius)
        r2 = nx * nx + ny * ny
        sphere = r2 <= 1.0
        nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

        lat = np.arcsin(np.clip(ny, -1.0, 1.0))
        lon = np.arctan2(nx, nz + 1e-6) + 0.92 * phase

        n1 = np.sin(2.5 * lon + 1.1 * np.sin(1.9 * lat))
        n2 = np.sin(4.2 * lat - 1.7 * lon + 1.3)
        n3 = np.sin(7.3 * lon + 5.2 * lat + 0.4)
        elev = 0.54 * n1 + 0.31 * n2 + 0.15 * n3

        abs_lat = np.abs(lat)
        lat_norm = abs_lat / (math.pi * 0.5)

        ocean_b = 106.0 + 40.0 * (1.0 - lat_norm)
        ocean_g = 64.0 + 36.0 * (1.0 - lat_norm)
        ocean_r = 22.0 + 10.0 * (1.0 - lat_norm)

        image = np.stack([ocean_b, ocean_g, ocean_r], axis=-1).astype(np.float32)

        land = elev > 0.11
        tropical = np.exp(-np.square(lat / 0.86))
        highland = np.clip((elev - 0.11) / 0.55, 0.0, 1.0)

        land_b = 44.0 + 18.0 * tropical + 14.0 * highland
        land_g = 90.0 + 82.0 * tropical - 24.0 * highland
        land_r = 52.0 + 40.0 * (1.0 - tropical) + 44.0 * highland

        image[..., 0] = np.where(land, land_b, image[..., 0])
        image[..., 1] = np.where(land, land_g, image[..., 1])
        image[..., 2] = np.where(land, land_r, image[..., 2])

        desert = land & (abs_lat < 0.45) & (elev < 0.24)
        image[..., 0] = np.where(desert, 70.0, image[..., 0])
        image[..., 1] = np.where(desert, 145.0, image[..., 1])
        image[..., 2] = np.where(desert, 196.0, image[..., 2])

        coast = np.abs(elev - 0.11) < 0.028
        image[..., 0] = np.where(coast, 76.0, image[..., 0])
        image[..., 1] = np.where(coast, 168.0, image[..., 1])
        image[..., 2] = np.where(coast, 176.0, image[..., 2])

        polar = abs_lat > 1.08
        image[polar] = image[polar] * 0.35 + np.array([240.0, 246.0, 252.0], dtype=np.float32) * 0.65

        cloud_base = (
            0.48 * np.sin(8.4 * (lon + 0.35 * cloud_phase) - 2.1 * lat)
            + 0.33 * np.sin(15.2 * (lon + 0.52 * cloud_phase) + 10.6 * lat + 0.7)
            + 0.19 * np.sin(20.0 * (lon + 0.44 * cloud_phase) - 5.7 * lat + 2.0)
        )
        clouds = cloud_base > 0.56

        light_dir = np.array([-0.42, -0.35, 0.84], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip(light_dir[0] * nx + light_dir[1] * ny + light_dir[2] * nz, 0.0, 1.0)
        shade = 0.21 + 0.79 * diffuse
        image *= shade[..., None]

        dark = diffuse < 0.11
        city_noise = np.sin(13.4 * lon + 8.2 * lat) * np.sin(9.6 * lon - 11.0 * lat + 1.2)
        cities = land & dark & (abs_lat < 1.25) & (city_noise > 0.84)
        image[cities] += np.array([32.0, 78.0, 172.0], dtype=np.float32) * emissive

        hx = nx + 0.30
        hy = ny + 0.27
        spec = np.exp(-((hx * hx + hy * hy) / 0.023))
        spec *= (~land).astype(np.float32)
        image += spec[..., None] * np.array([80.0, 125.0, 190.0], dtype=np.float32)

        cloud_alpha = np.where(clouds, 0.16 + 0.54 * diffuse, 0.0).astype(np.float32)
        image = image * (1.0 - cloud_alpha[..., None]) + 255.0 * cloud_alpha[..., None]

        halo = (r2 > 1.0) & (r2 <= 1.11 * 1.11)
        halo_strength = np.zeros_like(r2, dtype=np.float32)
        halo_strength[halo] = (1.11 - np.sqrt(r2[halo])) / 0.11
        image[..., 0] += 120.0 * halo_strength
        image[..., 1] += 82.0 * halo_strength
        image[..., 2] += 26.0 * halo_strength

        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        alpha = np.zeros_like(r2, dtype=np.float32)
        alpha[sphere] = 0.95
        alpha = np.maximum(alpha, 0.32 * halo_strength)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha[~(sphere | halo)] = 0.0

        cv2.circle(image, (cx, cy), radius, (230, 238, 252), 1, cv2.LINE_AA)
        return Patch(image=image, alpha=alpha, center=pad)

    def draw(self, frame: np.ndarray, center: Tuple[int, int], radius: int, main_phase: float, cloud_phase: float, params: BodyParams) -> None:
        if radius < 8:
            return
        patch = self._get_patch(radius, main_phase, cloud_phase, params)
        _blit_patch(frame, patch, center)


class StarRenderer:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, int, int, int], Patch] = {}

    def _key(self, radius: int, main_phase: float, flare_phase: float, params: BodyParams) -> Tuple[int, int, int, int, int]:
        rq = max(8, int(round(radius / float(RADIUS_QUANT)) * RADIUS_QUANT))
        pb = int((main_phase % (2.0 * math.pi)) / (2.0 * math.pi) * PHASE_BUCKETS) % PHASE_BUCKETS
        fb = int((flare_phase % (2.0 * math.pi)) / (2.0 * math.pi) * PHASE_BUCKETS) % PHASE_BUCKETS
        tb = int(np.clip(params.color_temp / 250.0, 0.0, 200.0))
        sb = int(np.clip(params.spin * 10.0, 0.0, 300.0))
        return rq, pb, fb, tb, sb

    def _get_patch(self, radius: int, main_phase: float, flare_phase: float, params: BodyParams) -> Patch:
        key = self._key(radius, main_phase, flare_phase, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        patch = self._build_patch(*key)
        self._cache[key] = patch
        if len(self._cache) > CACHE_MAX:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        return patch

    def _build_patch(self, radius: int, phase_bucket: int, flare_bucket: int, temp_bucket: int, spin_bucket: int) -> Patch:
        main_phase = (2.0 * math.pi * phase_bucket) / float(PHASE_BUCKETS)
        flare_phase = (2.0 * math.pi * flare_bucket) / float(PHASE_BUCKETS)
        color_temp = 250.0 * temp_bucket
        spin = 0.1 * spin_bucket

        b0, g0, r0 = _temp_to_bgr(color_temp)

        pad = int(radius * 2.0) + 6
        size = 2 * pad + 1
        cy = cx = pad
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx.astype(np.float32) - cx) / float(radius)
        ny = (yy.astype(np.float32) - cy) / float(radius)
        r2 = nx * nx + ny * ny
        rr = np.sqrt(r2)
        sphere = rr <= 1.0

        gran = (
            0.50 * np.sin((11.0 + 0.3 * spin) * nx + 8.0 * ny + 2.2 * main_phase)
            + 0.32 * np.sin((17.0 + 0.2 * spin) * nx - 13.0 * ny - 1.1 * main_phase)
            + 0.18 * np.sin(24.0 * nx + 6.0 * ny + 0.6)
        )
        gran = 0.5 + 0.5 * gran
        core = np.exp(-(r2 / 0.58))
        limb = np.clip(1.0 - r2, 0.0, 1.0)

        image = np.zeros((size, size, 3), dtype=np.float32)
        image[..., 0] = b0 * (0.58 + 0.42 * gran) + 35.0 * core
        image[..., 1] = g0 * (0.58 + 0.42 * gran) + 35.0 * core
        image[..., 2] = r0 * (0.58 + 0.42 * gran) + 18.0 * core
        image *= (0.48 + 0.52 * limb)[..., None]

        # Micro flares.
        flare_dir = np.arctan2(ny, nx)
        flare_gate = np.exp(-np.square((rr - 1.0) / 0.12))
        flare_pattern = (
            np.sin(6.0 * flare_dir + 3.6 * flare_phase)
            + 0.6 * np.sin(11.0 * flare_dir - 2.8 * flare_phase)
        )
        flare_strength = np.clip(0.5 + 0.5 * flare_pattern, 0.0, 1.0) * flare_gate
        image[..., 0] += 85.0 * flare_strength
        image[..., 1] += 95.0 * flare_strength
        image[..., 2] += 55.0 * flare_strength

        halo = np.exp(-(r2 / 2.25))
        image[..., 0] += 90.0 * halo
        image[..., 1] += 110.0 * halo
        image[..., 2] += 85.0 * halo

        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        alpha = np.zeros_like(rr, dtype=np.float32)
        alpha[sphere] = 0.96
        alpha += 0.45 * halo * (rr <= 1.85)
        alpha = np.clip(alpha, 0.0, 1.0)
        return Patch(image=image, alpha=alpha, center=pad)

    def draw(self, frame: np.ndarray, center: Tuple[int, int], radius: int, main_phase: float, flare_phase: float, params: BodyParams) -> None:
        if radius < 8:
            return
        patch = self._get_patch(radius, main_phase, flare_phase, params)
        _blit_patch(frame, patch, center)


class BlackHoleRenderer:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, int, int, int], Patch] = {}

    def _key(self, radius: int, phase: float, disk_phase: float, params: BodyParams) -> Tuple[int, int, int, int, int]:
        rq = max(8, int(round(radius / float(RADIUS_QUANT)) * RADIUS_QUANT))
        pb = int((phase % (2.0 * math.pi)) / (2.0 * math.pi) * PHASE_BUCKETS) % PHASE_BUCKETS
        db = int((disk_phase % (2.0 * math.pi)) / (2.0 * math.pi) * PHASE_BUCKETS) % PHASE_BUCKETS
        sb = int(np.clip(params.spin * 12.0, 0.0, 400.0))
        mb = int(np.clip(params.mass * 4.0, 0.0, 400.0))
        return rq, pb, db, sb, mb

    def _get_patch(self, radius: int, phase: float, disk_phase: float, params: BodyParams) -> Patch:
        key = self._key(radius, phase, disk_phase, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        patch = self._build_patch(*key)
        self._cache[key] = patch
        if len(self._cache) > CACHE_MAX:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        return patch

    def _build_patch(self, radius: int, phase_bucket: int, disk_bucket: int, spin_bucket: int, mass_bucket: int) -> Patch:
        phase = (2.0 * math.pi * phase_bucket) / float(PHASE_BUCKETS)
        disk_phase = (2.0 * math.pi * disk_bucket) / float(PHASE_BUCKETS)
        spin = 0.083 * spin_bucket
        mass = 0.25 * mass_bucket

        pad = int(radius * 2.35) + 8
        size = 2 * pad + 1
        cy = cx = pad
        yy, xx = np.mgrid[0:size, 0:size]
        nx = (xx.astype(np.float32) - cx) / float(radius)
        ny = (yy.astype(np.float32) - cy) / float(radius)
        r2 = nx * nx + ny * ny
        rr = np.sqrt(r2)

        image = np.zeros((size, size, 3), dtype=np.float32)

        # Event horizon.
        horizon = np.clip(1.0 - rr, 0.0, 1.0)
        image[..., 0] += 8.0 * horizon
        image[..., 1] += 8.0 * horizon
        image[..., 2] += 14.0 * horizon

        # Stronger photon ring with mass scaling.
        ring_pos = 1.02 + 0.02 * np.tanh((mass - 10.0) / 20.0)
        photon = np.exp(-np.square((rr - ring_pos) / 0.032))
        image[..., 0] += 110.0 * photon
        image[..., 1] += 160.0 * photon
        image[..., 2] += 230.0 * photon

        # Tilted accretion disk, with Doppler bright side.
        tilt = 0.38 + 0.06 * np.sin(0.35 * phase)
        c = math.cos(disk_phase * (1.0 + 0.05 * spin))
        s = math.sin(disk_phase * (1.0 + 0.05 * spin))
        xr = nx * c + ny * s
        yr = -nx * s + ny * c

        re = np.sqrt((xr / 1.62) ** 2 + (yr / tilt) ** 2)
        disk_band = np.exp(-np.square((re - 1.0) / 0.12))
        turbulence = (
            0.5
            + 0.5 * np.sin((18.0 + 0.4 * spin) * xr + 7.0 * yr + 2.4 * disk_phase)
            + 0.20 * np.sin(34.0 * xr - 10.0 * yr + 1.2 * phase)
        )
        turbulence = np.clip(turbulence, 0.0, 1.0)

        # Doppler beaming approximation: approaching side brighter.
        doppler = np.clip(0.48 + 0.52 * xr, 0.12, 1.1)
        disk = disk_band * (0.26 + 0.74 * turbulence) * doppler
        disk *= (rr >= 0.90)

        image[..., 0] += 58.0 * disk
        image[..., 1] += 170.0 * disk
        image[..., 2] += 255.0 * disk

        lens = np.exp(-np.square((rr - 1.35) / 0.28))
        image[..., 0] += 40.0 * lens
        image[..., 1] += 92.0 * lens
        image[..., 2] += 170.0 * lens

        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        alpha = np.zeros_like(rr, dtype=np.float32)
        alpha[rr <= 1.0] = 0.98
        alpha = np.maximum(alpha, 0.70 * np.clip(photon + disk + 0.62 * lens, 0.0, 1.0))
        alpha[rr > 2.05] = 0.0
        alpha = np.clip(alpha, 0.0, 1.0)

        return Patch(image=image, alpha=alpha, center=pad)

    def draw(self, frame: np.ndarray, center: Tuple[int, int], radius: int, main_phase: float, disk_phase: float, params: BodyParams) -> None:
        if radius < 8:
            return
        patch = self._get_patch(radius, main_phase, disk_phase, params)
        _blit_patch(frame, patch, center)


def _blit_patch(frame: np.ndarray, patch: Patch, center: Tuple[int, int]) -> None:
    h, w = frame.shape[:2]
    cx, cy = center
    top = cy - patch.center
    left = cx - patch.center
    bottom = top + patch.image.shape[0]
    right = left + patch.image.shape[1]

    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(w, right)
    y1 = min(h, bottom)
    if x1 <= x0 or y1 <= y0:
        return

    px0 = x0 - left
    py0 = y0 - top
    px1 = px0 + (x1 - x0)
    py1 = py0 + (y1 - y0)

    sub_img = patch.image[py0:py1, px0:px1].astype(np.float32)
    sub_alpha = patch.alpha[py0:py1, px0:px1][..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1.0 - sub_alpha) + sub_img * sub_alpha
    frame[y0:y1, x0:x1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    order = {"earth": 0, "star": 1, "blackhole": 2}
    return tuple(sorted((a, b), key=lambda k: order.get(k, 99)))


def _kind_to_id(kind: str) -> int:
    table = {"earth": 0, "star": 1, "blackhole": 2}
    return int(table.get(kind, 0))


def _send_render_bridge(
    bridge_sock: socket.socket | None,
    now_ts: float,
    left_state: BodyState,
    right_state: BodyState,
    left_kind: str,
    right_kind: str,
    interaction_state: InteractionState,
    frame_shape: Tuple[int, int, int],
) -> None:
    if bridge_sock is None:
        return

    h, w = frame_shape[:2]
    if w <= 0 or h <= 0:
        return

    lx = float(np.clip(left_state.x / w, 0.0, 1.0))
    ly = float(np.clip(left_state.y / h, 0.0, 1.0))
    rx = float(np.clip(right_state.x / w, 0.0, 1.0))
    ry = float(np.clip(right_state.y / h, 0.0, 1.0))
    msg = (
        f"{now_ts:.3f},{int(left_state.valid)},{lx:.4f},{ly:.4f},"
        f"{int(right_state.valid)},{rx:.4f},{ry:.4f},"
        f"{_kind_to_id(left_kind)},{_kind_to_id(right_kind)},"
        f"{int(interaction_state.active)},{float(np.clip(interaction_state.strength, 0.0, 1.0)):.3f}"
    )
    try:
        bridge_sock.sendto(msg.encode("ascii", errors="ignore"), (RENDER_BRIDGE_HOST, RENDER_BRIDGE_PORT))
    except OSError:
        # UDP bridge is optional; ignore transient socket errors.
        pass


def _pair_label(pair: Tuple[str, str]) -> str:
    labels = {
        ("earth", "earth"): "planet collision",
        ("earth", "star"): "planet evaporation",
        ("earth", "blackhole"): "spaghettification",
        ("star", "star"): "supernova",
        ("star", "blackhole"): "accretion burst",
        ("blackhole", "blackhole"): "bh merger",
    }
    return labels.get(pair, "interaction")


def _interaction_threshold_px(
    left_state: BodyState,
    right_state: BodyState,
    left_params: BodyParams,
    right_params: BodyParams,
) -> float:
    # Trigger a bit earlier than visual contact to avoid overlap jitter.
    return max(
        30.0,
        INTERACT_THRESHOLD_BOOST
        * 0.5
        * (
            left_state.radius * left_params.danger_radius
            + right_state.radius * right_params.danger_radius
        ),
    )


def _update_interaction_state(
    state: InteractionState,
    left_state: BodyState,
    right_state: BodyState,
    left_params: BodyParams,
    right_params: BodyParams,
    now_ts: float,
) -> Tuple[float, float, float]:
    """
    Updates interaction activation/deactivation with hysteresis.
    Returns (distance_px, normalized_distance, threshold_px).
    """
    if not (left_state.valid and right_state.valid):
        state.active = False
        state.pair = None
        state.label = "none"
        state.strength = 0.0
        state.close_since_ts = 0.0
        state.locked_pair = None
        return -1.0, -1.0, -1.0

    dx = right_state.x - left_state.x
    dy = right_state.y - left_state.y
    d = math.sqrt(dx * dx + dy * dy)
    threshold = _interaction_threshold_px(left_state, right_state, left_params, right_params)
    norm_d = d / max(1e-4, threshold)

    live_pair = _pair_key(left_params.kind, right_params.kind)
    enter = threshold * INTERACT_ENTER_RATIO
    exit_ = threshold * INTERACT_EXIT_RATIO

    should_activate = d <= enter
    should_deactivate = d >= exit_

    if not state.active:
        if should_activate:
            if state.close_since_ts <= 0.0:
                state.close_since_ts = now_ts
            if (now_ts - state.close_since_ts) >= INTERACT_DWELL_SEC:
                state.active = True
                state.locked_pair = live_pair
                state.pair = live_pair
                state.label = _pair_label(live_pair)
                state.x = 0.5 * (left_state.x + right_state.x)
                state.y = 0.5 * (left_state.y + right_state.y)
                state.radius = 0.62 * (left_state.radius + right_state.radius)
                state.strength = float(np.clip(1.0 - norm_d, 0.0, 1.0))
        else:
            state.close_since_ts = 0.0
    else:
        if should_deactivate:
            state.active = False
            state.pair = None
            state.label = "none"
            state.strength = 0.0
            state.close_since_ts = 0.0
            state.locked_pair = None
        else:
            locked = state.locked_pair or live_pair
            state.pair = locked
            state.label = _pair_label(locked)
            target_x = 0.5 * (left_state.x + right_state.x)
            target_y = 0.5 * (left_state.y + right_state.y)
            target_r = 0.62 * (left_state.radius + right_state.radius)
            target_s = float(np.clip(1.0 - norm_d, 0.0, 1.0))
            state.x = (1.0 - INTERACT_BLEND_ALPHA) * state.x + INTERACT_BLEND_ALPHA * target_x
            state.y = (1.0 - INTERACT_BLEND_ALPHA) * state.y + INTERACT_BLEND_ALPHA * target_y
            state.radius = (1.0 - INTERACT_BLEND_ALPHA) * state.radius + INTERACT_BLEND_ALPHA * target_r
            state.strength = (1.0 - INTERACT_BLEND_ALPHA) * state.strength + INTERACT_BLEND_ALPHA * target_s

    return d, norm_d, threshold


def _draw_interaction_result(
    frame: np.ndarray,
    state: InteractionState,
    earth_renderer: EarthRenderer,
    star_renderer: StarRenderer,
    blackhole_renderer: BlackHoleRenderer,
    phase_main: float,
    phase_cloud: float,
    phase_flare: float,
    phase_disk: float,
) -> None:
    if not state.active or state.pair is None:
        return

    cx = int(state.x)
    cy = int(state.y)
    r = int(max(10.0, state.radius))
    s = float(np.clip(state.strength, 0.0, 1.0))
    pair = state.pair

    # Reusable interaction params.
    p_earth = BodyParams("earth", "Earth", 1.0, 1.0, 1.0, 5778.0, 1.2 + 0.8 * s, 1.5)
    p_star = BodyParams("star", "Merger Star", 6.0, 1.15, 2.6, 12000.0, 1.8 + 0.9 * s, 2.0)
    p_bh = BodyParams("blackhole", "Interaction BH", 16.0, 1.2, 3.2, 0.0, 1.0 + 0.5 * s, 2.8)

    if pair == ("earth", "earth"):
        # Grazing-impact debris field with split rocky cores.
        off = int(0.34 * r)
        earth_renderer.draw(frame, (cx - off, cy), int(r * 0.68), phase_main * 1.2, phase_cloud * 1.2, p_earth)
        earth_renderer.draw(frame, (cx + off, cy), int(r * 0.68), phase_main * 1.35 + 0.7, phase_cloud * 1.1, p_earth)
        for i in range(26):
            a = phase_flare * 2.1 + i * (2.0 * math.pi / 26.0)
            rr = r * (0.95 + 0.50 * ((i % 6) / 5.0))
            ex = int(cx + rr * math.cos(a))
            ey = int(cy + rr * math.sin(a))
            cv2.circle(frame, (ex, ey), max(1, int(1 + 2 * s)), (80, 160, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(r * (1.55 + 0.20 * s)), (90, 180, 255), 1, cv2.LINE_AA)

    elif pair == ("earth", "star"):
        # Planet evaporation plume into a hot stellar photosphere.
        star_renderer.draw(frame, (cx, cy), int(r * 1.05), phase_main * 1.6, phase_flare * 2.0, p_star)
        plume_dir = phase_main * 0.35
        ux = math.cos(plume_dir)
        uy = math.sin(plume_dir)
        for i in range(34):
            u = i / 33.0
            jitter = 0.45 * math.sin(phase_flare * 4.2 + i * 0.6)
            rr = r * (1.0 + 1.8 * u)
            ex = int(cx - rr * ux + jitter * r * uy)
            ey = int(cy - rr * uy - jitter * r * ux)
            cv2.circle(frame, (ex, ey), max(1, int(1 + 2.2 * (1.0 - u))), (120, 205, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(r * 1.42), (140, 230, 255), 2, cv2.LINE_AA)

    elif pair == ("earth", "blackhole"):
        # Spaghettification stream: rocky fragments stretched into the event horizon.
        blackhole_renderer.draw(frame, (cx, cy), int(r * 1.02), phase_main * 1.8, phase_disk * 2.0, p_bh)
        for i in range(36):
            u = i / 35.0
            t = phase_disk * 3.6 + 10.0 * u
            rr = r * (2.2 - 1.7 * u)
            ex = int(cx + rr * math.cos(t))
            ey = int(cy + 0.62 * rr * math.sin(t))
            cv2.circle(frame, (ex, ey), max(1, int(1 + 3 * (1.0 - u))), (95, 185, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(r * 1.32), (95, 170, 255), 2, cv2.LINE_AA)

    elif pair == ("star", "star"):
        # Supernova: compact remnant + bright blast + expanding shell.
        blast_r = int(r * (1.25 + 1.15 * (0.5 + 0.5 * math.sin(phase_main * 4.8))))
        star_renderer.draw(frame, (cx, cy), int(r * 0.46), phase_main * 3.0, phase_flare * 3.1, p_star)
        cv2.circle(frame, (cx, cy), blast_r, (180, 245, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(blast_r * 0.70), (130, 215, 255), 2, cv2.LINE_AA)
        for i in range(30):
            a = phase_flare * 2.3 + i * (2.0 * math.pi / 30.0)
            rr = r * (1.2 + 1.1 * ((i % 7) / 6.0))
            ex = int(cx + rr * math.cos(a))
            ey = int(cy + rr * math.sin(a))
            cv2.circle(frame, (ex, ey), max(1, int(1 + 3 * s)), (145, 225, 255), -1, cv2.LINE_AA)
        # Pulsar-like beams from the remnant.
        beam_len = int(r * (2.1 + 0.6 * s))
        ba = phase_main * 1.9
        bx = int(beam_len * math.cos(ba))
        by = int(beam_len * math.sin(ba))
        cv2.line(frame, (cx - bx, cy - by), (cx + bx, cy + by), (150, 230, 255), 2, cv2.LINE_AA)

    elif pair == ("star", "blackhole"):
        # Tidal disruption + accretion jets.
        blackhole_renderer.draw(frame, (cx, cy), int(r * 1.10), phase_main * 2.0, phase_disk * 2.4, p_bh)
        for i in range(24):
            u = i / 23.0
            t = phase_disk * 3.5 + 8.8 * u
            rr = r * (2.0 - 1.2 * u)
            ex = int(cx + rr * math.cos(t))
            ey = int(cy + 0.55 * rr * math.sin(t))
            cv2.circle(frame, (ex, ey), max(1, int(1 + 2 * (1.0 - u))), (120, 205, 255), -1, cv2.LINE_AA)
        jet_len = int(r * (2.5 + 0.9 * s))
        for sign in (-1, 1):
            p0 = (cx, cy)
            p1 = (cx + int(0.22 * r), cy + sign * jet_len)
            cv2.line(frame, p0, p1, (125, 220, 255), 2 + int(2 * s), cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(r * 1.44), (130, 210, 255), 2, cv2.LINE_AA)

    elif pair == ("blackhole", "blackhole"):
        # BH merger remnant + propagating gravitational-wave ripples.
        blackhole_renderer.draw(frame, (cx, cy), int(r * 1.22), phase_main * 2.4, phase_disk * 2.6, p_bh)
        for k in range(4):
            rr = int(r * (1.25 + 0.42 * k + 0.18 * math.sin(phase_main * 4.6 + k)))
            cv2.circle(frame, (cx, cy), rr, (110, 190, 255), 1, cv2.LINE_AA)
        swirl_a = phase_main * 2.8
        sx = int(cx + 0.52 * r * math.cos(swirl_a))
        sy = int(cy + 0.52 * r * math.sin(swirl_a))
        cv2.circle(frame, (sx, sy), max(2, int(4 + 3 * s)), (130, 210, 255), -1, cv2.LINE_AA)
    else:
        star_renderer.draw(frame, (cx, cy), int(r * 1.1), phase_main, phase_flare, p_star)


def _interaction_overlay(
    frame: np.ndarray,
    left_state: BodyState,
    right_state: BodyState,
    left_params: BodyParams,
    right_params: BodyParams,
    suppress_visuals: bool = False,
) -> Tuple[float, float, float]:
    """
    Returns (distance_px, normalized_distance, potential_proxy).
    """
    if not (left_state.valid and right_state.valid):
        return -1.0, -1.0, 0.0

    dx = right_state.x - left_state.x
    dy = right_state.y - left_state.y
    d = math.sqrt(dx * dx + dy * dy)
    avg_r = max(1.0, 0.5 * (left_state.radius + right_state.radius))
    nd = d / avg_r

    potential = (left_params.mass * right_params.mass) / (d * d + 900.0)

    # Influence link.
    if (not suppress_visuals) and nd < 6.0:
        strength = float(np.clip((7.0 - nd) / 7.0, 0.0, 1.0))
        color = (
            int(85 + 120 * strength),
            int(90 + 95 * strength),
            int(140 + 90 * strength),
        )
        cv2.line(
            frame,
            (int(left_state.x), int(left_state.y)),
            (int(right_state.x), int(right_state.y)),
            color,
            1 + int(1.5 * strength),
            cv2.LINE_AA,
        )

    # Pre-collision warning rings.
    warning = float(np.clip((4.2 - nd) / 2.5, 0.0, 1.0))
    if (not suppress_visuals) and warning > 0.0:
        color = (int(40 + 180 * warning), int(70 + 170 * warning), int(255))
        lrad = int(left_state.radius * (1.10 + 0.15 * warning))
        rrad = int(right_state.radius * (1.10 + 0.15 * warning))
        cv2.circle(frame, (int(left_state.x), int(left_state.y)), lrad, color, 1, cv2.LINE_AA)
        cv2.circle(frame, (int(right_state.x), int(right_state.y)), rrad, color, 1, cv2.LINE_AA)

    return d, nd, potential


def _draw_body_card(frame: np.ndarray, x: int, y: int, title: str, active: bool, params: BodyParams, cycle: HandCycleState) -> None:
    w = 220
    h = 88
    bg = (18, 24, 34)
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 98, 124), 1)

    status = "ON" if active else "OFF"
    status_color = (120, 235, 180) if active else (130, 145, 165)

    cv2.putText(frame, title, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (232, 240, 252), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (x + w - 36, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{params.name}", (x + 8, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (176, 196, 226), 1, cv2.LINE_AA)
    cv2.putText(frame, f"M:{params.mass:.1f}  S:{params.spin:.1f}", (x + 8, y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 190, 220), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"DR:{params.danger_radius:.2f}  Arm:{'Y' if cycle.fist_latched else 'N'}  C:{cycle.curled_fingers}",
        (x + 8, y + 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (165, 185, 214),
        1,
        cv2.LINE_AA,
    )

    meter = float(np.clip(cycle.fist_score_ema, 0.0, 1.0))
    bar_x0 = x + 8
    bar_y0 = y + 74
    bar_w = w - 16
    bar_h = 8
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (55, 70, 96), -1)
    fill = int(bar_w * meter)
    if fill > 0:
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + fill, bar_y0 + bar_h), (90, 205, 255), -1)
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (96, 114, 146), 1)
    cv2.putText(
        frame,
        f"F:{cycle.fist_score_live:0.2f}/{cycle.fist_score_ema:0.2f}",
        (x + w - 84, y + 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (165, 185, 214),
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
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    use_tasks = False
    hands_solution = None
    hand_landmarker = None
    drawer = None
    conn_style = None
    lmk_style = None
    task_timestamp_ms = 0

    if hasattr(mp, "solutions"):
        hands_solution = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
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
            min_hand_detection_confidence=MP_MIN_DETECTION_CONF,
            min_hand_presence_confidence=MP_MIN_PRESENCE_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )
        hand_landmarker = vision.HandLandmarker.create_from_options(options)
        use_tasks = True
    else:
        print("Error: unsupported mediapipe build (expected solutions or tasks.vision).")
        cap.release()
        return 1

    mirror = True
    draw_skeleton = False
    left_state = BodyState()
    right_state = BodyState()
    left_track = HandTrackState()
    right_track = HandTrackState()
    interact_state = InteractionState()
    left_cycle = HandCycleState(object_index=0)
    right_cycle = HandCycleState(object_index=0)

    left_objects = ("earth", "star", "blackhole")
    right_objects = ("earth", "star", "blackhole")

    params_map: Dict[str, BodyParams] = {
        "earth": BodyParams(kind="earth", name="Earth", mass=1.0, radius_scale=1.00, spin=1.0, color_temp=5778.0, emissive=1.0, danger_radius=1.35),
        "star": BodyParams(kind="star", name="Star", mass=3.8, radius_scale=1.02, spin=1.8, color_temp=10200.0, emissive=1.35, danger_radius=1.90),
        "blackhole": BodyParams(kind="blackhole", name="Black Hole", mass=12.0, radius_scale=1.16, spin=2.8, color_temp=0.0, emissive=0.9, danger_radius=2.45),
    }

    global_size_scale = 1.0
    global_lift_scale = 1.5

    earth_renderer = EarthRenderer()
    star_renderer = StarRenderer()
    blackhole_renderer = BlackHoleRenderer()
    bridge_sock: socket.socket | None = None
    if RENDER_BRIDGE_ENABLED:
        try:
            bridge_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            bridge_sock.setblocking(False)
        except OSError:
            bridge_sock = None

    fps = 0.0
    last_ts = time.perf_counter()
    phase_main = 0.0
    phase_cloud = 0.0
    phase_flare = 0.0
    phase_disk = 0.0
    last_norm_dist = -1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        observations: List[HandObservation] = []

        if use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            task_timestamp_ms += 33
            result = hand_landmarker.detect_for_video(mp_image, task_timestamp_ms)
            observations = _extract_task_hands(result, mirror=mirror)
            if (
                draw_skeleton
                and result.hand_landmarks
                and landmark_pb2 is not None
                and hasattr(mp, "solutions")
            ):
                for hand_landmarks in result.hand_landmarks:
                    proto = landmark_pb2.NormalizedLandmarkList(
                        landmark=[
                            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                            for lm in hand_landmarks
                        ]
                    )
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        proto,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )
        else:
            result = hands_solution.process(rgb)
            observations = _extract_solution_hands(result, mirror=mirror)
            if draw_skeleton and result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    drawer.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        lmk_style,
                        conn_style,
                    )

        now_ts = time.perf_counter()
        left_hand, right_hand = _stable_assign_hands(observations, left_track, right_track, now_ts=now_ts)
        _update_track(left_track, left_hand, now_ts)
        _update_track(right_track, right_hand, now_ts)

        near_zone = (last_norm_dist > 0.0 and last_norm_dist < NEAR_FREEZE_NORM_DIST) or interact_state.active
        cycle_frozen = (
            interact_state.active
            or (last_norm_dist > 0.0 and last_norm_dist < CYCLE_FREEZE_NORM_DIST)
        )
        if not cycle_frozen:
            left_switched = _update_hand_cycle(left_hand, left_cycle, now_ts, len(left_objects))
            right_switched = _update_hand_cycle(right_hand, right_cycle, now_ts, len(right_objects))
            # If both switch in the same frame, keep only the stronger fist signal.
            if left_switched and right_switched:
                left_strong = (
                    left_cycle.curled_fingers >= 3
                    and left_cycle.fist_score_live >= (FIST_ON_SCORE + 0.08)
                )
                right_strong = (
                    right_cycle.curled_fingers >= 3
                    and right_cycle.fist_score_live >= (FIST_ON_SCORE + 0.08)
                )
                if left_strong != right_strong:
                    if left_strong:
                        _undo_cycle_switch(right_cycle, len(right_objects))
                    else:
                        _undo_cycle_switch(left_cycle, len(left_objects))
                elif abs(left_cycle.fist_score_ema - right_cycle.fist_score_ema) >= 0.08:
                    if left_cycle.fist_score_ema < right_cycle.fist_score_ema:
                        _undo_cycle_switch(left_cycle, len(left_objects))
                    else:
                        _undo_cycle_switch(right_cycle, len(right_objects))

        left_key = left_objects[left_cycle.object_index]
        right_key = right_objects[right_cycle.object_index]
        left_params = params_map[left_key]
        right_params = params_map[right_key]

        keep_left = left_hand is None and left_track.active and (now_ts - left_track.last_seen_ts) <= HAND_MISSING_GRACE_SEC
        keep_right = right_hand is None and right_track.active and (now_ts - right_track.last_seen_ts) <= HAND_MISSING_GRACE_SEC
        pos_alpha = ANCHOR_ALPHA * (CLOSE_RANGE_ALPHA_SCALE if near_zone else 1.0)
        _update_body_state(
            left_state,
            left_hand,
            frame.shape,
            left_params,
            global_size_scale,
            global_lift_scale,
            keep_previous_on_missing=keep_left,
            position_alpha=pos_alpha,
        )
        _update_body_state(
            right_state,
            right_hand,
            frame.shape,
            right_params,
            global_size_scale,
            global_lift_scale,
            keep_previous_on_missing=keep_right,
            position_alpha=pos_alpha,
        )

        dt = now_ts - last_ts
        last_ts = now_ts
        dt = float(np.clip(dt, 0.0, 0.08))

        phase_main += dt * 1.35
        phase_cloud += dt * 0.78
        phase_flare += dt * 2.6
        phase_disk += dt * 2.15

        dist_px, norm_dist, interact_threshold = _update_interaction_state(
            interact_state, left_state, right_state, left_params, right_params, now_ts
        )
        last_norm_dist = norm_dist
        _send_render_bridge(
            bridge_sock,
            now_ts,
            left_state,
            right_state,
            left_key,
            right_key,
            interact_state,
            frame.shape,
        )

        if interact_state.active:
            _draw_interaction_result(
                frame,
                interact_state,
                earth_renderer,
                star_renderer,
                blackhole_renderer,
                phase_main,
                phase_cloud,
                phase_flare,
                phase_disk,
            )
        else:
            if left_state.valid:
                if left_params.kind == "earth":
                    earth_renderer.draw(
                        frame,
                        (int(left_state.x), int(left_state.y)),
                        int(left_state.radius),
                        phase_main * left_params.spin,
                        phase_cloud * left_params.spin,
                        left_params,
                    )
                elif left_params.kind == "star":
                    star_renderer.draw(
                        frame,
                        (int(left_state.x), int(left_state.y)),
                        int(left_state.radius),
                        phase_main * left_params.spin,
                        phase_flare * left_params.spin,
                        left_params,
                    )
                else:
                    blackhole_renderer.draw(
                        frame,
                        (int(left_state.x), int(left_state.y)),
                        int(left_state.radius),
                        phase_main * left_params.spin,
                        phase_disk * left_params.spin,
                        left_params,
                    )

            if right_state.valid:
                if right_params.kind == "earth":
                    earth_renderer.draw(
                        frame,
                        (int(right_state.x), int(right_state.y)),
                        int(right_state.radius),
                        phase_main * right_params.spin,
                        phase_cloud * right_params.spin,
                        right_params,
                    )
                elif right_params.kind == "star":
                    star_renderer.draw(
                        frame,
                        (int(right_state.x), int(right_state.y)),
                        int(right_state.radius),
                        phase_main * right_params.spin,
                        phase_flare * right_params.spin,
                        right_params,
                    )
                else:
                    blackhole_renderer.draw(
                        frame,
                        (int(right_state.x), int(right_state.y)),
                        int(right_state.radius),
                        phase_main * right_params.spin,
                        phase_disk * right_params.spin,
                        right_params,
                    )

        # Keep proximity metrics + pre-interaction warning overlays.
        _, _, potential = _interaction_overlay(
            frame, left_state, right_state, left_params, right_params, suppress_visuals=interact_state.active
        )

        if dt > 0.0:
            inst = 1.0 / dt
            fps = inst if fps == 0.0 else (0.90 * fps + 0.10 * inst)

        # Header HUD (compact).
        cv2.rectangle(frame, (8, 8), (952, 42), (14, 18, 28), -1)
        summary = (
            f"fps:{fps:5.1f}  hands:{len(observations)}  mirror:{'on' if mirror else 'off'}"
            f"  L:{left_params.name}({'on' if left_state.valid else 'off'})"
            f"  R:{right_params.name}({'on' if right_state.valid else 'off'})"
        )
        cv2.putText(frame, summary, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (215, 232, 252), 1, cv2.LINE_AA)
        if dist_px >= 0.0:
            ia = interact_state.label if interact_state.active else "none"
            interaction_text = (
                f"dist:{dist_px:6.1f}px  nd:{norm_dist:4.2f}  thr:{interact_threshold:6.1f}"
                f"  pot:{potential:5.3f}  fx:{ia}"
            )
        else:
            interaction_text = "dist: --  nd: --  thr: --  pot: --  fx:none"
        cv2.putText(
            frame,
            f"Fist cycles hand object | {interaction_text}",
            (14, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (165, 188, 220),
            1,
            cv2.LINE_AA,
        )

        # Object cards.
        _draw_body_card(
            frame,
            10,
            50,
            "LEFT HAND",
            left_state.valid,
            left_params,
            left_cycle,
        )
        _draw_body_card(
            frame,
            frame.shape[1] - 230,
            50,
            "RIGHT HAND",
            right_state.valid,
            right_params,
            right_cycle,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            mirror = not mirror
        if key == ord("["):
            global_size_scale = max(0.45, global_size_scale - 0.05)
        if key == ord("]"):
            global_size_scale = min(2.6, global_size_scale + 0.05)
        if key == ord("-"):
            global_lift_scale = max(0.3, global_lift_scale - 0.08)
        if key == ord("="):
            global_lift_scale = min(3.0, global_lift_scale + 0.08)

    cap.release()
    cv2.destroyAllWindows()
    if hands_solution is not None:
        hands_solution.close()
    if hand_landmarker is not None:
        hand_landmarker.close()
    if bridge_sock is not None:
        bridge_sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
