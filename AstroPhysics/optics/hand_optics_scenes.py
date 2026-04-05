from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


CANVAS_W = 1320
CANVAS_H = 860
FIST_COOLDOWN_S = 0.60

SUPPORTED_MODES = {
    "bench": "AR Optics Bench",
    "prism": "Prism Spectroscopy Hands",
    "telescope": "Telescope Hands",
}

BENCH_ELEMENTS = (
    "convex lens",
    "concave lens",
    "mirror",
    "beam splitter",
    "prism",
)

PRISM_MATERIALS = (
    ("BK7 glass", 1.52, 0.12),
    ("flint glass", 1.66, 0.18),
    ("ice crystal", 1.31, 0.08),
)

TELESCOPE_TYPES = (
    "refractor",
    "reflector",
    "cassegrain",
)

SPECTRUM_COLORS = (
    (72, 92, 255),
    (88, 168, 255),
    (96, 232, 232),
    (84, 220, 132),
    (120, 210, 88),
    (176, 180, 70),
    (255, 142, 92),
)

BENCH_PALETTES = (
    ("white beam", (236, 238, 244)),
    ("cyan beam", (120, 220, 255)),
    ("amber beam", (255, 194, 128)),
)

STAR_MODES = (
    "single star",
    "double star",
    "cluster",
)


@dataclass
class SceneHand:
    side: str
    center_norm: np.ndarray
    angle_rad: float
    pinch: bool
    pinch_strength: float
    fist: bool
    span_norm: float
    valid: bool = True


@dataclass
class Segment:
    start: np.ndarray
    end: np.ndarray
    color: Tuple[int, int, int]
    width: int = 2


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float32)


def _perp(vec: np.ndarray) -> np.ndarray:
    return np.asarray([-vec[1], vec[0]], dtype=np.float32)


def _norm_to_px(point: np.ndarray) -> Tuple[int, int]:
    x = int(_clamp(float(point[0]), 0.0, 1.0) * CANVAS_W)
    y = int(_clamp(float(point[1]), 0.0, 1.0) * CANVAS_H)
    return x, y


def _ray_to_bounds(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    ts: List[float] = []
    dx = float(direction[0])
    dy = float(direction[1])
    if abs(dx) > 1e-6:
        tx = (1.0 - origin[0]) / dx if dx > 0.0 else (0.0 - origin[0]) / dx
        if tx > 0.0:
            ts.append(tx)
    if abs(dy) > 1e-6:
        ty = (1.0 - origin[1]) / dy if dy > 0.0 else (0.0 - origin[1]) / dy
        if ty > 0.0:
            ts.append(ty)
    if not ts:
        return origin.copy()
    return origin + direction * (min(ts) * 0.995)


def _intersect_ray_segment(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    half_length: float,
) -> tuple[np.ndarray | None, float]:
    denom = direction[0] * tangent[1] - direction[1] * tangent[0]
    if abs(denom) < 1e-6:
        return None, 0.0

    delta = center - origin
    t_ray = (delta[0] * tangent[1] - delta[1] * tangent[0]) / denom
    s_line = (delta[0] * direction[1] - delta[1] * direction[0]) / denom
    if t_ray <= 0.0 or abs(s_line) > half_length:
        return None, 0.0
    return origin + direction * t_ray, float(s_line)


def _reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = _normalize(normal)
    if float(np.dot(direction, n)) > 0.0:
        n = -n
    return _normalize(direction - 2.0 * float(np.dot(direction, n)) * n)


def _lens_image_point(
    source: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    focal_length: float,
) -> tuple[np.ndarray | None, float]:
    n = _normalize(normal)
    if float(np.dot(source - center, n)) > 0.0:
        n = -n

    u = -float(np.dot(source - center, n))
    if u < 1e-4:
        return None, 0.0
    y_obj = float(np.dot(source - center, tangent))
    inv_v = (1.0 / focal_length) - (1.0 / u)
    if abs(inv_v) < 1e-4:
        return None, 0.0

    v = 1.0 / inv_v
    mag = -v / u
    image = center + n * v + tangent * (mag * y_obj)
    return image.astype(np.float32), float(v)


def _draw_gradient(canvas: np.ndarray, now: float, mode: str) -> None:
    h, w = canvas.shape[:2]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    if mode == "prism":
        b = 18.0 + 40.0 * (1.0 - y) + 18.0 * x
        g = 10.0 + 28.0 * (1.0 - y)
        r = 18.0 + 24.0 * y
    elif mode == "telescope":
        b = 12.0 + 16.0 * (1.0 - y)
        g = 12.0 + 18.0 * (1.0 - y)
        r = 16.0 + 22.0 * x
    else:
        b = 18.0 + 22.0 * (1.0 - y) + 14.0 * x
        g = 12.0 + 20.0 * (1.0 - y)
        r = 16.0 + 18.0 * y

    canvas[..., 0] = np.clip(b, 0, 255).astype(np.uint8)
    canvas[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    canvas[..., 2] = np.clip(r, 0, 255).astype(np.uint8)

    overlay = canvas.copy()
    cv2.circle(
        overlay,
        (int(0.25 * w), int((0.22 + 0.02 * math.sin(0.7 * now)) * h)),
        220,
        (72, 36, 24),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        overlay,
        (int(0.78 * w), int((0.72 + 0.03 * math.cos(0.8 * now)) * h)),
        280,
        (24, 32, 72),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0.0, canvas)


def _draw_grid(canvas: np.ndarray, spacing: int = 64, color: tuple[int, int, int] = (34, 40, 58)) -> None:
    for x in range(0, CANVAS_W, spacing):
        cv2.line(canvas, (x, 0), (x, CANVAS_H), color, 1, cv2.LINE_AA)
    for y in range(0, CANVAS_H, spacing):
        cv2.line(canvas, (0, y), (CANVAS_W, y), color, 1, cv2.LINE_AA)


def _draw_beam_segments(canvas: np.ndarray, segments: Sequence[Segment], glow_alpha: float = 0.22) -> None:
    overlay = canvas.copy()
    for segment in segments:
        p0 = _norm_to_px(segment.start)
        p1 = _norm_to_px(segment.end)
        cv2.line(overlay, p0, p1, segment.color, max(1, segment.width + 5), cv2.LINE_AA)
    cv2.addWeighted(overlay, glow_alpha, canvas, 1.0 - glow_alpha, 0.0, canvas)
    for segment in segments:
        p0 = _norm_to_px(segment.start)
        p1 = _norm_to_px(segment.end)
        cv2.line(canvas, p0, p1, segment.color, segment.width, cv2.LINE_AA)


def _draw_beam_travelers(canvas: np.ndarray, segments: Sequence[Segment], now: float) -> None:
    overlay = canvas.copy()
    for idx, segment in enumerate(segments):
        p0 = segment.start.astype(np.float32)
        p1 = segment.end.astype(np.float32)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length < 0.03:
            continue

        direction = delta / length
        phase = (now * (0.52 + 0.06 * (idx % 5)) + idx * 0.11) % 1.0
        for offset in (0.0, 0.44):
            t = (phase + offset) % 1.0
            point = p0 + direction * (length * t)
            center = _norm_to_px(point)
            radius = max(2, segment.width + 1)
            cv2.circle(overlay, center, radius + 4, segment.color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, (255, 248, 214), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0.0, canvas)


def _draw_hand_marker(canvas: np.ndarray, hand: SceneHand, label: str, accent: tuple[int, int, int]) -> None:
    center = _norm_to_px(hand.center_norm)
    radius = int(20 + 80 * hand.span_norm)
    ring = canvas.copy()
    cv2.circle(ring, center, radius + 14, accent, -1, cv2.LINE_AA)
    cv2.addWeighted(ring, 0.10 if hand.valid else 0.05, canvas, 0.90 if hand.valid else 0.95, 0.0, canvas)
    cv2.circle(canvas, center, radius, accent, 2, cv2.LINE_AA)
    hand_dir = np.asarray([math.cos(hand.angle_rad), math.sin(hand.angle_rad)], dtype=np.float32)
    arrow_end = _norm_to_px(hand.center_norm + hand_dir * (0.06 + 0.18 * hand.span_norm))
    cv2.arrowedLine(canvas, center, arrow_end, accent, 2, cv2.LINE_AA, tipLength=0.30)
    text_color = (228, 234, 246) if hand.valid else (128, 140, 166)
    cv2.putText(canvas, label, (center[0] - 54, center[1] - radius - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_color, 1, cv2.LINE_AA)


def _bench_source_direction(hand: SceneHand) -> np.ndarray:
    if not hand.valid:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    # Treat an upright hand as the neutral pose, then bias emission to the right
    # and damp the steering response so large wrist rotation is not required.
    relative_angle = _wrap_angle(hand.angle_rad + 0.5 * math.pi)
    beam_angle = _clamp(relative_angle * 0.55, -0.90, 0.90)
    return _normalize(np.asarray([math.cos(beam_angle), math.sin(beam_angle)], dtype=np.float32))


def _draw_dashed_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_px: float = 14.0,
    gap_px: float = 10.0,
) -> None:
    p0 = np.asarray(start, dtype=np.float32)
    p1 = np.asarray(end, dtype=np.float32)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return
    direction = delta / length
    offset = 0.0
    while offset < length:
        a = offset
        b = min(length, offset + dash_px)
        pa = tuple(np.round(p0 + direction * a).astype(np.int32))
        pb = tuple(np.round(p0 + direction * b).astype(np.int32))
        cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)
        offset += dash_px + gap_px


def _draw_focus_bloom(
    canvas: np.ndarray,
    point: np.ndarray,
    color: tuple[int, int, int],
    now: float,
    radius_px: int = 30,
) -> None:
    center = _norm_to_px(point)
    glow = canvas.copy()
    cv2.circle(glow, center, radius_px, color, -1, cv2.LINE_AA)
    cv2.circle(glow, center, max(10, radius_px // 2), (255, 240, 188), -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.10, canvas, 0.90, 0.0, canvas)
    for idx in range(4):
        angle = now * 1.2 + 0.8 * idx
        dx = int((radius_px + 8) * math.cos(angle))
        dy = int((radius_px + 8) * math.sin(angle))
        cv2.line(canvas, (center[0] - dx, center[1] - dy), (center[0] + dx, center[1] + dy), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, center, max(6, radius_px // 3), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, center, 4, (255, 246, 210), -1, cv2.LINE_AA)


def _draw_bench_environment(canvas: np.ndarray, now: float) -> None:
    overlay = canvas.copy()
    table_top = int(CANVAS_H * 0.79)
    cv2.rectangle(overlay, (0, table_top), (CANVAS_W, CANVAS_H), (18, 22, 34), -1)
    cv2.rectangle(overlay, (0, table_top + 18), (CANVAS_W, CANVAS_H), (28, 18, 14), -1)
    cv2.addWeighted(overlay, 0.34, canvas, 0.66, 0.0, canvas)

    rail_y = table_top + 22
    cv2.line(canvas, (0, rail_y), (CANVAS_W, rail_y), (96, 106, 126), 3, cv2.LINE_AA)
    cv2.line(canvas, (0, rail_y + 26), (CANVAS_W, rail_y + 26), (66, 76, 94), 2, cv2.LINE_AA)
    for x in range(20, CANVAS_W, 42):
        tick_h = 8 if (x // 42) % 5 else 14
        cv2.line(canvas, (x, rail_y - 4), (x, rail_y + tick_h), (118, 128, 148), 1, cv2.LINE_AA)

    shimmer_x = int((0.18 + 0.64 * (0.5 + 0.5 * math.sin(now * 0.7))) * CANVAS_W)
    shimmer = canvas.copy()
    cv2.rectangle(shimmer, (shimmer_x - 70, table_top + 6), (shimmer_x + 70, table_top + 18), (180, 190, 215), -1)
    cv2.addWeighted(shimmer, 0.08, canvas, 0.92, 0.0, canvas)


def _draw_beam_volume(
    canvas: np.ndarray,
    source: np.ndarray,
    direction: np.ndarray,
    spread: float,
    color: tuple[int, int, int],
    now: float,
) -> None:
    perp = _perp(direction)
    overlay = canvas.copy()
    base_len = 0.72
    for idx, alpha in enumerate((0.14, 0.09, 0.06)):
        width_scale = 1.0 + 0.32 * idx
        pulse = 1.0 + 0.04 * math.sin(now * 2.4 + idx * 0.7)
        left_dir = _normalize(_rotate(direction, -spread * width_scale * pulse))
        right_dir = _normalize(_rotate(direction, spread * width_scale * pulse))
        p0 = source - perp * (0.005 + 0.008 * idx)
        p1 = source + perp * (0.005 + 0.008 * idx)
        p2 = source + right_dir * base_len
        p3 = source + left_dir * base_len
        poly = np.asarray([_norm_to_px(p0), _norm_to_px(p1), _norm_to_px(p2), _norm_to_px(p3)], dtype=np.int32)
        cv2.fillConvexPoly(overlay, poly, color, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, canvas)


def _draw_beam_wavefronts(canvas: np.ndarray, segments: Sequence[Segment], now: float) -> None:
    overlay = canvas.copy()
    for idx, segment in enumerate(segments):
        p0 = segment.start.astype(np.float32)
        p1 = segment.end.astype(np.float32)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length < 0.12:
            continue

        direction = delta / length
        normal = _perp(direction)
        count = 3 if segment.width >= 3 else 2
        for j in range(count):
            phase = (now * (0.34 + 0.03 * (idx % 4)) + 0.22 * j + 0.08 * idx) % 1.0
            if phase < 0.08 or phase > 0.92:
                continue
            center = p0 + direction * (length * phase)
            half = 0.010 + 0.004 * segment.width
            a = _norm_to_px(center - normal * half)
            b = _norm_to_px(center + normal * half)
            cv2.line(overlay, a, b, segment.color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0.0, canvas)


def _draw_optic_glints(canvas: np.ndarray, center: np.ndarray, tangent: np.ndarray, now: float, accent: tuple[int, int, int]) -> None:
    perp = _perp(tangent)
    glow = canvas.copy()
    for phase in (-0.32, 0.0, 0.32):
        offset = tangent * phase * 0.10
        twinkle = 0.5 + 0.5 * math.sin(now * 3.2 + phase * 12.0)
        pt = center + offset + perp * (0.008 * twinkle)
        px = _norm_to_px(pt)
        cv2.circle(glow, px, int(4 + 4 * twinkle), accent, -1, cv2.LINE_AA)
        cv2.line(canvas, (px[0] - 8, px[1]), (px[0] + 8, px[1]), (255, 248, 220), 1, cv2.LINE_AA)
        cv2.line(canvas, (px[0], px[1] - 8), (px[0], px[1] + 8), (255, 248, 220), 1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.10, canvas, 0.90, 0.0, canvas)


def _draw_bench_source_emitter(
    canvas: np.ndarray,
    source: np.ndarray,
    direction: np.ndarray,
    spread: float,
    beam_color: tuple[int, int, int],
    active: bool,
    now: float,
) -> None:
    perp = _perp(direction)
    core = source - direction * 0.050
    tip = source + direction * 0.015
    body_radius = 0.020
    overlay = canvas.copy()
    cone = np.asarray(
        [
            _norm_to_px(core - perp * (body_radius * 0.80)),
            _norm_to_px(core + perp * (body_radius * 0.80)),
            _norm_to_px(tip + perp * (0.018 + spread * 0.08)),
            _norm_to_px(tip - perp * (0.018 + spread * 0.08)),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(overlay, cone, beam_color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.12 if active else 0.06, canvas, 0.88 if active else 0.94, 0.0, canvas)

    core_px = _norm_to_px(core)
    source_px = _norm_to_px(source)
    cv2.circle(canvas, core_px, 20, (36, 42, 54), -1, cv2.LINE_AA)
    cv2.circle(canvas, core_px, 20, (98, 116, 148), 1, cv2.LINE_AA)
    cv2.circle(canvas, core_px, 11, beam_color, 1, cv2.LINE_AA)
    cv2.circle(canvas, source_px, 8, beam_color, -1, cv2.LINE_AA)
    cv2.line(canvas, core_px, source_px, beam_color, 2, cv2.LINE_AA)
    pulse_r = int(18 + 6 * (0.5 + 0.5 * math.sin(now * 4.5)))
    pulse = canvas.copy()
    cv2.circle(pulse, source_px, pulse_r, beam_color, 2, cv2.LINE_AA)
    cv2.addWeighted(pulse, 0.12 if active else 0.06, canvas, 0.88 if active else 0.94, 0.0, canvas)
    for idx in range(3):
        wave_r = int(26 + idx * 14 + 8 * ((now * 1.6 + idx * 0.23) % 1.0))
        cv2.ellipse(pulse, source_px, (wave_r, max(6, int(wave_r * 0.42))), 0.0, -24, 24, beam_color, 1, cv2.LINE_AA)
    cv2.addWeighted(pulse, 0.08 if active else 0.04, canvas, 0.92 if active else 0.96, 0.0, canvas)


def _draw_optic_mount(
    canvas: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    half_length: float,
    focal: float,
) -> None:
    axis_start = center - normal * max(0.40, focal + 0.20)
    axis_end = center + normal * max(0.46, focal + 0.26)
    _draw_dashed_line(canvas, _norm_to_px(axis_start), _norm_to_px(axis_end), (86, 98, 122), 1, 12.0, 10.0)

    center_px = _norm_to_px(center)
    base_y = CANVAS_H - 76
    stand_x = center_px[0]
    cv2.line(canvas, center_px, (stand_x, base_y), (76, 88, 112), 2, cv2.LINE_AA)
    cv2.line(canvas, (stand_x - 54, base_y), (stand_x + 54, base_y), (92, 104, 128), 4, cv2.LINE_AA)
    cap0 = _norm_to_px(center - tangent * (half_length + 0.018))
    cap1 = _norm_to_px(center + tangent * (half_length + 0.018))
    cv2.line(canvas, cap0, cap1, (90, 102, 126), 3, cv2.LINE_AA)


class OpticsHandScene:
    def __init__(self, mode: str) -> None:
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.left_fist_latched = False
        self.right_fist_latched = False
        self.last_left_toggle_ts = 0.0
        self.last_right_toggle_ts = 0.0
        self.source_palette_index = 0
        self.bench_element_index = 0
        self.prism_material_index = 0
        self.telescope_type_index = 0
        self.star_mode_index = 0
        self.last_metrics: Dict[str, str] = {}

    def reset(self) -> None:
        self.left_fist_latched = False
        self.right_fist_latched = False
        self.last_left_toggle_ts = 0.0
        self.last_right_toggle_ts = 0.0
        self.source_palette_index = 0
        self.bench_element_index = 0
        self.prism_material_index = 0
        self.telescope_type_index = 0
        self.star_mode_index = 0
        self.last_metrics = {}

    def _update_toggles(self, hands: Dict[str, SceneHand], now: float) -> None:
        left = hands["left"]
        right = hands["right"]

        if left.valid and left.fist and not self.left_fist_latched and (now - self.last_left_toggle_ts) > FIST_COOLDOWN_S:
            self.left_fist_latched = True
            self.last_left_toggle_ts = now
            if self.mode == "telescope":
                self.star_mode_index = (self.star_mode_index + 1) % len(STAR_MODES)
            else:
                self.source_palette_index = (self.source_palette_index + 1) % len(BENCH_PALETTES)
        elif not left.fist:
            self.left_fist_latched = False

        if right.valid and right.fist and not self.right_fist_latched and (now - self.last_right_toggle_ts) > FIST_COOLDOWN_S:
            self.right_fist_latched = True
            self.last_right_toggle_ts = now
            if self.mode == "bench":
                self.bench_element_index = (self.bench_element_index + 1) % len(BENCH_ELEMENTS)
            elif self.mode == "prism":
                self.prism_material_index = (self.prism_material_index + 1) % len(PRISM_MATERIALS)
            else:
                self.telescope_type_index = (self.telescope_type_index + 1) % len(TELESCOPE_TYPES)
        elif not right.fist:
            self.right_fist_latched = False

    def draw(self, canvas: np.ndarray, hands: Dict[str, SceneHand], now: float, fps: float) -> None:
        self._update_toggles(hands, now)
        _draw_gradient(canvas, now, self.mode)
        _draw_grid(canvas)

        if self.mode == "bench":
            self._draw_bench(canvas, hands, now, fps)
        elif self.mode == "prism":
            self._draw_prism(canvas, hands, fps)
        else:
            self._draw_telescope(canvas, hands, fps)

    def _default_hud(self, canvas: np.ndarray, title: str, subtitle: str, footer: str, stats: Sequence[str], fps: float) -> None:
        cv2.putText(canvas, title, (350, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.02, (236, 240, 248), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (350, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (176, 194, 220), 1, cv2.LINE_AA)

        panel_x = 346
        panel_y = CANVAS_H - 106
        panel_w = 720
        panel_h = 68
        panel = canvas.copy()
        cv2.rectangle(panel, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 16, 24), -1)
        cv2.addWeighted(panel, 0.70, canvas, 0.30, 0.0, canvas)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (86, 98, 122), 1, cv2.LINE_AA)

        x = panel_x + 22
        y = panel_y + 24
        for idx, line in enumerate(stats):
            cv2.putText(canvas, line, (x + (idx % 3) * 228, y + (idx // 3) * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (226, 232, 246), 1, cv2.LINE_AA)

        cv2.putText(canvas, footer, (16, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 170, 198), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"fps {fps:4.1f}", (CANVAS_W - 92, CANVAS_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 170, 198), 1, cv2.LINE_AA)

    def _draw_bench(self, canvas: np.ndarray, hands: Dict[str, SceneHand], now: float, fps: float) -> None:
        left = hands["left"]
        right = hands["right"]
        _draw_bench_environment(canvas, now)
        _draw_hand_marker(canvas, left, "source hand", (120, 220, 255))
        _draw_hand_marker(canvas, right, "optic hand", (255, 186, 120))

        element_name = BENCH_ELEMENTS[self.bench_element_index]
        palette_name, palette_color = BENCH_PALETTES[self.source_palette_index]

        source = left.center_norm
        source_dir = _bench_source_direction(left)
        optic_center = right.center_norm
        tangent = _normalize(np.asarray([math.cos(right.angle_rad), math.sin(right.angle_rad)], dtype=np.float32))
        normal = _perp(tangent)
        aperture = 0.08 + 0.35 * right.span_norm
        focal = 0.12 + 0.58 * _clamp(float(np.linalg.norm(right.center_norm - left.center_norm)), 0.04, 0.80)
        focal *= 0.85 + 0.35 * _clamp(right.pinch_strength, 0.0, 1.0)
        spread = 0.30 - 0.12 * _clamp(left.pinch_strength, 0.0, 1.0)

        _draw_beam_volume(canvas, source, source_dir, spread, palette_color, now)
        _draw_bench_source_emitter(canvas, source, source_dir, spread, palette_color, left.valid, now)
        _draw_optic_mount(canvas, optic_center, tangent, normal, aperture, focal)

        if element_name == "prism":
            self._draw_prism_shape(canvas, optic_center, tangent, aperture, fill=True)
        else:
            self._draw_optic_surface(canvas, optic_center, tangent, aperture, element_name)
        _draw_optic_glints(canvas, optic_center, tangent, now, (188, 228, 255) if "lens" in element_name or element_name == "beam splitter" else (255, 208, 156))

        segments: List[Segment] = []
        virtual_guides: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int]]] = []
        beam_count = 9 if left.valid else 5
        focus_point: np.ndarray | None = None
        image_distance = 0.0
        image_point: np.ndarray | None = None
        focal_plus = optic_center + normal * focal
        focal_minus = optic_center - normal * focal
        for marker in (focal_plus, focal_minus):
            if 0.02 <= float(marker[0]) <= 0.98 and 0.02 <= float(marker[1]) <= 0.98:
                center_px = _norm_to_px(marker)
                cv2.circle(canvas, center_px, 6, (180, 196, 146), 1, cv2.LINE_AA)
                cv2.line(canvas, (center_px[0] - 8, center_px[1]), (center_px[0] + 8, center_px[1]), (180, 196, 146), 1, cv2.LINE_AA)

        focal_signed = 0.0
        if "lens" in element_name:
            focal_signed = focal if element_name == "convex lens" else -focal
            image_point, image_distance = _lens_image_point(source, optic_center, tangent, normal, focal_signed)

        for i in range(beam_count):
            angle = _lerp(-spread, spread, i / max(1, beam_count - 1))
            ray_dir = _normalize(_rotate(source_dir, angle))
            hit, _ = _intersect_ray_segment(source, ray_dir, optic_center, tangent, aperture)
            ray_width = 3 if abs(i - beam_count // 2) <= 1 else 2
            if hit is None:
                segments.append(Segment(source.copy(), _ray_to_bounds(source, ray_dir), palette_color, ray_width))
                continue

            segments.append(Segment(source.copy(), hit.copy(), palette_color, ray_width))

            if element_name == "mirror":
                out_dir = _reflect(ray_dir, normal)
                segments.append(Segment(hit.copy(), _ray_to_bounds(hit, out_dir), (255, 196, 126), ray_width))
            elif element_name == "beam splitter":
                transmit = _ray_to_bounds(hit, ray_dir)
                reflect = _ray_to_bounds(hit, _reflect(ray_dir, normal))
                segments.append(Segment(hit.copy(), transmit, (180, 232, 255), ray_width))
                segments.append(Segment(hit.copy(), reflect, (255, 182, 118), ray_width))
            elif element_name == "prism":
                base_turn = 0.14 + 0.18 * _clamp(right.pinch_strength + right.span_norm, 0.0, 1.0)
                cross = ray_dir[0] * tangent[1] - ray_dir[1] * tangent[0]
                sign = -1.0 if cross >= 0.0 else 1.0
                for idx, color in enumerate(SPECTRUM_COLORS):
                    offset = _lerp(-0.07, 0.09, idx / max(1, len(SPECTRUM_COLORS) - 1))
                    out_dir = _normalize(_rotate(ray_dir, sign * (base_turn + offset)))
                    segments.append(Segment(hit.copy(), _ray_to_bounds(hit, out_dir), color, 2 if idx not in (2, 3, 4) else 3))
            else:
                if image_point is not None:
                    focus_point = image_point
                    if image_distance > 0.0:
                        out_dir = _normalize(image_point - hit)
                    else:
                        out_dir = _normalize(hit - image_point)
                        virtual_guides.append((_norm_to_px(hit), _norm_to_px(image_point), (96, 112, 148)))
                else:
                    out_dir = _normalize(_rotate(ray_dir, 0.06 if focal_signed > 0.0 else -0.06))
                segments.append(Segment(hit.copy(), _ray_to_bounds(hit, out_dir), (120, 226, 255) if focal_signed > 0.0 else (255, 188, 124), ray_width))

        _draw_beam_segments(canvas, segments)
        _draw_beam_wavefronts(canvas, segments, now)
        _draw_beam_travelers(canvas, segments, now)
        for start_px, end_px, color in virtual_guides:
            _draw_dashed_line(canvas, start_px, end_px, color, 1, 10.0, 8.0)

        if focus_point is not None and 0.02 <= float(focus_point[0]) <= 0.98 and 0.02 <= float(focus_point[1]) <= 0.98:
            _draw_focus_bloom(canvas, focus_point, (255, 232, 136), now, 34 if image_distance > 0.0 else 24)

        if image_point is not None and image_distance < 0.0 and 0.02 <= float(image_point[0]) <= 0.98 and 0.02 <= float(image_point[1]) <= 0.98:
            ghost_px = _norm_to_px(image_point)
            cv2.circle(canvas, ghost_px, 10, (144, 156, 188), 1, cv2.LINE_AA)
            _draw_dashed_line(canvas, (ghost_px[0] - 12, ghost_px[1]), (ghost_px[0] + 12, ghost_px[1]), (144, 156, 188), 1, 6.0, 5.0)
            _draw_dashed_line(canvas, (ghost_px[0], ghost_px[1] - 12), (ghost_px[0], ghost_px[1] + 12), (144, 156, 188), 1, 6.0, 5.0)

        stats = [
            f"element  {element_name}",
            f"source  {palette_name}",
            f"focal  {focal:0.2f}",
            f"aperture  {aperture:0.2f}",
            f"beam spread  {spread:0.2f}",
            f"image d  {image_distance:0.2f}",
        ]
        self._default_hud(
            canvas,
            "Floating Optics Hands",
            "Left hand steers the source. Right hand steers the optic. Left fist cycles beam palette, right fist cycles the element.",
            "[m] mirror  [f] full  [r] reset  [q] quit",
            stats,
            fps,
        )

    def _draw_prism(self, canvas: np.ndarray, hands: Dict[str, SceneHand], fps: float) -> None:
        left = hands["left"]
        right = hands["right"]
        _draw_hand_marker(canvas, left, "beam hand", (120, 220, 255))
        _draw_hand_marker(canvas, right, "prism hand", (255, 186, 120))

        material_name, base_index, material_dispersion = PRISM_MATERIALS[self.prism_material_index]
        source = left.center_norm
        beam_dir = _normalize(np.asarray([math.cos(left.angle_rad), math.sin(left.angle_rad)], dtype=np.float32))
        prism_center = right.center_norm
        tangent = _normalize(np.asarray([math.cos(right.angle_rad), math.sin(right.angle_rad)], dtype=np.float32))
        prism_size = 0.10 + 0.34 * right.span_norm
        detector_x = 0.90
        dispersion = material_dispersion + 0.10 * _clamp(float(np.linalg.norm(right.center_norm - left.center_norm)), 0.0, 1.0)
        dispersion += 0.08 * _clamp(right.pinch_strength, 0.0, 1.0)

        self._draw_prism_shape(canvas, prism_center, tangent, prism_size, fill=True)
        screen_color = (82, 96, 120)
        cv2.line(canvas, _norm_to_px(np.asarray([detector_x, 0.18], dtype=np.float32)), _norm_to_px(np.asarray([detector_x, 0.82], dtype=np.float32)), screen_color, 2, cv2.LINE_AA)

        face_center = prism_center - tangent * 0.02
        face_tangent = _rotate(tangent, math.radians(58.0))
        hit, _ = _intersect_ray_segment(source, beam_dir, face_center, face_tangent, prism_size * 0.9)
        segments: List[Segment] = []
        detector_hits: List[Tuple[float, tuple[int, int, int]]] = []
        white_color = BENCH_PALETTES[self.source_palette_index][1]
        if hit is None:
            segments.append(Segment(source.copy(), _ray_to_bounds(source, beam_dir), white_color, 2))
        else:
            segments.append(Segment(source.copy(), hit.copy(), white_color, 2))
            base_turn = 0.08 + 0.22 * dispersion + 0.08 * (base_index - 1.30)
            cross = beam_dir[0] * tangent[1] - beam_dir[1] * tangent[0]
            sign = -1.0 if cross >= 0.0 else 1.0
            for idx, color in enumerate(SPECTRUM_COLORS):
                extra = _lerp(-0.06, 0.12, idx / max(1, len(SPECTRUM_COLORS) - 1))
                out_dir = _normalize(_rotate(beam_dir, sign * (base_turn + extra)))
                end = _ray_to_bounds(hit, out_dir)
                segments.append(Segment(hit.copy(), end, color, 2))
                if abs(float(out_dir[0])) > 1e-5:
                    t = (detector_x - float(hit[0])) / float(out_dir[0])
                    if t > 0.0:
                        detector_y = float(hit[1] + out_dir[1] * t)
                        if 0.18 <= detector_y <= 0.82:
                            detector_hits.append((detector_y, color))

        _draw_beam_segments(canvas, segments)
        for detector_y, color in detector_hits:
            y_px = _norm_to_px(np.asarray([detector_x, detector_y], dtype=np.float32))[1]
            cv2.line(canvas, (int(detector_x * CANVAS_W) + 6, y_px), (int(detector_x * CANVAS_W) + 44, y_px), color, 3, cv2.LINE_AA)

        stats = [
            f"material  {material_name}",
            f"n base  {base_index:0.2f}",
            f"dispersion  {dispersion:0.2f}",
            f"prism size  {prism_size:0.2f}",
            f"detector lines  {len(detector_hits):d}",
            f"source  {BENCH_PALETTES[self.source_palette_index][0]}",
        ]
        self._default_hud(
            canvas,
            "Prism Spectroscopy Hands",
            "Left hand aims the beam. Right hand moves the prism. Left fist cycles source color, right fist cycles prism material.",
            "[m] mirror  [f] full  [r] reset  [q] quit",
            stats,
            fps,
        )

    def _draw_telescope(self, canvas: np.ndarray, hands: Dict[str, SceneHand], fps: float) -> None:
        left = hands["left"]
        right = hands["right"]
        _draw_hand_marker(canvas, left, "detector hand", (120, 220, 255))
        _draw_hand_marker(canvas, right, "scope hand", (255, 186, 120))

        telescope_type = TELESCOPE_TYPES[self.telescope_type_index]
        star_mode = STAR_MODES[self.star_mode_index]
        center = right.center_norm
        axis = _normalize(np.asarray([math.cos(right.angle_rad), math.sin(right.angle_rad)], dtype=np.float32))
        perp = _perp(axis)
        separation = float(np.linalg.norm(left.center_norm - right.center_norm))
        aperture = 0.06 + 0.18 * right.span_norm + 0.08 * separation
        focal = 0.16 + 0.34 * separation
        field_angle = _clamp((float(left.center_norm[1]) - float(right.center_norm[1])) * 1.10, -0.24, 0.24)
        detector_shift = _clamp((float(left.center_norm[0]) - float(right.center_norm[0])) * 0.32, -0.12, 0.12)
        detector_pos = center + axis * (focal + detector_shift)

        self._draw_telescope_body(canvas, center, axis, perp, aperture, telescope_type)

        incoming_dir = _normalize(axis + perp * field_angle)
        incoming_start = center - axis * 0.38
        ray_offsets = np.linspace(-aperture, aperture, 7, dtype=np.float32)
        segments: List[Segment] = []
        blur_metric = 0.0
        star_points = self._star_offsets(star_mode)
        for star_offset in star_points:
            star_axis = _normalize(incoming_dir + perp * star_offset)
            for offset in ray_offsets:
                start = incoming_start + perp * offset - star_axis * 0.24
                if telescope_type == "refractor":
                    lens_center = center - axis * 0.10
                    hit, _ = _intersect_ray_segment(start, star_axis, lens_center, perp, aperture * 0.95)
                    if hit is None:
                        continue
                    focus = center + axis * focal + perp * (field_angle + star_offset) * focal * 1.2
                    segments.append(Segment(start.copy(), hit.copy(), (190, 210, 255), 2))
                    segments.append(Segment(hit.copy(), _ray_to_bounds(hit, _normalize(focus - hit)), (120, 226, 255), 2))
                    blur_metric += float(np.linalg.norm(detector_pos - focus))
                elif telescope_type == "reflector":
                    mirror_center = center + axis * 0.16
                    hit = mirror_center + perp * offset * 0.92
                    focus = center - axis * (0.08 + focal * 0.52) + perp * (field_angle + star_offset) * focal * 0.9
                    segments.append(Segment(start.copy(), hit.copy(), (190, 210, 255), 2))
                    segments.append(Segment(hit.copy(), focus.copy(), (255, 192, 128), 2))
                    blur_metric += float(np.linalg.norm(detector_pos - focus))
                else:
                    primary = center + axis * 0.18 + perp * offset * 0.92
                    secondary = center - axis * 0.02 + perp * ((field_angle + star_offset) * 0.12)
                    focus = center + axis * (0.02 + focal * 0.34) + perp * (field_angle + star_offset) * focal * 0.65
                    segments.append(Segment(start.copy(), primary.copy(), (190, 210, 255), 2))
                    segments.append(Segment(primary.copy(), secondary.copy(), (255, 192, 128), 2))
                    segments.append(Segment(secondary.copy(), focus.copy(), (120, 226, 255), 2))
                    blur_metric += float(np.linalg.norm(detector_pos - focus))

        _draw_beam_segments(canvas, segments, glow_alpha=0.18)

        detector_px = _norm_to_px(detector_pos)
        cv2.line(
            canvas,
            _norm_to_px(detector_pos - perp * (aperture * 0.70)),
            _norm_to_px(detector_pos + perp * (aperture * 0.70)),
            (118, 196, 116),
            2,
            cv2.LINE_AA,
        )

        blur_metric /= max(1, len(ray_offsets) * max(1, len(star_points)))
        focus_quality = math.exp(-9.0 * blur_metric)
        blur_radius = int(12 + 40 * (1.0 - focus_quality))
        cv2.circle(canvas, detector_px, blur_radius, (120, 224, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, detector_px, max(2, blur_radius // 4), (255, 232, 140), -1, cv2.LINE_AA)

        stats = [
            f"scope  {telescope_type}",
            f"star field  {star_mode}",
            f"aperture  {aperture:0.2f}",
            f"focal  {focal:0.2f}",
            f"detector shift  {detector_shift:0.2f}",
            f"focus  {100.0 * focus_quality:3.0f}%",
        ]
        self._default_hud(
            canvas,
            "Telescope Hands",
            "Right hand moves the telescope. Left hand moves the detector and field angle. Left fist cycles star field, right fist cycles telescope type.",
            "[m] mirror  [f] full  [r] reset  [q] quit",
            stats,
            fps,
        )

    def _draw_optic_surface(self, canvas: np.ndarray, center: np.ndarray, tangent: np.ndarray, half_length: float, name: str) -> None:
        normal = _perp(tangent)
        if "lens" in name:
            color = (120, 220, 255)
            bulge = (0.020 + 0.026 * half_length) * (1.0 if name == "convex lens" else -0.82)
            thickness = 0.015
            samples = np.linspace(-1.0, 1.0, 28, dtype=np.float32)
            face0 = []
            face1 = []
            for s in samples:
                profile = 1.0 - s * s
                centerline = center + tangent * (half_length * s)
                offset = normal * (bulge * profile)
                face0.append(centerline - normal * thickness + offset)
                face1.append(centerline + normal * thickness - offset)
            poly = np.asarray([_norm_to_px(p) for p in face0 + face1[::-1]], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, poly, (56, 98, 142), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0.0, canvas)
            cv2.polylines(canvas, [np.asarray([_norm_to_px(p) for p in face0], dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.polylines(canvas, [np.asarray([_norm_to_px(p) for p in face1], dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.line(canvas, _norm_to_px(center - tangent * half_length), _norm_to_px(center + tangent * half_length), (86, 138, 182), 1, cv2.LINE_AA)
        elif name == "mirror":
            samples = np.linspace(-1.0, 1.0, 28, dtype=np.float32)
            sag = 0.026 + 0.022 * half_length
            front = []
            back = []
            for s in samples:
                centerline = center + tangent * (half_length * s)
                curve = normal * (sag * (s * s - 1.0))
                front.append(centerline + curve)
                back.append(centerline + curve + normal * 0.018)
            poly = np.asarray([_norm_to_px(p) for p in front + back[::-1]], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, poly, (92, 74, 54), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0.0, canvas)
            cv2.polylines(canvas, [np.asarray([_norm_to_px(p) for p in front], dtype=np.int32)], False, (255, 184, 120), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [np.asarray([_norm_to_px(p) for p in back], dtype=np.int32)], False, (136, 110, 84), 2, cv2.LINE_AA)
        elif name == "beam splitter":
            thickness = 0.012
            corners = np.asarray(
                [
                    _norm_to_px(center - tangent * half_length - normal * thickness),
                    _norm_to_px(center - tangent * half_length + normal * thickness),
                    _norm_to_px(center + tangent * half_length + normal * thickness),
                    _norm_to_px(center + tangent * half_length - normal * thickness),
                ],
                dtype=np.int32,
            )
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, corners, (78, 96, 132), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0.0, canvas)
            cv2.polylines(canvas, [corners], True, (188, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, _norm_to_px(center), 8, (255, 230, 140), 1, cv2.LINE_AA)

    def _draw_prism_shape(self, canvas: np.ndarray, center: np.ndarray, tangent: np.ndarray, size: float, fill: bool) -> None:
        normal = _perp(tangent)
        p0 = center + tangent * size
        p1 = center - tangent * size + normal * (size * 0.72)
        p2 = center - tangent * size - normal * (size * 0.72)
        poly = np.asarray([_norm_to_px(p0), _norm_to_px(p1), _norm_to_px(p2)], dtype=np.int32)
        if fill:
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, poly, (58, 76, 116), cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0.0, canvas)
        cv2.polylines(canvas, [poly], True, (176, 220, 255), 2, cv2.LINE_AA)

    def _draw_telescope_body(
        self,
        canvas: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        perp: np.ndarray,
        aperture: float,
        telescope_type: str,
    ) -> None:
        tube_half = 0.18
        c0 = center - axis * tube_half - perp * aperture
        c1 = center - axis * tube_half + perp * aperture
        c2 = center + axis * tube_half + perp * aperture
        c3 = center + axis * tube_half - perp * aperture
        poly = np.asarray([_norm_to_px(c0), _norm_to_px(c1), _norm_to_px(c2), _norm_to_px(c3)], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, poly, (26, 32, 44), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0.0, canvas)
        cv2.polylines(canvas, [poly], True, (110, 126, 152), 2, cv2.LINE_AA)

        front = center - axis * tube_half
        back = center + axis * tube_half
        cv2.line(canvas, _norm_to_px(front - perp * aperture), _norm_to_px(front + perp * aperture), (170, 210, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, _norm_to_px(back - perp * aperture), _norm_to_px(back + perp * aperture), (255, 192, 120), 2, cv2.LINE_AA)
        if telescope_type == "cassegrain":
            secondary = center - axis * 0.02
            cv2.line(canvas, _norm_to_px(secondary - perp * aperture * 0.26), _norm_to_px(secondary + perp * aperture * 0.26), (255, 222, 166), 2, cv2.LINE_AA)

    def _star_offsets(self, star_mode: str) -> List[float]:
        if star_mode == "single star":
            return [0.0]
        if star_mode == "double star":
            return [-0.045, 0.045]
        return [-0.08, -0.03, 0.0, 0.04, 0.08]
