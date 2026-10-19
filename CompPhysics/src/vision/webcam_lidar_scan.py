#!/usr/bin/env python3
"""
Pseudo-LiDAR webcam scanner.

This is not real depth sensing. It turns webcam edges and motion into a
rotating scan effect with an echo buffer so the scene feels like a LiDAR HUD.

Controls:
- q / ESC: quit
- f: toggle fullscreen
- m: toggle mirror
- p / SPACE: pause sweep
- c: toggle webcam inset
- r: clear echoes
- [ ]: sweep speed
- , .: beam width
- - =: sensitivity
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"Missing dependency: {exc}. Install with: pip install opencv-python numpy")
    raise SystemExit(1)

try:
    from AppKit import NSScreen
except Exception:
    NSScreen = None


WINDOW_NAME = "Pseudo LiDAR Scan"
CAPTURE_W = 1280
CAPTURE_H = 720
DEFAULT_WINDOW_W = 1440
DEFAULT_WINDOW_H = 900
HUD_MARGIN = 24
PANEL_W = 350
RING_COUNT = 5
SWEEP_TRAIL_DECAY = 0.90
ECHO_DECAY = 0.965
BG_TOP = (7, 13, 12)
BG_BOTTOM = (2, 5, 5)
PANEL_FILL = (10, 18, 17)
PANEL_BORDER = (54, 96, 75)
PANEL_TEXT = (226, 236, 232)
PANEL_MUTED = (146, 174, 161)
RADAR_GRID = (46, 92, 69)
RADAR_RING = (58, 118, 87)
RADAR_GLOW = (124, 255, 178)
RADAR_FRESH = (164, 255, 208)
RADAR_AGED = (66, 168, 116)
RADAR_SWEEP = (150, 255, 192)
RADAR_MOTION = (88, 194, 255)
RADAR_LOCK = (245, 252, 248)


@dataclass
class ScanState:
    sweep_angle: float = math.radians(18.0)
    sweep_speed: float = math.radians(78.0)
    beam_width: float = math.radians(7.0)
    sensitivity: float = 1.0
    paused: bool = False
    mirror: bool = True
    fullscreen: bool = True
    show_inset: bool = True


@dataclass
class Layout:
    width: int
    height: int
    radar_size: int
    radar_x: int
    radar_y: int
    center: Tuple[int, int]
    radius: int
    panel_x: int
    panel_y: int
    panel_w: int
    panel_h: int
    inset_w: int
    inset_h: int


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
    return DEFAULT_WINDOW_W, DEFAULT_WINDOW_H


def _compute_layout(width: int, height: int) -> Layout:
    radar_size = int(min(height - 2 * HUD_MARGIN, width - PANEL_W - 3 * HUD_MARGIN))
    radar_size = max(420, radar_size)
    radar_x = HUD_MARGIN
    radar_y = (height - radar_size) // 2
    panel_x = radar_x + radar_size + HUD_MARGIN
    panel_y = radar_y
    panel_w = width - panel_x - HUD_MARGIN
    panel_h = radar_size

    inset_w = max(260, panel_w - 28)
    inset_w = min(inset_w, panel_w - 28)
    inset_h = int(inset_w * 9 / 16)
    if inset_h > panel_h - 220:
        inset_h = max(180, panel_h - 220)
        inset_w = int(inset_h * 16 / 9)

    return Layout(
        width=width,
        height=height,
        radar_size=radar_size,
        radar_x=radar_x,
        radar_y=radar_y,
        center=(radar_size // 2, radar_size // 2),
        radius=radar_size // 2 - 12,
        panel_x=panel_x,
        panel_y=panel_y,
        panel_w=panel_w,
        panel_h=panel_h,
        inset_w=inset_w,
        inset_h=inset_h,
    )


def _make_maps(radar_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:radar_size, 0:radar_size].astype(np.float32)
    center = (radar_size - 1) * 0.5
    dx = xx - center
    dy = center - yy
    radius = np.sqrt(dx * dx + dy * dy) / max(center, 1.0)
    angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    inside = radius <= 1.0
    return angle, radius, inside


def _vertical_gradient(height: int, width: int, top: Tuple[int, int, int], bottom: Tuple[int, int, int]) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_color = np.asarray(top, dtype=np.float32)
    bottom_color = np.asarray(bottom, dtype=np.float32)
    row = ((1.0 - y) * top_color + y * bottom_color).astype(np.uint8)
    return np.repeat(row[:, None, :], width, axis=1)


def _apply_scanline_texture(image: np.ndarray, line_strength: float = 0.08, grain_strength: float = 0.03) -> np.ndarray:
    textured = image.astype(np.float32).copy()
    line_pattern = (1.0 - line_strength * ((np.arange(image.shape[0]) % 4) == 0).astype(np.float32))[:, None, None]
    textured *= line_pattern
    noise = np.random.normal(0.0, 255.0 * grain_strength, image.shape).astype(np.float32)
    textured += noise
    return np.clip(textured, 0.0, 255.0).astype(np.uint8)


def _build_radar_background(layout: Layout) -> np.ndarray:
    radar = np.zeros((layout.radar_size, layout.radar_size, 3), dtype=np.uint8)
    center = layout.center
    yy, xx = np.mgrid[0 : layout.radar_size, 0 : layout.radar_size].astype(np.float32)
    dx = xx - float(center[0])
    dy = yy - float(center[1])
    radial = np.clip(np.sqrt(dx * dx + dy * dy) / max(float(layout.radius), 1.0), 0.0, 1.0)

    core = np.asarray((10, 24, 18), dtype=np.float32)
    outer = np.asarray((1, 6, 5), dtype=np.float32)
    radar[:] = (((1.0 - radial[..., None]) * core) + (radial[..., None] * outer)).astype(np.uint8)

    for ring in range(RING_COUNT):
        t = (ring + 1) / RING_COUNT
        r = int(layout.radius * t)
        color = (
            int(16 + 22 * t),
            int(48 + 78 * t),
            int(16 + 34 * t),
        )
        cv2.circle(radar, center, r, color, 1, cv2.LINE_AA)

    for deg in range(0, 360, 10):
        angle = math.radians(deg)
        is_major = deg % 30 == 0
        inner = 0 if is_major else int(layout.radius * 0.12)
        x0 = int(center[0] + inner * math.cos(angle))
        y0 = int(center[1] - inner * math.sin(angle))
        x1 = int(center[0] + layout.radius * math.cos(angle))
        y1 = int(center[1] - layout.radius * math.sin(angle))
        cv2.line(radar, (x0, y0), (x1, y1), RADAR_GRID if is_major else (22, 48, 36), 1, cv2.LINE_AA)

    for deg, label in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        angle = math.radians(deg)
        tx = int(center[0] + (layout.radius + 18) * math.cos(angle))
        ty = int(center[1] - (layout.radius + 18) * math.sin(angle))
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
        cv2.putText(
            radar,
            label,
            (tx - size[0] // 2, ty + size[1] // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            PANEL_TEXT,
            1,
            cv2.LINE_AA,
        )

    cv2.circle(radar, center, 5, RADAR_GLOW, -1, cv2.LINE_AA)
    cv2.circle(radar, center, layout.radius, RADAR_RING, 2, cv2.LINE_AA)
    radar = _apply_scanline_texture(radar, line_strength=0.035, grain_strength=0.012)
    return radar


def _make_panel_background(layout: Layout) -> np.ndarray:
    panel = _vertical_gradient(layout.height, layout.width, BG_TOP, BG_BOTTOM)
    cv2.rectangle(
        panel,
        (layout.radar_x - 10, layout.radar_y - 10),
        (layout.radar_x + layout.radar_size + 10, layout.radar_y + layout.radar_size + 10),
        (8, 12, 12),
        -1,
    )
    cv2.rectangle(
        panel,
        (layout.radar_x - 10, layout.radar_y - 10),
        (layout.radar_x + layout.radar_size + 10, layout.radar_y + layout.radar_size + 10),
        PANEL_BORDER,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        panel,
        (layout.panel_x, layout.panel_y),
        (layout.panel_x + layout.panel_w, layout.panel_y + layout.panel_h),
        PANEL_FILL,
        -1,
    )
    cv2.rectangle(
        panel,
        (layout.panel_x, layout.panel_y),
        (layout.panel_x + layout.panel_w, layout.panel_y + layout.panel_h),
        PANEL_BORDER,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        panel,
        (layout.panel_x + 10, layout.panel_y + 84),
        (layout.panel_x + layout.panel_w - 10, layout.panel_y + layout.panel_h - 104),
        (8, 14, 13),
        1,
        cv2.LINE_AA,
    )
    return panel


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def _extract_features(
    frame_bgr: np.ndarray,
    radar_size: int,
    prev_gray_small: np.ndarray | None,
    sensitivity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    small = cv2.resize(frame_bgr, (radar_size, radar_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_low = int(np.clip(74.0 / sensitivity, 22.0, 130.0))
    edge_high = int(np.clip(168.0 / sensitivity, 60.0, 240.0))
    motion_threshold = int(np.clip(24.0 / sensitivity, 7.0, 42.0))

    edges = cv2.Canny(gray, edge_low, edge_high)

    if prev_gray_small is None:
        motion = np.zeros_like(gray)
    else:
        motion_delta = cv2.absdiff(gray, prev_gray_small)
        _, motion = cv2.threshold(motion_delta, motion_threshold, 255, cv2.THRESH_BINARY)

    combined = cv2.max(edges, motion)
    combined = cv2.dilate(combined, np.ones((3, 3), np.uint8), iterations=1)
    strength = cv2.GaussianBlur(combined.astype(np.float32) / 255.0, (0, 0), 1.1)
    return gray, edges, motion, np.clip(strength, 0.0, 1.0)


def _compose_radar(
    static_bg: np.ndarray,
    echo_map: np.ndarray,
    sweep_glow: np.ndarray,
    inside_mask: np.ndarray,
) -> np.ndarray:
    radar = static_bg.astype(np.float32).copy()

    radial_fade = np.where(inside_mask, 1.0, 0.0).astype(np.float32)
    echo = np.clip(echo_map, 0.0, 1.0) * radial_fade
    glow = np.clip(sweep_glow, 0.0, 1.0) * radial_fade

    fresh_echo = np.clip(echo ** 0.82, 0.0, 1.0)
    aged_echo = np.clip(echo ** 1.65, 0.0, 1.0)
    sweep_band = np.clip(glow ** 1.2, 0.0, 1.0)

    radar += aged_echo[..., None] * np.asarray(RADAR_AGED, dtype=np.float32) * 0.55
    radar += fresh_echo[..., None] * np.asarray(RADAR_FRESH, dtype=np.float32) * 0.72
    radar += sweep_band[..., None] * np.asarray(RADAR_SWEEP, dtype=np.float32) * 0.38

    radar = np.clip(radar, 0.0, 255.0)
    bright_pass = np.clip(radar - 128.0, 0.0, 255.0).astype(np.uint8)
    bloom = cv2.GaussianBlur(bright_pass, (0, 0), 3.6)
    radar = cv2.addWeighted(radar.astype(np.uint8), 1.0, bloom, 0.22, 0.0).astype(np.float32)

    yy, xx = np.mgrid[0 : radar.shape[0], 0 : radar.shape[1]].astype(np.float32)
    cx = (radar.shape[1] - 1) * 0.5
    cy = (radar.shape[0] - 1) * 0.5
    vignette = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(min(cx, cy), 1.0)
    vignette = np.clip(1.0 - 0.22 * vignette ** 1.45, 0.72, 1.0)
    radar *= vignette[..., None]
    return np.clip(radar, 0.0, 255.0).astype(np.uint8)


def _draw_nearest_lock(
    radar: np.ndarray,
    hit_mask: np.ndarray,
    radius_map: np.ndarray,
    center: Tuple[int, int],
) -> float | None:
    if not np.any(hit_mask):
        return None

    masked_radius = np.where(hit_mask, radius_map, 10.0)
    index = int(np.argmin(masked_radius))
    y, x = np.unravel_index(index, masked_radius.shape)
    distance = float(masked_radius[y, x])

    cv2.circle(radar, (int(x), int(y)), 8, RADAR_GLOW, 1, cv2.LINE_AA)
    cv2.circle(radar, (int(x), int(y)), 2, RADAR_LOCK, -1, cv2.LINE_AA)
    cv2.line(radar, center, (int(x), int(y)), RADAR_AGED, 1, cv2.LINE_AA)
    return distance


def _draw_sweep_line(
    radar: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    sweep_angle: float,
) -> None:
    end_x = int(center[0] + radius * math.cos(sweep_angle))
    end_y = int(center[1] - radius * math.sin(sweep_angle))
    for scale, alpha in ((1.0, 1.0), (0.94, 0.52), (0.88, 0.24)):
        sx = int(center[0] + radius * scale * math.cos(sweep_angle))
        sy = int(center[1] - radius * scale * math.sin(sweep_angle))
        color = tuple(int(alpha * c) for c in RADAR_SWEEP)
        cv2.line(radar, center, (sx, sy), color, 1 if alpha < 1.0 else 2, cv2.LINE_AA)
    cv2.circle(radar, (end_x, end_y), 6, RADAR_FRESH, 1, cv2.LINE_AA)


def _draw_webcam_inset(
    canvas: np.ndarray,
    frame_bgr: np.ndarray,
    edges: np.ndarray,
    motion: np.ndarray,
    layout: Layout,
) -> None:
    inset_x = layout.panel_x + 14
    inset_y = layout.panel_y + 120

    preview = frame_bgr.copy()
    overlay = preview.copy()
    edge_mask = cv2.resize(edges, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_LINEAR)
    motion_mask = cv2.resize(motion, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay[edge_mask > 0] = RADAR_GLOW
    overlay[motion_mask > 0] = RADAR_MOTION
    cv2.addWeighted(overlay, 0.42, preview, 0.58, 0.0, preview)
    preview = cv2.resize(preview, (layout.inset_w, layout.inset_h), interpolation=cv2.INTER_AREA)
    preview = _apply_scanline_texture(preview, line_strength=0.03, grain_strength=0.01)

    cv2.rectangle(
        canvas,
        (inset_x - 8, inset_y - 8),
        (inset_x + layout.inset_w + 8, inset_y + layout.inset_h + 32),
        (12, 18, 17),
        -1,
    )
    cv2.rectangle(
        canvas,
        (inset_x - 8, inset_y - 8),
        (inset_x + layout.inset_w + 8, inset_y + layout.inset_h + 32),
        PANEL_BORDER,
        1,
        cv2.LINE_AA,
    )
    canvas[inset_y : inset_y + layout.inset_h, inset_x : inset_x + layout.inset_w] = preview
    cv2.circle(canvas, (inset_x + 10, inset_y + layout.inset_h + 18), 5, RADAR_GLOW, -1, cv2.LINE_AA)
    cv2.putText(canvas, "edge", (inset_x + 22, inset_y + layout.inset_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, PANEL_MUTED, 1, cv2.LINE_AA)
    cv2.circle(canvas, (inset_x + 76, inset_y + layout.inset_h + 18), 5, RADAR_MOTION, -1, cv2.LINE_AA)
    cv2.putText(canvas, "motion", (inset_x + 88, inset_y + layout.inset_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, PANEL_MUTED, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "camera source",
        (inset_x + 160, inset_y + layout.inset_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        PANEL_TEXT,
        1,
        cv2.LINE_AA,
    )


def _draw_panel_text(
    canvas: np.ndarray,
    layout: Layout,
    fps: float,
    state: ScanState,
    hit_ratio: float,
    motion_ratio: float,
    nearest_lock: float | None,
    nearest_echo: float | None,
) -> None:
    x = layout.panel_x + 16
    y = layout.panel_y + 34

    def put(line: str, color: Tuple[int, int, int], scale: float = 0.58, dy: int = 28) -> None:
        nonlocal y
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)
        y += dy

    put("Pseudo LiDAR Scanner", PANEL_TEXT, 0.78, 34)
    put("monocular edge + motion sweep", PANEL_MUTED, 0.54, 24)
    y += 6
    put(f"FPS            {fps:5.1f}", PANEL_TEXT)
    put(f"SWEEP DEG/S    {math.degrees(state.sweep_speed):5.1f}", PANEL_TEXT)
    put(f"BEAM WIDTH     {math.degrees(state.beam_width):5.1f}", PANEL_TEXT)
    put(f"SENSITIVITY    {state.sensitivity:5.2f}", PANEL_TEXT)
    put(f"BEAM HITS      {100.0 * hit_ratio:5.1f} %", RADAR_FRESH)
    put(f"MOTION FILL    {100.0 * motion_ratio:5.1f} %", RADAR_MOTION)

    if nearest_lock is None:
        put("CLOSEST BEAM   --", PANEL_MUTED)
    else:
        put(f"CLOSEST BEAM   {(1.0 - nearest_lock) * 100.0:5.1f}", RADAR_FRESH)

    if nearest_echo is None:
        put("CLOSEST ECHO   --", PANEL_MUTED)
    else:
        put(f"CLOSEST ECHO   {(1.0 - nearest_echo) * 100.0:5.1f}", RADAR_AGED)

    put(f"MIRROR         {'ON' if state.mirror else 'OFF'}", PANEL_TEXT)
    put(f"SWEEP          {'PAUSED' if state.paused else 'LIVE'}", PANEL_TEXT)
    put(f"INSET          {'SHOWN' if state.show_inset else 'HIDDEN'}", PANEL_TEXT)

    footer_y = layout.panel_y + layout.panel_h - 84
    cv2.putText(
        canvas,
        "VISUAL CONTROLS",
        (x, footer_y - 16),
        cv2.FONT_HERSHEY_DUPLEX,
        0.54,
        PANEL_TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "[ ] speed   , . beam   - = sensitivity",
        (x, footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        PANEL_MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "[m] mirror   [f] full   [c] inset   [r] clear",
        (x, footer_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        PANEL_MUTED,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "[space] pause   [q] quit",
        (x, footer_y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        PANEL_MUTED,
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    screen_w, screen_h = _get_screen_size()
    state = ScanState()
    layout = _compute_layout(screen_w if state.fullscreen else DEFAULT_WINDOW_W, screen_h if state.fullscreen else DEFAULT_WINDOW_H)
    angle_map, radius_map, inside_mask = _make_maps(layout.radar_size)
    static_radar = _build_radar_background(layout)
    static_panel = _make_panel_background(layout)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (camera index 0).")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if state.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, layout.width, layout.height)

    prev_gray_small: np.ndarray | None = None
    echo_map = np.zeros((layout.radar_size, layout.radar_size), dtype=np.float32)
    sweep_glow = np.zeros_like(echo_map)

    prev_ts = time.perf_counter()
    fps_ema = 60.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read webcam frame.")
                break

            if state.mirror:
                frame = cv2.flip(frame, 1)

            gray_small, edges_small, motion_small, feature_strength = _extract_features(
                frame,
                layout.radar_size,
                prev_gray_small,
                state.sensitivity,
            )
            prev_gray_small = gray_small

            now = time.perf_counter()
            dt = max(now - prev_ts, 1.0 / 240.0)
            prev_ts = now
            fps_ema = 0.92 * fps_ema + 0.08 * (1.0 / dt)

            if not state.paused:
                state.sweep_angle = (state.sweep_angle + state.sweep_speed * dt) % (2.0 * np.pi)

            beam_delta = _angle_diff(angle_map, state.sweep_angle)
            beam_mask = np.clip(1.0 - beam_delta / max(state.beam_width, 1e-4), 0.0, 1.0) ** 1.75
            beam_mask *= inside_mask.astype(np.float32)

            hits = feature_strength * beam_mask
            hit_mask = hits > 0.08

            echo_map *= ECHO_DECAY
            echo_map = np.maximum(echo_map, hits * 1.85)
            echo_map *= inside_mask.astype(np.float32)

            sweep_glow *= SWEEP_TRAIL_DECAY
            sweep_glow = np.maximum(sweep_glow, beam_mask * 0.24)

            radar = _compose_radar(static_radar, echo_map, sweep_glow, inside_mask)
            nearest_lock = _draw_nearest_lock(radar, hit_mask, radius_map, layout.center)
            nearest_echo = _draw_nearest_lock(radar, echo_map > 0.22, radius_map, layout.center)
            _draw_sweep_line(radar, layout.center, layout.radius, state.sweep_angle)

            ring_label_color = (88, 148, 102)
            for ring in range(1, RING_COUNT + 1):
                label = f"{ring * 20:02d}"
                r = int(layout.radius * ring / RING_COUNT)
                cv2.putText(
                    radar,
                    label,
                    (layout.center[0] + 10, layout.center[1] - r + 18),
                cv2.FONT_HERSHEY_DUPLEX,
                0.46,
                PANEL_MUTED,
                1,
                cv2.LINE_AA,
            )

            canvas = static_panel.copy()
            canvas[
                layout.radar_y : layout.radar_y + layout.radar_size,
                layout.radar_x : layout.radar_x + layout.radar_size,
            ] = radar

            cv2.putText(
                canvas,
                "SCAN FIELD",
                (layout.radar_x + 10, layout.radar_y - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.56,
                PANEL_TEXT,
                1,
                cv2.LINE_AA,
            )

            if state.show_inset:
                _draw_webcam_inset(canvas, frame, edges_small, motion_small, layout)

            hit_ratio = float(np.count_nonzero(hit_mask)) / float(np.count_nonzero(inside_mask))
            motion_ratio = float(np.count_nonzero(motion_small)) / float(motion_small.size)
            _draw_panel_text(canvas, layout, fps_ema, state, hit_ratio, motion_ratio, nearest_lock, nearest_echo)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("m"):
                state.mirror = not state.mirror
            elif key == ord("c"):
                state.show_inset = not state.show_inset
            elif key in (ord("p"), ord(" ")):
                state.paused = not state.paused
            elif key == ord("r"):
                echo_map.fill(0.0)
                sweep_glow.fill(0.0)
            elif key == ord("["):
                state.sweep_speed = max(math.radians(18.0), state.sweep_speed - math.radians(10.0))
            elif key == ord("]"):
                state.sweep_speed = min(math.radians(220.0), state.sweep_speed + math.radians(10.0))
            elif key == ord(","):
                state.beam_width = max(math.radians(2.0), state.beam_width - math.radians(1.0))
            elif key == ord("."):
                state.beam_width = min(math.radians(18.0), state.beam_width + math.radians(1.0))
            elif key in (ord("-"), ord("_")):
                state.sensitivity = max(0.45, state.sensitivity - 0.1)
            elif key in (ord("="), ord("+")):
                state.sensitivity = min(2.4, state.sensitivity + 0.1)
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                if state.fullscreen:
                    layout = _compute_layout(screen_w, screen_h)
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    layout = _compute_layout(DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, layout.width, layout.height)

                angle_map, radius_map, inside_mask = _make_maps(layout.radar_size)
                static_radar = _build_radar_background(layout)
                static_panel = _make_panel_background(layout)
                prev_gray_small = None
                echo_map = np.zeros((layout.radar_size, layout.radar_size), dtype=np.float32)
                sweep_glow = np.zeros_like(echo_map)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
