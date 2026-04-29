#!/usr/bin/env python3
"""
Pseudo thermal camera.

This is not a real infrared sensor. It uses webcam brightness, motion,
and optional hand landmarks to synthesize a thermal-looking field with
diffusion, upward drift, and a false-color palette.

Controls:
- q / ESC: quit
- f: toggle fullscreen
- m: toggle mirror
- c: cycle thermal palette
- h: toggle hand heat injection
- i: toggle webcam inset
- p / SPACE: pause thermal evolution
- r: reset thermal field
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = os.path.join(tempfile.gettempdir(), "mplconfig")
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_dir

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


WINDOW_NAME = "Pseudo Thermal Camera"
CAPTURE_W = 1280
CAPTURE_H = 720
DEFAULT_WINDOW_W = 1480
DEFAULT_WINDOW_H = 920
FIELD_W = 240
FIELD_H = 160
HUD_MARGIN = 24
PANEL_W = 330
MODULE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES: Tuple[Path, ...] = (
    MODULE_DIR.parent / "vision" / "models" / "hand_landmarker.task",
    MODULE_DIR.parent / "DefensiveSys" / "models" / "hand_landmarker.task",
)
MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[0])

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
TIP_IDS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass(frozen=True)
class PaletteSpec:
    name: str
    stops: Tuple[Tuple[float, Tuple[int, int, int]], ...]


PALETTES: Tuple[PaletteSpec, ...] = (
    PaletteSpec(
        name="IRON",
        stops=(
            (0.00, (8, 4, 12)),
            (0.18, (42, 18, 84)),
            (0.40, (22, 86, 190)),
            (0.62, (58, 176, 255)),
            (0.80, (82, 232, 255)),
            (0.92, (168, 252, 255)),
            (1.00, (236, 255, 255)),
        ),
    ),
    PaletteSpec(
        name="WHITE HOT",
        stops=(
            (0.00, (6, 8, 10)),
            (0.25, (34, 38, 46)),
            (0.50, (84, 98, 126)),
            (0.72, (150, 178, 212)),
            (0.90, (210, 228, 248)),
            (1.00, (255, 255, 255)),
        ),
    ),
    PaletteSpec(
        name="EMBER",
        stops=(
            (0.00, (2, 8, 18)),
            (0.16, (0, 22, 52)),
            (0.36, (0, 70, 132)),
            (0.58, (18, 152, 232)),
            (0.76, (64, 212, 255)),
            (0.90, (148, 244, 255)),
            (1.00, (232, 255, 255)),
        ),
    ),
)


@dataclass
class SceneState:
    palette_index: int = 0
    paused: bool = False
    mirror: bool = True
    fullscreen: bool = True
    show_hands: bool = True
    show_inset: bool = True


@dataclass
class Layout:
    width: int
    height: int
    view_x: int
    view_y: int
    view_w: int
    view_h: int
    panel_x: int
    panel_y: int
    panel_w: int
    panel_h: int
    inset_x: int
    inset_y: int
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
    view_x = HUD_MARGIN
    view_y = HUD_MARGIN
    panel_x = width - PANEL_W - HUD_MARGIN
    panel_y = HUD_MARGIN
    panel_w = PANEL_W
    panel_h = height - 2 * HUD_MARGIN
    view_w = panel_x - 2 * HUD_MARGIN
    view_h = height - 2 * HUD_MARGIN
    inset_w = panel_w - 28
    inset_h = int(inset_w * 9 / 16)
    inset_x = panel_x + 14
    inset_y = panel_y + panel_h - inset_h - 34
    return Layout(
        width=width,
        height=height,
        view_x=view_x,
        view_y=view_y,
        view_w=view_w,
        view_h=view_h,
        panel_x=panel_x,
        panel_y=panel_y,
        panel_w=panel_w,
        panel_h=panel_h,
        inset_x=inset_x,
        inset_y=inset_y,
        inset_w=inset_w,
        inset_h=inset_h,
    )


def _build_palette_lut(spec: PaletteSpec) -> np.ndarray:
    xp = np.array([stop[0] * 255.0 for stop in spec.stops], dtype=np.float32)
    colors = np.array([stop[1] for stop in spec.stops], dtype=np.float32)
    x = np.arange(256, dtype=np.float32)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, 0, channel] = np.clip(
            np.interp(x, xp, colors[:, channel]),
            0.0,
            255.0,
        ).astype(np.uint8)
    return lut


def _vertical_gradient(height: int, width: int, top: Tuple[int, int, int], bottom: Tuple[int, int, int]) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_color = np.asarray(top, dtype=np.float32)
    bottom_color = np.asarray(bottom, dtype=np.float32)
    row = ((1.0 - y) * top_color + y * bottom_color).astype(np.uint8)
    return np.repeat(row[:, None, :], width, axis=1)


def _add_gaussian(field: np.ndarray, cx: float, cy: float, sigma: float, amount: float) -> None:
    if sigma <= 0.0 or amount <= 0.0:
        return
    radius = int(max(6.0, sigma * 3.2))
    x0 = max(0, int(cx) - radius)
    x1 = min(field.shape[1], int(cx) + radius + 1)
    y0 = max(0, int(cy) - radius)
    y1 = min(field.shape[0], int(cy) + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    gaussian = amount * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    field[y0:y1, x0:x1] += gaussian.astype(np.float32)


def _extract_hand_landmarks(
    result: object,
    field_w: int,
    field_h: int,
) -> List[List[Tuple[float, float]]]:
    hands: List[List[Tuple[float, float]]] = []
    if not result.multi_hand_landmarks:
        return hands
    for landmarks in result.multi_hand_landmarks:
        pts: List[Tuple[float, float]] = []
        for lm in landmarks.landmark:
            pts.append((lm.x * field_w, lm.y * field_h))
        hands.append(pts)
    return hands


def _extract_task_hand_landmarks(
    result: object,
    field_w: int,
    field_h: int,
) -> List[List[Tuple[float, float]]]:
    hands: List[List[Tuple[float, float]]] = []
    if not getattr(result, "hand_landmarks", None):
        return hands
    for landmarks in result.hand_landmarks:
        pts: List[Tuple[float, float]] = []
        for lm in landmarks:
            pts.append((lm.x * field_w, lm.y * field_h))
        hands.append(pts)
    return hands


def _inject_hand_heat(hand_sets: Sequence[Sequence[Tuple[float, float]]], heat: np.ndarray) -> int:
    if not hand_sets:
        return 0
    for hand in hand_sets:
        palm_x = 0.55 * hand[WRIST][0] + 0.45 * hand[MIDDLE_TIP][0]
        palm_y = 0.60 * hand[WRIST][1] + 0.40 * hand[MIDDLE_TIP][1]
        _add_gaussian(heat, palm_x, palm_y, 8.8, 0.64)
        for idx in TIP_IDS:
            _add_gaussian(heat, hand[idx][0], hand[idx][1], 5.0, 0.54)
    return len(hand_sets)


def _make_inset(frame_bgr: np.ndarray, hand_result: object, layout: Layout) -> np.ndarray:
    preview = frame_bgr.copy()
    if getattr(hand_result, "multi_hand_landmarks", None):
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                preview,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
    preview = cv2.resize(preview, (layout.inset_w, layout.inset_h), interpolation=cv2.INTER_AREA)
    return preview


def _draw_hud(
    canvas: np.ndarray,
    layout: Layout,
    palette: PaletteSpec,
    lut: np.ndarray,
    fps: float,
    motion_ratio: float,
    hands_seen: int,
    hotspot_level: float,
    ambient_level: float,
    state: SceneState,
) -> None:
    x = layout.panel_x + 18
    y = layout.panel_y + 38

    def put(text: str, color: Tuple[int, int, int], scale: float = 0.60, dy: int = 28) -> None:
        nonlocal y
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)
        y += dy

    panel_fill = (10, 14, 18)
    panel_border = (54, 74, 92)
    text_primary = (232, 238, 246)
    text_muted = (156, 174, 194)
    accent = (255, 218, 170)
    motion_color = (132, 206, 255)

    cv2.rectangle(
        canvas,
        (layout.panel_x, layout.panel_y),
        (layout.panel_x + layout.panel_w, layout.panel_y + layout.panel_h),
        panel_fill,
        -1,
    )
    cv2.rectangle(
        canvas,
        (layout.panel_x, layout.panel_y),
        (layout.panel_x + layout.panel_w, layout.panel_y + layout.panel_h),
        panel_border,
        1,
        cv2.LINE_AA,
    )

    put("Pseudo Thermal Camera", text_primary, 0.74, 34)
    put("brightness + motion + hand heat", text_muted, 0.52, 24)
    y += 6

    hotspot_temp = 18.0 + 402.0 * hotspot_level
    ambient_temp = 18.0 + 62.0 * ambient_level

    put(f"PALETTE       {palette.name}", accent)
    put(f"FPS           {fps:5.1f}", text_primary)
    put(f"MOTION        {100.0 * motion_ratio:5.1f} %", motion_color)
    put(f"HANDS         {hands_seen}", text_primary)
    put(f"HOTSPOT       {hotspot_temp:6.1f} C", accent)
    put(f"AMBIENT       {ambient_temp:6.1f} C", text_primary)
    put(f"MIRROR        {'ON' if state.mirror else 'OFF'}", text_primary)
    put(f"HAND HEAT     {'ON' if state.show_hands else 'OFF'}", text_primary)
    put(f"INSET         {'ON' if state.show_inset else 'OFF'}", text_primary)
    put(f"FIELD         {'PAUSED' if state.paused else 'LIVE'}", text_primary)

    legend_x = layout.panel_x + layout.panel_w - 70
    legend_y0 = layout.panel_y + 48
    legend_y1 = layout.panel_y + 276
    legend = cv2.resize(lut, (18, legend_y1 - legend_y0), interpolation=cv2.INTER_LINEAR)
    legend = np.flipud(legend)
    canvas[legend_y0:legend_y1, legend_x:legend_x + 18] = legend
    cv2.rectangle(canvas, (legend_x - 1, legend_y0 - 1), (legend_x + 19, legend_y1 + 1), panel_border, 1, cv2.LINE_AA)
    cv2.putText(canvas, "420C", (legend_x - 48, legend_y0 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "18C", (legend_x - 34, legend_y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_muted, 1, cv2.LINE_AA)

    footer_y = layout.panel_y + layout.panel_h - 96
    cv2.putText(canvas, "CONTROLS", (x, footer_y), cv2.FONT_HERSHEY_DUPLEX, 0.54, text_primary, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[c] palette   [h] hands   [i] inset", (x, footer_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[m] mirror   [f] full   [r] reset", (x, footer_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_muted, 1, cv2.LINE_AA)
    cv2.putText(canvas, "[space] pause   [q] quit", (x, footer_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.50, text_muted, 1, cv2.LINE_AA)


def main() -> int:
    screen_w, screen_h = _get_screen_size()
    state = SceneState()
    layout = _compute_layout(screen_w if state.fullscreen else DEFAULT_WINDOW_W, screen_h if state.fullscreen else DEFAULT_WINDOW_H)

    luts = tuple(_build_palette_lut(spec) for spec in PALETTES)

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

    heat_field = np.zeros((FIELD_H, FIELD_W), dtype=np.float32)
    prev_gray_small: np.ndarray | None = None
    prev_ts = time.perf_counter()
    fps_ema = 60.0

    use_tasks = False
    tracker_solution = None
    tracker_tasks = None
    task_timestamp_ms = 0

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        tracker_solution = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
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

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read webcam frame.")
                break

            if state.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(now - prev_ts, 1.0 / 240.0)
            prev_ts = now
            fps_ema = 0.92 * fps_ema + 0.08 * (1.0 / dt)

            small = cv2.resize(frame, (FIELD_W, FIELD_H), interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.GaussianBlur(gray_small, (5, 5), 0)
            gray_norm = cv2.GaussianBlur(gray_small.astype(np.float32) / 255.0, (0, 0), 1.0)

            if prev_gray_small is None:
                motion_norm = np.zeros_like(gray_norm)
            else:
                motion_delta = cv2.absdiff(gray_small, prev_gray_small).astype(np.float32) / 255.0
                motion_norm = cv2.GaussianBlur(np.clip((motion_delta - 0.05) * 3.8, 0.0, 1.0), (0, 0), 1.2)
            prev_gray_small = gray_small

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if use_tasks:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hand_result = tracker_tasks.detect_for_video(mp_image, task_timestamp_ms)
                task_timestamp_ms += max(1, int(dt * 1000.0))
                hand_sets = _extract_task_hand_landmarks(hand_result, FIELD_W, FIELD_H)
            else:
                hand_result = tracker_solution.process(rgb)
                hand_sets = _extract_hand_landmarks(hand_result, FIELD_W, FIELD_H)

            brightness_source = np.clip((gray_norm - 0.08) * 1.18, 0.0, 1.0) ** 1.35
            motion_source = np.clip(motion_norm * 1.85, 0.0, 1.0) ** 1.18
            source = 0.28 * brightness_source + 0.92 * motion_source

            hand_heat = np.zeros_like(heat_field)
            hands_seen = _inject_hand_heat(hand_sets, hand_heat) if state.show_hands else 0
            source += hand_heat
            source = np.clip(source, 0.0, 1.6)

            if not state.paused:
                diffused = cv2.GaussianBlur(heat_field, (0, 0), 1.35)
                risen = np.roll(diffused, -1, axis=0)
                risen[-1, :] = 0.0

                heat_field = 0.82 * heat_field + 0.10 * diffused + 0.08 * risen
                heat_field *= 0.987
                heat_field += source * 0.52
                heat_field = np.clip(heat_field, 0.0, 2.4)

            thermal_level = 1.0 - np.exp(-1.32 * np.clip(heat_field, 0.0, None))
            thermal_level = np.clip(thermal_level, 0.0, 1.0)

            thermal_u8 = (thermal_level * 255.0).astype(np.uint8)
            colorized = cv2.applyColorMap(thermal_u8, luts[state.palette_index])
            colorized = cv2.GaussianBlur(colorized, (0, 0), 0.8)

            bright_pass = np.clip(thermal_u8.astype(np.float32) - 154.0, 0.0, 255.0).astype(np.uint8)
            bloom_mask = cv2.GaussianBlur(bright_pass, (0, 0), 5.0)
            bloom = cv2.applyColorMap(bloom_mask, luts[state.palette_index])
            colorized = cv2.addWeighted(colorized, 1.0, bloom, 0.22, 0.0)

            contours = []
            for level in (110, 160, 210):
                mask = (thermal_u8 > level).astype(np.uint8) * 255
                found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.extend(found)
            contour_layer = np.zeros_like(colorized)
            if contours:
                cv2.drawContours(contour_layer, contours, -1, (240, 248, 255), 1, cv2.LINE_AA)
                colorized = cv2.addWeighted(colorized, 1.0, contour_layer, 0.18, 0.0)

            thermal_view = cv2.resize(colorized, (layout.view_w, layout.view_h), interpolation=cv2.INTER_CUBIC)
            grid_color = (24, 40, 54)
            for ix in range(0, layout.view_w, 64):
                cv2.line(thermal_view, (ix, 0), (ix, layout.view_h), grid_color, 1, cv2.LINE_AA)
            for iy in range(0, layout.view_h, 64):
                cv2.line(thermal_view, (0, iy), (layout.view_w, iy), grid_color, 1, cv2.LINE_AA)

            canvas = _vertical_gradient(layout.height, layout.width, (6, 10, 16), (2, 4, 7))
            canvas[
                layout.view_y : layout.view_y + layout.view_h,
                layout.view_x : layout.view_x + layout.view_w,
            ] = thermal_view
            cv2.rectangle(
                canvas,
                (layout.view_x - 10, layout.view_y - 10),
                (layout.view_x + layout.view_w + 10, layout.view_y + layout.view_h + 10),
                (18, 24, 32),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                canvas,
                "THERMAL FIELD",
                (layout.view_x + 8, layout.view_y - 10),
                cv2.FONT_HERSHEY_DUPLEX,
                0.60,
                (236, 240, 248),
                1,
                cv2.LINE_AA,
            )

            if state.show_inset:
                if use_tasks:
                    preview = cv2.resize(frame, (layout.inset_w, layout.inset_h), interpolation=cv2.INTER_AREA)
                else:
                    preview = _make_inset(frame, hand_result, layout)
                cv2.rectangle(
                    canvas,
                    (layout.inset_x - 8, layout.inset_y - 8),
                    (layout.inset_x + layout.inset_w + 8, layout.inset_y + layout.inset_h + 32),
                    (10, 14, 18),
                    -1,
                )
                cv2.rectangle(
                    canvas,
                    (layout.inset_x - 8, layout.inset_y - 8),
                    (layout.inset_x + layout.inset_w + 8, layout.inset_y + layout.inset_h + 32),
                    (56, 76, 96),
                    1,
                    cv2.LINE_AA,
                )
                canvas[
                    layout.inset_y : layout.inset_y + layout.inset_h,
                    layout.inset_x : layout.inset_x + layout.inset_w,
                ] = preview
                cv2.putText(
                    canvas,
                    "webcam reference",
                    (layout.inset_x + 8, layout.inset_y + layout.inset_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (176, 188, 204),
                    1,
                    cv2.LINE_AA,
                )

            hotspot_level = float(thermal_level.max())
            ambient_level = float(np.percentile(thermal_level, 40.0))
            motion_ratio = float(np.count_nonzero(motion_norm > 0.08)) / float(motion_norm.size)
            _draw_hud(
                canvas,
                layout,
                PALETTES[state.palette_index],
                luts[state.palette_index],
                fps_ema,
                motion_ratio,
                hands_seen,
                hotspot_level,
                ambient_level,
                state,
            )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key == ord("m"):
                state.mirror = not state.mirror
                prev_gray_small = None
            elif key == ord("c"):
                state.palette_index = (state.palette_index + 1) % len(PALETTES)
            elif key == ord("h"):
                state.show_hands = not state.show_hands
            elif key == ord("i"):
                state.show_inset = not state.show_inset
            elif key in (ord("p"), ord(" ")):
                state.paused = not state.paused
            elif key == ord("r"):
                heat_field.fill(0.0)
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                if state.fullscreen:
                    layout = _compute_layout(screen_w, screen_h)
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    layout = _compute_layout(DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, layout.width, layout.height)
    finally:
        if tracker_solution is not None:
            tracker_solution.close()
        if tracker_tasks is not None:
            tracker_tasks.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
