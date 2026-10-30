#!/usr/bin/env python3
"""Live gravitational lensing camera: move your hand, spread your pinch.

Mouse drag also moves the lens. +/- changes strength; H switches hand control;
Left pinch locks; two new left pinches resume right-hand control; L toggles lock.
G toggles the Einstein-radius guide; B toggles a stylized black center;
I toggles original inset; Space freezes the image; R resets; Q quits.
Use --demo for a camera-free animated scene, --no-hands for mouse only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

WINDOW = "Lensing Camera"
CYAN = (235, 215, 80)
INK = (23, 18, 13)
MODEL_PATH = Path(__file__).resolve().parents[2] / "resources/models/hand_landmarker.task"


class PointLens:
    """Inverse image sampling using beta = theta - theta_E^2 theta/|theta|^2.

    All coordinates are pixels in an illustrative source plane. This is the
    weak-field thin-lens point-mass mapping, not a near-horizon GR ray tracer.
    """

    def __init__(self, width, height):
        self.y, self.x = np.indices((height, width), dtype=np.float32)

    def maps(self, center, radius):
        if not np.isfinite((*center, radius)).all() or radius < 0:
            raise ValueError("Lens parameters must be finite; radius must be nonnegative")
        dx, dy = self.x - np.float32(center[0]), self.y - np.float32(center[1])
        r2 = dx*dx + dy*dy
        # The singular central pixel is hidden by the optional decorative mask.
        # Without it the center samples itself, a finite numerical convention.
        factor = np.float32(radius*radius) / np.maximum(r2, np.float32(0.01))
        return self.x - factor*dx, self.y - factor*dy, r2

    def apply(self, frame, center, radius, shadow=True):
        mx, my, r2 = self.maps(center, radius)
        image = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(4, 4, 6))
        if shadow and radius > 0:
            # Artistic occulting disk; NOT the Einstein ring or an event horizon.
            mask_radius = 0.24 * radius
            alpha = np.clip((np.sqrt(r2) - mask_radius + 1.5) / 3, 0, 1)
            image = (image * alpha[..., None]).astype(np.uint8)
        return image


@dataclass
class LensControl:
    center: np.ndarray
    radius: float = 115.0
    hand_enabled: bool = True
    last_hand: float = -math.inf
    points: tuple | None = None
    locked: bool = False
    left_pinched: bool = False
    unlock_first: float = -math.inf

    def set_locked(self, locked):
        self.locked = locked
        self.unlock_first = -math.inf
        self.points = None
        self.last_hand = -math.inf

    def update_left(self, landmarks, width, height, now):
        """Pinch edges, with hysteresis. Locking pinch never counts toward unlock.

        Missing/invalid observations preserve the latch: hiding a held pinch
        cannot produce a new tap. An observed release is required between taps.
        """
        if now - self.unlock_first > 0.8:
            self.unlock_first = -math.inf
        if landmarks is None:
            return
        points = np.asarray(landmarks, dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return
        palm = np.linalg.norm((points[5, :2] - points[17, :2]) * [width, height])
        if palm < 10:
            return
        ratio = np.linalg.norm((points[4, :2] - points[8, :2]) * [width, height]) / palm
        pinched = ratio < (0.55 if self.left_pinched else 0.35)
        pressed = pinched and not self.left_pinched
        self.left_pinched = pinched
        if not pressed:
            return
        if not self.locked:
            self.set_locked(True)
        elif now - self.unlock_first <= 0.8:
            self.set_locked(False)
        else:
            self.unlock_first = now

    def update_hands(self, hands, width, height, now, dt, allow_follow=True):
        # Process the stop gesture BEFORE right-hand smoothing on this frame.
        self.update_left(hands.get("left"), width, height, now)
        right = hands.get("right")
        if right is not None and allow_follow:
            return self.follow(right, width, height, now, dt)
        return False

    def follow(self, landmarks, width, height, now, dt):
        if self.locked:
            return False
        points = np.asarray(landmarks, dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return False
        thumb, index = points[[4, 8], :2] * [width, height]
        palm = np.linalg.norm((points[5, :2] - points[17, :2]) * [width, height])
        if palm < 10:
            return False
        ratio = np.linalg.norm(index - thumb) / palm
        target = (thumb + index) / 2
        target = np.clip(target, [0, 0], [width-1, height-1])
        # Spreading thumb/index increases Einstein radius, with no mass claim.
        radius = float(np.interp(ratio, [0.15, 1.8], [35, min(width, height)*0.36]))
        alpha = 1 - math.exp(-max(0.0, min(dt, 0.2)) / 0.09)
        self.center += alpha * (target - self.center)
        self.radius += alpha * (radius - self.radius)
        self.points = (thumb, index)
        self.last_hand = now
        return True


class HandInput:
    """Use the repository's existing MediaPipe landmark adapters, CPU only."""

    def __init__(self):
        cache = Path(__file__).resolve().parents[2] / "output/cache/matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        import mediapipe as mp
        if __package__:
            from . import two_hand_bridge as bridge
        else:
            import two_hand_bridge as bridge
        self.mp, self.bridge = mp, bridge
        self.timestamp = 0
        if hasattr(mp, "solutions"):
            self.tasks = False
            self.detector = mp.solutions.hands.Hands(
                max_num_hands=2, model_complexity=0,
                min_detection_confidence=0.55, min_tracking_confidence=0.5)
        else:
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Missing bundled hand model: {MODEL_PATH}")
            self.tasks = True
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
                                                  delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        small = cv2.resize(frame, (480, round(frame.shape[0]*480/frame.shape[1])))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.tasks:
            self.timestamp = max(self.timestamp+1, time.monotonic_ns() // 1_000_000)
            result = self.detector.detect_for_video(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
            hands = self.bridge._extract_task_hands(result, mirror=False)
        else:
            hands = self.bridge._extract_solution_hands(self.detector.process(rgb), mirror=False)
        # Input is already mirrored; MediaPipe's selfie handedness labels apply.
        selected = {}
        for side in ("left", "right"):
            hand = self.bridge._pick_best_by_label(hands, side)
            if hand is not None:
                selected[side] = hand.landmarks
        return selected

    def close(self):
        self.detector.close()


def text(image, message, xy, color=(225, 232, 236), size=0.5):
    cv2.putText(image, message, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def synthetic_scene(width, height, t):
    """Animated observatory test card; no generated concept image is used."""
    image = np.full((height, width, 3), (32, 25, 19), np.uint8)
    for x in range(0, width, 48):
        cv2.line(image, (x, 0), (x, height), (65, 55, 38), 1)
    for y in range(0, height, 48):
        cv2.line(image, (0, y), (width, y), (65, 55, 38), 1)
    rng = np.random.default_rng(72)
    for x, y, brightness in rng.random((160, 3)):
        cv2.circle(image, (int(x*width), int(y*height)), 1,
                   (int(150+100*brightness),)*3, -1)
    for i in range(7):
        x = int(width*(0.13+i*0.11))
        y = int(height*(0.45+0.16*math.sin(i*1.3+t*0.3)))
        cv2.circle(image, (x, y), 12+i*2, (185, 130+i*12, 80+i*15), -1, cv2.LINE_AA)
        cv2.ellipse(image, (x, y), (30+i*3, 8), -20, 0, 360, (115, 100, 70), 1, cv2.LINE_AA)
    cv2.rectangle(image, (width//2-3, height//2-3), (width//2+3, height//2+3), (255, 240, 215), -1)
    text(image, "SYNTHETIC SKY / camera-free preview", (25, height-25), CYAN, 0.6)
    return image


def compose(frame, lensed, control, guide, inset, shadow, frozen, fps, status, now):
    h, w = frame.shape[:2]
    view = lensed.copy()
    center = tuple(np.rint(control.center).astype(int))
    if guide:
        cv2.circle(view, center, round(control.radius), (135, 135, 80), 1, cv2.LINE_AA)
        label_pos = (max(10, min(w-185, center[0]+15)),
                     max(25, min(h-15, center[1]-round(control.radius)-12)))
        text(view, "Einstein radius (guide)", label_pos, CYAN, 0.43)
    if control.hand_enabled and control.points is not None and now-control.last_hand < 0.35:
        a, b = (tuple(np.rint(p).astype(int)) for p in control.points)
        cv2.line(view, a, b, CYAN, 1, cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(view, p, 5, CYAN, -1, cv2.LINE_AA)
        cv2.line(view, tuple(np.int32((np.array(a)+b)/2)), center, CYAN, 1, cv2.LINE_AA)
    if inset:
        iw = w//4
        ih = round(h*iw/w)
        original = cv2.resize(frame, (iw, ih))
        x, y = 18, h-ih-18
        view[y:y+ih, x:x+iw] = original
        cv2.rectangle(view, (x-1, y-1), (x+iw, y+ih), (175, 168, 145), 1)
        cv2.rectangle(view, (x, y+ih-26), (x+100, y+ih), INK, -1)
        text(view, "ORIGINAL", (x+8, y+ih-9), size=0.42)
    header = np.full((54, w, 3), INK, np.uint8)
    text(header, "L E N S I N G   C A M E R A", (20, 34), CYAN, 0.7)
    text(header, f"{fps:.0f} FPS" + (" / LOCKED" if control.locked else " / FROZEN" if frozen else " / LIVE"), (w-195, 33), size=0.48)
    footer = np.full((118, w, 3), INK, np.uint8)
    text(footer, status, (20, 24), CYAN)
    text(footer, f"Einstein radius {control.radius:.0f} px | Relative mass {(control.radius/115)**2:.2f}x | Center mask {'on (stylized)' if shadow else 'off'}", (20, 49), size=0.48)
    text(footer, "Drag: move lens | +/-: strength | H: hand/mouse | G: guide | B: mask | I: inset", (20, 74), size=0.47)
    text(footer, "Left pinch: lock | Left double pinch: resume | L: lock | Space: freeze | R: reset | Q: quit", (20, 99), (150, 165, 180), 0.46)
    return np.vstack((header, view, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--headless", action="store_true", help="demo only, no windows")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    if args.headless and (not args.demo or args.frames < 1):
        parser.error("--headless requires --demo and a positive --frames")
    cap = detector = None
    if not args.demo:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            cap.release()
            parser.exit(1, "Cannot open camera. Allow Terminal in macOS Privacy & Security > Camera, or try --camera 1.\n")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    control = LensControl(np.array([480., 270.]), hand_enabled=not (args.no_hands or args.demo))
    hand_status = "Mouse control: drag anywhere in the camera view; +/- changes strength"
    if control.hand_enabled:
        try:
            detector = HandInput()
        except Exception as exc:
            print(f"Hand tracking unavailable; using mouse controls: {exc}")
            control.hand_enabled = False
    lens = None
    frame = None
    view = None
    shape = None
    dragging = False
    guide, inset, shadow, frozen = False, True, True, False
    index, fps = 0, 30.0
    last_time = time.perf_counter()
    last_detection = last_time
    mouse_time = -math.inf
    hands = {}

    def mouse(event, x, y, flags, userdata):
        nonlocal dragging, mouse_time
        if event == cv2.EVENT_LBUTTONUP:
            dragging = False
        if control.locked or shape is None or not (0 <= y-54 < shape[0]):
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
        if dragging and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            control.center[:] = np.clip([x, y-54], [0, 0], [shape[1]-1, shape[0]-1])
            mouse_time = time.perf_counter()
            control.points = None
        if event == cv2.EVENT_MOUSEWHEEL:
            control.radius = float(np.clip(control.radius*(1.08 if flags > 0 else 1/1.08), 20, 230))

    try:
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, mouse)
        while True:
            now = time.perf_counter()
            dt = max(1e-4, now-last_time)
            last_time = now
            fps = 0.9*fps + 0.1/dt
            if not frozen or frame is None:
                if args.demo:
                    frame = synthetic_scene(960, 540, index/30)
                else:
                    ok, raw = cap.read()
                    if not ok:
                        print("Camera stopped delivering frames.")
                        return 1
                    frame = cv2.flip(cv2.resize(raw, (960, round(raw.shape[0]*960/raw.shape[1]))), 1)
            if frame.shape[:2] != shape:
                shape = frame.shape[:2]
                lens = PointLens(shape[1], shape[0])
                control.center[:] = [shape[1]/2, shape[0]/2]
                control.radius = min(115, shape[0]*0.3)
            if detector is not None and control.hand_enabled and not frozen:
                try:
                    hands = detector.detect(frame)
                except Exception as exc:
                    print(f"Hand detector stopped; mouse controls remain available: {exc}")
                    detector.close()
                    detector = None
                    control.hand_enabled = False
                    hands = {}
                elapsed = now-last_detection
                last_detection = now
                control.update_hands(hands, shape[1], shape[0], now, elapsed,
                                     allow_follow=not dragging and now-mouse_time > 1.5)
            if control.locked:
                pending = now-control.unlock_first <= 0.8
                hand_status = ("LOCKED / one more left pinch to resume" if pending else
                               "LOCKED / position + size held / double-pinch LEFT hand to resume (or L)")
            elif control.hand_enabled:
                if now-control.last_hand < 0.35:
                    hand_status = "FOLLOWING RIGHT HAND / left pinch locks position + size"
                else:
                    hand_status = "Show RIGHT thumb + index to steer / LEFT pinch locks / mouse drag also works"
            else:
                hand_status = "MOUSE CONTROL / drag the lens; +/- changes strength"
            if args.demo and not control.locked:
                hand_status = "SYNTHETIC SKY / drag the lens across a planet to see arcs and multiple images"
            lensed = lens.apply(frame, control.center, control.radius, shadow)
            view = compose(frame, lensed, control, guide, inset, shadow, frozen, fps, hand_status, now)
            index += 1
            if args.headless:
                if index >= args.frames:
                    break
                continue
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(15 if args.demo else 1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("+"), ord("="), ord("-"), ord("_")):
                if not control.locked:
                    control.radius = float(np.clip(control.radius*(1.08 if key in (43, 61) else 1/1.08), 20, 230))
            elif key == ord("l"):
                control.set_locked(not control.locked)
            elif key == ord("h"):
                control.hand_enabled = not control.hand_enabled
                control.points = None
                control.last_hand = -math.inf
                if control.hand_enabled and detector is None:
                    try:
                        detector = HandInput()
                    except Exception as exc:
                        print(f"Hand tracking unavailable: {exc}")
                        control.hand_enabled = False
            elif key == ord("g"):
                guide = not guide
            elif key == ord("b"):
                shadow = not shadow
            elif key == ord("i"):
                inset = not inset
            elif key == ord(" "):
                frozen = not frozen
                control.points = None
                control.last_hand = -math.inf
            elif key == ord("r"):
                control.set_locked(False)
                control.center[:] = [shape[1]/2, shape[0]/2]
                control.radius = min(115, shape[0]*0.3)
                control.points = None
                control.last_hand = -math.inf
        if args.snapshot is not None and view is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), view):
                raise OSError(f"Could not write {args.snapshot}")
        if args.headless:
            print(f"Rendered {index} frames, {shape[1]}x{shape[0]}; finite lens maps; no camera required")
        return 0
    finally:
        if detector is not None:
            detector.close()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
