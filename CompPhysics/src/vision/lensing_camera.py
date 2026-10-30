#!/usr/bin/env python3
"""Live gravitational lensing camera: move your hand, spread your pinch.

Mouse drag also moves the lens. +/- changes strength; H switches hand control;
Left pinch locks; two new left pinches resume right-hand control; L toggles lock.
G toggles the Einstein-radius guide; B toggles a stylized black center;
I toggles original inset; Space freezes the image; R resets; Q quits.
Both pinches + move hands apart/together: stretch/compress source wavelength.
S: camera/space; A: align source star; right-click: move star; C: color tint;
[/]: wavelength. Use --demo for a camera-free scene, --no-hands for mouse only.
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
    wavelength_nm: float = 550.0
    wave_active: bool = False
    wave_releasing: bool = False
    wave_anchor_distance: float = 1.0
    wave_anchor_nm: float = 550.0
    wave_points: tuple | None = None
    wave_left_closed: bool = False
    wave_right_closed: bool = False
    gesture_lock_time: float = -math.inf

    def end_wave(self):
        self.wave_active = False
        self.wave_releasing = True
        self.wave_points = None
        self.unlock_first = -math.inf

    def update_wave(self, hands, width, height, now, dt):
        geometries = {}
        for side in ("left", "right"):
            landmarks = hands.get(side)
            if landmarks is None:
                continue
            p = np.asarray(landmarks, dtype=float)
            if p.shape != (21, 3) or not np.isfinite(p).all():
                continue
            palm = np.linalg.norm((p[5, :2]-p[17, :2]) * [width, height])
            if palm < 10:
                continue
            tips = p[[4, 8], :2] * [width, height]
            geometries[side] = (tips.mean(axis=0), np.linalg.norm(tips[0]-tips[1])/palm)
        # Missing data never becomes another pinch edge in the lock gesture.
        for side in ("left", "right"):
            name = f"wave_{side}_closed"
            if side in geometries:
                setattr(self, name, geometries[side][1] < (0.55 if getattr(self, name) else 0.35))
        both = (len(geometries) == 2 and self.wave_left_closed and self.wave_right_closed)
        if both:
            a, b = geometries["left"][0], geometries["right"][0]
            distance = max(40.0, float(np.linalg.norm(a-b)))
            if not self.wave_active:
                # If left closed a frame before right, treat that as preparation
                # for this two-hand gesture, not an unintended permanent lock.
                if self.locked and now-self.gesture_lock_time < 0.35:
                    self.set_locked(False)
                self.wave_anchor_distance = distance
                self.wave_anchor_nm = self.wavelength_nm
            target = np.clip(self.wave_anchor_nm*distance/self.wave_anchor_distance, 380, 750)
            alpha = 1-math.exp(-max(0.0, min(dt, 0.2))/0.07)
            self.wavelength_nm += alpha*(target-self.wavelength_nm)
            self.wave_active, self.wave_releasing = True, False
            self.wave_points = (a, b)
            self.points = None
            self.left_pinched = True
            self.unlock_first = -math.inf
            return True
        if self.wave_active:
            self.end_wave()
        if self.wave_releasing:
            # Consume the release so exiting wavelength mode cannot lock/unlock
            # or shrink the lens with a still-closed right hand.
            if "left" in geometries:
                self.left_pinched = self.wave_left_closed
            closed_visible = any(getattr(self, f"wave_{side}_closed") for side in geometries)
            if not closed_visible:
                self.wave_releasing = False
            return True
        return False

    def set_locked(self, locked):
        self.locked = locked
        self.gesture_lock_time = -math.inf
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
            self.gesture_lock_time = now
        elif now - self.unlock_first <= 0.8:
            self.set_locked(False)
        else:
            self.unlock_first = now

    def update_hands(self, hands, width, height, now, dt, allow_follow=True):
        # Two-hand wavelength control takes priority over one-hand controls.
        if self.update_wave(hands, width, height, now, dt):
            return False
        # Process the stop gesture BEFORE right-hand smoothing on this frame.
        self.update_left(hands.get("left"), width, height, now)
        right = hands.get("right")
        if right is not None and allow_follow:
            return self.follow(right, width, height, now, dt)
        return False

    def follow(self, landmarks, width, height, now, dt):
        if self.locked or self.wave_active or self.wave_releasing:
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
        # Swap detector labels once here to match the user's physical hands
        # in this mirrored camera pipeline; downstream roles stay anatomical.
        selected = {}
        for side in ("left", "right"):
            detector_side = "right" if side == "left" else "left"
            hand = self.bridge._pick_best_by_label(hands, detector_side)
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



def wavelength_color(wavelength_nm):
    """Approximate visible-spectrum sRGB colors, returned in OpenCV BGR order."""
    wavelengths = [380, 440, 490, 510, 560, 590, 650, 750]
    rgb = np.array([[150, 0, 255], [40, 40, 255], [0, 220, 255], [0, 255, 70],
                    [180, 255, 0], [255, 190, 0], [255, 30, 0], [180, 0, 0]])
    return tuple(int(np.interp(wavelength_nm, wavelengths, rgb[:, c])) for c in (2, 1, 0))


def frequency_thz(wavelength_nm):
    return 299792.458 / wavelength_nm


def tint_source(frame, wavelength_nm):
    """Illustrative monochromatic treatment, not webcam spectroscopy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    color = np.asarray(wavelength_color(wavelength_nm), np.float32)/255
    tinted = gray[..., None]*color[None, None, :]
    return np.clip(0.25*frame+0.75*tinted, 0, 255).astype(np.uint8)


class SpaceScene:
    """Cached procedural starfield; source star moves independently of the lens."""

    def __init__(self, width, height):
        y, x = np.indices((height, width), dtype=np.float32)
        band = np.exp(-((y-height*0.48-0.18*(x-width/2))/(height*0.14))**2)
        cloud = band*(0.55+0.2*np.sin(x/82)+0.2*np.sin(y/55+x/135))
        self.base = np.stack((12+25*cloud, 7+12*cloud, 5+15*cloud), axis=2).astype(np.uint8)
        rng = np.random.default_rng(419)
        for sx, sy, brightness in rng.random((550, 3)):
            p = (int(sx*width), int(sy*height))
            level = int(50+200*brightness**2)
            cv2.circle(self.base, p, 1 if brightness < 0.97 else 2,
                       (level, int(level*0.94), int(level*0.89)), -1, cv2.LINE_AA)
        for i in range(4):
            p = (int(width*(0.15+i*0.22)), int(height*(0.25+(i%2)*0.46)))
            cv2.ellipse(self.base, p, (18+i*4, 5+i), -25+i*18, 0, 360, (80, 65, 55), 1, cv2.LINE_AA)
            cv2.circle(self.base, p, 2, (175, 155, 140), -1, cv2.LINE_AA)

    def render(self, star_position, wavelength_nm, star_visible=True):
        image = self.base.copy()
        if star_visible:
            x, y = np.rint(star_position).astype(int)
            h, w = image.shape[:2]
            x0, x1 = max(0, x-28), min(w, x+29)
            y0, y1 = max(0, y-28), min(h, y+29)
            if x0 < x1 and y0 < y1:
                yy, xx = np.indices((y1-y0, x1-x0), dtype=np.float32)
                r2 = (xx+x0-x)**2+(yy+y0-y)**2
                profile = 255*np.exp(-r2/10)+85*np.exp(-r2/130)
                color = np.asarray(wavelength_color(wavelength_nm), dtype=np.float32)/255
                patch = image[y0:y1, x0:x1].astype(np.float32)
                patch += profile[..., None]*color[None, None, :]
                image[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
        return image


def draw_wave_panel(footer, control, now):
    w = footer.shape[1]
    color = wavelength_color(control.wavelength_nm)
    text(footer, f"Wavelength {control.wavelength_nm:.0f} nm  |  Frequency {frequency_thz(control.wavelength_nm):.1f} THz  |  Both pinches: stretch/compress", (20, 70), color, 0.48)
    xs = np.arange(20, w-20)
    ys = 110+17*np.sin(2*np.pi*((xs-20)/(control.wavelength_nm*0.16)-now*550/control.wavelength_nm))
    cv2.polylines(footer, [np.column_stack((xs, ys)).astype(np.int32)], False, color, 2, cv2.LINE_AA)
    text(footer, "Spatial wave sketch / slowed animation", (20, 92), (130, 145, 160), 0.34)
    # Small spectrum ruler with an indicator for the current selection.
    for x in range(20, w-20, 2):
        wavelength = 380+(x-20)/(w-40)*370
        cv2.line(footer, (x, 142), (x, 150), wavelength_color(wavelength), 2)
    marker = int(20+(control.wavelength_nm-380)/370*(w-40))
    cv2.line(footer, (marker, 137), (marker, 155), (240, 240, 240), 2)


def compose(frame, lensed, control, guide, inset, shadow, frozen, fps, status, now, background="camera"):
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
    if control.wave_active and control.wave_points is not None:
        a, b = (tuple(np.rint(p).astype(int)) for p in control.wave_points)
        color = wavelength_color(control.wavelength_nm)
        cv2.line(view, a, b, color, 2, cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(view, p, 7, color, -1, cv2.LINE_AA)
    if inset:
        iw = w//4
        ih = round(h*iw/w)
        original = cv2.resize(frame, (iw, ih))
        x, y = 18, h-ih-18
        view[y:y+ih, x:x+iw] = original
        cv2.rectangle(view, (x-1, y-1), (x+iw, y+ih), (175, 168, 145), 1)
        cv2.rectangle(view, (x, y+ih-26), (x+100, y+ih), INK, -1)
        text(view, "CAMERA" if background == "space" else "ORIGINAL", (x+8, y+ih-9), size=0.42)
    header = np.full((54, w, 3), INK, np.uint8)
    text(header, "L E N S I N G   C A M E R A", (20, 34), CYAN, 0.7)
    state = "WAVELENGTH" if control.wave_active else "LOCKED" if control.locked else "FROZEN" if frozen else background.upper()
    text(header, f"{fps:.0f} FPS / {state}", (w-230, 33), size=0.48)
    footer = np.full((204, w, 3), INK, np.uint8)
    text(footer, status, (20, 24), CYAN)
    text(footer, f"Einstein radius {control.radius:.0f} px | Relative mass {(control.radius/115)**2:.2f}x | Center mask {'on (stylized)' if shadow else 'off'}", (20, 49), size=0.48)
    draw_wave_panel(footer, control, now)
    text(footer, "S: camera/space | Right-click: star | A: align star | C: tint | [/]: wavelength | +/-: lens", (20, 176), size=0.44)
    text(footer, "Left pinch: lock | Left double pinch: resume | L: lock | H: hands | Space: freeze | Q: quit", (20, 197), (150, 165, 180), 0.43)
    return np.vstack((header, view, footer))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--background", choices=("camera", "space"), default="camera")
    parser.add_argument("--wavelength", type=float, default=550, help="visible wavelength, 380-750 nm")
    parser.add_argument("--tint", action="store_true", help="illustrative source color treatment")
    parser.add_argument("--headless", action="store_true", help="demo only, no windows")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args(argv)
    if args.headless and (not args.demo or args.frames < 1):
        parser.error("--headless requires --demo and a positive --frames")
    if not np.isfinite(args.wavelength) or not 380 <= args.wavelength <= 750:
        parser.error("--wavelength must be between 380 and 750 nm")
    cap = detector = None
    if not args.demo:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            cap.release()
            parser.exit(1, "Cannot open camera. Allow Terminal in macOS Privacy & Security > Camera, or try --camera 1.\n")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    control = LensControl(np.array([480., 270.]), hand_enabled=not (args.no_hands or args.demo),
                          wavelength_nm=args.wavelength)
    hand_status = "Mouse control: drag anywhere in the camera view; +/- changes strength"
    if control.hand_enabled:
        try:
            detector = HandInput()
        except Exception as exc:
            print(f"Hand tracking unavailable; using mouse controls: {exc}")
            control.hand_enabled = False
    lens = sky = None
    star_position = np.array([480., 270.])
    space_seen = args.background == "space"
    background, colorize, star_visible = args.background, args.tint, True
    frame = None
    view = None
    shape = None
    dragging = star_dragging = False
    guide, inset, shadow, frozen = False, True, True, False
    index, fps = 0, 30.0
    last_time = time.perf_counter()
    last_detection = last_time
    mouse_time = -math.inf
    hands = {}

    def mouse(event, x, y, flags, userdata):
        nonlocal dragging, star_dragging, mouse_time
        if event == cv2.EVENT_LBUTTONUP:
            dragging = False
        if event == cv2.EVENT_RBUTTONUP:
            star_dragging = False
        if shape is None or not (0 <= y-54 < shape[0]):
            return
        if background == "space":
            if event == cv2.EVENT_RBUTTONDOWN:
                star_dragging = True
            if star_dragging and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_RBUTTONDOWN):
                star_position[:] = np.clip([x, y-54], [0, 0], [shape[1]-1, shape[0]-1])
        if control.locked or control.wave_active or control.wave_releasing:
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
                sky = SpaceScene(shape[1], shape[0])
                star_position[:] = [shape[1]/2, shape[0]/2]
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
                was_wave_active = control.wave_active
                control.update_hands(hands, shape[1], shape[0], now, elapsed,
                                     allow_follow=not dragging and now-mouse_time > 1.5)
                if control.wave_active and not was_wave_active:
                    colorize = True
            if control.wave_active:
                hand_status = "WAVELENGTH / hands apart = longer + redder / hands together = shorter + bluer"
            elif control.wave_releasing:
                hand_status = "Wavelength held / release BOTH pinches to return to lens controls"
            elif control.locked:
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
                hand_status = "DEMO / S switches space background / right-click places star / A aligns for a ring"
            source = sky.render(star_position, control.wavelength_nm, star_visible) if background == "space" else frame
            if colorize:
                source = tint_source(source, control.wavelength_nm)
            lensed = lens.apply(source, control.center, control.radius, shadow)
            view = compose(frame, lensed, control, guide, inset, shadow, frozen, fps, hand_status, now, background)
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
                if not control.locked and not control.wave_active and not control.wave_releasing:
                    control.radius = float(np.clip(control.radius*(1.08 if key in (43, 61) else 1/1.08), 20, 230))
            elif key == ord("s"):
                background = "space" if background == "camera" else "camera"
                if background == "space" and not space_seen:
                    star_position[:] = control.center
                    space_seen = True
                star_dragging = False
            elif key == ord("a") and background == "space":
                star_position[:] = control.center
            elif key == ord("t"):
                star_visible = not star_visible
            elif key == ord("c"):
                colorize = not colorize
            elif key in (ord("["), ord("]")):
                control.wavelength_nm = float(np.clip(control.wavelength_nm+(10 if key == ord("]") else -10), 380, 750))
                colorize = True
            elif key == ord("l"):
                control.set_locked(not control.locked)
            elif key == ord("h"):
                control.end_wave()
                control.wave_releasing = False
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
                control.end_wave()
                control.wave_releasing = False
                frozen = not frozen
                control.points = None
                control.last_hand = -math.inf
            elif key == ord("r"):
                control.end_wave()
                control.wave_releasing = False
                control.wavelength_nm = 550
                colorize = False
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
