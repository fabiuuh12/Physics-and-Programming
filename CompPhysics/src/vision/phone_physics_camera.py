#!/usr/bin/env python3
"""Track a selected phone and overlay estimated 2D motion, force and energy.

Run: python3 explore.py run phone_physics_camera
Select the phone with S. C freezes a frame: click two endpoints of a known
length, type that length in cm, then Enter. M accepts mass in grams.
Q quits; R clears the target. Keep the camera fixed and motion in the calibrated
plane. This is manual target selection, not semantic object recognition.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np

WINDOW = "Phone Physics Camera"
CYAN, GOLD, PINK = (240, 220, 70), (70, 210, 255), (190, 100, 255)


class MotionEstimator:
    """Local quadratic fit with actual timestamps; x right, y up.

    Derivatives are evaluated at the middle of a 0.45 s window, reducing
    differentiation noise at the cost of about 0.225 s of display lag.
    """

    def __init__(self):
        self.samples = deque()

    def clear(self):
        self.samples.clear()

    def add(self, timestamp, center):
        if self.samples and (timestamp <= self.samples[-1][0]
                             or timestamp - self.samples[-1][0] > 0.25):
            self.clear()
        self.samples.append((timestamp, np.array([center[0], -center[1]], dtype=float)))
        while self.samples and timestamp - self.samples[0][0] > 0.45:
            self.samples.popleft()
        if len(self.samples) < 6 or timestamp - self.samples[0][0] < 0.18:
            return None
        times = np.array([s[0] for s in self.samples])
        relative = times - (times[0] + times[-1]) / 2
        design = np.column_stack((np.ones_like(relative), relative, 0.5 * relative**2))
        fit = np.linalg.lstsq(design, np.array([s[1] for s in self.samples]), rcond=None)[0]
        return fit[1], fit[2]


class PhoneTracker:
    """Forward/backward optical flow and robust similarity transform.

    Lose lock instead of continuing with an unsupported position. A new
    selection is required after loss; there is no automatic identification.
    """

    def __init__(self):
        self.gray = self.points = self.corners = None
        self.quality = 0.0

    @property
    def active(self):
        return self.corners is not None

    def clear(self):
        self.__init__()

    def _features(self, gray):
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, self.corners.astype(np.int32), 255)
        return cv2.goodFeaturesToTrack(gray, 100, 0.015, 6, mask=mask, blockSize=5)

    def select(self, frame, box):
        self.clear()
        x, y, w, h = box
        if min(w, h) < 20:
            return False
        self.corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = self._features(self.gray)
        if self.points is None or len(self.points) < 8:
            self.clear()
            return False
        self.quality = 1.0
        return True

    def update(self, frame):
        if not self.active:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        options = dict(winSize=(25, 25), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, self.points, None, **options)
        if nxt is None or ok is None:
            self.clear()
            return None
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None, **options)
        if back is None or back_ok is None:
            self.clear()
            return None
        good = (ok.ravel() == 1) & (back_ok.ravel() == 1)
        good &= np.linalg.norm(back - self.points, axis=2).ravel() < 1.2
        old, new = self.points[good], nxt[good]
        if len(new) < 8:
            self.clear()
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.0)
        if matrix is None or inliers is None or inliers.sum() < 8 or inliers.mean() < 0.6:
            self.clear()
            return None
        scale = np.linalg.norm(matrix[:, 0])
        corners = cv2.transform(self.corners[None], matrix)[0]
        h, w = gray.shape
        center = corners.mean(axis=0)
        if not 0.90 < scale < 1.10 or not (0 <= center[0] < w and 0 <= center[1] < h):
            self.clear()
            return None
        self.corners = corners
        self.quality = float(inliers.sum() / len(self.points))
        self.points = new[inliers.ravel().astype(bool)].reshape(-1, 1, 2)
        self.gray = gray
        if len(self.points) < 45:
            fresh = self._features(gray)
            if fresh is not None and len(fresh) >= 8:
                self.points = fresh
        return center


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def physical_values(motion, meters_per_pixel, mass_kg):
    if motion is None:
        return None
    velocity, acceleration = motion
    if meters_per_pixel is None:
        return velocity, acceleration, None, None
    velocity, acceleration = velocity * meters_per_pixel, acceleration * meters_per_pixel
    force = None if mass_kg is None else mass_kg * acceleration
    energy = None if mass_kg is None else 0.5 * mass_kg * float(velocity @ velocity)
    return velocity, acceleration, force, energy


def label(frame, text, point, color=(225, 235, 240), size=0.55):
    cv2.putText(frame, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def arrow(frame, center, vector, gain, color, name):
    # Convert physics y-up back to image y-down. Gains differ by quantity.
    offset = np.array([vector[0], -vector[1]]) * gain
    length = np.linalg.norm(offset)
    if length < 2:
        return
    capped = length > 170
    offset *= min(1.0, 170 / length)
    start = tuple(np.rint(center).astype(int))
    end = tuple(np.rint(center + offset).astype(int))
    cv2.arrowedLine(frame, start, end, color, 3, cv2.LINE_AA, tipLength=0.18)
    end = (max(5, min(frame.shape[1]-90, end[0]+8)),
           max(25, min(frame.shape[0]-12, end[1]-8)))
    label(frame, name + (" [capped]" if capped else ""), end, color)


def render(frame, tracker, trail, motion, scale, mass, status, mode, typed, clicks):
    view = frame.copy()
    # Trail follows the tracked center, not the fitted/lagged position.
    for a, b in zip(list(trail)[:-1], list(trail)[1:]):
        cv2.line(view, tuple(np.int32(a)), tuple(np.int32(b)), CYAN, 2, cv2.LINE_AA)
    if tracker.active:
        cv2.polylines(view, [tracker.corners.astype(np.int32)], True, CYAN, 2, cv2.LINE_AA)
        center = tracker.corners.mean(axis=0)
        label(view, "PHONE | selected target", tuple(np.int32(tracker.corners[0] + [0, -12])), CYAN)
        if motion is not None:
            arrow(view, center, motion[0], 0.22, CYAN, "v")
            arrow(view, center, motion[1], 0.045, GOLD, "a")
            if scale is not None and mass is not None:
                arrow(view, center, motion[1] * scale * mass, 3000.0, PINK, "F net")
    # Separate readable dashboard below the camera image.
    panel = np.full((240, view.shape[1], 3), (24, 22, 18), np.uint8)
    label(panel, "PHONE PHYSICS / 2D MOTION ESTIMATES", (18, 27), CYAN, 0.65)
    label(panel, status, (18, 54), GOLD)
    values = physical_values(motion, scale, mass)
    units = "m" if scale is not None else "px"
    if values is None:
        label(panel, "Motion: waiting for a stable track (0.2 s minimum)", (18, 83))
    else:
        velocity, acceleration, force, energy = values
        label(panel, f"Speed {np.linalg.norm(velocity):.3f} {units}/s   |   Acceleration {np.linalg.norm(acceleration):.3f} {units}/s^2", (18, 83), CYAN)
        text = "F net / kinetic energy: set calibration [C] and mass [M]"
        if force is not None:
            text = f"Estimated F net {np.linalg.norm(force):.4f} N   |   In-plane kinetic energy {energy:.5f} J"
        label(panel, text, (18, 110), PINK)
    calibration = "uncalibrated" if scale is None else f"{scale * 1000:.3f} mm/px"
    mass_text = "unset" if mass is None else f"{mass * 1000:g} g"
    label(panel, f"Scale: {calibration}   |   Mass: {mass_text}   |   Fit lag ~0.22 s", (18, 138))
    label(panel, "S select phone   C calibrate   M mass   R reset target   Q quit", (18, 166))
    label(panel, "Fixed camera; move parallel to image plane. Rotation/depth can bias estimates.", (18, 190), (155, 170, 180), 0.48)
    label(panel, "Arrows: v 0.22 s; a 0.045 s^2; F 3000 px/N. Capped at 170 px.", (18, 213), (155, 170, 180), 0.48)
    if mode:
        cv2.rectangle(view, (0, 0), (view.shape[1], 88), (24, 22, 18), -1)
        instruction = "Click two endpoints of a known length" if mode == "calibrate" else (
            "Known length in cm" if mode == "length" else "Phone mass in grams")
        label(view, instruction + ": " + typed, (18, 32), GOLD, 0.65)
        label(view, "Enter confirms number / Esc cancels. Calibration uses a frozen frame.", (18, 62))
    for point in clicks:
        cv2.circle(view, point, 5, GOLD, -1)
    return np.vstack((view, panel))


def demo_frame(t):
    """Deterministic textured phone moving in a plane, for camera-free checks."""
    frame = np.full((540, 960, 3), (42, 35, 29), np.uint8)
    x = int(420 + 145 * np.sin(t * 1.4))
    y = int(155 + 65 * np.sin(t * 2.0))
    cv2.rectangle(frame, (x, y), (x+105, y+190), (210, 185, 130), -1)
    cv2.rectangle(frame, (x+7, y+8), (x+98, y+182), (52, 44, 33), -1)
    for row in range(6):
        for col in range(3):
            cv2.rectangle(frame, (x+15+col*27, y+25+row*24),
                          (x+29+col*27, y+38+row*24), (80+row*22, 170, 220), -1)
    return frame, (x, y, 105, 190)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mass-g", type=positive, help="actual phone mass in grams")
    parser.add_argument("--demo", action="store_true", help="synthetic phone; no camera")
    parser.add_argument("--headless", action="store_true", help="demo only, no window")
    parser.add_argument("--frames", type=int, default=180, help="headless demo duration")
    parser.add_argument("--snapshot", type=Path, help="save final rendered demo frame")
    args = parser.parse_args(argv)
    if args.headless and (not args.demo or args.frames < 1):
        parser.error("--headless needs --demo and --frames >= 1")
    cap = None
    if not args.demo:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            cap.release()
            parser.exit(1, "Cannot open webcam. Check camera permission or try --camera 1.\n")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    tracker, estimator = PhoneTracker(), MotionEstimator()
    trail = deque(maxlen=90)
    scale = None
    mass = args.mass_g / 1000 if args.mass_g else None
    status = "Press S and draw a tight box around your phone; Enter confirms."
    mode, typed, clicks = "", "", []
    frozen = None
    motion = None
    view = None

    def mouse(event, x, y, flags, userdata):
        nonlocal mode
        if mode == "calibrate" and event == cv2.EVENT_LBUTTONDOWN and y < frozen.shape[0]:
            clicks.append((x, y))
            if len(clicks) == 2:
                if np.linalg.norm(np.subtract(*clicks)) >= 10:
                    mode = "length"
                else:
                    clicks.clear()

    try:
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, mouse)
        index = 0
        while True:
            if args.demo:
                frame, box = demo_frame(index / 30)
                timestamp = index / 30
                if index == 0:
                    tracker.select(frame, box)
                    # Known synthetic dimensions; never used for the real camera.
                    scale = 0.15 / 190
                    status = "SYNTHETIC DEMO | known 15 cm phone height; no camera measurements"
            else:
                ok, frame = cap.read()
                timestamp = time.perf_counter()
                if not ok:
                    print("Camera stopped delivering frames.")
                    return 1
            if mode:
                frame = frozen.copy()
            else:
                was_active = tracker.active
                center = tracker.update(frame)
                if center is not None:
                    motion = estimator.add(timestamp, center)
                    trail.append(center.copy())
                else:
                    estimator.clear()
                    trail.clear()
                    motion = None
                    if was_active:
                        status = "Tracking lost. Press S to reselect; show a detailed face of the phone."
            view = render(frame, tracker, trail, motion, scale, mass, status, mode, typed, clicks)
            index += 1
            if args.headless:
                if index >= args.frames:
                    break
                continue
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(30 if args.demo else 1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if mode:
                if key == 27:
                    mode, typed, clicks = "", "", []
                    estimator.clear()
                    motion = None
                elif key in (10, 13) and mode in ("mass", "length"):
                    try:
                        value = positive(typed)
                    except (ValueError, argparse.ArgumentTypeError):
                        typed = ""
                        continue
                    if mode == "mass":
                        mass = value / 1000
                    else:
                        scale = value / 100 / np.linalg.norm(np.subtract(*clicks))
                    mode, typed, clicks = "", "", []
                    estimator.clear()
                    motion = None
                elif key in (8, 127):
                    typed = typed[:-1]
                elif mode != "calibrate" and key in b"0123456789." and len(typed) < 12:
                    typed += chr(key)
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                box = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
                cv2.setMouseCallback(WINDOW, mouse)
                success = tracker.select(frame, box)
                estimator.clear()
                trail.clear()
                motion = None
                status = ("Phone selected. Move slowly sideways/up-down. C calibrates; M sets mass."
                          if success else "Selection needs visible detail. Try the screen or camera lenses; press S.")
            elif key in (ord("c"), ord("m")):
                mode = "calibrate" if key == ord("c") else "mass"
                typed, clicks, frozen = "", [], frame.copy()
            elif key == ord("r"):
                tracker.clear()
                estimator.clear()
                trail.clear()
                motion = None
                status = "Target cleared. Press S to select your phone."
        if args.snapshot and view is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), view):
                raise OSError(f"Could not write {args.snapshot}")
        if args.headless:
            print(f"Demo: {index} frames; tracking={'locked' if tracker.active else 'lost'}; motion={'ready' if motion is not None else 'unavailable'}")
            return 0 if tracker.active and motion is not None else 1
        return 0
    finally:
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
