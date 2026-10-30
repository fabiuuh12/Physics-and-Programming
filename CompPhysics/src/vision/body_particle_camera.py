#!/usr/bin/env python3
"""Body particle lab: webcam silhouette with simulated alpha, beta and gamma beams.

1/2/3 selects particles. Drag Energy to compare monoenergetic beams.
Click/drag the camera view to aim the beam. Space pauses particles, M slows
animation, R clears tracks/counts, F fits the window, ? shows controls, Q quits.
Use --demo for a camera-free moving silhouette. No recordings are made by default.
"""
from __future__ import annotations

from functools import partial
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import cv2
import numpy as np

if __package__:
    from .body_particle_model import COLORS, ENERGIES, KINDS, Transport
    from .lensing_camera import DisplayArea, display_area, fit_window, rounded_rect, ui_text, inside
else:
    from body_particle_model import COLORS, ENERGIES, KINDS, Transport
    from lensing_camera import DisplayArea, display_area, fit_window, rounded_rect, ui_text, inside

WINDOW = "Particle Lab"
HEADER, FOOTER = 56, 86
MODEL_PATH = Path(__file__).resolve().parents[2] / "resources/models/pose_landmarker_lite.task"


class PoseInput:
    def __init__(self):
        cache = Path(__file__).resolve().parents[2] / "output/cache/matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        import mediapipe as mp
        self.mp = mp
        self.timestamp = 0
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"Missing bundled pose model: {MODEL_PATH}")
            self.tasks = True
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
                                                  delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1, min_pose_detection_confidence=.5,
                min_pose_presence_confidence=.5, min_tracking_confidence=.5,
                output_segmentation_masks=True)
            self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        else:
            self.tasks = False
            self.detector = mp.solutions.pose.Pose(enable_segmentation=True, model_complexity=0,
                                                   min_detection_confidence=.5,
                                                   min_tracking_confidence=.5)

    def detect(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (480, round(h*480/w)))
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if self.tasks:
            self.timestamp = max(self.timestamp+1, time.monotonic_ns()//1_000_000)
            result = self.detector.detect_for_video(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
            if not result.pose_landmarks or not result.segmentation_masks:
                return None
            probability = result.segmentation_masks[0].numpy_view().copy()
        else:
            result = self.detector.process(rgb)
            if result.pose_landmarks is None or result.segmentation_mask is None:
                return None
            probability = result.segmentation_mask
        probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(probability, 0, 1).astype(np.float32)

    def close(self):
        self.detector.close()


class Silhouette:
    """Short-lived mask cache. A missing detection immediately removes the target."""
    def __init__(self):
        self.probability = None
        self.updated = -math.inf

    def update(self, probability, now):
        if probability is None or not np.isfinite(probability).all() or np.count_nonzero(probability > .55) < 300:
            self.probability = None
            self.updated = -math.inf
            return
        if self.probability is None or self.probability.shape != probability.shape:
            self.probability = probability.copy()
        else:
            self.probability = .25*self.probability+.75*probability
        self.updated = now

    def mask(self, shape, now):
        if self.probability is None or now-self.updated > .3:
            return np.zeros(shape, dtype=bool)
        return self.probability > .55


def demo_scene(t):
    h, w = 540, 960
    mask = np.zeros((h, w), np.uint8)
    x = int(500+35*math.sin(t*.7))
    cv2.circle(mask, (x, 96), 42, 1, -1, cv2.LINE_8)
    cv2.fillConvexPoly(mask, np.array([[x-85, 160], [x+85, 160], [x+65, 355], [x-65, 355]]), 1)
    cv2.line(mask, (x, 130), (x, 180), 1, 45)
    for sign in (-1, 1):
        elbow = (x+sign*122, 238+int(40*math.sin(t+sign)))
        wrist = (x+sign*178, 190+int(55*math.sin(t+sign)))
        cv2.line(mask, (x+sign*70, 178), elbow, 1, 36)
        cv2.line(mask, elbow, wrist, 1, 30)
        cv2.circle(mask, wrist, 18, 1, -1)
        cv2.line(mask, (x+sign*36, 340), (x+sign*55, 535), 1, 49)
    frame = np.full((h, w, 3), (32, 30, 27), np.uint8)
    frame[mask > 0] = (138, 129, 117)
    return frame, mask.astype(np.float32)


def layout(w, h):
    start = w//2-130
    return {
        "alpha": (start, 12, start+82, 44),
        "beta": (start+88, 12, start+170, 44),
        "gamma": (start+176, 12, start+258, 44),
        "controls": (w-126, 12, w-18, 44),
        "energy": (w-205, HEADER+h+16, w-28, HEADER+h+39),
        "slow": (w-146, HEADER+h+53, w-20, HEADER+h+77),
        "panel": (w-378, HEADER+14, w-18, HEADER+min(h-14, 440)),
        "close": (w-54, HEADER+24, w-28, HEADER+50),
    }


def render(frame, mask, engine, source, tracked, paused, slow, controls, scale, demo=False):
    h, w = frame.shape[:2]
    size = (w*scale, h*scale)
    label = partial(ui_text, scale=scale)
    panel = partial(rounded_rect, scale=scale)
    point = lambda xy: tuple(round(float(value)*scale) for value in xy)
    # Keep the real body recognizable; the tint marks only the model target.
    view = (frame.astype(np.float32)*.55).astype(np.uint8)
    view[mask] = np.clip(view[mask].astype(float)*.95+np.array([24, 19, 10]), 0, 255)
    view = cv2.resize(view, size, interpolation=cv2.INTER_CUBIC)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, [c*scale for c in contours], -1, (151, 158, 155), max(1, scale), cv2.LINE_AA)
    glow = np.zeros_like(view)
    color = COLORS[engine.kind]
    for particle in engine.particles:
        if len(particle.trail) > 1:
            points = np.array([point(p) for p in particle.trail], np.int32)
            strength = (.35+.65*particle.energy/particle.initial_energy) if particle.alive else .45*max(0, particle.fade/.65)
            ink = tuple(int(c*strength) for c in color)
            cv2.polylines(glow, [points], False, ink, 2*scale, cv2.LINE_AA)
        if particle.alive:
            cv2.circle(glow, point(particle.pos), 3*scale, color, -1, cv2.LINE_AA)
    for position, life, event in engine.flashes:
        fraction = life/.45
        p = point(position)
        cv2.circle(glow, p, max(1, round((3+12*(1-fraction))*scale)),
                   tuple(int(225*fraction) for _ in range(3)), scale, cv2.LINE_AA)
        cv2.circle(glow, p, 3*scale, tuple(int(255*fraction) for _ in range(3)), -1, cv2.LINE_AA)
    view = cv2.add(view, glow)
    small = cv2.resize(glow, (w//2, h//2), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 2)
    view = cv2.addWeighted(view, 1, cv2.resize(blur, size), .55, 0)
    if tracked:
        x, y = point(source)
        cv2.circle(view, (x, y), 7*scale, color, scale, cv2.LINE_AA)
        cv2.line(view, (x+10*scale, y), (x+35*scale, y), color, scale, cv2.LINE_AA)
    else:
        label(view, "Step into view", (w//2-95, h//2-12), 26)
        label(view, "Your silhouette becomes the target", (w//2-131, h//2+24), 14, (167, 173, 184))
    header = np.full((HEADER*scale, w*scale, 3), (23, 22, 21), np.uint8)
    footer = np.full((FOOTER*scale, w*scale, 3), (23, 22, 21), np.uint8)
    label(header, "Particle Lab", (20, 17), 19)
    boxes = layout(w, h)
    for kind in KINDS:
        box = boxes[kind]
        panel(header, box, (62, 59, 56) if kind == engine.kind else (32, 30, 29), 7)
        label(header, kind.capitalize(), (box[0]+15, 22), 13,
              (235, 236, 239) if kind == engine.kind else (153, 157, 164))
    panel(header, boxes["controls"], (39, 37, 35), 7)
    label(header, "Controls  ?", (w-113, 22), 13, (184, 188, 194))
    for x, name, key in [(20, "Through", "through"), (155, "Scattered", "scattered"), (300, "Stopped", "stopped")]:
        label(footer, name, (x, 23), 12, (150, 156, 166))
        label(footer, str(engine.counts[key]), (x+78, 20), 16)
    label(footer, f"Energy  {engine.energy:.2f} MeV", (w-395, 23), 13)
    box = boxes["energy"]
    lo, hi, _ = ENERGIES[engine.kind]
    fraction = np.clip(math.log(engine.energy/lo)/math.log(hi/lo), 0, 1)
    x0, x1 = box[0], box[2]
    knob = x0+fraction*(x1-x0)
    cv2.line(footer, point((x0, 27)), point((x1, 27)), (62, 63, 68), 3*scale, cv2.LINE_AA)
    cv2.line(footer, point((x0, 27)), point((knob, 27)), (137, 142, 152), 3*scale, cv2.LINE_AA)
    cv2.circle(footer, point((knob, 27)), 5*scale, (230, 231, 234), -1, cv2.LINE_AA)
    state = "Paused" if paused else "Demo silhouette" if demo else "Body tracked" if tracked else "Waiting for body"
    label(footer, state, (20, 61), 12, (180, 186, 195))
    label(footer, "Water-like 2D target · simulated particles", (190, 61), 11, (128, 136, 148))
    panel(footer, (w-146, 53, w-20, 77), (57, 54, 50) if slow else (34, 32, 30), 6)
    label(footer, "Slow motion" if slow else "Normal speed", (w-133, 60), 11)
    canvas = np.vstack((header, view, footer))
    if controls:
        x0, y0, x1, y1 = boxes["panel"]
        panel(canvas, boxes["panel"], (35, 33, 31), 12)
        label(canvas, "Controls", (x0+20, y0+18), 18)
        label(canvas, "×", (x1-33, y0+15), 21, (165, 167, 173))
        rows = [("1 / 2 / 3", "Alpha / beta / gamma"), ("Energy slider", "Change beam energy"),
                ("Click / drag", "Aim beam"), ("Space", "Pause particles"), ("M", "Slow motion"),
                ("R", "Clear tracks and counts"), ("F", "Fit window"), ("Q", "Quit")]
        for i, (key, action) in enumerate(rows):
            y = y0+66+i*28
            label(canvas, key, (x0+20, y), 12, (156, 160, 167))
            label(canvas, action, (x0+150, y), 12)
        label(canvas, "Outside target: vacuum", (x0+20, y0+314), 11, (151, 157, 168))
        label(canvas, "Model scale: 1 px = 0.05 cm of water", (x0+20, y0+338), 11, (151, 157, 168))
        label(canvas, "Visual model, not a body scan or dose estimate.", (x0+20, y0+373), 10, (128, 136, 148))
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--type", choices=KINDS, default="alpha")
    parser.add_argument("--energy", type=float, help="beam energy in MeV, within selected type's range")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--headless", action="store_true", help="demo only")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--report", type=Path, help="write numeric demo/check results as JSON")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ui-scale", type=int, choices=(1, 2, 3))
    parser.add_argument("--show-controls", action="store_true")
    args = parser.parse_args(argv)
    if args.headless and (not args.demo or args.frames < 1):
        parser.error("--headless requires --demo and --frames >= 1")
    lo, hi, default = ENERGIES[args.type]
    energy = default if args.energy is None else args.energy
    if not np.isfinite(energy) or not lo <= energy <= hi:
        parser.error(f"{args.type} energy must be between {lo} and {hi} MeV")
    screen = DisplayArea() if args.headless else display_area()
    scale = args.ui_scale or screen.scale
    cap = detector = None
    engine = Transport(args.type, energy, seed=args.seed if args.seed is not None else 7 if args.demo else None)
    target = Silhouette()
    source = np.array([45., 265.])
    shape = None
    paused = slow = energy_dragging = aiming = manual_aim = False
    controls = args.show_controls
    fitted = False
    index = 0
    last = time.perf_counter()
    was_tracked = False
    view = None

    def choose(kind):
        nonlocal energy_dragging
        engine.reset(kind)
        energy_dragging = False

    def mouse(event, x, y, flags, userdata):
        nonlocal controls, energy_dragging, aiming, manual_aim, slow
        x, y = x/scale, y/scale
        if event == cv2.EVENT_LBUTTONUP:
            energy_dragging = aiming = False
        if shape is None:
            return
        h, w = shape
        boxes = layout(w, h)
        if event == cv2.EVENT_LBUTTONDOWN:
            if inside(boxes["controls"], x, y):
                controls = not controls
                return
            if controls and inside(boxes["close"], x, y):
                controls = False
                return
            for kind in KINDS:
                if inside(boxes[kind], x, y):
                    choose(kind)
                    return
            if inside(boxes["slow"], x, y):
                slow = not slow
                return
            if inside(boxes["energy"], x, y):
                energy_dragging = True
        if controls and inside(boxes["panel"], x, y):
            return
        if energy_dragging and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            lo, hi, _ = ENERGIES[engine.kind]
            box = boxes["energy"]
            fraction = float(np.clip((x-box[0])/(box[2]-box[0]), 0, 1))
            energy = round(lo*(hi/lo)**fraction, 3)
            if energy != engine.energy:
                engine.reset(energy=energy)
            return
        if 0 <= y-HEADER < h:
            if event == cv2.EVENT_LBUTTONDOWN:
                aiming = manual_aim = True
            if aiming and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
                source[:] = [np.clip(x, 15, w*.4), y-HEADER]

    try:
        if not args.demo:
            detector = PoseInput()
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                parser.exit(1, "Cannot open webcam. Allow camera access or try --camera 1.\n")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback(WINDOW, mouse)
        while True:
            now = time.perf_counter()
            dt = 1/30 if args.demo else min(.08, max(.001, now-last))
            last = now
            clock = index/30 if args.demo else now
            if args.demo:
                frame, probability = demo_scene(clock)
                target.update(probability, clock)
            else:
                ok, raw = cap.read()
                if not ok:
                    print("Camera stopped delivering frames.")
                    return 1
                frame = cv2.flip(cv2.resize(raw, (960, round(raw.shape[0]*960/raw.shape[1]))), 1)
                if index % 2 == 0:
                    target.update(detector.detect(frame), time.perf_counter())
            shape = frame.shape[:2]
            mask = target.mask(shape, clock if args.demo else time.perf_counter())
            tracked = bool(mask.any())
            if not tracked and was_tracked:
                engine.reset(energy=engine.energy)
                if not args.demo:
                    print("Body tracking lost; emission paused.", flush=True)
            elif tracked and not was_tracked and not args.demo:
                print("Body tracking active.", flush=True)
            if tracked and not manual_aim:
                ys, _ = np.nonzero(mask)
                source[1] = float(np.mean(ys))
            if tracked and not paused and not energy_dragging:
                engine.step(mask, dt*(.25 if slow else 1), source)
            was_tracked = tracked
            view = render(frame, mask, engine, source, tracked, paused, slow, controls, scale, args.demo)
            index += 1
            if args.headless:
                if index >= args.frames:
                    break
                continue
            cv2.imshow(WINDOW, view)
            if not fitted:
                fit_window(view.shape[1]/scale, view.shape[0]/scale, screen, WINDOW)
                fitted = True
            key = cv2.waitKey(15 if args.demo else 1) & 0xFF
            if key == 27 and controls:
                controls = False
                continue
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("1"), ord("2"), ord("3")):
                choose(KINDS[key-ord("1")])
            elif key in (ord("?"), ord("/")):
                controls = not controls
            elif key == ord(" "):
                paused = not paused
            elif key == ord("m"):
                slow = not slow
            elif key == ord("r"):
                engine.reset(energy=engine.energy)
                manual_aim = False
            elif key == ord("f"):
                fit_window(view.shape[1]/scale, view.shape[0]/scale, display_area(), WINDOW)
        if args.snapshot and view is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), view):
                raise OSError(f"Could not write {args.snapshot}")
        report = dict(particle=engine.kind, energy_MeV=engine.energy, frames=index,
                      counts=engine.counts, active=sum(p.alive for p in engine.particles),
                      target="homogeneous water-like 2D shape", cm_per_pixel=engine.cm_per_pixel)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2)+"\n")
        if args.headless:
            print(json.dumps(report))
        return 0
    except (ImportError, RuntimeError, FileNotFoundError) as exc:
        print(f"Cannot start body tracking: {exc}. Try --demo for a camera-free preview.")
        return 1
    finally:
        if detector is not None:
            detector.close()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
