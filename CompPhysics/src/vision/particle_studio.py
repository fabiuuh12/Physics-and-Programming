#!/usr/bin/env python3
"""Particle Studio: hand-controlled track chamber and toy collision display.

Tracks: hold a right pinch to release particles; spread left thumb/index for B.
Collisions: pinch both hands apart, then bring the pinches together to collide.
1/2 changes mode; click the scene or Enter releases particles/triggers an event.
V reverses B; P cycles species; C toggles camera background; Space pauses;
R clears; F fits the window; ? shows controls; Q quits. --demo needs no camera.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache, partial
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import cv2
import numpy as np

if __package__:
    from .lensing_camera import DisplayArea, HandInput, display_area, fit_window, inside, rounded_rect, ui_text
    from .particle_studio_model import DETECTOR_RADIUS, LAYERS, NEGATIVE, POSITIVE, SPECIES, Studio
else:
    from lensing_camera import DisplayArea, HandInput, display_area, fit_window, inside, rounded_rect, ui_text
    from particle_studio_model import DETECTOR_RADIUS, LAYERS, NEGATIVE, POSITIVE, SPECIES, Studio

WINDOW = "Particle Studio"
HEADER, FOOTER = 56, 96
MODES = ("tracks", "collisions")


def chamber_scale(w, h):
    return (min(h/2-28, w*.4))/DETECTOR_RADIUS


def to_world(pixel, w, h, limit=3.6):
    position = (np.asarray(pixel, dtype=float)-[w/2, h/2])/[chamber_scale(w, h), -chamber_scale(w, h)]
    length = np.linalg.norm(position)
    if length > limit:
        position *= limit/length
    return position


def geometry(landmarks, w, h):
    if landmarks is None:
        return None
    p = np.asarray(landmarks, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    palm = np.linalg.norm((p[5, :2]-p[17, :2])*[w, h])
    if palm < 10:
        return None
    tips = p[[4, 8], :2]*[w, h]
    point = tips.mean(axis=0)
    ratio = np.linalg.norm(tips[0]-tips[1])/palm
    direction = (p[8, :2]-p[0, :2])*[w, -h]
    return point, ratio, math.atan2(direction[1], direction[0])


@dataclass
class Gestures:
    left_closed: bool = False
    right_closed: bool = False
    armed: bool = False
    latched: bool = False
    next_release: float = 0.
    last_right: float = -math.inf
    points: tuple = ()

    def update(self, hands, mode, studio, w, h, now, dt):
        left, right = (geometry(hands.get(side), w, h) for side in ("left", "right"))
        if left is not None:
            self.left_closed = left[1] < (.55 if self.left_closed else .35)
        if right is not None:
            self.right_closed = right[1] < (.55 if self.right_closed else .35)
            self.last_right = now
        self.points = tuple(p[0] for p in (left, right) if p is not None)
        both = left is not None and right is not None and self.left_closed and self.right_closed
        # A collision gesture must not simultaneously dial the magnetic field to zero.
        if left is not None and not (mode == 'collisions' and (both or self.armed or self.latched)):
            magnitude = float(np.interp(left[1], [.2, 1.8], [0, 3]))
            sign = -1 if studio.magnetic_field < 0 else 1
            alpha = 1-math.exp(-min(.2, max(0, dt))/.12)
            studio.magnetic_field += alpha*(sign*magnitude-studio.magnetic_field)
        if mode == 'tracks':
            if right is not None and self.right_closed and now >= self.next_release:
                studio.release(to_world(right[0], w, h), right[2])
                self.next_release = now+.32
                return 'released'
            return None
        if left is None or right is None:
            if self.armed:
                self.latched = True
            self.armed = False
            return None
        if not self.left_closed and not self.right_closed:
            self.armed = self.latched = False
        if both and not self.latched:
            distance = float(np.linalg.norm(left[0]-right[0]))
            if distance > .23*w:
                self.armed = True
            elif self.armed and distance < .12*w:
                vertex = to_world((left[0]+right[0])/2, w, h, 1.5)
                delta = (right[0]-left[0])*[1, -1]
                studio.collide(vertex, math.atan2(delta[1], delta[0]))
                self.armed = False
                self.latched = True
                return 'collision'
        return None


def layout(w, h):
    return {
        'tracks': (w//2-104, 12, w//2-8, 44),
        'collisions': (w//2-2, 12, w//2+104, 44),
        'controls': (w-126, 12, w-18, 44),
        'action': (20, HEADER+h+12, 128, HEADER+h+44),
        'reverse': (w-166, HEADER+h+12, w-20, HEADER+h+44),
        'field': (150, HEADER+h+60, 400, HEADER+h+83),
        'energy': (690, HEADER+h+60, w-28, HEADER+h+83),
        'panel': (w-378, HEADER+14, w-18, HEADER+min(h-14, 480)),
        'close': (w-54, HEADER+24, w-28, HEADER+50),
    }


@lru_cache(maxsize=6)
def detector_background(w, h, scale):
    y, x = np.indices((h, w), dtype=np.float32)
    radius = np.hypot(x-w/2, y-h/2)
    light = np.exp(-(radius/(h*.5))**2)
    image = np.stack((12+8*light, 12+6*light, 12+4*light), axis=2).astype(np.uint8)
    image = cv2.resize(image, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    center = (w*scale//2, h*scale//2)
    unit = chamber_scale(w, h)*scale
    for i, ring in enumerate(LAYERS):
        r = round(ring*unit)
        cv2.circle(image, center, r, (42+2*i, 43+2*i, 43+2*i), scale, cv2.LINE_AA)
        for angle in np.linspace(0, 2*math.pi, 96, endpoint=False):
            a = np.array([math.cos(angle), math.sin(angle)])
            p0 = tuple(np.int32(np.array(center)+a*(r-2*scale)))
            p1 = tuple(np.int32(np.array(center)+a*(r+2*scale)))
            cv2.line(image, p0, p1, (64, 62, 57), scale, cv2.LINE_AA)
    cv2.circle(image, center, round(DETECTOR_RADIUS*unit), (77, 75, 70), scale, cv2.LINE_AA)
    return image


def render(frame, studio, mode, gestures, paused, controls, camera_background, scale, hands_available=True, demo=False):
    h, w = frame.shape[:2]
    label = partial(ui_text, scale=scale)
    panel = partial(rounded_rect, scale=scale)
    point = lambda p: tuple(round(float(a)*scale) for a in p)
    world = lambda p: point(np.asarray(p)*[chamber_scale(w, h), -chamber_scale(w, h)]+[w/2, h/2])
    view = detector_background(w, h, scale).copy()
    if camera_background:
        camera = cv2.resize(frame, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
        view = cv2.addWeighted(view, .8, camera, .23, 0)
    # Sparse dots/crosses indicate the field normal; no false 3D force arrows.
    for x in range(-2, 3):
        for y in range(-2, 3):
            if abs(studio.magnetic_field) > .02:
                p = world(np.array([x*1.2, y*1.2]))
                if studio.magnetic_field >= 0:
                    cv2.circle(view, p, scale, (75, 70, 61), -1, cv2.LINE_AA)
                else:
                    cv2.drawMarker(view, p, (75, 70, 61), cv2.MARKER_TILTED_CROSS, 4*scale, scale, cv2.LINE_AA)
    glow = np.zeros_like(view)
    for track in studio.tracks:
        points = np.array([world(p) for p in track.points], np.int32)
        fade = 1. if track.alive else max(0, track.fade/2.2)
        if len(points) > 1:
            for i, chunk in enumerate(np.array_split(np.arange(len(points)-1), 5)):
                if len(chunk):
                    ink = tuple(int(c*fade*(.22+.16*i)) for c in track.color)
                    cv2.polylines(glow, [points[chunk[0]:chunk[-1]+2]], False, ink, 2*scale, cv2.LINE_AA)
        if track.alive:
            cv2.circle(glow, world(track.pos), 3*scale, track.color, -1, cv2.LINE_AA)
    for p, color, life in studio.hits:
        cv2.circle(glow, world(p), 4*scale, tuple(int(c*life/1.2) for c in color), -1, cv2.LINE_AA)
    if studio.pending is not None:
        pending = studio.pending
        axis = np.array([math.cos(pending['axis']), math.sin(pending['axis'])])
        distance = 3.6*(1-min(1, pending['progress']))
        for sign in (-1, 1):
            p = pending['vertex']+sign*axis*distance
            tail = p+sign*axis*.6
            cv2.line(glow, world(tail), world(p), (180, 195, 220), 2*scale, cv2.LINE_AA)
            cv2.circle(glow, world(p), 4*scale, (240, 240, 245), -1, cv2.LINE_AA)
    if studio.event_flash > 0 and studio.last_event is not None:
        p = world(studio.last_event.vertex)
        fraction = studio.event_flash/.5
        r = round((4+28*(1-fraction))*scale)
        cv2.circle(glow, p, r, tuple(int(230*fraction) for _ in range(3)), scale, cv2.LINE_AA)
        cv2.circle(glow, p, 5*scale, (240, 245, 255), -1, cv2.LINE_AA)
    blur = cv2.GaussianBlur(cv2.resize(glow, (w//2, h//2)), (0, 0), 2)
    view = cv2.add(view, glow)
    view = cv2.addWeighted(view, 1, cv2.resize(blur, (w*scale, h*scale)), .8, 0)
    for p in gestures.points:
        cv2.circle(view, point(p), 3*scale, (200, 211, 218), -1, cv2.LINE_AA)
    if mode == 'collisions' and gestures.armed and len(gestures.points) == 2:
        cv2.line(view, point(gestures.points[0]), point(gestures.points[1]), (110, 117, 125), scale, cv2.LINE_AA)
    label(view, SPECIES[studio.species][0] if mode == 'tracks' else 'Charge-balanced toy events', (20, 18), 12, (147, 154, 166))
    label(view, f"{len(studio.tracks)} tracks", (20, 41), 11, (115, 124, 138))
    for i, (color, name) in enumerate(((POSITIVE, 'Positive'), (NEGATIVE, 'Negative'))):
        cv2.circle(view, point((w-115, 25+i*22)), 3*scale, color, -1, cv2.LINE_AA)
        label(view, name, (w-104, 19+i*22), 11, (162, 169, 180))
    # Camera stays visible even on the dark detector background.
    iw, ih = 176, round(h*176/w)
    camera = cv2.resize(frame, (iw*scale, ih*scale), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros_like(camera)
    panel(mask, (0, 0, iw-1, ih-1), (255, 255, 255), 8)
    alpha = mask.astype(np.float32)/255
    x, y = 20, h-ih-18
    roi = view[y*scale:(y+ih)*scale, x*scale:(x+iw)*scale]
    roi[:] = roi*(1-alpha)+camera*alpha
    roi[(ih-23)*scale:(ih-3)*scale, 4*scale:(iw-4)*scale] = (roi[(ih-23)*scale:(ih-3)*scale, 4*scale:(iw-4)*scale]*.45).astype(np.uint8)
    label(view, 'Demo' if demo else 'Camera', (x+10, y+ih-21), 10)
    header = np.full((HEADER*scale, w*scale, 3), (23, 22, 21), np.uint8)
    footer = np.full((FOOTER*scale, w*scale, 3), (23, 22, 21), np.uint8)
    label(header, 'Particle Studio', (20, 17), 19)
    boxes = layout(w, h)
    for value in MODES:
        box = boxes[value]
        panel(header, box, (62, 59, 56) if mode == value else (32, 30, 29), 7)
        label(header, value.capitalize(), (box[0]+14, 22), 13,
              (235, 236, 239) if mode == value else (153, 157, 164))
    panel(header, boxes['controls'], (39, 37, 35), 7)
    label(header, 'Controls  ?', (w-113, 22), 13, (184, 188, 194))
    panel(footer, (20, 12, 128, 44), (62, 59, 56), 7)
    label(footer, 'Release' if mode == 'tracks' else 'Collide', (39, 22), 13)
    status = ('Space to resume' if paused else 'Hold right pinch to release' if mode == 'tracks' else
              'Release both pinches to rearm' if gestures.latched else 'Bring pinches together' if gestures.armed else
              'Pinch both hands apart, then bring them together')
    if not hands_available and not paused:
        status = 'Click in the chamber or press Enter'
    label(footer, status, (145, 23), 12, (157, 165, 177))
    panel(footer, (w-166, 12, w-20, 44), (35, 33, 31), 7)
    label(footer, 'Reverse B   V', (w-151, 23), 12, (179, 185, 195))
    label(footer, f"Field {studio.magnetic_field:+.2f} T", (20, 67), 12)
    momentum = studio.momentum if mode == 'tracks' else studio.collision_energy
    caption = f"Momentum {momentum:.2f} GeV/c" if mode == 'tracks' else f"Next event {momentum:.1f} GeV"
    label(footer, caption, (475, 67), 12)
    fraction = math.log(momentum/.08)/math.log(2/.08) if mode == 'tracks' else (momentum-2)/10
    for name, value in (('field', abs(studio.magnetic_field)/3), ('energy', fraction)):
        box = boxes[name]
        x0, x1 = box[0], box[2]
        knob = x0+float(np.clip(value, 0, 1))*(x1-x0)
        cv2.line(footer, point((x0, 72)), point((x1, 72)), (62, 63, 68), 3*scale, cv2.LINE_AA)
        cv2.line(footer, point((x0, 72)), point((knob, 72)), (137, 142, 152), 3*scale, cv2.LINE_AA)
        cv2.circle(footer, point((knob, 72)), 5*scale, (230, 231, 234), -1, cv2.LINE_AA)
    canvas = np.vstack((header, view, footer))
    if controls:
        x0, y0, x1, y1 = boxes['panel']
        panel(canvas, boxes['panel'], (35, 33, 31), 12)
        label(canvas, 'Controls', (x0+20, y0+18), 18)
        label(canvas, '×', (x1-33, y0+15), 21, (165, 167, 173))
        rows = [('Right pinch', 'Release tracks'), ('Left spread', 'Magnetic field strength'),
                ('Both pinches', 'Apart, then together: collide'), ('1 / 2', 'Tracks / collisions'),
                ('Click / Enter', 'Release or collide'), ('V', 'Reverse field'), ('P', 'Next particle species'),
                ('C', 'Camera background'), ('Space', 'Pause'), ('R', 'Clear'), ('F', 'Fit window'), ('Q', 'Quit')]
        for i, (key, action) in enumerate(rows):
            label(canvas, key, (x0+20, y0+65+i*25), 12, (156, 160, 167))
            label(canvas, action, (x0+146, y0+65+i*25), 12)
        label(canvas, 'Vacuum tracks; slowed animation.', (x0+20, y0+390), 11, (139, 148, 161))
        label(canvas, 'Toy collisions conserve energy, momentum', (x0+20, y0+415), 11, (139, 148, 161))
        label(canvas, 'and charge at the vertex; no shower model.', (x0+20, y0+434), 11, (139, 148, 161))
    return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--mode', choices=MODES, default='tracks')
    parser.add_argument('--no-hands', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--headless', action='store_true', help='demo only')
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--ui-scale', type=int, choices=(1, 2, 3))
    parser.add_argument('--show-controls', action='store_true')
    args = parser.parse_args(argv)
    if args.headless and (not args.demo or args.frames < 1):
        parser.error('--headless requires --demo and --frames >= 1')
    screen = DisplayArea() if args.headless else display_area()
    scale = args.ui_scale or screen.scale
    studio, gestures = Studio(args.seed), Gestures()
    mode = args.mode
    cap = detector = None
    shape = (540, 960)
    controls, camera_background, paused = args.show_controls, False, False
    slider = None
    override_until = -math.inf
    fitted = False
    index = 0
    last = last_detection = time.perf_counter()
    view = None
    hand_link_reported = False

    def switch_mode(value):
        nonlocal mode, gestures, slider
        mode = value
        gestures = Gestures()
        slider = None
        studio.clear()
        if mode == 'collisions':
            studio.collide()
        else:
            for position, angle in [((-1., .1), 2.3), ((.8, .2), -.6), ((0., -.9), 1.5)]:
                studio.release(position, angle)

    def action(position=(0., 0.)):
        if mode == 'tracks':
            studio.release(position, studio.rng.uniform(-math.pi, math.pi))
        else:
            studio.collide(position)

    def mouse(event, x, y, flags, userdata):
        nonlocal controls, slider, override_until
        x, y = x/scale, y/scale
        if event == cv2.EVENT_LBUTTONUP:
            slider = None
        h, w = shape
        boxes = layout(w, h)
        if event == cv2.EVENT_LBUTTONDOWN:
            if inside(boxes['controls'], x, y):
                controls = not controls
                return
            for value in MODES:
                if inside(boxes[value], x, y):
                    switch_mode(value)
                    return
            if controls and inside(boxes['close'], x, y):
                controls = False
                return
            if inside(boxes['action'], x, y):
                action()
                return
            if inside(boxes['reverse'], x, y):
                studio.magnetic_field = -studio.magnetic_field if abs(studio.magnetic_field) > .001 else -1.4
                override_until = time.perf_counter()+1.5
                return
            for name in ('field', 'energy'):
                if inside(boxes[name], x, y):
                    slider = name
        if controls and inside(boxes['panel'], x, y):
            return
        if slider and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            box = boxes[slider]
            fraction = float(np.clip((x-box[0])/(box[2]-box[0]), 0, 1))
            if slider == 'field':
                sign = -1 if studio.magnetic_field < 0 else 1
                studio.magnetic_field = sign*3*fraction
                override_until = time.perf_counter()+1.5
            elif mode == 'tracks':
                studio.momentum = .08*(2/.08)**fraction
            else:
                studio.collision_energy = 2+10*fraction
            return
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= y-HEADER < h:
            action(to_world((x, y-HEADER), w, h, 3.6 if mode == 'tracks' else 1.5))

    try:
        if not args.demo:
            cap = cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                parser.exit(1, 'Cannot open camera. Allow camera access or try --camera 1.\n')
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            if not args.no_hands:
                try:
                    detector = HandInput()
                except Exception as exc:
                    print(f'Hand tracking unavailable; mouse controls remain active: {exc}', flush=True)
        switch_mode(mode)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.setMouseCallback(WINDOW, mouse)
        while True:
            now = time.perf_counter()
            dt = 1/30 if args.demo else min(.08, max(.001, now-last))
            last = now
            if args.demo:
                frame = np.full((540, 960, 3), (28, 27, 25), np.uint8)
                if not paused and index and index % (36 if mode == 'tracks' else 120) == 0:
                    action(studio.rng.uniform(-.8, .8, 2))
            else:
                ok, raw = cap.read()
                if not ok:
                    print('Camera stopped delivering frames.')
                    return 1
                frame = cv2.flip(cv2.resize(raw, (960, round(raw.shape[0]*960/raw.shape[1]))), 1)
            shape = frame.shape[:2]
            if detector is not None and index % 2 == 0:
                try:
                    hands = detector.detect(frame)
                    if hands and not hand_link_reported:
                        print('Hand tracking active.', flush=True)
                        hand_link_reported = True
                    if not paused:
                        saved_field = studio.magnetic_field
                        gestures.update(hands, mode, studio, shape[1], shape[0], now, now-last_detection)
                        if now < override_until:
                            studio.magnetic_field = saved_field
                    else:
                        gestures.points = tuple(value[0] for value in
                            (geometry(hands.get(side), shape[1], shape[0]) for side in ('left', 'right'))
                            if value is not None)
                    last_detection = now
                except Exception as exc:
                    print(f'Hand tracking stopped; using mouse controls: {exc}', flush=True)
                    detector.close()
                    detector = None
                    gestures = Gestures()
            if not paused:
                studio.step(dt)
            view = render(frame, studio, mode, gestures, paused, controls, camera_background, scale,
                          hands_available=detector is not None, demo=args.demo)
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
            if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('1'), ord('2')):
                switch_mode(MODES[key-ord('1')])
            elif key in (10, 13):
                action()
            elif key in (ord('?'), ord('/')):
                controls = not controls
            elif key == ord('v'):
                studio.magnetic_field = -studio.magnetic_field if abs(studio.magnetic_field) > .001 else -1.4
                override_until = now+1.5
            elif key == ord('p'):
                studio.species = (studio.species+1)%len(SPECIES)
            elif key == ord('c'):
                camera_background = not camera_background
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                studio.clear()
                gestures = Gestures()
            elif key == ord('f'):
                fit_window(view.shape[1]/scale, view.shape[0]/scale, display_area(), WINDOW)
        if args.snapshot and view is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), view):
                raise OSError(f'Could not write {args.snapshot}')
        event = studio.last_event
        report = dict(mode=mode, frames=index, tracks=len(studio.tracks), events=studio.events,
                      field_T=studio.magnetic_field,
                      event_energy_GeV=None if event is None else event.energy,
                      energy_residual_GeV=None if event is None else event.outgoing_energy-event.energy,
                      momentum_residual_GeV_c=None if event is None else event.momentum_residual.tolist())
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2)+'\n')
        if args.headless:
            print(json.dumps(report))
        return 0
    finally:
        if detector is not None:
            detector.close()
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
