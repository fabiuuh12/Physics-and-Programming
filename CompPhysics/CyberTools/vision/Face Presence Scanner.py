#!/usr/bin/env python3
"""Local webcam face-presence monitor.

Detects visible, front-facing faces. It does not identify people, compare
identities, save frames, or record video.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with: pip install -r requirements.txt")
    raise SystemExit(1)


WINDOW_NAME = "Face Presence Monitor"
PANEL_WIDTH = 340
WHITE = (238, 241, 244)
MUTED = (150, 158, 166)
BLUE = (224, 144, 55)
GREEN = (114, 196, 92)
AMBER = (73, 178, 226)
RED = (90, 105, 224)
PANEL = (32, 36, 40)
PANEL_DARK = (23, 26, 29)
DIVIDER = (62, 68, 73)
Box = tuple[int, int, int, int]


@dataclass
class PresenceTracker:
    """Add hysteresis so a single missed frame does not flicker the status."""

    confirm_frames: int = 2
    clear_frames: int = 8
    hit_streak: int = 0
    miss_streak: int = 0
    present: bool = False
    started_at: float | None = None

    def update(self, face_count: int, now: float) -> None:
        if face_count:
            self.hit_streak += 1
            self.miss_streak = 0
            if not self.present and self.hit_streak >= self.confirm_frames:
                self.present = True
                self.started_at = now
        else:
            self.miss_streak += 1
            self.hit_streak = 0
            if self.present and self.miss_streak >= self.clear_frames:
                self.present = False
                self.started_at = None

    def elapsed(self, now: float) -> float:
        return 0.0 if self.started_at is None else max(0.0, now - self.started_at)


@dataclass
class AppState:
    mirror: bool = True
    fullscreen: bool = False
    show_help: bool = True
    faces: tuple[Box, ...] = ()
    tracker: PresenceTracker = field(default_factory=PresenceTracker)
    fps_samples: deque[float] = field(default_factory=lambda: deque(maxlen=30))


@dataclass(frozen=True)
class ImageQuality:
    brightness: float
    sharpness: float
    lighting_label: str
    focus_label: str
    guidance: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect visible faces locally without identifying people."
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="requested capture width")
    parser.add_argument("--height", type=int, default=720, help="requested capture height")
    parser.add_argument("--no-mirror", action="store_true", help="start without mirroring")
    parser.add_argument("--detect-every", type=int, default=2, metavar="N", help="detect every N frames")
    return parser.parse_args()


def load_face_detector() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {path}")
    return detector


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam {index}. Close other camera apps or try --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def detect_faces(frame: np.ndarray, detector: cv2.CascadeClassifier) -> tuple[Box, ...]:
    height, width = frame.shape[:2]
    target_width = min(720, width)
    scale = target_width / width
    small = cv2.resize(frame, (target_width, max(1, round(height * scale))))
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_face = max(30, round(min(gray.shape[:2]) * 0.09))
    found = detector.detectMultiScale(
        gray, scaleFactor=1.12, minNeighbors=6, minSize=(min_face, min_face)
    )
    inverse = 1.0 / scale
    boxes = [tuple(round(int(value) * inverse) for value in box) for box in found]
    boxes.sort(key=lambda box: box[0])
    return tuple(boxes)


def smooth_boxes(previous: tuple[Box, ...], current: tuple[Box, ...]) -> tuple[Box, ...]:
    """Reduce bounding-box jitter when face count and ordering are stable."""
    if not previous or len(previous) != len(current):
        return current
    smoothed: list[Box] = []
    for old, new in zip(previous, current):
        old_center = (old[0] + old[2] / 2, old[1] + old[3] / 2)
        new_center = (new[0] + new[2] / 2, new[1] + new[3] / 2)
        distance = ((old_center[0] - new_center[0]) ** 2 + (old_center[1] - new_center[1]) ** 2) ** 0.5
        if distance < max(old[2], old[3]) * 0.8:
            smoothed.append(tuple(round(0.65 * a + 0.35 * b) for a, b in zip(old, new)))
        else:
            smoothed.append(new)
    return tuple(smoothed)


def measure_quality(frame: np.ndarray) -> ImageQuality:
    """Estimate lighting and focus to explain common detection problems."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sample_width = min(640, gray.shape[1])
    scale = sample_width / gray.shape[1]
    sample = cv2.resize(gray, (sample_width, max(1, round(gray.shape[0] * scale))))
    brightness = float(np.mean(sample))
    sharpness = float(cv2.Laplacian(sample, cv2.CV_64F).var())

    if brightness < 55:
        lighting, guidance = "Low", "Add light in front of the subject"
    elif brightness > 215:
        lighting, guidance = "Overexposed", "Reduce direct light or glare"
    else:
        lighting, guidance = "Good", "Camera conditions are suitable"
    focus = "Low" if sharpness < 45 else "Good"
    if focus == "Low" and lighting == "Good":
        guidance = "Hold still or clean the camera lens"
    return ImageQuality(brightness, sharpness, lighting, focus, guidance)


def format_duration(seconds: float) -> str:
    total = int(seconds)
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_face_box(image: np.ndarray, box: Box, index: int) -> None:
    x, y, width, height = box
    cv2.rectangle(image, (x, y), (x + width, y + height), GREEN, 2, cv2.LINE_AA)
    label = f"Face {index}"
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
    top = max(0, y - 29)
    cv2.rectangle(image, (x, top), (x + text_width + 18, top + 29), GREEN, -1)
    put_text(image, label, (x + 8, top + 20), 0.52, PANEL_DARK)


def draw_camera_view(frame: np.ndarray, faces: tuple[Box, ...]) -> np.ndarray:
    view = frame.copy()
    for index, box in enumerate(faces, start=1):
        draw_face_box(view, box, index)
    height, width = view.shape[:2]
    put_text(view, "LIVE", (20, 32), 0.55, WHITE, 2)
    cv2.circle(view, (77, 26), 5, RED, -1, cv2.LINE_AA)
    put_text(view, "Face detection only - no identity matching", (20, height - 18), 0.50, WHITE)
    cv2.rectangle(view, (0, 0), (width - 1, height - 1), DIVIDER, 1)
    return view


def draw_metric(canvas: np.ndarray, x: int, y: int, label: str, value: str, good: bool = True) -> None:
    put_text(canvas, label.upper(), (x, y), 0.43, MUTED)
    put_text(canvas, value, (x, y + 28), 0.62, WHITE if good else AMBER)


def draw_panel(
    canvas: np.ndarray,
    x: int,
    state: AppState,
    quality: ImageQuality,
    now: float,
    camera_index: int,
    fps: float,
) -> None:
    height = canvas.shape[0]
    cv2.rectangle(canvas, (x, 0), (canvas.shape[1], height), PANEL, -1)
    cv2.rectangle(canvas, (x, 0), (canvas.shape[1], 92), PANEL_DARK, -1)
    left = x + 26
    put_text(canvas, "FACE PRESENCE MONITOR", (left, 35), 0.62)
    put_text(canvas, "Local real-time analysis", (left, 63), 0.48, MUTED)

    status_color = GREEN if state.tracker.present else MUTED
    status = "Face detected" if state.tracker.present else "No face detected"
    cv2.circle(canvas, (left + 7, 128), 7, status_color, -1, cv2.LINE_AA)
    put_text(canvas, status, (left + 25, 135), 0.68, status_color, 2)

    cv2.line(canvas, (left, 166), (x + PANEL_WIDTH - 26, 166), DIVIDER, 1)
    draw_metric(canvas, left, 198, "Visible faces", str(len(state.faces)))
    draw_metric(canvas, left + 150, 198, "Present for", format_duration(state.tracker.elapsed(now)))
    draw_metric(canvas, left, 264, "Lighting", quality.lighting_label, quality.lighting_label == "Good")
    draw_metric(canvas, left + 150, 264, "Focus", quality.focus_label, quality.focus_label == "Good")
    draw_metric(canvas, left, 330, "Camera", str(camera_index))
    draw_metric(canvas, left + 150, 330, "Frame rate", f"{fps:.1f} fps")

    cv2.line(canvas, (left, 378), (x + PANEL_WIDTH - 26, 378), DIVIDER, 1)
    put_text(canvas, "CAPTURE GUIDANCE", (left, 408), 0.43, MUTED)
    put_text(canvas, quality.guidance, (left, 438), 0.48)
    put_text(canvas, "Face the camera and keep your face", (left, 466), 0.44, MUTED)
    put_text(canvas, "fully visible for best results.", (left, 488), 0.44, MUTED)

    cv2.line(canvas, (left, height - 126), (x + PANEL_WIDTH - 26, height - 126), DIVIDER, 1)
    put_text(canvas, "PRIVACY", (left, height - 96), 0.43, MUTED)
    put_text(canvas, "Frames are processed locally.", (left, height - 70), 0.46)
    put_text(canvas, "Nothing is recorded or saved.", (left, height - 46), 0.46)
    if state.show_help:
        put_text(canvas, "Q Quit   F Fullscreen   M Mirror   H Help", (left, height - 17), 0.40, BLUE)


def compose_display(
    frame: np.ndarray,
    state: AppState,
    quality: ImageQuality,
    now: float,
    camera_index: int,
    fps: float,
) -> np.ndarray:
    view = draw_camera_view(frame, state.faces)
    height, width = view.shape[:2]
    canvas = np.full((height, width + PANEL_WIDTH, 3), PANEL, dtype=np.uint8)
    canvas[:, :width] = view
    draw_panel(canvas, width, state, quality, now, camera_index, fps)
    return canvas


def main() -> int:
    args = parse_args()
    if args.detect_every < 1:
        print("Error: --detect-every must be at least 1.")
        return 2
    try:
        detector = load_face_detector()
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    state = AppState(mirror=not args.no_mirror)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(args.width + PANEL_WIDTH, 1600), min(args.height, 900))
    previous_time = time.perf_counter()
    frame_number = 0
    quality = ImageQuality(0.0, 0.0, "Checking", "Checking", "Evaluating camera conditions")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: the webcam stopped returning frames.")
                return 1
            if state.mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = max(now - previous_time, 1.0 / 240.0)
            previous_time = now
            state.fps_samples.append(1.0 / dt)
            fps = sum(state.fps_samples) / len(state.fps_samples)

            if frame_number % args.detect_every == 0:
                detected = detect_faces(frame, detector)
                state.faces = smooth_boxes(state.faces, detected)
                state.tracker.update(len(detected), now)
            if frame_number % 12 == 0:
                quality = measure_quality(frame)
            frame_number += 1

            cv2.imshow(WINDOW_NAME, compose_display(frame, state, quality, now, args.camera, fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                state.mirror = not state.mirror
            elif key == ord("h"):
                state.show_help = not state.show_help
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                mode = cv2.WINDOW_FULLSCREEN if state.fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
                if not state.fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, min(args.width + PANEL_WIDTH, 1600), min(args.height, 900))
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
