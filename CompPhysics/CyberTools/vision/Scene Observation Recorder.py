#!/usr/bin/env python3
"""Local people and body-landmark observation recorder.

Detects people and pose landmarks, derives clothing-color observations, and
optionally stores structured observations in SQLite. Camera frames are
never saved. This tool does not identify people or infer sensitive traits.

Controls: Q/Esc quit, R toggle data recording, F fullscreen, M mirror, H help.
"""

from __future__ import annotations

import argparse
import json
import socket
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with: pip install -r requirements.txt")
    raise SystemExit(1)


WINDOW_NAME = "Scene Observation Recorder"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
MODEL_DIR = PROJECT_DIR / "vision" / "models"
DEFAULT_POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
DEFAULT_DATABASE = SCRIPT_DIR / "data" / "scene_observations.sqlite3"
DEFAULT_POSE_UDP_HOST = "127.0.0.1"
DEFAULT_POSE_UDP_PORT = 50525

PANEL_WIDTH = 360
WHITE = (238, 241, 244)
MUTED = (150, 158, 166)
BLUE = (224, 144, 55)
GREEN = (114, 196, 92)
AMBER = (73, 178, 226)
RED = (90, 105, 224)
PANEL = (32, 36, 40)
PANEL_DARK = (23, 26, 29)
DIVIDER = (62, 68, 73)
COLOR_SWATCHES = {
    "black": (25, 25, 25), "white": (240, 240, 240), "gray": (135, 135, 135),
    "red": (55, 55, 215), "orange": (35, 135, 235), "yellow": (45, 215, 230),
    "green": (70, 175, 75), "blue": (205, 105, 45), "purple": (175, 75, 150),
    "pink": (175, 115, 230), "mixed": (160, 160, 160), "uncertain": (85, 85, 85),
}
POSE_LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)
DISPLAY_LANDMARKS = {
    11: "L shoulder", 12: "R shoulder", 13: "L elbow", 14: "R elbow",
    15: "L wrist", 16: "R wrist", 23: "L hip", 24: "R hip",
    25: "L knee", 26: "R knee", 27: "L ankle", 28: "R ankle",
}


@dataclass(frozen=True)
class ClothingObservation:
    person_index: int
    region: str
    primary_color: str
    secondary_color: str | None
    color_share: float
    sample_pixels: int
    center: tuple[float, float]


@dataclass
class AppState:
    mirror: bool = True
    fullscreen: bool = False
    show_help: bool = True
    recording: bool = False
    poses: list[list[object]] = field(default_factory=list)
    clothing: list[ClothingObservation] = field(default_factory=list)
    silhouette_contours: list[np.ndarray] = field(default_factory=list)
    silhouette_probability: np.ndarray | None = None
    records_written: int = 0
    last_saved_at: float = 0.0
    fps_samples: list[float] = field(default_factory=list)


class ObservationStore:
    """Lazy SQLite writer; no file is created until recording is enabled."""

    def __init__(self, path: Path, self_described_profile: dict[str, str] | None = None) -> None:
        self.path = path
        self.connection: sqlite3.Connection | None = None
        self.session_id = str(uuid.uuid4())
        self.self_described_profile = self_described_profile or {}

    def open(self) -> None:
        if self.connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_utc TEXT NOT NULL,
                app_version TEXT NOT NULL,
                notes TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                captured_utc TEXT NOT NULL,
                kind TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL,
                center_x REAL,
                center_y REAL,
                width REAL,
                height REAL,
                characteristics_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_observations_session
                ON observations(session_id, captured_utc);
            """
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO sessions VALUES (?, ?, ?, ?)",
            (
                self.session_id,
                utc_now(),
                "1.1",
                json.dumps(
                    {
                        "data_policy": "Local structured observations; no camera frames stored.",
                        "self_described_profile": self.self_described_profile,
                    }
                ),
            ),
        )
        self.connection.commit()

    def save(
        self,
        poses: list[list[object]],
        clothing: list[ClothingObservation],
    ) -> int:
        self.open()
        assert self.connection is not None
        captured = utc_now()
        rows: list[tuple[object, ...]] = []
        for pose_index, landmarks in enumerate(poses):
            visible = [
                {
                    "index": index,
                    "name": POSE_LANDMARK_NAMES[index],
                    "x": round(float(point.x), 5),
                    "y": round(float(point.y), 5),
                    "z": round(float(point.z), 5),
                    "visibility": round(float(point.visibility), 5),
                }
                for index, point in enumerate(landmarks)
                if point.visibility >= 0.35
            ]
            rows.append(
                (
                    self.session_id, captured, "pose", f"person_{pose_index + 1}", None,
                    None, None, None, None,
                    json.dumps({"landmarks": visible}),
                )
            )
        for item in clothing:
            rows.append(
                (
                    self.session_id, captured, "clothing", item.region, item.color_share,
                    item.center[0], item.center[1], None, None,
                    json.dumps(
                        {
                            "person_index": item.person_index,
                            "primary_color": item.primary_color,
                            "secondary_color": item.secondary_color,
                            "classification": "approximate_visible_color",
                            "sample_pixels": item.sample_pixels,
                        }
                    ),
                )
            )
        self.connection.executemany(
            """
            INSERT INTO observations (
                session_id, captured_utc, kind, label, confidence,
                center_x, center_y, width, height, characteristics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--sample-seconds", type=float, default=0.5, help="recording interval")
    parser.add_argument("--pose-every", type=int, default=3, metavar="N", help="pose inference interval")
    parser.add_argument("--max-people", type=int, default=8, metavar="N", help="maximum simultaneous poses")
    parser.add_argument(
        "--outline-smoothing",
        type=float,
        default=0.68,
        metavar="VALUE",
        help="outline stability from 0 (responsive) to 0.9 (very stable)",
    )
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--profile", type=Path, help="optional self-described JSON profile")
    parser.add_argument("--pose-udp-host", default=DEFAULT_POSE_UDP_HOST)
    parser.add_argument("--pose-udp-port", type=int, default=DEFAULT_POSE_UDP_PORT)
    parser.add_argument("--no-pose-stream", action="store_true", help="disable localhost avatar stream")
    return parser.parse_args()


def load_profile(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read profile {path}: {exc}") from exc
    if not isinstance(data, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in data.items()):
        raise RuntimeError("Profile must be a JSON object containing string keys and values.")
    allowed = {"profile_id", "race", "eye_color", "hair_color", "notes"}
    unknown = set(data) - allowed
    if unknown:
        raise RuntimeError(f"Unsupported profile fields: {', '.join(sorted(unknown))}")
    return data


def create_pose_detector(pose_model: Path, max_people: int = 8):
    if not pose_model.is_file():
        raise RuntimeError(f"Required model not found: {pose_model}")
    # Loading bytes avoids native-library path issues with OneDrive and Unicode
    # directory names on Windows.
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=pose_model.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=max_people,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {index}. Close other camera apps or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def classify_color_pixels(pixels_bgr: np.ndarray) -> tuple[str, str | None, float]:
    """Return the two most common coarse colors and the primary share."""
    if pixels_bgr.shape[0] < 80:
        return "uncertain", None, 0.0
    hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    labels = np.full(len(hsv), "pink", dtype="<U10")
    labels[(hue < 8) | (hue >= 172)] = "red"
    labels[(hue >= 8) & (hue < 22)] = "orange"
    labels[(hue >= 22) & (hue < 36)] = "yellow"
    labels[(hue >= 36) & (hue < 86)] = "green"
    labels[(hue >= 86) & (hue < 132)] = "blue"
    labels[(hue >= 132) & (hue < 160)] = "purple"
    labels[value < 45] = "black"
    neutral = (saturation < 28) & (value >= 45)
    labels[neutral] = "gray"
    labels[neutral & (value > 205)] = "white"
    names, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    primary = str(names[order[0]])
    share = float(counts[order[0]] / counts.sum())
    secondary = str(names[order[1]]) if len(order) > 1 and counts[order[1]] / counts.sum() >= 0.18 else None
    if share < 0.28:
        return "mixed", secondary, share
    return primary, secondary, share


def _inset_polygon(points: np.ndarray, amount: float = 0.12) -> np.ndarray:
    center = np.mean(points, axis=0)
    return np.rint(points * (1.0 - amount) + center * amount).astype(np.int32)


def sample_clothing_region(
    frame: np.ndarray,
    polygon: np.ndarray,
    person_index: int,
    region: str,
) -> ClothingObservation:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, _inset_polygon(polygon), 255, cv2.LINE_AA)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    pixels = frame[mask > 0]
    primary, secondary, share = classify_color_pixels(pixels)
    center_px = np.mean(polygon, axis=0)
    return ClothingObservation(
        person_index=person_index,
        region=region,
        primary_color=primary,
        secondary_color=secondary,
        color_share=share,
        sample_pixels=int(pixels.shape[0]),
        center=(float(center_px[0] / frame.shape[1]), float(center_px[1] / frame.shape[0])),
    )


def analyze_clothing(frame: np.ndarray, poses: list[list[object]]) -> list[ClothingObservation]:
    """Sample pose-defined clothing regions while avoiding face-based traits."""
    height, width = frame.shape[:2]
    output: list[ClothingObservation] = []
    for person_index, landmarks in enumerate(poses, start=1):
        upper_points = [_point(landmarks, index, width, height) for index in (11, 12, 24, 23)]
        if all(point is not None for point in upper_points):
            upper = np.asarray(upper_points, dtype=np.int32)
            # Move the shoulder edge downward to reduce neck/skin contamination.
            upper[0:2] = np.rint(upper[0:2] * 0.82 + upper[2:4][::-1] * 0.18).astype(np.int32)
            output.append(sample_clothing_region(frame, upper, person_index, "upper_clothing"))

        lower_points = [_point(landmarks, index, width, height) for index in (23, 24, 26, 25)]
        if all(point is not None for point in lower_points):
            output.append(
                sample_clothing_region(
                    frame, np.asarray(lower_points, dtype=np.int32), person_index, "lower_clothing"
                )
            )
    return output


def make_mp_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def detect_pose(image: mp.Image, pose_detector, timestamp_ms: int):
    result = pose_detector.detect_for_video(image, timestamp_ms)
    return (
        result.pose_landmarks,
        result.pose_world_landmarks,
        list(result.segmentation_masks or []),
    )


def send_pose_packets(
    sock: socket.socket,
    host: str,
    port: int,
    timestamp_ms: int,
    world_poses: list[list[object]],
) -> None:
    """Send one compact CSV datagram per person for the C++ avatar receiver."""
    for person_index, landmarks in enumerate(world_poses):
        if len(landmarks) != 33:
            continue
        parts = ["POSE", "1", str(timestamp_ms), str(person_index), "33"]
        for point in landmarks:
            parts.extend(
                (
                    f"{float(point.x):.5f}",
                    f"{float(point.y):.5f}",
                    f"{float(point.z):.5f}",
                    f"{float(point.visibility):.4f}",
                )
            )
        try:
            sock.sendto(",".join(parts).encode("ascii"), (host, port))
        except OSError:
            # Tracking must continue even if no avatar receiver is running.
            pass


def update_silhouette(
    previous: np.ndarray | None,
    masks: list[object],
    width: int,
    height: int,
    smoothing: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Temporally stabilize a union of all person masks and extract clean edges."""
    current = np.zeros((height, width), dtype=np.float32)
    for mask in masks:
        probability = np.asarray(mask.numpy_view(), dtype=np.float32)
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
        current = np.maximum(current, probability)

    if previous is None or previous.shape != current.shape:
        stabilized = current
    else:
        # New foreground appears quickly; disappearing foreground fades faster
        # than a symmetric average so the outline stays stable without ghosts.
        base_response = 1.0 - smoothing
        attack = min(0.92, base_response + 0.48)
        release = min(0.72, base_response + 0.20)
        response = np.where(current > previous, attack, release).astype(np.float32)
        stabilized = previous + (current - previous) * response

    stabilized = cv2.GaussianBlur(stabilized, (0, 0), 1.15)
    binary = np.where(stabilized >= 0.50, 255, 0).astype(np.uint8)
    scale = max(1, round(min(width, height) / 360))
    close_size = 2 * scale + 3
    open_size = 2 * scale + 1
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8)
    )
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_area = height * width * 0.0025
    cleaned = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    return stabilized, cleaned


def put_text(image, text: str, position: tuple[int, int], scale=0.52, color=WHITE, thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _point(landmarks: list[object], index: int, width: int, height: int) -> tuple[int, int] | None:
    landmark = landmarks[index]
    if landmark.visibility < 0.58:
        return None
    return round(landmark.x * width), round(landmark.y * height)


def _draw_capsule(
    frame: np.ndarray,
    start: tuple[int, int] | None,
    end: tuple[int, int] | None,
    label: str,
) -> None:
    """Outline a pose-defined body region without drawing a center skeleton line."""
    if start is None or end is None:
        return
    length = max(1, round(((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5))
    thickness = max(12, round(length * 0.24))
    region = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.line(region, start, end, 255, thickness, cv2.LINE_AA)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, AMBER, 2, cv2.LINE_AA)
    midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    put_text(frame, label, (midpoint[0] + 5, midpoint[1] - 5), 0.34, WHITE)


def draw_pose(frame: np.ndarray, poses: list[list[object]], silhouette_contours: list[np.ndarray]) -> None:
    height, width = frame.shape[:2]
    # A subtle dark under-stroke keeps the precise green edge readable against
    # both bright and dark backgrounds.
    cv2.drawContours(frame, silhouette_contours, -1, PANEL_DARK, 4, cv2.LINE_AA)
    cv2.drawContours(frame, silhouette_contours, -1, GREEN, 2, cv2.LINE_AA)

    # Landmarks divide the silhouette into named anatomical regions. Capsules
    # are only a visualization of those regions; detection still uses landmarks.
    segments = (
        (11, 13, "L upper arm"), (13, 15, "L forearm"),
        (12, 14, "R upper arm"), (14, 16, "R forearm"),
        (23, 25, "L thigh"), (25, 27, "L lower leg"),
        (24, 26, "R thigh"), (26, 28, "R lower leg"),
    )
    for landmarks in poses:
        for start_index, end_index, label in segments:
            _draw_capsule(
                frame,
                _point(landmarks, start_index, width, height),
                _point(landmarks, end_index, width, height),
                label,
            )

        torso_indices = (11, 12, 24, 23)
        torso = [_point(landmarks, index, width, height) for index in torso_indices]
        if all(point is not None for point in torso):
            polygon = np.asarray(torso, dtype=np.int32)
            cv2.polylines(frame, [polygon], True, AMBER, 2, cv2.LINE_AA)
            center = tuple(np.mean(polygon, axis=0).astype(int))
            put_text(frame, "torso", (center[0] - 18, center[1]), 0.38, WHITE)

        left_ear = _point(landmarks, 7, width, height)
        right_ear = _point(landmarks, 8, width, height)
        if left_ear is not None and right_ear is not None:
            center = ((left_ear[0] + right_ear[0]) // 2, (left_ear[1] + right_ear[1]) // 2)
            radius = max(14, round(abs(left_ear[0] - right_ear[0]) * 0.72))
            cv2.ellipse(frame, center, (radius, round(radius * 1.25)), 0, 0, 360, AMBER, 2, cv2.LINE_AA)
            put_text(frame, "head", (center[0] + radius + 4, center[1]), 0.36, WHITE)


def draw_clothing(frame: np.ndarray, clothing: list[ClothingObservation]) -> None:
    height, width = frame.shape[:2]
    for item in clothing:
        x, y = round(item.center[0] * width), round(item.center[1] * height)
        region = "upper" if item.region == "upper_clothing" else "lower"
        color_text = item.primary_color
        if item.secondary_color:
            color_text += f" + {item.secondary_color}"
        label = f"P{item.person_index} {region}: {color_text} ({item.color_share:.0%})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0]
        left = max(0, min(width - text_size[0] - 34, x - text_size[0] // 2))
        top = max(42, min(height - 34, y + 12))
        cv2.rectangle(frame, (left, top), (left + text_size[0] + 34, top + 27), PANEL_DARK, -1)
        cv2.rectangle(frame, (left + 6, top + 7), (left + 19, top + 20), COLOR_SWATCHES[item.primary_color], -1)
        put_text(frame, label, (left + 26, top + 19), 0.44, WHITE)


def draw_panel(canvas: np.ndarray, x: int, state: AppState, fps: float, database: Path) -> None:
    height = canvas.shape[0]
    cv2.rectangle(canvas, (x, 0), (canvas.shape[1], height), PANEL, -1)
    cv2.rectangle(canvas, (x, 0), (canvas.shape[1], 92), PANEL_DARK, -1)
    left = x + 26
    put_text(canvas, "SCENE OBSERVATION", (left, 35), 0.64, WHITE)
    put_text(canvas, "People + body landmarks", (left, 63), 0.48, MUTED)

    status_color = RED if state.recording else MUTED
    cv2.circle(canvas, (left + 7, 128), 7, status_color, -1, cv2.LINE_AA)
    put_text(canvas, "Recording data" if state.recording else "Preview only", (left + 25, 135), 0.66, status_color, 2)
    cv2.line(canvas, (left, 166), (x + PANEL_WIDTH - 26, 166), DIVIDER, 1)

    metrics = (
        ("CLOTHING", str(len(state.clothing)), left, 198),
        ("PEOPLE", str(len(state.poses)), left + 155, 198),
        ("ROWS SAVED", str(state.records_written), left, 268),
        ("FRAME RATE", f"{fps:.1f} fps", left + 155, 268),
    )
    for label, value, px, py in metrics:
        put_text(canvas, label, (px, py), 0.42, MUTED)
        put_text(canvas, value, (px, py + 29), 0.64, WHITE)

    cv2.line(canvas, (left, 326), (x + PANEL_WIDTH - 26, 326), DIVIDER, 1)
    put_text(canvas, "CLOTHING COLORS", (left, 358), 0.42, MUTED)
    y = 390
    for item in state.clothing[:4]:
        secondary = f" / {item.secondary_color}" if item.secondary_color else ""
        swatch = COLOR_SWATCHES[item.primary_color]
        cv2.rectangle(canvas, (left, y - 13), (left + 14, y + 1), swatch, -1)
        put_text(
            canvas,
            f"P{item.person_index} {item.region.replace('_', ' ')}: {item.primary_color}{secondary}",
            (left + 23, y), 0.43, WHITE,
        )
        y += 25
    if not state.clothing:
        put_text(canvas, "No clothing region available", (left, y), 0.47, MUTED)
        y += 25

    cv2.line(canvas, (left, height - 136), (x + PANEL_WIDTH - 26, height - 136), DIVIDER, 1)
    put_text(canvas, "DATA POLICY", (left, height - 105), 0.42, MUTED)
    put_text(canvas, "Structured observations only.", (left, height - 79), 0.46, WHITE)
    put_text(canvas, "No images, identity, or sensitive traits.", (left, height - 55), 0.43, WHITE)
    if state.show_help:
        put_text(canvas, "R Record   Q Quit   F Full   M Mirror", (left, height - 20), 0.42, BLUE)


def compose_display(frame: np.ndarray, state: AppState, fps: float, database: Path) -> np.ndarray:
    view = frame.copy()
    draw_pose(view, state.poses, state.silhouette_contours)
    draw_clothing(view, state.clothing)
    status = "REC" if state.recording else "LIVE"
    put_text(view, status, (20, 32), 0.55, RED if state.recording else WHITE, 2)
    put_text(view, "Local analysis - camera frames are not stored", (20, view.shape[0] - 18), 0.48, WHITE)
    height, width = view.shape[:2]
    canvas = np.full((height, width + PANEL_WIDTH, 3), PANEL, dtype=np.uint8)
    canvas[:, :width] = view
    draw_panel(canvas, width, state, fps, database)
    return canvas


def main() -> int:
    args = parse_args()
    if (
        args.sample_seconds <= 0
        or args.pose_every < 1
        or not 1 <= args.max_people <= 20
        or not 0.0 <= args.outline_smoothing <= 0.9
        or not 1 <= args.pose_udp_port <= 65535
    ):
        print("Error: invalid interval, max people (1-20), or smoothing (0-0.9).")
        return 2
    try:
        profile = load_profile(args.profile)
        pose_detector = create_pose_detector(DEFAULT_POSE_MODEL, args.max_people)
        cap = open_camera(args.camera, args.width, args.height)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    state = AppState(mirror=not args.no_mirror)
    store = ObservationStore(args.database.resolve(), profile)
    pose_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, min(args.width + PANEL_WIDTH, 1640), min(args.height, 900))
    started = time.perf_counter()
    previous = started
    frame_number = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: the webcam stopped returning frames.")
                return 1
            if state.mirror:
                frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            state.fps_samples.append(1.0 / max(now - previous, 1.0 / 240.0))
            state.fps_samples = state.fps_samples[-30:]
            previous = now

            run_pose = frame_number % args.pose_every == 0
            if run_pose:
                timestamp_ms = int((now - started) * 1000)
                image = make_mp_image(frame)
                state.poses, world_poses, masks = detect_pose(image, pose_detector, timestamp_ms)
                if not args.no_pose_stream:
                    send_pose_packets(
                        pose_socket,
                        args.pose_udp_host,
                        args.pose_udp_port,
                        timestamp_ms,
                        world_poses,
                    )
                state.silhouette_probability, state.silhouette_contours = update_silhouette(
                    state.silhouette_probability,
                    masks,
                    frame.shape[1],
                    frame.shape[0],
                    args.outline_smoothing,
                )
                state.clothing = analyze_clothing(frame, state.poses)
            frame_number += 1

            if state.recording and now - state.last_saved_at >= args.sample_seconds:
                state.records_written += store.save(state.poses, state.clothing)
                state.last_saved_at = now

            fps = sum(state.fps_samples) / len(state.fps_samples)
            cv2.imshow(WINDOW_NAME, compose_display(frame, state, fps, args.database))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                state.recording = not state.recording
                if state.recording:
                    store.open()
                    state.last_saved_at = 0.0
            elif key == ord("m"):
                state.mirror = not state.mirror
            elif key == ord("h"):
                state.show_help = not state.show_help
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                mode = cv2.WINDOW_FULLSCREEN if state.fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
                if not state.fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, min(args.width + PANEL_WIDTH, 1640), min(args.height, 900))
    finally:
        cap.release()
        pose_detector.close()
        pose_socket.close()
        store.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
