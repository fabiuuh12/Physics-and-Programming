#!/usr/bin/env python3
"""
Simple webcam fingertip tracker.

Draws red dots on the fingertip landmarks of detected hands.
Supports both legacy MediaPipe `solutions` and newer `tasks`-only builds.
Press 'q' to quit.
"""

import os
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# Some MediaPipe builds import matplotlib internally; keep its cache writable.
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

try:
    import cv2
    import mediapipe as mp
except ImportError as exc:
    print(
        "Missing dependency: "
        f"{exc}. Install with: pip install opencv-python mediapipe"
    )
    sys.exit(1)

TIP_IDS = [4, 8, 12, 16, 20]  # Thumb to pinky tips
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def ensure_hand_model(model_path: Path) -> Path:
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    if model_path.exists() and model_path.stat().st_size == 0:
        model_path.unlink()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand model to {model_path}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            print("SSL cert verification failed; retrying download without verification.")
            try:
                insecure_ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(MODEL_URL, context=insecure_ctx) as src:
                    with model_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Failed to download the hand model after SSL fallback."
                ) from fallback_exc
        else:
            raise RuntimeError(
                "Failed to download the hand model. "
                "Check internet access and retry."
            ) from exc
    except Exception as exc:
        raise RuntimeError("Failed to download the hand model.") from exc
    return model_path


def draw_fingertips(frame, normalized_points):
    h, w, _ = frame.shape
    for x_norm, y_norm in normalized_points:
        x_px = int(x_norm * w)
        y_px = int(y_norm * h)
        cv2.circle(frame, (x_px, y_px), 10, (0, 0, 255), -1)
        cv2.circle(frame, (x_px, y_px), 12, (255, 255, 255), 2)


def show_frame(frame) -> bool:
    cv2.putText(
        frame,
        "Finger tracker test - press q to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Webcam Finger Tracker", frame)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def run_with_solutions(cap) -> None:
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            points = []
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for tip_id in TIP_IDS:
                        landmark = hand_landmarks.landmark[tip_id]
                        points.append((landmark.x, landmark.y))

            draw_fingertips(frame, points)
            if not show_frame(frame):
                break


def run_with_tasks(cap) -> None:
    model_path = ensure_hand_model(MODEL_PATH)
    vision = mp.tasks.vision

    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        last_timestamp_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Warning: failed to read frame from webcam.")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = max(last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
            last_timestamp_ms = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            points = []
            for hand_landmarks in result.hand_landmarks:
                for tip_id in TIP_IDS:
                    landmark = hand_landmarks[tip_id]
                    points.append((landmark.x, landmark.y))

            draw_fingertips(frame, points)
            if not show_frame(frame):
                break


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (camera index 0).")
        return 1

    try:
        if hasattr(mp, "solutions"):
            run_with_solutions(cap)
        elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            run_with_tasks(cap)
        else:
            print(
                "Unsupported mediapipe build: expected either `solutions` or "
                "`tasks.vision` APIs."
            )
            return 1
    except RuntimeError as exc:
        error_text = str(exc)
        if "kGpuService" in error_text or "NSOpenGLPixelFormat" in error_text:
            print("Runtime error: MediaPipe failed to initialize OpenGL services.")
            print("Try running from a local GUI terminal session.")
            print(
                "If it still fails, use Python 3.11 in a virtualenv and reinstall "
                "opencv-python + mediapipe."
            )
        else:
            print(f"Runtime error: {exc}")
        return 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
